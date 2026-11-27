#!/usr/bin/env python
"""Stage-2 rollouts (docs/DATA_PLAN.md section 11): k samples per task on the SWE-smith and TermGrade pools through the
evaluation scaffolds against one served policy, with a continuous scheduler, per-family concurrency, resume by episode id
and a wall-clock cut-off.

Usage (from the repository root):
    .venv/train/bin/python scripts/data/run_rollouts.py --policy sft --api-base http://HOST:8000/v1 --model-name qwen35-sft \
        [--tasks-file data/pools/swesmith/tasks.jsonl --tasks-file data/pools/tgr_env/tasks.jsonl] [--split stage2] \
        [--round stage2] [--n 4] [--max-tasks swe=420,terminal=180] [--task-ids-file FILE] [--family swe|terminal] \
        [--max-sandboxes 96] [--family-concurrency swe=64,terminal=32] [--stop-launching-at ISO] [--cutoff ISO] \
        [--caps-override terminal.max_output_tokens=120000,terminal.wall_sec=2700] [--run-id ID --resume] [--dry-run]

Scaffolds (one worker process per episode, src/apex_data/rollout/; workers run in .venv/harbor):
    swe       mini-swe-agent 2.4.6 tool calls, SWE-bench Pro lock config + sampling   caps 50 steps / 16,384 per turn / 40K / 20 min
    terminal  Terminus-2 through `harbor trial start`, Terminal-Bench lock settings   caps 40 turns / lock per turn / 60K / 30 min
    (default caps, endpoint.STAGE2_CAPS; --caps-override changes them for a launch: the specs, the agent config, the proxy's
    output budget, the scheduler's per-item timeout and the recorded caps all follow)
Sample k of task t sends seed endpoint.sample_seed(t, k) with every request; episode_id = <task_id>#<round>#<policy>#<k>.
Tasks: split == --split (default stage2); SWE-smith tasks need a calibration with status ok (gold replay admitted them,
grade/swesmith) and their image; TermGrade tasks need extra.env_check.passed and their prepared task directory. Tasks
failing these are counted (skipped_tasks), never rolled out. --max-tasks takes the first N per family in pool order (the
pools are written in their seeded draw order). Priority (sample index, family, task order): a cut-off run covers every
task once before any task twice.
Cut-off (plan 12, drop order D): --stop-launching-at starts no episode after that time; --cutoff terminates running
episodes at that time (state cut_off; whatever is graded by the cut-off is the set).

Inputs : task JSONL (schema.Task) of the pools; the served model at --api-base (key: env var named by --api-key-env, a dummy
         for vLLM); docker; SWE-smith mirrors (cloned on first use under RAW/swesmith_mirrors); the prepared TermGrade dataset
         (--terminal-dataset-dir, default grade.harbor_tests.PREPARED_DIR).
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl (one line per episode), errors.log, resources.csv,
         rollout_summary.json, config_snapshot/{mini_swe_agent_stage2.yaml, lock copies}}
         WORK/<run_id>/episodes.jsonl (schema.Episode, append-only: the resume point and grade_episodes.py's input),
         state.json (scheduler table), requests/<episode>.jsonl.gz (every request body and response, no secrets),
         episodes/<episode>/ (spec.json, worker.log, episode.json, Harbor trial dir for terminal episodes).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.coordination import wait_for_docker_window  # noqa: E402
from apex_data.grade import harbor_tests  # noqa: E402
from apex_data.grade.swesmith import DEFAULT_MIRRORS_DIR, ensure_mirror, image_present, repo_name_of, slug  # noqa: E402
from apex_data.rollout import mini_swe, terminus  # noqa: E402
from apex_data.rollout.endpoint import (FAMILIES, FAMILY_OF_SCAFFOLD, LOCKS, Caps, Endpoint, caps_for, episode_id,  # noqa: E402
                                        lock_inference)
from apex_data.rollout.scheduler import Limits, Outcome, Scheduler, WorkItem, parse_time  # noqa: E402
from apex_data.schema import read_jsonl, task_from_dict  # noqa: E402
from apex_eval.common import (RunDir, command_line, compute_rates, git_info, hardware_info, new_run_id, resource_rows,  # noqa: E402
                              utc_now)

FAMILY_RANK = {"swe": 0, "terminal": 1}
SOURCE_FAMILY = {"swesmith": "swe", "tgr_env": "terminal"}
DEFAULT_TASK_FILES = (paths.POOLS / "swesmith" / "tasks.jsonl", paths.POOLS / "tgr_env" / "tasks.jsonl")
TIMEOUT_GRACE_SEC = {"swe": 1800, "terminal": terminus.GRACE_SEC + 600}   # on top of the wall cap before the scheduler kills a worker
TRANSIENT_ERRORS = ("SandboxError", "APIConnectionError", "ServiceUnavailableError", "InternalServerError", "Timeout", "APIError")


def parse_caps_override(text: str) -> dict[str, dict[str, int]]:
    """`family.field=value,...` -> {family: {field: value}} over endpoint.Caps fields (positive integers)."""
    fields = {f.name for f in dataclasses.fields(Caps)}
    out: dict[str, dict[str, int]] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        fam, _, field = key.strip().partition(".")
        if fam not in FAMILIES or field not in fields or not value.strip().isdigit() or int(value) < 1:
            raise ValueError(f"--caps-override {part.strip()!r}: expected family.field=positive int, family in {FAMILIES}, "
                             f"field in {sorted(fields)}")
        out.setdefault(fam, {})[field] = int(value)
    return out


def parse_kv(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = int(v)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-file", type=Path, action="append", default=None, help="schema.Task JSONL (repeatable; default: both pools)")
    p.add_argument("--split", default="stage2", help="task split to roll out (stage2; dev only for driver checks)")
    p.add_argument("--family", choices=FAMILIES, default=None, help="only this family")
    p.add_argument("--policy", required=True, choices=["base", "sft", "stage2"], help="the policy the endpoint serves (in episode ids)")
    p.add_argument("--round", dest="round_", default="stage2", choices=["stage2", "pilot", "dev"], help="episode round (stage2 = the set)")
    p.add_argument("--api-base", required=True, help="OpenAI-compatible base URL, e.g. http://172.18.0.1:8000/v1")
    p.add_argument("--model-name", required=True, help="served model id (litellm gets openai/<id>)")
    p.add_argument("--api-key-env", default="EVAL_API_KEY", help="NAME of the env var holding the endpoint key")
    p.add_argument("--policy-revision", default=None, help="what the endpoint serves (e.g. the SFT run id / adapter sha), for metadata")
    p.add_argument("--n", type=int, default=4, help="samples per task (plan: 4)")
    p.add_argument("--max-tasks", default="", help="first N eligible tasks per family, e.g. swe=420,terminal=180 (plan: 600 tasks)")
    p.add_argument("--task-ids-file", type=Path, default=None, help="only these task ids (one per line), e.g. a 10-task pilot")
    p.add_argument("--max-sandboxes", type=int, default=96, help="global cap on concurrent episodes (plan: 64 to 96)")
    p.add_argument("--family-concurrency", default="swe=64,terminal=32", help="per-family caps, family=N,...")
    p.add_argument("--load-pause", type=float, default=None, help="pause launches above this 1-min load average (default cpu_count + 8)")
    p.add_argument("--control-file", type=Path, default=None, help="JSON file re-read every 30 s while running: pause_launches, load_pause, tb_container_cap, max_sandboxes, per_family overlay the flags (removing it restores them); running episodes are never touched; every change is logged and recorded")
    p.add_argument("--tb-container-cap", type=int, default=None, help="while a reportable Terminal-Bench run is active "
                   "(logs/coordination/terminal_bench_active.json), start no episode once the host runs this many containers (default off)")
    p.add_argument("--cgroup-parent", default=None, help="systemd slice for the SWE containers")
    p.add_argument("--mirrors-dir", type=Path, default=DEFAULT_MIRRORS_DIR, help="SWE-smith mirror cache")
    p.add_argument("--terminal-dataset-dir", type=Path, default=harbor_tests.PREPARED_DIR, help="prepared TermGrade Harbor dataset")
    p.add_argument("--pull-missing", action="store_true", help="docker pull SWE task images that are not present (default: skip them)")
    p.add_argument("--max-env-retries", type=int, default=1, help="re-runs of an episode whose sandbox or endpoint failed (plan: once)")
    p.add_argument("--stop-launching-at", default=None, help="ISO time (UTC): start no episode after it")
    p.add_argument("--cutoff", default=None, help="ISO time (UTC): terminate running episodes at it (drop order D)")
    p.add_argument("--run-id", default=None, help="default: data_rollouts_<policy>_<round>_<UTC timestamp>")
    p.add_argument("--resume", action="store_true", help="continue --run-id: episodes already in episodes.jsonl are skipped")
    p.add_argument("--caps-override", default="", help="per-family Caps changes for this launch, family.field=value,... (fields "
                   "max_steps, max_tokens_per_turn, max_output_tokens, wall_sec); recorded as config.caps_override, a change on "
                   "--resume as caps_changes; every episode records its own extra.caps")
    p.add_argument("--work-dir", type=Path, default=None, help="default: WORK/<run_id>")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: write logs/ under this root")
    p.add_argument("--serving-gpus", type=int, default=1, help="GPUs serving the policy during the run (resources.csv)")
    p.add_argument("--note", default="", help="free text for metadata.json")
    p.add_argument("--dry-run", action="store_true", help="validate inputs, print the plan, start nothing")
    a = p.parse_args(argv)
    a.tasks_file = a.tasks_file or [f for f in DEFAULT_TASK_FILES if f.exists()]
    if not a.tasks_file:
        p.error("no --tasks-file given and no pool file exists yet")
    a.family_limits = {**{"swe": 64, "terminal": 32}, **parse_kv(a.family_concurrency)}
    a.max_tasks_by_family = parse_kv(a.max_tasks)
    try:
        a.caps_override = parse_caps_override(a.caps_override)
    except ValueError as e:
        p.error(str(e))
    a.caps = {f: dataclasses.replace(caps_for(f), **a.caps_override.get(f, {})) for f in FAMILIES}
    a.stop_at, a.cutoff_at = parse_time(a.stop_launching_at), parse_time(a.cutoff)
    a.load_pause = a.load_pause if a.load_pause is not None else float((os.cpu_count() or 1) + 8)
    return a


def eligibility(t: dict[str, Any], split: str) -> str | None:
    """Why a pool record is not rolled out (None = eligible)."""
    fam = SOURCE_FAMILY.get(t.get("source"))
    if fam is None:
        return f"source_{t.get('source')}_not_a_stage2_pool"
    if t.get("split") != split:
        return f"split_{t.get('split')}"
    extra = t.get("extra") or {}
    if fam == "swe":
        if (extra.get("calibration") or {}).get("status") != "ok":
            return "not_calibrated_ok"
        if not t.get("image") or not mini_swe.problem_statement(t).strip():
            return "missing_image_or_statement"
    elif not (extra.get("env_check") or {}).get("passed"):
        return "env_check_not_passed"
    return None


def load_tasks(a: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    wanted = {ln.strip() for ln in a.task_ids_file.read_text().splitlines() if ln.strip()} if a.task_ids_file else None
    skipped: Counter = Counter()
    seen: set[str] = set()
    per_family: Counter = Counter()
    out: list[dict[str, Any]] = []
    for f in a.tasks_file:
        for d in read_jsonl(f):
            t = task_from_dict(d)   # validates the record
            fam = SOURCE_FAMILY.get(t.source)
            if (a.family and fam != a.family) or (wanted is not None and t.task_id not in wanted):
                continue
            why = eligibility(d, a.split)
            if why:
                skipped[why] += 1
                continue
            if t.task_id in seen:
                raise SystemExit(f"{f}: duplicate task_id {t.task_id}")
            if fam in a.max_tasks_by_family and per_family[fam] >= a.max_tasks_by_family[fam]:
                continue
            seen.add(t.task_id)
            per_family[fam] += 1
            out.append(d)
    if wanted is not None and wanted - seen:
        skipped["requested_but_not_eligible_or_absent"] += len(wanted - seen)
    return out, skipped


def preflight(a: argparse.Namespace, tasks: list[dict[str, Any]], endpoint: Endpoint, log) -> dict[str, Any]:
    """Every problem at once: endpoint, docker, venvs, images, mirrors, task dirs. SWE tasks whose image is missing (and not
    pulled) are reported and removed from the run (counted as skipped), not silently dropped."""
    problems: list[str] = []
    health = endpoint.health()
    if not health["ok"]:
        problems.append(f"endpoint {a.api_base} does not serve {a.model_name!r}: {health.get('error') or health.get('models')}")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        problems.append("docker daemon not reachable")
    fams = {SOURCE_FAMILY[t["source"]] for t in tasks}
    if not mini_swe.HARBOR_PYTHON.exists():
        problems.append(f"{mini_swe.HARBOR_PYTHON} missing (scripts/setup_eval_env.sh)")
    if "terminal" in fams and not terminus.HARBOR.exists():
        problems.append(f"{terminus.HARBOR} missing")
    missing_images: list[str] = []
    for img in sorted({t["image"] for t in tasks if SOURCE_FAMILY[t["source"]] == "swe"}):
        if image_present(img):
            continue
        if a.pull_missing and not a.dry_run:
            wait_for_docker_window(f"docker pull {img}", log)   # no pull while a reportable Terminal-Bench run is active
            log(f"pulling {img}")
            if subprocess.run(["docker", "pull", "-q", img], capture_output=True, check=False).returncode == 0:
                continue
        missing_images.append(img)
    missing_dirs = [t["task_id"] for t in tasks if SOURCE_FAMILY[t["source"]] == "terminal"
                    and not (a.terminal_dataset_dir / t["upstream_id"] / "task.toml").exists()]
    repos = sorted({repo_name_of(t) for t in tasks if SOURCE_FAMILY[t["source"]] == "swe" and t["image"] not in missing_images})
    mirror_errors = {}
    if not a.dry_run:
        for r in repos:
            try:
                ensure_mirror(r, a.mirrors_dir)
            except Exception as e:  # noqa: BLE001
                mirror_errors[r] = str(e)[:300]
    for p in problems:
        log(f"preflight: {p}")
    return {"endpoint": health, "problems": problems, "missing_images": missing_images, "missing_task_dirs": missing_dirs,
            "mirror_errors": mirror_errors, "n_mirrors": len(repos)}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_items(tasks: list[dict[str, Any]], a: argparse.Namespace) -> list[WorkItem]:
    items = []
    for ti, t in enumerate(tasks):
        fam = SOURCE_FAMILY[t["source"]]
        timeout = a.caps[fam].wall_sec + TIMEOUT_GRACE_SEC[fam]
        for k in range(a.n):
            items.append(WorkItem(priority=(k, FAMILY_RANK[fam], ti), key=episode_id(t["task_id"], a.round_, a.policy, k), family=fam,
                                  payload={"task": t, "sample_index": k}, timeout_sec=timeout))
    return items


def summarise(episodes: list[dict[str, Any]], sched: Scheduler, wall_sec: float) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for fam in FAMILIES:
        eps = [e for e in episodes if FAMILY_OF_SCAFFOLD.get(e["scaffold"]) == fam]
        if not eps:
            continue
        by_family[fam] = {"n": len(eps), "exit_reasons": dict(Counter(e["exit_reason"] for e in eps)),
                          "mean_wall_sec": round(sum(e["wall_sec"] for e in eps) / len(eps), 1),
                          "output_tokens": sum(e["output_tokens"] for e in eps),
                          "mean_steps": round(sum(e["steps"] for e in eps) / len(eps), 1),
                          "episodes_per_hour": round(len(eps) / max(wall_sec / 3600, 1e-9), 1),
                          "container_minutes": round(sum(e["wall_sec"] for e in eps) / 60, 1)}
    states = sched.state
    return {"n_episodes": len(episodes), "by_family": by_family, "scheduler_counts": sched.counts(), "cut_off": sched.cut_off,
            "environment_failures_recorded": sum(1 for s in states.values() if s.get("retries_exhausted")),
            "failed_without_record": sorted(k for k, s in states.items() if s.get("status") == "failed"),
            "cut_off_items": sorted(k for k, s in states.items() if s.get("status") == "cut_off"),
            "scheduler_wall_sec": round(wall_sec, 1)}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_rollouts", a.policy, a.round_)
    work_dir = a.work_dir or (paths.WORK / run_id)
    tasks, skipped = load_tasks(a)
    if not tasks:
        raise SystemExit(f"no eligible task (skipped: {dict(skipped)})")
    endpoint = Endpoint(api_base=a.api_base, model_name=a.model_name, api_key_env=a.api_key_env)
    fams = sorted({SOURCE_FAMILY[t["source"]] for t in tasks})
    config = {"round": a.round_, "policy": a.policy, "n_samples": a.n, "split": a.split, "families": fams,
              "caps": {f: dataclasses.asdict(a.caps[f]) for f in fams}, "caps_override": a.caps_override or None,
              "lock_inference": {f: lock_inference(f) for f in fams},
              "seed_rule": f"endpoint.sample_seed(task_id, k) = sha256('{paths.SEED}:<task_id>:<k>')[:8] & 0x7fffffff, sent with every request",
              "limits": {"max_sandboxes": a.max_sandboxes, "per_family": a.family_limits, "load_pause": a.load_pause,
                         "max_env_retries": a.max_env_retries, "tb_container_cap": a.tb_container_cap,
                         "control_file": str(a.control_file) if a.control_file else None},
              "stop_launching_at": a.stop_launching_at, "cutoff": a.cutoff, "max_tasks": a.max_tasks_by_family,
              "mirrors_dir": str(a.mirrors_dir), "terminal_dataset_dir": str(a.terminal_dataset_dir), "work_dir": str(work_dir),
              "cgroup_parent": a.cgroup_parent, "swe_runtime_env": mini_swe.runtime_env(1, "4g"),
              "terminal_start": {"jitter_sec_max": terminus.START_JITTER_SEC, "probe_wait_sec_max": terminus.PROBE_WAIT_SEC,
                                 "probe_fast_sec": terminus.PROBE_FAST_SEC}}
    if a.dry_run:
        pf = preflight(a, tasks, endpoint, print)
        print(json.dumps({"run_id": run_id, "n_tasks": len(tasks), "by_family": dict(Counter(SOURCE_FAMILY[t["source"]] for t in tasks)),
                          "n_episodes": len(tasks) * a.n, "skipped_tasks": dict(skipped), "config": config, "preflight": pf}, indent=1, default=str))
        return 0 if not pf["problems"] else 1

    with RunDir(run_id, root=a.out_root, log_name="train.log", resume=a.resume) as run:
        run.log(f"run_id={run_id} policy={a.policy} round={a.round_} model={a.model_name} tasks={len(tasks)} x {a.n}; skipped {dict(skipped)}")
        pf = preflight(a, tasks, endpoint, run.log)
        if pf["problems"]:
            for p in pf["problems"]:
                run.error(f"preflight: {p}")
            raise SystemExit("preflight failed: " + "; ".join(pf["problems"]))
        unusable = set(pf["missing_images"]) | {r for r in pf["mirror_errors"]}
        dropped_tasks = [t["task_id"] for t in tasks if (SOURCE_FAMILY[t["source"]] == "swe" and (t["image"] in unusable or repo_name_of(t) in unusable))
                         or t["task_id"] in pf["missing_task_dirs"]]
        for tid in dropped_tasks:
            run.error(f"{tid}: not rolled out (image missing, mirror failed or task dir missing); counted in skipped_tasks")
        skipped["preflight_image_mirror_or_dir"] += len(dropped_tasks)
        tasks = [t for t in tasks if t["task_id"] not in set(dropped_tasks)]
        snap = run.run_dir / "config_snapshot"
        agent_cfg_path = snap / "mini_swe_agent_stage2.yaml"
        if "swe" in fams:
            mini_swe.agent_config(agent_cfg_path, caps=a.caps["swe"])
        for f in fams:
            run.snapshot_config(LOCKS[f])
        requests_dir, episodes_dir = work_dir / "requests", work_dir / "episodes"
        requests_dir.mkdir(parents=True, exist_ok=True)
        episodes_dir.mkdir(parents=True, exist_ok=True)
        from apex_eval.common import model_revision  # noqa: PLC0415
        prior = run.read_metadata() if a.resume else {}
        prior_caps = (prior.get("config") or {}).get("caps")
        if prior_caps and prior_caps != config["caps"]:   # a resumed launch with other caps: the change is kept in the record
            change = {"at": utc_now(), "from": prior_caps, "to": config["caps"], "caps_override": a.caps_override or None}
            run.write_metadata(caps_changes=[*(prior.get("caps_changes") or []), change])
            run.log(f"caps changed on resume: {json.dumps(change)}")
        run.write_metadata(run_id=run_id, kind="data_rollouts", step="stage2_rollouts", policy=a.policy, policy_revision=a.policy_revision,
                           model_name=a.model_name, endpoint=pf["endpoint"], base_model_revision=model_revision(), code=git_info(),
                           hardware=hardware_info(), seed=paths.SEED, command=command_line(), note=a.note, config=config,
                           data_version={"tasks_files": [{"path": str(f), "sha256": sha256_file(f)} for f in a.tasks_file],
                                         "n_tasks": len(tasks), "by_family": dict(Counter(SOURCE_FAMILY[t["source"]] for t in tasks)),
                                         "skipped_tasks": dict(skipped)},
                           outputs={"episodes": str(work_dir / "episodes.jsonl"), "requests": str(requests_dir)},
                           start_time=run.read_metadata().get("start_time") or utc_now(), status="running")

        def launcher(it: WorkItem) -> subprocess.Popen:
            t, k = it.payload["task"], it.payload["sample_index"]
            ep_dir = episodes_dir / slug(it.key, 120)
            ep_dir.mkdir(parents=True, exist_ok=True)
            if it.family == "swe":
                drv, spec = mini_swe, mini_swe.build_spec(t, endpoint, round_=a.round_, policy=a.policy, sample_index=k, episode_dir=ep_dir,
                                                          requests_dir=requests_dir, run_id=run_id, agent_config_path=agent_cfg_path,
                                                          docker={"cgroup_parent": a.cgroup_parent}, mirrors_dir=a.mirrors_dir,
                                                          size_runtimes=True, caps=a.caps["swe"])
            else:
                drv, spec = terminus, terminus.build_spec(t, endpoint, round_=a.round_, policy=a.policy, sample_index=k, episode_dir=ep_dir,
                                                          requests_dir=requests_dir, run_id=run_id, dataset_dir=a.terminal_dataset_dir,
                                                          caps=a.caps["terminal"])
            spec_path, out_path = ep_dir / "spec.json", ep_dir / "episode.json"
            spec_path.write_text(json.dumps(spec))
            if out_path.exists():   # an earlier attempt (environment failure): kept next to the new one, never deleted
                out_path.rename(ep_dir / f"episode.attempt{len(list(ep_dir.glob('episode.attempt*.json'))) + 1}.json")
            log = (ep_dir / "worker.log").open("a")
            return subprocess.Popen(drv.worker_command(spec_path, out_path), cwd=ROOT, env=drv.worker_env(), stdout=log, stderr=subprocess.STDOUT)

        def collector(it: WorkItem, rc: int | None) -> Outcome:
            out_path = episodes_dir / slug(it.key, 120) / "episode.json"
            if not out_path.exists():
                return Outcome(record=None, retry=True, note=f"worker exit {rc} without episode.json (see worker.log)")
            ep = json.loads(out_path.read_text())
            extra = ep.get("extra") or {}
            if it.family == "swe":
                transient = ep["exit_reason"] == "error" and any(x in str(extra.get("exit_status") or extra.get("error") or "") for x in TRANSIENT_ERRORS)
            else:
                transient = bool((extra.get("harbor") or {}).get("environment_failure"))
            if ep["exit_reason"] == "error":
                run.error(f"{it.key}: {(extra.get('error') or (extra.get('harbor') or {}).get('exception_message') or '')[:300]}")
            return Outcome(record=ep, retry=transient, note=ep["exit_reason"] + (" (environment)" if transient else ""))

        def on_record(ep: dict[str, Any]) -> None:
            extra = ep.get("extra") or {}
            run.metric({"episode_id": ep["episode_id"], "task_id": ep["task_id"], "scaffold": ep["scaffold"], "exit_reason": ep["exit_reason"],
                        "steps": ep["steps"], "output_tokens": ep["output_tokens"], "wall_sec": ep["wall_sec"], "seed": extra.get("seed"),
                        "patch_bytes": len(ep.get("patch") or ""), "reward": (extra.get("harbor") or {}).get("reward")})

        limits = Limits(global_slots=a.max_sandboxes, per_family=a.family_limits, max_env_retries=a.max_env_retries, load_pause=a.load_pause,
                        tb_container_cap=a.tb_container_cap)
        items = build_items(tasks, a)
        sched = Scheduler(work_dir, items, launcher, collector, limits, records_name="episodes.jsonl", log=run.log, on_record=on_record,
                          stop_launching_at=a.stop_at, deadline=a.cutoff_at, control_file=a.control_file,
                          on_limits=lambda ch: (run.metric({"event": "limits_changed", **ch}), run.write_metadata(limit_changes=sched.limit_changes)))
        t0 = time.time()
        sched.run()
        keys = {it.key for it in items}
        episodes = [e for e in read_jsonl(work_dir / "episodes.jsonl") if e["episode_id"] in keys] if (work_dir / "episodes.jsonl").exists() else []
        summary = summarise(episodes, sched, time.time() - t0)
        (run.run_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=1))
        rates = compute_rates()
        container_min = round(sum(e["wall_sec"] for e in episodes) / 60, 1)
        run.write_resources(resource_rows(run.elapsed(), serving_gpus=a.serving_gpus, gpu_usd_per_hour=rates["gpu_vm_usd_per_hour"], peak_gpu_mem_gb=None, orchestrator_cpus=os.cpu_count())
                            + [("container_minutes", container_min, "min", "sum of episode wall clocks (one sandbox each)"),
                               ("scheduler_wall_sec", summary["scheduler_wall_sec"], "s", "time inside the scheduler loop")]
                            + [(f"{f}_episodes_per_hour", s["episodes_per_hour"], "1/h", "throughput for the Stage 6 plan")
                               for f, s in summary["by_family"].items()])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="cut_off" if sched.cut_off else "finished",
                           summary=summary)
        run.finalize()
        run.log(f"done: {len(episodes)} episodes {json.dumps({f: s['exit_reasons'] for f, s in summary['by_family'].items()})}; "
                f"{container_min} container-minutes; episodes in {work_dir / 'episodes.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
