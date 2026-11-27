#!/usr/bin/env python
"""Grade stage-2 rollout episodes into schema.Verdict records (docs/DATA_PLAN.md sections 6 and 11: the reward), with the
pass@k histogram per pool. Runs after the rollouts or alongside them (--follow).

Usage (from the repository root):
    .venv/train/bin/python scripts/data/grade_episodes.py --episodes WORK/<rollout_run>/episodes.jsonl [--episodes ...] \
        [--tasks-file data/pools/swesmith/tasks.jsonl --tasks-file data/pools/tgr_env/tasks.jsonl] [--follow] \
        [--slots 32] [--cpus-per-slot 2] [--memory-per-slot 8g] [--cutoff ISO] [--run-id ID --resume] [--dry-run]

Verifiers (src/apex_data/grade/, imported, never modified here):
    mini_swe_toolcall  grade.swesmith (worker subprocess `python -m apex_data.grade.swesmith grade`): the agent's non-test
                       diff replayed on a pristine container of the task image with the harness-side tests restored; r_pass
                       = every FAIL_TO_PASS passes and every graded PASS_TO_PASS still passes, in two runs (the second only
                       after a first pass); tamper flags (tests/CI/test-config edits, skip markers, empty diff, no submit)
                       force 0. Containers `s2swesmith_*`, 2 CPUs / 8 GB, --network none.
    terminus2          grade.harbor_tests.grade_trial on the episode's Harbor trial directory (in process, no container):
                       the in-trial tests/test.sh reward plus the task_complete handshake and a consistent verifier
                       report; tamper flags force 0. An episode without a trial (environment error) gets r_pass 0 with
                       extra.no_trial.
Every episode gets a verdict (failed and errored episodes included; nothing is dropped). Sandbox failures during a replay
are retried once (--max-env-retries) and then recorded with extra.env_failure.
--follow keeps polling the episodes files and grades new episodes until every rollout run named by them has finished
(its logs/<run_id>/metadata.json status is no longer "running"); --cutoff stops at that time (drop order D: what is
graded by then is the set; build_stage2_set.py --cutoff applies the same time to graded_at).

Inputs : episodes.jsonl files from scripts/data/run_rollouts.py; the pool task files (task_id lookup, calibration).
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl (one line per verdict), errors.log, resources.csv,
         pass_at_k.json}; WORK/<run_id>/verdicts.jsonl (schema.Verdict, append-only, the resume point), state.json,
         grades/<episode>/{episode.json, task.json, verdict.json, worker.log, run1/, run2/}.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.grade import harbor_tests  # noqa: E402
from apex_data.grade.swesmith import DEFAULT_MIRRORS_DIR, VERIFIER_VERSION as SWESMITH_VERSION, slug  # noqa: E402
from apex_data.rollout.endpoint import FAMILY_OF_SCAFFOLD  # noqa: E402
from apex_data.rollout.scheduler import Limits, Outcome, Scheduler, WorkItem, parse_time  # noqa: E402
from apex_data.schema import Verdict, read_jsonl  # noqa: E402
from apex_eval.common import RunDir, command_line, compute_rates, git_info, hardware_info, new_run_id, read_json, resource_rows, utc_now  # noqa: E402

DEFAULT_TASK_FILES = (paths.POOLS / "swesmith" / "tasks.jsonl", paths.POOLS / "tgr_env" / "tasks.jsonl")
GRADE_TIMEOUT_SEC = 3 * 3600
TERMINAL_ONE = "_grade-terminal-episode"     # internal worker form: <script> _grade-terminal-episode EPISODE.json VERDICT.json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, action="append", required=True, help="episodes.jsonl of a rollout run (repeatable)")
    p.add_argument("--tasks-file", type=Path, action="append", default=None, help="schema.Task JSONL (repeatable; default: both pools)")
    p.add_argument("--family", choices=["swe", "terminal"], default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N episodes (smoke tests)")
    p.add_argument("--follow", action="store_true", help="keep grading new episodes until the rollout runs have finished")
    p.add_argument("--poll-sec", type=float, default=60.0, help="--follow: seconds between reads of the episodes files")
    p.add_argument("--rollout-logs-root", type=Path, default=paths.LOGS, help="--follow: logs/ directory of the rollout runs")
    p.add_argument("--slots", type=int, default=32, help="concurrent SWE replays (plan: 32 to 40 grading slots)")
    p.add_argument("--cpus-per-slot", type=float, default=2.0)
    p.add_argument("--memory-per-slot", default="8g")
    p.add_argument("--cgroup-parent", default=None, help="systemd slice for the grading containers")
    p.add_argument("--network", default="none", help="docker network of the replays (none = hermetic)")
    p.add_argument("--mirrors-dir", type=Path, default=DEFAULT_MIRRORS_DIR)
    p.add_argument("--no-second-run", action="store_true", help="debugging only: one-run verdicts are not positives")
    p.add_argument("--max-env-retries", type=int, default=1, help="re-runs of a replay whose sandbox failed (plan: once)")
    p.add_argument("--load-pause", type=float, default=None, help="pause new replays above this 1-min load average (default cpu_count + 8)")
    p.add_argument("--control-file", type=Path, default=None, help="JSON file re-read every 30 s while running: pause_launches, load_pause, tb_container_cap, max_sandboxes, per_family overlay the flags (removing it restores them); running episodes are never touched; every change is logged and recorded")
    p.add_argument("--tb-container-cap", type=int, default=None, help="while a reportable Terminal-Bench run is active "
                   "(logs/coordination/terminal_bench_active.json), start no replay once the host runs this many containers (default off)")
    p.add_argument("--cutoff", default=None, help="ISO time (UTC): stop grading at it (drop order D)")
    p.add_argument("--run-id", default=None, help="default: data_grade_<UTC timestamp>")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--work-dir", type=Path, default=None, help="default: WORK/<run_id>")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: logs/ under this root")
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan")
    a = p.parse_args(argv)
    a.tasks_file = a.tasks_file or [f for f in DEFAULT_TASK_FILES if f.exists()]
    a.cutoff_at = parse_time(a.cutoff)
    return a


def family_of(ep: dict[str, Any]) -> str:
    return FAMILY_OF_SCAFFOLD.get(ep.get("scaffold"), str(ep.get("scaffold")))


def grade_terminal(ep: dict[str, Any], graded_at: str) -> Verdict:
    """harbor_tests.grade_trial on the episode's trial directory; an episode without one gets r_pass 0 (no_trial)."""
    trial = ((ep.get("extra") or {}).get("harbor") or {}).get("trial_dir")
    if trial and (Path(trial) / "result.json").exists():
        v = harbor_tests.grade_trial(trial, episode_id=ep["episode_id"], task_id=ep["task_id"], graded_at=graded_at)
        v.extra["episode_exit_reason"] = ep.get("exit_reason")
        return v
    v = Verdict(episode_id=ep["episode_id"], task_id=ep["task_id"], r_pass=0, runs=[], tamper_flags=[], verifier="harbor_tests",
                verifier_version=harbor_tests.VERIFIER_VERSION, graded_at=graded_at,
                extra={"no_trial": True, "episode_exit_reason": ep.get("exit_reason"),
                       "env_failure": bool(((ep.get("extra") or {}).get("harbor") or {}).get("environment_failure")),
                       "skipped_reason": "no Harbor trial directory (the episode errored before or during the trial)"})
    v.validate()
    return v


def pass_at_k(episodes: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per pool (task source): tasks by number of passing samples among their graded samples, plus coverage counts.
    A pass = r_pass 1 (tamper flags already force 0)."""
    graded: dict[str, list[int]] = defaultdict(list)
    for e in episodes:
        v = verdicts.get(e["episode_id"])
        if v is not None:
            graded[e["task_id"]].append(int(v["r_pass"]))
    out: dict[str, Any] = {}
    for source in sorted({tasks.get(t, {}).get("source", t.split(":", 1)[0]) for t in graded}):
        ids = [t for t in graded if tasks.get(t, {}).get("source", t.split(":", 1)[0]) == source]
        k_max = max(len(graded[t]) for t in ids)
        hist = Counter(sum(graded[t]) for t in ids)
        out[source] = {"n_tasks": len(ids), "samples_per_task": dict(sorted(Counter(len(graded[t]) for t in ids).items())),
                       "histogram": {str(k): hist.get(k, 0) for k in range(k_max + 1)},
                       "tasks_with_a_pass": sum(1 for t in ids if sum(graded[t]) > 0),
                       "n_episodes": sum(len(graded[t]) for t in ids), "n_pass": sum(sum(graded[t]) for t in ids)}
    return out


class EpisodeFeed:
    """Episodes of the given files not yet seen (incremental reads for --follow)."""

    def __init__(self, files: list[Path], family: str | None, limit: int | None, logs_root: Path = paths.LOGS):
        self.files, self.family, self.limit, self.logs_root = files, family, limit, logs_root
        self.offsets = {f: 0 for f in files}
        self.episodes: dict[str, dict[str, Any]] = {}

    def read_new(self) -> list[dict[str, Any]]:
        new = []
        for f in self.files:
            if not f.exists():
                continue
            with f.open("r", encoding="utf-8") as fh:
                fh.seek(self.offsets[f])
                while True:
                    pos = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):     # a line still being written
                        fh.seek(pos)
                        break
                    if line.strip():
                        e = json.loads(line)
                        if (self.family is None or family_of(e) == self.family) and e["episode_id"] not in self.episodes:
                            if self.limit is not None and len(self.episodes) >= self.limit:
                                continue
                            self.episodes[e["episode_id"]] = e
                            new.append(e)
                self.offsets[f] = fh.tell()
        return new

    def rollouts_finished(self) -> bool:
        """Every rollout run whose work dir holds one of the files has left status 'running' (logs/<run_id>/metadata.json;
        a run without metadata there, e.g. one written with --out-root, counts as finished)."""
        for f in self.files:
            meta = read_json(self.logs_root / f.parent.name / "metadata.json", {}) or {}
            if meta.get("status") == "running":
                return False
        return True


def grade_terminal_file(episode_path: Path, out_path: Path) -> int:
    v = grade_terminal(json.loads(episode_path.read_text()), utc_now())
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(v), ensure_ascii=False))
    tmp.replace(out_path)
    print(json.dumps({"episode_id": v.episode_id, "r_pass": v.r_pass, "tamper_flags": v.tamper_flags}))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [TERMINAL_ONE]:
        return grade_terminal_file(Path(argv[1]), Path(argv[2]))
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_grade")
    work_dir = a.work_dir or (paths.WORK / run_id)
    tasks = {t["task_id"]: t for f in a.tasks_file for t in read_jsonl(f)}
    feed = EpisodeFeed(a.episodes, a.family, a.limit, a.rollout_logs_root)
    first = feed.read_new()
    missing = sorted({e["task_id"] for e in first if e["task_id"] not in tasks})
    if missing:
        raise SystemExit(f"{len(missing)} episode task_id(s) not in the task files: {missing[:5]}")
    if a.dry_run:
        print(json.dumps({"run_id": run_id, "work_dir": str(work_dir), "n_episodes_now": len(first),
                          "by_family": dict(Counter(family_of(e) for e in first)), "slots": a.slots, "cpus_per_slot": a.cpus_per_slot,
                          "second_run": not a.no_second_run, "follow": a.follow, "cutoff": a.cutoff}, indent=1))
        return 0

    with RunDir(run_id, root=a.out_root, log_name="train.log", resume=a.resume) as run:
        grades_dir, verdicts_path = work_dir / "grades", work_dir / "verdicts.jsonl"
        grades_dir.mkdir(parents=True, exist_ok=True)
        run.log(f"run_id={run_id} episodes now={len(first)} follow={a.follow} slots={a.slots}x{a.cpus_per_slot}cpu cutoff={a.cutoff}")
        run.write_metadata(run_id=run_id, kind="data_grade", step="stage2_grading", code=git_info(), hardware=hardware_info(), seed=paths.SEED,
                           command=command_line(), note=a.note,
                           config={"slots": a.slots, "cpus_per_slot": a.cpus_per_slot, "memory_per_slot": a.memory_per_slot,
                                   "cgroup_parent": a.cgroup_parent, "network": a.network, "second_run": not a.no_second_run,
                                   "max_env_retries": a.max_env_retries, "follow": a.follow, "cutoff": a.cutoff, "work_dir": str(work_dir),
                                   "tb_container_cap": a.tb_container_cap, "control_file": str(a.control_file) if a.control_file else None,
                                   "verifiers": {"swe": SWESMITH_VERSION, "terminal": harbor_tests.VERIFIER_VERSION}},
                           data_version={"episodes_files": [str(p) for p in a.episodes], "tasks_files": [str(p) for p in a.tasks_file]},
                           outputs={"verdicts": str(verdicts_path)}, start_time=run.read_metadata().get("start_time") or utc_now(), status="running")

        def on_record(v: dict[str, Any]) -> None:
            run.metric({"episode_id": v["episode_id"], "task_id": v["task_id"], "r_pass": v["r_pass"], "verifier": v["verifier"],
                        "tamper_flags": v["tamper_flags"], "n_runs": len(v["runs"]), "env_failure": bool((v.get("extra") or {}).get("env_failure")),
                        "seconds": round(sum(float(r.get("seconds") or 0) for r in v["runs"]), 1)})

        def to_items(eps: list[dict[str, Any]]) -> list[WorkItem]:
            items = []
            for e in eps:
                if e["task_id"] not in tasks:
                    run.error(f"{e['episode_id']}: task {e['task_id']} not in the task files; not graded")
                    continue
                items.append(WorkItem(priority=(int(e.get("sample_index", 0)), len(items)), key=e["episode_id"], family="grade" if family_of(e) == "swe" else "trial",
                                      payload={"episode": e, "task": tasks[e["task_id"]]}, timeout_sec=GRADE_TIMEOUT_SEC))
            return items

        def launcher(it: WorkItem) -> subprocess.Popen:
            e, t = it.payload["episode"], it.payload["task"]
            d = grades_dir / slug(it.key, 120)
            d.mkdir(parents=True, exist_ok=True)
            (d / "episode.json").write_text(json.dumps(e, ensure_ascii=False))
            (d / "task.json").write_text(json.dumps(t, ensure_ascii=False))
            if (d / "verdict.json").exists():   # an earlier attempt: kept, never deleted
                (d / "verdict.json").rename(d / f"verdict.attempt{len(list(d.glob('verdict.attempt*.json'))) + 1}.json")
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            if it.family == "trial":    # terminal: read the trial directory in a short subprocess (uniform scheduling)
                cmd = [sys.executable, str(Path(__file__).resolve()), TERMINAL_ONE, str(d / "episode.json"), str(d / "verdict.json")]
            else:
                cmd = [sys.executable, "-m", "apex_data.grade.swesmith", "grade", "--episode", str(d / "episode.json"), "--task", str(d / "task.json"),
                       "--out", str(d / "verdict.json"), "--work-dir", str(d), "--cpus", str(a.cpus_per_slot), "--memory", a.memory_per_slot,
                       "--network", a.network, "--mirrors-dir", str(a.mirrors_dir), "--run-id", run_id]
                if a.cgroup_parent:
                    cmd += ["--cgroup-parent", a.cgroup_parent]
                if a.no_second_run:
                    cmd.append("--no-second-run")
            return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=(d / "worker.log").open("a"), stderr=subprocess.STDOUT)

        def collector(it: WorkItem, rc: int | None) -> Outcome:
            path = grades_dir / slug(it.key, 120) / "verdict.json"
            if not path.exists():
                return Outcome(record=None, retry=True, note=f"grader exit {rc} without verdict.json (see worker.log)")
            v = json.loads(path.read_text())
            v["graded_at"] = v.get("graded_at") or utc_now()
            if any(r.get("retryable") for r in v.get("runs", [])):   # the sandbox, not the patch: retry; recorded when retries run out
                v.setdefault("extra", {})["env_failure"] = True
                return Outcome(record=v, retry=True, note="sandbox failure: " + str([r.get("error") for r in v["runs"] if r.get("error")])[:200])
            return Outcome(record=v, note=f"r_pass={v['r_pass']}")

        limits = Limits(global_slots=a.slots + 8, per_family={"grade": a.slots, "trial": 8}, max_env_retries=a.max_env_retries,
                        load_pause=a.load_pause if a.load_pause is not None else (os.cpu_count() or 1) + 8, poll_sec=0.5,
                        tb_container_cap=a.tb_container_cap)
        sched = Scheduler(work_dir, to_items(first), launcher, collector, limits, records_name="verdicts.jsonl", log=run.log, on_record=on_record,
                          deadline=a.cutoff_at, feeder=(lambda: to_items(feed.read_new())) if a.follow else None,
                          feeder_done=feed.rollouts_finished if a.follow else None, feed_every_sec=a.poll_sec, control_file=a.control_file,
                          on_limits=lambda ch: (run.metric({"event": "limits_changed", **ch}), run.write_metadata(limit_changes=sched.limit_changes)))
        t0 = time.time()
        sched.run()
        if a.follow:
            feed.read_new()
        all_eps = list(feed.episodes.values())
        verdicts = {v["episode_id"]: v for v in read_jsonl(verdicts_path)} if verdicts_path.exists() else {}
        mine = {k: v for k, v in verdicts.items() if k in feed.episodes}
        hist = pass_at_k(all_eps, mine, tasks)
        (run.run_dir / "pass_at_k.json").write_text(json.dumps(hist, indent=1))
        flags = Counter(f for v in mine.values() for f in v["tamper_flags"])
        flaky = sum(1 for v in mine.values() if len(v["runs"]) == 2 and v["runs"][0].get("pass") and not v["runs"][1].get("pass"))
        summary = {"n_episodes": len(all_eps), "n_verdicts": len(mine), "n_pass": sum(v["r_pass"] for v in mine.values()),
                   "by_verifier": {ver: {"n": sum(1 for v in mine.values() if v["verifier"] == ver),
                                         "r_pass": sum(v["r_pass"] for v in mine.values() if v["verifier"] == ver)}
                                   for ver in sorted({v["verifier"] for v in mine.values()})},
                   "tamper_flags": dict(flags), "first_pass_second_fail": flaky,
                   "environment_failures": sum(1 for v in mine.values() if (v.get("extra") or {}).get("env_failure")),
                   "ungraded": sorted(set(feed.episodes) - set(mine)), "cut_off": sched.cut_off, "pass_at_k": hist,
                   "scheduler_counts": sched.counts()}
        run_minutes = round(sum(float(r.get("seconds") or 0) for v in mine.values() for r in v["runs"]) / 60, 1)
        rates = compute_rates()
        run.write_resources(resource_rows(run.elapsed(), serving_gpus=0, gpu_usd_per_hour=rates["gpu_vm_usd_per_hour"], peak_gpu_mem_gb=None, orchestrator_cpus=os.cpu_count())
                            + [("container_minutes", run_minutes, "min", "sum of replay wall clocks across verdict runs"),
                               ("scheduler_wall_sec", round(time.time() - t0, 1), "s", "time inside the scheduler loop")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="cut_off" if sched.cut_off else "finished",
                           summary=summary)
        run.finalize()
        run.log(f"done: {summary['n_pass']}/{summary['n_verdicts']} r_pass=1; pass@k {json.dumps({s: h['histogram'] for s, h in hist.items()})}; "
                f"flags {dict(flags)}; env failures {summary['environment_failures']}; verdicts in {verdicts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
