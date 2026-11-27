"""Run ASI-Bench (60 tasks x the lock's prompt levels; B1 only since 2026-10-12) with the official CLI on seed42 (official,
Portal-scored; the only reportable split).

Usage (from the repository root; or via scripts/evaluate.sh asibench ...):
  PYTHONPATH=src .venv/asibench/bin/python -m apex_eval.asibench --version base \
      --model-name <served model id> --api-base http://<vllm-host>:8000/v1 [--repo seed42|seed31415] [--tasks a.b,c.d] [--prompt-levels b1,b2]

Inputs : configs/eval/asibench.yaml (lock); third_party/ASI-Bench (pinned checkout: CLI + tasks/ scoring contracts);
         third_party/asibench_instances/<repo> (pinned instances); opencode on PATH (~/.npm-global/bin); a running endpoint;
         OPENAI_API_KEY in the environment for the GPT-5.5 judge that 13 task contracts call at local scoring time.
Outputs: logs/<run_id>/            metadata.json, eval.log, metrics.jsonl (one line per task x level), errors.log, resources.csv,
                                   config_snapshot/, asibench_{run,score,report,submit}.log, asibench/run/ (raw produce-only output incl. workspaces)
         results/<version>/asibench/<run_id>/   per_task.jsonl (shared schema, sample = prompt level, group = domain), summary.json,
                                   local_score_seed31415.json, report.csv, report.txt, run_metadata.json, submission_<ts>.tar.gz (seed42),
                                   tasks/<task>/<level>/{result.json, agent_model_output.*, agent_stderr.log, agent_stdout.jsonl, trajectory.json, model_calls.json}
Protocol: produce-only `asibench run` with the opencode_cli adapter in the Docker os sandbox (endpoint via env-named config, so no
          secret on the command line). --repo seed42 (the official split, default): the submission bundle is built with
          `asibench submit --no-upload`; with --submit it is uploaded to the Portal (needs `asibench login` or ASIBENCH_SUBMIT_TOKEN),
          and the official score arrives from the Portal, not from this runner (records: pending_external). --repo seed31415:
          the official local scorer (`asibench score`, explicitly non-official) and `asibench report` run here; the judge key is
          only given to scoring, through a local forwarding proxy (apex_eval.judge_proxy) that removes the `temperature` the
          contracts set for gpt-5.5, which the OpenAI API rejects. Nothing is dropped silently.
          The seed31415 companion runs for stage 2 only, as an unofficial fallback (configs/eval/asibench.yaml
          decision_2026-10-12_companion); a seed31415 run is always marked not reportable, so it is never the ASI-Bench cell.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from apex_eval.common import (AGENT_ERROR, BUDGET_EXHAUSTED, MISSING, OK, PENDING_EXTERNAL, REPO_ROOT, SCORING_FAILURE, SKIPPED,
                              RunPaths, add_common_args, checkout_matches, command_line, copy_if_exists, failure_summary, git_info,
                              hardware_info, lock_deviations, model_revision, preflight, read_json, record, run_logged, serving_info,
                              sha256_file, utc_now)

ASIBENCH = REPO_ROOT / ".venv" / "asibench" / "bin" / "asibench"
ASIBENCH_SRC = REPO_ROOT / "third_party" / "ASI-Bench"
LOCK = REPO_ROOT / "configs" / "eval" / "asibench.yaml"
TASKS_DIR = ASIBENCH_SRC / "tasks"
INSTANCES_ROOT = REPO_ROOT / "third_party" / "asibench_instances"
# The host-side `asibench run` process gets a minimal environment: the sandbox builds its own, and no judge or upload
# credential must be visible to the agent phase.
AGENT_ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL")
AGENT_ENV_PREFIXES = ("DOCKER_", "XDG_", "NPM_")
SIDE_FILES = (".agent_stderr.log", ".agent_stdout.jsonl", ".trajectory.json", ".model_calls.json", ".llm_response.md", ".selected_code.py")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    lock = yaml.safe_load(LOCK.read_text())
    rm = lock["run_mode"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--repo", default=lock["source"]["primary_split"], choices=["seed42", "seed31415"],
                   help="seed42 = official split (Portal-scored, the default); seed31415 = unofficial local companion (stage 2 only), never reportable")
    p.add_argument("--submit", action="store_true", help="seed42 only: upload the submission bundle to the Portal (requires `asibench login` or ASIBENCH_SUBMIT_TOKEN)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids for pilots/smoke tests (default: all 60; marks the run not reportable)")
    p.add_argument("--prompt-levels", default=",".join(lock["task_set"]["prompt_levels"]), help="lock value; a subset is allowed for pilots")
    p.add_argument("--parallel", type=int, default=int(rm["parallel"]), help="lock value; override for pilots only")
    p.add_argument("--timeout", type=int, default=int(rm["timeout_sec_per_instance"]), help="lock value; override for smoke tests only")
    p.add_argument("--tool-mode", default=rm["tool_mode"], choices=["restricted", "search", "unrestricted"], help="lock value")
    p.add_argument("--score-parallel", type=int, default=4, help="scorer workers (seed31415 only; does not affect scores)")
    p.add_argument("--judge-api-base", default=None, help="judge gateway for the local scorer instead of the local proxy (requires the two flags below)")
    p.add_argument("--judge-api-key-env", default=None, help="NAME of the env var holding the judge key (never the key)")
    p.add_argument("--judge-api-protocol", default=None, choices=["native", "openai"])
    p.add_argument("--skip-score", action="store_true", help="debugging only; produce-only run without scoring, report or bundle")
    a = p.parse_args(argv)
    if a.judge_api_base and not (a.judge_api_key_env and a.judge_api_protocol):
        p.error("--judge-api-base requires --judge-api-key-env and --judge-api-protocol (the scorer rejects a partial triple)")
    a.lock = lock
    a.levels = [x.strip() for x in a.prompt_levels.split(",") if x.strip()]
    a.task_list = [x.strip() for x in a.tasks.split(",") if x.strip()] if a.tasks else None
    a.instances = INSTANCES_ROOT / a.repo
    a.local_scoring = a.repo == "seed31415" and not a.skip_score
    return a


def expected_instances(instances: Path, task_list: list[str] | None) -> list[tuple[str, str]]:
    """(task_id, instance_id) pairs from the pulled instance directories, optionally restricted to task_list (which must all exist)."""
    pairs = [(d.name.split("__")[0], d.name) for d in sorted(instances.iterdir()) if d.is_dir() and "__" in d.name]
    if task_list is not None:
        unknown = sorted(set(task_list) - {t for t, _ in pairs})
        if unknown:
            raise SystemExit(f"--tasks names unknown task ids: {unknown}")
        pairs = [(t, i) for t, i in pairs if t in task_list]
    return pairs


class JudgeProxy:
    """The local forwarding proxy for the scorer's GPT-5.5 judge, as a context manager (port chosen by the OS, always stopped)."""

    def __init__(self, paths: RunPaths):
        self.paths = paths
        self.requests_log = paths.run_dir / "judge_proxy_requests.jsonl"
        self.proc: subprocess.Popen | None = None
        self.stdout = None
        self.api_base = ""

    def __enter__(self) -> JudgeProxy:
        with socket.socket() as sck:
            sck.bind(("127.0.0.1", 0))
            port = sck.getsockname()[1]
        self.stdout = (self.paths.run_dir / "judge_proxy.log").open("a")
        self.proc = subprocess.Popen([sys.executable, "-m", "apex_eval.judge_proxy", "--port", str(port), "--log", str(self.requests_log)],
                                     cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                                     stdout=self.stdout, stderr=subprocess.STDOUT)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("judge proxy did not start listening within 15 s (see judge_proxy.log)")
        self.api_base = f"http://127.0.0.1:{port}/v1"
        self.paths.log(f"judge proxy started on port {port} (drops the unsupported temperature for gpt-5.x; log: {self.requests_log.name})")
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.stdout is not None:
            self.stdout.close()
        n_calls = sum(1 for _ in self.requests_log.open()) if self.requests_log.exists() else 0
        self.paths.log(f"judge proxy stopped; {n_calls} judge request(s) forwarded")


def collect(paths: RunPaths, a: argparse.Namespace, run_out: Path, score_file: Path | None) -> list[dict[str, Any]]:
    """One record per (task, level) from the produce-only output and, for the companion split, the local score file."""
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    if score_file and score_file.exists():
        for r in read_json(score_file, {}).get("results", []):
            scores[(r["task_id"], r["prompt_level"])] = r
    pending = a.repo == "seed42" and not a.skip_score
    records = []
    for task_id, inst in expected_instances(a.instances, a.task_list):
        for level in a.levels:
            domain = task_id.split(".")[0]
            res_path = run_out / task_id / f"{inst}__{level}.json"
            d = read_json(res_path)
            if not isinstance(d, dict):
                paths.error(f"{task_id} {level}: no result file")
                records.append(record("asibench", task_id, sample=level, group=domain, run_status=MISSING,
                                      error_reason="no result file produced by asibench run", instance_id=inst))
                continue
            ao = d.get("agent_output") or {}
            cost = d.get("cost") or {}
            s = scores.get((task_id, level))
            scored = bool(s) and s.get("evaluation_status") == "completed" and not s.get("scorer_internal_error")
            extra: dict[str, Any] = dict(instance_id=inst, agent_status=d.get("status"), timeout_phase=ao.get("timeout_phase"),
                                         process_exit_code=ao.get("process_exit_code"),
                                         sandbox_mode=((d.get("provenance") or {}).get("sandbox") or {}).get("effective_mode"),
                                         n_code_files=len(ao.get("code_files") or []), n_data_files=len(ao.get("data_files") or []))
            if s:
                extra.update(evaluation_status=s.get("evaluation_status"), scorer_internal_error=s.get("scorer_internal_error"),
                             hard_gates_passed=s.get("hard_gates_passed"), soft_gate_failures=s.get("soft_gate_failures"),
                             gates=[{"scorer": g.get("scorer_name"), "passed": g.get("passed"), "score": g.get("score"), "max": g.get("max_score")}
                                    for g in (s.get("gate_results") or [])],
                             scorers=[{"scorer": g.get("scorer_name"), "score": g.get("score"), "max": g.get("max_score")}
                                      for g in (s.get("score_results") or [])])
            agent_error = (ao.get("error_message") or "")[:300] or ao.get("timeout_phase")
            if d.get("status") == "timeout":
                status, error = BUDGET_EXHAUSTED, agent_error
            elif d.get("status") != "completed":
                status, error = AGENT_ERROR, agent_error
                paths.error(f"{task_id} {level}: agent {d.get('status')}: {error}")
            elif score_file is None:
                status, error = (PENDING_EXTERNAL, None) if pending else (SKIPPED, "scoring skipped by request")
            elif s is None:
                status, error = SCORING_FAILURE, "no entry in the local score file"
                paths.error(f"{task_id} {level}: {error}")
            elif not scored:
                status, error = SCORING_FAILURE, f"evaluation_status={s.get('evaluation_status')} scorer_internal_error={s.get('scorer_internal_error')}"
                paths.error(f"{task_id} {level}: {error}")
            else:
                status, error = OK, None
            out = paths.results_dir / "tasks" / task_id / level
            copy_if_exists(res_path, out / "result.json")
            for suffix in SIDE_FILES:
                copy_if_exists(run_out / task_id / f"{inst}__{level}{suffix}", out / suffix.lstrip("."))
            for f in (run_out / task_id).glob(f"{inst}__{level}.agent_model_output.*"):
                copy_if_exists(f, out / f.name.split(f"{inst}__{level}.", 1)[1])
            records.append(record("asibench", task_id, sample=level, group=domain, score=s.get("final_score") if scored else None,
                                  max_score=s.get("max_score") if s else None, run_status=status,
                                  harness_status=f"{d.get('status')}/{s.get('evaluation_status') if s else 'unscored'}", error_reason=error,
                                  time_sec=d.get("execution_time_seconds"), input_tokens=cost.get("input_tokens"),
                                  output_tokens=cost.get("output_tokens"), artifacts=f"tasks/{task_id}/{level}", **extra))
    return records


def summarize(records: list[dict[str, Any]], levels: list[str], repo: str, score_file: Path | None) -> dict[str, Any]:
    scored = [r for r in records if r["run_status"] == OK and r["score"] is not None]

    def mean(xs: list[float]) -> float | None:
        return round(statistics.fmean(xs), 2) if xs else None

    official = read_json(score_file, {}) if score_file else {}
    out = {
        "reported_metric": ("official seed42 Portal score (pending; produced by the maintainers, not by this runner)" if repo == "seed42"
                            else "overall_mean_score: unweighted mean of final_score (0-100) over scored (task, level) results; non-official seed31415 local scoring"),
        "n_results_expected": len(records), "n_scored": len(scored),
        "n_agent_completed": sum(1 for r in records if r["extra"].get("agent_status") == "completed"),
        "overall_mean_score": mean([r["score"] for r in scored]),
        "by_prompt_level": {lv: mean([r["score"] for r in scored if r["sample"] == lv]) for lv in levels},
        "by_domain": {dm: mean([r["score"] for r in scored if r["group"] == dm]) for dm in sorted({r["group"] for r in records})},
        "asibench_score_file": {k: official.get(k) for k in ("instance_count", "scored_instance_count", "scorer_error_count", "total_score", "total_max_score", "mean_percent")},
        **failure_summary(records),
    }
    return out


def asibench_cmd(paths: RunPaths, name: str, args: list[str], env: dict[str, str]) -> int:
    return run_logged(paths, name, [str(ASIBENCH), *args], env, REPO_ROOT, log_path=paths.run_dir / f"asibench_{name}.log")


def score_locally(paths: RunPaths, a: argparse.Namespace, run_out: Path, env: dict[str, str], score_file: Path) -> int:
    """`asibench score` on the seed31415 companion, through the judge proxy unless a judge gateway was given."""
    judge_env_name = a.judge_api_key_env or "OPENAI_API_KEY"

    def run(route: dict[str, Any]) -> int:
        cmd = ["score", "--repo", "seed31415", "--results-dir", str(run_out), "--instances-dir", str(a.instances),
               "--tasks-dir", str(TASKS_DIR), "--parallel", str(a.score_parallel), "--output", str(score_file)]
        for flag, key in (("--judge-api-base", "api_base"), ("--judge-api-key-env", "api_key_env"), ("--judge-api-protocol", "api_protocol")):
            if route.get(key):
                cmd += [flag, route[key]]
        paths.write_metadata(judge_route=route)
        return asibench_cmd(paths, "score", cmd, env)

    if a.judge_api_base:
        return run({"api_base": a.judge_api_base, "api_key_env": a.judge_api_key_env, "api_protocol": a.judge_api_protocol})
    with JudgeProxy(paths) as proxy:
        return run({"proxy": "apex_eval.judge_proxy", "api_base": proxy.api_base, "api_key_env": judge_env_name, "api_protocol": "openai",
                    "note": "temperature removed for gpt-5.x because the OpenAI API only accepts the default for that model"})


def build_bundle(paths: RunPaths, a: argparse.Namespace, run_out: Path, env: dict[str, str]) -> tuple[int, Path | None]:
    """`asibench submit` for the official split: bundle only, or uploaded with --submit. The bundle is copied next to the results."""
    cmd = ["submit", "--results-dir", str(run_out), "--benchmark-repo", a.lock["source"]["seed42_dataset"],
           "--note", f"qwen3.5-9b-sft {a.version} {a.model_name} run {paths.run_id}"]
    if not a.submit:
        cmd.append("--no-upload")
    rc = asibench_cmd(paths, "submit", cmd, env)
    if rc != 0:
        paths.error(f"asibench submit exited {rc} ({'upload' if a.submit else 'bundle only'})")
    bundles = sorted(run_out.glob("submission_*.tar.gz"))   # the CLI names it submission_<timestamp>.tar.gz next to the run
    bundle = None
    if bundles:
        bundle = paths.results_dir / bundles[-1].name
        copy_if_exists(bundles[-1], bundle)
    paths.write_metadata(official_submission={
        "bundle": str(bundle) if bundle else None, "bundle_sha256": sha256_file(bundle) if bundle else None,
        "uploaded": bool(a.submit and rc == 0),
        "note": "the official seed42 score is produced by the ASI-Bench Portal after browser confirmation; recorded in results/summary.csv when it arrives"})
    return rc, bundle


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if not ASIBENCH.exists() or not TASKS_DIR.exists() or not a.instances.exists():
        sys.exit("run scripts/setup_eval_env.sh first (asibench venv, ASI-Bench checkout or instances missing)")
    path_env = f"{Path.home() / '.npm-global' / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["PATH"] = path_env
    with RunPaths("asibench", a.version, a.run_id, root=a.out_root, resume=a.resume) as paths:
        if a.resume:
            raise SystemExit("--resume is not implemented for ASI-Bench; start a new run (the CLI's own --resume is not wired in)")
        pairs = expected_instances(a.instances, a.task_list)
        paths.log(f"run_id={paths.run_id} version={a.version} repo={a.repo} model={a.model_name} tasks={len(pairs)} levels={a.levels} "
                  f"parallel={a.parallel} timeout={a.timeout}")
        ep = preflight(paths, a, docker=True, binaries=("opencode",), env_vars=(("OPENAI_API_KEY",) if a.local_scoring and not a.judge_api_base else ()))
        asibench_commit = checkout_matches(paths, "ASI-Bench", ASIBENCH_SRC, a.lock["source"]["commit"])
        lock_copy = paths.snapshot_config(LOCK)
        rm, src = a.lock["run_mode"], a.lock["source"]
        agent_config = {**rm["agent_config"], "model": a.model_name}
        deviations = lock_deviations({"parallel": a.parallel, "timeout": a.timeout, "tool_mode": a.tool_mode, "prompt_levels": a.levels},
                                     {"parallel": rm["parallel"], "timeout": rm["timeout_sec_per_instance"], "tool_mode": rm["tool_mode"],
                                      "prompt_levels": list(a.lock["task_set"]["prompt_levels"])})
        reportable = not deviations and a.task_list is None and not a.skip_score and a.repo == "seed42"
        if not reportable:
            paths.log(f"NOT REPORTABLE: deviations={deviations} tasks={a.task_list} skip_score={a.skip_score} repo={a.repo}")
        paths.write_metadata(
            run_id=paths.run_id, benchmark="asibench", version=a.version, model_name=a.model_name,
            benchmark_version=f"{a.repo} @ {src['seed42_revision'] if a.repo == 'seed42' else src['dataset_revision']}",
            split_role="official (Portal-scored)" if a.repo == "seed42" else "local companion (non-official scoring)",
            endpoint=ep, serving=serving_info(a.serving_info), code=git_info(), hardware=hardware_info(),
            config={"lock_file": str(lock_copy.relative_to(paths.run_dir)), "agent": "opencode_cli", "agent_config": agent_config,
                    "sandbox": rm["sandbox"], "tool_mode": a.tool_mode, "timeout_sec": a.timeout, "parallel": a.parallel, "prompt_levels": a.levels,
                    "asibench_commit": asibench_commit, "sampling": a.lock["inference"]["sampling"], "max_tokens_per_turn": a.lock["inference"]["max_tokens_per_turn"]},
            data_version={"repo": a.repo, "instances_dir": str(a.instances), "instances_revision": (a.instances / ".revision").read_text().strip() if (a.instances / ".revision").exists() else None,
                          "n_tasks": len(pairs), "tasks": a.task_list},
            model_revision=model_revision(), seed=f"none (opencode sends no seed; {a.repo} is the dataset name, not a sampling seed)",
            command=command_line(), note=a.note, lock_deviations=deviations, reportable=reportable, start_time=utc_now(), status="running",
        )
        run_out = paths.run_dir / "asibench" / "run"
        run_out.mkdir(parents=True)
        agent_env = {k: v for k, v in os.environ.items() if k in AGENT_ENV_KEEP or k.startswith(AGENT_ENV_PREFIXES)}
        agent_env.update(EVAL_API_BASE=a.api_base, EVAL_API_KEY=a.api_key)
        cmd = ["run", "--instances-dir", str(a.instances), "--tasks-dir", str(TASKS_DIR), "--prompt-levels", ",".join(a.levels),
               "--sandbox", rm["sandbox"], "--agent", "opencode_cli", "--agent-config", json.dumps(agent_config), "--tool-mode", a.tool_mode,
               "--timeout", str(a.timeout), "--parallel", str(a.parallel), "--output-dir", str(run_out)]
        if a.task_list:
            cmd += ["--tasks", ",".join(a.task_list)]
        exit_codes: dict[str, int | None] = {"run": asibench_cmd(paths, "run", cmd, agent_env), "score": None, "report": None, "submit": None}
        if exit_codes["run"] != 0:
            paths.error(f"asibench run exited {exit_codes['run']}")
        host_env = dict(os.environ)
        score_file = None
        bundle = None
        if a.local_scoring:
            score_file = paths.results_dir / "local_score_seed31415.json"
            exit_codes["score"] = score_locally(paths, a, run_out, host_env, score_file)
            if exit_codes["score"] != 0:
                paths.error(f"asibench score exited {exit_codes['score']}")
        if not a.skip_score:
            exit_codes["report"] = asibench_cmd(paths, "report", ["report", "--results-dir", str(run_out), "--per-task", "--csv",
                                                                   str(paths.results_dir / "report.csv")], host_env)
            copy_if_exists(paths.run_dir / "asibench_report.log", paths.results_dir / "report.txt")
            if a.repo == "seed42":
                exit_codes["submit"], bundle = build_bundle(paths, a, run_out, host_env)
        copy_if_exists(run_out / "run_metadata.json", paths.results_dir / "run_metadata.json")
        records = collect(paths, a, run_out, score_file)
        paths.write_records(records)
        summary = {"run_id": paths.run_id, "benchmark": "asibench", "split": a.repo, "version": a.version, "model_name": a.model_name,
                   "reportable": reportable, "lock_deviations": deviations, **summarize(records, a.levels, a.repo, score_file)}
        paths.finish(a, summary, exit_codes=exit_codes)
        if a.repo == "seed42":
            paths.log(f"done: {summary['n_agent_completed']}/{summary['n_results_expected']} agent runs completed, {summary['n_run_failures']} run failures; "
                      f"bundle {bundle}; official score pending from the Portal; results in {paths.results_dir}")
        else:
            paths.log(f"done: overall mean {summary['overall_mean_score']} over {summary['n_scored']}/{summary['n_results_expected']} scored results; "
                      f"{summary['n_run_failures']} run failures; results in {paths.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
