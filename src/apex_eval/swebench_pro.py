"""Run SWE-bench Pro v2 HARD-51 under the locked protocol against an OpenAI-compatible endpoint.

Usage (from the repository root; or via scripts/evaluate.sh swebench_pro ...):
  PYTHONPATH=src .venv/harbor/bin/python -m apex_eval.swebench_pro --version base \
      --model-name <served model id> --api-base http://<vllm-host>:8000/v1 [--limit N] [--run-id ID --resume]

Inputs : configs/eval/swebench_pro_v2_hard.yaml (lock), configs/eval/swebench_pro_hard51_ids.txt,
         third_party/SWE-bench_Pro-os at the pinned commit (scripts/setup_eval_env.sh), the 51 images,
         and a running OpenAI-compatible server at --api-base.
Outputs: logs/<run_id>/            metadata.json, eval.log, metrics.jsonl (one line per task), errors.log, resources.csv,
                                   config_snapshot/ (lock + generated mini-swe-agent config), harbor/{agent,regrade}*/ (raw harbor jobs)
         results/<version>/swebench_pro/<run_id>/   per_task.jsonl (shared schema, see common.py), summary.json,
                                   tasks/<instance_id>/{model.patch, trajectory.json, agent_result.json, replay.json, verifier outputs}
Protocol: phase 1 = host-side mini-swe-agent in an offline sandbox (agent budget from task.toml); phase 2 = patch_replay re-grades
          every captured model.patch on a pristine image. The re-graded reward is the reported one. Failed tasks are never dropped.
          Lock values are CLI-overridable for pilots only: any deviation is recorded and marks the run as not reportable.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from apex_eval.common import (BUDGET_EXHAUSTED, MISSING, OK, REPO_ROOT, SCORING_FAILURE, RunPaths, add_common_args,
                              checkout_matches, command_line, copy_if_exists, failure_summary, git_info, hardware_info,
                              lock_deviations, model_revision, preflight, read_json, read_task_ids, record, serving_info, utc_now)
from apex_eval.harbor_utils import (HARBOR, classify, consolidate_trials, exception, harbor_run_resilient, job_dirs, load_trials,
                                    retry_args, reward, seconds_between, task_args)

SBP = REPO_ROOT / "third_party" / "SWE-bench_Pro-os"
TASKS = Path("v2/tasks")   # relative to SBP, which is harbor's working directory for this benchmark
LOCK = REPO_ROOT / "configs" / "eval" / "swebench_pro_v2_hard.yaml"
IDS = REPO_ROOT / "configs" / "eval" / "swebench_pro_hard51_ids.txt"
_ID_RE = re.compile(r"^instance_(?P<owner>.+?)__(?P<repo>.+?)-[0-9a-f]{40}")
AGENT_FILES = (("model.patch", "model.patch"), ("result.json", "agent_result.json"), ("mini-swe-agent.trajectory.json", "trajectory.json"))
VERIFIER_FILES = ("reward.txt", "test-stdout.txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    lock = yaml.safe_load(LOCK.read_text())
    retries = lock["evaluator"]["infrastructure_retries"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--task-ids-file", type=Path, default=IDS)
    p.add_argument("--limit", type=int, default=None, help="only the first N task ids (smoke tests; marks the run not reportable)")
    p.add_argument("--n-concurrent", type=int, default=int(lock["evaluator"]["n_concurrent"]), help="lock value; override for pilots only")
    p.add_argument("--step-limit", type=int, default=int(lock["agent"]["step_limit"]), help="lock value; override for pilots only")
    p.add_argument("--max-retries", type=int, default=int(retries["harbor_max_retries"]), help="harbor retries on the lock's exception list")
    p.add_argument("--skip-regrade", action="store_true", help="debugging only; the reported number requires the re-grade")
    a = p.parse_args(argv)
    a.lock = lock
    a.retry_include = list(retries["retry_include"])
    a.rerun_passes = int(retries["rerun_passes"])
    return a


def build_agent_config(lock: dict[str, Any], step_limit: int, out_path: Path) -> dict[str, Any]:
    """The locked mini-swe-agent config with this experiment's sampling settings injected (identical for all versions)."""
    base = yaml.safe_load((REPO_ROOT / lock["agent"]["config_file"]).read_text())
    inf, s = lock["inference"], lock["inference"]["sampling"]
    base["model"]["model_kwargs"] = {
        "temperature": s["temperature"], "top_p": s["top_p"], "presence_penalty": s["presence_penalty"],
        "max_tokens": inf["max_tokens_per_turn"], "seed": inf.get("seed"), "drop_params": True,
        "extra_body": {"top_k": s["top_k"], "min_p": s["min_p"], "repetition_penalty": s["repetition_penalty"]},
    }
    base["agent"]["step_limit"] = step_limit
    base["agent"]["cost_limit"] = 0   # no cost cap for a self-hosted endpoint; the budget is steps + the task.toml timeout
    if lock["agent"].get("wall_time_limit_seconds"):
        base["agent"]["wall_time_limit_seconds"] = int(lock["agent"]["wall_time_limit_seconds"])   # lets the agent stop and save its patch before harbor's hard timeout
    out_path.write_text(yaml.safe_dump(base, sort_keys=False))
    return base


def token_usage(traj_path: Path) -> tuple[int | None, int | None, int]:
    """(input_tokens, output_tokens, n_assistant_turns) summed over the litellm responses recorded in the mini-swe-agent trajectory."""
    traj = read_json(traj_path, {})
    tin = tout = n_turns = 0
    found = False
    for m in traj.get("messages", []):
        if m.get("role") != "assistant":
            continue
        n_turns += 1
        usage = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
        if usage:
            found = True
            tin += int(usage.get("prompt_tokens") or 0)
            tout += int(usage.get("completion_tokens") or 0)
    return (tin, tout, n_turns) if found else (None, None, n_turns)


def collect(paths: RunPaths, task_ids: list[str], skip_regrade: bool) -> list[dict[str, Any]]:
    """One record per task from the agent job(s) and the re-grade job(s); artifacts copied under results/.../tasks/<id>/."""
    agent_trials = load_trials(*job_dirs(paths, "agent"))
    regrade_trials = {} if skip_regrade else load_trials(*job_dirs(paths, "regrade"))
    records: list[dict[str, Any]] = []
    for tid in task_ids:
        m = _ID_RE.match(tid)
        repo = f"{m['owner']}/{m['repo']}" if m else None
        at = agent_trials.get(tid)
        if at is None:
            paths.error(f"{tid}: no agent trial found")
            records.append(record("swebench_pro", tid, group=repo, resolved=False, run_status=MISSING,
                                  error_reason="no harbor trial produced a result for this task"))
            continue
        tdir, d, n_results = at
        exc_t, exc_m = exception(d)
        ag = tdir / "agent"
        ares = read_json(ag / "result.json", {})
        tin, tout, n_turns = token_usage(ag / "mini-swe-agent.trajectory.json")
        ex = d.get("agent_execution") or {}
        patch_bytes = (ag / "model.patch").stat().st_size if (ag / "model.patch").exists() else 0
        out_dir = paths.results_dir / "tasks" / tid
        for src, dst in AGENT_FILES:
            copy_if_exists(ag / src, out_dir / dst)
        copy_if_exists(tdir / "result.json", out_dir / "harbor_agent_result.json")
        for name in VERIFIER_FILES:
            copy_if_exists(tdir / "verifier" / name, out_dir / "in_sandbox_verifier" / name)
        status = classify(exc_t, exc_m)
        if status == OK and ares.get("exit_status") == "LimitsExceeded":   # stopped at step_limit: graded on its patch like a
            status = BUDGET_EXHAUSTED                                       # submission, counted as budget exhausted
        error = f"{exc_t}: {exc_m}" if exc_t else None
        if exc_t:
            paths.error(f"{tid}: agent phase {error}")
        extra: dict[str, Any] = dict(trial_name=tdir.name, n_results=n_results, repo=repo, exit_status=ares.get("exit_status"),
                                     n_calls=ares.get("n_calls"), n_turns=n_turns, patch_bytes=patch_bytes, in_sandbox_reward=reward(d),
                                     agent_started_at=ex.get("started_at"), agent_finished_at=ex.get("finished_at"),
                                     timeout_without_patch=(exc_t == "AgentTimeoutError" and patch_bytes == 0))
        final = extra["in_sandbox_reward"]
        if not skip_regrade:
            rt = regrade_trials.get(tid)
            if rt is None:
                extra.update(regrade_reward=None)
                if status in (OK, BUDGET_EXHAUSTED):
                    status, error = SCORING_FAILURE, "regrade: no trial produced a result"
                    paths.error(f"{tid}: {error}")
                final = None
            else:
                rdir, rd, rn = rt
                rexc_t, rexc_m = exception(rd)
                replay = read_json(rdir / "agent" / "replay.json", {})
                extra.update(regrade_reward=reward(rd), regrade_apply_rc=replay.get("apply_rc"), regrade_n_results=rn)
                copy_if_exists(rdir / "result.json", out_dir / "harbor_regrade_result.json")
                copy_if_exists(rdir / "agent" / "replay.json", out_dir / "replay.json")
                for name in VERIFIER_FILES:
                    copy_if_exists(rdir / "verifier" / name, out_dir / "regrade_verifier" / name)
                if rexc_t:
                    paths.error(f"{tid}: regrade {rexc_t}: {rexc_m}")
                    if status in (OK, BUDGET_EXHAUSTED):
                        status, error = SCORING_FAILURE, f"regrade: {rexc_t}: {rexc_m}"
                final = extra["regrade_reward"]
        records.append(record("swebench_pro", tid, group=repo, resolved=final == 1.0, score=final, max_score=1.0, run_status=status,
                              harness_status=exc_t, error_reason=error, time_sec=seconds_between(ex.get("started_at"), ex.get("finished_at")),
                              input_tokens=tin, output_tokens=tout, artifacts=f"tasks/{tid}", **extra))
    return records


def summarize(records: list[dict[str, Any]], skip_regrade: bool) -> dict[str, Any]:
    n = len(records)
    resolved = sum(1 for r in records if r["resolved"])
    return {
        "reported_metric": "resolved_rate_pct (IN-SANDBOX ONLY, not reportable)" if skip_regrade else "resolved_rate_pct (re-graded on a pristine image)",
        "n_tasks": n, "n_resolved": resolved, "resolved_rate_pct": round(100 * resolved / n, 2) if n else None,
        "in_sandbox_resolved": sum(1 for r in records if r["extra"].get("in_sandbox_reward") == 1.0),
        "timeouts_without_patch": sum(1 for r in records if r["extra"].get("timeout_without_patch")),
        "timeouts_with_in_sandbox_pass": sum(1 for r in records if r["run_status"] == BUDGET_EXHAUSTED and r["extra"].get("in_sandbox_reward") == 1.0),
        "by_repo": {repo: {"n": sum(1 for r in records if r["group"] == repo), "resolved": sum(1 for r in records if r["group"] == repo and r["resolved"])}
                    for repo in sorted({r["group"] for r in records if r["group"]})},
        **failure_summary(records),
    }


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if not HARBOR.exists() or not SBP.exists():
        sys.exit("run scripts/setup_eval_env.sh first (harbor venv or SWE-bench Pro checkout missing)")
    task_ids = read_task_ids(a.task_ids_file, a.limit)
    with RunPaths("swebench_pro", a.version, a.run_id, root=a.out_root, resume=a.resume) as paths:
        paths.log(f"run_id={paths.run_id} version={a.version} model={a.model_name} tasks={len(task_ids)} n_concurrent={a.n_concurrent}")
        ep = preflight(paths, a, docker=True)
        sbp_commit = checkout_matches(paths, "SWE-bench Pro", SBP, a.lock["source"]["commit"])
        lock_copy = paths.snapshot_config(LOCK)
        paths.snapshot_config(a.task_ids_file)
        cfg_path = paths.run_dir / "config_snapshot" / "mini_swe_agent_config.yaml"
        agent_cfg = build_agent_config(a.lock, a.step_limit, cfg_path)
        deviations = lock_deviations({"n_concurrent": a.n_concurrent, "step_limit": a.step_limit, "max_retries": a.max_retries},
                                     {"n_concurrent": a.lock["evaluator"]["n_concurrent"], "step_limit": a.lock["agent"]["step_limit"],
                                      "max_retries": a.lock["evaluator"]["infrastructure_retries"]["harbor_max_retries"]})
        reportable = not deviations and not a.limit and not a.skip_regrade and a.task_ids_file == IDS
        if not reportable:
            paths.log(f"NOT REPORTABLE: deviations={deviations} limit={a.limit} skip_regrade={a.skip_regrade} task_ids_file={a.task_ids_file}")
        paths.write_metadata(
            run_id=paths.run_id, benchmark="swebench_pro", benchmark_version="v2.0.0 HARD-51", version=a.version,
            model_name=a.model_name, endpoint=ep, serving=serving_info(a.serving_info), code=git_info(), hardware=hardware_info(),
            config={"lock_file": str(lock_copy.relative_to(paths.run_dir)), "agent_config": str(cfg_path.relative_to(paths.run_dir)),
                    "step_limit": a.step_limit, "n_concurrent": a.n_concurrent, "max_retries": a.max_retries, "retry_include": a.retry_include,
                    "rerun_passes": a.rerun_passes, "sampling": agent_cfg["model"]["model_kwargs"],
                    "wall_time_limit_seconds": agent_cfg["agent"].get("wall_time_limit_seconds"), "swebench_pro_commit": sbp_commit},
            data_version={"task_ids_file": str(a.task_ids_file), "n_tasks": len(task_ids), "limit": a.limit},
            model_revision=model_revision(), seed=a.lock["inference"].get("seed"), command=command_line(), note=a.note,
            lock_deviations=deviations, reportable=reportable, start_time=utc_now(), status="running",
        )
        env = {**os.environ, "OPENAI_API_BASE": a.api_base, "OPENAI_BASE_URL": a.api_base, "OPENAI_API_KEY": a.api_key,
               "MSWEA_COST_TRACKING": "ignore_errors", "PYTHONPATH": str(SBP / "v2" / "tooling")}
        base_args = ["-n", str(a.n_concurrent)] + retry_args(a.max_retries, a.retry_include)
        agent_args = ["-a", "host_mini_swe:HostMiniSweAgent", "-m", f"openai/{a.model_name}", "--ak", f"config_file={cfg_path}"]
        rcs_agent = harbor_run_resilient(paths, "agent", lambda ids: task_args(TASKS, ids) + base_args + agent_args, env, SBP, task_ids, a.rerun_passes)
        rcs_regrade: list[int] = []
        if not a.skip_regrade:
            source = consolidate_trials(job_dirs(paths, "agent"), paths.run_dir / "harbor" / "agent_consolidated")
            regrade_args = ["-a", "patch_replay:PatchReplayAgent", "-m", "replay", "--ak", f"source_job={source}"]
            rcs_regrade = harbor_run_resilient(paths, "regrade", lambda ids: task_args(TASKS, ids) + base_args + regrade_args, env, SBP, task_ids, a.rerun_passes)
        for name, rcs in (("agent", rcs_agent), ("regrade", rcs_regrade)):
            if any(rc != 0 for rc in rcs):
                paths.error(f"harbor {name} job exit codes {rcs} (per-task exceptions, if any, are recorded above)")
        records = collect(paths, task_ids, a.skip_regrade)
        paths.write_records(records)
        summary = {"run_id": paths.run_id, "benchmark": "swebench_pro", "version": a.version, "model_name": a.model_name,
                   "reportable": reportable, "lock_deviations": deviations, **summarize(records, a.skip_regrade)}
        paths.finish(a, summary, harbor_exit_codes={"agent": rcs_agent, "regrade": rcs_regrade})
        paths.log(f"done: {summary['n_resolved']}/{summary['n_tasks']} resolved ({summary['resolved_rate_pct']}%), "
                  f"{summary['n_run_failures']} run failures, {summary['n_budget_exhausted']} budget exhausted; results in {paths.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
