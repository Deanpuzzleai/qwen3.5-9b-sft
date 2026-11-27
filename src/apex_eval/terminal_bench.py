"""Run Terminal-Bench 2.0 with Harbor + Terminus-2 against an OpenAI-compatible endpoint, the lock's seeds per model version.

Usage (from the repository root; or via scripts/evaluate.sh terminal_bench ...):
  PYTHONPATH=src .venv/harbor/bin/python -m apex_eval.terminal_bench --version base \
      --model-name <served model id> --api-base http://<vllm-host>:8000/v1 [--limit N] [--seeds 1 2] [--run-id ID --resume]

Inputs : configs/eval/terminal_bench_2.yaml (lock), configs/eval/terminal_bench_task_ids.txt, the pinned dataset under
         third_party/terminal-bench/terminal-bench and its 89 images (scripts/setup_eval_env.sh), a running endpoint at --api-base.
Outputs: logs/<run_id>/            metadata.json, eval.log, metrics.jsonl (one line per task x seed), errors.log, resources.csv,
                                   config_snapshot/ (lock, task list, generated harbor job config per seed), harbor/seed_<s>*/ (raw jobs)
         results/<version>/terminal_bench/<run_id>/   per_task.jsonl (shared schema, sample = seed_<s>), per_seed.json, summary.json,
                                   tasks/<task>/seed_<s>/{trajectory.json, harbor_result.json, verifier/...}
Protocol: Terminal-Bench 2.0 leaderboard rules (lock decision_2026-10-12): each task's own task.toml agent timeout and
          resources, no overrides, timeout_multiplier 1.0; temperature/top_p/top_k/max_tokens and a per-request seed passed
          to Terminus-2; Terminus-2 defaults otherwise; same seeds for every version. The lock can still carry an agent
          timeout or resource override (null = none); a set value is passed to Harbor through the job config.
          Reported metric = mean and sample std of the per-seed success rate. An agent that uses its whole task timeout is
          still graded (budget_exhausted, not a run failure); environment and harness failures count as 0 and are listed.
          Tasks the lock lists as unverifiable on this date (harness_check_*.verifier_failures) are recorded as
          environment failures with resolved = false for every version, whatever their verifier returned (the raw
          reward is kept in `extra`), so the denominator is unchanged and the run-failure table shows them; tasks whose
          reference solution is stale (reference_solution_failures) only carry a `lock_note` in `extra`.
          Lock values are CLI-overridable for pilots only: any deviation is recorded and marks the run as not reportable.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from apex_eval.common import (ENVIRONMENT_FAILURE, MISSING, REPO_ROOT, RunPaths, add_common_args, command_line, copy_if_exists, failure_summary,
                              git_info, hardware_info, lock_deviations, model_revision, preflight, read_task_ids, record,
                              serving_info, utc_now, write_json)
from apex_eval.harbor_utils import (HARBOR, agent_tokens, classify, exception, harbor_run_resilient, job_dirs, load_trials,
                                    lock_known_failures, retry_args, reward, seconds_between, task_args)

LOCK = REPO_ROOT / "configs" / "eval" / "terminal_bench_2.yaml"
IDS = REPO_ROOT / "configs" / "eval" / "terminal_bench_task_ids.txt"
VERIFIER_FILES = ("reward.txt", "test-stdout.txt", "output.json")


def opt(value: Any, cast: type) -> Any:
    """A lock value that may be null (null = no override)."""
    return None if value is None else cast(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    lock = yaml.safe_load(LOCK.read_text())
    res = lock["evaluator"]["resources_per_task"]
    retries = lock["evaluator"]["infrastructure_retries"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--task-ids-file", type=Path, default=IDS)
    p.add_argument("--limit", type=int, default=None, help="only the first N task ids (smoke tests / pilots; marks the run not reportable)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(lock["protocol"]["seeds"]), help="lock value; a subset is allowed for pilots")
    p.add_argument("--n-concurrent", type=int, default=int(lock["evaluator"]["n_concurrent"]), help="lock value; override for pilots only")
    p.add_argument("--agent-timeout-sec", type=float, default=opt(lock["agent"]["agent_timeout_sec"], float),
                   help="lock value (default: none, each task.toml timeout applies); override for smoke tests only")
    p.add_argument("--override-cpus", type=int, default=opt(res["override_cpus"], int), help="lock value (default: none, task.toml cpus)")
    p.add_argument("--override-memory-mb", type=int, default=opt(res["override_memory_mb"], int), help="lock value (default: none, task.toml memory)")
    p.add_argument("--max-retries", type=int, default=int(retries["harbor_max_retries"]), help="harbor retries on the lock's exception list")
    a = p.parse_args(argv)
    a.lock = lock
    a.retry_include = list(retries["retry_include"])
    a.rerun_passes = int(retries["rerun_passes"])
    a.dataset = REPO_ROOT / lock["source"]["local_path"]
    a.unverifiable, a.stale_reference = lock_known_failures(lock)   # {task: cause}; see the module docstring
    return a


def job_config(a: argparse.Namespace, seed: int | None) -> dict[str, Any]:
    """Harbor JobConfig fragment carrying everything the CLI cannot express (agent kwargs, and any agent timeout or resource
    override the lock sets; with the leaderboard setting there are none and each task.toml applies)."""
    s, agent = a.lock["inference"]["sampling"], a.lock["agent"]
    llm_kwargs: dict[str, Any] = {"max_tokens": int(a.lock["inference"]["max_tokens_per_turn"]), "top_p": s["top_p"],
                                  "extra_body": {"top_k": s["top_k"], "min_p": s["min_p"]}}
    if seed is not None:
        llm_kwargs["seed"] = seed
    agent_cfg: dict[str, Any] = {
        "name": "terminus-2",
        "model_name": f"openai/{a.model_name}",
        "kwargs": {
            "api_base": a.api_base,
            "model_info": agent["model_info"],            # context window / output limit declared to Terminus-2
            "parser_name": agent["parser"],              # Terminus-2 defaults, passed explicitly so the protocol does not depend on them
            "enable_summarize": bool(agent["context_summarization"]),
            "max_turns": agent["max_turns"],
            "temperature": s["temperature"],
            "llm_call_kwargs": llm_kwargs,
        },
    }
    if a.agent_timeout_sec is not None:
        agent_cfg["override_timeout_sec"] = a.agent_timeout_sec
    cfg: dict[str, Any] = {"agents": [agent_cfg]}
    environment = {k: v for k, v in (("override_cpus", a.override_cpus), ("override_memory_mb", a.override_memory_mb)) if v is not None}
    if environment:
        cfg["environment"] = environment
    return cfg


def run_seed(paths: RunPaths, a: argparse.Namespace, env: dict[str, str], seed: int, task_ids: list[str]) -> list[int]:
    """One harbor job (plus re-runs of pending trials) for one seed; returns the harbor exit codes."""
    cfg_path = paths.run_dir / "config_snapshot" / f"harbor_job_seed_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(job_config(a, seed), sort_keys=False))
    base_args = ["-c", str(cfg_path), "-n", str(a.n_concurrent)] + retry_args(a.max_retries, a.retry_include)
    return harbor_run_resilient(paths, f"seed_{seed}", lambda ids: base_args + task_args(a.dataset, ids), env, REPO_ROOT, task_ids, a.rerun_passes)


def collect_seed(paths: RunPaths, seed: int, task_ids: list[str], unverifiable: dict[str, str] | None = None,
                 stale_reference: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One record per task for one seed; trial artifacts copied under results/.../tasks/<task>/seed_<s>/. `unverifiable`
    and `stale_reference` are the lock's drift lists (harbor_utils.lock_known_failures); an unverifiable task is recorded
    as an environment failure with resolved = false whatever its verifier returned (lock handling rule)."""
    unverifiable, stale_reference = unverifiable or {}, stale_reference or {}
    trials = load_trials(*job_dirs(paths, f"seed_{seed}"))
    sample = f"seed_{seed}"
    records = []
    for tid in task_ids:
        t = trials.get(tid)
        if t is None:
            paths.error(f"{sample} {tid}: no trial found")
            records.append(record("terminal_bench", tid, sample=sample, resolved=False, run_status=MISSING,
                                  error_reason="no harbor trial produced a result for this task"))
            continue
        tdir, d, n_results = t
        exc_t, exc_m = exception(d)
        tin, tout = agent_tokens(d)
        ex = d.get("agent_execution") or {}
        r = reward(d)
        if exc_t:
            paths.error(f"{sample} {tid}: {exc_t}: {exc_m}")
        status, harness_status, reason = classify(exc_t, exc_m), exc_t, (f"{exc_t}: {exc_m}" if exc_t else None)
        resolved, score, extra = r == 1.0, r, {}
        if tid in stale_reference:
            extra["lock_note"] = f"reference solution stale, task and verifier intact ({stale_reference[tid]})"
        if tid in unverifiable:   # lock handling rule: unresolved for every version, listed as an environment failure
            status, harness_status = ENVIRONMENT_FAILURE, exc_t or "lock:verifier_failures"
            reason = f"verifier cannot run on this date ({unverifiable[tid]})" + (f"; harbor: {exc_t}: {exc_m}" if exc_t else "")
            resolved, score, extra["verifier_reward_before_lock_rule"] = False, 0.0, r
            paths.error(f"{sample} {tid}: environment failure by the lock's handling rule (verifier cannot run on this date); "
                        f"harbor reward {r}")
        out = paths.results_dir / "tasks" / tid / sample
        copy_if_exists(tdir / "result.json", out / "harbor_result.json")
        copy_if_exists(tdir / "agent" / "trajectory.json", out / "trajectory.json")
        for name in VERIFIER_FILES:
            copy_if_exists(tdir / "verifier" / name, out / "verifier" / name)
        records.append(record("terminal_bench", tid, sample=sample, resolved=resolved, score=score, max_score=1.0,
                              run_status=status, harness_status=harness_status, error_reason=reason,
                              time_sec=seconds_between(ex.get("started_at"), ex.get("finished_at")), input_tokens=tin, output_tokens=tout,
                              artifacts=f"tasks/{tid}/{sample}", trial_name=tdir.name, n_results=n_results,
                              agent_started_at=ex.get("started_at"), agent_finished_at=ex.get("finished_at"), **extra))
    return records


def summarize(seeds: list[int], task_ids: list[str], records: list[dict[str, Any]],
              unverifiable: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """(per-seed rows, per-task pass counts, summary) from the task x seed records; `unverifiable` = the lock's list of
    tasks whose verifier cannot run on this date (they are inside every seed's denominator and inside the run failures)."""
    unverifiable = {t: c for t, c in (unverifiable or {}).items() if t in task_ids}
    per_seed = []
    for seed in seeds:
        recs = [r for r in records if r["sample"] == f"seed_{seed}"]
        n_ok = sum(1 for r in recs if r["resolved"])
        per_seed.append({"seed": seed, "n_tasks": len(recs), "n_resolved": n_ok,
                         "success_rate_pct": round(100 * n_ok / len(recs), 2) if recs else None,
                         **{k: v for k, v in failure_summary(recs).items() if k in ("n_run_failures", "n_budget_exhausted", "run_status_counts", "failed")}})
    rates = [s["success_rate_pct"] for s in per_seed if s["success_rate_pct"] is not None]
    pass_counts = {tid: sum(1 for r in records if r["task_id"] == tid and r["resolved"]) for tid in task_ids}
    summary = {
        "reported_metric": ("mean +/- sample std of the per-seed task success rate (%)" if len(seeds) > 1 else
                            "task success rate (%) of the single seed; per-task outcomes support paired comparisons across versions"),
        "n_tasks": len(task_ids), "n_seeds": len(seeds), "per_seed_success_rate_pct": rates,
        "mean_success_rate_pct": round(statistics.fmean(rates), 2) if rates else None,
        "std_success_rate_pct": round(statistics.stdev(rates), 2) if len(rates) > 1 else None,
        "tasks_passed_in_all_seeds": sum(1 for v in pass_counts.values() if v == len(seeds)),
        "tasks_passed_in_no_seed": sum(1 for v in pass_counts.values() if v == 0),
        "n_unverifiable_tasks_from_lock": len(unverifiable),
        "unverifiable_tasks_from_lock": unverifiable,   # environment failures for every version, unresolved in every seed (lock handling rule)
        **failure_summary(records),
    }
    return per_seed, pass_counts, summary


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if not HARBOR.exists() or not a.dataset.exists():
        sys.exit("run scripts/setup_eval_env.sh first (harbor venv or Terminal-Bench dataset missing)")
    task_ids = read_task_ids(a.task_ids_file, a.limit)
    env = {**os.environ, "OPENAI_API_KEY": a.api_key, "OPENAI_API_BASE": a.api_base, "OPENAI_BASE_URL": a.api_base}
    with RunPaths("terminal_bench", a.version, a.run_id, root=a.out_root, resume=a.resume) as paths:
        paths.log(f"run_id={paths.run_id} version={a.version} model={a.model_name} tasks={len(task_ids)} seeds={a.seeds} n_concurrent={a.n_concurrent}")
        ep = preflight(paths, a, docker=True)
        lock_copy = paths.snapshot_config(LOCK)
        paths.snapshot_config(a.task_ids_file)
        locked = {"n_concurrent": a.lock["evaluator"]["n_concurrent"], "agent_timeout_sec": opt(a.lock["agent"]["agent_timeout_sec"], float),
                  "override_cpus": a.lock["evaluator"]["resources_per_task"]["override_cpus"],
                  "override_memory_mb": a.lock["evaluator"]["resources_per_task"]["override_memory_mb"],
                  "max_retries": a.lock["evaluator"]["infrastructure_retries"]["harbor_max_retries"], "seeds": list(a.lock["protocol"]["seeds"])}
        deviations = lock_deviations({"n_concurrent": a.n_concurrent, "agent_timeout_sec": a.agent_timeout_sec, "override_cpus": a.override_cpus,
                                      "override_memory_mb": a.override_memory_mb, "max_retries": a.max_retries, "seeds": list(a.seeds)}, locked)
        reportable = not deviations and not a.limit and a.task_ids_file == IDS
        if not reportable:
            paths.log(f"NOT REPORTABLE: deviations={deviations} limit={a.limit} task_ids_file={a.task_ids_file}")
        paths.write_metadata(
            run_id=paths.run_id, benchmark="terminal_bench", benchmark_version="2.0", version=a.version, model_name=a.model_name,
            endpoint=ep, serving=serving_info(a.serving_info), code=git_info(), hardware=hardware_info(),
            config={"lock_file": str(lock_copy.relative_to(paths.run_dir)), "agent": "terminus-2", "agent_timeout_sec": a.agent_timeout_sec,
                    "override_cpus": a.override_cpus, "override_memory_mb": a.override_memory_mb, "n_concurrent": a.n_concurrent,
                    "max_retries": a.max_retries, "retry_include": a.retry_include, "rerun_passes": a.rerun_passes,
                    "unverifiable_tasks_from_lock": sorted(t for t in a.unverifiable if t in task_ids),
                    "agent_kwargs": job_config(a, None)["agents"][0]["kwargs"], "per_seed_job_configs": "config_snapshot/harbor_job_seed_<s>.yaml",
                    "dataset_git_commit_lock": a.lock["source"]["git_commit"]},
            data_version={"task_ids_file": str(a.task_ids_file), "n_tasks": len(task_ids), "limit": a.limit},
            model_revision=model_revision(), seed=a.seeds, command=command_line(), note=a.note,
            lock_deviations=deviations, reportable=reportable, start_time=utc_now(), status="running",
        )
        exit_codes: dict[str, list[int]] = {}
        records: list[dict[str, Any]] = []
        if a.resume:   # metrics.jsonl and per_task.jsonl are rebuilt from the trial results on resume
            (paths.results_dir / "per_task.jsonl").unlink(missing_ok=True)
        for seed in a.seeds:
            rcs = run_seed(paths, a, env, seed, task_ids)
            exit_codes[f"seed_{seed}"] = rcs
            if any(rc != 0 for rc in rcs):
                paths.error(f"harbor job seed_{seed} exit codes {rcs}")
            recs = collect_seed(paths, seed, task_ids, a.unverifiable, a.stale_reference)
            paths.write_records(recs, append=True)   # streamed per seed, so an interrupted run still has its finished seeds on disk
            records.extend(recs)
            fs = failure_summary(recs)
            paths.log(f"seed {seed}: {sum(1 for r in recs if r['resolved'])}/{len(recs)} resolved, {fs['n_run_failures']} run failures, "
                      f"{fs['n_budget_exhausted']} budget exhausted")
        per_seed, pass_counts, stats = summarize(a.seeds, task_ids, records, a.unverifiable)
        summary = {"run_id": paths.run_id, "benchmark": "terminal_bench", "version": a.version, "model_name": a.model_name,
                   "reportable": reportable, "lock_deviations": deviations, **stats}
        write_json(paths.results_dir / "per_seed.json", {"per_seed": per_seed, "pass_counts": pass_counts})
        paths.finish(a, summary, harbor_exit_codes=exit_codes)
        paths.log(f"done: mean {summary['mean_success_rate_pct']}% +/- {summary['std_success_rate_pct']} over {len(a.seeds)} seeds; "
                  f"{summary['n_run_failures']} run failures, {summary['n_budget_exhausted']} budget exhausted; results in {paths.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
