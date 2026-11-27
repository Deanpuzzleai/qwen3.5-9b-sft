"""Shared helpers for the Harbor-based benchmarks (SWE-bench Pro v2, Terminal-Bench 2.0)."""
from __future__ import annotations

import datetime as dt
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apex_eval.common import (AGENT_ERROR, BUDGET_EXHAUSTED, ENVIRONMENT_FAILURE, OK, REPO_ROOT, VERIFIER_FAILURE, RunPaths,
                              read_json, run_logged)

HARBOR = REPO_ROOT / ".venv" / "harbor" / "bin" / "harbor"
HARBOR_PYTHON = REPO_ROOT / ".venv" / "harbor" / "bin" / "python"

# Harbor exception types (harbor 0.24.0) that mean the sandbox, not the model, failed. Harbor itself retries the first
# list when passed as --retry-include; the compose failure is a plain RuntimeError that harbor cannot retry selectively,
# so the runners re-run those trials themselves (harbor_run_resilient).
ENVIRONMENT_EXCEPTIONS = ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError",
                          "SandboxLikelyOutOfMemoryError", "MemoryLimitExceededError")
ENVIRONMENT_MESSAGE_PATTERNS = (("RuntimeError", "Docker compose command failed"),)
VERIFIER_EXCEPTIONS = ("VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError")
AGENT_BUDGET_EXCEPTION = "AgentTimeoutError"
DEFAULT_RETRY_INCLUDE = list(ENVIRONMENT_EXCEPTIONS[:3])

# Harbor decides once per process whether the docker host supports offline sandboxes by running a 30 s container probe.
# Under load (image pulls, many containers starting) that probe can time out, after which every trial in the job is
# rejected with this message before the agent starts and no result.json is written.
PROBE_MSG = "network_mode='no-network' is not supported"

Trial = tuple[Path, dict[str, Any], int]   # trial dir, its result.json, number of result files seen for the task across passes


def harbor_run(paths: RunPaths, job_name: str, args: list[str], env: dict[str, str], cwd: Path) -> int:
    """`harbor run <args>` as job <job_name> under <run_dir>/harbor, full output in <run_dir>/harbor_<job_name>.log."""
    cmd = [str(HARBOR), "run", *args, "--quiet", "--yes", "--job-name", job_name, "-o", str(paths.run_dir / "harbor")]
    return run_logged(paths, f"harbor {job_name}", cmd, env, cwd, log_path=paths.run_dir / f"harbor_{job_name}.log")


def task_args(dataset_path: Path, task_ids: list[str]) -> list[str]:
    out = ["-p", str(dataset_path)]
    for tid in task_ids:
        out += ["-i", tid]
    return out


def retry_args(max_retries: int, retry_include: list[str]) -> list[str]:
    out = ["--max-retries", str(max_retries)]
    for exc in retry_include:
        out += ["--retry-include", exc]
    return out


def load_trials(*job_dirs: Path) -> dict[str, Trial]:
    """task_name -> (trial_dir, result.json, n_results) over one or more job dirs; the latest finished result wins."""
    out: dict[str, Trial] = {}
    for rj in sorted(rj for job_dir in job_dirs for rj in job_dir.glob("*/result.json")):
        d = read_json(rj)
        if not isinstance(d, dict) or not d.get("task_name"):
            continue
        name = d["task_name"].split("/")[-1]
        prev = out.get(name)
        n = (prev[2] if prev else 0) + 1
        if prev is None or (d.get("finished_at") or "") >= (prev[1].get("finished_at") or ""):
            out[name] = (rj.parent, d, n)
        else:
            out[name] = (prev[0], prev[1], n)
    return out


def seconds_between(a: str | None, b: str | None) -> float | None:
    try:
        return (dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).total_seconds()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def reward(d: dict[str, Any]) -> float | None:
    r = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return None if r is None else float(r)


def exception(d: dict[str, Any]) -> tuple[str | None, str | None]:
    ei = d.get("exception_info") or {}
    msg = (ei.get("exception_message") or ei.get("exception_traceback") or "")[:500]
    return ei.get("exception_type"), (msg or None)


def agent_tokens(d: dict[str, Any]) -> tuple[int | None, int | None]:
    ar = d.get("agent_result") or {}
    return ar.get("n_input_tokens"), ar.get("n_output_tokens")


def classify(exc_type: str | None, exc_msg: str | None = None) -> str:
    """Harbor trial exception -> shared run_status vocabulary (common.py)."""
    if not exc_type:
        return OK
    if exc_type == AGENT_BUDGET_EXCEPTION:
        return BUDGET_EXHAUSTED
    if exc_type in ENVIRONMENT_EXCEPTIONS or any(exc_type == t and p in (exc_msg or "") for t, p in ENVIRONMENT_MESSAGE_PATTERNS):
        return ENVIRONMENT_FAILURE
    if exc_type in VERIFIER_EXCEPTIONS:
        return VERIFIER_FAILURE
    return AGENT_ERROR


def lock_known_failures(lock: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Tasks a benchmark lock's harness_check_* blocks list as drifted since the benchmark's release, as two {task: cause}
    maps: tasks whose verifier cannot run on this date (the lock's handling rule: an environment failure for every model
    version, counted as unresolved) and tasks whose reference solution is stale while task and verifier are intact
    (informational only; the task stays solvable). Locks without such lists yield two empty maps."""
    unverifiable: dict[str, str] = {}
    stale_reference: dict[str, str] = {}
    for key, block in lock.items():
        if not (key.startswith("harness_check_") and isinstance(block, dict)):
            continue
        for e in block.get("verifier_failures") or []:
            unverifiable[e["task"]] = f"{key}: {e['cause']}"
        for e in block.get("reference_solution_failures") or []:
            stale_reference[e["task"]] = f"{key}: {e['cause']}"
    return unverifiable, stale_reference


def validation_verdict(oracle: dict[str, dict[str, Any]], nop: dict[str, dict[str, Any]], n_tasks: int, nop_sample: int,
                       expected_failures: dict[str, str]) -> dict[str, Any]:
    """PASS/FAIL of a harness check (scripts/validate_harness.sh). PASS = every task has an oracle trial, the oracle resolves
    every task the lock does not list as drifted (expected_failures, from lock_known_failures), and the empty agent ran on
    its whole sample and resolved nothing. `oracle` / `nop`: task -> {reward, exception, attempts} of the latest attempt."""
    resolved = {k for k, v in oracle.items() if v.get("reward") == 1.0}
    unresolved = sorted(k for k in oracle if k not in resolved)
    unexpected = sorted(k for k in unresolved if k not in expected_failures)
    spurious = sorted(k for k, v in nop.items() if v.get("reward") == 1.0)
    return {
        "oracle": {"n_tasks": n_tasks, "n_trials_found": len(oracle), "n_resolved": len(resolved),
                   "unresolved_or_errored": unresolved,
                   "expected_failures_from_lock": {k: expected_failures[k] for k in unresolved if k in expected_failures},
                   "unexpected_failures": unexpected,
                   "expected_but_resolved": sorted(k for k in expected_failures if k in resolved),
                   "exceptions": {k: v["exception"] for k, v in oracle.items() if v.get("exception")},
                   "retried_tasks": {k: v["attempts"] for k, v in oracle.items() if v.get("attempts", 1) > 1}},
        "nop": {"n_tasks": nop_sample, "n_trials_found": len(nop), "n_resolved": len(spurious), "spurious_passes": spurious},
        "pass": not unexpected and len(oracle) == n_tasks and len(nop) == nop_sample and not spurious,
    }


def wait_for_docker_probe(paths: RunPaths, attempts: int = 6, sleep_sec: int = 20) -> bool:
    """Run harbor's own kernel probe in a fresh process until it passes, so a job is not started while the daemon is swamped."""
    code = ("from harbor.environments.docker.docker import DockerEnvironment as D; import sys; "
            "sys.exit(0 if D._egress_control_kernel_support() else 1)")
    for i in range(1, attempts + 1):
        try:
            r = subprocess.run([str(HARBOR_PYTHON), "-c", code], capture_output=True, text=True, timeout=120, check=False)
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0:
            return True
        if r is not None and "Traceback" in r.stderr:
            raise RuntimeError(f"harbor's docker probe crashed (broken harbor venv?):\n{r.stderr[-800:]}")
        paths.log(f"docker egress probe did not pass (attempt {i}/{attempts}); waiting {sleep_sec}s for the daemon to settle")
        time.sleep(sleep_sec)
    return False


def job_dirs(paths: RunPaths, job_name: str) -> list[Path]:
    base = paths.run_dir / "harbor" / job_name
    return [d for d in [base, *sorted(base.parent.glob(f"{job_name}_rerun*"))] if d.exists()]


def pending_tasks(dirs: list[Path], task_ids: list[str]) -> list[str]:
    """Tasks with no trial result in these job dirs (harbor rejected them before the trial ran, or the runner was
    interrupted) plus tasks whose latest result is an environment failure (the sandbox, not the model, failed)."""
    done = load_trials(*dirs)
    return [t for t in task_ids if t not in done or classify(*exception(done[t][1])) == ENVIRONMENT_FAILURE]


def harbor_run_resilient(paths: RunPaths, job_name: str, args_for: Callable[[list[str]], list[str]], env: dict[str, str],
                         cwd: Path, task_ids: list[str], max_passes: int = 3) -> list[int]:
    """Run a harbor job for the tasks that still need a result. Tasks that produced no result (probe rejection, harness
    crash, interrupted run) or an environment failure are re-run in follow-up jobs named <job_name>_rerun<N>; an existing
    job (--resume) is continued the same way. Returns the exit codes of the passes run now. Nothing is ever dropped:
    tasks still pending after the last pass are reported by the caller as missing / environment failures."""
    existing = job_dirs(paths, job_name)
    remaining = pending_tasks(existing, task_ids) if existing else list(task_ids)
    if existing:
        paths.log(f"{job_name}: {len(task_ids) - len(remaining)} task(s) already have a result; {len(remaining)} pending")
    rcs: list[int] = []
    for p in range(1, max_passes + 1):
        if not remaining:
            break
        n_prev = len(job_dirs(paths, job_name))
        name = job_name if n_prev == 0 else f"{job_name}_rerun{n_prev}"
        if not wait_for_docker_probe(paths):
            paths.error(f"{name}: docker egress probe kept failing; starting the job anyway")
        rcs.append(harbor_run(paths, name, args_for(remaining), env, cwd))
        log = paths.run_dir / f"harbor_{name}.log"
        probe_hit = log.exists() and PROBE_MSG in log.read_text(errors="replace")
        remaining = pending_tasks(job_dirs(paths, job_name), remaining)
        if remaining:
            why = " (harbor rejected offline sandboxes: egress probe timed out)" if probe_hit else ""
            what = "re-running them" if p < max_passes else "reported as missing / environment failures"
            paths.error(f"{name}: {len(remaining)} task(s) without a usable result{why}; {what}")
    return rcs


def consolidate_trials(dirs: list[Path], out_dir: Path) -> Path:
    """Symlink the winning trial dir per task (latest finished across the given job dirs) into one directory, so
    consumers that scan a single job dir (e.g. patch_replay's source_job) see exactly one trial per task."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():   # rebuilt from scratch (links only): a re-run task must not appear twice
        if old.is_symlink():
            old.unlink()
    for tdir, _result, _n in load_trials(*dirs).values():
        (out_dir / tdir.name).symlink_to(tdir.resolve(), target_is_directory=True)
    return out_dir
