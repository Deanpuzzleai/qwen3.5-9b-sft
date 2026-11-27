"""Shared run bookkeeping, record schema and helpers for the evaluation runners (and the run-directory base class
the training side builds on).

Layout: every run gets a run_id and writes logs/<run_id>/{metadata.json, eval.log | train.log, metrics.jsonl,
errors.log, resources.csv}; evaluation results go to results/<version>/<benchmark>/<run_id>/.

Per-task record schema (one JSON line per task in per_task.jsonl, identical keys for all four benchmarks; see `record`):
  benchmark, task_id, sample (seed / prompt level / null), group (repo / domain / category / null), resolved (binary verdict
  or null), score, max_score, run_status (vocabulary below), harness_status (the harness's own status or exception type),
  error_reason, time_sec, input_tokens, output_tokens, reasoning_tokens, artifacts (dir relative to the results dir), extra.
run_status vocabulary: ok | budget_exhausted (the agent used its whole time budget; its output is still graded) |
  agent_error | environment_failure | verifier_failure | scoring_failure | missing (no trial produced) |
  pending_external (official scoring happens elsewhere) | skipped (debugging flags). Run failures = agent_error,
  environment_failure, verifier_failure, scoring_failure, missing; budget_exhausted is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_LOCK = REPO_ROOT / "configs" / "eval" / "model.yaml"
COMPUTE_CONFIG = REPO_ROOT / "configs" / "compute.yaml"
VERSIONS = ("base", "sft", "stage2")

OK, BUDGET_EXHAUSTED, AGENT_ERROR, ENVIRONMENT_FAILURE, VERIFIER_FAILURE, SCORING_FAILURE, MISSING, PENDING_EXTERNAL, SKIPPED = (
    "ok", "budget_exhausted", "agent_error", "environment_failure", "verifier_failure", "scoring_failure", "missing",
    "pending_external", "skipped")
RUN_FAILURE_STATUSES = frozenset({AGENT_ERROR, ENVIRONMENT_FAILURE, VERIFIER_FAILURE, SCORING_FAILURE, MISSING})
METRIC_KEYS = ("task_id", "sample", "resolved", "score", "run_status", "time_sec", "input_tokens", "output_tokens")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def new_run_id(*parts: str) -> str:
    """`<part>_<part>_<UTC timestamp>`, e.g. new_run_id('hle', 'base') or new_run_id('sft', 'r2')."""
    return "_".join([*parts, timestamp()])


# --- small file / process helpers ---------------------------------------------------------------------------------
def read_json(path: Path, default: Any = None) -> Any:
    """Parse a JSON file; return `default` when it is missing or malformed (callers record that as a run failure)."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (a kill mid-write never leaves a truncated file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_task_ids(path: Path, limit: int | None = None) -> list[str]:
    """Non-empty lines of a task-id file, in file order; `limit` keeps the first N (smoke tests and pilots)."""
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return ids[:limit] if limit else ids


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def model_lock() -> dict[str, str]:
    """{'repo', 'revision'} of the locked base model (configs/eval/model.yaml, read-only)."""
    m = yaml.safe_load(MODEL_LOCK.read_text())["model"]
    return {"repo": str(m["repo"]), "revision": str(m["revision"])}


def model_revision() -> str:
    return model_lock()["revision"]


def compute_rates() -> dict[str, float]:
    """Cost assumptions shared by every resources.csv (configs/compute.yaml)."""
    c = yaml.safe_load(COMPUTE_CONFIG.read_text())
    return {"gpu_vm_usd_per_hour": float(c["gpu_vm_usd_per_hour"]), "cpu_vm_usd_per_hour": float(c["cpu_vm_usd_per_hour"]),
            "default_serving_gpus": int(c["default_serving_gpus"])}


def git_head(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], repo)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """stdout of a short informational command, or 'unavailable (...)'; never raises (metadata must not break a run)."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({type(exc).__name__})"


# --- provenance ---------------------------------------------------------------------------------------------------
def git_info() -> dict[str, Any]:
    dirty = _run(["git", "status", "--porcelain"], REPO_ROOT)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT),
        "dirty": bool(dirty) and not dirty.startswith("unavailable"),
    }


def hardware_info() -> dict[str, Any]:
    mem_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    gpus: Any = "none"
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        gpus = [g.strip() for g in out.splitlines() if g.strip()] or out
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "mem_total_gb": mem_gb,
        "gpus": gpus,
        "python": sys.version.split()[0],
    }


def endpoint_info(api_base: str) -> dict[str, Any]:
    """Best-effort description of the OpenAI-compatible server (served model ids, vLLM version)."""
    info: dict[str, Any] = {"api_base": api_base}
    root = api_base.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    for key, url in (("models", f"{api_base.rstrip('/')}/models"), ("version", f"{root}/version")):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310
                body = json.loads(r.read().decode())
            info[key] = [m.get("id") for m in body.get("data", [])] if key == "models" else body
        except Exception as exc:  # noqa: BLE001 - informational only
            info[key] = f"unavailable ({type(exc).__name__})"
    return info


def checkout_matches(run: RunDir, name: str, repo: Path, locked_commit: str) -> str:
    """Record a pinned checkout's HEAD and abort the run if it is not the commit the lock names."""
    head = git_head(repo)
    if head != locked_commit:
        run.error(f"{name} checkout is at {head}, lock says {locked_commit}; run scripts/setup_eval_env.sh")
        raise SystemExit(f"{name} checkout does not match the lock ({head} != {locked_commit})")
    return head


def command_line() -> str:
    """argv as a shell line, with the value of --api-key redacted."""
    out, redact = [], False
    for x in sys.argv:
        if redact:
            out.append("***")
        elif x.startswith("--api-key="):
            out.append("--api-key=***")
        else:
            out.append(shlex.quote(x))
        redact = x == "--api-key"
    return " ".join(out)


# --- CLI flags shared by the four evaluation runners -------------------------------------------------------------
def add_common_args(p: argparse.ArgumentParser) -> None:
    """Flags every runner takes, with identical names and defaults (model endpoint, run identity, resource accounting)."""
    rates = compute_rates()
    p.add_argument("--version", required=True, choices=VERSIONS, help="which model version is being evaluated")
    p.add_argument("--model-name", required=True, help="served model id on the endpoint")
    p.add_argument("--api-base", required=True, help="OpenAI-compatible base URL, e.g. http://10.0.0.5:8000/v1")
    p.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "EMPTY"), help="key for the model endpoint (dummy for vLLM; prefer EVAL_API_KEY)")
    p.add_argument("--run-id", default=None, help="default: <benchmark>_<version>_<UTC timestamp>")
    p.add_argument("--resume", action="store_true", help="continue an existing --run-id: finished work is kept, unfinished tasks are re-run")
    p.add_argument("--serving-info", type=Path, default=None, help="metadata.json written by scripts/serve_vllm.sh, embedded in this run's metadata")
    p.add_argument("--serving-gpus", type=int, default=rates["default_serving_gpus"], help="GPUs serving the model during this run (resources.csv)")
    p.add_argument("--gpu-usd-per-hour", type=float, default=rates["gpu_vm_usd_per_hour"], help="per serving GPU (configs/compute.yaml)")
    p.add_argument("--peak-gpu-mem-gb", type=float, default=None, help="peak GPU memory on the serving VM, if measured")
    p.add_argument("--note", default="", help="free-text note stored in metadata.json")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: write logs/ and results/ under this root instead of the repository")


def lock_deviations(effective: dict[str, Any], locked: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Locked settings whose effective value differs from the lock (CLI overrides). Non-empty => the run is not reportable."""
    return {k: {"lock": locked[k], "effective": effective[k]} for k in locked if effective.get(k) != locked[k]}


def serving_info(path: Path | None) -> Any:
    return read_json(path, f"unreadable: {path}") if path else None


# --- per-task records -----------------------------------------------------------------------------------------------
def record(benchmark: str, task_id: str, *, run_status: str, sample: str | None = None, group: str | None = None,
           resolved: bool | None = None, score: float | None = None, max_score: float | None = None,
           harness_status: str | None = None, error_reason: str | None = None, time_sec: float | None = None,
           input_tokens: int | None = None, output_tokens: int | None = None, reasoning_tokens: int | None = None,
           artifacts: str | None = None, **extra: Any) -> dict[str, Any]:
    """One per-task record in the shared schema (module docstring). Benchmark-specific fields go under `extra`."""
    return {"benchmark": benchmark, "task_id": task_id, "sample": sample, "group": group, "resolved": resolved,
            "score": score, "max_score": max_score, "run_status": run_status, "harness_status": harness_status,
            "error_reason": error_reason, "time_sec": None if time_sec is None else round(time_sec, 3),
            "input_tokens": input_tokens, "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens,
            "artifacts": artifacts, "extra": extra}


def failure_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-failure accounting shared by all benchmarks (failures recorded and counted separately)."""
    n = len(records)
    failures = [r for r in records if r["run_status"] in RUN_FAILURE_STATUSES]
    times = [r["time_sec"] for r in records if r.get("time_sec") is not None]
    tin = [r["input_tokens"] for r in records if r.get("input_tokens") is not None]
    tout = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    return {
        "n_records": n,
        "n_run_failures": len(failures), "run_failure_rate_pct": round(100 * len(failures) / n, 2) if n else None,
        "n_budget_exhausted": sum(1 for r in records if r["run_status"] == BUDGET_EXHAUSTED),
        "run_status_counts": dict(Counter(r["run_status"] for r in records)),
        "failed": [{"task_id": r["task_id"], "sample": r["sample"], "run_status": r["run_status"], "error_reason": r["error_reason"]} for r in failures],
        "mean_time_sec": round(fmean(times), 1) if times else None,
        "total_input_tokens": sum(tin) if tin else None, "total_output_tokens": sum(tout) if tout else None,
        "tokens_known_for_n_records": len(tout),
    }


# --- run directories: the base class and the evaluation variant ---------------------------------------------------
class RunDir:
    """Append-only run directory <root>/logs/<run_id>/ with the five run files. Use as a context manager so a
    crash or SIGTERM is still recorded: errors.log gets the traceback, metadata.status becomes crashed/aborted, the
    exception propagates. Never overwrites an existing run_id unless `resume` is set."""

    def __init__(self, run_id: str, root: Path | None = None, *, log_name: str = "eval.log", resume: bool = False):
        self.run_id, self.log_name = run_id, log_name
        # a non-default root is for smoke tests only; absolute, because harbor runs with the benchmark checkout as cwd
        self.root = Path(root).resolve() if root else REPO_ROOT
        self.run_dir = self.root / "logs" / run_id
        if self.run_dir.exists() and not resume:
            raise FileExistsError(f"run_id {run_id} already exists; never overwrite logs (use --resume to continue it)")
        if resume and not (self.run_dir / "metadata.json").exists():
            raise FileNotFoundError(f"--resume: no previous run found at {self.run_dir}")
        self.resumed = resume and self.run_dir.exists()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        if self.resumed:   # duration (and so GPU hours and cost) spans from the original start; a gap before the resume counts too
            first = read_json(self.run_dir / "metadata.json", {}).get("start_time")
            if first:
                self.started = dt.datetime.fromisoformat(first.replace("Z", "+00:00")).timestamp()
        self._prev_sigterm: Any = None

    def __enter__(self):
        def on_sigterm(signum, frame):  # noqa: ARG001
            raise SystemExit(143)
        self._prev_sigterm = signal.signal(signal.SIGTERM, on_sigterm)
        if self.resumed:
            meta = self.write_metadata(resumed_at=[*read_json(self.run_dir / "metadata.json", {}).get("resumed_at", []), utc_now()], status="running")
            self.log(f"resuming run {self.run_id} (resume #{len(meta['resumed_at'])})")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        signal.signal(signal.SIGTERM, self._prev_sigterm or signal.SIG_DFL)
        if exc is not None and not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            if isinstance(exc, SystemExit):
                reason = "terminated (SIGTERM)" if exc.code == 143 else f"aborted: {exc.code}"
                self.error(reason)
                status = "aborted"
            else:
                self.error("CRASH: " + "".join(traceback.format_exception(exc_type, exc, tb)).rstrip())
                reason, status = f"{exc_type.__name__}: {exc}", "crashed"
            self.write_metadata(end_time=utc_now(), duration_sec=round(self.elapsed(), 1), status=status, failure=reason)
            if not (self.run_dir / "resources.csv").exists():
                self.write_resources([("duration_sec", round(self.elapsed(), 1), "s", f"run {status}; see errors.log")])
            self.finalize()
        return False

    def elapsed(self) -> float:
        return time.time() - self.started

    # --- append-only logs ---
    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with (self.run_dir / self.log_name).open("a") as f:
            f.write(line + "\n")

    def error(self, msg: str) -> None:
        with (self.run_dir / "errors.log").open("a") as f:
            f.write(f"[{utc_now()}] {msg}\n")

    def metric(self, rec: dict[str, Any]) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"ts": utc_now(), **rec}, default=str) + "\n")

    # --- metadata / resources ---
    def read_metadata(self) -> dict[str, Any]:
        return read_json(self.run_dir / "metadata.json", {})

    def write_metadata(self, **fields: Any) -> dict[str, Any]:
        meta = self.read_metadata()
        meta.update(fields)
        write_json(self.run_dir / "metadata.json", meta)
        return meta

    def snapshot_config(self, src: Path) -> Path:
        dst = self.run_dir / "config_snapshot" / src.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def write_resources(self, rows: list[tuple[str, Any, str, str]]) -> None:
        with (self.run_dir / "resources.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "unit", "note"])
            w.writerows(rows)

    def finalize(self) -> None:
        """The five files exist on every exit path; errors.log says 'none' when nothing was recorded."""
        for name in (self.log_name, "metrics.jsonl"):
            (self.run_dir / name).touch()
        err = self.run_dir / "errors.log"
        if not err.exists() or err.stat().st_size == 0:
            err.write_text("none\n")


def run_segment(meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """One finished or interrupted segment of a run, from its metadata before a --resume overwrites it. A segment that
    never wrote end_time (killed, VM stopped) ends at the last file write under its run directory."""
    end, source = meta.get("end_time"), "end_time"
    if not end:
        last = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=None)
        end = dt.datetime.fromtimestamp(last, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if last else meta.get("start_time")
        source = "last file write"
    seconds = (lambda t: dt.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
    return {"start_time": meta.get("start_time"), "end_time": end, "end_source": source,
            "duration_sec": round(max(0.0, seconds(end) - seconds(meta["start_time"])), 1) if meta.get("start_time") and end else 0.0,
            **{k: meta.get(k) for k in ("status", "failure", "endpoint", "serving", "command", "note", "code")}}


class RunPaths(RunDir):
    """An evaluation run: logs/<run_id>/ (eval.log) plus results/<version>/<benchmark>/<run_id>/.

    A --resume run keeps each earlier segment in metadata `segments` (start, end, endpoint, serving, command, note, code)
    before the runner rewrites those fields for the new segment. Its duration (and so GPU hours and cost) is the sum of
    the segments' own durations: time between segments, when no part of the run was active, is not counted."""

    def __init__(self, benchmark: str, version: str, run_id: str | None = None, root: Path | None = None, resume: bool = False):
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
        self.benchmark, self.version = benchmark, version
        run_id = run_id or new_run_id(benchmark, version)
        results_dir = (Path(root).resolve() if root else REPO_ROOT) / "results" / version / benchmark / run_id
        if results_dir.exists() and not resume:
            raise FileExistsError(f"results for run_id {run_id} already exist; never overwrite results (use --resume to continue)")
        super().__init__(run_id, root, log_name="eval.log", resume=resume)
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.segments: list[dict[str, Any]] = []
        if self.resumed:
            prev = self.read_metadata()
            self.segments = [*prev.get("segments", []), run_segment(prev, self.run_dir)]
            self.started = time.time()

    def __enter__(self):
        if self.segments:
            self.write_metadata(segments=self.segments)
        return super().__enter__()

    def elapsed(self) -> float:
        return sum(s["duration_sec"] for s in self.segments) + time.time() - self.started

    def write_records(self, records: list[dict[str, Any]], append: bool = False) -> None:
        """Write per-task records to per_task.jsonl and mirror METRIC_KEYS of each into metrics.jsonl."""
        with (self.results_dir / "per_task.jsonl").open("a" if append else "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                self.metric({k: r.get(k) for k in METRIC_KEYS})

    def finish(self, a: argparse.Namespace, summary: dict[str, Any], **extra: Any) -> None:
        """Common end of a successful run: resources.csv from the shared flags, final metadata, summary.json, errors.log."""
        duration = self.elapsed()
        self.write_resources(resource_rows(duration, serving_gpus=a.serving_gpus, gpu_usd_per_hour=a.gpu_usd_per_hour,
                                           peak_gpu_mem_gb=a.peak_gpu_mem_gb,
                                           orchestrator_cpus=os.cpu_count()))
        self.write_metadata(end_time=utc_now(), duration_sec=round(duration, 1), status="finished", summary=summary,
                            results_dir=str(self.results_dir), **extra)
        write_json(self.results_dir / "summary.json", summary)
        self.finalize()


def preflight(paths: RunPaths, a: argparse.Namespace, *, docker: bool = False, binaries: tuple[str, ...] = (),
              env_vars: tuple[str, ...] = ()) -> dict[str, Any]:
    """Fail fast, with the reason in errors.log, on the conditions every real run needs: the model id is served at
    --api-base, docker answers (Harbor benchmarks), required binaries are on PATH, required env vars are set."""
    ep = endpoint_info(a.api_base)
    problems = []
    if isinstance(ep.get("models"), list) and a.model_name not in ep["models"]:
        problems.append(f"model {a.model_name!r} is not served at {a.api_base} (served: {ep['models']})")
    elif not isinstance(ep.get("models"), list):
        problems.append(f"endpoint {a.api_base} did not answer /models ({ep.get('models')})")
    if docker and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        problems.append("docker daemon not reachable")
    for b in binaries:
        if not shutil.which(b):
            problems.append(f"{b} not found on PATH")
    for v in env_vars:
        if not os.environ.get(v):
            problems.append(f"{v} is not set")
    for pr in problems:
        paths.error(f"preflight: {pr}")
    if problems:
        raise SystemExit("preflight failed: " + "; ".join(problems))
    return ep


def run_logged(run: RunDir, name: str, cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path | None = None) -> int:
    """Run a subprocess with its combined output captured to a log file; the command and exit code go to the run log.
    If the runner is interrupted (SIGTERM/Ctrl-C), the child is terminated instead of being left orphaned."""
    log_path = log_path or (run.run_dir / f"{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run.log(f"{name}: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    with log_path.open("a") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    run.log(f"{name}: exit {rc} after {time.time() - t0:.0f}s")
    return rc


def resource_rows(duration_sec: float, *, serving_gpus: int, gpu_usd_per_hour: float,
                  peak_gpu_mem_gb: float | None, orchestrator_cpus: int | None,
                  cpu_usd_per_hour: float | None = None) -> list[tuple[str, Any, str, str]]:
    """Resource accounting for an evaluation run whose model is served on remote GPU VM(s). Cost is GPU time only: the CPU
    VM is always on and is not charged to runs; cpu_usd_per_hour is accepted from older callers and ignored."""
    hours = duration_sec / 3600
    gpu_hours = hours * serving_gpus
    cost = gpu_hours * gpu_usd_per_hour
    return [
        ("duration_sec", round(duration_sec, 1), "s", "wall clock of this run on the CPU VM"),
        ("orchestrator_cpus", orchestrator_cpus, "vcpu", "CPU box running sandboxes and the agent loop"),
        ("serving_gpus", serving_gpus, "count", "GPUs serving the model during this run (from --serving-gpus)"),
        ("peak_gpu_memory_gb", peak_gpu_mem_gb if peak_gpu_mem_gb is not None else "n/a",
         "GB", "from the serving VM (nvidia-smi) if provided via --peak-gpu-mem-gb; n/a = not measured"),
        ("gpu_hours", round(gpu_hours, 3), "h", "estimated = duration x serving_gpus (GPU may be shared with other runs)"),
        ("cost_usd", round(cost, 2), "USD", f"estimated GPU cost = gpu_hours x {gpu_usd_per_hour} (configs/compute.yaml); CPU VM not charged"),
    ]
