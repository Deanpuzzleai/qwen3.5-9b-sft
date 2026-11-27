"""Run bookkeeping for the GPU bring-up checks (helpers of scripts/gpu_checks.sh and scripts/serve_vllm.sh).

Every check is a run: it gets a run_id and writes logs/<run_id>/{metadata.json, eval.log|train.log, metrics.jsonl,
errors.log, resources.csv}. This mirrors src/apex_eval/common.RunPaths (same files, same append-only
discipline) without creating results/ directories, because a bring-up check is not a benchmark evaluation.
git_info / hardware_info / utc_now are reused from apex_eval.common.

Inputs : none at import time (reads configs/eval/model.yaml for the model revision when metadata is written).
Outputs: the five files above under <repo>/logs/<run_id>/; `CheckRun.finish()` prints one PASS/FAIL line and returns
         the process exit code (0 PASS or DRYRUN_OK, 1 FAIL, 3 SKIP, 75 OOM).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from apex_eval.common import compute_rates, git_info, hardware_info, utc_now  # noqa: E402

GPU_USD_PER_HOUR = float(os.environ.get("GPU_USD_PER_HOUR", compute_rates()["gpu_vm_usd_per_hour"]))   # configs/compute.yaml unless overridden
CPU_USD_PER_HOUR = float(os.environ.get("CPU_USD_PER_HOUR", "0"))   # 0 on the GPU VM; gpu_checks.sh exports the yaml's apex-cpu rate under --cpu-dry-run

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_SKIP, EXIT_OOM = 0, 1, 2, 3, 75
STATUS_EXIT = {"PASS": EXIT_PASS, "DRYRUN_OK": EXIT_PASS, "FAIL": EXIT_FAIL, "DRYRUN_FAIL": EXIT_FAIL,
               "SKIP": EXIT_SKIP, "OOM": EXIT_OOM, "ERROR": EXIT_FAIL}


def new_run_id(name: str) -> str:
    """`<name>_<UTC timestamp>`; names look like gpucheck_t4 or serve_s6 (dryrun_ prefix for CPU dry runs)."""
    return f"{name}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def model_revision() -> dict[str, Any]:
    """Locked base-model repo/revision from configs/eval/model.yaml (read-only; best effort)."""
    try:
        import yaml

        m = yaml.safe_load((REPO_ROOT / "configs/eval/model.yaml").read_text())["model"]
        return {"repo": m.get("repo"), "revision": m.get("revision")}
    except Exception as exc:  # noqa: BLE001
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def torch_env_info() -> dict[str, Any]:
    """torch / CUDA facts for metadata.json (works without CUDA and without torch)."""
    info: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                     "cuda_version": torch.version.cuda})
        if torch.cuda.is_available():
            info.update({"device_name": torch.cuda.get_device_name(0),
                         "capability": list(torch.cuda.get_device_capability(0)),
                         "device_count": torch.cuda.device_count()})
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable ({type(exc).__name__})"
    for pkg in ("transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:  # noqa: BLE001
            info[pkg] = None
    return info


def cuda_peaks_gb() -> tuple[float | None, float | None]:
    """(max_memory_allocated, max_memory_reserved) in GB on cuda:0, or (None, None) without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return (round(torch.cuda.max_memory_allocated() / 1e9, 3), round(torch.cuda.max_memory_reserved() / 1e9, 3))
    except Exception:  # noqa: BLE001
        pass
    return None, None


class CheckRun:
    """Directories and append-only log files for one check run (see module docstring)."""

    def __init__(self, name: str, *, log_name: str = "eval.log", run_id: str | None = None,
                 root: Path | None = None, config: dict[str, Any] | None = None, seed: int | None = None):
        if log_name not in ("eval.log", "train.log"):
            raise ValueError("log_name must be eval.log or train.log")
        self.name = name
        self.run_id = run_id or new_run_id(name)
        self.run_dir = (Path(root) if root else REPO_ROOT) / "logs" / self.run_id
        if self.run_dir.exists():
            raise FileExistsError(f"run_id {self.run_id} already exists; never overwrite logs")
        self.run_dir.mkdir(parents=True)
        self.log_name = log_name
        self.t0 = time.time()
        self.start_time = utc_now()
        self.n_errors = 0
        self.write_metadata(run_id=self.run_id, kind=name, status="RUNNING", command=" ".join(sys.argv),
                            cwd=os.getcwd(), code=git_info(), hardware=hardware_info(), env=torch_env_info(),
                            model=model_revision(), config=config or {}, seed=seed, data_version="synthetic (own)",
                            start_time=self.start_time, end_time=None)

    # --- append-only logs -------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with (self.run_dir / self.log_name).open("a") as f:
            f.write(line + "\n")

    def error(self, msg: str) -> None:
        self.n_errors += 1
        self.log(f"ERROR: {msg}")
        with (self.run_dir / "errors.log").open("a") as f:
            f.write(f"[{utc_now()}] {msg}\n")

    def metric(self, record: dict[str, Any]) -> None:
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"ts": utc_now(), **record}, default=str) + "\n")

    # --- metadata / resources ------------------------------------------------------------------
    def write_metadata(self, **fields: Any) -> dict[str, Any]:
        path = self.run_dir / "metadata.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        meta.update(fields)
        path.write_text(json.dumps(meta, indent=2, default=str))
        return meta

    def write_resources(self, rows: list[tuple[str, Any, str, str]]) -> None:
        with (self.run_dir / "resources.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "unit", "note"])
            w.writerows(rows)

    def resource_rows(self, *, peak_alloc_gb: float | None = None, peak_reserved_gb: float | None = None,
                      gpu_count: int | None = None, extra: list[tuple[str, Any, str, str]] | None = None
                      ) -> list[tuple[str, Any, str, str]]:
        """Duration, peak GPU memory, GPU hours and cost for this run (resources.csv)."""
        duration = time.time() - self.t0
        if gpu_count is None:
            gpu_count = 1 if torch_env_info().get("cuda_available") else 0
        gpu_hours = duration / 3600 * gpu_count
        cost = gpu_hours * GPU_USD_PER_HOUR + duration / 3600 * CPU_USD_PER_HOUR
        if peak_alloc_gb is None and peak_reserved_gb is None:
            peak_alloc_gb, peak_reserved_gb = cuda_peaks_gb()
        rows: list[tuple[str, Any, str, str]] = [
            ("duration_sec", round(duration, 1), "s", "wall clock of this check"),
            ("peak_gpu_memory_allocated_gb", peak_alloc_gb if peak_alloc_gb is not None else "n/a", "GB",
             "torch.cuda.max_memory_allocated on cuda:0; n/a = no CUDA"),
            ("peak_gpu_memory_reserved_gb", peak_reserved_gb if peak_reserved_gb is not None else "n/a", "GB",
             "torch.cuda.max_memory_reserved on cuda:0 (what the allocator holds); n/a = no CUDA"),
            ("gpu_count", gpu_count, "count", "GPUs used by this check"),
            ("gpu_hours", round(gpu_hours, 4), "h", "duration x gpu_count"),
            ("cost_usd", round(cost, 3), "USD",
             f"gpu_hours x {GPU_USD_PER_HOUR} + cpu_hours x {CPU_USD_PER_HOUR} (GPU_USD_PER_HOUR / CPU_USD_PER_HOUR)"),
        ]
        return rows + (extra or [])

    def finish(self, status: str, summary: dict[str, Any], *, peak_alloc_gb: float | None = None,
               peak_reserved_gb: float | None = None, gpu_count: int | None = None,
               extra_resources: list[tuple[str, Any, str, str]] | None = None) -> int:
        """Write resources.csv, close metadata, finalize errors.log, print the verdict; return the exit code."""
        if status not in STATUS_EXIT:
            raise ValueError(f"unknown status {status!r}")
        self.write_resources(self.resource_rows(peak_alloc_gb=peak_alloc_gb, peak_reserved_gb=peak_reserved_gb,
                                                gpu_count=gpu_count, extra=extra_resources))
        self.write_metadata(status=status, end_time=utc_now(), summary=summary, n_errors=self.n_errors)
        err = self.run_dir / "errors.log"
        if not err.exists() or err.stat().st_size == 0:
            err.write_text("none\n")
        self.metric({"event": "finish", "status": status, **summary})
        brief = " ".join(f"{k}={v}" for k, v in summary.items() if not isinstance(v, (dict, list)))
        print(f"[{self.name}] {status} {brief}", flush=True)
        print(f"RUN_DIR={self.run_dir}", flush=True)
        return STATUS_EXIT[status]


def verdict(ok: bool, dry_run: bool) -> str:
    """PASS/FAIL, or DRYRUN_OK/DRYRUN_FAIL when GPU-only criteria were not evaluated on this box."""
    if dry_run:
        return "DRYRUN_OK" if ok else "DRYRUN_FAIL"
    return "PASS" if ok else "FAIL"


def fmt_gb(n_bytes: int | float) -> float:
    return round(n_bytes / 1e9, 3)
