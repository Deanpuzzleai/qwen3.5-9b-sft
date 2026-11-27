#!/usr/bin/env python
"""T1 (K1/G1) CUDA stack: torch build, CUDA version, device capability, arch list, free memory, driver, Triton.

Run from the repository root (normally via scripts/gpu_checks.sh t1):
    .venv/train/bin/python scripts/dev/gpu_checks/t1_cuda_stack.py [--expect-torch 2.14.1+cu130]
        [--min-free-gb auto] [--cpu-dry-run] [--run-name gpucheck_t1]
Inputs : none (reads the CUDA device). Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log,
         resources.csv}; prints PASS/FAIL with the measured numbers.
Pass (REPORT.md T1): torch == expected build (default 2.14.1+cu130), capability (10,0) B200 or (9,0) H200, the
matching sm_100 / sm_90 in torch.cuda.get_arch_list(), free memory >= 170 GB (B200) / 135 GB (H200), Triton importable.
GPU-only: with --cpu-dry-run the facts are logged and the GPU criteria are reported as not evaluated.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from runlog import CheckRun, verdict

CAPABILITIES = {(10, 0): ("B200", "sm_100", 170.0), (9, 0): ("H200", "sm_90", 135.0)}


def nvidia_smi() -> dict[str, str]:
    """Driver version and GPU name/memory from nvidia-smi (empty if absent)."""
    if not shutil.which("nvidia-smi"):
        return {}
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    parts = [p.strip() for p in out.splitlines()[0].split(",")] if out.strip() else []
    keys = ("gpu_name", "driver_version", "memory_total_mib", "memory_used_mib")
    return dict(zip(keys, parts))


def collect() -> dict:
    import torch

    facts: dict = {"torch": torch.__version__, "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    try:
        import triton

        facts["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        facts["triton"] = f"unavailable ({type(exc).__name__})"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        facts.update({"device_name": torch.cuda.get_device_name(0), "capability": tuple(torch.cuda.get_device_capability(0)),
                      "arch_list": torch.cuda.get_arch_list(), "free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2),
                      "device_count": torch.cuda.device_count()})
    facts.update(nvidia_smi())
    return facts


def evaluate(facts: dict, expect_torch: str, min_free_gb: float | None) -> tuple[bool, dict[str, bool]]:
    cap = tuple(facts.get("capability", ()))
    card = CAPABILITIES.get(cap)
    checks = {
        "torch_build": facts["torch"] == expect_torch,
        "cuda_available": bool(facts["cuda_available"]),
        "capability_known": card is not None,
        "arch_in_list": bool(card) and card[1] in facts.get("arch_list", []),
        "triton_importable": not str(facts["triton"]).startswith("unavailable"),
    }
    threshold = min_free_gb if min_free_gb is not None else (card[2] if card else None)
    checks["free_memory"] = threshold is not None and facts.get("free_gb", 0) >= threshold
    return all(checks.values()), checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-torch", default="2.14.1+cu130")
    ap.add_argument("--min-free-gb", type=float, default=None, help="override the per-card threshold (170 B200 / 135 H200)")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t1")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args))
    facts = collect()
    run.log(f"facts: {facts}")
    run.metric({"event": "facts", **facts})
    ok, checks = evaluate(facts, args.expect_torch, args.min_free_gb)
    for k, v in checks.items():
        run.log(f"check {k}: {'ok' if v else 'FAILED'}")
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    summary = {"torch": facts["torch"], "capability": facts.get("capability"), "free_gb": facts.get("free_gb"),
               "arch_ok": checks["arch_in_list"], "triton": facts["triton"], "checks": checks}
    status = verdict(ok if not args.cpu_dry_run else True, args.cpu_dry_run)
    return run.finish(status, summary)


if __name__ == "__main__":
    sys.exit(main())
