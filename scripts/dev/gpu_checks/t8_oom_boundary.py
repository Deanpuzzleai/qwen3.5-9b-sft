#!/usr/bin/env python
"""T8 (K6) OOM boundary: repeat the T7 run with doubling sequence length (and optionally micro-batch 2) until OOM.

Run from the repository root (normally via scripts/gpu_checks.sh t8):
    .venv/train/bin/python scripts/dev/gpu_checks/t8_oom_boundary.py --model-dir <ckpt> --planned-seq-len 8192
        [--seq-lens 49152,65536,98304] [--max-seq-len 131072] [--steps 3] [--min-pass-ratio 1.5] [--try-batch-2] [--cpu-dry-run]
Inputs : the checkpoint directory; each attempt is a fresh subprocess of t7_lora_smoke.py (clean CUDA context)
         with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and --no-criteria (only OOM/no-OOM matters).
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl (one row per attempt: seq_len, batch, status,
         child run_dir, peak reserved GB), errors.log, resources.csv (boundary)}; each child writes its own run dir.
Pass: the largest passing sequence length >= planned T x --min-pass-ratio (default 1.5, REPORT.md T8); the boundary is
recorded. scripts/gpu_checks.sh caps runs 49152,65536,98304 with planned 49152 and ratio 1.0 (at least the fallback cap
trains) and leaves the cap choice to cap_decision.py.
GPU-only; --cpu-dry-run runs two short lengths on the tiny twin.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from runlog import EXIT_OOM, CheckRun, verdict

T7 = Path(__file__).resolve().parent / "t7_lora_smoke.py"
RUN_DIR_RE = re.compile(r"^RUN_DIR=(.+)$", re.MULTILINE)


def child_peaks(run_dir: Path) -> dict[str, Any]:
    """peak reserved/allocated GB from a child's resources.csv (n/a -> None)."""
    out: dict[str, Any] = {}
    path = run_dir / "resources.csv"
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        if row["metric"] in ("peak_gpu_memory_reserved_gb", "peak_gpu_memory_allocated_gb", "tokens_per_s_mean"):
            out[row["metric"]] = None if row["value"] == "n/a" else float(row["value"])
    return out


def attempt(run: CheckRun, args: argparse.Namespace, seq_len: int, batch: int) -> dict[str, Any]:
    """One subprocess T7 run; returns {seq_len, batch, status, run_dir, peaks}."""
    cmd = [sys.executable, str(T7), "--model-dir", args.model_dir, "--seq-len", str(seq_len), "--batch", str(batch),
           "--max-steps", str(args.steps), "--rank", str(args.rank), "--alpha", str(args.alpha), "--no-criteria",
           "--allow-fallback", "--run-name", f"{args.run_name}_child_T{seq_len}_b{batch}"]
    if args.cpu_dry_run:
        cmd.append("--cpu-dry-run")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    run.log(f"attempt seq_len={seq_len} batch={batch}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    m = RUN_DIR_RE.search(proc.stdout)
    child_dir = Path(m.group(1)) if m else None
    status = "PASS" if proc.returncode == 0 else ("OOM" if proc.returncode == EXIT_OOM else "ERROR")
    if status == "ERROR":
        run.error(f"child failed rc={proc.returncode}: {proc.stderr[-1500:]}")
    rec = {"event": "attempt", "seq_len": seq_len, "batch": batch, "status": status, "child_run_dir": str(child_dir),
           **(child_peaks(child_dir) if child_dir else {})}
    run.metric(rec)
    run.log(f"result seq_len={seq_len} batch={batch}: {status} peak_reserved_gb={rec.get('peak_gpu_memory_reserved_gb')}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--planned-seq-len", type=int, default=8192)
    ap.add_argument("--seq-lens", default=None, help="explicit comma list; default doubles from the planned length")
    ap.add_argument("--max-seq-len", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--min-pass-ratio", type=float, default=1.5, help="pass when the largest passing length >= planned x this")
    ap.add_argument("--try-batch-2", action="store_true", help="also try micro-batch 2 at the planned length")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t8")
    args = ap.parse_args()
    run = CheckRun(args.run_name, log_name="train.log", config=vars(args))
    if args.seq_lens:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
    else:
        seq_lens, t = [], args.planned_seq_len
        while t <= args.max_seq_len:
            seq_lens.append(t)
            t *= 2
    attempts: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        rec = attempt(run, args, seq_len, 1)
        attempts.append(rec)
        if rec["status"] != "PASS":
            break
    if args.try_batch_2:
        attempts.append(attempt(run, args, args.planned_seq_len, 2))
    passing = [a["seq_len"] for a in attempts if a["status"] == "PASS" and a["batch"] == 1]
    largest = max(passing) if passing else 0
    first_oom = next((a["seq_len"] for a in attempts if a["status"] == "OOM"), None)
    errors = [a for a in attempts if a["status"] == "ERROR"]
    ok = largest >= args.min_pass_ratio * args.planned_seq_len and not errors
    if not ok and not args.cpu_dry_run:
        run.error(f"largest passing seq_len {largest} < {args.min_pass_ratio} x planned {args.planned_seq_len} or child errors {len(errors)}")
    summary = {"largest_passing_seq_len": largest, "first_oom_seq_len": first_oom, "planned_seq_len": args.planned_seq_len,
               "attempts": [(a["seq_len"], a["batch"], a["status"], a.get("peak_gpu_memory_reserved_gb")) for a in attempts]}
    extra = [("largest_passing_seq_len", largest, "tokens", "micro-batch 1, 3 steps, expandable_segments"),
             ("first_oom_seq_len", first_oom if first_oom else "n/a", "tokens", "first attempt that hit CUDA OOM")]
    return run.finish(verdict(ok, args.cpu_dry_run), summary, gpu_count=0 if args.cpu_dry_run else 1, extra_resources=extra)


if __name__ == "__main__":
    sys.exit(main())
