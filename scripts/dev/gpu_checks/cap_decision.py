#!/usr/bin/env python
"""Row-cap decision of milestone M4.1 from the `caps` step of scripts/gpu_checks.sh (docs/DATA_PLAN.md section 4, "Row cap").

Run from the repository root (normally via scripts/gpu_checks.sh caps, which runs T7 at both caps and T8 first):
    .venv/train/bin/python scripts/dev/gpu_checks/cap_decision.py --t7 49152=logs/<t7 run> --t7 65536=logs/<t7 run>
        --t8 logs/<t8 run> [--target 65536] [--fallback 49152] [--headroom 0.90] [--gpu-total-gb GB] [--cpu-dry-run]
        [--run-name gpucheck_caps]
Inputs : the run directories of T7 (t7_lora_smoke.py, one per cap: 20 LoRA steps on rows of exactly the cap) and of T8
         (t8_oom_boundary.py over fallback, target and 1.5 x target); the GPU's total memory (torch.cuda.mem_get_info on
         cuda:0, or --gpu-total-gb).
Rule   : the target cap (64K) whenever it fits with headroom, else the fallback (48K); if neither fits, FAIL (no cap). A cap
         fits when T7 passed at it, T8's attempt at it passed, and the peak reserved GPU memory at it (the larger of the T7
         and T8 values) is at most --headroom of the GPU's total memory: the rest absorbs allocator fragmentation over a
         multi-hour run with variable row lengths (the training scripts run with expandable_segments, as T8 does). The
         1.5 x target attempt is reported as extra margin and is not required. Dry runs on the CPU twin evaluate the
         statuses only (no GPU memory).
Outputs: logs/<run_id>/{metadata.json (summary: cap, rule, per-cap status, peak reserved GB, tokens/s mean and steady
         median), eval.log, metrics.jsonl (one record per cap), errors.log, resources.csv}; prints `CAP=<n>` and the
         tokens/s at each cap. Exit 0 when a cap was chosen, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from runlog import CheckRun, verdict

OK_STATUSES = ("PASS", "DRYRUN_OK")
WARMUP_STEPS = 2   # T7's first steps include kernel compilation and allocator growth; the steady rate skips them


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def t7_facts(run_dir: Path) -> dict[str, Any]:
    """Status, peak reserved GB and tokens/s (mean over all steps; steady = median after the warm-up steps) of one T7 run."""
    meta = read_json(run_dir / "metadata.json")
    summary = meta.get("summary") or {}
    steps = [r for r in read_jsonl(run_dir / "metrics.jsonl") if r.get("event") == "train" and r.get("tokens_per_s")]
    steady = [r["tokens_per_s"] for r in steps if int(r.get("step", 0)) > WARMUP_STEPS]
    return {"run_dir": str(run_dir), "status": meta.get("status", "MISSING"), "steps": summary.get("steps"),
            "peak_reserved_gb": summary.get("peak_reserved_gb"), "tokens_per_s_mean": summary.get("mean_tokens_per_s"),
            "tokens_per_s_steady": round(statistics.median(steady), 1) if steady else None}


def t8_attempts(run_dir: Path) -> dict[int, dict[str, Any]]:
    """seq_len -> {status, peak_reserved_gb} of T8's micro-batch-1 attempts (attempts stop at the first failure)."""
    out: dict[int, dict[str, Any]] = {}
    for r in read_jsonl(run_dir / "metrics.jsonl"):
        if r.get("event") == "attempt" and int(r.get("batch", 1)) == 1:
            out[int(r["seq_len"])] = {"status": r.get("status"), "peak_reserved_gb": r.get("peak_gpu_memory_reserved_gb")}
    return out


def decide(t7: dict[int, dict[str, Any]], t8: dict[int, dict[str, Any]], *, target: int, fallback: int, margin: int,
           headroom: float, gpu_total_gb: float | None) -> dict[str, Any]:
    """Apply the rule; returns {cap (or None), per_cap: {...}, margin_attempt, rule}."""
    limit = headroom * gpu_total_gb if gpu_total_gb else None
    per_cap: dict[int, dict[str, Any]] = {}
    for cap in (target, fallback):
        a, b = t7.get(cap, {"status": "MISSING"}), t8.get(cap, {"status": "NOT_RUN"})
        peaks = [p for p in (a.get("peak_reserved_gb"), b.get("peak_reserved_gb")) if isinstance(p, (int, float))]
        peak = max(peaks) if peaks else None
        reasons = []
        if a["status"] not in OK_STATUSES:
            reasons.append(f"T7 {a['status']}")
        if b["status"] not in OK_STATUSES:
            reasons.append(f"T8 {b['status']}")
        if limit is not None and (peak is None or peak > limit):
            reasons.append(f"peak reserved {peak} GB > {headroom:.0%} of {gpu_total_gb} GB")
        per_cap[cap] = {"fits": not reasons, "why_not": reasons, "peak_reserved_gb": peak, "headroom_limit_gb": round(limit, 1) if limit else None,
                        "t7_status": a["status"], "t8_status": b["status"], "tokens_per_s_mean": a.get("tokens_per_s_mean"),
                        "tokens_per_s_steady": a.get("tokens_per_s_steady"), "t7_run_dir": a.get("run_dir")}
    cap = target if per_cap[target]["fits"] else (fallback if per_cap[fallback]["fits"] else None)
    return {"cap": cap, "per_cap": per_cap, "margin_attempt": {"seq_len": margin, **t8.get(margin, {"status": "NOT_RUN"})},
            "rule": f"{target} if it fits with headroom (T7 and T8 pass, peak reserved <= {headroom} x GPU memory), else {fallback}, else none"}


def gpu_total_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.mem_get_info(0)[1] / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def parse_t7(items: list[str]) -> dict[int, Path]:
    out = {}
    for item in items:
        seq, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--t7 {item!r}: expected SEQ_LEN=RUN_DIR")
        out[int(seq)] = Path(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t7", action="append", default=[], metavar="SEQ_LEN=RUN_DIR", help="a T7 run at one cap (repeat per cap)")
    ap.add_argument("--t8", required=True, metavar="RUN_DIR", help="the T8 boundary run over fallback, target and margin")
    ap.add_argument("--target", type=int, default=65536)
    ap.add_argument("--fallback", type=int, default=49152)
    ap.add_argument("--margin", type=int, default=98304, help="the 1.5 x target attempt, reported only")
    ap.add_argument("--headroom", type=float, default=0.90, help="max share of GPU memory the peak reserved may use at the chosen cap")
    ap.add_argument("--gpu-total-gb", type=float, default=None, help="default: torch.cuda.mem_get_info on cuda:0")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_caps")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args))
    gpu_total = args.gpu_total_gb if args.gpu_total_gb else (None if args.cpu_dry_run else gpu_total_memory_gb())
    if gpu_total is None and not args.cpu_dry_run:
        run.error("GPU total memory unknown (no CUDA and no --gpu-total-gb): the headroom criterion cannot be evaluated")
        return run.finish("FAIL", {"cap": None, "reason": "no GPU memory figure"})
    t7 = {seq: t7_facts(path) for seq, path in parse_t7(args.t7).items()}
    t8 = t8_attempts(Path(args.t8))
    result = decide(t7, t8, target=args.target, fallback=args.fallback, margin=args.margin, headroom=args.headroom, gpu_total_gb=gpu_total)
    for cap, facts in result["per_cap"].items():
        run.metric({"event": "cap", "seq_len": cap, **facts})
        run.log(f"cap {cap}: {'fits' if facts['fits'] else 'does not fit: ' + '; '.join(facts['why_not'])}; peak reserved "
                f"{facts['peak_reserved_gb']} GB (limit {facts['headroom_limit_gb']}), tokens/s mean {facts['tokens_per_s_mean']} steady "
                f"{facts['tokens_per_s_steady']}")
    run.log(f"margin attempt {args.margin}: {result['margin_attempt']}")
    if result["cap"] is None:
        run.error("neither cap fits: stop for a manual decision (DATA_PLAN section 4: the gate decides the cap)")
    else:
        run.log(f"decision: CAP={result['cap']} (configs/sft.yaml data.max_length and configs/stage2.yaml filtered_sft.max_length; "
                f"tokens/s {result['per_cap'][result['cap']]['tokens_per_s_steady']} for the budget rule)")
    status = verdict(result["cap"] is not None, args.cpu_dry_run)
    code = run.finish(status, {"cap": result["cap"], "gpu_total_gb": gpu_total, **result})
    print(f"CAP={result['cap']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
