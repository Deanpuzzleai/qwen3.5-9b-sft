#!/usr/bin/env python
"""T5 (K4/G4) full load of the checkpoint into the served class on one GPU.

Run from the repository root (normally via scripts/gpu_checks.sh t5):
    .venv/train/bin/python scripts/dev/gpu_checks/t5_full_load.py --model-dir <ckpt> [--expect-unexpected 15]
        [--expect-alloc-gb 18.82 --alloc-tol 0.03] [--cpu-dry-run] [--run-name gpucheck_t5]
Inputs : the checkpoint directory. Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log,
         resources.csv (load seconds, allocated GB)}.
Pass (REPORT.md T5): AutoModelForImageTextToText.from_pretrained(dir, dtype=bf16, device_map="cuda",
output_loading_info=True) -> Qwen3_5ForConditionalGeneration, missing_keys == [], the checkpoint's 15 `mtp.*` tensors
dropped and nothing else (transformers 5.19 drops them silently through the class's `_keys_to_ignore_on_load_unexpected
= ['^mtp.*']`, so they are counted from the checkpoint's own keys; a version that reports them must report exactly those),
no "newly initialized" warning, _weight_conversions == [], sdpa selected, ~18.8 GB
allocated (the bf16 weights of the HF class without mtp.*: 18,819,635,168 B computed from the index; the measured
allocation is confirmed at the first GPU check), load time logged. GPU-only; --cpu-dry-run loads the tiny twin
(0 unexpected, no memory check).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from loading import CG_CLASS, WarningCapture, auto_class_for, load_model, pick_device
from runlog import CheckRun, cuda_peaks_gb, verdict


def checkpoint_mtp_keys(model_dir: str) -> list[str]:
    """`mtp.*` tensor names of the checkpoint, from model.safetensors.index.json or the single file's header."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        keys = list(json.loads(index.read_text())["weight_map"])
    else:
        with (d / "model.safetensors").open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    return sorted(k for k in keys if k.startswith("mtp."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--expect-unexpected", type=int, default=15, help="number of mtp.* keys the checkpoint holds and the load must drop (0 for merged dirs / twin)")
    ap.add_argument("--expect-alloc-gb", type=float, default=18.82, help="bf16 weight bytes of the HF class without mtp.* (from the index)")
    ap.add_argument("--alloc-tol", type=float, default=0.03, help="relative tolerance on allocated GB")
    ap.add_argument("--allow-text-only-class", action="store_true", help="dry runs on the text-only twin only")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t5")
    args = ap.parse_args()
    import torch

    run = CheckRun(args.run_name, config=vars(args))
    device = pick_device(args.cpu_dry_run)
    auto_class = auto_class_for(args.model_dir)
    with WarningCapture() as cap:
        model, info, seconds = load_model(args.model_dir, device=device,
                                          dtype_name="bfloat16" if device == "cuda" else "float32", auto_class=auto_class)
    alloc = round(torch.cuda.memory_allocated() / 1e9, 2) if device == "cuda" else None
    init_warnings = cap.grep(r"newly initialized|not initialized|were not used")
    run.log(f"class={info['class']} seconds={info['seconds']} missing={len(info['missing'])} unexpected={len(info['unexpected'])} "
            f"(non-mtp {len(info['unexpected_non_mtp'])}) mismatched={len(info['mismatched'])} conversions={info['weight_conversions']} "
            f"attn={info['attn_implementation']} alloc_gb={alloc}")
    run.metric({"event": "load", **{k: v for k, v in info.items() if k not in ("missing", "unexpected")},
                "n_missing": len(info["missing"]), "n_unexpected": len(info["unexpected"]), "allocated_gb": alloc,
                "init_warnings": init_warnings[:5]})
    # The checkpoint's mtp.* tensors must be dropped: either reported as unexpected or ignored by the class pattern.
    mtp_keys = checkpoint_mtp_keys(args.model_dir)
    ignore = getattr(type(model), "_keys_to_ignore_on_load_unexpected", None) or []
    if info["unexpected"]:
        mtp_dropped = len(info["unexpected"]) if sorted(info["unexpected"]) == mtp_keys else -1
    else:
        mtp_dropped = len(mtp_keys) if all(any(re.search(p, k) for p in ignore) for k in mtp_keys) else -1
    run.log(f"checkpoint mtp.* keys={len(mtp_keys)} reported_unexpected={len(info['unexpected'])} class_ignore={ignore} "
            f"-> dropped={mtp_dropped}")
    checks = {
        "served_class": info["class"] == CG_CLASS or (args.allow_text_only_class and args.cpu_dry_run),
        "no_missing": not info["missing"],
        "unexpected_only_mtp": not info["unexpected_non_mtp"],
        "unexpected_count": mtp_dropped == args.expect_unexpected,
        "no_mismatch": not info["mismatched"],
        "no_init_warning": not init_warnings,
        "no_weight_conversions": not info["weight_conversions"],
        "sdpa": info["attn_implementation"] == "sdpa",
    }
    if device == "cuda":
        checks["allocated_gb"] = abs(alloc - args.expect_alloc_gb) <= args.alloc_tol * args.expect_alloc_gb
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}; missing[:5]={info['missing'][:5]} "
                  f"unexpected_non_mtp[:5]={info['unexpected_non_mtp'][:5]}")
    peak_alloc, peak_reserved = cuda_peaks_gb()
    summary = {"class": info["class"], "load_seconds": info["seconds"], "n_missing": len(info["missing"]),
               "n_unexpected": len(info["unexpected"]), "allocated_gb": alloc, "checks": checks}
    extra = [("load_seconds", info["seconds"], "s", "from_pretrained wall time"),
             ("allocated_gb_after_load", alloc if alloc is not None else "n/a", "GB", "torch.cuda.memory_allocated right after load")]
    return run.finish(verdict(ok, args.cpu_dry_run), summary, peak_alloc_gb=peak_alloc, peak_reserved_gb=peak_reserved,
                      extra_resources=extra)


if __name__ == "__main__":
    sys.exit(main())
