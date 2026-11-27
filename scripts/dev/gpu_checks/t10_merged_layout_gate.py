#!/usr/bin/env python
"""T10 (G11) merged-layout gate: key set and shapes of a merged save == the base index minus mtp.*, plus config checks.

Run from the repository root (normally via scripts/gpu_checks.sh t10; also the pre-upload gate of the merge script):
    .venv/train/bin/python scripts/dev/gpu_checks/t10_merged_layout_gate.py --merged-dir <dir>
        [--reference-dir /mnt/apex-shared/models/Qwen3.5-9B@c2022362...] [--expect-architecture
        Qwen3_5ForConditionalGeneration] [--check-dtype] [--cpu-dry-run] [--run-name gpucheck_t10]
Inputs : safetensors headers only (never tensor data) of both directories: model.safetensors.index.json + shards,
         or a single model.safetensors; config.json of the merged dir. Header scan lifted from
         scan_disk() in probe 00_env_check.py of the archived 35B training-stack study.
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log, resources.csv}.
Pass (REPORT.md T10): identical key set and shapes vs the reference minus the 15 `mtp.*` tensors; dense 2-D
mlp.{gate,up,down}_proj in every decoder layer; `architectures ==
["Qwen3_5ForConditionalGeneration"]`; `text_config.mamba_ssm_dtype == "float32"` present;
`model.visual.*` present; tokenizer / chat_template.jinja copied next to the weights; generation_config.json
mirrors the reference (present and identical iff the reference has one, as apex_train.merge writes it)
(vLLM serves the directory as-is). Without --check-dtype a T9 save shows 48 dtype mismatches against the real base
(linear_attn.A_log / linear_attn.norm.weight are fp32 on the Hub and bf16 after a bf16 load); the production merge
(src/apex_train/merge.py) restores them and its gate requires per-key dtype equality. CPU-safe (headers only);
--cpu-dry-run only relaxes the architecture expectation for the tiny twin when passed explicitly.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

from loading import CG_CLASS, DEFAULT_MODEL_DIR, MTP_RE
from runlog import CheckRun, verdict

REQUIRED_AUX = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MIRRORED_AUX = ("generation_config.json",)   # in the merged dir iff in the reference, and then identical (apex_train.merge)
OPTIONAL_AUX = ("vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json")
MLP_RE = re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def scan_headers(model_dir: Path) -> dict[str, dict[str, Any]]:
    """{key: {dtype, shape}} from the index + shard headers (or the single model.safetensors)."""
    index = model_dir / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    tensors: dict[str, dict[str, Any]] = {}
    for shard in shards:
        with (model_dir / shard).open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                tensors[k] = {"dtype": v["dtype"], "shape": list(v["shape"]), "shard": shard}
    if index.exists() and set(tensors) != set(json.loads(index.read_text())["weight_map"]):
        raise RuntimeError(f"{model_dir}: index weight_map and shard headers disagree")
    return tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--reference-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--expect-architecture", default=CG_CLASS)
    ap.add_argument("--check-dtype", action="store_true", help="also require identical dtypes (bf16 base vs bf16 merge)")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t10")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args))
    merged_dir, ref_dir = Path(args.merged_dir), Path(args.reference_dir)
    ref = {k: v for k, v in scan_headers(ref_dir).items() if not MTP_RE.match(k)}
    mer = scan_headers(merged_dir)
    missing = sorted(set(ref) - set(mer))
    extra = sorted(set(mer) - set(ref))
    common = sorted(set(ref) & set(mer))
    shape_mismatch = [k for k in common if ref[k]["shape"] != mer[k]["shape"]]
    dtype_mismatch = [k for k in common if ref[k]["dtype"] != mer[k]["dtype"]]
    cfg = json.loads((merged_dir / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    mlp_keys = [k for k in mer if MLP_RE.search(k)]
    dense = bool(mlp_keys) and all(len(mer[k]["shape"]) == 2 for k in mlp_keys)
    aux_missing = [f for f in REQUIRED_AUX if not (merged_dir / f).exists()]
    aux_missing += [f for f in MIRRORED_AUX if (ref_dir / f).exists() != (merged_dir / f).exists()
                    or ((ref_dir / f).exists() and (ref_dir / f).read_bytes() != (merged_dir / f).read_bytes())]
    aux_optional_missing = [f for f in OPTIONAL_AUX if not (merged_dir / f).exists()]
    facts = {"n_reference_keys": len(ref), "n_merged_keys": len(mer), "n_missing": len(missing), "n_extra": len(extra),
             "n_shape_mismatch": len(shape_mismatch), "n_dtype_mismatch": len(dtype_mismatch), "architectures": cfg.get("architectures"),
             "mamba_ssm_dtype": text_cfg.get("mamba_ssm_dtype"), "n_visual_keys": sum(k.startswith("model.visual.") for k in mer),
             "mtp_keys_in_merged": sum(bool(MTP_RE.match(k)) for k in mer), "dense_mlp": dense, "n_mlp_tensors": len(mlp_keys),
             "aux_missing": aux_missing, "aux_optional_missing": aux_optional_missing,
             "merged_dtypes": sorted({v["dtype"] for v in mer.values()})}
    run.log(f"facts: {facts}")
    run.metric({"event": "diff", **facts, "missing_sample": missing[:5], "extra_sample": extra[:5], "shape_mismatch_sample": shape_mismatch[:5]})
    checks = {"same_key_set": not missing and not extra, "same_shapes": not shape_mismatch,
              "same_dtypes": not dtype_mismatch or not args.check_dtype, "dense_mlp": dense,
              "architecture": cfg.get("architectures") == [args.expect_architecture],
              "mamba_ssm_dtype_float32": text_cfg.get("mamba_ssm_dtype") == "float32",
              "visual_present": facts["n_visual_keys"] > 0 or args.expect_architecture != CG_CLASS,
              "aux_files": not aux_missing}
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}; missing[:5]={missing[:5]} extra[:5]={extra[:5]} "
                  f"shape_mismatch[:5]={shape_mismatch[:5]} aux_missing={aux_missing}")
    summary = {"n_missing": len(missing), "n_extra": len(extra), "n_shape_mismatch": len(shape_mismatch),
               "architectures": cfg.get("architectures"), "dense_mlp": dense, "checks": checks}
    return run.finish(verdict(ok, args.cpu_dry_run), summary, gpu_count=0)


if __name__ == "__main__":
    sys.exit(main())
