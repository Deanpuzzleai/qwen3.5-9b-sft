#!/usr/bin/env python
"""T4 (K3/G3 + SDPA) one real decoder layer: forward+backward at the planned lengths, memory per token, SDPA backend, scaling.

Run from the repository root (normally via scripts/gpu_checks.sh t4):
    .venv/train/bin/python scripts/dev/gpu_checks/t4_decoder_layer.py [--config-dir <ckpt or snapshot>]
        [--seq-lens 4096,8192] [--layers 0,3] [--max-kb-per-token N] [--cpu-dry-run] [--run-name gpucheck_t4]
Inputs : the text config of the checkpoint (shapes only; weights are random, std 0.02). No checkpoint is loaded.
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl (one row per layer x T: seconds, peak bytes/token),
         errors.log, resources.csv}.
Pass (REPORT.md T4 in its dense form): Qwen3_5DecoderLayer(cfg, 0) (Gated DeltaNet + dense MLP) and (cfg, 3) (gated full
attention + dense MLP) in bf16 at T=4096 and 8192, forward+backward: loss and grads finite; the attention layer runs
under sdpa_kernel([FLASH, CUDNN, EFFICIENT]) without "No available kernel" (MATH would materialise the T x T scores);
peak scales <= 2.2x from 4K to 8K. Peak bytes/token are recorded per layer kind (gdn_torch / gdn_fla / attention); no
per-token budget exists for Qwen3.5-9B yet (to be measured at the first GPU check), so --max-kb-per-token turns the
recorded value into a gate once a budget is set. The previous base's grouped_mm-vs-eager comparison has no dense equivalent.
Method lifted from probe 05b_cpu_activation_probe.py (build_layer / run_layer) of the archived 35B training-stack study.
GPU-only for the
memory and SDPA criteria; --cpu-dry-run uses the shrunken config at short T and evaluates only finiteness.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from typing import Any

from loading import SNAPSHOT_DIR, build_tiny_cg_config, pick_device, seed_everything, text_config_of
from runlog import CheckRun, verdict

SCALING_MAX = 2.2


def build_layer(cfg, layer_idx: int, device: str, dtype, seed: int):
    """One decoder layer with random weights (lifted from 05b: normal std 0.02, bf16, frozen) + rotary embedding."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    torch.manual_seed(seed)
    with torch.device(device):
        layer = Qwen3_5DecoderLayer(cfg, layer_idx)
    for p in layer.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
        p.requires_grad_(False)
    return layer.to(dtype), Qwen3_5TextRotaryEmbedding(cfg).to(device)


def run_layer(layer, rot, cfg, seq_len: int, device: str, dtype, seed: int, sdpa_backends) -> dict[str, Any]:
    """forward+backward through the activations at T=seq_len; returns finiteness, seconds, peak bytes."""
    import torch

    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, cfg.hidden_size, device=device, dtype=dtype).requires_grad_(True)
    pos = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    cos, sin = rot(x, pos)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    ctx = sdpa_backends if sdpa_backends is not None else _nullcontext()
    with ctx:
        out = layer(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos[0])
        loss = out.float().pow(2).mean()
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base_mem) if device == "cuda" else None
    return {"grad_finite": bool(torch.isfinite(x.grad).all()), "loss_finite": bool(torch.isfinite(loss)),
            "seconds": round(seconds, 3), "peak_bytes": peak, "bytes_per_token": (peak / seq_len) if peak is not None else None}


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sdpa_context(device: str):
    """FLASH / CUDNN / EFFICIENT only (no MATH) on CUDA; None on CPU."""
    if device != "cuda":
        return None
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(SNAPSHOT_DIR), help="checkpoint or snapshot dir with config.json")
    ap.add_argument("--seq-lens", default="4096,8192")
    ap.add_argument("--layers", default="0,3", help="decoder layer indices: 0 = Gated DeltaNet, 3 = full attention")
    ap.add_argument("--max-kb-per-token", type=float, default=None,
                    help="gate peak KB/token per layer (no budget yet for Qwen3.5-9B: record first, then set it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t4")
    args = ap.parse_args()
    import torch

    run = CheckRun(args.run_name, config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    seed_everything(args.seed)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cfg = build_tiny_cg_config().text_config if args.cpu_dry_run else text_config_of(args.config_dir)
    cfg._attn_implementation = "sdpa"
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    layers = [int(s) for s in args.layers.split(",")]
    fla_present = importlib.util.find_spec("fla") is not None
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for layer_idx in layers:
        layer, rot = build_layer(cfg, layer_idx, device, dtype, args.seed)
        kind = "attention" if cfg.layer_types[layer_idx] == "full_attention" else ("gdn_fla" if fla_present else "gdn_torch")
        for seq_len in seq_lens:
            try:
                r = run_layer(layer, rot, cfg, seq_len, device, dtype, args.seed, sdpa_context(device) if kind == "attention" else None)
            except RuntimeError as exc:
                failures.append(f"layer{layer_idx} T={seq_len}: {exc}")
                run.error(f"layer {layer_idx} T={seq_len} failed: {str(exc)[:300]}")
                r = {"grad_finite": False, "loss_finite": False, "seconds": None, "bytes_per_token": None}
            row = {"event": "layer", "layer_idx": layer_idx, "kind": kind, "seq_len": seq_len, "seconds": r["seconds"],
                   "bytes_per_token": r["bytes_per_token"], "grads_finite": bool(r["grad_finite"]), "loss_finite": bool(r["loss_finite"])}
            rows.append(row)
            run.metric(row)
            run.log(f"layer {layer_idx} ({kind}) T={seq_len}: {r['seconds']}s, peak "
                    f"{r['bytes_per_token'] and round(r['bytes_per_token'] / 1e3)} KB/token, finite={row['grads_finite'] and row['loss_finite']}")
            if device == "cuda":
                torch.cuda.empty_cache()
        del layer, rot
    checks = {"no_kernel_errors": not failures, "finite": all(r["grads_finite"] and r["loss_finite"] for r in rows)}
    if device == "cuda":   # memory criteria are meaningful only at real size on the GPU
        if args.max_kb_per_token is not None:
            checks["bytes_per_token"] = all(r["bytes_per_token"] <= args.max_kb_per_token * 1e3 for r in rows if r["bytes_per_token"])
        scaling_ok = True
        for layer_idx in layers:
            peaks = {r["seq_len"]: r["bytes_per_token"] * r["seq_len"] for r in rows if r["layer_idx"] == layer_idx and r["bytes_per_token"]}
            if len(peaks) >= 2:   # <= 2.2x per doubling of T
                lo, hi = min(peaks), max(peaks)
                scaling_ok &= peaks[hi] / peaks[lo] <= SCALING_MAX ** math.log2(hi / lo)
        checks["scaling_le_2.2x"] = scaling_ok
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    summary = {"fla_present": fla_present, "rows": len(rows),
               "kb_per_token": {f"L{r['layer_idx']}_T{r['seq_len']}": round(r["bytes_per_token"] / 1e3) for r in rows if r["bytes_per_token"]},
               "kb_per_token_budget": args.max_kb_per_token if args.max_kb_per_token is not None else "none yet (first GPU check records it)",
               "checks": checks}
    return run.finish(verdict(ok, args.cpu_dry_run), summary)


if __name__ == "__main__":
    sys.exit(main())
