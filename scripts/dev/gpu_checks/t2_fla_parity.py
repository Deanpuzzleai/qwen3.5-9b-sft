#!/usr/bin/env python
"""T2 (K2/G2/G12) flash-linear-attention + causal-conv1d: imports, kernel-vs-reference parity, no fallback warning.

Run from the repository root (normally via scripts/gpu_checks.sh t2):
    .venv/train/bin/python scripts/dev/gpu_checks/t2_fla_parity.py [--batch 2 --seq-len 512 --heads 32 --head-dim 128]
        [--allow-missing-causal-conv1d] [--cpu-dry-run] [--run-name gpucheck_t2]
Inputs : none (random bf16 tensors; a tiny GatedDeltaNet layer built from the shrunken config).
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log, resources.csv}.
Pass (REPORT.md T2): `import fla, triton, causal_conv1d` succeed; transformers' torch reference
`torch_chunk_gated_delta_rule` (unwrapped from the hub-kernel decorator) vs `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
on bf16 inputs B=2, T=512, 32 heads x 128 (the 9B's 32 value heads x 128; q/k already repeated to the value-head count, as the layer does),
use_qk_l2norm_in_kernel=True, random g <= 0 and beta in (0,1): max abs diff < 2e-2, relative Frobenius < 1e-2,
both backward passes finite; and a GatedDeltaNet forward emits no "falling back to its reference PyTorch
implementation" warning for chunk_gated_delta_rule (nor for causal_conv1d_fn unless --allow-missing-causal-conv1d).
GPU-only: the kernel needs Triton. --cpu-dry-run runs the reference only (plumbing test; the fallback warning is
then expected and recorded, not failed).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import time

from loading import WarningCapture, build_tiny_cg_config, callable_kwargs, pick_device, seed_everything
from runlog import CheckRun, verdict

MAX_ABS_TOL, REL_FRO_TOL = 2e-2, 1e-2


def try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, str(getattr(mod, "__version__", "ok"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def reference_fn():
    """transformers' pure-torch chunked gated delta rule, unwrapped from use_kernel_func_from_hub_with_fallback."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm

    fn = inspect.unwrap(mm.torch_chunk_gated_delta_rule)
    assert fn is not mm.torch_chunk_gated_delta_rule, "unwrap did not reach the reference function"
    return fn


def kernel_fn():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def make_inputs(batch: int, seq_len: int, heads: int, head_dim: int, device: str, seed: int) -> dict:
    """Inputs with the layer's dtypes: q/k/v bf16 leaves, g fp32 <= 0 (log-decay), beta = sigmoid(.) bf16."""
    import torch

    seed_everything(seed)
    dt = torch.bfloat16 if device == "cuda" else torch.float32   # CPU reference-only path stays fp32
    shape = (batch, seq_len, heads, head_dim)
    q = torch.randn(shape, device=device, dtype=dt).requires_grad_(True)
    k = torch.randn(shape, device=device, dtype=dt).requires_grad_(True)
    v = torch.randn(shape, device=device, dtype=dt).requires_grad_(True)
    g = -torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads, device=device)).to(dt)
    upstream = torch.randn(shape, device=device, dtype=torch.float32)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "upstream": upstream}


def run_fn(fn, inp: dict) -> tuple:
    """Forward + backward through fn with the transformers call convention; returns (out fp32, grads, seconds)."""
    import torch

    q, k, v = (t.detach().clone().requires_grad_(True) for t in (inp["q"], inp["k"], inp["v"]))
    kwargs = callable_kwargs(fn, {"g": inp["g"], "beta": inp["beta"], "initial_state": None,
                                  "output_final_state": False, "use_qk_l2norm_in_kernel": True})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(q, k, v, **kwargs)
    out = out[0] if isinstance(out, tuple) else out
    (out.float() * inp["upstream"]).sum().backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out.detach().float(), [t.grad.detach().float() for t in (q, k, v)], time.perf_counter() - t0


def compare(ref_out, ker_out, ref_grads, ker_grads) -> dict:
    import torch

    diff = (ref_out - ker_out).abs()
    rel = (diff.pow(2).sum().sqrt() / ref_out.pow(2).sum().sqrt().clamp_min(1e-12)).item()
    grad_rel = [((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt().clamp_min(1e-12)).item()
                for a, b in zip(ref_grads, ker_grads)]
    return {"max_abs_diff": diff.max().item(), "rel_frobenius": rel, "ref_grads_finite": all(torch.isfinite(g).all().item() for g in ref_grads),
            "kernel_grads_finite": all(torch.isfinite(g).all().item() for g in ker_grads), "grad_rel_frobenius_qkv": grad_rel}


def fallback_probe(device: str) -> list[str]:
    """Forward one tiny Qwen3_5GatedDeltaNet and return the kernels that fell back to the torch reference."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    tc = build_tiny_cg_config().text_config
    tc._attn_implementation = "sdpa"
    layer = Qwen3_5GatedDeltaNet(tc, 0).to(device, torch.bfloat16 if device == "cuda" else torch.float32)
    x = torch.randn(1, 64, tc.hidden_size, device=device, dtype=next(layer.parameters()).dtype)
    with WarningCapture() as cap:
        layer(x)
    return cap.fallback_functions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--heads", type=int, default=32, help="value heads (q/k are repeated to this count by the layer)")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing-causal-conv1d", action="store_true", help="causal_conv1d is optional (cuDNN conv fallback)")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t2")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    imports = {name: try_import(name) for name in ("fla", "triton", "causal_conv1d")}
    for name, (ok, detail) in imports.items():
        run.log(f"import {name}: {'ok ' + detail if ok else 'MISSING ' + detail}")
    run.metric({"event": "imports", **{k: v[1] for k, v in imports.items()}})

    inp = make_inputs(args.batch, args.seq_len, args.heads, args.head_dim, device, args.seed)
    ref_out, ref_grads, ref_s = run_fn(reference_fn(), inp)
    run.log(f"reference fwd+bwd {ref_s:.3f}s, output finite={bool(ref_out.isfinite().all())}")
    parity: dict = {"evaluated": False}
    if imports["fla"][0]:
        ker_out, ker_grads, ker_s = run_fn(kernel_fn(), inp)
        parity = {"evaluated": True, "kernel_seconds": round(ker_s, 4), "reference_seconds": round(ref_s, 4),
                  **compare(ref_out, ker_out, ref_grads, ker_grads)}
        run.log(f"parity: {parity}")
    else:
        run.log("fla not importable: kernel parity not evaluated")
    run.metric({"event": "parity", **parity})

    fallbacks = fallback_probe(device)
    run.log(f"GatedDeltaNet forward fallbacks: {fallbacks or 'none'}")
    run.metric({"event": "fallback_probe", "fallback_functions": fallbacks})

    checks = {
        "fla_import": imports["fla"][0], "triton_import": imports["triton"][0],
        "causal_conv1d_import": imports["causal_conv1d"][0] or args.allow_missing_causal_conv1d,
        "parity_max_abs": parity.get("evaluated", False) and parity["max_abs_diff"] < MAX_ABS_TOL,
        "parity_rel_fro": parity.get("evaluated", False) and parity["rel_frobenius"] < REL_FRO_TOL,
        "grads_finite": parity.get("evaluated", False) and parity["ref_grads_finite"] and parity["kernel_grads_finite"],
        "no_gdn_fallback": "chunk_gated_delta_rule" not in fallbacks,
        "no_conv_fallback": "causal_conv1d_fn" not in fallbacks or args.allow_missing_causal_conv1d,
    }
    ok = all(checks.values())
    if args.cpu_dry_run:
        ok = bool(ref_out.isfinite().all()) and parity["evaluated"] is False   # plumbing only
    elif not ok:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    summary = {"max_abs_diff": parity.get("max_abs_diff"), "rel_frobenius": parity.get("rel_frobenius"),
               "fallbacks": fallbacks, "fla": imports["fla"][1], "triton": imports["triton"][1], "checks": checks}
    return run.finish(verdict(ok, args.cpu_dry_run), summary)


if __name__ == "__main__":
    sys.exit(main())
