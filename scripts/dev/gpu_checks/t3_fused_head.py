#!/usr/bin/env python
"""T3 (G9) TRL fused LM head on the GPU: real Triton kernel importable, parity vs the plain loss, 2 SFTTrainer steps.

Run from the repository root (normally via scripts/gpu_checks.sh t3):
    .venv/train/bin/python scripts/dev/gpu_checks/t3_fused_head.py --model-dir <tiny CG twin> [--seq-len 256]
        [--trainer-steps 2] [--cpu-dry-run] [--run-name gpucheck_t3]
Inputs : the tiny ConditionalGeneration twin (make_tiny_cg_twin.py); synthetic own chats.
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log, resources.csv}.
Pass (REPORT.md T3): `trl.kernels.ChunkedLogProbFunction` imports (Triton present), `shim_installed == False`,
fused-vs-plain parity on the twin (loss diff < 1e-3 in bf16, log-prob diff < 2e-2, grads finite; fp32 tolerances
1e-5 / 1e-5 on CPU), and a 2-step SFTTrainer run (fused head in compute_loss) with finite losses.
Parity method lifted from probe 03_tiny_e2e_training.py (check 1c) of the archived 35B training-stack study. GPU-only;
--cpu-dry-run installs
the torch shim (then shim_installed is True by design and only the plumbing is verified).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from loading import load_tokenizer, pick_device, render_ids, seed_everything, synthetic_conversations
from runlog import REPO_ROOT, CheckRun, verdict
from t7_lora_smoke import (StepLogger, build_lora_config, build_sft_config, build_trainer, fused_head_status, load_base,
                           make_synthetic_dataset)


def kernel_import() -> tuple[bool, str]:
    try:
        from trl.kernels import ChunkedLogProbFunction  # noqa: F401

        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def parity_batch(tok, device: str, seq_len: int):
    """Two rendered own chats, right-padded; labels -100 on padding."""
    import torch

    rows = [render_ids(tok, c["messages"], c["tools"], add_generation_prompt=False)[:seq_len]
            for c in synthetic_conversations()[:2]]
    width = max(len(r) for r in rows)
    pad = tok.pad_token_id
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=device)
    attn = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
    labels = ids.masked_fill(attn == 0, -100)
    return ids, attn, labels


def fused_vs_plain(model, ids, attn, labels) -> dict[str, Any]:
    """Loss / per-token log-prob / gradient / entropy agreement between fused_lm_head=True and the plain forward."""
    import torch
    from trl.trainer.utils import add_fused_lm_head

    model.zero_grad(set_to_none=True)
    add_fused_lm_head(model, outputs=("log_probs", "entropy", "is_top1"))
    plain = model(input_ids=ids, attention_mask=attn, labels=labels, fused_lm_head=False)
    fused = model(input_ids=ids, attention_mask=attn, labels=labels, fused_lm_head=True)
    fused_loss = -fused.log_probs.sum() / fused.label_mask.sum()
    shifted = labels[:, 1:]
    lp_ref = torch.log_softmax(plain.logits[:, :-1].float(), -1).gather(-1, shifted.clamp(min=0)[..., None]).squeeze(-1)
    lp_ref = lp_ref * (shifted != -100)
    d_lp = (lp_ref - fused.log_probs).abs().max().item()
    fused_loss.backward()
    g_fused = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    plain.loss.backward()
    g_plain = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    d_grad = max(((g_fused[n] - g_plain[n]).abs().max().item() for n in g_fused if n in g_plain), default=float("nan"))
    ent_ref = torch.distributions.Categorical(logits=plain.logits[:, :-1].float()).entropy() * (shifted != -100)
    d_ent = (ent_ref - fused.entropy).abs().max().item()
    return {"loss_fused": fused_loss.item(), "loss_plain": plain.loss.item(),
            "loss_abs_diff": abs(fused_loss.item() - plain.loss.item()), "max_abs_diff_logprobs": d_lp,
            "max_abs_diff_grads": d_grad, "max_abs_diff_entropy": d_ent, "n_grad_tensors": len(g_fused),
            "grads_finite": all(torch.isfinite(g).all().item() for g in g_fused.values()),
            "mean_token_accuracy_fused": (fused.is_top1.sum() / fused.label_mask.sum()).item()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="tiny ConditionalGeneration twin")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--trainer-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t3")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    seed_everything(args.seed)
    imp_ok, imp_detail = kernel_import()
    head = fused_head_status(args.cpu_dry_run)
    run.log(f"trl.kernels import: {imp_ok} ({imp_detail}); fused head: {head}")
    if not head["trl_kernel_available"] and not head["shim_installed"]:
        run.error("no fused-head kernel and not a dry run")
        return run.finish("FAIL", {"reason": "triton missing"})
    tok = load_tokenizer(args.model_dir)
    model, info = load_base(args.model_dir, device)
    run.log(f"loaded {info['class']} ({info['dtype']})")
    bf16 = device == "cuda"
    tol_loss, tol_lp = (1e-3, 2e-2) if bf16 else (1e-5, 1e-5)
    ids, attn, labels = parity_batch(tok, device, args.seq_len)
    parity = fused_vs_plain(model, ids, attn, labels)
    run.log(f"parity: {parity}")
    run.metric({"event": "parity", **parity})

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "checkpoints/gpu_checks" / run.run_id
    sft_cfg = build_sft_config(work / "trainer", max_steps=args.trainer_steps, seq_len=args.seq_len, lr=2e-4,
                               grad_accum=1, seed=args.seed, save_steps=0, device=device)
    ds = make_synthetic_dataset(tok, args.seq_len, n_rows=args.trainer_steps)
    logger = StepLogger(run, args.seq_len, tag="trainer")
    trainer_ok, err = True, None
    try:
        trainer = build_trainer(model, tok, build_lora_config(8, 16, 0.0), sft_cfg, ds, [logger.callback])
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        trainer_ok, err = False, f"{type(exc).__name__}: {exc}"
        run.error(f"SFTTrainer micro-run failed: {err}")
    losses = logger.losses
    checks = {"kernel_import": imp_ok or args.cpu_dry_run, "shim_not_installed": not head["shim_installed"] or args.cpu_dry_run,
              "loss_parity": parity["loss_abs_diff"] < tol_loss, "logprob_parity": parity["max_abs_diff_logprobs"] < tol_lp,
              "grads_finite": parity["grads_finite"],
              "trainer_steps_finite": trainer_ok and len(losses) == args.trainer_steps and all(math.isfinite(x) for x in losses)}
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    summary = {"loss_abs_diff": parity["loss_abs_diff"], "max_abs_diff_logprobs": parity["max_abs_diff_logprobs"],
               "trainer_losses": losses, "shim_installed": head["shim_installed"], "checks": checks}
    return run.finish(verdict(ok, args.cpu_dry_run), summary)


if __name__ == "__main__":
    sys.exit(main())
