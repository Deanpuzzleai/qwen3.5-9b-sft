#!/usr/bin/env python
"""T7 (K5/G6/G10) 20-step LoRA SFT run with TRL SFTTrainer at the chosen sequence length (TRL-direct variant).

Run from the repository root (normally via scripts/gpu_checks.sh t7):
    .venv/train/bin/python scripts/dev/gpu_checks/t7_lora_smoke.py --model-dir <ckpt> --seq-len 8192 [--max-steps 20]
        [--rank 32 --alpha 64 --dropout 0.05] [--grad-accum 1] [--lr 2e-4] [--save-steps 0]
        [--budget-gb GB] [--work-dir checkpoints/gpu_checks/<run_id>] [--cpu-dry-run] [--run-name gpucheck_t7]
Inputs : the checkpoint directory (served class, bf16) and synthetic own data: identical rows of exactly --seq-len
         tokens built from a chat-template render (so the loss must fall on repeated data).
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl (per step: loss, grad_norm, lr, step_time, tok/s,
         CUDA peaks), errors.log, resources.csv (peak allocated/reserved, tok/s, GPU hours)}; trainer output
         (adapter, checkpoints) under --work-dir (git-ignored checkpoints/). Prints RUN_DIR=<path> last.
Settings (configs/sft.yaml): Qwen3_5ForConditionalGeneration via AutoModelForImageTextToText, bf16 weights, LoRA
recipe R2 in its dense form (r=32, alpha=64, dropout 0.05, 12 explicit target names: attention, Gated DeltaNet and
dense MLP projections; vision tower, embeddings and LM head frozen), sdpa attention, gradient checkpointing
(non-reentrant), micro-batch 1, packing/padding_free off, adamw_torch, TRL fused LM head (no full logits),
assistant_only_loss=False with own assistant_masks.
Pass (REPORT.md T7): loss finite and decreasing; tok/s and peak reserved memory recorded; no "falling back" kernel
warnings; peak reserved <= --budget-gb + 10 % when a budget is given (none exists for Qwen3.5-9B until the first GPU
check records it). Exit 75 on CUDA OOM (used by T8). GPU-only; --cpu-dry-run trains the tiny twin in fp32 with the
pure-torch fused-head shim (never installed outside dry runs).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from loading import (WarningCapture, auto_class_for, load_model, load_tokenizer, pick_device, render_ids,
                     seed_everything, synthetic_conversations)
from runlog import REPO_ROOT, CheckRun, cuda_peaks_gb, verdict

# Recipe R2 in its dense form (configs/sft.yaml lora.target_modules): attention q/k/v/o (8 full-attention layers),
# Gated DeltaNet in_proj_qkv/z/b/a + out_proj (24 linear-attention layers), dense MLP gate/up/down (all 32 layers).
R2_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                     "out_proj", "gate_proj", "up_proj", "down_proj"]
# No peak-memory budget table exists for Qwen3.5-9B (table 3a of memory_plan.md was measured on the previous base and does not transfer); the first
# GPU check records peak reserved GB per sequence length, after which --budget-gb gates later runs.
BUDGET_HEADROOM = 1.10


class TorchChunkedLogProb:
    """Pure-torch stand-in for trl.kernels.ChunkedLogProbFunction (Triton), CPU dry runs only.

    Lifted from probe 03_tiny_e2e_training.py of the archived 35B training-stack study (parity 1.9e-6 vs the plain
    loss). Same signature and outputs (log_prob, entropy, log(sum p^2), mean logit, is_top1), same transform order
    (logit_scale -> softcap -> 1/temperature), fp32 math, plain autograd. Never installed on the GPU VM.
    """

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None, logit_scale=1.0,
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        import torch

        n, vocab = hidden.shape[0], weight.shape[0]
        if n == 0:
            z = hidden.new_zeros((0,), dtype=torch.float32)
            return (z, z.clone() if "entropy" in outputs else None, z.clone() if "log_sum_sq_probs" in outputs else None,
                    z.clone() if "mean_logits" in outputs else None,
                    torch.zeros((0,), dtype=torch.bool, device=hidden.device) if "is_top1" in outputs else None)
        logits = hidden.float() @ weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        logits = logits * logit_scale
        if final_logit_softcapping is not None:
            logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
        logits = logits / temperature
        log_z = torch.logsumexp(logits, dim=-1)
        target_logit = logits.gather(1, targets[:, None]).squeeze(1)
        logprobs = target_logit - log_z
        entropy = log_sum_sq = mean_logits = is_top1 = None
        if "entropy" in outputs or "log_sum_sq_probs" in outputs:
            p = torch.softmax(logits, dim=-1)
            if "entropy" in outputs:
                entropy = log_z - (p * logits).sum(-1)
            if "log_sum_sq_probs" in outputs:
                log_sum_sq = torch.log((p * p).sum(-1)).detach()
        if "mean_logits" in outputs:
            mean_logits = logits.mean(-1).detach()
        if "is_top1" in outputs:
            with torch.no_grad():
                running_max = logits.max(-1).values
                before = torch.arange(vocab, device=logits.device)[None, :] < targets[:, None]
                max_before_target = logits.masked_fill(~before, float("-inf")).max(-1).values
                is_top1 = (target_logit >= running_max) & (target_logit > max_before_target)
        return logprobs, entropy, log_sum_sq, mean_logits, is_top1


def fused_head_status(cpu_dry_run: bool) -> dict[str, Any]:
    """TRL fused-LM-head kernel availability; installs the torch shim only for --cpu-dry-run when Triton is absent."""
    import trl.trainer.utils as tu

    info = {"trl_kernel_available": tu._ChunkedLogProbFunction is not None, "shim_installed": False}
    if tu._ChunkedLogProbFunction is None and cpu_dry_run:
        tu._ChunkedLogProbFunction = TorchChunkedLogProb
        info["shim_installed"] = True
    return info


def build_lora_config(rank: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", init_lora_weights=True,
                      target_modules=list(R2_TARGET_MODULES))


def build_sft_config(out_dir: Path, *, max_steps: int, seq_len: int, lr: float, grad_accum: int, seed: int,
                     save_steps: int, device: str, batch: int = 1):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(out_dir), max_steps=max_steps, num_train_epochs=1, per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch", logging_steps=1, logging_first_step=True,
        save_strategy="steps" if save_steps else "no", save_steps=save_steps or 0, save_total_limit=None,
        report_to="none", seed=seed, data_seed=seed, max_length=seq_len, truncation_mode="keep_start",
        packing=False, padding_free=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=device == "cuda", fp16=False, use_cpu=device == "cpu",
        dataloader_num_workers=0, dataloader_pin_memory=device == "cuda", disable_tqdm=True, remove_unused_columns=True,
        assistant_only_loss=False, completion_only_loss=None,
    )


def make_synthetic_dataset(tok, seq_len: int, n_rows: int, prompt_tokens: int = 64):
    """n identical rows of exactly seq_len tokens: a rendered own tool episode tiled to length; the first
    prompt_tokens positions are prompt (mask 0), the rest are trained (mask 1)."""
    from datasets import Dataset

    conv = synthetic_conversations()[1]
    ids = render_ids(tok, conv["messages"], conv["tools"], add_generation_prompt=False)
    ids = (ids * math.ceil(seq_len / len(ids)))[:seq_len]
    prompt_tokens = min(prompt_tokens, seq_len // 2)
    mask = [0] * prompt_tokens + [1] * (seq_len - prompt_tokens)
    return Dataset.from_dict({"input_ids": [ids] * n_rows, "assistant_masks": [mask] * n_rows})


class StepLogger:
    """TrainerCallback: per-step wall time, tokens/s and CUDA peaks into metrics.jsonl; keeps the loss history."""

    def __init__(self, run: CheckRun, tokens_per_step: int, tag: str = "train"):
        from transformers import TrainerCallback

        run_, tps, tag_ = run, tokens_per_step, tag
        logger = self

        class _CB(TrainerCallback):
            def on_step_begin(self, args, state, control, **kw):
                logger._t = time.perf_counter()

            def on_step_end(self, args, state, control, **kw):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.step_time = time.perf_counter() - logger._t

            def on_log(self, args, state, control, logs=None, **kw):
                logs = dict(logs or {})
                if "loss" not in logs:
                    return
                alloc, reserved = cuda_peaks_gb()
                rec = {"event": tag_, "step": state.global_step, "step_time_s": round(logger.step_time, 3),
                       "tokens_per_s": round(tps / logger.step_time, 1) if logger.step_time else None,
                       "max_memory_allocated_gb": alloc, "max_memory_reserved_gb": reserved, **logs}
                logger.records.append(rec)
                run_.metric(rec)
                run_.log(f"step {state.global_step}: loss={logs.get('loss')} grad_norm={logs.get('grad_norm')} "
                         f"tok/s={rec['tokens_per_s']} reserved_gb={reserved}")

        self.callback = _CB()
        self.records: list[dict] = []
        self.step_time = 0.0
        self._t = time.perf_counter()

    @property
    def losses(self) -> list[float]:
        return [float(r["loss"]) for r in self.records]


def load_base(model_dir: str, device: str) -> tuple[Any, dict]:
    """Served-class bf16 model on `device` (fp32 on CPU) with use_cache off; returns (model, loading_info)."""
    model, info, _ = load_model(model_dir, device=device, dtype_name="bfloat16" if device == "cuda" else "float32",
                                auto_class=auto_class_for(model_dir))
    model.config.use_cache = False
    return model, info


def build_trainer(model, tok, lora_cfg, sft_cfg, dataset, callbacks: list):
    from trl import SFTTrainer

    return SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset, processing_class=tok, peft_config=lora_cfg,
                      callbacks=callbacks)


def trainable_counts(model) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 4)}


def is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-gb", type=float, default=None,
                    help="peak reserved budget in GB (none is known for Qwen3.5-9B until the first GPU check records it)")
    ap.add_argument("--no-criteria", action="store_true", help="only run and record (T8 children): no loss/budget gate")
    ap.add_argument("--allow-fallback", action="store_true", help="do not fail on kernel fallback warnings")
    ap.add_argument("--work-dir", default=None, help="trainer output dir (default checkpoints/gpu_checks/<run_id>)")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t7")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args()
    run = CheckRun(args.run_name, log_name="train.log", config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    seed_everything(args.seed)
    head = fused_head_status(args.cpu_dry_run)
    run.log(f"fused LM head: {head}")
    if not head["trl_kernel_available"] and not head["shim_installed"]:
        run.error("TRL fused LM head kernel unavailable (no Triton) and not a dry run")
        return run.finish("FAIL", {"reason": "no fused head kernel"})
    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "checkpoints/gpu_checks" / run.run_id
    work.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model_dir)
    model, info = load_base(args.model_dir, device)
    run.log(f"loaded {info['class']} in {info['seconds']}s; unexpected(non-mtp)={info['unexpected_non_mtp']}")
    lora_cfg = build_lora_config(args.rank, args.alpha, args.dropout)
    sft_cfg = build_sft_config(work / "trainer", max_steps=args.max_steps, seq_len=args.seq_len, lr=args.lr,
                               grad_accum=args.grad_accum, seed=args.seed, save_steps=args.save_steps, device=device,
                               batch=args.batch)
    tokens_per_step = args.batch * args.grad_accum * args.seq_len
    ds = make_synthetic_dataset(tok, args.seq_len, n_rows=max(args.max_steps * args.batch * args.grad_accum, 1))
    logger = StepLogger(run, tokens_per_step)
    status, train_s = "PASS", None
    with WarningCapture() as cap:
        try:
            trainer = build_trainer(model, tok, lora_cfg, sft_cfg, ds, [logger.callback])
            counts = trainable_counts(trainer.model)
            run.log(f"trainable: {counts} recipe=R2 r={args.rank}")
            run.write_metadata(trainable=counts, lora=lora_cfg.to_dict(), sft_config=sft_cfg.to_dict(),
                               fused_head=head, loading=info, work_dir=str(work))
            t0 = time.time()
            trainer.train()
            train_s = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            if is_oom(exc):
                run.error(f"CUDA OOM at seq_len={args.seq_len}: {str(exc)[:300]}")
                status = "OOM"
            else:
                run.error(f"{type(exc).__name__}: {exc}")
                status = "ERROR"
    alloc, reserved = cuda_peaks_gb()
    losses = logger.losses
    tps = [r["tokens_per_s"] for r in logger.records if r.get("tokens_per_s")]
    budget = args.budget_gb
    summary: dict[str, Any] = {"seq_len": args.seq_len, "steps": len(losses), "first_loss": losses[0] if losses else None,
                               "last_loss": losses[-1] if losses else None, "peak_reserved_gb": reserved,
                               "peak_allocated_gb": alloc, "budget_gb": budget,
                               "mean_tokens_per_s": round(sum(tps) / len(tps), 1) if tps else None,
                               "train_seconds": round(train_s, 1) if train_s else None,
                               "fallbacks": cap.fallback_functions, "recipe": "R2"}
    if status == "PASS":
        checks = {"finite": bool(losses) and all(math.isfinite(x) for x in losses),
                  "decreasing": len(losses) >= 2 and losses[-1] < losses[0],
                  "budget": device == "cpu" or budget is None or reserved is None or reserved <= budget * BUDGET_HEADROOM,
                  "no_fallback": not cap.fallback_functions or args.allow_fallback,
                  "tok_s_recorded": bool(tps)}
        summary["checks"] = checks
        ok = all(checks.values()) if not args.no_criteria else checks["finite"]
        if not ok and not args.cpu_dry_run:
            run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
        status = verdict(ok, args.cpu_dry_run)
    extra = [("tokens_per_s_mean", summary["mean_tokens_per_s"], "tok/s", "trained tokens per second (micro-batch x accumulation x seq_len per step)"),
             ("budget_gb", budget if budget is not None else "n/a", "GB", "--budget-gb (none yet for Qwen3.5-9B: the first GPU check records peak reserved); pass = reserved <= budget x 1.10")]
    return run.finish(status, summary, peak_alloc_gb=alloc, peak_reserved_gb=reserved, extra_resources=extra)


if __name__ == "__main__":
    sys.exit(main())
