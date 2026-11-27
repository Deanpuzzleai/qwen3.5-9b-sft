#!/usr/bin/env python
"""T9 (K7/G7/G13) save / resume / merge: adapter save, bit-for-bit resume, merge_and_unload, greedy parity, timings.

Run from the repository root (normally via scripts/gpu_checks.sh t9):
    .venv/train/bin/python scripts/dev/gpu_checks/t9_save_resume_merge.py --model-dir <ckpt> [--seq-len 8192]
        [--num-prompts 3 --max-new-tokens 64] [--resume-tol 0.05] [--expect-merged-gb 18.82] [--reload-check]
        [--merged-dir checkpoints/gpu_checks/<run_id>/merged] [--cpu-dry-run] [--run-name gpucheck_t9]
Inputs : the checkpoint directory, synthetic own rows (T7 builder), own prompts for the greedy comparison.
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv (timings)}; the adapter,
         trainer checkpoints and the merged checkpoint under checkpoints/gpu_checks/<run_id>/ (git-ignored).
         Prints MERGED_DIR=<path> (input of T10 and of `scripts/serve_vllm.sh start` for S9).
Flow (REPORT.md "merge/save/serve"): 6 LoRA steps with save_steps=3 -> trainer.save_model(adapter) -> fresh trainer
resumes from checkpoint-3 for steps 4-6 -> greedy (base+adapter) on own prompts -> merge_and_unload() -> greedy
(merged) -> default save_pretrained(safe_serialization=True) + copy of tokenizer / chat_template.jinja /
generation_config.json / preprocessor configs from the source dir -> optional reload-from-disk greedy check.
Pass (REPORT.md T9): resumed losses == uninterrupted within bf16 noise (|diff| <= --resume-tol), adapter keys under
base_model.model.model.language_model.layers.* (served class), merged size == 18.82 GB (HF class bf16 without mtp.*, +-0.5 %),
greedy outputs identical for >= 64 tokens, timings logged. GPU-only; --cpu-dry-run runs on the tiny twin (fp32).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import sys
import time
from pathlib import Path

from loading import CG_CLASS, auto_class_for, load_model, load_tokenizer, pick_device, read_prompts, seed_everything
from runlog import REPO_ROOT, CheckRun, cuda_peaks_gb, verdict
from t6_decode_sanity import DEFAULT_PROMPTS, greedy
from t7_lora_smoke import StepLogger, build_lora_config, build_sft_config, build_trainer, fused_head_status, load_base, make_synthetic_dataset

HF_CLASS_BF16_BYTES = 18_819_635_168   # Qwen3_5ForConditionalGeneration, bf16, no mtp.* (summed from the index headers, 2026-10-11)
AUX_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
             "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
ADAPTER_PREFIX = {CG_CLASS: "base_model.model.model.language_model.layers.", "Qwen3_5ForCausalLM": "base_model.model.model.layers."}


def safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


def dir_bytes(d: Path, suffix: str = ".safetensors") -> int:
    return sum(p.stat().st_size for p in d.glob(f"*{suffix}"))


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--resume-tol", type=float, default=0.05, help="max |loss diff| per resumed step (bf16 noise)")
    ap.add_argument("--expect-merged-gb", type=float, default=HF_CLASS_BF16_BYTES / 1e9, help="0 = compare with the source dir size instead")
    ap.add_argument("--merged-dir", default=None)
    ap.add_argument("--reload-check", action="store_true", help="reload the merged dir from disk and compare greedy again")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t9")
    args = ap.parse_args()
    run = CheckRun(args.run_name, log_name="train.log", config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    seed_everything(args.seed)
    head = fused_head_status(args.cpu_dry_run)
    if not head["trl_kernel_available"] and not head["shim_installed"]:
        run.error("no fused-head kernel and not a dry run")
        return run.finish("FAIL", {"reason": "triton missing"})
    work = REPO_ROOT / "checkpoints/gpu_checks" / run.run_id
    merged_dir = Path(args.merged_dir) if args.merged_dir else work / "merged"
    adapter_dir = work / "adapter"
    timings: dict[str, float] = {}
    tok = load_tokenizer(args.model_dir)
    model, info = load_base(args.model_dir, device)
    cls = info["class"]
    run.log(f"loaded {cls} in {info['seconds']}s")
    lora = build_lora_config(args.rank, args.alpha, args.dropout)
    ds = make_synthetic_dataset(tok, args.seq_len, n_rows=6)

    # ---- run A: 6 steps, checkpoint at 3, adapter save
    log_a = StepLogger(run, args.seq_len, tag="runA")
    trainer = build_trainer(model, tok, lora, build_sft_config(work / "runA", max_steps=6, seq_len=args.seq_len, lr=2e-4, grad_accum=1,
                                                                 seed=args.seed, save_steps=3, device=device), ds, [log_a.callback])
    trainer.train()
    losses_a = log_a.losses
    t0 = time.time()
    trainer.save_model(str(adapter_dir))
    timings["save_adapter_s"] = round(time.time() - t0, 2)
    adapter_keys = safetensors_keys(adapter_dir / "adapter_model.safetensors")
    prefix = ADAPTER_PREFIX.get(cls, "base_model.model.")
    keys_ok = bool(adapter_keys) and all(k.startswith(prefix) for k in adapter_keys)
    run.log(f"adapter: {len(adapter_keys)} tensors, prefix ok={keys_ok} ({prefix}), e.g. {adapter_keys[:2]}")
    ckpt3 = work / "runA" / "checkpoint-3"
    base = trainer.model.unload()           # LoRA modules removed, base weights untouched; avoids a second 19 GB load
    del trainer
    free_cuda()

    # ---- resume from checkpoint-3 with a fresh trainer
    log_r = StepLogger(run, args.seq_len, tag="resume")
    trainer = build_trainer(base, tok, lora, build_sft_config(work / "runR", max_steps=6, seq_len=args.seq_len, lr=2e-4, grad_accum=1,
                                                                seed=args.seed, save_steps=3, device=device), ds, [log_r.callback])
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(ckpt3))
    timings["resume_train_s"] = round(time.time() - t0, 2)
    losses_r = log_r.losses
    tail_a = losses_a[-len(losses_r):] if losses_r else []
    resume_diff = max((abs(a - b) for a, b in zip(tail_a, losses_r)), default=float("inf"))
    run.log(f"resume: steps 4-6 uninterrupted={tail_a} resumed={losses_r} max|diff|={resume_diff}")
    run.metric({"event": "resume", "uninterrupted": tail_a, "resumed": losses_r, "max_abs_diff": resume_diff})

    # ---- greedy base+adapter vs merged
    peft_model = trainer.model.eval()
    prompts = read_prompts(Path(args.prompts), kind="plain", limit=args.num_prompts)
    kwargs = {"enable_thinking": True}
    outs_adapter = [greedy(peft_model, tok, p["messages"], args.max_new_tokens, device, kwargs)["output_token_ids"] for p in prompts]
    t0 = time.time()
    merged = peft_model.merge_and_unload()
    timings["merge_and_unload_s"] = round(time.time() - t0, 2)
    merged.eval()
    outs_merged = [greedy(merged, tok, p["messages"], args.max_new_tokens, device, kwargs)["output_token_ids"] for p in prompts]
    prefixes = [common_prefix(a, b) for a, b in zip(outs_adapter, outs_merged)]
    need = [min(args.max_new_tokens, len(a)) for a in outs_adapter]
    greedy_ok = all(p >= n for p, n in zip(prefixes, need))
    run.log(f"greedy adapter-vs-merged identical prefix per prompt: {prefixes} (needed {need})")
    run.metric({"event": "greedy_parity", "identical_prefix": prefixes, "needed": need, "ok": greedy_ok})

    # ---- default save + aux files
    t0 = time.time()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    timings["save_merged_s"] = round(time.time() - t0, 2)
    for name in AUX_FILES:
        src = Path(args.model_dir) / name
        if src.exists():
            shutil.copy2(src, merged_dir / name)
    merged_bytes = dir_bytes(merged_dir)
    expect_bytes = args.expect_merged_gb * 1e9 if args.expect_merged_gb and not args.cpu_dry_run else dir_bytes(Path(args.model_dir))
    size_ok = abs(merged_bytes - expect_bytes) <= 0.005 * expect_bytes
    run.log(f"merged save: {merged_bytes / 1e9:.3f} GB (expected {expect_bytes / 1e9:.3f}) in {timings['save_merged_s']}s -> {merged_dir}")

    reload_ok = True
    if args.reload_check:
        del trainer, peft_model, merged, base, model
        free_cuda()
        t0 = time.time()
        reloaded, rinfo, _ = load_model(merged_dir, device=device, dtype_name="bfloat16" if device == "cuda" else "float32",
                                        auto_class=auto_class_for(merged_dir))
        timings["reload_merged_s"] = round(time.time() - t0, 2)
        outs_reload = [greedy(reloaded.eval(), tok, p["messages"], args.max_new_tokens, device, kwargs)["output_token_ids"] for p in prompts]
        reload_prefixes = [common_prefix(a, b) for a, b in zip(outs_adapter, outs_reload)]
        reload_ok = all(p >= n for p, n in zip(reload_prefixes, need)) and not rinfo["missing"] and not rinfo["unexpected_non_mtp"]
        run.log(f"reload: class={rinfo['class']} missing={len(rinfo['missing'])} unexpected={len(rinfo['unexpected'])} prefixes={reload_prefixes}")
        run.metric({"event": "reload", "identical_prefix": reload_prefixes, "missing": len(rinfo["missing"]), "unexpected": len(rinfo["unexpected"])})

    checks = {"resume_within_tol": resume_diff <= args.resume_tol, "adapter_prefix": keys_ok,
              "adapter_prefix_is_language_model": prefix == ADAPTER_PREFIX[CG_CLASS] or args.cpu_dry_run,
              "merged_size": size_ok, "greedy_identical": greedy_ok, "reload": reload_ok}
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    for k, v in timings.items():
        run.metric({"event": "timing", "name": k, "seconds": v})
    alloc, reserved = cuda_peaks_gb()
    summary = {"resume_max_abs_diff": resume_diff, "adapter_tensors": len(adapter_keys), "merged_gb": round(merged_bytes / 1e9, 3),
               "greedy_identical_prefix": prefixes, "timings": timings, "merged_dir": str(merged_dir), "checks": checks}
    run.write_metadata(merged_dir=str(merged_dir), adapter_dir=str(adapter_dir), timings=timings)
    extra = [(k, v, "s", "wall time") for k, v in timings.items()] + [("merged_gb", round(merged_bytes / 1e9, 3), "GB", "sum of *.safetensors in the merged dir")]
    code = run.finish(verdict(ok, args.cpu_dry_run), summary, peak_alloc_gb=alloc, peak_reserved_gb=reserved, extra_resources=extra)
    print(f"MERGED_DIR={merged_dir}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
