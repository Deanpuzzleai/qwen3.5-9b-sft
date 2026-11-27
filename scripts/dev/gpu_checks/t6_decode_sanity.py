#!/usr/bin/env python
"""T6 (G5) decode sanity: greedy generation with thinking on, and the HF greedy reference file for S7/S9.

Run from the repository root (normally via scripts/gpu_checks.sh t6):
    .venv/train/bin/python scripts/dev/gpu_checks/t6_decode_sanity.py --model-dir <ckpt> [--num-prompts 1]
        [--max-new-tokens 64] [--prompts scripts/dev/gpu_checks/prompts_own.jsonl] [--cpu-dry-run] [--run-name gpucheck_t6]
    # S7 reference: --num-prompts 50 --max-new-tokens 128 (REPORT.md S7: 50 own prompts, T=0, 128 tokens)
Inputs : the checkpoint directory and the own prompt set (plain prompts only; never benchmark data).
Outputs: logs/<run_id>/{metadata.json, eval.log, metrics.jsonl, errors.log, resources.csv} and
         logs/<run_id>/hf_greedy_reference.jsonl: one row per prompt {id, messages, chat_template_kwargs,
         prompt_token_ids, output_token_ids, output_text, eos_reached} consumed by `scripts/serve_vllm.sh s7`.
Pass (REPORT.md T6): the generation prompt opens `<think>` (id 248068), generation finishes without error, the output
is finite/non-degenerate (unique-token ratio >= 0.15) and, when --require-eos, ends with <|im_end|> (248046);
"coherent" is for the operator: the first output is printed. GPU-only; --cpu-dry-run decodes the random tiny twin
(degeneracy not evaluated).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loading import IM_END_ID, THINK_ID, auto_class_for, load_model, load_tokenizer, pick_device, read_prompts, seed_everything
from runlog import CheckRun, cuda_peaks_gb, verdict

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts_own.jsonl"
MIN_UNIQUE_RATIO = 0.15


def greedy(model, tok, messages: list[dict], max_new_tokens: int, device: str, kwargs: dict) -> dict:
    """Greedy decode of one chat prompt; returns prompt ids, output ids/text, eos flag, seconds."""
    import torch

    enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                  return_tensors="pt", **kwargs)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, use_cache=True, pad_token_id=tok.pad_token_id)
    seconds = time.perf_counter() - t0
    prompt_ids = input_ids[0].tolist()
    gen = out[0, len(prompt_ids):].tolist()
    return {"prompt_token_ids": prompt_ids, "output_token_ids": gen, "output_text": tok.decode(gen, skip_special_tokens=False),
            "eos_reached": IM_END_ID in gen or tok.pad_token_id in gen, "seconds": round(seconds, 2),
            "think_opened": THINK_ID in prompt_ids[-3:],
            "unique_ratio": round(len(set(gen)) / max(len(gen), 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--num-prompts", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--no-thinking", action="store_true", help="render with enable_thinking=False (default: thinking on)")
    ap.add_argument("--require-eos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--run-name", default="gpucheck_t6")
    args = ap.parse_args()
    run = CheckRun(args.run_name, config=vars(args), seed=args.seed)
    device = pick_device(args.cpu_dry_run)
    seed_everything(args.seed)
    tok = load_tokenizer(args.model_dir)
    model, info, _ = load_model(args.model_dir, device=device, dtype_name="bfloat16" if device == "cuda" else "float32",
                                auto_class=auto_class_for(args.model_dir))
    model.eval()
    run.log(f"loaded {info['class']} in {info['seconds']}s")
    kwargs = {"enable_thinking": not args.no_thinking}
    prompts = read_prompts(Path(args.prompts), kind="plain", limit=args.num_prompts)
    ref_path = run.run_dir / "hf_greedy_reference.jsonl"
    rows, errors = [], 0
    for p in prompts:
        try:
            r = greedy(model, tok, p["messages"], args.max_new_tokens, device, kwargs)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            run.error(f"prompt {p['id']}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        run.metric({"event": "greedy", "id": p["id"], "n_prompt": len(r["prompt_token_ids"]), "n_out": len(r["output_token_ids"]),
                    "eos_reached": r["eos_reached"], "think_opened": r["think_opened"], "unique_ratio": r["unique_ratio"], "seconds": r["seconds"]})
        with ref_path.open("a") as f:
            f.write(json.dumps({"id": p["id"], "messages": p["messages"], "chat_template_kwargs": kwargs, "model_dir": args.model_dir,
                                **{k: r[k] for k in ("prompt_token_ids", "output_token_ids", "output_text", "eos_reached")}}) + "\n")
    if rows:
        run.log(f"first output ({len(rows[0]['output_token_ids'])} tokens, eos={rows[0]['eos_reached']}):\n{rows[0]['output_text'][:1500]}")
    checks = {"no_errors": errors == 0 and bool(rows),
              "think_opened": all(r["think_opened"] for r in rows) if not args.no_thinking else True,
              "not_degenerate": args.cpu_dry_run or all(r["unique_ratio"] >= MIN_UNIQUE_RATIO for r in rows),
              "eos": (not args.require_eos) or all(r["eos_reached"] for r in rows)}
    ok = all(checks.values())
    if not ok and not args.cpu_dry_run:
        run.error(f"failed checks: {[k for k, v in checks.items() if not v]}")
    tok_s = [len(r["output_token_ids"]) / r["seconds"] for r in rows if r["seconds"]]
    alloc, reserved = cuda_peaks_gb()
    summary = {"prompts": len(rows), "eos_reached": sum(r["eos_reached"] for r in rows), "mean_decode_tok_s": round(sum(tok_s) / len(tok_s), 1) if tok_s else None,
               "reference_file": str(ref_path), "checks": checks}
    run.write_metadata(reference_file=str(ref_path), chat_template_kwargs=kwargs)
    return run.finish(verdict(ok, args.cpu_dry_run), summary, peak_alloc_gb=alloc, peak_reserved_gb=reserved)


if __name__ == "__main__":
    sys.exit(main())
