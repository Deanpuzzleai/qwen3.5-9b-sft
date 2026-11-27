#!/usr/bin/env python
"""Build the shrunken Qwen3_5ForConditionalGeneration twin used by every CPU dry run (scripts/gpu_checks.sh --cpu-dry-run,
scripts/train_sft.sh --smoke, scripts/train_stage2.sh --smoke, the merge/gate dry run).

The twin has the 9B's structure at toy width (loading.build_tiny_cg_config: 4 decoder layers = 3 Gated DeltaNet + 1 full
attention, dense MLP, gated attention, 1-block vision tower, the real 248,320 vocab), random init (fp32, std 0.02), and
the REAL tokenizer + chat template copied from the verified snapshot of the locked revision (tokenizer.json,
tokenizer_config.json, chat_template.jinja, vocab.json, merges.txt). Saved with the default save_pretrained, so its
keys follow the real layout (model.language_model.*, model.visual.*, lm_head.weight; asserted after the save).

generation_config.json: the locked revision ships none, so the file written here is OURS, not the Hub's. It holds the
model card's thinking-mode sampling defaults for general tasks (temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
presence_penalty 1.5, repetition_penalty 1.0; the values configs/eval/model.yaml passes to vLLM through
--override-generation-config) plus the eos / pad ids, and says so in its `apex_note`. It gives HF generate() on the
twin sane defaults and the CPU checks a complete tokenizer directory; the merge never copies it into a
served checkpoint (src/apex_train/merge.py: generation_config.json mirrors the snapshot, which has none).

Run from the repository root:
    .venv/train/bin/python scripts/dev/gpu_checks/make_tiny_cg_twin.py [--out checkpoints/gpu_checks/tiny_cg_twin]
        [--tokenizer-dir <verified snapshot>] [--tests-dir logs/train_stack_check_qwen35_9b/tiny_base] [--seed 0] [--force]
Inputs : configs/eval/model_snapshot/config.json (shapes shrunk) and the tokenizer files of the verified snapshot
         (default: the mount of configs/eval/model.yaml model.local_dir, else --tests-dir when it already holds them).
Outputs: <out>/{config.json, model.safetensors, generation_config.json, tokenizer.json, tokenizer_config.json,
         chat_template.jinja, vocab.json, merges.txt}, and the same tokenizer files + generation_config.json in
         --tests-dir (the stable tokenizer location of the data_format tests; files already byte-identical are left
         alone, differing ones are replaced and reported). Idempotent: an existing complete twin is left alone unless
         --force. Prints the parameter count.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import struct
import sys
from pathlib import Path

from loading import CG_CLASS, DEFAULT_MODEL_DIR, TEST_TOKENIZER_DIR, build_tiny_cg_config, seed_everything

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
EXPECTED_KEY_PREFIXES = ("model.language_model.", "model.visual.", "lm_head.")
GENERATION_CONFIG = {
    "apex_note": "Written by scripts/dev/gpu_checks/make_tiny_cg_twin.py; NOT a Hub file. Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a "
                 "ships no generation_config.json; these are the model card's thinking-mode sampling defaults for general tasks, "
                 "the values configs/eval/model.yaml passes to vLLM through --override-generation-config.",
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "eos_token_id": [248046, 248044],   # <|im_end|> (the tokenizer's eos), <|endoftext|> (text_config.eos_token_id)
    "pad_token_id": 248044,
}


def safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


def default_tokenizer_dir() -> Path:
    """The verified snapshot on the mount when present, else the tests directory (already populated on this box)."""
    for cand in (DEFAULT_MODEL_DIR, TEST_TOKENIZER_DIR):
        if all((cand / name).is_file() for name in TOKENIZER_FILES):
            return cand
    return DEFAULT_MODEL_DIR


def sync_tokenizer_files(src: Path, dst: Path) -> list[str]:
    """Copy the tokenizer files and write our generation_config.json into dst; returns the names (re)written.
    Files already byte-identical are left alone (dst may be shared with other tooling)."""
    missing = [name for name in TOKENIZER_FILES if not (src / name).is_file()]
    if missing:
        raise FileNotFoundError(f"tokenizer files missing in {src}: {missing} (fetch the snapshot with scripts/fetch_base_model.sh)")
    dst.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in TOKENIZER_FILES:
        if (dst / name).is_file() and filecmp.cmp(src / name, dst / name, shallow=False):
            continue
        if (dst / name).is_file():
            print(f"[make_tiny_cg_twin] {dst / name} differed from {src / name}; replaced", flush=True)
        shutil.copyfile(src / name, dst / name)
        written.append(name)
    gen = json.dumps(GENERATION_CONFIG, indent=2) + "\n"
    if not (dst / "generation_config.json").is_file() or (dst / "generation_config.json").read_text() != gen:
        (dst / "generation_config.json").write_text(gen)
        written.append("generation_config.json")
    return written


def build(out: Path, tokenizer_dir: Path, seed: int) -> int:
    """Create the twin under `out`; returns the number of parameters."""
    import torch
    from transformers import AutoModelForImageTextToText

    seed_everything(seed)
    cfg = build_tiny_cg_config()
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)
    assert type(model).__name__ == CG_CLASS, type(model).__name__
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    keys = safetensors_keys(out / "model.safetensors")
    odd = [k for k in keys if not k.startswith(EXPECTED_KEY_PREFIXES)]
    assert keys and not odd, f"twin keys do not follow the served layout: {odd[:5]}"
    assert any(".linear_attn." in k for k in keys) and any(".self_attn." in k for k in keys)
    sync_tokenizer_files(tokenizer_dir, out)
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="checkpoints/gpu_checks/tiny_cg_twin", help="destination directory of the twin (git-ignored)")
    ap.add_argument("--tokenizer-dir", default=None, help="where to copy the tokenizer/template files from (default: the verified snapshot)")
    ap.add_argument("--tests-dir", default=str(TEST_TOKENIZER_DIR), help="also keep the tokenizer files + generation_config.json here ('' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild even if the twin exists")
    args = ap.parse_args()
    out = Path(args.out)
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else default_tokenizer_dir()
    if args.tests_dir:
        written = sync_tokenizer_files(tokenizer_dir, Path(args.tests_dir))
        if written:
            print(f"[make_tiny_cg_twin] tests dir {args.tests_dir}: wrote {written}")
    complete = all((out / name).is_file() for name in ("config.json", "model.safetensors", "generation_config.json", *TOKENIZER_FILES))
    if complete and not args.force:
        print(f"[make_tiny_cg_twin] exists: {out}")
        return 0
    n = build(out, tokenizer_dir, args.seed)
    print(f"[make_tiny_cg_twin] wrote {out} ({n:,d} params, {CG_CLASS}, tokenizer from {tokenizer_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
