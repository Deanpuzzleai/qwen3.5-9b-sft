#!/usr/bin/env python
"""Fill a model-card template (models/cards/*.md) from a training run's records, never by hand.

Run from the repository root:
    .venv/train/bin/python scripts/fill_model_card.py --template models/cards/sft_adapter.md \
        --run-dir logs/sft_main_<ts> --mixture data/manifests/mixture.json --data-card data/README.md \
        --hf-user deanoshuang --repo-name qwen3.5-9b-agent-sft-lora --out models/cards/sft_adapter.filled.md \
        [--set checkpoint="final (step 350)" --set hf_commit=<sha> --set results_table=@results.md ...]

Inputs : the template's {{key}} placeholders; logs/<run>/metadata.json (config: LR, batch, epochs; code commit; run
         id; last validation loss) and resources.csv (hours, tokens/s, memory, GPU-h, cost); the mixture manifest (rows,
         loss tokens, per-slice loss shares, row cap, train-file sha256, mixture run id, freeze time); the data card's
         path and sources table;
         `git remote get-url origin`.
         --set key=value fills any placeholder (value @path reads a file); later --set wins.
Outputs: the filled card at --out. Placeholders without a value stay as {{key}} and are listed on stderr, so a draft
         can be written before the evaluation numbers exist; --strict exits 1 if any remain.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def resources(run_dir: Path) -> dict[str, str]:
    with (run_dir / "resources.csv").open() as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def sources_table(data_card: Path) -> str:
    """The markdown table of the data card's sources section (first build: section "2."; revision 5: "Sources and licences")."""
    lines = data_card.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## 2.") or ln.startswith("## Sources"))
    table, in_table = [], False
    for ln in lines[start + 1:]:
        if ln.startswith("|"):
            table.append(ln)
            in_table = True
        elif in_table:
            break
    if not table:
        raise SystemExit(f"{data_card}: no table in section 2")
    return "\n".join(table)


def values(args: argparse.Namespace) -> dict[str, str]:
    meta = json.loads((args.run_dir / "metadata.json").read_text())
    res = resources(args.run_dir)
    mix = json.loads(args.mixture.read_text())
    optim = (meta.get("config") or {}).get("optim") or {}
    slices = mix["train"]["slices"]
    train_file = mix["files"]["train_rows"]
    pct = lambda s: f"{100 * slices[s]['loss_share']:.1f}"
    v = {
        "repo_name": args.repo_name,
        "hf_user": args.hf_user,
        "github_repo_url": subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True,
                                          check=True).stdout.strip().removesuffix(".git"),
        "n_rows": f"{train_file['rows']:,}",
        "loss_tokens": f"{train_file['loss_tokens']:,}",
        "share_swe": pct("swe"),
        "share_terminal": pct("terminal"),
        "share_retention": pct("retention"),
        "row_cap": f"{mix['cap']['max_length'] if 'cap' in mix else mix['max_length']:,}",   # first build: cap block; v5: top level
        "data_sha256": train_file["sha256"],
        "learning_rate": f"{optim['learning_rate']:.0e}".replace("e-0", "e-"),
        "effective_batch": str(optim["effective_batch_size"]),
        "grad_accum": str(optim["gradient_accumulation_steps"]),
        "num_epochs": f"{optim['num_train_epochs']} epoch{'s' if optim['num_train_epochs'] != 1 else ''}",
        "code_commit": meta["code"]["commit"],
        "optimizer_steps": res["optimizer_steps"],
        "train_hours": f"{float(res['train_runtime_sec']) / 3600:.2f}",
        "tokens_per_s": f"{float(res['tokens_per_s_steady_median']):,.0f}",
        "peak_gpu_mem_gb": f"{float(res['peak_gpu_memory_reserved_gb']):.1f}",
        "gpu_hours": f"{float(res['gpu_hours']):.2f}",
        "cost_usd": f"{float(res['cost_usd']):.0f}",
        "run_id": meta["run_id"],
        "sources_table": sources_table(args.data_card),
        "data_card": str(args.data_card),
        "data_run_id": mix["run_id"],
    }
    if (mix.get("freeze") or {}).get("frozen_at"):
        v["data_frozen_at"] = mix["freeze"]["frozen_at"]
    evals = (meta.get("summary") or {}).get("eval") or []
    if evals:   # the trainer's last validation pass (step, loss, shard rows)
        v["final_eval_step"] = str(evals[-1]["step"])
        v["final_eval_loss"] = f"{evals[-1]['eval_loss']:.4f}"
        v["eval_rows"] = str(evals[-1]["eval_rows"])
    for item in args.set:
        key, _, val = item.partition("=")
        v[key] = Path(val[1:]).read_text().rstrip("\n") if val.startswith("@") else val
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mixture", type=Path, default=Path("data/manifests/mixture.json"))
    p.add_argument("--data-card", type=Path, default=Path("data/README.md"))
    p.add_argument("--hf-user", required=True)
    p.add_argument("--repo-name", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="fill a placeholder (VALUE @path reads a file)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true", help="exit 1 if any placeholder stays unfilled")
    args = p.parse_args(argv)
    v = values(args)
    text = PLACEHOLDER.sub(lambda m: v.get(m.group(1), m.group(0)), args.template.read_text())
    left = sorted(set(PLACEHOLDER.findall(text)))
    args.out.write_text(text)
    print(f"wrote {args.out}; unfilled: {', '.join(left) or 'none'}", file=sys.stderr)
    return 1 if (left and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
