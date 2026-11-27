#!/usr/bin/env bash
# Merge a LoRA adapter into the base checkpoint, gate a merged directory, or verify an adapter. Run from the repository root.
#   scripts/merge_adapter.sh merge --base <base dir> --adapter <adapter dir> --out <merged dir> \
#       [--snapshot configs/eval/model_snapshot] [--merge-dtype fp32|bf16] [--device cpu|cuda] [--max-shard-size 5GB] [--run-id <id>] [--note TEXT]
#   scripts/merge_adapter.sh gate --out <dir> --base <base dir> [--snapshot configs/eval/model_snapshot] [--run-id <id>]
#   scripts/merge_adapter.sh verify-adapter --adapter <adapter dir> [--fix-adapter-config] [--run-id <id>]
#   scripts/merge_adapter.sh <subcommand> --help     # all flags
# Inputs : the base checkpoint directory (read-only; the real one is under /mnt/apex-shared/models/), a PEFT adapter
#          directory, the pinned snapshot configs/eval/model_snapshot/, the lock configs/eval/model.yaml (repo + revision),
#          the training venv .venv/train (torch 2.14.1, transformers 5.19.0, peft 0.21.2). Secrets in the git-ignored .env
#          are exported like scripts/evaluate.sh does (HF_TOKEN for a later upload; nothing here needs them).
# Outputs: <out>/ (merged bf16 checkpoint in the base layout + tokenizer/template/generation/preprocessor files + LICENSE
#          + merge_provenance.json) and logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv}.
# Exit   : 0 all checks passed; 1 a check failed or the step aborted (see logs/<run_id>/errors.log); 2 usage error.
# Details: src/apex_train/merge.py docstring (memory, the layout gate, the adapter checks).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
exec env PYTHONPATH=src .venv/train/bin/python -m apex_train.merge "$@"
