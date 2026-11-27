#!/usr/bin/env bash
# LoRA SFT of Qwen/Qwen3.5-9B with TRL + PEFT (configs/sft.yaml; docs/DATA_PLAN.md sections 4, 9, 10). Run from the repository root.
#   scripts/train_sft.sh                                   # main run on the frozen rows (data/processed/sft/{train,dev}_rows.parquet)
#   scripts/train_sft.sh --preset pilot                    # the 10 percent pilot (data/processed/sft/pilot_rows.parquet), same recipe
#   scripts/train_sft.sh --override optim.learning_rate=5e-5   # the LR fallback after an unstable pilot (M4.2)
#   scripts/train_sft.sh --max-steps 20 --override data.train_rows=<rows.parquet> --override data.dev_rows=null   # a check run
#   scripts/train_sft.sh --resume checkpoints/<run_id>/trainer           # resume from the latest checkpoint-N of a run (new run_id)
#   scripts/train_sft.sh --smoke [--preset pilot] [--threads 16] [--out-root DIR]   # CPU dry run on the tiny twin (dryrun_ run id)
#   scripts/train_sft.sh --smoke --resume <checkpoints>/<dryrun id>/trainer/checkpoint-6 --compare-steps-with <logs>/<dryrun id>
#   scripts/train_sft.sh --help                                          # all flags
# Inputs : configs/sft.yaml (model, lora, data, optim, eval, logging, run, presets, smoke sections), the base checkpoint
#          directory (read-only; APEX_BASE_MODEL_DIR overrides the default mount path), rows files from
#          `python -m apex_train.data_format` (APEX_SFT_TRAIN_ROWS / APEX_SFT_DEV_ROWS / APEX_SFT_PILOT_ROWS override the yaml
#          paths), data/manifests/mixture.json (optional, embedded in metadata.json), configs/eval/model.yaml (repo + revision
#          lock, read-only), the training venv .venv/train (torch 2.14.1, transformers 5.19.0, peft 0.21.2, trl 1.15.0; on the
#          GPU VM created by scripts/gpu_setup_train_env.sh). Secrets in the git-ignored .env are exported like
#          scripts/evaluate.sh does (HF_TOKEN for a later upload; training itself needs none).
# Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv, config_snapshot/sft.yaml},
#          checkpoints/<run_id>/{trainer/checkpoint-N/, adapter/ (+ base_model.json sidecar)},
#          curves/source/<run_id>_train.csv and <run_id>_eval.csv. --out-root DIR puts all three under DIR (smoke tests).
# Environment: PYTHON (default .venv/train/bin/python); PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (rows of
#          varying length up to the cap fragment the caching allocator otherwise; T7/T8 measure with the same setting).
# Exit   : 0 finished; 1 a check failed or the run aborted (see logs/<run_id>/errors.log); 2 usage error.
# Details: src/apex_train/sft.py docstring (checks, resume, the labels == masks self-check, the CPU-only fused-head shim).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
PYTHON=${PYTHON:-.venv/train/bin/python}
if [[ ! -x "$PYTHON" ]]; then echo "training interpreter not found: $PYTHON (set PYTHON=... or run scripts/gpu_setup_train_env.sh)" >&2; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec env PYTHONPATH=src "$PYTHON" -m apex_train.sft "$@"
