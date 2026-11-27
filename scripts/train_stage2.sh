#!/usr/bin/env bash
# Stage 2 (configs/stage2.yaml): turn-level DPO by default (revision 5); --method filtered_sft runs the earlier revision-4.3
# design (docs/DATA_PLAN.md section 11): continue the SFT adapter on the execution-filtered rows of one rejection-sampling
# round, one arm per run. Run from the repository root.
#   scripts/train_stage2.sh --base <merged SFT dir> --pairs <pairs.jsonl> --override model.init_adapter=null   # DPO
#   scripts/train_stage2.sh --method filtered_sft --arm positives --init-adapter checkpoints/<sft run_id>/adapter     # arm (a)
#   scripts/train_stage2.sh --method filtered_sft --arm control --init-adapter checkpoints/<sft run_id>/adapter \
#       --match-steps-of logs/<arm a run_id>                                                           # arm (b), arm (a)'s steps
#   scripts/train_stage2.sh --method filtered_sft --arm positives [--rows <rows>] [--replay-rows <rows>] [--max-steps N] [--override key.path=value ...]
#   scripts/train_stage2.sh --method filtered_sft --smoke --arm positives --init-adapter <checkpoints>/<dryrun_sft_* id>/adapter [--out-root DIR]
#                                                       # CPU dry run on the tiny twin, continuing an SFT --smoke adapter
#   scripts/train_stage2.sh --smoke --method dpo        # the DPO path on the twin (fresh adapter, 8 synthetic pairs)
#   scripts/train_stage2.sh --help                      # all flags
# Inputs : configs/stage2.yaml (every field commented), the raw base (model.base; APEX_BASE_MODEL_DIR), the SFT adapter
#          (model.init_adapter; APEX_SFT_ADAPTER_DIR or --init-adapter), the arm's rows and the shared replay rows
#          (apex_train.data_format rows; APEX_STAGE2_{POSITIVES,CONTROL,REPLAY}_ROWS override the yaml paths), the stage-2 set
#          manifest (optional), configs/eval/model.yaml (lock), the training venv .venv/train (torch 2.14.1, transformers 5.19.0,
#          peft 0.21.2, trl 1.15.0; on the GPU VM the venv built by scripts/gpu_setup_train_env.sh). Secrets in the git-ignored
#          .env are exported like scripts/evaluate.sh does (HF_TOKEN for a later adapter upload; training itself needs none).
# Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv, config_snapshot/},
#          curves/source/<run_id>_train.csv, checkpoints/stage2/<run_id>/{trainer/checkpoint-N (50 and 100 percent), adapter/}.
#          run_id = stage2_positives_<ts> / stage2_control_<ts> (dryrun_ prefix for smokes; --out-root DIR keeps them out of
#          the repository).
# Environment: PYTHON (default .venv/train/bin/python); PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (as in SFT).
# Exit   : 0 finished; 1 crashed (see logs/<run_id>/errors.log); 2 usage or config error (nothing written).
# Details: src/apex_train/stage2.py docstring (continuation checks, arms, the DPO path).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
PYTHON=${PYTHON:-.venv/train/bin/python}
if [[ ! -x "$PYTHON" ]]; then echo "training interpreter not found: $PYTHON (set PYTHON=... or run scripts/gpu_setup_train_env.sh)" >&2; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec env PYTHONPATH=src "$PYTHON" -m apex_train.stage2 "$@"
