#!/usr/bin/env bash
# gpu_checks.sh -- training-VM bring-up checks T1-T10 and the row-cap gate (milestone M4.1, docs/DATA_PLAN.md sections 4
# and 9) as subcommands. Check design: REPORT.md section 3c of the archived 35B training-stack study, re-targeted to the
# dense Qwen3.5-9B.
#
# Run from the repository root on the GPU VM (after scripts/gpu_setup_train_env.sh):
#   scripts/gpu_checks.sh all                         # T1-T6, caps (T7 + T8 at 48K and 64K, the cap decision), T9, T10
#   scripts/gpu_checks.sh caps                        # T7 at 49152 and 65536, T8 over 49152,65536,98304, decision CAP=<n>
#   scripts/gpu_checks.sh t1|t2|...|t10 [options] [-- extra args for that step's helper]
#   T7_ENTRYPOINT=apex_train scripts/gpu_checks.sh t7 --rows data/processed/sft/train_rows.parquet [--seq-len 65536] [--steps 20]
#                                                     # T7 through `python -m apex_train.sft` on real rows (the run's own logs/<run_id>/)
#   scripts/gpu_checks.sh all --cpu-dry-run           # CPU plumbing test on the shrunken twin (no GPU)
# Options:
#   --cpu-dry-run         run on the tiny ConditionalGeneration twin with the pure-torch fused-head shim; GPU-only
#                         criteria are reported as not evaluated (status DRYRUN_OK/DRYRUN_FAIL, run_id prefix dryrun_);
#                         lengths shrink to the twin's scale (caps 192/256/384, T7 256, T9 256)
#   --model-dir DIR       checkpoint to check (default: $MODEL_DIR or the locked base revision on /mnt/apex-shared)
#   --seq-len N           sequence length of the single t7 step (default: the target cap 65536)
#   --steps N             optimizer steps of T7 (default 20; 4 in dry runs)
#   --rows PATH           T7_ENTRYPOINT=apex_train: the rows file to train on (data.train_rows; no dev evaluation); every row
#                         must fit --seq-len (apex_train.sft refuses longer rows)
#   --merged-dir DIR      merged checkpoint for T10 (default: the MERGED_DIR printed by T9 in this invocation)
#   --rank N              LoRA rank for T7/T8/T9 (default 32)
#   --keep-going          'all' continues after a FAIL
# Environment: PYTHON (default .venv/train/bin/python), MODEL_DIR, REFERENCE_DIR (T10 reference, default MODEL_DIR),
#   T7_ENTRYPOINT=trl|apex_train (default trl: scripts/dev/gpu_checks/t7_lora_smoke.py on synthetic rows of exactly the
#   length), CAP_TARGET / CAP_FALLBACK / CAP_MARGIN (65536 / 49152 / 98304), CAP_HEADROOM (0.90, share of GPU memory the
#   peak may use at the chosen cap), T9_SEQ_LEN (8192), PYTORCH_CUDA_ALLOC_CONF (default expandable_segments:True, as in
#   training), GPU_USD_PER_HOUR and CPU_USD_PER_HOUR (defaults: the GPU rate of configs/compute.yaml; 0, or the yaml's CPU
#   rate under --cpu-dry-run because apex-cpu is billed meanwhile), DRYRUN_TWIN_DIR.
#   Secrets are read from .env like scripts/evaluate.sh (HF_TOKEN; nothing here needs network access).
# Inputs : the checkpoint directory (read-only), the own prompt set scripts/dev/gpu_checks/prompts_own.jsonl, for
#   T7_ENTRYPOINT=apex_train a rows parquet from apex_train.data_format.
# Outputs: one logs/<run_id>/ per step (metadata.json, eval.log or train.log, metrics.jsonl, errors.log, resources.csv)
#   written by the helpers in scripts/dev/gpu_checks/ (T7 via apex_train: logs/gpucheck_t7_apex_train_T<len>_<ts>/ by
#   apex_train.sft, adapter and checkpoints under checkpoints/<that run_id>/); other big artifacts (adapter, trainer
#   checkpoints, merged checkpoint) under checkpoints/gpu_checks/<run_id>/ (git-ignored). Prints PASS/FAIL per step, the
#   cap decision (CAP=<n>, tokens/s per cap) and a summary.
# Steps and pass criteria: T1 CUDA stack; T2 fla/causal-conv1d imports + kernel parity + no fallback warning; T3 TRL fused
#   head; T4 one decoder layer (finite fwd+bwd, KB/token recorded, SDPA backend); T5 full load (15 mtp.* unexpected,
#   ~18.8 GB bf16); T6 greedy decode + HF reference for S7; T7 20-step LoRA run at one length (loss finite and falling,
#   peak memory and tok/s recorded); T8 OOM boundary (micro-batch 1, 3 steps per length, stops at the first OOM);
#   caps = T7 at CAP_FALLBACK and CAP_TARGET + T8 over CAP_FALLBACK,CAP_TARGET,CAP_MARGIN + the rule of
#   cap_decision.py: CAP_TARGET whenever it fits with headroom (T7 and T8 pass there and peak reserved <= CAP_HEADROOM x
#   GPU memory), else CAP_FALLBACK, else FAIL; T9 save/resume/merge (+ greedy parity) at T9_SEQ_LEN; T10 merged layout
#   gate vs the base index minus mtp.*.
# GPU-only: T1, T2 (kernel), T4 (memory/SDPA criteria), T5 (memory), T6-T9 and caps at real size. T10 is CPU-safe (headers).
# Exit code: 0 if every requested step passed (or DRYRUN_OK), 1 otherwise; 2 usage error.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # what training uses (scripts/train_sft.sh)

PYTHON=${PYTHON:-.venv/train/bin/python}
MODEL_DIR=${MODEL_DIR:-/mnt/apex-shared/models/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a}   # configs/eval/model.yaml model.local_dir
REFERENCE_DIR=${REFERENCE_DIR:-}
HELPERS=scripts/dev/gpu_checks
CAP_TARGET=${CAP_TARGET:-65536}; CAP_FALLBACK=${CAP_FALLBACK:-49152}; CAP_MARGIN=${CAP_MARGIN:-98304}; CAP_HEADROOM=${CAP_HEADROOM:-0.90}
SEQ_LEN=${SEQ_LEN:-}
T9_SEQ_LEN=${T9_SEQ_LEN:-8192}
STEPS=${STEPS:-}
RANK=${RANK:-32}
T7_ENTRYPOINT=${T7_ENTRYPOINT:-trl}
ROWS=${ROWS:-}
DRYRUN_TWIN_DIR=${DRYRUN_TWIN_DIR:-checkpoints/gpu_checks/tiny_cg_twin}   # git-ignored; rebuilt when missing
DRY=0; KEEP_GOING=0; MERGED_DIR=${MERGED_DIR:-}
EXTRA=()

usage() { sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; }
step=${1:-}; [[ -n "$step" ]] || { usage; exit 2; }; shift
case "$step" in -h|--help|help) usage; exit 0 ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-dry-run) DRY=1; shift ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --seq-len) SEQ_LEN=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --rows) ROWS=$2; shift 2 ;;
    --merged-dir) MERGED_DIR=$2; shift 2 ;;
    --rank) RANK=$2; shift 2 ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON (run scripts/gpu_setup_train_env.sh)" >&2; exit 2; }
case "$step" in all|caps) [[ ${#EXTRA[@]} -eq 0 ]] || { echo "'-- extra args' apply to a single step, not to $step" >&2; exit 2; } ;; esac

PREFIX=gpucheck
DRYFLAG=()
if [[ $DRY -eq 1 ]]; then
  PREFIX=dryrun_gpucheck; DRYFLAG=(--cpu-dry-run)
  CAP_TARGET=256; CAP_FALLBACK=192; CAP_MARGIN=384; T9_SEQ_LEN=256   # the twin's scale, same ratios as 64K / 48K / 96K
  export CPU_USD_PER_HOUR=${CPU_USD_PER_HOUR:-$("$PYTHON" -c 'import yaml; print(yaml.safe_load(open("configs/compute.yaml"))["cpu_vm_usd_per_hour"])')}
  "$PYTHON" $HELPERS/make_tiny_cg_twin.py --out "$DRYRUN_TWIN_DIR" >/dev/null
  MODEL_DIR=$DRYRUN_TWIN_DIR
fi
SEQ_LEN=${SEQ_LEN:-$CAP_TARGET}
STEPS=${STEPS:-$([[ $DRY -eq 1 ]] && echo 4 || echo 20)}
REFERENCE_DIR=${REFERENCE_DIR:-$MODEL_DIR}
TWIN_DIR=$DRYRUN_TWIN_DIR
# T3 always runs on the tiny twin (fast; the real kernel is exercised on the GPU, the real size in T7)
[[ $DRY -eq 1 ]] || "$PYTHON" $HELPERS/make_tiny_cg_twin.py --out "$TWIN_DIR" >/dev/null
UNEXPECTED=15; [[ $DRY -eq 1 ]] && UNEXPECTED=0   # mtp.* tensors of the real checkpoint (none in the twin)

declare -A RESULT
ORDER=()
LAST_OUT=$(mktemp)
trap 'rm -f "$LAST_OUT"' EXIT

record() {  # name, exit code -> RESULT[name] (PASS/DRYRUN_OK, SKIP, OOM or FAIL) in invocation order
  local ok=PASS; [[ $DRY -eq 1 ]] && ok=DRYRUN_OK
  [[ -n "${RESULT[$1]:-}" ]] || ORDER+=("$1")
  case $2 in 0) RESULT[$1]=$ok ;; 3) RESULT[$1]=SKIP ;; 75) RESULT[$1]=OOM ;; *) RESULT[$1]=FAIL ;; esac
}
run_helper() {  # name, helper script, args...; tees the helper output to LAST_OUT, records the result, returns its exit code
  local name=$1; shift
  echo "=== $name: $PYTHON $HELPERS/$* ${DRYFLAG[*]:-} --run-name ${PREFIX}_${name}"
  set +e
  "$PYTHON" "$HELPERS/$1" "${@:2}" "${DRYFLAG[@]}" --run-name "${PREFIX}_${name}" "${EXTRA[@]}" 2>&1 | tee "$LAST_OUT"
  local rc=${PIPESTATUS[0]}
  set -e
  record "$name" "$rc"
  return "$rc"
}
last_run_dir() { /usr/bin/grep -E '^RUN_DIR=' "$LAST_OUT" | tail -1 | cut -d= -f2- || true; }

t1()  { run_helper t1 t1_cuda_stack.py; }
t2()  { run_helper t2 t2_fla_parity.py; }
t3()  { run_helper t3 t3_fused_head.py --model-dir "$TWIN_DIR"; }
t4()  { if [[ $DRY -eq 1 ]]; then run_helper t4 t4_decoder_layer.py --seq-lens 256,512;
        else run_helper t4 t4_decoder_layer.py --config-dir "$MODEL_DIR" --seq-lens 4096,8192; fi; }
t5()  { run_helper t5 t5_full_load.py --model-dir "$MODEL_DIR" --expect-unexpected "$UNEXPECTED"; }
t6()  { if [[ $DRY -eq 1 ]]; then run_helper t6 t6_decode_sanity.py --model-dir "$MODEL_DIR" --num-prompts 2 --max-new-tokens 16;
        else run_helper t6 t6_decode_sanity.py --model-dir "$MODEL_DIR" --num-prompts 50 --max-new-tokens 128; fi; }
t7_trl() {  # name, seq_len: T7 on synthetic rows of exactly seq_len tokens
  local fallback=(); [[ $DRY -eq 1 ]] && fallback=(--allow-fallback)
  run_helper "$1" t7_lora_smoke.py --model-dir "$MODEL_DIR" --seq-len "$2" --max-steps "$STEPS" --rank "$RANK" "${fallback[@]}"
}
t7_apex_train() {  # T7 through the real entry point: apex_train.sft on a rows file (or the --smoke twin rows in a dry run)
  local cmd=(env PYTHONPATH=src "$PYTHON" -m apex_train.sft --max-steps "$STEPS")
  if [[ $DRY -eq 1 ]]; then
    cmd+=(--smoke)   # the twin; the smoke's own row cap (1024) applies
  else
    [[ -n "$ROWS" ]] || { echo "T7_ENTRYPOINT=apex_train needs --rows <rows parquet>" >&2; record t7 2; return 2; }
    cmd+=(--override "data.max_length=$SEQ_LEN" --run-id "${PREFIX}_t7_apex_train_T${SEQ_LEN}_$(date -u +%Y%m%d_%H%M%S)")
  fi
  [[ -z "$ROWS" ]] || cmd+=(--override "data.train_rows=$ROWS" --override data.dev_rows=null --override data.mixture_manifest=null)
  echo "=== t7 via apex_train.sft: ${cmd[*]} ${EXTRA[*]:-}"
  set +e; "${cmd[@]}" "${EXTRA[@]}"; local rc=$?; set -e
  record t7 "$rc"; return "$rc"
}
t7()  { if [[ "$T7_ENTRYPOINT" == "apex_train" ]]; then t7_apex_train; else t7_trl t7 "$SEQ_LEN"; fi; }
t8_boundary() {  # name: OOM boundary over fallback, target, margin; passes when at least the fallback cap trains
  run_helper "$1" t8_oom_boundary.py --model-dir "$MODEL_DIR" --planned-seq-len "$CAP_FALLBACK" --min-pass-ratio 1.0 \
    --seq-lens "$CAP_FALLBACK,$CAP_TARGET,$CAP_MARGIN" --steps "$([[ $DRY -eq 1 ]] && echo 2 || echo 3)" --rank "$RANK"
}
t8()  { t8_boundary t8; }
caps() {  # M4.1: T7 at both caps, the T8 boundary, then the decision rule (cap_decision.py prints CAP=<n>)
  # A T7 or T8 failure at the target is the fallback's case, not a failed gate: the decision's exit code is the step's.
  local c t7_args=() t8_dir
  for c in "$CAP_FALLBACK" "$CAP_TARGET"; do
    t7_trl "t7_T$c" "$c" || true
    t7_args+=(--t7 "$c=$(last_run_dir)")
  done
  t8_boundary t8_caps || true
  t8_dir=$(last_run_dir)
  run_helper caps cap_decision.py "${t7_args[@]}" --t8 "$t8_dir" --target "$CAP_TARGET" --fallback "$CAP_FALLBACK" \
    --margin "$CAP_MARGIN" --headroom "$CAP_HEADROOM"
}
t9()  {
  local rc=0
  if [[ $DRY -eq 1 ]]; then run_helper t9 t9_save_resume_merge.py --model-dir "$MODEL_DIR" --seq-len "$T9_SEQ_LEN" --num-prompts 2 --max-new-tokens 16 --reload-check --rank "$RANK" || rc=$?
  else run_helper t9 t9_save_resume_merge.py --model-dir "$MODEL_DIR" --seq-len "$T9_SEQ_LEN" --rank "$RANK" || rc=$?; fi
  local m; m=$(/usr/bin/grep -E '^MERGED_DIR=' "$LAST_OUT" | tail -1 | cut -d= -f2- || true)
  [[ -n "$m" ]] && MERGED_DIR=$m
  return "$rc"
}
t10() {
  [[ -n "$MERGED_DIR" ]] || { echo "t10 needs --merged-dir (or run t9 first in the same invocation)" >&2; record t10 2; return 2; }
  run_helper t10 t10_merged_layout_gate.py --merged-dir "$MERGED_DIR" --reference-dir "$REFERENCE_DIR"
}

overall=0
run_step() { local s=$1; if ! "$s"; then overall=1; [[ $KEEP_GOING -eq 1 ]] || return 1; fi; return 0; }
case "$step" in
  t1|t2|t3|t4|t5|t6|t7|t8|t9|t10|caps) run_step "$step" || true ;;
  all) for s in t1 t2 t3 t4 t5 t6 caps t9 t10; do run_step "$s" || break; done ;;
  *) echo "unknown step: $step" >&2; usage; exit 2 ;;
esac
echo "=== gpu_checks summary (${PREFIX}; model=$MODEL_DIR)"
for s in "${ORDER[@]}"; do printf '  %-12s %s\n' "$s" "${RESULT[$s]}"; done
[[ -n "$MERGED_DIR" ]] && echo "  merged_dir: $MERGED_DIR"
exit $overall
