#!/usr/bin/env bash
# Rebuild the stage-1 SFT data, version v7 (the data of the final models), by calling the real data builders in order.
# A thin wrapper: every step is one of the scripts under scripts/data/, each of which documents its own inputs and
# outputs in its docstring / --help and writes logs/<run_id>/ (metadata.json, train.log, metrics.jsonl, errors.log,
# resources.csv). Run from the repository root.
#
#   scripts/prepare_data.sh               # run every step (CPU only; hours, ~100 GB of raw downloads)
#   scripts/prepare_data.sh --dry-run     # print the commands without running them
#   scripts/prepare_data.sh --from STEP   # start at step STEP (1-9), e.g. after an interrupted run
#
# Inputs : the pinned Hugging Face sources of data/manifests/raw_sources.json (fetched by step 1 into $APEX_DATA_RAW,
#          default /mnt/apex-shared/apex_data/raw), the frozen dev-set manifests data/manifests/dev_*.json, the repository
#          and episode exclusions data/manifests/decontam_exclusions.json, the Qwen3.5-9B tokenizer (configs/eval/model.yaml
#          revision), the v6 selection data/manifests/v6/selection.json (v7 reuses v6's 861 retention rows), and the
#          decontamination screen index that scripts/data/run_screens.py reads (not distributed with this package).
#          Secrets come from the environment or the git-ignored .env: HF_TOKEN (rate limits), GITHUB_TOKEN (fork-graph
#          lookups of the repository screen); none is printed or stored.
# Outputs: data/pools/{retention,retention_v5,swe_v5,swe_v7}/, data/processed/v7/{terminal,sft}/ (rows Parquet files,
#          git-ignored), data/manifests/v7/{mixture,selection,split,gate_report,swe_exclusions,terminal_slice}.json,
#          data/manifests/screens_v7_{p1_pools,p2_rows}.json, data/README_v7.md (the data card) and the freeze marker
#          logs/coordination/v7/data_frozen_v7.json with the sha256 of every rows file and manifest. Steps 7-9 overwrite
#          the shipped v7 manifests: compare the new marker with data/manifests/v7/data_frozen_v7.json (the frozen v7r1).
#          Compare the sha256 of the rows files; manifests that record descriptive text (mixture.json, gate_report.json,
#          the data card) can differ from the frozen copies in that text.
#
# Exit   : non-zero at the first failing step (set -e); `build` exits 3 when a screen dropped a drawn SWE task (then
#          re-run step 5 with the updated data/manifests/v7/swe_exclusions.json and steps 7-9; see build_mixture_v5.py).
#
# Earlier data versions (same builders; exact arguments in each run's logs/<run_id>/metadata.json `command`):
#   v1 (frozen 2026-10-12 15:42 UTC, data/manifests/v1/): fetch_raw.py; build_dev_set.py --freeze;
#       build_teacher_leg.py (SWE pool data/pools/swe_teacher); build_terminal_slice.py (data/pools/terminal_teacher);
#       build_retention_slice.py (data/pools/retention) + merge_retention_topup.py (top-up merge, data card v1 section 5.3);
#       build_mixture.py items; run_screens.py --pass-name p1_pools ...; build_mixture.py build --cap 64k;
#       run_screens.py --pass-name p2_rows --pools '' --rows data/processed/sft/train_rows.parquet --rows data/processed/sft/dev_rows.parquet;
#       write_data_card.py; build_mixture.py freeze.
#   v5 (frozen 2026-10-13 07:29 UTC, data/manifests/v5/): build_swe_slice.py --out-dir data/pools/swe_v5
#       --keep-all-bands hard,medium --band-share easy=0.45 --repo-cap go=5 --repo-cap ts=5 --second-attempt-bands medium
#       --max-second-share 0.3 --exclude-task-ids data/manifests/v5/swe_exclusions.json; build_terminal_slice_v5.py --workers 32;
#       build_retention_slice.py --compose-v5 5250000 --slice-tokens 3500000 --out-dir data/pools/retention_v5;
#       build_mixture_v5.py items | build | report | freeze (with the run_screens.py v5_p1_pools / v5_p2_rows passes).
#   v6r2 (frozen 2026-10-13 17:46 UTC, data/manifests/v6/): build_swe_masks_v6.py (v5 SWE rows with read / unverified-
#       submit masks -> data/pools/swe_v6); build_terminal_slice_v6.py --workers 48; build_mixture_v5.py items --version v6;
#       build_mixture_v5.py build --version v6 --swe-trim-fraction 0.6 --retention-share 0.28 --retention-keep-whole hard_math
#       --revision v6r2; build_mixture_v5.py report / freeze --version v6 --revision v6r2 --decision ... (v6_p1/p2 screens).
#   stage 2 (designed, not trained): build_stage2_pairs.py (states, samples, V/R rules), audit_stage2_pairs.py,
#       freeze_stage2_pairs.py (v5 pairs, data/manifests/stage2/pairs_frozen_v5.json); the earlier filtered-SFT round:
#       calibrate_tasks.py, build_terminal_envs.py, run_rollouts.py, grade_episodes.py, build_stage2_set.py.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

PY=${PYTHON:-.venv/train/bin/python}
export PYTHONPATH=src
SCREEN_ENV=(env OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false)
DRY=0
FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --from) FROM=${2:?--from needs a step number}; shift 2 ;;
    -h|--help) sed -n '2,47p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ $DRY == 1 || -x "$PY" ]] || { echo "training interpreter not found: $PY (scripts/gpu_setup_train_env.sh or set PYTHON=...)" >&2; exit 2; }

step() {   # step N "description" command...
  local n=$1 desc=$2; shift 2
  (( n < FROM )) && { echo "[skip] step $n: $desc"; return 0; }
  echo "[step $n] $desc"
  echo "  \$ $(printf '%q ' "$@")"
  [[ $DRY == 1 ]] || "$@"
}

# The decisions recorded in the v7 gate report and marker (data/manifests/v7/gate_report.json `decisions`).
DECISION_1="v7, option 'same dose': teacher-only stage-1 data at v6's size; SWE one random passing attempt per task, recovery trials first, the adaptive thinking masks (routine turns <= 1,000 thinking tokens; decision-point turns up to 8K when not repetitive and ending in a tool call; > 8K never); retention 28 % with v6's rows."
DECISION_2="SWE decision points: a decision point needs a real failure before it (non-zero returncode or a strong failure marker) or is the turn before the first edit; long turns kept first when below 1 % of trained turns; the SWE slice is revision 2 (after a second quality read)."
DECISION_3="terminal slice: half DeepSeek-V4-Pro / half Kimi-K3 passing trials with per-turn guards (thinking <= 4K, parsed command, not repetitive, no confirmation turn); this replaces the terminal 2 % / 10 % long-turn cap."

# 1. Pinned raw sources (Open-SWE-Traces v1.2, termgrade-trajectories, Nemotron retention sets, auxiliary inputs);
#    every file is checked against data/manifests/raw_sources.json.
step 1 "fetch the pinned raw sources" "$PY" scripts/data/fetch_raw.py
# 2. Dev set (frozen 2026-10-12 before any training row was drawn): rebuild and check against data/manifests/dev_freeze.json
#    and the admitted lists; dev repositories and task ids are excluded by every builder below.
step 2 "check the frozen dev set" "$PY" scripts/data/build_dev_set.py
step 2 "check the frozen dev admission lists" "$PY" scripts/data/freeze_dev_admission.py --check
# 3. Retention pool: the v1 pool (Nemotron math / science / instruction following / chat), the ~70 hard-math rows, and the
#    revision-5 ranked composition that v5, v6 and v7 draw from (v7 keeps v6's 861 rows, step 8).
step 3 "retention pool (v1 build)" "$PY" scripts/data/build_retention_slice.py
step 3 "retention hard-math rows" "$PY" scripts/data/build_retention_slice.py --convergence-replay 1210000 --convergence-profile hard_math_v5 --out-dir data/pools/retention_v5 --workers 16
step 3 "retention v5 composition" "$PY" scripts/data/build_retention_slice.py --compose-v5 5250000 --slice-tokens 3500000 --out-dir data/pools/retention_v5
# 4. SWE supply: index, band and filter every Open-SWE-Traces Qwen3.8-27B mini-swe-agent attempt (licence allowlist,
#    repository and decontamination exclusions, dev exclusions, clean patch, reasoning on every turn); the v7 SWE builder
#    draws from this run's profiles (its --source-run).
step 4 "SWE supply and v5 SWE pool" "$PY" scripts/data/build_swe_slice.py --out-dir data/pools/swe_v5 --keep-all-bands hard,medium --band-share easy=0.45 --repo-cap go=5 --repo-cap ts=5 --second-attempt-bands medium --max-second-share 0.3 --exclude-task-ids data/manifests/v5/swe_exclusions.json
# 5. v7 SWE slice: one random passing attempt per task, recovery trials first, the adaptive thinking masks, v6's dose.
#    Pass --source-run <the run id of step 4> when it differs from the builder's default.
step 5 "v7 SWE slice" "$PY" scripts/data/build_swe_slice_v7.py --exclude-task-ids data/manifests/v7/swe_exclusions.json
# 6. v7 terminal slice: half DeepSeek-V4-Pro, half the Kimi-K3 body, per-turn guards.
step 6 "v7 terminal slice" "$PY" scripts/data/build_terminal_slice_v7.py --workers 48
# 7. Screen items and the p1 screens (task-level: canary strings, repository exclusion, task-name screen, n-gram and
#    embedding overlap, exact / near deduplication, train-vs-dev lineage). `items` prints the exact p1 command.
if (( FROM <= 7 )); then
  echo "[step 7] screen items and p1 screens"
  if [[ $DRY == 1 ]]; then
    echo "  \$ $PY scripts/data/build_mixture_v5.py items --version v7   (then the run_screens.py --pass-name v7_p1_pools command it prints)"
  else
    P1_CMD=$("$PY" scripts/data/build_mixture_v5.py items --version v7 | tail -n 1)
    echo "  \$ $P1_CMD"
    bash -c "$P1_CMD"
  fi
else
  echo "[skip] step 7: screen items and p1 screens"
fi
# 8. The mixture: units, retention rows from v6, the task-grouped 5 % validation shard; then the p2 screens over the
#    rendered rows (repeat 8 until p2 drops nothing; drops are sticky).
step 8 "v7 mixture" "$PY" scripts/data/build_mixture_v5.py build --version v7 --retention-rows-from data/manifests/v6/selection.json --revision v7r1
step 8 "p2 screens over the v7 rows" "${SCREEN_ENV[@]}" "$PY" scripts/data/run_screens.py --pass-name v7_p2_rows --pools '' --workers 24 --rows data/processed/v7/sft/train_rows.parquet --rows data/processed/v7/sft/dev_rows.parquet
# 9. Gate report and data card, freeze (sha256 of every rows file and manifest), marker.
step 9 "gate report and data card" "$PY" scripts/data/build_mixture_v5.py report --version v7 --decision "$DECISION_1" --decision "$DECISION_2" --decision "$DECISION_3"
step 9 "freeze" "$PY" scripts/data/build_mixture_v5.py freeze --version v7 --revision v7r1 --no-marker --decision "$DECISION_1" --decision "$DECISION_2" --decision "$DECISION_3"
step 9 "freeze marker" "$PY" scripts/data/build_mixture_v5.py marker --version v7
echo "done: compare logs/coordination/v7/data_frozen_v7.json with the frozen v7r1 marker data/manifests/v7/data_frozen_v7.json"
