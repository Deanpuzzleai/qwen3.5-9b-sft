#!/usr/bin/env bash
# Dispatch one evaluation run (works from any directory).
#   scripts/evaluate.sh swebench_pro   --version base --model-name <served id> --api-base http://<vllm>:8000/v1 [runner args]
#   scripts/evaluate.sh terminal_bench --version base --model-name <served id> --api-base http://<vllm>:8000/v1 [runner args]
#   scripts/evaluate.sh asibench       --version base --model-name <served id> --api-base http://<vllm>:8000/v1 [runner args]
#   scripts/evaluate.sh hle            --version base --model-name <served id> --api-base http://<vllm>:8000/v1 [runner args]
# Inputs : configs/eval/*.yaml locks, third_party/ checkouts from scripts/setup_eval_env.sh, a running model endpoint,
#          OPENAI_API_KEY in the git-ignored .env for the HLE judge and the ASI-Bench local-scoring judge (exported to the runner).
# Outputs: logs/<run_id>/ and results/<version>/<benchmark>/<run_id>/ (see src/apex_eval/<benchmark>.py docstrings and README).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
bench=${1:?usage: scripts/evaluate.sh <swebench_pro|terminal_bench|asibench|hle> [args]  (scripts/evaluate.sh <benchmark> --help lists the runner options)}; shift
case "$bench" in -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
case "$bench" in
  swebench_pro)   venv=harbor ;;
  terminal_bench) venv=harbor ;;
  asibench)       venv=asibench ;;
  hle)            venv=hle ;;
  *) echo "unknown benchmark: $bench" >&2; exit 2 ;;
esac
[[ -x ".venv/$venv/bin/python" ]] || { echo ".venv/$venv missing: run scripts/setup_eval_env.sh first" >&2; exit 1; }
exec env PYTHONPATH=src ".venv/$venv/bin/python" -m "apex_eval.$bench" "$@"
