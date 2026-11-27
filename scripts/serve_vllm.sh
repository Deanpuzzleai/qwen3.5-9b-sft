#!/usr/bin/env bash
# serve_vllm.sh -- serve one Qwen3.5-9B checkpoint directory (Base snapshot, or a merged SFT / stage-2 save) with
# the pinned image vllm/vllm-openai:v0.31.0 on one GPU, with run bookkeeping, and run the serving-VM checks S2-S7 and S9
# (REPORT.md section 3c of the archived 35B training-stack study; README, repository layout notes) against it. Production
# version of that study's probe serve_vllm_draft.sh.
#
# Run from the repository root on the serving GPU VM:
#   scripts/serve_vllm.sh start --ckpt-dir DIR --served-name qwen35-base [--port 8000] [--gpu 0] [--max-num-seqs 64]
#                               [--max-num-batched-tokens 8192] [--health-timeout 2400] [--extra-args "..."] [--no-digest-pin]
#                               [--bind 127.0.0.1]   # host address the port is published on; loopback by default (reach it from the
#                                                    # CPU host with scripts/ssh_tunnel_endpoint.sh; the endpoint has no auth)
#   scripts/serve_vllm.sh stop   --served-name NAME [--port 8000] | --run-dir logs/serve_<name>_<ts>
#   scripts/serve_vllm.sh status
#   scripts/serve_vllm.sh smoke  --served-name NAME [--port 8000] [--tool-choice auto|forced]
#   scripts/serve_vllm.sh s2     --run-dir logs/serve_<name>_<ts>                 # base-load criteria from the start log
#   scripts/serve_vllm.sh s3     --served-name NAME --ckpt-dir DIR [--prompt-tokens 250000]
#   scripts/serve_vllm.sh s4     --served-name NAME --ckpt-dir DIR [--concurrency 32,64,96] [--input-len 4096] [--output-len 1024] [--num-prompts 192]
#   scripts/serve_vllm.sh s5     --served-name NAME                                 # parsers live: 20 tool + 20 plain prompts
#   scripts/serve_vllm.sh s6     --served-name NAME --ckpt-dir DIR                  # /tokenize vs HF apply_chat_template
#   scripts/serve_vllm.sh s7     --served-name NAME --hf-reference logs/<T6 run>/hf_greedy_reference.jsonl
#   scripts/serve_vllm.sh s9     --served-name NAME --hf-reference <T6 run on the merged dir>   # merged checkpoint served like the base
# Inputs : the checkpoint directory (config.json, *.safetensors, tokenizer files, chat_template.jinja, generation_config.json),
#          Docker with the NVIDIA runtime, configs/eval/model.yaml (image, pinned digest, serving.flags, flags_not_used),
#          configs/compute.yaml (GPU rate), scripts/dev/gpu_checks/serve_checks.py (+ a Python with pyyaml and transformers
#          for the flag parsing and s3/s6: PYTHON, default .venv/train). HF_TOKEN is not needed: the container runs HF offline.
# Outputs: start -> logs/serve_<name>_<ts>/{metadata.json (image digest, full flag list, sha256 of config.json and the
#          safetensors index, hardware, start/end), eval.log (this script + `docker logs -f` of the container), metrics.jsonl
#          (health, S2 facts), errors.log, resources.csv (GPU hours while up, peak GPU memory from nvidia-smi sampled every
#          60 s in gpu_samples.csv), models.json}; stop finalizes resources.csv/metadata; every check writes its own
#          logs/serve_<check>_<ts>/ (five files) and prints PASS/FAIL with the measured numbers.
# Serving flags come from configs/eval/model.yaml serving.flags (identical for every version) plus --max-num-seqs /
#   --max-num-batched-tokens (serving.flags_fixed_after_pilot; the CLI values are recorded until the lock pins them).
#   Anything listed under serving.flags_not_used (speculative decoding, --language-model-only, ...) is refused in --extra-args.
# Served names: qwen35-{base,sft,stage2} (configs/eval/model.yaml served_model_names); names containing gpt, o1, o3, gemini-, claude, kimi or trinity are refused
#   (opencode / HLE key behaviour on them).
# Pass criteria: S2 /health < 20 min, no missing-weight warning, "Maximum concurrency for 262,144 tokens" >= 4, Mamba
#   cache mode/block lines logged; S3 250K prompt without OOM/preemption, TTFT logged; S4 zero errors and the largest
#   concurrency with p99 TTFT < 30 s (= recommended --max-num-seqs); S5 >= 95 % parsable tool calls, no leaks;
#   S6 identical token ids; S7 >= 68 % identical first 64 tokens, >= 76 % per-token (neutral sampling; base model 78 % / 81 %); S9 = no key warnings + S7 on the merged dir.
# GPU-only: start/stop/s2/s3/s4 and the live checks need the serving VM. On this CPU box only `bash -n`, the argument
#   handling and the Python request/parse plumbing (against a fake endpoint) were exercised.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

LOCK=configs/eval/model.yaml
HELPER=scripts/dev/gpu_checks/serve_checks.py
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/train/bin/python ]]; then PYTHON=.venv/train/bin/python; else PYTHON=python3; fi
fi
lock_get() { "$PYTHON" -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {"d": d}))' "$LOCK" "$1"; }
VLLM_IMAGE=$(lock_get 'd["serving"]["image"]')
IMAGE_DIGEST=$(lock_get 'd["serving"]["image_digest_amd64"]')
GPU_USD_PER_HOUR=$("$PYTHON" -c 'import yaml; print(yaml.safe_load(open("configs/compute.yaml"))["gpu_vm_usd_per_hour"])')
# each lock flag is "--name" or "--name value"; the value is kept verbatim (the generation-config value is JSON with spaces)
mapfile -t LOCK_FLAGS < <("$PYTHON" -c 'import sys, yaml; s = yaml.safe_load(open(sys.argv[1]))["serving"]; [print(x) for f in s["flags"] for x in f.split(None, 1)]' "$LOCK")
mapfile -t NOT_USED < <("$PYTHON" -c 'import sys, yaml; s = yaml.safe_load(open(sys.argv[1]))["serving"]; [print(x.split(":")[0].strip()) for x in s["flags_not_used"]]' "$LOCK")
LOCK_MAX_NUM_SEQS=$(lock_get 'd["serving"]["flags_fixed_after_pilot"]["max_num_seqs"]')
LOCK_MAX_BATCHED=$(lock_get 'd["serving"]["flags_fixed_after_pilot"]["max_num_batched_tokens"]')
VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-$HOME/.cache/vllm}

usage() { sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'; }
sub=${1:-}; [[ -n "$sub" ]] || { usage; exit 2; }; shift
case "$sub" in -h|--help|help) usage; exit 0 ;; esac
CKPT_DIR=""; SERVED_NAME=""; PORT=8000; GPU=0; BIND=127.0.0.1
MAX_NUM_SEQS=$([[ "$LOCK_MAX_NUM_SEQS" != None ]] && echo "$LOCK_MAX_NUM_SEQS" || echo 64)
MAX_BATCHED=$([[ "$LOCK_MAX_BATCHED" != None ]] && echo "$LOCK_MAX_BATCHED" || echo 8192)
HEALTH_TIMEOUT=2400; EXTRA_ARGS=""; DIGEST_PIN=1; RUN_DIR=""; TOOL_CHOICE=auto; PROMPT_TOKENS=250000
CONCURRENCY=32,64,96; INPUT_LEN=4096; OUTPUT_LEN=1024; NUM_PROMPTS=192; HF_REFERENCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir) CKPT_DIR=$2; shift 2 ;;
    --served-name) SERVED_NAME=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --bind) BIND=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS=$2; shift 2 ;;
    --max-num-batched-tokens) MAX_BATCHED=$2; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT=$2; shift 2 ;;
    --extra-args) EXTRA_ARGS=$2; shift 2 ;;
    --no-digest-pin) DIGEST_PIN=0; shift ;;
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --tool-choice) TOOL_CHOICE=$2; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS=$2; shift 2 ;;
    --concurrency) CONCURRENCY=$2; shift 2 ;;
    --input-len) INPUT_LEN=$2; shift 2 ;;
    --output-len) OUTPUT_LEN=$2; shift 2 ;;
    --num-prompts) NUM_PROMPTS=$2; shift 2 ;;
    --hf-reference) HF_REFERENCE=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
BASE_URL="http://127.0.0.1:${PORT}"
CONTAINER="vllm-${SERVED_NAME}-${PORT}"

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_DIR/eval.log"; }
record_error() { echo "[$(ts)] $*" >> "$RUN_DIR/errors.log"; log "ERROR: $*"; }
metric() { printf '{"ts":"%s",%s}\n' "$(ts)" "$1" >> "$RUN_DIR/metrics.jsonl"; }
need() { [[ -n "${!1:-}" ]] || { echo "--$(echo "$1" | tr 'A-Z_' 'a-z-') is required for $sub" >&2; exit 2; }; }
json_get() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2"; }
container_run_dir() { docker inspect --format '{{index .Config.Labels "apex.run_dir"}}' "$CONTAINER" 2>/dev/null || true; }
image_ref() {  # the amd64 digest pinned in the lock (falls back to the tag with a warning when --no-digest-pin is given)
  if [[ $DIGEST_PIN -eq 1 ]]; then echo "${VLLM_IMAGE%%:*}@${IMAGE_DIGEST}"; else echo "$VLLM_IMAGE"; fi
}
new_check_run() {  # name -> sets RUN_DIR for a check run and writes the metadata skeleton
  RUN_DIR="logs/serve_${1}_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_DIR"; : > "$RUN_DIR/errors.log"
  python3 - "$RUN_DIR" "$1" "$SERVED_NAME" "$BASE_URL" "$(ts)" "$0 $sub $*" <<'PY'
import json, os, socket, subprocess, sys
d, kind, name, url, start, cmd = sys.argv[1:7]
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
smi = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null"], capture_output=True, text=True).stdout.strip()
json.dump({"run_id": os.path.basename(d), "kind": f"serve_{kind}", "status": "RUNNING", "served_model_name": name, "endpoint": url,
           "command": cmd, "code_commit": git, "host": socket.gethostname(), "gpus": smi.splitlines(), "start_time": start, "end_time": None,
           "seed": 0, "data_version": "synthetic (own prompts)"}, open(os.path.join(d, "metadata.json"), "w"), indent=2)
PY
  T0_CHECK=$(date +%s)
}
finish_check_run() {  # status, summary-json
  local dur=$(( $(date +%s) - T0_CHECK ))
  printf 'metric,value,unit,note\nduration_sec,%s,s,wall time of this check\ngpu_count,1,count,the serving GPU stays allocated during the check\ngpu_hours,%s,h,duration x 1\ncost_usd,%s,USD,gpu_hours x %s\n' \
    "$dur" "$(python3 -c "print(round($dur/3600,4))")" "$(python3 -c "print(round($dur/3600*$GPU_USD_PER_HOUR,3))")" "$GPU_USD_PER_HOUR" > "$RUN_DIR/resources.csv"
  [[ -s "$RUN_DIR/errors.log" ]] || echo none > "$RUN_DIR/errors.log"
  python3 - "$RUN_DIR/metadata.json" "$1" "$2" "$(ts)" <<'PY'
import json, sys
p, status, summary, end = sys.argv[1:5]
m = json.load(open(p)); m.update(status=status, end_time=end, summary=json.loads(summary)); json.dump(m, open(p, "w"), indent=2)
PY
  echo "[serve_vllm $sub] $1 $2"; echo "RUN_DIR=$RUN_DIR"
  [[ "$1" == PASS ]]
}
delegate() {  # python helper subcommand with the serve run dir cross-reference; its exit code is the verdict
  local srd; srd=$(container_run_dir)
  "$PYTHON" "$HELPER" "$@" --base-url "$BASE_URL" --model "$SERVED_NAME" ${srd:+--serve-run-dir "$srd"}
}
s2_checks() {  # start-run dir -> prints a JSON summary of the base-load criteria; exit 1 on failure
  python3 - "$1" <<'PY'
import json, re, sys
d = sys.argv[1]
log = open(f"{d}/eval.log", errors="replace").read()
meta = json.load(open(f"{d}/metadata.json"))
m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log)
conc = float(m.group(2)) if m else None
mamba = [ln for ln in log.splitlines() if re.search(r"Mamba cache mode|mamba_block_size|kv cache group sizes|kv lcm block sizes|GPU KV cache size|Available KV cache memory", ln)]
bad = [ln for ln in log.splitlines() if re.search(r"not initialized from checkpoint|missing keys|Missing keys|weights were not used|Unexpected weights", ln)]
secs = meta.get("seconds_to_healthy")
checks = {"healthy_under_20min": secs is not None and secs < 1200, "no_missing_weight_warning": not bad,
          "max_concurrency_ge_4": conc is not None and conc >= 4, "mamba_lines_logged": bool(mamba)}
out = {"seconds_to_healthy": secs, "max_concurrency_262k": conc, "n_mamba_lines": len(mamba), "bad_lines": bad[:3], "checks": checks,
       "mamba_lines": mamba[:6]}
print(json.dumps(out))
sys.exit(0 if all(checks.values()) else 1)
PY
}

case "$sub" in
# ------------------------------------------------------------------------------------------------------ start
start)
  need CKPT_DIR; need SERVED_NAME
  [[ -f "$CKPT_DIR/config.json" ]] || { echo "no config.json in $CKPT_DIR" >&2; exit 2; }
  if echo "$SERVED_NAME" | /usr/bin/grep -q -i -E 'gpt|(^|[^a-z])o1|(^|[^a-z])o3|gemini-|claude|kimi|trinity'; then
    echo "served name '$SERVED_NAME' contains a substring that changes opencode/HLE behaviour; use qwen35-{base,sft,stage2}" >&2; exit 2; fi
  for flag in "${NOT_USED[@]}" --speculative-config; do
    if [[ -n "$EXTRA_ARGS" ]] && echo "$EXTRA_ARGS" | /usr/bin/grep -q -- "$flag"; then echo "$flag is refused: configs/eval/model.yaml serving.flags_not_used" >&2; exit 2; fi
  done
  if docker ps -a --format '{{.Names}}' | /usr/bin/grep -qx "$CONTAINER"; then echo "container $CONTAINER exists; run 'scripts/serve_vllm.sh stop --served-name $SERVED_NAME --port $PORT' first" >&2; exit 2; fi
  RUN_DIR="logs/serve_${SERVED_NAME}_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_DIR"; : > "$RUN_DIR/errors.log"
  IMAGE=$(image_ref); START_TS=$(ts); T0=$(date +%s)
  [[ "$IMAGE" == "$VLLM_IMAGE" ]] && log "WARNING: image digest not pinned (manifest list missing or --no-digest-pin)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "pulling $IMAGE"; docker pull "$IMAGE" >> "$RUN_DIR/eval.log" 2>&1; }
  LOCAL_DIGEST=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo unknown)
  VLLM_ARGS=(/model --served-model-name "$SERVED_NAME" "${LOCK_FLAGS[@]}" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED"
             --host 0.0.0.0 --port 8000)
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_ARGS" ]] && VLLM_ARGS+=($EXTRA_ARGS)
  mkdir -p "$VLLM_CACHE_DIR"
  python3 - "$RUN_DIR/metadata.json" "$RUN_DIR" "$IMAGE" "$LOCAL_DIGEST" "$CKPT_DIR" "$SERVED_NAME" "$PORT" "$GPU" "$START_TS" "$CONTAINER" "$(printf '%s\n' "${VLLM_ARGS[@]}")" <<'PY'
import hashlib, json, os, socket, subprocess, sys
p, run_dir, image, local_digest, ckpt, name, port, gpu, start, container, flags = sys.argv[1:12]
def sha(f):
    path = os.path.join(ckpt, f)
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
smi = subprocess.run(["bash", "-c", f"nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i {gpu} 2>/dev/null"], capture_output=True, text=True).stdout.strip()
cfg = json.load(open(os.path.join(ckpt, "config.json")))
meta = {"run_id": os.path.basename(run_dir), "kind": "vllm_serve", "status": "STARTING", "image": image, "image_local_repo_digest": local_digest,
        "checkpoint_dir": os.path.abspath(ckpt), "checkpoint_sha256": {"config.json": sha("config.json"), "model.safetensors.index.json": sha("model.safetensors.index.json"),
        "chat_template.jinja": sha("chat_template.jinja"), "generation_config.json": sha("generation_config.json")},
        "architectures": cfg.get("architectures"), "served_model_name": name, "port": int(port), "gpu_index": int(gpu), "container": container,
        "vllm_args": flags.splitlines(), "flags_source": "configs/eval/model.yaml serving.flags", "speculative_decoding": "none (locked)", "hf_offline": True,
        "hardware": {"host": socket.gethostname(), "gpu": smi}, "code_commit": git, "command": " ".join(sys.argv[:1] + ["start"]),
        "start_time": start, "end_time": None, "seed": None, "data_version": None}
json.dump(meta, open(p, "w"), indent=2)
PY
  log "docker run $IMAGE as $CONTAINER: ckpt=$CKPT_DIR name=$SERVED_NAME gpu=$GPU port=$BIND:$PORT"
  log "vllm args: ${VLLM_ARGS[*]}"
  docker run -d --name "$CONTAINER" --gpus "\"device=${GPU}\"" --ipc=host --shm-size 16g -p "${BIND}:${PORT}:8000" \
    --label "apex.run_dir=$RUN_DIR" --label "apex.served_name=$SERVED_NAME" --label "apex.port=$PORT" \
    -v "$(realpath "$CKPT_DIR"):/model:ro" -v "$VLLM_CACHE_DIR:/root/.cache/vllm" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
    "$IMAGE" "${VLLM_ARGS[@]}" >> "$RUN_DIR/eval.log"
  # background: nvidia-smi sampler (every 60 s) and the container log follower, both into the run dir
  echo "timestamp,memory_used_mib,utilization_gpu_pct" > "$RUN_DIR/gpu_samples.csv"
  setsid nohup bash -c "while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ' >> '$RUN_DIR/gpu_samples.csv'; sleep 60; done" >/dev/null 2>&1 &
  echo $! > "$RUN_DIR/sampler.pid"
  setsid nohup docker logs -f "$CONTAINER" >> "$RUN_DIR/eval.log" 2>&1 &
  echo $! > "$RUN_DIR/logs.pid"
  log "waiting for $BASE_URL/health (timeout ${HEALTH_TIMEOUT}s; bf16 load + compile + cudagraph capture)"
  while true; do
    if curl -sf -o /dev/null --max-time 5 "$BASE_URL/health"; then SECS=$(( $(date +%s) - T0 )); break; fi
    if ! docker ps --format '{{.Names}}' | /usr/bin/grep -qx "$CONTAINER"; then
      record_error "container exited before becoming healthy (see eval.log)"; python3 - "$RUN_DIR/metadata.json" <<'PY'
import json, sys; p = sys.argv[1]; m = json.load(open(p)); m["status"] = "FAIL"; json.dump(m, open(p, "w"), indent=2)
PY
      exit 1
    fi
    if (( $(date +%s) - T0 > HEALTH_TIMEOUT )); then record_error "health timeout after ${HEALTH_TIMEOUT}s"; exit 1; fi
    sleep 15
  done
  log "healthy after ${SECS}s"
  metric "\"event\":\"healthy\",\"seconds_to_healthy\":$SECS"
  curl -s --max-time 10 "$BASE_URL/v1/models" > "$RUN_DIR/models.json" || true
  curl -s --max-time 10 "$BASE_URL/version" > "$RUN_DIR/version.json" || true
  python3 - "$RUN_DIR/metadata.json" "$SECS" "$RUN_DIR/version.json" <<'PY'
import json, sys
p, secs, vfile = sys.argv[1:4]
m = json.load(open(p)); m.update(status="RUNNING", seconds_to_healthy=int(secs))
try: m["vllm_version"] = json.load(open(vfile))
except Exception: m["vllm_version"] = None
json.dump(m, open(p, "w"), indent=2)
PY
  sleep 2   # let the log follower flush the startup lines before S2 reads them
  S2=$(s2_checks "$RUN_DIR" || true); metric "\"event\":\"s2_facts\",\"facts\":$S2"
  log "S2 facts: $S2"
  printf 'metric,value,unit,note\nseconds_to_healthy,%s,s,docker run -> /health 200\nstatus,RUNNING,,finalized by stop\n' "$SECS" > "$RUN_DIR/resources.csv"
  echo "API_BASE=$BASE_URL/v1"; echo "RUN_DIR=$RUN_DIR"
  ;;
# ------------------------------------------------------------------------------------------------------- stop
stop)
  if [[ -z "$RUN_DIR" ]]; then need SERVED_NAME; RUN_DIR=$(container_run_dir); fi
  [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "cannot find the start run dir (container $CONTAINER not running? pass --run-dir)" >&2; exit 1; }
  CONTAINER=$(json_get "$RUN_DIR/metadata.json" "d['container']")
  log "stopping $CONTAINER"
  docker stop -t 30 "$CONTAINER" >/dev/null 2>&1 || log "container was not running"
  for f in sampler.pid logs.pid; do pid=$(cat "$RUN_DIR/$f" 2>/dev/null || true); [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
  docker logs "$CONTAINER" > "$RUN_DIR/container_final.log" 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
  python3 - "$RUN_DIR" "$GPU_USD_PER_HOUR" "$(ts)" <<'PY'
import csv, datetime as dt, json, sys
d, rate, end = sys.argv[1], float(sys.argv[2]), sys.argv[3]
m = json.load(open(f"{d}/metadata.json"))
t0 = dt.datetime.strptime(m["start_time"], "%Y-%m-%dT%H:%M:%SZ"); t1 = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
dur = (t1 - t0).total_seconds()
peak, n = None, 0
try:
    for row in csv.DictReader(open(f"{d}/gpu_samples.csv")):
        n += 1; v = float(row["memory_used_mib"]); peak = v if peak is None or v > peak else peak
except Exception:
    pass
rows = [("duration_sec", round(dur, 1), "s", "docker run -> stop"), ("seconds_to_healthy", m.get("seconds_to_healthy"), "s", "docker run -> /health 200"),
        ("peak_gpu_memory_gb", round(peak / 1024, 2) if peak is not None else "n/a", "GB", f"max of {n} nvidia-smi samples (60 s)"),
        ("gpu_count", 1, "count", "one GPU per served checkpoint"), ("gpu_hours", round(dur / 3600, 4), "h", "duration x 1"),
        ("cost_usd", round(dur / 3600 * rate, 3), "USD", f"gpu_hours x {rate} (GPU_USD_PER_HOUR)"), ("status", "STOPPED", "", "finalized by stop")]
with open(f"{d}/resources.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "value", "unit", "note"]); w.writerows(rows)
m.update(status="STOPPED", end_time=end, peak_gpu_memory_gb=rows[2][1]); json.dump(m, open(f"{d}/metadata.json", "w"), indent=2)
PY
  [[ -s "$RUN_DIR/errors.log" ]] || echo none > "$RUN_DIR/errors.log"
  log "finalized $RUN_DIR/resources.csv"
  ;;
status)
  docker ps --filter 'label=apex.served_name' --format 'table {{.Names}}\t{{.Status}}\t{{.Label "apex.port"}}\t{{.Label "apex.run_dir"}}'
  ;;
# ------------------------------------------------------------------------------------------------- live checks
smoke) need SERVED_NAME; delegate smoke --tool-choice "$TOOL_CHOICE" ;;
s3) need SERVED_NAME; need CKPT_DIR; delegate s3 --tokenizer "$CKPT_DIR" --prompt-tokens "$PROMPT_TOKENS" --container "$CONTAINER" ;;
s5) need SERVED_NAME; delegate s5 ;;
s6) need SERVED_NAME; need CKPT_DIR; delegate s6 --tokenizer "$CKPT_DIR" ;;
s7) need SERVED_NAME; need HF_REFERENCE; delegate s7 --hf-reference "$HF_REFERENCE" ;;
s2)
  [[ -n "$RUN_DIR" ]] || { need SERVED_NAME; RUN_DIR=$(container_run_dir); }
  [[ -d "$RUN_DIR" ]] || { echo "start run dir not found; pass --run-dir" >&2; exit 2; }
  SRC=$RUN_DIR; new_check_run s2 "$@"
  log "evaluating base-load criteria of $SRC"
  if S2=$(s2_checks "$SRC"); then finish_check_run PASS "$S2"; else record_error "S2 criteria failed: $S2"; finish_check_run FAIL "$S2"; fi
  ;;
s4)
  need SERVED_NAME; need CKPT_DIR
  new_check_run s4 "$@"
  IMAGE=$(image_ref); mkdir -p "$RUN_DIR/bench"
  summary="{"; best=""; all_ok=1
  for C in ${CONCURRENCY//,/ }; do
    out="s4_c${C}.json"
    log "vllm bench serve: random ${INPUT_LEN}/${OUTPUT_LEN}, ${NUM_PROMPTS} prompts, concurrency ${C}"
    docker run --rm --network host --entrypoint vllm -v "$(realpath "$CKPT_DIR"):/model:ro" -v "$(realpath "$RUN_DIR/bench"):/results" \
      -e HF_HUB_OFFLINE=1 "$IMAGE" bench serve --backend openai-chat --base-url "$BASE_URL" --endpoint /v1/chat/completions \
      --model /model --served-model-name "$SERVED_NAME" --tokenizer /model --dataset-name random --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" --max-concurrency "$C" --ignore-eos --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --result-dir /results --result-filename "$out" \
      >> "$RUN_DIR/eval.log" 2>&1 || { record_error "bench at concurrency $C failed"; all_ok=0; continue; }
    row=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps({k:d.get(k) for k in ('completed','p99_ttft_ms','mean_ttft_ms','p99_tpot_ms','output_throughput','request_throughput','duration')}|{'n_errors':sum(1 for e in d.get('errors',[]) if e)}))" "$RUN_DIR/bench/$out")
    metric "\"event\":\"bench\",\"concurrency\":$C,\"result\":$row"; log "concurrency $C: $row"
    summary+="\"c$C\":$row,"
    ok=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(int(d['completed']==$NUM_PROMPTS and d['n_errors']==0 and (d['p99_ttft_ms'] or 1e9)<30000))" "$row")
    [[ "$ok" == 1 ]] && best=$C
    python3 -c "import json,sys;d=json.loads(sys.argv[1]);sys.exit(0 if d['completed']==$NUM_PROMPTS and d['n_errors']==0 else 1)" "$row" || all_ok=0
  done
  summary+="\"recommended_max_num_seqs\":${best:-null}}"
  if [[ $all_ok -eq 1 && -n "$best" ]]; then finish_check_run PASS "$summary"; else record_error "S4: errors or no concurrency with p99 TTFT < 30 s"; finish_check_run FAIL "$summary"; fi
  ;;
s9)
  need SERVED_NAME; need HF_REFERENCE
  SRC=$(container_run_dir); [[ -d "$SRC" ]] || { echo "merged checkpoint not served as $CONTAINER; start it first: scripts/serve_vllm.sh start --ckpt-dir <merged> --served-name $SERVED_NAME" >&2; exit 2; }
  new_check_run s9 "$@"
  log "S9: merged checkpoint served from $SRC; key-warning check + greedy agreement vs $HF_REFERENCE"
  keyfail=0; S2=$(s2_checks "$SRC" || true)
  python3 -c "import json,sys;sys.exit(0 if json.loads(sys.argv[1])['checks']['no_missing_weight_warning'] else 1)" "$S2" || { keyfail=1; record_error "merged load produced key warnings: $S2"; }
  if delegate s7 --hf-reference "$HF_REFERENCE" --run-name serve_s9_s7 && [[ $keyfail -eq 0 ]]; then finish_check_run PASS "{\"s2\":$S2}"; else finish_check_run FAIL "{\"s2\":$S2}"; fi
  ;;
*) echo "unknown subcommand: $sub" >&2; usage; exit 2 ;;
esac
