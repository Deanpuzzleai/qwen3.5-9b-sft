#!/usr/bin/env bash
# gpu_setup_train_env.sh -- create the pinned training environment on a fresh GPU VM and write the dependency locks.
#
# Run from the repository root on the GPU VM (Ubuntu, NVIDIA driver present, Docker present):
#   bash scripts/gpu_setup_train_env.sh                    # torch 2.14.1 CUDA 13 build + pins + fla + causal-conv1d source build
#   bash scripts/gpu_setup_train_env.sh --fallback-torch210 # REPORT.md fallback: torch 2.10.x+cu130 with prebuilt cu13torch2.10 wheels
#   bash scripts/gpu_setup_train_env.sh --skip-causal-conv1d | --causal-conv1d-in-docker | --print-only
# Inputs : configs/train_stack_check/env/freeze.txt (the CPU-validated pins: transformers 5.19.0, peft 0.21.2, trl 1.15.0,
#          accelerate 1.15.0, datasets 5.1.0, ...), network access to PyPI / download.pytorch.org / GitHub (and Docker
#          Hub for --causal-conv1d-in-docker), uv (installed into ~/.local/bin if missing).
# Outputs: .venv/train (python 3.12), requirements.lock (`uv pip freeze` of the GPU env = the project's dependency lock),
#          and one logs/gpu_setup_<ts>/{metadata.json, eval.log, metrics.jsonl (phase durations), errors.log, resources.csv,
#          toolchain_versions.txt (torch/cuda/driver/GPU/kernels import status)}.
# Variables (override in the environment): TORCH_VERSION=2.14.1, CUDA_VARIANT=cu130 (torch 2.14.1 ships cu130/cu132/
#   cu126, no cu128; cu126 has no sm_100, so Blackwell needs cu130: memory_throughput_plan/FINDINGS.md 5.1),
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/$CUDA_VARIANT, FLA_VERSION=0.5.2, CAUSAL_CONV1D_VERSION=1.7.0,
#   MAX_JOBS (ninja parallelism for the causal-conv1d build, default min(nproc,16)), VENV=.venv/train,
#   FALLBACK_TORCH_SPEC="torch==2.10.*", DEVEL_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel (nvcc 13.0 for the
#   in-docker wheel build), PINS_SRC=configs/train_stack_check/env/freeze.txt.
# Decisions (REPORT.md section 1 item 4): no flash-attn (sdpa), no `kernels` hub builds, no bitsandbytes; fla 0.5.2 is
#   pure Python on Triton 3.8 (T2 parity is the acceptance test); causal-conv1d has no torch-2.14 wheel -> sdist build
#   with ninja (optional: the cuDNN depthwise conv fallback is correct, only slower).
# Idempotent: existing venv/packages are verified and left alone; lock files are regenerated on every run.
# GPU-only: the torch CUDA build and the causal-conv1d compile need the GPU VM; on this CPU box only --print-only
#   and `bash -n` were exercised.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

TORCH_VERSION=${TORCH_VERSION:-2.14.1}
CUDA_VARIANT=${CUDA_VARIANT:-cu130}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_VARIANT}}
FLA_VERSION=${FLA_VERSION:-0.5.2}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-1.7.0}
VENV=${VENV:-.venv/train}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
PINS_SRC=${PINS_SRC:-configs/train_stack_check/env/freeze.txt}
FALLBACK_TORCH_SPEC=${FALLBACK_TORCH_SPEC:-torch==2.10.*}
DEVEL_IMAGE=${DEVEL_IMAGE:-pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel}
GPU_USD_PER_HOUR=${GPU_USD_PER_HOUR:-$(sed -n 's/^gpu_vm_usd_per_hour:[[:space:]]*\([0-9.]*\).*/\1/p' configs/compute.yaml)}   # configs/compute.yaml (a fresh VM has no pyyaml)
CAUSAL_CONV1D_RELEASES=https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}
nproc_=$(nproc 2>/dev/null || echo 8); MAX_JOBS=${MAX_JOBS:-$(( nproc_ < 16 ? nproc_ : 16 ))}
FALLBACK=0; SKIP_CC1D=0; CC1D_DOCKER=0; PRINT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --fallback-torch210) FALLBACK=1 ;;
    --skip-causal-conv1d) SKIP_CC1D=1 ;;
    --causal-conv1d-in-docker) CC1D_DOCKER=1 ;;
    --print-only) PRINT_ONLY=1 ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
[[ -f README.md && -f "$PINS_SRC" ]] || { echo "run from the repository root ($PINS_SRC must exist)" >&2; exit 1; }
if [[ $FALLBACK -eq 1 ]]; then TORCH_SPEC="$FALLBACK_TORCH_SPEC"; TORCH_TAG=torch2.10; else TORCH_SPEC="torch==${TORCH_VERSION}"; TORCH_TAG="torch${TORCH_VERSION%.*}"; fi

RUN_ID="gpu_setup_$(date -u +%Y%m%d_%H%M%S)"; RUN_DIR="logs/$RUN_ID"; mkdir -p "$RUN_DIR"; : > "$RUN_DIR/errors.log"
T_START=$(date +%s)
ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_DIR/eval.log"; }
record_error() { echo "[$(ts)] $*" >> "$RUN_DIR/errors.log"; log "ERROR: $*"; }
metric() { printf '{"ts":"%s",%s}\n' "$(ts)" "$1" >> "$RUN_DIR/metrics.jsonl"; }
run() {  # phase name, command...: logs, times, records the phase; honours --print-only
  local name=$1; shift
  log "[$name] $*"
  [[ $PRINT_ONLY -eq 1 ]] && return 0
  local t0; t0=$(date +%s)
  if "$@" >> "$RUN_DIR/eval.log" 2>&1; then metric "\"phase\":\"$name\",\"seconds\":$(( $(date +%s) - t0 )),\"ok\":true"
  else local rc=$?; metric "\"phase\":\"$name\",\"seconds\":$(( $(date +%s) - t0 )),\"ok\":false"; record_error "$name failed (rc=$rc): $*"; return $rc; fi
}
UVPY() { "$VENV/bin/python" "$@"; }
finish() {  # status
  local dur=$(( $(date +%s) - T_START ))
  printf 'metric,value,unit,note\nduration_sec,%s,s,setup wall time\ngpu_count,%s,count,GPUs on this VM\ngpu_hours,%s,h,duration x gpu_count (VM billed while setting up)\ncost_usd,%s,USD,gpu_hours x %s (GPU_USD_PER_HOUR)\n' \
    "$dur" "${GPU_COUNT:-0}" "$(python3 -c "print(round($dur/3600*${GPU_COUNT:-0},4))")" "$(python3 -c "print(round($dur/3600*${GPU_COUNT:-0}*$GPU_USD_PER_HOUR,3))")" "$GPU_USD_PER_HOUR" > "$RUN_DIR/resources.csv"
  [[ -s "$RUN_DIR/errors.log" ]] || echo none > "$RUN_DIR/errors.log"
  python3 - "$RUN_DIR/metadata.json" "$1" "$RUN_ID" "$TORCH_SPEC" "$TORCH_INDEX_URL" "$VENV" "$FALLBACK" "$(ts)" "$*" <<'PY'
import json, os, socket, subprocess, sys
p, status, run_id, torch_spec, index, venv, fallback, end = sys.argv[1:9]
prev = json.load(open(p)) if os.path.exists(p) else {}
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
smi = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null"], capture_output=True, text=True).stdout.strip()
meta = {**prev, "run_id": run_id, "kind": "gpu_setup_train_env", "status": status, "command": " ".join(os.environ.get("SETUP_ARGV", "").split()),
        "torch_spec": torch_spec, "torch_index_url": index, "venv": venv, "fallback_torch210": fallback == "1",
        "code_commit": git, "host": socket.gethostname(), "gpus": smi.splitlines(), "end_time": end, "seed": None, "data_version": None,
        "start_time": prev.get("start_time")}
json.dump(meta, open(p, "w"), indent=2)
PY
  log "status=$status run dir $RUN_DIR"
}
export SETUP_ARGV="$0 $*"
python3 - "$RUN_DIR/metadata.json" "$(ts)" <<'PY'
import json, sys
json.dump({"start_time": sys.argv[2]}, open(sys.argv[1], "w"))
PY
metric "\"phase\":\"start\",\"print_only\":$PRINT_ONLY,\"fallback_torch210\":$FALLBACK,\"torch_spec\":\"$TORCH_SPEC\",\"torch_index_url\":\"$TORCH_INDEX_URL\",\"max_jobs\":$MAX_JOBS"

# ---- 1. preflight: driver, docker, uv, python 3.12 -----------------------------------------------------------
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  log "nvidia-smi: $GPU_COUNT GPU(s), driver $DRIVER ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1))"
  if [[ "$CUDA_VARIANT" == cu130 && "${DRIVER%%.*}" -lt 580 ]]; then record_error "driver $DRIVER < R580: the cu130 torch build needs an R580+ driver (upgrade the driver or use CUDA_VARIANT=cu126 on non-Blackwell cards)"; fi
else
  log "WARNING: nvidia-smi not found (no GPU here?); continuing so --print-only works"
fi
command -v docker >/dev/null 2>&1 || log "WARNING: docker not found (needed only for --causal-conv1d-in-docker and for serving)"
# Triton JIT-compiles its CUDA driver stub against Python.h at first use (fla's GDN kernels, the causal-conv1d build); a venv
# on the distro Python without its -dev package fails there (seen 2026-10-12 on Nebius Ubuntu 24.04 CUDA images)
PY_INC=$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null || true)
if [[ -n "$PY_INC" && ! -f "$PY_INC/Python.h" ]]; then
  if [[ $PRINT_ONLY -eq 0 ]] && sudo -n true 2>/dev/null; then
    run install_python_dev sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-dev
  else
    record_error "Python.h missing under $PY_INC: install python3.12-dev (Triton and the causal-conv1d build need it)"
  fi
fi
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  run install_uv bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
export PATH="$HOME/.local/bin:$PATH"
log "uv: $(uv --version 2>/dev/null || echo 'not installed (print-only)')"

# ---- 2. venv ---------------------------------------------------------------------------------------------------
if [[ -x "$VENV/bin/python" ]]; then
  have_torch=$(UVPY -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
  if [[ $FALLBACK -eq 1 && "$have_torch" == 2.14.* ]] || [[ $FALLBACK -eq 0 && "$have_torch" == 2.10.* ]]; then
    bak="${VENV}.bak-$(date -u +%Y%m%d_%H%M%S)"; log "existing venv has torch $have_torch; moving it aside to $bak (nothing is deleted)"
    run move_venv_aside mv "$VENV" "$bak"
  fi
fi
[[ -x "$VENV/bin/python" ]] || run create_venv uv venv --python "$PYTHON_VERSION" "$VENV"

# ---- 3. torch (CUDA build from the PyTorch index), then the CPU-validated pins minus torch ----------------------
run install_torch uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
PINS="$RUN_DIR/requirements-train.in"
{ echo "# pins from $PINS_SRC (CPU-validated 2026-10-10) without torch; torch comes from $TORCH_INDEX_URL ($TORCH_SPEC)";
  /usr/bin/grep -v -E '^torch==' "$PINS_SRC"; echo "ninja"; echo "wheel"; } > "$PINS"
run install_pins uv pip install --python "$VENV/bin/python" -r "$PINS"
run pip_check uv pip check --python "$VENV/bin/python"

# ---- 4. flash-linear-attention (pure Python, Triton from torch; no [cuda] extra: it would re-resolve torch) ------
run install_fla uv pip install --python "$VENV/bin/python" "flash-linear-attention==${FLA_VERSION}" "fla-core==${FLA_VERSION}"

# ---- 5. causal-conv1d: prebuilt wheel (fallback env) or sdist build with ninja (host nvcc or the devel image) -----
CC1D_STATUS="skipped (--skip-causal-conv1d)"
if [[ $SKIP_CC1D -eq 0 ]]; then
  pyver=$(UVPY -c "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')" 2>/dev/null || echo cp312)
  wheel_url="${CAUSAL_CONV1D_RELEASES}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu13${TORCH_TAG}cxx11abiTRUE-${pyver}-${pyver}-linux_x86_64.whl"
  if [[ $FALLBACK -eq 1 ]] && run install_causal_conv1d_wheel uv pip install --python "$VENV/bin/python" "$wheel_url"; then
    CC1D_STATUS="prebuilt wheel $wheel_url"
  elif [[ $CC1D_DOCKER -eq 1 ]]; then
    mkdir -p third_party/wheels
    # Build inside the CUDA 13.0 devel image (nvcc) against the same torch build, then install the wheel into the venv.
    run build_causal_conv1d_docker docker run --rm -v "$PWD/third_party/wheels:/out" -e MAX_JOBS="$MAX_JOBS" -e CAUSAL_CONV1D_FORCE_BUILD=TRUE \
      "$DEVEL_IMAGE" bash -lc "pip install -q uv && uv venv /tmp/b --python ${PYTHON_VERSION} && . /tmp/b/bin/activate && \
        uv pip install --index-url ${TORCH_INDEX_URL} ${TORCH_SPEC} ninja wheel setuptools packaging pip && \
        pip wheel --no-build-isolation --no-deps causal-conv1d==${CAUSAL_CONV1D_VERSION} -w /out" \
      && run install_causal_conv1d_wheel bash -c "uv pip install --python '$VENV/bin/python' third_party/wheels/causal_conv1d-${CAUSAL_CONV1D_VERSION}*.whl" \
      && CC1D_STATUS="built in $DEVEL_IMAGE (MAX_JOBS=$MAX_JOBS)" || CC1D_STATUS="docker build FAILED (see errors.log); fallback conv in use"
  elif command -v nvcc >/dev/null 2>&1 || [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    log "nvcc: $(nvcc --version | tail -1)"
    if run build_causal_conv1d_host env CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS" uv pip install --python "$VENV/bin/python" --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}"; then
      CC1D_STATUS="built from sdist on the host (nvcc, MAX_JOBS=$MAX_JOBS)"
    else CC1D_STATUS="host build FAILED (see errors.log); fallback conv in use"; fi
  else
    CC1D_STATUS="not installed: no nvcc on the host (install cuda-toolkit-13-0 or re-run with --causal-conv1d-in-docker); cuDNN fallback conv in use"
    log "WARNING: $CC1D_STATUS"
  fi
fi
log "causal-conv1d: $CC1D_STATUS"

# ---- 6. verify imports and write the locks ---------------------------------------------------------------------
if [[ $PRINT_ONLY -eq 0 ]]; then
  VERIFY=$(UVPY - <<'PY'
import importlib, json
out = {}
import torch
out["torch"] = torch.__version__; out["cuda"] = torch.version.cuda; out["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    out["gpu"] = torch.cuda.get_device_name(0); out["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    out["arch_list"] = torch.cuda.get_arch_list()
for name in ("triton", "fla", "causal_conv1d", "transformers", "peft", "trl", "accelerate", "datasets"):
    try:
        m = importlib.import_module(name); out[name] = getattr(m, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001
        out[name] = f"MISSING ({type(exc).__name__})"
print(json.dumps(out))
PY
)
  log "verify: $VERIFY"
  metric "\"phase\":\"verify\",\"result\":$VERIFY"
  python3 - "$VERIFY" "$FALLBACK" <<'PY' || record_error "verification failed (see verify line)"
import json, sys
v = json.loads(sys.argv[1]); fallback = sys.argv[2] == "1"
problems = []
if not v.get("cuda_available"): problems.append("torch.cuda.is_available() is False")
if "+cu" not in v.get("torch", ""): problems.append(f"torch build is not a CUDA build: {v.get('torch')}")
if not fallback and not v.get("torch", "").startswith("2.14.1"): problems.append(f"torch {v.get('torch')} != 2.14.1")
for k, want in (("transformers", "5.19.0"), ("peft", "0.21.2"), ("trl", "1.15.0"), ("accelerate", "1.15.0"), ("datasets", "5.1.0")):
    if v.get(k) != want: problems.append(f"{k} {v.get(k)} != {want}")
for k in ("triton", "fla"):
    if str(v.get(k, "")).startswith("MISSING"): problems.append(f"{k} missing")
if "sm_100" not in v.get("arch_list", []) and v.get("capability", "").startswith("10."): problems.append("sm_100 not in torch arch list on a Blackwell GPU")
print("problems:", problems or "none")
sys.exit(1 if problems else 0)
PY
  { echo "# requirements.lock -- training environment on the GPU VM, generated by scripts/gpu_setup_train_env.sh on $(ts) ($(hostname))";
    echo "# torch from $TORCH_INDEX_URL ($TORCH_SPEC); pins from $PINS_SRC; fla $FLA_VERSION; causal-conv1d: $CC1D_STATUS";
    echo "# recreate: bash scripts/gpu_setup_train_env.sh (the file below is 'uv pip freeze' of $VENV)";
    uv pip freeze --python "$VENV/bin/python"; } > requirements.lock
  { echo "# generated by scripts/gpu_setup_train_env.sh on $(ts) on $(hostname) (run $RUN_ID)";
    echo "python=$(UVPY --version | awk '{print $2}')"; echo "uv=$(uv --version | awk '{print $2}')";
    python3 - "$VERIFY" <<'PY'
import json, sys
v = json.loads(sys.argv[1])
for k in ("torch", "cuda", "gpu", "capability", "triton", "fla", "causal_conv1d", "transformers", "peft", "trl", "accelerate", "datasets"):
    print(f"{k}={v.get(k)}")
print("arch_list=" + ",".join(v.get("arch_list", [])))
PY
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)";
    echo "nvcc=$(nvcc --version 2>/dev/null | tail -1 | sed 's/^ *//' || echo none)";
    echo "torch_index_url=$TORCH_INDEX_URL"; echo "fallback_torch210=$FALLBACK"; echo "causal_conv1d_install=$CC1D_STATUS";
    echo "flash_attn=not installed (attention via sdpa; REPORT.md item 4)"; echo "kernels_hub=not installed"; echo "bitsandbytes=not installed (quantization: none)";
    echo "requirements_lock_sha256=$(sha256sum requirements.lock | cut -c1-16)"; echo "kernel=$(uname -r)";
  } > "$RUN_DIR/toolchain_versions.txt"
  log "wrote requirements.lock and $RUN_DIR/toolchain_versions.txt"
fi
status=PASS; [[ -s "$RUN_DIR/errors.log" ]] && status=FAIL
[[ $PRINT_ONLY -eq 1 ]] && status=PRINT_ONLY
finish "$status"
[[ "$status" != FAIL ]]
