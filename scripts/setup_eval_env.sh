#!/usr/bin/env bash
# Reproduce the evaluation toolchain on the CPU VM.
#
# Run from the repository root:
#   bash scripts/setup_eval_env.sh            # everything, including the 51 SWE-bench Pro and 89 Terminal-Bench images (~250 GB on disk)
#   bash scripts/setup_eval_env.sh --skip-images
#   bash scripts/setup_eval_env.sh --apply-docker-pools   # host setting for >30 concurrent sandboxes (sudo, restarts docker when idle)
#
# Inputs : network access to GitHub, PyPI, ghcr.io, Docker Hub and Hugging Face; `uv`, `docker`, `npm` on PATH;
#          HF_TOKEN in .env for the gated HLE dataset (the only value read from .env); passwordless sudo only for
#          the docker-buildx apt install and for --apply-docker-pools.
# Outputs: .venv/harbor, .venv/asibench, .venv/hle   from configs/eval/requirements-{harbor,asibench,hle}.lock
#                                                   (written by the first install, then reproduced exactly)
#          third_party/SWE-bench_Pro-os   at the pinned commit, SHA256SUMS verified
#          third_party/ASI-Bench          at the pinned commit (+ the one-line patch from configs/eval/patches/)
#          third_party/hle                official HLE prediction/judge scripts at the pinned commit
#          third_party/hle_data           gated HLE dataset at the locked revision, checksum-verified (needs HF_TOKEN)
#          third_party/terminal-bench     Terminal-Bench 2.0 tasks (89), checksums verified against configs/eval/terminal_bench_manifest.tsv
#          third_party/asibench_instances/{seed42,seed31415}   instances at the pinned dataset revisions (.revision marker);
#          seed31415 is pulled for local debugging only and is not run in this experiment (configs/eval/asibench.yaml)
#          ~/.npm-global/bin/opencode     opencode CLI (pinned): asibench installs it into the task image itself, but its CLI
#                                         preflight also requires the binary on the host PATH
#          logs/setup_eval_env/<UTC time>.txt         what was actually installed on this host (versions, commits, docker)
#          (image pulls are compared against configs/eval/{swebench_pro,terminal_bench}_image_digests.tsv; a mismatch fails)
# Idempotent: re-running verifies pins and skips work already done. Never writes under configs/eval/ except the lock files
# on their first creation.
set -euo pipefail

# ---- pins (change only together with the lock files in configs/eval/) ----
HARBOR_VERSION=0.24.0
MINI_SWE_AGENT_VERSION=2.4.6
ASIBENCH_PATCH=configs/eval/patches/asibench-overlay-home-perms.patch   # one-line sandbox-permissions fix (see configs/eval/asibench.yaml)
OPENCODE_VERSION=1.17.15
SWEBENCH_PRO_REPO=https://github.com/scaleapi/SWE-bench_Pro-os.git
SWEBENCH_PRO_COMMIT=66f92766bba642462d4bbe5479e83f91f9211862   # release v2.0.0, 2026-09-22
ASIBENCH_REPO=https://github.com/apexin-ai/ASI-Bench.git
ASIBENCH_COMMIT=f13175a89dc9b4873f6306a3d31e46927c38f1a9       # 2026-10-09
ASIBENCH_SEED31415_REVISION=0fa14219cafdbab634d8b3cfbce238a8735a214f  # HF dataset Apexintelligence-AI/ASI-Bench-seed31415 (public references; local debugging only, not run)
ASIBENCH_SEED42_REVISION=4533397770a85ef79afe31e0361cbd9c42d5b50d     # HF dataset Apexintelligence-AI/ASI-Bench-seed42 (private references; official Portal scoring)
TERMINAL_BENCH_DATASET=terminal-bench@2.0                      # Harbor legacy registry -> laude-institute/terminal-bench-2
TERMINAL_BENCH_COMMIT=69671fbaac6d67a7ef0dfec016cc38a64ef7a77c  # the commit every registry entry points at (89 tasks)
HLE_REPO=https://github.com/centerforaisafety/hle.git
HLE_COMMIT=22ed3074b1e7b134bcbc09028d0ba320839b0655            # 2026-09-23; official prediction + judge scripts
HLE_DATASET_REVISION=5a81a4c7271a2a2a312b9a690f0c2fde837e4c29  # HF dataset cais/hle (gated: HF_TOKEN from an account that accepted the terms)
SWEBENCH_PRO_IMAGE_REPO=ghcr.io/scaleapi/swe-bench_pro-v2
PYTHON_VERSION=3.12
LOCKS=configs/eval   # requirements-<venv>.lock live here

SKIP_IMAGES=0; APPLY_DOCKER_POOLS=0
for arg in "$@"; do
  case "$arg" in
    --skip-images) SKIP_IMAGES=1 ;;
    --apply-docker-pools) APPLY_DOCKER_POOLS=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

[[ -d configs/eval && -d src/apex_eval ]] || { echo "run from the repository root" >&2; exit 1; }
ROOT=$(pwd)
mkdir -p third_party third_party/.setup_logs logs/setup_eval_env
log() { printf '[setup] %s\n' "$*"; }
fail() { echo "[setup] ERROR: $*" >&2; exit 1; }

# ---- prerequisites ----
command -v uv >/dev/null || fail "uv not found (https://docs.astral.sh/uv/)"
command -v npm >/dev/null || fail "npm not found (needed for the opencode CLI)"
docker info >/dev/null 2>&1 || fail "docker daemon not reachable (is the user in the docker group?)"
# The only secret this script needs: HF_TOKEN for the gated HLE dataset. Read just that value from .env.
HF_TOKEN=$( [[ -f .env ]] && { . ./.env >/dev/null 2>&1; printf '%s' "${HF_TOKEN:-}"; } || true )
export HF_TOKEN

# ---- python environments: exact lock files, written on the first install ----
make_venv() {  # name, lock file, then the top-level specs
  local name=$1 lock=$2; shift 2
  [[ -x ".venv/$name/bin/python" ]] || uv venv --quiet --python "$PYTHON_VERSION" ".venv/$name"
  if [[ -s "$lock" ]]; then
    uv pip install --quiet --python ".venv/$name/bin/python" -r "$lock" "$@"
  else
    uv pip install --quiet --python ".venv/$name/bin/python" "$@"
    uv pip freeze --python ".venv/$name/bin/python" | /usr/bin/grep -v -E '^(-e |asibench @ )' > "$lock"
    log "wrote $lock from the first install; commit it so later installs reproduce this environment exactly"
  fi
}
log "harbor venv"; make_venv harbor "$LOCKS/requirements-harbor.lock" "harbor==$HARBOR_VERSION" "mini-swe-agent==$MINI_SWE_AGENT_VERSION"

# ---- pinned checkouts ----
clone_pinned() {  # url, dir, commit
  local url=$1 dir=$2 commit=$3
  if [[ ! -d "$dir/.git" ]]; then git clone --quiet "$url" "$dir"; fi
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$commit"
  fi
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || fail "pin mismatch in $dir"
}
log "SWE-bench Pro checkout"; clone_pinned "$SWEBENCH_PRO_REPO" third_party/SWE-bench_Pro-os "$SWEBENCH_PRO_COMMIT"
log "verifying v2/SHA256SUMS"; (cd third_party/SWE-bench_Pro-os/v2 && sha256sum --quiet -c SHA256SUMS)
cmp -s third_party/SWE-bench_Pro-os/v2/hard51_ids.txt configs/eval/swebench_pro_hard51_ids.txt \
  || fail "hard51_ids.txt differs from the locked copy in configs/eval"
log "ASI-Bench checkout"; clone_pinned "$ASIBENCH_REPO" third_party/ASI-Bench "$ASIBENCH_COMMIT"
if ! /usr/bin/grep -q 'chmod -R a+rwX /home/agent' third_party/ASI-Bench/ai4sci_bench/runner/task_image.py; then
  log "applying $ASIBENCH_PATCH"; git -C third_party/ASI-Bench apply "$ROOT/$ASIBENCH_PATCH"
fi
log "asibench venv (editable install of the pinned, patched checkout, with the [full] extras)"
make_venv asibench "$LOCKS/requirements-asibench.lock" -e "third_party/ASI-Bench[full]"
log "HLE evaluation code checkout"; clone_pinned "$HLE_REPO" third_party/hle "$HLE_COMMIT"
log "hle venv (the official scripts' pinned requirements + pyarrow for the dataset)"
make_venv hle "$LOCKS/requirements-hle.lock" -r third_party/hle/requirements.txt "pyarrow==18.1.0" "pyyaml==6.0.3"   # pyarrow pinned: newer wheels need numpy>=2, the HLE requirements pin numpy 1.26.4

# ---- HLE dataset (gated): downloaded only when HF_TOKEN is available; verified against the locked revision ----
if [[ -n "$HF_TOKEN" ]]; then
  HD=third_party/hle_data; mkdir -p "$HD/data"
  for f in data/test-00000-of-00001.parquet README.md; do   # the parquet is the evaluation set; README.md is the dataset card (licence, canary)
    if [[ ! -s "$HD/$f" ]]; then
      log "downloading HLE $f"
      curl -sS -fL --retry 3 --retry-delay 5 --max-time 1800 -K - -o "$HD/$f.part" \
        "https://huggingface.co/datasets/cais/hle/resolve/$HLE_DATASET_REVISION/$f" <<< "header = \"Authorization: Bearer $HF_TOKEN\"" \
        || { rm -f "$HD/$f.part"; fail "HLE download failed for $f"; }
      mv "$HD/$f.part" "$HD/$f"
    fi
  done
  [[ "$(sha256sum "$HD/data/test-00000-of-00001.parquet" | cut -c1-64)" == "$(cut -d' ' -f1 configs/eval/hle_parquet.sha256)" ]] \
    || fail "HLE parquet checksum mismatch vs configs/eval/hle_parquet.sha256 (delete third_party/hle_data/data and re-run)"
else
  echo "[setup] WARNING: HF_TOKEN not in .env; skipping the gated HLE dataset download (needed for the HLE evaluation only)" >&2
fi

# ---- Terminal-Bench 2.0 task set (verified against the committed manifest) ----
TB=third_party/terminal-bench/terminal-bench
if [[ "$(ls "$TB" 2>/dev/null | wc -l)" -ne 89 ]]; then
  log "downloading $TERMINAL_BENCH_DATASET"; mkdir -p third_party/terminal-bench
  .venv/harbor/bin/harbor download "$TERMINAL_BENCH_DATASET" -o third_party/terminal-bench --export > third_party/.setup_logs/terminal_bench_download.log 2>&1 \
    || fail "harbor download failed, see third_party/.setup_logs/terminal_bench_download.log"
fi
log "verifying Terminal-Bench task checksums against configs/eval/terminal_bench_manifest.tsv"
while IFS=$'\t' read -r task img sha_toml sha_instr _rest; do
  [[ "$task" == \#* || -z "$task" ]] && continue
  [[ "$(sha256sum "$TB/$task/task.toml" | cut -c1-64)" == "$sha_toml" ]] || fail "task.toml checksum mismatch: $task"
  [[ "$(sha256sum "$TB/$task/instruction.md" | cut -c1-64)" == "$sha_instr" ]] || fail "instruction.md checksum mismatch: $task"
done < configs/eval/terminal_bench_manifest.tsv

# ---- docker daemon: enough network address pools for many concurrent sandboxes ----
# Every Harbor environment is a compose project with its own bridge network. Docker's built-in pools allow exactly 30 of
# them on a host (measured 2026-10-10); the 31st fails with "all predefined address pools have been fully subnetted".
# The pools below (172.18-172.31 and 192.168/16 carved into /24s = 3840 networks) avoid docker0's 172.17.0.0/16 and
# this VM's 10.x address. It is a host setting: with --apply-docker-pools the script merges the key into
# /etc/docker/daemon.json (keeping a backup) and restarts the daemon ONLY when no container is running.
if ! docker info --format '{{json .DefaultAddressPools}}' 2>/dev/null | /usr/bin/grep -q '172.18.0.0'; then
  if [[ $APPLY_DOCKER_POOLS -eq 1 ]]; then
    [[ -z "$(docker ps -q)" ]] || fail "containers are running; stop them before --apply-docker-pools (the restart kills them)"
    log "configuring docker default-address-pools (sudo) and restarting the daemon"
    tmp=$(mktemp)
    python3 - > "$tmp" <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
cfg = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
cfg["default-address-pools"] = [{"base": "172.18.0.0/15", "size": 24}, {"base": "172.20.0.0/14", "size": 24},
                                {"base": "172.24.0.0/13", "size": 24}, {"base": "192.168.0.0/16", "size": 24}]
print(json.dumps(cfg, indent=2))
PY
    sudo -n install -m 0644 -b "$tmp" /etc/docker/daemon.json; rm -f "$tmp"   # -b keeps daemon.json~ as a backup
    sudo -n systemctl restart docker
    for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    docker info --format '{{json .DefaultAddressPools}}' | /usr/bin/grep -q '172.18.0.0' || fail "docker pools not applied"
  else
    echo "[setup] WARNING: docker is using its built-in address pools (30 concurrent sandbox networks per host). To run more," >&2
    echo "        re-run with --apply-docker-pools while no container is running (needs sudo, restarts docker)." >&2
  fi
fi

# ---- docker buildx (harbor builds its egress-control sidecar with buildx) ----
if ! docker buildx version >/dev/null 2>&1; then
  log "installing docker-buildx via apt (sudo)"; sudo -n apt-get install -y -q docker-buildx
fi
docker buildx version >/dev/null

# ---- opencode CLI (user-level npm prefix, no sudo): required on the host PATH by asibench's preflight check ----
export NPM_CONFIG_PREFIX="$HOME/.npm-global"; mkdir -p "$NPM_CONFIG_PREFIX"
if [[ "$("$NPM_CONFIG_PREFIX/bin/opencode" --version 2>/dev/null || true)" != "$OPENCODE_VERSION" ]]; then
  log "installing opencode-ai@$OPENCODE_VERSION"; npm install -g --silent "opencode-ai@$OPENCODE_VERSION"
fi

# ---- ASI-Bench instances: seed42 (official, scored by the Portal) and seed31415 (public references; local debugging only) ----
for pair in "seed42:$ASIBENCH_SEED42_REVISION" "seed31415:$ASIBENCH_SEED31415_REVISION"; do
  repo=${pair%%:*}; rev=${pair#*:}; INST=third_party/asibench_instances/$repo
  if [[ "$(cat "$INST/.revision" 2>/dev/null)" != "$rev" || "$(ls "$INST" 2>/dev/null | wc -l)" -lt 60 ]]; then
    log "pulling $repo instances @ $rev"
    .venv/asibench/bin/asibench task pull --repo "$repo" --revision "$rev" --output-dir "$INST" > "third_party/.setup_logs/asibench_pull_$repo.log" 2>&1 \
      || fail "asibench task pull failed for $repo, see third_party/.setup_logs/asibench_pull_$repo.log"
    printf '%s\n' "$rev" > "$INST/.revision"
  fi
done

# ---- container images: pulled, present, and content-identical to the digests recorded at lock time ----
if [[ $SKIP_IMAGES -eq 0 ]]; then
  log "pulling the 51 SWE-bench Pro images (4 in parallel; present images are skipped by docker)"
  xargs -P 4 -I{} docker pull --platform linux/amd64 -q "$SWEBENCH_PRO_IMAGE_REPO:{}" \
    < configs/eval/swebench_pro_hard51_ids.txt > third_party/.setup_logs/image_pull.log 2>&1 || true   # presence is checked below
  log "pulling the 89 Terminal-Bench images (Docker Hub; anonymous pulls are rate-limited, 'docker login' first if this fails)"
  /usr/bin/grep -v '^#' configs/eval/terminal_bench_manifest.tsv | cut -f2 | xargs -P 4 -I{} docker pull --platform linux/amd64 -q {} \
    > third_party/.setup_logs/terminal_bench_image_pull.log 2>&1 || true
  problems=0
  while read -r id; do
    docker image inspect "$SWEBENCH_PRO_IMAGE_REPO:$id" >/dev/null 2>&1 || { echo "missing image: $id" >&2; problems=1; }
  done < configs/eval/swebench_pro_hard51_ids.txt
  while IFS=$'\t' read -r task img _rest; do
    [[ "$task" == \#* || -z "$task" ]] && continue
    docker image inspect "$img" >/dev/null 2>&1 || { echo "missing image: $img" >&2; problems=1; }
  done < configs/eval/terminal_bench_manifest.tsv
  # Tags can be moved upstream; the pulled content digest must be the one recorded at lock time (any of the image's repo digests).
  for f in configs/eval/swebench_pro_image_digests.tsv configs/eval/terminal_bench_image_digests.tsv; do
    while IFS=$'\t' read -r img want; do
      [[ "$img" == \#* || -z "$img" ]] && continue
      have=$(docker image inspect --format '{{join .RepoDigests "\n"}}' "$img" 2>/dev/null || true)
      /usr/bin/grep -qxF "$want" <<< "$have" || { echo "DIGEST MISMATCH (tag moved upstream?): $img locked=$want pulled=${have//$'\n'/,}" >&2; problems=1; }
    done < "$f"
  done
  [[ $problems -eq 0 ]] || fail "image set differs from the lock (see messages above); pull logs in third_party/.setup_logs/"
  log "all 140 images present with the locked digests"
fi

# ---- record what is actually installed on this host (a log, not a setting) ----
REC=logs/setup_eval_env/$(date -u +%Y%m%dT%H%M%SZ).txt
{
  echo "# generated by scripts/setup_eval_env.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  echo "venv_locks=$LOCKS/requirements-{harbor,asibench,hle}.lock"
  echo "harbor=$(.venv/harbor/bin/harbor --version 2>/dev/null | tr -d '\n')"
  echo "mini-swe-agent=$(.venv/harbor/bin/python -c 'import importlib.metadata as m;print(m.version("mini-swe-agent"))')"
  echo "litellm=$(.venv/harbor/bin/python -c 'import importlib.metadata as m;print(m.version("litellm"))')"
  echo "asibench=$(.venv/asibench/bin/asibench --version | awk '{print $NF}') (editable from third_party/ASI-Bench @ $ASIBENCH_COMMIT + $ASIBENCH_PATCH sha256 $(sha256sum "$ASIBENCH_PATCH" | cut -c1-16))"
  echo "opencode=$("$NPM_CONFIG_PREFIX/bin/opencode" --version)"
  echo "swebench_pro_commit=$(git -C third_party/SWE-bench_Pro-os rev-parse HEAD)"
  echo "hle_commit=$(git -C third_party/hle rev-parse HEAD)"
  echo "hle_dataset=$([[ -s third_party/hle_data/data/test-00000-of-00001.parquet ]] && echo "$HLE_DATASET_REVISION (parquet sha256 $(cut -c1-16 configs/eval/hle_parquet.sha256))" || echo "not downloaded")"
  echo "terminal_bench_dataset=$TERMINAL_BENCH_DATASET (git $TERMINAL_BENCH_COMMIT; manifest sha256 $(sha256sum configs/eval/terminal_bench_manifest.tsv | cut -c1-16))"
  echo "asibench_seed42_revision=$ASIBENCH_SEED42_REVISION"
  echo "asibench_seed31415_revision=$ASIBENCH_SEED31415_REVISION"
  echo "docker=$(docker version --format '{{.Server.Version}}') buildx=$(docker buildx version | awk '{print $2}') compose=$(docker compose version --short)"
  echo "docker_address_pools=$(docker info --format '{{json .DefaultAddressPools}}')"
  echo "python=$(.venv/harbor/bin/python --version | awk '{print $2}') node=$(node --version) kernel=$(uname -r)"
} > "$REC"
log "done; this host's toolchain record: $REC"
