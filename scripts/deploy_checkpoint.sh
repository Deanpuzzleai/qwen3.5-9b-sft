#!/usr/bin/env bash
# deploy_checkpoint.sh -- put one checkpoint directory (the Base snapshot or a merged SFT / stage-2 save) onto a GPU VM, verify
# it, serve it there with the locked image and flags (scripts/serve_vllm.sh, loopback only), and expose it on this host through an
# SSH tunnel (scripts/ssh_tunnel_endpoint.sh) on the docker0 address, so host-side runners and ASI-Bench's sandboxes reach it.
#
# Run from the repository root on the CPU VM:
#   scripts/deploy_checkpoint.sh --host gpu-e1 --src /mnt/apex-shared/checkpoints/<run>/merged --served-name qwen35-sft \
#                                --local-port 8000 [--bind 172.18.0.1] [--tunnel-name e1_sft] [--dest-name <dir name on the VM>]
# Inputs : --src (config.json, model*.safetensors, model.safetensors.index.json, tokenizer files, chat_template.jinja); SSH access
#          to --host (alias in ~/.ssh/config); this repository (synced to ~/qwen3.5-9b-sft on the VM without .env, data or logs).
# Outputs: the checkpoint at /mnt/models/<dest-name> on the VM with its sha256 list (<dest-name>.sha256); a serve run there
#          (logs/serve_<name>_<ts>/ on the VM); a tunnel run here (logs/tunnel_<tunnel-name>_<ts>/); prints API_BASE=... on success.
# Any server already running on the VM (qwen35-base / qwen35-sft / qwen35-stage2) is stopped first, and any open tunnel of the
# same name or on the same local address (an earlier deploy's tunnel under another name) is closed. Fails (exit 1) on a
# checksum mismatch, an unhealthy server or a tunnel that does not open; nothing is deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=""; SRC=""; NAME=""; LPORT=""; BIND=172.18.0.1; TUNNEL=""; DEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --served-name) NAME=$2; shift 2 ;;
    --local-port) LPORT=$2; shift 2 ;;
    --bind) BIND=$2; shift 2 ;;
    --tunnel-name) TUNNEL=$2; shift 2 ;;
    --dest-name) DEST=$2; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$HOST" && -n "$SRC" && -n "$NAME" && -n "$LPORT" ]] || { echo "--host, --src, --served-name and --local-port are required" >&2; exit 2; }
[[ -f "$SRC/config.json" && -f "$SRC/model.safetensors.index.json" ]] || { echo "$SRC is not a checkpoint directory" >&2; exit 2; }
DEST=${DEST:-$(basename "$(dirname "$SRC")")_$(basename "$SRC")}
TUNNEL=${TUNNEL:-${HOST#gpu-}_${NAME#qwen35-}}
say() { echo "[$(date -u +%FT%TZ)] $*"; }

say "checksums of $SRC"
SUMS=$(mktemp); (cd "$SRC" && sha256sum model*.safetensors model.safetensors.index.json config.json chat_template.jinja tokenizer.json 2>/dev/null) > "$SUMS"
say "copying to $HOST:/mnt/models/$DEST"
ssh "$HOST" "sudo mkdir -p /mnt/models && sudo chown \$(id -u):\$(id -g) /mnt/models"
rsync -a "$SRC/" "$HOST:/mnt/models/$DEST/"
scp -q "$SUMS" "$HOST:/mnt/models/$DEST.sha256"; rm -f "$SUMS"
ssh "$HOST" "cd /mnt/models/$DEST && sha256sum --quiet -c ../$DEST.sha256" || { echo "checksum mismatch on $HOST" >&2; exit 1; }
say "checksums verified on $HOST"

say "syncing the repository code to $HOST"
rsync -a --exclude .venv --exclude third_party --exclude logs --exclude data --exclude results --exclude checkpoints \
  --exclude curves --exclude .cache --exclude .env --exclude '*.env' --exclude __pycache__ ./ "$HOST:~/qwen3.5-9b-sft/"

for n in qwen35-base qwen35-sft qwen35-stage2; do
  ssh "$HOST" "cd ~/qwen3.5-9b-sft && docker ps --format '{{.Names}}' | grep -qx vllm-$n-8000 && scripts/serve_vllm.sh stop --served-name $n" || true
done
say "serving $NAME on $HOST"
ssh "$HOST" "cd ~/qwen3.5-9b-sft && scripts/serve_vllm.sh start --ckpt-dir /mnt/models/$DEST --served-name $NAME" | tail -3
ssh "$HOST" "curl -sf http://127.0.0.1:8000/health >/dev/null" || { echo "server on $HOST is not healthy" >&2; exit 1; }

scripts/ssh_tunnel_endpoint.sh close --name "$TUNNEL" >/dev/null 2>&1 || true
for f in logs/tunnel_*/tunnel.json; do   # open tunnels of other names on this address (closing a finalized run is a no-op)
  [[ -f "$f" ]] || continue
  if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (str(d.get('local_port')), d.get('bind', '127.0.0.1')) == (sys.argv[2], sys.argv[3]) else 1)" "$f" "$LPORT" "$BIND"; then
    scripts/ssh_tunnel_endpoint.sh close --run-dir "$(dirname "$f")" >/dev/null 2>&1 || true
  fi
done
scripts/ssh_tunnel_endpoint.sh open --host "$HOST" --name "$TUNNEL" --bind "$BIND" --local-port "$LPORT" | tail -1
