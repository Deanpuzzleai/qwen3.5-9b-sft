#!/usr/bin/env bash
# ssh_tunnel_endpoint.sh -- open / close / check an SSH port forward from this CPU host to a remote GPU VM's
# vLLM port, so evaluation runners on this host can use http://<bind>:<local-port>/v1 as --api-base.
#
# Run from the repository root:
#   scripts/ssh_tunnel_endpoint.sh open  --host <gpu-vm> [--user ubuntu] [--remote-port 8000] [--local-port 8000]
#                                        [--name base] [--key ~/.ssh/id_ed25519] [--ssh-port 22] [--health-timeout 90] [--no-autossh]
#                                        [--bind 127.0.0.1]   # local listen address; use the docker0 gateway address (e.g. 172.18.0.1)
#                                                             # when agents inside Docker sandboxes must reach the endpoint (ASI-Bench's
#                                                             # opencode runs in a bridge-network container, where 127.0.0.1 is the container)
#   scripts/ssh_tunnel_endpoint.sh close --name <name> | --run-dir logs/tunnel_<name>_<ts>
#   scripts/ssh_tunnel_endpoint.sh status [--name <name>]
#   scripts/ssh_tunnel_endpoint.sh check --local-port 8000          # health probe of whatever listens locally
# Inputs : SSH access to the GPU VM (key-based, BatchMode: no password prompts); the VM's vLLM listening on
#          127.0.0.1:<remote-port> (scripts/serve_vllm.sh start publishes the container port on the VM host).
#          No secrets are read; SSH keys stay in ~/.ssh. .env is sourced like scripts/evaluate.sh for parity only.
# Outputs: logs/tunnel_<name>_<ts>/{metadata.json (host, ports, pid, backend, start/end), eval.log, metrics.jsonl
#          (health probes: ts, http status, latency), errors.log, resources.csv (tunnel duration; no GPU),
#          tunnel.json (state used by close/status), tunnel.pid, models.json (/v1/models at open)}.
# Mechanism: autossh -M 0 (restarts the ssh on failure) when installed, else plain `ssh -N` kept alive with
#   ServerAliveInterval; the health loop polls http://<bind>:<local-port>/health until 200 or --health-timeout, then
#   saves /v1/models. open refuses (exit 1, no run directory) while a tunnel of the same name is alive or anything
#   listens on <bind>:<local-port>; close leaves an already finalized run untouched.
# GPU-only: nothing here needs a GPU; the remote endpoint does (serve_vllm.sh). Tested on apex-cpu against a live
#   gpu-e2 server (2026-10-13): open/close under a new name, a second close, both refusals; plus the argument handling,
#   the failure path (unreachable host) and status/close bookkeeping.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }
sub=${1:-}; [[ -n "$sub" ]] || { usage; exit 2; }; shift
case "$sub" in -h|--help|help) usage; exit 0 ;; esac
HOST=""; USER_=""; REMOTE_PORT=8000; LOCAL_PORT=8000; NAME=""; KEY=""; SSH_PORT=22; HEALTH_TIMEOUT=90; USE_AUTOSSH=1; RUN_DIR=""; BIND=127.0.0.1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --user) USER_=$2; shift 2 ;;
    --remote-port) REMOTE_PORT=$2; shift 2 ;;
    --local-port) LOCAL_PORT=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --key) KEY=$2; shift 2 ;;
    --ssh-port) SSH_PORT=$2; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT=$2; shift 2 ;;
    --no-autossh) USE_AUTOSSH=0; shift ;;
    --bind) BIND=$2; shift 2 ;;
    --run-dir) RUN_DIR=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_DIR/eval.log"; }
record_error() { echo "[$(ts)] $*" >> "$RUN_DIR/errors.log"; log "ERROR: $*"; }
metric() { printf '{"ts":"%s",%s}\n' "$(ts)" "$1" >> "$RUN_DIR/metrics.jsonl"; }
probe() {  # local port [bind address] -> prints "<http code> <seconds>" (curl prints 000 itself on connection failure)
  local out; out=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 5 "http://${2:-127.0.0.1}:$1/health" 2>/dev/null) || true
  echo "${out:-000 0}"
}
alive() {  # pid -> true while it is a running ssh/autossh (a stale pid file may name a reused pid; close kills it)
  [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null && [[ "$(cat "/proc/$1/comm" 2>/dev/null)" =~ ^(auto)?ssh$ ]]
}
name_run_dirs() {  # name -> its run dirs logs/tunnel_<name>_<YYYYMMDD>_<HHMMSS>, oldest first (exactly this name, so
  local d rest     # `e1_sft` never matches `e1_sft_ep2`)
  for d in logs/tunnel_"$1"_*/; do
    rest=${d#logs/tunnel_"$1"_}
    if [[ "$rest" =~ ^[0-9]{8}_[0-9]{6}/$ ]]; then echo "${d%/}"; fi   # an if, so a last non-match does not fail under set -e
  done | sort
}
latest_run_dir() {  # name -> its newest run dir
  name_run_dirs "$1" | tail -1
}
write_resources() {  # run dir, status
  local start end dur
  start=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['start_epoch'])" "$1/tunnel.json")
  end=$(date +%s); dur=$(( end - start ))
  printf 'metric,value,unit,note\nduration_sec,%s,s,tunnel open time\ngpu_count,0,count,no GPU on this host\ngpu_hours,0,h,tunnel only\ncost_usd,%s,USD,cpu_hours x %s (CPU_USD_PER_HOUR)\nstatus,%s,,final state\n' \
    "$dur" "$(python3 -c "print(round($dur/3600*${CPU_USD_PER_HOUR:-0},4))")" "${CPU_USD_PER_HOUR:-0}" "$2" > "$1/resources.csv"
}
finalize() {  # run dir, status
  write_resources "$1" "$2"
  python3 - "$1/metadata.json" "$2" "$(ts)" <<'PY'
import json, sys
p, status, end = sys.argv[1:4]
m = json.load(open(p)); m.update(status=status, end_time=end); json.dump(m, open(p, "w"), indent=2)
PY
  [[ -s "$1/errors.log" ]] || echo none > "$1/errors.log"
}

case "$sub" in
open)
  [[ -n "$HOST" ]] || { echo "--host is required" >&2; exit 2; }
  NAME=${NAME:-$(echo "$HOST" | tr -c 'A-Za-z0-9\n' '_')}
  # Refused before any run directory exists: a live tunnel of this name (any of its run directories), or anything
  # listening on the address. A second forward would exit at once (ExitOnForwardFailure) while the health probe
  # answered through the first one, so the open would look healthy.
  for d in $(name_run_dirs "$NAME"); do
    p=$(cat "$d/tunnel.pid" 2>/dev/null || true)
    if alive "$p"; then echo "a tunnel named $NAME is already open ($d, pid $p); close it first" >&2; exit 1; fi
  done
  LISTENER=$(ss -ltnpH "( sport = :${LOCAL_PORT} )" 2>/dev/null | grep -F -e "${BIND}:${LOCAL_PORT} " -e "0.0.0.0:${LOCAL_PORT} " -e "*:${LOCAL_PORT} " -e "[::]:${LOCAL_PORT} " || true)
  if [[ -n "$LISTENER" ]]; then echo "${BIND}:${LOCAL_PORT} is already in use: $(echo "$LISTENER" | tr -s ' ' | head -1)" >&2; exit 1; fi
  RUN_DIR="logs/tunnel_${NAME}_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_DIR"; : > "$RUN_DIR/errors.log"
  target="${USER_:+$USER_@}$HOST"
  # an explicit bind address needs GatewayPorts=yes on the client; the forward still listens only on that one address
  SSH_OPTS=(-N -L "${BIND}:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" -p "$SSH_PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15
            -o ServerAliveCountMax=3 -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
  [[ "$BIND" != 127.0.0.1 ]] && SSH_OPTS+=(-o GatewayPorts=yes)
  [[ -n "$KEY" ]] && SSH_OPTS+=(-i "$KEY")
  backend=ssh
  if [[ $USE_AUTOSSH -eq 1 ]] && command -v autossh >/dev/null 2>&1; then
    backend=autossh
    AUTOSSH_GATETIME=0 AUTOSSH_PIDFILE="$PWD/$RUN_DIR/tunnel.pid" AUTOSSH_LOGFILE="$PWD/$RUN_DIR/ssh.log" \
      autossh -M 0 -f "${SSH_OPTS[@]}" "$target"
    sleep 1
  else
    setsid nohup ssh "${SSH_OPTS[@]}" "$target" > "$RUN_DIR/ssh.log" 2>&1 &
    echo $! > "$RUN_DIR/tunnel.pid"
  fi
  PID=$(cat "$RUN_DIR/tunnel.pid" 2>/dev/null || echo "")
  python3 - "$RUN_DIR" "$NAME" "$HOST" "$USER_" "$REMOTE_PORT" "$LOCAL_PORT" "$PID" "$backend" "$SSH_PORT" "$BIND" <<'PY'
import json, os, socket, subprocess, sys, time
d, name, host, user, rport, lport, pid, backend, sshport, bind = sys.argv[1:11]
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
state = {"name": name, "host": host, "user": user or None, "remote_port": int(rport), "local_port": int(lport), "bind": bind, "pid": int(pid or 0),
         "backend": backend, "ssh_port": int(sshport), "start_epoch": int(time.time()), "run_dir": d}
json.dump(state, open(os.path.join(d, "tunnel.json"), "w"), indent=2)
meta = {"run_id": os.path.basename(d), "kind": "ssh_tunnel", "status": "RUNNING", "command": " ".join(sys.argv), **state,
        "api_base": f"http://{bind}:{lport}/v1", "code_commit": git, "host_local": socket.gethostname(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "end_time": None, "seed": None, "data_version": None}
json.dump(meta, open(os.path.join(d, "metadata.json"), "w"), indent=2)
PY
  log "tunnel $NAME: $backend pid=$PID $BIND:$LOCAL_PORT -> $target:127.0.0.1:$REMOTE_PORT (run dir $RUN_DIR)"
  t0=$(date +%s)
  while true; do
    read -r code secs <<< "$(probe "$LOCAL_PORT" "$BIND")"
    metric "\"event\":\"health\",\"http\":$code,\"seconds\":$secs"
    if [[ "$code" == "200" ]]; then
      log "healthy after $(( $(date +%s) - t0 ))s"; curl -s --max-time 10 "http://$BIND:$LOCAL_PORT/v1/models" > "$RUN_DIR/models.json" || true
      log "served models: $(python3 -c "import json,sys;print([m['id'] for m in json.load(open(sys.argv[1])).get('data',[])])" "$RUN_DIR/models.json" 2>/dev/null || echo '?')"
      write_resources "$RUN_DIR" RUNNING; echo "API_BASE=http://$BIND:$LOCAL_PORT/v1"; exit 0
    fi
    if ! alive "$PID"; then
      record_error "tunnel process exited: $(tail -3 "$RUN_DIR/ssh.log" 2>/dev/null | tr '\n' ' ')"; finalize "$RUN_DIR" FAIL; exit 1
    fi
    if (( $(date +%s) - t0 >= HEALTH_TIMEOUT )); then
      record_error "no 200 from /health within ${HEALTH_TIMEOUT}s (is vLLM up on the VM? scripts/serve_vllm.sh status there)"
      kill "$PID" 2>/dev/null || true; finalize "$RUN_DIR" FAIL; exit 1
    fi
    sleep 3
  done
  ;;
close)
  [[ -n "$RUN_DIR" ]] || { [[ -n "$NAME" ]] || { echo "close needs --name or --run-dir" >&2; exit 2; }; RUN_DIR=$(latest_run_dir "$NAME"); }
  [[ -n "$RUN_DIR" && -f "$RUN_DIR/tunnel.json" ]] || { echo "no tunnel state found for ${NAME:-$RUN_DIR}" >&2; exit 1; }
  STATUS=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" "$RUN_DIR/metadata.json" 2>/dev/null || echo "")
  if [[ -n "$STATUS" && "$STATUS" != RUNNING ]]; then   # finished runs keep their recorded end time and duration
    echo "[$(ts)] $RUN_DIR is already finalized ($STATUS); nothing to close"; exit 0
  fi
  PID=$(cat "$RUN_DIR/tunnel.pid" 2>/dev/null || echo "")
  if alive "$PID"; then kill "$PID" && log "closed tunnel pid $PID"; sleep 1; alive "$PID" && kill -9 "$PID" 2>/dev/null || true
  else log "tunnel pid ${PID:-?} already gone"; fi
  finalize "$RUN_DIR" CLOSED; log "finalized $RUN_DIR/resources.csv"
  ;;
status)
  found=0
  for f in logs/tunnel_*/tunnel.json; do
    [[ -f "$f" ]] || continue
    d=$(dirname "$f"); n=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['name'],d['pid'],d['local_port'],d['host'],d['remote_port'],d.get('bind','127.0.0.1'))" "$f")
    read -r n_name n_pid n_lport n_host n_rport n_bind <<< "$n"
    [[ -n "$NAME" && "$NAME" != "$n_name" ]] && continue
    if alive "$n_pid"; then found=1; read -r code secs <<< "$(probe "$n_lport" "$n_bind")"; echo "OPEN  $n_name pid=$n_pid $n_bind:$n_lport -> $n_host:$n_rport health=$code ($secs s) $d"; fi
  done
  [[ $found -eq 1 ]] || echo "no open tunnel${NAME:+ named $NAME}"
  ;;
check)
  read -r code secs <<< "$(probe "$LOCAL_PORT" "$BIND")"
  echo "http://$BIND:$LOCAL_PORT/health -> $code (${secs}s)"
  [[ "$code" == "200" ]] && curl -s --max-time 10 "http://127.0.0.1:$LOCAL_PORT/v1/models"; echo
  [[ "$code" == "200" ]]
  ;;
*) echo "unknown subcommand: $sub" >&2; usage; exit 2 ;;
esac
