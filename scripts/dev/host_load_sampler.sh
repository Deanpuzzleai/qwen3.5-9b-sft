#!/usr/bin/env bash
# host_load_sampler.sh -- record the evaluation host's load every INTERVAL seconds, so each benchmark run's timeout rate can
# be read against the CPU and I/O pressure it ran under (the evaluation sandboxes share apex-cpu with data building).
#
# Run from the repository root (normally inside tmux, for the whole evaluation window):
#   scripts/dev/host_load_sampler.sh [--interval 30] [--out logs/host_load_<UTC date>]
# Inputs : /proc/loadavg, /proc/stat (CPU busy / iowait shares between samples), `docker ps` (running container count).
# Outputs: <out>/host_load.csv with columns
#            ts_utc, load1, load5, load15, cpu_busy_pct, cpu_iowait_pct, running_procs, blocked_procs, containers
#          and <out>/metadata.json (host, interval, start time, command). Appends to an existing CSV; stop with Ctrl-C or
#          by killing the tmux session. Nothing else is read or written.
set -euo pipefail
cd "$(dirname "$0")/../.."
INTERVAL=30; OUT="logs/host_load_$(date -u +%Y%m%d)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"
CSV="$OUT/host_load.csv"
[[ -f "$CSV" ]] || echo "ts_utc,load1,load5,load15,cpu_busy_pct,cpu_iowait_pct,running_procs,blocked_procs,containers" > "$CSV"
python3 - "$OUT/metadata.json" "$INTERVAL" "$0 $*" <<'PY'
import json, socket, sys, time
path, interval, cmd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
meta = {"kind": "host_load_sampler", "host": socket.gethostname(), "interval_sec": interval, "command": cmd,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nproc": __import__("os").cpu_count()}
json.dump(meta, open(path, "w"), indent=2)
PY
cpu_counters() { awk '/^cpu /{idle=$5; iow=$6; tot=0; for (i=2; i<=NF; i++) tot+=$i; print tot, idle, iow}' /proc/stat; }
read -r t0 i0 w0 <<< "$(cpu_counters)"
while true; do
  sleep "$INTERVAL"
  read -r t1 i1 w1 <<< "$(cpu_counters)"
  dt=$(( t1 - t0 )); (( dt > 0 )) || dt=1
  busy=$(awk -v d="$dt" -v i="$(( i1 - i0 ))" -v w="$(( w1 - w0 ))" 'BEGIN{printf "%.1f", 100*(d-i-w)/d}')
  iow=$(awk -v d="$dt" -v w="$(( w1 - w0 ))" 'BEGIN{printf "%.1f", 100*w/d}')
  read -r l1 l5 l15 _ < /proc/loadavg
  run=$(awk '/^procs_running/{print $2}' /proc/stat); blk=$(awk '/^procs_blocked/{print $2}' /proc/stat)
  ctr=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date -u +%FT%TZ),$l1,$l5,$l15,$busy,$iow,$run,$blk,$ctr" >> "$CSV"
  t0=$t1; i0=$i1; w0=$w1
done
