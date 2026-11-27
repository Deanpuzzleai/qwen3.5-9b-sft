#!/usr/bin/env bash
# Sanity-check a Harbor benchmark on this host WITHOUT a model: the reference solution (oracle) must resolve every task the
# lock does not list as drifted since release (harness_check_*.verifier_failures / reference_solution_failures), and the
# empty agent (nop) must resolve none. This is the benchmark authors' own release gate, re-run on our infrastructure with
# the same sandbox limits and agent timeout the evaluation runner applies.
#
#   scripts/validate_harness.sh swebench_pro|terminal_bench [N_CONCURRENT] [NOP_SAMPLE]
# Inputs : third_party checkouts and images from scripts/setup_eval_env.sh; configs/eval task-id lists and locks.
# Outputs: logs/harness_checks/<benchmark>_<UTC time>/{oracle/,nop/} raw harbor jobs and summary.json (counts, the expected
#          and unexpected oracle failures, PASS/FAIL from apex_eval.harbor_utils.validation_verdict); exit 0 on PASS, 1 on FAIL.
# Note   : the oracle run applies each task's reference solution inside the official harness; nothing from it is read or
#          reused beyond the pass/fail count (integrity rules). The verifier transcripts it produces are
#          git-ignored for the same reason.
set -euo pipefail
cd "$(dirname "$0")/.."
bench=${1:?usage: scripts/validate_harness.sh swebench_pro|terminal_bench [N_CONCURRENT] [NOP_SAMPLE]}
case "$bench" in -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
n_conc=${2:-16}; nop_sample=${3:-5}
[[ "$n_conc" =~ ^[0-9]+$ && "$nop_sample" =~ ^[0-9]+$ ]] || { echo "N_CONCURRENT and NOP_SAMPLE must be integers" >&2; exit 2; }
# Resource, timeout and retry settings mirror the evaluation runs: SWE-bench Pro uses each task's own task.toml limits
# (locked protocol); Terminal-Bench applies its lock's timeout and resource values (since 2026-10-12 none: each task's own
# task.toml timeout and resources, the leaderboard rule);
# infrastructure retries use the same exception list as src/apex_eval (harbor_utils.DEFAULT_RETRY_INCLUDE).
HARBOR=.venv/harbor/bin/harbor
PY=.venv/harbor/bin/python
[[ -x "$HARBOR" ]] || { echo "run scripts/setup_eval_env.sh first (harbor venv missing)" >&2; exit 1; }
LOCK_OVERRIDES=""
case "$bench" in
  swebench_pro)   DATASET=third_party/SWE-bench_Pro-os/v2/tasks; IDS=configs/eval/swebench_pro_hard51_ids.txt; LOCK=configs/eval/swebench_pro_v2_hard.yaml ;;
  terminal_bench) DATASET=third_party/terminal-bench/terminal-bench; IDS=configs/eval/terminal_bench_task_ids.txt; LOCK=configs/eval/terminal_bench_2.yaml
                  read -r cpus mem timeout < <("$PY" -c 'import yaml; l = yaml.safe_load(open("configs/eval/terminal_bench_2.yaml")); r = l["evaluator"]["resources_per_task"]; print(r["override_cpus"], r["override_memory_mb"], l["agent"]["agent_timeout_sec"])')
                  # null in the lock = no override (leaderboard setting since 2026-10-12): task.toml values apply and no job config is needed
                  if [[ "$cpus $mem $timeout" != "None None None" ]]; then
                    LOCK_OVERRIDES="override_cpus $cpus, override_memory_mb $mem, agent override_timeout_sec $timeout (lock values, via a harbor job config)"
                  fi ;;
  *) echo "unknown benchmark: $bench" >&2; exit 2 ;;
esac
mapfile -t RETRY < <("$PY" -c 'import sys; sys.path.insert(0, "src"); from apex_eval.harbor_utils import DEFAULT_RETRY_INCLUDE as r; print("\n".join(r))')
retry_args=(--max-retries 2); for exc in "${RETRY[@]}"; do retry_args+=(--retry-include "$exc"); done
# Harbor decides once per job whether offline sandboxes are supported via a 30 s docker probe; on a busy daemon it can time
# out and every trial is rejected. Wait until the probe passes before starting each job (same guard as src/apex_eval).
wait_probe() {
  for i in 1 2 3 4 5 6; do
    "$PY" -c 'from harbor.environments.docker.docker import DockerEnvironment as D; import sys; sys.exit(0 if D._egress_control_kernel_support() else 1)' && return 0
    echo "[validate] docker egress probe not passing (attempt $i/6); waiting 20s"; sleep 20
  done
  return 1
}
OUT=logs/harness_checks/${bench}_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
# `harbor run` cannot take an absolute agent timeout on the command line (only a multiplier), so any Terminal-Bench override the
# lock sets goes through a job config exactly as src/apex_eval/terminal_bench.py passes it (null values are left out); with no
# override, and for SWE-bench Pro, the task.toml values apply.
agent_args() {  # <oracle|nop> -> the harbor arguments selecting that agent (plus the lock's overrides for Terminal-Bench)
  if [[ -z "$LOCK_OVERRIDES" ]]; then echo "-a $1"; return; fi
  local cfg=$OUT/harbor_job_$1.yaml
  "$PY" - "$cfg" "$1" "$cpus" "$mem" "$timeout" <<'PY'
import sys, yaml
path, agent, cpus, mem, timeout = sys.argv[1:]
cfg = {"agents": [{"name": agent, **({"override_timeout_sec": float(timeout)} if timeout != "None" else {})}]}
env = {k: int(v) for k, v in (("override_cpus", cpus), ("override_memory_mb", mem)) if v != "None"}
if env:
    cfg["environment"] = env
yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
PY
  echo "-c $cfg"
}
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mapfile -t ids < <(/usr/bin/grep -v '^\s*$' "$IDS")
id_args=(); for i in "${ids[@]}"; do id_args+=(-i "$i"); done
echo "[validate] $bench: oracle on ${#ids[@]} tasks, n_concurrent=$n_conc -> $OUT/oracle"
wait_probe || echo "[validate] probe kept failing; starting anyway"
$HARBOR run -p "$DATASET" "${id_args[@]}" $(agent_args oracle) -n "$n_conc" "${retry_args[@]}" --quiet --yes --job-name oracle -o "$OUT" > "$OUT/oracle.log" 2>&1 || true
nop_args=(); for i in "${ids[@]:0:$nop_sample}"; do nop_args+=(-i "$i"); done
echo "[validate] $bench: nop on $nop_sample tasks -> $OUT/nop"
wait_probe || echo "[validate] probe kept failing; starting anyway"
$HARBOR run -p "$DATASET" "${nop_args[@]}" $(agent_args nop) -n "$n_conc" "${retry_args[@]}" --quiet --yes --job-name nop -o "$OUT" > "$OUT/nop.log" 2>&1 || true
"$PY" - "$OUT" "$bench" "${#ids[@]}" "$nop_sample" "$started" "$n_conc" "$($HARBOR --version 2>/dev/null)" "${LOCK_OVERRIDES:-task.toml defaults}" "$LOCK" <<'PY'
import datetime, json, sys
from pathlib import Path
import yaml
sys.path.insert(0, "src")
from apex_eval.harbor_utils import lock_known_failures, validation_verdict
out, bench, n_tasks, nop_sample, started, n_conc, harbor_version, resources, lock_path = (
    Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), sys.argv[7], sys.argv[8], sys.argv[9])
def tally(job):
    """task -> latest finished attempt (reward, exception) and the number of attempts seen."""
    res = {}
    for rj in (out / job).glob("*/result.json"):
        d = json.loads(rj.read_text()); name = (d.get("task_name") or rj.parent.name).split("/")[-1]
        vr = d.get("verifier_result") or {}; r = (vr.get("rewards") or {}).get("reward")
        exc = (d.get("exception_info") or {}).get("exception_type")
        prev = res.get(name); attempts = (prev["attempts"] if prev else 0) + 1
        if prev is None or (d.get("finished_at") or "") >= prev["finished_at"]:
            res[name] = {"reward": r, "exception": exc, "finished_at": d.get("finished_at") or "", "attempts": attempts}
        else:
            prev["attempts"] = attempts
    return res
oracle, nop = tally("oracle"), tally("nop")
unverifiable, stale_reference = lock_known_failures(yaml.safe_load(open(lock_path)))
verdict = validation_verdict(oracle, nop, n_tasks, nop_sample, {**unverifiable, **stale_reference})
summary = {"benchmark": bench, "harbor_version": harbor_version, "started": started, "n_concurrent": n_conc, "resource_overrides": resources,
           "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pass_rule": "every task has an oracle trial; the oracle resolves every task not listed as drifted in the lock; the empty agent resolves none",
           "expected_failures_source": lock_path, **verdict}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
o, n = summary["oracle"], summary["nop"]
print(f"[validate] {bench}: oracle {o['n_resolved']}/{n_tasks} resolved ({len(o['expected_failures_from_lock'])} expected, "
      f"{len(o['unexpected_failures'])} unexpected failures), nop {n['n_resolved']}/{nop_sample} resolved -> "
      f"{'PASS' if summary['pass'] else 'FAIL'}; details in {out}/summary.json")
sys.exit(0 if summary["pass"] else 1)
PY
