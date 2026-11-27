#!/usr/bin/env python3
"""Dashboard and health probe for a training run of apex_train.sft / apex_train.stage2 (the training monitor's files).

Reads (read-only, nothing is written):
  logs/<run_id>/status.json     heartbeat of src/apex_train/monitor.py (rewritten every optimizer step and every
                                monitor.watchdog_every_s seconds while the process lives)
  logs/<run_id>/alerts.jsonl    the alerts (code, severity, step, value, threshold, message, action)
  logs/<run_id>/metadata.json   monitor_config (watchdog period, for the staleness limit) and the run status for runs that
                                have no status.json (written before the monitor existed)
Works on a live run (on the GPU VM, or on apex-cpu after rsync of logs/<run_id>/) and on a finished one. Standard library only.

Usage (from the repository root):
  python scripts/monitor_training.py --run logs/<run_id>            # dashboard once
  python scripts/monitor_training.py --run <run_id> --watch 60      # refresh every 60 s until the run ends (Ctrl-C stops)
  python scripts/monitor_training.py --run logs/<run_id> --json     # machine-readable status + derived fields + exit code
  python scripts/monitor_training.py --latest --oneline             # newest run under logs/ that has a status.json, one line
  ssh gpu-t 'cd ~/qwen3.5-9b-sft && .venv/train/bin/python scripts/monitor_training.py --run logs/<run_id>'

Exit code (for until-loops): 0 OK, 1 WARN, 2 CRITICAL (incl. a run that stopped, failed or was aborted), 3 stale (the
status has not been updated for --stale-after seconds while the run claims to be alive: the process is gone or frozen) or
no status at all. Health is the worst alert severity of the last monitor.health_window_s seconds (CRITICAL while stalled).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ("finished", "stopped", "failed", "aborted")
EXIT = {"OK": 0, "WARN": 1, "CRITICAL": 2, "STALE": 3}


def resolve_run(arg: str | None, latest: bool, logs_root: Path) -> Path:
    if latest:
        cands = [p.parent for p in logs_root.glob("*/status.json")]
        if not cands:
            raise SystemExit(f"no run with a status.json under {logs_root}")
        return max(cands, key=lambda p: (p / "status.json").stat().st_mtime)
    if not arg:
        raise SystemExit("pass --run logs/<run_id> (or a run_id) or --latest")
    path = Path(arg)
    for cand in (path, logs_root / arg, REPO_ROOT / arg):
        if (cand / "status.json").is_file() or (cand / "metadata.json").is_file():
            return cand
    raise SystemExit(f"{arg}: no status.json or metadata.json there (looked in {path}, {logs_root / arg})")


def read_json(path: Path) -> dict[str, Any] | None:
    for _ in range(3):   # status.json is replaced atomically; a retry covers a file system without atomic rename
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
    return None


def read_alerts(path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0 or not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def evaluate(run_dir: Path, stale_after: float | None, now: float | None = None) -> dict[str, Any]:
    """{status, meta, health, verdict (OK/WARN/CRITICAL/STALE), exit_code, age_s, reason}."""
    now = now or time.time()
    status = read_json(run_dir / "status.json")
    meta = read_json(run_dir / "metadata.json") or {}
    watchdog = float((meta.get("monitor_config") or {}).get("watchdog_every_s") or 30.0)
    limit = stale_after if stale_after is not None else max(300.0, 10 * watchdog)
    if status is None:
        st = meta.get("status")
        verdict = "OK" if st == "finished" else "CRITICAL" if st in ("crashed", "aborted", "stopped") else "STALE"
        return {"status": None, "meta": meta, "health": None, "verdict": verdict, "exit_code": EXIT[verdict], "age_s": None,
                "stale_after_s": limit, "reason": f"no status.json; metadata status {st!r}"}
    age = now - float(status.get("updated_unix") or 0)
    phase, health = status.get("phase"), status.get("health") or "OK"
    if phase not in TERMINAL and age > limit:
        verdict, reason = "STALE", f"status not updated for {age:.0f} s (> {limit:.0f} s) in phase {phase}: process gone or frozen"
    elif phase in ("stopped", "failed", "aborted"):
        verdict, reason = "CRITICAL", f"run {phase}: {status.get('stop_reason')}"
    else:
        verdict, reason = health if health in EXIT else "OK", f"health {health}, phase {phase}"
    return {"status": status, "meta": meta, "health": health, "verdict": verdict, "exit_code": EXIT[verdict], "age_s": round(age, 1),
            "stale_after_s": limit, "reason": reason}


# --- formatting --------------------------------------------------------------------------------------------------------
def f(x: Any, fmt: str = ".4f", none: str = "-") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return none if x is None else str(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return format(x, fmt)
    return str(x)


def hms(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return "-"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def bar(pct: Any, width: int = 30) -> str:
    p = max(0.0, min(100.0, float(pct))) if isinstance(pct, (int, float)) else 0.0
    n = round(width * p / 100)
    return "[" + "#" * n + "-" * (width - n) + "]"


def slices_text(d: dict[str, Any] | None) -> str:
    return "  ".join(f"{k} {f(v)}" for k, v in sorted((d or {}).items())) or "-"


def dashboard(ev: dict[str, Any], alerts: list[dict[str, Any]]) -> str:
    s, meta = ev["status"], ev["meta"]
    lines = []
    if s is None:
        lines.append(f"run {meta.get('run_id', '?')}: {ev['reason']}")
        lines.append(f"verdict {ev['verdict']} (exit {ev['exit_code']})")
        return "\n".join(lines)
    tag = " ".join(x for x in (s.get("stage"), s.get("method"), s.get("arm")) if x)
    lines.append(f"run {s['run_id']}  [{tag}]  phase {s.get('phase')}  host {s.get('host')} pid {s.get('pid')}  device {s.get('device')}")
    lines.append(f"verdict {ev['verdict']} (exit {ev['exit_code']}): {ev['reason']}; status updated {f(ev['age_s'], '.0f')} s ago")
    total = s.get("total_steps")
    lines.append(f"progress  {bar(s.get('percent'))} {s.get('step')}/{total if total else '?'} ({f(s.get('percent'), '.1f')}%)  "
                 f"elapsed {hms(s.get('elapsed_s'))}  ETA {hms(s.get('eta_s'))}" + (f" (~{s['eta_utc']})" if s.get("eta_utc") else "")
                 + f"  step wall {f(s.get('step_wall_s_median'), '.1f')} s (median)  since progress {f(s.get('seconds_since_progress'), '.0f')} s")
    lo = s.get("loss") or {}
    lines.append(f"loss      last {f(lo.get('last'))}  EMA {f(lo.get('ema'))}  min {f(lo.get('min'))} @ {lo.get('min_step', '-')}  "
                 f"raw (unfiltered) {f(lo.get('observed'))}  normalisation {s.get('loss_normalisation') or '-'}")
    lines.append(f"train     per slice (token-weighted, last 10 steps): {slices_text(s.get('slices_train'))}")
    e = s.get("eval")
    if e:
        best_rows, best_tok = ((s.get("best_eval") or {}).get(k) or {} for k in ("overall", "overall_tokens"))
        tokens = s.get("loss_normalisation") == "tokens" and e.get("slices_tokens")
        lines.append(f"eval      @ step {e.get('step')}: eval_loss {f(e.get('loss'))} row-weighted (best {f(best_rows.get('loss'))} @ "
                     f"{best_rows.get('step', '-')}), token-weighted {f(e.get('loss_tokens'))} (best {f(best_tok.get('loss'))} @ "
                     f"{best_tok.get('step', '-')}), {e.get('rows', '-')} rows in {f(e.get('wall_s'), '.0f')} s")
        lines.append(f"          per slice ({'token' if tokens else 'row'}-weighted): "
                     f"{slices_text(e.get('slices_tokens') if tokens else e.get('slices'))}")
    else:
        lines.append("eval      none yet")
    lr, g = s.get("lr") or {}, s.get("grad_norm") or {}
    clip = g.get("clip_rate")
    lines.append(f"optim     lr {f(lr.get('last'), '.3e')} (planned {f(lr.get('planned'), '.3e')})  grad norm {f(g.get('last'), '.3g')} "
                 f"(median {f(g.get('median'), '.3g')}, clipped {f(clip * 100 if isinstance(clip, (int, float)) else None, '.0f')}% "
                 f"at {f(g.get('max_grad_norm'), 'g')})")
    tok, ts = s.get("tokens") or {}, s.get("token_stats") or {}
    lines.append(f"tokens    {f(tok.get('seq_per_s'), '.0f')} tok/s (trained {f(tok.get('trained_per_s'), '.0f')} tok/s)  total "
                 f"{f(tok.get('seq_total'), ',.0f')} seq / {f(tok.get('trained_total'), ',.0f')} trained  entropy {f(ts.get('entropy'), '.3f')}  "
                 f"token acc {f(ts.get('accuracy'), '.3f')}")
    m = s.get("memory") or {}
    tot = m.get("total_gb")
    peak = m.get("peak_reserved_gb")
    share = f" ({100 * peak / tot:.0f}% of {tot:.0f} GB)" if isinstance(peak, (int, float)) and isinstance(tot, (int, float)) and tot else ""
    lines.append(f"memory    GPU alloc {f(m.get('alloc_gb'), '.1f')} GB  reserved {f(m.get('reserved_gb'), '.1f')} GB  peak alloc "
                 f"{f(m.get('peak_alloc_gb'), '.1f')} / reserved {f(peak, '.1f')} GB{share}  host RSS {f(m.get('rss_gb'), '.1f')} GB")
    gpu = s.get("gpu")
    if gpu:
        lines.append(f"gpu       util {f(gpu.get('gpu_util_pct'), '.0f')}%  power {f(gpu.get('power_w'), '.0f')}/{f(gpu.get('power_limit_w'), '.0f')} W  "
                     f"temp {f(gpu.get('temp_c'), '.0f')} C  SM {f(gpu.get('sm_clock_mhz'), '.0f')} MHz  mem used {f(gpu.get('mem_used_gb'), '.1f')} GB  "
                     f"reasons {gpu.get('throttle_reasons') or '-'}  (sample at {gpu.get('utc', '-')})")
    else:
        lines.append(f"gpu       no telemetry ({s.get('gpu_telemetry')})")
    lora = s.get("lora")
    if lora:
        fam = "  ".join(f"{k} {f(v, '.3g')}" for k, v in sorted((lora.get("update_ratio_by_family") or {}).items()))
        rel = lora.get("delta_w_rel")
        lines.append(f"lora      @ step {lora.get('step')}: update ratio {f(lora.get('update_ratio'), '.3g')} ({fam})  |adapter| "
                     f"{f(lora.get('norm'), '.4g')}  |dW|/|W0| {f(rel * 100 if isinstance(rel, (int, float)) else None, '.2f')}%  "
                     f"finite {lora.get('finite')}")
    lines.append(f"ckpts     {' '.join(map(str, s.get('checkpoints') or [])) or '-'}   disk free {f(s.get('disk_free_gb'), '.0f')} GB   "
                 f"monitor {f((s.get('monitor_s_mean') or 0) * 1000, '.1f')} ms/step")
    counts = (s.get("alerts") or {}).get("counts") or {}
    lines.append(f"alerts    CRITICAL {counts.get('CRITICAL', 0)}  WARN {counts.get('WARN', 0)}  INFO {counts.get('INFO', 0)}  "
                 f"(suppressed repeats {(s.get('alerts') or {}).get('suppressed', 0)})")
    for a in alerts:
        lines.append(f"  {a.get('ts', '')} {a.get('severity', ''):8s} {a.get('code', '')} @ step {a.get('step')}: {a.get('message', '')}")
        if a.get("severity") != "INFO" and a.get("action"):
            lines.append(f"      action: {a['action']}")
    if s.get("stop_reason"):
        lines.append(f"STOPPED: {s['stop_reason']} (adapter weights finite at stop: {s.get('weights_finite_at_stop')})")
    return "\n".join(lines)


def oneline(ev: dict[str, Any]) -> str:
    s = ev["status"]
    if s is None:
        return f"{ev['verdict']} {ev['reason']}"
    lo = s.get("loss") or {}
    return (f"{ev['verdict']} {s['run_id']} {s.get('phase')} step {s.get('step')}/{s.get('total_steps') or '?'} loss {f(lo.get('last'))} "
            f"ema {f(lo.get('ema'))} tok/s {f((s.get('tokens') or {}).get('seq_per_s'), '.0f')} ETA {hms(s.get('eta_s'))} "
            f"alerts W{((s.get('alerts') or {}).get('counts') or {}).get('WARN', 0)}/C{((s.get('alerts') or {}).get('counts') or {}).get('CRITICAL', 0)} "
            f"age {f(ev['age_s'], '.0f')}s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run directory logs/<run_id> (or the run_id)")
    p.add_argument("--latest", action="store_true", help="the newest run under --logs-root that has a status.json")
    p.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs", help="where run directories live (default logs/)")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="refresh every SECONDS until the run ends or Ctrl-C")
    p.add_argument("--json", action="store_true", help="print status.json + derived fields (verdict, exit_code, age_s) as JSON")
    p.add_argument("--oneline", action="store_true", help="one summary line instead of the dashboard")
    p.add_argument("--alerts", type=int, default=5, help="last N alerts to show (default 5)")
    p.add_argument("--stale-after", type=float, default=None, metavar="SECONDS",
                   help="status age that counts as stale (default max(300, 10 x monitor.watchdog_every_s))")
    a = p.parse_args(argv)
    run_dir = resolve_run(a.run, a.latest, a.logs_root)
    while True:
        ev = evaluate(run_dir, a.stale_after)
        if a.json:
            print(json.dumps({"run_dir": str(run_dir), "verdict": ev["verdict"], "exit_code": ev["exit_code"], "age_s": ev["age_s"],
                              "stale_after_s": ev["stale_after_s"], "reason": ev["reason"], "status": ev["status"]}, indent=1))
        elif a.oneline:
            print(oneline(ev), flush=True)
        else:
            text = dashboard(ev, read_alerts(run_dir / "alerts.jsonl", a.alerts))
            if a.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            width = shutil.get_terminal_size((160, 40)).columns
            print(text if not a.watch else text + "\n" + "-" * min(width, 160), flush=True)
        done = ev["status"] is None or (ev["status"] or {}).get("phase") in TERMINAL or ev["verdict"] == "STALE"
        if not a.watch or done:
            return ev["exit_code"]
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            return ev["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
