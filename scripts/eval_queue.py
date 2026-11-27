#!/usr/bin/env python3
"""Run a queue of benchmark evaluations against one model endpoint, starting each next benchmark as soon as the current
one is draining (its stragglers make few or no model calls), so a single slow task never leaves the GPU idle.

Run from the repository root (normally inside tmux, one queue per model server):
    .venv/train/bin/python scripts/eval_queue.py --name e1_sft --version sft --model-name qwen35-sft \
        --api-base http://172.18.0.1:8000/v1 --queue asibench,swebench_pro,hle [--label v5] [--asi-submit] [--note TEXT] \
        [--start-after pid:N | file:PATH | drain:BENCH:RUN_ID | at:UTC ...] [--require COND ...] \
        [--gate BENCH=file:PATH | BENCH=cpu_below:PCT ...] \
        [--yield-to PATH ...] [--extra terminal_bench="--limit 1"] [--out-root DIR] [--poll 30]

Inputs : scripts/evaluate.sh and the locks it reads; a serving endpoint (scripts/deploy_checkpoint.sh starts one).
Outputs: each benchmark is a normal evaluation run (logs/<run_id>/, results/<version>/<benchmark>/<run_id>/), started in its own
         process group so nothing here can stop it; the queue itself writes logs/eval_queue_<name>_<UTC>/ with the five files
         (metadata.json with the queue, run ids and start/drain/end times; eval.log; metrics.jsonl one line per event; errors.log;
         resources.csv) and <benchmark>.out (each runner's stdout).
Start: immediately, or with --start-after once ANY listed condition holds: `pid:N` (that process has exited, e.g. the dev run
         sharing the endpoint), `file:PATH` (a marker exists, e.g. logs/coordination/sft_dev_swe_qa.exited),
         `drain:BENCH:RUN_ID` (a run started outside this queue on the same endpoint is draining, by the rules below) or
         `at:YYYY-MM-DDTHH:MM:SSZ` (that UTC time has passed, e.g. a time cap on a marker wait); with --require (same
         syntax, repeatable) EVERY such condition must hold too, e.g. two co-located runs both draining; the wait is logged.
Draining rules (the next benchmark starts when the current run's state says so, or when its process exits):
         asibench       at most 1 instance left without a result file (logs/<run_id>/asibench/run/*/*__b1.json)
         terminal_bench at most 1 trial left without finished_at (logs/<run_id>/harbor/seed_*/*/result.json)
         swebench_pro   the agent phase is over (eval.log shows the first "harbor regrade:" launch; the re-grade makes no model calls)
         hle            the prediction passes are over (eval.log shows the first "judge_pass1:" launch; the judge calls OpenAI only)
Gates (--gate, repeatable): a gated benchmark is launched only once ANY of its conditions holds: `file:PATH` exists, or
         `cpu_below:PCT` (this host's CPU busy share over one --poll interval is below PCT), e.g. container-heavy runs that
         may start only after the data freeze or while apex-cpu is under 50 % busy. The wait and the condition met are logged.
Yield (--yield-to, repeatable): once any PATH exists (higher-priority work is waiting for this GPU), the queue launches
         nothing more; started runs finish, the benchmarks not started are recorded in metadata.json and the queue ends
         with status "yielded".
Coordination: while a terminal_bench run is active the queue keeps logs/coordination/terminal_bench_active.json (run id, start,
         endpoint); the data-building sessions pause Docker pulls and builds while it exists. It is removed when the run exits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = ("asibench", "swebench_pro", "terminal_bench", "hle")
COORD = REPO / "logs" / "coordination" / "terminal_bench_active.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_time(text: str) -> dt.datetime | None:
    """YYYY-MM-DDTHH:MM:SSZ -> an aware UTC datetime (None if malformed)."""
    try:
        return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


class CpuSampler:
    """This host's CPU busy share between two calls, from the aggregate `cpu` line of /proc/stat (idle = idle + iowait)."""

    def __init__(self, stat: Path = Path("/proc/stat")):
        self.stat = stat
        self.last = self.read()

    def read(self) -> tuple[int, int] | None:
        try:
            v = [int(x) for x in self.stat.read_text().splitlines()[0].split()[1:9]]
        except (OSError, ValueError, IndexError):
            return None
        return sum(v), v[3] + v[4]

    def busy_pct(self) -> float | None:
        """Busy share in percent since the previous call (None without two readings or elapsed ticks)."""
        cur, prev = self.read(), self.last
        self.last = cur
        if cur is None or prev is None or cur[0] <= prev[0]:
            return None
        total, idle = cur[0] - prev[0], cur[1] - prev[1]
        return 100.0 * (total - idle) / total


class Queue:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dir = REPO / "logs" / f"eval_queue_{a.name}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.dir.mkdir(parents=True)
        (self.dir / "errors.log").write_text("")
        self.t0 = time.time()
        self.meta = {"run_id": self.dir.name, "kind": "eval_queue", "name": a.name, "version": a.version, "model_name": a.model_name,
                     "api_base": a.api_base, "label": a.label, "queue": a.queue, "extra": a.extra, "out_root": a.out_root, "asi_submit": a.asi_submit,
                     "command": " ".join(map(shlex.quote, sys.argv)), "host": socket.gethostname(), "start_time": now(),
                     "end_time": None, "status": "running", "runs": [], "start_after": a.start_after, "start_reason": None,
                     "require": a.require, "gates": a.gate, "yield_to": a.yield_to, "not_started": [],
                     "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()}
        self.save()

    def save(self) -> None:
        (self.dir / "metadata.json").write_text(json.dumps(self.meta, indent=2))

    def log(self, msg: str, **event) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with (self.dir / "eval.log").open("a") as f:
            f.write(line + "\n")
        if event:
            with (self.dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"ts": now(), **event}) + "\n")

    def error(self, msg: str) -> None:
        with (self.dir / "errors.log").open("a") as f:
            f.write(f"[{now()}] {msg}\n")
        self.log(f"ERROR: {msg}")

    # ------------------------------------------------------------------------------------------------------ runs
    def launch(self, bench: str) -> dict:
        label = f"_{self.a.label}" if self.a.label else ""
        run_id = f"{bench}_{self.a.version}{label}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        cmd = ["scripts/evaluate.sh", bench, "--version", self.a.version, "--model-name", self.a.model_name,
               "--api-base", self.a.api_base, "--run-id", run_id, "--note", f"{self.a.note} (queue {self.dir.name})".strip()]
        if bench == "asibench" and self.a.asi_submit:
            cmd.append("--submit")
        if self.a.out_root:
            cmd += ["--out-root", self.a.out_root]
        cmd += shlex.split(self.a.extra.get(bench, ""))
        out = (self.dir / f"{bench}.out").open("a")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        run = {"benchmark": bench, "run_id": run_id, "pid": proc.pid, "command": " ".join(map(shlex.quote, cmd)),
               "started": now(), "drained": None, "ended": None, "exit_code": None}
        self.meta["runs"].append(run)
        self.save()
        self.log(f"started {bench} run {run_id} (pid {proc.pid})", event="start", benchmark=bench, run_id=run_id)
        if bench == "terminal_bench":
            coord = self.coord()
            coord.parent.mkdir(parents=True, exist_ok=True)
            coord.write_text(json.dumps({"run_id": run_id, "started": run["started"], "api_base": self.a.api_base, "queue": self.dir.name}, indent=2))
            self.log(f"coordination marker written: {coord} (Docker pause for the data-building sessions)")
        return {"proc": proc, "run": run}

    def coord(self) -> Path:
        """The Terminal-Bench marker; under --out-root for dry runs, so a dry run never pauses the other sessions."""
        return (Path(self.a.out_root).resolve() / "logs" / "coordination" / COORD.name) if self.a.out_root else COORD

    def run_root(self, run_id: str) -> Path:
        return (Path(self.a.out_root).resolve() if self.a.out_root else REPO) / "logs" / run_id

    def draining(self, bench: str, run_id: str) -> bool:
        root = self.run_root(run_id)
        log = (root / "eval.log").read_text(errors="replace") if (root / "eval.log").exists() else ""
        if bench == "swebench_pro":
            return "harbor regrade:" in log
        if bench == "hle":
            return "judge_pass1:" in log or "done: accuracy" in log
        if bench == "asibench":
            expected = self.expected(bench, log)
            done = len(list((root / "asibench" / "run").glob("*/*__b1.json")))
            return expected is not None and done >= expected - 1
        if bench == "terminal_bench":
            expected = self.expected(bench, log)
            done = 0
            for rf in (root / "harbor").glob("seed_*/*/result.json"):
                try:
                    done += bool(json.loads(rf.read_text()).get("finished_at"))
                except (json.JSONDecodeError, OSError):
                    pass
            return expected is not None and done >= expected - 1
        return False

    @staticmethod
    def expected(bench: str, log: str) -> int | None:
        """The run's own task count from its first log line (e.g. 'tasks=60', 'tasks=89 seeds=[1]')."""
        for line in log.splitlines():
            if "run_id=" in line and " tasks=" in line:
                n = int(line.split(" tasks=")[1].split()[0])
                if bench == "terminal_bench" and "seeds=[" in line:
                    n *= len(line.split("seeds=[")[1].split("]")[0].split(","))
                return n
        return None

    def finish_run(self, h: dict) -> None:
        run = h["run"]
        run["ended"], run["exit_code"] = now(), h["proc"].returncode
        self.save()
        self.log(f"{run['benchmark']} run {run['run_id']} exited {run['exit_code']}", event="end", benchmark=run["benchmark"],
                 run_id=run["run_id"], exit_code=run["exit_code"])
        if run["exit_code"] != 0:
            self.error(f"{run['benchmark']} run {run['run_id']} exited {run['exit_code']} (see {run['benchmark']}.out)")
        coord = self.coord()
        if run["benchmark"] == "terminal_bench" and coord.exists() and json.loads(coord.read_text()).get("run_id") == run["run_id"]:
            coord.unlink()
            self.log("coordination marker removed (Docker work may resume)")

    @staticmethod
    def start_condition(conds: list[str]) -> str | None:
        """The first --start-after condition that holds (pid:N exited, file:PATH exists, at:UTC passed), or None."""
        for c in conds:
            kind, _, val = c.partition(":")
            if kind == "file" and Path(val).exists():
                return c
            if kind == "at" and dt.datetime.now(dt.timezone.utc) >= utc_time(val):
                return c
            if kind == "pid":
                try:
                    os.kill(int(val), 0)
                except ProcessLookupError:
                    return c
                except PermissionError:
                    pass
        return None

    def drain_condition(self, conds: list[str]) -> str | None:
        """The first drain:BENCH:RUN_ID condition whose run is draining, or None."""
        for c in conds:
            kind, _, val = c.partition(":")
            if kind == "drain":
                bench, _, run_id = val.partition(":")
                if self.draining(bench, run_id):
                    return c
        return None

    def gate_condition(self, bench: str, cpu_busy: float | None) -> str | None:
        """The first --gate condition of `bench` that holds (None while it is closed); ungated benchmarks pass at once."""
        conds = [c for b, c in self.a.gate if b == bench]
        if not conds:
            return "ungated"
        for c in conds:
            kind, _, val = c.partition(":")
            if kind == "file" and Path(val).exists():
                return c
            if kind == "cpu_below" and cpu_busy is not None and cpu_busy < float(val):
                return f"{c} (busy {cpu_busy:.0f} %)"
        return None

    def yield_condition(self) -> str | None:
        return next((p for p in self.a.yield_to if Path(p).exists()), None)

    def holds(self, cond: str) -> bool:
        return (self.start_condition([cond]) or self.drain_condition([cond])) is not None

    def start_hit(self) -> str | None:
        """The start reason once --start-after (any) and --require (all) are satisfied, else None."""
        hit = (self.start_condition(self.a.start_after) or self.drain_condition(self.a.start_after)) if self.a.start_after else None
        if self.a.start_after and hit is None:
            return None
        if not all(self.holds(c) for c in self.a.require):
            return None
        return " and ".join(filter(None, [hit, ", ".join(self.a.require)]))

    def wait_start(self) -> None:
        if not self.a.start_after and not self.a.require:
            return
        parts = ([f"any of: {', '.join(self.a.start_after)}"] if self.a.start_after else []) + \
                ([f"all of: {', '.join(self.a.require)}"] if self.a.require else [])
        self.log(f"waiting to start until {' and '.join(parts)}")
        while (hit := self.start_hit()) is None:
            time.sleep(self.a.poll)
        self.meta["start_reason"] = f"{hit} at {now()}"
        self.save()
        self.log(f"start condition met: {hit}", event="start_condition", condition=hit)

    def go(self) -> int:
        self.wait_start()
        pending, active, waiting_gate = list(self.a.queue), [], None
        cpu = CpuSampler()
        while pending or active:
            busy = cpu.busy_pct()
            if pending and (hit := self.yield_condition()):
                self.meta.update(status="yielding", not_started=pending)
                self.save()
                self.log(f"yielding to {hit}: not starting {', '.join(pending)}", event="yield", condition=hit, not_started=pending)
                pending = []
            for h in [h for h in active if h["proc"].poll() is not None]:
                self.finish_run(h)
                active.remove(h)
            # the newest active run gates the next start
            gate = active[-1] if active else None
            if gate and not gate["run"]["drained"] and self.draining(gate["run"]["benchmark"], gate["run"]["run_id"]):
                gate["run"]["drained"] = now()
                self.save()
                self.log(f"{gate['run']['benchmark']} is draining; the next benchmark may start", event="drain",
                         benchmark=gate["run"]["benchmark"], run_id=gate["run"]["run_id"])
            if pending and (gate is None or gate["run"]["drained"]):
                if (opened := self.gate_condition(pending[0], busy)) is None:
                    if waiting_gate != pending[0]:
                        waiting_gate = pending[0]
                        self.log(f"{pending[0]} waits for its gate: any of {', '.join(c for b, c in self.a.gate if b == pending[0])}")
                else:
                    if opened != "ungated":
                        self.log(f"{pending[0]} gate open: {opened}", event="gate", benchmark=pending[0], condition=opened)
                    active.append(self.launch(pending.pop(0)))
            time.sleep(self.a.poll if (pending or active) else 0)
        dur = time.time() - self.t0
        self.meta.update(end_time=now(), status="yielded" if self.meta["not_started"] else "finished")
        self.save()
        (self.dir / "resources.csv").write_text("metric,value,unit,note\n"
                                                f"duration_sec,{dur:.1f},s,queue wall clock (each run records its own GPU hours and cost)\n")
        self.log(f"queue {self.a.name} finished in {dur / 3600:.2f} h")
        return 0 if all(r["exit_code"] == 0 for r in self.meta["runs"]) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="queue name, e.g. e1_sft")
    p.add_argument("--version", required=True, choices=["base", "sft", "stage2"])
    p.add_argument("--model-name", required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--queue", required=True, help=f"comma-separated, in order; from {BENCHMARKS}")
    p.add_argument("--asi-submit", action="store_true", help="upload ASI-Bench's bundle as a Portal draft when its run ends")
    p.add_argument("--extra", action="append", default=[], metavar="BENCH=ARGS", help="extra runner args for one benchmark (pilots and dry runs only)")
    p.add_argument("--out-root", default=None, help="DRY RUNS ONLY: passed to every runner (see src/apex_eval/common.py)")
    p.add_argument("--start-after", action="append", default=[], metavar="pid:N|file:PATH|drain:BENCH:RUN_ID|at:UTC",
                   help="start the first benchmark only once any such condition holds (repeatable)")
    p.add_argument("--require", action="append", default=[], metavar="COND",
                   help="like --start-after, but every --require condition must hold before the first benchmark starts (repeatable)")
    p.add_argument("--gate", action="append", default=[], metavar="BENCH=file:PATH|BENCH=cpu_below:PCT",
                   help="launch BENCH only once any of its gate conditions holds (repeatable)")
    p.add_argument("--yield-to", action="append", default=[], metavar="PATH",
                   help="launch nothing more once PATH exists; started runs finish (repeatable)")
    p.add_argument("--label", default="", help="model build in every run id, e.g. v5 -> asibench_sft_v5_<UTC> (several builds of one version)")
    p.add_argument("--note", default="")
    p.add_argument("--poll", type=float, default=30.0)
    a = p.parse_args()
    a.queue = [b.strip() for b in a.queue.split(",") if b.strip()]
    bad = [b for b in a.queue if b not in BENCHMARKS]
    if bad:
        p.error(f"unknown benchmarks: {bad}")
    a.extra = dict(e.split("=", 1) for e in a.extra)
    bad = [c for c in a.start_after + a.require if c.partition(":")[0] not in ("pid", "file", "drain", "at") or not c.partition(":")[2]
           or (c.startswith("at:") and utc_time(c[3:]) is None)
           or (c.startswith("pid:") and not c[4:].isdigit())
           or (c.startswith("drain:") and (c.split(":")[1] not in BENCHMARKS or len(c.split(":")) != 3 or not c.split(":")[2]))]
    if bad:
        p.error(f"--start-after / --require take pid:N, file:PATH, drain:BENCH:RUN_ID or at:YYYY-MM-DDTHH:MM:SSZ, got {bad}")
    gates = [g.partition("=") for g in a.gate]
    bad = [g for g, (b, _, c) in zip(a.gate, gates) if b not in a.queue or c.partition(":")[0] not in ("file", "cpu_below")
           or not c.partition(":")[2] or (c.startswith("cpu_below:") and not c[10:].replace(".", "", 1).isdigit())]
    if bad:
        p.error(f"--gate takes BENCH=file:PATH or BENCH=cpu_below:PCT for a queued benchmark, got {bad}")
    a.gate = [(b, c) for b, _, c in gates]
    q = Queue(a)
    signal.signal(signal.SIGTERM, lambda *_: (q.error("queue terminated; started runs keep running in their own sessions"), sys.exit(143)))
    return q.go()


if __name__ == "__main__":
    sys.exit(main())
