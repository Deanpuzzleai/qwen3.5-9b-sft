"""Continuous priority scheduler for the rollout and grading workers (docs/DATA_PLAN.md section 11: 64 to 96 rollout
sandboxes on apex-cpu, grading in parallel; section 12 drop order D: whatever is graded by the cut-off is the set).

Every work item is one worker subprocess that writes one JSON record (an Episode or a Verdict, keyed by episode_id). The
scheduler keeps the global slot budget filled in priority order (callers use (sample_index, family, task order), so a
cut-off run covers every task once before any task twice) under per-family caps. Finished records are appended to
<work_dir>/<records_name> (fsynced per line); the per-item state table goes to <work_dir>/state.json after every
transition. A restart with the same work_dir skips every item that already has a record (resume by episode id).

Environment failures (the collector returns retry=True) are re-queued up to max_env_retries times; when retries run
out the last record is written as it is. Nothing is dropped: an item without any record ends `failed` in state.json.
Time limits: `stop_launching_at` (epoch seconds) starts nothing new after it; `deadline` terminates the running workers
(state `cut_off`, as are items never launched) and returns. SIGTERM / Ctrl-C terminate the running workers (state
`interrupted`; their workers remove their containers) and propagate, so the RunDir records an aborted run. A `feeder`
callback adds items while the loop runs (grade_episodes.py --follow); the loop ends when the heap and the running set
are empty and `feeder_done()` is true. A 1-minute load-average ceiling pauses new launches, and so does the Terminal-Bench
container cap: while a reportable Terminal-Bench run is active (apex_data.coordination marker), no launch starts once the
host runs `tb_container_cap` or more containers (`docker ps -q`, counted at most every 10 s and incremented per launch).
A `control_file` (JSON) changes the limits of a running scheduler: it is re-read every `control_poll_sec` (30 s) when it
exists, and its keys overlay the starting limits: pause_launches (bool), load_pause (float or null), tb_container_cap (int
or null), max_sandboxes (int, the global slots), per_family ({family: int}, merged over the starting values). Removing the
file restores the starting limits. Running episodes are never touched; an unreadable or invalid file is ignored with a
warning; every change is logged and recorded (`limit_changes`, in state.json and through `on_limits`).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import heapq
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from apex_data.coordination import terminal_bench_active

PENDING, RUNNING, DONE, FAILED, INTERRUPTED, CUT_OFF = "pending", "running", "done", "failed", "interrupted", "cut_off"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str | None) -> float | None:
    """ISO-8601 time (UTC when no offset is given, 'Z' allowed) -> epoch seconds; None stays None."""
    if not text:
        return None
    t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).timestamp()


@dataclass(order=True)
class WorkItem:
    priority: tuple[Any, ...]
    key: str = field(compare=False)                  # record key: the episode_id
    family: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict, repr=False)
    slots: int = field(compare=False, default=1)
    timeout_sec: float | None = field(compare=False, default=None)   # hard kill after this many seconds


@dataclass
class Outcome:
    record: dict[str, Any] | None        # appended to the records file when not None
    retry: bool = False                  # environment failure: re-queue while retries remain
    note: str = ""


@dataclass
class Limits:
    global_slots: int = 96
    per_family: dict[str, int] = field(default_factory=lambda: {"swe": 64, "terminal": 32, "grade": 32})
    load_pause: float | None = None
    max_env_retries: int = 1             # plan section 11: operational failures are retried once, then recorded
    poll_sec: float = 0.5
    tb_container_cap: int | None = None  # while Terminal-Bench is active: no launch at >= this many host containers (None = off)


CONTAINER_COUNT_TTL_SEC = 10.0
CONTROL_KEYS = ("pause_launches", "load_pause", "tb_container_cap", "max_sandboxes", "per_family")


def running_containers() -> int | None:
    """Containers running on this host (`docker ps -q`), None when docker does not answer."""
    try:
        r = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return len(r.stdout.split()) if r.returncode == 0 else None


Launcher = Callable[[WorkItem], subprocess.Popen]
Collector = Callable[[WorkItem, int | None], Outcome]


class Scheduler:
    def __init__(self, work_dir: Path, items: Iterable[WorkItem], launcher: Launcher, collector: Collector, limits: Limits | None = None, *,
                 records_name: str = "episodes.jsonl", log: Callable[[str], None] = print,
                 on_record: Callable[[dict[str, Any]], None] | None = None, stop_launching_at: float | None = None,
                 deadline: float | None = None, feeder: Callable[[], list[WorkItem]] | None = None,
                 feeder_done: Callable[[], bool] | None = None, feed_every_sec: float = 30.0, control_file: Path | None = None,
                 control_poll_sec: float = 30.0, on_limits: Callable[[dict[str, Any]], None] | None = None):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.work_dir / records_name
        self.state_path = self.work_dir / "state.json"
        self.launcher, self.collector, self.limits = launcher, collector, limits or Limits()
        self.base_limits = dataclasses.replace(self.limits, per_family=dict(self.limits.per_family))
        self.pause_launches = False
        self.control_file, self.control_poll_sec, self.on_limits = (Path(control_file) if control_file else None), control_poll_sec, on_limits
        self.limit_changes: list[dict[str, Any]] = []
        self._control_seen: Any = None
        self._last_control = self._last_pause_log = 0.0
        self.log, self.on_record = log, on_record
        self.stop_launching_at, self.deadline = stop_launching_at, deadline
        self.feeder, self.feeder_done, self.feed_every_sec = feeder, feeder_done, feed_every_sec
        self.done_keys = self._existing_keys()
        self.state: dict[str, dict[str, Any]] = self._load_state()
        self.heap: list[WorkItem] = []
        self.queued: set[str] = set()
        self.running: dict[str, tuple[subprocess.Popen, WorkItem, float]] = {}
        self.n_skipped = 0
        self.cut_off = False
        self._last_load_log = self._last_feed = self._last_tb_log = 0.0
        self._containers: tuple[float, int | None] = (0.0, None)   # (counted at, count) for the Terminal-Bench cap
        self.add_items(items)

    # --- persistence ---
    def _existing_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.records_path.exists():
            with self.records_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        keys.add(json.loads(line)["episode_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return keys

    def _load_state(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text()).get("items", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"updated_at": utc_now(), "counts": self.counts(), "limit_changes": self.limit_changes,
                                   "items": self.state}, indent=1))
        tmp.replace(self.state_path)

    def _append_record(self, rec: dict[str, Any]) -> None:
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done_keys.add(rec["episode_id"])
        if self.on_record:
            self.on_record(rec)

    # --- queue ---
    def add_items(self, items: Iterable[WorkItem]) -> int:
        n = 0
        for it in items:
            if it.key in self.done_keys:
                if self.state.get(it.key, {}).get("status") != DONE:
                    self.n_skipped += 1
                    self.state[it.key] = {**self.state.get(it.key, {}), "status": DONE, "resumed": True}
                continue
            if it.key in self.queued or it.key in self.running:
                continue
            heapq.heappush(self.heap, it)
            self.queued.add(it.key)
            st = self.state.setdefault(it.key, {"attempts": 0})
            st.update(status=PENDING, family=it.family)
            n += 1
        return n

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for st in self.state.values():
            c[st.get("status", "?")] = c.get(st.get("status", "?"), 0) + 1
        c["running_slots"] = sum(it.slots for _, it, _ in self.running.values())
        return c

    def _used(self, family: str | None = None) -> int:
        return sum(it.slots for _, it, _ in self.running.values() if family is None or it.family == family)

    def _can_launch(self, it: WorkItem) -> bool:
        cap = self.limits.per_family.get(it.family, self.limits.global_slots)
        return self._used() + it.slots <= self.limits.global_slots and self._used(it.family) + it.slots <= cap

    def _load_ok(self) -> bool:
        if self.limits.load_pause is None:
            return True
        load = os.getloadavg()[0]
        if load > self.limits.load_pause:
            if time.time() - self._last_load_log > 60:
                self.log(f"scheduler: load average {load:.0f} > {self.limits.load_pause:.0f}; pausing new launches")
                self._last_load_log = time.time()
            return False
        return True

    def _effective_limits(self) -> dict[str, Any]:
        lim = self.limits
        return {"pause_launches": self.pause_launches, "load_pause": lim.load_pause, "tb_container_cap": lim.tb_container_cap,
                "max_sandboxes": lim.global_slots, "per_family": dict(lim.per_family)}

    def _apply_control(self, force: bool = False) -> None:
        """Re-read the control file (module docstring) and apply a change of limits; never touches running episodes."""
        if self.control_file is None or (not force and time.time() - self._last_control < self.control_poll_sec):
            return
        self._last_control = time.time()
        try:
            content = json.loads(self.control_file.read_text()) if self.control_file.exists() else {}
            if not isinstance(content, dict):
                raise ValueError("not a JSON object")
            unknown = sorted(set(content) - set(CONTROL_KEYS))
            if unknown:
                raise ValueError(f"unknown keys {unknown} (allowed: {list(CONTROL_KEYS)})")
            new = dataclasses.replace(self.base_limits, per_family=dict(self.base_limits.per_family))
            pause = bool(content.get("pause_launches", False))
            if "load_pause" in content:
                new.load_pause = None if content["load_pause"] is None else float(content["load_pause"])
            if "tb_container_cap" in content:
                new.tb_container_cap = None if content["tb_container_cap"] is None else int(content["tb_container_cap"])
            if "max_sandboxes" in content:
                new.global_slots = int(content["max_sandboxes"])
            if "per_family" in content:
                if not isinstance(content["per_family"], dict):
                    raise ValueError("per_family must be an object {family: slots}")
                new.per_family.update({str(k): int(v) for k, v in content["per_family"].items()})
        except (OSError, ValueError, TypeError) as e:
            if self._control_seen != ("error", str(e)):
                self.log(f"scheduler: control file {self.control_file} ignored: {e}")
                self._control_seen = ("error", str(e))
            return
        before = self._effective_limits()
        self.limits, self.pause_launches = new, pause
        after = self._effective_limits()
        self._control_seen = after
        if after != before:
            change = {"at": utc_now(), "control_file": str(self.control_file), "present": self.control_file.exists(), "limits": after,
                      "changed": {k: [before[k], after[k]] for k in after if before[k] != after[k]}}
            self.limit_changes.append(change)
            self.log(f"scheduler: limits changed by {self.control_file}: {json.dumps(change['changed'])}")
            if self.on_limits:
                self.on_limits(change)
            self._persist()

    def _tb_ok(self) -> bool:
        """False while a reportable Terminal-Bench run is active and the host runs >= tb_container_cap containers (an
        unanswered docker count also pauses: launches would fail anyway)."""
        cap = self.limits.tb_container_cap
        if cap is None or terminal_bench_active() is None:
            return True
        now = time.time()
        if now - self._containers[0] > CONTAINER_COUNT_TTL_SEC:
            self._containers = (now, running_containers())
        n = self._containers[1]
        if n is None or n >= cap:
            if now - self._last_tb_log > 60:
                self.log(f"scheduler: Terminal-Bench is active and the host runs {n if n is not None else 'an unknown number of'} containers "
                         f"(cap {cap}); pausing new launches")
                self._last_tb_log = now
            return False
        return True

    def plan(self) -> dict[str, Any]:
        by_family: dict[str, int] = {}
        for it in self.heap:
            by_family[it.family] = by_family.get(it.family, 0) + 1
        return {"pending": len(self.heap), "skipped_done": self.n_skipped, "by_family": by_family,
                "limits": {"global_slots": self.limits.global_slots, "per_family": self.limits.per_family, "load_pause": self.limits.load_pause,
                           "max_env_retries": self.limits.max_env_retries, "tb_container_cap": self.limits.tb_container_cap},
                "control_file": str(self.control_file) if self.control_file else None,
                "stop_launching_at": self.stop_launching_at, "deadline": self.deadline}

    # --- main loop ---
    def _feeding(self) -> bool:
        return self.feeder is not None and not (self.feeder_done() if self.feeder_done else True)

    def run(self) -> dict[str, int]:
        self._apply_control(force=True)
        self.log(f"scheduler: {len(self.heap)} pending, {self.n_skipped} already done, limits {self.plan()['limits']}"
                 + (f", pause_launches {self.pause_launches} (control file {self.control_file})" if self.control_file else ""))
        try:
            while True:
                now = time.time()
                self._apply_control()
                if self.deadline is not None and now >= self.deadline:
                    self._cut_off("deadline reached")
                    break
                if self.feeder is not None and now - self._last_feed >= self.feed_every_sec:
                    self._last_feed = now
                    added = self.add_items(self.feeder())
                    if added:
                        self.log(f"scheduler: {added} new item(s) fed")
                if not (self.heap or self.running or self._feeding()):
                    # the producers have finished, but the feed may have grown since its last read: drain it once more
                    added = self.add_items(self.feeder()) if self.feeder is not None else 0
                    if not added:
                        break
                    self.log(f"scheduler: {added} new item(s) fed after the producers finished")
                    continue
                if self.stop_launching_at is None or now < self.stop_launching_at:
                    self._launch_ready()
                elif not self.running:          # past the launch cut-off and nothing left running
                    self._cut_off("launch cut-off reached")
                    break
                self._collect_finished()
                time.sleep(self.limits.poll_sec)
        except BaseException:
            self._terminate_running(INTERRUPTED)
            raise
        finally:
            self._persist()
        return self.counts()

    def _launch_ready(self) -> None:
        if self.pause_launches and self.heap:
            if time.time() - self._last_pause_log > 300:
                self.log(f"scheduler: launches paused by {self.control_file} ({len(self.heap)} pending, {len(self.running)} running)")
                self._last_pause_log = time.time()
            return
        if not self.heap or not self._load_ok():
            return
        deferred: list[WorkItem] = []
        launched = 0
        while self.heap and self._used() < self.limits.global_slots and self._tb_ok():
            it = heapq.heappop(self.heap)
            if not self._can_launch(it):
                deferred.append(it)
                continue
            self.queued.discard(it.key)
            st = self.state.setdefault(it.key, {"attempts": 0})
            st["attempts"] = st.get("attempts", 0) + 1
            try:
                proc = self.launcher(it)
            except Exception as e:  # noqa: BLE001 - a launcher error is an environment failure of that item
                self._finish(it, Outcome(record=None, retry=True, note=f"launch failed: {type(e).__name__}: {e}"[:300]))
                continue
            self.running[it.key] = (proc, it, time.time())
            if self._containers[1] is not None:   # until the next count, each launch adds at least one container
                self._containers = (self._containers[0], self._containers[1] + 1)
            st.update(status=RUNNING, started_at=utc_now(), pid=proc.pid, family=it.family)
            launched += 1
        for it in deferred:
            heapq.heappush(self.heap, it)
        if launched:
            self._persist()

    def _collect_finished(self) -> None:
        finished = []
        for key, (proc, it, t0) in list(self.running.items()):
            rc = proc.poll()
            if rc is None and it.timeout_sec is not None and time.time() - t0 > it.timeout_sec:
                self.log(f"scheduler: {key} exceeded {it.timeout_sec:.0f}s; terminating")
                _terminate(proc)
                rc = proc.poll()
                self.state[key]["killed_after_sec"] = it.timeout_sec
            if rc is not None:
                finished.append((key, it, rc))
        for key, it, rc in finished:
            del self.running[key]
            try:
                outcome = self.collector(it, rc)
            except Exception as e:  # noqa: BLE001 - a collector crash must not take the scheduler down
                outcome = Outcome(record=None, retry=True, note=f"collect failed: {type(e).__name__}: {e}"[:300])
            self._finish(it, outcome, rc)
        if finished:
            self._persist()

    def _finish(self, it: WorkItem, outcome: Outcome, rc: int | None = None) -> None:
        st = self.state.setdefault(it.key, {"attempts": 0})
        st.update(finished_at=utc_now(), rc=rc, note=outcome.note or st.get("note", ""))
        if outcome.retry and st.get("attempts", 0) <= self.limits.max_env_retries:
            self.log(f"scheduler: {it.key} environment failure (attempt {st['attempts']}): {outcome.note}; re-queued")
            st["status"] = PENDING
            heapq.heappush(self.heap, it)
            self.queued.add(it.key)
            return
        if outcome.record is not None:   # retries exhausted: the last outcome is recorded as it is, never dropped
            if outcome.retry:
                st["retries_exhausted"] = True
                self.log(f"scheduler: {it.key} environment failure after {st['attempts']} attempt(s); recording it")
            self._append_record(outcome.record)
            st["status"] = DONE
        else:
            st["status"] = FAILED
            self.log(f"scheduler: {it.key} FAILED without a record: {outcome.note}")

    def _cut_off(self, why: str) -> None:
        self.cut_off = True
        n_pending = len(self.heap)
        self._terminate_running(CUT_OFF)
        for it in self.heap:
            self.state.setdefault(it.key, {})["status"] = CUT_OFF
        self.heap.clear()
        self.queued.clear()
        self.log(f"scheduler: {why}; {n_pending} item(s) never launched recorded as cut_off")

    def _terminate_running(self, status: str) -> None:
        if not self.running:
            return
        self.log(f"scheduler: terminating {len(self.running)} running worker(s) ({status})")
        for proc, _it, _t0 in self.running.values():
            proc.terminate()
        deadline = time.time() + 90
        for key, (proc, _it, _t0) in self.running.items():
            try:
                proc.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
            self.state[key].update(status=status, finished_at=utc_now())
        self.running.clear()


def _terminate(proc: subprocess.Popen, grace_sec: int = 60) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
