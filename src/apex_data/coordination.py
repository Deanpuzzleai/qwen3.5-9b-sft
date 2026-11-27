"""Coordination with the evaluation session on apex-cpu's shared Docker daemon.

While a reportable Terminal-Bench run is active, the evaluation queues (scripts/eval_queue.py) keep the marker file
logs/coordination/terminal_bench_active.json; it disappears when the run ends. Image pulls and builds of the data and
stage-2 tooling must not start while it exists (containers already running may finish): `wait_for_docker_window` is called
right before every `docker pull` and `docker build` and blocks until the marker is gone. Reading the marker never raises:
an unreadable marker counts as active.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from apex_data import paths

TB_ACTIVE = paths.LOGS / "coordination" / "terminal_bench_active.json"


def terminal_bench_active(path: Path = TB_ACTIVE) -> dict[str, Any] | None:
    """The marker's content while a reportable Terminal-Bench run is active, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {"marker": str(d)}
    except (OSError, ValueError):
        return {"marker": "unreadable"}


def wait_for_docker_window(what: str, log: Callable[[str], None] = print, poll_sec: float = 60.0, path: Path = TB_ACTIVE) -> float:
    """Block before `what` (a pull or build) while Terminal-Bench is active; returns the seconds waited."""
    t0 = time.time()
    last_note = 0.0
    while (marker := terminal_bench_active(path)) is not None:
        if time.time() - last_note >= 600:
            log(f"docker pause: Terminal-Bench is active ({path}: {json.dumps(marker)[:200]}); {what} waits")
            last_note = time.time()
        time.sleep(poll_sec)
    waited = time.time() - t0
    if waited >= 1:
        log(f"docker pause over after {waited:.0f} s; {what} starts")
    return waited
