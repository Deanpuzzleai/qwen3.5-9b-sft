"""Task-grouped, stratified draws over mixture units: the validation shard and the pilot (docs/DATA_PLAN.md revision 4.3
sections 4 and 9; WP7, unit U2.4).

Both draws work on tasks, never rows: every unit of a task (an SWE instance's row, a TermGrade task's one or two
episodes with their two trained turns each, a retention prompt) lands on one side, so no task straddles train and
validation and the pilot holds whole tasks. Strata are the cells (slice x quota part / source / retention part): each
cell gets round(fraction x its rows) rows by largest remainder over all cells, drawn as whole tasks in seeded order
until the cell's target is reached. The slice shares of the draw therefore follow the mixture's.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from apex_data.paths import SEED
from apex_data.render.select import SLICES, Unit

VAL_FRACTION = 0.05
PILOT_FRACTION = 0.10


def cell_of(u: Unit) -> str:
    return f"{u.slice}/{u.cell}"


def cell_targets(rows_per_cell: dict[str, int], fraction: float) -> dict[str, int]:
    """round(fraction x total rows) distributed over the cells by largest remainder (ties by cell name)."""
    total = sum(rows_per_cell.values())
    want = int(round(fraction * total))
    quota = {c: fraction * n for c, n in rows_per_cell.items()}
    take = {c: int(math.floor(q)) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: (-(quota[c] - take[c]), c))[: max(0, want - sum(take.values()))]:
        take[c] += 1
    return take


@dataclass
class Draw:
    """Tasks drawn per cell (`drawn`) and the rest (`rest`), as unit lists, with the arithmetic."""

    drawn: list[Unit]
    rest: list[Unit]
    report: dict[str, Any]


def task_grouped_draw(units: Sequence[Unit], fraction: float, *, seed: int = SEED, label: str = "split") -> Draw:
    """Draw whole tasks per cell until each cell's row target is reached. A task's units must all sit in one cell (true
    for every pool: a task belongs to one source and one part)."""
    tasks: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        tasks[u.task_id].append(u)
    cell_of_task: dict[str, str] = {}
    for t, us in tasks.items():
        cells = {cell_of(u) for u in us}
        if len(cells) != 1:
            raise ValueError(f"task {t} spans cells {sorted(cells)}")
        cell_of_task[t] = cells.pop()
    rows_per_cell: Counter[str] = Counter()
    for u in units:
        rows_per_cell[cell_of(u)] += len(u.row_keys)
    targets = cell_targets(dict(rows_per_cell), fraction)
    chosen: set[str] = set()
    got: Counter[str] = Counter()
    for cell in sorted(targets):
        ts = sorted(t for t, c in cell_of_task.items() if c == cell)
        random.Random(f"{seed}:{label}:{cell}").shuffle(ts)
        for t in ts:
            if got[cell] >= targets[cell]:
                break
            chosen.add(t)
            got[cell] += sum(len(u.row_keys) for u in tasks[t])
    drawn = [u for u in units if u.task_id in chosen]
    rest = [u for u in units if u.task_id not in chosen]
    per_slice = {}
    for s in SLICES:
        n_all = sum(len(u.row_keys) for u in units if u.slice == s)
        n_d = sum(len(u.row_keys) for u in drawn if u.slice == s)
        per_slice[s] = {"rows": n_all, "rows_drawn": n_d, "share_drawn": round(n_d / n_all, 4) if n_all else 0.0,
                        "tasks_drawn": len({u.task_id for u in drawn if u.slice == s})}
    report = {"fraction": fraction, "seed": seed, "label": label, "rows": sum(rows_per_cell.values()),
              "rows_drawn": sum(len(u.row_keys) for u in drawn), "tasks": len(tasks), "tasks_drawn": len(chosen),
              "units_drawn": len(drawn), "per_slice": per_slice,
              "per_cell": {c: {"rows": rows_per_cell[c], "target_rows": targets[c], "rows_drawn": got[c]} for c in sorted(targets)}}
    return Draw(drawn, rest, report)


def straddling_tasks(train: Iterable[Unit], dev: Iterable[Unit]) -> list[str]:
    """Tasks with units on both sides (must be empty)."""
    return sorted({u.task_id for u in train} & {u.task_id for u in dev})
