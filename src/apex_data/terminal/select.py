"""Episode selection of the terminal slice (docs/DATA_PLAN.md 3.3): 1,000 episodes, at most 500 from TerminalHorizon by
the category quota (one episode per environment), the rest (at least 500) TermGrade verifier passes.

TerminalHorizon. Each of the 25 environment categories belongs to one Terminal-Bench 2.0 category group (`CATEGORY_GROUP`,
the table of plan 3.3; the groups follow Terminal-Bench 2.0's public category list). Per environment one admissible
episode is chosen uniformly at random (never the shorter one by rule); per group, environments are drawn uniformly at
random up to the group's quota. A group short of supply passes its shortfall to software engineering (`ROLLOVER_GROUP`),
which draws further environments from its remaining supply; what software engineering cannot absorb stays unfilled and
is reported. Every random stream is `random.Random(f"{seed}:<purpose>:<key>")`, so a group's draw does not depend on the
other groups.

TermGrade. Tasks that all six labelled models solve at every counted attempt are skipped; dev tasks are excluded; at
most `TGR_PER_TASK` episodes per task. The draw covers tasks first: tasks in a seeded random order, one random episode
each, then a second random episode per task in the same order, until the target is met; so the 500 episodes span as many
tasks as the supply allows. If the supply after the dev exclusion is below the target, the skipped all-six-solve tasks are
re-admitted (plan 3.3 fallback) in the same way and counted.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from apex_data.schema import CATEGORY_GROUPS

TOTAL_EPISODES = 1_000
TH3K_MAX = 500
TGR_PER_TASK = 2
ROLLOVER_GROUP = "software_engineering"

# plan 3.3 quota table: Terminal-Bench 2.0 group -> TerminalHorizon categories, and the quota per group (sum 500)
GROUP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "software_engineering": ("Software engineering and developer tools", "Programming languages, compilers and runtimes",
                             "Operating systems and low-level systems"),
    "security": ("Cybersecurity, cryptography and privacy",),
    "system_administration": ("Networking and communications", "Cloud platforms and distributed infrastructure",
                              "Databases and storage systems"),
    "scientific_computing": ("Life sciences, medicine and health", "Earth, geospatial and environmental sciences",
                             "Physics, astronomy and instrumentation", "Chemistry, materials and molecular simulation",
                             "Mechanical, thermal and aerospace engineering", "Energy and public infrastructure",
                             "Robotics and autonomous systems", "Mathematics, statistics and formal methods"),
    "machine_learning": ("Machine learning and artificial intelligence",),
    "data_processing": ("Data engineering and analytics", "Finance, accounting and economic operations",
                        "Transport, logistics and operational decisions",
                        "Knowledge organization, humanities and public affairs"),
    "rest": ("Graphics, media and content tools", "Chips, electronics and embedded systems",
             "Industrial manufacturing and production engineering", "Agriculture, forestry and fisheries",
             "Education and educational assessment"),
}
QUOTAS: dict[str, int] = {"software_engineering": 150, "security": 50, "system_administration": 78,
                          "scientific_computing": 78, "machine_learning": 39, "data_processing": 72, "rest": 33}
CATEGORY_GROUP: dict[str, str] = {c: g for g, cats in GROUP_CATEGORIES.items() for c in cats}
assert tuple(GROUP_CATEGORIES) == CATEGORY_GROUPS and sum(QUOTAS.values()) == TH3K_MAX and len(CATEGORY_GROUP) == 25


def group_of(category: str | None) -> str | None:
    """Quota group of a TerminalHorizon category; None for an unknown or missing category (not eligible, counted)."""
    return CATEGORY_GROUP.get(category or "")


def _rng(seed: int, purpose: str, key: str = "") -> random.Random:
    return random.Random(f"{seed}:{purpose}:{key}")


def _shuffled(items: Iterable[str], seed: int, purpose: str, key: str = "") -> list[str]:
    out = sorted(items)
    _rng(seed, purpose, key).shuffle(out)
    return out


# --------------------------------------------------------------------------------------------- TerminalHorizon
def one_per_environment(candidates: dict[str, list[str]], seed: int) -> dict[str, str]:
    """{environment: one episode id drawn uniformly from its admissible episodes}."""
    return {env: _rng(seed, "th3k:env", env).choice(sorted(eps)) for env, eps in candidates.items() if eps}


@dataclass
class QuotaResult:
    selected: dict[str, list[str]] = field(default_factory=dict)      # group -> environments in draw order
    table: list[dict[str, Any]] = field(default_factory=list)
    unfilled: int = 0

    @property
    def environments(self) -> list[str]:
        return [e for g in self.selected for e in self.selected[g]]


def quota_fill(supply: dict[str, Sequence[str]], quotas: dict[str, int], seed: int,
               rollover_group: str = ROLLOVER_GROUP) -> QuotaResult:
    """Draw environments per group up to its quota; shortfalls roll into `rollover_group` (plan 3.3)."""
    if rollover_group not in quotas:
        raise ValueError(f"rollover group {rollover_group!r} has no quota")
    order = {g: _shuffled(supply.get(g, ()), seed, "th3k:quota", g) for g in quotas}
    res = QuotaResult()
    shortfall: dict[str, int] = {}
    for g, q in quotas.items():
        res.selected[g] = order[g][:q]
        shortfall[g] = q - len(res.selected[g])
    passed = sum(v for g, v in shortfall.items() if g != rollover_group)
    spare = order[rollover_group][quotas[rollover_group]:]
    extra = spare[:passed]
    res.selected[rollover_group] = res.selected[rollover_group] + extra
    res.unfilled = passed - len(extra) + shortfall[rollover_group]
    for g, q in quotas.items():
        res.table.append({"group": g, "quota": q, "supply_environments": len(order[g]),
                          "filled_own": min(q, len(order[g])), "shortfall": shortfall[g],
                          "rollover_out": shortfall[g] if g != rollover_group else 0,
                          "rollover_in": len(extra) if g == rollover_group else 0,
                          "selected": len(res.selected[g])})
    return res


# --------------------------------------------------------------------------------------------------- TermGrade
@dataclass
class TgrSelection:
    selected: list[str]
    counts: dict[str, Any]


def _task_first_order(by_task: dict[str, list[str]], seed: int, per_task: int, purpose: str) -> list[str]:
    tasks = _shuffled(by_task, seed, f"tgr:{purpose}:tasks")
    capped = {t: _shuffled(by_task[t], seed, "tgr:task", t)[:per_task] for t in tasks}
    return [capped[t][k] for k in range(per_task) for t in tasks if k < len(capped[t])]


def select_termgrade(episodes: dict[str, str], *, target: int, skip_tasks: set[str], dev_tasks: set[str], seed: int,
                     per_task: int = TGR_PER_TASK, readmit_skipped: bool = True) -> TgrSelection:
    """`episodes` = {episode_id: task} of the admissible verifier passes. Returns the drawn episode ids (draw order)."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for eid, task in episodes.items():
        by_task[task].append(eid)
    dev = {t: v for t, v in by_task.items() if t in dev_tasks}
    skipped = {t: v for t, v in by_task.items() if t in skip_tasks and t not in dev_tasks}
    eligible = {t: v for t, v in by_task.items() if t not in dev_tasks and t not in skip_tasks}
    order = _task_first_order(eligible, seed, per_task, "eligible")
    selected = order[:target]
    readmitted: list[str] = []
    if len(selected) < target and readmit_skipped:
        readmitted = _task_first_order(skipped, seed, per_task, "readmitted")[:target - len(selected)]
        selected += readmitted
    counts = {
        "admissible_episodes": len(episodes), "admissible_tasks": len(by_task),
        "dev_tasks_present": len(dev), "dev_episodes_excluded": sum(map(len, dev.values())),
        "all_six_solve_tasks_present": len(skipped), "all_six_solve_episodes_skipped": sum(map(len, skipped.values())),
        "eligible_tasks": len(eligible), "eligible_episodes": sum(map(len, eligible.values())),
        "eligible_episodes_after_per_task_cap": len(order), "per_task_cap": per_task, "target": target,
        "selected": len(selected), "selected_tasks": len({episodes[e] for e in selected}),
        "tasks_with_2_selected": sum(v == 2 for v in _count(episodes[e] for e in selected).values()),
        "readmitted_all_six_solve_episodes": len(readmitted), "shortfall": max(0, target - len(selected)),
    }
    return TgrSelection(selected, counts)


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return out
