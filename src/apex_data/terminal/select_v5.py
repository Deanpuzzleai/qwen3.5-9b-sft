"""Revision-5 selection of the terminal slice (data plan v5, 1.3 and 3.2; terminal revision 3):
TermGrade trials of several teachers, at most two per task, drawn by difficulty band in loss tokens.

Choice per task. `trial_order` ranks a task's passing trials that clear every profiler gate (efficiency.gate_failures:
the band limits over the trained turns, zero wasted actions, verified completion) by teacher priority (TEACHERS: accuracy,
then passes per output token on trials.parquet), then by mean thinking per trained turn, lowest first (ties: trial id).
The builder renders them in that order; the first whose rows pass the render gates is the task's first trial (rank 0),
the next one its second trial (rank 1; easy and medium tasks only, SECOND_BANDS).

The draw (`draw`), in loss tokens of the rendered rows (the trainer's token-averaged loss, plan 1.3); new tasks come
before second trials (diversity: TermGrade has no task-type labels, so distinct tasks are the measure):
  1. first trials. Hard: every hard task, once. The rest of the budget: medium MEDIUM_SHARE of it; easy takes what
     medium leaves, so a medium shortfall moves to easy (plan 1.3 supply rule) and an easy shortfall stays unfilled;
  2. second trials only fill budget left after the first trials, at most SECOND_SHARE_MAX of all episodes: medium's up
     to its quota first (band spread), then easy's, then medium's again, then hard's when `second_bands` allows them;
  3. within a band and rank, the trials split at the median of their mean thinking (`halves`): the lower half in a seeded
     random order, then the upper half (plan 1.3: the lower half is a preference, not a limit). Trials are taken in that
     order until the quota is reached; the one that reaches it is kept, so a band overshoots by less than one episode.
Each random stream is random.Random(f"{seed}:terminal_v5:<band>:<half>") (second trials: seed f"{seed}:second") over the
task ids sorted first, so a band's order does not depend on the other bands or on the input order.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BUDGET_LOSS_TOKENS = 2_500_000       # terminal rev 3 (plan 3.2 had ~1.5M)
MEDIUM_SHARE = 0.55                  # plan 3.2 / D1: the rest medium 55 / easy 45
SECOND_SHARE_MAX = 0.25              # second trials <= 25% of the terminal episodes
SECOND_BANDS = ("medium", "easy")    # hard tasks: once (plan 3.2, no forced exposure)
BANDS = ("hard", "medium", "easy")   # draw order
# teacher priority: Kimi-K3 > Qwen3.6-27B > Gemma-4 with reasoning > GLM-5.2 > DeepSeek-V4-Pro (TermGrade model keys)
TEACHERS = ("kimi", "qwen", "gemma_think", "glm", "deepseek")


@dataclass(frozen=True)
class TaskChoice:
    task: str
    band: str
    trial: str
    thinking_mean: float
    loss_tokens: int
    rank: int = 0                    # 0 = the task's first trial, 1 = its second


@dataclass
class Drawn:
    choice: TaskChoice
    order: int                       # position in the draw (0 = first)
    half: str                        # "all" (hard), "lower" or "upper"


@dataclass
class Draw:
    selected: list[Drawn] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)

    @property
    def loss_tokens(self) -> int:
        return sum(d.choice.loss_tokens for d in self.selected)


def trial_order(trials: Mapping[str, tuple[float, Sequence[str]]], teacher_of: Mapping[str, str] | None = None,
                teachers: Sequence[str] = TEACHERS) -> list[str]:
    """{trial: (mean thinking per trained turn, gate failures)} -> the gate-passing trials by teacher priority (when
    `teacher_of` is given), then lowest mean thinking first, then trial id."""
    prio = {t: i for i, t in enumerate(teachers)}
    key = (lambda t, m: (prio[teacher_of[t]], m, t)) if teacher_of is not None else (lambda t, m: (0, m, t))
    return [t for t, _ in sorted(((t, m) for t, (m, fails) in trials.items() if not fails), key=lambda x: key(*x))]


def halves(choices: Iterable[TaskChoice]) -> tuple[list[TaskChoice], list[TaskChoice]]:
    """(lower, upper) by mean thinking (ties: task id); the lower half takes the middle task of an odd count."""
    ranked = sorted(choices, key=lambda c: (c.thinking_mean, c.task))
    k = (len(ranked) + 1) // 2
    return ranked[:k], ranked[k:]


def _shuffled(items: list[TaskChoice], seed: Any, band: str, half: str) -> list[TaskChoice]:
    out = sorted(items, key=lambda c: c.task)
    random.Random(f"{seed}:terminal_v5:{band}:{half}").shuffle(out)
    return out


def band_order(choices: Iterable[TaskChoice], band: str, seed: Any) -> list[tuple[TaskChoice, str]]:
    """The draw order of one band: (choice, half) for the shuffled lower half, then the shuffled upper half."""
    lower, upper = halves(choices)
    return [(c, "lower") for c in _shuffled(lower, seed, band, "lower")] + \
           [(c, "upper") for c in _shuffled(upper, seed, band, "upper")]


def draw(choices: Iterable[TaskChoice], *, seed: int, budget: int = BUDGET_LOSS_TOKENS,
         medium_share: float = MEDIUM_SHARE, second_share_max: float = SECOND_SHARE_MAX,
         second_bands: Sequence[str] = SECOND_BANDS, hard_cap: bool = False) -> Draw:
    """The draw over every task's chosen trials (at most one per (task, rank); bands outside BANDS are ignored).
    `second_bands` names the bands whose tasks may have a second trial (v6 allows all three; hard ones come last).
    With `hard_cap` the budget is a ceiling (v6): medium, easy and second trials stop before a trial that would take
    the total over it, instead of keeping the one that crosses a quota."""
    by: dict[tuple[str, int], list[TaskChoice]] = {(b, r): [] for b in BANDS for r in (0, 1)}
    seen: set[tuple[str, int]] = set()
    for c in choices:
        if (c.task, c.rank) in seen:
            raise ValueError(f"task {c.task} has more than one trial of rank {c.rank}")
        if c.rank not in (0, 1) or (c.rank == 1 and c.band not in second_bands and c.band in BANDS):
            raise ValueError(f"task {c.task}: rank {c.rank} is not drawn in band {c.band}")
        seen.add((c.task, c.rank))
        if c.band in BANDS:
            by[(c.band, c.rank)].append(c)
    out = Draw()
    got = {b: 0 for b in BANDS}
    halves_taken: dict[tuple[str, int], list[str]] = {k: [] for k in by}

    def take(c: TaskChoice, half: str) -> None:
        out.selected.append(Drawn(c, len(out.selected), half))
        got[c.band] += c.loss_tokens
        halves_taken[(c.band, c.rank)].append(half)

    for c in sorted(by[("hard", 0)], key=lambda c: c.task):
        take(c, "all")
    rest = max(0, budget - got["hard"])
    quota = {"hard": None, "medium": round(rest * medium_share)}
    for band in ("medium", "easy"):
        if band == "easy":
            quota["easy"] = rest - got["medium"]
        for c, half in band_order(by[(band, 0)], band, seed):
            if got[band] >= quota[band] or (hard_cap and sum(got.values()) + c.loss_tokens > budget):
                break
            take(c, half)
    firsts = {b: got[b] for b in BANDS}
    allowance = int(len(out.selected) * second_share_max / (1 - second_share_max)) if second_share_max < 1 else 1 << 30
    seconds = {b: band_order(by[(b, 1)], b, f"{seed}:second") for b in BANDS}
    pos = {b: 0 for b in BANDS}
    for band, until in (("medium", quota["medium"]), ("easy", None), ("medium", None), ("hard", None)):
        if band not in second_bands:
            continue
        while pos[band] < len(seconds[band]):
            if sum(got.values()) >= budget or allowance <= 0 or (until is not None and got[band] >= until) or \
                    (hard_cap and sum(got.values()) + seconds[band][pos[band]][0].loss_tokens > budget):
                break
            take(*seconds[band][pos[band]])
            pos[band] += 1
            allowance -= 1
    for band in BANDS:
        supply = by[(band, 0)]
        lower, _ = halves(supply)
        n0 = sum(d.choice.band == band and d.choice.rank == 0 for d in out.selected)
        n1 = sum(d.choice.band == band and d.choice.rank == 1 for d in out.selected)
        row = {"band": band, "supply_tasks": len(supply), "supply_loss_tokens": sum(c.loss_tokens for c in supply),
               "supply_second_trials": len(by[(band, 1)]),
               "supply_second_loss_tokens": sum(c.loss_tokens for c in by[(band, 1)]),
               "quota_loss_tokens": quota[band], "drawn_tasks": n0, "drawn_second_trials": n1, "drawn_episodes": n0 + n1,
               "drawn_first_loss_tokens": firsts[band], "drawn_loss_tokens": got[band]}
        if band != "hard":
            row.update(lower_half_tasks=len(lower),
                       lower_half_max_thinking_mean=round(lower[-1].thinking_mean, 1) if lower else None,
                       drawn_from_lower_half=halves_taken[(band, 0)].count("lower"),
                       drawn_from_upper_half=halves_taken[(band, 0)].count("upper"),
                       shortfall_loss_tokens=max(0, quota[band] - got[band]))
        out.table.append(row)
    out.table[1]["moved_to_easy_loss_tokens"] = max(0, quota["medium"] - firsts["medium"])
    out.table[2]["planned_quota_loss_tokens"] = rest - quota["medium"]
    return out
