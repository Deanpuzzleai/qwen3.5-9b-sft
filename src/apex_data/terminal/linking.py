"""TerminalHorizon episode -> environment linking by instruction word 6-gram shingles (docs/DATA_PLAN.md 3.3).

TerminalHorizon-3K ships no environment id; `shunzou05/TerminalHorizon-Environment` `metadata/tasks.parquet` gives each of
its 1,535 environments a `task_id`, `category`, `subcategory` and `instruction`. An episode's instruction (the text of its
first prompt between "Task Description:" and the initial terminal state) is linked to the environment whose instruction
shares the largest fraction of its word 6-grams (lowercase `[a-z0-9_]+` words, the representation of
`logs/data_plan_rev4/probes/terminal_recheck/link_probe.py`). A link needs `score >= MIN_SCORE` (shared shingles over
the environment's shingles) and a strictly better score than the runner-up; anything else stays unlinked and is not
eligible for the category quota (counted).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

SHINGLE_N = 6
MIN_SCORE = 0.5
_WORD = re.compile(r"[a-z0-9_]+")


def words(text: str | None) -> list[str]:
    return _WORD.findall((text or "").lower())


def shingles(text: str | None, n: int = SHINGLE_N) -> set[str]:
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


@dataclass(frozen=True)
class Link:
    env: str | None          # environment task_id, None when unlinked
    score: float             # shared shingles / environment shingles of the best match
    score_episode: float     # shared shingles / episode shingles of the best match
    runner_up: float         # second-best score
    reason: str              # linked | no_shingles | no_overlap | below_threshold | tie


class EnvIndex:
    """Inverted shingle index over the environment instructions."""

    def __init__(self, envs: Sequence[dict[str, Any]], n: int = SHINGLE_N) -> None:
        self.n = n
        self.envs = {e["task_id"]: e for e in envs}
        if len(self.envs) != len(envs):
            raise ValueError("duplicate environment task_id in the metadata")
        self.ids = [e["task_id"] for e in envs]
        self.sizes = []
        self.inverted: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(envs):
            sh = shingles(e.get("instruction"), n)
            self.sizes.append(len(sh))
            for g in sh:
                self.inverted[g].append(i)

    def link(self, instruction: str | None, min_score: float = MIN_SCORE) -> Link:
        sh = shingles(instruction, self.n)
        if not sh:
            return Link(None, 0.0, 0.0, 0.0, "no_shingles")
        shared: Counter = Counter()
        for g in sh:
            for i in self.inverted.get(g, ()):
                shared[i] += 1
        if not shared:
            return Link(None, 0.0, 0.0, 0.0, "no_overlap")
        scored = sorted(((c / max(1, self.sizes[i]), c, i) for i, c in shared.items()), reverse=True)
        best, count, i = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        ep_score = count / len(sh)
        if best < min_score:
            return Link(None, round(best, 4), round(ep_score, 4), round(runner, 4), "below_threshold")
        if runner >= best:
            return Link(None, round(best, 4), round(ep_score, 4), round(runner, 4), "tie")
        return Link(self.ids[i], round(best, 4), round(ep_score, 4), round(runner, 4), "linked")

    def meta(self, env: str | None) -> dict[str, Any]:
        e = self.envs.get(env or "") or {}
        return {"category": e.get("category"), "subcategory": e.get("subcategory")}


def link_summary(links: Iterable[Link]) -> dict[str, Any]:
    """Counts for the manifest: reasons, per-environment multiplicity, score quantiles of the linked episodes."""
    links = list(links)
    reasons = Counter(l.reason for l in links)
    per_env = Counter(l.env for l in links if l.env)
    scores = sorted(l.score for l in links if l.env)

    def q(p: float) -> float | None:
        return scores[int(p * (len(scores) - 1))] if scores else None

    return {"n_episodes": len(links), "reasons": dict(sorted(reasons.items())), "linked": reasons.get("linked", 0),
            "distinct_environments": len(per_env), "max_episodes_per_environment": max(per_env.values(), default=0),
            "environments_with_2plus": sum(v >= 2 for v in per_env.values()),
            "score_quantiles": {"p0": q(0), "p1": q(0.01), "p10": q(0.1), "p50": q(0.5)},
            "runner_up_max": max((l.runner_up for l in links if l.env), default=None)}
