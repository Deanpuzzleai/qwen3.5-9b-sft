"""Seeded stratified draws of the retention slice (docs/DATA_PLAN.md sections 3.4 and 11).

Strata per part:
  math     reasoning-length bands <= 4K / 4-7K / 7-10K tokens (MATH_BAND_SHARES = 55 / 25 / 20 percent, the plan's
           shares; the natural shares of the eligible pool are measured and reported next to them)
  chat     classifier topic group: CHAT_HSS_SHARE of the rows from the humanities / social science / knowledge group
           (the plan's floor is 40 percent; the draw aims at 50 percent because the classifier's hss precision on the
           hand-labelled sample is about 0.8, so the true share stays above the floor)
  sci_mcq  option count: rows with >= 5 options first, rows with 4 options only if the first stratum runs short
  if, sci_rqa  one stratum
Each part is drawn twice from the same shuffled survivors: the SFT rows first, then the replay reserve from the rows
left over, so the two sets share no source row; the dedup screen keeps one row per duplicate cluster before the draw,
so they share no dedup group either. Every shuffle is `random.Random(f"{seed}:{part}:{stratum}")` over rows sorted by
task id, so a rebuild draws the same rows.

Top-up (`draw_topup`, `interleave`): extra SFT rows that follow the SFT rows and the reserve without changing them. Per
stratum the candidate order is the survivors after the SFT and reserve quotas in that same shuffle (the next survivors),
then rows measured beyond the original oversample in their own shuffle (`f"{seed}:{part}:{stratum}:topup"`); quotas
follow the same stratum rules. `interleave` then orders the top-up across (part, stratum) cells by the Webster divisor
sequence, so every prefix of the emitted order holds each cell's share up to rounding (`prefix_deviation` measures it).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from fractions import Fraction
from typing import Any

from apex_data.retention.common import apportion

MATH_BANDS = (("le4k", 4096), ("4k_7k", 7168), ("7k_10k", 10240))
MATH_BAND_SHARES = {"le4k": 0.55, "4k_7k": 0.25, "7k_10k": 0.20}
CHAT_HSS_SHARE = 0.50
CHAT_HSS_FLOOR = 0.40
MCQ_PREFERENCE = ("ge5", "lt5")


def math_band(reasoning_tokens: int) -> str | None:
    for name, upper in MATH_BANDS:
        if reasoning_tokens <= upper:
            return name
    return None


def stratum_of(part: str, c: dict[str, Any]) -> str:
    if part == "math":
        return c["meta"]["length_band"]
    if part == "chat":
        return c["meta"]["topic_group"]
    if part == "sci_mcq":
        return "ge5" if c["meta"]["n_options"] >= 5 else "lt5"
    return "all"


def stratum_shares(part: str) -> dict[str, float] | None:
    """Fixed shares per stratum, or None for parts drawn by preference order (MCQ) or without strata."""
    if part == "math":
        return dict(MATH_BAND_SHARES)
    if part == "chat":
        return {"hss": CHAT_HSS_SHARE, "other": 1 - CHAT_HSS_SHARE}
    return None


def quotas(part: str, n: int, supply: dict[str, int] | None = None) -> dict[str, int]:
    """Rows per stratum for a draw of n rows (by fixed shares, or by preference order against `supply`)."""
    shares = stratum_shares(part)
    if shares is not None:
        return apportion(n, shares)
    if part == "sci_mcq":
        supply = supply or {}
        first = min(n, supply.get("ge5", 0))
        return {"ge5": first, "lt5": n - first}
    return {"all": n}


def shuffled(rows: Sequence[dict[str, Any]], seed: int, *tags: str) -> list[dict[str, Any]]:
    out = sorted(rows, key=lambda r: r["task_id"])
    random.Random(":".join([str(seed), *tags])).shuffle(out)
    return out


def draw_sft_and_reserve(part: str, survivors: Sequence[dict[str, Any]], n_sft: int, n_reserve: int, seed: int
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(SFT rows, reserve rows, report). Raises ValueError when a stratum cannot fill its SFT quota; a reserve
    shortfall is reported, not raised."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in survivors:
        by[stratum_of(part, r)].append(r)
    supply = {s: len(v) for s, v in by.items()}
    q_sft = quotas(part, n_sft, supply)
    left_supply = {s: supply.get(s, 0) - q_sft.get(s, 0) for s in supply}
    q_res = quotas(part, n_reserve, left_supply)
    sft: list[dict[str, Any]] = []
    res: list[dict[str, Any]] = []
    short: dict[str, int] = {}
    for s in sorted(set(q_sft) | set(q_res)):
        pool = shuffled(by.get(s, []), seed, part, s)
        if len(pool) < q_sft.get(s, 0):
            raise ValueError(f"{part}: stratum {s} has {len(pool)} surviving rows for an SFT quota of {q_sft[s]}")
        sft += pool[:q_sft.get(s, 0)]
        rest = pool[q_sft.get(s, 0):]
        take = rest[:q_res.get(s, 0)]
        if len(take) < q_res.get(s, 0):
            short[s] = q_res[s] - len(take)
        res += take
    groups = Counter(r["screens"].get("dedup_group") or r["task_id"] for r in sft + res)
    if any(v > 1 for v in groups.values()):
        raise ValueError(f"{part}: a dedup group occurs more than once among the drawn rows")
    report = {"supply": dict(sorted(supply.items())), "sft_quota": q_sft, "reserve_quota": q_res,
              "reserve_shortfall": short, "sft_strata": dict(Counter(stratum_of(part, r) for r in sft)),
              "reserve_strata": dict(Counter(stratum_of(part, r) for r in res))}
    return sft, res, report


def draw_topup(part: str, survivors: Sequence[dict[str, Any]], n_sft: int, n_reserve: int,
               new_survivors: Sequence[dict[str, Any]], n_topup: int, seed: int, exclude: Collection[str] = ()
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(top-up rows, report): n_topup rows that come after the SFT rows and the reserve rows that
    `draw_sft_and_reserve(part, survivors, n_sft, n_reserve, seed)` draws, which stay unchanged.

    Per stratum the candidate order is the survivors left after the SFT and reserve quotas in that draw's shuffle (the
    next survivors), followed by `new_survivors` (rows measured beyond the original oversample, screened and gated the
    same way) in their own shuffle `random.Random(f"{seed}:{part}:{stratum}:topup")`; task ids in `exclude` are skipped.
    The quotas are `quotas(part, n_topup, supply)` over that order (math bands and chat topic groups by fixed shares,
    MCQ by option-count preference). The rows are returned stratum by stratum, each in its candidate order. Raises
    ValueError when a stratum cannot fill its quota, when a new survivor is also an original survivor, or when a dedup
    group occurs twice among the top-up rows or in both the top-up and the SFT or reserve rows."""
    old_ids = {r["task_id"] for r in survivors}
    if any(r["task_id"] in old_ids for r in new_survivors):
        raise ValueError(f"{part}: a new survivor is also an original survivor")
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in survivors:
        by[stratum_of(part, r)].append(r)
    new_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in new_survivors:
        new_by[stratum_of(part, r)].append(r)
    supply = {s: len(v) for s, v in by.items()}
    q_sft = quotas(part, n_sft, supply)
    q_res = quotas(part, n_reserve, {s: supply.get(s, 0) - q_sft.get(s, 0) for s in supply})
    taken: list[dict[str, Any]] = []
    order: dict[str, list[dict[str, Any]]] = {}
    left: dict[str, int] = {}
    n_excluded = 0
    for s in sorted(set(by) | set(new_by) | set(q_sft) | set(q_res)):
        pool = shuffled(by.get(s, []), seed, part, s)
        cut = q_sft.get(s, 0) + q_res.get(s, 0)
        taken += pool[:cut]
        cands = pool[cut:] + shuffled(new_by.get(s, []), seed, part, s, "topup")
        order[s] = [r for r in cands if r["task_id"] not in exclude]
        n_excluded += len(cands) - len(order[s])
        left[s] = sum(r["task_id"] in old_ids for r in order[s])
    q_top = quotas(part, n_topup, {s: len(v) for s, v in order.items()})
    rows: list[dict[str, Any]] = []
    for s in sorted(q_top):
        if len(order.get(s, [])) < q_top[s]:
            raise ValueError(f"{part}: stratum {s} has {len(order.get(s, []))} top-up candidates for a quota of {q_top[s]}")
        rows += order[s][:q_top[s]]
    def group(r: dict[str, Any]) -> str:
        return r["screens"].get("dedup_group") or r["task_id"]

    top_groups = Counter(group(r) for r in rows)
    if any(v > 1 for v in top_groups.values()) or set(top_groups) & {group(r) for r in taken}:
        raise ValueError(f"{part}: a dedup group occurs twice among the top-up rows or also among the SFT / reserve rows")
    report = {"quota": q_top, "strata": dict(Counter(stratum_of(part, r) for r in rows)),
              "supply_next_survivors": {s: v for s, v in sorted(left.items()) if v},
              "supply_new": {s: len(v) for s, v in sorted(new_by.items())}, "n_excluded": n_excluded,
              "from_next_survivors": dict(Counter(stratum_of(part, r) for r in rows if r["task_id"] in old_ids)),
              "from_new": dict(Counter(stratum_of(part, r) for r in rows if r["task_id"] not in old_ids))}
    return rows, report


def interleave(cells: Mapping[str, Sequence[Any]], seed: int, tag: str = "topup_interleave") -> list[tuple[str, Any]]:
    """[(cell, item)] in a proportional order: the j-th item (0-based) of a cell of n items gets the key (2j + 1) / 2n
    (the Webster / Sainte-Lague divisor sequence, exact fractions) and items are emitted by ascending key, ties broken by
    a seeded rank of the cell (`random.Random(f"{seed}:{tag}:{cell}")`). Each cell keeps its own order, so the first k
    items hold a prefix of every cell's order with about k x n_cell / N items of it (`prefix_deviation` measures the
    worst case). Empty cells are skipped."""
    keyed = []
    for cell, items in cells.items():
        n = len(items)
        tie = random.Random(f"{seed}:{tag}:{cell}").random()
        keyed += [(Fraction(2 * j + 1, 2 * n), tie, cell, j, item) for j, item in enumerate(items)]
    keyed.sort(key=lambda x: x[:4])
    return [(cell, item) for _, _, cell, _, item in keyed]


def prefix_deviation(labels: Sequence[str]) -> dict[str, float]:
    """{label: max over prefixes k of |count of label in labels[:k] - k x its overall share|}."""
    total = Counter(labels)
    n = len(labels)
    seen: Counter = Counter()
    worst = {lab: 0.0 for lab in total}
    for k, lab in enumerate(labels, start=1):
        seen[lab] += 1
        for c, m in total.items():
            worst[c] = max(worst[c], abs(seen[c] - k * m / n))
    return {c: round(v, 4) for c, v in sorted(worst.items())}
