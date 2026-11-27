"""The revision-5 SWE draw (data plan 1.3, 3.1; execution plan 0.2): pure functions over the admitted
instances, one candidate per instance. scripts/data/build_swe_slice.py builds the candidates (bands, gates, rendering);
the first build's size-stratified draw (apex_data.teacher.filter.draw_cap) is parked, not used by this module.

Candidates. An instance is admitted when one of its resolved attempts passes every gate (row filters, the rendered-episode
gates, the profiler gates of its band and the band's row cap). Its candidate is the gate-passing attempt with the lowest
mean thinking per turn (`choose_attempt`; ties by trajectory id). Bands come from all of the instance's attempts
(efficiency.swe_band).

The draw, in layers:
  1. Bands by loss tokens, drawn hard first. Two modes:
     - shares of a total (plan 3.1): hard 20 / medium 50 / easy 30 percent of `total_loss`. A band stops at the first
       pick that reaches its target; the difference to the target (a shortfall, or the small overshoot of the last
       episode) moves to the next easier band (the supply rule of plan 1.3). An easy shortfall is reported.
     - keep-all: the bands of `keep_all` (hard, medium) take every admissible candidate; each remaining band b then
       gets band_shares[b] x total, where the total solves
       total = kept loss tokens + sum of those targets (easy at 45 percent of the SWE loss tokens).
  2. Languages within a band, in episodes: Go 36 / Python 34 / TypeScript 18 / JavaScript 6 / other (Rust, Java, PHP) 6.
     Each pick goes to the language group with the largest deficit (share x (picks + 1) - its picks; ties in GROUPS
     order). A group that runs out of admissible candidates gives its share to Go, else Python, else to the remaining
     groups in proportion.
  3. Within a (band, language group) cell, the candidate order (`cell_tiers`): the lower half by thinking_mean (ceil(n/2))
     in a seeded random order, then the upper half in a seeded random order (plan 1.3: the lower half is a preference,
     not a limit). Switch `lower_half` off: one seeded random order.
  4. Recovery preference: while the cell's recovery share would stay below RECOVERY_MIN after this pick
     (recovery picks < RECOVERY_MIN x (cell picks + 1)), the pick is the first recovery candidate of the first tier that
     has one; with none left, the order of 3 applies and the pick is counted as a recovery supply limit.
  5. Category round-robin within the cell: among the candidates of the first tier that has any (after 4), the category
     with the fewest picks in the cell, skipping a category that already holds CATEGORY_CAP of the band's picks while
     another is available; ties by the earliest candidate in the tier order.
  6. Diversity caps across the slice: at most MAX_PER_REPO instances per repository (`repo_caps` per language group
     overrides it: Go and TypeScript 5) and one per (repository, PR).
  7. Second attempts (`max_second_share` > 0): an instance's attempt-2 candidate is admissible once its attempt 1 is
     drawn and while second attempts stay at most `max_second_share` of all episodes drawn so far (counting this pick).
     It counts as an episode for the language deficits, the recovery share and the category round-robin, and not as a
     new instance for the repository and PR caps.

Switches (`DrawSwitches`) follow the must-have ladder (execution plan 0.2): the nice-to-haves are dropped in the order
category round-robin, recovery preference, lower half. Seeded: the only randomness is the seeded order of 3.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

BANDS = ("hard", "medium", "easy")                               # draw order
BAND_LOSS_SHARES = {"hard": 0.20, "medium": 0.50, "easy": 0.30}
TARGET_LOSS_TOKENS = 9_000_000
LANGUAGE_CODES = {"go": "go", "python": "py", "ts": "ts", "js": "js", "rust": "rust", "java": "java", "php": "php"}
GROUP_OF = {"go": "go", "py": "py", "ts": "ts", "js": "js", "rust": "other", "java": "other", "php": "other"}
GROUPS = ("go", "py", "ts", "js", "other")
LANGUAGE_SHARES = {"go": 0.36, "py": 0.34, "ts": 0.18, "js": 0.06, "other": 0.06}
GAP_ORDER = ("go", "py")                                         # a short language's share goes to Go, then Python
CATEGORIES = ("bug-fix", "feature-request", "enhancement", "other")
CATEGORY_CAP = 0.50
RECOVERY_MIN = 0.20
MAX_PER_REPO = 3
LOSS_CAP_PER_ROW = 32_768                                        # plan 3.4: at most 32K loss tokens per row


@dataclass(frozen=True)
class Candidate:
    """An admitted instance, represented by its chosen attempt (attempt 1), or the instance's next gate-passing attempt
    by thinking_mean (attempt 2: an alternative trajectory of the same task, drawn only after attempt 1)."""

    subsource: str
    instance_id: str
    trajectory_id: str
    band: str                    # BANDS
    language: str                # LANGUAGE_CODES value
    category: str
    repo: str                    # normalised owner/name
    pr: str | None
    loss_tokens: int             # n_assistant_tokens of the rendered row
    seq_tokens: int              # n_tokens of the rendered row
    thinking_mean: float         # reasoning tokens per assistant turn, mean
    recovery: bool               # recovery arc B (efficiency.recovery_arc)
    attempt: int = 1             # 1: the lowest-thinking gate-passing attempt; 2: the next one
    priority: int = 0            # without the lower-half preference, a cell draws higher priorities first (v7: a long tail turn)

    @property
    def key(self) -> str:
        """The instance (task) key."""
        return f"{self.subsource}:{self.instance_id}"

    @property
    def unit_key(self) -> str:
        return f"{self.key}#{self.attempt}"

    @property
    def group(self) -> str:
        return GROUP_OF[self.language]

    @property
    def lineage(self) -> tuple[str, str] | None:
        return (self.repo, self.pr) if self.pr else None


@dataclass(frozen=True)
class DrawSwitches:
    category_round_robin: bool = True
    recovery_preference: bool = True
    lower_half: bool = True

    def dropped(self) -> list[str]:
        return [k for k, v in asdict(self).items() if not v]


def choose_attempt(attempts: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The gate-passing attempt with the lowest thinking_mean (ties by trajectory_id); None when no attempt passes.
    Each attempt is a mapping with `ok`, `thinking_mean` and `trajectory_id`."""
    ok = [a for a in attempts if a["ok"]]
    return min(ok, key=lambda a: (a["thinking_mean"], a["trajectory_id"])) if ok else None


def _seeded(cands: Sequence[Candidate], seed: int, *labels: str) -> list[Candidate]:
    out = sorted(cands, key=lambda c: c.unit_key)
    random.Random(":".join([str(seed), *labels])).shuffle(out)
    return out


def cell_tiers(cands: Sequence[Candidate], seed: int, band: str, group: str, *, lower_half: bool = True) -> list[list[Candidate]]:
    """The cell's candidates as ordered tiers: [lower half, upper half] by thinking_mean (ceil(n/2) in the lower half;
    ties at the boundary by key), each in a seeded random order; one seeded tier when `lower_half` is off."""
    if not lower_half:                               # one seeded tier per priority, highest first (all 0: one tier)
        levels = sorted({c.priority for c in cands}, reverse=True) or [0]
        return [_seeded([c for c in cands if c.priority == lv], seed, band, group, "all" if lv == 0 else f"priority{lv}") for lv in levels]
    ranked = sorted(cands, key=lambda c: (c.thinking_mean, c.unit_key))
    k = math.ceil(len(ranked) / 2)
    return [_seeded(ranked[:k], seed, band, group, "lower"), _seeded(ranked[k:], seed, band, group, "upper")]


@dataclass
class Pick:
    candidate: Candidate
    order: int                   # position in the whole draw
    tier: int                    # 0 lower half, 1 upper half (0 when lower_half is off)
    recovery_preferred: bool     # the pick was made under the recovery preference


@dataclass
class BandResult:
    band: str
    target: float                # loss tokens, including the carry from the harder band
    carry_in: float
    loss_tokens: int = 0
    seq_tokens: int = 0
    picks: list[Pick] = field(default_factory=list)
    exhausted: list[str] = field(default_factory=list)          # groups that ran out, in order
    recovery_short: int = 0                                     # picks made without a recovery candidate left while wanted
    share_moves: list[dict[str, Any]] = field(default_factory=list)
    supply_exhausted: bool = False

    @property
    def shortfall(self) -> float:
        return max(0.0, self.target - self.loss_tokens)


@dataclass
class DrawResult:
    total_loss: int
    switches: DrawSwitches
    seed: int
    bands: dict[str, BandResult]
    skipped: Counter                                             # repo_cap / lineage_duplicate per walk step
    keep_all: tuple[str, ...] = ()
    repo_caps: dict[str, int] = field(default_factory=dict)
    max_second_share: float = 0.0

    @property
    def picks(self) -> list[Pick]:
        return [p for b in BANDS for p in self.bands[b].picks]


class _Cell:
    """Remaining candidates of one (band, group) cell and its running counts."""

    def __init__(self, tiers: list[list[Candidate]]) -> None:
        self.tiers = tiers
        self.n = 0
        self.recovery = 0
        self.recovery_short = 0          # picks that wanted a recovery candidate when none was left (the supply limit)
        self.categories: Counter = Counter()


def draw(candidates: Iterable[Candidate], *, total_loss: int = TARGET_LOSS_TOKENS, seed: int, switches: DrawSwitches = DrawSwitches(),
         band_shares: Mapping[str, float] = BAND_LOSS_SHARES, language_shares: Mapping[str, float] = LANGUAGE_SHARES,
         max_per_repo: int = MAX_PER_REPO, repo_caps: Mapping[str, int] | None = None,
         keep_all: Sequence[str] = (), max_second_share: float = 0.0, recovery_min: float = RECOVERY_MIN) -> DrawResult:
    """The whole SWE draw (module docstring). `candidates` holds one candidate per instance. With `keep_all`, those bands
    take every admissible candidate and `total_loss` is unused; the other bands' `band_shares` are shares of the total."""
    cands = list(candidates)
    keys = [c.unit_key for c in cands]
    if len(set(keys)) != len(keys):
        raise ValueError("one candidate per instance and attempt")
    firsts = {c.key for c in cands if c.attempt == 1}
    if any(c.attempt not in (1, 2) or (c.attempt == 2 and c.key not in firsts) for c in cands):
        raise ValueError("attempts are 1 or 2, and an attempt 2 needs its instance's attempt 1")
    if max_second_share <= 0:                      # no second attempts: they take no part in the cell tiers either
        cands = [c for c in cands if c.attempt == 1]
    unknown = {c.band for c in cands} - set(BANDS)
    if unknown:
        raise ValueError(f"candidates with bands outside {BANDS}: {sorted(unknown)}")
    keep_all = tuple(keep_all)
    if keep_all and tuple(BANDS[:len(keep_all)]) != keep_all:
        raise ValueError(f"keep_all must be the leading bands of {BANDS}, got {keep_all}")
    rest_share = sum(band_shares[b] for b in BANDS if b not in keep_all)
    if keep_all and not 0 <= rest_share < 1:
        raise ValueError(f"the shares of the bands after {keep_all} must sum below 1, got {rest_share}")
    caps = dict(repo_caps or {})

    def cap_of(c: Candidate) -> int:
        return caps.get(c.group, max_per_repo)
    by_cell: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        by_cell[(c.band, c.group)].append(c)
    repo_counts: Counter = Counter()
    lineage: set[tuple[str, str]] = set()
    skipped: Counter = Counter()
    results: dict[str, BandResult] = {}
    picked: set[str] = set()                        # instances whose attempt 1 is drawn
    dead: set[str] = set()                          # instances whose attempt 1 the caps excluded
    carry = 0.0
    order = 0
    n_second = 0

    def capped(c: Candidate) -> str | None:
        """Why an attempt 1 can never be drawn any more (repo_cap, lineage_duplicate), else None."""
        if repo_counts[c.repo] >= cap_of(c):
            return "repo_cap"
        return "lineage_duplicate" if c.lineage is not None and c.lineage in lineage else None

    def admissible(c: Candidate) -> bool:
        if c.attempt == 1:
            return True                              # prune() removed the capped ones
        return c.key in picked and n_second + 1 <= max_second_share * (order + 1)

    def prune(cell: _Cell) -> None:
        """Drop candidates that can never be drawn: capped attempts 1 (counted once each) and their attempts 2."""
        for t, tier in enumerate(cell.tiers):
            kept = []
            for c in tier:
                why = capped(c) if c.attempt == 1 else None
                if why:
                    skipped[why] += 1
                    dead.add(c.key)
                elif c.attempt == 2 and (c.key in dead or max_second_share <= 0):
                    continue
                else:
                    kept.append(c)
            cell.tiers[t] = kept

    def next_pick(cell: _Cell, band_cats: Counter, band_n: int) -> tuple[Candidate, int, bool] | None:
        prune(cell)
        tiers = [(t, avail) for t, tier in enumerate(cell.tiers) if (avail := [c for c in tier if admissible(c)])]
        if not tiers:
            return None
        want_rec = switches.recovery_preference and cell.recovery < recovery_min * (cell.n + 1)
        pool: list[Candidate] | None = None
        t_pick, rec_pref = tiers[0][0], False
        if want_rec:
            for t, tier in tiers:
                rec = [c for c in tier if c.recovery]
                if rec:
                    pool, t_pick, rec_pref = rec, t, True
                    break
            else:
                cell.recovery_short += 1
        if pool is None:
            pool = tiers[0][1]
        if switches.category_round_robin:
            present = list(dict.fromkeys(c.category for c in pool))
            capped = [k for k in present if (band_cats[k] + 1) / (band_n + 1) > CATEGORY_CAP]
            allowed = [k for k in present if k not in capped] or present
            low = min(cell.categories[k] for k in allowed)
            chosen = next(c for c in pool if c.category in allowed and cell.categories[c.category] == low)
        else:
            chosen = pool[0]
        return chosen, t_pick, rec_pref

    for band in BANDS:
        if band in keep_all:
            target = math.inf
        elif keep_all:
            kept = sum(results[b].loss_tokens for b in keep_all)
            target = band_shares[band] * kept / (1 - rest_share)
        else:
            target = band_shares[band] * total_loss + carry
        res = results[band] = BandResult(band, target, carry)
        shares = {g: float(language_shares.get(g, 0.0)) for g in GROUPS}
        cells = {g: _Cell(cell_tiers(by_cell.get((band, g), []), seed, band, g, lower_half=switches.lower_half)) for g in GROUPS}
        counts: Counter = Counter()
        band_cats: Counter = Counter()
        live = [g for g in GROUPS if shares[g] > 0]
        while res.loss_tokens < res.target and live:
            n = len(res.picks)
            g = max(live, key=lambda x: (shares[x] * (n + 1) - counts[x], -GROUPS.index(x)))
            got = next_pick(cells[g], band_cats, n)
            if got is None:
                live.remove(g)
                res.exhausted.append(g)
                heir = next((h for h in GAP_ORDER if h in live), None)
                moved = shares[g]
                if heir is not None:
                    shares[heir] += moved
                    res.share_moves.append({"from": g, "to": heir, "share": round(moved, 4), "after_picks": n})
                elif live:
                    rest = sum(shares[h] for h in live)
                    for h in live:
                        shares[h] += moved * (shares[h] / rest if rest else 1 / len(live))
                    res.share_moves.append({"from": g, "to": "remaining", "share": round(moved, 4), "after_picks": n})
                shares[g] = 0.0
                continue
            c, tier, rec_pref = got
            for t in cells[g].tiers:
                if c in t:
                    t.remove(c)
                    break
            cell = cells[g]
            cell.n += 1
            cell.recovery += c.recovery
            cell.categories[c.category] += 1
            band_cats[c.category] += 1
            counts[g] += 1
            if c.attempt == 1:
                picked.add(c.key)
                repo_counts[c.repo] += 1
                if c.lineage is not None:
                    lineage.add(c.lineage)
            else:
                n_second += 1
            res.picks.append(Pick(c, order, tier, rec_pref))
            res.loss_tokens += c.loss_tokens
            res.seq_tokens += c.seq_tokens
            order += 1
        res.supply_exhausted = res.loss_tokens < res.target
        res.recovery_short = sum(c.recovery_short for c in cells.values())
        carry = 0.0 if keep_all else res.target - res.loss_tokens     # keep-all: kept bands have no target to carry
    return DrawResult(total_loss, switches, seed, results, skipped, keep_all, caps, max_second_share)


# ------------------------------------------------------------------------------------------------------ reporting
def share(n: float, d: float) -> float:
    return round(n / d, 4) if d else 0.0


def supply_table(candidates: Iterable[Candidate]) -> dict[str, Any]:
    """Admitted supply per (language group, band) and per (language, band): instances, loss tokens, recovery and
    category counts, thinking_mean median."""
    cells: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    langs: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        cells[(c.group, c.band)].append(c)
        langs[(c.language, c.band)].append(c)

    def summary(cs: list[Candidate]) -> dict[str, Any]:
        tm = sorted(c.thinking_mean for c in cs)
        return {"instances": len(cs), "loss_tokens": sum(c.loss_tokens for c in cs), "seq_tokens": sum(c.seq_tokens for c in cs),
                "recovery": sum(c.recovery for c in cs), "repositories": len({c.repo for c in cs}),
                "categories": dict(Counter(c.category for c in cs)),
                "thinking_mean_median": round(tm[len(tm) // 2], 1) if tm else None}

    return {"by_group_band": {f"{g}/{b}": summary(cells[(g, b)]) for g in GROUPS for b in BANDS},
            "by_language_band": {f"{lang}/{b}": summary(cs) for (lang, b), cs in sorted(langs.items())},
            "by_band": {b: summary([c for (g, bb), cs in cells.items() if bb == b for c in cs]) for b in BANDS}}


def draw_report(result: DrawResult) -> dict[str, Any]:
    """Per band: target, achieved loss tokens, shortfall, episodes, language / category / recovery shares; overall
    shares by band (loss tokens), language group and category (episodes), recovery (episodes)."""
    picks = result.picks
    total = sum(p.candidate.loss_tokens for p in picks)
    bands = {}
    for b, r in result.bands.items():
        ps = [p.candidate for p in r.picks]
        n = len(ps)
        finite = math.isfinite(r.target)
        bands[b] = {"target_loss_tokens": round(r.target) if finite else None, "carry_in": round(r.carry_in), "loss_tokens": r.loss_tokens,
                    "seq_tokens": r.seq_tokens, "episodes": n, "instances": len({c.key for c in ps}),
                    "second_attempts": sum(c.attempt == 2 for c in ps), "repositories": len({c.repo for c in ps}),
                    "shortfall": round(r.shortfall) if finite else 0,
                    "supply_exhausted": r.supply_exhausted, "keep_all": b in result.keep_all,
                    "groups": {g: sum(c.group == g for c in ps) for g in GROUPS},
                    "group_shares": {g: share(sum(c.group == g for c in ps), n) for g in GROUPS},
                    "categories": dict(Counter(c.category for c in ps)),
                    "max_category_share": share(max(Counter(c.category for c in ps).values(), default=0), n),
                    "recovery": sum(c.recovery for c in ps), "recovery_share": share(sum(c.recovery for c in ps), n),
                    "recovery_preferred_picks": sum(p.recovery_preferred for p in r.picks),
                    "recovery_supply_short_picks": r.recovery_short,
                    "upper_half_picks": sum(p.tier == 1 for p in r.picks),
                    "exhausted_groups": r.exhausted, "share_moves": r.share_moves}
    n = len(picks)
    return {"mode": "keep_all" if result.keep_all else "shares", "keep_all": list(result.keep_all), "repo_caps": result.repo_caps,
            "total_target": None if result.keep_all else result.total_loss, "loss_tokens": total, "seq_tokens": sum(p.candidate.seq_tokens for p in picks),
            "episodes": n, "instances": len({p.candidate.key for p in picks}),
            "second_attempts": sum(p.candidate.attempt == 2 for p in picks),
            "second_attempt_share": share(sum(p.candidate.attempt == 2 for p in picks), n),
            "max_second_share": result.max_second_share, "seed": result.seed, "switches": asdict(result.switches),
            "dropped_nice_to_haves": result.switches.dropped(),
            "band_loss_shares": {b: share(result.bands[b].loss_tokens, total) for b in BANDS},
            "group_shares": {g: share(sum(p.candidate.group == g for p in picks), n) for g in GROUPS},
            "language_counts": dict(Counter(p.candidate.language for p in picks)),
            "category_shares": {k: share(sum(p.candidate.category == k for p in picks), n) for k in CATEGORIES},
            "recovery_share": share(sum(p.candidate.recovery for p in picks), n),
            "max_rows_per_repository": max(Counter(p.candidate.repo for p in picks).values(), default=0),
            "max_instances_per_repository": max(Counter(p.candidate.repo for p in picks if p.candidate.attempt == 1).values(),
                                                default=0),
            "repositories": len({p.candidate.repo for p in picks}),
            "skipped": dict(result.skipped), "bands": bands}
