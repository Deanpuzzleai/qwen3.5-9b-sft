"""Row filters, exclusions and the size-stratified draw of the SWE teacher slice (data plan 3.2 filters 1 to 8, section 8).

Cheap filters (`row_filters`, every trajectory of the index, in plan order; each step counts n_in / n_dropped / n_out in
total and per subsource x language, and every dropped trajectory keeps its reason):
  1. resolved == 1; language in {go, py, js, ts}; repository licence on the plan-7 allowlist; Scale-SWE rows only on
     instances listed in PrimeIntellect/Scale-SWE-Verified;
  2. D2 (apex_data.screens.d2_repos: the SWE-bench Pro repositories, predecessors, basenames, identifiers, the GitHub fork
     graph) and the repositories of data/manifests/decontam_exclusions.json;
  3. tool definition equal to our mini-swe-agent 2.4.6 `bash` schema; the model patch is non-empty and touches no test,
     CI or test-configuration path (apex_data.grade.tamper.classify_path, the graders' path classes);
  4. (index pre-check) non-empty reasoning on every assistant turn; the full filter-4 gates run on rendered episodes;
  6. dev exclusions by instance id and repository (data/manifests/dev_exclusions.json), applied here because they are cheap.
The rendered-episode filters (4: parse round trip, thrashing, reference-patch lines in the prompt; 5: length within the
cap) run in the build script through apex_data.teacher.render and apex_data.render.filters.

The draw (`draw_cap`, pure; run once per cap because the supply differs by cap):
  * an instance is admitted at cap C when one of its trajectories passed every gate and renders within C tokens; which
    of those represents it is the trajectory choice (`choose_trajectory`, `TRAJECTORY_CHOICES`), which never changes
    admission: `random_not_shortest` (the default) draws at random among them, never the shortest when two or more fit
    (the random key is a hash of seed and trajectory id, so both caps pick the same trajectory whenever it is eligible
    at both); `fewest_turns_verified` takes the one with the fewest assistant turns among those that ran a test after
    their last edit (apex_data.teacher.verify), ties by rendered tokens then trajectory id, and falls back to
    `random_not_shortest` when none did;
  * quotas Go 470 / TS 260 / JS 120 / Py 120 SWE-rebench-V2 rows: 65 percent of each quota from large instances first
    (big in seeded random order, then feature), the rest at random from all admitted instances; Py 200 Scale-SWE rows
    ranked by non-test patch size (big first, then feature, then by added lines);
  * at most 5 rows per repository and one row per (repository, PR) across the whole slice;
  * the walk is lazy: it stops at the first instance whose trajectories are not evaluated yet and returns the next
    instances to evaluate, so evaluating more never changes a completed draw;
  * pinned picks (rows kept from an earlier selection) count toward their part, phase, repository and PR first; the
    walks then fill only the slots left, in the same seeded orders. Under `random_not_shortest` a pin keeps its
    trajectory; under another choice it keeps its instance, part and phase and its trajectory is chosen again by the rule.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_data import paths
from apex_data.grade.tamper import classify_path
from apex_data.pools.common import repo_key
from apex_data.screens import d2_repos
from apex_data.screens.licence import ALLOWLIST
from apex_data.teacher.load import LANGUAGE_CODES, IndexRow, size_stratum
from apex_train.data_format import normalise_tools

MAX_PER_REPO = 5
BIG_SHARE = 0.65
FLAG_OF_CLASS = {"test": "tests_touched", "ci": "ci_touched", "test_config": "test_config_touched"}


@dataclass(frozen=True)
class PartSpec:
    """One quota cell of the draw."""

    name: str
    subsource: str
    language: str                 # schema code
    quota: int
    mode: str                     # stratified (65 % large first, 35 % random) | ranked (by non-test patch size)


PARTS = (PartSpec("go", "swe-rebench-v2", "go", 470, "stratified"),
         PartSpec("ts", "swe-rebench-v2", "ts", 260, "stratified"),
         PartSpec("js", "swe-rebench-v2", "js", 120, "stratified"),
         PartSpec("py_scale", "scale-swe", "py", 200, "ranked"),
         PartSpec("py_rebench", "swe-rebench-v2", "py", 120, "stratified"))


# ------------------------------------------------------------------------------------------- small screens
def tamper_flags(touched: Iterable[str]) -> dict[str, list[str]]:
    """{flag: [paths]} over the three path classes of classify_path; empty when clean."""
    flags: dict[str, list[str]] = defaultdict(list)
    for p in touched:
        for kind in sorted(classify_path(p)):
            flags[FLAG_OF_CLASS[kind]].append(p)
    return dict(flags)


def tool_schema_matches(tools: Any, expected: dict[str, Any]) -> tuple[bool, str]:
    """(ok, detail): exactly one tool whose normalised schema (name, description, parameters) equals `expected`."""
    try:
        if isinstance(tools, str):
            tools = json.loads(tools)
        parsed = [json.loads(t) if isinstance(t, str) else t for t in (tools or [])]
        got = normalise_tools(parsed, "tools") or []
        want = normalise_tools([expected], "expected") or []
    except Exception as exc:  # noqa: BLE001 - any malformed definition is a mismatch, reported not raised
        return False, f"unparseable tool definition: {type(exc).__name__}: {exc}"
    if len(got) != 1:
        return False, f"expected exactly one tool, got {len(got)}"
    for key in ("name", "parameters", "description"):
        if got[0]["function"].get(key) != want[0]["function"].get(key):
            return False, f"tool {key} differs"
    return True, "ok"


def normalise_repo(name: str | None) -> str:
    """`owner/name`, lower case; accepts GitHub URLs and SWE-smith style names (`owner__name`, `owner__name.<sha>`)."""
    s = (name or "").strip().lower()
    s = re.sub(r"^https?://github\.com/", "", s).removesuffix(".git").strip("/")
    tail = s.rsplit("/", 1)[-1]
    if "__" in tail and ("/" not in s or re.search(r"\.[0-9a-f]{6,}$", tail)):
        s = re.sub(r"\.[0-9a-f]{6,}$", "", tail).replace("__", "/", 1)
    return repo_key(s) or s


@dataclass
class DevExclusions:
    repos: set[str] = field(default_factory=set)
    instance_ids: set[str] = field(default_factory=set)
    status: str = "absent"                       # applied | absent
    source: str | None = None
    sha256: str | None = None


def load_dev_exclusions(path: Path | None = None) -> DevExclusions:
    """`swe_repositories` and `swe_task_ids` of data/manifests/dev_exclusions.json (WP6); status `absent` without it."""
    path = Path(path) if path else paths.DEV_EXCLUSIONS
    if not path.exists():
        return DevExclusions()
    raw = path.read_bytes()
    obj = json.loads(raw)
    ids = {str(x).split(":", 1)[-1] for x in obj.get("swe_task_ids") or []}
    return DevExclusions({normalise_repo(r) for r in obj.get("swe_repositories") or []}, ids, "applied",
                         str(path.relative_to(paths.REPO)) if path.is_relative_to(paths.REPO) else str(path),
                         hashlib.sha256(raw).hexdigest())


# ----------------------------------------------------------------------------------------------- the funnel
@dataclass
class Funnel:
    """Sequential survival table: per step n_in / n_dropped / n_out in total and per `subsource/language` cell."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)

    def step(self, name: str, before: Sequence[Any], after: Sequence[Any], cell: Callable[[Any], str], **extra: Any) -> None:
        cin, cout = Counter(map(cell, before)), Counter(map(cell, after))
        self.steps.append({"filter": name, "n_in": len(before), "n_dropped": len(before) - len(after), "n_out": len(after),
                           "by_cell": {k: {"n_in": cin[k], "n_dropped": cin[k] - cout[k], "n_out": cout[k]} for k in sorted(cin)},
                           **extra})


def cell_of(r: IndexRow) -> str:
    return f"{r.subsource}/{r.language_code or r.language}"


@dataclass
class Drop:
    subsource: str
    instance_id: str
    trajectory_id: str
    repo: str
    language: str
    stage: str
    reason: str
    detail: str = ""


def row_filters(index: Sequence[IndexRow], *, bash_tool: dict[str, Any], verified: set[str], locked: d2_repos.LockedRepos,
                fork_infos: dict[str, Any] | None, fork_lookup_enabled: bool, dev: DevExclusions,
                funnel: Funnel | None = None, languages: Iterable[str] = tuple(LANGUAGE_CODES)
                ) -> tuple[list[IndexRow], Funnel, list[Drop], dict[str, Any]]:
    """The cheap filters in plan order. Returns the survivors, the funnel, one Drop per dropped trajectory and the D2
    report (mechanism counts, fork-graph states, excluded-repository counts). `languages` are the dataset language values
    admitted by filter 1 (default the first build's four; the revision-5 slice adds rust, java and php)."""
    funnel = funnel or Funnel()
    drops: list[Drop] = []
    rows = list(index)
    funnel.step("0_rows_in_shards", rows, rows, cell_of, note="teacher identity, thinking mode and upstream dataset asserted per row")

    def keep(name: str, verdict: Callable[[IndexRow], tuple[str, str] | None]) -> None:
        nonlocal rows
        kept = []
        for r in rows:
            v = verdict(r)
            if v is None:
                kept.append(r)
            else:
                drops.append(Drop(r.subsource, r.instance_id, r.trajectory_id, r.repo, r.language, name, v[0], v[1]))
                funnel.reasons[v[0]] += 1
        funnel.step(name, rows, kept, cell_of)
        rows = kept

    keep("1_resolved", lambda r: None if r.resolved == 1 else ("not_resolved", str(r.resolved)))
    admitted_languages = set(languages)
    keep("1_language", lambda r: None if r.language in admitted_languages else ("language_out_of_quota", r.language))
    keep("1_licence", lambda r: None if r.license in ALLOWLIST else ("licence_not_allowlisted", r.license))
    keep("1_scale_swe_verified", lambda r: None if r.subsource != "scale-swe" or r.instance_id in verified
         else ("scale_swe_not_verified", "not in PrimeIntellect/Scale-SWE-Verified"))
    marks: dict[tuple[str, str], d2_repos.D2Result] = {}
    for r in rows:
        if (r.repo, r.instance_id) not in marks:
            res = d2_repos.screen(r.repo, locked=locked, upstream_id=r.instance_id)
            res.add_fork_graph((fork_infos or {}).get(repo_key(r.repo) or ""), locked, lookup_enabled=fork_lookup_enabled)
            marks[(r.repo, r.instance_id)] = res

    def d2(r: IndexRow) -> tuple[str, str] | None:
        m = marks[(r.repo, r.instance_id)]
        if m.locked_hit:
            return "d2_locked_repository", ",".join(m.mechanisms)
        if m.excluded_hit:
            return "decontam_excluded_repository", ",".join(m.mechanisms)
        return None

    before_d2 = list(rows)
    keep("2_d2_and_decontam_exclusions", d2)
    d2_report = d2_repos.counts(marks.values())
    d2_report["trajectories_dropped_locked"] = sum(1 for d in drops if d.reason == "d2_locked_repository")
    d2_report["trajectories_dropped_excluded"] = sum(1 for d in drops if d.reason == "decontam_excluded_repository")
    d2_report["excluded_repositories"] = {repo: {"trajectories": n, "instances": len({r.instance_id for r in before_d2
                                                                                     if normalise_repo(r.repo) == repo})}
                                          for repo, n in Counter(normalise_repo(r.repo) for r in before_d2
                                                                 if normalise_repo(r.repo) in locked.excluded).items()}
    schema_cache: dict[str, tuple[bool, str]] = {}

    def schema(r: IndexRow) -> tuple[str, str] | None:
        if r.tools_json not in schema_cache:
            schema_cache[r.tools_json] = tool_schema_matches(json.loads(r.tools_json), bash_tool)
        ok, detail = schema_cache[r.tools_json]
        return None if ok else ("tool_schema_mismatch", detail)

    keep("3_tool_schema", schema)

    def tamper(r: IndexRow) -> tuple[str, str] | None:
        if r.model_patch_empty:
            return "model_patch_empty", ""
        flags = tamper_flags(r.model_patch_paths)
        return ("model_patch_tamper", json.dumps(flags, sort_keys=True)[:300]) if flags else None

    keep("3_model_patch_clean", tamper)
    keep("4_reasoning_every_turn", lambda r: None if r.n_assistant > 0 and r.n_assistant_without_reasoning == 0
         else ("reasoning_missing", f"{r.n_assistant_without_reasoning} of {r.n_assistant} assistant turns"))
    keep("6_dev_exclusions", lambda r: ("dev_excluded_instance", r.instance_id) if r.instance_id in dev.instance_ids
         else (("dev_excluded_repository", r.repo) if normalise_repo(r.repo) in dev.repos else None))
    return rows, funnel, drops, d2_report


# -------------------------------------------------------------------------------------------- instances
@dataclass(frozen=True)
class Instance:
    subsource: str
    instance_id: str
    language: str                       # schema code
    repo: str                           # normalised owner/name (the per-repository cap)
    files: int                          # non-test files changed by the reference patch
    added: int                          # added lines of the reference patch without test files
    pr: str | None
    trajectories: tuple[str, ...]       # trajectory ids that passed the cheap filters

    @property
    def key(self) -> str:
        return f"{self.subsource}:{self.instance_id}"

    @property
    def stratum(self) -> str:
        return size_stratum(self.files, self.added)

    @property
    def lineage(self) -> tuple[str, str] | None:
        return (self.repo, self.pr) if self.pr else None


@dataclass(frozen=True)
class Outcome:
    """Gate outcome of one trajectory (cap independent: n_tokens decides the cap). `n_turns` (assistant turns of the
    rendered row) and `verified` (apex_data.teacher.verify) are read only by the `fewest_turns_verified` choice."""

    trajectory_id: str
    ok: bool
    n_tokens: int | None = None
    stage: str | None = None
    reason: str | None = None
    n_turns: int | None = None
    verified: bool | None = None


PENDING = object()
RANDOM_NOT_SHORTEST = "random_not_shortest"
FEWEST_TURNS_VERIFIED = "fewest_turns_verified"
TRAJECTORY_CHOICES = {
    RANDOM_NOT_SHORTEST: "random among the instance's trajectories that pass every gate and fit the cap, never the shortest "
                         "when two or more fit; the key is a hash of seed and trajectory id",
    FEWEST_TURNS_VERIFIED: "among the instance's trajectories that pass every gate and fit the cap, the one with the fewest "
                           "assistant turns among those that ran a test command after their last file edit "
                           "(apex_data.teacher.verify); ties by rendered tokens, then trajectory id; when none of them did, "
                           "random_not_shortest. A pinned instance keeps its slot and gets its trajectory by this rule",
}


def _rank(seed: int, *parts: str) -> int:
    return int(hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()[:16], 16)


def choose_trajectory(inst: Instance, outcomes: dict[str, Outcome], cap: int, seed: int,
                      rule: str = RANDOM_NOT_SHORTEST) -> Any:
    """The instance's row at `cap`: PENDING while a trajectory is unevaluated, None when nothing fits (both independent of
    `rule`), else the trajectory id of `rule` (TRAJECTORY_CHOICES) among the fitting ones."""
    if rule not in TRAJECTORY_CHOICES:
        raise ValueError(f"unknown trajectory choice {rule!r}")
    if any(t not in outcomes for t in inst.trajectories):
        return PENDING
    fit = sorted((outcomes[t].n_tokens, t) for t in inst.trajectories if outcomes[t].ok and outcomes[t].n_tokens <= cap)
    if not fit:
        return None
    if rule == FEWEST_TURNS_VERIFIED:
        if any(outcomes[t].verified is None or outcomes[t].n_turns is None for _, t in fit):
            raise ValueError(f"{inst.key}: turns or test-after-edit features missing for a fitting trajectory")
        verified = [(outcomes[t].n_turns, n, t) for n, t in fit if outcomes[t].verified]
        if verified:
            return min(verified)[2]
    if len(fit) >= 2:
        fit = fit[1:]
    return min(fit, key=lambda x: _rank(seed, "trajectory", x[1]))[1]


def seeded_order(instances: Iterable[Instance], seed: int, *labels: str) -> list[Instance]:
    out = sorted(instances, key=lambda i: i.key)
    random.Random(":".join([str(seed), *labels])).shuffle(out)
    return out


@dataclass(frozen=True)
class Orders:
    large_first: tuple[Instance, ...]   # big (seeded order), then feature (seeded order); ranked parts: by size
    random: tuple[Instance, ...]        # every instance of the part, seeded order (empty for ranked parts)


STRATUM_RANK = {"big": 0, "feature": 1, "random": 2}


def part_orders(part: PartSpec, instances: Iterable[Instance], seed: int) -> Orders:
    insts = [i for i in instances if i.subsource == part.subsource and i.language == part.language]
    if part.mode == "ranked":
        ranked = sorted(insts, key=lambda i: (STRATUM_RANK[i.stratum], -i.added, -i.files, _rank(seed, part.name, i.key)))
        return Orders(tuple(ranked), ())
    big = seeded_order((i for i in insts if i.stratum == "big"), seed, part.name, "big")
    feature = seeded_order((i for i in insts if i.stratum == "feature"), seed, part.name, "feature")
    return Orders(tuple(big + feature), tuple(seeded_order(insts, seed, part.name, "random")))


@dataclass
class Pick:
    part: str
    phase: str                          # large_first | random | ranked
    instance: Instance
    trajectory_id: str


@dataclass
class PartResult:
    part: PartSpec
    picks: list[Pick] = field(default_factory=list)
    blocked: bool = False
    shortfall: int = 0
    walked: Counter = field(default_factory=Counter)      # admitted / not_admitted / repo_cap / lineage_duplicate


@dataclass
class CapDraw:
    cap: int
    parts: dict[str, PartResult]
    pending: list[Instance]

    @property
    def complete(self) -> bool:
        return not self.pending

    @property
    def picks(self) -> list[Pick]:
        return [p for r in self.parts.values() for p in r.picks]


def draw_cap(cap: int, parts: Sequence[PartSpec], orders: dict[str, Orders], outcomes: dict[str, Outcome], seed: int, *,
             max_per_repo: int = MAX_PER_REPO, big_share: float = BIG_SHARE, lookahead: Callable[[int], int] | None = None,
             pinned: dict[str, list[Pick]] | None = None, trajectory_choice: str = RANDOM_NOT_SHORTEST) -> CapDraw:
    """Run every part in order with shared per-repository and lineage counters. A part that reaches an unevaluated
    instance stops there (blocked) and contributes the next instances of its walk to `pending`; later parts still run so
    their evaluations can proceed in parallel, but the draw is complete only when nothing is pending. `pinned` picks
    (per part) keep their instance, part and phase and only the remaining slots of each phase are drawn; their
    trajectory is kept under `random_not_shortest` and chosen again under another `trajectory_choice` (a pinned instance
    with an unevaluated trajectory is then pending). `trajectory_choice` decides only which trajectory represents an
    admitted instance, never which instances are admitted."""
    if trajectory_choice not in TRAJECTORY_CHOICES:
        raise ValueError(f"unknown trajectory choice {trajectory_choice!r}")
    lookahead = lookahead or (lambda need: max(24, 3 * need))
    repo_counts: Counter = Counter()
    lineage: set[tuple[str, str]] = set()
    taken: set[str] = set()
    results: dict[str, PartResult] = {}
    pending: dict[str, Instance] = {}
    if pinned and trajectory_choice != RANDOM_NOT_SHORTEST:
        rechosen: dict[str, list[Pick]] = {}
        for name, picks in pinned.items():
            rechosen[name] = []
            for p in picks:
                choice = choose_trajectory(p.instance, outcomes, cap, seed, trajectory_choice)
                if choice is PENDING:
                    pending.setdefault(p.instance.key, p.instance)
                    choice = p.trajectory_id
                elif choice is None:
                    raise ValueError(f"pinned instance {p.instance.key} has no trajectory within {cap} tokens")
                rechosen[name].append(Pick(p.part, p.phase, p.instance, choice))
        pinned = rechosen
    for p in (p for picks in (pinned or {}).values() for p in picks):
        taken.add(p.instance.key)
        repo_counts[p.instance.repo] += 1
        if p.instance.lineage is not None:
            lineage.add(p.instance.lineage)

    def walk(res: PartResult, order: Sequence[Instance], need: int, phase: str) -> int:
        """Take up to `need` admitted instances in `order`; returns the number short (0 when filled or blocked)."""
        got = 0
        for pos, inst in enumerate(order):
            if got >= need:
                return 0
            if inst.key in taken:
                continue
            if repo_counts[inst.repo] >= max_per_repo:
                res.walked["repo_cap"] += 1
                continue
            if inst.lineage is not None and inst.lineage in lineage:
                res.walked["lineage_duplicate"] += 1
                continue
            choice = choose_trajectory(inst, outcomes, cap, seed, trajectory_choice)
            if choice is PENDING:
                res.blocked = True
                want = lookahead(need - got)
                for nxt in order[pos:]:
                    if want <= 0:
                        break
                    if nxt.key in taken or repo_counts[nxt.repo] >= max_per_repo:
                        continue
                    if any(t not in outcomes for t in nxt.trajectories):
                        pending.setdefault(nxt.key, nxt)
                        want -= 1
                return 0
            if choice is None:
                res.walked["not_admitted"] += 1
                continue
            res.walked["admitted"] += 1
            res.picks.append(Pick(res.part.name, phase, inst, choice))
            taken.add(inst.key)
            repo_counts[inst.repo] += 1
            if inst.lineage is not None:
                lineage.add(inst.lineage)
            got += 1
        return need - got

    for part in parts:
        res = results[part.name] = PartResult(part, picks=list((pinned or {}).get(part.name, [])))
        o = orders[part.name]
        if part.mode == "ranked":
            res.shortfall = walk(res, o.large_first, part.quota - len(res.picks), "ranked")
            continue
        k = int(round(big_share * part.quota)) - sum(p.phase == "large_first" for p in res.picks)
        # a short large stratum is filled at random (plan 3.2 fallback): recorded, not a shortfall of the quota
        res.walked["large_first_short"] = walk(res, o.large_first, max(0, k), "large_first")
        if res.blocked:
            continue
        res.shortfall = walk(res, o.random, part.quota - len(res.picks), "random")
    return CapDraw(cap, results, list(pending.values()))


# ----------------------------------------------------------------------------------------------- reporting
def choice_counts(picks: Sequence[Pick], outcomes: dict[str, Outcome], cap: int) -> dict[str, int]:
    """How `fewest_turns_verified` represented each pick: by a verified trajectory, or by the random_not_shortest fallback
    (no fitting trajectory verified), split by whether the instance had one or several fitting trajectories."""
    out = dict.fromkeys(("verified", "fallback_one_fitting", "fallback_several_fitting"), 0)
    for p in picks:
        if outcomes[p.trajectory_id].verified:
            out["verified"] += 1
        else:
            n_fit = sum(1 for t in p.instance.trajectories if outcomes[t].ok and outcomes[t].n_tokens <= cap)
            out["fallback_one_fitting" if n_fit == 1 else "fallback_several_fitting"] += 1
    return out


def share(values: Iterable[bool]) -> float:
    v = list(values)
    return round(sum(v) / len(v), 3) if v else 0.0


def median(values: Iterable[int]) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    m = len(v) // 2
    return float(v[m]) if len(v) % 2 else (v[m - 1] + v[m]) / 2


ADDED_BINS = (0, 10, 25, 50, 100, 200, 500, 1000)
FILE_BINS = (1, 2, 3, 5, 10, 20)


def bin_counts(values: Iterable[int], bins: Sequence[int]) -> dict[str, int]:
    labels = [f"<={b}" for b in bins] + [f">{bins[-1]}"]
    out = dict.fromkeys(labels, 0)
    for x in values:
        out[next((f"<={b}" for b in bins if x <= b), labels[-1])] += 1
    return out


def shape_summary(picks: Sequence[Pick], tokens: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Median added lines, share >= 100 added, share >= 3 files, stratum shares, histograms, sequence and loss tokens
    (`tokens`: trajectory id -> (n_tokens, n_assistant_tokens))."""
    added = [p.instance.added for p in picks]
    files = [p.instance.files for p in picks]
    seq = sum(tokens[p.trajectory_id][0] for p in picks)
    loss = sum(tokens[p.trajectory_id][1] for p in picks)
    repos = Counter(p.instance.repo for p in picks)
    return {"n": len(picks), "median_added_lines": median(added), "share_added_ge100": share(a >= 100 for a in added),
            "share_files_ge3": share(f >= 3 for f in files),
            "strata": {s: share(p.instance.stratum == s for p in picks) for s in STRATUM_RANK},
            "phases": dict(Counter(p.phase for p in picks)),
            "sequence_tokens": seq, "loss_tokens": loss, "mean_tokens": round(seq / len(picks)) if picks else 0,
            "repositories": len(repos), "max_rows_per_repository": max(repos.values(), default=0),
            "added_lines_histogram": bin_counts(added, ADDED_BINS), "files_histogram": bin_counts(files, FILE_BINS)}
