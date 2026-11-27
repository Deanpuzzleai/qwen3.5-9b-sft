"""Lineage keys and the cross-pool / cross-split rule (plan 7): the same upstream pull request or base commit must not
appear twice across the SWE pools and the dev manifests.

Keys. `repo_pr` = (repo, PR number) for every record with a PR (Open-SWE-Traces rows, whose instance ids end in the PR
number: `owner__name-<n>` for SWE-rebench-V2, `owner_name_pr<n>` for Scale-SWE; and the SWE-smith pr_mirror tasks, whose
`upstream_id` ends in `.pr_<n>`): the cross-pool dedup key, applied within and across sources. `repo_commit` =
(repo, base_commit) for every record, but a `repo_commit` group counts only when it spans two or more sources: every
SWE-smith task of a repository shares one mirror commit, so within a source the key is one per repository and says nothing
(the per-task exact key with the touched-file set is applied within a source by scripts/data/run_screens.py). Repositories
are normalised to `owner/name` (URL, `.git`, case and the SWE-smith `owner__name` form), PRs to digits, commits to their
7-character prefix (SWE-smith stores 8 characters, the real-issue sources 40).

Rule. In a group that spans train and dev: before the dev freeze the dev members leave dev; after it the train members
leave train (the dev set is immutable). Among training pools the member of the highest-priority source is kept (oswt >
swesmith: the SFT teacher rows are fixed first, a stage-2 task on the same change is redundant), ties by smallest id.
Dev-internal groups are reported, never resolved here.

Dev manifests (`data/manifests/dev_*.json`) are read tolerantly: a list, or a dict whose `items` / `tasks` / `records` /
`ids` / `task_ids` is a list, of ids or of dicts with `task_id` (or `id` / `instance_id`), optional `repo`, `pr`,
`base_commit`, `source`, `split`; `frozen: true` or `frozen_at` marks the freeze.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POOL_PRIORITY = ("oswt", "swesmith")
CROSS_SOURCE_ONLY = frozenset({"repo_commit"})   # kinds whose groups count only when they span >= 2 sources
PR_MIRROR_RE = re.compile(r"\.pr_(\d+)$")        # SWE-smith pr_mirror tasks
OSWT_PR_RE = re.compile(r"(?:_pr|-)(\d+)$")      # Open-SWE-Traces: SWE-rebench-V2 owner__name-<n>, Scale-SWE owner_name_pr<n>
_PR_DIGITS = re.compile(r"(\d+)\s*$")
_HEX = re.compile(r"^[0-9a-f]{7,40}$")
_MANIFEST_LIST_KEYS = ("items", "tasks", "records", "ids", "task_ids")


def normalise_repo(repo: Any) -> str | None:
    if not repo or not isinstance(repo, str):
        return None
    s = repo.strip().lower()
    s = re.sub(r"^(https?://|git@)", "", s)
    s = re.sub(r"^(www\.)?github\.com[:/]", "", s)
    s = s.strip("/")
    if s.endswith(".git"):
        s = s[:-4]
    if "/" not in s and "__" in s:                 # SWE-smith style owner__name
        s = s.replace("__", "/", 1)
    parts = [p for p in s.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def normalise_pr(pr: Any) -> str | None:
    """Digits of `123`, `#123`, `pr_123`, `.../pull/123`; None when there is no number."""
    if pr is None or pr is False:
        return None
    if isinstance(pr, bool):
        return None
    if isinstance(pr, int):
        return str(pr) if pr > 0 else None
    m = _PR_DIGITS.search(str(pr).strip())
    return m.group(1) if m else None


def normalise_commit(commit: Any) -> str | None:
    if not commit or not isinstance(commit, str):
        return None
    s = commit.strip().lower()
    return s[:7] if _HEX.match(s) else None


@dataclass(frozen=True)
class LineageRecord:
    task_id: str
    source: str              # schema.SOURCES code or "dev"
    split: str               # train | dev | ...
    repo: str | None
    pr: str | None
    base_commit: str | None

    def lineage_keys(self) -> list[tuple[str, str]]:
        out = []
        if self.repo and self.pr:
            out.append(("repo_pr", f"{self.repo}#{self.pr}"))
        if self.repo and self.base_commit:
            out.append(("repo_commit", f"{self.repo}@{self.base_commit}"))
        return out


def from_task(task: Mapping[str, Any]) -> LineageRecord:
    """A pool record (schema.Task as a dict, or a screen item). Without `pr`, SWE-smith tasks take it from a `.pr_<n>`
    upstream id (pr_mirror) and Open-SWE-Traces rows from the trailing PR number of their instance id."""
    source = str(task.get("source") or str(task.get("task_id") or "").split(":", 1)[0])
    pr = normalise_pr(task.get("pr"))
    upstream = str(task.get("upstream_id") or str(task.get("task_id") or "").split(":", 1)[-1])
    if pr is None and source in ("swesmith", "oswt"):
        m = (PR_MIRROR_RE if source == "swesmith" else OSWT_PR_RE).search(upstream)
        pr = m.group(1) if m else None
    return LineageRecord(str(task["task_id"]), source, str(task.get("split") or "train"), normalise_repo(task.get("repo")),
                         pr, normalise_commit(task.get("base_commit")))


def from_manifest_entry(entry: Any, *, source: str = "dev", split: str = "dev") -> LineageRecord | None:
    if isinstance(entry, str):
        return LineageRecord(entry, source, split, None, None, None)
    if isinstance(entry, Mapping):
        tid = entry.get("task_id") or entry.get("id") or entry.get("instance_id")
        if not tid:
            return None
        return LineageRecord(str(tid), str(entry.get("source") or source), str(entry.get("split") or split),
                             normalise_repo(entry.get("repo")), normalise_pr(entry.get("pr")), normalise_commit(entry.get("base_commit")))
    return None


def load_manifest(path: Path, *, source: str = "dev", split: str = "dev") -> tuple[list[LineageRecord], bool]:
    """(records, frozen) from a dev manifest in any of the tolerated shapes."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = False
    entries: Any = obj
    if isinstance(obj, Mapping):
        frozen = bool(obj.get("frozen") or obj.get("frozen_at"))
        source = str(obj.get("source") or source)
        split = str(obj.get("split") or split)
        entries = next((obj[k] for k in _MANIFEST_LIST_KEYS if isinstance(obj.get(k), list)), [])
    records = [r for r in (from_manifest_entry(e, source=source, split=split) for e in entries) if r is not None]
    return records, frozen


def lineage_groups(records: Iterable[LineageRecord]) -> dict[tuple[str, str], list[LineageRecord]]:
    """{(kind, key): members} for keys shared by >= 2 distinct task ids; `repo_commit` groups only across sources."""
    groups: dict[tuple[str, str], dict[str, LineageRecord]] = defaultdict(dict)
    for r in records:
        for k in r.lineage_keys():
            groups[k].setdefault(r.task_id, r)
    out = {}
    for k, v in groups.items():
        if len(v) > 1 and (k[0] not in CROSS_SOURCE_ONLY or len({m.source for m in v.values()}) > 1):
            out[k] = sorted(v.values(), key=lambda r: r.task_id)
    return out


@dataclass(frozen=True)
class LineageDrop:
    task_id: str
    split: str        # the split the task leaves
    rule: str         # repo_pr | repo_commit
    kept_id: str
    reason: str       # dev_overlap_before_freeze | dev_overlap_after_freeze | pool_duplicate


def _rank(r: LineageRecord, priority: Sequence[str]) -> tuple[int, str]:
    return (priority.index(r.source) if r.source in priority else len(priority), r.task_id)


def resolve(groups: Mapping[tuple[str, str], Sequence[LineageRecord]], *, dev_frozen: bool,
            priority: Sequence[str] = POOL_PRIORITY) -> list[LineageDrop]:
    """One drop per (task, rule) following the module rule; a task in several groups may appear once per rule."""
    drops: dict[tuple[str, str], LineageDrop] = {}
    for (kind, _key), members in sorted(groups.items()):
        dev = [m for m in members if m.split == "dev"]
        train = sorted((m for m in members if m.split != "dev"), key=lambda r: _rank(r, priority))
        if dev and train:
            if dev_frozen:
                for m in train:
                    drops.setdefault((m.task_id, kind), LineageDrop(m.task_id, m.split, kind, dev[0].task_id, "dev_overlap_after_freeze"))
                continue
            for m in dev:
                drops.setdefault((m.task_id, kind), LineageDrop(m.task_id, "dev", kind, train[0].task_id, "dev_overlap_before_freeze"))
        for m in train[1:]:
            drops.setdefault((m.task_id, kind), LineageDrop(m.task_id, m.split, kind, train[0].task_id, "pool_duplicate"))
    return sorted(drops.values(), key=lambda d: (d.task_id, d.rule))


def counts(records: Sequence[LineageRecord], groups: Mapping[tuple[str, str], Sequence[LineageRecord]],
           drops: Sequence[LineageDrop]) -> dict[str, Any]:
    """Counts only: records per source, keyed records, groups per rule, source pairs that overlap, drops per reason."""
    pair_counter: Counter[str] = Counter()
    for (kind, _), members in groups.items():
        sources = sorted({m.source for m in members})
        pair_counter[f"{kind}:{'+'.join(sources)}"] += 1
    return {
        "n_records": len(records), "by_source": dict(Counter(r.source for r in records)),
        "n_with_repo_pr": sum(1 for r in records if any(k == "repo_pr" for k, _ in r.lineage_keys())),
        "n_with_repo_commit": sum(1 for r in records if any(k == "repo_commit" for k, _ in r.lineage_keys())),
        "n_groups": len(groups), "groups_by_rule": dict(Counter(k for k, _ in groups)),
        "groups_by_source_set": dict(pair_counter),
        "n_dev_internal_groups": sum(1 for members in groups.values() if all(m.split == "dev" for m in members)),
        "n_drops": len(drops), "drops_by_reason": dict(Counter(d.reason for d in drops)), "drops_by_split": dict(Counter(d.split for d in drops)),
    }
