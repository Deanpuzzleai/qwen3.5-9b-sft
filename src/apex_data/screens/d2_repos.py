"""D2 repository exclusion (plan 7 D2): the SWE-bench Pro repositories at every commit, plus the repository-level
exclusions of data/manifests/decontam_exclusions.json.

Two lists, kept apart because the plan treats them differently and the M2.1 check counts them separately:

Locked (the 11 SWE-bench Pro HARD-51 repositories): names derived from configs/eval/swebench_pro_hard51_ids.txt
(instance_<owner>__<name>-<sha>) plus the `split.repos` names of configs/eval/swebench_pro_v2_hard.yaml (both read-only),
the predecessors (matrix-org/matrix-react-sdk, vector-im/element-web, ansible-collections/*, NodeBB/nodebb-plugin-*) and
the basenames. Mechanisms, recorded by name in the `d2` mark:
  metadata     the record's repository is a locked repository (case-insensitive)
  predecessor  the repository matches a predecessor pattern (ansible-collections/*, nodebb/nodebb-plugin-*)
  basename     the repository name equals a locked basename (ansible, openlibrary, flipt, teleport, webclients,
               element-web, vuls, qutebrowser, navidrome, nodebb, tutanota, matrix-react-sdk) under another owner
  identifier   a locked basename is a whole token of the upstream id or the image reference (SWE-smith mirrors)
  signature    a module-path signature in the text (github.com/gravitational/teleport, go.flipt.io/flipt,
               from/import ansible | openlibrary | qutebrowser, @proton/, @tutao/, matrix-react-sdk, nodebb, ...);
               a repository that merely imports a locked one is dropped too (plan 7: a dependency import is a drop)
  fork_graph   the GitHub API says the repository is a fork of, derives from, or was renamed from a locked repository
Excluded (`exclude.repositories` of decontam_exclusions.json, e.g. bottlepy/bottle, the codebase of a Terminal-Bench 2.0
task): a repository-level exclusion only, with no signature or dependency-import rule (a project that imports the library is
not its codebase). Mechanisms: excluded_repo (metadata), excluded_basename (same name under another owner),
excluded_identifier (whole token of the upstream id or image), excluded_fork_graph (fork / source / rename).

Fork graph: needs the GitHub API (`licence.RepoLookup`, GITHUB_TOKEN from the environment or .env, cache under
WORK/licence_cache/). Without a token and without a cached answer the step is recorded as "not_run" per record.
Both lists are hashed into every mark (`locked_sha256`), so a later, larger list is a visible change.

Interface for the pool builders: `locked = load_locked()`; per record `screen(repo, text=..., upstream_id=..., image=...,
locked=locked)` -> D2Result (mechanisms, `locked_hit`, `excluded_hit`, `drop`, `as_mark()`); `fork_graph(info, locked)` adds
the fork mechanisms from a RepoInfo. Task-level helpers (`annotate_static`, `annotate_fork_graph`, the `*_reason` survival
predicates) work on schema.Task records.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from apex_data import paths
from apex_data.pools.common import Task, repo_key, sha256_hex
from apex_data.screens.licence import RepoInfo

IDS_FILE = paths.CONFIGS / "eval" / "swebench_pro_hard51_ids.txt"
LOCK_YAML = paths.CONFIGS / "eval" / "swebench_pro_v2_hard.yaml"
EXCLUSIONS_FILE = paths.DECONTAM_EXCLUSIONS
PLAN_BASENAMES = ("ansible", "openlibrary", "flipt", "teleport", "webclients", "element-web", "vuls", "qutebrowser",
                  "navidrome", "nodebb", "tutanota")
PREDECESSORS = ("matrix-org/matrix-react-sdk", "vector-im/element-web")
PREDECESSOR_PREFIXES = ("ansible-collections/", "nodebb/nodebb-plugin-")
SIGNATURES = (
    ("teleport", r"github\.com/gravitational/teleport"),
    ("vuls", r"github\.com/future-architect/vuls"),
    ("navidrome", r"github\.com/navidrome/navidrome"),
    ("flipt", r"go\.flipt\.io/flipt|github\.com/flipt-io/flipt"),
    ("openlibrary", r"github\.com/internetarchive/openlibrary|\b(?:from|import)\s+openlibrary\b"),
    ("ansible", r"\b(?:from|import)\s+ansible\b"),
    ("qutebrowser", r"\b(?:from|import)\s+qutebrowser\b"),
    ("webclients", r"@proton/"),
    ("tutanota", r"@tutao/"),
    ("element-web", r"\bmatrix-react-sdk\b"),
    ("nodebb", r"\bnodebb\b"),
)
_SIGNATURES = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in SIGNATURES]
_ID = re.compile(r"^instance_([^_]+)__(.+?)-[0-9a-f]{40}")
LOCKED_KINDS = ("metadata", "predecessor", "basename", "identifier", "signature", "fork_graph")
EXCLUDED_KINDS = ("excluded_repo", "excluded_basename", "excluded_identifier", "excluded_fork_graph")
# Plan 7 D2 names the 11 SWE-bench Pro repositories and their predecessors. Checked names-only on 2026-10-11 against the
# public sources below: none names a repository beyond the 11 locked ones. Recorded in the screen reports.
UNION_SOURCES = {
    "checked_on": "2026-10-11",
    "method": "repository names only, from public dataset cards, repository metadata and the paper",
    "sources": [
        {"source": "ScaleAI/SWE-bench_Pro dataset card", "public_tasks": 642, "repositories": 11,
         "repositories_beyond_locked": 0},
        {"source": "scaleapi/SWE-bench_Pro-os README", "repositories": 11, "repositories_beyond_locked": 0},
        {"source": "paper arXiv:2509.16941", "public_tasks": 731, "public_repositories": 11, "commercial_tasks": 276,
         "commercial_repositories": 18, "held_out_tasks": 858, "held_out_repositories": 12,
         "note": "commercial and held-out repositories are unnamed and private", "repositories_beyond_locked": 0},
    ],
    "union": "11 locked HARD-51 repositories + 2 predecessors (matrix-org/matrix-react-sdk, vector-im/element-web) = 13",
}


@dataclass(frozen=True)
class LockedRepos:
    repos: frozenset[str]              # lowercase owner/name, predecessors included
    basenames: frozenset[str]
    excluded: frozenset[str]           # lowercase owner/name from decontam_exclusions.json (repository level only)
    excluded_basenames: frozenset[str]
    sources: dict[str, list[str]]
    sha256: str

    def basename_patterns(self) -> dict[str, re.Pattern[str]]:
        return _basename_patterns(self.basenames)

    def excluded_patterns(self) -> dict[str, re.Pattern[str]]:
        return _basename_patterns(self.excluded_basenames)


@lru_cache(maxsize=16)
def _basename_patterns(basenames: frozenset[str]) -> dict[str, re.Pattern[str]]:
    return {b: re.compile(r"(?<![a-z0-9])" + re.escape(b) + r"(?![a-z0-9])") for b in sorted(basenames)}


def load_exclusions(path: Path = EXCLUSIONS_FILE) -> list[str]:
    """`exclude.repositories` of data/manifests/decontam_exclusions.json as lowercase owner/name (empty when absent)."""
    if not Path(path).exists():
        return []
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for r in (obj.get("exclude") or {}).get("repositories") or []:
        k = repo_key(str(r))
        if k is None:
            raise ValueError(f"{path}: repository {r!r} is not of the form owner/name")
        out.append(k)
    return sorted(set(out))


def load_locked(ids_file: Path = IDS_FILE, lock_yaml: Path = LOCK_YAML, exclusions_file: Path = EXCLUSIONS_FILE) -> LockedRepos:
    from_ids = sorted({f"{m.group(1)}/{m.group(2)}".lower()
                       for line in Path(ids_file).read_text().splitlines() if (m := _ID.match(line.strip()))})
    from_yaml: list[str] = []
    if Path(lock_yaml).exists():
        cfg = yaml.safe_load(Path(lock_yaml).read_text()) or {}
        from_yaml = sorted(str(r).lower() for r in ((cfg.get("split") or {}).get("repos") or {}))
    repos = set(from_ids) | set(from_yaml) | set(PREDECESSORS)
    basenames = {r.split("/")[1] for r in repos} | set(PLAN_BASENAMES)
    excluded = set(load_exclusions(exclusions_file)) - repos
    excluded_basenames = {r.split("/")[1] for r in excluded}
    listing = (sorted(repos) + [f"basename:{b}" for b in sorted(basenames)] + [f"prefix:{p}" for p in PREDECESSOR_PREFIXES]
               + [f"excluded:{r}" for r in sorted(excluded)])
    sources = {"ids_file": from_ids, "lock_yaml": from_yaml, "plan_basenames": list(PLAN_BASENAMES),
               "predecessors": list(PREDECESSORS), "predecessor_prefixes": list(PREDECESSOR_PREFIXES),
               "decontam_exclusions": sorted(excluded)}
    return LockedRepos(frozenset(repos), frozenset(basenames), frozenset(excluded), frozenset(excluded_basenames),
                       sources, sha256_hex("\n".join(listing)))


def _token_hits(text: str | None, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    if not text:
        return []
    s = text.lower().replace("_", "-")
    return [b for b, p in patterns.items() if p.search(s)]


def identifier_hits(text: str | None, locked: LockedRepos) -> list[str]:
    """Locked basenames occurring as whole tokens ('_' counts as '-') in an identifier such as an image reference."""
    return _token_hits(text, locked.basename_patterns())


def repo_mechanisms(repo: str | None, upstream_id: str | None, image: str | None, locked: LockedRepos) -> list[str]:
    """Repository-identity mechanisms of both lists (no text signatures, no network)."""
    hits: list[str] = []
    key = repo_key(repo)
    if key:
        name = key.split("/")[1].replace("_", "-")
        if key in locked.repos:
            hits.append(f"metadata:{key}")
        hits += [f"predecessor:{p}*" for p in PREDECESSOR_PREFIXES if key.startswith(p)]
        if name in locked.basenames and key not in locked.repos:
            hits.append(f"basename:{name}")
        if key in locked.excluded:
            hits.append(f"excluded_repo:{key}")
        elif name in locked.excluded_basenames:
            hits.append(f"excluded_basename:{name}")
    for label, value in (("upstream_id", upstream_id), ("image", image)):
        hits += [f"identifier:{label}:{b}" for b in identifier_hits(value, locked)]
        hits += [f"excluded_identifier:{label}:{b}" for b in _token_hits(value, locked.excluded_patterns())]
    return hits


def text_signatures(text: str | None) -> list[str]:
    """Module-path signatures of the locked repositories (never applied to the exclusion list)."""
    return [name for name, rx in _SIGNATURES if rx.search(text)] if text else []


def fork_mechanisms(info: RepoInfo | None, locked: LockedRepos) -> list[str]:
    if info is None or info.status != "found":
        return []
    hits = []
    for label, value in (("parent", info.parent), ("source", info.source), ("renamed", info.full_name)):
        k = repo_key(value)
        if not k or (label == "renamed" and k == info.repo):
            continue
        if k in locked.repos:
            hits.append(f"fork_graph:{label}:{k}")
        elif k in locked.excluded:
            hits.append(f"excluded_fork_graph:{label}:{k}")
    return hits


def kind_of(mechanism: str) -> str:
    return mechanism.split(":", 1)[0]


@dataclass
class D2Result:
    mechanisms: list[str] = field(default_factory=list)
    fork_graph: str = "not_run"          # not_run | clear | hit | not_found | error | rate_limited | unresolved
    locked_sha256: str = ""

    @property
    def locked_hit(self) -> bool:
        return any(kind_of(m) in LOCKED_KINDS for m in self.mechanisms)

    @property
    def excluded_hit(self) -> bool:
        return any(kind_of(m) in EXCLUDED_KINDS for m in self.mechanisms)

    @property
    def drop(self) -> bool:
        return bool(self.mechanisms)

    def add_fork_graph(self, info: RepoInfo | None, locked: LockedRepos, *, lookup_enabled: bool) -> None:
        """Fork / rename check from a cached or fetched GitHub answer; `lookup_enabled` False and no answer -> not_run."""
        if info is None:
            self.fork_graph = "unresolved" if lookup_enabled else "not_run"
            return
        if info.status != "found":
            self.fork_graph = info.status
            return
        hits = fork_mechanisms(info, locked)
        self.fork_graph = "hit" if hits else "clear"
        self.mechanisms += [h for h in hits if h not in self.mechanisms]

    def as_mark(self) -> dict[str, Any]:
        return {"hit": self.locked_hit, "excluded": self.excluded_hit, "mechanisms": list(self.mechanisms),
                "fork_graph": self.fork_graph, "locked_sha256": self.locked_sha256}


def screen(repo: str | None = None, *, locked: LockedRepos, text: str | None = None, upstream_id: str | None = None,
           image: str | None = None) -> D2Result:
    """Static D2 for one record: repository identity of both lists plus the locked signatures in `text` (the issue text,
    the task instruction or a rendered row). The fork step is added by `D2Result.add_fork_graph`."""
    mech = repo_mechanisms(repo, upstream_id, image, locked)
    mech += [f"signature:{s}" for s in text_signatures(text)]
    return D2Result(mech, "not_run", locked.sha256[:16])


def counts(results: Iterable[D2Result]) -> dict[str, Any]:
    """Counts only: records with a locked hit / an exclusion hit / either, hits per mechanism kind, fork-graph states."""
    results = list(results)
    by_kind: Counter[str] = Counter()
    for r in results:
        by_kind.update({kind_of(m) for m in r.mechanisms})
    return {"n_records": len(results), "n_locked_hit": sum(r.locked_hit for r in results),
            "n_excluded_hit": sum(r.excluded_hit for r in results), "n_dropped": sum(r.drop for r in results),
            "records_by_mechanism": dict(sorted(by_kind.items())),
            "fork_graph": dict(Counter(r.fork_graph for r in results))}


# --- schema.Task helpers ---------------------------------------------------------------------------------------------
def task_text(t: Task) -> str | None:
    """The task statement of a pool record: extra.problem_statement, else extra.text (schema.Task keeps text in extra)."""
    return t.extra.get("problem_statement") or t.extra.get("text")


def annotate_static(tasks: Iterable[Task], locked: LockedRepos, text_of: Callable[[Task], str | None] = task_text) -> Counter:
    """Mechanisms that need no network, written to task.screens["d2"]. Returns records per mechanism kind."""
    c: Counter = Counter()
    for t in tasks:
        r = screen(t.repo, locked=locked, text=text_of(t), upstream_id=t.upstream_id, image=t.image)
        t.screens["d2"] = r.as_mark()
        c.update({kind_of(m) for m in r.mechanisms})
    return c


def annotate_fork_graph(tasks: Iterable[Task], locked: LockedRepos, infos: dict[str, RepoInfo], *,
                        lookup_enabled: bool = True) -> Counter:
    """Fork / rename check from the GitHub answers; the mark's fork_graph becomes clear | hit | not_found | error |
    rate_limited | unresolved, or not_run when no lookup was possible. Returns the state counts."""
    c: Counter = Counter()
    for t in tasks:
        mark = t.screens.get("d2") or screen(t.repo, locked=locked).as_mark()
        r = D2Result(list(mark.get("mechanisms", [])), mark.get("fork_graph", "not_run"), locked.sha256[:16])
        r.add_fork_graph(infos.get(repo_key(t.repo) or ""), locked, lookup_enabled=lookup_enabled)
        t.screens["d2"] = r.as_mark()
        c[r.fork_graph] += 1
    return c


def _mechanisms(t: Task, kinds: tuple[str, ...]) -> list[str]:
    return [m for m in (t.screens.get("d2") or {}).get("mechanisms", []) if kind_of(m) in kinds]


def repository_reason(t: Task) -> str | None:
    m = _mechanisms(t, ("metadata", "predecessor", "basename", "identifier"))
    return ", ".join(m) if m else None


def signature_reason(t: Task) -> str | None:
    m = _mechanisms(t, ("signature",))
    return ", ".join(m) if m else None


def fork_graph_reason(t: Task) -> str | None:
    m = _mechanisms(t, ("fork_graph",))
    return ", ".join(m) if m else None


def exclusion_reason(t: Task) -> str | None:
    """decontam_exclusions.json repositories (any excluded_* mechanism, fork graph included)."""
    m = _mechanisms(t, EXCLUDED_KINDS)
    return ", ".join(m) if m else None


def hit_records(tasks: Iterable[Task]) -> list[dict[str, Any]]:
    return [{"task_id": t.task_id, "repo": t.repo, "mechanisms": t.screens["d2"]["mechanisms"]}
            for t in tasks if (t.screens.get("d2") or {}).get("mechanisms")]
