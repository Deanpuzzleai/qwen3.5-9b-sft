"""SWE dev slice (docs/DATA_PLAN.md sections 6 and 10): 100 SWE-smith tasks from 8 held-out Python and Go repositories.

Inputs: the parquet files of `SWE-bench/SWE-smith-py` and `SWE-bench/SWE-smith-go` (source `swesmith`; RAW/swesmith_py,
RAW/swesmith_go); `data/manifests/decontam_exclusions.json` (repository exclusions); the 11 repository names of
`configs/eval/swebench_pro_v2_hard.yaml` (read-only; the D2 repository list); GitHub repository metadata (licence, fork
parent) for the repositories considered, cached under WORK/licence_cache/.

Task filters (plan section 6, survival table per language in the manifest): bug type known; >= 1 FAIL_TO_PASS;
>= 1 PASS_TO_PASS; gold patch <= 400 changed lines; <= 50 FAIL_TO_PASS tests; a non-empty problem statement (the agent's
only task description); repository not excluded (decontam_exclusions.json, the 11 locked repositories, or a shared
basename with either).

Draw (the revision-3 rule, restricted to Python and Go): per language 4 repositories and 50 tasks (13, 13, 12, 12).
A repository is eligible with >= 13 surviving tasks. Repositories are tiered by their count of PR-mirror and LM-rewrite
tasks (tier 1: >= 13; tier 2: 1 to 12; tier 3: none), shuffled with a seeded stream inside a tier, and taken in order
when their GitHub licence is on the allowlist and they are not a fork of an excluded repository (repositories passed
over are recorded with the reason). Inside a repository, tasks are taken by bug family, hardest first (PR mirror,
LM rewrite, combine, LM modify, procedural with multi-line before one-line bugs), seeded shuffle inside each group.
The manifest lists the 100 tasks in a seeded permutation, so any prefix is a random subsample.

Not done here (plan section 6, WP8): image pulls and the three-replay smoke per repository.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from apex_data import paths
from apex_data.devset.common import largest_remainder, rng_for, sample_counts, sample_plan, sha256_bytes, sha256_file

SOURCE = "swesmith"
DATASETS = {"py": ("swesmith_py", "SWE-bench/SWE-smith-py"), "go": ("swesmith_go", "SWE-bench/SWE-smith-go")}
LANGUAGES = ("py", "go")
DEV_REPOS = {"py": 4, "go": 4}
DEV_TASKS = {"py": 50, "go": 50}
MAX_PATCH_LINES = 400
MAX_F2P = 50
ONE_LINER_MAX_LINES = 2
PREFERRED_FAMILIES = ("pr_mirror", "lm_rewrite")
FILL_ORDER = ("pr_mirror", "lm_rewrite", "combine", "lm_modify", "procedural")
LICENCE_ALLOWLIST = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT-0", "CC0-1.0",
                               "Unlicense", "MPL-2.0"})
LOCKED_CONFIG = paths.CONFIGS / "eval" / "swebench_pro_v2_hard.yaml"
LICENCE_CACHE = paths.WORK / "licence_cache"
COLUMNS = ("instance_id", "repo", "image_name", "problem_statement", "patch", "FAIL_TO_PASS", "PASS_TO_PASS")
_PR = re.compile(r"^pr_(\d+)$")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/", re.M)
_PLUS_FILE = re.compile(r"^\+\+\+ b/(\S+)", re.M)


# --- parsing --------------------------------------------------------------------------------------------------------
def classify_bug_type(bug_type: str) -> str:
    if bug_type.startswith("func_pm_"):
        return "procedural"
    if bug_type in ("lm_modify", "func_basic"):
        return "lm_modify"
    if bug_type == "lm_rewrite":
        return "lm_rewrite"
    if bug_type.startswith("combine_"):
        return "combine"
    if _PR.match(bug_type):
        return "pr_mirror"
    return "other"


def parse_instance_id(instance_id: str, mirror_repo: str) -> dict[str, str]:
    """'owner__name.commit8.bug_type[__suffix]' against the mirror 'swesmith/owner__name.commit8' (names may hold dots)."""
    mirror = mirror_repo.rsplit("/", 1)[-1]
    if not instance_id.startswith(mirror + "."):
        raise ValueError(f"instance_id {instance_id!r} does not start with mirror {mirror!r}")
    rest = instance_id[len(mirror) + 1:]
    bug_type = rest.rsplit("__", 1)[0] if "__" in rest else rest
    owner_name, commit8 = mirror.rsplit(".", 1)
    owner, _, name = owner_name.partition("__")
    return {"repo": f"{owner}/{name}".lower(), "commit8": commit8, "bug_type": bug_type,
            "bug_family": classify_bug_type(bug_type)}


def patch_stats(patch: str | None) -> dict[str, int]:
    """Changed (added + removed) lines and distinct files of a unified diff."""
    patch = patch or ""
    changed = sum(1 for line in patch.splitlines() if line[:1] in "+-" and not line.startswith(("+++ ", "--- ")))
    files = _DIFF_HEADER.findall(patch) or _PLUS_FILE.findall(patch)
    return {"changed_lines": changed, "files": len(set(files))}


def basename(repo: str) -> str:
    return repo.rsplit("/", 1)[-1].lower()


# --- inputs ---------------------------------------------------------------------------------------------------------
def parquet_files(directory: Path) -> list[Path]:
    files = sorted(p for p in Path(directory).rglob("*.parquet") if ".cache" not in p.parts)
    if not files:
        raise FileNotFoundError(f"no parquet files under {directory}")
    return files


def load_tasks(lang: str, directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(one record per row with the fields the filters need; the input files with sha256). Test lists become counts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out, files = [], []
    for f in parquet_files(directory):
        files.append({"file": str(f.relative_to(directory)), "sha256": sha256_file(f)})
        local = 0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=list(COLUMNS)):
            cols = {c: (pc.list_value_length(batch.column(c)).to_pylist() if c in ("FAIL_TO_PASS", "PASS_TO_PASS")
                        else batch.column(c).to_pylist()) for c in COLUMNS}
            for k in range(batch.num_rows):
                iid, mirror = cols["instance_id"][k], cols["repo"][k]
                parsed = parse_instance_id(iid, mirror)
                stats = patch_stats(cols["patch"][k])
                ps = cols["problem_statement"][k] or ""
                out.append({"instance_id": iid, "language": lang, "swesmith_repo": mirror, **parsed,
                            "image_name": cols["image_name"][k], "patch_changed_lines": stats["changed_lines"],
                            "patch_files": stats["files"], "f2p_count": int(cols["FAIL_TO_PASS"][k] or 0),
                            "p2p_count": int(cols["PASS_TO_PASS"][k] or 0), "problem_statement_chars": len(ps.strip()),
                            "problem_statement_sha256": sha256_bytes(ps.encode("utf-8")),
                            "gold_patch_sha256": sha256_bytes((cols["patch"][k] or "").encode("utf-8")),
                            "raw": {"file": str(f.relative_to(directory)), "row": local + k}})
            local += batch.num_rows
    return out, files


def excluded_repositories() -> dict[str, str]:
    """Repository -> reason: decontam_exclusions.json plus the 11 locked repositories of the SWE-bench Pro config."""
    out = {r.lower(): "data/manifests/decontam_exclusions.json"
           for r in json.loads(paths.DECONTAM_EXCLUSIONS.read_text())["exclude"]["repositories"]}
    locked = yaml.safe_load(LOCKED_CONFIG.read_text())["split"]["repos"]
    out.update({r.lower(): "locked repository (configs/eval/swebench_pro_v2_hard.yaml)" for r in locked})
    return out


# --- filters --------------------------------------------------------------------------------------------------------
def filters(excluded: dict[str, str]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    bases = {basename(r) for r in excluded}
    return [
        ("bug_type_known", lambda t: t["bug_family"] != "other"),
        ("f2p_ge_1", lambda t: t["f2p_count"] >= 1),
        ("p2p_ge_1", lambda t: t["p2p_count"] >= 1),
        ("patch_le_400_changed_lines", lambda t: t["patch_changed_lines"] <= MAX_PATCH_LINES),
        ("f2p_le_50", lambda t: t["f2p_count"] <= MAX_F2P),
        ("problem_statement_present", lambda t: t["problem_statement_chars"] > 0),
        ("repository_not_excluded", lambda t: t["repo"] not in excluded and basename(t["repo"]) not in bases),
    ]


def apply_filters(tasks: list[dict[str, Any]], rules) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, table = list(tasks), []
    for name, ok in rules:
        nxt = [t for t in kept if ok(t)]
        table.append({"rule": name, "n_in": len(kept), "n_dropped": len(kept) - len(nxt), "n_out": len(nxt)})
        kept = nxt
    return kept, table


# --- GitHub repository metadata (licence, fork parent) ---------------------------------------------------------------
def github_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    env = paths.REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            line = line[len("export "):] if line.startswith("export ") else line
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class RepoMeta:
    """Cache-first GET https://api.github.com/repos/<owner>/<name>: {status, spdx, fork, parent, source}. Definitive
    answers (found / not_found) are cached as one JSON file per repository; offline mode never calls the API."""

    def __init__(self, cache_dir: Path = LICENCE_CACHE, token: str | None = None, offline: bool = False,
                 fetch: Callable[[str], tuple[int, dict[str, Any]]] | None = None):
        self.cache_dir, self.token, self.offline = Path(cache_dir), token, offline
        self._fetch = fetch or self._urllib
        self.api_calls = 0

    def _urllib(self, repo: str) -> tuple[int, dict[str, Any]]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "apex-data-devset"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                    return r.status, json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (404, 451):
                    return e.code, {}
                if attempt == 2:
                    return e.code, {}
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    return 0, {}
            time.sleep(2 * (attempt + 1))
        return 0, {}

    def get(self, repo: str) -> dict[str, Any]:
        path = self.cache_dir / (repo.replace("/", "__") + ".json")
        if path.exists():
            c = json.loads(path.read_text())
            return {"status": c.get("status"), "spdx": c.get("spdx"), "fork": c.get("fork"), "parent": c.get("parent"),
                    "source": c.get("source"), "full_name": c.get("full_name")}
        if self.offline:
            return {"status": "not_cached", "spdx": None, "fork": None, "parent": None, "source": None, "full_name": None}
        self.api_calls += 1
        code, body = self._fetch(repo)
        if code == 200:
            info = {"status": "found", "spdx": (body.get("license") or {}).get("spdx_id"), "fork": body.get("fork"),
                    "parent": (body.get("parent") or {}).get("full_name"),
                    "source": (body.get("source") or {}).get("full_name"), "full_name": body.get("full_name")}
        elif code == 404:
            info = {"status": "not_found", "spdx": None, "fork": None, "parent": None, "source": None, "full_name": None}
        else:
            return {"status": f"error_{code}", "spdx": None, "fork": None, "parent": None, "source": None, "full_name": None}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"repo": repo, **info, "http_status": code}, indent=1))
        os.replace(tmp, path)
        return info


def repo_reason(meta: dict[str, Any], excluded: dict[str, str]) -> str | None:
    """None when the repository may be a dev repository, else why not. A lookup that gave no definitive answer stops
    the build: passing the repository over would make the draw depend on the network."""
    if meta["status"] not in ("found", "not_found"):
        raise RuntimeError(f"GitHub metadata unavailable ({meta['status']}); re-run online with GITHUB_TOKEN set")
    if meta["status"] == "not_found":
        return "repository not found on GitHub (licence unknown)"
    if meta["spdx"] not in LICENCE_ALLOWLIST:
        return f"licence {meta['spdx']} not on the allowlist"
    bases = {basename(r) for r in excluded}
    for rel in (meta.get("parent"), meta.get("source"), meta.get("full_name")):
        if rel and (rel.lower() in excluded or basename(rel) in bases):
            return f"fork graph or rename reaches excluded repository {rel}"
    return None


# --- draw -----------------------------------------------------------------------------------------------------------
def group_by_repo(tasks: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        out.setdefault(t["repo"], []).append(t)
    return out


def pick_tasks(tasks: list[dict[str, Any]], n: int, rng) -> list[dict[str, Any]]:
    """FILL_ORDER families first (multi-line procedural before one-liners), seeded shuffle inside each group."""
    picked: list[dict[str, Any]] = []
    for fam in FILL_ORDER:
        members = [t for t in tasks if t["bug_family"] == fam]
        groups = ([t for t in members if t["patch_changed_lines"] > ONE_LINER_MAX_LINES],
                  [t for t in members if t["patch_changed_lines"] <= ONE_LINER_MAX_LINES]) if fam == "procedural" \
            else (members,)
        for group in groups:
            pool = sorted(group, key=lambda t: t["instance_id"])
            rng.shuffle(pool)
            picked.extend(pool[: n - len(picked)])
            if len(picked) >= n:
                return picked
    return picked


def pick_repositories(lang: str, cands: list[dict[str, Any]], meta: RepoMeta, excluded: dict[str, str], seed: int
                      ) -> tuple[list[str], dict[str, Any]]:
    k, need = DEV_REPOS[lang], DEV_TASKS[lang]
    per_repo = -(-need // k)
    by_repo = group_by_repo(cands)
    eligible = sorted(r for r, ts in by_repo.items() if len(ts) >= per_repo)
    preferred = {r: sum(t["bug_family"] in PREFERRED_FAMILIES for t in by_repo[r]) for r in eligible}
    tiers = ([r for r in eligible if preferred[r] >= per_repo],
             [r for r in eligible if 0 < preferred[r] < per_repo],
             [r for r in eligible if preferred[r] == 0])
    rng = rng_for(seed, SOURCE, "dev", "repos", lang)
    chosen: list[str] = []
    considered: list[dict[str, Any]] = []
    for tier_no, tier in enumerate(tiers, 1):
        order = list(tier)
        rng.shuffle(order)
        for r in order:
            if len(chosen) >= k:
                break
            m = meta.get(r)
            reason = repo_reason(m, excluded)
            considered.append({"repo": r, "tier": tier_no, "candidate_tasks": len(by_repo[r]),
                               "preferred_tasks": preferred[r], "spdx": m["spdx"], "fork_parent": m.get("parent"),
                               "chosen": reason is None, "passed_over": reason})
            if reason is None:
                chosen.append(r)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        raise RuntimeError(f"{lang}: only {len(chosen)} of {k} dev repositories found; considered {considered}")
    return chosen, {"per_repo_min_tasks": per_repo, "eligible_repos": len(eligible),
                    "tier_sizes": [len(t) for t in tiers], "considered": considered}


def build(raw_root: Path = paths.RAW, seed: int = paths.SEED, meta: RepoMeta | None = None,
          revisions: dict[str, str | None] | None = None, log: Callable[[str], Any] = print) -> dict[str, Any]:
    meta = meta or RepoMeta(token=github_token())
    revisions = revisions or {}
    excluded = excluded_repositories()
    rules = filters(excluded)
    inputs, survival, languages, repositories, items = {}, {}, {}, {}, []
    for lang in LANGUAGES:
        raw_name, hf_repo = DATASETS[lang]
        loaded, files = load_tasks(lang, Path(raw_root) / raw_name)
        inputs[raw_name] = {"repo": hf_repo, "revision": revisions.get(raw_name), "files": files}
        dup = [k for k, v in Counter(t["instance_id"] for t in loaded).items() if v > 1]
        if dup:
            raise ValueError(f"{raw_name}: duplicate instance ids, e.g. {dup[:3]}")
        cands, survival[lang] = apply_filters(loaded, rules)
        chosen, report = pick_repositories(lang, cands, meta, excluded, seed)
        shares = largest_remainder(DEV_TASKS[lang], {r: 1 for r in chosen})
        by_repo = group_by_repo(cands)
        picked = []
        for r in chosen:
            got = pick_tasks(by_repo[r], shares[r], rng_for(seed, SOURCE, "dev", "tasks", r))
            if len(got) != shares[r]:
                raise RuntimeError(f"{r}: {len(got)} tasks for a share of {shares[r]}")
            picked += got
            if len({t["swesmith_repo"] for t in by_repo[r]}) != 1:
                raise RuntimeError(f"{r}: tasks from more than one SWE-smith mirror")
            first = by_repo[r][0]
            m = next(c for c in report["considered"] if c["repo"] == r)
            repositories[r] = {"language": lang, "tasks": shares[r], "spdx": m["spdx"], "tier": m["tier"],
                               "candidate_tasks": m["candidate_tasks"], "preferred_tasks": m["preferred_tasks"],
                               "swesmith_mirror": first["swesmith_repo"], "commit8": first["commit8"],
                               "families": dict(sorted(Counter(t["bug_family"] for t in got).items()))}
        languages[lang] = {"repos": len(chosen), "tasks": len(picked), **report}
        items += picked
        log(f"dev_swe: {lang}: {len(loaded)} loaded, {len(cands)} after filters, repositories {chosen}")
    order = sorted(items, key=lambda t: t["instance_id"])
    rng_for(seed, SOURCE, "dev", "order").shuffle(order)
    out_items = [{"task_id": f"{SOURCE}:{t['instance_id']}", "upstream_id": t["instance_id"], "repo": t["repo"],
                  "language": t["language"], "swesmith_repo": t["swesmith_repo"], "image_name": t["image_name"],
                  "bug_type": t["bug_type"], "bug_family": t["bug_family"],
                  "patch_changed_lines": t["patch_changed_lines"], "patch_files": t["patch_files"],
                  "f2p_count": t["f2p_count"], "p2p_count": t["p2p_count"],
                  "problem_statement_sha256": t["problem_statement_sha256"], "gold_patch_sha256": t["gold_patch_sha256"],
                  "raw": {"dataset": DATASETS[t["language"]][0], **t["raw"]},
                  "samples": sample_plan("swe", f"{SOURCE}:{t['instance_id']}", seed)} for t in order]
    return {
        "slice": "swe", "source": SOURCE, "frozen": True, "seed": seed,
        "plan": "docs/DATA_PLAN.md section 10: 8 held-out SWE-smith repositories (Python and Go), 100 tasks x 4 samples "
                "(mini-swe-agent toolcall with the SWE-bench Pro lock's configuration); reduced set = 100 x 2 (sample "
                "indices 0 and 1)",
        "inputs": inputs,
        "excluded_repositories": dict(sorted(excluded.items())),
        "licence_allowlist": sorted(LICENCE_ALLOWLIST),
        "draw": {"rule": "per language 4 repositories x (13, 13, 12, 12) tasks; eligible with >= 13 surviving tasks; tiers "
                         "by PR-mirror + LM-rewrite count; seeded shuffle inside a tier; licence and fork check in that "
                         "order; tasks hardest bug family first, seeded shuffle inside a family; manifest order is a "
                         "seeded permutation",
                 "rng_labels": f"'{seed}:{SOURCE}:dev:repos:<language>', '{seed}:{SOURCE}:dev:tasks:<repository>', "
                               f"'{seed}:{SOURCE}:dev:order'",
                 "languages": languages},
        "survival": survival,
        "repositories": dict(sorted(repositories.items())),
        "pending_task_checks": "image pulls and the three-replay smoke per repository (gold passes, empty diff fails, "
                               "a reverted test is flagged; plan section 6) run in the SWE-smith task pipeline (WP8) "
                               "before the first dev evaluation; a failing task stays in the set and its samples are "
                               "recorded as environment failures",
        "counts": {**sample_counts(out_items), "languages": dict(sorted(Counter(t["language"] for t in out_items).items())),
                   "repositories": len(repositories)},
        "items": out_items,
    }
