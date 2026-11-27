"""SWE-smith task pool for the SWE dev slice and the stage-2 round (data plan sections 6, 10 and 11; WP8).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python -m apex_data.pools.swe_smith [--n-stage2 520] [--seed 42] \
        [--topup-language go --topup-n 220] [--run-id ID] [--dry-run]

Inputs:  RAW/swesmith_py and RAW/swesmith_go (SWE-bench/SWE-smith-py and -go parquet files; revisions in
         data/manifests/raw_sources.json under the same keys); data/manifests/dev_swe.json (the frozen dev task ids) and
         data/manifests/dev_exclusions.json (dev repositories), both read-only; data/manifests/decontam_exclusions.json via
         the D2 screen; the GitHub API for repository licences and the fork graph (apex_data.screens.licence, GITHUB_TOKEN
         from the environment or .env, answers cached under WORK/licence_cache/); the toolkit's profile registry
         (.venv/swesmith, swesmith 0.0.9).
Outputs: data/pools/swesmith/tasks.jsonl: schema.Task records of the dev tasks (split "dev") and the stage-2 candidates
         (split "stage2"), each with the F2P/P2P id lists, the bug patch (extra.bug_patch; the gold fix is its reverse,
         grade.swesmith.gold_patch, hashed in gold_patch_sha) and the issue text (extra.problem_statement). A rebuild keeps
         the calibration fields (extra.calibration, extra.eval_spec, a calibration exclusion) of records whose content is
         unchanged, so a larger --n-stage2 tops the pool up without re-calibrating the prefix of the same seeded order.
         data/pools/swesmith/manifest.json: revisions and sha256 of the inputs, the funnel (n_in / n_dropped / n_out per
         rule), the draw, histograms per split / language / repository, the outputs' sha256; the calibration step
         (scripts/data/calibrate_tasks.py) adds its admitted / excluded counts to the same manifest.
         WORK/<run_id>/candidates.jsonl: every screened task id with its repository and language (the draw's frame).
         logs/<run_id>/ with the five run files.

Task-level filters (plan section 6), applied in this order with counts: a non-empty bug patch and issue text; >= 1
FAIL_TO_PASS; >= 1 PASS_TO_PASS; bug patch <= 400 changed lines; FAIL_TO_PASS count <= 50; the bug patch touches only
non-test paths (the grader strips agent changes on test / CI / test-config paths, so a fix there could never pass); a
toolkit profile exists for the repository; D2 (the SWE-bench Pro repositories and forks, plus the repositories of
decontam_exclusions.json, through apex_data.screens.d2_repos); the repository licence is on the allowlist
(apex_data.screens.licence; unresolved licences are dropped). Image availability is checked at calibration (a task whose
image is not present is excluded there and counted). The dev tasks come from dev_swe.json verbatim; the stage-2 frame
excludes the dev repositories and the dev task ids, and the draw is uniform over the frame: the frame sorted by task_id,
shuffled with random.Random("<seed>:swesmith:stage2"), the first --n-stage2 taken (a larger N extends the same order).
A language top-up (--topup-language L --topup-n K) adds the next K tasks of language L in the same order after the first
--n-stage2, also as split "stage2", with draw_rank = their position in that order and extra.draw_topup = L (manifest
draw.topup); the first --n-stage2 tasks and every carried calibration are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.compute as pc
import pyarrow.parquet as pq

from apex_data import paths
from apex_data.grade.swesmith import SWESMITH_PYTHON, reverse_patch, sha256_text
from apex_data.grade.tamper import classify_path, parse_diff
from apex_data.pools.common import exclude, repo_key
from apex_data.schema import Task, read_jsonl, task_id as make_task_id, write_jsonl
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now

SOURCE = "swesmith"
DATASETS = {"py": ("swesmith_py", "SWE-bench/SWE-smith-py"), "go": ("swesmith_go", "SWE-bench/SWE-smith-go")}
DATASET_LICENCE = "MIT"                           # dataset cards of SWE-bench/SWE-smith-py and -go
COLUMNS = ("instance_id", "repo", "image_name", "problem_statement", "patch")
LIST_COLUMNS = ("FAIL_TO_PASS", "PASS_TO_PASS")
MAX_PATCH_LINES = 400
MAX_F2P = 50
N_STAGE2 = 520                                    # ~420 admitted after calibration (plan 11), ~80 % admission in the sprint pilot
POOL_DIR = paths.POOLS / "swesmith"
POOL_FILE = POOL_DIR / "tasks.jsonl"                # what the rollout and dev-evaluation drivers read
DEV_SWE = paths.MANIFESTS / "dev_swe.json"
CARRY_KEYS = ("calibration", "eval_spec", "split_before_calibration")
_PR = re.compile(r"^pr_(\d+)$")


# --- raw rows ---------------------------------------------------------------------------------------------------------
def raw_revisions(path: Path = paths.RAW_SOURCES) -> dict[str, Any]:
    """{key: entry} of data/manifests/raw_sources.json ({} when absent); entries carry revision, repo id and files."""
    if not path.exists():
        return {}
    obj = json.loads(path.read_text())
    entries = obj.get("sources", obj) if isinstance(obj, dict) else {}
    return entries if isinstance(entries, dict) else {}


def parquet_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def file_record(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def iter_rows(files: list[Path]) -> Iterable[dict[str, Any]]:
    """Rows of the parquet files with the scalar columns, list lengths of F2P/P2P and a (file, row) locator."""
    for f in files:
        table = pq.read_table(f, columns=[*COLUMNS, *LIST_COLUMNS])
        lengths = {c: pc.list_value_length(table.column(c)).to_pylist() for c in LIST_COLUMNS}
        scalars = {c: table.column(c).to_pylist() for c in COLUMNS}
        for i in range(table.num_rows):
            row = {c: scalars[c][i] for c in COLUMNS}
            row.update({c: lengths[c][i] or 0 for c in LIST_COLUMNS})
            row.update(_file=f.name, _row=i)
            yield row


def read_lists(files: list[Path], wanted: dict[str, set[int]]) -> dict[tuple[str, int], dict[str, list[str]]]:
    """F2P/P2P id lists of the wanted rows ({file name: row indices})."""
    out: dict[tuple[str, int], dict[str, list[str]]] = {}
    for f in files:
        rows = wanted.get(f.name)
        if not rows:
            continue
        table = pq.read_table(f, columns=list(LIST_COLUMNS)).take(sorted(rows))
        cols = {c: table.column(c).to_pylist() for c in LIST_COLUMNS}
        for j, i in enumerate(sorted(rows)):
            out[(f.name, i)] = {c: list(cols[c][j] or []) for c in LIST_COLUMNS}
    return out


# --- records ----------------------------------------------------------------------------------------------------------
def parse_instance_id(instance_id: str, swesmith_repo: str) -> dict[str, Any]:
    """Split '<owner>__<name>.<commit8>.<bug_type>[__<suffix>]' against the mirror name 'swesmith/<owner>__<name>.<commit8>'
    (repository names may contain dots, so the mirror prefix is matched, not the dots)."""
    mirror = swesmith_repo.rsplit("/", 1)[-1]
    if not instance_id.startswith(mirror + "."):
        raise ValueError(f"instance_id {instance_id!r} does not start with mirror {mirror!r}")
    rest = instance_id[len(mirror) + 1:]
    bug_type = rest.rsplit("__", 1)[0] if "__" in rest else rest
    owner_name, commit8 = mirror.rsplit(".", 1)
    owner, _, name = owner_name.partition("__")
    m = _PR.match(bug_type)
    return {"upstream": f"{owner}/{name}", "repo_name": mirror, "commit8": commit8, "bug_type": bug_type, "pr": m.group(1) if m else None}


def patch_stats(patch: str) -> dict[str, Any]:
    """Changed (added + removed) lines and touched files of a unified diff; `test_paths` = touched test/CI/config paths."""
    changes = parse_diff(patch)
    touched = sorted({p for fc in changes for p in fc.paths})
    return {"changed_lines": sum(len(fc.added) + len(fc.removed) for fc in changes), "files": len(touched), "touched": touched,
            "test_paths": [p for p in touched if classify_path(p)]}


def make_task(lang: str, revision: str | None, row: dict[str, Any]) -> Task:
    iid = row["instance_id"]
    parsed = parse_instance_id(iid, row["repo"])
    bug = row["patch"] or ""
    ps = row["problem_statement"] or ""
    stats = patch_stats(bug)
    gold = reverse_patch(bug) if bug.strip() else ""
    dataset_key, hf_repo = DATASETS[lang]
    return Task(
        task_id=make_task_id(SOURCE, iid), source=SOURCE, family="swe", language=lang, repo=parsed["upstream"], upstream_id=iid,
        hf_revision=revision, base_commit=parsed["commit8"], pr=parsed["pr"], image=row["image_name"],
        problem_statement_sha=sha256_text(ps) if ps else None,
        tests={"f2p": None, "p2p": None, "f2p_count": int(row["FAIL_TO_PASS"]), "p2p_count": int(row["PASS_TO_PASS"]),
               "runner": "swesmith profile of extra.swesmith_repo"},
        gold_patch_sha=sha256_text(gold) if gold else None,
        licence={"dataset": DATASET_LICENCE, "repo_spdx": None, "status": "pending", "method": None},
        split="stage2",
        stratum={"language": lang, "repo": repo_key(parsed["upstream"]), "bug_type": parsed["bug_type"]},
        extra={"swesmith_repo": row["repo"], "repo_name": parsed["repo_name"], "bug_type": parsed["bug_type"], "hf_repo": hf_repo,
               "problem_statement": ps, "bug_patch": bug, "bug_patch_sha256": sha256_text(bug),
               "patch_lines": stats["changed_lines"], "patch_files": stats["files"], "touched_files": stats["touched"],
               "patch_test_paths": stats["test_paths"], "raw": {"dataset": dataset_key, "file": row["_file"], "row": row["_row"]}},
    )


def load(revisions: dict[str, Any], raw_root: Path = paths.RAW, log: Callable[[str], Any] = print) -> tuple[list[Task], dict[str, Any]]:
    """Every row of the two language sets as Task records (lists not hydrated), plus the input file records."""
    tasks: list[Task] = []
    inputs: dict[str, Any] = {}
    for lang, (key, hf_repo) in DATASETS.items():
        files = parquet_files(raw_root / key)
        if not files:
            raise SystemExit(f"no parquet files under {raw_root / key} (fetch {hf_repo} first: scripts/data/fetch_raw.py)")
        rev = (revisions.get(key) or {}).get("revision")
        n0 = len(tasks)
        tasks.extend(make_task(lang, rev, row) for row in iter_rows(files))
        inputs[key] = {"hf_repo": hf_repo, "revision": rev, "rows": len(tasks) - n0, "files": [file_record(f) for f in files]}
        log(f"loaded {len(tasks) - n0} rows of {hf_repo} @ {rev} from {len(files)} file(s)")
    return tasks, inputs


def hydrate(tasks: list[Task], raw_root: Path = paths.RAW) -> None:
    """F2P/P2P id lists of the given tasks from the raw rows (checked against the load-time counts)."""
    by_dataset: dict[str, dict[str, set[int]]] = {}
    for t in tasks:
        raw = t.extra["raw"]
        by_dataset.setdefault(raw["dataset"], {}).setdefault(raw["file"], set()).add(raw["row"])
    for dataset, wanted in by_dataset.items():
        lists = read_lists(parquet_files(raw_root / dataset), wanted)
        for t in tasks:
            raw = t.extra["raw"]
            if raw["dataset"] != dataset:
                continue
            v = lists[(raw["file"], raw["row"])]
            t.tests["f2p"], t.tests["p2p"] = v["FAIL_TO_PASS"], v["PASS_TO_PASS"]
            if len(t.tests["f2p"]) != t.tests["f2p_count"] or len(t.tests["p2p"]) != t.tests["p2p_count"]:
                raise RuntimeError(f"{t.task_id}: hydrated test lists disagree with the load-time counts")


# --- funnel -------------------------------------------------------------------------------------------------------------
class Funnel:
    """Ordered survival table: each rule records n_in / n_dropped / n_out; dropped tasks are marked split 'excluded' with
    the rule and detail (pools.common.exclude) and kept in `dropped` with a reason histogram per rule."""

    def __init__(self, n_loaded: int):
        self.n_loaded, self.rows, self.dropped = n_loaded, [], []

    def apply(self, rule: str, tasks: list[Task], reason: Callable[[Task], str | None]) -> list[Task]:
        kept: list[Task] = []
        details: Counter = Counter()
        for t in tasks:
            r = reason(t)
            if r is None:
                kept.append(t)
            else:
                exclude(t, rule, r)
                self.dropped.append(t)
                details[r.split(":", 1)[0] if rule in ("licence_allowlist", "d2_repository") else rule] += 1
        self.rows.append({"rule": rule, "n_in": len(tasks), "n_dropped": len(tasks) - len(kept), "n_out": len(kept),
                          **({"top_reasons": dict(details.most_common(8))} if rule in ("licence_allowlist", "d2_repository") and details else {})})
        return kept

    def table(self) -> dict[str, Any]:
        return {"n_loaded": self.n_loaded, "rules": list(self.rows), "n_out": self.rows[-1]["n_out"] if self.rows else self.n_loaded}


def toolkit_repo_names() -> set[str]:
    """Every repo_name the toolkit's profile registry knows (one call into .venv/swesmith)."""
    code = "import json; from swesmith.profiles import registry; print(json.dumps(sorted(k for k in registry.keys() if '/' not in k)))"
    with tempfile.TemporaryDirectory() as cwd:   # not the repository root: the toolkit's load_dotenv() would read .env
        r = subprocess.run([str(SWESMITH_PYTHON), "-c", code], cwd=cwd, capture_output=True, text=True, timeout=300, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"toolkit registry listing failed: {r.stderr[-500:]}")
    return set(json.loads(r.stdout.strip().splitlines()[-1]))


def static_filters(tasks: list[Task], funnel: Funnel, profiles: set[str]) -> list[Task]:
    kept = funnel.apply("bug_patch_present", tasks, lambda t: None if t.extra["bug_patch"].strip() else "empty bug patch")
    kept = funnel.apply("issue_text_present", kept, lambda t: None if t.extra["problem_statement"].strip() else "empty problem statement")
    kept = funnel.apply("f2p_ge_1", kept, lambda t: None if t.tests["f2p_count"] >= 1 else "no FAIL_TO_PASS tests")
    kept = funnel.apply("p2p_ge_1", kept, lambda t: None if t.tests["p2p_count"] >= 1 else "no PASS_TO_PASS tests")
    kept = funnel.apply("patch_le_400_lines", kept,
                        lambda t: None if t.extra["patch_lines"] <= MAX_PATCH_LINES else f"{t.extra['patch_lines']} changed lines")
    kept = funnel.apply("f2p_le_50", kept, lambda t: None if t.tests["f2p_count"] <= MAX_F2P else f"{t.tests['f2p_count']} FAIL_TO_PASS tests")
    kept = funnel.apply("fix_on_source_paths_only", kept,
                        lambda t: None if not t.extra["patch_test_paths"] else "bug patch touches " + ",".join(t.extra["patch_test_paths"][:3]))
    kept = funnel.apply("toolkit_profile", kept, lambda t: None if t.extra["repo_name"] in profiles else f"no profile for {t.extra['repo_name']}")
    return kept


def screen_filters(tasks: list[Task], funnel: Funnel, log: Callable[[str], Any]) -> tuple[list[Task], dict[str, Any]]:
    """D2 (static mechanisms + fork graph) and the licence allowlist through the shared screens, one GitHub lookup per
    repository. Returns the survivors and a summary for the manifest."""
    from apex_data.screens import d2_repos, licence  # noqa: PLC0415 - the screens import pools.common

    locked = d2_repos.load_locked()
    lookup = licence.RepoLookup(token=licence.github_token())
    infos, lookup_summary = lookup.lookup((repo_key(t.repo) for t in tasks), log=log)
    d2_repos.annotate_static(tasks, locked)
    fork_states = d2_repos.annotate_fork_graph(tasks, locked, infos, lookup_enabled=lookup.enabled)
    kept = funnel.apply("d2_repository", tasks, lambda t: ", ".join(t.screens["d2"]["mechanisms"]) or None)
    status = licence.annotate_licence(kept, infos)
    kept = funnel.apply("licence_allowlist", kept,
                        lambda t: None if t.licence.get("status") == "ok" else f"{t.licence.get('method') or 'unresolved'}:{t.licence.get('repo_spdx')}")
    spdx = Counter(str(t.licence.get("repo_spdx")) for t in tasks if t.licence.get("method"))
    return kept, {"lookup": lookup_summary, "d2_locked_sha256": locked.sha256[:16], "d2_fork_graph": dict(fork_states),
                  "licence_allowlist": sorted(licence.ALLOWLIST), "licence_status": dict(status), "spdx_counts_after_d2": dict(spdx.most_common())}


# --- dev and stage-2 ----------------------------------------------------------------------------------------------------
def upstream_of(task_id: str) -> str:
    return task_id.split(":", 1)[1] if ":" in task_id else task_id


def read_dev(dev_file: Path = DEV_SWE, exclusions_file: Path = paths.DEV_EXCLUSIONS) -> tuple[list[str], set[str], dict[str, Any]]:
    """(dev upstream ids, dev repositories (lowercase owner/name), provenance) from the frozen dev manifests."""
    if not dev_file.exists():
        raise SystemExit(f"{dev_file} not found: the dev set is frozen by WP6 before the stage-2 pool is drawn")
    dev = json.loads(dev_file.read_text())
    listed = dev.get("task_ids") or [it["task_id"] for it in dev.get("items") or []]   # WP6 writes `items`; `task_ids` is the older form
    ids = [upstream_of(t) for t in listed]
    repos = {repo_key(r) for r in (dev.get("repositories") or {}) if repo_key(r)}
    excl: dict[str, Any] = {}
    if exclusions_file.exists():
        excl = json.loads(exclusions_file.read_text())
        swe = excl.get("swe") if isinstance(excl.get("swe"), dict) else excl
        for key in ("repositories", "swe_repositories", "repos"):
            vals = swe.get(key)
            if isinstance(vals, dict):
                vals = list(vals)
            for r in vals or []:
                if repo_key(r):
                    repos.add(repo_key(r))
        for key in ("task_ids", "swe_task_ids"):
            ids += [upstream_of(t) for t in swe.get(key) or [] if str(t).startswith(("swesmith:", "T1:")) or ":" not in str(t)]
    ids = list(dict.fromkeys(ids))
    prov = {"dev_swe": {**file_record(dev_file), "n_task_ids": len(listed), "frozen": dev.get("frozen")},
            "dev_exclusions": file_record(exclusions_file) if exclusions_file.exists() else None,
            "n_dev_repositories": len(repos)}
    return ids, repos, prov


def draw_order(frame: list[Task], seed: int) -> list[Task]:
    """The draw order: the frame sorted by task_id, shuffled with Random('<seed>:swesmith:stage2')."""
    order = sorted(frame, key=lambda t: t.task_id)
    random.Random(f"{seed}:swesmith:stage2").shuffle(order)
    return order


def draw_stage2(frame: list[Task], n: int, seed: int) -> list[Task]:
    """Uniform seeded draw: the first n of the draw order."""
    return draw_order(frame, seed)[:n]


def draw_topup(order: list[Task], n_stage2: int, language: str, n: int) -> list[tuple[int, Task]]:
    """(draw rank, task) of the next n tasks of `language` in the draw order after its first n_stage2."""
    return [(r, t) for r, t in enumerate(order) if r >= n_stage2 and t.language == language][:n]


def carry_over(new: list[Task], old_path: Path) -> int:
    """Copy the calibration fields of unchanged records from an existing pool file (same task_id, bug patch and lists)."""
    if not old_path.exists():
        return 0
    old = {r["task_id"]: r for r in read_jsonl(old_path)}
    n = 0
    for t in new:
        o = old.get(t.task_id)
        if not o:
            continue
        same = (o.get("extra", {}).get("bug_patch_sha256") == t.extra["bug_patch_sha256"]
                and (o.get("tests") or {}).get("f2p") == t.tests["f2p"] and (o.get("tests") or {}).get("p2p") == t.tests["p2p"]
                and o.get("image") == t.image)
        if not same:
            continue
        for k in CARRY_KEYS:
            if k in (o.get("extra") or {}):
                t.extra[k] = o["extra"][k]
        exc = (o.get("extra") or {}).get("exclusion") or {}
        if o.get("split") == "excluded" and exc.get("rule") in ("calibration", "smoke"):
            t.extra["split_before_calibration"] = t.split
            t.split, t.extra["exclusion"] = "excluded", exc
        n += 1
    return n


def histogram(values: Iterable[Any]) -> dict[str, int]:
    c = Counter(str(v) for v in values)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def build(n_stage2: int = N_STAGE2, seed: int = paths.SEED, raw_root: Path = paths.RAW, pool_dir: Path = POOL_DIR,
          work_dir: Path | None = None, dev_file: Path = DEV_SWE, exclusions_file: Path = paths.DEV_EXCLUSIONS,
          log: Callable[[str], Any] = print, dry_run: bool = False, profiles: set[str] | None = None,
          topup_language: str | None = None, topup_n: int = 0,
          screens: Callable[[list[Task], Funnel, Callable[[str], Any]], tuple[list[Task], dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Build the pool; returns the manifest (also written unless dry_run). `profiles` and `screens` are injectable for tests."""
    t0 = time.time()
    revisions = raw_revisions()
    tasks, inputs = load(revisions, raw_root, log)
    funnel = Funnel(len(tasks))
    by_upstream = {t.upstream_id: t for t in tasks}
    dev_ids, dev_repos, dev_prov = read_dev(dev_file, exclusions_file)
    missing_dev = [i for i in dev_ids if i not in by_upstream]
    if missing_dev:
        raise SystemExit(f"{len(missing_dev)} dev task ids are not in the raw SWE-smith rows (e.g. {missing_dev[:3]})")
    dev_tasks = [by_upstream[i] for i in dev_ids]
    dev_set = set(dev_ids)
    kept = static_filters(tasks, funnel, profiles if profiles is not None else toolkit_repo_names())
    kept, screen_summary = (screens or screen_filters)(kept, funnel, log)
    dev_failing = [{"task_id": t.task_id, "rule": t.extra["exclusion"]["rule"], "detail": t.extra["exclusion"]["detail"]}
                   for t in dev_tasks if t.split == "excluded"]
    for t in dev_tasks:                    # the dev set is frozen: its tasks stay dev; a failed task-level filter is reported
        if t.split == "excluded":
            t.extra["dev_filter_flag"] = t.extra.pop("exclusion")
        t.split = "dev"
    frame_rules = Funnel(len(kept))
    frame = [t for t in kept if t.upstream_id not in dev_set]           # dev tasks are not candidates (never marked excluded)
    frame_rules.rows.append({"rule": "not_dev_task", "n_in": len(kept), "n_dropped": len(kept) - len(frame), "n_out": len(frame)})
    frame = frame_rules.apply("not_dev_repository", frame, lambda t: f"dev repository {repo_key(t.repo)}" if repo_key(t.repo) in dev_repos else None)
    order = draw_order(frame, seed)
    topup = draw_topup(order, n_stage2, topup_language, topup_n) if topup_language else []
    for rank, t in [*enumerate(order[:n_stage2]), *topup]:
        t.split = "stage2"
        t.extra["draw_rank"] = rank        # calibration runs in this order, so a partial run covers a uniform prefix
    for _, t in topup:
        t.extra["draw_topup"] = topup_language
    picked = order[:n_stage2] + [t for _, t in topup]
    out_tasks = sorted(dev_tasks, key=lambda t: t.task_id) + sorted(picked, key=lambda t: t.task_id)
    hydrate(out_tasks, raw_root)
    for t in out_tasks:
        t.validate()
    pool_file = pool_dir / "tasks.jsonl"
    carried = carry_over(out_tasks, pool_file)
    run_work = work_dir or (paths.WORK / f"swesmith_pool_seed{seed}")
    manifest: dict[str, Any] = {
        "pool": "swesmith", "source": SOURCE, "built_at": utc_now(), "seed": seed,
        "inputs": {**inputs, **dev_prov}, "dataset_licence": DATASET_LICENCE,
        "funnel": {"task_level": funnel.table(), "stage2_frame": frame_rules.table()},
        "screens": screen_summary,
        "dev": {"n": len(dev_tasks), "repositories": histogram(repo_key(t.repo) for t in dev_tasks),
                "languages": histogram(t.language for t in dev_tasks), "failing_task_level_filters": dev_failing},
        "draw": {"method": "uniform over the screened frame; frame sorted by task_id, shuffled with random.Random('<seed>:swesmith:stage2'), first n",
                 "frame_size": len(frame), "n_requested": n_stage2, "n_drawn": len(picked),
                 "frame_languages": histogram(t.language for t in frame), "frame_repositories": len({repo_key(t.repo) for t in frame}),
                 "topup": {"method": f"the next n tasks of language {topup_language!r} in the same draw order after its first n_stage2",
                           "language": topup_language, "after_rank": n_stage2, "n_requested": topup_n, "n_drawn": len(topup),
                           "draw_ranks": [topup[0][0], topup[-1][0]] if topup else None,
                           "repositories": histogram(repo_key(t.repo) for _, t in topup)} if topup_language else None},
        "stage2": {"n": len(picked), "languages": histogram(t.language for t in picked),
                   "repositories": histogram(repo_key(t.repo) for t in picked), "n_repositories": len({repo_key(t.repo) for t in picked}),
                   "bug_types": histogram(re.sub(r"_\d+$", "", t.extra["bug_type"]) for t in picked)},
        "images": {"n_distinct": len({t.image for t in out_tasks}), "dev": len({t.image for t in dev_tasks}), "stage2": len({t.image for t in picked})},
        "carried_calibrations": carried,
        "seconds": round(time.time() - t0, 1),
    }
    if dry_run:
        return manifest
    run_work.mkdir(parents=True, exist_ok=True)
    cand = run_work / "candidates.jsonl"
    write_jsonl(cand, ({"task_id": t.task_id, "repo": repo_key(t.repo), "language": t.language} for t in sorted(frame, key=lambda t: t.task_id)))
    write_jsonl(pool_file, out_tasks)
    manifest["outputs"] = {"tasks": {**file_record(pool_file), "n_records": len(out_tasks), "splits": histogram(t.split for t in out_tasks)},
                           "candidates": {**file_record(cand), "n_records": len(frame)}}
    old = json.loads((pool_dir / "manifest.json").read_text()) if (pool_dir / "manifest.json").exists() else {}
    if old.get("calibration"):
        manifest["calibration_before_rebuild"] = old["calibration"]
    if old.get("smoke"):                 # per-repository smoke results stay valid for unchanged records (calibrate_tasks merges)
        manifest["smoke"] = old["smoke"]
    tmp = pool_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(pool_dir / "manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-stage2", type=int, default=N_STAGE2, help="stage-2 candidates to draw (default %(default)s)")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="default data_swesmith_pool_<UTC timestamp>")
    p.add_argument("--pool-dir", type=Path, default=POOL_DIR, help="pool directory to rebuild (tasks.jsonl, manifest.json; "
                   "default %(default)s); a staging copy calibrates a top-up before it replaces the live pool")
    p.add_argument("--topup-language", default=None, help="add the next --topup-n tasks of this language (py, go) in the draw order")
    p.add_argument("--topup-n", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print the manifest without writing the pool")
    a = p.parse_args(argv)
    if (a.topup_language is None) != (a.topup_n <= 0):
        p.error("--topup-language and a positive --topup-n go together")
    run_id = a.run_id or new_run_id("data_swesmith_pool")
    with RunDir(run_id, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, kind="data_swesmith_pool", command=command_line(), code=git_info(), hardware=hardware_info(),
                           seed=a.seed, config={"n_stage2": a.n_stage2, "topup_language": a.topup_language, "topup_n": a.topup_n, "pool_dir": str(a.pool_dir),
                                                "max_patch_lines": MAX_PATCH_LINES, "max_f2p": MAX_F2P, "dry_run": a.dry_run},
                           start_time=utc_now(), status="running")
        man = build(a.n_stage2, a.seed, pool_dir=a.pool_dir, work_dir=paths.WORK / run_id, log=run.log, dry_run=a.dry_run,
                    topup_language=a.topup_language, topup_n=a.topup_n)
        for row in man["funnel"]["task_level"]["rules"] + man["funnel"]["stage2_frame"]["rules"]:
            run.metric(row)
        run.log(json.dumps({k: man[k] for k in ("funnel", "draw", "stage2", "images", "carried_calibrations")}, indent=1) if a.dry_run else
                f"pool written: {man['outputs']['tasks']}; draw {man['draw']['n_drawn']} of {man['draw']['frame_size']}")
        run.write_metadata(data_version={"inputs": man["inputs"]}, end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished",
                           summary={"funnel_out": man["funnel"]["task_level"]["n_out"], "frame": man["draw"]["frame_size"],
                                    "dev": man["dev"]["n"], "stage2": man["stage2"]["n"], "images": man["images"]})
        run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "wall clock"), ("gpu_hours", 0, "h", "CPU-only step"),
                             ("cost_usd", "n/a", "USD", "inside the apex-cpu VM's hourly cost")])
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
