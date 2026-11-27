#!/usr/bin/env python
"""Run the screens of docs/DATA_PLAN.md section 7 over the pools (pass p1_pools) and the rendered rows (pass p2_rows):
D1 canary, D2 repository exclusion, D4 structural, dedup (exact keys, MinHash-LSH), lineage (cross-pool and train-vs-dev)
and, when the sealed artefacts exist, the D3 / D5 sealed inputs-only screen.

Run from the repository root:
    .venv/train/bin/python scripts/data/run_screens.py --pass-name p1_pools [--pools swe_teacher,terminal_teacher,retention]
        [--pool NAME=PATH ...] [--scan PATH ...] [--dev-frozen] [--lsh-bands 25 --lsh-rows 10] [--no-github]
    .venv/train/bin/python scripts/data/run_screens.py --pass-name p2_rows --pools '' --rows 'data/processed/sft/*.parquet'

Inputs
  pool items   --pool NAME=PATH (repeatable; PATH is a .jsonl or .parquet file, or a directory whose *.jsonl and *.parquet
               files are read in name order). Without --pool, for each name in --pools: <pools-dir>/<name>/tasks.jsonl, else
               <pools-dir>/<name>/tasks.parquet (absent pools are logged and skipped). One record per task; records that
               repeat a task_id (several episodes of one task) are merged, the first record's fields win. Fields:
                 task_id      "<source>:<upstream_id>" (required; the source code decides the kind when `kind` is absent)
                 kind         swe | terminal | prompt (else from row_kind / family / the source code)
                 text         the issue text (swe), the task instruction or the raw first Terminus-2 user message (terminal;
                              the harness template is cut away), the prompt (prompt); also read from problem_statement,
                              instruction, prompt, extra.problem_statement, extra.text
                 repo, upstream_id, image, pr, base_commit   (D2, lineage)
                 patch, touched_files                        (exact patch and (repo, commit, files) keys, fix families)
                 split        train by default; ids listed in data/manifests/dev_*.json count as dev
  rendered rows --rows GLOB (repeatable): Parquet files of apex_train.data_format rows with the provenance columns
               (input_ids decoded with --tokenizer, default the base model); screened by D1, D2 (the row's `repo` column and
               the signatures in the decoded text) and the D3 n-gram rules (no cosine rule on full rows).
  --scan PATH  D1 file scan of raw dumps and task trees (directory, .zip, .parquet, .jsonl); reported, never dropping.
  Also read: data/manifests/dev_*.json (dev ids and `frozen`), data/manifests/decontam_exclusions.json (D2 repositories),
  configs/eval/{swebench_pro_hard51_ids.txt,swebench_pro_v2_hard.yaml,terminal_bench_task_ids.txt} (public names for
  D2 and D4), data/decontam/sealed/ (D3 only when its manifest exists; never written here), GITHUB_TOKEN from the
  environment or .env (D2 fork graph; recorded as not_run without it or with --no-github).
Outputs
  WORK/<run_id>/screens/ (or --work-dir): d1_hits.jsonl, d2_hits.jsonl, d4_hits.jsonl, dedup_pairs.jsonl,
  lineage_drops.jsonl, d3_decisions.jsonl, marks.jsonl (one line per item: marks, drop reasons, `dropped`);
  data/manifests/screens_<pass-name>.json (parameters, input sha256s, per-pool survival table, counts per screen, every
  dropped id with its rules, audit samples of <= 50 ids per rule); data/decontam/report.json (D1, D2, D4 and D3 counts,
  merged per pass; counts only); logs/<run_id>/ (data_screens_<UTC>). Pool files are never rewritten: consumers read the
  dropped ids from the screens manifest.
Drop rules: d1_canary, d2_locked, d2_excluded_repo, d4_structural, exact_text, exact_patch, exact_repo_commit_files,
  near_dup_swe, near_dup_terminal, near_dup_prompt, cross_split_dev, cross_split_train, lineage_repo_pr,
  lineage_repo_commit, d3_ngram13, d3_ngram8_50pct, d3_cosine085. Reported without dropping (plan 7 names no drop for
  them): fix families, items that are > 50 % shared boilerplate, D1 file-scan hits, SWE-smith (repo, commit, files)
  collisions. Ids only, never text; nothing is deleted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from apex_data import paths  # noqa: E402
from apex_data.schema import read_jsonl, write_jsonl  # noqa: E402
from apex_data.pools.common import repo_key  # noqa: E402
from apex_data.screens import d1_canary, d2_repos, dedup, licence, lineage  # noqa: E402
from apex_data.screens import d3_match as dm  # noqa: E402
from apex_data.screens import d4_structural as d4  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, sha256_file, utc_now, write_json  # noqa: E402

DEFAULT_POOLS = ("swe_teacher", "terminal_teacher", "retention")
KINDS = ("swe", "terminal", "prompt")
KIND_BY_SOURCE = {"oswt": "swe", "swesmith": "swe", "th3k": "terminal", "tgr": "terminal", "tgr_env": "terminal",
                  "nm_math": "prompt", "nm_ifchat": "prompt", "nm_sci": "prompt", "devqa": "prompt"}
KIND_BY_FAMILY = {"swe": "swe", "terminal": "terminal", "qa": "prompt", "chat": "prompt", "swe_episode": "swe",
                  "terminal_turn": "terminal", "retention": "prompt"}
TEXT_KEYS = ("text", "problem_statement", "instruction", "prompt")
ITEM_FIELDS = ("task_id", "kind", "row_kind", "family", "source", *TEXT_KEYS, "repo", "upstream_id", "image", "pr",
               "base_commit", "patch", "touched_files", "split", "stratum", "extra")
ROW_ID_COLUMNS = ("task_id", "sample_id", "repo", "row_kind")
FILES_KEY_SOURCES = ("oswt",)   # real-issue rows: (repo, base_commit, touched_files) identifies one fix; in SWE-smith many
                                # distinct synthetic bugs share one file at the mirror commit, so there it is only counted
NEAR_CFG = {"swe": (dedup.ISSUE_NGRAM, dedup.ISSUE_THRESHOLD, 0, True),
            "terminal": (dedup.TERMINAL_NGRAM, dedup.TERMINAL_THRESHOLD, 0, False),
            "prompt": (dedup.PROMPT_NGRAM, dedup.PROMPT_THRESHOLD, dedup.PROMPT_EXACT_ONLY_UNDER, True)}  # ngram, threshold, min tokens, boilerplate
AUDIT_SAMPLE = 50


# --- items ---------------------------------------------------------------------------------------------------------------
def _first_text(rec: dict[str, Any]) -> str:
    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    for value in [*(rec.get(k) for k in TEXT_KEYS), extra.get("problem_statement"), extra.get("text")]:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalise_item(rec: dict[str, Any]) -> dict[str, Any]:
    """A pool record -> the screen item (kind, text and the identity fields; see --help)."""
    task_id = str(rec["task_id"])
    source = str(rec.get("source") or task_id.split(":", 1)[0])
    kind = rec.get("kind") or KIND_BY_FAMILY.get(str(rec.get("row_kind") or rec.get("family") or "")) or KIND_BY_SOURCE.get(source)
    if kind not in KINDS:
        raise ValueError(f"{task_id}: cannot tell the item kind (set `kind` to one of {KINDS})")
    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    text = _first_text(rec)
    return {"task_id": task_id, "source": source, "kind": kind,
            "text": dedup.terminal_task_text(text) if kind == "terminal" else text,
            "repo": rec.get("repo"), "upstream_id": rec.get("upstream_id") or task_id.split(":", 1)[-1],
            "image": rec.get("image"), "pr": rec.get("pr"), "base_commit": rec.get("base_commit"),
            "patch": rec.get("patch") or extra.get("patch") or extra.get("gold_patch"),
            "touched_files": rec.get("touched_files") or extra.get("touched_files"),
            "split": rec.get("split") or "train", "stratum": rec.get("stratum")}


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if c in ITEM_FIELDS]
        for batch in pf.iter_batches(batch_size=1024, columns=cols):
            yield from batch.to_pylist()
    else:
        raise ValueError(f"unsupported pool file {path} (use .jsonl or .parquet)")


def pool_files(spec: Path) -> list[Path]:
    if spec.is_dir():
        return sorted(p for p in spec.iterdir() if p.suffix in (".jsonl", ".parquet") and p.is_file())
    return [spec] if spec.exists() else []


def resolve_pools(a: argparse.Namespace) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for spec in a.pool:
        name, sep, path = spec.partition("=")
        if not sep or not name:
            raise SystemExit(f"--pool expects NAME=PATH, got {spec!r}")
        out[name] = pool_files(Path(path))
    for name in (n for n in a.pools.split(",") if n and n not in out):
        for cand in (a.pools_dir / name / "tasks.jsonl", a.pools_dir / name / "tasks.parquet"):
            if cand.exists():
                out[name] = [cand]
                break
        else:
            out[name] = []
    return out


def load_pools(specs: dict[str, list[Path]], run: RunDir) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """(items per pool, raw records per pool for the D1 scan, input descriptions)."""
    items, raw, inputs = {}, {}, {}
    seen: dict[str, str] = {}
    for name, files in specs.items():
        if not files:
            run.log(f"pool {name}: no input file, skipped")
            continue
        merged: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        described = []
        for f in files:
            n = 0
            for rec in read_records(f):
                records.append(rec)
                n += 1
                item = normalise_item(rec)
                if item["task_id"] in merged:
                    merged[item["task_id"]]["n_records"] += 1
                    continue
                if item["task_id"] in seen:
                    raise SystemExit(f"task_id {item['task_id']} appears in pools {seen[item['task_id']]} and {name}")
                seen[item["task_id"]] = name
                merged[item["task_id"]] = {**item, "pool": name, "n_records": 1}
            described.append({"path": str(f), "sha256": sha256_file(f), "n_records": n})
        items[name], raw[name] = list(merged.values()), records
        inputs[name] = {"files": described, "n_records": len(records), "n_items": len(merged),
                        "by_kind": dict(Counter(i["kind"] for i in merged.values()))}
        run.log(f"pool {name}: {len(merged)} items from {len(records)} records in {len(files)} file(s)")
    return items, raw, inputs


def iter_row_items(patterns: list[str], tokenizer_dir: Path, run: RunDir) -> Iterator[dict[str, Any]]:
    """Rendered rows decoded back to text: {id, task_id, repo, text, file}."""
    import pyarrow.parquet as pq

    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        return
    from apex_train.data_format import load_tokenizer

    tok = load_tokenizer(tokenizer_dir)
    for f in files:
        pf = pq.ParquetFile(f)
        cols = ["input_ids", *(c for c in ROW_ID_COLUMNS if c in pf.schema_arrow.names)]
        i = 0
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            for row in batch.to_pylist():
                rid = str(row.get("sample_id") or f"{Path(f).name}:{i}")
                yield {"id": f"row:{rid}", "task_id": row.get("task_id"), "repo": row.get("repo"),
                       "text": tok.decode(row["input_ids"]), "file": f}
                i += 1
        run.log(f"rows {f}: {i} rows decoded")


# --- marks ---------------------------------------------------------------------------------------------------------------
class Marks:
    """Per-item marks and drop reasons (ids only)."""

    def __init__(self) -> None:
        self.m: dict[str, dict[str, Any]] = defaultdict(lambda: {"drop_reasons": []})

    def mark(self, item_id: str, **fields: Any) -> None:
        self.m[item_id].update(fields)

    def drop(self, item_id: str, rule: str, **info: Any) -> None:
        self.m[item_id]["drop_reasons"].append({"rule": rule, **info})

    def rules(self, item_id: str) -> set[str]:
        return {d["rule"] for d in self.m[item_id]["drop_reasons"]} if item_id in self.m else set()


def write_lines(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    return write_jsonl(path, list(rows))


# --- stages --------------------------------------------------------------------------------------------------------------
def stage_d1(raw: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]], scan_paths: list[Path], workers: int,
             drop_rules: set[str], out_dir: Path, marks: Marks, run: RunDir) -> dict[str, Any]:
    results: dict[str, d1_canary.ScanResult] = {}
    for name, records in raw.items():
        results[f"pool:{name}"] = d1_canary.scan_records(records, source=f"pool:{name}", id_keys=("task_id",))
    if rows:
        results["rows"] = d1_canary.scan_records(({"id": r["id"], "text": r["text"]} for r in rows), source="rows", id_keys=("id",))
    for sp in scan_paths:
        t0 = time.time()
        results[f"scan:{sp}"] = d1_canary.scan_path(Path(sp), workers=workers)
        run.log(f"d1 scan {sp}: {results[f'scan:{sp}'].counts()} in {time.time() - t0:.0f}s")
    hits = [h for r in results.values() for h in r.hits]
    d1_canary.write_hits(hits, out_dir / "d1_hits.jsonl")
    rules_of: dict[str, set[str]] = defaultdict(set)
    for h in hits:
        if (h.path.startswith("pool:") or h.path == "rows") and h.record:
            rules_of[h.record].add(f"{h.rule}:{h.level}")
    for item_id, rules in rules_of.items():
        marks.mark(item_id, d1={"hit": True, "rules": sorted(rules)})
        if any(r.split(":")[0] in drop_rules for r in rules):
            marks.drop(item_id, "d1_canary")
    return {"drop_rules": sorted(drop_rules), "n_items_dropped": sum(1 for r in rules_of.values()
                                                                      if any(x.split(":")[0] in drop_rules for x in r)),
            **{name: r.counts() for name, r in results.items()}}


def stage_d2(items: list[dict[str, Any]], locked: d2_repos.LockedRepos, lookup: licence.RepoLookup, out_dir: Path,
             marks: Marks, run: RunDir) -> dict[str, Any]:
    repos = sorted({k for it in items if (k := repo_key(it.get("repo")))})
    infos, lookup_summary = lookup.lookup(repos, log=run.log) if repos else ({}, {"requested": 0})
    results = []
    hit_lines = []
    for it in items:
        r = d2_repos.screen(it.get("repo"), locked=locked, text=it["text"], upstream_id=it.get("upstream_id"), image=it.get("image"))
        if repo_key(it.get("repo")):
            r.add_fork_graph(infos.get(repo_key(it.get("repo"))), locked, lookup_enabled=lookup.enabled)
        results.append(r)
        marks.mark(it["id"], d2=r.as_mark())
        if r.locked_hit:
            marks.drop(it["id"], "d2_locked", mechanisms=[m for m in r.mechanisms if d2_repos.kind_of(m) in d2_repos.LOCKED_KINDS])
        if r.excluded_hit:
            marks.drop(it["id"], "d2_excluded_repo", mechanisms=[m for m in r.mechanisms if d2_repos.kind_of(m) in d2_repos.EXCLUDED_KINDS])
        if r.mechanisms:
            hit_lines.append({"id": it["id"], "repo": it.get("repo"), "mechanisms": r.mechanisms})
    write_lines(out_dir / "d2_hits.jsonl", hit_lines)
    return {**d2_repos.counts(results), "locked_sha256": locked.sha256, "lists": locked.sources,
            "union_sources": d2_repos.UNION_SOURCES, "repo_lookup": lookup_summary,
            "fork_step": "run" if lookup.enabled else "not_run (no GITHUB_TOKEN or --no-github; cache only)"}


def stage_d4(items: list[dict[str, Any]], out_dir: Path, marks: Marks) -> dict[str, Any]:
    pats = d4.slug_patterns(d4.load_tb_slugs())
    lines, n_hit, by_reason, flags = [], 0, Counter(), Counter()
    terminal = [it for it in items if it["kind"] == "terminal"]
    for it in terminal:
        r = d4.screen_text(it["text"], pats, task_name=it.get("upstream_id"))
        d4_reasons = [x for x in r.reasons() if x.startswith("d4:")]
        flags.update({"boilerplate_lines": r.boilerplate_lines, "tb_name_mention": len(r.tb_name_mention_files)})
        marks.mark(it["id"], d4={"hit": r.d4_hit, "reasons": d4_reasons})
        if r.d4_hit:
            n_hit += 1
            by_reason.update(x.split(":")[1] for x in d4_reasons)
            marks.drop(it["id"], "d4_structural", reasons=d4_reasons)
            lines.append({"id": it["id"], "reasons": d4_reasons})
    write_lines(out_dir / "d4_hits.jsonl", lines)
    return {"n_terminal_items": len(terminal), "n_hit": n_hit, "by_reason": dict(by_reason), "recorded_flags": dict(flags)}


def _resolve_groups(groups: Iterable[list[str]], rule: str, split_of: dict[str, str], dev_frozen: bool, marks: Marks) -> dict[str, int]:
    """Within a group: the cross-split rule first, then one representative per split (smallest id) is kept."""
    n_groups = n_train = n_dev = n_cross = 0
    for members in groups:
        members = sorted(set(members))
        if len(members) < 2:
            continue
        n_groups += 1
        for m in members:
            marks.mark(m, dedup_group=members[0])
        for d in dedup.resolve_cross_split([members], split_of, dev_frozen, rule):
            marks.drop(d.id, f"cross_split_{d.split}", matched_id=d.matched_id, via=rule)
            n_cross += 1
        train = [m for m in members if split_of.get(m, "train") != "dev"]
        dev = [m for m in members if split_of.get(m, "train") == "dev"]
        for m in train[1:]:
            marks.drop(m, rule, kept_id=train[0])
            n_train += 1
        if not dev_frozen:
            for m in dev[1:]:
                marks.drop(m, rule, kept_id=dev[0])
                n_dev += 1
    return {"n_groups": n_groups, "n_train_drops": n_train, "n_dev_drops": n_dev, "n_cross_split_drops": n_cross}


def stage_dedup(items: list[dict[str, Any]], split_of: dict[str, str], dev_frozen: bool, bands: int, rows: int,
                out_dir: Path, marks: Marks, run: RunDir) -> dict[str, Any]:
    hasher = dedup.MinHasher()
    counts: dict[str, Any] = {"params": {"bands": bands, "rows": rows, "num_perm": dedup.NUM_PERM,
                                         "near": {k: {"ngram": v[0], "threshold": v[1], "exact_only_under": v[2], "boilerplate_stripped": v[3]}
                                                  for k, v in NEAR_CFG.items()}}}
    # exact keys; the terminal text is already the cut Terminus-2 task text, so exact_text_key is its task-text key
    text_keys = {it["id"]: dedup.exact_text_key(it["text"]) for it in items}
    patch_keys = {it["id"]: dedup.patch_key(it["patch"]) for it in items if it.get("patch")}
    rcf_keys, rcf_outside = {}, {}
    for it in items:
        files = it.get("touched_files") or dedup.touched_files(it.get("patch"))
        key = dedup.repo_commit_files_key(it.get("repo"), it.get("base_commit"), files)
        if key:
            (rcf_keys if it["source"] in FILES_KEY_SOURCES else rcf_outside)[it["id"]] = f"{it['source']}:{key}"
    for rule, keys in (("exact_text", text_keys), ("exact_patch", patch_keys), ("exact_repo_commit_files", rcf_keys)):
        groups = dedup.exact_groups(keys)
        counts[rule] = {"n_keyed": sum(1 for v in keys.values() if v), **_resolve_groups(groups.values(), rule, split_of, dev_frozen, marks)}
    outside = dedup.exact_groups(rcf_outside)
    counts["exact_repo_commit_files"].update({"sources": list(FILES_KEY_SOURCES), "outside_scope_reported_only": {
        "n_keyed": len(rcf_outside), "n_groups": len(outside), "n_items_in_groups": sum(len(v) for v in outside.values())}})
    pairs_out = []
    for kind, (ngram, threshold, min_tokens, strip) in NEAR_CFG.items():
        kind_items = [dedup.DedupItem(it["id"], it["text"], block=it.get("repo"), split=split_of[it["id"]]) for it in items if it["kind"] == kind]
        if not kind_items:
            continue
        t0 = time.time()
        bp = dedup.BoilerplateIndex.build(dedup.tokenize(x.text) for x in kind_items) if strip else None
        reps = dedup.build_representations(kind_items, ngram, bp)
        n_flagged = sum(r.boilerplate_flag for r in reps)
        res = dedup.near_duplicates(reps, threshold, bands=bands, rows=rows, hasher=hasher, min_tokens=min_tokens)
        rule = f"near_dup_{kind}"
        pairs_out.extend({"rule": rule, "a": x, "b": y, "jaccard": j} for x, y, j in res.pairs)
        counts[rule] = {**res.counts(), "ngram": ngram, "threshold": threshold,
                        "boilerplate": {**bp.counts(), "n_items_gt50_reported": n_flagged} if bp else None,
                        **_resolve_groups(res.groups.values(), rule, split_of, dev_frozen, marks), "seconds": round(time.time() - t0, 1)}
        run.log(f"dedup {rule}: {counts[rule]}")
    patch_items = [dedup.DedupItem(it["id"], dedup.canonical_patch(it["patch"]), block=lineage.normalise_repo(it.get("repo")))
                   for it in items if it.get("patch") and it.get("repo")]
    if patch_items:
        ff = dedup.fix_families(dedup.build_representations(patch_items, dedup.ISSUE_NGRAM), hasher=hasher)
        for tid, fam in ff.family_of.items():
            marks.mark(tid, fix_family=fam)
        pairs_out.extend({"rule": "fix_family", "a": x, "b": y, "jaccard": j} for x, y, j in ff.pairs)
        counts["fix_family_reported_only"] = {**ff.counts(), "n_patches": len(patch_items), "threshold": dedup.FIX_FAMILY_THRESHOLD}
    write_lines(out_dir / "dedup_pairs.jsonl", pairs_out)
    return counts


def stage_lineage(items: list[dict[str, Any]], dev_records: list[lineage.LineageRecord], dev_frozen: bool, split_of: dict[str, str],
                  out_dir: Path, marks: Marks) -> dict[str, Any]:
    records: list[lineage.LineageRecord] = []
    for it in items:
        r = lineage.from_task(it)
        records.append(lineage.LineageRecord(r.task_id, r.source, split_of[it["id"]], r.repo, r.pr, r.base_commit))
    pool_ids = {r.task_id for r in records}
    records.extend(r for r in dev_records if r.task_id not in pool_ids)
    groups = lineage.lineage_groups(records)
    drops = lineage.resolve(groups, dev_frozen=dev_frozen)
    for (kind, key), members in groups.items():
        for m in members:
            if m.task_id in pool_ids:
                marks.m[m.task_id].setdefault("lineage_keys", []).append(f"{kind}:{key}")
    for d in drops:
        if d.task_id in pool_ids:
            marks.drop(d.task_id, f"lineage_{d.rule}", kept_id=d.kept_id, reason=d.reason)
    write_lines(out_dir / "lineage_drops.jsonl", ({"task_id": d.task_id, "split": d.split, "rule": d.rule, "kept_id": d.kept_id,
                                                   "reason": d.reason} for d in drops))
    return lineage.counts(records, groups, drops)


def stage_d3(items: list[dict[str, Any]], rows: list[dict[str, Any]], a: argparse.Namespace, out_dir: Path, marks: Marks,
             run: RunDir) -> dict[str, Any]:
    index = dm.SealedIndex.load(a.sealed_dir, allow_fixture=a.allow_fixture_sealed)
    embedder = None if a.no_embeddings else dm.Embedder(a.model_dir)
    matcher = dm.Matcher(index, embedder)
    screened = [dm.Item(it["id"], it["text"], kind="prompt" if it["kind"] == "prompt" else "task", stratum=it.get("stratum"))
                for it in items if it["text"]]
    screened += [dm.Item(r["id"], r["text"], kind="row") for r in rows]
    t0 = time.time()
    decisions = matcher.screen(screened)
    for d in decisions:
        marks.m[d.item_id].setdefault("d3", []).append({"benchmark": d.benchmark, "rule": d.rule, "benchmark_item": d.benchmark_item, "score": d.score})
        marks.drop(d.item_id, f"d3_{d.rule}", benchmark=d.benchmark)
    write_lines(out_dir / "d3_decisions.jsonl", dm.decisions_to_dicts(decisions))
    counts = {**dm.summarise(decisions, len(screened)), "n_task_items": len(screened) - len(rows), "n_row_items": len(rows),
              "benchmarks": sorted(index.benchmarks), "sealed_manifest_sha256": index.manifest_sha256,
              "sealed_fixture": bool(index.manifest.get("fixture")), "embeddings": embedder is not None,
              "embedding_model": embedder.info if embedder else None, "seconds": round(time.time() - t0, 1),
              "rules": {"ngram13": "any shared 13-gram", "ngram8_50pct": f">= {dm.SHORT_FRACTION:.0%} of >= {dm.MIN_SHORT_NGRAMS} 8-grams in one item",
                        "cosine085": f"cosine >= {dm.COSINE_THRESHOLD} (task and prompt items only)"}}
    run.log(f"d3: {counts['items_hit']} of {len(screened)} items hit; by rule {counts['by_rule']}")
    return counts


# --- reports -------------------------------------------------------------------------------------------------------------
def survival(groups: dict[str, list[str]], marks: Marks) -> dict[str, Any]:
    out = {}
    for name, ids in groups.items():
        by_rule: Counter[str] = Counter()
        n_dropped = 0
        for i in ids:
            rules = marks.rules(i)
            by_rule.update(rules)
            n_dropped += bool(rules)
        out[name] = {"n_in": len(ids), "n_dropped": n_dropped, "n_out": len(ids) - n_dropped, "by_rule": dict(sorted(by_rule.items()))}
    return out


def audit_samples(marks: Marks, seed: int = paths.SEED) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    by_rule: dict[str, list[str]] = defaultdict(list)
    for item_id in marks.m:
        for r in marks.rules(item_id):
            by_rule[r].append(item_id)
    return {rule: sorted(np.array(sorted(ids))[rng.permutation(len(ids))[:AUDIT_SAMPLE]].tolist()) for rule, ids in sorted(by_rule.items())}


def load_dev(manifests_dir: Path, run: RunDir) -> tuple[list[lineage.LineageRecord], bool]:
    records, frozen = [], False
    for p in sorted(manifests_dir.glob("dev_*.json")):
        recs, fz = lineage.load_manifest(p)
        records.extend(recs)
        frozen = frozen or fz
        run.log(f"dev manifest {p.name}: {len(recs)} ids, frozen={fz}")
    return records, frozen


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pass-name", default="p1_pools", help="pass name in data/decontam/report.json and the screens manifest name")
    p.add_argument("--pools", default=",".join(DEFAULT_POOLS), help="pool names under --pools-dir ('' for none)")
    p.add_argument("--pools-dir", type=Path, default=paths.POOLS)
    p.add_argument("--pool", action="append", default=[], help="NAME=PATH pool item file or directory (repeatable; overrides --pools-dir)")
    p.add_argument("--rows", action="append", default=[], help="glob of rendered-row Parquet files (repeatable)")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR, help="tokenizer directory to decode --rows input_ids")
    p.add_argument("--scan", type=Path, action="append", default=[], help="D1 file scan root (directory, .zip, .parquet, .jsonl); repeatable")
    p.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 4)), help="processes for the D1 file scan")
    p.add_argument("--d1-drop-rules", default=",".join(d1_canary.RULES), help="D1 rules whose hit on an item drops it (default all four)")
    p.add_argument("--manifests-dir", type=Path, default=paths.MANIFESTS, help="dev_*.json are read here; screens_<pass>.json is written here")
    p.add_argument("--lsh-bands", type=int, default=dedup.DEFAULT_BANDS, help="MinHash-LSH bands (plan: 25)")
    p.add_argument("--lsh-rows", type=int, default=dedup.DEFAULT_ROWS, help="rows per band (plan: 10); bands x rows <= 250")
    p.add_argument("--dev-frozen", action="store_true", help="overlaps drop the training item (also read from the dev manifests' `frozen`)")
    p.add_argument("--no-github", action="store_true", help="D2 fork graph from the cache only (recorded as not_run where uncached)")
    p.add_argument("--repo-cache", type=Path, default=licence.DEFAULT_CACHE_DIR, help="GitHub repository answers cache")
    p.add_argument("--sealed-dir", type=Path, default=paths.SEALED, help="sealed benchmark artefacts; D3 is skipped when its manifest is absent")
    p.add_argument("--allow-fixture-sealed", action="store_true", help="TESTS ONLY: accept a fixture-mode sealed directory")
    p.add_argument("--no-embeddings", action="store_true", help="D3 n-gram rules only (no cosine rule; recorded in the report)")
    p.add_argument("--model-dir", type=Path, default=dm.BGE_MODEL_DIR, help="bge-small-en-v1.5 directory for the cosine rule")
    p.add_argument("--report", type=Path, default=dm.REPORT_PATH, help="decontamination counts report (tracked)")
    p.add_argument("--work-dir", type=Path, default=None, help="default WORK/<run_id>/screens")
    p.add_argument("--run-root", type=Path, default=None, help="TESTS ONLY: write logs/ under this root instead of the repository")
    a = p.parse_args(argv)
    if a.lsh_bands * a.lsh_rows > dedup.NUM_PERM:
        p.error(f"--lsh-bands x --lsh-rows must be <= {dedup.NUM_PERM}")
    a.drop_rules = {r for r in a.d1_drop_rules.split(",") if r}
    if a.drop_rules - set(d1_canary.RULES):
        p.error(f"--d1-drop-rules must be a subset of {d1_canary.RULES}")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = new_run_id("data_screens")
    work = a.work_dir or (paths.WORK / run_id / "screens")
    t_start = time.time()
    with RunDir(run_id, root=a.run_root, log_name="train.log") as run:
        config = {k: (str(v) if isinstance(v, Path) else sorted(v) if isinstance(v, set) else v) for k, v in vars(a).items()}
        run.write_metadata(run_id=run_id, step="run_screens", command=command_line(), code=git_info(), hardware=hardware_info(),
                           config=config, seed=paths.SEED, start_time=utc_now(), status="running", work_dir=str(work))
        work.mkdir(parents=True, exist_ok=True)
        pools, raw, inputs = load_pools(resolve_pools(a), run)
        rows = list(iter_row_items(a.rows, a.tokenizer, run)) if a.rows else []
        for name, f_rows in Counter(r["file"] for r in rows).items():
            inputs[f"rows:{name}"] = {"files": [{"path": name, "sha256": sha256_file(Path(name)), "n_records": f_rows}]}
        items = [{**it, "id": it["task_id"]} for its in pools.values() for it in its]
        dev_records, frozen = load_dev(a.manifests_dir, run)
        dev_frozen = a.dev_frozen or frozen
        dev_ids = {r.task_id for r in dev_records}
        split_of = {it["id"]: "dev" if it["id"] in dev_ids else it["split"] for it in items}
        marks = Marks()
        token = None if a.no_github else licence.github_token()
        if token is None:
            run.log(licence.NOTICE_NO_TOKEN)
        lookup = licence.RepoLookup(a.repo_cache, token=token)
        locked = d2_repos.load_locked()
        counts: dict[str, Any] = {"dev_frozen": dev_frozen, "n_dev_manifest_ids": len(dev_ids), "n_items": len(items), "n_rows": len(rows)}
        run.log(f"run_id={run_id} pass={a.pass_name} pools={list(pools)} items={len(items)} rows={len(rows)} dev_frozen={dev_frozen} "
                f"lsh={a.lsh_bands}x{a.lsh_rows} fork_graph={'api' if lookup.enabled else 'cache only'}")

        counts["d1"] = stage_d1(raw, rows, a.scan, a.workers, a.drop_rules, work, marks, run)
        counts["d2"] = stage_d2(items + [{**r, "kind": "row"} for r in rows], locked, lookup, work, marks, run)
        if items:
            counts["d4"] = stage_d4(items, work, marks)
            counts["dedup"] = stage_dedup(items, split_of, dev_frozen, a.lsh_bands, a.lsh_rows, work, marks, run)
            counts["lineage"] = stage_lineage(items, dev_records, dev_frozen, split_of, work, marks)
        if dm.SealedIndex.exists(a.sealed_dir):
            counts["d3"] = stage_d3(items, rows, a, work, marks, run)
        else:
            counts["d3"] = {"skipped": f"no sealed manifest under {a.sealed_dir} (written by scripts/decontam/seal_benchmark_inputs.py)"}
            run.log(f"d3 skipped: {counts['d3']['skipped']}")
        for stage in ("d1", "d2", "d4", "dedup", "lineage", "d3"):
            if stage in counts:
                run.metric({"stage": stage, **{k: v for k, v in counts[stage].items() if k not in ("params", "lists", "union_sources", "rules", "embedding_model")}})

        groups = {name: [it["task_id"] for it in its] for name, its in pools.items()}
        if rows:
            groups["rows"] = [r["id"] for r in rows]
        surv = survival(groups, marks)
        all_ids = [i for ids in groups.values() for i in ids]
        dropped = {i: sorted(marks.rules(i)) for i in all_ids if marks.rules(i)}
        write_lines(work / "marks.jsonl", ({"id": i, "split": split_of.get(i, "train"), "dropped": i in dropped,
                                            **marks.m.get(i, {"drop_reasons": []})} for i in all_ids))
        decontam = {k: counts[k] for k in ("d1", "d2", "d4", "d3") if k in counts}
        dm.write_report(a.pass_name, {"run_id": run_id, "survival": surv, **decontam}, a.report)
        manifest = {"run_id": run_id, "pass_name": a.pass_name, "written_at": utc_now(), "work_dir": str(work),
                    "params": {"lsh_bands": a.lsh_bands, "lsh_rows": a.lsh_rows, "dev_frozen": dev_frozen, "num_perm": dedup.NUM_PERM,
                               "d1_drop_rules": sorted(a.drop_rules), "scan": [str(s) for s in a.scan], "rows": a.rows},
                    "inputs": inputs, "survival": surv, "counts": counts, "dropped": dropped,
                    "audit_samples": audit_samples(marks), "audit_sample_size": AUDIT_SAMPLE}
        a.manifests_dir.mkdir(parents=True, exist_ok=True)
        write_json(a.manifests_dir / f"screens_{a.pass_name}.json", manifest)
        duration = time.time() - t_start
        run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock on the CPU VM"),
                             ("orchestrator_cpus", os.cpu_count(), "vcpu", "apex-cpu"),
                             ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
                             ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(duration, 1), status="finished", summary={"survival": surv},
                           manifest=str(a.manifests_dir / f"screens_{a.pass_name}.json"))
        run.finalize()
        run.log(f"survival: {json.dumps(surv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
