#!/usr/bin/env python
"""Build the SWE teacher slice (WP3, unit U2.1, milestone M2.1): Open-SWE-Traces v1.2 mini-swe-agent Qwen3.8-27B episodes
from both task-source folders, re-rendered under our mini-swe-agent 2.4.6 harness templates, filtered (data plan 3.2
filters 1 to 6), and drawn to the language quotas with the size-stratified rule (filters 7 and 8) separately at the 64K
target cap and the 48K fallback cap. Rows are rendered once (at the 64K cap, recording fits_48k) for the union of both
selections. Spec: docs/DATA_PLAN.md 3.2, 4, 6, 7, 8; src/apex_data/README.md (source `oswt`, provenance fields).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_teacher_leg.py [--workers 24] [--limit-shards N]
        [--quota-scale F] [--out-dir data/pools/swe_teacher] [--repo-root /app] [--keep-selections DIR]
        [--trajectory-choice {random_not_shortest,fewest_turns_verified}] [--features PATH]
        [--resume RUN_ID] [--draw-from RUN_ID] [--max-turn-tokens N] [--no-fork-graph] [--seed 42]

Repository path: the teacher's episodes assume the repository at /testbed; our SWE-bench Pro harness starts the agent with
it at /app and the prompt never names it. --repo-root (default /app; /testbed keeps the teacher's path) rewrites the
/testbed path token in every message before any filter reads it (apex_data.teacher.render.RootRewriter); an episode
whose commands make the rewrite unsafe is dropped (`repo_root_ambiguous`).
--keep-selections DIR keeps every row of DIR/selection_{64k,48k}.json that still passes every filter (same trajectory,
part and phase) and draws only the vacated slots, through the same walks, quotas and caps (filter.draw_cap `pinned`).

Trajectory choice (which gate-passing trajectory within the cap represents an admitted instance; admission, quotas,
strata, caps, the lazy walk and the kept rows do not depend on it): `random_not_shortest` (default, the plan's rule) or
`fewest_turns_verified` (fewest assistant turns among the trajectories that ran a test after their last edit,
apex_data.teacher.verify; random_not_shortest when none did; kept rows keep their instance and get their trajectory by
this rule). The second needs the test-after-edit features of every gate-passing trajectory: read from the raw rows once
and cached in --features (default WORK/<run_id>/evals/trajectory_features.jsonl).
--max-turn-tokens N fails a gate-passing trajectory whose rendered row has an assistant turn of more than N trained tokens
(the turn's contiguous assistant_masks span: reasoning, content, tool call, end token; stage `turn_length`, reason
`turn_too_long`), measured on the cached rendered rows, so the cache stays independent of N; default: no limit.
--draw-from RUN_ID draws again on the index, gate outcomes and rendered rows of WORK/<RUN_ID>/ (another trajectory
choice or output directory without rendering again): they are read, never written; a trajectory the draw needs and that
run has not evaluated is evaluated into this run's own WORK/<run_id>/ (e.g. the other trajectories of kept instances
when the trajectory choice is not random_not_shortest).

Inputs (read-only):
  RAW/oswt/data/minisweagent/qwen38_27b/{swe-rebench-v2,scale-swe}/*.parquet   teacher trajectories (pinned revision)
  RAW/swe_rebench_v2/data/*.parquet, RAW/scale_swe_verified/data/*.parquet     reference patches (size strata), the
                                                                               Scale-SWE-Verified admission list
  data/manifests/raw_sources.json        revisions of the three inputs
  data/manifests/decontam_exclusions.json, configs/eval/ (through apex_data.screens.d2_repos)   D2 and repository exclusions
  data/manifests/dev_exclusions.json     SWE dev repositories and task ids (filter 6; recorded as absent when missing)
  the base tokenizer and chat template (apex_data.paths.BASE_MODEL_DIR); mini-swe-agent 2.4.6 in .venv/harbor (templates
  and BASH_TOOL) and the runner-written harness config the templates are asserted against (--lock-config)
  GITHUB_TOKEN (environment or .env, never printed): D2 fork-graph lookups, cached under WORK/licence_cache/
Outputs:
  <out-dir>/rows.parquet          trainer rows (apex_train.data_format.ROW_COLUMNS) + schema.PROVENANCE_FIELDS + row_key,
                                  selected_64k, selected_48k, repo_root, repo_root_rewrites,
                                  repo_root_cd_prefixes_deleted: the union of both selections
                                  (git-ignored; sha256 in the
                                  manifest; a different previous file is renamed to rows.superseded_<UTC>.parquet)
  <out-dir>/selection_64k.json, selection_48k.json   the two draws: per row instance, trajectory, stratum, phase, tokens
  <out-dir>/manifest.json         funnel (n_in / n_dropped / n_out per filter, per subsource x language), quotas per cap,
                                  stratum shares, added-line and file-count histograms, tokens, D2 and exclusion counts,
                                  M2.1 checks, inputs, templates, revisions
  <out-dir>/dropped.jsonl.gz      every dropped trajectory with its stage and reason (nothing is deleted)
  WORK/<run_id>/                  index.parquet, evals/outcomes.jsonl and evals/rows/*.parquet (the gate outcome and the
                                  rendered row of every evaluated trajectory; --resume reuses them; with --draw-from only
                                  the trajectories that run lacks, and the index is read from there)
  --features                      with fewest_turns_verified: test-after-edit features per gate-passing trajectory (jsonl)
  logs/<run_id>/                  metadata.json, train.log, metrics.jsonl, errors.log, resources.csv (+ copies of the
                                  manifest and the selections)
Seeded (--seed, default apex_data.paths.SEED): the draw orders and the per-instance trajectory choice are hashes of the
seed; the evaluation order does not change a completed draw.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from apex_data import paths  # noqa: E402
from apex_data.pools.common import repo_key  # noqa: E402
from apex_data.render.adapters import AdapterError, adapt  # noqa: E402
from apex_data.render.filters import CAP_FALLBACK, CAP_TARGET, GATES, FilterConfig, filter_episodes, qwen3_roundtrip  # noqa: E402
from apex_data.schema import PROVENANCE_FIELDS, validate_provenance  # noqa: E402
from apex_data.screens import d2_repos  # noqa: E402
from apex_data.screens.licence import RepoLookup, github_token  # noqa: E402
from apex_data.teacher import filter as tfilter  # noqa: E402
from apex_data.teacher import load, render, verify  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import (ROW_COLUMNS, TURN_TOO_LONG, BuildOptions, SampleValidationError, arrow_schema,  # noqa: E402
                                    build_rows, encode_messages, histogram, load_tokenizer, turn_span_lengths)

CAPS = {"64k": CAP_TARGET, "48k": CAP_FALLBACK}
STAGES = ("rebuild", "repo_root_ambiguous", "reference_patch_leak", "d2_text", "length_precheck", *(f"gate_{g}" for g in GATES))
TURN_STAGE = "turn_length"       # --max-turn-tokens: applied after the gates, on the cached rendered rows
PROVENANCE_TYPES = {"patch_files": pa.int64(), "patch_added_lines": pa.int64(), "fits_48k": pa.bool_()}
EXTRA_COLUMNS = (("row_key", pa.string()), ("selected_64k", pa.bool_()), ("selected_48k", pa.bool_()),
                 ("repo_root", pa.string()), ("repo_root_rewrites", pa.int32()), ("repo_root_cd_prefixes_deleted", pa.int32()))
CACHE_SCHEMA = pa.schema([*arrow_schema(), ("trajectory_id", pa.string()), ("episode_json", pa.large_string())])
_W: dict[str, Any] = {}          # worker-process globals


# ------------------------------------------------------------------------------------------------ workers
def _worker_init(tokenizer_dir: str, bundle: dict[str, Any], rows_dir: str, hf_revision: str, leak_fraction: float,
                 repo_root: str) -> None:
    import transformers

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()          # the >262,144-token warning is expected for long candidates
    _W.update(tok=load_tokenizer(tokenizer_dir), bundle=render.TemplateBundle.from_dict(bundle), rows_dir=Path(rows_dir),
              locked=d2_repos.load_locked(), revision=hf_revision, leak_fraction=leak_fraction, repo_root=repo_root)


def _outcome(it: dict[str, Any], stage: str | None, reason: str | None, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"trajectory_id": it["trajectory_id"], "instance_id": it["instance_id"], "subsource": it["subsource"],
            "language": it["language"], "repo": it["repo"], "ok": stage is None, "stage": stage, "reason": reason,
            "detail": detail[:300], **extra}


def leaks(n_matched: int, n_lines: int, leak_fraction: float) -> bool:
    """The reference-patch rule: drop when some line matched and the matched share reaches `leak_fraction` (0 = any)."""
    return n_matched > 0 and n_matched / max(1, n_lines) >= leak_fraction


def evaluate_one(raw: dict[str, Any], it: dict[str, Any], tok: Any, bundle: render.TemplateBundle, locked: Any,
                 revision: str, max_tokens: int, leak_fraction: float = 0.0, repo_root: str | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One trajectory through the rendered-episode filters, every check on the rewritten text; returns (outcome record,
    cache row or None)."""
    task_id = f"{load.SOURCE}:{it['instance_id']}"
    try:
        ep, st = render.build_episode(raw, bundle, task_id=task_id, episode_id=it["trajectory_id"], repo_root=repo_root)
    except render.RenderDrop as exc:
        stage = "repo_root_ambiguous" if exc.reason == "repo_root_ambiguous" else "rebuild"
        return _outcome(it, stage, exc.reason, exc.detail), None
    ref = ((raw.get("metadata") or {}).get("reference_patch") or {}).get("patch")
    matched, n_added = render.reference_patch_overlap(render.prompt_texts(ep), ref)
    common_extra = {"rebuild": st, "reference_patch_lines": n_added, "reference_patch_lines_in_prompt": len(matched)}
    if leaks(len(matched), n_added, leak_fraction):
        return _outcome(it, "reference_patch_leak", "reference_patch_line_in_prompt",
                        f"{len(matched)} of {n_added} non-test added lines", **common_extra), None
    d2 = d2_repos.screen(it["repo"], locked=locked, text=render.episode_text(ep.messages), upstream_id=it["instance_id"])
    if d2.drop:
        return _outcome(it, "d2_text", "d2_locked_repository" if d2.locked_hit else "decontam_excluded_repository",
                        ",".join(d2.mechanisms), **common_extra), None
    try:                     # exact full-render length first: an over-cap episode skips the per-turn passes of the gates
        sample = adapt(ep).sample
        n = len(encode_messages(tok, sample.template_messages(), sample.tools, enable_thinking=True))
        if n > max_tokens:
            return _outcome(it, "length_precheck", "too_long", f"{n} > {max_tokens}", n_tokens=n, **common_extra), None
    except (AdapterError, SampleValidationError):
        pass                 # the adapt gate below records the reason
    task = {"task_id": task_id, "source": load.SOURCE, "family": "swe", "language": it["language"], "repo": it["repo"],
            "hf_revision": revision, "licence": {"repo_spdx": it["license"], "dataset": load.DATASET_LICENSE},
            "screens": {"d2": d2.as_mark()},
            "extra": {"provenance": {"hf_repo": load.HF_REPO, "generator": load.GENERATOR, "label_provenance": "verifier",
                                     "oswt_subsource": it["subsource"], "size_stratum": it["stratum"],
                                     "patch_files": it["files"], "patch_added_lines": it["added"]}}}
    verdict = {"r_pass": 1, "tamper_flags": [], "verifier": "upstream_resolved"}
    res = filter_episodes([ep], {ep.episode_id: verdict}, {task_id: task}, tok, FilterConfig(max_row_tokens=max_tokens))
    if res.dropped:
        d = res.dropped[0]
        n_tok = None
        if d.reason == "too_long":
            n_tok = int(d.detail.split(" > ")[0])
        return _outcome(it, f"gate_{d.gate}", d.reason, d.detail, n_tokens=n_tok, **common_extra), None
    kept = res.kept[0]
    row = {**kept.row, "trajectory_id": it["trajectory_id"],
           "episode_json": json.dumps({"messages": ep.messages, "tools": ep.tools, "extra": ep.extra}, ensure_ascii=False)}
    return _outcome(it, None, None, n_tokens=kept.row["n_tokens"], n_assistant_tokens=kept.row["n_assistant_tokens"],
                    n_assistant_turns=kept.row["n_assistant_turns"], provenance=kept.provenance, **common_extra), row


def _worker_eval(task: tuple[str, str, list[dict[str, Any]], int]) -> list[dict[str, Any]]:
    subsource, shard_path, items, max_tokens = task
    raws = load.read_rows(subsource, Path(shard_path), [it["row"] for it in items])
    outcomes, rows = [], []
    for it, raw in zip(items, raws, strict=True):
        if raw["trajectory_id"] != it["trajectory_id"]:
            raise RuntimeError(f"{shard_path} row {it['row']}: trajectory id {raw['trajectory_id']} != {it['trajectory_id']}")
        out, row = evaluate_one(raw, it, _W["tok"], _W["bundle"], _W["locked"], _W["revision"], max_tokens, _W["leak_fraction"],
                                _W["repo_root"])
        outcomes.append(out)
        if row is not None:
            rows.append(row)
    if rows:
        name = hashlib.sha256("|".join(sorted(r["trajectory_id"] for r in rows)).encode()).hexdigest()[:20]
        path = _W["rows_dir"] / f"{subsource}_{Path(shard_path).stem}_{name}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=CACHE_SCHEMA), tmp)
        tmp.replace(path)
        for o in outcomes:
            if o["ok"]:
                o["row_file"] = path.name
    return outcomes


def _worker_scan(job: tuple[str, str]) -> list[load.IndexRow]:
    return load.scan_shard(job[0], Path(job[1]))


def _worker_features(job: tuple[str, str, list[tuple[int, str]]]) -> list[dict[str, Any]]:
    subsource, shard_path, items = job
    raws = load.read_rows(subsource, Path(shard_path), [row for row, _ in items], columns=("trajectory_id", "messages"))
    for (row, t), raw in zip(items, raws, strict=True):
        if raw["trajectory_id"] != t:
            raise RuntimeError(f"{shard_path} row {row}: trajectory id {raw['trajectory_id']} != {t}")
    return [verify.trajectory_features(raw) for raw in raws]


# ------------------------------------------------------------------------------------------------ helpers
def shard_fingerprint(shards: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [{"subsource": s, "name": p.name, "bytes": p.stat().st_size, "mtime": int(p.stat().st_mtime)} for s, p in shards]


def build_or_load_index(shards: list[tuple[str, Path]], work: Path, workers: int, run: common.RunDir,
                        read_only: bool = False) -> list[load.IndexRow]:
    cache, meta = work / "index.parquet", work / "index_meta.json"
    fp = shard_fingerprint(shards)
    if cache.exists() and common.read_json(meta, {}).get("shards") == fp:
        index = load.index_from_table(pq.read_table(cache))
        run.log(f"index resumed from {cache} ({len(index)} rows, {len(shards)} shards)")
        return index
    if read_only:
        raise RuntimeError(f"{cache} is missing or was built from other shards; it is read-only here (--draw-from)")
    t0 = time.time()
    index: list[load.IndexRow] = []
    with mp.get_context("fork").Pool(min(workers, len(shards))) as pool:
        for rows in pool.imap(_worker_scan, [(s, str(p)) for s, p in shards]):
            index.extend(rows)
    tmp = cache.with_suffix(".parquet.tmp")
    pq.write_table(load.index_to_table(index), tmp)
    tmp.replace(cache)
    common.write_json(meta, {"shards": fp, "n_rows": len(index), "built": common.utc_now(), "seconds": round(time.time() - t0, 1)})
    run.log(f"index built: {len(index)} rows from {len(shards)} shards in {time.time() - t0:.0f}s -> {cache}")
    return index


def instances_from(rows: list[load.IndexRow], upstream: dict[tuple[str, str], load.UpstreamTask]) -> list[tfilter.Instance]:
    out = []
    for (sub, iid), trajs in sorted(load.group_by_instance(rows).items()):
        r0 = trajs[0]
        up = upstream.get((sub, iid))
        files, added = (up.size.files, up.size.added) if up else (r0.ref_files, r0.ref_added)
        out.append(tfilter.Instance(sub, iid, r0.language_code, tfilter.normalise_repo(r0.repo), files, added,
                                    up.pr if up and up.pr else load.pr_number(iid),
                                    tuple(sorted(t.trajectory_id for t in trajs))))
    return out


def load_outcomes(path: Path, leak_fraction: float) -> tuple[dict[str, dict[str, Any]], Counter]:
    """The last outcome per trajectory, reconciled with the current reference-patch rule: a cached leak drop the rule
    now admits is evaluated again; a cached pass the rule now rejects becomes a leak drop."""
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["trajectory_id"]] = rec
    changed: Counter = Counter()
    for t, rec in list(out.items()):
        if "reference_patch_lines" not in rec:
            continue
        drop = leaks(rec["reference_patch_lines_in_prompt"], rec["reference_patch_lines"], leak_fraction)
        if rec["stage"] == "reference_patch_leak" and not drop:
            del out[t]
            changed["re_evaluate"] += 1
        elif drop and rec["stage"] != "reference_patch_leak":     # a pass, or a drop at a later stage
            out[t] = {**rec, "ok": False, "stage": "reference_patch_leak", "reason": "reference_patch_line_in_prompt",
                      "detail": f"{rec['reference_patch_lines_in_prompt']} of {rec['reference_patch_lines']} non-test added lines"}
            changed["now_dropped"] += 1
    return out, changed


def update_features(path: Path, wanted: list[str], by_traj: dict[str, load.IndexRow], shard_dirs: dict[str, Path],
                    workers: int, run: common.RunDir) -> dict[str, dict[str, Any]]:
    """Test-after-edit features (apex_data.teacher.verify) of the `wanted` trajectories: the records of the current rule id
    cached in the jsonl at `path`, the rest read from the raw rows (two columns per shard) and appended there. Returns the
    records of `wanted` only."""
    feats: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() and (rec := json.loads(line))["rule_id"] == verify.RULE_ID:
                    feats[rec["trajectory_id"]] = rec
    missing = sorted(t for t in set(wanted) if t not in feats)
    if not missing:
        return {t: feats[t] for t in wanted}
    by_shard: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for t in missing:
        by_shard[(by_traj[t].subsource, by_traj[t].shard)].append((by_traj[t].row, t))
    jobs = [(sub, str(shard_dirs[sub] / shard), sorted(items)) for (sub, shard), items in sorted(by_shard.items())]
    t0 = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool, path.open("a", encoding="utf-8") as f:
        for recs in pool.imap_unordered(_worker_features, jobs):
            for rec in recs:
                feats[rec["trajectory_id"]] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
    run.log(f"trajectory features ({verify.RULE_ID}): {len(missing)} computed from {len(jobs)} shards in {time.time() - t0:.0f}s, "
            f"{len(feats)} cached -> {path}")
    return {t: feats[t] for t in wanted}


def eval_tasks(trajs: list[load.IndexRow], meta: dict[str, dict[str, Any]], shard_dirs: dict[str, Path], per_task: int,
               max_tokens: int) -> list[tuple[str, str, list[dict[str, Any]], int]]:
    by_shard: dict[tuple[str, str], list[load.IndexRow]] = defaultdict(list)
    for r in trajs:
        by_shard[(r.subsource, r.shard)].append(r)
    tasks = []
    for (sub, shard), rs in sorted(by_shard.items()):
        rs = sorted(rs, key=lambda r: r.row)
        for k in range(0, len(rs), per_task):
            items = [{"row": r.row, "trajectory_id": r.trajectory_id, "license": r.license, **meta[r.trajectory_id]}
                     for r in rs[k:k + per_task]]
            tasks.append((sub, str(shard_dirs[sub] / shard), items, max_tokens))
    return tasks


def staged_funnel(outcomes: list[dict[str, Any]], stages: tuple[str, ...] = STAGES) -> list[dict[str, Any]]:
    """Sequential table over the evaluated trajectories: each stage's n_in counts those that reached it."""
    steps, alive = [], list(outcomes)
    for stage in stages:
        dropped = [o for o in alive if o["stage"] == stage]
        cin = Counter(f"{o['subsource']}/{o['language']}" for o in alive)
        cdrop = Counter(f"{o['subsource']}/{o['language']}" for o in dropped)
        steps.append({"filter": stage, "n_in": len(alive), "n_dropped": len(dropped), "n_out": len(alive) - len(dropped),
                      "reasons": dict(Counter(o["reason"] for o in dropped)),
                      "by_cell": {k: {"n_in": cin[k], "n_dropped": cdrop[k], "n_out": cin[k] - cdrop[k]} for k in sorted(cin)}})
        alive = [o for o in alive if o["stage"] != stage]
    return steps


def publish(tmp: Path, path: Path, run: common.RunDir) -> None:
    """Move a finished output into place. An identical previous output is kept as is (the new copy is dropped); a
    different one is renamed to <name>.superseded_<UTC>, never deleted."""
    if path.exists():
        if common.sha256_file(path) == common.sha256_file(tmp):
            tmp.unlink()
            run.log(f"{path} unchanged")
            return
        old = path.with_name(f"{path.name.split('.')[0]}.superseded_{common.timestamp()}{''.join(path.suffixes)}")
        path.rename(old)
        run.log(f"superseded {path} -> {old}")
    tmp.replace(path)


def longest_turns(records: dict[str, dict[str, Any]], wanted: Iterable[str], rows_dirs: list[Path], cache: dict[str, int]) -> None:
    """cache[t] = the longest trained span (assistant_masks run) of t's cached rendered row, for the `wanted` gate-passing
    trajectories not cached yet (read from the row files the outcomes name; first of `rows_dirs` that has the file)."""
    by_file: dict[str, set[str]] = defaultdict(set)
    for t in wanted:
        if t not in cache:
            by_file[records[t]["row_file"]].add(t)
    for fname, ts in sorted(by_file.items()):
        path = next((d / fname for d in rows_dirs if (d / fname).exists()), None)
        if path is None:
            raise FileNotFoundError(f"rendered-row file {fname} not in {[str(d) for d in rows_dirs]}")
        table = pq.read_table(path, columns=["trajectory_id", "assistant_masks"])
        masks = table.column("assistant_masks").combine_chunks()
        offsets, values = masks.offsets.to_numpy(), masks.values.to_numpy(zero_copy_only=False).astype(np.int8)
        for k, t in enumerate(table.column("trajectory_id").to_pylist()):
            if t in ts:
                edges = np.diff(np.concatenate(([0], values[offsets[k]:offsets[k + 1]] != 0, [0])).astype(np.int8))
                cache[t] = int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max(initial=0))
        missing = ts - cache.keys()
        if missing:
            raise RuntimeError(f"{path}: rows of {sorted(missing)[:3]} missing")


def apply_turn_cap(records: dict[str, dict[str, Any]], longest: dict[str, int], cap: int) -> int:
    """Fail (in memory; the cached outcomes are not rewritten) every gate-passing trajectory whose longest trained turn
    exceeds `cap`. Returns the number newly failed."""
    n = 0
    for t, rec in records.items():
        if rec["ok"] and t in longest and longest[t] > cap:
            records[t] = {**rec, "ok": False, "stage": TURN_STAGE, "reason": TURN_TOO_LONG,
                          "detail": f"longest trained turn {longest[t]} tokens > {cap}", "max_turn_tokens": longest[t]}
            n += 1
    return n


def trained_spans(mask: list[int]) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, m in enumerate(mask + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((start, i))
            start = None
    return spans


def roundtrip_check(rows: list[dict[str, Any]], tok: Any, n: int, seed: int) -> dict[str, Any]:
    """M2.1: decode every trained span of `n` seeded rows, assert one span per assistant turn, the turn's reasoning
    inside the span before `</think>`, the qwen3 / qwen3_coder parse of the span equal to the turn, and a re-render of the
    stored episode reproducing the row token for token."""
    picked = random.Random(f"{seed}:m21").sample(rows, min(n, len(rows)))
    failures: list[str] = []
    turns = 0
    for row in picked:
        ep = json.loads(row["episode_json"])
        adapted = adapt({"episode_id": row["sample_id"], "task_id": row["task_id"], "scaffold": render.SCAFFOLD,
                         "messages": ep["messages"], "tools": ep["tools"], "exit_reason": "submit"},
                        {"source": row["source"], "language": row["language"], "license": row["license"]})
        sample = adapted.sample
        rebuilt, _ = build_rows(sample, tok, BuildOptions(mode="episode"))
        if not rebuilt or rebuilt[0]["input_ids"] != row["input_ids"] or rebuilt[0]["assistant_masks"] != row["assistant_masks"]:
            failures.append(f"{row['sample_id']}: re-render differs")
            continue
        spans = trained_spans(list(row["assistant_masks"]))
        asst = [m for m in sample.messages if m.role == "assistant"]
        if len(spans) != len(asst):
            failures.append(f"{row['sample_id']}: {len(spans)} trained spans for {len(asst)} assistant turns")
            continue
        for (a, b), msg in zip(spans, asst):
            turns += 1
            text = tok.decode(row["input_ids"][a:b])
            head = text.split("</think>", 1)[0]
            if "</think>" not in text or msg.reasoning_content.strip() not in head or not head.strip():
                failures.append(f"{row['sample_id']}: reasoning not inside the trained span")
                break
            if qwen3_roundtrip(text, msg) is not None:
                failures.append(f"{row['sample_id']}: parser round trip {qwen3_roundtrip(text, msg)}")
                break
    return {"rows_checked": len(picked), "assistant_turns_checked": turns, "failures": failures, "pass": not failures and bool(picked)}


def load_kept_selections(directory: Path) -> dict[str, Any]:
    """The two selection files of an earlier build, read before this run writes its own."""
    out: dict[str, Any] = {"dir": str(directory), "entries": {}, "sha256": {}}
    for name in CAPS:
        path = directory / f"selection_{name}.json"
        sel = json.loads(path.read_text(encoding="utf-8"))
        out["entries"][name], out["sha256"][name] = sel["rows"], common.sha256_file(path)
        out["source_run_id"] = sel["run_id"]
    return out


def pinned_picks(entries: list[dict[str, Any]], cap: int, inst_of: dict[str, tfilter.Instance], records: dict[str, Any]
                 ) -> tuple[dict[str, list[tfilter.Pick]], set[str], dict[str, str]]:
    """(kept picks per part, trajectories still to evaluate, reasons of the entries that are not kept)."""
    kept: dict[str, list[tfilter.Pick]] = defaultdict(list)
    pending, lost = set(), {}
    for e in entries:
        t, o = e["trajectory_id"], records.get(e["trajectory_id"])
        if t not in inst_of:
            lost[t] = "row_filters"
        elif o is None:
            pending.add(t)
        elif not o["ok"]:
            lost[t] = o["reason"]
        elif o["n_tokens"] > cap:
            lost[t] = "over_cap"
        else:
            kept[e["part"]].append(tfilter.Pick(e["part"], e["phase"], inst_of[t], t))
    return dict(kept), pending, lost


def teacher_success(index: list[load.IndexRow]) -> dict[tuple[str, str], str]:
    """all_pass | mixed | single per instance over the recorded attempts (resolved 1 / 0; unknown labels excluded)."""
    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in index:
        if r.resolved in (0, 1):
            per[(r.subsource, r.instance_id)].append(r.resolved)
    return {k: "single" if len(v) < 2 else ("all_pass" if all(v) else "mixed") for k, v in per.items()}


# --------------------------------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oswt-root", type=Path, default=None, help="Open-SWE-Traces root (default RAW/oswt)")
    p.add_argument("--rebench-root", type=Path, default=None, help="SWE-rebench-V2 root (default RAW/swe_rebench_v2)")
    p.add_argument("--scale-verified-root", type=Path, default=None, help="Scale-SWE-Verified root (default RAW/scale_swe_verified)")
    p.add_argument("--out-dir", type=Path, default=paths.POOLS / "swe_teacher")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--harbor-python", type=Path, default=None, help="interpreter with mini-swe-agent 2.4.6 (default .venv/harbor/bin/python)")
    p.add_argument("--lock-config", type=Path, default=None, help="runner-written mini-swe-agent config the templates are asserted "
                   "against (default the snapshot under configs/train_stack_check/harness_request_shapes/captures/)")
    p.add_argument("--workers", type=int, default=24, help="render processes (the host is shared: keep <= 24)")
    p.add_argument("--per-task", type=int, default=6, help="trajectories per worker task (one shard read per task)")
    p.add_argument("--limit-shards", type=int, default=None, help="first N shards of each folder (small-scale runs)")
    p.add_argument("--quota-scale", type=float, default=1.0, help="scale every quota (small-scale runs; 1.0 = the plan)")
    p.add_argument("--repo-root", default="/app", help="repository path our harness uses; the teacher's /testbed token is "
                   "rewritten to it in every message before the filters (default /app; /testbed keeps the teacher's path)")
    p.add_argument("--keep-selections", type=Path, default=None, metavar="DIR",
                   help="keep the rows of DIR/selection_{64k,48k}.json that still pass and draw only the vacated slots")
    p.add_argument("--trajectory-choice", choices=tuple(tfilter.TRAJECTORY_CHOICES), default=tfilter.RANDOM_NOT_SHORTEST,
                   help="which gate-passing trajectory within the cap represents an admitted instance (default "
                        "random_not_shortest, the plan's rule; fewest_turns_verified: fewest assistant turns among those that "
                        "ran a test after their last edit, else random_not_shortest)")
    p.add_argument("--features", type=Path, default=None, metavar="PATH",
                   help="test-after-edit feature cache for fewest_turns_verified (jsonl keyed by trajectory id; default "
                        "WORK/<run_id>/evals/trajectory_features.jsonl)")
    p.add_argument("--leak-min-fraction", type=float, default=0.0,
                   help="drop a trajectory when this share of the reference patch's substantive non-test added lines occurs in "
                        "the prompt (0 = any line, the plan's literal rule); --resume re-applies a changed rule to the cache")
    p.add_argument("--max-turn-tokens", type=int, default=None, metavar="N",
                   help="fail a trajectory whose rendered row has an assistant turn of more than N trained tokens (stage "
                        f"{TURN_STAGE}, reason {TURN_TOO_LONG}; measured on the cached rows; default: no limit)")
    p.add_argument("--no-fork-graph", action="store_true", help="skip GitHub fork-graph lookups (cache only; recorded)")
    p.add_argument("--roundtrip-rows", type=int, default=20)
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="run id (default data_swe_teacher_<UTC>)")
    p.add_argument("--resume", default=None, metavar="RUN_ID", help="continue a run: reuse WORK/<RUN_ID>/ and logs/<RUN_ID>/")
    p.add_argument("--draw-from", default=None, metavar="RUN_ID",
                   help="draw on the index, outcomes and rendered rows of WORK/<RUN_ID>/ (read-only); trajectories it lacks "
                        "are evaluated into this run's WORK/<run_id>/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.resume or a.run_id or common.new_run_id("data_swe_teacher")
    work = paths.WORK / run_id
    base = paths.WORK / a.draw_from if a.draw_from else None          # read-only cache of an earlier run
    if base and not (base / "evals" / "outcomes.jsonl").exists():
        raise SystemExit(f"--draw-from {a.draw_from}: {base}/evals/outcomes.jsonl not found")
    (work / "evals" / "rows").mkdir(parents=True, exist_ok=True)
    rows_dirs = [work / "evals" / "rows", *([base / "evals" / "rows"] if base else [])]
    features_path = a.features or paths.WORK / run_id / "evals" / "trajectory_features.jsonl"
    use_features = a.trajectory_choice != tfilter.RANDOM_NOT_SHORTEST
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with common.RunDir(run_id, log_name="train.log", resume=bool(a.resume)) as run:
        revisions = load.raw_revisions()
        oswt_rev = revisions["oswt"]["revision"]
        if not oswt_rev:
            raise RuntimeError("data/manifests/raw_sources.json has no `oswt` revision; run scripts/data/fetch_raw.py first")
        parts = [tfilter.PartSpec(p.name, p.subsource, p.language, max(1, round(p.quota * a.quota_scale)), p.mode)
                 for p in tfilter.PARTS]
        run.write_metadata(run_id=run_id, step="data_swe_teacher", command=common.command_line(), git=common.git_info(),
                           config=json.loads(json.dumps(vars(a), default=str)), seed=a.seed, hardware=common.hardware_info(),
                           data_versions=revisions, model_revision=common.model_revision(), work_dir=str(work),
                           base_work_dir=str(base) if base else None,
                           quotas={p.name: p.quota for p in parts}, caps=CAPS, start_time=common.utc_now(), status="running")
        bundle = render.load_templates(a.harbor_python, lock_config=a.lock_config)
        run.log(f"mini-swe-agent {bundle.version} templates from {bundle.config_path} equal the lock reference "
                f"{bundle.lock_reference['path']}; bash tool sha256 {bundle.bash_tool_sha256[:12]}; uname vars {bundle.template_vars}")
        tok = load_tokenizer(a.tokenizer)
        template_sha = hashlib.sha256(tok.chat_template.encode()).hexdigest()
        run.write_metadata(tokenizer={"path": str(a.tokenizer), "chat_template_sha256": template_sha},
                           templates={k: v for k, v in bundle.to_dict().items() if not k.endswith("_template")})

        # ---- index and side inputs --------------------------------------------------------------------------------
        shards = load.list_shards(a.oswt_root, limit=a.limit_shards)
        shard_dirs = {s: p.parent for s, p in shards}
        index = build_or_load_index(shards, base or work, a.workers, run, read_only=base is not None)
        if len({r.trajectory_id for r in index}) != len(index):
            raise RuntimeError("trajectory ids are not unique across the shards")
        verified = load.scale_swe_verified_ids(a.scale_verified_root)
        upstream = load.load_upstream(a.rebench_root, a.scale_verified_root)
        dev = tfilter.load_dev_exclusions()
        locked = d2_repos.load_locked()
        run.log(f"Scale-SWE-Verified ids {len(verified)}; upstream task rows {len(upstream)}; dev exclusions {dev.status} "
                f"({len(dev.repos)} repositories, {len(dev.instance_ids)} task ids); D2 list sha {locked.sha256[:16]} "
                f"({len(locked.repos)} locked repositories, excluded {sorted(locked.excluded)})")
        if dev.status != "applied":
            run.error("data/manifests/dev_exclusions.json absent: filter 6 (dev repositories) not applied; re-run after WP6")
        pre = [r for r in index if r.resolved == 1 and r.language in load.LANGUAGE_CODES]
        repos = sorted({k for r in pre if (k := repo_key(r.repo))})
        token = None if a.no_fork_graph else github_token()
        lookup = RepoLookup(cache_dir=paths.WORK / "licence_cache", token=token, enabled=not a.no_fork_graph)
        fork_infos, fork_summary = lookup.lookup(repos, log=run.log)
        rows, funnel, drops, d2_report = tfilter.row_filters(
            index, bash_tool=bundle.bash_tool, verified=verified, locked=locked, fork_infos=fork_infos,
            fork_lookup_enabled=lookup.enabled, dev=dev)
        d2_report["fork_graph_lookup"] = fork_summary
        for s in funnel.steps:
            run.metric({"stage": "row_filter", **{k: v for k, v in s.items() if k != "by_cell"}})
        run.log("row filters: " + json.dumps([(s["filter"], s["n_out"]) for s in funnel.steps]))
        instances = instances_from(rows, upstream)
        by_traj = {r.trajectory_id: r for r in rows}
        inst_of = {t: inst for inst in instances for t in inst.trajectories}
        meta = {t: {"instance_id": inst.instance_id, "subsource": inst.subsource, "language": inst.language,
                    "repo": by_traj[t].repo, "stratum": inst.stratum, "files": inst.files, "added": inst.added}
                for t, inst in inst_of.items()}
        orders = {p.name: tfilter.part_orders(p, instances, a.seed) for p in parts}
        run.log("instances after row filters: " + json.dumps(dict(Counter(f"{i.subsource}/{i.language}/{i.stratum}" for i in instances))))

        # ---- lazy evaluation and draw (with kept rows pinned when --keep-selections) ------------------------------
        keep = load_kept_selections(a.keep_selections) if a.keep_selections else None
        outcomes_path = work / "evals" / "outcomes.jsonl"
        records, reconciled = load_outcomes(outcomes_path, a.leak_min_fraction)
        if base:                                     # this run's own outcomes win over the cached ones
            cached, reconciled_base = load_outcomes(base / "evals" / "outcomes.jsonl", a.leak_min_fraction)
            records, reconciled = {**cached, **records}, reconciled + reconciled_base
        run.log(f"evaluated trajectories on disk: {len(records)}; reconciled with the reference-patch rule: {dict(reconciled)}; "
                f"repository root {a.repo_root}; trajectory choice {a.trajectory_choice}"
                + (f"; keeping the selections of {keep['dir']}" if keep else "") + (f"; cached outcomes of {base}" if base else ""))
        pool = None                                  # render processes, started when something needs evaluating
        lost: dict[str, dict[str, str]] = {}
        pins_by_cap: dict[str, dict[str, list[tfilter.Pick]]] = {}
        feats: dict[str, dict[str, Any]] = {}
        longest: dict[str, int] = {}                  # --max-turn-tokens: longest trained turn per gate-passing trajectory
        try:
            iteration = 0
            while True:
                if a.max_turn_tokens is not None:
                    longest_turns(records, [t for t, r in records.items() if r["ok"] and t in meta], rows_dirs, longest)
                    n_cap = apply_turn_cap(records, longest, a.max_turn_tokens)
                    run.log(f"turn cap {a.max_turn_tokens}: {n_cap} more gate-passing trajectories fail ({TURN_TOO_LONG}); "
                            f"{len(longest)} measured")
                if use_features:
                    feats = update_features(features_path, [t for t, r in records.items() if r["ok"] and t in meta], by_traj,
                                            shard_dirs, a.workers, run)
                outcomes = {t: tfilter.Outcome(t, r["ok"], r.get("n_tokens"), r["stage"], r["reason"], r.get("n_assistant_turns"),
                                               feats[t]["verified"] if t in feats else None) for t, r in records.items()}
                pending: set[str] = set()
                draws = {}
                for name, cap in CAPS.items():
                    pins = None
                    if keep:
                        pins, wait, lost[name] = pinned_picks(keep["entries"][name], cap, inst_of, records)
                        pending |= wait
                        pins_by_cap[name] = pins
                    draws[name] = tfilter.draw_cap(cap, parts, orders, outcomes, a.seed, pinned=pins,
                                                   trajectory_choice=a.trajectory_choice)
                pending |= {t for d in draws.values() for inst in d.pending for t in inst.trajectories if t not in records}
                if not pending:
                    break
                if pool is None:
                    pool = mp.get_context("fork").Pool(a.workers, initializer=_worker_init,
                                                       initargs=(str(a.tokenizer), bundle.to_dict(), str(work / "evals" / "rows"),
                                                                 oswt_rev, a.leak_min_fraction, a.repo_root))
                iteration += 1
                tasks = eval_tasks([by_traj[t] for t in sorted(pending)], meta, shard_dirs, a.per_task, CAP_TARGET)
                t1, n_ok = time.time(), 0
                with outcomes_path.open("a", encoding="utf-8") as f:
                    for outs in pool.imap_unordered(_worker_eval, tasks):
                        for o in outs:
                            records[o["trajectory_id"]] = o
                            n_ok += o["ok"]
                            f.write(json.dumps(o, ensure_ascii=False) + "\n")
                        f.flush()
                state = {name: {p: len(r.picks) for p, r in d.parts.items()} for name, d in draws.items()}
                run.log(f"iteration {iteration}: evaluated {len(pending)} trajectories in {len(tasks)} tasks, {time.time() - t1:.0f}s, "
                        f"{n_ok} rendered; picks so far {state}")
                run.metric({"stage": "evaluate", "iteration": iteration, "n": len(pending), "n_ok": n_ok,
                            "seconds": round(time.time() - t1, 1), "picks": state})
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if keep:
            run.log("kept selections: " + json.dumps({name: {"kept": len(keep["entries"][name]) - len(lost[name]),
                                                             "replaced": dict(Counter(lost[name].values()))} for name in CAPS}))

        # ---- selections ------------------------------------------------------------------------------------------
        evaluated = [o for o in records.values() if o["trajectory_id"] in meta]   # outcomes of rows the filters still admit
        tokens = {o["trajectory_id"]: (o["n_tokens"], o["n_assistant_tokens"]) for o in evaluated if o["ok"]}
        success = teacher_success(index)
        selections: dict[str, dict[str, Any]] = {}
        # kept rows by instance: under a trajectory choice other than random_not_shortest a kept row's trajectory may change
        kept_ids = {name: {p.instance.key for ps in (pins_by_cap[name] or {}).values() for p in ps} for name in CAPS} if keep else {}
        for name, d in draws.items():
            picks = d.picks
            entries = [{"task_id": f"{load.SOURCE}:{p.instance.instance_id}", "instance_id": p.instance.instance_id,
                        "subsource": p.instance.subsource, "language": p.instance.language, "repo": p.instance.repo,
                        "size_stratum": p.instance.stratum, "patch_files": p.instance.files, "patch_added_lines": p.instance.added,
                        "part": p.part, "phase": p.phase, "trajectory_id": p.trajectory_id, "row_key": f"{p.trajectory_id}:-1",
                        "n_tokens": tokens[p.trajectory_id][0], "n_assistant_tokens": tokens[p.trajectory_id][1],
                        "teacher_success": success.get((p.instance.subsource, p.instance.instance_id), "single"),
                        **({"kept_from": keep["source_run_id"]} if keep and p.instance.key in kept_ids[name] else {})}
                       for p in picks]
            by_lang = defaultdict(list)
            for p in picks:
                by_lang[p.instance.language].append(p)
            selections[name] = {
                "cap": d.cap, "run_id": run_id, "seed": a.seed, "complete": d.complete,
                "draw": {"trajectory_choice": a.trajectory_choice,
                         **({"choice_counts": tfilter.choice_counts(picks, outcomes, d.cap)} if use_features else {})},
                "quotas": {pn: {"quota": r.part.quota, "filled": len(r.picks), "shortfall": r.shortfall, "met": r.shortfall == 0
                                and len(r.picks) == r.part.quota, "walk": dict(r.walked)} for pn, r in d.parts.items()},
                "summary": {"all": tfilter.shape_summary(picks, tokens),
                            **{lang: tfilter.shape_summary(ps, tokens) for lang, ps in sorted(by_lang.items())},
                            **{pn: tfilter.shape_summary(r.picks, tokens) for pn, r in d.parts.items() if pn.startswith("py_")}},
                "teacher_success": dict(Counter(e["teacher_success"] for e in entries)),
                "rows": entries}
            run.log(f"selection {name}: " + json.dumps({pn: q["filled"] for pn, q in selections[name]["quotas"].items()})
                    + f"; {json.dumps({k: v for k, v in selections[name]['summary']['all'].items() if 'histogram' not in k})}")

        # ---- rows of the union --------------------------------------------------------------------------------------
        sel_sets = {name: {e["trajectory_id"] for e in s["rows"]} for name, s in selections.items()}
        union = sorted(set().union(*sel_sets.values()))
        files = sorted({records[t]["row_file"] for t in union})
        cache_rows: dict[str, dict[str, Any]] = {}
        for fname in files:
            row_path = next((d / fname for d in rows_dirs if (d / fname).exists()), None)
            if row_path is None:
                raise FileNotFoundError(f"rendered-row file {fname} not in {[str(d) for d in rows_dirs]}")
            for r in pq.read_table(row_path).to_pylist():
                if r["trajectory_id"] in sel_sets["64k"] | sel_sets["48k"]:
                    cache_rows[r["trajectory_id"]] = r
        out_rows = []
        d2_final = Counter()
        root_stats: dict[str, Any] = {"rewrites": [], "prefixes": [], "counts": Counter(), "root_tokens_left": 0,
                                      "testbed_substrings_left": 0, "tokens_delta": [], "fits_48k_changed": 0}
        kept_tokens = {e["trajectory_id"]: e["n_tokens"] for name in CAPS for e in keep["entries"][name]
                       if e["trajectory_id"] not in lost[name]} if keep else {}
        for t in union:
            r, prov = cache_rows[t], records[t]["provenance"]
            validate_provenance(prov)
            ep_text = render.episode_text(json.loads(r["episode_json"])["messages"])
            recheck = d2_repos.screen(prov["repo"], locked=locked, text=ep_text, upstream_id=records[t]["instance_id"])
            d2_final["locked_hit"] += recheck.locked_hit
            d2_final["excluded_hit"] += recheck.excluded_hit
            rb = records[t]["rebuild"]
            counts = rb.get("repo_root_counts") or {}
            root_stats["rewrites"].append(counts.get("paths_rewritten", 0))
            root_stats["prefixes"].append(counts.get("cd_prefixes_deleted", 0))
            root_stats["counts"].update(counts)
            root_stats["root_tokens_left"] += len(render.ROOT_TOKEN_RE.findall(ep_text))
            root_stats["testbed_substrings_left"] += ep_text.count(render.TEACHER_REPO_ROOT)
            if t in kept_tokens:
                old_n = kept_tokens[t]
                root_stats["tokens_delta"].append(r["n_tokens"] - old_n)
                root_stats["fits_48k_changed"] += (old_n <= CAP_FALLBACK) != (r["n_tokens"] <= CAP_FALLBACK)
            out_rows.append({**{k: r[k] for k in ROW_COLUMNS}, **{k: prov[k] for k in PROVENANCE_FIELDS}, "row_key": prov["row_key"],
                             "selected_64k": t in sel_sets["64k"], "selected_48k": t in sel_sets["48k"],
                             "repo_root": rb.get("repo_root", render.TEACHER_REPO_ROOT),
                             "repo_root_rewrites": counts.get("paths_rewritten", 0),
                             "repo_root_cd_prefixes_deleted": counts.get("cd_prefixes_deleted", 0)})
        schema = pa.schema([*arrow_schema(), *[(k, PROVENANCE_TYPES.get(k, pa.string())) for k in PROVENANCE_FIELDS],
                            *EXTRA_COLUMNS])
        rows_path = a.out_dir / "rows.parquet"
        tmp = rows_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), tmp)
        publish(tmp, rows_path, run)
        rows_sha = common.sha256_file(rows_path)
        run.log(f"rows -> {rows_path}: {len(out_rows)} rows (union of {len(sel_sets['64k'])} at 64K and {len(sel_sets['48k'])} at 48K), "
                f"sha256 {rows_sha[:16]}")
        def per_row(v: list[int]) -> dict[str, Any]:
            return {"mean": round(sum(v) / len(v), 1) if v else 0.0, "max": max(v, default=0), "total": sum(v)}

        delta = root_stats["tokens_delta"]
        repo_root_report = {
            "repo_root": a.repo_root, "teacher_root": render.TEACHER_REPO_ROOT,
            "paths_rewritten_per_row": per_row(root_stats["rewrites"]),
            "cd_prefixes_deleted_per_row": per_row(root_stats["prefixes"]),
            "counts": dict(sorted(root_stats["counts"].items())),
            "ambiguous_drops": {"trajectories": sum(1 for o in records.values() if o["stage"] == "repo_root_ambiguous"),
                                "by_reason": dict(sum((Counter(json.loads(o["detail"])) for o in records.values()
                                                       if o["stage"] == "repo_root_ambiguous"), Counter()))},
            "root_tokens_left": root_stats["root_tokens_left"],
            "testbed_substrings_left": root_stats["testbed_substrings_left"],
            "kept_rows_tokens_delta_vs_source": per_row(delta) | {"min": min(delta, default=0)} if keep else None,
            "kept_rows_fits_48k_changed": root_stats["fits_48k_changed"] if keep else None,
            "rule": "commands lose a leading `cd /testbed && ` (or ; or newline) and a leading `cd /testbed/<sub> &&` becomes "
                    "`cd <sub> &&`; then /testbed at a path start (start, whitespace, quote, =, :, (, `, [, {, comma, <, >) and "
                    "a name end becomes the repository root in every message (observations on the parsed output); episodes "
                    "with a command the rewrite could change (bare word testbed, a skipped /testbed path, cd /testbed && cd ..) "
                    "are dropped as repo_root_ambiguous"}
        run.log(f"repository root: {json.dumps(repo_root_report)}")

        # ---- M2.1 checks -------------------------------------------------------------------------------------------
        rt = roundtrip_check([{**cache_rows[t], "task_id": records[t]["provenance"]["task_id"]} for t in union], tok,
                             a.roundtrip_rows, a.seed)
        bottle = {name: sum(1 for e in s["rows"] if e["repo"] in locked.excluded) for name, s in selections.items()}
        checks = {
            "roundtrip_20_rows": rt,
            "quotas_met": {name: all(q["met"] for q in s["quotas"].values()) for name, s in selections.items()},
            "scale_swe_rows": {name: s["quotas"]["py_scale"]["filled"] for name, s in selections.items()},
            "max_rows_per_repository": {name: s["summary"]["all"]["max_rows_per_repository"] for name, s in selections.items()},
            "one_row_per_instance": {name: len({e["task_id"] for e in s["rows"]}) == len(s["rows"]) for name, s in selections.items()},
            "d2_hits_selected_rows": d2_final["locked_hit"], "decontam_excluded_repository_rows": d2_final["excluded_hit"],
            "excluded_repository_rows_by_cap": bottle,
            "provenance_valid": True,
            "repo_root_tokens_left": root_stats["root_tokens_left"],
            "repo_root_ambiguous_rows": sum(1 for t in union if records[t]["stage"] == "repo_root_ambiguous"),
        }
        if keep:
            checks["kept_selections"] = {name: {"kept": len(kept_ids[name]), "replaced": len(lost[name]),
                                                "replaced_by_reason": dict(Counter(lost[name].values()))} for name in CAPS}
        if a.max_turn_tokens is not None:
            row_longest = max((max(turn_span_lengths(r["assistant_masks"]), default=0) for r in out_rows), default=0)
            checks["max_trained_turn_tokens"] = {"cap": a.max_turn_tokens, "rows_max": row_longest,
                                                 "pass": row_longest <= a.max_turn_tokens}
        checks["pass"] = (rt["pass"] and all(checks["quotas_met"].values()) and all(checks["one_row_per_instance"].values())
                          and checks.get("max_trained_turn_tokens", {}).get("pass", True)
                          and max(checks["max_rows_per_repository"].values()) <= tfilter.MAX_PER_REPO
                          and d2_final["locked_hit"] == 0 and d2_final["excluded_hit"] == 0 and not any(bottle.values())
                          and root_stats["root_tokens_left"] == 0 and checks["repo_root_ambiguous_rows"] == 0)
        run.log(f"M2.1 checks: {json.dumps({k: v for k, v in checks.items() if k != 'roundtrip_20_rows'})}; round trip "
                f"{rt['rows_checked']} rows / {rt['assistant_turns_checked']} turns, failures {rt['failures'][:3]}")
        if not checks["pass"]:
            run.error(f"M2.1 checks failed: {json.dumps(checks)[:2000]}")

        # ---- dropped records and manifest ------------------------------------------------------------------------
        dropped_path = a.out_dir / "dropped.jsonl.gz"
        with gzip.GzipFile(dropped_path.with_suffix(".gz.tmp"), "wb", mtime=0) as gz, \
                io.TextIOWrapper(gz, encoding="utf-8") as f:                    # mtime 0: identical content, identical bytes
            for d in drops:
                f.write(json.dumps(asdict(d)) + "\n")
            for o in evaluated:
                if not o["ok"]:
                    f.write(json.dumps({k: o.get(k) for k in ("subsource", "instance_id", "trajectory_id", "repo", "language",
                                                              "stage", "reason", "detail", "n_tokens")}) + "\n")
        publish(dropped_path.with_suffix(".gz.tmp"), dropped_path, run)
        not_reached = len(rows) - len(evaluated)
        stage_steps = staged_funnel(evaluated, STAGES + ((TURN_STAGE,) if a.max_turn_tokens is not None else ()))
        per_cap_instances = {}
        for name, cap in CAPS.items():
            evaluated_inst = [i for i in instances if all(t in records for t in i.trajectories)]
            admitted = [i for i in evaluated_inst if tfilter.choose_trajectory(i, outcomes, cap, a.seed, a.trajectory_choice)
                        not in (None, tfilter.PENDING)]
            per_cap_instances[name] = {"instances_fully_evaluated": len(evaluated_inst), "admitted_at_cap": len(admitted),
                                       "admitted_by_cell": dict(Counter(f"{i.subsource}/{i.language}/{i.stratum}" for i in admitted))}
        all_tokens = [o["n_tokens"] for o in evaluated if o.get("n_tokens")]
        leak_by_stratum: dict[str, dict[str, Any]] = {}
        for o in evaluated:
            if o["stage"] == "rebuild":
                continue
            c = leak_by_stratum.setdefault(meta[o["trajectory_id"]]["stratum"], {"reached": 0, "dropped": 0})
            c["reached"] += 1
            c["dropped"] += o["stage"] == "reference_patch_leak"
        for c in leak_by_stratum.values():
            c["share"] = round(c["dropped"] / c["reached"], 3) if c["reached"] else 0.0
        manifest = {
            "pool": "swe_teacher", "source": load.SOURCE, "run_id": run_id, "built": common.utc_now(), "seed": a.seed,
            "code_commit": common.git_info().get("commit"),
            "dataset": {"repo": load.HF_REPO, "revision": oswt_rev, "folders": [str(load.SHARD_ROOT / s) for s in load.SUBSOURCES],
                        "licence": load.DATASET_LICENSE + " (per-row repository SPDX in `repo_license`)",
                        "shards": {"n": len(shards), "limited": a.limit_shards is not None}},
            "side_inputs": {"swe_rebench_v2": revisions["swe_rebench_v2"], "scale_swe_verified": revisions["scale_swe_verified"],
                            "scale_swe_verified_ids": len(verified),
                            "reference_patch_agreement": dict(Counter(
                                ("upstream_equal" if upstream[(r.subsource, r.instance_id)].patch_sha256 == r.ref_patch_sha256
                                 else "upstream_differs") if (r.subsource, r.instance_id) in upstream else "no_upstream_row"
                                for r in rows))},
            "generator": {"name": load.GENERATOR, "thinking": True, "reasoning_effort": sorted({r.reasoning_effort for r in rows}),
                          "label_provenance": "verifier (upstream resolved == 1, NVIDIA's execution label)"},
            "templates": {k: v for k, v in bundle.to_dict().items() if not k.endswith("_template")},
            "tokenizer": {"path": str(a.tokenizer), "chat_template_sha256": template_sha},
            "render": {"row_kind": "swe_episode", "mode": "episode", "enable_thinking": True, "cap_target": CAP_TARGET,
                       "cap_fallback": CAP_FALLBACK, "truncation": "none at render time; observations carry the harness's own "
                       ">= 10,000-character elision", "submit_command": "the teacher's final `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT "
                       "&& cat patch.txt` is rewritten to the lone `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` our harness prompt asks "
                       "for; every other turn is verbatim",
                       "reference_patch_rule": (f"a trajectory is dropped when its prompt (system + user text) contains "
                                                f"{'any' if a.leak_min_fraction == 0 else f'>= {a.leak_min_fraction:.0%} of the'} "
                                                f"non-test added lines of >= {render.LEAK_MIN_CHARS} non-blank characters of the "
                                                "reference patch"),
                       "reference_patch_drops_by_stratum": leak_by_stratum},
            "repo_root": repo_root_report,
            "quotas": {p.name: {"subsource": p.subsource, "language": p.language, "quota": p.quota, "mode": p.mode} for p in parts},
            "draw": {"big_share": tfilter.BIG_SHARE, "max_rows_per_repository": tfilter.MAX_PER_REPO,
                     "strata": "big = >= 3 non-test files and >= 100 added lines; feature = >= 3 files or >= 50 added lines; "
                               "random = the rest (reference patch without test files, apex_data.grade.tamper.classify_path)",
                     "trajectory_choice": a.trajectory_choice,
                     "trajectory_choice_rule": tfilter.TRAJECTORY_CHOICES[a.trajectory_choice],
                     **({"trajectory_features": {
                         "rule_id": verify.RULE_ID, "test_pattern": verify.TEST_RE.pattern, "edit_pattern": verify.EDIT_RE.pattern,
                         "source": "assistant tool-call arguments of the raw Open-SWE-Traces messages, in order",
                         "cache": str(features_path), "n_trajectories": len(feats),
                         "verified_share": tfilter.share(f["verified"] for f in feats.values())}} if use_features else {}),
                     "lineage": "one row per (repository, PR) across both subsources"},
            "draw_source": ({"mode": "kept selections, vacated slots drawn", "kept_from": keep["dir"],
                             "source_run_id": keep["source_run_id"], "source_selection_sha256": keep["sha256"],
                             "kept": {name: len(kept_ids[name]) for name in CAPS},
                             "replaced": {name: {"n": len(lost[name]), "by_reason": dict(Counter(lost[name].values())),
                                                 "by_part": dict(Counter(e["part"] for e in keep["entries"][name]
                                                                         if e["trajectory_id"] in lost[name]))}
                                          for name in CAPS}}
                            if keep else {"mode": "drawn"}),
            **({"max_turn_completion_tokens": {
                "cap": a.max_turn_tokens,
                "rule": "a gate-passing trajectory whose rendered row has an assistant turn of more than the cap in trained "
                        f"tokens (its contiguous assistant_masks span) fails at stage {TURN_STAGE} ({TURN_TOO_LONG}); kept "
                        "selections whose trajectory fails are replaced like any other failing kept row",
                "measured_trajectories": len(longest),
                "failed_trajectories": sum(1 for o in evaluated if o["stage"] == TURN_STAGE)}}
               if a.max_turn_tokens is not None else {}),
            "funnel": {"row_filters": funnel.steps, "row_filter_reasons": dict(sorted(funnel.reasons.items())),
                       "rendered_filters": stage_steps, "not_reached_by_the_draw": not_reached,
                       "instances": {"after_row_filters": len(instances),
                                     "by_cell": dict(Counter(f"{i.subsource}/{i.language}/{i.stratum}" for i in instances))},
                       "per_cap": per_cap_instances},
            "evaluated_length_histogram": histogram(all_tokens),
            "selections": {name: {k: v for k, v in s.items() if k != "rows"} for name, s in selections.items()},
            "d2": {**d2_report, "text_signature_drops": sum(1 for o in evaluated if o["stage"] == "d2_text"),
                   "selected_rows_rescreened": d2_final, "locked_list_sha256": locked.sha256,
                   "fork_graph_source": "GitHub API, cache WORK/licence_cache" if lookup.enabled else "cache only (lookups disabled)"},
            "dev_exclusions": {"status": dev.status, "source": dev.source, "sha256": dev.sha256, "repositories": sorted(dev.repos),
                               "task_ids": len(dev.instance_ids),
                               "rows_in_dev_repositories_all": dict(Counter(tfilter.normalise_repo(r.repo) for r in index
                                                                            if tfilter.normalise_repo(r.repo) in dev.repos)),
                               "trajectories_dropped_at_filter_6": dict(Counter(d.reason for d in drops if d.stage == "6_dev_exclusions"))},
            "checks_m2_1": checks,
            "outputs": {"rows_parquet": str(rows_path.relative_to(paths.REPO)) if rows_path.is_relative_to(paths.REPO) else str(rows_path),
                        "rows_sha256": rows_sha, "n_rows": len(out_rows), "rows_64k": len(sel_sets["64k"]), "rows_48k": len(sel_sets["48k"]),
                        "rows_in_both": len(sel_sets["64k"] & sel_sets["48k"]),
                        "selection_files": ["selection_64k.json", "selection_48k.json"], "dropped": "dropped.jsonl.gz",
                        "dropped_sha256": common.sha256_file(dropped_path) if dropped_path.exists() else None, "work_dir": str(work),
                        **({"base_work_dir": str(base)} if base else {})},
        }
        for name, s in selections.items():
            common.write_json(a.out_dir / f"selection_{name}.json", s)
            common.write_json(run.run_dir / f"selection_{name}.json", s)
        common.write_json(a.out_dir / "manifest.json", manifest)
        common.write_json(run.run_dir / "manifest.json", manifest)

        dur = time.time() - t_start
        rss_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1e6
        run.write_resources([
            ("duration_sec", round(dur, 1), "s", "wall clock on the CPU VM"),
            ("workers", a.workers, "proc", "render processes"),
            ("trajectories_evaluated", len(evaluated), "rows", "rendered-episode filters run"),
            ("peak_rss_gb", round(rss_gb, 2), "GB", "parent + largest child max RSS"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"), ("gpu_hours", 0.0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)")])
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(dur, 1), status="ok" if checks["pass"] else "checks_failed",
                           n_rows=len(out_rows), manifest=str(a.out_dir / "manifest.json"), rows_parquet=str(rows_path),
                           rows_sha256=rows_sha)
        run.finalize()
        return 0 if checks["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
