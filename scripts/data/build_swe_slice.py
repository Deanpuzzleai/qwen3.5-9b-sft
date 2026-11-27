#!/usr/bin/env python
"""Build the revision-5 SWE slice (data/manifests/v5/DATA_CARD_v5.md, SWE slice): Open-SWE-Traces
v1.2 mini-swe-agent Qwen3.8-27B episodes, banded by the teacher's attempts, gated by the efficiency profiler, re-rendered
under our mini-swe-agent 2.4.6 harness, and drawn by loss tokens (apex_data.teacher.select_v5).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_swe_slice.py [--workers 24] [--limit-shards N]
        [--target-loss-tokens 9000000] [--out-dir data/pools/swe_v5] [--seed 42] [--resume RUN_ID]
        [--exclude-task-ids PATH] [--no-category-round-robin] [--no-recovery-preference] [--no-lower-half]
        [--keep-all-bands hard,medium] [--band-share easy=0.45] [--repo-cap go=5 --repo-cap ts=5] [--note TEXT]
        [--second-attempt-bands medium] [--max-second-share 0.3]
        [--no-fork-graph] [--repo-root /app]

Pipeline (each stage counts what it drops, nothing is deleted):
  1. index    every trajectory of both folders (apex_data.teacher.load.scan_shard; teacher identity asserted per row).
  2. bands    per instance from all its attempts with a known outcome (efficiency.swe_band): hard 1 of 3, medium 2 of 3
              or 1 of 2, easy 3 of 3 or 2 of 2; unsolved and single-attempt instances are not used.
  3. row filters (apex_data.teacher.filter.row_filters): resolved, language (go, python, ts, js, rust, java, php),
              licence allowlist, Scale-SWE-Verified, D2 and decontamination exclusions, the bash tool schema, a clean
              non-empty model patch, reasoning on every turn, dev exclusions; then the band filter.
  4. raw profile (prefilter) on the stored messages: the band's step limit, zero wasted actions, a verified completion,
              the tamper screen of the model patch (apex_data.grade.tamper.swe_flags: tests, CI, test config, skip
              markers), and the thinking limits with RAW_SLACK (5 %) because the rendered episode, not the raw one, is
              authoritative (the /testbed -> /app rewrite changes reasoning token counts slightly).
  5. render + gates per trajectory that passed 4: the episode rebuilt under our templates (repository root rewritten),
              the reference-patch leak rule, the D2 text screen, the shared gates of apex_data.render.filters (verdict,
              tamper, adapt, exit, format-error turn, finish length, reasoning present, language, parse round trip,
              render; the `thrashing` gate is replaced by the profiler's wasted-action detector, plan 5.1) at the band's
              row cap (32K / 48K / 64K), at most 32K loss tokens, and the profiler gates on the rendered episode
              (efficiency.gate_failures: band limits, zero wasted actions, verify before completion). Authoritative.
  6. candidates: per instance the gate-passing attempt with the lowest mean thinking per turn; in the bands of
              --second-attempt-bands also the next gate-passing attempt (attempt 2, drawn after attempt 1, at most
              --max-second-share of the episodes; revision 5 uses medium only).
  7. draw     (select_v5.draw): bands by loss tokens 20 / 50 / 30 of --target-loss-tokens with the supply rule, language
              quotas in episodes, lower-half random draw, recovery preference (>= 20 %), category round-robin, <= 3
              instances per repository, one per (repository, PR). --keep-all-bands takes every admissible instance of
              those bands and --band-share sets the other bands as shares of the total (revision 5: hard and
              medium kept whole, easy 45 %); --repo-cap sets the repository cap per language group (Go and TypeScript
              5). --exclude-task-ids removes instances first (screen drops in P1e; the draw is then run again on the
              same cached candidates).

Inputs (read-only):
  RAW/oswt/data/minisweagent/qwen38_27b/{swe-rebench-v2,scale-swe}/*.parquet   teacher trajectories (pinned revision)
  RAW/swe_rebench_v2, RAW/scale_swe_verified    reference-patch sizes (reported), PR numbers, the Scale-SWE-Verified list
  data/manifests/raw_sources.json, data/manifests/decontam_exclusions.json, data/manifests/dev_exclusions.json, configs/eval/
  the Qwen3.5-9B tokenizer (apex_data.paths.BASE_MODEL_DIR); mini-swe-agent 2.4.6 in .venv/harbor and the harness-config
  lock snapshot (apex_data.teacher.render); GITHUB_TOKEN (environment or .env, never printed) for D2 fork-graph lookups
  --exclude-task-ids PATH   optional: task ids to exclude (a JSON list, a JSON object with a `task_ids` list, or one id per line)
Outputs:
  <out-dir>/rows.parquet        the drawn rows in draw order: apex_train.data_format.ROW_COLUMNS (difficulty = band) +
                                schema.PROVENANCE_FIELDS (category = the task type) + ROW_EXTRA columns (git-ignored)
  <out-dir>/candidates.jsonl    every admitted instance with its chosen attempt (the supply the draw reads)
  <out-dir>/selection.json      the picks in draw order and the draw report (targets, shortfalls, shares, switches)
  <out-dir>/supply.json         admitted supply per (language group, band) and (language, band), and the funnel
  <out-dir>/manifest.json       funnel, gate counts, draw report, checks, inputs, templates, revisions, output sha256
  <out-dir>/dropped.jsonl.gz    every dropped trajectory with its stage and reason
  WORK/<run_id>/                index.parquet, evals/raw_profiles.jsonl, evals/outcomes.jsonl, evals/rows/*.parquet
                                (--resume reuses every record of the current RECORD_VERSION)
  logs/<run_id>/                metadata.json, train.log, metrics.jsonl, errors.log, resources.csv (+ manifest copy)
Seeded (--seed, default apex_data.paths.SEED): the draw's cell orders.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import build_teacher_leg as legacy  # noqa: E402  (index cache, leak rule, round-trip check, publish)
from apex_data import efficiency, paths  # noqa: E402
from apex_data.grade import tamper  # noqa: E402
from apex_data.pools.common import repo_key  # noqa: E402
from apex_data.render.adapters import AdapterError, adapt  # noqa: E402
from apex_data.render.filters import FilterConfig, filter_episodes  # noqa: E402
from apex_data.schema import PROVENANCE_FIELDS, validate_provenance  # noqa: E402
from apex_data.screens import d2_repos  # noqa: E402
from apex_data.screens.licence import RepoLookup, github_token  # noqa: E402
from apex_data.teacher import filter as tfilter  # noqa: E402
from apex_data.teacher import load, render  # noqa: E402
from apex_data.teacher import select_v5 as sv  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import (ROW_COLUMNS, SampleValidationError, arrow_schema, encode_messages, histogram,  # noqa: E402
                                    load_tokenizer)

RECORD_VERSION = f"swe_v5:1|{efficiency.VERSION}"
RAW_SLACK = 0.05
SKIP_GATES = ("thrashing",)          # plan 5.1: the profiler's wasted-action detector replaces it
STAGES = ("raw_profile", "rebuild", "repo_root_ambiguous", "reference_patch_leak", "d2_text", "band_row_cap",
          *(f"gate_{g}" for g in ("verdict", "tamper", "adapt", "exit", "format_error_turn", "steps", "finish_length",
                                  "reasoning", "think_length", "language", "parse", "repetition", "constraints", "render",
                                  "row_length")), "profile")
PROVENANCE_TYPES = {"patch_files": pa.int64(), "patch_added_lines": pa.int64(), "fits_48k": pa.bool_()}
ROW_EXTRA = (("row_key", pa.string()), ("band", pa.string()), ("language_group", pa.string()),
             ("thinking_mean", pa.float64()), ("thinking_max", pa.int32()), ("steps", pa.int32()),
             ("recovery_arc", pa.bool_()), ("draw_order", pa.int32()), ("draw_tier", pa.int32()),
             ("recovery_preferred", pa.bool_()), ("repo_root", pa.string()), ("attempt", pa.int32()))
CACHE_SCHEMA = pa.schema([*arrow_schema(), ("trajectory_id", pa.string()), ("episode_json", pa.large_string())])
_W: dict[str, Any] = {}              # worker-process globals


# ------------------------------------------------------------------------------------------------ profiling
def profile_summary(p: efficiency.Profile) -> dict[str, Any]:
    return {"steps": p.steps, "thinking": p.thinking, "thinking_mean": round(p.thinking_mean, 2), "thinking_max": p.thinking_max,
            "output_tokens": p.output_tokens, "n_actions": p.n_actions, "wasted_actions": p.wasted_actions,
            "no_action_turns": p.no_action_turns, "completed": p.completed, "verified": p.verified,
            "recovery": p.recovery_arc is not None, "recovery_arc": p.recovery_arc}


def raw_prefilter_failures(p: efficiency.Profile, band: str, slack: float = RAW_SLACK) -> list[str]:
    """Stage 4: the profiler gates on the stored messages, thinking limits widened by `slack` (the rendered episode
    decides); steps, wasted actions and the verified completion exactly."""
    lim = efficiency.SWE_LIMITS[band]
    out = []
    if p.thinking_max > lim.turn_ceiling * (1 + slack):
        out.append("turn_ceiling")
    if p.thinking_mean > lim.mean_max * (1 + slack):
        out.append("mean_above")
    if p.thinking_mean < lim.mean_min * (1 - slack):
        out.append("mean_below")
    if p.steps > lim.steps_max:
        out.append("steps")
    if p.wasted_actions:
        out.append("wasted_actions")
    if not p.completed:
        out.append("not_completed")
    elif not p.verified:
        out.append("unverified_completion")
    return out


def model_patch(raw: dict[str, Any]) -> str | None:
    return ((raw.get("metadata") or {}).get("model_patch") or {}).get("patch")


def raw_record(raw: dict[str, Any], trajectory_id: str, band: str) -> dict[str, Any]:
    patch = model_patch(raw)
    p = efficiency.profile(raw["messages"], "swe", patch=patch, root=efficiency.TEACHER_REPO_ROOT)
    flags = tamper.swe_flags(patch, "submit")
    fails = raw_prefilter_failures(p, band) + (["tamper"] if flags else [])
    return {"trajectory_id": trajectory_id, "band": band, "ok": not fails, "fails": fails, "tamper_flags": flags,
            "profile": profile_summary(p), "version": RECORD_VERSION}


def _worker_raw(job: tuple[str, list[tuple[int, str, str]]]) -> list[dict[str, Any]]:
    shard_path, items = job
    table = pq.read_table(shard_path, columns=["trajectory_id", "messages", "metadata"])
    table = table.take(pa.array([row for row, _, _ in items], type=pa.int64()))
    out = []
    for k, (row, t, band) in enumerate(items):
        raw = table.slice(k, 1).to_pylist()[0]
        if raw["trajectory_id"] != t:
            raise RuntimeError(f"{shard_path} row {row}: trajectory id {raw['trajectory_id']} != {t}")
        out.append(raw_record(raw, t, band))
    return out


def _worker_raw_init() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    efficiency._qwen_tokenizer()


# ------------------------------------------------------------------------------------------------ rendering
def _outcome(it: dict[str, Any], stage: str | None, reason: str | None, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"trajectory_id": it["trajectory_id"], "instance_id": it["instance_id"], "subsource": it["subsource"],
            "language": it["language"], "band": it["band"], "ok": stage is None, "stage": stage, "reason": reason,
            "detail": detail[:300], "version": RECORD_VERSION, **extra}


def evaluate(raw: dict[str, Any], it: dict[str, Any], tok: Any, bundle: render.TemplateBundle, locked: Any, revision: str,
             repo_root: str | None, *, profile_gates: bool = True, row_max: int | None = None
             ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Stage 5 for one trajectory: (outcome record, cache row or None). `it` carries the instance fields (instance_id,
    subsource, language, repo, license, band, category, stratum, files, added). `profile_gates` off (v7) skips the
    band limits, zero wasted actions and the verified completion and the 32K loss cap (the caller masks turns and
    checks the loss cap afterwards); `row_max` overrides the band's row cap."""
    band = it["band"]
    lim = efficiency.SWE_LIMITS[band]
    cap = row_max or lim.row_max
    task_id = f"{load.SOURCE}:{it['instance_id']}"
    try:
        ep, st = render.build_episode(raw, bundle, task_id=task_id, episode_id=it["trajectory_id"], repo_root=repo_root)
    except render.RenderDrop as exc:
        return _outcome(it, "repo_root_ambiguous" if exc.reason == "repo_root_ambiguous" else "rebuild", exc.reason, exc.detail), None
    ref = ((raw.get("metadata") or {}).get("reference_patch") or {}).get("patch")
    matched, n_added = render.reference_patch_overlap(render.prompt_texts(ep), ref)
    extra = {"rebuild": st, "reference_patch_lines": n_added, "reference_patch_lines_in_prompt": len(matched)}
    if legacy.leaks(len(matched), n_added, 0.0):
        return _outcome(it, "reference_patch_leak", "reference_patch_line_in_prompt",
                        f"{len(matched)} of {n_added} non-test added lines", **extra), None
    d2 = d2_repos.screen(it["repo"], locked=locked, text=render.episode_text(ep.messages), upstream_id=it["instance_id"])
    if d2.drop:
        return _outcome(it, "d2_text", "d2_locked_repository" if d2.locked_hit else "decontam_excluded_repository",
                        ",".join(d2.mechanisms), **extra), None
    try:                     # exact full-render length first: an over-cap episode skips the per-turn passes of the gates
        sample = adapt(ep).sample
        n = len(encode_messages(tok, sample.template_messages(), sample.tools, enable_thinking=True))
        if n > cap:
            return _outcome(it, "band_row_cap", "too_long", f"{n} > {cap}", n_tokens=n, **extra), None
    except (AdapterError, SampleValidationError):
        pass                 # the adapt gate below records the reason
    task = {"task_id": task_id, "source": load.SOURCE, "family": "swe", "language": it["language"], "repo": it["repo"],
            "hf_revision": revision, "licence": {"repo_spdx": it["license"], "dataset": load.DATASET_LICENSE},
            "screens": {"d2": d2.as_mark()},
            "extra": {"provenance": {"hf_repo": load.HF_REPO, "generator": load.GENERATOR, "label_provenance": "verifier",
                                     "oswt_subsource": it["subsource"], "size_stratum": it["stratum"],
                                     "patch_files": it["files"], "patch_added_lines": it["added"], "category": it["category"]}}}
    verdict = {"r_pass": 1, "tamper_flags": tamper.swe_flags(ep.patch, ep.exit_reason), "verifier": "upstream_resolved"}
    res = filter_episodes([ep], {ep.episode_id: verdict}, {task_id: task}, tok, FilterConfig(max_row_tokens=cap),
                          skip_gates=SKIP_GATES)
    if res.dropped:
        d = res.dropped[0]
        return _outcome(it, f"gate_{d.gate}", d.reason, d.detail, **extra), None
    kept = res.kept[0]
    row = kept.row
    tokens = {"n_tokens": row["n_tokens"], "n_assistant_tokens": row["n_assistant_tokens"],
              "n_assistant_turns": row["n_assistant_turns"]}
    if not profile_gates:
        cache = {**row, "difficulty": band, "trajectory_id": it["trajectory_id"],
                 "episode_json": json.dumps({"messages": ep.messages, "tools": ep.tools, "extra": ep.extra}, ensure_ascii=False)}
        return _outcome(it, None, None, provenance=kept.provenance, **tokens, **extra), cache
    p = efficiency.profile(ep.messages, "swe", patch=ep.patch, root=repo_root or efficiency.TEACHER_REPO_ROOT)
    fails = efficiency.gate_failures(p, band, row_tokens=row["n_tokens"])
    if row["n_assistant_tokens"] > sv.LOSS_CAP_PER_ROW:
        fails.append("loss_cap")
    if fails:
        return _outcome(it, "profile", ",".join(fails), profile=profile_summary(p), **tokens, **extra), None
    cache = {**row, "difficulty": band, "trajectory_id": it["trajectory_id"],
             "episode_json": json.dumps({"messages": ep.messages, "tools": ep.tools, "extra": ep.extra}, ensure_ascii=False)}
    return _outcome(it, None, None, profile=profile_summary(p), provenance=kept.provenance, **tokens, **extra), cache


def _worker_render_init(tokenizer_dir: str, bundle: dict[str, Any], rows_dir: str, revision: str, repo_root: str) -> None:
    import transformers

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()          # the >262,144-token warning is expected for long candidates
    _W.update(tok=load_tokenizer(tokenizer_dir), bundle=render.TemplateBundle.from_dict(bundle), rows_dir=Path(rows_dir),
              locked=d2_repos.load_locked(), revision=revision, repo_root=repo_root)
    efficiency._qwen_tokenizer()


def _worker_render(job: tuple[str, str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    subsource, shard_path, items = job
    raws = load.read_rows(subsource, Path(shard_path), [it["row"] for it in items])
    outcomes, rows = [], []
    for it, raw in zip(items, raws, strict=True):
        if raw["trajectory_id"] != it["trajectory_id"]:
            raise RuntimeError(f"{shard_path} row {it['row']}: trajectory id {raw['trajectory_id']} != {it['trajectory_id']}")
        out, row = evaluate(raw, it, _W["tok"], _W["bundle"], _W["locked"], _W["revision"], _W["repo_root"])
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


# ------------------------------------------------------------------------------------------------ helpers
def instance_bands(index: list[load.IndexRow]) -> dict[tuple[str, str], str | None]:
    """{(subsource, instance_id): band} from every attempt of the instance (resolved 1 / 0 known, -1 unknown)."""
    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in index:
        per[(r.subsource, r.instance_id)].append(r.resolved)
    return {k: efficiency.swe_band(v) for k, v in per.items()}


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """The last record per trajectory of the current RECORD_VERSION."""
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() and (rec := json.loads(line)).get("version") == RECORD_VERSION:
                    out[rec["trajectory_id"]] = rec
    return out


def chunks(items: list[Any], n: int) -> list[list[Any]]:
    return [items[k:k + n] for k in range(0, len(items), n)]


def run_pool(run: common.RunDir, name: str, jobs: list[Any], worker: Any, workers: int, out_path: Path,
             records: dict[str, dict[str, Any]], initializer: Any = None, initargs: tuple = ()) -> None:
    """Map `worker` over `jobs` in a fork pool; every returned record is appended to `out_path` and `records`."""
    if not jobs:
        run.log(f"{name}: nothing to do ({len(records)} records cached)")
        return
    t0, n, n_ok = time.time(), 0, 0
    with mp.get_context("fork").Pool(min(workers, len(jobs)), initializer=initializer, initargs=initargs) as pool, \
            out_path.open("a", encoding="utf-8") as f:
        for k, recs in enumerate(pool.imap_unordered(worker, jobs), 1):
            for rec in recs:
                records[rec["trajectory_id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                n_ok += rec["ok"]
            f.flush()
            if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                run.log(f"{name}: {k}/{len(jobs)} jobs, {n} trajectories, {n_ok} pass, {time.time() - t0:.0f}s")
    run.metric({"stage": name, "jobs": len(jobs), "n": n, "n_ok": n_ok, "seconds": round(time.time() - t0, 1)})


def read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {line.strip() for line in text.splitlines() if line.strip()}
    ids = obj.get("task_ids", []) if isinstance(obj, dict) else obj
    return {str(x) for x in ids}


def sel_entry(p: sv.Pick, rec: dict[str, Any]) -> dict[str, Any]:
    c = p.candidate
    prof = rec["profile"]
    return {"order": p.order, "task_id": f"{load.SOURCE}:{c.instance_id}", "instance_id": c.instance_id,
            "subsource": c.subsource, "trajectory_id": c.trajectory_id, "row_key": f"{c.trajectory_id}:-1", "band": c.band,
            "language": c.language, "group": c.group, "category": c.category, "repo": c.repo, "pr": c.pr,
            "loss_tokens": c.loss_tokens, "seq_tokens": c.seq_tokens, "thinking_mean": c.thinking_mean,
            "thinking_max": prof["thinking_max"], "steps": prof["steps"], "recovery": c.recovery, "tier": p.tier,
            "recovery_preferred": p.recovery_preferred, "attempt": c.attempt}


def per_turn_quantile(values: list[int], q: float) -> float:
    v = sorted(values)
    return float(v[min(len(v) - 1, int(q * len(v)))]) if v else 0.0


# --------------------------------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oswt-root", type=Path, default=None, help="Open-SWE-Traces root (default RAW/oswt)")
    p.add_argument("--out-dir", type=Path, default=paths.POOLS / "swe_v5")
    p.add_argument("--harbor-python", type=Path, default=None, help="interpreter with mini-swe-agent 2.4.6 (default .venv/harbor/bin/python)")
    p.add_argument("--lock-config", type=Path, default=None, help="runner-written mini-swe-agent config the templates are asserted against")
    p.add_argument("--workers", type=int, default=24, help="worker processes (the host is shared: keep <= 32)")
    p.add_argument("--per-task-raw", type=int, default=400, help="trajectories per raw-profile job (one shard read per job)")
    p.add_argument("--per-task-render", type=int, default=80, help="trajectories per render job (one shard read per job)")
    p.add_argument("--limit-shards", type=int, default=None,
                   help="profile and render only the first N shards of each folder (smoke runs; bands still use every shard)")
    p.add_argument("--target-loss-tokens", type=int, default=sv.TARGET_LOSS_TOKENS, help="SWE loss-token target (plan: ~9M)")
    p.add_argument("--repo-root", default="/app", help="repository path our harness uses (the teacher's /testbed is rewritten)")
    p.add_argument("--exclude-task-ids", type=Path, default=None, help="task ids (oswt:<instance>) removed before the draw")
    p.add_argument("--no-category-round-robin", action="store_true", help="ladder step 1 (execution plan 0.2)")
    p.add_argument("--no-recovery-preference", action="store_true", help="ladder step 2")
    p.add_argument("--no-lower-half", action="store_true", help="ladder step 3")
    p.add_argument("--keep-all-bands", default="", help="comma-separated leading bands drawn whole (e.g. hard,medium)")
    p.add_argument("--band-share", action="append", default=[], metavar="BAND=SHARE",
                   help="override a band's loss-token share (with --keep-all-bands: its share of the SWE total)")
    p.add_argument("--repo-cap", action="append", default=[], metavar="GROUP=N",
                   help=f"instances per repository for a language group (default {sv.MAX_PER_REPO} for every group)")
    p.add_argument("--note", default=None, help="the decision behind non-default draw settings (recorded in the manifest)")
    p.add_argument("--second-attempt-bands", default="", help="comma-separated bands whose instances may add their next "
                   "gate-passing attempt (default none: one trajectory per instance)")
    p.add_argument("--max-second-share", type=float, default=0.3, help="second attempts as a share of all episodes, at most")
    p.add_argument("--no-fork-graph", action="store_true", help="skip GitHub fork-graph lookups (cache only; recorded)")
    p.add_argument("--roundtrip-rows", type=int, default=20)
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="run id (default data_swe_v5_<UTC>)")
    p.add_argument("--resume", default=None, metavar="RUN_ID", help="continue a run: reuse WORK/<RUN_ID>/ and logs/<RUN_ID>/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.resume or a.run_id or common.new_run_id("data_swe_v5")
    work = paths.WORK / run_id
    rows_dir = work / "evals" / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    switches = sv.DrawSwitches(not a.no_category_round_robin, not a.no_recovery_preference, not a.no_lower_half)
    keep_all = tuple(b for b in a.keep_all_bands.split(",") if b)
    band_shares = {**sv.BAND_LOSS_SHARES, **{k: float(v) for k, v in (x.split("=", 1) for x in a.band_share)}}
    repo_caps = {k: int(v) for k, v in (x.split("=", 1) for x in a.repo_cap)}
    second_bands = tuple(b for b in a.second_attempt_bands.split(",") if b)
    max_second = a.max_second_share if second_bands else 0.0
    draw_settings = {"keep_all_bands": list(keep_all), "band_shares": band_shares, "repo_caps": repo_caps,
                     "max_per_repository": sv.MAX_PER_REPO, "target_loss_tokens": None if keep_all else a.target_loss_tokens,
                     "second_attempt_bands": list(second_bands), "max_second_share": max_second,
                     "note": a.note}
    t_start = time.time()
    with common.RunDir(run_id, log_name="train.log", resume=bool(a.resume)) as run:
        revisions = load.raw_revisions()
        oswt_rev = revisions["oswt"]["revision"]
        if not oswt_rev:
            raise RuntimeError("data/manifests/raw_sources.json has no `oswt` revision; run scripts/data/fetch_raw.py first")
        run.write_metadata(run_id=run_id, step="data_swe_v5", command=common.command_line(), git=common.git_info(),
                           config=json.loads(json.dumps(vars(a), default=str)), seed=a.seed, hardware=common.hardware_info(),
                           data_versions=revisions, model_revision=common.model_revision(), work_dir=str(work),
                           record_version=RECORD_VERSION, switches=asdict(switches), draw_settings=draw_settings,
                           start_time=common.utc_now(), status="running")
        bundle = render.load_templates(a.harbor_python, lock_config=a.lock_config)
        run.log(f"mini-swe-agent {bundle.version} templates from {bundle.config_path} equal the lock reference "
                f"{bundle.lock_reference['path']}; bash tool sha256 {bundle.bash_tool_sha256[:12]}")
        tok = load_tokenizer(paths.BASE_MODEL_DIR)
        template_sha = hashlib.sha256(tok.chat_template.encode()).hexdigest()
        run.write_metadata(tokenizer={"path": str(paths.BASE_MODEL_DIR), "chat_template_sha256": template_sha},
                           templates={k: v for k, v in bundle.to_dict().items() if not k.endswith("_template")})

        # ---- 1-3: index, bands, row filters -----------------------------------------------------------------------
        shards = load.list_shards(a.oswt_root)           # bands need every attempt: an instance's attempts span shards
        shard_dirs = {s: p.parent for s, p in shards}
        processed = {(s, p.name) for s, p in load.list_shards(a.oswt_root, limit=a.limit_shards)}
        index = legacy.build_or_load_index(shards, work, a.workers, run)
        if len({r.trajectory_id for r in index}) != len(index):
            raise RuntimeError("trajectory ids are not unique across the shards")
        bands = instance_bands(index)
        run.log("instance bands (all attempts): " + json.dumps(dict(Counter(str(b) for b in bands.values()))))
        verified = load.scale_swe_verified_ids()
        upstream = load.load_upstream()
        dev = tfilter.load_dev_exclusions()
        if dev.status != "applied":
            raise RuntimeError("data/manifests/dev_exclusions.json is missing: dev tasks must be excluded (plan 5.3)")
        locked = d2_repos.load_locked()
        pre = [r for r in index if r.resolved == 1 and r.language in sv.LANGUAGE_CODES]
        repos = sorted({k for r in pre if (k := repo_key(r.repo))})
        lookup = RepoLookup(cache_dir=paths.WORK / "licence_cache", token=None if a.no_fork_graph else github_token(),
                            enabled=not a.no_fork_graph)
        fork_infos, fork_summary = lookup.lookup(repos, log=run.log)
        rows, funnel, drops, d2_report = tfilter.row_filters(
            index, bash_tool=bundle.bash_tool, verified=verified, locked=locked, fork_infos=fork_infos,
            fork_lookup_enabled=lookup.enabled, dev=dev, languages=tuple(sv.LANGUAGE_CODES))
        d2_report["fork_graph_lookup"] = fork_summary
        if a.limit_shards is not None:                                           # smoke runs only
            limited = [r for r in rows if (r.subsource, r.shard) in processed]
            funnel.step("smoke_limit_shards", rows, limited, tfilter.cell_of)
            rows = limited
        banded = [r for r in rows if bands[(r.subsource, r.instance_id)] in sv.BANDS]
        for r in rows:
            if bands[(r.subsource, r.instance_id)] not in sv.BANDS:
                b = bands[(r.subsource, r.instance_id)]
                drops.append(tfilter.Drop(r.subsource, r.instance_id, r.trajectory_id, r.repo, r.language, "7_banded",
                                          "single_attempt" if b is None else f"band_{b}"))
        funnel.step("7_banded", rows, banded, tfilter.cell_of)
        for s in funnel.steps:
            run.metric({"stage": "row_filter", **{k: v for k, v in s.items() if k != "by_cell"}})
        run.log("row filters: " + json.dumps([(s["filter"], s["n_out"]) for s in funnel.steps]))

        def instance_meta(r: load.IndexRow) -> dict[str, Any]:
            up = upstream.get((r.subsource, r.instance_id))
            files, added = (up.size.files, up.size.added) if up else (r.ref_files, r.ref_added)
            return {"instance_id": r.instance_id, "subsource": r.subsource, "language": sv.LANGUAGE_CODES[r.language],
                    "repo": r.repo, "license": r.license, "band": bands[(r.subsource, r.instance_id)], "category": r.category,
                    "stratum": load.size_stratum(files, added), "files": files, "added": added,
                    "pr": up.pr if up and up.pr else load.pr_number(r.instance_id)}

        meta = {r.trajectory_id: instance_meta(r) for r in banded}

        # ---- 4: raw profile -----------------------------------------------------------------------------------------
        raw_path = work / "evals" / "raw_profiles.jsonl"
        raw_recs = read_records(raw_path)
        missing: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for r in banded:
            if r.trajectory_id not in raw_recs:
                missing[str(shard_dirs[r.subsource] / r.shard)].append((r.row, r.trajectory_id, meta[r.trajectory_id]["band"]))
        jobs = [(shard, part) for shard, items in sorted(missing.items()) for part in chunks(sorted(items), a.per_task_raw)]
        run_pool(run, "raw_profile", jobs, _worker_raw, a.workers, raw_path, raw_recs, initializer=_worker_raw_init)
        raw_ok = [r for r in banded if raw_recs[r.trajectory_id]["ok"]]
        raw_fail_reasons = Counter(f for r in banded for f in raw_recs[r.trajectory_id]["fails"])
        run.log(f"raw profile: {len(raw_ok)} of {len(banded)} pass; failures by reason {dict(raw_fail_reasons)}")

        # ---- 5: render + gates --------------------------------------------------------------------------------------
        out_path = work / "evals" / "outcomes.jsonl"
        records = read_records(out_path)
        todo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in raw_ok:
            if r.trajectory_id not in records:
                todo[(r.subsource, r.shard)].append({"row": r.row, "trajectory_id": r.trajectory_id, **meta[r.trajectory_id]})
        jobs = [(sub, str(shard_dirs[sub] / shard), part) for (sub, shard), items in sorted(todo.items())
                for part in chunks(sorted(items, key=lambda x: x["row"]), a.per_task_render)]
        run_pool(run, "render", jobs, _worker_render, a.workers, out_path, records, initializer=_worker_render_init,
                 initargs=(str(paths.BASE_MODEL_DIR), bundle.to_dict(), str(rows_dir), oswt_rev, a.repo_root))
        evaluated = [records[r.trajectory_id] for r in raw_ok]
        run.log(f"render + gates: {sum(o['ok'] for o in evaluated)} of {len(evaluated)} pass; by stage "
                + json.dumps(dict(Counter(o["stage"] for o in evaluated if not o["ok"]))))

        # ---- 6-7: candidates and draw -------------------------------------------------------------------------------
        attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for o in evaluated:
            attempts[(o["subsource"], o["instance_id"])].append(
                {"ok": o["ok"], "trajectory_id": o["trajectory_id"], "thinking_mean": (o.get("profile") or {}).get("thinking_mean", 0.0)})
        excluded = read_exclusions(a.exclude_task_ids)
        candidates: list[sv.Candidate] = []
        n_excluded = 0
        for (sub, iid), atts in sorted(attempts.items()):
            best = sv.choose_attempt(atts)
            if best is None:
                continue
            if f"{load.SOURCE}:{iid}" in excluded:
                n_excluded += 1
                continue
            o, m = records[best["trajectory_id"]], meta[best["trajectory_id"]]
            first = sv.Candidate(sub, iid, o["trajectory_id"], m["band"], m["language"], m["category"], tfilter.normalise_repo(m["repo"]),
                                 m["pr"], o["n_assistant_tokens"], o["n_tokens"], o["profile"]["thinking_mean"], o["profile"]["recovery"])
            candidates.append(first)
            nxt = sv.choose_attempt([x for x in atts if x["trajectory_id"] != best["trajectory_id"]]) if m["band"] in second_bands else None
            if nxt is not None:
                o2 = records[nxt["trajectory_id"]]
                candidates.append(sv.Candidate(**{**asdict(first), "trajectory_id": o2["trajectory_id"],
                                                  "loss_tokens": o2["n_assistant_tokens"], "seq_tokens": o2["n_tokens"],
                                                  "thinking_mean": o2["profile"]["thinking_mean"],
                                                  "recovery": o2["profile"]["recovery"], "attempt": 2}))
        supply = sv.supply_table([c for c in candidates if c.attempt == 1])
        supply["second_attempts_by_band"] = dict(Counter(c.band for c in candidates if c.attempt == 2))
        run.log(f"candidates: {sum(c.attempt == 1 for c in candidates)} admitted instances (+ {sum(c.attempt == 2 for c in candidates)} "
                f"second attempts) ({n_excluded} excluded by --exclude-task-ids); by band "
                + json.dumps({b: supply["by_band"][b]["instances"] for b in sv.BANDS}))
        result = sv.draw(candidates, total_loss=a.target_loss_tokens, seed=a.seed, switches=switches, band_shares=band_shares,
                         repo_caps=repo_caps, keep_all=keep_all, max_second_share=max_second)
        report = sv.draw_report(result)
        run.log("draw: " + json.dumps({k: report[k] for k in ("episodes", "loss_tokens", "band_loss_shares", "group_shares",
                                                               "recovery_share", "category_shares")}))
        for b in sv.BANDS:
            br = report["bands"][b]
            run.log(f"band {b}: {br['episodes']} episodes, {br['loss_tokens']} of {br['target_loss_tokens']} loss tokens, "
                    f"groups {br['groups']}, recovery {br['recovery_share']}, exhausted {br['exhausted_groups']}")

        # ---- outputs ---------------------------------------------------------------------------------------------
        picks = result.picks
        sel_ids = {p.candidate.trajectory_id for p in picks}
        cache_rows: dict[str, dict[str, Any]] = {}
        for fname in sorted({records[t]["row_file"] for t in sel_ids}):
            for r in pq.read_table(rows_dir / fname).to_pylist():
                if r["trajectory_id"] in sel_ids:
                    cache_rows[r["trajectory_id"]] = r
        out_rows, d2_final, root_left = [], Counter(), 0
        for p in picks:
            c, t = p.candidate, p.candidate.trajectory_id
            r, prov, prof = cache_rows[t], records[t]["provenance"], records[t]["profile"]
            validate_provenance(prov)
            ep_text = render.episode_text(json.loads(r["episode_json"])["messages"])
            recheck = d2_repos.screen(prov["repo"], locked=locked, text=ep_text, upstream_id=c.instance_id)
            d2_final["locked_hit"] += recheck.locked_hit
            d2_final["excluded_hit"] += recheck.excluded_hit
            root_left += len(render.ROOT_TOKEN_RE.findall(ep_text))
            out_rows.append({**{k: r[k] for k in ROW_COLUMNS}, **{k: prov[k] for k in PROVENANCE_FIELDS},
                             "row_key": prov["row_key"], "band": c.band, "language_group": c.group,
                             "thinking_mean": c.thinking_mean, "thinking_max": prof["thinking_max"], "steps": prof["steps"],
                             "recovery_arc": c.recovery, "draw_order": p.order, "draw_tier": p.tier,
                             "recovery_preferred": p.recovery_preferred,
                             "repo_root": records[t]["rebuild"].get("repo_root", render.TEACHER_REPO_ROOT), "attempt": c.attempt})
        schema = pa.schema([*arrow_schema(), *[(k, PROVENANCE_TYPES.get(k, pa.string())) for k in PROVENANCE_FIELDS], *ROW_EXTRA])
        rows_path = a.out_dir / "rows.parquet"
        tmp = rows_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), tmp)
        legacy.publish(tmp, rows_path, run)

        # ---- checks -------------------------------------------------------------------------------------------------
        rt = legacy.roundtrip_check([{**cache_rows[p.candidate.trajectory_id],
                                      "task_id": records[p.candidate.trajectory_id]["provenance"]["task_id"]} for p in picks],
                                    tok, a.roundtrip_rows, a.seed)
        within = []
        for p in picks:
            o = records[p.candidate.trajectory_id]
            prof = o["profile"]
            lim = efficiency.SWE_LIMITS[p.candidate.band]
            within.append(prof["thinking_max"] <= lim.turn_ceiling and lim.mean_min <= prof["thinking_mean"] <= lim.mean_max
                          and prof["steps"] <= lim.steps_max and o["n_tokens"] <= lim.row_max
                          and o["n_assistant_tokens"] <= sv.LOSS_CAP_PER_ROW)
        lineage = Counter((p.candidate.repo, p.candidate.pr) for p in picks if p.candidate.pr and p.candidate.attempt == 1)
        per_repo = Counter(p.candidate.repo for p in picks if p.candidate.attempt == 1)       # instances, not episodes
        cap_of_repo = {p.candidate.repo: repo_caps.get(p.candidate.group, sv.MAX_PER_REPO) for p in picks}
        over_cap = sorted(r for r, n in per_repo.items() if n > cap_of_repo[r])
        checks = {
            "roundtrip": rt,
            "band_limits_all_rows": all(within),
            "zero_wasted_actions": all(records[t]["profile"]["wasted_actions"] == 0 for t in sel_ids),
            "verified_completion": all(records[t]["profile"]["completed"] and records[t]["profile"]["verified"] for t in sel_ids),
            "attempts_per_instance": all(n == 1 or (n == 2 and band in second_bands) for (_, band), n in
                                         Counter((p.candidate.key, p.candidate.band) for p in picks).items())
                                     and all(p.candidate.attempt == 1 or any(q.candidate.key == p.candidate.key and q.candidate.attempt == 1
                                                                             and q.order < p.order for q in picks) for p in picks),
            "max_rows_per_repository": report["max_rows_per_repository"], "repositories_over_cap": over_cap,
            "one_row_per_repo_pr": max(lineage.values(), default=0) <= 1,
            "d2_hits_selected_rows": d2_final["locked_hit"], "decontam_excluded_repository_rows": d2_final["excluded_hit"],
            "repo_root_tokens_left": root_left,
            "dev_exclusions": dev.status,
        }
        checks["pass"] = (rt["pass"] and checks["band_limits_all_rows"] and checks["zero_wasted_actions"]
                          and checks["verified_completion"] and checks["attempts_per_instance"]
                          and not over_cap and checks["one_row_per_repo_pr"]
                          and d2_final["locked_hit"] == 0 and d2_final["excluded_hit"] == 0 and root_left == 0)
        run.log(f"checks: {json.dumps({k: v for k, v in checks.items() if k != 'roundtrip'})}; round trip {rt['rows_checked']} "
                f"rows / {rt['assistant_turns_checked']} turns, failures {rt['failures'][:3]}")
        if not checks["pass"]:
            run.error(f"P1b checks failed: {json.dumps(checks)[:2000]}")

        # ---- records ------------------------------------------------------------------------------------------------
        dropped_path = a.out_dir / "dropped.jsonl.gz"
        with gzip.GzipFile(dropped_path.with_suffix(".gz.tmp"), "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8") as f:
            for d in drops:
                f.write(json.dumps(asdict(d)) + "\n")
            for r in banded:
                rr = raw_recs[r.trajectory_id]
                if not rr["ok"]:
                    f.write(json.dumps({"subsource": r.subsource, "instance_id": r.instance_id, "trajectory_id": r.trajectory_id,
                                        "repo": r.repo, "language": r.language, "stage": "raw_profile",
                                        "reason": ",".join(rr["fails"]), "detail": json.dumps(rr["tamper_flags"])}) + "\n")
            for o in evaluated:
                if not o["ok"]:
                    f.write(json.dumps({k: o.get(k) for k in ("subsource", "instance_id", "trajectory_id", "language", "band",
                                                              "stage", "reason", "detail", "n_tokens")}) + "\n")
        legacy.publish(dropped_path.with_suffix(".gz.tmp"), dropped_path, run)
        with (a.out_dir / "candidates.jsonl.tmp").open("w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps({**asdict(c), "task_id": f"{load.SOURCE}:{c.instance_id}", "group": c.group}) + "\n")
        legacy.publish(a.out_dir / "candidates.jsonl.tmp", a.out_dir / "candidates.jsonl", run)

        stage_rows = [{"subsource": r.subsource, "language": r.language, "stage": None if raw_recs[r.trajectory_id]["ok"] else
                       "raw_profile", "reason": None} for r in banded if not raw_recs[r.trajectory_id]["ok"]]
        rendered_funnel = legacy.staged_funnel(stage_rows + evaluated, STAGES)
        profile_reasons = Counter(x for o in evaluated if o["stage"] == "profile" for x in o["reason"].split(","))
        sel_thinking = [x for t in sel_ids for x in records[t]["profile"]["thinking"]]
        selection = {"run_id": run_id, "seed": a.seed, "report": report,
                     "rows": [sel_entry(p, records[p.candidate.trajectory_id]) for p in picks]}
        supply_doc = {"run_id": run_id, "record_version": RECORD_VERSION, "admitted": supply,
                      "funnel": {"row_filters": [{k: v for k, v in s.items() if k != "by_cell"} for s in funnel.steps],
                                 "raw_profile": {"n_in": len(banded), "n_out": len(raw_ok), "reasons": dict(raw_fail_reasons)},
                                 "rendered": {"n_in": len(evaluated), "n_out": sum(o["ok"] for o in evaluated),
                                              "by_stage": dict(Counter(o["stage"] for o in evaluated if not o["ok"])),
                                              "profile_reasons": dict(profile_reasons)},
                                 "instances_admitted": len(candidates), "excluded_task_ids": n_excluded}}
        common.write_json(a.out_dir / "selection.json", selection)
        common.write_json(a.out_dir / "supply.json", supply_doc)
        manifest = {
            "pool": "swe_v5", "source": load.SOURCE, "run_id": run_id, "built": common.utc_now(), "seed": a.seed,
            "plan": "data/manifests/v5/DATA_CARD_v5.md (SWE slice)", "code_commit": common.git_info().get("commit"),
            "record_version": RECORD_VERSION,
            "dataset": {"repo": load.HF_REPO, "revision": oswt_rev, "licence": load.DATASET_LICENSE,
                        "shards": {"n": len(shards), "processed": len(processed), "limited": a.limit_shards is not None}},
            "generator": {"name": load.GENERATOR, "thinking": True, "label_provenance": "verifier (upstream resolved == 1)"},
            "templates": {k: v for k, v in bundle.to_dict().items() if not k.endswith("_template")},
            "tokenizer": {"path": str(paths.BASE_MODEL_DIR), "chat_template_sha256": template_sha},
            "rules": {"bands": "efficiency.swe_band over every attempt with a known outcome; < 2 known = not used",
                      "limits": {b: asdict(l) for b, l in efficiency.SWE_LIMITS.items()},
                      "loss_cap_per_row": sv.LOSS_CAP_PER_ROW, "raw_prefilter_slack": RAW_SLACK, "skip_gates": list(SKIP_GATES),
                      "draw_settings": draw_settings, "language_shares": sv.LANGUAGE_SHARES,
                      "language_groups": sv.GROUP_OF, "gap_order": list(sv.GAP_ORDER), "recovery_min": sv.RECOVERY_MIN,
                      "category_cap": sv.CATEGORY_CAP,
                      "lineage": "one instance per (repository, PR)", "repo_root": a.repo_root,
                      "attempt_choice": "per instance the gate-passing attempt with the lowest rendered thinking_mean; in "
                                        "second_attempt_bands also the next one (attempt 2)"},
            "funnel": supply_doc["funnel"], "rendered_funnel": rendered_funnel,
            "band_counts_all_instances": dict(Counter(str(b) for b in bands.values())),
            "draw": report, "supply": supply,
            "selected_thinking_per_turn": {"mean": round(sum(sel_thinking) / len(sel_thinking), 1) if sel_thinking else 0.0,
                                           "p90": per_turn_quantile(sel_thinking, 0.9), "max": max(sel_thinking, default=0)},
            "selected_length_histogram": histogram([o["n_tokens"] for o in out_rows]),
            "d2": {**d2_report, "selected_rows_rescreened": dict(d2_final), "locked_list_sha256": locked.sha256},
            "dev_exclusions": {"status": dev.status, "source": dev.source, "sha256": dev.sha256},
            "checks": checks,
            "outputs": {"rows_parquet": str(rows_path), "rows_sha256": common.sha256_file(rows_path), "n_rows": len(out_rows),
                        "selection_sha256": common.sha256_file(a.out_dir / "selection.json"),
                        "supply_sha256": common.sha256_file(a.out_dir / "supply.json"),
                        "candidates_sha256": common.sha256_file(a.out_dir / "candidates.jsonl"),
                        "dropped_sha256": common.sha256_file(dropped_path), "work_dir": str(work)},
        }
        common.write_json(a.out_dir / "manifest.json", manifest)
        common.write_json(run.run_dir / "manifest.json", manifest)
        dur = time.time() - t_start
        rss_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1e6
        run.write_resources([
            ("duration_sec", round(dur, 1), "s", "wall clock on the CPU VM"),
            ("workers", a.workers, "proc", "profile and render processes"),
            ("trajectories_profiled", len(banded), "rows", "raw profile (stage 4)"),
            ("trajectories_rendered", len(evaluated), "rows", "render + gates (stage 5)"),
            ("peak_rss_gb", round(rss_gb, 2), "GB", "parent + largest child max RSS"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"), ("gpu_hours", 0.0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only")])
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(dur, 1), status="ok" if checks["pass"] else "checks_failed",
                           n_rows=len(out_rows), manifest=str(a.out_dir / "manifest.json"), rows_parquet=str(rows_path),
                           rows_sha256=manifest["outputs"]["rows_sha256"])
        run.finalize()
        return 0 if checks["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
