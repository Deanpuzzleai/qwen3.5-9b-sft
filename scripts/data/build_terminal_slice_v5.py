#!/usr/bin/env python
"""Build the revision-5 terminal slice (data/manifests/v5/DATA_CARD_v5.md, "Terminal slice"): TermGrade trials of the
teachers in priority order (terminal revision 3: Kimi-K3 > Qwen3.6-27B > Gemma-4 with
reasoning > GLM-5.2 > DeepSeek-V4-Pro), one trial per task plus a second for easy and medium tasks (<= 25% of the
episodes), every turn rendered as a per-turn row, drawn by difficulty band in loss tokens (select_v5, 2.5M). Revision 2
(Kimi-K3 only, one trial per task, 1.5M) is `--teachers kimi --budget 1500000 --no-second-trials`. The first build's
slice (DeepSeek-V4-Pro and TerminalHorizon, two turns per episode) is scripts/data/build_terminal_slice.py.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_terminal_slice_v5.py [--workers 32]
        [--limit-tasks 60 --budget 60000] [--run-id ID --resume]

Inputs (RAW = /mnt/apex-shared/apex_data/raw; every file's sha256 is checked against data/manifests/raw_sources.json)
  --messages-dir  RAW/tgr/data/messages/<teacher>.parquet  trajectories_messages per --teachers key
                                                          (ai-and/termgrade-trajectories)
  --trials    RAW/tgr/data/trials.parquet          per-trial rewards, joined on (task, model, trial)
  --labels    RAW/tgr_env/labels.jsonl             per-model pass rates; their six-model mean gives the band
  data/manifests/dev_exclusions.json (terminal_task_ids), configs/eval/terminal_bench_task_ids.txt (the D4 slug screen),
  the Qwen3.5-9B tokenizer (paths.BASE_MODEL_DIR), Harbor 0.24.0's Terminus-2 parser in .venv/harbor (round trip).
Outputs
  --out-dir (default data/processed/v5/terminal, git-ignored; with --limit-tasks: WORK/<run_id>/preview)
    rows.parquet      the drawn trials' rows in draw order: data_format.ROW_COLUMNS (difficulty = band) +
                      schema.PROVENANCE_FIELDS + bookkeeping (apex_data.terminal.rows; dup_group = the task-text cluster)
    episodes.jsonl    one line per drawn trial: task, trial, teacher, rank (0 first, 1 second), band, draw position and
                      half, profile, rows, loss tokens
    tasks.jsonl       one line per distinct drawn task {task_id, kind, text}: the scripts/data/run_screens.py --pool input
    candidates.jsonl  one line per task with a passing trial: band, every passing trial's teacher, stage, reason, gate
                      failures and thinking per turn, and the chosen trials (tasks not drawn stay stage-2 states, 3.2)
    supply.json       the supply table per band (labelled tasks -> drawn loss tokens), gate failures per band, the draw,
                      and the thinking per turn of the admitted, rejected and drawn trials
    manifest.json     inputs, parameters, funnel, supply, draw, token and data-card statistics, checks
  WORK/<run_id>/render/   render shards (every task's chosen trials; reused by --resume)
  logs/<run_id>/          metadata.json, train.log, metrics.jsonl, errors.log, resources.csv, render_check.jsonl,
                          dropped.jsonl (every trial not drawn, with the stage and reason that stopped it)

Steps:
  1. load (terminal.loaders, per teacher key: reward 1 in both files, scored, counted toward k, every turn parsed;
     user #1 re-rendered under our Harbor template), the per-episode screens (D1 on the task text, D4, identity,
     credentials) and the profile (efficiency.profile in Qwen3.5-9B tokens), in a process pool;
  2. D1 whole-file scans, dev exclusion, task-text dedup (screens.dedup: a near-duplicate of a dev task is out; one task
     per cluster);
  3. band per task (efficiency.terminal_band of the six-model mean) and the profiler gates (efficiency.gate_failures:
     band limits over the trained turns (plan 1.3: limits are per trained turn), zero wasted actions, verified
     completion);
  4. per task, the gate-passing trials by teacher priority, then lowest mean thinking (select_v5.trial_order), rendered
     with the render/filters gates at the band's row cap, every turn trained (the confirmation reply only in the seeded
     TERMINAL_CONFIRMATION_SHARE of episodes), until one (hard) or two (easy, medium) render whole with every row <= 32K
     loss tokens and pass the gates again with the thinking read from their rows (efficiency.trained_thinking);
  5. draw (select_v5.draw: new tasks first: hard once, the rest medium 55 / easy 45 by loss tokens, lower half by
     thinking first; then second trials into the budget left, <= 25% of the episodes);
  6. rows and the checks: row render and parser round trip on every row, the 20-row render check, D1 on the drawn
     episodes, the gates re-run with the rendered row length, confirmation share <= 10%, band row caps, the 32K loss
     cap, dev absence, at most one trial per hard task and two per other task, second trials <= 25%. Exit 1 when a
     check fails.
Seeded by --seed (apex_data.paths.SEED).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import efficiency, paths  # noqa: E402
from apex_data.render.adapters import TERMINAL_CONFIRMATION_SHARE, confirmation_replies, trains_confirmation  # noqa: E402
from apex_data.render.filters import FilterConfig, filter_episodes  # noqa: E402
from apex_data.schema import validate_provenance  # noqa: E402
from apex_data.screens import d1_canary, d4_structural, dedup  # noqa: E402
from apex_data.terminal import admission, checks, harness, loaders, rows as rowsio, select_v5  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402
from apex_train.data_format import load_tokenizer  # noqa: E402

CONFIRMATION_SHARE_MAX = 0.10       # plan 3.2 / 5.1: at most 10% pure confirmation turns
ROW_LOSS_CAP = 32_768               # plan 3.4: at most 32K loss tokens per row
CHUNK = 4                           # tasks per render job
RENDER_CHECK_ROWS = 20


# --------------------------------------------------------------------------------------------------------- inputs
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--messages-dir", type=Path, default=paths.RAW / "tgr" / "data" / "messages")
    p.add_argument("--teachers", default=",".join(select_v5.TEACHERS),
                   help="TermGrade model keys in priority order (default %(default)s)")
    p.add_argument("--no-second-trials", action="store_true",
                   help="one trial per task (revision 2); default: a second trial per easy and medium task")
    p.add_argument("--trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet")
    p.add_argument("--labels", type=Path, default=paths.RAW / "tgr_env" / "labels.jsonl")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES)
    p.add_argument("--dev-exclusions", type=Path, default=paths.DEV_EXCLUSIONS)
    p.add_argument("--allow-unpinned", action="store_true",
                   help="development only: run on files that do not match raw_sources.json (recorded; not final)")
    p.add_argument("--budget", type=int, default=select_v5.BUDGET_LOSS_TOKENS, help="loss-token budget (default %(default)s)")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default data/processed/v5/terminal (WORK/<run_id>/preview with --limit-tasks)")
    p.add_argument("--work-dir", type=Path, default=None, help="default WORK/<run_id>")
    p.add_argument("--workers", type=int, default=32, help="processes for loading, profiling, rendering and checks")
    p.add_argument("--limit-tasks", type=int, default=None, metavar="N",
                   help="a seeded random sample of N tasks (small end-to-end runs; not final)")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="logs/<run_id>/ (default data_terminal_slice_v5_<UTC timestamp>)")
    p.add_argument("--resume", action="store_true", help="continue --run-id, reusing its render shards")
    p.add_argument("--log-root", type=Path, default=None, help="write logs/<run_id>/ under this root (tests)")
    a = p.parse_args(argv)
    a.teachers = tuple(t for t in a.teachers.split(",") if t)
    unknown = [t for t in a.teachers if t not in loaders.TGR_TEACHERS]
    if unknown or not a.teachers:
        p.error(f"unknown teacher keys {unknown}; known: {sorted(loaders.TGR_TEACHERS)}")
    return a


def verify_inputs(inputs: dict[str, Path], manifest: Path) -> dict[str, Any]:
    """sha256 of every input against its raw_sources.json entry (the file under the source's `dest`)."""
    sources = json.loads(manifest.read_text()).get("sources", {}) if manifest.exists() else {}
    out: dict[str, Any] = {}
    for name, path in inputs.items():
        path = Path(path).resolve()
        rec: dict[str, Any] = {"path": str(path), "sha256": rowsio.sha256_file(path), "source": None,
                               "revision": None, "pinned_sha256": None}
        for key, src in sources.items():
            dest = Path(src.get("dest") or "/nonexistent").resolve()
            if dest in path.parents:
                rel = str(path.relative_to(dest))
                hit = next((f for f in src.get("files") or [] if f.get("path") == rel), None)
                rec.update(source=key, revision=src.get("revision"), pinned_sha256=hit and hit.get("sha256"))
        rec["matches"] = rec["sha256"] == rec["pinned_sha256"]
        out[name] = rec
    return out


def six_model_bands(labels_path: Path) -> tuple[dict[str, str], dict[str, float]]:
    """{task: band}, {task: six-model mean pass rate} (efficiency.terminal_band; the probes' and dev set's rule)."""
    rates: dict[str, list[float]] = defaultdict(list)
    for line in labels_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rates[r["task"]].append(float(r["pass_rate"]))
    mean = {t: sum(v) / len(v) for t, v in rates.items()}
    return {t: efficiency.terminal_band(m) for t, m in mean.items()}, mean


def write_json(path: Path, obj: Any) -> str:
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records))
    tmp.replace(path)
    return rowsio.sha256_file(path)


# -------------------------------------------------------------------------------------------- per-trial state
@dataclass
class Trial:
    loaded: loaders.Loaded
    screens: dict[str, Any] | None
    profile: efficiency.Profile | None
    trained: efficiency.Profile | None = None   # the profile with `thinking` over the trained turns only
    stage: str = "candidate"            # the step that stopped it, or candidate / chosen / selected
    reason: str = ""
    detail: str = ""
    gate_failures: list[str] = field(default_factory=list)
    teacher: str = select_v5.TEACHERS[0]

    @property
    def task(self) -> str:
        return self.loaded.info["task"]

    @property
    def trial(self) -> str:
        return self.loaded.upstream_id

    @property
    def eid(self) -> str | None:
        return self.loaded.episode["episode_id"] if self.loaded.episode else None

    def drop(self, stage: str, reason: str, detail: str = "") -> None:
        if self.stage == "candidate":
            self.stage, self.reason, self.detail = stage, reason, detail


_G: dict[str, Any] = {}     # worker globals (fork-inherited for loading, set by the initializer for rendering)


def _token_counts(texts: list[str]) -> list[int]:
    if "tok" not in _G:
        _G["tok"] = load_tokenizer(_G["tokenizer_dir"])
    texts = [t or "" for t in texts]
    return [len(ids) for ids in _G["tok"](texts, add_special_tokens=False)["input_ids"]] if texts else []


def _load_worker(item: tuple[str, dict[str, Any], float | None, str | None]) -> Trial:
    teacher, row, reward, started = item
    loaded = loaders.load_tgr_record(row, revision=_G["revision"], trial_reward=reward, started_at=started,
                                     model_key=teacher)
    screens = prof = trained = None
    if loaded.episode is not None:
        msgs = loaded.episode["messages"]
        screens = admission.screen_episode(msgs, loaded.info.get("instruction") or "", loaded.info["task"],
                                           _G["slug_patterns"])
        prof = efficiency.profile(msgs, "terminal", count=_token_counts)
        asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        trained = replace(prof, thinking=[prof.thinking[asst.index(i)] for i in trainable_turns(loaded.episode)])
    return Trial(loaded, screens, prof, trained, teacher=teacher)


def load_all(a: argparse.Namespace, run: RunDir, revision: str, tasks: set[str] | None,
             files: dict[str, Path]) -> list[Trial]:
    _G.update(revision=revision, tokenizer_dir=str(a.tokenizer),
              slug_patterns=d4_structural.slug_patterns(d4_structural.load_tb_slugs()))
    items = ((t, *x) for t in a.teachers for x in loaders.read_tgr(files[t], a.trials)
             if tasks is None or x[0]["task"] in tasks)
    out: list[Trial] = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, t in enumerate(pool.imap(_load_worker, items, chunksize=4), 1):
            out.append(t)
            if n % 1000 == 0:
                run.log(f"loaded and profiled {n} trials")
    return out


# ---------------------------------------------------------------------------------------------- funnel steps
def apply_exclusions(trials: list[Trial], d1_trials: set[str], dev_tasks: set[str]) -> None:
    for t in trials:
        if t.loaded.drop:
            t.drop("load", t.loaded.drop, t.loaded.detail)
        elif t.task in dev_tasks:
            t.drop("exclusions", "dev_task")
        else:
            reason = admission.screen_drop_reason(t.screens, t.trial in d1_trials)
            if reason:
                t.drop("screens", reason, json.dumps({k: t.screens[k] for k in ("d1_reasons", "d4_reasons", "identity",
                                                                                "credentials")}))


def apply_dedup(trials: list[Trial], dev_tasks: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Task-text clusters (screens.dedup.terminal_task_clusters) over one item per task (any trial's instruction; dev
    tasks as the dev side). A task clustered with a dev task is out; within a cluster of training tasks only the
    representative (smallest id) stays. Returns {task: cluster representative} and the counts."""
    text: dict[str, str] = {}
    for t in trials:
        if t.loaded.info.get("instruction"):
            text.setdefault(t.task, t.loaded.info["instruction"])
    items = [dedup.DedupItem(task, txt, split="dev" if task in dev_tasks else "train") for task, txt in sorted(text.items())]
    cl = dedup.terminal_task_clusters(items)
    members: dict[str, list[str]] = defaultdict(list)
    for task in text:
        members[cl.group_of.get(task, task)].append(task)
    out_reason: dict[str, tuple[str, str]] = {}
    stats = Counter()
    for rep, tasks in members.items():
        stats["clusters"] += 1
        if len(tasks) < 2:
            continue
        stats["clusters_multi_task"] += 1
        dev = [x for x in tasks if x in dev_tasks]
        stats["clusters_with_dev_and_train"] += bool(dev) and len(dev) < len(tasks)
        for x in tasks:
            if x in dev_tasks:
                continue
            if dev:
                out_reason[x] = ("near_duplicate_of_dev_task", ",".join(sorted(dev)))
            elif x != min(t for t in tasks if t not in dev_tasks):
                out_reason[x] = ("near_duplicate_task", rep)
    for t in trials:
        if t.task in out_reason:
            t.drop("dedup", *out_reason[t.task])
    rep_of = {task: cl.group_of.get(task, task) for task in text}
    return rep_of, {"method": "screens.dedup.terminal_task_clusters (exact normalised task text + MinHash-LSH, Jaccard >= "
                              f"{dedup.TERMINAL_THRESHOLD} on word {dedup.TERMINAL_NGRAM}-grams), one item per task",
                    "tasks": len(items), "dev_tasks": sum(i.split == "dev" for i in items), **dict(stats),
                    "tasks_dropped": dict(Counter(r for r, _ in out_reason.values())),
                    **{f"minhash_{k}": v for k, v in cl.counts().items()}}


def apply_gates(trials: list[Trial], bands: dict[str, str]) -> None:
    for t in trials:
        if t.stage != "candidate":
            continue
        band = bands.get(t.task)
        if band not in efficiency.TERMINAL_LIMITS:
            t.drop("band", f"band_{band}")
            continue
        t.gate_failures = efficiency.gate_failures(t.trained, band)
        if t.gate_failures:
            t.drop("profile_gates", ",".join(t.gate_failures))


# ------------------------------------------------------------------------------------------ render (worker pool)
def trainable_turns(episode: dict[str, Any]) -> list[int]:
    """Message indices of the turns the rows train: every assistant turn, the confirmation reply only in the seeded
    TERMINAL_CONFIRMATION_SHARE of episodes (render/adapters.adapt_terminus)."""
    msgs = episode["messages"]
    conf = set(confirmation_replies(msgs))
    keep = trains_confirmation(episode["episode_id"])
    return [i for i, m in enumerate(msgs) if m["role"] == "assistant" and (keep or i not in conf)]


def _render_init(tokenizer_dir: str) -> None:
    _G["tok"] = load_tokenizer(tokenizer_dir)


def _render_one(entry: dict[str, Any], t: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str, str]:
    """Rows of one trial, or None with the failing step and detail."""
    ep = t["episode"]
    msgs = ep["messages"]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    cfg = FilterConfig(max_row_tokens=entry["cap"], terminal_turns_per_episode=len(asst))
    res = filter_episodes([ep], {ep["episode_id"]: t["verdict"]}, {t["task"]["task_id"]: t["task"]}, _G["tok"], cfg)
    if res.dropped:
        d = res.dropped[0]
        return None, f"{d.gate}:{d.reason}", d.detail[:200]
    want = trainable_turns(ep)
    got = sorted(r.row["turn_index"] for r in res.kept)
    if got != want:
        stats = res.survival["terminal_turn"].stats
        why = sorted(k for k in stats if k.startswith("row_drop_") and k not in ("row_drop_not_candidate",
                                                                                  "row_drop_not_selected"))
        return None, "render:turn_not_rendered", f"missing turns {sorted(set(want) - set(got))}: {why}"
    over = [r.row["n_assistant_tokens"] for r in res.kept if r.row["n_assistant_tokens"] > ROW_LOSS_CAP]
    if over:
        return None, "render:row_loss_cap", f"{max(over)} > {ROW_LOSS_CAP}"
    spans = [efficiency.trained_thinking(r.row["input_ids"], r.row["assistant_masks"]) for r in res.kept]
    if any(len(x) != 1 or x[0] is None for x in spans):
        return None, "render:trained_span", f"{[len(x) for x in spans]} spans, closed {[x[:1] for x in spans]}"
    thinking = [x[0] for x in spans]
    fails = efficiency.gate_failures(replace(t["profile"], thinking=thinking), entry["band"],
                                     row_tokens=max(r.row["n_tokens"] for r in res.kept), limits=entry.get("limits"))
    if fails:
        return None, "render:rendered_gates", ",".join(fails)
    conf = set(confirmation_replies(msgs))
    records = []
    for r in res.kept:
        validate_provenance(r.provenance)
        rec = rowsio.flat_row(r.row, r.provenance)
        rec["difficulty"] = entry["band"]
        rec["dup_group"] = entry["dup_group"]
        i = r.row["turn_index"]
        rec["_meta"] = {"turn_index": i, "n_tokens": rec["n_tokens"], "n_assistant_tokens": rec["n_assistant_tokens"],
                        "thinking_tokens": thinking[len(records)], "confirmation": i in conf}
        records.append(rec)
    return records, "rendered", ""


def _render_worker(job: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    shard, entries = job
    shard_path, meta_path = Path(shard + ".parquet"), Path(shard + ".json")
    records: list[dict[str, Any]] = []
    metas = []
    for entry in entries:
        tries = []
        chosen = []
        for t in entry["trials"]:
            if len(chosen) == entry["want"]:
                break
            rows, outcome, detail = _render_one(entry, t)
            tries.append({"trial": t["trial"], "outcome": outcome, "detail": detail})
            if rows is not None:
                chosen.append({"trial": t["trial"], "teacher": t["teacher"], "episode_id": t["episode"]["episode_id"],
                               "rows": [r.pop("_meta") for r in rows]})
                records.extend(rows)
        metas.append({"task": entry["task"], "band": entry["band"], "chosen": chosen, "tries": tries})
    rowsio.write_table(records, shard_path)
    tmp = meta_path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(metas))
    tmp.replace(meta_path)
    return metas


def render_all(entries: list[dict[str, Any]], a: argparse.Namespace, work: Path, run: RunDir) -> tuple[list[dict[str, Any]], list[Path]]:
    """Render every task's candidates in order; shards are keyed by their tasks and trials, so --resume skips them."""
    out_dir = work / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: e["task"])
    jobs, done, shards = [], [], []
    for i in range(0, len(entries), CHUNK):
        part = entries[i:i + CHUNK]
        key = hashlib.sha256(json.dumps([[e["task"], e["cap"], e["want"], [t["trial"] for t in e["trials"]]]
                                         for e in part]).encode()).hexdigest()[:12]
        shard = str(out_dir / f"chunk_{i // CHUNK:05d}_{key}")
        shards.append(Path(shard + ".parquet"))
        if Path(shard + ".json").exists() and Path(shard + ".parquet").exists():
            done.extend(json.loads(Path(shard + ".json").read_text()))
            continue
        jobs.append((shard, part))
    run.log(f"render: {len(jobs)} shards to build, {len(shards) - len(jobs)} reused ({len(entries)} tasks)")
    with mp.get_context("spawn").Pool(a.workers, initializer=_render_init, initargs=(str(a.tokenizer),)) as pool:
        for n, metas in enumerate(pool.imap_unordered(_render_worker, jobs), 1):
            done.extend(metas)
            if n % 50 == 0 or n == len(jobs):
                run.log(f"render: {n}/{len(jobs)} shards")
    return done, shards


# ----------------------------------------------------------------------------------------------- statistics
def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def card_block(eps: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for e in eps for r in e["rows"]]
    think = [r["thinking_tokens"] for r in rows]
    loss = sum(r["n_assistant_tokens"] for r in rows)
    return {"episodes": len(eps), "rows": len(rows), "sequence_tokens": sum(r["n_tokens"] for r in rows),
            "loss_tokens": loss, "loss_tokens_per_episode_mean": round(loss / len(eps), 1) if eps else None,
            "thinking_per_trained_turn_mean": round(statistics.fmean(think), 1) if think else None,
            "thinking_per_trained_turn_p90": _pct(think, 0.9),
            "steps_p50": _pct([e["profile"]["steps"] for e in eps], 0.5),
            "steps_mean": round(statistics.fmean(e["profile"]["steps"] for e in eps), 2) if eps else None,
            "output_tokens_per_trajectory_p50": _pct([e["profile"]["output_tokens"] for e in eps], 0.5),
            "row_tokens_max": max((r["n_tokens"] for r in rows), default=0),
            "loss_tokens_per_row_max": max((r["n_assistant_tokens"] for r in rows), default=0),
            "confirmation_rows": sum(r["confirmation"] for r in rows)}


GATED_OUT = ("load", "exclusions", "screens", "dedup", "band")     # stages before the profiler gates


def thinking_distribution(trials: list[Trial], bands: dict[str, str]) -> dict[str, Any]:
    """The teachers' reasoning tokens per trained turn (and per trial, their mean) of the trials that reached the
    profiler gates: admitted (passed them), rejected (failed them) and drawn, overall, per band and per teacher. One
    unit for all three groups: the profiler's count of reasoning_content (rejected trials are never rendered)."""
    def block(ts: list[Trial]) -> dict[str, Any]:
        turns = [x for t in ts for x in t.trained.thinking]
        means = [t.trained.thinking_mean for t in ts]
        q = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        return {"trials": len(ts), "turns": len(turns),
                "per_turn_mean": round(statistics.fmean(turns), 1) if turns else None,
                "per_turn": {f"p{round(x * 100)}": _pct(turns, x) for x in q},
                "trial_mean": {f"p{round(x * 100)}": round(_pct(means, x), 1) if means else None for x in q}}

    reached = [t for t in trials if t.stage not in GATED_OUT and t.trained is not None]
    groups = {"admitted": [t for t in reached if t.stage != "profile_gates"],
              "rejected": [t for t in reached if t.stage == "profile_gates"],
              "drawn": [t for t in reached if t.stage == "selected"]}
    return {"unit": "Qwen3.5-9B tokens of reasoning_content per trained turn (efficiency.profile; the untrained "
                    "confirmation replies excluded)",
            "all": {g: block(ts) for g, ts in groups.items()},
            **{b: {g: block([t for t in ts if bands.get(t.task) == b]) for g, ts in groups.items()} for b in select_v5.BANDS},
            "by_teacher": {k: {g: block([t for t in ts if t.teacher == k]) for g, ts in groups.items()}
                           for k in sorted({t.teacher for t in reached})}}


def supply_table(trials: list[Trial], bands: dict[str, str], dev_tasks: set[str],
                 chosen: dict[tuple[str, int], dict[str, Any]], dr: select_v5.Draw) -> list[dict[str, Any]]:
    by_band: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_band[bands.get(t.task)].append(t)
    draw_row = {r["band"]: r for r in dr.table}
    out = []
    for band in (*select_v5.BANDS, "unsolved", None):
        ts = by_band.get(band, [])
        labelled = [x for x, b in bands.items() if b == band]
        loaded = [t for t in ts if t.loaded.drop is None]
        gate_fail = Counter(f for t in ts if t.stage == "profile_gates" for f in t.gate_failures)
        passing = [t for t in loaded if t.stage not in (*GATED_OUT, "profile_gates")]
        row: dict[str, Any] = {
            "band": band, "labelled_tasks": len(labelled), "dev_tasks": sum(x in dev_tasks for x in labelled),
            "tasks_with_teacher_pass": len({t.task for t in loaded if t.task not in dev_tasks}),
            "passing_trials": sum(t.task not in dev_tasks for t in loaded),
            "passing_trials_by_teacher": dict(Counter(t.teacher for t in loaded if t.task not in dev_tasks)),
            "gate_passing_trials_by_teacher": dict(Counter(t.teacher for t in passing)),
            "tasks_after_screens_and_dedup": len({t.task for t in loaded if t.stage not in ("exclusions", "screens", "dedup")}),
            "trials_after_screens_and_dedup": sum(t.stage not in ("exclusions", "screens", "dedup") for t in loaded),
            "gate_passing_trials": sum(t.stage not in (*GATED_OUT, "profile_gates") for t in loaded),
            "tasks_with_gate_passing_trial": len({t.task for t in loaded if t.stage not in (*GATED_OUT, "profile_gates")}),
            "gate_failures_by_trial": dict(gate_fail.most_common()),
            "tasks_rendered": sum(1 for (_, rank), c in chosen.items() if rank == 0 and c["band"] == band),
            "second_trials_rendered": sum(1 for (_, rank), c in chosen.items() if rank == 1 and c["band"] == band),
        }
        if band in draw_row:
            row.update({k: v for k, v in draw_row[band].items() if k != "band"})
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------- checks
def _check_worker(job: tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]) -> list[tuple[str, list[str]]]:
    rows, msgs = job
    return [(r["row_key"], checks.check_row(_G["tok"], r, msgs[r["episode_id"]])) for r in rows]


def check_rows(rows: list[dict[str, Any]], msgs: dict[str, list[dict[str, Any]]], a: argparse.Namespace) -> dict[str, Any]:
    jobs = [(rows[i:i + 8], {r["episode_id"]: msgs[r["episode_id"]] for r in rows[i:i + 8]}) for i in range(0, len(rows), 8)]
    problems: Counter = Counter()
    bad: list[dict[str, Any]] = []
    n = 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_render_init, initargs=(str(a.tokenizer),)) as pool:
        for res in pool.imap_unordered(_check_worker, jobs):
            for key, probs in res:
                n += 1
                problems.update(probs)
                if probs:
                    bad.append({"row_key": key, "problems": probs})
    return {"harbor_parser": str(harness.HARBOR_PARSER_FILE) if checks.harbor_parser() else "absent",
            "rows_checked": n, "rows_clean": n - len(bad), "problems": dict(problems), "examples": bad[:10]}


def render_check(rows: list[dict[str, Any]], msgs: dict[str, list[dict[str, Any]]], a: argparse.Namespace,
                 path: Path) -> dict[str, Any]:
    picks = random.Random(f"{a.seed}:terminal_v5:render_check").sample(rows, min(RENDER_CHECK_ROWS, len(rows)))
    tok = load_tokenizer(a.tokenizer)
    ex = [checks.render_excerpt(tok, r, msgs[r["episode_id"]]) for r in picks]
    write_jsonl(path, ex)
    return {"rows": len(ex), "history_think_blocks_total": sum(e["history_think_blocks"] for e in ex),
            "rows_with_history": sum(e["history_assistant_turns"] > 0 for e in ex),
            "trained_turn_with_reasoning": sum(e["trained_reasoning_chars"] > 0 for e in ex),
            "rows_with_problems": sum(bool(e["problems"]) for e in ex), "file": str(path)}


# ---------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_terminal_slice_v5")
    work = (a.work_dir or paths.WORK / run_id).resolve()
    out_dir = (a.out_dir or (work / "preview" if a.limit_tasks else paths.PROCESSED / "v5" / "terminal")).resolve()
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        second = not a.no_second_trials
        files = {t: a.messages_dir / f"{t}.parquet" for t in a.teachers}
        inputs = {**{f"messages_{t}": f for t, f in files.items()}, "trials": a.trials, "labels": a.labels}
        input_check = verify_inputs(inputs, a.raw_sources)
        mismatched = [k for k, v in input_check.items() if not v["matches"]]
        if mismatched and not a.allow_unpinned:
            raise SystemExit(f"inputs not matching raw_sources.json sha256: {mismatched}; --allow-unpinned for development")
        revision = input_check[f"messages_{a.teachers[0]}"]["revision"] or "unpinned"
        drift = harness.template_drift()
        if drift:
            raise SystemExit(drift)
        dev_tasks = set(json.loads(a.dev_exclusions.read_text()).get("terminal_task_ids") or [])
        if not dev_tasks:
            raise SystemExit(f"{a.dev_exclusions} lists no terminal_task_ids")
        git = git_info()
        run.write_metadata(run_id=run_id, step="data_terminal_slice_v5", command=command_line(), git=git,
                           code_commit=git["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={k: v["revision"] for k, v in input_check.items()},
                           inputs={k: str(v) for k, v in inputs.items()}, model_revision=str(a.tokenizer),
                           hardware=hardware_info(), seed=a.seed, start_time=utc_now(), status="running",
                           work_dir=str(work), out_dir=str(out_dir), teachers=list(a.teachers), second_trials=second)
        work.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        bands, mean_rate = six_model_bands(a.labels)
        run.log(f"inputs: {len(bands)} labelled tasks {dict(Counter(bands.values()))}; {len(dev_tasks)} dev tasks")

        limit = None
        if a.limit_tasks:
            import pyarrow.parquet as pq
            all_tasks = sorted(set(pq.read_table(files[a.teachers[0]], columns=["task"]).column("task").to_pylist()))
            limit = set(random.Random(f"{a.seed}:terminal_v5:limit").sample(all_tasks, min(a.limit_tasks, len(all_tasks))))

        # 1. load, screens, profile
        trials = load_all(a, run, revision, limit, files)
        run.log(f"loaded {len(trials)} trials; passing {dict(Counter(t.teacher for t in trials if t.loaded.drop is None))}")

        # 2. D1 whole files, exclusions, screens, dedup
        d1_files = {t: d1_canary.scan_parquet(files[t], id_keys=("trial",)) for t in a.teachers}
        apply_exclusions(trials, {h.record for r in d1_files.values() for h in r.hits}, dev_tasks)
        rep_of, dedup_report = apply_dedup(trials, dev_tasks)
        run.log(f"D1 whole-file hits {sum(r.counts()['n_hits'] for r in d1_files.values())}; "
                f"dedup {dedup_report.get('tasks_dropped')}")

        # 3. bands and profiler gates
        apply_gates(trials, bands)
        run.metric({"step": "gates", **Counter(t.stage for t in trials)})

        # 4. render per task: teacher priority, then lowest thinking; one trial (hard) or two (easy, medium)
        by_task: dict[str, list[Trial]] = defaultdict(list)
        for t in trials:
            if t.stage == "candidate":
                by_task[t.task].append(t)
        entries = []
        for task, ts in by_task.items():
            band = bands[task]
            idx = {t.trial: t for t in ts}
            order = select_v5.trial_order({t.trial: (t.trained.thinking_mean, t.gate_failures) for t in ts},
                                          {t.trial: t.teacher for t in ts}, a.teachers)
            entries.append({"task": task, "band": band, "cap": efficiency.TERMINAL_LIMITS[band].row_max,
                            "want": 2 if second and band in select_v5.SECOND_BANDS else 1,
                            "dup_group": f"tgr:{rep_of.get(task, task)}",
                            "trials": [{"trial": tr, "teacher": idx[tr].teacher, "episode": idx[tr].loaded.episode,
                                        "verdict": idx[tr].loaded.verdict, "task": idx[tr].loaded.task,
                                        "profile": idx[tr].profile} for tr in order]})
        metas, shards = render_all(entries, a, work, run)
        by_trial = {(t.task, t.trial): t for t in trials}
        chosen: dict[tuple[str, int], dict[str, Any]] = {}
        for m in metas:
            for tr in m["tries"]:
                if tr["outcome"] != "rendered":
                    by_trial[(m["task"], tr["trial"])].drop("render", tr["outcome"], tr["detail"])
            for rank, c in enumerate(m["chosen"]):
                chosen[(m["task"], rank)] = {**c, "band": m["band"]}
                by_trial[(m["task"], c["trial"])].stage = "chosen"
        for t in trials:
            if t.stage == "candidate":
                t.drop("choice", "higher_priority_trial_chosen")
        run.log(f"rendered: {sum(r == 0 for _, r in chosen)} of {len(entries)} tasks have a trial "
                f"{dict(Counter(c['band'] for (_, r), c in chosen.items() if r == 0))}; "
                f"{sum(r == 1 for _, r in chosen)} second trials")

        # 5. draw: new tasks first, then second trials
        choices = [select_v5.TaskChoice(task, c["band"], c["trial"], statistics.fmean(r["thinking_tokens"] for r in c["rows"]),
                                        sum(r["n_assistant_tokens"] for r in c["rows"]), rank)
                   for (task, rank), c in chosen.items()]
        dr = select_v5.draw(choices, seed=a.seed, budget=a.budget)
        drawn = {(d.choice.task, d.choice.rank): d for d in dr.selected}
        for key, c in chosen.items():
            t = by_trial[(key[0], c["trial"])]
            if key in drawn:
                t.stage = "selected"
            else:
                t.stage, t.reason = "draw", "not_drawn"
        distinct = {task for task, _ in drawn}
        run.log(f"drawn: {len(drawn)} episodes ({len(distinct)} tasks), {dr.loss_tokens} loss tokens; " +
                ", ".join(f"{r['band']} {r['drawn_episodes']} / {r['drawn_loss_tokens']}" for r in dr.table))

        # 6. rows and episode records
        eids = {chosen[key]["episode_id"]: key for key in drawn}
        rows = [r for s in shards for r in rowsio.read_table(s, episode_ids=set(eids))]
        rows.sort(key=lambda r: (drawn[eids[r["episode_id"]]].order, r["turn_index"]))
        files_out: dict[str, Any] = {}
        sha = rowsio.write_table(rows, out_dir / "rows.parquet")
        files_out["rows.parquet"] = {"sha256": sha, "bytes": (out_dir / "rows.parquet").stat().st_size, "rows": len(rows)}
        msgs = {by_trial[(key[0], chosen[key]["trial"])].eid: by_trial[(key[0], chosen[key]["trial"])].loaded.episode["messages"]
                for key in drawn}
        eps = []
        for d in dr.selected:
            c = d.choice
            t = by_trial[(c.task, c.trial)]
            eps.append({"episode_id": t.eid, "task_id": t.loaded.task["task_id"], "task": c.task, "trial": c.trial,
                        "teacher": t.teacher, "rank": c.rank, "band": c.band,
                        "six_model_mean": round(mean_rate[c.task], 4), "draw_order": d.order, "half": d.half,
                        "generator": t.loaded.task["extra"]["provenance"]["generator"],
                        "licence": loaders.TGR_TEACHERS[t.teacher].licence,
                        "dup_group": f"tgr:{rep_of.get(c.task, c.task)}", "loss_tokens": c.loss_tokens,
                        "trained_thinking_mean": round(c.thinking_mean, 1), "profile": t.profile.to_dict(),
                        "rows": chosen[(c.task, c.rank)]["rows"]})
        files_out["episodes.jsonl"] = {"sha256": write_jsonl(out_dir / "episodes.jsonl", eps), "rows": len(eps)}
        seen_tasks: set[str] = set()
        task_items = []
        for e in eps:
            if e["task"] not in seen_tasks:
                seen_tasks.add(e["task"])
                task_items.append({"task_id": e["task_id"], "kind": "terminal", "upstream_id": e["task"],
                                   "text": by_trial[(e["task"], e["trial"])].loaded.info["instruction"]})
        files_out["tasks.jsonl"] = {"sha256": write_jsonl(out_dir / "tasks.jsonl", task_items), "rows": len(task_items)}
        cand = []
        loaded_by_task: dict[str, list[Trial]] = defaultdict(list)
        for t in trials:
            if t.loaded.drop is None:
                loaded_by_task[t.task].append(t)
        for task in sorted(loaded_by_task):
            ts = sorted(loaded_by_task[task], key=lambda t: (a.teachers.index(t.teacher), t.trial))
            cand.append({"task": task, "band": bands.get(task),
                         "chosen": [chosen[(task, r)]["trial"] for r in (0, 1) if (task, r) in chosen],
                         "drawn": [chosen[(task, r)]["trial"] for r in (0, 1) if (task, r) in drawn],
                         "trials": [{"trial": t.trial, "teacher": t.teacher, "stage": t.stage, "reason": t.reason,
                                     "gate_failures": t.gate_failures,
                                     "trained_thinking_mean": round(t.trained.thinking_mean, 1) if t.trained else None,
                                     "thinking_per_turn": t.profile.thinking if t.profile else None,
                                     "steps": t.profile.steps if t.profile else None} for t in ts]})
        files_out["candidates.jsonl"] = {"sha256": write_jsonl(out_dir / "candidates.jsonl", cand), "rows": len(cand)}

        # 7. checks
        row_checks = check_rows(rows, msgs, a)
        run.log(f"row checks: {row_checks['rows_clean']}/{row_checks['rows_checked']} clean {row_checks['problems']}")
        rc = render_check(rows, msgs, a, run.run_dir / "render_check.jsonl")
        final_d1 = d1_canary.scan_records(({"episode_id": e, "messages": m} for e, m in msgs.items()),
                                          source="drawn_episodes").counts()
        regate = {e["episode_id"]: efficiency.gate_failures(
            replace(by_trial[(e["task"], e["trial"])].profile, thinking=[r["thinking_tokens"] for r in e["rows"]]),
            e["band"], row_tokens=max(r["n_tokens"] for r in e["rows"])) for e in eps}
        conf_rows = sum(r["confirmation"] for e in eps for r in e["rows"])
        n_rows = sum(len(e["rows"]) for e in eps)
        per_task = Counter(e["task"] for e in eps)
        n_second = sum(e["rank"] == 1 for e in eps)
        checks_out = {
            "row_checks": row_checks, "render_check_20": rc, "d1_drawn_episodes": final_d1,
            "gates_rerun_with_row_length": {"failing_episodes": {k: v for k, v in regate.items() if v}},
            "confirmation_share": {"rows": conf_rows, "of": n_rows, "share": round(conf_rows / n_rows, 4) if n_rows else 0,
                                   "max": CONFIRMATION_SHARE_MAX},
            "rows_over_band_cap": sum(r["n_tokens"] > efficiency.TERMINAL_LIMITS[r["difficulty"]].row_max for r in rows),
            "rows_over_loss_cap": sum(r["n_assistant_tokens"] > ROW_LOSS_CAP for r in rows),
            "dev_tasks_drawn": sorted({e["task"] for e in eps if e["task"] in dev_tasks}),
            "distinct_trials": len(eps) == len({e["trial"] for e in eps}),
            "trials_per_task": {"max": max(per_task.values(), default=0),
                                "hard_tasks_with_2": sum(per_task[e["task"]] > 1 for e in eps if e["band"] == "hard"),
                                "ok": all(n <= (1 if bands[t] == "hard" else 2 if second else 1) for t, n in per_task.items())},
            "second_trials": {"episodes": n_second, "share": round(n_second / len(eps), 4) if eps else 0,
                              "max": select_v5.SECOND_SHARE_MAX if second else 0.0},
            "rows_match_episode_records": len(rows) == n_rows,
            "verified_outcomes": all(by_trial[(e["task"], e["trial"])].loaded.verdict["r_pass"] == 1 for e in eps),
        }
        ok = (row_checks["rows_checked"] == row_checks["rows_clean"] == len(rows) and not rc["rows_with_problems"]
              and final_d1["n_hits"] == 0 and not checks_out["gates_rerun_with_row_length"]["failing_episodes"]
              and checks_out["confirmation_share"]["share"] <= CONFIRMATION_SHARE_MAX
              and not checks_out["rows_over_band_cap"] and not checks_out["rows_over_loss_cap"]
              and not checks_out["dev_tasks_drawn"] and checks_out["distinct_trials"]
              and checks_out["trials_per_task"]["ok"]
              and checks_out["second_trials"]["share"] <= checks_out["second_trials"]["max"]
              and checks_out["rows_match_episode_records"] and checks_out["verified_outcomes"])
        checks_out["all_pass"] = ok

        # 8. reports
        with (run.run_dir / "dropped.jsonl").open("w") as f:
            for t in trials:
                if t.stage != "selected":
                    f.write(json.dumps({"task": t.task, "trial": t.trial, "episode_id": t.eid, "stage": t.stage,
                                        "reason": t.reason, "detail": t.detail[:200]}) + "\n")
        stages = ["load", "exclusions", "screens", "dedup", "band", "profile_gates", "render", "choice", "draw"]
        teacher_block = {}
        for k in a.teachers:
            te = [e for e in eps if e["teacher"] == k]
            teacher_block[k] = {
                "model": loaders.TGR_TEACHERS[k].generator, "licence": loaders.TGR_TEACHERS[k].licence,
                "priority": a.teachers.index(k) + 1, "trials": sum(t.teacher == k for t in trials),
                "passing": sum(t.teacher == k and t.loaded.drop is None for t in trials),
                "gate_passing": sum(t.teacher == k and t.stage not in (*GATED_OUT, "profile_gates", "load") for t in trials),
                "drawn_tasks": len({e["task"] for e in te}), "drawn_episodes": len(te),
                "drawn_second_trials": sum(e["rank"] == 1 for e in te),
                "drawn_loss_tokens": sum(e["loss_tokens"] for e in te),
                "drawn_episodes_by_band": {b: sum(e["band"] == b for e in te) for b in select_v5.BANDS}}
        n = len(trials)
        funnel = []
        for st in stages:
            dropped = [t for t in trials if t.stage == st]
            funnel.append({"step": st, "n_in": n, "n_dropped": len(dropped), "n_out": n - len(dropped),
                           "reasons": dict(Counter(t.reason for t in dropped).most_common(25))})
            n -= len(dropped)
        supply = {"plan": "data/manifests/v5/DATA_CARD_v5.md (terminal slice); terminal revision 3", "run_id": run_id,
                  "teachers": list(a.teachers), "second_trials": second,
                  "budget_loss_tokens": a.budget, "medium_share_of_rest": select_v5.MEDIUM_SHARE,
                  "second_share_max": select_v5.SECOND_SHARE_MAX if second else 0.0,
                  "drawn_loss_tokens": dr.loss_tokens, "drawn_episodes": len(drawn), "drawn_tasks": len(distinct),
                  "drawn_second_trials": n_second, "by_teacher": teacher_block,
                  "loss_share_by_band": {b: round(sum(e["loss_tokens"] for e in eps if e["band"] == b) / dr.loss_tokens, 4)
                                         if dr.loss_tokens else 0 for b in select_v5.BANDS},
                  "table": supply_table(trials, bands, dev_tasks, chosen, dr),
                  "thinking_distribution": thinking_distribution(trials, bands),
                  "limits": {b: vars(v) for b, v in efficiency.TERMINAL_LIMITS.items()}}
        files_out["supply.json"] = {"sha256": write_json(out_dir / "supply.json", supply)}
        final = (not a.limit_tasks and not mismatched and a.budget == select_v5.BUDGET_LOSS_TOKENS
                 and a.teachers == select_v5.TEACHERS and second)
        manifest = {
            "slice": "terminal_v5", "plan": "data/manifests/v5/DATA_CARD_v5.md", "run_id": run_id,
            "created": utc_now(), "code_commit": git["commit"], "code_dirty": git.get("dirty"), "seed": a.seed,
            "final": final, "development_flags": {"limit_tasks": a.limit_tasks, "unpinned_inputs": mismatched,
                                                  "budget": a.budget, "teachers": list(a.teachers),
                                                  "second_trials": second},
            "decision": "terminal revision 3: teachers by priority Kimi-K3 > "
                        "Qwen3.6-27B > Gemma-4 with reasoning > GLM-5.2 > DeepSeek-V4-Pro, a second trial per easy and "
                        "medium task (<= 25% of episodes, new tasks first), 2.5M loss tokens; gates unchanged",
            "teachers": teacher_block, "dataset_licence": loaders.DATASET_LICENSE,
            "kimi_serving": "the dataset authors served Kimi-K3's open weights on their own endpoints "
                            "(trials.serving_stack h200 / amd), so no API terms apply",
            "inputs": input_check, "efficiency_version": efficiency.VERSION,
            "exclusions": {"dev_terminal_task_ids": len(dev_tasks),
                           "dev_exclusions_sha256": rowsio.sha256_file(a.dev_exclusions)},
            "files": files_out, "d1_whole_file": {t: r.counts() for t, r in d1_files.items()}, "dedup": dedup_report,
            "screens": admission.screen_counts([t.screens for t in trials if t.screens is not None]),
            "funnel": funnel, "supply": supply["table"], "draw": dr.table,
            "rules": {"choice": "per task, the gate-passing trials by teacher priority, then lowest mean thinking per "
                                "trained turn, that render whole: one for hard tasks, two for easy and medium",
                      "draw": "new tasks first (hard once; medium 55 / easy 45 of the rest), then second trials into "
                              "the budget left, at most 25% of the episodes",
                      "turns": "every assistant turn is a per-turn row; the confirmation reply only in the seeded "
                               f"{TERMINAL_CONFIRMATION_SHARE:.0%} of episodes (render/adapters.trains_confirmation)",
                      "row_caps": {b: v.row_max for b, v in efficiency.TERMINAL_LIMITS.items()},
                      "row_loss_cap": ROW_LOSS_CAP},
            "card": {"all": card_block(eps), **{b: card_block([e for e in eps if e["band"] == b]) for b in select_v5.BANDS},
                     "by_teacher": {k: card_block([e for e in eps if e["teacher"] == k]) for k in a.teachers
                                    if any(e["teacher"] == k for e in eps)},
                     "distinct_tasks": len(distinct)},
            "stage2_states": "TermGrade tasks not in episodes.jsonl (and not dev) stay available as stage-2 states "
                             "(plan 3.2, 4.4); candidates.jsonl lists every task's trials",
            "checks": checks_out,
        }
        files_out["manifest.json"] = {"sha256": write_json(out_dir / "manifest.json", manifest)}
        run.metric({"step": "done", "drawn_episodes": len(drawn), "drawn_tasks": len(distinct), "rows": len(rows),
                    "loss_tokens": dr.loss_tokens, "checks_pass": ok})
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("workers", a.workers, "processes", "loading, profiling, rendering and checks"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if ok else "failed",
                           outcome="terminal v5 slice built" + ("" if final else " (development run, not final)"),
                           summary={"drawn_episodes": len(drawn), "drawn_tasks": len(distinct), "rows": len(rows),
                                    "loss_tokens": dr.loss_tokens,
                                    "by_band": {r["band"]: [r["drawn_episodes"], r["drawn_loss_tokens"]] for r in dr.table}},
                           outputs={k: {"path": str(out_dir / k), **v} for k, v in files_out.items()})
        if not ok:
            run.error(f"checks failed: {json.dumps({k: v for k, v in checks_out.items() if k != 'render_check_20'}, default=str)[:2000]}")
        run.finalize()
        run.log(f"done in {run.elapsed():.0f} s; outputs in {out_dir}; checks {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
