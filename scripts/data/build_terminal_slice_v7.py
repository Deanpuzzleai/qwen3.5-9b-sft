#!/usr/bin/env python
"""Build the v7 terminal slice: half DeepSeek-V4-Pro, half the Kimi-K3 body (replacing an earlier 2% / 10% long-turn
cap; data/manifests/v7/DATA_CARD_v7.md). Reuses build_terminal_slice_v6 (admission, render, role order), the v7 supply
probe's turn features and the strong failure detector.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_terminal_slice_v7.py [--workers 48] [--limit-tasks 60]

Halves (each ~--budget / 2 loss tokens, drawn with select_v5.draw: hard once, medium 55 / easy 45, hard cap):
  kimi      Kimi-K3 > Qwen3.6-27B > Gemma-4 with reasoning; v6 admission (graded pass, completed, verified, zero wasted
            actions, steps <= 20); v6's trial choice with debugging-arc trials switched in up to 40% of first trials; a
            second trial (>= 6 steps) as v6. Row caps by band (32K / 48K / 64K).
  deepseek  DeepSeek-V4-Pro graded-pass trials (the render gates still require the completion handshake and clean turns);
            per task the debugging-arc trials first, then the lowest mean thinking; one trial per task. Rows up to
            65,536 tokens, so late turns with long contexts are trained.
Turns (both halves; thinking read from the rendered row, efficiency.trained_thinking). A turn is trainable when:
  - it is not the confirmation reply, its reasoning is closed, its row has <= 32K loss tokens;
  - thinking <= THINK_MAX (4,000); kimi half only: thinking <= ROUTINE_MAX (1,000) unless a decision point (right after
    a genuine failure, apex_data.terminal.failures, or the first substantive command);
  - its content parses as Terminus-2 JSON (ends in a parsed command);
  - its reasoning is not repetitive: no 30-word run occurs twice, and distinct / all lines of >= 20 characters >= 0.8;
  - it does not send empty keystrokes to a screen stuck at a `>` continuation prompt.
  Per episode: turns that send C-c to a stuck prompt or a running command are always trained (when trainable), then
  v6's turn_roles order (verify_before_complete, the hand-in turn, after_error / long_job / recovery round robin) and a
  seeded fill, up to TRAINED_TURNS (C-c turns may exceed it); the hand-in turn exactly once (a trial whose hand-in turn
  is not trainable is skipped).

Inputs: RAW/tgr/data/messages/{kimi,qwen,gemma_think,deepseek}.parquet, trials.parquet, RAW/tgr_env/labels.jsonl
  (sha256 against raw_sources.json), data/manifests/dev_exclusions.json, the Qwen3.5-9B tokenizer.
Outputs: --out-dir (default data/processed/v7/terminal, git-ignored): rows.parquet (kimi half then deepseek half, draw
  order), episodes.jsonl (half, teacher, rank, band, steps, arc, trained turns with kind, index and features),
  tasks.jsonl, supply.json, manifest.json (report and checks; copied to data/manifests/v7/terminal_slice.json);
  logs/<run_id>/: the five run files, render_check.jsonl, quality_read.jsonl (8 DeepSeek trained turns, 4 of them at
  index >= 15, and 4 C-c turns, with row keys and excerpts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import resource
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import build_terminal_slice_v5 as B  # noqa: E402
import build_terminal_slice_v6 as V6  # noqa: E402
import probe_terminal_supply_v7 as P  # noqa: E402
from apex_data import efficiency as E, paths  # noqa: E402
from apex_data.render import turn_roles  # noqa: E402
from apex_data.render.filters import terminus_parse  # noqa: E402
from apex_data.screens import d1_canary  # noqa: E402
from apex_data.terminal import admission, failures, harness, loaders, rows as rowsio, select_v5  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

KIMI_TEACHERS = ("kimi", "qwen", "gemma_think")
DEEPSEEK = "deepseek"
HALVES = ("kimi", "deepseek")
BUDGET_LOSS_TOKENS = 3_300_000
THINK_MAX = 4_000
ROUTINE_MAX = 1_000
DEEPSEEK_ROW_MAX = 65_536          # max_length: late turns with ~50K contexts are trained
ARC_SHARE_MAX = 0.40               # kimi half
LATE_INDEX = 15                    # assistant-turn ordinal counted as "late" in the report
TRAINED_TURNS = V6.TRAINED_TURNS
WORD_RUN = 30
LONG_LINE = 20
DISTINCT_MIN = 0.8
CHUNK = 4
DETECTORS = {"strong": failures.failure_lines, "loose": turn_roles.error_lines}
_CONTINUATION = re.compile(r"^\s*>(\s.*)?$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--messages-dir", type=Path, default=paths.RAW / "tgr" / "data" / "messages")
    p.add_argument("--trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet")
    p.add_argument("--labels", type=Path, default=paths.RAW / "tgr_env" / "labels.jsonl")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES)
    p.add_argument("--dev-exclusions", type=Path, default=paths.DEV_EXCLUSIONS)
    p.add_argument("--budget", type=int, default=BUDGET_LOSS_TOKENS, help="both halves (default %(default)s)")
    p.add_argument("--failure-detector", choices=("strong", "loose"), default="strong")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--manifest-copy", type=Path, default=paths.MANIFESTS / "v7" / "terminal_slice.json")
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit-tasks", type=int, default=None, metavar="N")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-root", type=Path, default=None)
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------------------- guards
def repetitive(text: str) -> bool:
    """A 30-word run occurs twice without overlapping, or distinct / all lines of >= 20 characters < 0.8."""
    words = text.split()
    first: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - WORD_RUN + 1):
        j = first.setdefault(tuple(words[i:i + WORD_RUN]), i)
        if i - j >= WORD_RUN:
            return True
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= LONG_LINE]
    return bool(lines) and len(set(lines)) / len(lines) < DISTINCT_MIN


def continuation_prompt(output: str) -> bool:
    """The screen's last non-blank line is a `>` continuation prompt (heredoc, unclosed quote)."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return bool(lines) and bool(_CONTINUATION.match(lines[-1]))


def stuck_or_running(output: str) -> bool:
    """The screen does not end at a shell prompt: a continuation / REPL prompt, a pager, or a command still running."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return bool(lines) and not turn_roles._PROMPT_LINE.match(lines[-1])


def trainable(row: dict[str, Any], half: str) -> tuple[bool, str]:
    """The v7 guards on a rendered row's meta (thinking from the row; features from the transcript)."""
    t = row["thinking_tokens"]
    if row["confirmation"]:
        return False, "confirmation"
    if t is None:
        return False, "unclosed_reasoning"
    if row["n_assistant_tokens"] > V6.ROW_LOSS_CAP:
        return False, "row_loss_cap"
    if t > THINK_MAX:
        return False, "over_4000"
    if half == "kimi" and t > ROUTINE_MAX and row["kind"] != "decision":
        return False, "routine_over_1000"
    if not row["parsed"]:
        return False, "unparsed"
    if row["repetitive"]:
        return False, "repetitive"
    if row["empty_keys"] and row["at_continuation"]:
        return False, "empty_keys_at_continuation"
    return True, row["kind"]


# -------------------------------------------------------------------------------------------- render (worker pool)
def _augment(msgs: list[dict[str, Any]], meta: dict[str, Any], feats: dict[str, Any]) -> None:
    by_i = {t["i"]: t for t in feats["turns"]}
    for r in meta["rows"]:
        i = r["turn_index"]
        f = by_i[i]
        p = terminus_parse(msgs[i].get("content") or "")
        keys = [k for k, _ in p.commands] if not p.error else []
        prev = turn_roles.terminal_output(msgs[i - 1].get("content") or "") if msgs[i - 1]["role"] == "user" else ""
        r.update(k=f["k"], after_error=f["after_error"], first_substantive=f["first_substantive"],
                 kind="decision" if f["after_error"] or f["first_substantive"] else "routine", parsed=not p.error,
                 empty_keys=not p.error and not p.task_complete and not any(k.strip() for k in keys),
                 at_continuation=continuation_prompt(prev),
                 ctrl_c=any(k.strip() == "C-c" for k in keys) and stuck_or_running(prev),
                 repetitive=repetitive((msgs[i].get("reasoning_content") or "").strip()))
    meta["debug_arc"] = feats["debug_arc"]


def _worker(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> list[dict[str, Any]]:
    shard, pairs = job
    records, metas = [], []
    for entry, t in pairs:
        rows, meta = V6._render_trial(entry, t)
        if meta["outcome"] == "rendered":
            msgs = t["episode"]["messages"]
            _augment(msgs, meta, P.turn_features(t["teacher"], t["trial"], msgs, t["thinking"], DETECTORS[entry["detector"]]))
        records.extend(rows)
        metas.append({**meta, "task": entry["task"], "band": entry["band"], "half": entry["half"]})
    rowsio.write_table(records, Path(shard + ".parquet"))
    Path(shard + ".json").write_text(json.dumps(metas))
    return metas


def render_all(pairs: list[tuple[dict[str, Any], dict[str, Any]]], a: argparse.Namespace, work: Path,
               run: RunDir) -> tuple[list[dict[str, Any]], list[Path]]:
    out_dir = work / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(pairs, key=lambda x: x[1]["trial"])
    jobs, shards = [], []
    for k in range(0, len(pairs), CHUNK):
        part = pairs[k:k + CHUNK]
        key = hashlib.sha256(json.dumps([[e["cap"], t["trial"]] for e, t in part]).encode()).hexdigest()[:12]
        shard = str(out_dir / f"chunk_{k // CHUNK:05d}_{key}")
        jobs.append((shard, part))
        shards.append(Path(shard + ".parquet"))
    jobs.sort(key=lambda j: -sum(len(json.dumps(t["episode"]["messages"])) for _, t in j[1]))
    done: list[dict[str, Any]] = []
    with mp.get_context("spawn").Pool(a.workers, initializer=B._render_init, initargs=(str(a.tokenizer),)) as pool:
        for n, metas in enumerate(pool.imap_unordered(_worker, jobs), 1):
            done.extend(metas)
            if n % 500 == 0 or n == len(jobs):
                run.log(f"render: {n}/{len(jobs)} shards")
    return done, shards


# ----------------------------------------------------------------------------------------------------- selection
def select_turns(meta: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]] | None, str]:
    """C-c turns first, then v6's order and a seeded fill, up to TRAINED_TURNS; the hand-in turn must be trainable."""
    ok = {r["turn_index"]: r for r in meta["rows"] if trainable(r, meta["half"])[0]}
    if meta["hand_in"] is None or meta["hand_in"] not in ok:
        return None, "hand_in_not_trainable"
    picked: dict[int, str] = {i: "ctrl_c" for i in sorted(ok) if ok[i]["ctrl_c"]}
    for i, role in meta["order"]:
        if len(picked) >= TRAINED_TURNS and meta["hand_in"] in picked:
            break
        if i in ok and i not in picked:
            picked[i] = "hand_in" if i == meta["hand_in"] else role
    rest = sorted(set(ok) - set(picked))
    random.Random(f"{seed}:{meta['episode_id']}").shuffle(rest)
    for i in rest[:max(0, TRAINED_TURNS - len(picked))]:
        picked[i] = "random"
    return [{**ok[i], "selected_by": picked[i]} for i in sorted(picked)], "selected"


def choose(metas: dict[str, dict[str, Any]], trials_by_task: dict[str, list[B.Trial]], bands: dict[str, str],
           half: str, seed: int) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """kimi: v6's first trial, arc trials switched in up to ARC_SHARE_MAX, v6's second trial. deepseek: arc trials
    first, then the lowest mean thinking; one trial per task."""
    sel: dict[str, tuple[list[dict[str, Any]] | None, str]] = {}

    def selected(trial: str) -> list[dict[str, Any]] | None:
        if trial not in sel:
            m = metas.get(trial)
            sel[trial] = select_turns(m, seed) if m and m["outcome"] == "rendered" else (None, "not_rendered")
        return sel[trial][0]

    plain: dict[str, list[B.Trial]] = {}
    first: dict[str, B.Trial] = {}
    best_arc: dict[str, B.Trial] = {}
    for task, ts in trials_by_task.items():
        if half == "kimi":
            ranked = sorted(ts, key=lambda t: (KIMI_TEACHERS.index(t.teacher), V6.order_mean(t), t.trial))
        else:
            ranked = sorted(ts, key=lambda t: (not metas[t.trial].get("debug_arc"), V6.order_mean(t), t.trial))
        ok = [t for t in ranked if selected(t.trial)]
        if not ok:
            continue
        plain[task], first[task] = ok, ok[0]
        arc = next((t for t in ok if metas[t.trial]["debug_arc"]), None)
        if arc is not None:
            best_arc[task] = arc
    n_arc = sum(metas[t.trial]["debug_arc"] for t in first.values())
    switched = 0
    if half == "kimi":
        target = int(ARC_SHARE_MAX * len(first))
        switch = sorted(t for t in best_arc if not metas[first[t].trial]["debug_arc"])
        random.Random(f"{seed}:terminal_v7:arc").shuffle(switch)
        for task in switch:
            if n_arc >= target:
                break
            first[task] = best_arc[task]
            n_arc += 1
            switched += 1
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for task, t in first.items():
        second = next((x for x in plain[task] if x.trial != t.trial and metas[x.trial]["steps"] >= V6.SECOND_MIN_STEPS),
                      None) if half == "kimi" else None
        for rank, x in enumerate([t] + ([second] if second else [])):
            m = metas[x.trial]
            chosen[(task, rank)] = {"trial": x.trial, "teacher": x.teacher, "band": bands[task], "episode_id": m["episode_id"],
                                    "steps": m["steps"], "debug_arc": m["debug_arc"], "rows": sel[x.trial][0]}
    skips = Counter(why for rows, why in sel.values() if rows is None)
    return chosen, {"tasks": len(first), "arc_first_trials": n_arc, "arc_share_first": round(n_arc / len(first), 4) if first else 0,
                    "switched_to_arc": switched, "tasks_with_an_arc_trial": len(best_arc), "trial_skips": dict(skips)}


def tstats(xs: list[int]) -> dict[str, Any]:
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1) if xs else None, "p50": B._pct(xs, 0.5),
            "p90": B._pct(xs, 0.9), "p99": B._pct(xs, 0.99), "max": max(xs, default=None)}


def half_report(eps: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for e in eps for r in e["rows"]]
    loss = sum(r["n_assistant_tokens"] for r in rows)
    th = [r["thinking_tokens"] for r in rows]
    return {"episodes": len(eps), "tasks": len({e["task"] for e in eps}), "rows": len(rows), "loss_tokens": loss,
            "second_trials": sum(e["rank"] == 1 for e in eps),
            "thinking_per_trained_turn": tstats(th),
            "routine_thinking": tstats([r["thinking_tokens"] for r in rows if r["kind"] == "routine"]),
            "share_over_2k": {"turns": round(sum(x > 2_000 for x in th) / len(th), 4) if th else 0,
                              "loss": round(sum(r["n_assistant_tokens"] for r in rows if r["thinking_tokens"] > 2_000) / loss, 4)
                              if loss else 0},
            "context_tokens_of_trained_rows": tstats([r["n_tokens"] for r in rows]),
            "late_turns": {"rows_at_index_15_plus": sum(r["k"] >= LATE_INDEX for r in rows),
                           "episodes_with_one": sum(any(r["k"] >= LATE_INDEX for r in e["rows"]) for e in eps)},
            "debugging_arc_share": round(sum(e["debug_arc"] for e in eps) / len(eps), 4) if eps else 0,
            "ctrl_c_turns_trained": sum(r["ctrl_c"] for r in rows),
            "after_failure_turns": sum(r["after_error"] for r in rows),
            "done_turn_share": round(sum(r["selected_by"] == "hand_in" for r in rows) / len(rows), 4) if rows else 0,
            "steps": tstats([e["steps"] for e in eps]),
            "by_band": {b: {"episodes": sum(e["band"] == b for e in eps),
                            "loss_tokens": sum(e["loss_tokens"] for e in eps if e["band"] == b)} for b in select_v5.BANDS},
            "by_teacher": {k: {"tasks": len({e["task"] for e in eps if e["teacher"] == k}),
                               "episodes": sum(e["teacher"] == k for e in eps),
                               "loss_tokens": sum(e["loss_tokens"] for e in eps if e["teacher"] == k)}
                           for k in sorted({e["teacher"] for e in eps})}}


# ---------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_terminal_slice_v7")
    work = (a.work_dir or paths.WORK / run_id).resolve()
    out_dir = (a.out_dir or (work / "preview" if a.limit_tasks else paths.PROCESSED / "v7" / "terminal")).resolve()
    teachers = (*KIMI_TEACHERS, DEEPSEEK)
    with RunDir(run_id, root=a.log_root, log_name="train.log") as run:
        files = {t: a.messages_dir / f"{t}.parquet" for t in teachers}
        check = B.verify_inputs({**{f"messages_{t}": f for t, f in files.items()}, "trials": a.trials, "labels": a.labels},
                                a.raw_sources)
        bad = [k for k, v in check.items() if not v["matches"]]
        if bad:
            raise SystemExit(f"inputs not matching raw_sources.json sha256: {bad}")
        if harness.template_drift():
            raise SystemExit(harness.template_drift())
        dev = set(json.loads(a.dev_exclusions.read_text()).get("terminal_task_ids") or [])
        git = git_info()
        run.write_metadata(run_id=run_id, step="data_terminal_slice_v7", command=command_line(), git=git,
                           code_commit=git["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={k: v["revision"] for k, v in check.items()}, model_revision=str(a.tokenizer),
                           hardware=hardware_info(), seed=a.seed, start_time=utc_now(), status="running",
                           out_dir=str(out_dir), decision="half DeepSeek-V4-Pro / half Kimi body")
        work.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        bands, mean_rate = B.six_model_bands(a.labels)
        limit = None
        if a.limit_tasks:
            import pyarrow.parquet as pq
            all_tasks = sorted(set(pq.read_table(files["kimi"], columns=["task"]).column("task").to_pylist()))
            limit = set(random.Random(f"{a.seed}:terminal_v7:limit").sample(all_tasks, min(a.limit_tasks, len(all_tasks))))

        # 1. load, screens, dedup, dev exclusion
        ns = argparse.Namespace(teachers=teachers, trials=a.trials, tokenizer=a.tokenizer, workers=a.workers)
        trials = B.load_all(ns, run, check["messages_kimi"]["revision"], limit, files)
        d1 = {t: d1_canary.scan_parquet(files[t], id_keys=("trial",)) for t in teachers}
        B.apply_exclusions(trials, {h.record for r in d1.values() for h in r.hits}, dev)
        rep_of, dedup_report = B.apply_dedup(trials, dev)

        # 2. admission: kimi half = v6's correctness gates; deepseek half = graded pass (the render gates follow)
        for t in trials:
            if t.stage != "candidate":
                continue
            if bands.get(t.task) not in E.TERMINAL_LIMITS:
                t.drop("band", f"band_{bands.get(t.task)}")
                continue
            if t.teacher != DEEPSEEK:
                t.gate_failures = V6.admission_failures(t.profile)
                if t.gate_failures:
                    t.drop("correctness_gates", ",".join(t.gate_failures))
        admitted = [t for t in trials if t.stage == "candidate"]
        run.log(f"admitted {dict(Counter(t.teacher for t in admitted))}")

        def pair(t: B.Trial) -> tuple[dict[str, Any], dict[str, Any]]:
            half = "deepseek" if t.teacher == DEEPSEEK else "kimi"
            cap = DEEPSEEK_ROW_MAX if half == "deepseek" else E.TERMINAL_LIMITS[bands[t.task]].row_max
            return ({"task": t.task, "band": bands[t.task], "cap": cap, "half": half, "detector": a.failure_detector,
                     "dup_group": f"tgr:{rep_of.get(t.task, t.task)}"},
                    {"trial": t.trial, "teacher": t.teacher, "episode": t.loaded.episode, "verdict": t.loaded.verdict,
                     "task": t.loaded.task, "thinking": t.profile.thinking})

        # 3. render every admitted trial (every turn), then select and draw per half
        metas_list, shards = render_all([pair(t) for t in admitted], a, work, run)
        metas = {m["trial"]: m for m in metas_list}
        by_half: dict[str, dict[str, list[B.Trial]]] = {h: defaultdict(list) for h in HALVES}
        for t in admitted:
            if metas[t.trial]["outcome"] != "rendered":
                t.drop("render", metas[t.trial]["outcome"], metas[t.trial].get("detail", ""))
            else:
                by_half[metas[t.trial]["half"]][t.task].append(t)
        eps: list[dict[str, Any]] = []
        choice_report, draws = {}, {}
        for half in HALVES:
            chosen, choice_report[half] = choose(metas, by_half[half], bands, half, a.seed)
            choices = [select_v5.TaskChoice(task, c["band"], c["trial"],
                                            statistics.fmean(r["thinking_tokens"] for r in c["rows"]),
                                            sum(r["n_assistant_tokens"] for r in c["rows"]), rank)
                       for (task, rank), c in chosen.items()]
            dr = select_v5.draw(choices, seed=a.seed, budget=a.budget // 2, second_bands=select_v5.BANDS, hard_cap=True)
            draws[half] = dr.table
            for d in dr.selected:
                c = chosen[(d.choice.task, d.choice.rank)]
                eps.append({"half": half, "task": d.choice.task, "trial": c["trial"], "teacher": c["teacher"],
                            "rank": d.choice.rank, "band": c["band"], "episode_id": c["episode_id"], "steps": c["steps"],
                            "debug_arc": c["debug_arc"], "draw_order": len(eps), "loss_tokens": d.choice.loss_tokens,
                            "rows": c["rows"]})
            run.log(f"{half}: {sum(e['half'] == half for e in eps)} episodes, {dr.loss_tokens} loss tokens, "
                    f"{choice_report[half]}")
        drawn = {e["trial"] for e in eps}
        for t in admitted:
            if t.stage == "candidate":
                t.stage, t.reason = ("selected", "") if t.trial in drawn else ("choice_or_draw", "not_drawn")

        # 4. rows and records
        want = {e["episode_id"]: {r["turn_index"] for r in e["rows"]} for e in eps}
        order = {e["episode_id"]: e["draw_order"] for e in eps}
        rows = [r for s in shards for r in rowsio.read_table(s, episode_ids=set(want)) if r["turn_index"] in want[r["episode_id"]]]
        rows.sort(key=lambda r: (order[r["episode_id"]], r["turn_index"]))
        files_out: dict[str, Any] = {"rows.parquet": {"sha256": rowsio.write_table(rows, out_dir / "rows.parquet"), "rows": len(rows)}}
        by_trial = {t.trial: t for t in trials}
        for e in eps:
            t = by_trial[e["trial"]]
            e.update(task_id=t.loaded.task["task_id"], six_model_mean=round(mean_rate[e["task"]], 4),
                     licence=loaders.TGR_TEACHERS[t.teacher].licence, dup_group=f"tgr:{rep_of.get(e['task'], e['task'])}")
        files_out["episodes.jsonl"] = {"sha256": B.write_jsonl(out_dir / "episodes.jsonl", eps), "rows": len(eps)}
        seen: set[str] = set()
        tasks_out = []
        for e in eps:
            if e["task"] not in seen:
                seen.add(e["task"])
                tasks_out.append({"task_id": e["task_id"], "kind": "terminal", "upstream_id": e["task"],
                                  "text": by_trial[e["trial"]].loaded.info["instruction"]})
        files_out["tasks.jsonl"] = {"sha256": B.write_jsonl(out_dir / "tasks.jsonl", tasks_out), "rows": len(tasks_out)}

        # 5. checks
        msgs = {e["episode_id"]: by_trial[e["trial"]].loaded.episode["messages"] for e in eps}
        row_checks = B.check_rows(rows, msgs, a)
        rc = B.render_check(rows, msgs, a, run.run_dir / "render_check.jsonl")
        final_d1 = d1_canary.scan_records(({"episode_id": k, "messages": m} for k, m in msgs.items()), source="drawn").counts()
        meta_rows = [(e, r) for e in eps for r in e["rows"]]
        reread = {(r["episode_id"], r["turn_index"]): E.trained_thinking(r["input_ids"], r["assistant_masks"]) for r in rows}
        think_of = {(e["episode_id"], r["turn_index"]): r["thinking_tokens"] for e, r in meta_rows}
        cap_of = {e["episode_id"]: DEEPSEEK_ROW_MAX if e["half"] == "deepseek" else E.TERMINAL_LIMITS[e["band"]].row_max for e in eps}
        loss_total = sum(r["n_assistant_tokens"] for r in rows)
        checks_out = {
            "row_checks": row_checks, "render_check_20": rc, "d1_drawn": final_d1,
            "guard_rerun_failures": sum(not trainable(r, e["half"])[0] for e, r in meta_rows),
            "thinking_reread_mismatch": sum(1 for k, v in reread.items() if v != [think_of[k]]),
            "turns_over_4000": sum(1 for v in reread.values() if v and v[0] and v[0] > THINK_MAX),
            "kimi_routine_over_1000": sum(e["half"] == "kimi" and r["kind"] == "routine" and r["thinking_tokens"] > ROUTINE_MAX
                                          for e, r in meta_rows),
            "empty_keys_at_continuation_trained": sum(r["empty_keys"] and r["at_continuation"] for _, r in meta_rows),
            "repetitive_trained": sum(r["repetitive"] for _, r in meta_rows),
            "confirmation_rows": sum(r["confirmation"] for _, r in meta_rows),
            "hand_in_rows_per_episode": dict(Counter(sum(r["selected_by"] == "hand_in" for r in e["rows"]) for e in eps)),
            "rows_over_cap": sum(r["n_tokens"] > cap_of[r["episode_id"]] for r in rows),
            "rows_over_max_length": sum(r["n_tokens"] > DEEPSEEK_ROW_MAX for r in rows),
            "rows_over_loss_cap": sum(r["n_assistant_tokens"] > V6.ROW_LOSS_CAP for r in rows),
            "dev_tasks": sorted({e["task"] for e in eps if e["task"] in dev}),
            "trials_per_task_per_half_max": max(Counter((e["half"], e["task"]) for e in eps).values(), default=0),
            "budget": {"loss_tokens": loss_total, "max": a.budget},
            "rows_match_records": len(rows) == len(meta_rows),
        }
        ok = (row_checks["rows_checked"] == row_checks["rows_clean"] == len(rows) and not rc["rows_with_problems"]
              and final_d1["n_hits"] == 0 and not checks_out["guard_rerun_failures"]
              and not checks_out["thinking_reread_mismatch"] and not checks_out["turns_over_4000"]
              and not checks_out["kimi_routine_over_1000"] and not checks_out["empty_keys_at_continuation_trained"]
              and not checks_out["repetitive_trained"] and not checks_out["confirmation_rows"]
              and set(checks_out["hand_in_rows_per_episode"]) <= {1} and not checks_out["rows_over_cap"]
              and not checks_out["rows_over_loss_cap"] and not checks_out["dev_tasks"]
              and checks_out["trials_per_task_per_half_max"] <= 2 and loss_total <= a.budget
              and checks_out["rows_match_records"])
        checks_out["all_pass"] = ok

        # 6. report, quality sample, outputs
        untrainable = {h: dict(Counter(trainable(r, h)[1] for m in metas.values() if m["half"] == h and m["outcome"] == "rendered"
                                       for r in m["rows"] if not trainable(r, h)[0])) for h in HALVES}
        report = {"rows": len(rows), "episodes": len(eps), "tasks": len({e["task"] for e in eps}), "loss_tokens": loss_total,
                  "sequence_tokens": sum(r["n_tokens"] for r in rows),
                  "tasks_in_both_halves": len({e["task"] for e in eps if e["half"] == "kimi"} & {e["task"] for e in eps if e["half"] == "deepseek"}),
                  "halves": {h: half_report([e for e in eps if e["half"] == h]) for h in HALVES},
                  "choice": choice_report, "untrainable_turns_by_reason": untrainable,
                  "empty_keys_at_continuation_dropped": {h: untrainable[h].get("empty_keys_at_continuation", 0) for h in HALVES},
                  "ctrl_c_turns_trainable": {h: sum(r["ctrl_c"] and trainable(r, h)[0] for m in metas.values()
                                                    if m["half"] == h and m["outcome"] == "rendered" for r in m["rows"]) for h in HALVES}}
        rng = random.Random(f"{a.seed}:terminal_v7:quality")
        ds = [(e, r) for e, r in meta_rows if e["half"] == "deepseek"]
        late = [x for x in ds if x[1]["k"] >= LATE_INDEX]
        early = [x for x in ds if x[1]["k"] < LATE_INDEX]
        cc = [x for x in meta_rows if x[1]["ctrl_c"]]
        picks = ([("deepseek_late", x) for x in rng.sample(late, min(4, len(late)))] +
                 [("deepseek", x) for x in rng.sample(early, min(8 - min(4, len(late)), len(early)))] +
                 [("ctrl_c", x) for x in rng.sample(cc, min(4, len(cc)))])
        ctx = {(r["episode_id"], r["turn_index"]): r["n_tokens"] for r in rows}

        def excerpt(kind: str, e: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
            m = msgs[e["episode_id"]][r["turn_index"]]
            return {"kind": kind, "row_key": f"{e['episode_id']}:{r['turn_index']}", "half": e["half"], "teacher": e["teacher"],
                    "band": e["band"], "turn_ordinal": r["k"], "context_tokens": ctx[(e["episode_id"], r["turn_index"])],
                    "thinking_tokens": r["thinking_tokens"], "after_failure": r["after_error"],
                    "previous_screen_tail": (msgs[e["episode_id"]][r["turn_index"] - 1].get("content") or "")[-800:],
                    "reasoning_head": (m.get("reasoning_content") or "")[:1200],
                    "reasoning_tail": (m.get("reasoning_content") or "")[-600:], "action": m.get("content")}

        B.write_jsonl(run.run_dir / "quality_read.jsonl", [excerpt(k, e, r) for k, (e, r) in picks])
        supply = {"run_id": run_id, "budget": a.budget, "draws": draws, "choice": choice_report,
                  "admission": {k: dict(Counter(t.stage for t in trials if t.teacher == k)) for k in teachers}}
        files_out["supply.json"] = {"sha256": B.write_json(out_dir / "supply.json", supply)}
        manifest = {"slice": "terminal_v7", "decision": "half DeepSeek-V4-Pro / half Kimi body; "
                                                         "replaces the 2% / 10% long-turn cap",
                    "run_id": run_id, "created": utc_now(), "code_commit": git["commit"], "seed": a.seed,
                    "final": not a.limit_tasks and a.budget == BUDGET_LOSS_TOKENS,
                    "rules": {"think_max": THINK_MAX, "kimi_routine_max": ROUTINE_MAX, "deepseek_row_max": DEEPSEEK_ROW_MAX,
                              "kimi_arc_share_max": ARC_SHARE_MAX, "trained_turns": TRAINED_TURNS,
                              "repetition": f"no {WORD_RUN}-word run twice; distinct / all lines >= {LONG_LINE} chars >= {DISTINCT_MIN}",
                              "failure_detector": a.failure_detector, "kimi_teachers": list(KIMI_TEACHERS),
                              "deepseek_admission": "graded pass + render gates (no v6 correctness gates)"},
                    "inputs": check, "dedup": dedup_report, "d1_whole_file": {t: r.counts() for t, r in d1.items()},
                    "screens": admission.screen_counts([t.screens for t in trials if t.screens is not None]),
                    "files": files_out, "report": report, "checks": checks_out,
                    "quality_read": str(run.run_dir / "quality_read.jsonl")}
        files_out["manifest.json"] = {"sha256": B.write_json(out_dir / "manifest.json", manifest)}
        if not a.limit_tasks:
            a.manifest_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_dir / "manifest.json", a.manifest_copy)
        run.metric({"step": "done", "rows": len(rows), "loss_tokens": loss_total, "checks_pass": ok})
        run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "CPU only"),
                             ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main"),
                             ("gpu_hours", 0, "h", "CPU only"), ("cost_usd", 0, "USD", "CPU only")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if ok else "failed",
                           outputs={k: {"path": str(out_dir / k), **v} for k, v in files_out.items()})
        if not ok:
            run.error(f"checks failed: {json.dumps({k: v for k, v in checks_out.items() if k not in ('render_check_20', 'row_checks')}, default=str)[:2000]}")
        run.finalize()
        run.log(f"done in {run.elapsed():.0f} s; {out_dir}; checks {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
