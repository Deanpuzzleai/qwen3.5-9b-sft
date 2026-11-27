#!/usr/bin/env python
"""Build the v6 terminal slice (data/manifests/v6/DATA_CARD_v6.md, "Terminal slice"):
the existing per-turn builder and turn-role order, correctness gates only, at most 6 trained turns per episode, each
within its band's per-turn thinking ceiling. The v5 slice is scripts/data/build_terminal_slice_v5.py; this script
reuses its loading, screens, dedup, render and check functions unchanged.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_terminal_slice_v6.py [--workers 48] [--limit-tasks 60]

Admission (per trial): the loader's graded pass (reward 1 in both files, scored, counted toward k, every turn parsed),
the screens, the dev exclusion and the task-text dedup as in v5; then the correctness gates of the efficiency profiler,
unchanged: completed, verified completion, zero wasted actions, and at most MAX_STEPS assistant turns. No episode-level
thinking or output gate. Teachers by priority (select_v5.TEACHERS); within a teacher, the lowest mean thinking per
non-confirmation turn first.
Rows (per trial): every turn is rendered with render/filters at the band's row cap (v5's _render_one path without the
episode limits). A turn is eligible when it is not a confirmation reply, its thinking read from its row
(efficiency.trained_thinking) is within the band's per-turn ceiling (efficiency.TERMINAL_LIMITS turn_ceiling x the
ceiling scale) and its row has <= 32K loss tokens. The hand-in turn (the first task_complete turn) must be eligible,
else the trial is skipped (counted). Up to TRAINED_TURNS eligible turns are trained, in the existing
render/turn_roles.priority_turns order (verify_before_complete, the hand-in turn, then a round robin over after_error,
long_job and recovery), filled by a seeded random draw over the other eligible turns. Turns not trained stay in later
rows' history exactly as the harness rendered them.
Per task: the first trial (teacher priority, then thinking) that yields rows, and a second one with >= SECOND_MIN_STEPS
steps. Draw: select_v5.draw with --budget (3.5M, a ceiling): new tasks first (hard once, medium 55 / easy 45 of the
rest), then second trials into the budget left, <= 25% of the episodes (medium, easy, medium, then hard).
Acceptance (no overthinking): mean thinking per trained turn <= ACCEPT per band and overall. If it fails, the ceilings
are lowered uniformly by 20% (CEILING_FALLBACK) and the selection and draw run again (the render is reused).

Inputs: as build_terminal_slice_v5.py (RAW/tgr/data/messages/<teacher>.parquet, trials.parquet, RAW/tgr_env/labels.jsonl,
  sha256 against data/manifests/raw_sources.json; data/manifests/dev_exclusions.json; the Qwen3.5-9B tokenizer).
Outputs:
  --out-dir (default data/processed/v6/terminal, git-ignored; with --limit-tasks: WORK/<run_id>/preview)
    rows.parquet      the trained rows in draw order (ROW_COLUMNS, difficulty = band, + provenance and bookkeeping)
    episodes.jsonl    one line per drawn trial: task, trial, teacher, rank, band, steps, trained turns with their role,
                      thinking and loss tokens, the skipped over-ceiling turns
    tasks.jsonl       one line per distinct drawn task {task_id, kind, text}: the run_screens.py --pool input
    supply.json       admission funnel, supply and draw per band and teacher, acceptance attempts
    manifest.json     inputs, rules, report (rows, episodes, tasks, loss tokens, steps distribution, done-turn and
                      after-error shares, thinking by band), checks
  WORK/<run_id>/render/   every admitted trial's rows (reused by --resume)
  logs/<run_id>/          metadata.json, train.log, metrics.jsonl, errors.log, resources.csv, render_check.jsonl,
                          dropped.jsonl
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
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import build_terminal_slice_v5 as B  # noqa: E402
from apex_data import efficiency as E, paths  # noqa: E402
from apex_data.render import turn_roles  # noqa: E402
from apex_data.render.adapters import confirmation_replies  # noqa: E402
from apex_data.render.filters import FilterConfig, filter_episodes, terminus_parse  # noqa: E402
from apex_data.schema import validate_provenance  # noqa: E402
from apex_data.screens import d1_canary  # noqa: E402
from apex_data.terminal import admission, harness, loaders, rows as rowsio, select_v5  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402
from apex_train.data_format import LATEST_FITTING  # noqa: E402

BUDGET_LOSS_TOKENS = 3_500_000      # v6: terminal <= 3.5M loss tokens
MAX_STEPS = 20
SECOND_MIN_STEPS = 6
TRAINED_TURNS = 6
CEILINGS = {b: lim.turn_ceiling for b, lim in E.TERMINAL_LIMITS.items()}       # easy 1,200 / medium 1,800 / hard 2,500
CEILING_FALLBACK = 0.8
ACCEPT = {"easy": 450, "medium": 750, "hard": 1000, "all": 400}
ROW_LOSS_CAP = B.ROW_LOSS_CAP
CHUNK = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--messages-dir", type=Path, default=paths.RAW / "tgr" / "data" / "messages")
    p.add_argument("--teachers", default=",".join(select_v5.TEACHERS), help="model keys in priority order")
    p.add_argument("--trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet")
    p.add_argument("--labels", type=Path, default=paths.RAW / "tgr_env" / "labels.jsonl")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES)
    p.add_argument("--dev-exclusions", type=Path, default=paths.DEV_EXCLUSIONS)
    p.add_argument("--budget", type=int, default=BUDGET_LOSS_TOKENS, help="loss-token budget (default %(default)s)")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit-tasks", type=int, default=None, metavar="N", help="seeded sample of N tasks (not final)")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true", help="continue --run-id, reusing its render shards")
    p.add_argument("--log-root", type=Path, default=None)
    a = p.parse_args(argv)
    a.teachers = tuple(t for t in a.teachers.split(",") if t)
    if not a.teachers or any(t not in loaders.TGR_TEACHERS for t in a.teachers):
        p.error(f"teachers must be keys of {sorted(loaders.TGR_TEACHERS)}")
    return a


# ---------------------------------------------------------------------------------------------------- admission
def admission_failures(p: E.Profile) -> list[str]:
    """The v6 correctness gates (the profiler's detectors, unchanged): completed, verified, zero wasted, steps."""
    out = []
    if not p.completed:
        out.append("not_completed")
    elif not p.verified:
        out.append("unverified_completion")
    if p.wasted_actions:
        out.append("wasted_actions")
    if p.steps > MAX_STEPS:
        out.append("steps")
    return out


def order_mean(t: B.Trial) -> float:
    """Mean thinking over the non-confirmation turns (the within-teacher order)."""
    msgs = t.loaded.episode["messages"]
    conf = set(confirmation_replies(msgs))
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    xs = [t.profile.thinking[k] for k, i in enumerate(asst) if i not in conf]
    return statistics.fmean(xs) if xs else 0.0


def hand_in_index(messages: list[dict[str, Any]]) -> int | None:
    """Message index of the first assistant turn that sets task_complete (the hand-in turn)."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            p = terminus_parse(m.get("content") or "")
            if not p.error and p.task_complete:
                return i
    return None


# ------------------------------------------------------------------------------------------- render (worker pool)
def _render_trial(entry: dict[str, Any], t: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every renderable turn of one trial as a row (v5's path without the episode limits) and the trial's meta."""
    ep = t["episode"]
    msgs = ep["messages"]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    res = filter_episodes([ep], {ep["episode_id"]: t["verdict"]}, {t["task"]["task_id"]: t["task"]}, B._G["tok"],
                          FilterConfig(max_row_tokens=entry["cap"], terminal_turns_per_episode=len(asst)))
    meta: dict[str, Any] = {"trial": t["trial"], "teacher": t["teacher"], "episode_id": ep["episode_id"],
                            "steps": len(asst), "rows": []}
    if res.dropped:
        d = res.dropped[0]
        return [], {**meta, "outcome": f"{d.gate}:{d.reason}", "detail": d.detail[:200]}
    conf = set(confirmation_replies(msgs))
    roles = turn_roles.classify(msgs)
    hand_in = hand_in_index(msgs)
    order = [(hand_in if i == LATEST_FITTING else i, role) for i, role in turn_roles.priority_turns(msgs)]
    records = []
    for r in res.kept:
        validate_provenance(r.provenance)
        rec = rowsio.flat_row(r.row, r.provenance)
        rec["difficulty"] = entry["band"]
        rec["dup_group"] = entry["dup_group"]
        spans = E.trained_thinking(r.row["input_ids"], r.row["assistant_masks"])
        i = r.row["turn_index"]
        meta["rows"].append({"turn_index": i, "n_tokens": rec["n_tokens"], "n_assistant_tokens": rec["n_assistant_tokens"],
                             "thinking_tokens": spans[0] if len(spans) == 1 else None, "confirmation": i in conf,
                             "role": roles.get(i, "hand_in" if i == hand_in else None)})
        records.append(rec)
    return records, {**meta, "outcome": "rendered", "hand_in": hand_in, "order": order}


def _render_worker(job: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> list[dict[str, Any]]:
    shard, pairs = job
    records, metas = [], []
    for entry, t in pairs:
        rows, meta = _render_trial(entry, t)
        records.extend(rows)
        metas.append({**meta, "task": entry["task"], "band": entry["band"]})
    rowsio.write_table(records, Path(shard + ".parquet"))
    tmp = Path(shard + f".json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(metas))
    tmp.replace(Path(shard + ".json"))
    return metas


def render_all(pairs: list[tuple[dict[str, Any], dict[str, Any]]], a: argparse.Namespace, work: Path,
               run: RunDir) -> tuple[list[dict[str, Any]], list[Path]]:
    out_dir = work / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(pairs, key=lambda x: x[1]["trial"])
    jobs, done, shards = [], [], []
    for k in range(0, len(pairs), CHUNK):
        part = pairs[k:k + CHUNK]
        key = hashlib.sha256(json.dumps([[e["cap"], t["trial"]] for e, t in part]).encode()).hexdigest()[:12]
        shard = str(out_dir / f"chunk_{k // CHUNK:05d}_{key}")
        shards.append(Path(shard + ".parquet"))
        if Path(shard + ".json").exists() and Path(shard + ".parquet").exists():
            done.extend(json.loads(Path(shard + ".json").read_text()))
        else:
            jobs.append((shard, part))
    run.log(f"render: {len(jobs)} shards to build, {len(shards) - len(jobs)} reused ({len(pairs)} trials)")
    jobs.sort(key=lambda j: -sum(len(json.dumps(t["episode"]["messages"])) for _, t in j[1]))   # long jobs first
    with mp.get_context("spawn").Pool(a.workers, initializer=B._render_init, initargs=(str(a.tokenizer),)) as pool:
        for n, metas in enumerate(pool.imap_unordered(_render_worker, jobs), 1):
            done.extend(metas)
            if n % 100 == 0 or n == len(jobs):
                run.log(f"render: {n}/{len(jobs)} shards")
    return done, shards


# ----------------------------------------------------------------------------------------------------- selection
def select_turns(meta: dict[str, Any], band: str, scale: float, seed: int) -> tuple[list[dict[str, Any]] | None, str]:
    """The trained rows of one rendered trial (role order, eligible turns only, <= TRAINED_TURNS), or None and why."""
    ceiling = CEILINGS[band] * scale
    eligible = {r["turn_index"]: r for r in meta["rows"]
                if not r["confirmation"] and r["thinking_tokens"] is not None and r["thinking_tokens"] <= ceiling
                and r["n_assistant_tokens"] <= ROW_LOSS_CAP}
    if meta["hand_in"] is None or meta["hand_in"] not in eligible:
        over = any(r["turn_index"] == meta["hand_in"] and (r["thinking_tokens"] or 0) > ceiling for r in meta["rows"])
        return None, "hand_in_over_ceiling" if over else "hand_in_not_trainable"
    picked: dict[int, str] = {}
    for i, role in meta["order"]:
        if len(picked) == TRAINED_TURNS:
            break
        if i in eligible and i not in picked:
            picked[i] = "hand_in" if i == meta["hand_in"] else role
    rest = sorted(set(eligible) - set(picked))
    random.Random(f"{seed}:{meta['episode_id']}").shuffle(rest)
    for i in rest[:TRAINED_TURNS - len(picked)]:
        picked[i] = "random"
    return [{**eligible[i], "selected_by": picked[i]} for i in sorted(picked)], "selected"


def choose(metas: dict[str, dict[str, Any]], trials_by_task: dict[str, list[B.Trial]], bands: dict[str, str],
           scale: float, seed: int, teachers: tuple[str, ...]) -> tuple[dict[tuple[str, int], dict[str, Any]], Counter]:
    """Per task, the first trial (teacher priority, then thinking) with trained rows, and a second with >= 6 steps."""
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    skips: Counter = Counter()
    for task, ts in trials_by_task.items():
        band = bands[task]
        ranked = sorted(ts, key=lambda t: (teachers.index(t.teacher), order_mean(t), t.trial))
        for t in ranked:
            m = metas.get(t.trial)
            if m is None or m["outcome"] != "rendered":
                continue
            rank = sum(1 for r in (0, 1) if (task, r) in chosen)
            if rank == 2:
                break
            if rank == 1 and m["steps"] < SECOND_MIN_STEPS:
                skips["second_trial_under_6_steps"] += 1
                continue
            rows, why = select_turns(m, band, scale, seed)
            if rows is None:
                skips[why] += 1
                continue
            chosen[(task, rank)] = {"trial": t.trial, "teacher": t.teacher, "band": band, "episode_id": m["episode_id"],
                                    "steps": m["steps"], "rows": rows,
                                    "skipped_turns": [r["turn_index"] for r in m["rows"] if not r["confirmation"]
                                                      and r["turn_index"] not in {x["turn_index"] for x in rows}]}
    return chosen, skips


def thinking_stats(xs: list[int]) -> dict[str, Any]:
    return {"turns": len(xs), "mean": round(statistics.fmean(xs), 1) if xs else None, "p50": B._pct(xs, 0.5),
            "p90": B._pct(xs, 0.9), "p99": B._pct(xs, 0.99), "max": max(xs, default=None)}


def acceptance(eps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in (*select_v5.BANDS, "all"):
        xs = [r["thinking_tokens"] for e in eps if b in ("all", e["band"]) for r in e["rows"]]
        mean = round(statistics.fmean(xs), 1) if xs else 0.0
        out[b] = {"mean": mean, "max": ACCEPT[b], "pass": mean <= ACCEPT[b]}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def draw_and_records(chosen: dict[tuple[str, int], dict[str, Any]], a: argparse.Namespace) -> tuple[select_v5.Draw, list[dict[str, Any]]]:
    choices = [select_v5.TaskChoice(task, c["band"], c["trial"],
                                    statistics.fmean(r["thinking_tokens"] for r in c["rows"]),
                                    sum(r["n_assistant_tokens"] for r in c["rows"]), rank) for (task, rank), c in chosen.items()]
    dr = select_v5.draw(choices, seed=a.seed, budget=a.budget, second_bands=select_v5.BANDS, hard_cap=True)
    eps = []
    for d in dr.selected:
        c = chosen[(d.choice.task, d.choice.rank)]
        eps.append({"task": d.choice.task, "trial": c["trial"], "teacher": c["teacher"], "rank": d.choice.rank,
                    "band": c["band"], "episode_id": c["episode_id"], "steps": c["steps"], "draw_order": d.order,
                    "half": d.half, "loss_tokens": d.choice.loss_tokens, "rows": c["rows"],
                    "skipped_turns": c["skipped_turns"]})
    return dr, eps


def report(eps: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for e in eps for r in e["rows"]]
    n = len(rows)
    return {
        "rows": n, "episodes": len(eps), "tasks": len({e["task"] for e in eps}),
        "second_trials": sum(e["rank"] == 1 for e in eps), "loss_tokens": sum(r["n_assistant_tokens"] for r in rows),
        "sequence_tokens": sum(r["n_tokens"] for r in rows),
        "steps_distribution": dict(sorted(Counter(e["steps"] for e in eps).items())),
        "steps_p50": B._pct([e["steps"] for e in eps], 0.5),
        "trained_turns_per_episode": dict(sorted(Counter(len(e["rows"]) for e in eps).items())),
        "done_turn_share": round(sum(r["selected_by"] == "hand_in" for r in rows) / n, 4) if n else 0,
        "after_error_turn_share": round(sum(r["role"] == "after_error" for r in rows) / n, 4) if n else 0,
        "rows_by_selection": dict(Counter(r["selected_by"] for r in rows).most_common()),
        "rows_by_role": dict(Counter(r["role"] or "none" for r in rows).most_common()),
        "over_ceiling_turns_skipped": sum(len(e["skipped_turns"]) for e in eps),
        "thinking_per_trained_turn": {"all": thinking_stats([r["thinking_tokens"] for r in rows]),
                                      **{b: thinking_stats([r["thinking_tokens"] for e in eps if e["band"] == b
                                                            for r in e["rows"]]) for b in select_v5.BANDS}},
        "by_band": {b: {"tasks": len({e["task"] for e in eps if e["band"] == b}),
                        "episodes": sum(e["band"] == b for e in eps),
                        "rows": sum(len(e["rows"]) for e in eps if e["band"] == b),
                        "loss_tokens": sum(e["loss_tokens"] for e in eps if e["band"] == b)} for b in select_v5.BANDS},
        "by_teacher": {k: {"tasks": len({e["task"] for e in eps if e["teacher"] == k}),
                           "episodes": sum(e["teacher"] == k for e in eps),
                           "loss_tokens": sum(e["loss_tokens"] for e in eps if e["teacher"] == k),
                           "episodes_by_band": {b: sum(e["teacher"] == k and e["band"] == b for e in eps)
                                                for b in select_v5.BANDS}}
                       for k in sorted({e["teacher"] for e in eps})},
    }


# ---------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_terminal_slice_v6")
    work = (a.work_dir or paths.WORK / run_id).resolve()
    out_dir = (a.out_dir or (work / "preview" if a.limit_tasks else paths.PROCESSED / "v6" / "terminal")).resolve()
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        files = {t: a.messages_dir / f"{t}.parquet" for t in a.teachers}
        inputs = {**{f"messages_{t}": f for t, f in files.items()}, "trials": a.trials, "labels": a.labels}
        check = B.verify_inputs(inputs, a.raw_sources)
        bad = [k for k, v in check.items() if not v["matches"]]
        if bad:
            raise SystemExit(f"inputs not matching raw_sources.json sha256: {bad}")
        drift = harness.template_drift()
        if drift:
            raise SystemExit(drift)
        dev_tasks = set(json.loads(a.dev_exclusions.read_text()).get("terminal_task_ids") or [])
        if not dev_tasks:
            raise SystemExit(f"{a.dev_exclusions} lists no terminal_task_ids")
        git = git_info()
        run.write_metadata(run_id=run_id, step="data_terminal_slice_v6", command=command_line(), git=git,
                           code_commit=git["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={k: v["revision"] for k, v in check.items()},
                           inputs={k: str(v) for k, v in inputs.items()}, model_revision=str(a.tokenizer),
                           hardware=hardware_info(), seed=a.seed, start_time=utc_now(), status="running",
                           work_dir=str(work), out_dir=str(out_dir), plan="data/manifests/v6/DATA_CARD_v6.md (terminal slice)")
        work.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        bands, mean_rate = B.six_model_bands(a.labels)
        limit = None
        if a.limit_tasks:
            import pyarrow.parquet as pq
            all_tasks = sorted(set(pq.read_table(files[a.teachers[0]], columns=["task"]).column("task").to_pylist()))
            limit = set(random.Random(f"{a.seed}:terminal_v6:limit").sample(all_tasks, min(a.limit_tasks, len(all_tasks))))

        # 1. load, screens, profile; D1, exclusions, dedup (v5 functions)
        b_args = argparse.Namespace(teachers=a.teachers, trials=a.trials, tokenizer=a.tokenizer, workers=a.workers)
        trials = B.load_all(b_args, run, check[f"messages_{a.teachers[0]}"]["revision"], limit, files)
        run.log(f"loaded {len(trials)} trials; passing {dict(Counter(t.teacher for t in trials if t.loaded.drop is None))}")
        d1_files = {t: d1_canary.scan_parquet(files[t], id_keys=("trial",)) for t in a.teachers}
        B.apply_exclusions(trials, {h.record for r in d1_files.values() for h in r.hits}, dev_tasks)
        rep_of, dedup_report = B.apply_dedup(trials, dev_tasks)

        # 2. correctness gates
        for t in trials:
            if t.stage != "candidate":
                continue
            if bands.get(t.task) not in E.TERMINAL_LIMITS:
                t.drop("band", f"band_{bands.get(t.task)}")
                continue
            t.gate_failures = admission_failures(t.profile)
            if t.gate_failures:
                t.drop("correctness_gates", ",".join(t.gate_failures))
        admitted = [t for t in trials if t.stage == "candidate"]
        run.log(f"admitted {len(admitted)} trials {dict(Counter(t.teacher for t in admitted))}")

        # 3. render every admitted trial
        pairs = [({"task": t.task, "band": bands[t.task], "cap": E.TERMINAL_LIMITS[bands[t.task]].row_max,
                   "dup_group": f"tgr:{rep_of.get(t.task, t.task)}"},
                  {"trial": t.trial, "teacher": t.teacher, "episode": t.loaded.episode, "verdict": t.loaded.verdict,
                   "task": t.loaded.task}) for t in admitted]
        metas_list, shards = render_all(pairs, a, work, run)
        metas = {m["trial"]: m for m in metas_list}
        for t in admitted:
            m = metas[t.trial]
            if m["outcome"] != "rendered":
                t.drop("render", m["outcome"], m.get("detail", ""))
        trials_by_task: dict[str, list[B.Trial]] = defaultdict(list)
        for t in admitted:
            if t.stage == "candidate":
                trials_by_task[t.task].append(t)

        # 4. selection, draw, acceptance (ceilings lowered by 20% once if it fails)
        attempts = []
        for scale in (1.0, CEILING_FALLBACK):
            chosen, skips = choose(metas, trials_by_task, bands, scale, a.seed, a.teachers)
            dr, eps = draw_and_records(chosen, a)
            acc = acceptance(eps)
            attempts.append({"ceiling_scale": scale, "ceilings": {b: round(c * scale) for b, c in CEILINGS.items()},
                             "chosen_first": sum(r == 0 for _, r in chosen), "chosen_second": sum(r == 1 for _, r in chosen),
                             "trial_skips": dict(skips), "drawn_episodes": len(eps), "loss_tokens": dr.loss_tokens,
                             "acceptance": acc})
            run.log(f"scale {scale}: {len(eps)} episodes, {dr.loss_tokens} loss tokens, skips {dict(skips)}, "
                    f"acceptance {'pass' if acc['pass'] else 'FAIL'} " +
                    " ".join(f"{b} {acc[b]['mean']}" for b in (*select_v5.BANDS, 'all')))
            if acc["pass"]:
                break
        drawn_trials = {e["trial"] for e in eps}
        chosen_trials = {c["trial"] for c in chosen.values()}
        for t in admitted:
            if t.stage != "candidate":
                continue
            t.stage, t.reason = (("selected", "") if t.trial in drawn_trials else ("draw", "not_drawn")
                                 if t.trial in chosen_trials else ("choice", "not_chosen"))

        # 5. rows
        want = {e["episode_id"]: {r["turn_index"] for r in e["rows"]} for e in eps}
        order = {e["episode_id"]: e["draw_order"] for e in eps}
        rows = [r for s in shards for r in rowsio.read_table(s, episode_ids=set(want)) if r["turn_index"] in want[r["episode_id"]]]
        rows.sort(key=lambda r: (order[r["episode_id"]], r["turn_index"]))
        files_out: dict[str, Any] = {}
        files_out["rows.parquet"] = {"sha256": rowsio.write_table(rows, out_dir / "rows.parquet"), "rows": len(rows)}
        by_trial = {t.trial: t for t in trials}
        for e in eps:
            t = by_trial[e["trial"]]
            e.update(task_id=t.loaded.task["task_id"], six_model_mean=round(mean_rate[e["task"]], 4),
                     generator=t.loaded.task["extra"]["provenance"]["generator"],
                     licence=loaders.TGR_TEACHERS[t.teacher].licence, dup_group=f"tgr:{rep_of.get(e['task'], e['task'])}",
                     profile=t.profile.to_dict())
        files_out["episodes.jsonl"] = {"sha256": B.write_jsonl(out_dir / "episodes.jsonl", eps), "rows": len(eps)}
        seen: set[str] = set()
        tasks_out = []
        for e in eps:
            if e["task"] not in seen:
                seen.add(e["task"])
                tasks_out.append({"task_id": e["task_id"], "kind": "terminal", "upstream_id": e["task"],
                                  "text": by_trial[e["trial"]].loaded.info["instruction"]})
        files_out["tasks.jsonl"] = {"sha256": B.write_jsonl(out_dir / "tasks.jsonl", tasks_out), "rows": len(tasks_out)}

        # 6. checks
        msgs = {e["episode_id"]: by_trial[e["trial"]].loaded.episode["messages"] for e in eps}
        row_checks = B.check_rows(rows, msgs, a)
        run.log(f"row checks: {row_checks['rows_clean']}/{row_checks['rows_checked']} clean {row_checks['problems']}")
        rc = B.render_check(rows, msgs, a, run.run_dir / "render_check.jsonl")
        final_d1 = d1_canary.scan_records(({"episode_id": k, "messages": m} for k, m in msgs.items()),
                                          source="drawn_episodes").counts()
        scale = attempts[-1]["ceiling_scale"]
        thinking_by_row = {(r["episode_id"], r["turn_index"]): E.trained_thinking(r["input_ids"], r["assistant_masks"])
                           for r in rows}
        per_task = Counter(e["task"] for e in eps)
        band_of_episode = {e["episode_id"]: e["band"] for e in eps}
        rep = report(eps)
        checks_out = {
            "row_checks": row_checks, "render_check_20": rc, "d1_drawn_episodes": final_d1,
            "confirmation_rows": sum(r["confirmation"] for e in eps for r in e["rows"]),
            "hand_in_rows_per_episode": dict(Counter(sum(r["selected_by"] == "hand_in" for r in e["rows"]) for e in eps)),
            "max_trained_turns_per_episode": max((len(e["rows"]) for e in eps), default=0),
            "rows_over_ceiling": sum(1 for (eid, _), v in thinking_by_row.items()
                                     if len(v) != 1 or v[0] is None or v[0] > CEILINGS[band_of_episode[eid]] * scale),
            "rows_over_band_cap": sum(r["n_tokens"] > E.TERMINAL_LIMITS[r["difficulty"]].row_max for r in rows),
            "rows_over_loss_cap": sum(r["n_assistant_tokens"] > ROW_LOSS_CAP for r in rows),
            "admission_rerun_failures": sum(bool(admission_failures(by_trial[e["trial"]].profile)) for e in eps),
            "dev_tasks_drawn": sorted({e["task"] for e in eps if e["task"] in dev_tasks}),
            "trials_per_task_max": max(per_task.values(), default=0),
            "second_trials_under_6_steps": sum(e["rank"] == 1 and e["steps"] < SECOND_MIN_STEPS for e in eps),
            "second_trial_share": round(rep["second_trials"] / len(eps), 4) if eps else 0,
            "rows_match_episode_records": len(rows) == rep["rows"],
            "budget": {"loss_tokens": rep["loss_tokens"], "max": a.budget},
            "acceptance": attempts[-1]["acceptance"],
        }
        ok = (row_checks["rows_checked"] == row_checks["rows_clean"] == len(rows) and not rc["rows_with_problems"]
              and final_d1["n_hits"] == 0 and checks_out["confirmation_rows"] == 0
              and set(checks_out["hand_in_rows_per_episode"]) == {1}
              and checks_out["max_trained_turns_per_episode"] <= TRAINED_TURNS and checks_out["rows_over_ceiling"] == 0
              and not checks_out["rows_over_band_cap"] and not checks_out["rows_over_loss_cap"]
              and not checks_out["admission_rerun_failures"] and not checks_out["dev_tasks_drawn"]
              and checks_out["trials_per_task_max"] <= 2 and not checks_out["second_trials_under_6_steps"]
              and checks_out["second_trial_share"] <= select_v5.SECOND_SHARE_MAX
              and checks_out["rows_match_episode_records"] and rep["loss_tokens"] <= a.budget
              and checks_out["acceptance"]["pass"])
        checks_out["all_pass"] = ok

        # 7. reports
        with (run.run_dir / "dropped.jsonl").open("w") as f:
            for t in trials:
                if t.stage != "selected":
                    f.write(json.dumps({"task": t.task, "trial": t.trial, "teacher": t.teacher, "stage": t.stage,
                                        "reason": t.reason, "detail": t.detail[:200]}) + "\n")
        funnel = {k: {"trials": sum(t.teacher == k for t in trials),
                      "passing": sum(t.teacher == k and t.loaded.drop is None for t in trials),
                      "stages": dict(Counter(t.stage for t in trials if t.teacher == k)),
                      "correctness_gate_failures": dict(Counter(f for t in trials if t.teacher == k
                                                                and t.stage == "correctness_gates"
                                                                for f in t.gate_failures)),
                      "model": loaders.TGR_TEACHERS[k].generator, "licence": loaders.TGR_TEACHERS[k].licence}
                  for k in a.teachers}
        supply = {"plan": "data/manifests/v6/DATA_CARD_v6.md (terminal slice)", "run_id": run_id, "teachers": list(a.teachers),
                  "budget_loss_tokens": a.budget, "funnel": funnel, "attempts": attempts, "draw": dr.table,
                  "supply_tasks_by_band": {b: sum(1 for t in trials_by_task if bands[t] == b) for b in select_v5.BANDS}}
        files_out["supply.json"] = {"sha256": B.write_json(out_dir / "supply.json", supply)}
        manifest = {
            "slice": "terminal_v6", "plan": "data/manifests/v6/DATA_CARD_v6.md (terminal slice)",
            "run_id": run_id, "created": utc_now(), "code_commit": git["commit"], "seed": a.seed,
            "final": not a.limit_tasks and a.budget == BUDGET_LOSS_TOKENS and a.teachers == select_v5.TEACHERS,
            "inputs": check, "efficiency_version": E.VERSION, "turn_roles_version": turn_roles.VERSION,
            "rules": {"admission": f"graded pass; completed; verified completion; zero wasted actions; steps <= {MAX_STEPS} "
                                   "(no episode-level thinking or output gate)",
                      "turns": f"<= {TRAINED_TURNS} trained turns per episode in turn_roles.priority_turns order, then a "
                               "seeded fill; only turns within the band's per-turn ceiling (thinking read from the row); "
                               "never the confirmation reply; the hand-in turn exactly once (a trial whose hand-in turn "
                               "is over the ceiling is skipped)",
                      "trials": f"<= 2 per task; a second trial has >= {SECOND_MIN_STEPS} steps; teachers by priority",
                      "draw": "select_v5.draw: new tasks first (hard once, medium 55 / easy 45), then second trials "
                              "<= 25% of episodes", "ceilings": attempts[-1]["ceilings"], "acceptance": ACCEPT,
                      "row_caps": {b: v.row_max for b, v in E.TERMINAL_LIMITS.items()}, "row_loss_cap": ROW_LOSS_CAP},
            "exclusions": {"dev_terminal_task_ids": len(dev_tasks)}, "d1_whole_file": {t: r.counts() for t, r in d1_files.items()},
            "dedup": dedup_report, "screens": admission.screen_counts([t.screens for t in trials if t.screens is not None]),
            "files": files_out, "report": rep, "attempts": attempts, "checks": checks_out,
        }
        files_out["manifest.json"] = {"sha256": B.write_json(out_dir / "manifest.json", manifest)}
        run.metric({"step": "done", "episodes": rep["episodes"], "rows": rep["rows"], "loss_tokens": rep["loss_tokens"],
                    "checks_pass": ok})
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("workers", a.workers, "processes", "loading, profiling, rendering and checks"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"), ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if ok else "failed",
                           summary={k: rep[k] for k in ("rows", "episodes", "tasks", "loss_tokens", "done_turn_share",
                                                        "after_error_turn_share")},
                           outputs={k: {"path": str(out_dir / k), **v} for k, v in files_out.items()})
        if not ok:
            run.error(f"checks failed: {json.dumps({k: v for k, v in checks_out.items() if k not in ('render_check_20', 'row_checks')}, default=str)[:2000]}")
        run.finalize()
        run.log(f"done in {run.elapsed():.0f} s; outputs in {out_dir}; checks {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
