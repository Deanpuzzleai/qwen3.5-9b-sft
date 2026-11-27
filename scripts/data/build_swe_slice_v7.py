#!/usr/bin/env python
"""Build the v7 SWE slice: Open-SWE-Traces v1.2 mini-swe-agent Qwen3.8-27B, one random passing attempt per task,
recovery trials first, the adaptive thinking masks, v6's dose.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_swe_slice_v7.py [--source-run data_swe_v5_20261013_060512]
        [--out-dir data/pools/swe_v7] [--target-loss-tokens 3700000] [--presample-factor 4] [--seed 42] [--workers 24]

Pool: the resolved attempts that passed the row filters of scripts/data/build_swe_slice.py (licence, Scale-SWE-Verified,
D2 and decontamination repository exclusions, tool schema, clean model patch, reasoning on every turn, dev exclusions)
and are banded from all of the task's attempts: WORK/<source-run>/evals/raw_profiles.jsonl (its stage-4 records hold
the raw arc-B recovery flag; their thinking gates are not used).
Steps:
  1. pre-sample (seeded): per (band, language group), --presample-factor x the episodes the dose needs at v6r2's band
     and language shares (at least 12 tasks per cell when supplied); half of each cell's sample from tasks with a
     recovery (arc B) attempt, which then takes a random recovery attempt; the rest a random passing attempt;
  2. render + gates (build_swe_slice.evaluate without its profile gates: the episode rebuilt under our templates,
     the reference-patch leak rule, the D2 text screen, the shared render/filters gates, row cap 65,536); an attempt
     that fails is replaced by another random attempt of its task (up to 3 tries);
  3. masks: apex_data.render.swe_masks rule v7 (S1 reads, S2 unverified submits, routine turns <= 1,000, decision
     turns <= 8K when not repetitive and ending in a tool call, > 8K never); rows over 32K loss tokens after the masks
     are dropped;
  4. draw (apex_data.teacher.select_v5.draw, shares mode): --target-loss-tokens by masked loss tokens at v6r2's band
     shares (hard 26.8 / medium 27.7 / easy 45.5), language quotas at v6r2's shares (Go 28.3 / Python 46.1 /
     TypeScript 16.2 / JavaScript 5.0 / other 4.4; a short group's gap to Go, then Python), recovery preference to
     --recovery-share per cell, category round-robin, repository caps (Go and TypeScript 5, others 3), one task per
     (repository, PR); random order within a cell (no lower-half preference);
  5. tail cap: trained turns above 2K thinking hold at most --tail-cap of the slice's loss tokens; above it, such
     turns are untrained from easy episodes first, then medium, then hard (last drawn first, longest turn first).
Outputs (<out-dir>): rows.parquet (rows in draw order with the v7 masks, provenance and turn columns; git-ignored),
manifest.json, selection.json, supply.json, samples.json (the quality-read sample: 15 tail turns, 10 recovery
episodes, 10 routine turns near 1,000), dropped.jsonl.gz; WORK/<run_id>/ (render cache); logs/<run_id>/ (the five
run files). Seeded by --seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import json
import math
import multiprocessing as mp
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import build_swe_slice as bss  # noqa: E402  (evaluate, worker init, record helpers)
from apex_data import paths  # noqa: E402
from apex_data.render import swe_masks as sm  # noqa: E402
from apex_data.schema import PROVENANCE_FIELDS, validate_provenance  # noqa: E402
from apex_data.teacher import filter as tfilter  # noqa: E402
from apex_data.teacher import load, render  # noqa: E402
from apex_data.teacher import select_v5 as sv  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import ROW_COLUMNS, arrow_schema, load_tokenizer  # noqa: E402

V6_BAND_SHARES = {"hard": 0.268, "medium": 0.277, "easy": 0.455}
V6_LANGUAGE_SHARES = {"go": 0.283, "py": 0.461, "ts": 0.162, "js": 0.050, "other": 0.044}
REPO_CAPS = {"go": 5, "ts": 5}
ROW_CAP = 65_536
TAIL_MIN = 2048
EST_LOSS_FRACTION = 0.6            # masked loss per episode ~ this share of the raw output tokens (the 500-task simulation)
EXTRA = (("row_key", pa.string()), ("band", pa.string()), ("language_group", pa.string()), ("steps", pa.int32()),
         ("recovery_arc", pa.bool_()), ("draw_order", pa.int32()), ("recovery_preferred", pa.bool_()),
         ("repo_root", pa.string()), ("attempt", pa.int32()), ("thinking_trained_max", pa.int32()),
         ("thinking_trained_mean", pa.float64()), ("trained_turns_over_2k", pa.int32()), ("loss_tokens_over_2k", pa.int32()),
         ("submit_masked", pa.bool_()), ("turn_kinds", pa.string()), ("turns_json", pa.large_string()))
_W: dict[str, Any] = {}


def _worker_render(job: tuple[str, str, list[dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    subsource, shard_path, items = job
    raws = load.read_rows(subsource, Path(shard_path), [it["row"] for it in items])
    out = []
    for it, raw in zip(items, raws, strict=True):
        if raw["trajectory_id"] != it["trajectory_id"]:
            raise RuntimeError(f"{shard_path} row {it['row']}: trajectory id mismatch")
        out.append(bss.evaluate(raw, it, _W["tok"], _W["bundle"], _W["locked"], _W["revision"], _W["repo_root"],
                                profile_gates=False, row_max=ROW_CAP))
    return out


def _init(tokenizer_dir: str, bundle: dict[str, Any], revision: str) -> None:
    bss._worker_render_init(tokenizer_dir, bundle, "/dev/null", revision, "/app")
    _W.update(bss._W)


def turn_letter(t: sm.Turn) -> str:
    letter = {"read": "r", "edit": "e", "run": "u", "submit": "s", "other": "o"}[t.kind]
    return (letter.upper() if not t.trained else letter) + ("*" if t.decision else "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run", default="data_swe_v5_20261013_060512", help="the build_swe_slice run whose WORK cache holds the pool")
    p.add_argument("--out-dir", type=Path, default=paths.POOLS / "swe_v7")
    p.add_argument("--target-loss-tokens", type=int, default=3_700_000)
    p.add_argument("--presample-factor", default="hard=10,medium=10,easy=4",
                   help="per band: tasks pre-sampled per episode the dose needs (hard and medium attempts often fail the 64K row cap)")
    p.add_argument("--easy-max-share", type=float, default=0.45, help="easy at most this share of the SWE loss tokens (trimmed from "
                   "the end of its draw order after the draw)")
    p.add_argument("--recovery-share", type=float, default=0.35)
    p.add_argument("--tail-cap", type=float, default=0.10, help="trained turns > 2K thinking: at most this share of the loss tokens")
    p.add_argument("--tail-min-share", type=float, default=0.01, help="when trained turns > 2K are below this share of the trained "
                   "turns, draw again with the episodes holding one first in every cell")
    p.add_argument("--exclude-task-ids", type=Path, default=None, help="task ids (oswt:<instance>) removed before the draw, after "
                   "rendering, so the seeded pre-sample stays the same (screen drops of an earlier mixture build)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or common.new_run_id("data_swe_v7")
    src = paths.WORK / a.source_run
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="data_swe_v7", command=common.command_line(), git=common.git_info(), seed=a.seed,
                           hardware=common.hardware_info(), config={k: str(v) for k, v in vars(a).items()},
                           data_versions=load.raw_revisions(), model_revision=common.model_revision(),
                           start_time=common.utc_now(), status="running")
        rng = random.Random(a.seed)
        prof = {}
        with (src / "evals" / "raw_profiles.jsonl").open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                prof[p["trajectory_id"]] = p
        index = {r.trajectory_id: r for r in load.index_from_table(pq.read_table(src / "index.parquet")) if r.trajectory_id in prof}
        upstream = load.load_upstream()
        shard_dirs = {s: p.parent for s, p in load.list_shards()}
        meta: dict[str, dict[str, Any]] = {}
        for t, r in index.items():
            up = upstream.get((r.subsource, r.instance_id))
            files, added = (up.size.files, up.size.added) if up else (r.ref_files, r.ref_added)
            meta[t] = {"instance_id": r.instance_id, "subsource": r.subsource, "language": sv.LANGUAGE_CODES[r.language],
                       "repo": r.repo, "license": r.license, "band": prof[t]["band"], "category": r.category,
                       "stratum": load.size_stratum(files, added), "files": files, "added": added,
                       "pr": up.pr if up and up.pr else load.pr_number(r.instance_id)}
        tasks: dict[tuple[str, str], dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
        for t, m in meta.items():
            tasks[(m["band"], sv.GROUP_OF[m["language"]])][(m["subsource"], m["instance_id"])].append(t)
        run.log(f"pool: {len(meta)} passing attempts, {sum(len(v) for v in tasks.values())} tasks from {a.source_run}")

        # ---- 1. pre-sample -------------------------------------------------------------------------------------
        out_mean = {b: sum(prof[t]["profile"]["output_tokens"] for t in meta if meta[t]["band"] == b) /
                    max(1, sum(meta[t]["band"] == b for t in meta)) for b in sv.BANDS}
        need = {b: a.target_loss_tokens * V6_BAND_SHARES[b] / (EST_LOSS_FRACTION * out_mean[b]) for b in sv.BANDS}
        factor = {k: float(v) for k, v in (x.split("=") for x in a.presample_factor.split(","))}
        chosen: dict[tuple[str, str], dict[str, Any]] = {}
        for (band, group), tk in sorted(tasks.items()):
            n = min(len(tk), max(12, math.ceil(factor[band] * need[band] * V6_LANGUAGE_SHARES[group])))
            keys = sorted(tk)
            rng.shuffle(keys)
            rec = [k for k in keys if any(prof[t]["profile"]["recovery"] for t in tk[k])]
            take_rec = rec[:math.ceil(n / 2)]
            rest = [k for k in keys if k not in set(take_rec)][: n - len(take_rec)]
            for k in take_rec + rest:
                pool_ = sorted(t for t in tk[k] if prof[t]["profile"]["recovery"]) if k in set(take_rec) else sorted(tk[k])
                rng.shuffle(pool_)
                chosen[k] = {"order": pool_, "others": [t for t in sorted(tk[k]) if t not in pool_], "tries": 0}
        run.log(f"pre-sample: {len(chosen)} tasks (need estimate {json.dumps({b: round(v) for b, v in need.items()})})")

        # ---- 2. render + gates ---------------------------------------------------------------------------------
        bundle = render.load_templates()
        rev = load.raw_revisions()["oswt"]["revision"]
        passed: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        failed: list[dict[str, Any]] = []
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(str(paths.BASE_MODEL_DIR), bundle.to_dict(), rev)) as pool:
            for attempt_round in range(3):
                todo = [(k, c) for k, c in chosen.items() if k not in passed and (c["order"] or c["others"])]
                if not todo:
                    break
                by_shard: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
                for k, c in todo:
                    t = c["order"].pop(0) if c["order"] else c["others"].pop(0)
                    r = index[t]
                    by_shard[(r.subsource, r.shard)].append({"row": r.row, "trajectory_id": t, "task": k, **meta[t]})
                jobs = [(sub, str(shard_dirs[sub] / shard), items[i:i + 60]) for (sub, shard), items in sorted(by_shard.items())
                        for i in range(0, len(items), 60)]
                n_ok = 0
                for res in pool.imap_unordered(_worker_render, jobs):
                    for out, row in res:
                        k = (out["subsource"], out["instance_id"])
                        if out["ok"]:
                            passed[k] = (out, row)
                            n_ok += 1
                        else:
                            failed.append({k2: out.get(k2) for k2 in ("subsource", "instance_id", "trajectory_id", "band", "stage", "reason", "detail")})
                run.log(f"render round {attempt_round + 1}: {len(todo)} attempts, {n_ok} pass")

        # ---- 3. masks ------------------------------------------------------------------------------------------
        tok = load_tokenizer(paths.BASE_MODEL_DIR)
        cands, masked_rows, turn_info, dropped_loss_cap = [], {}, {}, 0
        for k, (out, row) in passed.items():
            m = meta[out["trajectory_id"]]
            new, turns = sm.mask_row(row, tok.decode, rule="v7")
            if new["n_assistant_tokens"] > sv.LOSS_CAP_PER_ROW:
                dropped_loss_cap += 1
                failed.append({"subsource": k[0], "instance_id": k[1], "trajectory_id": out["trajectory_id"], "band": m["band"],
                               "stage": "loss_cap", "reason": "loss_cap_after_masks", "detail": str(new["n_assistant_tokens"])})
                continue
            masked_rows[out["trajectory_id"]] = new
            turn_info[out["trajectory_id"]] = turns
            cands.append(sv.Candidate(k[0], k[1], out["trajectory_id"], m["band"], m["language"], m["category"],
                                      tfilter.normalise_repo(m["repo"]), m["pr"], new["n_assistant_tokens"], new["n_tokens"],
                                      0.0, bool(prof[out["trajectory_id"]]["profile"]["recovery"])))
        excluded = bss.read_exclusions(a.exclude_task_ids)
        n_before = len(cands)
        cands = [c for c in cands if f"oswt:{c.instance_id}" not in excluded]
        supply = sv.supply_table(cands)
        run.log(f"candidates after masks: {len(cands)} ({dropped_loss_cap} over the 32K loss cap; {n_before - len(cands)} excluded by "
                f"--exclude-task-ids)")

        # ---- 4. draw -------------------------------------------------------------------------------------------
        switches = sv.DrawSwitches(category_round_robin=True, recovery_preference=True, lower_half=False)

        def run_draw(cs: list[sv.Candidate]) -> sv.DrawResult:
            return sv.draw(cs, total_loss=a.target_loss_tokens, seed=a.seed, switches=switches, band_shares=V6_BAND_SHARES,
                           language_shares=V6_LANGUAGE_SHARES, repo_caps=REPO_CAPS, recovery_min=a.recovery_share)

        def long_share(res: sv.DrawResult) -> float:
            tr = [t for p in res.picks for t in turn_info[p.candidate.trajectory_id] if t.trained]
            return sum(t.thinking > TAIL_MIN for t in tr) / len(tr) if tr else 0.0
        has_tail = {t: any(x.trained and x.thinking > TAIL_MIN for x in turns) for t, turns in turn_info.items()}
        result = run_draw(cands)
        tail_preference = {"share_first_draw": round(long_share(result), 4), "applied": False}
        if tail_preference["share_first_draw"] < a.tail_min_share:
            result = run_draw([dataclasses.replace(c, priority=1 if has_tail[c.trajectory_id] else 0) for c in cands])
            tail_preference.update(applied=True, share_after=round(long_share(result), 4),
                                   candidates_with_tail=sum(has_tail[c.trajectory_id] for c in cands))
        run.log(f"tail preference: {json.dumps(tail_preference)}")
        picks = result.picks
        # easy at most --easy-max-share of the loss tokens: the last drawn easy episodes go when a harder band ran short
        easy_cut = []
        while True:
            total = sum(p.candidate.loss_tokens for p in picks)
            easy = [p for p in picks if p.candidate.band == "easy"]
            if not easy or sum(p.candidate.loss_tokens for p in easy) <= a.easy_max_share * total:
                break
            last = max(easy, key=lambda p: p.order)
            easy_cut.append(last.candidate.trajectory_id)
            picks = [p for p in picks if p is not last]
        for b in result.bands.values():
            b.picks = [p for p in b.picks if p.candidate.trajectory_id not in set(easy_cut)]
            b.loss_tokens = sum(p.candidate.loss_tokens for p in b.picks)
            b.seq_tokens = sum(p.candidate.seq_tokens for p in b.picks)
        picks = result.picks
        report = sv.draw_report(result)
        report["easy_cap"] = {"max_share": a.easy_max_share, "episodes_removed": len(easy_cut)}
        run.log("draw: " + json.dumps({k: report[k] for k in ("episodes", "loss_tokens", "band_loss_shares", "group_shares", "recovery_share")}))

        # ---- 5. tail cap ---------------------------------------------------------------------------------------
        def tail_loss() -> tuple[int, int]:
            tail = sum(t.end - t.start for p in picks for t in turn_info[p.candidate.trajectory_id] if t.trained and t.thinking > TAIL_MIN)
            return tail, sum(masked_rows[p.candidate.trajectory_id]["n_assistant_tokens"] for p in picks)
        untrained = Counter()
        tail, total = tail_loss()
        tail_before = (tail, total)
        for band in ("easy", "medium", "hard"):
            for p in sorted((p for p in picks if p.candidate.band == band), key=lambda p: -p.order):
                if tail <= a.tail_cap * total:
                    break
                tid = p.candidate.trajectory_id
                for t in sorted((t for t in turn_info[tid] if t.trained and t.thinking > TAIL_MIN), key=lambda t: -t.thinking):
                    if tail <= a.tail_cap * total:
                        break
                    t.trained, t.reason = False, "tail_cap"
                    row = masked_rows[tid]
                    row["assistant_masks"][t.start:t.end] = [0] * (t.end - t.start)
                    row["n_assistant_tokens"] -= t.end - t.start
                    row["n_assistant_turns"] -= 1
                    tail -= t.end - t.start
                    total -= t.end - t.start
                    untrained[band] += 1
        run.log(f"tail cap {a.tail_cap}: > 2K loss {tail_before[0]:,} of {tail_before[1]:,} -> {tail:,} of {total:,}; untrained {dict(untrained)}")

        # ---- outputs -------------------------------------------------------------------------------------------
        v5_man = json.loads((paths.POOLS / "swe_v5" / "manifest.json").read_text(encoding="utf-8"))
        out_rows, problems = [], []
        stats = defaultdict(list)
        for p in picks:
            c, t = p.candidate, p.candidate.trajectory_id
            row, turns, rec = masked_rows[t], turn_info[t], passed[(c.subsource, c.instance_id)][0]
            prov = rec["provenance"]
            validate_provenance(prov)
            trained = [x for x in turns if x.trained]
            for x in trained:
                stats["all"].append(x.thinking)
                (stats["decision"] if x.decision else stats["routine"]).append(x.thinking)
                if row["input_ids"][x.end - 1] != sm.IM_END:
                    problems.append(f"{t}: a trained span does not end with <|im_end|>")
                if x.thinking > sm.DECISION_MAX or (not x.decision and x.thinking > sm.ROUTINE_MAX) or not any(cc.strip() for cc in x.commands):
                    problems.append(f"{t}: turn {x.index} breaks the v7 rule")
            if not trained:
                problems.append(f"{t}: no trained turn")
            big = [x for x in trained if x.thinking > TAIL_MIN]
            out_rows.append({**{k: row[k] for k in ROW_COLUMNS}, **{k: prov[k] for k in PROVENANCE_FIELDS}, "difficulty": c.band,
                             "row_key": prov["row_key"], "band": c.band, "language_group": c.group,
                             "steps": len(turns), "recovery_arc": c.recovery, "draw_order": p.order,
                             "recovery_preferred": p.recovery_preferred, "repo_root": "/app", "attempt": 1,
                             "thinking_trained_max": max((x.thinking for x in trained), default=0),
                             "thinking_trained_mean": round(sum(x.thinking for x in trained) / len(trained), 2) if trained else 0.0,
                             "trained_turns_over_2k": len(big), "loss_tokens_over_2k": sum(x.end - x.start for x in big),
                             "submit_masked": any(x.kind == "submit" and not x.trained for x in turns),
                             "turn_kinds": "".join(turn_letter(x) for x in turns), "turns_json": sm.dumps_turns(turns)})
        schema = pa.schema([*arrow_schema(), *[(k, bss.PROVENANCE_TYPES.get(k, pa.string())) for k in PROVENANCE_FIELDS], *EXTRA])
        rows_path = a.out_dir / "rows.parquet"
        tmp = rows_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), tmp)
        bss.legacy.publish(tmp, rows_path, run)

        def pct(v: list[int], q: float) -> float:
            v = sorted(v)
            return float(v[min(len(v) - 1, int(q * len(v)))]) if v else 0.0
        loss = sum(r["n_assistant_tokens"] for r in out_rows)
        big_loss = sum(r["loss_tokens_over_2k"] for r in out_rows)
        trained_stats = {"trained_turns": len(stats["all"]), "p50": pct(stats["all"], .5), "p90": pct(stats["all"], .9),
                         "p99": pct(stats["all"], .99), "max": max(stats["all"], default=0), "routine_p90": pct(stats["routine"], .9),
                         "decision_p90": pct(stats["decision"], .9), "share_turns_over_2k": round(sum(x > TAIL_MIN for x in stats["all"]) / len(stats["all"]), 4),
                         "loss_tokens": loss, "share_loss_over_2k": round(big_loss / loss, 4),
                         "share_episodes_with_trained_over_2k": round(sum(r["trained_turns_over_2k"] > 0 for r in out_rows) / len(out_rows), 4)}
        reasons = Counter(x.reason for p in picks for x in turn_info[p.candidate.trajectory_id] if x.reason)
        long_by: Counter = Counter()
        for p in picks:
            turns = turn_info[p.candidate.trajectory_id]
            fe = next((i for i, x in enumerate(turns) if x.kind == "edit"), None)
            for x in turns:
                if x.trained and x.thinking > TAIL_MIN:
                    failure = any(sm.shows_failure(o) for o in x.observations)
                    long_by[f"{x.reason or x.kind}|{'after_failure' if failure else ('before_first_edit' if fe is not None and x.index == fe - 1 else 'neither')}"] += 1
        # quality-read sample (seeded)
        srng = random.Random(f"{a.seed}:samples")
        tails = [(r["row_key"], x["index"], x["thinking"]) for r in out_rows for x in json.loads(r["turns_json"]) if x["trained"] and x["thinking"] > TAIL_MIN]
        routine = [(r["row_key"], x["index"], x["thinking"]) for r in out_rows for x in json.loads(r["turns_json"])
                   if x["trained"] and not x["decision"] and 800 <= x["thinking"] <= 1000]
        recov = [r["row_key"] for r in out_rows if r["recovery_arc"]]
        samples = {"rows_file": str(rows_path), "note": "row_key = <trajectory_id>:-1; turn = the 0-based assistant turn (turns_json index)",
                   "tail_turns": [{"row_key": k, "turn": i, "thinking": th} for k, i, th in srng.sample(tails, min(8, len(tails)))],
                   "recovery_episodes": [{"row_key": k} for k in srng.sample(recov, min(10, len(recov)))],
                   "routine_turns_near_1000": [{"row_key": k, "turn": i, "thinking": th} for k, i, th in srng.sample(routine, min(10, len(routine)))]}
        rp = Counter(r["band"] for r in out_rows)
        checks = {"problems": problems[:50], "rows": len(out_rows), "tail_cap_ok": big_loss <= a.tail_cap * loss + 1,
                  "max_instances_per_repository": report["max_instances_per_repository"], "pass": not problems and big_loss <= a.tail_cap * loss + 1}
        selection = {"run_id": run_id, "seed": a.seed, "report": report,
                     "rows": [{"order": p.order, "task_id": f"oswt:{p.candidate.instance_id}", "trajectory_id": p.candidate.trajectory_id,
                               "band": p.candidate.band, "language": p.candidate.language, "group": p.candidate.group,
                               "category": p.candidate.category, "repo": p.candidate.repo, "recovery": p.candidate.recovery,
                               "loss_tokens": masked_rows[p.candidate.trajectory_id]["n_assistant_tokens"]} for p in picks]}
        manifest = {"pool": "swe_v7", "source": "oswt", "run_id": run_id, "built": common.utc_now(), "seed": a.seed,
                    "plan": "data/manifests/v7/DATA_CARD_v7.md", "code_commit": common.git_info().get("commit"),
                    "source_run": a.source_run, "dataset": v5_man.get("dataset"), "generator": v5_man.get("generator"),
                    "templates": v5_man.get("templates"), "tokenizer": v5_man.get("tokenizer"),
                    "rules": {"masks": {k: v for k, v in sm.rules().items() if k != "doc"} | {"rule": "v7", "routine_max": sm.ROUTINE_MAX,
                                                                                          "decision_max": sm.DECISION_MAX},
                              "draw_settings": {"band_shares": V6_BAND_SHARES, "language_shares": V6_LANGUAGE_SHARES,
                                                "repo_caps": REPO_CAPS, "max_per_repository": sv.MAX_PER_REPO,
                                                "target_loss_tokens": a.target_loss_tokens, "recovery_share": a.recovery_share,
                                                "attempt": "one random passing attempt per task (a random recovery attempt for the recovery-first tasks)",
                                                "lower_half": False},
                              "tail_cap": a.tail_cap, "tail_min_share": a.tail_min_share, "easy_max_share": a.easy_max_share, "presample_factor": factor, "row_cap": ROW_CAP, "loss_cap_per_row": sv.LOSS_CAP_PER_ROW},
                    "funnel": {"pool_attempts": len(meta), "presample_tasks": len(chosen), "render_pass": len(passed),
                               "render_fail_by_stage": dict(Counter(f["stage"] for f in failed)), "candidates": len(cands),
                               "loss_cap_drops": dropped_loss_cap, "excluded_task_ids": n_before - len(cands),
                               "exclusions_file": str(a.exclude_task_ids) if a.exclude_task_ids else None},
                    "draw": report, "supply": supply, "tail_cap": {"before": tail_before, "after": [tail, total], "untrained_by_band": dict(untrained)},
                    "trained_turns": trained_stats, "mask_reasons": dict(reasons), "episodes_by_band": dict(rp),
                    "long_turns_by_reason": dict(long_by), "tail_preference": tail_preference,
                    "selected_thinking_per_turn": {"mean": round(sum(stats["all"]) / len(stats["all"]), 1), "p90": trained_stats["p90"],
                                                   "max": trained_stats["max"]},
                    "checks": checks, "outputs": {"rows": str(rows_path), "rows_sha256": common.sha256_file(rows_path), "n_rows": len(out_rows)}}
        common.write_json(a.out_dir / "selection.json", selection)
        common.write_json(a.out_dir / "supply.json", {"run_id": run_id, "admitted": supply, "funnel": manifest["funnel"]})
        common.write_json(a.out_dir / "samples.json", samples)
        common.write_json(a.out_dir / "manifest.json", manifest)
        with gzip.GzipFile(a.out_dir / "dropped.jsonl.gz", "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8") as f:
            for d in failed:
                f.write(json.dumps(d) + "\n")
        run.log(f"rows {len(out_rows)} -> {rows_path}; trained turns {json.dumps(trained_stats)}; checks {json.dumps({k: v for k, v in checks.items() if k != 'problems'})}")
        if problems:
            run.error(f"checks: {problems[:5]}")
        run.write_resources([("duration_sec", round(time.time() - t0, 1), "s", "wall clock"), ("workers", a.workers, "proc", "render"),
                             ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "parent"),
                             ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"), ("cost_usd", 0, "USD", "CPU only")])
        run.write_metadata(end_time=common.utc_now(), status="ok" if checks["pass"] else "checks_failed", trained_turns=trained_stats)
        run.finalize()
        return 0 if checks["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
