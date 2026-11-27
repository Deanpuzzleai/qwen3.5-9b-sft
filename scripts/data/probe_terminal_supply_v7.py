#!/usr/bin/env python
"""v7 planning probe, analysis only: per-teacher supply of the TermGrade terminal pool after the v6 screens,
decontamination and dev exclusion; per-turn thinking; "tail" candidates; command repeats and debugging arcs; and the
command-repeat share of the v6 slice's trained turns. Writes nothing under data/.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/probe_terminal_supply_v7.py [--workers 48]
Inputs: RAW/tgr/data/messages/{kimi,qwen,gemma_think,glm,deepseek}.parquet, trials.parquet, RAW/tgr_env/labels.jsonl
  (sha256 against raw_sources.json), data/manifests/dev_exclusions.json, the Qwen3.5-9B tokenizer, and the built v6 slice
  data/processed/v6/terminal/episodes.jsonl (read only).
Outputs: logs/<run_id>/ (default data_terminal_supply_v7_<UTC>): stats.json and the five run files; --report (default
  logs/coordination/v7/supply_terminal.md).

Definitions (Qwen3.5-9B tokens; a turn = an assistant message; passing = graded pass that survives the screens, the
dedup and the dev exclusion, as v6 admits before its correctness gates):
  pass rate        reward == 1 over the trials counted toward k on non-dev tasks (trials.parquet)
  thinking         tokens of reasoning_content (efficiency.profile); output per pass = reasoning + content per trial
  after error      the observation before the turn shows an error line (render/turn_roles.error_lines, the v6
                   after_error detector)
  first substantive  the first turn whose keystrokes contain an edit, write, run or inline segment (efficiency.segments)
  tail candidate   thinking in [2,000, 8,000), (a) after error or first substantive, (b) not self-repetitive: no 200-char
                   chunk of the reasoning occurs twice without overlap and distinct non-empty lines / lines >= 0.8,
                   (c) the content parses (Terminus-2 JSON) with at least one non-empty keystroke; loss tokens =
                   reasoning + content tokens
  command repeat   the turn's normalised keystrokes equal the previous turn's (turns with keystrokes, from the second on)
  debugging arc    a turn whose observation shows an error, then a later turn with a different command that edits or
                   writes, then (at or after it) a turn that runs or inlines a check whose observation shows no error
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
from collections import Counter
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
from apex_data.screens import d1_canary  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

TEACHERS = ("kimi", "qwen", "deepseek", "gemma_think", "glm")
NAMES = {"kimi": "Kimi-K3", "qwen": "Qwen3.6-27B", "deepseek": "DeepSeek-V4-Pro", "gemma_think": "Gemma-4-31B (reasoning on)",
         "glm": "GLM-5.2", "gemma": "Gemma-4-31B (reasoning off)"}
TAIL = (2_000, 8_000)
CHUNK = 200
DISTINCT_MIN = 0.8
CHANGE, CHECK = {"edit", "write"}, {"run", "inline"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--messages-dir", type=Path, default=paths.RAW / "tgr" / "data" / "messages")
    p.add_argument("--trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet")
    p.add_argument("--labels", type=Path, default=paths.RAW / "tgr_env" / "labels.jsonl")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES)
    p.add_argument("--dev-exclusions", type=Path, default=paths.DEV_EXCLUSIONS)
    p.add_argument("--v6-episodes", type=Path, default=paths.PROCESSED / "v6" / "terminal" / "episodes.jsonl")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--report", type=Path, default=ROOT / "logs" / "coordination" / "v7" / "supply_terminal.md")
    p.add_argument("--run-id", default=None)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------- turn features
def repeated_chunk(text: str, n: int = CHUNK) -> bool:
    """Some n-char chunk occurs twice without overlapping itself."""
    first: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        w = text[i:i + n]
        j = first.setdefault(w, i)
        if i - j >= n:
            return True
    return False


def distinct_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(set(lines)) / len(lines) if lines else 1.0


def _features(item: tuple[str, str, list[dict[str, Any]], list[int]]) -> dict[str, Any]:
    return turn_features(*item)


def turn_features(teacher: str, trial: str, msgs: list[dict[str, Any]], thinking: list[int],
                  error_fn: Any = turn_roles.error_lines) -> dict[str, Any]:
    """Per-turn features of one trial; `error_fn(terminal output) -> lines` decides "after error" and the arcs (this
    probe: render/turn_roles.error_lines; v7 build: apex_data.terminal.failures.failure_lines)."""
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    conf = set(confirmation_replies(msgs))
    content_tokens = B._token_counts([msgs[i].get("content") or "" for i in asst])
    turns = []
    first_sub = None
    for k, i in enumerate(asst):
        cmd, _done, parsed = E.terminal_command(msgs[i].get("content") or "")
        kinds = {s.kind for s in E.segments(cmd, family="terminal")} if cmd else set()
        prev = msgs[i - 1].get("content") or "" if i > 0 and msgs[i - 1]["role"] == "user" else ""
        nxt = msgs[i + 1].get("content") or "" if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user" else ""
        if first_sub is None and kinds & (CHANGE | CHECK):
            first_sub = k
        turns.append({"k": k, "i": i, "thinking": thinking[k], "content": content_tokens[k], "parsed": parsed,
                      "has_cmd": bool(cmd.strip()), "cmd": E.normalise(cmd), "kinds": sorted(kinds),
                      "after_error": bool(error_fn(turn_roles.terminal_output(prev))),
                      "obs_error": bool(error_fn(turn_roles.terminal_output(nxt))),
                      "confirmation": i in conf})
    for t in turns:
        t["first_substantive"] = t["k"] == first_sub
        t["repeat_prev"] = t["k"] > 0 and t["has_cmd"] and t["cmd"] == turns[t["k"] - 1]["cmd"]
        t["tail"] = None
        if TAIL[0] <= t["thinking"] < TAIL[1]:
            text = (msgs[t["i"]].get("reasoning_content") or "").strip()
            t["tail"] = {"a": t["after_error"] or t["first_substantive"],
                         "b": not repeated_chunk(text) and distinct_line_ratio(text) >= DISTINCT_MIN,
                         "c": t["parsed"] and t["has_cmd"]}
    arc = False
    for a in turns:
        if not a["obs_error"]:
            continue
        for b in turns[a["k"] + 1:]:
            if b["cmd"] != a["cmd"] and set(b["kinds"]) & CHANGE and any(
                    set(c["kinds"]) & CHECK and not c["obs_error"] for c in turns[b["k"]:]):
                arc = True
                break
        if arc:
            break
    for t in turns:
        t.pop("cmd")
    return {"teacher": teacher, "trial": trial, "turns": turns, "debug_arc": arc}


def pct(xs: list[int], q: float) -> int | None:
    return B._pct(xs, q)


# ---------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_terminal_supply_v7")
    with RunDir(run_id, log_name="train.log") as run:
        files = {t: a.messages_dir / f"{t}.parquet" for t in TEACHERS}
        check = B.verify_inputs({**{f"messages_{t}": f for t, f in files.items()}, "trials": a.trials, "labels": a.labels},
                                a.raw_sources)
        bad = [k for k, v in check.items() if not v["matches"]]
        if bad:
            raise SystemExit(f"inputs not matching raw_sources.json sha256: {bad}")
        dev = set(json.loads(a.dev_exclusions.read_text()).get("terminal_task_ids") or [])
        git = git_info()
        run.write_metadata(run_id=run_id, step="data_terminal_supply_v7", command=command_line(), git=git,
                           code_commit=git["commit"], seed=paths.SEED, start_time=utc_now(), status="running",
                           hardware=hardware_info(), model_revision=str(a.tokenizer),
                           data_versions={k: v["revision"] for k, v in check.items()}, config={k: str(v) for k, v in vars(a).items()})
        bands, _ = B.six_model_bands(a.labels)

        # load + screens + dev + dedup, as v6
        ns = argparse.Namespace(teachers=TEACHERS, trials=a.trials, tokenizer=a.tokenizer, workers=a.workers)
        trials = B.load_all(ns, run, check["messages_kimi"]["revision"], None, files)
        d1 = {t: d1_canary.scan_parquet(files[t], id_keys=("trial",)) for t in TEACHERS}
        B.apply_exclusions(trials, {h.record for r in d1.values() for h in r.hits}, dev)
        B.apply_dedup(trials, dev)
        passing = [t for t in trials if t.stage == "candidate"]
        run.log(f"passing after screens, dedup and dev exclusion: {dict(Counter(t.teacher for t in passing))}")

        # per-turn features
        B._G["tokenizer_dir"] = str(a.tokenizer)
        items = [(t.teacher, t.trial, t.loaded.episode["messages"], t.profile.thinking) for t in passing]
        with mp.get_context("fork").Pool(a.workers) as pool:
            feats = list(pool.imap(_features, items, chunksize=8))
        by_trial = {f["trial"]: f for f in feats}
        prof = {t.trial: t.profile for t in passing}

        # pass rates (trials counted toward k on non-dev tasks)
        import pyarrow.parquet as pq
        tr = pq.read_table(a.trials, columns=["task", "model", "reward", "counts_toward_k"]).to_pylist()
        rate: dict[str, dict[str, Any]] = {}
        for m in (*TEACHERS, "gemma"):
            rs = [r for r in tr if r["model"] == m and r["counts_toward_k"] and r["task"] not in dev]
            rate[m] = {"counted_trials": len(rs), "passes": sum(r["reward"] == 1 for r in rs),
                       "pass_rate": round(sum(r["reward"] == 1 for r in rs) / len(rs), 4) if rs else None,
                       "tasks_with_a_pass": len({r["task"] for r in rs if r["reward"] == 1})}

        stats: dict[str, Any] = {}
        for m in TEACHERS:
            fs = [f for f in feats if f["teacher"] == m]
            turns = [t for f in fs for t in f["turns"]]
            think = [t["thinking"] for t in turns]
            tails = [t for t in turns if t["tail"] is not None]
            cand = [t for t in tails if all(t["tail"].values())]
            with_cmd = [t for t in turns if t["k"] > 0 and t["has_cmd"]]
            stats[m] = {
                "model": NAMES[m], "tasks_covered": len({t.task for t in passing if t.teacher == m}),
                "passing_trials": len(fs), **{f"pool_{k}": v for k, v in rate[m].items()},
                "turns_per_passing_trial_mean": round(statistics.fmean(prof[f["trial"]].steps for f in fs), 2) if fs else None,
                "output_tokens_per_pass_mean": round(statistics.fmean(prof[f["trial"]].output_tokens for f in fs)) if fs else None,
                "output_tokens_per_pass_p50": pct([prof[f["trial"]].output_tokens for f in fs], 0.5),
                "thinking_per_turn": {"turns": len(think), "p50": pct(think, 0.5), "p90": pct(think, 0.9),
                                      "p99": pct(think, 0.99), "max": max(think, default=None),
                                      "turns_2k_4k": sum(2_000 <= x < 4_000 for x in think),
                                      "turns_4k_8k": sum(4_000 <= x < 8_000 for x in think),
                                      "turns_8k_plus": sum(x >= 8_000 for x in think)},
                "tail": {"turns_2k_8k": len(tails),
                         "a_after_error_or_first_substantive": sum(t["tail"]["a"] for t in tails),
                         "b_not_self_repetitive": sum(t["tail"]["b"] for t in tails),
                         "c_parsed_command": sum(t["tail"]["c"] for t in tails),
                         "candidates_a_b_c": len(cand),
                         "candidates_after_error": sum(t["after_error"] for t in cand),
                         "candidates_first_substantive": sum(t["first_substantive"] for t in cand),
                         "candidate_trials": sum(any(t["tail"] is not None and all(t["tail"].values()) for t in f["turns"])
                                                 for f in fs),
                         "candidate_loss_tokens": sum(t["thinking"] + t["content"] for t in cand),
                         "candidate_thinking_mean": round(statistics.fmean(t["thinking"] for t in cand)) if cand else None},
                "command_repeat_share": round(sum(t["repeat_prev"] for t in with_cmd) / len(with_cmd), 4) if with_cmd else None,
                "command_repeat_turns": sum(t["repeat_prev"] for t in with_cmd),
                "debugging_arc_share": round(sum(f["debug_arc"] for f in fs) / len(fs), 4) if fs else None,
            }
        stats["gemma"] = {"model": NAMES["gemma"], **{f"pool_{k}": v for k, v in rate["gemma"].items()},
                          "note": "no reasoning (non-thinking); messages not fetched; counts from trials.parquet only"}

        # v6 slice: trained turns repeating the previous turn's command
        v6: dict[str, Any] = {"file": str(a.v6_episodes)}
        if a.v6_episodes.exists():
            n = rep = n_cmd = 0
            for line in a.v6_episodes.read_text().splitlines():
                e = json.loads(line)
                f = by_trial.get(e["trial"])
                if f is None:
                    continue
                at = {t["i"]: t for t in f["turns"]}
                for r in e["rows"]:
                    t = at[r["turn_index"]]
                    n += 1
                    n_cmd += t["k"] > 0 and t["has_cmd"]
                    rep += t["repeat_prev"]
            v6.update(trained_turns=n, with_command_after_first=n_cmd, repeat_prev=rep,
                      share_of_trained_turns=round(rep / n, 4) if n else None,
                      share_of_turns_with_command=round(rep / n_cmd, 4) if n_cmd else None)
        out = {"run_id": run_id, "created": utc_now(), "definitions": __doc__.split("Definitions", 1)[1].strip(),
               "teachers": stats, "v6_slice": v6, "dev_tasks": len(dev),
               "screens_note": "loader graded pass, D1 (file and task text), D4, identity, credentials, dev exclusion, "
                               "task-text dedup: the v6 admission before its correctness gates"}
        B.write_json(run.run_dir / "stats.json", out)
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(markdown(out))
        run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", 0, "USD", "CPU only")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished")
        run.finalize()
        run.log(f"done in {run.elapsed():.0f} s; {a.report}")
    return 0


def markdown(o: dict[str, Any]) -> str:
    T = o["teachers"]
    ks = [k for k in TEACHERS]
    L = [f"# v7 planning: terminal supply per teacher (analysis only), {o['created'][:16]} UTC", "",
         f"Probe run `logs/{o['run_id']}/` (stats.json; code scripts/data/probe_terminal_supply_v7.py). Pool: "
         "ai-and/termgrade-trajectories at the pinned revision after the v6 screens, decontamination, dedup and dev "
         "exclusion (no correctness gates). Tokens are Qwen3.5-9B tokens. Definitions at the end.", "",
         "## 1. Supply", "",
         "| Teacher | Tasks covered | Passing trials | Pass rate (counted, non-dev) | Turns per passing trial | Output tokens per pass (mean / p50) |",
         "|---|---|---|---|---|---|"]
    for k in ks:
        t = T[k]
        L.append(f"| {t['model']} | {t['tasks_covered']} | {t['passing_trials']:,} | {t['pool_pass_rate']:.1%} "
                 f"({t['pool_passes']:,} / {t['pool_counted_trials']:,}) | {t['turns_per_passing_trial_mean']} | "
                 f"{t['output_tokens_per_pass_mean']:,} / {t['output_tokens_per_pass_p50']:,} |")
    g = T["gemma"]
    L.append(f"| {g['model']} | {g['pool_tasks_with_a_pass']} (tasks with a pass) | n/a | {g['pool_pass_rate']:.1%} "
             f"({g['pool_passes']:,} / {g['pool_counted_trials']:,}) | n/a | n/a (no reasoning) |")
    L += ["", "## 2. Thinking per turn (passing trials)", "",
          "| Teacher | Turns | p50 | p90 | p99 | max | 2K-4K turns | 4K-8K turns | >= 8K turns |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ks:
        h = T[k]["thinking_per_turn"]
        L.append(f"| {T[k]['model']} | {h['turns']:,} | {h['p50']:,} | {h['p90']:,} | {h['p99']:,} | {h['max']:,} | "
                 f"{h['turns_2k_4k']:,} | {h['turns_4k_8k']:,} | {h['turns_8k_plus']:,} |")
    L += ["", "## 3. Tail candidates (2K-8K thinking)", "",
          "| Teacher | 2K-8K turns | (a) after error / first substantive | (b) not self-repetitive | (c) parsed command | a+b+c | of which after error / first substantive | trials | loss tokens | thinking mean |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ks:
        x = T[k]["tail"]
        L.append(f"| {T[k]['model']} | {x['turns_2k_8k']:,} | {x['a_after_error_or_first_substantive']:,} | "
                 f"{x['b_not_self_repetitive']:,} | {x['c_parsed_command']:,} | **{x['candidates_a_b_c']:,}** | "
                 f"{x['candidates_after_error']:,} / {x['candidates_first_substantive']:,} | {x['candidate_trials']:,} | "
                 f"{x['candidate_loss_tokens']:,} | {x['candidate_thinking_mean'] or 'n/a'} |")
    L += ["", "## 4. Command repeats and debugging arcs (passing trials)", "",
          "| Teacher | Turns repeating the previous command | Passing trials with a debugging arc |", "|---|---|---|"]
    for k in ks:
        L.append(f"| {T[k]['model']} | {T[k]['command_repeat_share']:.1%} ({T[k]['command_repeat_turns']:,}) | "
                 f"{T[k]['debugging_arc_share']:.1%} |")
    v = o["v6_slice"]
    L += ["", "## 5. v6 terminal slice as built", ""]
    if "trained_turns" in v:
        L.append(f"Trained turns repeating the previous turn's command: {v['repeat_prev']} of {v['trained_turns']:,} "
                 f"({v['share_of_trained_turns']:.1%}; {v['share_of_turns_with_command']:.1%} of the "
                 f"{v['with_command_after_first']:,} trained turns that have a command and a previous turn).")
    L += ["", "## Definitions", "", "```", o["definitions"], "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
