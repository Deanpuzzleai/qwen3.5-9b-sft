#!/usr/bin/env python
"""Build the stage-2 turn-level preference pairs.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_stage2_pairs.py --api-base http://172.18.0.1:8002/v1 \
        --model qwen35-base --n-swe 50 --n-terminal 50 [--swe-run WORK/<swe slice run>] [--exclude-swe FILE]
        [--exclude-terminal FILE] [--samples 6] [--workers 8] [--run-id ID --resume] [--out-dir DIR]
        [--reuse-from RUN_ID]      (relabel: copy that run's states and candidates, sample nothing new)
        [--states-from RUN_ID]     (copy that run's states only, then sample them anew: a new model on the same states)
        [--states-only]            (step 1 only; a later --resume of the same run id samples and labels)
        [--endpoints-file F]       (endpoints that join a running sampler: re-read every 30 s)

Steps (each writes into WORK/<run_id>/, so --resume continues where a run stopped):
  1 states   apex_data.pairs.states: SWE states from the gate-passing trajectories of --swe-run, terminal states from
             Kimi-K3's passing and failing TermGrade trials; dev ids and the --exclude-* ids (stage-1 instances and tasks)
             never enter; --n-swe / --n-terminal states per family; then scripts/data/run_screens.py p1 over the states'
             tasks (D1, D2, D3 sealed, D4, dedup and lineage against dev; --no-screens skips it): a dropped task loses
             its states                                                                      -> states.jsonl
  2 sample   per state, --samples completions from --model at --api-base through /v1/completions with the state's token
             ids (apex_data.pairs.candidates); states already sampled are skipped on --resume -> candidates.jsonl
  3 label    parse and label every candidate and the teacher's turn (apex_data.pairs.rules.label_turn), pick the pairs
             (rules V, R, D, P, E with the teacher fallback), validate each pair with apex_train.stage2's
             PreferencePair.from_dict and render_pair under configs/stage2.yaml's dpo settings (a failure is dropped and
             counted), then the set constraints (rules.balance: E <= 40 percent, families 60 / 40)
Inputs   --swe-run (default the first build's SWE run, WORK/data_swe_teacher_app_refill_20261012_074847: evals/outcomes.jsonl,
         evals/rows/*.parquet, index.parquet), RAW/tgr/data/messages/kimi.parquet, RAW/tgr_env/labels.jsonl,
         data/manifests/dev_exclusions.json, configs/stage2.yaml (dpo), configs/eval/{swebench_pro_v2_hard,
         terminal_bench_2}.yaml (sampling), the Qwen3.5-9B tokenizer (paths.BASE_MODEL_DIR). --exclude-swe /
         --exclude-terminal: text files with one instance id / TermGrade task name per line, or JSON lists.
Outputs  <out-dir> (default WORK/<run_id>/): pairs.jsonl (apex_train.stage2's pair format: id, prompt, tools, chosen,
         rejected, metadata with rule, family, band, state, chosen source and thinking tokens), report.json (supply,
         candidate labels, yield per rule and family, drops, balance), examples.md (5 pairs per rule, for the audit);
         logs/<run_id>/: metadata.json, train.log, metrics.jsonl, errors.log, resources.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.pairs import candidates as cand_mod, rules, states as st  # noqa: E402
from apex_data.rollout.endpoint import Endpoint  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

DEFAULT_SWE_RUN = paths.WORK / "data_swe_teacher_app_refill_20261012_074847"
KIMI = paths.RAW / "tgr" / "data" / "messages" / "kimi.parquet"
LABELS = paths.RAW / "tgr_env" / "labels.jsonl"
DEV = paths.REPO / "data" / "manifests" / "dev_exclusions.json"
GPU_RATE_USD = 8.26                    # per GPU hour (single B200 VM)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-base", required=True, action="append", help="OpenAI-compatible endpoint, e.g. "
                   "http://172.18.0.1:8002/v1; repeat it to sample from several servers of the same model (a shared queue "
                   "in the states' order, --workers requests in flight per endpoint)")
    p.add_argument("--model", required=True, action="append", help="served model name (qwen35-base, qwen35-sft); one for "
                   "every --api-base, or one per --api-base in the same order")
    p.add_argument("--endpoint-deadline", action="append", default=[], metavar="URL=UTC",
                   help="no request to URL after UTC (ISO, e.g. 2026-10-13T10:30:00Z): new requests stop --deadline-margin-s "
                        "before it, requests in flight time out at it (vLLM aborts a request whose client disconnects) and "
                        "their states go back to the queue for the other endpoints")
    p.add_argument("--deadline-margin-s", type=float, default=300, help="see --endpoint-deadline (default %(default)s)")
    p.add_argument("--endpoints-file", type=Path, default=None,
                   help="JSON list of endpoints that join the running sampler, re-read every 30 s: [{\"api_base\", \"model\", "
                        "\"deadline\": UTC ISO | null, \"cap_s\": seconds after joining | null}]; an entry joins once its "
                        "/models answers; the earlier of deadline and join + cap_s acts as --endpoint-deadline")
    p.add_argument("--progress-marker", type=Path, default=None,
                   help="JSON rewritten every 25 sampled states: done / total, per endpoint, rate and ETA (UTC)")
    p.add_argument("--max-endpoints", type=int, default=4, help="thread pool size = --workers x this (default %(default)s)")
    p.add_argument("--sampling-done-marker", type=Path, default=None, help="write this marker (JSON) when sampling ends, "
                   "before labelling: the endpoints are free from then on")
    p.add_argument("--n-swe", default="50", help="SWE states, or `auto`: --swe-share of all states (default %(default)s)")
    p.add_argument("--n-terminal", default="50", help="terminal states, or `all`: every held-out task (default %(default)s)")
    p.add_argument("--swe-share", type=float, default=0.4, help="SWE share of the states with --n-swe auto, and of the "
                   "sampling order (default %(default)s)")
    p.add_argument("--per-episode-swe", type=int, default=2, help="states per SWE episode (default %(default)s)")
    p.add_argument("--min-prompt-swe", type=int, default=0, help="SWE states only at turns whose prompt has at least this "
                   "many tokens (the P4a top-up: later turns, where runaways concentrate; default %(default)s = any)")
    p.add_argument("--terminal-trials-per-task", type=int, default=1, help="terminal episodes per task (default %(default)s)")
    p.add_argument("--exclude-terminal-episodes", type=Path, default=None,
                   help="terminal episode ids (tgr:<trial>) never used, e.g. the trials of an earlier state set")
    p.add_argument("--per-episode-terminal", type=int, default=3, help="states per terminal episode (default %(default)s)")
    p.add_argument("--time-cap-s", type=float, default=0, help="stop sending new sampling requests after this many seconds "
                   "(0 = none); the states are sent in an order that keeps --swe-share, so a cut keeps the share")
    p.add_argument("--family-balance", action="store_true", help="cut the larger family to the plan's 60 / 40 (off by "
                   "default; the natural split is reported)")
    p.add_argument("--samples", type=int, default=6, help="samples per state (default %(default)s)")
    p.add_argument("--workers", type=int, default=8, help="concurrent requests (default %(default)s)")
    p.add_argument("--swe-run", type=Path, default=DEFAULT_SWE_RUN, help="SWE slice build in WORK (default %(default)s)")
    p.add_argument("--exclude-swe", type=Path, default=None, help="stage-1 SWE instance ids to exclude")
    p.add_argument("--swe-rollouts", type=Path, default=None, help="extra SWE states: SWE-smith rollout episodes.jsonl")
    p.add_argument("--swe-rollout-share", type=float, default=0.5, help="share of the SWE states from --swe-rollouts "
                   "(default %(default)s)")
    p.add_argument("--exclude-terminal", type=Path, default=None, help="stage-1 TermGrade task names to exclude")
    p.add_argument("--kimi", type=Path, default=KIMI)
    p.add_argument("--labels", type=Path, default=LABELS)
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="default data_stage2_pairs_<UTC>")
    p.add_argument("--resume", action="store_true", help="continue --run-id: reuse its states, skip sampled states")
    p.add_argument("--out-dir", type=Path, default=None, help="pairs, report and examples (default WORK/<run_id>)")
    p.add_argument("--log-root", type=Path, default=None, help="logs/<run_id>/ under this root (tests)")
    p.add_argument("--no-sample", action="store_true", help="label the candidates there are, send no request (with --reuse-from: "
                   "a preview or the audit sample of a run still sampling)")
    p.add_argument("--states-only", action="store_true", help="build and screen the states, then stop (no endpoint needed)")
    p.add_argument("--screen-pool", action="append", default=[], metavar="NAME=PATH",
                   help="reference pool for the p1 pass (run_screens --pool; e.g. the stage-1 slices' and the dev screen "
                        "items of P1e): a state task that is one of its items, or in a dedup group with one, is dropped")
    p.add_argument("--no-screens", action="store_true", help="skip the run_screens.py p1 pass over the states' tasks (tests)")
    p.add_argument("--examples", type=int, default=5, help="example pairs per rule in examples.md (30 for the audit)")
    p.add_argument("--rules-v6", action="store_true",
                   help="v6 rule set: D = D-SWE (submit after an unverified edit, S2 classifier), P = read-only repeat without "
                        "change; no E; priority V > R > D-SWE > P")
    p.add_argument("--p-teacher-fallback", action="store_true",
                   help="rule P (v6): the teacher's recorded turn as chosen when every valid sample repeats (see rules.py)")
    p.add_argument("--r-fallback-degenerate-only", action="store_true",
                   help="rule R: a different-action chosen side only for a degenerate rejected turn (unclosed or repetitive; v6)")
    p.add_argument("--drop-rules", default="", help="comma-separated rules to leave out (a rule that failed its audit)")
    p.add_argument("--checkpoint", default=None, help="the sampled model's merged checkpoint (recorded in metadata.json)")
    p.add_argument("--states-from", default=None, metavar="RUN_ID",
                   help="copy WORK/RUN_ID/{states.jsonl, states_report.json} (read only) and sample them (v6: the frozen v5 states)")
    p.add_argument("--reuse-from", default=None, metavar="RUN_ID",
                   help="copy WORK/RUN_ID/{states.jsonl, states_report.json, candidates.jsonl} into this run (read only) and "
                        "label them again; only states missing from its candidates are sampled")
    a = p.parse_args(argv)
    if len(a.model) not in (1, len(a.api_base)):
        p.error("--model: one name, or one per --api-base")
    a.endpoints = list(zip(a.api_base, a.model * len(a.api_base) if len(a.model) == 1 else a.model))
    from datetime import datetime
    a.deadlines = {}
    for d in a.endpoint_deadline:
        url, _, when = d.partition("=")
        if url not in a.api_base or not when:
            p.error(f"--endpoint-deadline {d!r}: URL must be one of --api-base, followed by =<UTC time>")
        a.deadlines[url] = datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp()
    if (a.reuse_from or a.states_from) and a.resume or (a.reuse_from and a.states_from):
        p.error("--reuse-from / --states-from start a new run: one of them, without --resume")
    if not (a.n_swe == "auto" or a.n_swe.isdigit()) or not (a.n_terminal == "all" or a.n_terminal.isdigit()):
        p.error("--n-swe is a count or auto; --n-terminal is a count or all")
    if a.n_swe == "auto" and a.n_terminal != "all" or a.samples < 2 or a.workers < 1 or not 0 < a.swe_share < 1:
        p.error("--n-swe auto needs --n-terminal all; --samples >= 2, --workers >= 1, 0 < --swe-share < 1")
    return a


def read_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text()
    if text.lstrip().startswith("["):
        return {str(x) for x in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------- states
def build_states(a: argparse.Namespace, run: RunDir, tok: Any, opts: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dev = json.loads(DEV.read_text())
    report: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    order = ("terminal", "swe") if a.n_swe == "auto" else ("swe", "terminal")
    for fam in order:
        if fam == "terminal":
            n_want = 10 ** 9 if a.n_terminal == "all" else int(a.n_terminal)
        else:
            n_terminal = sum(s["family"] == "terminal" for s in out)
            n_want = round(a.swe_share / (1 - a.swe_share) * n_terminal) if a.n_swe == "auto" else int(a.n_swe)
        if not n_want:
            continue
        got: list[dict[str, Any]] = []
        if fam == "swe":
            exclude = read_ids(a.exclude_swe) | set(dev["swe_task_ids"])
            wanted = st.swe_candidates(a.swe_run, exclude, a.seed)
            episodes = st.swe_episodes(a.swe_run, wanted[:max((6 if a.min_prompt_swe else 2) * n_want, 50)])
            drops: dict[str, int] = {}
            supply = {"instances": len(wanted), "by_band": dict(Counter(w["band"] for w in wanted))}
            if a.swe_rollouts:
                ro, drops = st.swe_rollout_episodes(a.swe_rollouts, exclude, {r.lower() for r in dev["swe_repositories"]}, a.seed)
                supply["rollout_tasks"] = len(ro)
                n_ro = round(a.swe_rollout_share * n_want)
                for ep in ro:
                    if len(got) >= n_ro:
                        break
                    got += st.episode_states(ep, fam, ep["band"], tok, opts, a.seed,
                                             per_episode=min(a.per_episode_swe, n_ro - len(got)), min_prompt=a.min_prompt_swe)[0]
        else:
            exclude = read_ids(a.exclude_terminal) | set(dev["terminal_task_ids"])
            eps, drops = st.terminal_episodes(a.kimi, a.labels, exclude, a.seed, trials_per_task=a.terminal_trials_per_task,
                                              exclude_episodes=read_ids(a.exclude_terminal_episodes))
            episodes = iter(eps)
            supply = {"tasks": len(eps), "by_band": dict(Counter(e["band"] for e in eps))}
        skipped: Counter = Counter()
        for ep in episodes:
            if len(got) >= n_want:
                break
            per = a.per_episode_swe if fam == "swe" else a.per_episode_terminal
            states, why = st.episode_states(ep, fam, ep["band"], tok, opts, a.seed, per_episode=min(per, n_want - len(got)),
                                            min_prompt=a.min_prompt_swe if fam == "swe" else 0)
            if why:
                skipped[why] += 1
            got += states
        report[fam] = {"supply": supply, "excluded_ids": len(exclude), "load_drops": drops, "episodes_skipped": dict(skipped),
                       "states": len(got), "by_band": dict(Counter(s["band"] for s in got)),
                       "by_kind": dict(Counter(s["kind"] for s in got)),
                       "by_source": dict(Counter(s["source"].get("kind", "teacher") for s in got)),
                       "n_prompt_max": max((s["n_prompt"] for s in got), default=0)}
        run.log(f"[states] {fam}: {report[fam]}")
        out += got
    return family_order(out, a.swe_share), report


def family_order(states: list[dict[str, Any]], swe_share: float) -> list[dict[str, Any]]:
    """States in sampling order: the families merged so that every prefix holds about swe_share SWE states (a time cap
    then keeps the share); each family keeps its own order."""
    by = {f: [s for s in states if s["family"] == f] for f in ("swe", "terminal")}
    share = {"swe": swe_share, "terminal": 1 - swe_share}
    out, taken = [], {"swe": 0, "terminal": 0}
    while len(out) < len(states):
        live = [f for f in by if taken[f] < len(by[f])]
        f = min(live, key=lambda f: ((taken[f] + 1) / share[f], f))
        out.append(by[f][taken[f]])
        taken[f] += 1
    return out


def screen_states(a: argparse.Namespace, run: RunDir, states: list[dict[str, Any]], work: Path
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """run_screens.py p1 over the states' tasks (subprocess, logged); states of a dropped task are removed."""
    import shutil
    import subprocess

    ref: set[str] = set()                                  # the reference pools' task ids (stage-1 slices, dev)
    for sp in a.screen_pool:
        ref |= {r["task_id"] for r in read_jsonl(Path(sp.split("=", 1)[1]))}
    in_ref = {s["task_id"] for s in states} & ref          # run_screens refuses a task id in two pools: drop these first
    items = {s["task_id"]: st.screen_item(s) for s in states if s["task_id"] not in in_ref}
    items_path = work / "screen_items.jsonl"
    write_jsonl(items_path, list(items.values()))
    sdir = work / "screens"
    sdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(paths.MANIFESTS.glob("dev_*.json")):
        shutil.copy2(f, sdir / f.name)
    cmd = [sys.executable, str(ROOT / "scripts/data/run_screens.py"), "--pools", "", "--no-github", "--workers", str(a.workers),
           "--manifests-dir", str(sdir), "--report", str(sdir / "decontam_report.json"), "--pass-name", "p1_stage2_states",
           "--pool", f"stage2_states={items_path}", *[x for sp in a.screen_pool for x in ("--pool", sp)]]
    out_json = sdir / "screens_p1_stage2_states.json"
    if not out_json.exists():
        run.log(f"[screens] {' '.join(cmd)}")
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(ROOT / "src")}
        with (sdir / "p1_stage2_states.log").open("w") as lf:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0 or not out_json.exists():
            raise RuntimeError(f"run_screens failed (exit {proc.returncode}); see {sdir / 'p1_stage2_states.log'}")
    manifest = json.loads(out_json.read_text())
    dropped = {(k.split(":", 1)[1] if k.startswith("row:") else k): rules_ for k, rules_ in manifest.get("dropped", {}).items()}
    hit = {t: dropped[t] for t in items if t in dropped}
    hit.update({t: ["reference_pool_item"] for t in in_ref})
    groups: dict[str, set[str]] = defaultdict(set)
    with (Path(manifest["work_dir"]) / "marks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("dedup_group"):
                groups[m["dedup_group"]].add(m["id"])
    for members in groups.values():
        if members & ref:
            for t in members & items.keys():
                hit.setdefault(t, []).append("dup_of_reference_pool")
    kept = [s for s in states if s["task_id"] not in hit]
    rep = {"tasks": len(items) + len(in_ref), "reference_pool_items": len(in_ref), "tasks_dropped": len(hit), "by_rule": dict(Counter(x for v in hit.values() for x in v)),
           "states_dropped": len(states) - len(kept)}
    run.log(f"[screens] {rep}")
    return kept, rep


# ------------------------------------------------------------------------------------------------------- sample
def sample_states(a: argparse.Namespace, run: RunDir, states: list[dict[str, Any]], tok: Any, opts: Any, path: Path) -> dict[str, Any]:
    done = {r["state_id"] for r in read_jsonl(path)}
    todo = [s for s in states if s["state_id"] not in done]
    run.log(f"[sample] {len(todo)} states to sample ({len(done)} already done), {a.samples} each, {a.workers} workers")
    if not todo:
        return {"sampled": 0, "errors": 0, "seconds": 0.0, "endpoints": None}
    from datetime import datetime

    eps = [Endpoint(url, model) for url, model in a.endpoints]
    health = [e.health() for e in eps]
    bad = [h for h in health if not h["ok"]]
    if bad:
        raise RuntimeError(f"endpoints not serving their model: {bad}")
    joined: dict[str, str] = {e.api_base: utc_now() for e in eps}
    join_t = [time.time() for _ in eps]
    last_poll = 0.0

    def poll_joiners() -> None:
        """Add the --endpoints-file entries that are not in use yet and answer /models (at most every 30 s)."""
        nonlocal last_poll
        if a.endpoints_file is None or time.time() - last_poll < 30 or not a.endpoints_file.exists():
            return
        last_poll = time.time()
        try:
            entries = json.loads(a.endpoints_file.read_text())
        except (OSError, ValueError) as exc:
            run.error(f"[sample] {a.endpoints_file}: {exc}")
            return
        for ent in entries:
            if ent["api_base"] in joined or len(eps) >= a.max_endpoints:
                continue
            e = Endpoint(ent["api_base"], ent["model"])
            h = e.health()
            if not h["ok"]:
                continue
            ends = [t for t in (datetime.fromisoformat(ent["deadline"].replace("Z", "+00:00")).timestamp() if ent.get("deadline")
                                else None, time.time() + ent["cap_s"] if ent.get("cap_s") else None) if t is not None]
            eps.append(e)
            health.append(h)
            busy.append(0)
            deadline.append(min(ends) if ends else None)
            joined[e.api_base] = utc_now()
            join_t.append(time.time())
            run.log(f"[sample] endpoint joined: {e.api_base} ({e.model_name}), deadline "
                    f"{datetime.fromtimestamp(deadline[-1]).isoformat() if deadline[-1] else 'none'}")

    def body(s: dict[str, Any], model: str) -> dict[str, Any]:     # built at submission: token-id prompts are large
        ids = st.prompt_ids(s["state_id"], s["prompt"], s["tools"], tok, opts)
        return cand_mod.request_body(model, ids, s["family"], a.samples, st.sample_budget(s, opts.max_length), s["state_id"])

    t0, n_err, n_done, n_unsent, n_requeued = time.time(), 0, 0, 0, 0
    queue = list(todo)
    by_id = {s["state_id"]: s for s in todo}
    n_todo = len(todo)
    per_ep: Counter = Counter()
    deadline = [a.deadlines.get(e.api_base) for e in eps]

    def open_slot() -> int | None:
        """The open endpoint with the most free slots (an endpoint closes --deadline-margin-s before its deadline)."""
        now = time.time()
        free = [(busy[k], k) for k in range(len(eps)) if busy[k] < a.workers
                and (deadline[k] is None or now < deadline[k] - a.deadline_margin_s)]
        return min(free)[1] if free else None

    def all_closed() -> bool:
        now = time.time()
        return all(d is not None and now >= d - a.deadline_margin_s for d in deadline)

    busy = [0] * len(eps)
    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers * a.max_endpoints) as pool:
        futs: dict[Any, tuple[str, int]] = {}
        while queue or futs:
            poll_joiners()
            while queue:
                if a.time_cap_s and time.time() - t0 > a.time_cap_s or all_closed():
                    n_unsent = len(queue)
                    run.log(f"[sample] {'time cap' if not all_closed() else 'every endpoint past its deadline'}: "
                            f"{n_unsent} states not sent")
                    queue = []
                    break
                k = open_slot()
                if k is None:
                    break
                s = queue.pop(0)
                timeout = 3600.0 if deadline[k] is None else max(1.0, deadline[k] - time.time())
                retries = 3 if deadline[k] is None else 0
                futs[pool.submit(cand_mod.complete, eps[k], body(s, eps[k].model_name), timeout=timeout, retries=retries)] = (
                    s["state_id"], k)
                busy[k] += 1
            if not futs:
                if queue:                                  # every open endpoint is past its margin: wait is pointless
                    continue
                break
            done_now, _ = wait(futs, timeout=30, return_when=FIRST_COMPLETED)
            if not done_now:
                continue
            fut = done_now.pop()
            sid, k = futs.pop(fut)
            busy[k] -= 1
            try:
                resp = fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded per state; the state is retried on --resume
                if deadline[k] is not None and any(d is None for d in deadline):
                    queue.insert(0, by_id[sid])            # back to the queue for an endpoint without a deadline
                    n_requeued += 1
                    run.log(f"[sample] {sid} requeued after {type(exc).__name__} on {eps[k].api_base}")
                    continue
                n_err += 1
                run.error(f"[sample] {sid}: {type(exc).__name__}: {exc}")
                continue
            choices = sorted(resp.get("choices") or [], key=lambda c: c.get("index", 0))
            per_ep[eps[k].api_base] += 1
            f.write(json.dumps({"state_id": sid, "endpoint": eps[k].api_base, "usage": resp.get("usage"),
                                "samples": [{"text": c.get("text") or "", "finish_reason": c.get("finish_reason")} for c in choices]},
                               ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 25 == 0 or not (queue or futs):
                el = time.time() - t0
                eta = time.time() + el / n_done * (n_todo - n_done) if n_done else None
                run.metric({"step": "sample", "states_done": n_done, "of": n_todo, "errors": n_err, "elapsed_s": round(el, 1)})
                run.log(f"[sample] {n_done}/{n_todo} states, {n_err} errors, {el:.0f}s")
                if a.progress_marker:
                    from datetime import datetime, timezone
                    a.progress_marker.write_text(json.dumps({
                        "producer": "scripts/data/build_stage2_pairs.py (v5 stage-2 pairs)", "utc": utc_now(), "run_id": run.run_id,
                        "states_done": n_done, "of": n_todo, "share": round(n_done / n_todo, 3), "errors": n_err,
                        "requeued": n_requeued, "by_endpoint": dict(per_ep), "elapsed_s": round(el),
                        "states_per_min": round(60 * n_done / el, 2) if el else None,
                        "eta_utc": datetime.fromtimestamp(eta, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if eta else None,
                        "notes": "ETA at the current rate of the endpoints in use; an endpoint joining or leaving moves it"},
                        indent=1) + "\n")
    rep = {"sampled": n_done, "errors": n_err, "not_sent_time_cap": n_unsent, "requeued": n_requeued,
           "seconds": round(time.time() - t0, 1), "by_endpoint": dict(per_ep), "endpoints": health, "joined": joined,
           "endpoint_seconds": {e.api_base: round(min(time.time(), deadline[k] or float("inf")) - join_t[k], 1)
                                for k, e in enumerate(eps)},
           "deadlines": {e.api_base: deadline[k] for k, e in enumerate(eps) if deadline[k] is not None}}
    if a.sampling_done_marker:
        a.sampling_done_marker.write_text(json.dumps({
            "producer": "scripts/data/build_stage2_pairs.py (v5 stage-2 pairs)", "utc": utc_now(), "artifacts": {},
            "run_id": run.run_id, "sampled_states": n_done, "errors": n_err, "not_sent_time_cap": n_unsent,
            "by_endpoint": dict(per_ep), "seconds": rep["seconds"],
            "notes": "sampling finished; the endpoints are free (labelling and the audit run on apex-cpu)"}, indent=1) + "\n")
        run.log(f"[sample] wrote {a.sampling_done_marker}")
    return rep


# ------------------------------------------------------------------------------------------------------- label
def label_and_pair(a: argparse.Namespace, run: RunDir, states: list[dict[str, Any]], cand_rows: list[dict[str, Any]],
                   tok: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    from apex_data.efficiency import qwen_token_counts
    from apex_data.render.filters import FilterConfig, repeated_window
    from apex_train.data_format import DataFormatError
    from apex_train.stage2 import PreferencePair, render_pair

    train_opts = st.dpo_options()
    rep_cfg = FilterConfig()                               # the stage-1 repetition-loop gate: a 50-token window x 3
    dropped_rules = {r.strip() for r in a.drop_rules.split(",") if r.strip()}
    by_state = {r["state_id"]: r for r in cand_rows}
    labels: dict[str, Counter] = defaultdict(Counter)
    yield_: dict[str, Counter] = defaultdict(Counter)
    drops: Counter = Counter()
    pairs: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for s in states:
        row = by_state.get(s["state_id"])
        if row is None:
            continue
        fam = s["family"]
        cands = []
        for k, smp in enumerate(row["samples"]):
            parsed = cand_mod.parse_completion(smp["text"], smp["finish_reason"], fam)
            lab = rules.label_turn(s, parsed, qwen_token_counts)
            lab["repetitive"] = bool((not lab["closed"] or lab["thinking"] > rules.ceiling(s)) and repeated_window(
                tok.encode(parsed["reasoning"], add_special_tokens=False), rep_cfg.repetition_window, rep_cfg.repetition_repeats))
            lab["loop"] = not lab["closed"] and lab["repetitive"]       # a degenerate loop: never closed, repetitive
            cands.append({"k": k, "message": parsed["message"], "labels": lab})
            c = labels[fam]
            c["candidates"] += 1
            c["valid"] += lab["valid"]
            c["clean"] += lab["clean"]
            c["over_ceiling"] += lab["valid"] and lab["thinking"] > rules.ceiling(s)
            c["cut_off"] += not lab["finished"] and not lab["closed"]
            c["unclosed_stopped"] += lab["finished"] and not lab["closed"]
            c["loop"] += lab["loop"]
            c["premature"] += lab["premature"]
            c["repeat"] += lab["repeat"]
            if not lab["valid"]:
                c[f"invalid:{lab['invalid_reason']}"] += 1
        teacher = None
        if s["teacher"] is not None:
            tparsed = cand_mod.teacher_parsed(s["teacher"], fam)
            teacher = {"message": tparsed["message"], "labels": rules.label_turn(s, tparsed, qwen_token_counts)}
            labels[fam]["teacher_clean"] += teacher["labels"]["clean"]
        labels[fam]["states"] += 1
        picked = rules.pick_pairs(s, cands, teacher, a.seed, skip=dropped_rules,
                                  degenerate_fallback_only=a.r_fallback_degenerate_only, rules_v6=a.rules_v6,
                                  p_teacher_fallback=a.p_teacher_fallback)
        yield_[fam]["states_with_pair"] += bool(picked)
        seen_rule: Counter = Counter()
        for p in picked:
            ch = teacher if p["chosen"] == "teacher" else cands[p["chosen"]]
            rej = cands[p["rejected"]]
            seen_rule[p["rule"]] += 1                    # R may make several pairs per state: #R, #R2, #R3
            pid = f"{s['state_id']}#{p['rule']}" + (str(seen_rule[p["rule"]]) if seen_rule[p["rule"]] > 1 else "")
            meta = {"id": pid, "rule": p["rule"], "family": fam, "band": s["band"], "state_id": s["state_id"], "kind": s["kind"],
                    "task_id": s["task_id"], "turn": s["turn"], "chosen_source": "teacher" if p["chosen"] == "teacher" else "candidate",
                    "chosen_thinking": ch["labels"]["thinking"], "rejected_thinking": rej["labels"]["thinking"],
                    "rejected_closed": rej["labels"]["closed"], "rejected_loop": rej["labels"]["loop"],
                    "rejected_repetitive": rej["labels"].get("repetitive", False),
                    "detail": p["detail"], "model": a.endpoints[0][1]}
            obj = {"id": pid, "prompt": s["prompt"], "tools": s["tools"], "chosen": ch["message"], "rejected": rej["message"],
                   "metadata": meta}
            try:
                rendered = render_pair(PreferencePair.from_dict(obj, default_id=pid, thinking=train_opts.thinking), tok, train_opts)
                meta.update(n_prompt=rendered["n_prompt"], n_chosen=rendered["n_chosen"], n_rejected=rendered["n_rejected"])
            except DataFormatError as exc:
                drops[f"{p['rule']}:{exc.reason}"] += 1
                continue
            yield_[fam][p["rule"]] += 1
            yield_[fam]["teacher_fallback"] += p["chosen"] == "teacher"
            yield_[fam][f"{p['rule']}_rejected_unclosed"] += not rej["labels"]["closed"]
            yield_[fam][f"{p['rule']}_rejected_loop"] += rej["labels"]["loop"]
            pairs.append({**obj, "family": fam, "rule": p["rule"]})
            examples.append({"pair": obj, "chosen_labels": ch["labels"], "rejected_labels": rej["labels"]})
    dup = [k for k, n in Counter(p["id"] for p in pairs).items() if n > 1]
    if dup:
        raise RuntimeError(f"pair ids not unique: {dup[:3]}")
    kept, bal = rules.balance(pairs, a.seed, rules.FAMILY_SHARES if a.family_balance else None)
    out = [{k: v for k, v in p.items() if k not in ("family", "rule")} for p in kept]
    n_states = Counter(s["family"] for s in states if s["state_id"] in by_state)
    n_states = {f: n for f, n in n_states.items() if n}
    loops = {r: {"pairs": sum(p["metadata"]["rule"] == r for p in out),
                 "rejected_unclosed": sum(p["metadata"]["rule"] == r and not p["metadata"]["rejected_closed"] for p in out),
                 "rejected_loop": sum(p["metadata"]["rule"] == r and p["metadata"]["rejected_loop"] for p in out)}
             for r in ("V", "R")}
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in out:
        cells[f"{p['metadata']['family']}/{p['metadata']['rule']}"].append(p["metadata"])
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0.0  # noqa: E731
    tokens = {c: {"pairs": len(ms), "mean_prompt": mean([m["n_prompt"] for m in ms]),
                  "mean_chosen": mean([m["n_chosen"] for m in ms]), "mean_rejected": mean([m["n_rejected"] for m in ms]),
                  "mean_row": mean([m["n_prompt"] + max(m["n_chosen"], m["n_rejected"]) for m in ms]),
                  "mean_two_sequences": mean([2 * m["n_prompt"] + m["n_chosen"] + m["n_rejected"] for m in ms])}
              for c, ms in sorted(cells.items())}
    r_detail = Counter(f"{p['metadata']['family']}/{p['metadata']['detail']}" for p in out if p["metadata"]["rule"] == "R")
    rep = {"labels": {f: dict(c) for f, c in labels.items()}, "yield": {f: dict(c) for f, c in yield_.items()},
           "loops_in_V_R": loops, "R_by_family_and_chosen": dict(sorted(r_detail.items())), "tokens_by_family_rule": tokens,
           "pairs_before_balance": len(pairs), "pairs": len(out), "pairs_per_state": {
               f: round(sum(v for k, v in yield_[f].items() if k in rules.RULES) / n_states[f], 3) for f in n_states},
           "by_rule": dict(Counter(p["metadata"]["rule"] for p in out)),
           "by_family": dict(Counter(p["metadata"]["family"] for p in out)), "validator_drops": dict(drops), "balance": bal}
    run.log(f"[label] {json.dumps(rep, default=str)[:1500]}")
    return out, rep, examples


def _short(m: dict[str, Any] | None, family: str, n: int = 400, n_action: int = 1200) -> str:
    if not m:
        return "(none)"
    calls = m.get("tool_calls") or []
    action = " ;; ".join(c["function"]["arguments"]["command"] for c in calls) if calls else (m.get("content") or "")
    if family == "terminal" and not calls:
        from apex_data.efficiency import terminal_command
        keys, done, ok = terminal_command(action)
        action = f"keystrokes {keys!r} task_complete={done}" if ok else action
    r = (m.get("reasoning_content") or "").replace("\n", " ")
    return f"reasoning ({len(r)} chars): {r[:n]}{' ...' if len(r) > n else ''}\n    action: {action[:n_action]!r}"


def _history_tail(prompt: list[dict[str, Any]], family: str, k: int = 3, n: int = 300) -> list[str]:
    """The last k actions before the state with the head of their observations (the context rules D and P judge)."""
    from apex_data.efficiency import actions

    return [f"    - turn {x.turn}: {x.command[:n]!r} -> {x.observation[:n]!r}" for x in actions(prompt, family)[-k:]]


def write_examples(path: Path, examples: list[dict[str, Any]], seed: int, per_rule: int) -> None:
    """examples.md: per rule, per_rule seeded pairs with their labels and the state's last actions (the audit file)."""
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in examples:
        by_rule[e["pair"]["metadata"]["rule"]].append(e)
    lines = [f"# Stage-2 pair examples ({per_rule} per rule, seeded)", ""]
    for rule in rules.RULES:
        xs = by_rule.get(rule, [])
        random.Random(f"{seed}:examples:{rule}").shuffle(xs)
        lines.append(f"## Rule {rule} ({len(xs)} pairs)")
        for e in xs[:per_rule]:
            m, fam = e["pair"]["metadata"], e["pair"]["metadata"]["family"]
            cl, rl = e["chosen_labels"], e["rejected_labels"]
            lines += [f"- `{m['id']}` {fam}/{m['band']} turn {m['turn']} ({m['kind']}), chosen from {m['chosen_source']}, "
                      f"thinking {m['chosen_thinking']} vs {m['rejected_thinking']}; {m['detail']}",
                      "  - last actions before the state:", *_history_tail(e["pair"]["prompt"], fam),
                      f"  - chosen: {_short(e['pair']['chosen'], fam)}",
                      f"    labels: completion {cl['completion']}, verifies {cl['verifies']}",
                      f"  - rejected: {_short(e['pair']['rejected'], fam)}",
                      f"    labels: valid {rl['valid']} ({rl['invalid_reason']}), completion {rl['completion']}, "
                      f"premature {rl['premature']}, repeat {rl['repeat']}, verifies {rl['verifies']}"]
        lines.append("")
    path.write_text("\n".join(lines))


# ------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    from apex_train.data_format import load_tokenizer

    run_id = a.run_id or new_run_id("data_stage2_pairs")
    work = paths.WORK / run_id
    out_dir = a.out_dir or work
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        start = utc_now()
        if not a.resume:
            from apex_data.rollout.endpoint import lock_inference
            sampling = {fam: {**lock_inference(fam), "n": a.samples, "seed": "rollout.endpoint.sample_seed(state_id, 0)",
                              "max_tokens": f"min(lock max_tokens_per_turn, dpo.max_length - prompt - {st.RETOKENIZE_MARGIN})",
                              "stop_token_ids": "[<|im_end|>]", "skip_special_tokens": False}
                        for fam in ("swe", "terminal")}
            run.write_metadata(run_id=run_id, step="data_build_stage2_pairs", command=command_line(), code=git_info(),
                               config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                               data_version={"states_from": a.states_from, "swe_run": a.swe_run.name, "kimi": str(a.kimi),
                                             "labels": str(a.labels)},
                               model_revision=",".join(sorted(set(a.model))), checkpoint=a.checkpoint, sampling=sampling,
                               hardware=hardware_info(), seed=a.seed, start_time=start, status="running")
        work.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        tok = load_tokenizer(a.tokenizer)
        opts = st.dpo_options(on_too_long="drop")
        states_path, cand_path = work / "states.jsonl", work / "candidates.jsonl"
        if a.reuse_from:
            import shutil
            src = paths.WORK / a.reuse_from
            for name in ("states.jsonl", "states_report.json"):
                shutil.copyfile(src / name, work / name)
            text = (src / "candidates.jsonl").read_text(encoding="utf-8")   # complete lines only: the source may be growing
            (work / "candidates.jsonl").write_text(text[:text.rfind("\n") + 1], encoding="utf-8")
            run.log(f"[reuse] states and candidates copied from {src}")
        if a.states_from:
            import shutil
            src = paths.WORK / a.states_from
            for name in ("states.jsonl", "states_report.json"):
                shutil.copyfile(src / name, work / name)
            run.log(f"[states-from] states copied from {src}; sampling them anew")
        if (a.resume or a.reuse_from or a.states_from) and states_path.exists():
            states = read_jsonl(states_path)
            states_rep = json.loads((work / "states_report.json").read_text())
            run.log(f"[states] reusing {len(states)} states from {states_path}")
        else:
            states, states_rep = build_states(a, run, tok, opts)
            if not a.no_screens:
                states, states_rep["screens"] = screen_states(a, run, states, work)
            write_jsonl(states_path, states)
            (work / "states_report.json").write_text(json.dumps(states_rep, indent=1) + "\n")
        if a.states_only:
            run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
                                 ("gpu_hours", 0, "h", "states only: no sampling yet"), ("cost_usd", 0, "USD", "CPU only")])
            run.write_metadata(status="states_built", counts={"states": len(states)})
            run.finalize()
            run.log(f"done: {len(states)} states -> {states_path} (--states-only; --resume samples them)")
            return 0
        sample_rep = ({"sampled": 0, "errors": 0, "seconds": 0.0, "endpoints": None, "note": "--no-sample"} if a.no_sample
                      else sample_states(a, run, states, tok, opts, cand_path))
        pairs, label_rep, examples = label_and_pair(a, run, states, read_jsonl(cand_path), tok)
        write_jsonl(out_dir / "pairs.jsonl", pairs)
        write_examples(out_dir / "examples.md", examples, a.seed, a.examples)
        report = {"run_id": run_id, "created": utc_now(), "endpoints": a.endpoints, "samples": a.samples,
                  "states": states_rep, "sample": sample_rep, **label_rep,
                  "probe_gate": {"rule": ">= 1 pair per state on average (data plan 4.4)",
                                 "pairs_per_state": label_rep["pairs_per_state"]}}
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        if sample_rep["errors"]:
            rc = 1
        gpu_h = round(sum((sample_rep.get("endpoint_seconds") or {}).values()) / 3600, 3)
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("sampling_sec", sample_rep["seconds"], "s", f"requests to the endpoints {sorted(sample_rep.get('joined') or {})}"),
            ("peak_gpu_memory_gb", "n/a", "GB", "the endpoint's server; not measured here"),
            ("gpu_hours", gpu_h, "h", "per endpoint (one GPU each): from joining to the end of sampling or its deadline"),
            ("cost_usd", round(gpu_h * GPU_RATE_USD, 2), "USD", f"gpu_hours x {GPU_RATE_USD} USD per GPU hour"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if rc == 0 else "failed",
                           outputs={"pairs": str(out_dir / "pairs.jsonl"), "report": str(out_dir / "report.json")},
                           counts={"states": len(states), "pairs": len(pairs), "by_rule": label_rep["by_rule"]})
        run.finalize()
        run.log(f"done: {len(pairs)} pairs -> {out_dir}")
    return rc


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
