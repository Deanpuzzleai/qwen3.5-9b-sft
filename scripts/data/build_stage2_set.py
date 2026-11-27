#!/usr/bin/env python
"""Freeze the stage-2 set (docs/DATA_PLAN.md section 11, milestone M6.2): arm (a) `positives` (verified positives), arm (b)
`control` (the loss-token-matched unfiltered control) and the retention replay rows both arms share, in the files and
manifest configs/stage2.yaml and apex_train.stage2 read.

Usage (from the repository root):
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_stage2_set.py \
        --episodes WORK/<rollout_run>/episodes.jsonl --verdicts WORK/<grade_run>/verdicts.jsonl \
        [--tasks-file data/pools/swesmith/tasks.jsonl --tasks-file data/pools/tgr_env/tasks.jsonl] \
        [--replay-reserve data/pools/retention/replay_reserve.parquet] [--policy sft] [--cutoff ISO] \
        [--max-row-tokens 65536] [--max-turn-tokens 8192] [--terminal-turns-per-episode 6 --terminal-turn-roles] \
        [--positives-per-task 2] [--replay-share 0.20] [--effective-batch 16] [--seed 42] \
        [--out-dir data/processed/stage2] [--manifest data/manifests/stage2_mixture.json] [--run-id ID]

Selection (seeded with --seed; every choice is recorded):
  graded      episodes of --round/--policy with a verdict whose graded_at <= --cutoff (later verdicts are "late": recorded,
              never added); every episode without a verdict is listed as ungraded.
  positives   r_pass = 1, tamper-clean, every render gate of apex_data.render.filters (parse-valid with the real parsers,
              explicit submit / task_complete handshake, no format-error turn, no finish_reason length, reasoning on every
              turn, step caps 50 / 40, row cap --max-row-tokens), then <= --positives-per-task episodes per task drawn at
              random. Rendered as in stage 1: SWE episode rows, terminal per-turn rows with two trained turns (the
              latest fitting turn and a seeded random earlier one).
  terminal    --terminal-turns-per-episode N trains N turns per terminal episode (both arms). With --terminal-turn-roles
  turns       the turns are role-chosen (apex_data.render.turn_roles, the same rule for both arms): (1)
              verify_before_complete, the assistant turn right before the first task_complete turn; (2) latest, the
              latest fitting turn (the hand-in turn, as in stage 1); (3) a round robin over after_error (the turn
              answering an observation with an error), long_job (background job, polling or a long wait, or writing a
              task-named path in the first half of the episode) and recovery (a control key, or a screen stuck at a
              continuation/REPL prompt or a pager): each role's latest turn in that order, then each role's next
              latest; (4) random, the seeded draw over the remaining turns. Turns that are not candidates (confirmation
              replies) or do not fit the row cap or --max-turn-tokens are skipped. Every terminal row's provenance gets
              turn_role; the manifest's `roles` has the rule text and the rows and loss tokens per role and arm.
  quality     --quality (revision 4.3.16; apex_data.render.quality, the same rule for both arms, after the render gates):
              an episode whose agent issued the same command and got the same observation back >= 3 turns in a row is
              dropped from both arms (gate quality_loop). Per family a process check: SWE = tests run after the last edit
              (stage 1's apex_data.teacher.verify rule) and no added non-test junk file in the final patch; terminal = a
              verification command (running the deliverable, a script or tests, or inspecting output files) in the first
              task_complete turn or the three turns before it. When >= 70 percent of the family's positives pass, the
              check is a hard gate in both arms (gate quality_<family>); otherwise it is a preference: the per-task
              positive pick takes passing episodes first and the control's loss-token draw takes passing episodes first
              (each group in seeded random order). Counts per check, family and arm are in the manifest's `quality`;
              every rollout row's provenance gets quality_pass.
  control     the same gates with the verdict gate skipped (filter_episodes skip_gates=("verdict",): format-valid rollouts,
              failures included, tamper flags still drop), no per-task cap; episodes drawn in seeded random order until
              their rollout loss tokens (sum of n_assistant_tokens) match the positives arm's per family (--match-by source:
              per family and task source, e.g. terminal/tgr_env) within --match-tolerance
              (an episode that would overshoot the upper bound is skipped).
  replay      rows of the replay reserve (WP5; disjoint from the SFT rows) taken until they carry --replay-share of the
              stage-2 loss tokens: replay = share / (1 - share) x the positives' rollout loss tokens. A reserve with a
              replay_rank column (the convergence replay) is taken in rank order, a prefix that keeps the reserve's cell
              shares (near the target a row that would overshoot the tolerance is skipped); otherwise rows are drawn in
              seeded random order. The order used is manifest replay.order ("rank" / "seeded_random"). The same rows go
              with both arms.
  fixed       --fixed-rows PATH --fixed-share F (revision 4.3.16; e.g. the terminal coverage slice at 0.15): pre-rendered
              data_format rows shared by both arms like the replay, taken in --fixed-rank-column order (a prefix) until
              they carry F of the stage-2 loss tokens: with replay share R, total = positives' rollout loss tokens /
              (1 - R - F), replay = R x total, fixed = F x total. A row above --max-row-tokens or with a trained span above
              --max-turn-tokens is skipped and counted. Written to <out-dir>/fixed_rows.parquet; the trainer takes it as a
              second shared file: filtered_sft.replay_rows = [replay_rows.parquet, fixed_rows.parquet] (manifest
              arms.<arm>.shared_rows_files).
  steps       planned optimizer steps = ceil((rollout rows + replay rows) / --effective-batch) per arm; the control trains the
              positives' number of steps (apex_train.stage2 --arm control --match-steps-of logs/<positives run_id>),
              recorded as matched_optimizer_steps.

Inputs : episodes.jsonl (run_rollouts.py), verdicts.jsonl (grade_episodes.py), the pool task files, the replay reserve
         (data_format rows; extra provenance columns are carried into the provenance file), the Qwen3.5-9B tokenizer
         (--tokenizer, default apex_data.paths.BASE_MODEL_DIR).
Outputs (refused when the manifest or a rows file already exists: the set is frozen once): <out-dir>/positives_rows.parquet
         (arm a), control_rows.parquet (arm b), replay_rows.parquet (shared replay rows, a
         separate file): apex_train.data_format rows = configs/stage2.yaml filtered_sft.arms.{positives,control}.rows and
         filtered_sft.replay_rows; training: scripts/train_stage2.sh --arm positives --rows <positives_rows> --replay-rows
         <replay_rows>, then --arm control --rows <control_rows> --replay-rows <replay_rows> --match-steps-of logs/<positives run>.
         <out-dir>/rows_provenance.jsonl (one line per row: file, row_key, schema.PROVENANCE_FIELDS, r_pass, tokens),
         <out-dir>/dropped.jsonl (every filtered episode with arm, gate and reason), <out-dir>/selection.json (episode ids per
         arm, late and ungraded ids, replay row keys).
         --manifest (data/manifests/stage2_mixture.json = filtered_sft.manifest): per rows file path, sha256, rows, loss tokens
         and planned optimizer steps; per arm its rows and replay files with sha256, rows, loss tokens, replay share and planned
         steps (control: matched_optimizer_steps); inputs with sha256, counts, pass@k per pool, survival tables, per-task cap
         effects, the matched-token check.
         logs/<run_id>/ with the five run files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apex_data import paths  # noqa: E402
from apex_data.render import quality, turn_roles  # noqa: E402
from apex_data.render.filters import FilterConfig, Rendered, filter_episodes  # noqa: E402
from apex_data.rollout.endpoint import FAMILY_OF_SCAFFOLD, STAGE2_CAPS  # noqa: E402
from apex_data.rollout.scheduler import parse_time  # noqa: E402
from apex_data.schema import PROVENANCE_FIELDS, SchemaError, read_jsonl  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402
from apex_train.data_format import ROW_COLUMNS, write_rows  # noqa: E402
from grade_episodes import pass_at_k  # noqa: E402

DEFAULT_TASK_FILES = (paths.POOLS / "swesmith" / "tasks.jsonl", paths.POOLS / "tgr_env" / "tasks.jsonl")
DEFAULT_REPLAY = paths.POOLS / "retention" / "replay_reserve.parquet"
POOL_HF_REPO = {"tgr_env": "ai-and/termgrade-environments"}
ARMS = ("positives", "control")                       # configs/stage2.yaml filtered_sft.arms
ROWS_FILES = {"positives": "positives_rows.parquet", "control": "control_rows.parquet", "replay": "replay_rows.parquet"}
FIXED_FILE = "fixed_rows.parquet"                      # --fixed-rows: a second rows file shared by both arms
POLICY_GENERATOR = {"base": "Qwen3.5-9B", "sft": "Qwen3.5-9B + SFT adapter", "stage2": "Qwen3.5-9B + stage-2 adapter"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, action="append", required=True, help="episodes.jsonl (repeatable)")
    p.add_argument("--verdicts", type=Path, action="append", required=True, help="verdicts.jsonl (repeatable)")
    p.add_argument("--tasks-file", type=Path, action="append", default=None, help="pool task JSONL (repeatable; default both pools)")
    p.add_argument("--replay-reserve", type=Path, default=DEFAULT_REPLAY, help="WP5 replay reserve rows (parquet)")
    p.add_argument("--allow-no-replay", action="store_true", help="SMOKE TESTS ONLY: build without a replay reserve (recorded)")
    p.add_argument("--round", dest="round_", default="stage2", help="episode round to use")
    p.add_argument("--policy", default="sft", help="episode policy to use")
    p.add_argument("--cutoff", default=None, help="ISO time (UTC): verdicts graded later are recorded as late, never added")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR, help="directory with the Qwen3.5-9B tokenizer")
    p.add_argument("--max-row-tokens", type=int, default=65_536, help="row cap = configs/stage2.yaml filtered_sft.max_length (M4.1)")
    p.add_argument("--max-turn-tokens", type=int, default=None, help="trained tokens per agent turn (render FilterConfig."
                   "max_turn_completion_tokens; revision 4.3.14: 8192): SWE episodes with a longer turn drop as turn_too_long, "
                   "terminal per-turn picks skip longer turns; default no limit")
    p.add_argument("--terminal-turns-per-episode", type=int, default=None, help="trained turns per terminal episode, both "
                   "arms (default: the adapter's 2)")
    p.add_argument("--family-targets", default="",
                   help="family=N,... (revision 4.3.21): a family's positives above N (after the per-task cap) are trimmed to N in "
                        "seeded-hash order of the episode id (dropped gate 'family_target')")
    p.add_argument("--exclude-episodes", type=Path, default=None,
                   help="episode ids (one per line) left out of both arms, e.g. the rows a p2 screen flagged (dropped gate 'screen')")
    p.add_argument("--quality", action="store_true", help="revision 4.3.16 process-quality rule (apex_data.render.quality): "
                   "loop drop in both arms, then per family a hard gate (>= 70 percent of positives pass) or a preference")
    p.add_argument("--terminal-turn-roles", action="store_true", help="role-chosen terminal turns (apex_data.render.turn_roles; "
                   "rule above), both arms")
    p.add_argument("--positives-per-task", type=int, default=2)
    p.add_argument("--replay-share", type=float, default=0.20, help="replay share of stage-2 loss tokens")
    p.add_argument("--fixed-rows", type=Path, default=None, help="pre-rendered rows shared by both arms (e.g. the terminal coverage slice)")
    p.add_argument("--fixed-share", type=float, default=0.0, help="--fixed-rows share of stage-2 loss tokens")
    p.add_argument("--fixed-rank-column", default="coverage_rank", help="--fixed-rows are taken in this column's order")
    p.add_argument("--effective-batch", type=int, default=16, help="rows per optimizer step (configs/stage2.yaml)")
    p.add_argument("--match-by", choices=("family", "source"), default="family",
                   help="the control's loss-token match per family (default) or per (family, task source)")
    p.add_argument("--match-tolerance", type=float, default=0.005, help="relative tolerance of the control's loss-token match")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--out-dir", type=Path, default=paths.PROCESSED / "stage2")
    p.add_argument("--manifest", type=Path, default=paths.MANIFESTS / "stage2_mixture.json", help="= configs/stage2.yaml filtered_sft.manifest")
    p.add_argument("--run-id", default=None, help="default: data_stage2_set_<UTC timestamp>")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: logs/ under this root")
    a = p.parse_args(argv)
    a.tasks_file = a.tasks_file or [f for f in DEFAULT_TASK_FILES if f.exists()]
    a.cutoff_at = parse_time(a.cutoff)
    if not 0 <= a.replay_share < 1:
        p.error("--replay-share must be in [0, 1)")
    try:
        a.family_target = {k.strip(): int(v) for k, v in (x.split("=", 1) for x in a.family_targets.split(",") if x.strip())}
    except ValueError:
        p.error("--family-targets takes family=N,...")
    if (a.fixed_rows is None) != (a.fixed_share == 0):
        p.error("--fixed-rows and a positive --fixed-share go together")
    if not 0 <= a.fixed_share < 1 - a.replay_share:
        p.error("--replay-share + --fixed-share must be below 1")
    if a.terminal_turns_per_episode is not None and a.terminal_turns_per_episode < 1:
        p.error("--terminal-turns-per-episode must be a positive integer")
    return a


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_of(ep: dict[str, Any]) -> str:
    return FAMILY_OF_SCAFFOLD.get(ep.get("scaffold"), str(ep.get("scaffold")))


def with_provenance(task: dict[str, Any], policy: str, served: str | None) -> dict[str, Any]:
    """A copy of a pool record with extra.provenance for stage-2 rollout rows (filters.provenance_for reads it)."""
    extra = dict(task.get("extra") or {})
    prov = {"hf_repo": extra.get("hf_repo") or POOL_HF_REPO.get(task.get("source")),
            "generator": f"{POLICY_GENERATOR.get(policy, policy)} ({served})" if served else POLICY_GENERATOR.get(policy, policy),
            "label_provenance": "verifier", "category": (task.get("stratum") or {}).get("category")}
    if task.get("family") == "swe":
        prov["patch_files"] = extra.get("patch_files")
    extra["provenance"] = {k: v for k, v in prov.items() if k in PROVENANCE_FIELDS}
    return {**task, "extra": extra}


def units_of(kept: list[Rendered]) -> dict[str, list[Rendered]]:
    """Rendered rows grouped by episode (the selection unit), in row order."""
    out: dict[str, list[Rendered]] = defaultdict(list)
    for r in kept:
        out[r.provenance["episode_id"]].append(r)
    return dict(out)


def longest_trained_span(row: dict[str, Any]) -> int:
    """Longest run of trained tokens (assistant_masks == 1) in a data_format row: its longest trained assistant turn."""
    best = run = 0
    for m in row["assistant_masks"]:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def loss_tokens(rows: list[dict[str, Any]] | list[Rendered]) -> int:
    return sum(int((r.row if isinstance(r, Rendered) else r)["n_assistant_tokens"]) for r in rows)


def cap_per_task(units: dict[str, list[Rendered]], k: int, rng: random.Random,
                 prefer: set[str] | frozenset[str] = frozenset()) -> tuple[list[str], dict[str, Any]]:
    """<= k episodes per task, drawn uniformly (sorted ids shuffled by rng; episodes in `prefer` first, each group keeping
    its shuffled order); returns (kept episode ids, cap report)."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for eid in sorted(units):
        by_task[units[eid][0].provenance["task_id"]].append(eid)
    kept: list[str] = []
    for tid in sorted(by_task):
        ids = list(by_task[tid])
        rng.shuffle(ids)
        ids.sort(key=lambda e: e not in prefer)          # stable: without preferred episodes the shuffled order is unchanged
        kept.extend(sorted(ids[:k]))
    before = Counter(len(v) for v in by_task.values())
    after = Counter(min(len(v), k) for v in by_task.values())
    return sorted(kept), {"episodes_before": len(units), "episodes_after": len(kept), "tasks": len(by_task),
                          "tasks_capped": sum(1 for v in by_task.values() if len(v) > k),
                          "episodes_per_task_before": dict(sorted(before.items())), "episodes_per_task_after": dict(sorted(after.items()))}


def apply_quality(pos_units: dict[str, list[Rendered]], pool_units: dict[str, list[Rendered]], episodes: dict[str, dict[str, Any]]
                  ) -> tuple[dict[str, Any], set[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The revision-4.3.16 process-quality rule on the rendered units of both arms (in place): loop drops, then per family a
    hard gate or a preference. Returns (manifest report, preferred episode ids, drop records, per-episode checks)."""
    ids = sorted(set(pos_units) | set(pool_units))
    checks = {eid: quality.episode_quality(episodes[eid], family_of(episodes[eid])) for eid in ids}
    drops: list[dict[str, Any]] = []

    def drop(eid: str, gate: str, reason: str) -> None:
        for arm, units in (("positives", pos_units), ("control", pool_units)):
            if eid in units:
                kind = units[eid][0].provenance.get("row_kind", "")
                del units[eid]
                drops.append({"arm": arm, "episode_id": eid, "task_id": episodes[eid]["task_id"], "row_kind": kind, "gate": gate,
                              "reason": reason, "detail": json.dumps({k: v for k, v in checks[eid].items() if k != "version"})})

    for eid in ids:
        if checks[eid]["loop_drop"]:
            drop(eid, "quality_loop", f"{checks[eid]['loop']} identical command + observation turns in a row")
    report: dict[str, Any] = {"version": quality.VERSION, "loop_repeats": quality.LOOP_REPEATS, "verify_window": quality.VERIFY_WINDOW,
                              "hard_share": quality.HARD_SHARE,
                              "loop_dropped": dict(Counter(d["arm"] for d in drops)), "families": {}}
    prefer: set[str] = set()
    for fam in sorted({family_of(episodes[e]) for e in ids}):
        pos = [e for e in pos_units if family_of(episodes[e]) == fam]
        pool = [e for e in pool_units if family_of(episodes[e]) == fam]
        share = sum(checks[e]["pass"] for e in pos) / len(pos) if pos else None
        mode = "hard" if share is not None and share >= quality.HARD_SHARE else "preference"
        sub = ("swe_tests_after_edit", "swe_clean_diff") if fam == "swe" else ("terminal_verify",)
        fam_rep: dict[str, Any] = {"mode": mode, "positives_pass_share": round(share, 4) if share is not None else None}
        for arm, eps in (("positives", pos), ("control", pool)):
            fam_rep[arm] = {"episodes": len(eps), "pass": sum(checks[e]["pass"] for e in eps),
                            **{f"fail_{c}": sum(not checks[e][c] for e in eps) for c in sub}}
        before = len(drops)
        for e in sorted(set(pos) | set(pool)):
            if not checks[e]["pass"]:
                if mode == "hard":
                    drop(e, f"quality_{fam}", "fails " + ", ".join(c for c in sub if not checks[e][c]))
            else:
                prefer.add(e)
        fam_rep["dropped"] = dict(Counter(d["arm"] for d in drops[before:]))
        report["families"][fam] = fam_rep
    return report, (prefer if any(f["mode"] == "preference" for f in report["families"].values()) else set()), drops, checks


def trim_to_targets(pos_ids: list[str], episodes: dict[str, dict[str, Any]], targets: dict[str, int], seed: int
                    ) -> tuple[list[str], list[str], dict[str, Any]]:
    """Revision 4.3.21: a family's positives above its target keep the first N in the order of sha256(f"{seed}:{episode_id}")
    (fixed in advance, blind to length, quality and outcome). Returns (kept ids, trimmed ids, report per family)."""
    kept: list[str] = []
    trimmed: list[str] = []
    report: dict[str, Any] = {}
    for fam in sorted({family_of(episodes[e]) for e in pos_ids} | set(targets)):
        ids = [e for e in pos_ids if family_of(episodes[e]) == fam]
        n = targets.get(fam)
        order = sorted(ids, key=lambda e: hashlib.sha256(f"{seed}:{e}".encode()).hexdigest())
        keep = order if n is None else order[:n]
        kept += keep
        trimmed += [e for e in order if e not in set(keep)]
        report[fam] = {"target": n, "before": len(ids), "after": len(keep), "shortfall": max(0, n - len(ids)) if n is not None else None,
                       "order": "sha256(seed:episode_id)"}
    return sorted(kept), sorted(trimmed), report


def fill_to_tokens(candidates: list[tuple[str, int]], target: int, tolerance: float, rng: random.Random | None) -> tuple[list[str], int]:
    """Candidates (id, loss tokens) in seeded random order (rng None: in the given order), added while they keep the total
    <= target x (1 + tolerance), until the total reaches target x (1 - tolerance). Returns (chosen ids, total)."""
    if target <= 0:
        return [], 0
    order = list(candidates)
    if rng is not None:
        order.sort()
        rng.shuffle(order)
    hi, lo = target * (1 + tolerance), target * (1 - tolerance)
    chosen, total = [], 0
    for cid, n in order:
        if total >= lo:
            break
        if total + n <= hi:
            chosen.append(cid)
            total += n
    return sorted(chosen), total


def file_record(path: Path, rows: list[dict[str, Any]], batch: int) -> dict[str, Any]:
    """Manifest entry of one rows file: path, sha256, rows, loss tokens (sum of n_assistant_tokens), tokens, planned steps."""
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows), "loss_tokens": loss_tokens(rows),
            "tokens": sum(int(r["n_tokens"]) for r in rows), "planned_optimizer_steps": math.ceil(len(rows) / batch) if rows else 0}


def load_replay(path: Path) -> list[dict[str, Any]]:
    """Replay reserve rows: the data_format columns plus whatever provenance columns the reserve carries. A replay_rank
    column must hold distinct integers."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = pq.read_table(path).to_pylist()
    missing = [c for c in ROW_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: replay reserve lacks data_format columns {missing}")
    if rows and "replay_rank" in rows[0]:
        ranks = [r["replay_rank"] for r in rows]
        if any(not isinstance(k, int) for k in ranks) or len(set(ranks)) != len(ranks):
            raise SystemExit(f"{path}: replay_rank must hold distinct integers")
    return rows


def role_report(rendered: list[Rendered]) -> dict[str, Any]:
    """Terminal rows and loss tokens per turn_role, and the trained turns per episode."""
    term = [r for r in rendered if "turn_role" in r.provenance]
    return {"rows": {k: n for k, n in Counter(r.provenance["turn_role"] for r in term).most_common()},
            "loss_tokens": {k: sum(int(r.row["n_assistant_tokens"]) for r in term if r.provenance["turn_role"] == k)
                            for k in turn_roles.ALL_ROLES if any(r.provenance["turn_role"] == k for r in term)},
            "episodes": len({r.provenance["episode_id"] for r in term}),
            "turns_per_episode": dict(sorted(Counter(Counter(r.provenance["episode_id"] for r in term).values()).items()))}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_stage2_set")
    existing = [str(f) for f in (a.manifest, *(a.out_dir / n for n in (*ROWS_FILES.values(), FIXED_FILE))) if f.exists()]
    if existing:   # the stage-2 set is frozen once (M6.2): never overwrite it; a rebuild goes to a new --out-dir / --manifest
        raise SystemExit(f"stage-2 set files exist ({', '.join(existing)}); the set is frozen once and never overwritten "
                         "(pass a new --out-dir and --manifest for another build)")
    with RunDir(run_id, root=a.out_root, log_name="train.log") as run:
        t0 = time.time()
        inputs = {"episodes": [{"path": str(p), "sha256": sha256_file(p)} for p in a.episodes],
                  "verdicts": [{"path": str(p), "sha256": sha256_file(p)} for p in a.verdicts],
                  "tasks": [{"path": str(p), "sha256": sha256_file(p)} for p in a.tasks_file],
                  "replay_reserve": {"path": str(a.replay_reserve), "sha256": sha256_file(a.replay_reserve)} if a.replay_reserve.exists() else None,
                  **({"fixed_rows": {"path": str(a.fixed_rows), "sha256": sha256_file(a.fixed_rows)}} if a.fixed_rows else {})}
        config = {"round": a.round_, "policy": a.policy, "cutoff": a.cutoff, "max_row_tokens": a.max_row_tokens, "max_turn_tokens": a.max_turn_tokens,
                  "terminal_turns_per_episode": a.terminal_turns_per_episode,
                  "terminal_turn_roles": turn_roles.VERSION if a.terminal_turn_roles else None,
                  "quality": quality.VERSION if a.quality else None,
                  **({"family_targets": a.family_target} if a.family_target else {}),
                  **({"exclude_episodes": {"path": str(a.exclude_episodes), "sha256": sha256_file(a.exclude_episodes)}}
                     if a.exclude_episodes else {}),
                  "positives_per_task": a.positives_per_task, "replay_share": a.replay_share, "effective_batch": a.effective_batch,
                  "match_tolerance": a.match_tolerance,
                  **({"match_by": a.match_by} if a.match_by != "family" else {}),
                  **({"fixed_share": a.fixed_share, "fixed_rank_column": a.fixed_rank_column} if a.fixed_rows else {}), "seed": a.seed, "max_steps": {f: c.max_steps for f, c in STAGE2_CAPS.items()},
                  "tokenizer": str(a.tokenizer)}
        run.write_metadata(run_id=run_id, kind="data_stage2_set", step="build_stage2_set", command=command_line(), code=git_info(),
                           hardware=hardware_info(), seed=a.seed, config=config, data_version=inputs, start_time=utc_now(), status="running")
        if inputs["replay_reserve"] is None and not a.allow_no_replay:
            run.error(f"replay reserve {a.replay_reserve} missing")
            raise SystemExit(f"replay reserve {a.replay_reserve} missing (WP5 builds it; --allow-no-replay only for smoke tests)")

        # --- inputs ---
        tasks = {t["task_id"]: t for f in a.tasks_file for t in read_jsonl(f)}
        episodes: dict[str, dict[str, Any]] = {}
        for f in a.episodes:
            for e in read_jsonl(f):
                if e.get("round") == a.round_ and e.get("policy") == a.policy:
                    episodes[e["episode_id"]] = e
        verdicts_all: dict[str, dict[str, Any]] = {}
        for f in a.verdicts:
            for v in read_jsonl(f):
                verdicts_all[v["episode_id"]] = v          # a later record of the same episode (re-grade) wins
        late = sorted(k for k, v in verdicts_all.items() if k in episodes and a.cutoff_at is not None
                      and (parse_time(v.get("graded_at")) or float("inf")) > a.cutoff_at)
        verdicts = {k: v for k, v in verdicts_all.items() if k in episodes and k not in set(late)}
        ungraded = sorted(set(episodes) - set(verdicts) - set(late))
        unknown_tasks = sorted({e["task_id"] for k, e in episodes.items() if k in verdicts and e["task_id"] not in tasks})
        if unknown_tasks:
            run.error(f"{len(unknown_tasks)} graded task(s) missing from the task files: {unknown_tasks[:5]}")
            raise SystemExit("task files do not cover the graded episodes")
        excluded: set[str] = set()
        if a.exclude_episodes is not None:
            excluded = {ln.strip() for ln in a.exclude_episodes.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
            run.log(f"--exclude-episodes: {len(excluded & set(verdicts))} of {len(excluded)} listed episodes are graded here and leave both arms")
        graded = [episodes[k] for k in sorted(verdicts) if k not in excluded]
        served = next((str((e.get("extra") or {}).get("model")) for e in graded if (e.get("extra") or {}).get("model")), None)
        tasks_p = {tid: with_provenance(tasks[tid], a.policy, served) for tid in {e["task_id"] for e in graded}}
        hist = pass_at_k(graded, verdicts, tasks)
        run.log(f"episodes {len(episodes)} ({a.round_}/{a.policy}); graded {len(graded)}; late {len(late)}; ungraded {len(ungraded)}; "
                f"pass@k {json.dumps({s: h['histogram'] for s, h in hist.items()})}")

        # --- render: arm (a) candidates (positives) and the format-valid pool (arm (b)), per family ---
        from apex_train.data_format import load_tokenizer  # noqa: PLC0415

        tok = load_tokenizer(a.tokenizer)
        survival: dict[str, Any] = {arm: {} for arm in ARMS}
        dropped: list[dict[str, Any]] = [{"arm": "both", "episode_id": k, "task_id": episodes[k]["task_id"], "row_kind": family_of(episodes[k]),
                                          "gate": "screen", "reason": f"listed in {a.exclude_episodes}", "detail": ""}
                                         for k in sorted(excluded & set(verdicts))]
        pos_units: dict[str, list[Rendered]] = {}
        pool_units: dict[str, list[Rendered]] = {}
        for fam in sorted({family_of(e) for e in graded}):
            cfg = FilterConfig(max_row_tokens=a.max_row_tokens, max_steps=STAGE2_CAPS[fam].max_steps, max_turn_completion_tokens=a.max_turn_tokens,
                               terminal_turns_per_episode=a.terminal_turns_per_episode,
                               terminal_turn_roles=turn_roles.priority_turns if a.terminal_turn_roles else None)
            fam_eps = [e for e in graded if family_of(e) == fam]
            positives = [e for e in fam_eps if int(verdicts[e["episode_id"]]["r_pass"]) == 1]
            for arm, eps, skip in (("positives", positives, ()), ("control", fam_eps, ("verdict",))):
                try:
                    res = filter_episodes(eps, verdicts, tasks_p, tok, cfg, skip_gates=skip)
                except SchemaError as e:   # the provenance contract (schema._REQUIRED) rejects this row kind
                    run.error(f"{fam} {arm}: provenance contract: {e}")
                    raise SystemExit(f"schema.validate_provenance rejects stage-2 {fam} rows ({e}); the contract in "
                                     "src/apex_data/README.md must allow them before the set can be built") from e
                survival[arm][fam] = {k: s.to_dict() for k, s in res.survival.items()}
                dropped += [{"arm": arm, **d.__dict__} for d in res.dropped]
                (pos_units if arm == "positives" else pool_units).update(units_of(res.kept))
                run.log(f"{fam} {arm}: {len(eps)} episodes in, {len(units_of(res.kept))} kept ({len(res.kept)} rows), {len(res.dropped)} dropped "
                        f"{dict(Counter(d.reason for d in res.dropped).most_common(6))}")

        quality_report, prefer, checks = None, set(), {}
        if a.quality:
            quality_report, prefer, qdrops, checks = apply_quality(pos_units, pool_units, episodes)
            dropped += qdrops
            run.log(f"quality {quality.VERSION}: loop drops {quality_report['loop_dropped']}; "
                    + "; ".join(f"{f} {r['mode']} (positives pass {r['positives_pass_share']}), dropped {r['dropped']}"
                                for f, r in quality_report["families"].items()))
        rng = random.Random(f"{a.seed}:stage2_set")
        pos_ids, cap_report = cap_per_task(pos_units, a.positives_per_task, rng, prefer)
        target_report: dict[str, Any] | None = None
        if a.family_target:
            pos_ids, trimmed, target_report = trim_to_targets(pos_ids, episodes, a.family_target, a.seed)
            dropped += [{"arm": "positives", "episode_id": e, "task_id": episodes[e]["task_id"], "row_kind": family_of(episodes[e]),
                         "gate": "family_target", "reason": f"above the {family_of(episodes[e])} target {a.family_target[family_of(episodes[e])]}",
                         "detail": ""} for e in trimmed]
            run.log(f"family targets {json.dumps({f: (r['before'], r['after']) for f, r in target_report.items()})}")
        arms: dict[str, dict[str, list[Rendered]]] = {"positives": {eid: pos_units[eid] for eid in pos_ids}, "control": {}}
        match: dict[str, Any] = {}
        def match_key(eid: str) -> str:      # --match-by source: one match per (family, task source), e.g. terminal/tgr_env
            fam_ = family_of(episodes[eid])
            return fam_ if a.match_by == "family" else f"{fam_}/{episodes[eid]['task_id'].split(':', 1)[0]}"

        for fam in sorted({match_key(eid) for eid in pool_units} | {match_key(eid) for eid in pos_ids}):
            target = sum(loss_tokens(arms["positives"][eid]) for eid in pos_ids if match_key(eid) == fam)
            cands = [(eid, loss_tokens(u)) for eid, u in pool_units.items() if match_key(eid) == fam]
            if any(eid in prefer for eid, _ in cands):   # quality preference: passing episodes first, each group shuffled
                cands.sort()
                rng.shuffle(cands)
                cands.sort(key=lambda c: c[0] not in prefer)
                chosen, total = fill_to_tokens(cands, target, a.match_tolerance, None)
            else:
                chosen, total = fill_to_tokens(cands, target, a.match_tolerance, rng)
            arms["control"].update({eid: pool_units[eid] for eid in chosen})
            match[fam] = {"target_loss_tokens": target, "control_loss_tokens": total, "rel_diff": round((total - target) / target, 5) if target else 0.0,
                          "within_tolerance": abs(total - target) <= a.match_tolerance * target, "pool_episodes": len(cands),
                          "pool_loss_tokens": sum(n for _, n in cands), "control_episodes": len(chosen),
                          "control_r_pass": dict(Counter(str(verdicts[eid]["r_pass"]) for eid in chosen))}
            if not match[fam]["within_tolerance"]:
                run.error(f"{fam}: control matched {total} of {target} loss tokens (pool {match[fam]['pool_loss_tokens']}); outside tolerance")

        # --- replay ---
        rollout_a = sum(loss_tokens(u) for u in arms["positives"].values())
        replay_target = round(rollout_a * a.replay_share / (1 - a.replay_share - a.fixed_share))
        replay_rows: list[dict[str, Any]] = []
        replay_order: dict[str, Any] = {"order": None}
        if inputs["replay_reserve"] is not None:
            reserve = load_replay(a.replay_reserve)
            keyed = {f"{r['sample_id']}:{r['turn_index']}": r for r in reserve}
            if len(keyed) != len(reserve):
                raise SystemExit(f"{a.replay_reserve}: replay row keys sample_id:turn_index are not unique")
            ranked = bool(reserve) and "replay_rank" in reserve[0]
            cands = [(k, int(r["n_assistant_tokens"])) for k, r in keyed.items()]
            if ranked:
                cands.sort(key=lambda c: keyed[c[0]]["replay_rank"])
            ids, got = fill_to_tokens(cands, replay_target, a.match_tolerance, None if ranked else rng)
            replay_rows = [keyed[k] for k in ids]
            replay_order = {"order": "rank" if ranked else "seeded_random"}
            if ranked:   # a pure prefix skips nothing below the last rank taken
                chosen = set(ids)
                last = max((keyed[k]["replay_rank"] for k in chosen), default=None)
                replay_order.update(last_rank=last, skipped_below_last_rank=sum(
                    1 for k, _ in cands if k not in chosen and last is not None and keyed[k]["replay_rank"] < last))
            if got < replay_target * (1 - a.match_tolerance):   # the reserve is too small: the replay share ends below --replay-share
                run.error(f"replay reserve short: {got:,d} of the {replay_target:,d} loss tokens a {a.replay_share:.0%} replay share needs "
                          f"(reserve {sum(int(r['n_assistant_tokens']) for r in reserve):,d}); every reserve row is used")
        else:
            run.error("no replay reserve: replay rows empty (--allow-no-replay; not a frozen set)")

        # --- fixed rows (--fixed-rows): a rank prefix shared by both arms ---
        fixed_rows: list[dict[str, Any]] = []
        fixed_report: dict[str, Any] | None = None
        if a.fixed_rows is not None:
            if not a.fixed_rows.exists():
                raise SystemExit(f"--fixed-rows {a.fixed_rows} missing")
            fixed_all = load_replay(a.fixed_rows)
            fkeyed = {f"{r['sample_id']}:{r['turn_index']}": r for r in fixed_all}
            if len(fkeyed) != len(fixed_all):
                raise SystemExit(f"{a.fixed_rows}: row keys sample_id:turn_index are not unique")
            if fixed_all and a.fixed_rank_column not in fixed_all[0]:
                raise SystemExit(f"{a.fixed_rows}: no {a.fixed_rank_column} column")
            if len({r[a.fixed_rank_column] for r in fixed_all}) != len(fixed_all):
                raise SystemExit(f"{a.fixed_rows}: {a.fixed_rank_column} values are not distinct")
            too_long = {k for k, r in fkeyed.items() if int(r["n_tokens"]) > a.max_row_tokens}
            long_turn = {k for k, r in fkeyed.items() if a.max_turn_tokens is not None and longest_trained_span(r) > a.max_turn_tokens} - too_long
            fixed_target = round(rollout_a * a.fixed_share / (1 - a.replay_share - a.fixed_share))
            cands = sorted(((k, int(r["n_assistant_tokens"])) for k, r in fkeyed.items() if k not in too_long | long_turn),
                           key=lambda c: fkeyed[c[0]][a.fixed_rank_column])
            ids, got = fill_to_tokens(cands, fixed_target, a.match_tolerance, None)
            fixed_rows = sorted((fkeyed[k] for k in ids), key=lambda r: r[a.fixed_rank_column])
            fixed_report = {"source": inputs["fixed_rows"], "share": a.fixed_share, "rank_column": a.fixed_rank_column,
                            "target_loss_tokens": fixed_target, "loss_tokens": got, "rows": len(fixed_rows),
                            "last_rank": max((r[a.fixed_rank_column] for r in fixed_rows), default=None),
                            "skipped_row_cap": len(too_long), "skipped_turn_cap": len(long_turn),
                            "file_rows": len(fixed_all), "file_loss_tokens": sum(int(r["n_assistant_tokens"]) for r in fixed_all)}
            for col in ("task_type", "row_kind", "fixed_part", "type", "domain"):
                if fixed_rows and col in fixed_rows[0]:
                    fixed_report[f"by_{col}"] = {str(k): {"rows": n, "loss_tokens": sum(int(r["n_assistant_tokens"]) for r in fixed_rows if r.get(col) == k)}
                                                 for k, n in Counter(r.get(col) for r in fixed_rows).items()}
            if got < fixed_target * (1 - a.match_tolerance):
                run.error(f"fixed rows short: {got:,d} of the {fixed_target:,d} loss tokens a {a.fixed_share:.0%} share needs")
            run.log(f"fixed rows: {len(fixed_rows)} rows, {got:,d} of {fixed_target:,d} loss tokens (skipped: row cap {len(too_long)}, "
                    f"turn cap {len(long_turn)})")

        # --- write: the trainer's three rows files (configs/stage2.yaml filtered_sft.arms.*.rows and replay_rows) ---
        out = a.out_dir
        out.mkdir(parents=True, exist_ok=True)
        files = {"positives": out / ROWS_FILES["positives"], "control": out / ROWS_FILES["control"], "replay": out / ROWS_FILES["replay"]}
        prov_lines: list[dict[str, Any]] = []
        file_report: dict[str, Any] = {}
        for arm in ARMS:
            rendered = sorted((r for u in arms[arm].values() for r in u), key=lambda r: r.provenance["row_key"])
            write_rows([r.row for r in rendered], files[arm])
            prov_lines += [{"file": files[arm].name, "part": "rollout", "arm": arm, **r.provenance,
                            "r_pass": int(verdicts[r.provenance["episode_id"]]["r_pass"]),
                            **({"quality_pass": bool(checks[r.provenance["episode_id"]]["pass"])} if checks else {})} for r in rendered]
            file_report[arm] = file_record(files[arm], [r.row for r in rendered], a.effective_batch)
        write_rows([{k: r[k] for k in ROW_COLUMNS} for r in replay_rows], files["replay"])
        prov_lines += [{"file": files["replay"].name, "part": "replay", "arm": "both", "row_key": f"{r['sample_id']}:{r['turn_index']}",
                        **{k: v for k, v in r.items() if k not in ROW_COLUMNS}, "n_tokens": r["n_tokens"], "n_assistant_tokens": r["n_assistant_tokens"]}
                       for r in replay_rows]
        file_report["replay"] = file_record(files["replay"], replay_rows, a.effective_batch)
        replay_tokens = file_report["replay"]["loss_tokens"]
        fixed_tokens = 0
        if a.fixed_rows is not None:
            files["fixed"] = out / FIXED_FILE
            write_rows([{k: r[k] for k in ROW_COLUMNS} for r in fixed_rows], files["fixed"])
            prov_lines += [{"file": files["fixed"].name, "part": "fixed", "arm": "both", "row_key": f"{r['sample_id']}:{r['turn_index']}",
                            **{k: v for k, v in r.items() if k not in ROW_COLUMNS}, "n_tokens": r["n_tokens"],
                            "n_assistant_tokens": r["n_assistant_tokens"]} for r in fixed_rows]
            file_report["fixed"] = file_record(files["fixed"], fixed_rows, a.effective_batch)
            fixed_tokens = file_report["fixed"]["loss_tokens"]
        arm_report: dict[str, Any] = {}
        for arm in ARMS:
            tokens, n_rows = file_report[arm]["loss_tokens"], file_report[arm]["rows"] + len(replay_rows) + len(fixed_rows)
            arm_report[arm] = {"rows_file": file_report[arm]["path"], "rows_sha256": file_report[arm]["sha256"],
                               "replay_rows_file": file_report["replay"]["path"], "replay_rows_sha256": file_report["replay"]["sha256"],
                               "episodes": len(arms[arm]), "tasks": len({episodes[e]["task_id"] for e in arms[arm]}),
                               "rollout_rows": file_report[arm]["rows"], "replay_rows": len(replay_rows), "rows": n_rows,
                               "rollout_loss_tokens": tokens, "replay_loss_tokens": replay_tokens, "loss_tokens": tokens + replay_tokens,
                               "replay_share": round(replay_tokens / (tokens + replay_tokens + fixed_tokens), 4) if tokens + replay_tokens else None,
                               **({"fixed_rows_file": file_report["fixed"]["path"], "fixed_rows_sha256": file_report["fixed"]["sha256"],
                                   "fixed_rows": len(fixed_rows), "fixed_loss_tokens": fixed_tokens,
                                   "fixed_share": round(fixed_tokens / (tokens + replay_tokens + fixed_tokens), 4) if tokens else None,
                                   "loss_tokens": tokens + replay_tokens + fixed_tokens,
                                   "shared_rows_files": [file_report["replay"]["path"], file_report["fixed"]["path"]]}
                                  if a.fixed_rows is not None else {}),
                               "planned_optimizer_steps": math.ceil(n_rows / a.effective_batch) if n_rows else 0,
                               "r_pass": dict(Counter(str(verdicts[e]["r_pass"]) for e in arms[arm])),
                               "by_family": {f: {"episodes": sum(1 for e in arms[arm] if family_of(episodes[e]) == f),
                                                 "rows": sum(len(u) for e, u in arms[arm].items() if family_of(episodes[e]) == f),
                                                 "loss_tokens": sum(loss_tokens(u) for e, u in arms[arm].items() if family_of(episodes[e]) == f)}
                                             for f in sorted({family_of(episodes[e]) for e in arms[arm]})}}
        arm_report["control"]["matched_optimizer_steps"] = arm_report["positives"]["planned_optimizer_steps"]
        arm_report["control"]["steps_note"] = ("the control trains the positives arm's planned steps "
                                               "(apex_train.stage2 --arm control --match-steps-of logs/<positives run_id>)")
        (out / "rows_provenance.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in prov_lines))
        (out / "dropped.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dropped))
        selection = {"positives": sorted(arms["positives"]), "control": sorted(arms["control"]), "late": late, "ungraded": ungraded,
                     "replay_row_keys": [f"{r['sample_id']}:{r['turn_index']}" for r in replay_rows]}
        (out / "selection.json").write_text(json.dumps(selection, indent=1))
        side = {name: {"path": str(out / f), "sha256": sha256_file(out / f)}
                for name, f in (("rows_provenance", "rows_provenance.jsonl"), ("dropped", "dropped.jsonl"), ("selection", "selection.json"))}
        manifest = {"what": "stage-2 set (docs/DATA_PLAN.md section 11, M6.2): arm positives = verified positives, arm control = "
                            "loss-token-matched unfiltered format-valid rollouts; each arm trains on its rows file plus replay_rows "
                            "(configs/stage2.yaml filtered_sft.arms / replay_rows)",
                    "run_id": run_id, "frozen_at": utc_now(), "config": config, "inputs": inputs,
                    "files": {f"{k}_rows": v for k, v in file_report.items()}, "arms": arm_report,
                    "counts": {"episodes": len(episodes), "graded": len(graded), "late_graded": len(late), "ungraded": len(ungraded),
                               "by_family": dict(Counter(family_of(e) for e in graded)),
                               "r_pass": dict(Counter(str(v["r_pass"]) for v in verdicts.values())),
                               "dropped_by_arm_gate": {f"{arm}/{gate}": n for (arm, gate), n in Counter((d["arm"], d["gate"]) for d in dropped).items()}},
                    "pass_at_k": hist, "survival": survival, "positives_cap": cap_report, "control_match": match,
                    "replay": {"target_loss_tokens": replay_target, "loss_tokens": replay_tokens, "rows": len(replay_rows),
                               "source": inputs["replay_reserve"], **replay_order},
                    "roles": {"version": turn_roles.VERSION, "order": turn_roles.ORDER,
                              "terminal_turns_per_episode": a.terminal_turns_per_episode,
                              **{arm: role_report([r for u in arms[arm].values() for r in u]) for arm in ARMS}}
                             if a.terminal_turn_roles else None,
                    "quality": quality_report,
                    **({"family_targets": target_report} if a.family_target else {}),
                    **({"fixed": fixed_report} if a.fixed_rows is not None else {}),
                    "side_files": side}
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
        tmp.replace(a.manifest)
        for arm in ARMS:
            run.metric({"arm": arm, **{k: v for k, v in arm_report[arm].items() if k not in ("by_family", "rows_file")}})
            run.log(f"{arm}: {arm_report[arm]['episodes']} episodes, {arm_report[arm]['rollout_rows']} rollout rows "
                    f"({arm_report[arm]['rollout_loss_tokens']:,d} loss tokens) + {len(replay_rows)} replay rows ({replay_tokens:,d}); "
                    f"replay share {arm_report[arm]['replay_share']}; planned steps {arm_report[arm]['planned_optimizer_steps']}")
        run.log(f"replay order {replay_order}" + (f"; terminal turn roles {json.dumps({arm: manifest['roles'][arm]['rows'] for arm in ARMS})}"
                                                  if manifest["roles"] else ""))
        run.log(f"control match {json.dumps({f: (m['target_loss_tokens'], m['control_loss_tokens'], m['rel_diff']) for f, m in match.items()})}; "
                f"manifest {a.manifest} sha256 {sha256_file(a.manifest)}")
        duration = time.time() - t0
        run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock"), ("orchestrator_cpus", os.cpu_count(), "vcpu", "CPU-only step"),
                             ("gpu_hours", 0.0, "h", "no GPU"), ("peak_gpu_memory_gb", "n/a", "GB", "no GPU"),
                             ("cost_usd", 0.0, "USD", "GPU cost only; CPU-only step, the always-on CPU VM is not charged")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(duration, 1), status="finished",
                           summary={"arms": {k: v for k, v in arm_report.items()}, "control_match": match,
                                    "manifest": {"path": str(a.manifest), "sha256": sha256_file(a.manifest)}})
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
