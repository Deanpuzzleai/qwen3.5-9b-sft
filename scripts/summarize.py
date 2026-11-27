#!/usr/bin/env python3
"""Aggregate tables and confidence intervals from the raw per-task results (benchmarks) and the dev-set outputs (dev).

Aggregate tables and confidence intervals come only from this script, run on raw results.

Usage (from the repository root):
  .venv/train/bin/python scripts/summarize.py [benchmarks] [--run base:hle=<run_id>] [--portal-score base=<file>] [...]
  .venv/train/bin/python scripts/summarize.py dev --run <dev run dir | jsonl> [--baseline <dev run dir | jsonl>] [...]
  PYTHONPATH=src .venv/train/bin/python scripts/summarize.py quickcheck --run sft_v5=<smoke run dir> --run base=<dev run dir> \
      [--run sft_v1=<part 1 dir>+<part 2 dir>] [--compare A:B ...] [--out PATH]

benchmarks mode (the default)
  Inputs (read-only). These are the only files opened; in particular never the HLE predictions or the judge's output
  store (third_party/.hle_judge_outputs/, which embeds the answer key), never task artifacts under results/.../tasks/:
    results/<version>/<benchmark>/<run_id>/per_task.jsonl   per-task records in the shared schema (src/apex_eval/common.py
                                                            `record`); version base|sft|stage2; benchmark swebench_pro |
                                                            terminal_bench | hle | asibench
    results/<version>/<benchmark>/<run_id>/summary.json     the runner's summary: `reportable`; its presence marks a finished run
    results/<version>/asibench/<run_id>/portal_score.json   optional ASI-Bench Portal result (schema below; or --portal-score)
    results/<version>/<benchmark>/<run_id>/superseded.json  optional: the run was replaced after a lock change
                                                            ({"superseded_by", "decision", "reason"}); it stays in results/
    logs/<run_id>/metadata.json                             `reportable` / `status` when summary.json is absent
    logs/<run_id>/resources.csv                             duration_sec, gpu_hours, cost_usd of the evaluation run
    configs/eval/swebench_pro_hard51_ids.txt, terminal_bench_task_ids.txt, terminal_bench_2.yaml (seeds), hle.yaml
    (split and its question count), asibench.yaml (task count, prompt levels): the locked task sets a reportable run covers
  Outputs:
    results/summary.csv         one row per benchmark metric x version (SWE-bench Pro resolved rate, Terminal-Bench success
                                rate, HLE accuracy, ASI-Bench Portal score and ASI-Bench all-tasks score, macro-average) with
                                the interval, the paired deltas vs Base and vs SFT, run-failure counts by kind, time, tokens, cost
    analysis/uncertainty.json   every number above plus the run inventory, McNemar counts, bootstrap details and the rules
    stdout                      the same tables, readable
    logs/<run_id>/              metadata.json (inputs with sha256), eval.log, metrics.jsonl, errors.log (warnings), resources.csv

dev mode
  Inputs: --run and --baseline name a dev evaluation output: a JSONL file, or a directory holding per_item.jsonl (else
  per_task.jsonl). One JSON object per item: `slice` (or `benchmark`), `task_id`, `sample_index` (or `sample`), and a
  verdict: `resolved` (bool), or `score` with optional `max_score` (without max_score the score is a fraction in [0, 1]);
  optional `run_status` (shared vocabulary). Items without a verdict count as 0.
  --admitted PATH (data/manifests/dev_admitted.json, src/apex_data/devset/admission.py) restricts the SWE and terminal
  slices of both outputs to the frozen admitted dev tasks; record lines of other tasks are left out and counted under
  `admitted.n_lines_not_admitted`.
  Outputs: stdout tables; logs/<run_id>/dev_summary.json (and --out PATH if given); the usual five log files.
  Per slice: score with a Wilson interval over items and a task-clustered bootstrap interval; against the baseline, items
  are matched by (task_id, sample_index): exact McNemar, a task-clustered paired bootstrap of the difference, and (with
  --guard-resolutions) the data plan's do-no-harm guard (point >= -3 pp, lower bound >= -resolution); a macro-average
  over --macro-slices (default: every slice present in both).

quickcheck mode (training plan v5, sections 2 and 4: the SFT quick check and the stage-2 arms' no-harm check)
  Inputs: --run LABEL=PATH (repeatable) names dev outputs as in dev mode ('+' joins the part runs of one evaluation); the
  first run's items (normally a `run_dev_eval.py --size smoke` run) are the item set, and every run is measured on them.
  For the SWE and terminal slices the item's agent transcript is read: episode.json in its `artifacts` directory or up to
  three levels above (the dev work root, outside results/; QA outputs and judge files are never opened).
  Outputs: stdout table; logs/<run_id>/quickcheck_summary.json (and --out PATH); the usual five log files.
  Agent items also have endpoint request logs (requests/*.jsonl.gz beside the transcript): every model response the
  harness received, including those it rejected and left out of the transcript (e.g. a response cut at the per-turn
  token limit without a tool call); requests the server refused (non-200, e.g. context overflow) are counted apart.
  Per run and slice: items, passes, thinking tokens per model response (mean and nearest-rank p90; agent slices: the
  server's reasoning-token count of every logged response, else the transcript's turns with the Qwen3.5-9B tokenizer;
  QA: the runner's reasoning tokens), truncated responses (cut at the per-turn token limit), SWE responses without a tool
  call, malformed turns (the runner's format failures over transcript turns), wasted actions (same normalised command and
  observation as an earlier action; apex_data.efficiency on the transcript), items whose longest run of one repeated
  command (normalised, consecutive) reaches --repeat-run (default 20; column `repeat`), verified share of completed episodes,
  budget-exhausted items, run failures and output tokens per item. Outcome measures (including the v7
  overthinking guard's): output tokens per pass (all items' output tokens over the passes), loop endings (exit_reason
  format_errors: three consecutive replies without a parsable tool call), step-cap endings, thinking p50, normal-reply
  thinking p50/p90 (responses not cut at the per-turn limit), the share of responses with thinking > 3,000 tokens, the
  share of those that still carry a tool call (back-to-acting; tool-calling harnesses only) and steps per episode (mean,
  p50; the record's agent steps). Per --compare A:B (default:
  the first run against each other run), on the items both have: the pass difference with items gained and lost, and the
  differences of the rates. No intervals: a quick check is read, not tested. --read-items N lists, per agent slice, the N
  items with the most output tokens in one run (--read-items-from, default the first) with each run's verdict, output
  tokens, turns and episode.json path, for reading transcripts side by side on the same items.

Rules (also in analysis/uncertainty.json under "rules")
  * Nothing is dropped. Every unit (task x sample) counts in the denominator; a unit without a positive verdict counts as
    unresolved (0), whatever its run status, and the statuses are counted separately: environment_failure,
    agent_error (model / agent failures), verifier_failure, scoring_failure, missing (run failures) and budget_exhausted
    (timeouts: the agent used its budget; still graded). A reportable run is checked against the locked task set: units
    the lock names but the run lacks are added as `missing`.
  * Absolute rates of binary benchmarks carry Wilson 95 percent intervals over units. ASI-Bench scores are 0-100 means
    with a percentile bootstrap over tasks.
  * Paired comparisons (SFT vs Base, stage 2 vs SFT, stage 2 vs Base) use the units both runs share: exact two-sided
    McNemar on discordant pairs (binary benchmarks) and a percentile bootstrap of the mean difference that resamples
    tasks (clusters) with replacement, so several samples of one task move together. Seeded; default 10,000 resamples.
  * ASI-Bench: `portal_score` is the Portal's aggregate (mean over scored results; NOT SCORED / evaluation_invalid
    excluded, the lock's primary metric); `all_tasks_score` is the mean over all 60 B1 results with NOT SCORED,
    evaluation_invalid and absent results counted as 0. Paired intervals need per-task Portal scores; with only the
    aggregate, the Portal score is reported without an interval.
  * Macro-average: equal-weight mean of the four headline metrics (resolved rate, success rate, accuracy, Portal score);
    its intervals combine the per-benchmark bootstrap resamples (independent across benchmarks; a benchmark without
    per-task data enters as a constant). With fewer than four headline metrics it is reported as macro_average_partial.
  * Run choice per (version, benchmark): --run pins a run; otherwise the newest finished run marked reportable;
    --include-unreportable falls back to the newest finished, then the newest run with records (flagged in the table).
    A run with superseded.json (replaced after a lock change) is listed in the inventory but chosen only when pinned.

portal_score.json (written by hand from the Portal's result page; every field optional except one score source):
  {"submission_id": "...", "status": "scored", "aggregate_score": 12.3, "n_scored": 57,
   "per_task": [{"task_id": "math.x", "prompt_level": "b1", "final_score": 40.0, "max_score": 100.0,
                 "evaluation_status": "completed" | "evaluation_invalid"}, ...]}
  The asibench CLI score-report shape (`mean_percent`, `scored_instance_count`, `results`) is accepted too. Scores are
  read as 100 x final_score / max_score (final_score as-is without max_score).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from apex_eval.common import (AGENT_ERROR, BUDGET_EXHAUSTED, ENVIRONMENT_FAILURE, MISSING, OK, PENDING_EXTERNAL,  # noqa: E402
                              RUN_FAILURE_STATUSES, SCORING_FAILURE, SKIPPED, VERIFIER_FAILURE, VERSIONS, RunDir, command_line,
                              compute_rates, git_info, hardware_info, model_lock, new_run_id, read_task_ids, sha256_file, utc_now,
                              write_json)

EVAL_LOCKS = REPO_ROOT / "configs" / "eval"
# benchmark -> (display name, headline metric, binary verdicts); README order
BENCHMARKS: dict[str, tuple[str, str, bool]] = {
    "swebench_pro": ("SWE-bench Pro v2 HARD-51", "resolved_rate", True),
    "terminal_bench": ("Terminal-Bench 2.0", "success_rate", True),
    "hle": ("HLE text-only", "accuracy", True),
    "asibench": ("ASI-Bench seed42 B1", "portal_score", False),
}
ASI_METRICS = ("portal_score", "all_tasks_score")
COMPARISONS = (("sft", "base"), ("stage2", "sft"), ("stage2", "base"))   # (treatment, reference)
STATUSES = (OK, BUDGET_EXHAUSTED, AGENT_ERROR, ENVIRONMENT_FAILURE, VERIFIER_FAILURE, SCORING_FAILURE, MISSING, PENDING_EXTERNAL, SKIPPED)
RESULT_FILES = frozenset({"per_task.jsonl", "summary.json", "portal_score.json", "superseded.json"})   # the only files read under results/
LOG_FILES = frozenset({"metadata.json", "resources.csv"})                           # the only files read under logs/<run_id>/
DEV_FILES = ("per_item.jsonl", "per_task.jsonl")
MAX_LISTED_DISCORDANT = 200   # discordant task ids are listed for benchmarks up to this size (not HLE)
RUNNER_METRIC_KEYS = ("resolved_rate_pct", "mean_success_rate_pct", "accuracy_pct")
# the rules section of the module docstring, one string per bullet (written to analysis/uncertainty.json)
RULES = [" ".join(b.split()) for b in __doc__.split("Rules (also in")[1].split("portal_score.json (written")[0].split("\n  * ")[1:]]

Key = tuple[str, Any]   # (task_id, sample)


# --- statistics ----------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials, in percent."""
    if n <= 0:
        return None
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of b vs c discordant pairs at p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, float(Fraction(2 * tail, 2 ** n)))


def cluster_bootstrap(terms: list[tuple[np.ndarray, np.ndarray, float]], n_boot: int, rng: np.random.Generator,
                      chunk: int = 1000) -> np.ndarray:
    """Resamples of sum_i sign_i * sum(num_i[idx]) / sum(den_i[idx]); every term shares the resampled task indices idx
    (paired). num/den are per-task (cluster) sums over the task's units, so a task's samples move together."""
    n_tasks = len(terms[0][0])
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n_tasks, size=(m, n_tasks))
        stat = np.zeros(m)
        with np.errstate(invalid="ignore", divide="ignore"):
            for num, den, sign in terms:
                stat += sign * num[idx].sum(axis=1) / den[idx].sum(axis=1)
        out[start:start + m] = stat
    return out


def percentile_ci(boot: np.ndarray | None, alpha: float) -> tuple[float, float] | None:
    if boot is None or not np.isfinite(boot).any():
        return None
    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rnd(x: Any, nd: int = 2) -> Any:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def rnd_pair(x: tuple[float, float] | None, nd: int = 2) -> list[float] | None:
    return None if x is None else [rnd(x[0], nd), rnd(x[1], nd)]


def clusters(values: dict[Key, float | None], keys: list[Key], scored_only: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-task (sum of unit values, unit count) over `keys`, task ids sorted. scored_only: units without a value are left
    out of both sums (Portal-style mean); otherwise they count as 0."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for k in keys:
        t, v = str(k[0]), values[k]
        num.setdefault(t, 0.0)
        den.setdefault(t, 0.0)
        if v is not None or not scored_only:
            num[t] += v or 0.0
            den[t] += 1.0
    tasks = sorted(num)
    return np.array([num[t] for t in tasks]), np.array([den[t] for t in tasks]), tasks


def ratio(num: np.ndarray, den: np.ndarray) -> float | None:
    return float(num.sum() / den.sum()) if den.sum() > 0 else None


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    """An independent, reproducible random stream per (benchmark, comparison) so results do not depend on call order."""
    return np.random.default_rng([seed, *stream])


# --- guarded file access -------------------------------------------------------------------------------------------
def read_result_file(path: Path) -> str | None:
    """Text of an allowed results/ or logs/ file (RESULT_FILES / LOG_FILES), None when absent. Any other name is a bug."""
    if path.name not in RESULT_FILES | LOG_FILES:
        raise ValueError(f"summarize.py does not read {path.name} (allowed: {sorted(RESULT_FILES | LOG_FILES)})")
    try:
        return path.read_text()
    except OSError:
        return None


def read_json_file(path: Path) -> Any:
    text = read_result_file(path)
    try:
        return json.loads(text) if text is not None else None
    except json.JSONDecodeError:
        return None


def read_resources(path: Path) -> dict[str, float]:
    """logs/<run_id>/resources.csv (metric,value,unit,note) -> {metric: float} for the numeric rows."""
    text = read_result_file(path)
    out: dict[str, float] = {}
    for row in csv.DictReader((text or "").splitlines()):
        try:
            out[row["metric"]] = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    os.replace(tmp, path)


# --- locked task sets ----------------------------------------------------------------------------------------------
def locked_task_sets() -> dict[str, dict[str, Any]]:
    """benchmark -> {"keys": unit keys or None, "n": unit count, "source": ...} from the evaluation locks (read-only)."""
    tb_lock = yaml.safe_load((EVAL_LOCKS / "terminal_bench_2.yaml").read_text())
    hle_lock = yaml.safe_load((EVAL_LOCKS / "hle.yaml").read_text())
    asi_lock = yaml.safe_load((EVAL_LOCKS / "asibench.yaml").read_text())
    swe_ids = read_task_ids(EVAL_LOCKS / "swebench_pro_hard51_ids.txt")
    tb_keys = [(t, f"seed_{s}") for s in tb_lock["protocol"]["seeds"] for t in read_task_ids(EVAL_LOCKS / "terminal_bench_task_ids.txt")]
    split = hle_lock["protocol"]["split"]
    n_hle = int(hle_lock["dataset_facts"]["text_only" if split == "text-only" else "n_questions"])
    n_asi = int(asi_lock["task_set"]["n_tasks"]) * len(asi_lock["task_set"]["prompt_levels"])
    return {
        "swebench_pro": {"keys": [(t, None) for t in swe_ids], "n": len(swe_ids), "source": "configs/eval/swebench_pro_hard51_ids.txt"},
        "terminal_bench": {"keys": tb_keys, "n": len(tb_keys), "source": "configs/eval/terminal_bench_task_ids.txt x terminal_bench_2.yaml seeds"},
        "hle": {"keys": None, "n": n_hle, "source": f"configs/eval/hle.yaml dataset_facts ({split})"},
        "asibench": {"keys": None, "n": n_asi, "source": "configs/eval/asibench.yaml task_set (n_tasks x prompt_levels)"},
    }


# --- runs ------------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    version: str
    benchmark: str
    run_id: str
    results_dir: Path
    logs_dir: Path
    summary: dict[str, Any] | None
    metadata: dict[str, Any] | None
    superseded: dict[str, Any] | None = None   # superseded.json: replaced after a lock change, chosen only when pinned
    pinned: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)
    values: dict[Key, float | None] = field(default_factory=dict)   # unit -> 0-100 value (None: no score, ASI-Bench only)
    statuses: dict[Key, str] = field(default_factory=dict)
    n_absent: int = 0                     # units the lock counts but the run lacks, where the lock gives no ids (HLE, ASI)
    has_scores: bool = False              # per-unit values exist (always for binary benchmarks; ASI-Bench: Portal or local scores)
    portal: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_records(self) -> bool:
        return (self.results_dir / "per_task.jsonl").exists()

    @property
    def finished(self) -> bool:
        return self.summary is not None

    @property
    def reportable(self) -> bool | None:
        for src in (self.summary, self.metadata):
            if src and src.get("reportable") is not None:
                return bool(src["reportable"])
        return None

    @property
    def order(self) -> tuple[str, str]:
        m = re.search(r"(\d{8}_\d{6})$", self.run_id)
        return (m.group(1) if m else "", self.run_id)

    def inventory(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "finished": self.finished, "reportable": self.reportable, "has_records": self.has_records,
                "status": (self.metadata or {}).get("status"),
                "superseded_by": self.superseded.get("superseded_by") if isinstance(self.superseded, dict) else None}


def discover(results_root: Path, logs_root: Path, version: str, benchmark: str) -> list[Run]:
    bench_dir = results_root / version / benchmark
    if not bench_dir.is_dir():
        return []
    def superseded(d: Path) -> dict[str, Any] | None:   # a marker that does not parse still marks the run
        return (read_json_file(d / "superseded.json") or {}) if (d / "superseded.json").exists() else None
    return [Run(version, benchmark, d.name, d, logs_root / d.name, read_json_file(d / "summary.json"),
                read_json_file(logs_root / d.name / "metadata.json"), superseded(d))
            for d in sorted(bench_dir.iterdir()) if d.is_dir()]


def select_run(cands: list[Run], pinned: str | None, include_unreportable: bool) -> tuple[Run | None, str]:
    """(selected run, why) under the run-choice rule of the module docstring."""
    if pinned:
        hit = [c for c in cands if c.run_id == pinned]
        if not hit or not hit[0].has_records:
            raise SystemExit(f"--run pins {pinned}, which has no per_task.jsonl under {cands[0].results_dir.parent if cands else 'results/'}")
        hit[0].pinned = True
        return hit[0], "pinned with --run"
    newest = lambda runs: max(runs, key=lambda r: r.order)  # noqa: E731
    cands = [c for c in cands if c.superseded is None]
    tiers = [([c for c in cands if c.has_records and c.finished and c.reportable is True], "newest finished reportable run")]
    if include_unreportable:
        tiers += [([c for c in cands if c.has_records and c.finished], "newest finished run (NOT REPORTABLE)"),
                  ([c for c in cands if c.has_records], "newest run with records (UNFINISHED)")]
    for runs, why in tiers:
        if runs:
            return newest(runs), why
    return None, "no eligible run"


def load_records(run: Run) -> None:
    """per_task.jsonl -> run.records; a repeated (task_id, sample) keeps its last line, a malformed line is reported."""
    by_key: dict[Key, dict[str, Any]] = {}
    n_dup = n_bad = 0
    for line in (read_result_file(run.results_dir / "per_task.jsonl") or "").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            key = (str(r["task_id"]), r.get("sample"))
        except (json.JSONDecodeError, KeyError, TypeError):
            n_bad += 1
            continue
        n_dup += key in by_key
        by_key[key] = r
    run.records = list(by_key.values())
    if n_dup:
        run.notes.append(f"{n_dup} repeated (task_id, sample) lines: the last one is used")
    if n_bad:
        run.notes.append(f"{n_bad} malformed lines in per_task.jsonl (a reportable run counts their units as missing)")


def portal_result(d: Any) -> dict[str, Any] | None:
    """Normalise a portal_score.json (module docstring) to {aggregate_pct, n_scored, per_task {(task, level): pct|None}, ...}."""
    if not isinstance(d, dict):
        return None
    per: dict[Key, float | None] = {}
    for e in d.get("per_task") or d.get("results") or []:
        tid = e.get("task_id") or str(e.get("instance_id", "")).split("__")[0]
        level = str(e.get("prompt_level") or e.get("sample") or "").lower() or None
        fs, mx = e.get("final_score"), e.get("max_score")
        status = e.get("evaluation_status") or ("completed" if fs is not None else "evaluation_invalid")
        per[(tid, level)] = None if status != "completed" or fs is None else (100 * float(fs) / float(mx) if mx else float(fs))
    agg = d.get("aggregate_score", d.get("mean_percent"))
    n_scored = d.get("n_scored", d.get("scored_instance_count"))
    return {"submission_id": d.get("submission_id"), "status": d.get("status"), "aggregate_pct": None if agg is None else float(agg),
            "n_scored": None if n_scored is None else int(n_scored), "per_task": per or None}


def build_units(run: Run, locked: dict[str, Any] | None) -> None:
    """Unit values and statuses of a selected run; a reportable run is completed against its locked task set."""
    binary = BENCHMARKS[run.benchmark][2]
    portal_per = (run.portal or {}).get("per_task")
    matched: set[Key] = set()
    for r in run.records:
        key = (str(r["task_id"]), r.get("sample"))
        run.statuses[key] = str(r.get("run_status"))
        if binary:
            run.values[key] = 100.0 if r.get("resolved") is True else 0.0
        elif portal_per is not None:
            pkey = next((k for k in ((key[0], str(key[1]).lower() if key[1] is not None else None), (key[0], None)) if k in portal_per), None)
            run.values[key] = portal_per[pkey] if pkey else None
            if pkey:
                matched.add(pkey)
        elif r.get("score") is not None and r.get("run_status") == OK:   # a local score (non-official companion split only)
            run.values[key] = 100 * float(r["score"]) / float(r["max_score"]) if r.get("max_score") else float(r["score"])
        else:
            run.values[key] = None
    run.has_scores = binary or portal_per is not None or any(v is not None for v in run.values.values())
    if portal_per is not None:
        unmatched = sorted(set(portal_per) - matched, key=str)
        if unmatched:
            run.notes.append(f"{len(unmatched)} Portal per-task entries match no record (e.g. {unmatched[0]})")
    if not (locked and run.reportable):
        if run.reportable is not True:
            run.notes.append("NOT REPORTABLE: lock deviation, task subset or skipped scoring (see the runner's summary.json)")
        return
    if locked["keys"] is not None:
        absent = [k for k in locked["keys"] if k not in run.values]
        lock_keys = set(locked["keys"])
        extra = [k for k in run.values if k not in lock_keys]
        for k in absent:
            run.values[k], run.statuses[k] = (0.0 if binary else None), MISSING
        if absent:
            run.notes.append(f"{len(absent)} locked units absent from per_task.jsonl: counted as missing and unresolved")
        if extra:
            run.notes.append(f"{len(extra)} units not in the locked task set (kept in the denominator), e.g. {extra[0]}")
    else:
        run.n_absent = max(0, locked["n"] - len(run.values))
        if run.n_absent:
            run.notes.append(f"{run.n_absent} of the lock's {locked['n']} units absent from per_task.jsonl: counted as missing and as 0")
        elif len(run.values) > locked["n"]:
            run.notes.append(f"{len(run.values)} units, the lock counts {locked['n']}")


# --- per-run aggregation ---------------------------------------------------------------------------------------------
def failure_block(run: Run) -> dict[str, Any]:
    counts = Counter(run.statuses.values())
    counts[MISSING] += run.n_absent
    n = len(run.statuses) + run.n_absent
    n_fail = sum(counts[s] for s in RUN_FAILURE_STATUSES)
    return {"n_units": n, "status_counts": {s: counts[s] for s in STATUSES if counts[s]} | {s: c for s, c in counts.items() if s not in STATUSES},
            "n_run_failures": n_fail, "run_failure_rate_pct": rnd(100 * n_fail / n) if n else None,
            "by_kind": {"environment_failure": counts[ENVIRONMENT_FAILURE], "model_agent_error": counts[AGENT_ERROR],
                        "verifier_failure": counts[VERIFIER_FAILURE], "scoring_failure": counts[SCORING_FAILURE],
                        "missing": counts[MISSING], "timeout_budget_exhausted": counts[BUDGET_EXHAUSTED],
                        "pending_external": counts[PENDING_EXTERNAL], "skipped": counts[SKIPPED]}}


def cost_block(run: Run, n_units: int) -> dict[str, Any]:
    res = read_resources(run.logs_dir / "resources.csv")
    times = [float(r["time_sec"]) for r in run.records if r.get("time_sec") is not None]
    tin = [float(r["input_tokens"]) for r in run.records if r.get("input_tokens") is not None]
    tout = [float(r["output_tokens"]) for r in run.records if r.get("output_tokens") is not None]
    dur, gpu_h = res.get("duration_sec"), res.get("gpu_hours")
    # GPU cost only: gpu_hours x the GPU rate; the always-on CPU host is not charged to runs, and
    # older resources.csv files whose cost_usd still includes it are not read for cost
    cost = gpu_h * compute_rates()["gpu_vm_usd_per_hour"] if gpu_h is not None else None
    return {"mean_time_sec": rnd(fmean(times), 1) if times else None, "time_known_for_n": len(times),
            "mean_input_tokens": rnd(fmean(tin), 0) if tin else None, "mean_output_tokens": rnd(fmean(tout), 0) if tout else None,
            "tokens_known_for_n": len(tout), "total_input_tokens": int(sum(tin)) if tin else None,
            "total_output_tokens": int(sum(tout)) if tout else None,
            "duration_h": rnd(dur / 3600, 3) if dur is not None else None,
            "wall_sec_per_unit": rnd(dur / n_units, 1) if dur is not None and n_units else None,
            "gpu_hours": rnd(res.get("gpu_hours"), 3), "cost_usd": rnd(cost),
            "cost_usd_per_unit": rnd(cost / n_units, 4) if cost is not None and n_units else None,
            "source": "logs/<run_id>/resources.csv gpu_hours x configs/compute.yaml gpu_vm_usd_per_hour (GPU cost only)" if res else "resources.csv missing"}


def absolute(run: Run, bench_idx: int, version_idx: int, n_boot: int, seed: int, alpha: float) -> tuple[dict[str, Any], dict[str, np.ndarray | float | None]]:
    """(the run's table entry, metric -> bootstrap resamples (array), a constant (no per-task data) or None)."""
    name, headline, binary = BENCHMARKS[run.benchmark]
    keys = sorted(run.values, key=str)
    fb = failure_block(run)
    metrics: dict[str, Any] = {}
    boots: dict[str, np.ndarray | float | None] = {}
    absent = [(f"__absent_{i}", None) for i in range(run.n_absent)]   # count-only missing units: own clusters, value 0
    values = run.values | {k: (0.0 if binary else None) for k in absent}
    all_keys = keys + absent
    if binary:
        k = sum(1 for key in all_keys if values[key] == 100.0)
        n = len(all_keys)
        num, den, _ = clusters(values, all_keys, scored_only=False)
        boots[headline] = cluster_bootstrap([(num, den, 1.0)], n_boot, rng_for(seed, bench_idx, 100 + version_idx)) if n else None
        metrics[headline] = {"value_pct": rnd(100 * k / n) if n else None, "ci": rnd_pair(wilson(k, n, alpha)), "ci_method": "wilson",
                             "n_units": n, "n_successes": k}
    else:
        portal = run.portal or {}
        n_scored_units = sum(1 for key in keys if values[key] is not None)
        for mi, metric in enumerate(ASI_METRICS):
            scored_only = metric == "portal_score"
            num, den, _ = clusters(values, all_keys, scored_only=scored_only)
            per_task_value = ratio(num, den) if run.has_scores else None
            boot = cluster_bootstrap([(num, den, 1.0)], n_boot, rng_for(seed, bench_idx, 100 + version_idx, mi)) if per_task_value is not None else None
            entry: dict[str, Any] = {"value_pct": rnd(per_task_value), "ci": rnd_pair(percentile_ci(boot, alpha)),
                                     "ci_method": "task bootstrap" if boot is not None else None,
                                     "n_units": int(den.sum()) if scored_only else len(all_keys), "n_scored": n_scored_units,
                                     "source": ("Portal per-task scores" if portal.get("per_task") else "per_task.jsonl scores (local scorer)")
                                     if per_task_value is not None else None}
            if metric == "portal_score" and portal.get("aggregate_pct") is not None:
                if per_task_value is not None and abs(portal["aggregate_pct"] - per_task_value) > 0.05:
                    run.notes.append(f"Portal aggregate {portal['aggregate_pct']:.2f} differs from the per-task mean {per_task_value:.2f}; "
                                     "the table shows the Portal's number, the interval is around the per-task mean")
                entry.update(value_pct=rnd(portal["aggregate_pct"]), source="Portal aggregate")
                if boot is None:
                    entry["ci_method"] = "none (aggregate only; no per-task Portal scores)"
            if metric == "all_tasks_score" and per_task_value is None and portal.get("aggregate_pct") is not None and portal.get("n_scored") is not None:
                n_units = len(all_keys)
                entry.update(value_pct=rnd(portal["aggregate_pct"] * portal["n_scored"] / n_units) if n_units else None,
                             source="Portal aggregate x n_scored / n_units", ci_method="none (aggregate only)")
            if entry["value_pct"] is None:
                pending = fb["status_counts"].get(PENDING_EXTERNAL, 0)
                entry["source"] = f"pending: no Portal score yet ({pending} results pending_external)" if pending else "no scores"
            metrics[metric] = entry
            boots[metric] = boot if boot is not None else (float(entry["value_pct"]) if entry["value_pct"] is not None else None)
        if run.portal:
            metrics["portal"] = {k: run.portal.get(k) for k in ("submission_id", "status", "aggregate_pct", "n_scored")} | {
                "n_per_task": len(run.portal["per_task"] or {}) if run.portal.get("per_task") else 0}
    per_sample = None
    samples = sorted({str(k[1]) for k in keys if k[1] is not None})
    if len(samples) > 1:
        per_sample = {}
        for s in samples:
            vs = [values[k] for k in keys if str(k[1]) == s]
            sv = [v for v in vs if v is not None]
            per_sample[s] = {"n": len(vs), "value_pct": rnd(fmean(sv)) if sv else None}
        rates = [d["value_pct"] for d in per_sample.values() if d["value_pct"] is not None]
        per_sample["std_across_samples_pp"] = rnd(stdev(rates)) if len(rates) > 1 else None
    runner_metric = next((run.summary[k] for k in RUNNER_METRIC_KEYS if run.summary and run.summary.get(k) is not None), None)
    mine = metrics[headline]["value_pct"] if binary else None
    if runner_metric is not None and mine is not None and abs(float(runner_metric) - mine) > 0.01:
        run.notes.append(f"the runner's summary reports {runner_metric}; this table uses the locked denominator")
    entry = {"run_id": run.run_id, "results_dir": str(run.results_dir), "finished": run.finished, "reportable": run.reportable,
             "pinned": run.pinned, "model_name": (run.summary or {}).get("model_name") or (run.metadata or {}).get("model_name"),
             "benchmark": name, "metrics": metrics, "per_sample": per_sample, "failures": fb,
             "time_tokens_cost": cost_block(run, fb["n_units"]), "notes": run.notes}
    return entry, boots


def compare(bench: str, run_b: Run, run_a: Run, bench_idx: int, cmp_idx: int, n_boot: int, seed: int,
            alpha: float) -> tuple[dict[str, Any], np.ndarray | float | None]:
    """Paired comparison of run_b (treatment) vs run_a (reference) on their shared units, headline metric (plus the
    all-tasks score for ASI-Bench). Returns (details, the headline difference's resamples or constant or None)."""
    name, headline, binary = BENCHMARKS[bench]
    shared = sorted(set(run_a.values) & set(run_b.values), key=str)
    out: dict[str, Any] = {"treatment": run_b.version, "reference": run_a.version, "run_treatment": run_b.run_id, "run_reference": run_a.run_id,
                           "n_shared_units": len(shared), "n_shared_tasks": len({k[0] for k in shared}),
                           "n_units_only_reference": len(set(run_a.values) - set(run_b.values)) + run_a.n_absent,
                           "n_units_only_treatment": len(set(run_b.values) - set(run_a.values)) + run_b.n_absent, "metrics": {}}
    if not shared:
        out["note"] = "no shared units"
        return out, None
    headline_boot: np.ndarray | float | None = None
    for mi, metric in enumerate((headline,) if binary else ASI_METRICS):
        scored_only = metric == "portal_score"
        num_a, den_a, _ = clusters(run_a.values, shared, scored_only)
        num_b, den_b, _ = clusters(run_b.values, shared, scored_only)
        va, vb = ratio(num_a, den_a), ratio(num_b, den_b)
        m: dict[str, Any] = {}
        if va is not None and vb is not None and run_a.has_scores and run_b.has_scores:
            boot = cluster_bootstrap([(num_b, den_b, 1.0), (num_a, den_a, -1.0)], n_boot, rng_for(seed, bench_idx, cmp_idx, mi))
            m = {"reference_pct": rnd(va), "treatment_pct": rnd(vb), "diff_pp": rnd(vb - va), "ci": rnd_pair(percentile_ci(boot, alpha)),
                 "ci_method": f"paired task-clustered percentile bootstrap, {n_boot} resamples", "boot_se_pp": rnd(float(np.nanstd(boot)), 3),
                 "frac_resamples_above_0": rnd(float(np.nanmean(boot > 0)), 4)}
            if metric == headline:
                headline_boot = boot
        elif metric == headline:
            pa, pb = portal_value(run_a), portal_value(run_b)
            if pa is not None and pb is not None:
                m = {"reference_pct": rnd(pa), "treatment_pct": rnd(pb), "diff_pp": rnd(pb - pa), "ci": None,
                     "ci_method": "none: per-task Portal scores are missing on at least one side"}
                headline_boot = pb - pa
            else:
                m = {"diff_pp": None, "note": "no scores on at least one side (Portal pending)"}
        if binary and m:
            gained = [k for k in shared if run_a.values[k] == 0.0 and run_b.values[k] == 100.0]
            lost = [k for k in shared if run_a.values[k] == 100.0 and run_b.values[k] == 0.0]
            m["mcnemar"] = {"n_lost_reference_only": len(lost), "n_gained_treatment_only": len(gained),
                            "p_exact_two_sided": rnd(mcnemar_exact(len(lost), len(gained)), 5),
                            "note": "pairs are units; exact for one sample per task" if len({k[1] for k in shared}) <= 1 else
                                    "pairs are units (several samples per task); the bootstrap interval is the clustered one"}
            if len(shared) <= MAX_LISTED_DISCORDANT:
                m["discordant_tasks"] = {"gained": [list(k) for k in gained], "lost": [list(k) for k in lost]}
        out["metrics"][metric] = m
    return out, headline_boot


def portal_value(run: Run) -> float | None:
    """ASI-Bench headline of one run: the Portal's aggregate, else the mean over scored per-task results, else None."""
    agg = (run.portal or {}).get("aggregate_pct")
    if agg is not None or not run.has_scores:
        return agg
    num, den, _ = clusters(run.values, list(run.values), scored_only=True)
    return ratio(num, den)


def macro_entry(per_bench: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Equal-weight mean over benchmarks of values (floats) with resamples (arrays) or constants (floats)."""
    covered = [b for b in BENCHMARKS if per_bench.get(b) is not None and per_bench[b][0] is not None]
    if not covered:
        return {"value_pct": None, "ci": None, "covered": [], "complete": False}
    value = fmean(per_bench[b][0] for b in covered)
    boots = [per_bench[b][1] for b in covered]
    arrays = [x for x in boots if isinstance(x, np.ndarray)]
    ci = None
    if arrays and all(x is not None for x in boots):
        n = len(arrays[0])
        ci = percentile_ci(np.mean([x if isinstance(x, np.ndarray) else np.full(n, float(x)) for x in boots], axis=0), alpha)
    return {"value_pct": rnd(value), "ci": rnd_pair(ci), "covered": covered, "complete": len(covered) == len(BENCHMARKS),
            "ci_note": "constants enter benchmarks without per-task data" if any(not isinstance(x, np.ndarray) for x in boots) else None}


# --- benchmarks mode -------------------------------------------------------------------------------------------------
CSV_COLUMNS = ["benchmark", "metric", "version", "run_id", "reportable", "value_pct", "ci_low", "ci_high", "ci_method", "n_units",
               "n_successes", "delta_vs_base_pp", "delta_vs_base_ci_low", "delta_vs_base_ci_high", "mcnemar_p_vs_base",
               "delta_vs_sft_pp", "delta_vs_sft_ci_low", "delta_vs_sft_ci_high", "mcnemar_p_vs_sft", "n_paired_vs_base",
               "n_paired_vs_sft", "n_units_total", "n_run_failures", "run_failure_rate_pct", "n_environment_failure", "n_agent_error",
               "n_verifier_failure", "n_scoring_failure", "n_missing", "n_budget_exhausted", "n_pending_external", "n_skipped",
               "mean_time_sec", "wall_sec_per_unit", "mean_input_tokens", "mean_output_tokens", "duration_h", "gpu_hours",
               "cost_usd", "cost_usd_per_unit", "notes"]


def parse_pins(items: list[str], what: str, pattern: str) -> dict[tuple[str, ...], str]:
    out: dict[tuple[str, ...], str] = {}
    for item in items:
        m = re.fullmatch(pattern, item)
        if not m:
            raise SystemExit(f"{what}: cannot parse {item!r}")
        *key, value = m.groups()
        if key[0] not in VERSIONS or (len(key) > 1 and key[1] not in BENCHMARKS):
            raise SystemExit(f"{what}: unknown version or benchmark in {item!r}")
        out[tuple(key)] = value
    return out


def run_benchmarks(a: argparse.Namespace, log: RunDir) -> dict[str, Any]:
    pins = parse_pins(a.run, "--run", r"(\w+):(\w+)=(\S+)")
    portal_files = parse_pins(a.portal_score, "--portal-score", r"(\w+)=(\S+)")
    locks = locked_task_sets()
    inventory: dict[str, dict[str, Any]] = {}
    selected: dict[tuple[str, str], Run] = {}
    inputs: list[dict[str, Any]] = []
    for bench in BENCHMARKS:
        for version in VERSIONS:
            cands = discover(a.results_root, a.logs_root, version, bench)
            run, why = select_run(cands, pins.get((version, bench)), a.include_unreportable)
            inventory.setdefault(bench, {})[version] = {"selected": run.run_id if run else None, "why": why,
                                                        "candidates": [c.inventory() for c in cands]}
            if run is None:
                continue
            load_records(run)
            if bench == "asibench":
                pfile = Path(portal_files[(version,)]) if (version,) in portal_files else run.results_dir / "portal_score.json"
                if pfile.exists():
                    if pfile.name != "portal_score.json":
                        raise SystemExit(f"--portal-score: the file must be named portal_score.json, got {pfile}")
                    run.portal = portal_result(read_json_file(pfile))
                    inputs.append({"path": str(pfile), "sha256": sha256_file(pfile)})
                    if run.portal is None:
                        run.notes.append(f"{pfile} is not a JSON object; ignored")
            build_units(run, locks[bench])
            if why != "newest finished reportable run":
                run.notes.append(f"run choice: {why}")
            n_reportable = sum(1 for c in cands if c.finished and c.reportable is True and c.superseded is None)
            if n_reportable > 1 and not run.pinned:
                run.notes.append(f"{n_reportable} finished reportable runs exist; the newest is used (pin with --run to choose)")
            selected[(version, bench)] = run
            inputs.append({"path": str(run.results_dir / "per_task.jsonl"), "sha256": sha256_file(run.results_dir / "per_task.jsonl"),
                           "n_records": len(run.records)})

    absolute_out: dict[str, dict[str, Any]] = {}
    headline: dict[str, dict[str, tuple[float | None, Any]]] = {v: {} for v in VERSIONS}
    for bi, bench in enumerate(BENCHMARKS):
        for vi, version in enumerate(VERSIONS):
            run = selected.get((version, bench))
            if run is None:
                continue
            entry, boots = absolute(run, bi, vi, a.n_boot, a.seed, a.alpha)
            absolute_out.setdefault(bench, {})[version] = entry
            hm = BENCHMARKS[bench][1]
            headline[version][bench] = (entry["metrics"][hm]["value_pct"], boots.get(hm))
            log.metric({"event": "absolute", "benchmark": bench, "version": version, "run_id": run.run_id,
                        **{m: e.get("value_pct") for m, e in entry["metrics"].items() if isinstance(e, dict) and "value_pct" in e}})

    comparisons_out: dict[str, dict[str, Any]] = {}
    deltas: dict[tuple[str, str], dict[str, tuple[float | None, Any]]] = {}
    for bi, bench in enumerate(BENCHMARKS):
        for ci, (vb, va) in enumerate(COMPARISONS):
            rb, ra = selected.get((vb, bench)), selected.get((va, bench))
            if rb is None or ra is None:
                continue
            det, boot = compare(bench, rb, ra, bi, ci, a.n_boot, a.seed, a.alpha)
            comparisons_out.setdefault(bench, {})[f"{vb}_vs_{va}"] = det
            hm = det["metrics"].get(BENCHMARKS[bench][1], {})
            deltas.setdefault((vb, va), {})[bench] = (hm.get("diff_pp"), boot)
            log.metric({"event": "paired", "benchmark": bench, "comparison": f"{vb}_vs_{va}", "diff_pp": hm.get("diff_pp"), "ci": hm.get("ci"),
                        "mcnemar_p": (hm.get("mcnemar") or {}).get("p_exact_two_sided")})

    macro = {"absolute": {v: macro_entry(headline[v], a.alpha) for v in VERSIONS},
             "paired": {f"{vb}_vs_{va}": macro_entry(deltas.get((vb, va), {}), a.alpha) for vb, va in COMPARISONS}}
    warnings = [f"{v}/{b} {run.run_id}: {n}" for (v, b), run in selected.items() for n in run.notes]
    missing_cells = [f"{v}/{b}" for b in BENCHMARKS for v in VERSIONS if (v, b) not in selected]
    for w in warnings:
        log.error(w)
    result = {
        "generated_by": "scripts/summarize.py", "summarize_run_id": log.run_id, "generated_at": utc_now(),
        "config": {"n_boot": a.n_boot, "seed": a.seed, "alpha": a.alpha, "confidence": 1 - a.alpha,
                   "include_unreportable": a.include_unreportable, "results_root": str(a.results_root)},
        "rules": RULES, "runs": inventory, "absolute": absolute_out, "comparisons": comparisons_out, "macro_average": macro,
        "cells_without_run": missing_cells, "warnings": warnings,
    }
    rows = table_rows(absolute_out, comparisons_out, macro)
    write_csv_atomic(a.out_csv, CSV_COLUMNS, rows)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out_json, result)
    print_tables(rows)
    log.log(f"wrote {a.out_csv} ({len(rows)} rows) and {a.out_json}; {len(selected)} of {len(BENCHMARKS) * len(VERSIONS)} cells have a run; "
            f"{len(warnings)} warnings (errors.log)")
    return {"inputs": inputs, "outputs": [str(a.out_csv), str(a.out_json)], "n_cells": len(selected)}


def table_rows(absolute_out: dict[str, Any], comparisons_out: dict[str, Any], macro: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metric_list = [(b, BENCHMARKS[b][1]) for b in BENCHMARKS if b != "asibench"] + [("asibench", m) for m in ASI_METRICS]
    for bench, metric in metric_list:
        for version in VERSIONS:
            e = absolute_out.get(bench, {}).get(version)
            row: dict[str, Any] = {"benchmark": bench, "metric": metric, "version": version}
            if e is None:
                rows.append(row | {"notes": "no run"})
                continue
            m = e["metrics"][metric]
            fb, ct = e["failures"], e["time_tokens_cost"]
            row.update(run_id=e["run_id"], reportable=e["reportable"], value_pct=m["value_pct"], ci_method=m.get("ci_method"),
                       n_units=m["n_units"], n_successes=m.get("n_successes", m.get("n_scored")),
                       n_units_total=fb["n_units"], n_run_failures=fb["n_run_failures"], run_failure_rate_pct=fb["run_failure_rate_pct"],
                       n_environment_failure=fb["by_kind"]["environment_failure"], n_agent_error=fb["by_kind"]["model_agent_error"],
                       n_verifier_failure=fb["by_kind"]["verifier_failure"], n_scoring_failure=fb["by_kind"]["scoring_failure"],
                       n_missing=fb["by_kind"]["missing"], n_budget_exhausted=fb["by_kind"]["timeout_budget_exhausted"],
                       n_pending_external=fb["by_kind"]["pending_external"], n_skipped=fb["by_kind"]["skipped"],
                       **{k: ct[k] for k in ("mean_time_sec", "wall_sec_per_unit", "mean_input_tokens", "mean_output_tokens", "duration_h",
                                             "gpu_hours", "cost_usd", "cost_usd_per_unit")},
                       notes="; ".join(([m["source"]] if m.get("source") else []) + e["notes"]))
            if m.get("ci"):
                row["ci_low"], row["ci_high"] = m["ci"]
            for ref in ("base", "sft"):
                c = comparisons_out.get(bench, {}).get(f"{version}_vs_{ref}")
                cm = (c or {}).get("metrics", {}).get(metric) or {}
                if not cm:
                    continue
                row[f"delta_vs_{ref}_pp"] = cm.get("diff_pp")
                if cm.get("ci"):
                    row[f"delta_vs_{ref}_ci_low"], row[f"delta_vs_{ref}_ci_high"] = cm["ci"]
                row[f"mcnemar_p_vs_{ref}"] = (cm.get("mcnemar") or {}).get("p_exact_two_sided")
                row[f"n_paired_vs_{ref}"] = c["n_shared_units"]
            rows.append(row)
    for version in VERSIONS:
        ma = macro["absolute"][version]
        row = {"benchmark": "macro", "metric": "macro_average" if ma["complete"] else "macro_average_partial", "version": version,
               "value_pct": ma["value_pct"], "ci_method": "combined per-benchmark bootstrap" if ma["ci"] else None,
               "n_units": len(ma["covered"]), "notes": f"covers {', '.join(ma['covered']) or 'nothing'}" + (f"; {ma['ci_note']}" if ma.get("ci_note") else "")}
        if ma["ci"]:
            row["ci_low"], row["ci_high"] = ma["ci"]
        for ref in ("base", "sft"):
            mp = macro["paired"].get(f"{version}_vs_{ref}")
            if mp and mp["value_pct"] is not None:
                row[f"delta_vs_{ref}_pp"] = mp["value_pct"]
                if mp["ci"]:
                    row[f"delta_vs_{ref}_ci_low"], row[f"delta_vs_{ref}_ci_high"] = mp["ci"]
                row[f"n_paired_vs_{ref}"] = len(mp["covered"])
                if not mp["complete"]:
                    row["notes"] += f"; delta vs {ref} covers {', '.join(mp['covered'])}"
        rows.append(row)
    return rows


def fmt(x: Any, nd: int = 1) -> str:
    return "-" if x is None or x == "" else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_tables(rows: list[dict[str, Any]]) -> None:
    def ci(r: dict[str, Any], p: str) -> str:
        return f"[{fmt(r.get(p + 'low'))}, {fmt(r.get(p + 'high'))}]" if r.get(p + "low") is not None else ""

    print("\nScores (percent) with 95% intervals; deltas are paired (pp) with McNemar p")
    print(f"{'benchmark':15} {'metric':22} {'version':7} {'value':>6} {'interval':16} {'vs base':>7} {'interval':16} {'p':>7} "
          f"{'vs sft':>7} {'interval':16} {'p':>7}  reportable")
    for r in rows:
        print(f"{r['benchmark']:15} {r['metric']:22} {r['version']:7} {fmt(r.get('value_pct')):>6} {ci(r, 'ci_'):16} "
              f"{fmt(r.get('delta_vs_base_pp')):>7} {ci(r, 'delta_vs_base_ci_'):16} {fmt(r.get('mcnemar_p_vs_base'), 4):>7} "
              f"{fmt(r.get('delta_vs_sft_pp')):>7} {ci(r, 'delta_vs_sft_ci_'):16} {fmt(r.get('mcnemar_p_vs_sft'), 4):>7}  {fmt(r.get('reportable'))}")
    print("\nRun status (counts), time per unit (agent s; run wall s), mean tokens per unit and run cost")
    print(f"{'benchmark':15} {'version':7} {'units':>5} {'fail%':>6} {'env':>4} {'agent':>5} {'verif':>5} {'score':>5} {'miss':>4} "
          f"{'tmout':>5} {'pend':>4} {'agent s':>7} {'wall s':>7} {'in tok':>9} {'out tok':>8} {'GPU h':>6} {'USD':>7} {'USD/u':>6}")
    seen = set()
    for r in rows:
        if r["benchmark"] == "macro" or (r["benchmark"], r["version"]) in seen or not r.get("run_id"):
            continue
        seen.add((r["benchmark"], r["version"]))
        print(f"{r['benchmark']:15} {r['version']:7} {fmt(r.get('n_units_total')):>5} {fmt(r.get('run_failure_rate_pct')):>6} "
              f"{fmt(r.get('n_environment_failure')):>4} {fmt(r.get('n_agent_error')):>5} {fmt(r.get('n_verifier_failure')):>5} "
              f"{fmt(r.get('n_scoring_failure')):>5} {fmt(r.get('n_missing')):>4} {fmt(r.get('n_budget_exhausted')):>5} "
              f"{fmt(r.get('n_pending_external')):>4} {fmt(r.get('mean_time_sec'), 0):>7} {fmt(r.get('wall_sec_per_unit'), 0):>7} "
              f"{fmt(r.get('mean_input_tokens'), 0):>9} "
              f"{fmt(r.get('mean_output_tokens'), 0):>8} {fmt(r.get('gpu_hours'), 2):>6} {fmt(r.get('cost_usd'), 2):>7} "
              f"{fmt(r.get('cost_usd_per_unit'), 2):>6}")
    notes = [(r["benchmark"], r["version"], r["notes"]) for r in rows if r.get("run_id") and r.get("notes")]
    if notes:
        print("\nNotes")
        for b, v, n in dict.fromkeys(notes):
            print(f"  {b}/{v}: {n}")
    empty = [f"{r['version']}/{r['benchmark']}" for r in rows if r["benchmark"] != "macro" and not r.get("run_id")]
    if empty:
        print("\nCells without a run: " + ", ".join(dict.fromkeys(empty)))


# --- dev mode --------------------------------------------------------------------------------------------------------
def load_dev(path: Path, admitted: dict[str, set[str]] | None = None) -> tuple[Path, dict[str, dict[Key, float]], dict[str, Counter], int, Counter]:
    """A dev output (file or directory; module docstring) -> (file, slice -> {(task_id, sample_index): 0-100}, status
    counts, number of repeated items, items left out per slice because their task is not in `admitted`; the last line of
    a repeated (slice, task_id, sample_index) is used)."""
    file = path
    if path.is_dir():
        file = next((path / n for n in DEV_FILES if (path / n).exists()), None)
        if file is None:
            raise SystemExit(f"{path}: no {' or '.join(DEV_FILES)}")
    values: dict[str, dict[Key, float]] = {}
    statuses: dict[str, Counter] = {}
    left_out: Counter = Counter()
    n_dup = 0
    for i, line in enumerate(file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        sl = r.get("slice") or r.get("benchmark")
        sample = r.get("sample_index", r.get("sample"))
        if sl is None or r.get("task_id") is None:
            raise SystemExit(f"{file}:{i}: an item needs `slice` (or `benchmark`) and `task_id`")
        if admitted is not None and str(sl) in admitted and str(r["task_id"]) not in admitted[str(sl)]:
            left_out[str(sl)] += 1
            continue
        if isinstance(r.get("resolved"), bool):
            v = 100.0 * r["resolved"]
        elif r.get("score") is not None:
            v = 100 * float(r["score"]) / float(r["max_score"]) if r.get("max_score") else 100 * float(r["score"])
        else:
            v = 0.0
        key = (str(r["task_id"]), sample)
        n_dup += key in values.get(str(sl), {})
        values.setdefault(str(sl), {})[key] = v
        statuses.setdefault(str(sl), Counter())[str(r.get("run_status", "unknown"))] += 1
    return file, values, statuses, n_dup, left_out


def load_admitted(path: Path) -> tuple[dict[str, set[str]], str]:
    """The frozen admitted dev task lists: ({slice: admitted task ids}, admitted_sha256); the file's hash is verified."""
    from apex_data.devset import admission  # noqa: PLC0415

    frozen = admission.load(path)
    if frozen is None:
        raise SystemExit(f"--admitted {path}: no such file")
    return admission.admitted_set(frozen), frozen["admitted_sha256"]


def dev_slice(vals: dict[Key, float], n_boot: int, rng: np.random.Generator, alpha: float) -> tuple[dict[str, Any], np.ndarray | None]:
    """Score of one slice: Wilson over items (binary verdicts; assumes independent items) and a task-clustered bootstrap."""
    keys = sorted(vals, key=str)
    num, den, tasks = clusters(vals, keys, scored_only=False)
    binary = all(v in (0.0, 100.0) for v in vals.values())
    boot = cluster_bootstrap([(num, den, 1.0)], n_boot, rng) if keys else None
    k = sum(1 for v in vals.values() if v == 100.0)
    return {"n_items": len(keys), "n_tasks": len(tasks), "value_pct": rnd(ratio(num, den)) if keys else None,
            "wilson_ci": rnd_pair(wilson(k, len(keys), alpha)) if binary else None, "task_bootstrap_ci": rnd_pair(percentile_ci(boot, alpha))}, boot


def run_dev(a: argparse.Namespace, log: RunDir) -> dict[str, Any]:
    admitted, admitted_sha = load_admitted(a.admitted) if a.admitted else (None, None)
    run_file, run_vals, run_status, run_dup, run_out = load_dev(a.dev_run, admitted)
    base_file, base_vals, base_status, base_dup, base_out = load_dev(a.baseline, admitted) if a.baseline else (None, {}, {}, 0, Counter())
    for f, n in ((run_file, run_dup), (base_file, base_dup)):
        if n:
            log.error(f"{f}: {n} repeated (slice, task_id, sample_index) items; the last line of each is used")
    resolutions = {}
    for item in filter(None, (a.guard_resolutions or "").split(",")):
        name, _, pp = item.partition("=")
        resolutions[name.strip()] = float(pp)
    slices = sorted(set(run_vals) | set(base_vals))
    macro_slices = [s for s in (a.macro_slices.split(",") if a.macro_slices else slices) if s in run_vals and s in base_vals]
    out: dict[str, Any] = {"run": {"path": str(run_file), "label": a.label},
                           "baseline": {"path": str(base_file) if base_file else None, "label": a.baseline_label},
                           "config": {"n_boot": a.n_boot, "seed": a.seed, "alpha": a.alpha, "guard_point_pp": a.guard_point_pp, "resolutions_pp": resolutions},
                           "admitted": {"path": str(a.admitted) if a.admitted else None, "admitted_sha256": admitted_sha,
                                        "n_lines_not_admitted": {"run": dict(run_out), "baseline": dict(base_out)}},
                           "slices": {}}
    paired_boots: dict[str, tuple[float | None, Any]] = {}
    abs_boots: dict[str, dict[str, tuple[float | None, Any]]] = {"run": {}, "baseline": {}}
    for si, sl in enumerate(slices):
        e: dict[str, Any] = {}
        for side, vals, st, sidx in (("run", run_vals, run_status, 0), ("baseline", base_vals, base_status, 1)):
            if sl in vals:
                e[side], boot = dev_slice(vals[sl], a.n_boot, rng_for(a.seed, si, sidx), a.alpha)
                e[side]["status_counts"] = dict(st[sl])
                abs_boots[side][sl] = (e[side]["value_pct"], boot)
        if sl in run_vals and sl in base_vals:
            shared = sorted(set(run_vals[sl]) & set(base_vals[sl]), key=str)
            num_b, den_b, tasks = clusters(run_vals[sl], shared, False)
            num_a, den_a, _ = clusters(base_vals[sl], shared, False)
            p: dict[str, Any] = {"n_shared_items": len(shared), "n_shared_tasks": len(tasks),
                                 "n_items_only_run": len(set(run_vals[sl]) - set(base_vals[sl])),
                                 "n_items_only_baseline": len(set(base_vals[sl]) - set(run_vals[sl]))}
            if shared:
                boot = cluster_bootstrap([(num_b, den_b, 1.0), (num_a, den_a, -1.0)], a.n_boot, rng_for(a.seed, si, 2))
                diff = ratio(num_b, den_b) - ratio(num_a, den_a)
                ci = percentile_ci(boot, a.alpha)
                p.update(diff_pp=rnd(diff), ci=rnd_pair(ci), ci_method=f"paired task-clustered percentile bootstrap, {a.n_boot} resamples")
                if all(v in (0.0, 100.0) for k in shared for v in (run_vals[sl][k], base_vals[sl][k])):
                    lost = sum(1 for k in shared if base_vals[sl][k] == 100.0 and run_vals[sl][k] == 0.0)
                    gained = sum(1 for k in shared if base_vals[sl][k] == 0.0 and run_vals[sl][k] == 100.0)
                    p["mcnemar"] = {"n_lost": lost, "n_gained": gained, "p_exact_two_sided": rnd(mcnemar_exact(lost, gained), 5)}
                if sl in resolutions and ci is not None:
                    p["guard"] = {"point_ok": diff >= -a.guard_point_pp, "lower_bound_ok": ci[0] >= -resolutions[sl],
                                  "resolution_pp": resolutions[sl], "rule": f"diff >= -{a.guard_point_pp} pp and lower bound >= -{resolutions[sl]} pp"}
                paired_boots[sl] = (diff, boot)
            e["paired"] = p
        out["slices"][sl] = e
        log.metric({"event": "dev_slice", "slice": sl, **{side: e[side]["value_pct"] for side in ("run", "baseline") if side in e},
                    "diff_pp": (e.get("paired") or {}).get("diff_pp")})
    out["macro"] = {"slices": macro_slices,
                    "run": macro_of({s: abs_boots["run"][s] for s in macro_slices if s in abs_boots["run"]}, a.alpha),
                    "baseline": macro_of({s: abs_boots["baseline"][s] for s in macro_slices if s in abs_boots["baseline"]}, a.alpha),
                    "paired": macro_of({s: paired_boots[s] for s in macro_slices if s in paired_boots}, a.alpha)}
    out_path = log.run_dir / "dev_summary.json"
    write_json(out_path, out)
    if a.out:
        write_json(a.out, out)
    print_dev(out)
    log.log(f"dev summary of {run_file} vs {base_file}: {len(slices)} slices; wrote {out_path}" + (f" and {a.out}" if a.out else ""))
    inputs = [{"path": str(f), "sha256": sha256_file(f)} for f in (run_file, base_file, a.admitted) if f]
    return {"inputs": inputs, "outputs": [str(out_path)] + ([str(a.out)] if a.out else [])}


def macro_of(per_slice: dict[str, tuple[float | None, Any]], alpha: float) -> dict[str, Any]:
    vals = [v for v, _ in per_slice.values() if v is not None]
    boots = [b for _, b in per_slice.values() if isinstance(b, np.ndarray)]
    if not vals or len(vals) != len(per_slice):
        return {"value_pct": None, "ci": None}
    ci = percentile_ci(np.mean(boots, axis=0), alpha) if len(boots) == len(per_slice) else None
    return {"value_pct": rnd(fmean(vals)), "ci": rnd_pair(ci)}


def print_dev(out: dict[str, Any]) -> None:
    run_l, base_l = out["run"]["label"], out["baseline"]["label"]
    print(f"\nDev set: {run_l} vs {base_l} (percent; 95% intervals; paired by (task_id, sample_index))")
    print(f"{'slice':12} {'items':>5} {run_l[:10]:>10} {'interval':16} {base_l[:10]:>10} {'diff':>6} {'interval':16} {'McNemar p':>9}  guard")
    for sl, e in out["slices"].items():
        r, b, p = e.get("run") or {}, e.get("baseline") or {}, e.get("paired") or {}
        ci = r.get("task_bootstrap_ci")
        pci = p.get("ci")
        g = p.get("guard")
        print(f"{sl:12} {fmt(r.get('n_items')):>5} {fmt(r.get('value_pct')):>10} {(f'[{ci[0]}, {ci[1]}]' if ci else ''):16} "
              f"{fmt(b.get('value_pct')):>10} {fmt(p.get('diff_pp')):>6} {(f'[{pci[0]}, {pci[1]}]' if pci else ''):16} "
              f"{fmt((p.get('mcnemar') or {}).get('p_exact_two_sided'), 4):>9}  "
              f"{'' if not g else ('pass' if g['point_ok'] and g['lower_bound_ok'] else 'FAIL')}")
    m = out["macro"]
    print(f"{'macro':12} {'':>5} {fmt(m['run']['value_pct']):>10} {'':16} {fmt(m['baseline']['value_pct']):>10} "
          f"{fmt(m['paired']['value_pct']):>6} {(str(m['paired']['ci']) if m['paired']['ci'] else ''):16}  over {', '.join(m['slices'])}")


# --- quickcheck mode -------------------------------------------------------------------------------------------------
AGENT_SLICES = ("swe", "terminal")
QC_EPISODE_UP = 4   # an item's episode.json sits in its artifacts dir (SWE) or up to 3 levels above (terminal Harbor trial)
LONG_THINK = 3000   # "past 3K": the v7 overthinking guard's long reply and the back-to-acting threshold
LOOP_ENDING = "format_errors"   # the SWE harness's exit after three consecutive replies without a parsable tool call
STEP_CAP = "step_cap"


def parse_named_runs(items: list[str]) -> list[tuple[str, list[Path]]]:
    """LABEL=PATH[+PATH...] -> (label, paths); part runs of one evaluation are joined with '+'."""
    out = []
    for it in items:
        label, sep, paths = it.partition("=")
        if not sep or not label or not paths:
            raise SystemExit(f"--run takes LABEL=PATH[+PATH...], got {it!r}")
        out.append((label, [Path(p) for p in paths.split("+")]))
    if len({lbl for lbl, _ in out}) != len(out):
        raise SystemExit("--run labels must be unique")
    return out


def load_items(paths: list[Path]) -> tuple[list[Path], dict[tuple[str, str, Any], dict[str, Any]]]:
    """Dev records keyed (slice, task_id, sample_index); the last line of a repeated key wins (resumed runs)."""
    files, recs = [], {}
    for path in paths:
        f = next((path / n for n in DEV_FILES if (path / n).exists()), None) if path.is_dir() else path
        if f is None or not f.exists():
            raise SystemExit(f"{path}: no {' or '.join(DEV_FILES)}")
        files.append(f)
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(str(r.get("slice") or r.get("benchmark")), str(r["task_id"]), r.get("sample_index", r.get("sample")))] = r
    return files, recs


def find_episode(artifacts: str | None) -> Path | None:
    """A dev item's episode.json: in its artifacts directory or up to three levels above (the dev work root's agent
    transcripts, which hold no answer keys; never results/ files)."""
    d = Path(artifacts) if artifacts else None
    for _ in range(QC_EPISODE_UP if d else 0):
        if (d / "episode.json").exists():
            return d / "episode.json"
        d = d.parent
    return None


def episode_messages(path: Path | None) -> list[dict[str, Any]] | None:
    """The transcript (`messages`) of an episode.json, with tool calls stored as text parsed back to a list."""
    if path is None:
        return None
    try:
        msgs = json.loads(path.read_text()).get("messages") or []
    except (OSError, ValueError, AttributeError):
        return None
    for m in msgs:
        if isinstance(m.get("tool_calls"), str):
            try:
                m["tool_calls"] = json.loads(m["tool_calls"])
            except ValueError:
                m["tool_calls"] = None
    return msgs


def find_requests(artifacts: str | None) -> list[Path]:
    """The endpoint request logs of a dev item (requests/*.jsonl[.gz] in its artifacts directory or above): every model
    response the harness received, including the ones it rejected and left out of the transcript."""
    d = Path(artifacts) if artifacts else None
    for _ in range(QC_EPISODE_UP if d else 0):
        if (d / "requests").is_dir():
            return sorted(p for p in (d / "requests").iterdir() if p.name.endswith((".jsonl", ".jsonl.gz")))
        d = d.parent
    return []


def response_measures(files: list[Path], count: Any) -> list[dict[str, Any]]:
    """Per logged response: reasoning tokens (the server's usage count, else the counter on reasoning_content), whether
    it was cut at the per-turn token limit, and (tool-calling harnesses) whether it carried a tool call (`acted`; None
    for text-command harnesses). A request the server refused (non-200, e.g. the prompt no longer fits the context) is
    marked as an error, not a response."""
    import gzip  # noqa: PLC0415

    out = []
    for f in files:
        with (gzip.open(f, "rt") if f.name.endswith(".gz") else f.open()) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                resp = r.get("response") or {}
                if r.get("status") not in (None, 200) or "message" not in resp:   # a refused request (e.g. context overflow)
                    out.append({"error": True})
                    continue
                msg = resp.get("message") or {}
                usage = (resp.get("usage") or {}).get("completion_tokens_details") or {}
                think = usage.get("reasoning_tokens")
                if think is None:
                    think = count([msg.get("reasoning_content") or ""])[0]
                acted = bool(msg.get("tool_calls")) if r.get("has_tools") else None
                out.append({"error": False, "think": int(think), "truncated": resp.get("finish_reason") == "length",
                            "no_tool_call": acted is False, "acted": acted})
    return out


def item_measures(sl: str, r: dict[str, Any], count: Any) -> dict[str, Any]:
    """Per-item quantities the quick-check table aggregates. Agent slices: thinking per model response from the request
    logs (every response, also those the harness rejected; the transcript when no log exists), and actions, wasted
    actions and verification from the transcript with apex_data.efficiency (the stage-1 gates' and stage-2 rules'
    detectors). QA: one response whose reasoning tokens and truncation the runner recorded."""
    tm, ex = r.get("turn_metrics") or {}, r.get("extra") or {}
    steps = next((ex[k] for k in ("n_agent_steps", "steps") if ex.get(k) is not None), tm.get("n_turns"))
    m: dict[str, Any] = {"resolved": bool(r.get("resolved")), "status": str(r.get("run_status") or "unknown"),
                         "turns": int(tm.get("n_turns") or 0), "format_failures": int(tm.get("n_format_failures") or 0),
                         "exit_reason": ex.get("exit_reason"), "steps": int(steps) if steps is not None else None,
                         "output_tokens": r.get("output_tokens"), "thinking": [], "responses": [], "truncated": 0, "no_tool_call": 0,
                         "think_source": "none", "request_errors": 0, "actions": 0, "wasted": 0, "repeat_run": 0, "completed": False, "verified": False,
                         "transcript": None, "episode": None}
    if sl in AGENT_SLICES:
        from apex_data import efficiency  # noqa: PLC0415 - tokenizer and detectors, only in this mode

        ep = find_episode(r.get("artifacts"))
        msgs = episode_messages(ep)
        m["transcript"], m["episode"] = msgs is not None, str(ep) if msgs is not None else None
        if msgs:
            p = efficiency.profile(msgs, sl, count=count, root=efficiency.HARNESS_REPO_ROOT)
            m.update(thinking=list(p.thinking), think_source="transcript", actions=p.n_actions, wasted=p.wasted_actions,
                     completed=p.completed, verified=p.completed and p.verified,
                     responses=[{"think": t, "truncated": False, "acted": None} for t in p.thinking],   # accepted turns only
                     repeat_run=longest_repeat_run([efficiency.normalise(x.command) for x in efficiency.actions(msgs, sl)]))
        logged = response_measures(find_requests(r.get("artifacts")), count)
        m["request_errors"] = sum(x["error"] for x in logged)
        if resp := [x for x in logged if not x["error"]]:
            m.update(thinking=[x["think"] for x in resp], think_source="requests", truncated=sum(x["truncated"] for x in resp),
                     no_tool_call=sum(x["no_tool_call"] for x in resp), responses=resp)
    else:
        think = tm.get("think_tokens") if tm.get("think_unit") == "tokens" else r.get("reasoning_tokens")
        m.update(thinking=[int(think)] if think is not None else [], think_source="record", truncated=int(tm.get("n_truncated") or 0))
        m["responses"] = [{"think": t, "truncated": m["truncated"] > 0, "acted": None} for t in m["thinking"]]
    return m


def longest_repeat_run(commands: list[str]) -> int:
    """The longest run of one non-empty normalised command issued consecutively (an agent looping on one action)."""
    best = cur = 0
    prev = None
    for c in commands:
        cur = cur + 1 if c and c == prev else (1 if c else 0)
        prev = c
        best = max(best, cur)
    return best


def nearest_rank(xs: list[float], q: float) -> float | None:
    s = sorted(xs)
    return float(s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]) if s else None


def quick_aggregate(ms: list[dict[str, Any]], repeat_min: int = 20) -> dict[str, Any]:
    think = [t for m in ms for t in m["thinking"]]
    normal = [x["think"] for m in ms for x in m["responses"] if not x["truncated"]]
    long_ = [x for m in ms for x in m["responses"] if x["think"] > LONG_THINK]
    long_known = [x for x in long_ if x["acted"] is not None]
    turns, fails = sum(m["turns"] for m in ms), sum(m["format_failures"] for m in ms)
    acts, wasted = sum(m["actions"] for m in ms), sum(m["wasted"] for m in ms)
    completed = sum(m["completed"] for m in ms)
    statuses = Counter(m["status"] for m in ms)
    exits = Counter(str(m["exit_reason"]) for m in ms)
    out_tok = [m["output_tokens"] for m in ms if m["output_tokens"] is not None]
    steps = [m["steps"] for m in ms if m["steps"] is not None]
    n_pass = sum(m["resolved"] for m in ms)
    return {"n_items": len(ms), "n_pass": n_pass,
            "think_per_turn_mean": rnd(fmean(think), 1) if think else None, "think_per_turn_p50": nearest_rank(think, 0.5),
            "think_per_turn_p90": nearest_rank(think, 0.9),
            "normal_think_p50": nearest_rank(normal, 0.5), "normal_think_p90": nearest_rank(normal, 0.9), "n_normal_responses": len(normal),
            "long_think_min": LONG_THINK, "n_long_responses": len(long_),
            "long_response_pct": rnd(100 * len(long_) / len(think)) if think else None,
            "back_to_acting_pct": rnd(100 * sum(x["acted"] for x in long_known) / len(long_known)) if long_known else None,
            "n_loop_endings": exits[LOOP_ENDING], "n_step_cap": exits[STEP_CAP], "exit_reasons": dict(exits),
            "tokens_per_pass": rnd(sum(out_tok) / n_pass, 0) if n_pass and out_tok else None,
            "steps_per_episode_mean": rnd(fmean(steps), 1) if steps else None, "steps_per_episode_p50": nearest_rank(steps, 0.5),
            "n_responses": len(think), "truncated_response_pct": rnd(100 * sum(m["truncated"] for m in ms) / len(think)) if think else None,
            "no_tool_call_response_pct": rnd(100 * sum(m["no_tool_call"] for m in ms) / len(think)) if think else None,
            "think_sources": dict(Counter(m["think_source"] for m in ms)), "n_request_errors": sum(m["request_errors"] for m in ms),
            "n_turns": turns, "malformed_turn_pct": rnd(100 * fails / turns) if turns else None,
            "n_actions": acts, "wasted_action_pct": rnd(100 * wasted / acts) if acts else None,
            "repeat_run_min": repeat_min, "n_items_repeat_run": sum(1 for m in ms if m["repeat_run"] >= repeat_min),
            "longest_repeat_run": max((m["repeat_run"] for m in ms), default=0),
            "n_completed": completed, "verified_of_completed_pct": rnd(100 * sum(m["verified"] for m in ms) / completed) if completed else None,
            "n_budget_exhausted": statuses[BUDGET_EXHAUSTED], "n_run_failures": sum(statuses[s] for s in RUN_FAILURE_STATUSES),
            "status_counts": dict(statuses), "output_tokens_per_item": rnd(fmean(out_tok), 0) if out_tok else None,
            "n_missing_transcripts": sum(1 for m in ms if m["transcript"] is False)}


def run_quickcheck(a: argparse.Namespace, log: RunDir, count: Any = None) -> dict[str, Any]:
    """Quick-check table (training plan v5, sections 2 and 4): the first --run defines the items; every run is measured on
    them and each comparison on the items both runs have."""
    if count is None:
        from apex_data.efficiency import qwen_token_counts as count  # noqa: PLC0415
    runs = parse_named_runs(a.qc_runs)
    labels = [lbl for lbl, _ in runs]
    pairs = [tuple(c.split(":", 1)) for c in a.compare] if a.compare else [(labels[0], lbl) for lbl in labels[1:]]
    if any(len(p) != 2 or p[0] not in labels or p[1] not in labels for p in pairs):
        raise SystemExit(f"--compare takes A:B with labels from --run ({labels}), got {a.compare}")
    loaded = {lbl: load_items(paths) for lbl, paths in runs}
    items = sorted(loaded[labels[0]][1], key=str)
    slices = [s for s in ("swe", "terminal", "qa") if any(k[0] == s for k in items)]
    measures: dict[str, dict[tuple, dict[str, Any]]] = {}
    for lbl in labels:
        recs = loaded[lbl][1]
        measures[lbl] = {k: item_measures(k[0], recs[k], count) for k in items if k in recs}
        log.log(f"{lbl}: {len(measures[lbl])} of {len(items)} quick-check items present")
    out: dict[str, Any] = {"runs": {lbl: {"paths": [str(f) for f in loaded[lbl][0]]} for lbl in labels}, "items_from": labels[0],
                           "n_items": {s: sum(1 for k in items if k[0] == s) for s in slices}, "think_unit": "tokens (Qwen3.5-9B tokenizer)",
                           "per_run": {}, "comparisons": []}
    for lbl in labels:
        per = {s: quick_aggregate([m for k, m in measures[lbl].items() if k[0] == s], a.repeat_run) for s in slices}
        for s in slices:
            per[s]["n_quickcheck_items_missing"] = out["n_items"][s] - per[s]["n_items"]
        out["per_run"][lbl] = per
    for x, y in pairs:
        cmp_: dict[str, Any] = {"run": x, "reference": y, "slices": {}}
        for s in slices:
            shared = [k for k in items if k[0] == s and k in measures[x] and k in measures[y]]
            ax, ay = (quick_aggregate([measures[lbl][k] for k in shared], a.repeat_run) for lbl in (x, y))
            diff = lambda key: None if ax[key] is None or ay[key] is None else rnd(ax[key] - ay[key])  # noqa: E731
            cmp_["slices"][s] = {"n_shared": len(shared), "pass_diff": ax["n_pass"] - ay["n_pass"],
                                 "gained": sum(1 for k in shared if measures[x][k]["resolved"] and not measures[y][k]["resolved"]),
                                 "lost": sum(1 for k in shared if measures[y][k]["resolved"] and not measures[x][k]["resolved"]),
                                 "malformed_turn_pct_diff": diff("malformed_turn_pct"), "think_per_turn_mean_diff": diff("think_per_turn_mean"),
                                 "truncated_response_pct_diff": diff("truncated_response_pct"),
                                 "n_items_repeat_run_diff": ax["n_items_repeat_run"] - ay["n_items_repeat_run"],
                                 "n_loop_endings_diff": ax["n_loop_endings"] - ay["n_loop_endings"],
                                 "normal_think_p90_diff": diff("normal_think_p90"), "long_response_pct_diff": diff("long_response_pct"),
                                 "tokens_per_pass_diff": diff("tokens_per_pass"),
                                 "think_per_turn_p90_diff": diff("think_per_turn_p90"), "wasted_action_pct_diff": diff("wasted_action_pct")}
        sv = cmp_["slices"].values()
        cmp_["total"] = {"pass_diff": sum(v["pass_diff"] for v in sv), "gained": sum(v["gained"] for v in sv), "lost": sum(v["lost"] for v in sv)}
        out["comparisons"].append(cmp_)
    if a.read_items:
        src = a.read_items_from or labels[0]
        if src not in labels:
            raise SystemExit(f"--read-items-from {src!r} is not a --run label ({labels})")
        out["read_items"] = {"from": src, "rule": f"per agent slice, the {a.read_items} items with the most output tokens in {src}",
                             "items": []}
        for s in (sl for sl in AGENT_SLICES if sl in slices):
            ranked = sorted((k for k in items if k[0] == s and k in measures[src]),
                            key=lambda k: (-(measures[src][k]["output_tokens"] or 0), str(k)))[:a.read_items]
            for k in ranked:
                out["read_items"]["items"].append({"slice": s, "task_id": k[1], "sample_index": k[2], "runs": {
                    lbl: {f: measures[lbl][k][f] for f in ("resolved", "status", "output_tokens", "turns", "episode")}
                    for lbl in labels if k in measures[lbl]}})
    out_path = log.run_dir / "quickcheck_summary.json"
    write_json(out_path, out)
    if a.out:
        write_json(a.out, out)
    print_quickcheck(out)
    log.log(f"quick check of {', '.join(labels)} on {len(items)} items; wrote {out_path}" + (f" and {a.out}" if a.out else ""))
    inputs = [{"path": str(f), "sha256": sha256_file(f)} for lbl in labels for f in loaded[lbl][0]]
    return {"inputs": inputs, "outputs": [str(out_path)] + ([str(a.out)] if a.out else [])}


def print_quickcheck(out: dict[str, Any]) -> None:
    print(f"\nQuick check on the items of {out['items_from']} ({out['n_items']}); thinking in {out['think_unit']}")
    print(f"{'slice':9} {'run':14} {'items':>5} {'pass':>4} {'think/turn':>10} {'p90':>7} {'trunc%':>6} {'notool%':>7} "
          f"{'malformed%':>10} {'wasted%':>8} {'repeat':>6} {'verified%':>9} {'budget':>6} {'fail':>4} {'out tok/item':>12}")
    for sl in out["n_items"]:
        for lbl, per in out["per_run"].items():
            e = per[sl]
            print(f"{sl:9} {lbl[:14]:14} {e['n_items']:>5} {e['n_pass']:>4} {fmt(e['think_per_turn_mean'], 0):>10} "
                  f"{fmt(e['think_per_turn_p90'], 0):>7} {fmt(e['truncated_response_pct']):>6} {fmt(e['no_tool_call_response_pct']):>7} "
                  f"{fmt(e['malformed_turn_pct']):>10} {fmt(e['wasted_action_pct']):>8} {e['n_items_repeat_run']:>6} "
                  f"{fmt(e['verified_of_completed_pct']):>9} {e['n_budget_exhausted']:>6} {e['n_run_failures']:>4} "
                  f"{fmt(e['output_tokens_per_item'], 0):>12}")
    long_min = next((e["long_think_min"] for per in out["per_run"].values() for e in per.values()), LONG_THINK)
    print(f"\nOutcome measures (normal = responses not cut at the turn limit; long = thinking > {long_min} tokens; "
          f"back-to-acting = long responses that carry a tool call)")
    print(f"{'slice':9} {'run':14} {'pass':>4} {'tok/pass':>9} {'loops':>5} {'stepcap':>7} {'repeat':>6} {'timeouts':>8} "
          f"{'p50':>5} {'normal p50':>10} {'normal p90':>10} {'long%':>6} {'back-to-act%':>12} {'steps/ep':>8} {'p50':>5}")
    for sl in out["n_items"]:
        for lbl, per in out["per_run"].items():
            e = per[sl]
            print(f"{sl:9} {lbl[:14]:14} {e['n_pass']:>4} {fmt(e['tokens_per_pass'], 0):>9} {e['n_loop_endings']:>5} {e['n_step_cap']:>7} "
                  f"{e['n_items_repeat_run']:>6} {e['n_budget_exhausted']:>8} {fmt(e['think_per_turn_p50'], 0):>5} "
                  f"{fmt(e['normal_think_p50'], 0):>10} {fmt(e['normal_think_p90'], 0):>10} {fmt(e['long_response_pct']):>6} "
                  f"{fmt(e['back_to_acting_pct']):>12} {fmt(e['steps_per_episode_mean'], 1):>8} {fmt(e['steps_per_episode_p50'], 0):>5}")
    for c in out["comparisons"]:
        parts = [f"{sl} {v['pass_diff']:+d} (+{v['gained']}/-{v['lost']} of {v['n_shared']})" for sl, v in c["slices"].items()]
        print(f"{c['run']} vs {c['reference']}: passes {'; '.join(parts)}; total {c['total']['pass_diff']:+d} "
              f"(+{c['total']['gained']}/-{c['total']['lost']})")
    if out.get("read_items"):
        print(f"\nItems to read ({out['read_items']['rule']}):")
        for it in out["read_items"]["items"]:
            verdicts = ", ".join(f"{lbl} {'pass' if r['resolved'] else 'fail'} {fmt(r['output_tokens'], 0)} tok" for lbl, r in it["runs"].items())
            print(f"  {it['slice']:8} {it['task_id'][:60]:60} {verdicts}")


# --- entry point -----------------------------------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples (default 10,000)")
    common.add_argument("--seed", type=int, default=20261012, help="bootstrap seed (numpy; per-benchmark streams are derived from it)")
    common.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level of every interval")
    common.add_argument("--run-id", default=None, help="run id of this summarize run (default summarize_<UTC timestamp>)")
    common.add_argument("--log-root", type=Path, default=REPO_ROOT, help="SMOKE TESTS ONLY: write logs/<run_id>/ under this root")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("benchmarks", parents=[common], help="the 4 x 3 benchmark table (default mode)",
                       description="benchmarks mode; see `scripts/summarize.py --help`")
    b.add_argument("--results-root", type=Path, default=REPO_ROOT / "results", help="root of results/<version>/<benchmark>/<run_id>/")
    b.add_argument("--logs-root", type=Path, default=None, help="root of logs/<run_id>/ (default: <results-root>/../logs)")
    b.add_argument("--run", action="append", default=[], metavar="VERSION:BENCHMARK=RUN_ID", help="pin the run of one cell (repeatable)")
    b.add_argument("--portal-score", action="append", default=[], metavar="VERSION=PATH",
                   help="ASI-Bench Portal result file (named portal_score.json) for a version (default: inside the selected run's results dir)")
    b.add_argument("--include-unreportable", action="store_true", help="pilots and smoke tests: fall back to finished, then unfinished, unreportable runs")
    b.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "summary.csv")
    b.add_argument("--out-json", type=Path, default=REPO_ROOT / "analysis" / "uncertainty.json")
    d = sub.add_parser("dev", parents=[common], help="per-slice dev-set scores and the paired comparison against a baseline",
                       description="dev mode; see `scripts/summarize.py --help`")
    d.add_argument("--run", dest="dev_run", type=Path, required=True, help="dev output (directory with per_item.jsonl / per_task.jsonl, or a JSONL file)")
    d.add_argument("--baseline", type=Path, default=None, help="baseline dev output in the same format (e.g. Base's full dev run)")
    d.add_argument("--label", default="run")
    d.add_argument("--baseline-label", default="baseline")
    d.add_argument("--macro-slices", default=None, help="comma-separated slices of the macro-average (default: every slice in both; e.g. swe,terminal,qa)")
    d.add_argument("--guard-resolutions", default=None, help="slice=pp,... design resolutions for the do-no-harm guard, e.g. swe=7,terminal=12,qa=6")
    d.add_argument("--guard-point-pp", type=float, default=3.0, help="largest allowed point drop vs the baseline (data plan section 1)")
    d.add_argument("--out", type=Path, default=None, help="also write the dev summary JSON here")
    d.add_argument("--admitted", type=Path, default=None,
                   help="frozen admitted dev task lists (data/manifests/dev_admitted.json): SWE and terminal items of other tasks are left out")
    q = sub.add_parser("quickcheck", parents=[common], help="quick-check table of dev smoke runs: behaviour metrics, paired pass counts",
                       description="quickcheck mode; see `scripts/summarize.py --help`")
    q.add_argument("--run", dest="qc_runs", action="append", required=True, metavar="LABEL=PATH[+PATH]",
                   help="dev output (dir or JSONL; '+' joins part runs); the first --run defines the items (repeatable)")
    q.add_argument("--compare", action="append", default=[], metavar="A:B",
                   help="paired comparison of run A against reference B (default: the first run against each other run)")
    q.add_argument("--read-items", type=int, default=0, metavar="N",
                   help="list, per agent slice, the N items with the most output tokens, with every run's verdict and transcript path")
    q.add_argument("--read-items-from", default=None, metavar="LABEL",
                   help="the run whose output tokens choose the items to read (default: the first run), so later runs are read on the same items")
    q.add_argument("--repeat-run", type=int, default=20, metavar="N",
                   help="count items whose agent issues one command N or more times in a row (default 20)")
    q.add_argument("--out", type=Path, default=None, help="also write the quick-check JSON here")
    if not argv or (argv[0] not in ("benchmarks", "dev", "quickcheck", "-h", "--help")):
        argv = ["benchmarks", *argv]
    a = p.parse_args(argv)
    if a.n_boot < 1 or not 0 < a.alpha < 1:
        p.error("--n-boot must be positive and --alpha in (0, 1)")
    if a.mode == "benchmarks" and a.logs_root is None:
        a.logs_root = a.results_root.resolve().parent / "logs"
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(list(sys.argv[1:] if argv is None else argv))
    run_id = a.run_id or new_run_id({"benchmarks": "summarize", "dev": "summarize_dev", "quickcheck": "summarize_qc"}[a.mode])
    with RunDir(run_id, a.log_root, log_name="eval.log") as log:
        log.write_metadata(run_id=run_id, script="scripts/summarize.py", mode=a.mode, code=git_info(), hardware=hardware_info(),
                           config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                           model_revision={"base_lock": model_lock(), "note": "per-version model names are in analysis/uncertainty.json"},
                           seed=a.seed, command=command_line(), start_time=utc_now(), status="running")
        out = {"benchmarks": run_benchmarks, "dev": run_dev, "quickcheck": run_quickcheck}[a.mode](a, log)
        duration = log.elapsed()
        log.write_resources([("duration_sec", round(duration, 1), "s", "wall clock (CPU only)"),
                             ("peak_gpu_memory_gb", "n/a", "GB", "no GPU used"), ("gpu_hours", 0, "h", "no GPU used"),
                             ("cost_usd", 0, "USD", "no GPU used; the CPU host is not charged to runs")])
        log.write_metadata(data_version={"inputs": out["inputs"]}, outputs=out["outputs"], end_time=utc_now(),
                           duration_sec=round(duration, 1), status="finished")
        log.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
