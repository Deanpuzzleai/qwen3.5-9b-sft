"""Mixture accounting and the budget rule (docs/DATA_PLAN.md revision 4.3 section 4; WP7, unit U2.4).

Shares are stated in loss tokens, the tokens that carry gradient (`n_assistant_tokens`, the sum of the assistant mask):
the trainer's loss is token-weighted (sum of token losses / trained tokens per accumulation window), so a slice's
gradient share is its loss-token share. Sequence tokens set the training time. Targets: SWE ~67 / terminal ~18 / retention ~15 percent, checked at
+-2 percentage points.

The budget rule: the main SFT run must finish inside `max_hours` (plan: ~7.5 to 8 h) at the throughput measured at the
chosen cap (M4.1, T7). Every sequence token of the mixture (training rows and validation shard, as the plan's sizing
table counts them) is costed at that throughput. When the mixture does not fit, whole units are trimmed in this order:
1. Python SWE rows: SWE-rebench-V2 Python first, then Scale-SWE, each the last drawn first (the draw puts large tasks
   first, so the random-phase rows go first); Go / JS / TS rows are never trimmed;
2. retention rows, proportionally across its parts, down to a 12 percent loss-token floor;
3. whole terminal episodes, TerminalHorizon before TermGrade, proportionally across the category-quota groups.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from apex_data.paths import SEED
from apex_data.render.select import SLICES, Unit, rank

TARGET_LOSS_SHARES = {"swe": 0.67, "terminal": 0.18, "retention": 0.15}
TOLERANCE = 0.02
RETENTION_FLOOR = 0.12
THROUGHPUT_TOKENS_PER_S = 9000.0          # measured on real 64K rows on GPU-T (M4.1's synthetic T7 gave 8.8K)
MAX_SFT_HOURS = 8.0
EVAL_STEPS = 25                           # configs/sft.yaml eval.eval_steps
EFFECTIVE_BATCH = 16                      # configs/sft.yaml optim.effective_batch_size
EVAL_SPEEDUP = 3.0                        # [I] a forward-only eval pass vs a training step with gradient checkpointing
TOKEN_BINS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)
PROVENANCE_KEYS = ("hf_repo", "generator", "label_provenance", "dataset_license", "oswt_subsource", "size_stratum",
                   "language", "repo_license", "retention_part", "category_group", "source")
NORMALISATION_NOTE = ("The trainer's loss is token-weighted: the sum of token losses divided by the trained (assistant-mask) tokens "
                      "of the accumulation window, so a slice's gradient share equals its loss-token share; sequence-token shares "
                      "set the training time. Both are recorded.")


def _share(part: float, total: float) -> float:
    return round(part / total, 4) if total else 0.0


def histogram(values: Iterable[int], bins: Sequence[int] = TOKEN_BINS) -> dict[str, int]:
    labels = [f"<={b}" for b in bins] + [f">{bins[-1]}"]
    c: Counter[str] = Counter()
    for v in values:
        c[next((f"<={b}" for b in bins if v <= b), labels[-1])] += 1
    return {k: c[k] for k in labels}


def account(units: Iterable[Unit], rows: Mapping[str, Mapping[str, Any]], *, targets: Mapping[str, float] = TARGET_LOSS_SHARES,
            tolerance: float = TOLERANCE, provenance: bool = True) -> dict[str, Any]:
    """Rows, units, tasks, sequence and loss tokens per slice and per cell; loss and sequence shares; the tolerance check;
    per-provenance row counts and loss-token shares; sequence-length histograms."""
    units = list(units)
    per: dict[str, dict[str, Any]] = {}
    cells: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"units": 0, "rows": 0, "sequence_tokens": 0, "loss_tokens": 0}))
    for s in SLICES:
        us = [u for u in units if u.slice == s]
        per[s] = {"rows": sum(len(u.row_keys) for u in us), "units": len(us), "tasks": len({u.task_id for u in us}),
                  "sequence_tokens": sum(u.seq_tokens for u in us), "loss_tokens": sum(u.loss_tokens for u in us)}
        for u in us:
            c = cells[s][u.cell]
            c["units"] += 1
            c["rows"] += len(u.row_keys)
            c["sequence_tokens"] += u.seq_tokens
            c["loss_tokens"] += u.loss_tokens
    tot = {k: sum(p[k] for p in per.values()) for k in ("rows", "units", "tasks", "sequence_tokens", "loss_tokens")}
    shares_loss = {s: _share(per[s]["loss_tokens"], tot["loss_tokens"]) for s in SLICES}
    shares_seq = {s: _share(per[s]["sequence_tokens"], tot["sequence_tokens"]) for s in SLICES}
    deviation = {s: round(shares_loss[s] - targets[s], 4) for s in SLICES if s in targets}
    out: dict[str, Any] = {
        "slices": {s: {**per[s], "loss_share": shares_loss[s], "sequence_share": shares_seq[s],
                       "loss_tokens_per_row": round(per[s]["loss_tokens"] / per[s]["rows"], 1) if per[s]["rows"] else 0.0,
                       "sequence_tokens_per_row": round(per[s]["sequence_tokens"] / per[s]["rows"], 1) if per[s]["rows"] else 0.0,
                       "cells": {c: dict(v) for c, v in sorted(cells[s].items())}} for s in SLICES},
        "totals": tot, "shares_by_loss_tokens": shares_loss, "shares_by_sequence_tokens": shares_seq,
        "targets": dict(targets), "tolerance_pp": round(100 * tolerance, 1),
        "deviation_pp": {s: round(100 * d, 2) for s, d in deviation.items()},
        "within_tolerance": all(abs(d) <= tolerance + 1e-9 for d in deviation.values()),
    }
    if provenance:
        out["per_provenance"] = provenance_counts(units, rows)
        out["sequence_length_histogram"] = {s: histogram(int(rows[k]["n_tokens"]) for u in units if u.slice == s for k in u.row_keys)
                                            for s in SLICES}
    return out


def provenance_counts(units: Iterable[Unit], rows: Mapping[str, Mapping[str, Any]], keys: Sequence[str] = PROVENANCE_KEYS) -> dict[str, Any]:
    """{key: {value: {rows, loss_tokens, row_share, loss_share}}} over the rows of `units` (None values as "none")."""
    keyset = [k for u in units for k in u.row_keys]
    total_loss = sum(int(rows[k]["n_assistant_tokens"]) for k in keyset) or 1
    out: dict[str, Any] = {}
    for key in keys:
        c: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "loss_tokens": 0})
        for k in keyset:
            v = rows[k].get(key)
            label = "none" if v is None else str(v)
            c[label]["rows"] += 1
            c[label]["loss_tokens"] += int(rows[k]["n_assistant_tokens"])
        out[key] = {v: {**d, "row_share": _share(d["rows"], len(keyset)), "loss_share": _share(d["loss_tokens"], total_loss)}
                    for v, d in sorted(c.items())}
    return out


def source_table(units: Iterable[Unit], rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows and tokens per (slice, dataset, revision, dataset licence, generator, label provenance): the data card's
    sources table, taken from the rows' own provenance."""
    keyf = ("hf_repo", "hf_revision", "dataset_license", "generator", "label_provenance")
    acc: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"rows": 0, "sequence_tokens": 0, "loss_tokens": 0})
    for u in units:
        for k in u.row_keys:
            r = rows[k]
            sub = r.get("retention_part") or r.get("oswt_subsource") or r.get("source")
            a = acc[(u.slice, *(r.get(f) for f in keyf), sub)]
            a["rows"] += 1
            a["sequence_tokens"] += int(r["n_tokens"])
            a["loss_tokens"] += int(r["n_assistant_tokens"])
    return [{"slice": key[0], **dict(zip(keyf, key[1:-1])), "part": key[-1], **v} for key, v in sorted(acc.items(), key=lambda kv: tuple(map(str, kv[0])))]


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2


ADDED_BINS = (0, 10, 25, 50, 100, 200, 500, 1000)
FILES_BINS = (1, 2, 3, 5, 10, 20)


def swe_profile(units: Iterable[Unit], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """SWE rows by language x size stratum, repositories, patch-size medians and histograms (plan section 8 tables)."""
    rs = [rows[k] for u in units if u.slice == "swe" for k in u.row_keys]
    by_lang: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rs:
        by_lang[str(r.get("language"))].append(r)
    by_lang["all"] = rs

    def prof(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        added = [int(r.get("patch_added_lines") or 0) for r in group]
        files = [int(r.get("patch_files") or 0) for r in group]
        repos = Counter(str(r.get("repo")) for r in group)
        return {"rows": len(group), "strata": dict(sorted(Counter(str(r.get("size_stratum")) for r in group).items())),
                "subsources": dict(sorted(Counter(str(r.get("oswt_subsource")) for r in group).items())),
                "repositories": len(repos), "max_rows_per_repository": max(repos.values(), default=0),
                "median_added_lines": _median(added), "share_added_ge100": _share(sum(a >= 100 for a in added), len(group)),
                "share_files_ge3": _share(sum(f >= 3 for f in files), len(group)),
                "added_lines_histogram": histogram(added, ADDED_BINS), "files_histogram": histogram(files, FILES_BINS)}

    return {lang: prof(g) for lang, g in sorted(by_lang.items())}


# --- budget rule -------------------------------------------------------------------------------------------------------
@dataclass
class BudgetResult:
    kept: list[Unit]
    report: dict[str, Any]
    removed: list[Unit] = field(default_factory=list)


def predicted_hours(seq_tokens: int, throughput: float) -> float:
    return round(seq_tokens / throughput / 3600, 3) if throughput > 0 else math.inf


def eval_overhead(train_rows: int, dev_seq_tokens: int, throughput: float, *, eval_steps: int = EVAL_STEPS,
                  effective_batch: int = EFFECTIVE_BATCH, speedup: float = EVAL_SPEEDUP) -> dict[str, Any]:
    """Estimate of the eval-loss passes over the validation shard (every `eval_steps` optimizer steps and at the end);
    outside the budget rule, reported next to it."""
    steps = math.ceil(train_rows / effective_batch) if train_rows else 0
    n_evals = steps // eval_steps + (1 if steps % eval_steps else 0)
    tokens = n_evals * dev_seq_tokens
    return {"optimizer_steps": steps, "eval_passes": n_evals, "eval_forward_tokens": tokens, "assumed_speedup": speedup,
            "hours": round(tokens / (throughput * speedup) / 3600, 3) if throughput > 0 else math.inf,
            "note": "estimate [I]: forward-only passes assumed ~3x faster per token than a training step with gradient checkpointing"}


def _proportional_order(units: Sequence[Unit], group_of: Any, seed: int, label: str) -> list[Unit]:
    """Removal order that keeps the groups' shares: repeatedly take the group with the largest remaining fraction of its
    original size, its last unit in seeded order."""
    groups: dict[str, list[Unit]] = defaultdict(list)
    for u in sorted(units, key=lambda u: (rank(seed, label, u.unit_id), u.unit_id)):
        groups[str(group_of(u))].append(u)
    size = {g: len(v) for g, v in groups.items()}
    order = []
    while any(groups.values()):
        g = max((g for g in groups if groups[g]), key=lambda g: (len(groups[g]) / size[g], g))
        order.append(groups[g].pop())
    return order


def apply_budget_rule(units: Sequence[Unit], *, throughput: float = THROUGHPUT_TOKENS_PER_S, max_hours: float = MAX_SFT_HOURS,
                      retention_floor: float = RETENTION_FLOOR, seed: int = SEED) -> BudgetResult:
    """Trim whole units in the plan's order until every sequence token of the mixture fits `max_hours` at `throughput`."""
    budget = int(throughput * max_hours * 3600)
    kept = list(units)
    seq = sum(u.seq_tokens for u in kept)
    loss = {s: sum(u.loss_tokens for u in kept if u.slice == s) for s in SLICES}
    report: dict[str, Any] = {"throughput_tokens_per_s": throughput, "max_hours": max_hours, "budget_sequence_tokens": budget,
                              "sequence_tokens_before": seq, "predicted_hours_before": predicted_hours(seq, throughput),
                              "retention_floor": retention_floor, "trimmed": seq > budget, "stages": []}
    removed: list[Unit] = []
    removed_ids: set[str] = set()

    def stage(name: str, order: Sequence[Unit], allow: Any = None) -> None:
        nonlocal seq
        n = rows = s_tok = l_tok = 0
        for u in order:
            if seq <= budget:
                break
            if allow is not None and not allow(u):
                break
            removed_ids.add(u.unit_id)
            removed.append(u)
            seq -= u.seq_tokens
            loss[u.slice] -= u.loss_tokens
            n, rows, s_tok, l_tok = n + 1, rows + len(u.row_keys), s_tok + u.seq_tokens, l_tok + u.loss_tokens
        report["stages"].append({"stage": name, "units_removed": n, "rows_removed": rows, "sequence_tokens_removed": s_tok,
                                 "loss_tokens_removed": l_tok})

    if seq > budget:
        for cell in ("py_rebench", "py_scale"):
            order = sorted((u for u in kept if u.slice == "swe" and u.cell == cell), key=lambda u: -u.order)
            stage(f"swe_{cell}", order)
        ret = [u for u in kept if u.slice == "retention"]

        def above_floor(u: Unit) -> bool:
            total = sum(loss.values()) - u.loss_tokens
            return total > 0 and (loss["retention"] - u.loss_tokens) / total >= retention_floor

        stage("retention_to_floor", _proportional_order(ret, lambda u: u.cell, seed, "budget_retention"), allow=above_floor)
        for src in ("th3k", "tgr"):
            eps = [u for u in kept if u.slice == "terminal" and u.cell == src]
            stage(f"terminal_{src}", _proportional_order(eps, lambda u: u.stratum or u.cell, seed, f"budget_terminal_{src}"))
    kept = [u for u in kept if u.unit_id not in removed_ids]
    report.update({"sequence_tokens_after": seq, "predicted_hours_after": predicted_hours(seq, throughput),
                   "budget_met": seq <= budget, "units_removed": len(removed),
                   "rows_removed": sum(len(u.row_keys) for u in removed)})
    return BudgetResult(kept, report, removed)
