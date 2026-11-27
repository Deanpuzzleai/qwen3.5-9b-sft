#!/usr/bin/env python3
"""Rebuild the training figures and reports from the curve source data in curves/source/.

Curves are rebuilt from source data in curves/source/ by this script. It works with whatever runs exist (SFT
only, SFT and one stage-2 arm, ...), with runs still in progress and with runs written before the training monitor existed:
every figure is drawn from the sources that exist and skipped (logged) when its source is missing.

Usage (from the repository root):
  .venv/train/bin/python scripts/plot_curves.py [--source curves/source] [--out analysis/figures] [--runs RUN_ID ...]
      [--label RUN_ID=NAME ...] [--smooth 10] [--bin-steps 10] [--checkpoint-every 25] [--include-dryrun]
      [--compare NAME=RUN_ID,RUN_ID[,...] ...] [--no-reports]

Inputs (read-only):
  curves/source/<run_id>_train.csv   one row per optimizer step (src/apex_train/common.py BASE_TRAIN_COLUMNS plus method
                                     extras): step, epoch, loss, learning_rate, grad_norm, tokens_this_step,
                                     tokens_cumulative, tokens_per_s, step_time_s, elapsed_s, peak_gpu_mem_allocated_gb,
                                     peak_gpu_mem_reserved_gb, rss_gb[, entropy, mean_token_accuracy] (SFT and stage 2 alike).
  curves/source/<run_id>_eval.csv    one row per validation-loss evaluation: step, epoch, eval_loss, eval_entropy,
                                     eval_mean_token_accuracy, eval_runtime_s, elapsed_s
  curves/source/<run_id>_health.csv  the training monitor's per-step fields (src/apex_train/monitor.py HEALTH_COLUMNS): loss
                                     EMA, raw micro-batch loss sum, planned LR, clip threshold, trained (loss) tokens, data wait,
                                     GPU memory (current and peak), latest GPU sample
  curves/source/<run_id>_slices.csv  train loss per step and slice and eval loss per evaluation and slice, both weightings
                                     (loss_tokens, loss_rows); the eval slice `all` is the whole evaluation; SLICE_COLUMNS
  curves/source/<run_id>_rows.csv    one record per trained / evaluated row (row loss = mean NLL per loss token); ROW_COLUMNS
  curves/source/<run_id>_lora.csv    adapter norms, update ratios and |delta W| / |W0| per module family; LORA_COLUMNS
  curves/source/<run_id>_gpu.csv     nvidia-smi samples (util, power, temperature, SM clock, memory, throttle reasons)
  logs/<run_id>/metrics.jsonl        optional: the run's `checkpoint` events (apex_train TrainMetrics.on_save) place the
                                     checkpoint markers. Without them the markers follow the configured strategy: every
                                     --checkpoint-every steps and the last step for SFT runs (configs/sft.yaml save_steps 25),
                                     50 and 100 percent of the steps for stage-2 runs (docs/DATA_PLAN.md section 11).
  logs/<run_id>/alerts.jsonl         optional: the monitor's alerts (alert timeline on the loss figure, report tables)
  logs/<run_id>/{metadata.json, status.json, resources.csv}   optional: the summary and health of the HTML reports
  Runs are grouped by run_id prefix into panels: sft_* (the SFT pilot and main run), stage2_* (the stage-2 arms
  stage2_positives_* and stage2_control_*), others.
  dryrun_* runs (CPU smoke tests) are skipped unless --include-dryrun.
Outputs (under --out, default analysis/figures/, overwritten on every run):
  loss_vs_step.png / .csv            training loss per step (faint) with its trailing mean over --smooth steps, validation
                                     loss, checkpoint markers; one panel per group, every run in its own colour
  loss_vs_tokens.png / .csv          the same against cumulative training tokens (validation points at the tokens seen by
                                     their step)
  training_dynamics.png / .csv       learning rate, gradient norm, throughput (tokens/s) and peak reserved GPU memory vs step
  runs_summary.csv                   per run: steps, epochs, tokens, wall time, median tokens/s, peak GPU memory, final
                                     training loss (smoothed), best and final validation loss, checkpoints
  <run_id>/ (one per run; not with --no-reports)
    loss.png                         training loss (raw and EMA) vs step with eval loss (overall and per slice), checkpoints
                                     and the alert timeline; the same vs trained (loss) tokens
    slice_loss.png                   train loss per slice binned over --bin-steps steps (from the per-row records) with the eval
                                     loss per slice; eval loss per slice relative to its first evaluation (the forgetting check)
    row_loss.png                     row loss vs row length (coloured by slice) and its distribution per slice in the early /
                                     middle / late third of the run
    lr_grad.png                      learning rate vs the planned schedule; gradient norm (log) with the clip threshold and
                                     clipped steps; clip rate
    lora.png                         |lora_A|, |lora_B|, update ratio and |delta W| / |W0| per module family
    throughput.png                   sequence and trained tokens/s, step time, data-wait share
    gpu.png                          GPU memory (allocated / reserved / peak, GPU total) vs step; utilisation, power,
                                     temperature, SM clock vs time
    token_stats.png                  token entropy and accuracy (train and eval), when logged
    report.html                      summary table, health, alerts table, every figure embedded (base64 PNG)
  compare_<name>/ (per --compare, else automatic: all SFT runs when there are two or more, all stage-2 runs when two or more)
    compare_loss.png, compare_slices.png, compare_dynamics.png, report.html
  Every figure's .csv holds exactly the plotted numbers in long format:
  figure, panel, run_id, experiment, series, x_name, x, y_name, y.
  logs/<run_id>/                     metadata.json (inputs with sha256), eval.log, metrics.jsonl, errors.log, resources.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from apex_eval.common import (RunDir, command_line, compute_rates, git_info, hardware_info, new_run_id, sha256_file,  # noqa: E402
                              utc_now)

KINDS = ("train", "eval", "health", "slices", "rows", "lora", "gpu")   # TrainMetrics (train, eval) + TrainingMonitor (the rest)
CURVE_RE = re.compile(rf"^(?P<run_id>.+)_(?P<kind>{'|'.join(KINDS)})\.csv$")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")
GROUPS = (("sft_", "SFT"), ("stage2_", "Stage 2 (continued SFT arms)"))
# categorical slots in fixed order (one colour per run, the same in every figure); validated reference palette, light mode
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE, TEXT, TEXT_2, GRID, MARKER_NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#52514e"
MUTED = "#898781"
STATUS = {"INFO": MUTED, "WARN": "#fab219", "CRITICAL": "#d03b3b", "OK": "#0ca30c"}   # reserved status colours (never a series)
SLICE_ORDER = ("swe_episode", "terminal_turn", "retention")   # fixed slots 1-3 (the all-pairs-validated three)
FAMILY_ORDER = ("attention", "gdn", "mlp", "other")
DYNAMICS = (("learning_rate", "learning rate", False), ("grad_norm", "gradient norm", True),
            ("tokens_per_s", "throughput (tokens/s)", True), ("peak_gpu_mem_reserved_gb", "peak reserved GPU memory (GB)", False))
LONG_COLUMNS = ["figure", "panel", "run_id", "experiment", "series", "x_name", "x", "y_name", "y"]


@dataclass
class Curve:
    run_id: str
    experiment: str
    group: str
    color: str = ""
    train: dict[str, np.ndarray] = field(default_factory=dict)     # column -> values, rows sorted by step (last row per step)
    eval: dict[str, np.ndarray] = field(default_factory=dict)
    health: dict[str, np.ndarray] = field(default_factory=dict)    # monitor fields per step (same convention)
    slices: list[dict[str, str]] = field(default_factory=list)     # long tables as read (strings)
    rows: list[dict[str, str]] = field(default_factory=list)
    lora: list[dict[str, str]] = field(default_factory=list)
    gpu: list[dict[str, str]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    checkpoints: list[int] = field(default_factory=list)
    checkpoint_source: str = ""
    files: list[Path] = field(default_factory=list)

    @property
    def steps(self) -> np.ndarray:
        return self.train.get("step", np.array([]))


def num(x: Any) -> float:
    try:
        return float(x) if x not in (None, "") else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_curve_csv(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """A curves CSV -> {column: float array} sorted by step; a step logged twice (resumed run) keeps its last row."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_step: dict[float, dict[str, str]] = {}
    for r in rows:
        try:
            by_step[float(r["step"])] = r
        except (KeyError, TypeError, ValueError):
            continue
    ordered = [by_step[s] for s in sorted(by_step)]
    cols = list(rows[0]) if rows else []
    return {c: np.array([num(r.get(c)) for r in ordered], dtype=float) for c in cols}, len(rows) - len(ordered)


def read_table(path: Path) -> list[dict[str, str]]:
    """A long-format CSV as dicts of strings; a truncated last line (run in progress) is dropped."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if None not in r.values()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def trailing_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` finite values up to each point (NaN-aware); window 1 returns y."""
    if window <= 1 or len(y) == 0:
        return y.copy()
    out = np.full(len(y), math.nan)
    for i in range(len(y)):
        seg = y[max(0, i - window + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def group_of(run_id: str) -> str:
    return next((name for prefix, name in GROUPS if run_id.startswith(prefix)), "Other runs")


def checkpoint_steps(curve: Curve, logs_root: Path, every: int) -> None:
    """Checkpoint markers: the run's logged checkpoint events, else the configured strategy (module docstring)."""
    metrics = logs_root / curve.run_id / "metrics.jsonl"
    if metrics.exists():
        steps = [int(rec["step"]) for rec in read_jsonl(metrics) if rec.get("event") == "checkpoint" and rec.get("step") is not None]
        if steps:
            curve.checkpoints, curve.checkpoint_source = sorted(set(steps)), "logged checkpoint events (logs/<run_id>/metrics.jsonl)"
            return
    if not len(curve.steps):
        return
    last = int(np.nanmax(curve.steps))
    if curve.run_id.startswith("stage2_"):
        curve.checkpoints = sorted({max(1, round(last / 2)), last})
        curve.checkpoint_source = "strategy: 50 and 100 percent of the steps"
    else:
        curve.checkpoints = sorted({*range(every, last + 1, every), last})
        curve.checkpoint_source = f"strategy: every {every} steps and the last step"


def discover(a: argparse.Namespace, log: RunDir) -> list[Curve]:
    files: dict[str, dict[str, Path]] = {}
    for p in sorted(a.source.glob("*.csv")):
        m = CURVE_RE.match(p.name)
        if not m:
            log.error(f"{p.name}: not a <run_id>_<{'|'.join(KINDS)}>.csv curve file; skipped")
            continue
        files.setdefault(m["run_id"], {})[m["kind"]] = p
    curves = []
    for run_id in sorted(files):
        if a.runs and run_id not in a.runs:
            continue
        if run_id.startswith("dryrun_") and not a.include_dryrun:
            log.log(f"{run_id}: dry run (CPU smoke test); skipped (--include-dryrun to plot it)")
            continue
        if not {"train", "eval"} & set(files[run_id]):
            log.error(f"{run_id}: monitor files without a train or eval curve; skipped")
            continue
        exp = a.labels.get(run_id, TIMESTAMP_RE.sub("", run_id))
        c = Curve(run_id, exp, group_of(run_id))
        for kind, path in sorted(files[run_id].items(), key=lambda kv: KINDS.index(kv[0])):
            try:
                if kind in ("train", "eval", "health"):
                    data, n_dup = read_curve_csv(path)
                    setattr(c, kind, data)
                    if n_dup and kind != "health":
                        log.error(f"{path.name}: {n_dup} repeated steps (resumed run); the last row of each step is plotted")
                else:
                    setattr(c, kind, read_table(path))
                c.files.append(path)
            except (OSError, csv.Error, UnicodeDecodeError) as exc:
                log.error(f"{path.name}: unreadable ({exc}); skipped")
        if "train" not in files[run_id]:
            log.error(f"{run_id}: eval curve without a train curve; plotted without training loss")
        run_logs = a.logs_root / run_id
        alerts_path = run_logs / "alerts.jsonl"
        c.alerts = read_jsonl(alerts_path)
        if alerts_path.is_file():
            c.files.append(alerts_path)
        c.meta, c.status = read_json(run_logs / "metadata.json"), read_json(run_logs / "status.json")
        if (run_logs / "resources.csv").is_file():
            c.resources = {r["metric"]: r["value"] for r in read_table(run_logs / "resources.csv") if "metric" in r}
        checkpoint_steps(c, a.logs_root, a.checkpoint_every)
        curves.append(c)
    missing = sorted(set(a.runs or []) - {c.run_id for c in curves})
    if missing:
        log.error(f"--runs names runs without curve files: {missing}")
    if len(curves) > len(PALETTE):
        raise SystemExit(f"{len(curves)} runs exceed the {len(PALETTE)} distinguishable colours; select runs with --runs")
    for c, color in zip(curves, PALETTE):
        c.color = color
    return curves


# --- plotting --------------------------------------------------------------------------------------------------------
def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


class LongTable:
    """The plotted numbers of one figure, in the long format of the module docstring."""

    def __init__(self, figure: str):
        self.figure, self.rows = figure, []

    def add(self, panel: str, c: Curve, series: str, x_name: str, x: np.ndarray, y_name: str, y: np.ndarray) -> None:
        for xi, yi in zip(x, y):
            if math.isfinite(xi) and math.isfinite(yi):
                self.rows.append([self.figure, panel, c.run_id, c.experiment, series, x_name, repr(float(xi)), y_name, repr(float(yi))])

    def write(self, path: Path) -> None:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(LONG_COLUMNS)
            w.writerows(self.rows)


def new_figure(n_rows: int, n_cols: int, width: float = 6.4, height: float = 3.8) -> tuple[Any, np.ndarray]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False, facecolor=SURFACE)
    return fig, axes


def save(fig: Any, table: LongTable, out: Path, name: str, log: RunDir, tight: bool = True) -> list[Path]:
    import matplotlib.pyplot as plt
    if tight:   # gridspec figures with explicit spacing skip it
        fig.tight_layout()
    png, data = out / f"{name}.png", out / f"{name}.csv"
    fig.savefig(png, dpi=150, facecolor=SURFACE, bbox_inches="tight")   # legends sit below the panels
    plt.close(fig)
    table.write(data)
    log.log(f"wrote {png} and {data} ({len(table.rows)} plotted points)")
    return [png, data]


def loss_figure(curves: list[Curve], x_col: str, name: str, smooth: int, out: Path, log: RunDir) -> list[Path]:
    """Training (per step and smoothed) and validation loss per group panel, with checkpoint markers."""
    from matplotlib.lines import Line2D
    groups = list(dict.fromkeys(c.group for c in curves))
    fig, axes = new_figure(1, len(groups))
    table = LongTable(name)
    x_label = "optimizer step" if x_col == "step" else "training tokens (millions)"
    scale = 1.0 if x_col == "step" else 1e-6
    for ax, group in zip(axes[0], groups):
        handles = []
        for c in (c for c in curves if c.group == group):
            steps = c.steps
            x_train = c.train.get(x_col, np.full(len(steps), math.nan)) * scale
            if "loss" in c.train and len(steps):
                raw, sm = c.train["loss"], trailing_mean(c.train["loss"], smooth)
                ax.plot(x_train, raw, color=c.color, linewidth=0.8, alpha=0.3)
                ax.plot(x_train, sm, color=c.color, linewidth=2)
                table.add(group, c, "train_loss", x_label, x_train, "loss", raw)
                table.add(group, c, f"train_loss_mean{smooth}", x_label, x_train, "loss", sm)
                handles.append(Line2D([], [], color=c.color, linewidth=2, label=f"{c.experiment}: training loss (mean of {smooth} steps)"))
                logged_steps = set(steps.astype(int))
                ck = np.array([s for s in c.checkpoints if s in logged_steps], dtype=float)
                if len(ck):
                    idx = np.searchsorted(steps, ck)
                    ax.plot(x_train[idx], sm[idx], linestyle="none", marker="D", markersize=6, markerfacecolor=SURFACE,
                            markeredgecolor=c.color, markeredgewidth=1.6, zorder=4)
                    table.add(group, c, "checkpoint", x_label, x_train[idx], "loss", sm[idx])
            if "eval_loss" in c.eval and len(c.eval["eval_loss"]):
                e_steps = c.eval["step"]
                if x_col == "step":
                    x_eval = e_steps
                elif len(steps):   # tokens seen by the last training step at or before the evaluation
                    pos = np.searchsorted(steps, e_steps, side="right") - 1
                    x_eval = np.where(pos >= 0, x_train[np.clip(pos, 0, None)], math.nan)
                else:
                    x_eval = np.full(len(e_steps), math.nan)
                ax.plot(x_eval, c.eval["eval_loss"], color=c.color, linewidth=1.6, linestyle="--", marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
                table.add(group, c, "validation_loss", x_label, x_eval, "loss", c.eval["eval_loss"])
                handles.append(Line2D([], [], color=c.color, linewidth=1.6, linestyle="--", marker="o", markersize=5,
                                      label=f"{c.experiment}: validation loss"))
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=MARKER_NEUTRAL,
                              markeredgewidth=1.6, label="checkpoint"))
        style(ax, group, x_label, "loss (mean token cross-entropy)")
        ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    fig.suptitle("Training and validation loss" + (" vs step" if x_col == "step" else " vs training tokens"), color=TEXT, fontsize=12)
    return save(fig, table, out, name, log)


def dynamics_figure(curves: list[Curve], smooth: int, out: Path, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    groups = list(dict.fromkeys(c.group for c in curves if len(c.steps)))
    name = "training_dynamics"
    table = LongTable(name)
    if not groups:
        return []
    fig, axes = new_figure(len(DYNAMICS), len(groups), height=2.8)
    for col, group in enumerate(groups):
        members = [c for c in curves if c.group == group and len(c.steps)]
        for row, (column, label, smoothed) in enumerate(DYNAMICS):
            ax = axes[row][col]
            for c in members:
                y = c.train.get(column)
                if y is None or not np.isfinite(y).any():
                    continue
                panel = f"{group} / {label}"
                if smoothed:
                    sm = trailing_mean(y, smooth)
                    ax.plot(c.steps, y, color=c.color, linewidth=0.8, alpha=0.3)
                    ax.plot(c.steps, sm, color=c.color, linewidth=2)
                    table.add(panel, c, column, "optimizer step", c.steps, label, y)
                    table.add(panel, c, f"{column}_mean{smooth}", "optimizer step", c.steps, label, sm)
                else:
                    ax.plot(c.steps, y, color=c.color, linewidth=2)
                    table.add(panel, c, column, "optimizer step", c.steps, label, y)
            style(ax, f"{group}: {label}" if row == 0 else label, "optimizer step" if row == len(DYNAMICS) - 1 else "", label)
            if row == len(DYNAMICS) - 1:
                ax.legend(handles=[Line2D([], [], color=c.color, linewidth=2, label=c.experiment) for c in members], fontsize=8,
                          frameon=False, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3)
    fig.suptitle(f"Training dynamics (faint: per step; solid: mean of {smooth} steps where noisy)", color=TEXT, fontsize=12)
    return save(fig, table, out, name, log)


def runs_summary(curves: list[Curve], smooth: int, path: Path) -> None:
    def last_finite(y: np.ndarray | None) -> float | None:
        if y is None:
            return None
        f = y[np.isfinite(y)]
        return float(f[-1]) if len(f) else None

    def max_finite(y: np.ndarray | None) -> float | None:
        return float(np.nanmax(y)) if y is not None and np.isfinite(y).any() else None

    cols = ["run_id", "experiment", "group", "steps", "epochs", "tokens_total", "wall_h", "median_tokens_per_s",
            "peak_gpu_mem_reserved_gb", f"final_train_loss_mean{smooth}", "best_validation_loss", "best_validation_step",
            "final_validation_loss", "checkpoints", "checkpoint_source"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in curves:
            ev = c.eval.get("eval_loss")
            best = int(np.nanargmin(ev)) if ev is not None and np.isfinite(ev).any() else None
            tps = c.train.get("tokens_per_s")
            tps_f = [float(x) for x in tps if math.isfinite(x)] if tps is not None else []
            elapsed = max_finite(c.train.get("elapsed_s"))
            w.writerow({"run_id": c.run_id, "experiment": c.experiment, "group": c.group,
                        "steps": int(np.nanmax(c.steps)) if len(c.steps) else "", "epochs": max_finite(c.train.get("epoch")),
                        "tokens_total": max_finite(c.train.get("tokens_cumulative")),
                        "wall_h": round(elapsed / 3600, 3) if elapsed is not None else None,
                        "median_tokens_per_s": round(median(tps_f), 1) if tps_f else None,
                        "peak_gpu_mem_reserved_gb": max_finite(c.train.get("peak_gpu_mem_reserved_gb")),
                        f"final_train_loss_mean{smooth}": last_finite(trailing_mean(c.train["loss"], smooth)) if "loss" in c.train else None,
                        "best_validation_loss": float(ev[best]) if best is not None else None,
                        "best_validation_step": int(c.eval["step"][best]) if best is not None else None,
                        "final_validation_loss": last_finite(ev), "checkpoints": " ".join(map(str, c.checkpoints)),
                        "checkpoint_source": c.checkpoint_source})


# --- monitor data helpers --------------------------------------------------------------------------------------------
def color_map(names: list[str], order: tuple[str, ...]) -> dict[str, str]:
    """Fixed slots for the known names (slot i for order[i]), the free slots for the others in sorted order; past the eighth
    name: muted ink (never a generated hue)."""
    out = {n: PALETTE[i] for i, n in enumerate(order) if n in names and i < len(PALETTE)}
    free = [p for p in PALETTE if p not in out.values()]
    for n, p in zip(sorted(n for n in names if n not in out), free):
        out[n] = p
    return {n: out.get(n, MUTED) for n in names}


def column(rows: list[dict[str, str]], name: str, **where: str) -> np.ndarray:
    return np.array([num(r.get(name)) for r in rows if all(r.get(k) == v for k, v in where.items())], dtype=float)


def on_steps(src: dict[str, np.ndarray], col: str, steps: np.ndarray) -> np.ndarray:
    """Values of `col` of a per-step table (health) at `steps` (NaN where missing)."""
    if col not in src or "step" not in src:
        return np.full(len(steps), math.nan)
    lookup = dict(zip(src["step"].tolist(), src[col].tolist()))
    return np.array([lookup.get(float(s), math.nan) for s in steps], dtype=float)


def slice_key(name: str) -> tuple[int, str]:
    return (SLICE_ORDER.index(name) if name in SLICE_ORDER else len(SLICE_ORDER), name)


ALL_SLICE = "all"   # the monitor's eval slice holding the whole evaluation (not a data slice)


def slice_names(c: Curve) -> list[str]:
    names = {r["slice"] for r in c.slices if r.get("slice")} | {r["slice"] for r in c.rows if r.get("slice")}
    return sorted(names - {ALL_SLICE}, key=slice_key)


def loss_normalisation(c: Curve) -> str:
    """tokens / rows as the run recorded it (metadata loss_normalisation.mode, else status.json); rows for older runs."""
    mode = (c.meta.get("loss_normalisation") or {}).get("mode") if isinstance(c.meta.get("loss_normalisation"), dict) else None
    return mode or c.status.get("loss_normalisation") or "rows"


def binned_slice_loss(c: Curve, bin_steps: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """slice -> (bin centre step, token-weighted mean row loss of the bin's training rows); from the per-row records, or the
    per-step slice table when the run has none (the same numbers: both are token-weighted)."""
    src = [r for r in c.rows if r.get("split") == "train"]
    loss_key, w_key = ("loss", "n_loss_tokens") if src else ("loss_tokens", "n_loss_tokens")
    if not src:
        src = [r for r in c.slices if r.get("split") == "train"]
    acc: dict[str, dict[int, list[float]]] = {}
    for r in src:
        step, loss, w = num(r.get("step")), num(r.get(loss_key)), num(r.get(w_key))
        if not (math.isfinite(step) and math.isfinite(loss) and math.isfinite(w) and w > 0):
            continue
        b = int((step - 1) // bin_steps)
        v = acc.setdefault(r["slice"], {}).setdefault(b, [0.0, 0.0])
        v[0] += loss * w
        v[1] += w
    out = {}
    for name, bins in acc.items():
        keys = sorted(bins)
        out[name] = (np.array([k * bin_steps + (bin_steps + 1) / 2 for k in keys], dtype=float),
                     np.array([bins[k][0] / bins[k][1] for k in keys], dtype=float))
    return out


def eval_weighting(c: Curve) -> str:
    """The eval weighting that matches the run's training objective: tokens or rows."""
    return "tokens" if loss_normalisation(c) == "tokens" else "rows"


def eval_slices(c: Curve, weighting: str | None = None, *, include_all: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """slice -> (eval steps, eval loss) in `weighting` (default: the run's objective weighting); `all` only on request."""
    col = "loss_tokens" if (weighting or eval_weighting(c)) == "tokens" else "loss_rows"
    out: dict[str, list[tuple[float, float]]] = {}
    for r in c.slices:
        if r.get("split") == "eval" and (include_all or r.get("slice") != ALL_SLICE):
            out.setdefault(r["slice"], []).append((num(r.get("step")), num(r.get(col))))
    return {k: (np.array([p[0] for p in sorted(v)]), np.array([p[1] for p in sorted(v)])) for k, v in out.items()}


def eval_overall_tokens(c: Curve) -> tuple[np.ndarray, np.ndarray] | None:
    """The token-weighted eval loss over all evaluated rows (slice `all`), or None for runs that did not record it."""
    return eval_slices(c, "tokens", include_all=True).get(ALL_SLICE)


def trained_tokens_x(c: Curve) -> tuple[np.ndarray, str]:
    """x for 'vs tokens' panels: cumulative trained (loss) tokens from the monitor, else sequence tokens (millions)."""
    tt = on_steps(c.health, "tokens_trained_cumulative", c.steps)
    if np.isfinite(tt).any():
        return tt * 1e-6, "trained (loss) tokens (millions)"
    return c.train.get("tokens_cumulative", np.full(len(c.steps), math.nan)) * 1e-6, "sequence tokens (millions)"


def x_at_steps(steps: np.ndarray, x: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Cumulative-token x of evaluations at optimizer steps `at`: the value after the last logged step <= at; an
    evaluation before the first optimizer step (eval on start, step 0) sits at 0 tokens."""
    if not len(steps):
        return np.where(np.asarray(at) <= 0, 0.0, math.nan)
    pos = np.searchsorted(steps, at, side="right") - 1
    return np.where(pos >= 0, x[np.clip(pos, 0, None)], np.where(np.asarray(at) <= 0, 0.0, math.nan))


def loss_ema(c: Curve, smooth: int) -> tuple[np.ndarray, str]:
    ema = on_steps(c.health, "loss_ema", c.steps)
    if np.isfinite(ema).any():
        return ema, "EMA (monitor)"
    return trailing_mean(c.train.get("loss", np.full(len(c.steps), math.nan)), smooth), f"mean of {smooth} steps"


def legend_below(ax: Any, handles: list[Any], ncol: int = 3, offset: float = -0.2) -> None:
    if handles:
        ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(0, offset), ncol=ncol)


def no_data(ax: Any, title: str, text: str) -> None:
    style(ax, title, "", "")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=TEXT_2, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def title_of(c: Curve) -> str:
    return c.experiment if c.experiment == c.run_id else f"{c.experiment} ({c.run_id})"


def mark_alerts(ax: Any, c: Curve) -> None:
    """WARN / CRITICAL alerts as thin vertical lines in their status colour (labelled in the timeline strip)."""
    for al in c.alerts:
        if al.get("severity") in ("WARN", "CRITICAL") and isinstance(al.get("step"), (int, float)):
            ax.axvline(al["step"], color=STATUS[al["severity"]], linewidth=1, alpha=0.6, zorder=1)


# --- per-run figures -------------------------------------------------------------------------------------------------
def fig_loss(c: Curve, out: Path, a: argparse.Namespace, log: RunDir, slice_colors: dict[str, str]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not len(c.steps) and not c.eval:
        return []
    table = LongTable("loss")
    fig = plt.figure(figsize=(12.8, 10.0), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.0, 1.25, 4.0], hspace=0.55)
    ax, axa, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    steps, raw = c.steps, c.train.get("loss", np.array([]))
    ema, ema_label = loss_ema(c, a.smooth)
    x_tok, tok_label = trained_tokens_x(c)
    ev, ev_tok, weighting = eval_slices(c), eval_overall_tokens(c), eval_weighting(c)
    handles = []
    for axis, x, x_name in ((ax, steps, "optimizer step"), (axt, x_tok, tok_label)):
        panel = "vs step" if axis is ax else "vs tokens"
        if len(steps) and len(raw):
            axis.plot(x, raw, color=TEXT_2, linewidth=0.8, alpha=0.35)
            axis.plot(x, ema, color=TEXT, linewidth=2)
            table.add(panel, c, "train_loss", x_name, x, "loss", raw)
            table.add(panel, c, "train_loss_ema", x_name, x, "loss", ema)
            ck = np.array([s for s in c.checkpoints if s in set(steps.astype(int))], dtype=float)
            if len(ck):
                idx = np.searchsorted(steps, ck)
                axis.plot(x[idx], ema[idx], linestyle="none", marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=TEXT,
                          markeredgewidth=1.6, zorder=4)
                table.add(panel, c, "checkpoint", x_name, x[idx], "loss", ema[idx])
        if "eval_loss" in c.eval and len(c.eval["eval_loss"]):
            xe = c.eval["step"] if axis is ax else x_at_steps(steps, x, c.eval["step"])
            axis.plot(xe, c.eval["eval_loss"], color=TEXT_2, linewidth=1.6, linestyle="--", marker="o", markersize=5, markeredgecolor=SURFACE, zorder=3)
            table.add(panel, c, "eval_loss", x_name, xe, "loss", c.eval["eval_loss"])
        if ev_tok is not None:
            xe = ev_tok[0] if axis is ax else x_at_steps(steps, x, ev_tok[0])
            axis.plot(xe, ev_tok[1], color=TEXT, linewidth=1.6, linestyle="--", marker="s", markersize=5, markeredgecolor=SURFACE, zorder=3)
            table.add(panel, c, "eval_loss_tokens", x_name, xe, "loss", ev_tok[1])
        for name, (s_steps, s_loss) in sorted(ev.items(), key=lambda kv: slice_key(kv[0])):
            xe = s_steps if axis is ax else x_at_steps(steps, x, s_steps)
            axis.plot(xe, s_loss, color=slice_colors.get(name, MUTED), linewidth=1.2, linestyle=":", marker="o", markersize=4, zorder=3)
            table.add(panel, c, f"eval_loss_{name}", x_name, xe, "loss", s_loss)
    mark_alerts(ax, c)
    handles += [Line2D([], [], color=TEXT_2, linewidth=0.8, alpha=0.5, label="training loss per step"),
                Line2D([], [], color=TEXT, linewidth=2, label=f"training loss, {ema_label}")]
    if "eval_loss" in c.eval:
        handles.append(Line2D([], [], color=TEXT_2, linewidth=1.6, linestyle="--", marker="o", label="eval loss, row-weighted (Trainer eval_loss)"))
    if ev_tok is not None:
        handles.append(Line2D([], [], color=TEXT, linewidth=1.6, linestyle="--", marker="s", label="eval loss, token-weighted (all eval rows)"))
    handles += [Line2D([], [], color=slice_colors.get(n, MUTED), linewidth=1.2, linestyle=":", marker="o", label=f"eval loss ({weighting}-weighted): {n}")
                for n in sorted(ev, key=slice_key)]
    handles.append(Line2D([], [], linestyle="none", marker="D", markerfacecolor=SURFACE, markeredgecolor=TEXT, label="checkpoint"))
    if any(al.get("severity") in ("WARN", "CRITICAL") for al in c.alerts):
        handles += [Line2D([], [], color=STATUS[s], linewidth=1, label=f"{s} alert") for s in ("WARN", "CRITICAL")]
    style(ax, f"loss vs optimizer step (training loss normalisation: {loss_normalisation(c)})", "optimizer step", "loss (mean NLL per loss token)")
    style(axt, "loss vs " + tok_label.replace(" (millions)", ""), tok_label, "loss (mean NLL per loss token)")
    legend_below(axt, handles, ncol=3, offset=-0.18)
    # alert timeline strip, same x range as the step panel
    levels = {"INFO": 0, "WARN": 1, "CRITICAL": 2}
    seen: set[str] = set()
    span = max(float(np.nanmax(steps)) if len(steps) else 1.0, 1.0)
    last_label: dict[int, tuple[float, int]] = {}   # severity level -> (x of the last label, its stagger slot)
    for al in sorted(c.alerts, key=lambda x: (x.get("step") if isinstance(x.get("step"), (int, float)) else -1)):
        st, sev = al.get("step"), al.get("severity")
        if not isinstance(st, (int, float)) or sev not in levels:
            continue
        axa.plot([st], [levels[sev]], linestyle="none", marker="v" if sev == "CRITICAL" else "o", markersize=7,
                 color=STATUS[sev], markeredgecolor=TEXT if sev == "CRITICAL" else STATUS[sev])
        table.add("alert timeline", c, f"alert_{sev}_{al.get('code')}", "optimizer step", np.array([st], float), "severity level",
                  np.array([levels[sev]], float))
        if al.get("code") not in seen and sev != "INFO":
            seen.add(al.get("code"))
            prev = last_label.get(levels[sev])
            slot = (prev[1] + 1) % 3 if prev and st - prev[0] < 0.12 * span else 0
            last_label[levels[sev]] = (st, slot)
            axa.annotate(al.get("code", ""), (st, levels[sev]), textcoords="offset points", xytext=(5, 4 + 8 * slot), fontsize=6.5, color=TEXT_2)
    style(axa, f"alert timeline ({len(c.alerts)} alerts; logs/{c.run_id}/alerts.jsonl)", "optimizer step", "")
    axa.set_yticks([0, 1, 2], ["INFO", "WARN", "CRITICAL"])
    axa.set_ylim(-0.6, 3.4)
    if len(steps):
        ax.set_xlim(0, max(float(np.nanmax(steps)), 1.0) * 1.02)
        axa.set_xlim(ax.get_xlim())
    fig.suptitle(f"{title_of(c)}: training and eval loss", color=TEXT, fontsize=12)
    return save(fig, table, out, "loss", log, tight=False)


def fig_slice_loss(c: Curve, out: Path, a: argparse.Namespace, log: RunDir, slice_colors: dict[str, str]) -> list[Path]:
    from matplotlib.lines import Line2D
    train, ev = binned_slice_loss(c, a.bin_steps), eval_slices(c)
    if not train and not ev:
        return []
    table = LongTable("slice_loss")
    fig, axes = new_figure(1, 2, height=4.2)
    ax, axr = axes[0]
    handles = []
    for name in sorted(set(train) | set(ev), key=slice_key):
        col = slice_colors.get(name, MUTED)
        if name in train:
            x, y = train[name]
            ax.plot(x, y, color=col, linewidth=2, marker="." if len(x) < 30 else None)
            table.add("train per slice", c, f"train_{name}", f"optimizer step (bin centre, {a.bin_steps} steps)", x, "loss", y)
        if name in ev:
            x, y = ev[name]
            ax.plot(x, y, color=col, linewidth=1.2, linestyle="--", marker="o", markersize=5, markeredgecolor=SURFACE)
            table.add("train per slice", c, f"eval_{name}", "optimizer step", x, "loss", y)
            if len(y) and math.isfinite(y[0]) and y[0] != 0:
                rel = 100.0 * (y / y[0] - 1.0)
                axr.plot(x, rel, color=col, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
                table.add("eval vs first", c, f"eval_change_{name}", "optimizer step", x, "change vs first eval (%)", rel)
        handles.append(Line2D([], [], color=col, linewidth=2, label=name))
    thr = 100.0 * float(((c.meta.get("monitor_config") or {}).get("forgetting_rel_increase")) or 0.05)
    axr.axhline(0, color=MUTED, linewidth=0.8)
    axr.axhline(thr, color=STATUS["WARN"], linewidth=1.2, linestyle="--")
    handles_r = [Line2D([], [], color=STATUS["WARN"], linewidth=1.2, linestyle="--", label=f"WARN threshold +{thr:.0f}% (eval_slice_regression)")]
    style(ax, f"{title_of(c)}: loss per slice", "optimizer step", "loss (mean NLL per loss token)")
    style(axr, f"eval loss per slice ({eval_weighting(c)}-weighted) vs its first evaluation", "optimizer step", "change (%)")
    if not ev:
        axr.text(0.5, 0.5, "no per-slice evaluation", transform=axr.transAxes, ha="center", color=TEXT_2, fontsize=9)
    legend_below(ax, handles + [Line2D([], [], color=TEXT_2, linewidth=2, label=f"train, token-weighted, {a.bin_steps}-step bins"),
                                Line2D([], [], color=TEXT_2, linewidth=1.2, linestyle="--", marker="o", label=f"eval ({eval_weighting(c)}-weighted)")], ncol=3)
    legend_below(axr, handles_r, ncol=1)
    return save(fig, table, out, "slice_loss", log)


def fig_row_loss(c: Curve, out: Path, a: argparse.Namespace, log: RunDir, slice_colors: dict[str, str]) -> list[Path]:
    from matplotlib.lines import Line2D
    train = [r for r in c.rows if r.get("split") == "train" and math.isfinite(num(r.get("loss")))]
    evals = [r for r in c.rows if r.get("split") == "eval" and math.isfinite(num(r.get("loss")))]
    if not train and not evals:
        return []
    table = LongTable("row_loss")
    fig, axes = new_figure(1, 2, height=4.4)
    ax, axb = axes[0]
    names = sorted({r["slice"] for r in train + evals}, key=slice_key)
    for name in names:
        col = slice_colors.get(name, MUTED)
        tr = [r for r in train if r["slice"] == name]
        if tr:
            x, y = column(tr, "n_tokens"), column(tr, "loss")
            ax.scatter(x, y, s=10, color=col, alpha=0.55, linewidths=0)
            table.add("loss vs length", c, f"train_{name}", "row tokens", x, "row loss", y)
        er = [r for r in evals if r["slice"] == name]
        if er:
            x, y = column(er, "n_tokens"), column(er, "loss")
            ax.scatter(x, y, s=22, facecolors="none", edgecolors=col, linewidths=1.0)
            table.add("loss vs length", c, f"eval_{name}", "row tokens", x, "row loss", y)
    if train and np.nanmax(column(train, "n_tokens")) > 10 * max(np.nanmin(column(train, "n_tokens")), 1):
        ax.set_xscale("log")
    style(ax, f"{title_of(c)}: row loss vs row length", "row length (tokens)", "row loss (mean NLL per loss token)")
    legend_below(ax, [Line2D([], [], linestyle="none", marker="o", color=slice_colors.get(n, MUTED), label=n) for n in names]
                 + [Line2D([], [], linestyle="none", marker="o", color=TEXT_2, alpha=0.55, label="train row (filled)"),
                    Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=TEXT_2, label="eval row (hollow)")], ncol=3)
    if train:
        steps = column(train, "step")
        last = float(np.nanmax(steps))
        edges = (last / 3, 2 * last / 3)
        thirds = ("early", "middle", "late")
        data, pos, cols = [], [], []
        width = 0.8 / max(len(names), 1)
        for t_i, third in enumerate(thirds):
            for s_i, name in enumerate(names):
                ys = np.array([num(r["loss"]) for r in train if r["slice"] == name
                               and (0 if num(r["step"]) <= edges[0] else 1 if num(r["step"]) <= edges[1] else 2) == t_i], dtype=float)
                ys = ys[np.isfinite(ys)]
                if len(ys):
                    data.append(ys)
                    p = t_i + (s_i - (len(names) - 1) / 2) * width
                    pos.append(p)
                    cols.append(slice_colors.get(name, MUTED))
                    table.add("distribution by third", c, f"{third}_{name}", "third (0 early, 1 middle, 2 late)", np.full(len(ys), float(t_i)),
                              "row loss", ys)
        if data:
            bp = axb.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=True,
                             medianprops={"color": TEXT, "linewidth": 1.2}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.6)
                patch.set_edgecolor(col)
            axb.set_xticks(range(3), [f"{t}\n(steps {lo:.0f}-{hi:.0f})" for t, lo, hi in
                                      zip(thirds, (0, edges[0], edges[1]), (edges[0], edges[1], last))])
        style(axb, "training row loss by slice, early / middle / late third", "", "row loss")
        legend_below(axb, [Line2D([], [], linewidth=6, color=slice_colors.get(n, MUTED), alpha=0.6, label=n) for n in names], ncol=3, offset=-0.25)
    else:
        no_data(axb, "training row loss by third", "no training rows")
    return save(fig, table, out, "row_loss", log)


def fig_lr_grad(c: Curve, out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    steps = c.steps
    if not len(steps):
        return []
    table = LongTable("lr_grad")
    fig, axes = new_figure(3, 1, width=12.8, height=2.9)
    ax_lr, ax_g, ax_c = axes[:, 0]
    lr, planned = c.train.get("learning_rate", np.full(len(steps), math.nan)), on_steps(c.health, "lr_planned", steps)
    ax_lr.plot(steps, lr, color=c.color, linewidth=2)
    table.add("learning rate", c, "learning_rate", "optimizer step", steps, "learning rate", lr)
    hl = [Line2D([], [], color=c.color, linewidth=2, label="learning rate (logged)")]
    if np.isfinite(planned).any():
        ax_lr.plot(steps, planned, color=TEXT, linewidth=1.2, linestyle="--")
        table.add("learning rate", c, "lr_planned", "optimizer step", steps, "learning rate", planned)
        hl.append(Line2D([], [], color=TEXT, linewidth=1.2, linestyle="--", label="planned schedule (get_scheduler on the planned steps)"))
    style(ax_lr, f"{title_of(c)}: learning rate vs plan", "", "learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    legend_below(ax_lr, hl, ncol=2, offset=-0.12)
    g = c.train.get("grad_norm", np.full(len(steps), math.nan))
    mgn = on_steps(c.health, "max_grad_norm", steps)
    clip = float(np.nanmax(mgn)) if np.isfinite(mgn).any() else None
    pos = np.where(g > 0, g, math.nan)
    ax_g.plot(steps, pos, color=c.color, linewidth=0.8, alpha=0.4)
    ax_g.plot(steps, trailing_mean(pos, a.smooth), color=c.color, linewidth=2)
    table.add("gradient norm", c, "grad_norm", "optimizer step", steps, "gradient norm (pre-clip)", g)
    hg = [Line2D([], [], color=c.color, linewidth=2, label=f"gradient norm before clipping (mean of {a.smooth})")]
    if clip:
        ax_g.axhline(clip, color=TEXT_2, linewidth=1.2, linestyle="--")
        clipped = g > clip
        if clipped.any():
            ax_g.plot(steps[clipped], g[clipped], linestyle="none", marker="x", markersize=6, color=TEXT)
            table.add("gradient norm", c, "clipped_step", "optimizer step", steps[clipped], "gradient norm (pre-clip)", g[clipped])
        hg += [Line2D([], [], color=TEXT_2, linewidth=1.2, linestyle="--", label=f"clip threshold max_grad_norm {clip:g}"),
               Line2D([], [], linestyle="none", marker="x", color=TEXT, label="clipped step")]
    if np.isfinite(pos).any():
        ax_g.set_yscale("log")
    mark_alerts(ax_g, c)
    style(ax_g, "gradient norm (log scale)", "", "L2 norm")
    legend_below(ax_g, hg, ncol=3, offset=-0.12)
    rate = on_steps(c.health, "clip_rate", steps) * 100
    if np.isfinite(rate).any():
        ax_c.plot(steps, rate, color=c.color, linewidth=2)
        table.add("clip rate", c, "clip_rate", "optimizer step", steps, "clipped steps in the window (%)", rate)
        thr = 100 * float(((c.meta.get("monitor_config") or {}).get("clip_rate_warn")) or 0.5)
        ax_c.axhline(thr, color=STATUS["WARN"], linewidth=1.2, linestyle="--")
        ax_c.set_ylim(-2, 102)
        style(ax_c, f"share of clipped steps over the last {int(((c.meta.get('monitor_config') or {}).get('clip_rate_window')) or 50)} steps",
              "optimizer step", "clipped (%)")
        legend_below(ax_c, [Line2D([], [], color=STATUS["WARN"], linewidth=1.2, linestyle="--", label=f"WARN threshold {thr:.0f}%")], ncol=1,
                     offset=-0.3)
    else:
        no_data(ax_c, "clip rate", "no monitor data (curves/source/<run_id>_health.csv)")
    return save(fig, table, out, "lr_grad", log)


def fig_lora(c: Curve, out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    if not c.lora:
        return []
    table = LongTable("lora")
    fams = sorted({r["family"] for r in c.lora}, key=lambda f: (f == "all", FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 9, f))
    fam_colors = {**color_map([f for f in fams if f != "all"], FAMILY_ORDER), "all": TEXT}
    fig, axes = new_figure(2, 2, height=3.4)
    panels = ((axes[0][0], "A", "norm", "L2 norm of lora_A", "L2 norm", False, 1.0),
              (axes[0][1], "B", "norm", "L2 norm of lora_B (0 at init)", "L2 norm", False, 1.0),
              (axes[1][0], "all", "update_ratio", "update ratio |theta_t - theta_(t-N)| / |theta_(t-N)| (log)", "ratio", True, 1.0),
              (axes[1][1], "all", "delta_w_rel", "effective weight change |scaling B A| / |W0|", "%", False, 100.0))
    for ax, part, col, label, unit, logy, scale in panels:
        drawn = False
        for fam in fams:
            rows = [r for r in c.lora if r["family"] == fam and r["part"] == part]
            x, y = column(rows, "step"), column(rows, col) * scale
            ok = np.isfinite(y) & (y > 0 if logy else True)
            if ok.any():
                ax.plot(x[ok], y[ok], color=fam_colors[fam], linewidth=2.4 if fam == "all" else 1.6, marker="o", markersize=4)
                table.add(label, c, f"{fam}_{part}_{col}", "optimizer step", x, label, y)
                drawn = True
        if logy and drawn:
            ax.set_yscale("log")
        if col == "update_ratio" and drawn:
            thr = float(((c.meta.get("monitor_config") or {}).get("lora_update_ratio_warn")) or 0.5)
            ax.axhline(thr, color=STATUS["WARN"], linewidth=1.2, linestyle="--")
        style(ax, label, "optimizer step", unit)
        if not drawn:
            ax.text(0.5, 0.5, "not measured yet", transform=ax.transAxes, ha="center", color=TEXT_2, fontsize=9)
    every = (c.meta.get("monitor_config") or {}).get("lora_every_steps", "N")
    legend_below(axes[1][0], [Line2D([], [], color=fam_colors[f], linewidth=2, label="all adapters" if f == "all" else f) for f in fams]
                 + [Line2D([], [], color=STATUS["WARN"], linewidth=1.2, linestyle="--", label="WARN threshold (update ratio)")], ncol=3)
    fig.suptitle(f"{title_of(c)}: LoRA adapter health (every {every} steps; families: attention = q/k/v/o, gdn = Gated DeltaNet "
                 "in_proj_* / out_proj, mlp = gate/up/down)", color=TEXT, fontsize=11)
    return save(fig, table, out, "lora", log)


def fig_throughput(c: Curve, out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    steps = c.steps
    if not len(steps):
        return []
    table = LongTable("throughput")
    fig, axes = new_figure(3, 1, width=12.8, height=2.8)
    ax_t, ax_s, ax_w = axes[:, 0]
    seq = c.train.get("tokens_per_s", np.full(len(steps), math.nan))
    trained = on_steps(c.health, "tokens_trained_per_s", steps)
    handles = []
    for y, label, col in ((seq, "sequence tokens/s", c.color), (trained, "trained (loss) tokens/s", TEXT_2)):
        if np.isfinite(y).any():
            ax_t.plot(steps, y, color=col, linewidth=0.8, alpha=0.35)
            ax_t.plot(steps, trailing_mean(y, a.smooth), color=col, linewidth=2)
            table.add("throughput", c, label, "optimizer step", steps, "tokens/s", y)
            handles.append(Line2D([], [], color=col, linewidth=2, label=f"{label} (mean of {a.smooth})"))
    mark_alerts(ax_t, c)
    style(ax_t, f"{title_of(c)}: throughput (step time excludes data loading, eval and saving)", "", "tokens/s")
    legend_below(ax_t, handles, ncol=2, offset=-0.12)
    st = c.train.get("step_time_s", np.full(len(steps), math.nan))
    ax_s.plot(steps, st, color=c.color, linewidth=1.6)
    table.add("step time", c, "step_time_s", "optimizer step", steps, "seconds", st)
    style(ax_s, "step time (forward + backward + optimizer)", "", "seconds")
    wait = on_steps(c.health, "data_wait_s", steps)
    if np.isfinite(wait).any():
        share = 100 * wait / (wait + st)
        ax_w.plot(steps, share, color=c.color, linewidth=1.6)
        table.add("data wait", c, "data_wait_share", "optimizer step", steps, "share of step (%)", share)
        style(ax_w, "data-wait share: host time fetching and collating the step's micro-batches / (that + step time)", "optimizer step", "%")
    else:
        no_data(ax_w, "data-wait share", "no monitor data (curves/source/<run_id>_health.csv)")
    return save(fig, table, out, "throughput", log)


def fig_gpu(c: Curve, out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    steps = c.steps
    mem_cols = (("gpu_mem_alloc_gb", "allocated at step end"), ("gpu_mem_reserved_gb", "reserved at step end"),
                ("gpu_mem_peak_reserved_gb", "peak reserved (cumulative)"))
    has_mem = any(np.isfinite(on_steps(c.health, k, steps)).any() for k, _ in mem_cols) if len(steps) else False
    has_peak_train = len(steps) and np.isfinite(c.train.get("peak_gpu_mem_reserved_gb", np.array([math.nan]))).any()
    if not (has_mem or has_peak_train or c.gpu):
        return []
    table = LongTable("gpu")
    fig, axes = new_figure(5, 1, width=12.8, height=2.5)
    ax_m = axes[0][0]
    handles = []
    for (k, label), col in zip(mem_cols, (PALETTE[0], PALETTE[1], PALETTE[2])):
        y = on_steps(c.health, k, steps)
        if not np.isfinite(y).any() and k == "gpu_mem_peak_reserved_gb":
            y = c.train.get("peak_gpu_mem_reserved_gb", np.full(len(steps), math.nan))
        if np.isfinite(y).any():
            ax_m.plot(steps, y, color=col, linewidth=1.8)
            table.add("memory", c, k, "optimizer step", steps, "GB", y)
            handles.append(Line2D([], [], color=col, linewidth=1.8, label=label))
    total = on_steps(c.health, "gpu_mem_total_gb", steps)
    if np.isfinite(total).any():
        t = float(np.nanmax(total))
        frac = float(((c.meta.get("monitor_config") or {}).get("gpu_mem_reserved_warn_fraction")) or 0.92)
        ax_m.axhline(t, color=TEXT, linewidth=1.2)
        ax_m.axhline(frac * t, color=STATUS["WARN"], linewidth=1.2, linestyle="--")
        handles += [Line2D([], [], color=TEXT, linewidth=1.2, label=f"GPU total {t:.0f} GB"),
                    Line2D([], [], color=STATUS["WARN"], linewidth=1.2, linestyle="--", label=f"WARN {frac:.0%} of total")]
    style(ax_m, f"{title_of(c)}: GPU memory (torch allocator)", "optimizer step", "GB")
    legend_below(ax_m, handles, ncol=5, offset=-0.25)
    g = c.gpu
    gx = column(g, "elapsed_s") / 60 if g else np.array([])
    for ax, k, label, unit, extra in ((axes[1][0], "gpu_util_pct", "GPU utilisation", "%", None),
                                      (axes[2][0], "power_w", "power draw", "W", "power_limit_w"),
                                      (axes[3][0], "temp_c", "GPU temperature", "C", None),
                                      (axes[4][0], "sm_clock_mhz", "SM clock", "MHz", None)):
        y = column(g, k) if g else np.array([])
        if len(y) and np.isfinite(y).any():
            ax.plot(gx, y, color=c.color, linewidth=1.4)
            table.add(label, c, k, "minutes since run start", gx, unit, y)
            if extra:
                lim = column(g, extra)
                if np.isfinite(lim).any():
                    ax.plot(gx, lim, color=TEXT_2, linewidth=1.0, linestyle="--")
                    table.add(label, c, extra, "minutes since run start", gx, unit, lim)
            if k == "temp_c":
                ax.axhline(float(((c.meta.get("monitor_config") or {}).get("gpu_temp_warn_c")) or 87), color=STATUS["WARN"], linewidth=1.2, linestyle="--")
            for lo, hi in phase_spans(gx, [r.get("phase") for r in g], ("eval", "saving")):
                ax.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0, zorder=0)
            note = {"power_w": "; dashed: power limit", "temp_c": "; dashed: WARN threshold"}.get(k, "")
            style(ax, f"{label} (nvidia-smi samples; shaded: eval / saving{note})", "minutes since run start" if k == "sm_clock_mhz" else "", unit)
        else:
            no_data(ax, label, "no GPU telemetry (CPU run or nvidia-smi unavailable)")
    return save(fig, table, out, "gpu", log)


def phase_spans(x: np.ndarray, phases: list[Any], wanted: tuple[str, ...]) -> list[tuple[float, float]]:
    """Contiguous x intervals whose samples are in one of the `wanted` phases."""
    spans, start, prev = [], None, None
    for xi, ph in zip(x, phases):
        if ph in wanted and start is None:
            start = xi
        elif ph not in wanted and start is not None:
            spans.append((start, prev))
            start = None
        prev = xi
    if start is not None:
        spans.append((start, prev))
    return spans


def fig_token_stats(c: Curve, out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    steps = c.steps
    pairs = (("entropy", "eval_entropy", "token entropy (nats)"), ("mean_token_accuracy", "eval_mean_token_accuracy", "token accuracy (top-1)"))
    if not any(np.isfinite(c.train.get(k, np.array([math.nan]))).any() for k, _, _ in pairs):
        return []
    from matplotlib.lines import Line2D
    table = LongTable("token_stats")
    fig, axes = new_figure(1, 2, height=3.6)
    for ax, (k, ek, label) in zip(axes[0], pairs):
        y = c.train.get(k, np.full(len(steps), math.nan))
        ax.plot(steps, y, color=c.color, linewidth=0.8, alpha=0.35)
        ax.plot(steps, trailing_mean(y, a.smooth), color=c.color, linewidth=2)
        table.add(label, c, k, "optimizer step", steps, label, y)
        if ek in c.eval and np.isfinite(c.eval[ek]).any():
            ax.plot(c.eval["step"], c.eval[ek], color=TEXT, linewidth=1.4, linestyle="--", marker="o", markersize=5)
            table.add(label, c, ek, "optimizer step", c.eval["step"], label, c.eval[ek])
        style(ax, f"{title_of(c)}: {label}", "optimizer step", label)
        legend_below(ax, [Line2D([], [], color=c.color, linewidth=2, label=f"train (mean of {a.smooth})"),
                          Line2D([], [], color=TEXT, linewidth=1.4, linestyle="--", marker="o", label="eval")], ncol=2)
    return save(fig, table, out, "token_stats", log)


RUN_FIGURES = (("loss", fig_loss, True), ("slice_loss", fig_slice_loss, True), ("row_loss", fig_row_loss, True),
               ("lr_grad", fig_lr_grad, False), ("lora", fig_lora, False), ("throughput", fig_throughput, False),
               ("gpu", fig_gpu, False), ("token_stats", fig_token_stats, False))
CAPTIONS = {
    "loss": "Training loss per step (faint) and its EMA, eval loss overall (dashed) and per slice (dotted), checkpoints (diamonds), "
            "WARN / CRITICAL alerts (vertical lines, labelled in the timeline strip); below, the same against trained tokens.",
    "slice_loss": "Train loss per slice (token-weighted over bins of steps, from the per-row records) with the eval loss per slice; right: "
                  "each slice's eval loss relative to its first evaluation (forgetting check).",
    "row_loss": "Row loss (mean NLL per loss token of one row) against row length, train rows filled and eval rows hollow; right: the "
                "distribution per slice in the early / middle / late third of the run.",
    "lr_grad": "Logged learning rate against the planned schedule; gradient norm before clipping (log) with the clip threshold and the "
               "clipped steps; share of clipped steps over the monitor's window.",
    "lora": "LoRA adapter health per module family: norms of lora_A and lora_B, update ratio over the measurement interval, effective "
            "weight change |delta W| relative to the frozen base weights.",
    "throughput": "Sequence and trained tokens per second, step time, and the share of wall time spent fetching and collating batches.",
    "gpu": "GPU memory per step (allocator: allocated and reserved at step end, cumulative peak) against the GPU total and the WARN "
           "line; nvidia-smi utilisation, power, temperature and SM clock over time.",
    "token_stats": "Token entropy and top-1 token accuracy on the trained tokens (train per step, eval per evaluation).",
}


# --- comparison figures ----------------------------------------------------------------------------------------------
def fig_compare_loss(name: str, curves: list[Curve], out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    table = LongTable("compare_loss")
    fig, axes = new_figure(1, 2, height=4.2)
    tok_label = "trained (loss) tokens (millions)"
    for c in curves:
        if not len(c.steps):
            continue
        ema, _ = loss_ema(c, a.smooth)
        x_tok, tok_label = trained_tokens_x(c)
        for ax, x, x_name in ((axes[0][0], c.steps, "optimizer step"), (axes[0][1], x_tok, tok_label)):
            ax.plot(x, ema, color=c.color, linewidth=2)
            table.add(x_name, c, "train_loss_ema", x_name, x, "loss", ema)
            ev_tok = eval_overall_tokens(c) if eval_weighting(c) == "tokens" else None
            e_steps, e_loss, series = ((ev_tok[0], ev_tok[1], "eval_loss_tokens") if ev_tok is not None else
                                       (c.eval.get("step", np.array([])), c.eval.get("eval_loss", np.array([])), "eval_loss"))
            if len(e_loss):
                xe = e_steps if x is c.steps else x_at_steps(c.steps, x, e_steps)
                ax.plot(xe, e_loss, color=c.color, linewidth=1.4, linestyle="--", marker="o", markersize=5, markeredgecolor=SURFACE)
                table.add(x_name, c, series, x_name, xe, "loss", e_loss)
    style(axes[0][0], f"{name}: training loss (EMA) and eval loss vs step", "optimizer step", "loss (mean NLL per loss token)")
    style(axes[0][1], "vs tokens", tok_label, "loss (mean NLL per loss token)")
    legend_below(axes[0][0], [Line2D([], [], color=c.color, linewidth=2, label=c.experiment) for c in curves]
                 + [Line2D([], [], color=TEXT_2, linewidth=1.4, linestyle="--", marker="o",
                           label="eval loss (token-weighted for token-normalised runs, else the Trainer's eval_loss)")], ncol=3)
    return save(fig, table, out, "compare_loss", log)


def fig_compare_slices(name: str, curves: list[Curve], out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    per = {c.run_id: (binned_slice_loss(c, a.bin_steps), eval_slices(c)) for c in curves}
    names = sorted({s for tr, ev in per.values() for s in (*tr, *ev)}, key=slice_key)
    if not names:
        return []
    table = LongTable("compare_slices")
    fig, axes = new_figure(1, len(names), width=5.2, height=4.0)
    for ax, s in zip(axes[0], names):
        for c in curves:
            tr, ev = per[c.run_id]
            if s in tr:
                x, y = tr[s]
                ax.plot(x, y, color=c.color, linewidth=2)
                table.add(s, c, "train", f"optimizer step (bin centre, {a.bin_steps} steps)", x, "loss", y)
            if s in ev:
                ax.plot(*ev[s], color=c.color, linewidth=1.4, linestyle="--", marker="o", markersize=5, markeredgecolor=SURFACE)
                table.add(s, c, "eval", "optimizer step", ev[s][0], "loss", ev[s][1])
        style(ax, f"slice {s}", "optimizer step", "loss (mean NLL per loss token)")
    legend_below(axes[0][0], [Line2D([], [], color=c.color, linewidth=2, label=c.experiment) for c in curves]
                 + [Line2D([], [], color=TEXT_2, linewidth=2, label=f"train ({a.bin_steps}-step bins)"),
                    Line2D([], [], color=TEXT_2, linewidth=1.4, linestyle="--", marker="o", label="eval")], ncol=2)
    fig.suptitle(f"{name}: loss per slice", color=TEXT, fontsize=12)
    return save(fig, table, out, "compare_slices", log)


def fig_compare_dynamics(name: str, curves: list[Curve], out: Path, a: argparse.Namespace, log: RunDir) -> list[Path]:
    from matplotlib.lines import Line2D
    table = LongTable("compare_dynamics")
    fig, axes = new_figure(2, 2, height=3.4)
    specs = ((axes[0][0], "tokens_per_s", "throughput (sequence tokens/s)", True, False),
             (axes[0][1], "grad_norm", "gradient norm before clipping (log)", True, True),
             (axes[1][0], "learning_rate", "learning rate", False, False))
    for ax, k, label, smooth, logy in specs:
        for c in curves:
            y = c.train.get(k)
            if y is None or not np.isfinite(y).any():
                continue
            yy = trailing_mean(y, a.smooth) if smooth else y
            ax.plot(c.steps, yy, color=c.color, linewidth=2)
            table.add(label, c, k + (f"_mean{a.smooth}" if smooth else ""), "optimizer step", c.steps, label, yy)
        if logy:
            ax.set_yscale("log")
        style(ax, label, "optimizer step", label.split(" (")[0])
    ax = axes[1][1]
    drawn = False
    for c in curves:
        rows = [r for r in c.lora if r.get("family") == "all" and r.get("part") == "all"]
        x, y = column(rows, "step"), column(rows, "update_ratio")
        ok = np.isfinite(y) & (y > 0)
        if ok.any():
            ax.plot(x[ok], y[ok], color=c.color, linewidth=2, marker="o", markersize=4)
            table.add("lora update ratio", c, "update_ratio", "optimizer step", x, "update ratio", y)
            drawn = True
    if drawn:
        ax.set_yscale("log")
    style(ax, "LoRA update ratio (all adapters, log)", "optimizer step", "update ratio")
    legend_below(axes[1][0], [Line2D([], [], color=c.color, linewidth=2, label=c.experiment) for c in curves], ncol=3)
    fig.suptitle(f"{name}: training dynamics (mean of {a.smooth} steps where noisy)", color=TEXT, fontsize=12)
    return save(fig, table, out, "compare_dynamics", log)


# --- HTML reports ----------------------------------------------------------------------------------------------------
def summary_rows(c: Curve, a: argparse.Namespace) -> list[tuple[str, str]]:
    """Key facts of one run for the report (metadata, status, resources, curves)."""
    m, st, res = c.meta, c.status, c.resources
    mon = m.get("monitor") or {}
    planned = (m.get("planned") or {}).get("optimizer_steps")
    dv = m.get("data_version") or {}
    train_dv = dv.get("train") or dv.get("total") or {}
    cfg = m.get("config") or {}
    lr = (cfg.get("optim") or cfg.get("filtered_sft") or cfg.get("dpo") or {}).get("learning_rate")
    ema, _ = loss_ema(c, a.smooth)
    ev = c.eval.get("eval_loss")
    best = int(np.nanargmin(ev)) if ev is not None and np.isfinite(ev).any() else None
    tps = c.train.get("tokens_per_s", np.array([]))
    tps = tps[np.isfinite(tps)][2:] if len(tps) > 2 else tps[np.isfinite(tps)]
    counts = ((st.get("alerts") or {}).get("counts")) or (mon.get("alerts") or {}).get("counts") or {}
    if not counts and c.alerts:
        counts = {s: sum(1 for x in c.alerts if x.get("severity") == s) for s in ("INFO", "WARN", "CRITICAL")}
    best_slices = {}
    for name, (xs, ys) in eval_slices(c).items():
        if np.isfinite(ys).any():
            i = int(np.nanargmin(ys))
            best_slices[name] = {"loss": float(ys[i]), "step": int(xs[i])}
    trained = on_steps(c.health, "tokens_trained_cumulative", c.steps)

    def fmt(x: Any, f: str = ".4f") -> str:
        return format(x, f) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else ("-" if x is None else str(x))

    tok = eval_overall_tokens(c)
    best_tokens = "-"
    if tok is not None and np.isfinite(tok[1]).any():
        i = int(np.nanargmin(tok[1]))
        best_tokens = f"{fmt(float(tok[1][i]))} ({int(tok[0][i])})"

    last_ema = ema[np.isfinite(ema)][-1] if np.isfinite(ema).any() else None
    loss = c.train.get("loss", np.array([]))
    rows = [("run_id", c.run_id), ("experiment", c.experiment),
            ("stage / method / arm", " / ".join(str(x) for x in (m.get("stage"), m.get("method"), m.get("arm")) if x) or "-"),
            ("status", f"{m.get('status', '-')} (monitor phase {st.get('phase', '-')})"),
            ("steps", f"{int(np.nanmax(c.steps)) if len(c.steps) else 0} of {planned or '?'} planned"),
            ("train rows / loss tokens", f"{train_dv.get('n_rows', '-')} / {fmt(train_dv.get('n_assistant_tokens'), ',.0f')}"),
            ("learning rate (config)", fmt(lr, ".2e")), ("loss normalisation", loss_normalisation(c)),
            ("final training loss (EMA)", fmt(last_ema)), ("min training loss (per step)", fmt(float(np.nanmin(loss)) if np.isfinite(loss).any() else None)),
            ("best eval loss, row-weighted eval_loss (step)", f"{fmt(float(ev[best]))} ({int(c.eval['step'][best])})" if best is not None else "-"),
            ("best eval loss, token-weighted (step)", best_tokens),
            (f"best eval loss per slice, {eval_weighting(c)}-weighted (step)", ", ".join(f"{k} {fmt(v['loss'])} ({v['step']})" for k, v in sorted(best_slices.items(),
                                                                                                     key=lambda kv: slice_key(kv[0]))) or "-"),
            ("tokens/s (median after step 2)", fmt(float(np.median(tps)) if len(tps) else None, ".0f")),
            ("trained (loss) tokens", fmt(float(np.nanmax(trained)) if np.isfinite(trained).any() else None, ",.0f")),
            ("peak reserved GPU memory (GB)", fmt(float(np.nanmax(c.train["peak_gpu_mem_reserved_gb"])) if np.isfinite(c.train.get("peak_gpu_mem_reserved_gb", np.array([math.nan]))).any() else None, ".1f")),
            ("GPU hours / cost (USD)", f"{res.get('gpu_hours', '-')} / {res.get('cost_usd', '-')}"),
            ("alerts (CRITICAL / WARN / INFO)", f"{counts.get('CRITICAL', 0)} / {counts.get('WARN', 0)} / {counts.get('INFO', 0)}"),
            ("checkpoints", " ".join(map(str, c.checkpoints)) or "-"),
            ("monitor overhead", f"{fmt(((mon.get('overhead') or {}).get('share_of_step_time') or 0) * 100, '.3f')} % of step time"
             if mon.get("overhead") else "-")]
    if st.get("stop_reason"):
        rows.append(("stopped by the monitor", str(st["stop_reason"])))
    return rows


HTML_CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
        --grid: #e1e0d9; --ok: #0ca30c; --warn: #fab219; --crit: #d03b3b; }
body { background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 8px; } .sub { color: var(--ink2); margin: 0 0 16px; }
table { border-collapse: collapse; background: var(--surface); width: 100%; font-variant-numeric: tabular-nums; }
td, th { border-bottom: 1px solid var(--grid); padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; } .wrap { overflow-x: auto; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; border: 1px solid var(--grid); background: var(--surface); }
.badge::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--muted); }
.OK::before { background: var(--ok); } .WARN::before { background: var(--warn); } .CRITICAL::before { background: var(--crit); }
figure { margin: 18px 0; background: var(--surface); border: 1px solid var(--grid); border-radius: 6px; padding: 10px; }
figure img { max-width: 100%; height: auto; display: block; } figcaption { color: var(--ink2); font-size: 13px; margin-top: 6px; }
code { font-size: 12px; }
"""


def badge(level: str) -> str:
    return f'<span class="badge {html.escape(level)}">{html.escape(level)}</span>'


def write_html(path: Path, title: str, subtitle: str, sections: list[str], figures: list[tuple[Path, str]]) -> Path:
    parts = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
             f"<title>{html.escape(title)}</title><style>{HTML_CSS}</style></head><body><main>",
             f"<h1>{html.escape(title)}</h1><p class=\"sub\">{subtitle}</p>", *sections]
    if figures:
        parts.append("<h2>Figures</h2>")
    for png, caption in figures:
        data = base64.b64encode(png.read_bytes()).decode()
        csv_name = png.with_suffix(".csv").name
        parts.append(f"<figure><img alt=\"{html.escape(png.stem)}\" src=\"data:image/png;base64,{data}\"><figcaption><b>{html.escape(png.stem)}</b>: "
                     f"{html.escape(caption)} Plotted numbers: <a href=\"{html.escape(csv_name)}\">{html.escape(csv_name)}</a>.</figcaption></figure>")
    parts.append("</main></body></html>")
    path.write_text("\n".join(parts))
    return path


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell if isinstance(cell, Markup) else html.escape(str(cell))}</td>" for cell in r) + "</tr>" for r in rows)
    return f"<div class=\"wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


class Markup(str):
    """Trusted HTML fragment (a badge) inside table_html."""


def run_report(c: Curve, out: Path, figs: list[Path], a: argparse.Namespace) -> Path:
    st = c.status
    health = st.get("health") or ("CRITICAL" if c.meta.get("status") in ("crashed", "stopped") else "OK" if c.meta.get("status") == "finished" else "-")
    sections = ["<h2>Summary</h2>", table_html(["", ""], [[k, v] for k, v in summary_rows(c, a)])]
    if st:
        lo, ev = st.get("loss") or {}, st.get("eval") or {}
        sections += ["<h2>Health</h2>", table_html(["", ""], [
            ["health", Markup(badge(health))], ["phase", st.get("phase", "-")], ["status updated", st.get("updated_at", "-")],
            ["step", f"{st.get('step')} / {st.get('total_steps')}"], ["loss last / EMA", f"{lo.get('last')} / {lo.get('ema')}"],
            ["last eval", f"step {ev.get('step')}: {ev.get('loss')}; slices {ev.get('slices')}" if ev else "-"],
            ["GPU telemetry", st.get("gpu_telemetry") or "-"], ["stop reason", st.get("stop_reason") or "-"]])]
    alerts = [[Markup(badge(al.get("severity", "-"))), al.get("step", "-"), al.get("code", "-"), al.get("message", ""), al.get("action", ""),
               al.get("ts", "")] for al in c.alerts]
    sections += [f"<h2>Alerts ({len(c.alerts)})</h2>",
                 table_html(["severity", "step", "code", "message", "action", "time (UTC)"], alerts) if alerts else "<p>No alerts.</p>"]
    pngs = [(p, CAPTIONS.get(p.stem, "")) for p in figs if p.suffix == ".png"]
    sub = (f"{html.escape(c.group)}; generated {html.escape(utc_now())} by scripts/plot_curves.py from curves/source/{html.escape(c.run_id)}_*.csv "
           f"and logs/{html.escape(c.run_id)}/ (status {html.escape(str(c.meta.get('status', 'unknown')))})")
    return write_html(out / "report.html", f"{c.experiment}: training report", sub, sections, pngs)


def compare_report(name: str, curves: list[Curve], out: Path, figs: list[Path], a: argparse.Namespace) -> Path:
    keys = [k for k, _ in summary_rows(curves[0], a)]
    per = [dict(summary_rows(c, a)) for c in curves]
    rows = [[k, *[p.get(k, "-") for p in per]] for k in keys if k != "run_id"]
    sections = ["<h2>Runs</h2>", table_html(["", *[c.run_id for c in curves]], rows)]
    caps = {"compare_loss": "Training loss EMA (solid) and eval loss (dashed) of each run, vs step and vs trained tokens.",
            "compare_slices": "Train loss per slice (binned, token-weighted) and eval loss per slice of each run.",
            "compare_dynamics": "Throughput, gradient norm, learning rate and LoRA update ratio of each run."}
    pngs = [(p, caps.get(p.stem, "")) for p in figs if p.suffix == ".png"]
    return write_html(out / "report.html", f"Comparison: {name}", f"{len(curves)} runs; generated {html.escape(utc_now())} by scripts/plot_curves.py",
                      sections, pngs)


def comparisons(curves: list[Curve], a: argparse.Namespace, log: RunDir) -> dict[str, list[Curve]]:
    by_id = {c.run_id: c for c in curves}
    if a.compare:
        out = {}
        for name, ids in a.compare.items():
            sel = [by_id[i] for i in ids if i in by_id]
            missing = [i for i in ids if i not in by_id]
            if missing:
                log.error(f"--compare {name}: runs without curve files (or skipped dry runs): {missing}")
            if len(sel) >= 2:
                out[name] = sel
        return out
    auto = {"sft": [c for c in curves if c.run_id.startswith("sft_")], "stage2": [c for c in curves if c.run_id.startswith("stage2_")]}
    return {k: v for k, v in auto.items() if len(v) >= 2}


def safe(fn: Any, what: str, log: RunDir, *args: Any) -> list[Path]:
    """One figure; a failure (malformed source, run in progress) is logged and skipped, never fatal."""
    try:
        return fn(*args) or []
    except Exception as exc:  # noqa: BLE001
        import matplotlib.pyplot as plt
        plt.close("all")
        log.error(f"{what}: figure failed ({type(exc).__name__}: {exc}); skipped")
        return []


# --- entry point -----------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=REPO_ROOT / "curves" / "source", help="curve source directory")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "analysis" / "figures", help="figure directory")
    p.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs",
                   help="where logs/<run_id>/ live (metrics.jsonl checkpoint events, alerts.jsonl, metadata.json, status.json, resources.csv)")
    p.add_argument("--runs", nargs="+", default=None, metavar="RUN_ID", help="plot only these runs (default: every run in --source)")
    p.add_argument("--label", action="append", default=[], metavar="RUN_ID=NAME",
                   help="experiment name shown for a run (default: run_id without its timestamp)")
    p.add_argument("--smooth", type=int, default=10, help="trailing-mean window in steps for noisy series (1 = none)")
    p.add_argument("--bin-steps", type=int, default=10, help="bin width in steps for the per-slice train loss (per-run and comparison figures)")
    p.add_argument("--checkpoint-every", type=int, default=25, help="SFT checkpoint interval when the run logged no checkpoint events (configs/sft.yaml)")
    p.add_argument("--include-dryrun", action="store_true", help="also plot dryrun_* runs (CPU smoke tests)")
    p.add_argument("--compare", action="append", default=[], metavar="NAME=RUN_ID,RUN_ID",
                   help="a comparison report compare_<NAME>/ over these runs (repeatable; default: SFT runs, stage-2 runs when >= 2 each)")
    p.add_argument("--no-reports", action="store_true", help="only the four cross-run outputs (no per-run or comparison reports)")
    p.add_argument("--run-id", default=None, help="run id of this plotting run (default plot_curves_<UTC timestamp>)")
    p.add_argument("--log-root", type=Path, default=REPO_ROOT, help="SMOKE TESTS ONLY: write logs/<run_id>/ under this root")
    a = p.parse_args(argv)
    a.labels = {}
    for item in a.label:
        run_id, sep, name = item.partition("=")
        if not sep or not name:
            p.error(f"--label expects RUN_ID=NAME, got {item!r}")
        a.labels[run_id] = name
    comp = {}
    for item in a.compare:
        name, sep, ids = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or len(ids.split(",")) < 2:
            p.error(f"--compare expects NAME=RUN_ID,RUN_ID[,...] (NAME of letters, digits, _ . -), got {item!r}")
        comp[name] = [i.strip() for i in ids.split(",") if i.strip()]
    a.compare = comp
    if a.smooth < 1 or a.checkpoint_every < 1 or a.bin_steps < 1:
        p.error("--smooth, --bin-steps and --checkpoint-every must be >= 1")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    with RunDir(a.run_id or new_run_id("plot_curves"), a.log_root, log_name="eval.log") as log:
        log.write_metadata(run_id=log.run_id, script="scripts/plot_curves.py", code=git_info(), hardware=hardware_info(),
                           config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, model_revision="n/a (plots training curves)",
                           seed="n/a (deterministic)", command=command_line(), start_time=utc_now(), status="running")
        curves = discover(a, log) if a.source.is_dir() else []
        outputs: list[Path] = []
        if not curves:
            log.log(f"no curve source data in {a.source}; nothing to plot")
        else:
            a.out.mkdir(parents=True, exist_ok=True)
            for c in curves:
                log.log(f"{c.run_id} ({c.experiment}, {c.group}): {len(c.steps)} train rows, {len(c.eval.get('step', []))} eval rows, "
                        f"checkpoints {c.checkpoints} ({c.checkpoint_source}); monitor sources: "
                        f"{[k for k in KINDS[2:] if getattr(c, k)] or 'none'}, {len(c.alerts)} alerts")
                log.metric({"event": "run", "run_id": c.run_id, "experiment": c.experiment, "n_train_rows": len(c.steps),
                            "n_eval_rows": len(c.eval.get("step", [])), "checkpoints": c.checkpoints, "n_alerts": len(c.alerts)})
            outputs += loss_figure(curves, "step", "loss_vs_step", a.smooth, a.out, log)
            if any("tokens_cumulative" in c.train and np.isfinite(c.train["tokens_cumulative"]).any() for c in curves):
                outputs += loss_figure(curves, "tokens_cumulative", "loss_vs_tokens", a.smooth, a.out, log)
            outputs += dynamics_figure(curves, a.smooth, a.out, log)
            runs_summary(curves, a.smooth, a.out / "runs_summary.csv")
            outputs.append(a.out / "runs_summary.csv")
            if not a.no_reports:
                slice_colors = color_map(sorted({s for c in curves for s in slice_names(c)}, key=slice_key), SLICE_ORDER)
                for c in curves:
                    rdir = a.out / c.run_id
                    rdir.mkdir(parents=True, exist_ok=True)
                    figs: list[Path] = []
                    for fname, fn, needs_colors in RUN_FIGURES:
                        args = (c, rdir, a, log, slice_colors) if needs_colors else (c, rdir, a, log)
                        figs += safe(fn, f"{c.run_id}/{fname}", log, *args)
                    outputs += figs + [run_report(c, rdir, figs, a)]
                    log.log(f"wrote {rdir / 'report.html'} ({len([f for f in figs if f.suffix == '.png'])} figures)")
                for name, members in comparisons(curves, a, log).items():
                    cdir = a.out / f"compare_{name}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    figs = []
                    for fn in (fig_compare_loss, fig_compare_slices, fig_compare_dynamics):
                        figs += safe(fn, f"compare_{name}/{fn.__name__}", log, name, members, cdir, a, log)
                    outputs += figs + [compare_report(name, members, cdir, figs, a)]
                    log.log(f"wrote {cdir / 'report.html'} ({[c.run_id for c in members]})")
        duration = log.elapsed()
        rate = compute_rates()["cpu_vm_usd_per_hour"]
        log.write_resources([("duration_sec", round(duration, 1), "s", "wall clock (CPU only)"), ("peak_gpu_memory_gb", "n/a", "GB", "no GPU used"),
                             ("gpu_hours", 0, "h", "no GPU used"),
                             ("cost_usd", round(duration / 3600 * rate, 4), "USD", f"cpu_hours x {rate} (configs/compute.yaml)")])
        log.write_metadata(data_version={"inputs": [{"path": str(f), "sha256": sha256_file(f)} for c in curves for f in c.files]},
                           outputs=[str(o) for o in outputs], end_time=utc_now(), duration_sec=round(duration, 1), status="finished")
        log.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
