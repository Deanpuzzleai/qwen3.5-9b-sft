"""Training monitor for apex_train.sft and apex_train.stage2: per-step health records, per-row and per-slice losses, LoRA
health, GPU telemetry, alerts, a heartbeat and a stall watchdog. One implementation for both entry points, wired into the
TrainerCallback bridge of apex_train.common.TrainMetrics. It only observes: every value it reads is detached, and the
only control it ever exercises is "save a checkpoint and stop" after a CRITICAL non-finite or data alert
(monitor.stop_on_nonfinite / stop_on_data_error), when continuing would only burn GPU hours on a broken run.

Files (schema "monitor/v1"; append-only except status.json, which is rewritten atomically):
  logs/<run_id>/metrics.jsonl     TrainMetrics' per-step `train` record gains the monitor fields (MONITOR_STEP_FIELDS
                                  plus `slices` and `gpu`), the `eval` record gains `slices`; `lora_health` records
  logs/<run_id>/alerts.jsonl      one object per alert: ts, code, severity (INFO/WARN/CRITICAL), step, value,
                                  threshold, message, action, phase (+ context); WARN and CRITICAL are echoed to errors.log,
                                  every alert to train.log
  logs/<run_id>/status.json       heartbeat: phase, progress, ETA, last losses, eval per slice, throughput, memory, GPU,
                                  alert counts, last 3 alerts, health OK/WARN/CRITICAL, updated_at (every step and every
                                  monitor.watchdog_every_s seconds from the watchdog thread)
  logs/<run_id>/rows.jsonl        one record per trained row (split train) and per evaluated row (split eval)
  logs/<run_id>/stall_traceback_step<N>.txt   Python stacks of every thread when the watchdog declares a stall
  curves/source/<run_id>_health.csv   HEALTH_COLUMNS per optimizer step
  curves/source/<run_id>_slices.csv   SLICE_COLUMNS: per step and slice (split train) and per evaluation and slice (eval)
  curves/source/<run_id>_rows.csv     ROW_COLUMNS: the rows.jsonl records, flat
  curves/source/<run_id>_lora.csv     LORA_COLUMNS: adapter norms, update ratios and |delta W| per module family, every
                                      monitor.lora_every_steps steps
  curves/source/<run_id>_gpu.csv      GPU_COLUMNS: nvidia-smi samples every monitor.gpu_sample_every_s seconds (absent without
                                      nvidia-smi, e.g. CPU smokes)

What "row loss" means. A row's loss is its mean NLL per loss token, S_k / n_k (S_k = summed token NLL of row k, n_k = its
loss tokens counted after the causal shift, labels[..., 1:] != -100): comparable across rows, slices and with eval. How an
optimizer step combines its rows is the run's loss normalisation (optim.loss_normalisation / filtered_sft.loss_normalisation,
applied by apex_train.common.apply_loss_normalisation); the monitor records the mode the Trainer actually used (metadata
monitor_attach / monitor .loss_normalisation, status.json loss_normalisation) and warns if it differs from the configured one:
  tokens  (the default): the Trainer passes num_items_in_batch N = sum_k n_k of the accumulation window, TRL 1.15's
          SFTTrainer.compute_loss returns c_k = S_k / N, the Trainer backpropagates c_k unscaled and logs loss = sum_k c_k =
          sum S_k / sum n_k: the token-weighted mean NLL of the window.
  rows    no num_items_in_batch (Qwen3_5ForConditionalGeneration declares accepts_loss_kwargs = False): c_k = S_k / n_k, the
          Trainer backpropagates c_k / G (G = the window's micro-batches) and logs loss = mean_k c_k (row-weighted).
`loss_contribution` is the row's share of the logged step loss (c_k, resp. c_k / G); the shares sum to the logged loss up to
float rounding (checked every step: WARN loss_bookkeeping_mismatch), and in tokens mode N is checked against the counted loss
tokens (WARN loss_normalisation_mismatch, which also fires when the mode observed differs from the configured one).
Evaluation calls compute_loss without num_items_in_batch, so an eval row's value is S / n and the Trainer's eval_loss (kept
as it was) is the mean over rows (row-weighted). The monitor also writes the token-weighted eval loss (sum S / sum n over the
evaluated rows): `eval_loss_tokens` / `slices_tokens` in the metrics.jsonl eval record and status.json, and in
<run_id>_slices.csv, where every slice row carries both weightings (`loss_tokens`, `loss_rows`) and the eval slice named `all`
holds the whole evaluation (its loss_rows equals eval_loss). The eval detectors (overfitting, per-slice regression) use the
weighting of the training objective (token-weighted under tokens). The train slice loss of a step is token-weighted. With a
micro-batch of more than one row the per-row split is not identifiable: every row of the micro-batch gets the micro-batch
value and `shared_rows` > 1. DPO (stage2 --method dpo) has no row tracking. The raw micro-batch losses are what the monitor
checks for non-finite values: the Trainer's logging_nan_inf_filter replaces a non-finite micro-batch loss in the logged
`loss`, so `loss_observed` (the raw combination) is the one that shows it.

Overhead: per step a hash of each collated row (CPU), one stack of the 16 detached losses after the step's existing
synchronize, allocator counters, a few appended lines and one status.json write; every lora_every_steps steps one
device-to-host copy of the adapter (86.6M fp32 parameters, 346 MB kept on the host for the update ratio) and its norms.
The measured cost is recorded per step as `monitor_s` and summarised in metadata.json (monitor.overhead).

Smoke-only hook: under --smoke, APEX_MONITOR_INJECT_NONFINITE_STEP=N multiplies the first micro-batch loss of optimizer
step N by NaN, to exercise the stop-and-save path. Production runs never read it.
"""
from __future__ import annotations

import faulthandler
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import median
from typing import Any

from apex_eval.common import RunDir, utc_now
from apex_train.common import ConfigError, CsvAppender, _row_hash, current_rss_gb, peak_rss_gb, training_resource_rows

SCHEMA = "monitor/v1"
SEVERITIES = ("INFO", "WARN", "CRITICAL")
RANK = {s: i for i, s in enumerate(SEVERITIES)}
HEALTH = ("OK", "WARN", "CRITICAL")
INJECT_ENV = "APEX_MONITOR_INJECT_NONFINITE_STEP"
ROW_META_COLUMNS = ("sample_id", "source", "task_type", "mode", "turn_index", "n_tokens", "n_assistant_tokens")
# LoRA target name -> module family (configs/sft.yaml lora.target_modules): gated attention of the 8 full-attention layers,
# Gated DeltaNet of the 24 linear-attention layers, the dense MLP of all 32 layers
FAMILIES = {"q_proj": "attention", "k_proj": "attention", "v_proj": "attention", "o_proj": "attention",
            "in_proj_qkv": "gdn", "in_proj_z": "gdn", "in_proj_b": "gdn", "in_proj_a": "gdn", "out_proj": "gdn",
            "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
# NVML clock-event ("throttle") reason bits reported by nvidia-smi as a hex mask
THROTTLE_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
GPU_QUERY = ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "power.limit", "temperature.gpu",
             "clocks.sm", "clocks_throttle_reasons.active")
GPU_THROTTLE_FALLBACK = "clocks_event_reasons.active"   # the field's newer name; older drivers know only the first
HEALTH_COLUMNS = ("step", "epoch", "elapsed_s", "loss", "loss_observed", "loss_ema", "learning_rate", "lr_planned", "lr_rel_dev",
                  "grad_norm", "max_grad_norm", "grad_clipped", "clip_rate", "tokens_seq", "tokens_trained", "tokens_trained_cumulative",
                  "tokens_per_s", "tokens_trained_per_s", "step_time_s", "data_wait_s", "step_wall_s", "monitor_s",
                  "gpu_mem_alloc_gb", "gpu_mem_reserved_gb", "gpu_mem_peak_alloc_gb", "gpu_mem_peak_reserved_gb", "gpu_mem_total_gb",
                  "rss_gb", "gpu_util_pct", "gpu_power_w", "gpu_temp_c", "gpu_sm_clock_mhz", "entropy", "mean_token_accuracy",
                  "n_rows", "alerts_warn", "alerts_critical")
SLICE_COLUMNS = ("split", "step", "slice", "loss_tokens", "loss_rows", "n_rows", "n_tokens", "n_loss_tokens", "entropy", "token_accuracy",
                 "runtime_s")
ROW_COLUMNS = ("split", "step", "micro_step", "row_index", "sample_id", "slice", "source", "task_type", "mode", "turn_index", "n_tokens",
               "n_loss_tokens", "loss", "loss_contribution", "shared_rows", "entropy", "token_accuracy")
LORA_COLUMNS = ("step", "family", "part", "n_params", "norm", "update_norm", "update_ratio", "delta_w_norm", "base_w_norm", "delta_w_rel")
GPU_COLUMNS = ("utc", "elapsed_s", "step", "phase", "gpu_util_pct", "mem_util_pct", "mem_used_gb", "mem_total_gb", "power_w",
               "power_limit_w", "temp_c", "sm_clock_mhz", "throttle_mask", "throttle_reasons")
ACTIONS = {
    "nonfinite_loss": "the monitor saved checkpoint-<step> (forensic: it holds the state after the bad step) and stopped the run; resume "
                      "from the last checkpoint before this step with the LR fallback (--resume <that checkpoint> --override "
                      "optim.learning_rate=5e-5), after checking the rows of this step in rows.jsonl",
    "nonfinite_grad_norm": "as nonfinite_loss: forensic checkpoint saved, run stopped; resume from the last good checkpoint with LR 5e-5",
    "nonfinite_weights": "adapter weights are not finite: the run is stopped; resume from the last checkpoint whose weights are finite",
    "nonfinite_eval_loss": "check the adapter weights (lora_health) and the eval rows; the latest checkpoint is likely unusable",
    "loss_spike": "look at this step's rows in rows.jsonl (one long or unusual row?); no action if the EMA keeps falling",
    "loss_spike_repeated": "the loss is unstable (M4.2 criterion): stop and restart from the last checkpoint before the spikes with the "
                           "LR fallback 5e-5, or record why it is benign",
    "grad_norm_spike": "compare with the loss at this step; isolated spikes are clipped (max_grad_norm) and usually benign",
    "clip_rate_high": "most steps are clipped: the effective LR is lower than planned; if the loss also stalls, consider the LR fallback",
    "lr_schedule_deviation": "the scheduler does not follow the planned schedule (wrong total steps, warm-up or a resume mismatch): "
                             "check planned.optimizer_steps and the trainer's max_steps",
    "planned_steps_mismatch": "the Trainer plans a different number of optimizer steps than metadata.planned: the cosine schedule ends "
                              "elsewhere; check max_steps / epochs / row count",
    "throughput_drop": "check gpu_util, throttle reasons and kernel fallbacks in train.log; a long-row batch explains a single slow step",
    "stall": "no optimizer step for too long: see the stall_traceback file for the stacks; if hung, kill the process and resume from the "
             "latest checkpoint under a new run_id",
    "stall_recovered": "none (informational)",
    "gpu_mem_high": "close to OOM at the row cap: do not start anything else on the GPU; if it grows further, plan the 48K fallback",
    "gpu_mem_leak": "allocated memory at step end keeps growing: look for retained tensors; resume from a checkpoint would reset it",
    "host_rss_leak": "host memory keeps growing (data loading / logging): watch free memory; resume from a checkpoint resets it",
    "gpu_util_low": "the GPU is waiting (input bound or stalled): check data_wait_s and the host CPU",
    "gpu_temperature_high": "check cooling / the VM's GPU health; throughput may drop through thermal slowdown",
    "gpu_throttle": "clocks are reduced by thermal or hardware slowdown: check the VM; tok/s will drop",
    "eval_overfit": "eval loss rises while train loss falls: note the step; the guard rule selects among the 25-step checkpoints",
    "eval_slice_regression": "a dev slice got worse than at its first evaluation (forgetting): compare with the replay share and LR; "
                             "the guard rule decides on the dev benchmarks",
    "lora_update_ratio_high": "the adapter moves fast relative to its size: watch for loss spikes; consider the LR fallback",
    "disk_low": "free disk below the threshold under the checkpoint root: free space (never delete checkpoints: move them)",
    "data_zero_loss_tokens": "a row trains no token: rebuild the rows (should be impossible after the startup checks)",
    "data_row_too_long": "a row exceeds max_length: rebuild the rows at the cap (should be impossible after the startup checks)",
    "data_row_mismatch": "a collated row differs from its source row (truncation or a changed collator): stop and inspect",
    "row_unidentified": "a collated row matches no registered row: per-row records lack metadata (check the collator)",
    "loss_bookkeeping_mismatch": "the monitor's per-row decomposition disagrees with the logged loss: per-row losses are unreliable",
    "loss_normalisation_mismatch": "num_items_in_batch differs from the counted loss tokens: TRL / transformers changed the loss "
                                   "normalisation; per-row losses are unreliable",
}


class MonitorStop(RuntimeError):
    """Raised by the entry points after trainer.train() returned because the monitor stopped the run (CRITICAL alert)."""


# ------------------------------------------------------------------------------------------------------------ config
@dataclass
class MonitorConfig:
    """The `monitor:` block of configs/sft.yaml and configs/stage2.yaml (defaults here, documented there)."""

    enabled: bool = True
    slice_key: str = "task_type"
    row_records: bool = True
    eval_max_rows_per_slice: int | None = None
    eval_on_start: bool = False
    eval_rows: str | None = None
    eval_every_fraction: float = 0.25
    loss_ema_beta: float = 0.9
    lora_every_steps: int = 10
    gpu_sample_every_s: float = 15.0
    watchdog_every_s: float = 30.0
    alert_cooldown_steps: int = 25
    alert_cooldown_s: float = 900.0
    health_window_s: float = 1800.0
    stop_on_nonfinite: bool = True
    stop_on_data_error: bool = True
    spike_window: int = 50
    spike_min_history: int = 20
    loss_spike_mad_k: float = 6.0
    loss_spike_min_delta: float = 0.1
    loss_spike_repeat_n: int = 3
    loss_spike_repeat_window: int = 25
    grad_spike_factor: float = 5.0
    clip_rate_window: int = 50
    clip_rate_warn: float = 0.5
    lr_rel_tol: float = 0.01
    throughput_window: int = 20
    throughput_drop_fraction: float = 0.5
    throughput_drop_steps: int = 3
    stall_minutes: float = 20.0
    stall_step_factor: float = 5.0
    eval_stall_minutes: float = 60.0
    loading_stall_minutes: float = 45.0
    gpu_mem_reserved_warn_fraction: float = 0.92
    mem_leak_window: int = 50
    gpu_mem_leak_min_gb: float = 1.0
    rss_leak_min_gb: float = 8.0
    gpu_util_low_pct: float = 50.0
    gpu_util_low_minutes: float = 5.0
    gpu_temp_warn_c: float = 87.0
    throttle_warn_mask: int = 0xE8
    overfit_consecutive_evals: int = 2
    forgetting_rel_increase: float = 0.05
    lora_update_ratio_warn: float = 0.5
    disk_free_warn_gb: float = 50.0

    @classmethod
    def from_block(cls, block: dict[str, Any] | None) -> MonitorConfig:
        """Defaults overlaid with the yaml block; unknown keys and wrong types are ConfigErrors."""
        block = dict(block or {})
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(block) - set(known))
        if unknown:
            raise ConfigError(f"monitor: unknown keys {unknown} (known: {sorted(known)})")
        cfg = cls()
        for key, value in block.items():
            default = getattr(cfg, key)
            if value is None:
                if key not in ("eval_max_rows_per_slice", "eval_rows"):
                    raise ConfigError(f"monitor.{key} may not be null")
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ConfigError(f"monitor.{key} must be true or false, got {value!r}")
            elif key in ("eval_max_rows_per_slice",) or isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ConfigError(f"monitor.{key} must be an integer, got {value!r}")
            elif isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(f"monitor.{key} must be a number, got {value!r}")
                value = float(value)
            elif isinstance(default, str) or key == "eval_rows":
                value = str(value)
            setattr(cfg, key, value)
        for key in ("lora_every_steps", "spike_window", "spike_min_history", "loss_spike_repeat_n", "loss_spike_repeat_window",
                    "clip_rate_window", "throughput_window", "throughput_drop_steps", "mem_leak_window", "overfit_consecutive_evals"):
            if getattr(cfg, key) < 1:
                raise ConfigError(f"monitor.{key} must be >= 1")
        if cfg.eval_max_rows_per_slice is not None and cfg.eval_max_rows_per_slice < 1:
            raise ConfigError("monitor.eval_max_rows_per_slice must be >= 1 or null (all dev rows)")
        if not 0.0 < cfg.loss_ema_beta < 1.0:
            raise ConfigError("monitor.loss_ema_beta must be in (0, 1)")
        if not 0.0 < cfg.eval_every_fraction <= 1.0:
            raise ConfigError("monitor.eval_every_fraction must be in (0, 1]")
        if cfg.gpu_sample_every_s <= 0 or cfg.watchdog_every_s <= 0:
            raise ConfigError("monitor.gpu_sample_every_s and watchdog_every_s must be > 0")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------------------ alerts
@dataclass
class Alert:
    code: str
    severity: str
    step: int | None
    value: Any
    threshold: Any
    message: str
    action: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    key: str = ""

    def __post_init__(self) -> None:
        if self.severity not in RANK:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        self.action = self.action or ACTIONS.get(self.code, "")
        self.key = self.key or self.code

    def record(self, *, phase: str, run_id: str) -> dict[str, Any]:
        rec = {"ts": utc_now(), "schema": SCHEMA, "run_id": run_id, "code": self.code, "severity": self.severity, "step": self.step,
               "value": _clean(self.value), "threshold": _clean(self.threshold), "message": self.message, "action": self.action,
               "phase": phase}
        if self.context:
            rec["context"] = _clean(self.context)
        return rec


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _clean(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats -> their string ('nan', 'inf'), so files stay strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _r(x: Any, nd: int = 6) -> Any:
    return round(x, nd) if _finite(x) else x


class HealthChecks:
    """The detectors, as pure functions of the metric streams (no torch): each method takes the newest observation and
    returns the alerts it raises. Emission policy (cooldown, files, counts) belongs to TrainingMonitor."""

    def __init__(self, cfg: MonitorConfig, *, max_grad_norm: float | None = None) -> None:
        self.cfg = cfg
        self.max_grad_norm = max_grad_norm
        self.losses: deque[float] = deque(maxlen=cfg.spike_window)
        self.grads: deque[float] = deque(maxlen=cfg.spike_window)
        self.clipped: deque[bool] = deque(maxlen=cfg.clip_rate_window)
        self.tps: deque[float] = deque(maxlen=cfg.throughput_window)
        self.alloc: deque[float] = deque(maxlen=cfg.mem_leak_window)
        self.rss: deque[float] = deque(maxlen=cfg.mem_leak_window)
        self.spike_steps: deque[int] = deque(maxlen=max(cfg.loss_spike_repeat_n * 4, 16))
        self.tps_low_run = 0
        self.evals: list[tuple[int, float, float | None]] = []
        self.first_slice_eval: dict[str, tuple[int, float]] = {}
        self.util_low_since: float | None = None
        self.stalled_since_step: int | None = None

    @property
    def clip_rate(self) -> float | None:
        return sum(self.clipped) / len(self.clipped) if self.clipped else None

    # --- per optimizer step
    def on_step(self, r: dict[str, Any]) -> list[Alert]:
        c, out, step = self.cfg, [], r.get("step")
        loss = r.get("loss_observed") if r.get("loss_observed") is not None else r.get("loss")
        g = r.get("grad_norm")
        if loss is not None and not _finite(loss):
            out.append(Alert("nonfinite_loss", "CRITICAL", step, loss, "finite", f"training loss is {loss} at step {step}"))
        if g is not None and not _finite(g):
            out.append(Alert("nonfinite_grad_norm", "CRITICAL", step, g, "finite", f"gradient norm is {g} at step {step}"))
        if _finite(loss):
            if len(self.losses) >= c.spike_min_history:
                m = median(self.losses)
                sigma = 1.4826 * median(abs(x - m) for x in self.losses)
                thr = m + max(c.loss_spike_mad_k * sigma, c.loss_spike_min_delta)
                if loss > thr:
                    self.spike_steps.append(step)
                    out.append(Alert("loss_spike", "WARN", step, _r(loss), _r(thr),
                                     f"loss {loss:.4f} > rolling median {m:.4f} + max({c.loss_spike_mad_k} x 1.4826 MAD, "
                                     f"{c.loss_spike_min_delta}) over the last {len(self.losses)} steps"))
                    recent = [s for s in self.spike_steps if s is not None and step is not None and s > step - c.loss_spike_repeat_window]
                    if len(recent) >= c.loss_spike_repeat_n:
                        out.append(Alert("loss_spike_repeated", "CRITICAL", step, len(recent), c.loss_spike_repeat_n,
                                         f"{len(recent)} loss spikes within {c.loss_spike_repeat_window} steps (steps {recent})"))
            self.losses.append(loss)
        if _finite(g):
            if len(self.grads) >= c.spike_min_history:
                m = median(self.grads)
                if m > 0 and g > c.grad_spike_factor * m:
                    out.append(Alert("grad_norm_spike", "WARN", step, _r(g), _r(c.grad_spike_factor * m),
                                     f"grad norm {g:.4g} > {c.grad_spike_factor} x rolling median {m:.4g}"))
            self.grads.append(g)
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.clipped.append(g > self.max_grad_norm)
                rate = self.clip_rate
                if len(self.clipped) >= min(c.clip_rate_window, c.spike_min_history) and rate is not None and rate > c.clip_rate_warn:
                    out.append(Alert("clip_rate_high", "WARN", step, _r(rate, 4), c.clip_rate_warn,
                                     f"{rate:.0%} of the last {len(self.clipped)} steps clipped at max_grad_norm {self.max_grad_norm}"))
        lr, planned = r.get("learning_rate"), r.get("lr_planned")
        if _finite(lr) and _finite(planned):
            dev = abs(lr - planned) / max(abs(planned), 1e-30) if planned else (0.0 if lr == 0 else math.inf)
            if dev > c.lr_rel_tol and abs(lr - planned) > 1e-12:
                out.append(Alert("lr_schedule_deviation", "WARN", step, lr, planned,
                                 f"learning rate {lr:.4g} differs from the planned {planned:.4g} by {dev:.1%} (tolerance {c.lr_rel_tol:.1%})"))
        tps = r.get("tokens_per_s")
        if _finite(tps) and tps > 0:
            if len(self.tps) >= min(c.throughput_window, c.spike_min_history):
                m = median(self.tps)
                self.tps_low_run = self.tps_low_run + 1 if tps < c.throughput_drop_fraction * m else 0
                if self.tps_low_run >= c.throughput_drop_steps:
                    out.append(Alert("throughput_drop", "WARN", step, _r(tps, 1), _r(c.throughput_drop_fraction * m, 1),
                                     f"tokens/s {tps:.0f} below {c.throughput_drop_fraction:.0%} of the rolling median {m:.0f} for "
                                     f"{self.tps_low_run} consecutive steps"))
            self.tps.append(tps)
        total = r.get("gpu_mem_total_gb")
        used = max((v for v in (r.get("gpu_mem_reserved_gb"), r.get("gpu_mem_peak_reserved_gb")) if _finite(v)), default=None)
        if _finite(total) and total > 0 and used is not None and used / total > c.gpu_mem_reserved_warn_fraction:
            out.append(Alert("gpu_mem_high", "WARN", step, _r(used, 2), _r(c.gpu_mem_reserved_warn_fraction * total, 2),
                             f"reserved GPU memory {used:.1f} GB > {c.gpu_mem_reserved_warn_fraction:.0%} of {total:.1f} GB"))
        for key, series, min_gb, code, what in (("gpu_mem_alloc_gb", self.alloc, c.gpu_mem_leak_min_gb, "gpu_mem_leak", "allocated GPU memory at step end"),
                                                ("rss_gb", self.rss, c.rss_leak_min_gb, "host_rss_leak", "host RSS")):
            v = r.get(key)
            if _finite(v):
                series.append(v)
                growth = _monotone_growth(series, c.mem_leak_window)
                if growth is not None and growth > min_gb:
                    out.append(Alert(code, "WARN", step, _r(growth, 3), min_gb,
                                     f"{what} grew by {growth:.2f} GB over the last {c.mem_leak_window} steps, almost monotonically"))
        return out

    # --- per evaluation
    def on_eval(self, step: int, overall: float | None, slices: dict[str, float], train_loss_ema: float | None,
                weighting: str = "rows") -> list[Alert]:
        """Overfitting and per-slice regression on the eval losses of `weighting` (rows: the Trainer's eval_loss; tokens: the
        token-weighted eval loss), the one that matches the training objective."""
        c, out = self.cfg, []
        if overall is not None and not _finite(overall):
            out.append(Alert("nonfinite_eval_loss", "CRITICAL", step, overall, "finite", f"eval loss is {overall} at step {step}"))
        if _finite(overall):
            self.evals.append((step, overall, train_loss_ema))
            n = c.overfit_consecutive_evals
            if len(self.evals) > n:
                last = self.evals[-(n + 1):]
                rising = all(b[1] > a[1] for a, b in zip(last, last[1:]))
                emas = [e[2] for e in last]
                falling = all(_finite(e) for e in emas) and emas[-1] < emas[0]
                if rising and falling:
                    out.append(Alert("eval_overfit", "WARN", step, _r(overall), _r(last[0][1]),
                                     f"{weighting}-weighted eval loss rose for {n} consecutive evaluations ({' -> '.join(f'{e[1]:.4f}' for e in last)}, "
                                     f"steps {[e[0] for e in last]}) while the train loss EMA fell ({emas[0]:.4f} -> {emas[-1]:.4f})"))
        for name, value in sorted(slices.items()):
            if not _finite(value):
                continue
            first = self.first_slice_eval.setdefault(name, (step, value))
            limit = first[1] * (1 + c.forgetting_rel_increase)
            if step != first[0] and value > limit:
                out.append(Alert("eval_slice_regression", "WARN", step, _r(value), _r(limit),
                                 f"{weighting}-weighted eval loss of slice {name} is {value:.4f}, {value / first[1] - 1:+.1%} vs its first evaluation "
                                 f"{first[1]:.4f} at step {first[0]} (threshold +{c.forgetting_rel_increase:.0%})",
                                 context={"slice": name, "first_step": first[0], "weighting": weighting}, key=f"eval_slice_regression:{name}"))
        return out

    # --- LoRA health
    def on_lora(self, step: int, update_ratio: float | None, finite: bool) -> list[Alert]:
        out = []
        if not finite:
            out.append(Alert("nonfinite_weights", "CRITICAL", step, "non-finite", "finite", f"adapter weights contain NaN/inf at step {step}"))
        elif _finite(update_ratio) and update_ratio > self.cfg.lora_update_ratio_warn:
            out.append(Alert("lora_update_ratio_high", "WARN", step, _r(update_ratio, 4), self.cfg.lora_update_ratio_warn,
                             f"||theta_t - theta_(t-N)|| / ||theta_(t-N)|| = {update_ratio:.3g} over the last {self.cfg.lora_every_steps} steps"))
        return out

    # --- GPU telemetry sample (watchdog side)
    def on_gpu(self, s: dict[str, Any], *, phase: str, now: float, step: int | None) -> list[Alert]:
        c, out = self.cfg, []
        temp, mask, util = s.get("temp_c"), s.get("throttle_mask"), s.get("gpu_util_pct")
        if _finite(temp) and temp >= c.gpu_temp_warn_c:
            out.append(Alert("gpu_temperature_high", "WARN", step, temp, c.gpu_temp_warn_c, f"GPU temperature {temp:.0f} C"))
        if isinstance(mask, int) and mask & c.throttle_warn_mask:
            names = decode_throttle(mask & c.throttle_warn_mask)
            out.append(Alert("gpu_throttle", "WARN", step, hex(mask), hex(c.throttle_warn_mask), f"GPU clock reasons active: {', '.join(names)}"))
        if phase == "training" and _finite(util):
            if util < c.gpu_util_low_pct:
                self.util_low_since = self.util_low_since if self.util_low_since is not None else now
                low_for = now - self.util_low_since
                if low_for >= c.gpu_util_low_minutes * 60:
                    out.append(Alert("gpu_util_low", "WARN", step, util, c.gpu_util_low_pct,
                                     f"GPU utilisation below {c.gpu_util_low_pct:.0f} % for {low_for / 60:.1f} min while training"))
            else:
                self.util_low_since = None
        else:
            self.util_low_since = None
        return out

    # --- watchdog
    def stall_threshold_s(self, phase: str, median_step_s: float | None) -> float | None:
        c = self.cfg
        if phase == "training":
            return max(c.stall_minutes * 60, c.stall_step_factor * (median_step_s or 0.0))
        if phase in ("eval", "saving"):
            return c.eval_stall_minutes * 60
        if phase == "loading":
            return c.loading_stall_minutes * 60
        return None

    def on_watchdog(self, *, phase: str, idle_s: float, step: int | None, median_step_s: float | None) -> list[Alert]:
        thr = self.stall_threshold_s(phase, median_step_s)
        if thr is None:
            return []
        if idle_s > thr:
            first = self.stalled_since_step is None
            self.stalled_since_step = step if step is not None else -1
            return [Alert("stall", "CRITICAL", step, round(idle_s, 1), round(thr, 1),
                          f"no progress for {idle_s / 60:.1f} min in phase {phase} (threshold {thr / 60:.1f} min)"
                          + ("" if first else " (still stalled)"), context={"phase": phase})]
        if self.stalled_since_step is not None:
            self.stalled_since_step = None
            return [Alert("stall_recovered", "INFO", step, round(idle_s, 1), round(thr, 1), f"progress resumed in phase {phase}")]
        return []

    @property
    def stalled(self) -> bool:
        return self.stalled_since_step is not None

    def on_disk(self, free_gb: float, path: str, step: int | None) -> list[Alert]:
        if free_gb < self.cfg.disk_free_warn_gb:
            return [Alert("disk_low", "WARN", step, round(free_gb, 1), self.cfg.disk_free_warn_gb, f"{free_gb:.1f} GB free under {path}")]
        return []

    @staticmethod
    def on_rows(split: str, step: int | None, rows: Iterable[dict[str, Any]], max_length: int | None) -> list[Alert]:
        """Data sanity of collated rows: zero loss tokens, longer than max_length, different from the source row, unknown."""
        out = []
        for r in rows:
            sid = r.get("sample_id")
            if r.get("row_index") is None:
                out.append(Alert("row_unidentified", "WARN", step, r.get("n_tokens"), None,
                                 f"a collated {split} row of {r.get('n_tokens')} tokens matches no registered row"))
                continue
            if split == "train" and r.get("n_loss_tokens") == 0:
                out.append(Alert("data_zero_loss_tokens", "CRITICAL", step, 0, ">= 1", f"row {sid} trains zero tokens"))
            if max_length and _finite(r.get("n_tokens")) and r["n_tokens"] > max_length:
                out.append(Alert("data_row_too_long", "CRITICAL", step, r["n_tokens"], max_length, f"row {sid} has {r['n_tokens']} tokens"))
            src = r.get("source_n_tokens")
            if _finite(src) and src != r.get("n_tokens"):
                out.append(Alert("data_row_mismatch", "CRITICAL", step, r.get("n_tokens"), src,
                                 f"row {sid}: {r.get('n_tokens')} collated tokens vs {src} in the rows file"))
        return out


def _monotone_growth(series: Sequence[float], window: int) -> float | None:
    """Growth over the last `window` values when the series rises in >= 90 percent of its steps, else None."""
    if len(series) < window:
        return None
    vals = list(series)[-window:]
    ups = sum(b >= a for a, b in zip(vals, vals[1:])) / max(len(vals) - 1, 1)
    return vals[-1] - vals[0] if ups >= 0.9 else None


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit] or (["none"] if mask == 0 else [hex(mask)])


# -------------------------------------------------------------------------------------------------------- GPU telemetry
def _num(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.upper() in ("N/A", "NOT SUPPORTED", "[N/A]"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_gpu_line(line: str, query: Sequence[str]) -> dict[str, Any]:
    """One `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` line -> GPU_COLUMNS values (MiB -> GB)."""
    parts = [p.strip() for p in line.split(",")]
    raw = dict(zip(query, parts))
    mib = lambda k: None if _num(raw.get(k, "")) is None else round(_num(raw[k]) * 1.048576e-3, 3)  # noqa: E731
    mask_text = raw.get("clocks_throttle_reasons.active", raw.get(GPU_THROTTLE_FALLBACK, ""))
    try:
        mask = int(mask_text, 16) if mask_text.lower().startswith("0x") else None
    except ValueError:
        mask = None
    return {"gpu_util_pct": _num(raw.get("utilization.gpu", "")), "mem_util_pct": _num(raw.get("utilization.memory", "")),
            "mem_used_gb": mib("memory.used"), "mem_total_gb": mib("memory.total"), "power_w": _num(raw.get("power.draw", "")),
            "power_limit_w": _num(raw.get("power.limit", "")), "temp_c": _num(raw.get("temperature.gpu", "")),
            "sm_clock_mhz": _num(raw.get("clocks.sm", "")), "throttle_mask": mask,
            "throttle_reasons": "|".join(decode_throttle(mask)) if mask is not None else None}


def gpu_index() -> str:
    """nvidia-smi index of the training GPU: the first CUDA_VISIBLE_DEVICES entry when numeric, else 0."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return first if first.isdigit() else "0"


class GpuSampler:
    """Daemon thread: nvidia-smi every `every_s` seconds -> on_sample(dict). Tolerates a missing binary (CPU hosts) and drivers
    that only know the newer throttle field name. `query_fn(fields) -> (returncode, stdout, stderr)` is injectable."""

    def __init__(self, every_s: float, on_sample: Callable[[dict[str, Any]], None],
                 query_fn: Callable[[Sequence[str]], tuple[int, str, str]] | None = None) -> None:
        self.every_s, self.on_sample = every_s, on_sample
        self.query_fn = query_fn or self._nvidia_smi
        self.query: list[str] | None = list(GPU_QUERY)
        self.status = "not started"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="apex-monitor-gpu", daemon=True)

    @staticmethod
    def _nvidia_smi(query: Sequence[str]) -> tuple[int, str, str]:
        proc = subprocess.run(["nvidia-smi", "-i", gpu_index(), f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20, check=False)
        return proc.returncode, proc.stdout, proc.stderr

    def sample(self) -> dict[str, Any] | None:
        if self.query is None:
            return None
        for attempt in range(3):
            try:
                code, out, err = self.query_fn(self.query)
            except FileNotFoundError:
                self.query, self.status = None, "unavailable (nvidia-smi not found)"
                return None
            except (subprocess.TimeoutExpired, OSError) as exc:
                self.status = f"error: {exc}"
                return None
            if code == 0 and out.strip():
                self.status = "ok"
                return parse_gpu_line(out.strip().splitlines()[0], self.query)
            text = (out + err).lower()
            if attempt == 0 and "clocks_throttle_reasons" in text:
                self.query = [GPU_THROTTLE_FALLBACK if q == "clocks_throttle_reasons.active" else q for q in self.query]
            elif attempt <= 1 and any(q.startswith("clocks_") and "reasons" in q for q in self.query):
                self.query = [q for q in self.query if "reasons" not in q]
            else:
                self.status = f"error (exit {code}): {(out + err).strip()[:200]}"
                return None
        return None

    def start(self) -> None:
        self.status = "starting"
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            s = self.sample()
            if s is not None:
                try:
                    self.on_sample(s)
                except Exception as exc:  # noqa: BLE001 - telemetry must never kill training
                    self.status = f"callback error: {exc}"
            if self.query is None:
                return
            self.stop_event.wait(self.every_s)


# ------------------------------------------------------------------------------------------------------------ helpers
def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON to a temporary file in the same directory, fsync, os.replace: readers never see a partial file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w") as f:
        json.dump(_clean(obj), f, indent=1, default=str, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def task_key(sample_id: Any) -> str:
    """Task of a row for eval stratification: the sample id up to its episode suffix ('__'), else the sample id."""
    text = str(sample_id)
    return text.split("__", 1)[0]


def _stable(text: str) -> str:
    import hashlib

    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def stratified_eval_indices(slices: Sequence[str], sample_ids: Sequence[Any], turn_index: Sequence[Any], max_per_slice: int) -> list[int]:
    """Deterministic subset: per slice, rows are grouped by task and taken round-robin over tasks in a stable hash order
    (one row of every task before a second row of any task), up to max_per_slice rows. Returns sorted row indices."""
    by_slice: dict[str, dict[str, list[int]]] = {}
    for i, (s, sid) in enumerate(zip(slices, sample_ids)):
        by_slice.setdefault(s, {}).setdefault(task_key(sid), []).append(i)
    chosen: list[int] = []
    for s in sorted(by_slice):
        tasks = sorted(by_slice[s], key=lambda t: _stable(f"{s}|{t}"))
        queues = {t: sorted(by_slice[s][t], key=lambda i: _stable(f"{sample_ids[i]}|{turn_index[i]}|{i}")) for t in tasks}
        picked: list[int] = []
        rank = 0
        while len(picked) < max_per_slice and any(len(q) > rank for q in queues.values()):
            for t in tasks:
                if rank < len(queues[t]) and len(picked) < max_per_slice:
                    picked.append(queues[t][rank])
            rank += 1
        chosen += picked
    return sorted(chosen)


class RowTag:
    """Opaque per-micro-batch annotation carried in the collated batch under ROWS_KEY (an object without .to(), so
    accelerate / the Trainer pass it through untouched; the trainer subclass pops it before the forward)."""

    __slots__ = ("rows",)

    def __init__(self, rows: list[tuple[int | None, int, int]]) -> None:
        self.rows = rows   # (registry index or None, attention-mask tokens, shifted loss tokens) per row

    def __repr__(self) -> str:
        return f"RowTag({self.rows})"


ROWS_KEY = "apex_monitor_rows"


class _Tagger:
    """Collator wrapper: delegates, then tags the batch with each row's identity and token counts (CPU only)."""

    def __init__(self, inner: Callable[..., dict[str, Any]], monitor: TrainingMonitor) -> None:
        self.inner, self.monitor = inner, monitor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.inner(examples)
        t0 = time.perf_counter()
        try:
            batch[ROWS_KEY] = self.monitor._tag(batch)
        except Exception as exc:  # noqa: BLE001 - observation must never break training
            self.monitor._note_error(f"row tagging failed: {exc!r}")
        self.monitor._overhead_pending += time.perf_counter() - t0
        return batch


@dataclass
class _Micro:
    loss: Any
    tag: RowTag | None
    num_items: Any
    entropy: float | None
    accuracy: float | None


# ------------------------------------------------------------------------------------------------------------ LoRA
class LoraHealth:
    """Adapter statistics every N steps: L2 norms of lora_A / lora_B per module family, the update ratio
    ||theta_t - theta_(t-N)|| / ||theta_(t-N)|| against a host copy of the previous measurement, and the effective weight
    update |dW|_F = scaling * sqrt(trace((B^T B)(A A^T))) per module relative to the frozen base weight |W0|_F."""

    def __init__(self, model: Any) -> None:
        from peft.tuners.lora import LoraLayer

        self.entries: list[dict[str, Any]] = []
        for name, m in model.named_modules():
            if not isinstance(m, LoraLayer) or "default" not in getattr(m, "lora_A", {}):
                continue
            target = name.rsplit(".", 1)[-1]
            a, b = m.lora_A["default"].weight, m.lora_B["default"].weight
            if not (a.requires_grad or b.requires_grad):
                continue
            self.entries.append({"name": name, "family": FAMILIES.get(target, "other"), "A": a, "B": b,
                                 "scaling": float(m.scaling["default"]), "W": m.get_base_layer().weight})
        self.prev: list[Any] | None = None
        self.base_sq: dict[str, float] | None = None
        self.n_params = sum(e["A"].numel() + e["B"].numel() for e in self.entries)

    def _host_copy(self) -> list[Any]:
        import torch

        return [p.detach().to("cpu", dtype=torch.float32, copy=True) for e in self.entries for p in (e["A"], e["B"])]

    def snapshot(self) -> None:
        self.prev = self._host_copy()

    def base_norms(self) -> None:
        """|W0|_F^2 per family of the frozen base weights the adapters wrap; once, before training (no mid-run allocation)."""
        import torch

        with torch.no_grad():
            self.base_sq = {}
            for e in self.entries:
                w2 = float(torch.linalg.vector_norm(e["W"].detach(), dtype=torch.float32) ** 2)
                for fam in (e["family"], "all"):
                    self.base_sq[fam] = self.base_sq.get(fam, 0.0) + w2

    def measure(self) -> dict[str, Any]:
        """{"groups": {family: {part: {...}}}, "finite": bool, "update_ratio": float | None} (families incl. 'all')."""
        import torch

        if self.base_sq is None:
            self.base_norms()
        with torch.no_grad():
            cur = self._host_copy()
            acc: dict[tuple[str, str], list[float]] = {}   # (family, part) -> [n, norm^2, update^2, prev^2, dW^2]
            finite = True
            for i, e in enumerate(self.entries):
                a, b = cur[2 * i], cur[2 * i + 1]
                for part, t, j in (("A", a, 2 * i), ("B", b, 2 * i + 1)):
                    n2 = float(torch.sum(t * t))
                    finite &= math.isfinite(n2)
                    u2 = p2 = math.nan
                    if self.prev is not None:
                        u2 = float(torch.sum((t - self.prev[j]) ** 2))
                        p2 = float(torch.sum(self.prev[j] * self.prev[j]))
                    for key in ((e["family"], part), (e["family"], "all"), ("all", part), ("all", "all")):
                        v = acc.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
                        v[0] += t.numel()
                        v[1] += n2
                        v[2] += u2
                        v[3] += p2
                gram = float(torch.sum((b.t() @ b) * (a @ a.t()))) * e["scaling"] ** 2   # |s B A|_F^2 without forming B A
                for key in ((e["family"], "all"), ("all", "all")):
                    acc[key][4] += max(gram, 0.0)
            self.prev = cur
        groups: dict[str, dict[str, dict[str, Any]]] = {}
        for (fam, part), (n, n2, u2, p2, dw2) in sorted(acc.items()):
            rec = {"n_params": n, "norm": math.sqrt(n2) if n2 >= 0 else math.nan,
                   "update_norm": math.sqrt(u2) if math.isfinite(u2) else None,
                   "update_ratio": math.sqrt(u2 / p2) if math.isfinite(u2) and math.isfinite(p2) and p2 > 0 else None}
            if part == "all":
                base = (self.base_sq or {}).get(fam)
                rec.update(delta_w_norm=math.sqrt(dw2), base_w_norm=math.sqrt(base) if base else None,
                           delta_w_rel=math.sqrt(dw2 / base) if base else None)
            groups.setdefault(fam, {})[part] = rec
        return {"groups": groups, "finite": finite, "update_ratio": groups.get("all", {}).get("all", {}).get("update_ratio")}


# ------------------------------------------------------------------------------------------------------------ monitor
class TrainingMonitor:
    """See the module docstring. Life cycle (both entry points):
        mon = TrainingMonitor(run, cfg.get("monitor"), stage=..., device=..., curves_dir=..., checkpoint_root=...)   # phase loading
        mon.set_plan(planned_steps=..., max_length=...)
        dev = mon.select_eval_rows(dev)                       # optional stratified subset
        mon.register_rows("train", train_ds, index_by_hash)   # from check_prepared_labels
        Trainer = mon.trainer_class(SFTTrainer)               # thin compute_loss observer (or the class itself)
        metrics = TrainMetrics(..., monitor=mon); trainer = Trainer(..., callbacks=[metrics.callback(...)])
        mon.attach(trainer)                                   # tags collated rows, LoRA health, planned LR
        trainer.train(); mon.stop_reason -> MonitorStop
        mon.summary() / mon.resource_rows(); run exit -> mon.close(phase)
    A disabled monitor (monitor.enabled: false) is a no-op everywhere: the trainer class, collator and training are untouched."""

    def __init__(self, run: Any, block: dict[str, Any] | None, *, stage: str, device: str, curves_dir: Path, checkpoint_root: Path,
                 smoke: bool = False, resume: bool = False, row_tracking: bool = True, method: str | None = None,
                 arm: str | None = None) -> None:
        self.cfg = MonitorConfig.from_block(block)
        self.enabled = self.cfg.enabled
        self.run, self.stage, self.device, self.method, self.arm = run, stage, device, method, arm
        self.curves_dir, self.checkpoint_root = Path(curves_dir), Path(checkpoint_root)
        self.row_tracking = row_tracking and self.cfg.row_records
        self.lock = threading.RLock()
        self.t_start = time.time()
        self.phase = "loading"
        self.phase_since = time.time()
        self.step = 0
        self.total_steps: int | None = None
        self.max_length: int | None = None
        self.epoch: float | None = None
        self.stop_reason: str | None = None
        self.weights_finite_at_stop: bool | None = None
        self.closed = False
        self.checks = HealthChecks(self.cfg)
        self.registry_meta: list[dict[str, Any]] = []
        self.registry_hash: dict[str, int] = {}
        self.eval_subset: dict[str, Any] | None = None
        self.inject_step: int | None = None
        if smoke and os.environ.get(INJECT_ENV):
            self.inject_step = int(os.environ[INJECT_ENV])
        # per-step state
        self._micro: list[_Micro] = []
        self._eval_micro: list[_Micro] = []
        self._step_info: dict[str, Any] = {}
        self._overhead_pending = 0.0
        self._t_hooks_end: float | None = None
        self._t_last_step_end: float | None = None
        self._last_progress = time.time()
        self._data_wait: float | None = None
        self.step_walls: deque[float] = deque(maxlen=50)
        self.eval_walls: list[float] = []
        self.ema: float | None = None   # bias-corrected EMA of the step loss (loss_ema_beta)
        self._ema_raw, self._ema_n = 0.0, 0
        self.loss_min: tuple[float, int] | None = None
        self.last: dict[str, Any] = {}
        self.slice_window: dict[str, deque] = {}
        self.last_eval: dict[str, Any] | None = None
        self.best_eval: dict[str, tuple[float, int]] = {}
        self.checkpoints: list[int] = []
        self.tokens_trained_total = 0
        self.tokens_seq_total = 0
        self.monitor_s: list[float] = []
        self.data_wait_s: list[float] = []
        self.step_time_s: list[float] = []
        self.tps_seq: list[tuple[int, float]] = []
        self.tps_trained: list[tuple[int, float]] = []
        self.gpu_last: dict[str, Any] | None = None
        self.gpu_samples: list[dict[str, Any]] = []
        self.gpu_total_gb: float | None = None
        self.peak_alloc_gb: float | None = None
        self.peak_reserved_gb: float | None = None
        self.lora: LoraHealth | None = None
        self.lora_last: dict[str, Any] | None = None
        self.planned_lrs: list[float] | None = None
        self.loss_normalisation: str | None = None   # expected from the trainer at attach, confirmed at the first step
        self.eval_every: int | None = None
        self.disk_free_gb: float | None = None
        self.max_grad_norm: float | None = None
        self.alert_counts: Counter[str] = Counter()
        self.alerts_suppressed = 0
        self.alerts_recent: deque[tuple[float, dict[str, Any]]] = deque(maxlen=50)
        self._last_emit: dict[str, tuple[int, float, int]] = {}
        self._errors_noted: set[str] = set()
        self._stop_event = threading.Event()
        self.gpu_sampler: GpuSampler | None = None
        self.watchdog: threading.Thread | None = None
        self._last_disk_check = 0.0
        if not self.enabled:
            return
        run_dir = Path(run.run_dir)
        self.alerts_path, self.status_path, self.rows_path = run_dir / "alerts.jsonl", run_dir / "status.json", run_dir / "rows.jsonl"
        self.alerts_path.touch()
        rid = run.run_id
        self.health_csv = CsvAppender(self.curves_dir / f"{rid}_health.csv", HEALTH_COLUMNS, append=resume)
        self.slices_csv = CsvAppender(self.curves_dir / f"{rid}_slices.csv", SLICE_COLUMNS, append=resume)
        self.rows_csv = CsvAppender(self.curves_dir / f"{rid}_rows.csv", ROW_COLUMNS, append=resume) if self.row_tracking else None
        self.lora_csv = CsvAppender(self.curves_dir / f"{rid}_lora.csv", LORA_COLUMNS, append=resume)
        self.gpu_csv: CsvAppender | None = None
        self.gpu_csv_path = self.curves_dir / f"{rid}_gpu.csv"
        self._resume = resume
        run.write_metadata(monitor_config=self.cfg.to_dict(), monitor_files=self.files())
        self.write_status()
        self.gpu_sampler = GpuSampler(self.cfg.gpu_sample_every_s, self._on_gpu_sample)
        self.gpu_sampler.start()
        self.watchdog = threading.Thread(target=self._watchdog_loop, name="apex-monitor-watchdog", daemon=True)
        self.watchdog.start()
        run.log(f"monitor: schema {SCHEMA}, row tracking {self.row_tracking}, LoRA health every {self.cfg.lora_every_steps} steps, "
                f"GPU samples every {self.cfg.gpu_sample_every_s:g} s, watchdog every {self.cfg.watchdog_every_s:g} s; "
                f"status {self.status_path}" + (f"; TEST HOOK: non-finite loss injected at step {self.inject_step}" if self.inject_step else ""))

    # --- bookkeeping -------------------------------------------------------------------------------------------------
    def files(self) -> dict[str, str]:
        rid = self.run.run_id
        out = {"alerts": str(self.run.run_dir / "alerts.jsonl"), "status": str(self.run.run_dir / "status.json"),
               "health_csv": str(self.curves_dir / f"{rid}_health.csv"), "slices_csv": str(self.curves_dir / f"{rid}_slices.csv"),
               "lora_csv": str(self.curves_dir / f"{rid}_lora.csv"), "gpu_csv": str(self.curves_dir / f"{rid}_gpu.csv")}
        if self.row_tracking:
            out.update(rows=str(self.run.run_dir / "rows.jsonl"), rows_csv=str(self.curves_dir / f"{rid}_rows.csv"))
        return out

    def _note_error(self, msg: str) -> None:
        """An internal monitor problem: logged once per message, never raised (observation must not stop training)."""
        if msg in self._errors_noted:
            return
        self._errors_noted.add(msg)
        try:
            self.run.warn(f"monitor: {msg}")
        except Exception:  # noqa: BLE001
            pass

    def _set_phase(self, phase: str) -> None:
        with self.lock:
            if phase != self.phase:
                self.phase, self.phase_since = phase, time.time()
                self._last_progress = time.time()

    def set_plan(self, *, planned_steps: int | None, max_length: int | None, eval_every: int | None = None) -> None:
        """Planned optimizer steps (progress, ETA, the planned LR schedule), the row cap (data checks), the eval interval (ETA)."""
        self.total_steps, self.max_length, self.eval_every = planned_steps, max_length, eval_every

    # --- alerts --------------------------------------------------------------------------------------------------------
    def emit(self, alerts: Iterable[Alert]) -> list[dict[str, Any]]:
        """Cooldown per alert key (same or lower severity within alert_cooldown_steps steps AND alert_cooldown_s seconds is
        suppressed), then alerts.jsonl + train.log (+ errors.log for WARN / CRITICAL) + counts."""
        written = []
        for a in alerts:
            with self.lock:
                now, step = time.time(), self.step if a.step is None else a.step
                last = self._last_emit.get(a.key)
                if last is not None and RANK[a.severity] <= last[2] and step - last[0] < self.cfg.alert_cooldown_steps \
                        and now - last[1] < self.cfg.alert_cooldown_s:
                    self.alerts_suppressed += 1
                    continue
                self._last_emit[a.key] = (step, now, RANK[a.severity])
                rec = a.record(phase=self.phase, run_id=self.run.run_id)
                self.alert_counts[a.severity] += 1
                self.alerts_recent.append((now, rec))
                with self.alerts_path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            line = (f"ALERT {a.severity} {a.code} @ step {a.step}: {a.message} (value {rec['value']}, threshold {rec['threshold']})"
                    + (f"; action: {a.action}" if a.severity != "INFO" else ""))
            self.run.log(line)
            if a.severity != "INFO":
                RunDir.error(self.run, line)   # errors.log only: an alert is not a run error (TrainRun.n_errors stays untouched)
            written.append(rec)
        return written

    def health(self, now: float | None = None) -> str:
        now = now or time.time()
        if self.phase in ("stopped", "failed") or self.checks.stalled:
            return "CRITICAL"
        worst = max((RANK[rec["severity"]] for t, rec in self.alerts_recent if now - t <= self.cfg.health_window_s), default=0)
        return HEALTH[worst]

    # --- data registry -------------------------------------------------------------------------------------------------
    def select_eval_rows(self, ds: Any) -> Any:
        """Stratified-by-task subset of the dev rows (monitor.eval_max_rows_per_slice; None = all rows, unchanged)."""
        if ds is None or not self.enabled:
            return ds
        cols = [c for c in ("sample_id", "turn_index", self.cfg.slice_key) if c in ds.column_names]
        meta = ds.select_columns(cols).to_dict()
        slices = [str(v) if v is not None else "unknown" for v in meta.get(self.cfg.slice_key, ["unknown"] * len(ds))]
        before = dict(Counter(slices))
        if self.cfg.eval_max_rows_per_slice is None:
            self.eval_subset = {"selection": "all", "n_rows": len(ds), "rows_by_slice": before}
            return ds
        idx = stratified_eval_indices(slices, meta.get("sample_id", list(range(len(ds)))), meta.get("turn_index", [0] * len(ds)),
                                      self.cfg.eval_max_rows_per_slice)
        sub = ds.select(idx)
        after = dict(Counter(slices[i] for i in idx))
        self.eval_subset = {"selection": f"stratified by task, <= {self.cfg.eval_max_rows_per_slice} rows per slice", "n_rows": len(idx),
                            "n_rows_full": len(ds), "rows_by_slice": after, "rows_by_slice_full": before,
                            "indices_sha256": _stable(",".join(map(str, idx)))}
        self.run.log(f"monitor: eval subset {len(idx)} of {len(ds)} dev rows ({after}; full {before})")
        return sub

    def register_rows(self, split: str, ds: Any, index_by_hash: dict[str, int]) -> None:
        """Per-row metadata of a dataset the trainer will collate, keyed by the input_ids hash of check_prepared_labels."""
        if not (self.enabled and self.row_tracking) or ds is None:
            return
        cols = [c for c in dict.fromkeys((*ROW_META_COLUMNS, self.cfg.slice_key)) if c in ds.column_names]
        meta = ds.select_columns(cols).to_list()
        base = len(self.registry_meta)
        for i, m in enumerate(meta):
            v = m.get(self.cfg.slice_key)
            m.update(split=split, row_index=i, slice=str(v) if v not in (None, "") else "unknown")
            self.registry_meta.append(m)
        for h, i in index_by_hash.items():
            self.registry_hash.setdefault(h, base + i)
        self.run.log(f"monitor: registered {len(meta)} {split} rows by input_ids hash; slices {dict(Counter(m['slice'] for m in meta))}")

    def _tag(self, batch: dict[str, Any]) -> RowTag:
        ids, labels = batch["input_ids"], batch.get("labels")
        attn = batch.get("attention_mask")
        rows = []
        for i in range(ids.shape[0]):
            keep = attn[i].bool() if attn is not None else None
            row_ids = ids[i][keep] if keep is not None else ids[i]
            idx = self.registry_hash.get(_row_hash(row_ids.cpu().numpy()))
            n_loss = int((labels[i][1:] != -100).sum()) if labels is not None else 0
            rows.append((idx, int(row_ids.shape[0]), n_loss))
        return RowTag(rows)

    def trainer_class(self, base: type) -> type:
        """`base` itself when disabled or without row tracking; else a subclass whose compute_loss pops the row tag before the
        forward and records the detached loss the base class computed (no second forward, no change to the value)."""
        if not (self.enabled and self.row_tracking):
            return base
        monitor = self

        class Monitored(base):  # type: ignore[valid-type, misc]
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                tag = inputs.pop(ROWS_KEY, None)
                out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                loss = out[0] if return_outputs else out
                loss = monitor.observe_loss(self, loss, tag, num_items_in_batch)
                return (loss, out[1]) if return_outputs else loss

        Monitored.__name__ = Monitored.__qualname__ = f"Monitored{base.__name__}"
        return Monitored

    def attach(self, trainer: Any) -> None:
        """After the trainer exists (and after the label-check collator wrapper): tag collated rows, prepare LoRA health,
        the planned LR schedule and the clip threshold."""
        if not self.enabled:
            return
        args = trainer.args
        self.max_grad_norm = float(args.max_grad_norm) if args.max_grad_norm else None
        self.checks.max_grad_norm = self.max_grad_norm
        if self.row_tracking:
            trainer.data_collator = _Tagger(trainer.data_collator, self)
            scaled = bool(getattr(trainer, "model_accepts_loss_kwargs", False)) or getattr(trainer, "compute_loss_func", None) is not None
            self.loss_normalisation = "tokens" if scaled else "rows"
            self.run.log("monitor: expecting loss normalisation " + (
                "tokens (num_items_in_batch: the step loss is the token mean over the accumulation window)" if scaled else
                "rows (no num_items_in_batch: the step loss is the mean of the rows' token means; every row weighs the same)"))
        try:
            self.lora = LoraHealth(trainer.model)
            if not self.lora.entries:
                self.lora = None
            else:
                self.lora.base_norms()
        except Exception as exc:  # noqa: BLE001
            self._note_error(f"LoRA health disabled: {exc!r}")
            self.lora = None
        if self.total_steps:
            try:
                self.planned_lrs = planned_lr_schedule(args, self.total_steps)
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"planned LR schedule unavailable: {exc!r}")
        if self.device == "cuda":
            import torch

            self.gpu_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
        self.run.write_metadata(monitor_attach={"trainer_class": type(trainer).__name__, "loss_normalisation": self.loss_normalisation,
                                                "lora_modules": len(self.lora.entries) if self.lora else 0,
                                                "lora_params": self.lora.n_params if self.lora else 0, "max_grad_norm": self.max_grad_norm,
                                                "planned_lr_first_last": [self.planned_lrs[0], self.planned_lrs[-1]] if self.planned_lrs else None,
                                                "gpu_total_gb": self.gpu_total_gb, "eval_subset": self.eval_subset})

    # --- trainer hooks -------------------------------------------------------------------------------------------------
    def observe_loss(self, trainer: Any, loss: Any, tag: RowTag | None, num_items_in_batch: Any) -> Any:
        if not self.enabled:
            return loss
        t0 = time.perf_counter()
        training = trainer.model.training
        mode = "train" if training else "eval"
        ent = acc = None
        try:
            ent = trainer._metrics[mode]["entropy"][-1]
            acc = trainer._metrics[mode]["mean_token_accuracy"][-1]
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        if training:
            if self.inject_step is not None and trainer.state.global_step + 1 == self.inject_step and not self._micro:
                loss = loss * float("nan")   # TEST HOOK (--smoke only): the stop-and-save path
                self.inject_step = None
            self._micro.append(_Micro(loss.detach(), tag, num_items_in_batch, ent, acc))
        else:
            if self.phase != "eval":
                self._set_phase("eval")
            self._eval_micro.append(_Micro(loss.detach(), tag, None, ent, acc))
        self._overhead_pending += time.perf_counter() - t0
        return loss

    def on_train_begin(self, args: Any, state: Any) -> None:
        if not self.enabled:
            return
        t0 = time.perf_counter()
        self.step = int(state.global_step)
        self._set_phase("training")
        if self.total_steps and state.max_steps and int(state.max_steps) != int(self.total_steps):
            self.emit([Alert("planned_steps_mismatch", "WARN", self.step, int(state.max_steps), int(self.total_steps),
                             f"the Trainer runs max_steps {state.max_steps} but the plan has {self.total_steps} optimizer steps")])
        if self.lora is not None:
            try:
                self.lora.snapshot()
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"LoRA snapshot failed: {exc!r}")
                self.lora = None
        self._t_hooks_end = time.perf_counter()
        self._last_progress = time.time()
        self._overhead_pending += time.perf_counter() - t0
        self.write_status()

    def on_step_begin(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._data_wait = now - self._t_hooks_end if self._t_hooks_end is not None else None

    def on_step_end(self, state: Any, control: Any, step_time: float) -> None:
        """After the optimizer step (the bridge has synchronised): resolve the step's micro-batch losses into rows and slices,
        read memory counters, LoRA health every N steps, stop on non-finite raw losses or data errors."""
        if not self.enabled:
            return
        t0 = time.perf_counter()
        step = int(state.global_step)
        now = time.time()
        with self.lock:
            self.step, self.epoch = step, state.epoch
            self._last_progress = now
        wall = (t0 - self._t_last_step_end) if self._t_last_step_end is not None else None
        self._t_last_step_end = t0
        info: dict[str, Any] = {"step_time_s": step_time, "data_wait_s": self._data_wait, "step_wall_s": wall}
        alerts: list[Alert] = []
        micro, self._micro = self._micro, []
        if micro:
            try:
                with self.lock:
                    info.update(self._resolve_train_rows(step, micro, alerts))
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"row bookkeeping failed: {exc!r}")
        if self.device == "cuda":
            import torch

            info.update(gpu_mem_alloc_gb=round(torch.cuda.memory_allocated() / 1e9, 3), gpu_mem_reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 3),
                        gpu_mem_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                        gpu_mem_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3), gpu_mem_total_gb=self.gpu_total_gb)
            self.peak_alloc_gb, self.peak_reserved_gb = info["gpu_mem_peak_alloc_gb"], info["gpu_mem_peak_reserved_gb"]
        if self.lora is not None and step % self.cfg.lora_every_steps == 0:
            alerts += self._lora_step(step)
        self._step_info = info
        self.emit(alerts)
        if any(a.severity == "CRITICAL" and a.code.startswith("nonfinite") for a in alerts) and self.cfg.stop_on_nonfinite:
            self._request_stop(control, alerts)
        elif any(a.severity == "CRITICAL" and a.code.startswith("data_") for a in alerts) and self.cfg.stop_on_data_error:
            self._request_stop(control, alerts)
        self._overhead_pending += time.perf_counter() - t0

    def _resolve_train_rows(self, step: int, micro: list[_Micro], alerts: list[Alert]) -> dict[str, Any]:
        import torch

        losses = torch.stack([m.loss.float().reshape(()) for m in micro]).cpu().tolist()
        nib = next((m.num_items for m in micro if m.num_items is not None), None)
        n_norm = int(nib.item() if hasattr(nib, "item") else nib) if nib is not None else None
        observed_mode = "tokens" if n_norm is not None else "rows"
        if self.loss_normalisation not in (None, observed_mode):
            alerts.append(Alert("loss_normalisation_mismatch", "WARN", step, observed_mode, self.loss_normalisation,
                                f"the Trainer normalised the step by {observed_mode} although {self.loss_normalisation} was configured"))
        self.loss_normalisation = observed_mode
        rows, slices = [], {}
        counted = 0
        for k, (m, c) in enumerate(zip(micro, losses)):
            tag_rows = m.tag.rows if m.tag is not None else []
            n_loss_mb = sum(r[2] for r in tag_rows)
            counted += n_loss_mb
            mean_mb = c * n_norm / n_loss_mb if n_norm and n_loss_mb else c
            for idx, n_tok, n_loss in tag_rows:
                meta = self.registry_meta[idx] if idx is not None else {}
                row = {"split": "train", "step": step, "micro_step": k, "row_index": meta.get("row_index"), "sample_id": meta.get("sample_id"),
                       "slice": meta.get("slice", "unknown"), "source": meta.get("source"), "task_type": meta.get("task_type"),
                       "mode": meta.get("mode"), "turn_index": meta.get("turn_index"), "n_tokens": n_tok, "n_loss_tokens": n_loss,
                       "loss": mean_mb, "loss_contribution": _contribution(c, n_loss, n_loss_mb, len(tag_rows), n_norm, len(micro)),
                       "shared_rows": len(tag_rows), "entropy": m.entropy, "token_accuracy": m.accuracy,
                       "source_n_tokens": meta.get("n_tokens")}
                rows.append(row)
        alerts += HealthChecks.on_rows("train", step, rows, self.max_length)
        # normalised by num_items_in_batch: the Trainer logs sum(c_k); otherwise it divides each c_k by the accumulation steps
        observed = float(sum(losses)) if n_norm is not None else float(sum(losses)) / len(losses)
        if n_norm is not None and counted and n_norm != counted:
            alerts.append(Alert("loss_normalisation_mismatch", "WARN", step, n_norm, counted,
                                f"num_items_in_batch {n_norm} != loss tokens counted on the collated rows {counted}"))
        for r in rows:
            s = slices.setdefault(r["slice"], {"nll": 0.0, "n_loss": 0, "n_tokens": 0, "n_rows": 0, "row_losses": [], "ent": [], "acc": []})
            if _finite(r["loss"]) and r["n_loss_tokens"]:
                s["nll"] += r["loss"] * r["n_loss_tokens"]
                s["row_losses"].append(r["loss"])
            elif not _finite(r["loss"]):
                s["nll"] = math.nan
            s["n_loss"] += r["n_loss_tokens"]
            s["n_tokens"] += r["n_tokens"]
            s["n_rows"] += 1
            if r["entropy"] is not None:
                s["ent"].append(r["entropy"])
            if r["token_accuracy"] is not None:
                s["acc"].append(r["token_accuracy"])
        slice_recs = {}
        for name, s in sorted(slices.items()):
            rec = {"split": "train", "step": step, "slice": name, "loss_tokens": s["nll"] / s["n_loss"] if s["n_loss"] else None,
                   "loss_rows": sum(s["row_losses"]) / len(s["row_losses"]) if s["row_losses"] else None, "n_rows": s["n_rows"],
                   "n_tokens": s["n_tokens"], "n_loss_tokens": s["n_loss"], "entropy": _mean(s["ent"]), "token_accuracy": _mean(s["acc"]),
                   "runtime_s": None}
            slice_recs[name] = rec
            self.slices_csv.write(rec)
            win = self.slice_window.setdefault(name, deque(maxlen=10))
            win.append((rec["loss_tokens"], s["n_loss"]))
        self._write_rows(rows)
        self.tokens_trained_total += counted
        return {"loss_observed": observed, "tokens_trained": counted, "n_rows": len(rows),
                "slices": {k: {"loss": v["loss_tokens"], "rows": v["n_rows"], "loss_tokens": v["n_loss_tokens"], "tokens": v["n_tokens"]}
                           for k, v in slice_recs.items()},
                "rows_unidentified": sum(1 for r in rows if r["row_index"] is None)}

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows or not self.row_tracking:
            return
        with self.rows_path.open("a") as f:
            for r in rows:
                rec = {k: r.get(k) for k in ROW_COLUMNS}
                f.write(json.dumps(_clean({"schema": SCHEMA, **rec}), default=str) + "\n")
                self.rows_csv.write(rec)

    def _lora_step(self, step: int) -> list[Alert]:
        t0 = time.perf_counter()
        try:
            res = self.lora.measure()
        except Exception as exc:  # noqa: BLE001
            self._note_error(f"LoRA health failed: {exc!r}")
            return []
        dt = time.perf_counter() - t0
        for fam, parts in res["groups"].items():
            for part, rec in parts.items():
                self.lora_csv.write({"step": step, "family": fam, "part": part, **{k: _r(v, 8) for k, v in rec.items()}})
        self.lora_last = {"step": step, "update_ratio": res["update_ratio"], "finite": res["finite"], "seconds": round(dt, 3),
                          "groups": {f: {p: {k: _r(v, 6) for k, v in r.items()} for p, r in parts.items()} for f, parts in res["groups"].items()}}
        self.run.metric({"event": "lora_health", "schema": SCHEMA, **self.lora_last})
        return self.checks.on_lora(step, res["update_ratio"], res["finite"])

    def _request_stop(self, control: Any, alerts: list[Alert]) -> None:
        if self.stop_reason is not None or control is None:
            return
        crit = next(a for a in alerts if a.severity == "CRITICAL")
        self.stop_reason = f"{crit.code} at step {crit.step}: {crit.message}"
        control.should_save = True
        control.should_training_stop = True
        control.should_evaluate = False
        if self.lora is not None:
            try:
                self.weights_finite_at_stop = bool(self.lora.measure()["finite"])
            except Exception:  # noqa: BLE001
                self.weights_finite_at_stop = None
        self.run.log(f"monitor: STOP requested ({self.stop_reason}); saving checkpoint-{crit.step} (forensic; adapter weights finite: "
                     f"{self.weights_finite_at_stop}) and ending training")
        self._set_phase("saving")

    def on_train_log(self, step: int, logs: dict[str, Any], rec: dict[str, Any], control: Any) -> dict[str, Any]:
        """Monitor fields for TrainMetrics' per-step record; runs the step detectors; writes the health row and status."""
        if not self.enabled:
            return {}
        t0 = time.perf_counter()
        info, self._step_info = self._step_info, {}
        loss, observed = logs.get("loss"), info.get("loss_observed")
        x = observed if observed is not None else loss
        if _finite(x):
            b = self.cfg.loss_ema_beta
            self._ema_n += 1
            self._ema_raw = b * self._ema_raw + (1 - b) * x
            self.ema = self._ema_raw / (1 - b ** self._ema_n)
            if self.loss_min is None or x < self.loss_min[0]:
                self.loss_min = (x, step)
        lr = logs.get("learning_rate")
        planned = self.planned_lrs[step - 1] if self.planned_lrs and 1 <= step <= len(self.planned_lrs) else None
        g = logs.get("grad_norm")
        tokens_seq = rec.get("tokens_this_step") or 0
        self.tokens_seq_total = rec.get("tokens_cumulative") or self.tokens_seq_total
        st = info.get("step_time_s") or rec.get("step_time_s")
        trained = info.get("tokens_trained")
        tps_trained = round(trained / st, 1) if trained and st else None
        if _finite(rec.get("tokens_per_s")):
            self.tps_seq.append((step, rec["tokens_per_s"]))
        if tps_trained:
            self.tps_trained.append((step, tps_trained))
        with self.lock:
            if st:
                self.step_time_s.append(st)
            if info.get("data_wait_s") is not None:
                self.data_wait_s.append(info["data_wait_s"])
            if info.get("step_wall_s"):
                self.step_walls.append(info["step_wall_s"])
        gpu = dict(self.gpu_last or {})
        fields_ = {
            "schema": SCHEMA, "loss_observed": observed, "loss_ema": self.ema, "lr_planned": planned,
            "lr_rel_dev": (abs(lr - planned) / planned if _finite(lr) and planned else None),
            "max_grad_norm": self.max_grad_norm, "grad_clipped": bool(_finite(g) and self.max_grad_norm and g > self.max_grad_norm),
            "tokens_seq": tokens_seq, "tokens_trained": trained, "tokens_trained_cumulative": self.tokens_trained_total if trained is not None else None,
            "tokens_trained_per_s": tps_trained, "data_wait_s": _r(info.get("data_wait_s"), 4), "step_wall_s": _r(info.get("step_wall_s"), 3),
            **{k: info.get(k) for k in ("gpu_mem_alloc_gb", "gpu_mem_reserved_gb", "gpu_mem_peak_alloc_gb", "gpu_mem_peak_reserved_gb",
                                        "gpu_mem_total_gb", "n_rows", "rows_unidentified")},
            "slices": info.get("slices"), "gpu": {k: gpu.get(k) for k in ("gpu_util_pct", "power_w", "temp_c", "sm_clock_mhz", "throttle_reasons",
                                                                          "mem_used_gb")} if gpu else None,
        }
        check_rec = {**rec, **fields_, "step": step, "loss": loss, "grad_norm": g, "learning_rate": lr}
        with self.lock:
            alerts = self.checks.on_step(check_rec)
        if observed is not None and _finite(observed) and _finite(loss) and abs(observed - loss) > 1e-4 * max(1.0, abs(loss)):
            alerts.append(Alert("loss_bookkeeping_mismatch", "WARN", step, _r(observed), _r(loss),
                                f"sum of the step's micro-batch losses {observed:.6g} != logged loss {loss:.6g}"))
        emitted = self.emit(alerts)
        fields_["clip_rate"] = _r(self.checks.clip_rate, 4)
        fields_["alerts"] = [a["code"] for a in emitted] or None
        if any(a.severity == "CRITICAL" and a.code.startswith("nonfinite") for a in alerts) and self.cfg.stop_on_nonfinite:
            self._request_stop(control, alerts)
        if control is not None and self.stop_reason is None:
            if getattr(control, "should_evaluate", False):
                self._set_phase("eval")
            elif getattr(control, "should_save", False):
                self._set_phase("saving")
        self.last = {"step": step, "loss": loss, "loss_observed": observed, "loss_ema": self.ema, "learning_rate": lr, "lr_planned": planned,
                     "grad_norm": g, "tokens_per_s": rec.get("tokens_per_s"), "tokens_trained_per_s": tps_trained,
                     "entropy": logs.get("entropy"), "mean_token_accuracy": logs.get("mean_token_accuracy"), "rss_gb": rec.get("rss_gb"),
                     **{k: info.get(k) for k in ("gpu_mem_alloc_gb", "gpu_mem_reserved_gb")}}
        monitor_s = self._overhead_pending + (time.perf_counter() - t0)
        self._overhead_pending = 0.0
        self.monitor_s.append(monitor_s)
        fields_["monitor_s"] = round(monitor_s, 4)
        self.health_csv.write({**{k: rec.get(k) for k in ("step", "epoch", "elapsed_s", "loss", "learning_rate", "grad_norm", "tokens_per_s",
                                                         "step_time_s", "rss_gb", "entropy", "mean_token_accuracy")},
                               **{k: v for k, v in fields_.items() if k in HEALTH_COLUMNS},
                               "gpu_util_pct": gpu.get("gpu_util_pct"), "gpu_power_w": gpu.get("power_w"), "gpu_temp_c": gpu.get("temp_c"),
                               "gpu_sm_clock_mhz": gpu.get("sm_clock_mhz"), "alerts_warn": self.alert_counts["WARN"],
                               "alerts_critical": self.alert_counts["CRITICAL"]})
        self.write_status()
        self._t_hooks_end = time.perf_counter()
        return {k: v for k, v in fields_.items() if v is not None or k in ("loss_observed",)}

    def log_suffix(self, rec: dict[str, Any]) -> str:
        """Compact monitor part of the per-step train.log line."""
        if not self.enabled:
            return ""
        parts = []
        if _finite(rec.get("loss_ema")):
            parts.append(f"ema={rec['loss_ema']:.4f}")
        if rec.get("tokens_trained_per_s"):
            parts.append(f"trained tok/s={rec['tokens_trained_per_s']}")
        if _finite(rec.get("data_wait_s")):
            parts.append(f"wait={rec['data_wait_s']:.2f}s")
        if _finite(rec.get("gpu_mem_reserved_gb")):
            parts.append(f"reserved={rec['gpu_mem_reserved_gb']:.1f}GB")
        if rec.get("slices"):
            parts.append("slices " + " ".join(f"{k}={v['loss']:.3f}" for k, v in rec["slices"].items() if _finite(v.get("loss"))))
        return (", " + ", ".join(parts)) if parts else ""

    def on_eval_log(self, step: int, logs: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
        """Per-slice eval losses from the rows evaluated since the last evaluation; eval detectors; status."""
        if not self.enabled:
            return {}
        t0 = time.perf_counter()
        micro, self._eval_micro = self._eval_micro, []
        out: dict[str, Any] = {"schema": SCHEMA}
        slices: dict[str, dict[str, Any]] = {}
        alerts: list[Alert] = []
        if micro:
            import torch

            losses = torch.stack([m.loss.float().reshape(()) for m in micro]).cpu().tolist()
            rows = []
            for k, (m, c) in enumerate(zip(micro, losses)):
                tag_rows = m.tag.rows if m.tag is not None else []
                for idx, n_tok, n_loss in tag_rows:
                    meta = self.registry_meta[idx] if idx is not None else {}
                    rows.append({"split": "eval", "step": step, "micro_step": k, "row_index": meta.get("row_index"),
                                 "sample_id": meta.get("sample_id"), "slice": meta.get("slice", "unknown"), "source": meta.get("source"),
                                 "task_type": meta.get("task_type"), "mode": meta.get("mode"), "turn_index": meta.get("turn_index"),
                                 "n_tokens": n_tok, "n_loss_tokens": n_loss, "loss": c, "loss_contribution": None, "shared_rows": len(tag_rows),
                                 "entropy": m.entropy, "token_accuracy": m.accuracy, "source_n_tokens": meta.get("n_tokens")})
            alerts += HealthChecks.on_rows("eval", step, rows, self.max_length)
            for r in rows:
                s = slices.setdefault(r["slice"], {"losses": [], "nll": 0.0, "n_loss": 0, "n_tokens": 0, "ent": [], "acc": []})
                s["losses"].append(r["loss"])
                s["nll"] += r["loss"] * r["n_loss_tokens"]
                s["n_loss"] += r["n_loss_tokens"]
                s["n_tokens"] += r["n_tokens"]
                for key, col in (("ent", "entropy"), ("acc", "token_accuracy")):
                    if r[col] is not None:
                        s[key].append(r[col])
            self._write_rows(rows)
            if rows:
                out["eval_loss_from_rows"] = sum(r["loss"] for r in rows) / len(rows)
                out["eval_rows"] = len(rows)
        runtime = logs.get("eval_runtime")
        overall = logs.get("eval_loss")                        # the Trainer's eval_loss, row-weighted as before
        groups = dict(sorted(slices.items()))
        if slices:   # the whole evaluation as slice `all` (its loss_rows equals eval_loss)
            groups["all"] = {"losses": [x for g in slices.values() for x in g["losses"]], "nll": sum(g["nll"] for g in slices.values()),
                             "n_loss": sum(g["n_loss"] for g in slices.values()), "n_tokens": sum(g["n_tokens"] for g in slices.values()),
                             "ent": [x for g in slices.values() for x in g["ent"]], "acc": [x for g in slices.values() for x in g["acc"]]}
        slice_rows, slice_tokens = {}, {}
        for name, s in groups.items():
            srec = {"split": "eval", "step": step, "slice": name, "loss_tokens": s["nll"] / s["n_loss"] if s["n_loss"] else None,
                    "loss_rows": sum(s["losses"]) / len(s["losses"]), "n_rows": len(s["losses"]), "n_tokens": s["n_tokens"],
                    "n_loss_tokens": s["n_loss"], "entropy": _mean(s["ent"]), "token_accuracy": _mean(s["acc"]),
                    "runtime_s": runtime if name == "all" else None}
            self.slices_csv.write(srec)
            if name != "all":
                slice_rows[name], slice_tokens[name] = srec["loss_rows"], srec["loss_tokens"]
        overall_tokens = groups["all"]["nll"] / groups["all"]["n_loss"] if slices and groups["all"]["n_loss"] else None
        weighting = "tokens" if self.loss_normalisation == "tokens" else "rows"   # the detectors follow the training objective
        check_overall = overall_tokens if weighting == "tokens" and overall_tokens is not None else overall
        check_slices = slice_tokens if weighting == "tokens" else slice_rows
        for key, value in (("overall", overall), ("overall_tokens", overall_tokens), *check_slices.items()):
            best = self.best_eval.get(key)
            if _finite(value) and (best is None or value < best[0]):
                self.best_eval[key] = (value, step)
        out.update(eval_loss_tokens=overall_tokens, slices=slice_rows or None, slices_tokens=slice_tokens or None, eval_wall_s=runtime,
                   eval_check_weighting=weighting)
        if _finite(runtime):
            self.eval_walls.append(runtime)
        with self.lock:
            alerts += self.checks.on_eval(step, check_overall, check_slices, self.ema, weighting=weighting)
        out["alerts"] = [a["code"] for a in self.emit(alerts)] or None
        self.last_eval = {"step": step, "loss": overall, "loss_tokens": overall_tokens, "slices": slice_rows, "slices_tokens": slice_tokens,
                          "wall_s": runtime, "rows": out.get("eval_rows")}
        self._overhead_pending += time.perf_counter() - t0
        self.write_status()
        return {k: v for k, v in out.items() if v is not None}

    def on_evaluate(self, control: Any) -> None:
        if not self.enabled:
            return
        self._set_phase("saving" if getattr(control, "should_save", False) else "training")
        self._t_hooks_end = time.perf_counter()

    def on_save(self, step: int, checkpoint: Path) -> None:
        if not self.enabled:
            return
        self.checkpoints.append(step)
        if self.stop_reason is None:
            self._set_phase("training")
        self._t_hooks_end = time.perf_counter()
        self._check_disk(force=True)
        self.write_status()

    def on_train_end(self) -> None:
        """Final LoRA measurement (unless the last step had one), phase finishing (adapter save) or stopped."""
        if not self.enabled:
            return
        if self.lora is not None and (self.lora_last is None or self.lora_last["step"] != self.step) and self.step > 0:
            with self.lock:
                alerts = self._lora_step(self.step)
            self.emit(alerts)
        self._set_phase("stopped" if self.stop_reason else "finishing")
        self.write_status()

    # --- GPU, disk, watchdog -------------------------------------------------------------------------------------------
    def _on_gpu_sample(self, s: dict[str, Any]) -> None:
        now = time.time()
        with self.lock:
            s = {"utc": utc_now(), "elapsed_s": round(now - self.t_start, 1), "step": self.step, "phase": self.phase, **s}
            self.gpu_last = s
            if len(self.gpu_samples) < 100_000:
                self.gpu_samples.append({k: s.get(k) for k in ("gpu_util_pct", "power_w", "temp_c", "phase")})
            if self.gpu_csv is None:
                self.gpu_csv = CsvAppender(self.gpu_csv_path, GPU_COLUMNS, append=self._resume or self.gpu_csv_path.exists())
            self.gpu_csv.write({k: (hex(v) if k == "throttle_mask" and isinstance(v, int) else v) for k, v in s.items()})
            alerts = self.checks.on_gpu(s, phase=self.phase, now=now, step=self.step)
        self.emit(alerts)

    def _check_disk(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_disk_check < 600:
            return
        self._last_disk_check = now
        path = self.checkpoint_root if self.checkpoint_root.exists() else self.checkpoint_root.parent
        try:
            free = shutil.disk_usage(path).free / 1e9
        except OSError:
            return
        self.disk_free_gb = round(free, 1)
        self.emit(self.checks.on_disk(free, str(path), self.step))

    def median_step_s(self) -> float | None:
        return median(self.step_walls) if self.step_walls else (median(self.step_time_s[-20:]) if self.step_time_s else None)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self.cfg.watchdog_every_s):
            try:
                self.watchdog_tick()
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"watchdog error: {exc!r}")

    def watchdog_tick(self, now: float | None = None) -> list[dict[str, Any]]:
        """One watchdog pass: stall detection (with a stack dump on the first detection), disk, status heartbeat."""
        now = now or time.time()
        with self.lock:
            phase, idle = self.phase, now - self._last_progress
        was_stalled = self.checks.stalled
        alerts = self.checks.on_watchdog(phase=phase, idle_s=idle, step=self.step, median_step_s=self.median_step_s())
        if alerts and alerts[0].code == "stall" and not was_stalled:
            dump = self.run.run_dir / f"stall_traceback_step{self.step}.txt"
            try:
                with dump.open("a") as f:
                    f.write(f"[{utc_now()}] stall in phase {phase} after {idle:.0f} s without progress; stacks of all threads:\n")
                    f.flush()
                    faulthandler.dump_traceback(file=f, all_threads=True)
                alerts[0].context["traceback"] = str(dump)
            except OSError as exc:
                self._note_error(f"stall traceback dump failed: {exc!r}")
        written = self.emit(alerts)
        self._check_disk()
        self.write_status()
        return written

    # --- status --------------------------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            total = self.total_steps
            pct = round(100.0 * self.step / total, 2) if total else None
            med = self.median_step_s()
            remaining = max(total - self.step, 0) if total else None
            eta = None
            if remaining is not None and med and self.phase not in ("finished", "stopped", "failed", "aborted"):
                n_evals_left = 0
                if self.eval_walls and self.eval_every:
                    n_evals_left = remaining // self.eval_every
                eta = remaining * med + n_evals_left * (self.eval_walls[-1] if self.eval_walls else 0.0)
            slices_train = {}
            for name, win in self.slice_window.items():
                pairs = [(l, n) for l, n in win if _finite(l) and n]
                if pairs:
                    slices_train[name] = sum(l * n for l, n in pairs) / sum(n for _, n in pairs)
            return _clean({
                "schema": SCHEMA, "run_id": self.run.run_id, "stage": self.stage, "method": self.method, "arm": self.arm,
                "host": socket.gethostname(), "pid": os.getpid(), "device": self.device, "phase": self.phase,
                "phase_since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.phase_since)),
                "step": self.step, "total_steps": total, "percent": pct, "epoch": self.epoch,
                "elapsed_s": round(now - self.t_start, 1), "seconds_since_progress": round(now - self._last_progress, 1),
                "step_wall_s_median": _r(med, 2), "eta_s": round(eta, 0) if eta is not None else None,
                "eta_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + eta)) if eta is not None else None,
                "loss": {"last": self.last.get("loss"), "observed": self.last.get("loss_observed"), "ema": self.ema,
                         "min": self.loss_min[0] if self.loss_min else None, "min_step": self.loss_min[1] if self.loss_min else None},
                "lr": {"last": self.last.get("learning_rate"), "planned": self.last.get("lr_planned")},
                "grad_norm": {"last": self.last.get("grad_norm"), "median": median(self.checks.grads) if self.checks.grads else None,
                              "clip_rate": self.checks.clip_rate, "max_grad_norm": self.max_grad_norm},
                "tokens": {"seq_per_s": self.last.get("tokens_per_s"), "trained_per_s": self.last.get("tokens_trained_per_s"),
                           "seq_total": self.tokens_seq_total, "trained_total": self.tokens_trained_total},
                "token_stats": {"entropy": self.last.get("entropy"), "accuracy": self.last.get("mean_token_accuracy")},
                "slices_train": slices_train, "eval": self.last_eval,
                "best_eval": {k: {"loss": v[0], "step": v[1]} for k, v in self.best_eval.items()},
                "memory": {"alloc_gb": self.last.get("gpu_mem_alloc_gb"), "reserved_gb": self.last.get("gpu_mem_reserved_gb"),
                           "peak_alloc_gb": self.peak_alloc_gb, "peak_reserved_gb": self.peak_reserved_gb, "total_gb": self.gpu_total_gb,
                           "rss_gb": round(current_rss_gb(), 3)},
                "gpu": self.gpu_last, "gpu_telemetry": self.gpu_sampler.status if self.gpu_sampler else None,
                "lora": self._lora_status(),
                "checkpoints": self.checkpoints, "disk_free_gb": self.disk_free_gb,
                "alerts": {"counts": {s: self.alert_counts[s] for s in SEVERITIES}, "suppressed": self.alerts_suppressed,
                           "last": [rec for _, rec in list(self.alerts_recent)[-3:]]},
                "loss_normalisation": self.loss_normalisation, "health": self.health(now), "stop_reason": self.stop_reason, "weights_finite_at_stop": self.weights_finite_at_stop,
                "monitor_s_mean": _r(_mean(self.monitor_s), 4),
                "updated_at": utc_now(), "updated_unix": round(now, 3),
            })

    def _lora_status(self) -> dict[str, Any] | None:
        if not self.lora_last:
            return None
        overall = self.lora_last["groups"].get("all", {}).get("all", {})
        return {"step": self.lora_last["step"], "update_ratio": self.lora_last["update_ratio"], "finite": self.lora_last["finite"],
                "norm": overall.get("norm"), "delta_w_rel": overall.get("delta_w_rel"),
                "update_ratio_by_family": {f: p.get("all", {}).get("update_ratio") for f, p in self.lora_last["groups"].items() if f != "all"}}

    def write_status(self) -> None:
        if not self.enabled:
            return
        try:
            with self.lock:
                write_json_atomic(self.status_path, self.status())
        except Exception as exc:  # noqa: BLE001
            self._note_error(f"status.json write failed: {exc!r}")

    # --- end of run ----------------------------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """End-of-run block for metadata.json (`monitor`)."""
        if not self.enabled:
            return {"enabled": False}
        steady = [v for s, v in self.tps_seq if s > 2]
        steady_trained = [v for s, v in self.tps_trained if s > 2]
        total_step = sum(self.step_time_s)
        util = [g["gpu_util_pct"] for g in self.gpu_samples if g.get("phase") == "training" and _finite(g.get("gpu_util_pct"))]
        power = [g["power_w"] for g in self.gpu_samples if g.get("phase") == "training" and _finite(g.get("power_w"))]
        temps = [g["temp_c"] for g in self.gpu_samples if _finite(g.get("temp_c"))]
        return _clean({
            "schema": SCHEMA, "steps_logged": len(self.step_time_s), "final_step": self.step, "stop_reason": self.stop_reason,
            "loss_normalisation": self.loss_normalisation,
            "weights_finite_at_stop": self.weights_finite_at_stop,
            "loss": {"last": self.last.get("loss"), "last_observed": self.last.get("loss_observed"), "ema": self.ema,
                     "min": self.loss_min[0] if self.loss_min else None, "min_step": self.loss_min[1] if self.loss_min else None},
            "slices_train_last": {k: v for k, v in self.status().get("slices_train", {}).items()},
            "eval_last": self.last_eval, "best_eval": {k: {"loss": v[0], "step": v[1]} for k, v in self.best_eval.items()},
            "eval_subset": self.eval_subset, "eval_wall_s_total": round(sum(self.eval_walls), 1), "n_evals": len(self.eval_walls),
            "tokens": {"seq_total": self.tokens_seq_total, "trained_total": self.tokens_trained_total},
            "throughput": {"seq_tok_s_mean": _r(_mean([v for _, v in self.tps_seq]), 1), "seq_tok_s_steady_median": _r(_median(steady), 1),
                           "trained_tok_s_steady_median": _r(_median(steady_trained), 1),
                           "data_wait_share": _r(sum(self.data_wait_s) / (sum(self.data_wait_s) + total_step), 5) if total_step else None},
            "memory": {"peak_alloc_gb": self.peak_alloc_gb, "peak_reserved_gb": self.peak_reserved_gb, "total_gb": self.gpu_total_gb,
                       "peak_rss_gb": round(peak_rss_gb(), 2)},
            "gpu": {"util_pct_mean_training": _r(_mean(util), 1), "power_w_mean_training": _r(_mean(power), 1), "temp_c_max": _r(max(temps), 1) if temps else None,
                    "samples": len(self.gpu_samples), "telemetry": self.gpu_sampler.status if self.gpu_sampler else None},
            "lora_last": self.lora_last, "alerts": {"counts": {s: self.alert_counts[s] for s in SEVERITIES}, "suppressed": self.alerts_suppressed},
            "overhead": {"monitor_s_per_step_mean": _r(_mean(self.monitor_s), 4), "monitor_s_per_step_max": _r(max(self.monitor_s), 4) if self.monitor_s else None,
                         "share_of_step_time": _r(sum(self.monitor_s) / total_step, 5) if total_step else None,
                         "lora_health_s_last": self.lora_last.get("seconds") if self.lora_last else None},
            "checkpoints": self.checkpoints,
        })

    def resource_rows(self) -> list[tuple[str, Any, str, str]]:
        """Extra resources.csv rows (the monitor's measurements)."""
        if not self.enabled:
            return []
        s = self.summary()
        na = lambda v: "n/a" if v is None else v  # noqa: E731
        return [
            ("trained_tokens_processed", s["tokens"]["trained_total"], "tokens", "loss tokens of the rows trained by this run (monitor)"),
            ("tokens_per_s_steady_median", na(s["throughput"]["seq_tok_s_steady_median"]), "tok/s", "median sequence tokens/s after step 2"),
            ("trained_tokens_per_s_steady_median", na(s["throughput"]["trained_tok_s_steady_median"]), "tok/s", "median loss tokens/s after step 2"),
            ("data_wait_share", na(s["throughput"]["data_wait_share"]), "fraction", "host-side batch fetch time / (fetch + step time)"),
            ("eval_wall_sec_total", s["eval_wall_s_total"], "s", f"{s['n_evals']} evaluations (Trainer eval_runtime)"),
            ("gpu_util_pct_mean", na(s["gpu"]["util_pct_mean_training"]), "%", "nvidia-smi samples while training"),
            ("gpu_power_w_mean", na(s["gpu"]["power_w_mean_training"]), "W", "nvidia-smi samples while training"),
            ("gpu_temp_c_max", na(s["gpu"]["temp_c_max"]), "C", "nvidia-smi samples"),
            ("alerts_warn", s["alerts"]["counts"]["WARN"], "count", "logs/<run_id>/alerts.jsonl"),
            ("alerts_critical", s["alerts"]["counts"]["CRITICAL"], "count", "logs/<run_id>/alerts.jsonl"),
            ("monitor_overhead_sec_per_step", na(s["overhead"]["monitor_s_per_step_mean"]), "s", "time spent in the monitor per optimizer step"),
            ("monitor_overhead_share", na(s["overhead"]["share_of_step_time"]), "fraction", "monitor time / step time"),
        ]

    def close(self, phase: str, reason: str | None = None) -> None:
        """Stop the threads and write the final status (phase finished / stopped / failed / aborted)."""
        if not self.enabled or self.closed:
            return
        self.closed = True
        self._stop_event.set()
        if self.gpu_sampler is not None:
            self.gpu_sampler.stop()
        if self.watchdog is not None and self.watchdog is not threading.current_thread():
            self.watchdog.join(timeout=5)
        with self.lock:
            self.phase, self.phase_since = phase, time.time()
            if reason and not self.stop_reason and phase != "finished":
                self.stop_reason = reason
        try:
            write_json_atomic(self.status_path, self.status())
        except Exception as exc:  # noqa: BLE001
            self._note_error(f"final status.json write failed: {exc!r}")


def stop_monitored_run(run: Any, monitor: TrainingMonitor, metrics: Any, *, device: str, global_step: int,
                       trained_tokens: int | None) -> None:
    """The monitor stopped training (CRITICAL alert; its checkpoint is saved): record status `stopped` with resources and the
    monitor summary, no final adapter, then raise MonitorStop (exit 1). Shared by apex_train.sft and apex_train.stage2."""
    ckpts = [c["step"] for c in metrics.checkpoints]
    stop_step = monitor.step
    before = [s for s in ckpts if s < stop_step]
    advice = (f"resume from checkpoint-{before[-1]} (the last checkpoint before step {stop_step})" if before
              else "no checkpoint precedes the stop: restart the run")
    rows = training_resource_rows(run, metrics, device=device, trained_tokens=trained_tokens) + monitor.resource_rows()
    summary = {"global_step": global_step, "steps_this_run": len(metrics.train_records), "stopped_by_monitor": monitor.stop_reason,
               "forensic_checkpoint_step": ckpts[-1] if ckpts and ckpts[-1] == stop_step else None,
               "weights_finite_at_stop": monitor.weights_finite_at_stop, "advice": advice, "checkpoints": metrics.checkpoints,
               "eval": metrics.eval_records, "resources": {r[0]: r[1] for r in rows}}
    run.finish("stopped", rows, summary=summary, monitor=monitor.summary())
    raise MonitorStop(f"{monitor.stop_reason}; {advice}")


def _contribution(c: float, n_loss: int, n_loss_mb: int, rows_in_mb: int, n_norm: int | None, n_micro: int) -> float | None:
    """A row's share of the logged step loss: c_k itself (micro-batch 1, normalised), its token share of a multi-row micro-batch,
    and divided by the accumulation steps when the loss was not normalised by num_items_in_batch."""
    share = c if rows_in_mb == 1 else (c * n_loss / n_loss_mb if n_loss_mb else None)
    return share if share is None or n_norm is not None else share / n_micro


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else None


def _median(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if _finite(x)]
    return median(xs) if xs else None


def planned_lr_schedule(args: Any, total_steps: int) -> list[float]:
    """The learning rate the configured schedule gives each optimizer step (index s-1 = the LR logged at step s), built
    independently of the Trainer from the planned step count: transformers.get_scheduler on a one-parameter optimizer."""
    import torch
    from transformers import get_scheduler

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=float(args.learning_rate))
    sched = get_scheduler(args.lr_scheduler_type, opt, num_warmup_steps=args.get_warmup_steps(total_steps), num_training_steps=total_steps,
                          scheduler_specific_kwargs=args.lr_scheduler_kwargs or {})
    lrs = []
    for _ in range(total_steps):
        lrs.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sched.step()
    return lrs
