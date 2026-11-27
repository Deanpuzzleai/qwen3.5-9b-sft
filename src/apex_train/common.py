"""Shared helpers for the training side (apex_train.sft, apex_train.stage2, apex_train.merge).

Scope: YAML configs with `${VAR:-default}` expansion and dotted overrides; the training run directory (train.log
variant of the run-directory layout, built on apex_eval.common.RunDir so crash, SIGTERM, atomic metadata and the five
files behave exactly like evaluation runs); base-model and LoRA construction for the locked dense base Qwen/Qwen3.5-9B
(configs/sft.yaml `model:` / `lora:`; REPORT.md section 2 of the archived 35B training-stack study documents the same flow as
verified for the previous base); the TrainingArguments block both TRL trainers share; one TrainerCallback that mirrors every step into
metrics.jsonl and a curves CSV and drives the training monitor (apex_train.monitor: health checks, alerts, status.json, per-row and
per-slice losses); the labels == assistant_masks self-checks; adapter saving; resource accounting.

Nothing here imports torch, transformers, peft or trl at module level (so --help stays fast); the pure-torch
fused-LM-head shim is installed only for CPU dry runs when TRL's Triton kernel is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import re
import resource
import shlex
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from apex_eval.common import (REPO_ROOT, RunDir, compute_rates, dir_bytes, git_info, hardware_info, model_lock,
                              sha256_file, utc_now)

# Qwen/Qwen3.5-9B (configs/eval/model.yaml): dense hybrid decoder (32 layers = 8 x (3 Gated DeltaNet + 1 full attention),
# dense MLP) behind a vision wrapper. Training, merging and serving use the wrapper class; the text-only class is refused
# outside dry runs (it saves a text-only config that silently reloads with a random vision tower).
SERVED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TEXT_ONLY_ARCHITECTURE = "Qwen3_5ForCausalLM"
AUTO_CLASS_FOR_ARCH = {SERVED_ARCHITECTURE: "AutoModelForImageTextToText", TEXT_ONLY_ARCHITECTURE: "AutoModelForCausalLM"}
MTP_PREFIX = "mtp."   # the checkpoint's 15 multi-token-prediction tensors; unused by the HF class (unexpected keys on load)
# Adapter key prefix on the served class; the text-only class would yield base_model.model.model.layers.*
ADAPTER_KEY_PREFIX = "base_model.model.model.language_model.layers."
# Names LoRA must never target: `gate` (a router name that must not appear in a dense recipe), the GatedDeltaNet conv1d,
# embeddings, the LM head, and PEFT's all-linear (it would also wrap the vision tower's Linears).
FORBIDDEN_TARGETS = ("gate", "conv1d", "embed_tokens", "lm_head", "all-linear")
# Shrunken served-class twin for CPU dry runs (random init, real tokenizer); git-ignored, built on demand by the smokes.
TINY_TWIN_DIR = REPO_ROOT / "checkpoints" / "gpu_checks" / "tiny_cg_twin"
TINY_TWIN_BUILDER = REPO_ROOT / "scripts" / "dev" / "gpu_checks" / "make_tiny_cg_twin.py"
KERNEL_PACKAGES = {"fla": "flash-linear-attention (chunk_gated_delta_rule)", "causal_conv1d": "causal-conv1d (GDN conv)"}
FALLBACK_RE = re.compile(r"`(\w+)` is falling back to its reference PyTorch implementation")
CAPTURED_LOGGERS = ("transformers", "trl", "peft", "accelerate", "datasets", "torch", "py.warnings", "")
PACKAGES = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "safetensors")
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_REQUIRED = object()


class ConfigError(ValueError):
    """A YAML config is missing a field, has the wrong type, or asks for something the recipe refuses."""


# ------------------------------------------------------------------------------------------------- small utilities
def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # ru_maxrss is kB on Linux


def current_rss_gb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def torch_env() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                            "cuda_version": torch.version.cuda, "threads": torch.get_num_threads(),
                            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    if torch.cuda.is_available():
        info.update(device_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    device_count=torch.cuda.device_count(), total_memory_gb=round(torch.cuda.mem_get_info()[1] / 1e9, 1))
    return info


def gpu_peak_gb() -> tuple[float | None, float | None]:
    """(max_memory_allocated, max_memory_reserved) in GB on cuda:0, or (None, None) without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None, None
    return round(torch.cuda.max_memory_allocated() / 1e9, 3), round(torch.cuda.max_memory_reserved() / 1e9, 3)


def seed_everything(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)


def kernel_preflight() -> dict[str, bool]:
    """Importability of the GDN kernels (REPORT: without fla / causal-conv1d training is correct but far slower)."""
    return {name: importlib.util.find_spec(name) is not None for name in KERNEL_PACKAGES}


def run_cli(run: TrainRun, name: str, cmd: list[str]) -> None:
    """Run a helper command from the repository root with its output appended to train.log; raise on failure."""
    run.log(f"{name}: {shlex.join(cmd)}")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    with (run.run_dir / run.log_name).open("a") as f:
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit {proc.returncode}; see {run.log_name}")


def twin_revision(model_dir: Path) -> str:
    """Identity of a local checkpoint without a hub revision (the smoke twin): written as the adapter's `revision`."""
    return "local-config-sha256-" + sha256_file(Path(model_dir) / "config.json")[:12]


def ensure_tiny_twin(path: Path | str, run: TrainRun | None = None) -> Path:
    """Dry runs only: build the shrunken served-class twin (scripts/dev/gpu_checks/make_tiny_cg_twin.py) when `path`
    holds no weights yet. Logs through `run` when given (train.log), else prints."""
    path = resolve_path(path)
    if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    cmd = [sys.executable, str(TINY_TWIN_BUILDER), "--out", str(path)]
    if run is not None:
        run_cli(run, "build tiny twin", cmd)
        return path
    print(f"building the tiny twin: {shlex.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode != 0:
        raise RuntimeError(f"tiny twin build failed: {shlex.join(cmd)}")
    return path


# ------------------------------------------------------------------------------------------------- config loading
def expand_env(obj: Any) -> Any:
    """Expand `${VAR}` / `${VAR:-default}` in every string of a nested config; an unset VAR without default raises."""
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str):
        def sub(m: re.Match) -> str:
            value = os.environ.get(m.group(1))
            if value is None:
                if m.group(2) is None:
                    raise ConfigError(f"environment variable {m.group(1)} is not set and the config gives no default")
                return m.group(2)
            return value
        return ENV_PATTERN.sub(sub, obj)
    return obj


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Parse a YAML mapping and expand environment references; anything else is a ConfigError."""
    path = resolve_path(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config {path} is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"config {path} must be a YAML mapping at the top level (got {type(doc).__name__})")
    return expand_env(doc)


def get_dotted(cfg: dict[str, Any], key: str, default: Any = _REQUIRED) -> Any:
    """cfg['a']['b'] for key 'a.b'; ConfigError when a required key is missing."""
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _REQUIRED:
                raise ConfigError(f"config is missing required field '{key}'")
            return default
        node = node[part]
    return node


def set_dotted(cfg: dict[str, Any], key: str, value: Any) -> None:
    """Set cfg['a']['b'] = value for key 'a.b', creating intermediate mappings."""
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_override(item: str) -> tuple[str, Any]:
    """'a.b=VALUE' -> ('a.b', YAML-parsed VALUE), with int/float fallbacks for `1e-4`-style text; `x.y=null` -> None."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override {item!r} must look like key.path=value")
    try:
        value: Any = yaml.safe_load(raw) if raw.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {item!r}: value is not valid YAML: {exc}") from exc
    if isinstance(value, str):
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
    return key.strip(), value


def apply_overrides(cfg: dict[str, Any], mapping: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Set dotted keys in place; returns {key, from, to, source} records for metadata.json."""
    applied = []
    for key, value in mapping.items():
        applied.append({"key": key, "from": get_dotted(cfg, key, None), "to": value, "source": source})
        set_dotted(cfg, key, value)
    return applied


def require_type(cfg: dict[str, Any], key: str, kinds: type | tuple[type, ...]) -> Any:
    """get_dotted + isinstance check with a readable error (bool is rejected where a number is expected)."""
    value = get_dotted(cfg, key)
    kinds_t = kinds if isinstance(kinds, tuple) else (kinds,)
    ok = isinstance(value, kinds_t) and not (isinstance(value, bool) and bool not in kinds_t)
    if not ok:
        raise ConfigError(f"config field '{key}' must be {'|'.join(k.__name__ for k in kinds_t)}, got {type(value).__name__} ({value!r})")
    return value


# ------------------------------------------------------------------------------------------------- run bookkeeping
class LogCapture(logging.Handler):
    """WARNING+ records of the HF libraries and python warnings -> train.log, with counts and kernel-fallback names."""

    def __init__(self, path: Path) -> None:
        super().__init__(logging.WARNING)
        self.path = path
        self.counter: Counter[tuple[str, str, str]] = Counter()
        self.fallbacks: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.counter[(record.name, record.levelname, msg[:300])] += 1
        match = FALLBACK_RE.search(msg)
        if match:
            self.fallbacks.add(match.group(1))
        with self.path.open("a") as f:
            f.write(f"[{utc_now()}] {record.levelname} {record.name}: {msg}\n")

    def summary(self) -> list[dict[str, Any]]:
        return [{"n": n, "logger": k[0], "level": k[1], "message": k[2]} for k, n in self.counter.most_common(50)]


class TrainRun(RunDir):
    """A training-side run: logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv}.
    Captures the HF libraries' warnings into train.log and counts kernel fallbacks; everything else is RunDir."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        super().__init__(run_id, root, log_name="train.log")
        self.n_errors = 0
        self.monitor: Any = None   # apex_train.monitor.TrainingMonitor, closed with the run's final phase on exit
        self.capture = LogCapture(self.run_dir / "train.log")
        logging.captureWarnings(True)
        warnings.simplefilter("default")
        for name in CAPTURED_LOGGERS:
            logging.getLogger(name).addHandler(self.capture)

    def warn(self, msg: str) -> None:
        self.log("WARNING " + msg)

    def error(self, msg: str) -> None:
        self.n_errors += 1
        self.log("ERROR " + msg.splitlines()[0])
        super().error(msg)

    def start_metadata(self, **fields: Any) -> None:
        """The provenance block every training run writes first."""
        self.write_metadata(run_id=self.run_id, command=shlex.join(sys.argv), code=git_info(), hardware=hardware_info(),
                            versions=package_versions(), python=sys.version.split()[0], model_lock=model_lock(),
                            start_time=utc_now(), status="running", **fields)

    def finish(self, status: str, rows: list[tuple[str, Any, str, str]], **fields: Any) -> None:
        """resources.csv from `rows`, end time, status, the warning summary; errors.log 'none' when empty."""
        self.write_resources(rows)
        self.write_metadata(end_time=utc_now(), status=status, duration_sec=round(self.elapsed(), 1), n_errors=self.n_errors,
                            warnings_distinct=self.capture.summary(), kernel_fallbacks=sorted(self.capture.fallbacks), **fields)
        self.finalize()

    def attach_monitor(self, monitor: Any) -> None:
        self.monitor = monitor

    def __exit__(self, exc_type, exc, tb) -> bool:
        from apex_train.monitor import MonitorStop

        try:
            if isinstance(exc, MonitorStop):
                # the monitor stopped the run on a CRITICAL alert: status `stopped` (written by finish), the reason in errors.log
                self.error(f"stopped by the training monitor: {exc}")
                self.write_metadata(failure=f"MonitorStop: {exc}", end_time=utc_now())
                self.finalize()
                return super().__exit__(None, None, None)   # restores the SIGTERM handler; returns False, so MonitorStop propagates
            return super().__exit__(exc_type, exc, tb)
        finally:
            if self.monitor is not None:
                phase = ("finished" if exc is None else "stopped" if isinstance(exc, MonitorStop)
                         else "aborted" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed")
                self.monitor.close(phase, reason=None if exc is None else f"{type(exc).__name__}: {str(exc)[:300]}")
            for name in CAPTURED_LOGGERS:
                logging.getLogger(name).removeHandler(self.capture)


# ---------------------------------------------------------------------------------------- CPU shim for TRL's head
class TorchChunkedLogProb:
    """Pure-torch stand-in for `trl.kernels.ChunkedLogProbFunction` (a Triton kernel; absent on the CPU torch wheel,
    so TRL 1.15.0 sets `trl.trainer.utils._ChunkedLogProbFunction = None` and SFTTrainer / DPOTrainer crash in
    compute_loss, which always calls `model(..., fused_lm_head=True)`). Same signature and outputs as the kernel
    (per token: target log-prob, entropy, log(sum p^2), mean logit, is_top1 with the first-argmax tie rule), same
    transform order (logit_scale -> softcap -> 1/temperature), fp32 math, plain autograd; parity against the plain
    loss 1.9e-6, measured in the archived 35B training-stack study. CPU dry runs only."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None, logit_scale=1.0,  # noqa: ARG004 - kernel signature
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        import torch

        n, vocab = hidden.shape[0], weight.shape[0]
        if n == 0:
            z = hidden.new_zeros((0,), dtype=torch.float32)
            return (z, z.clone() if "entropy" in outputs else None, z.clone() if "log_sum_sq_probs" in outputs else None,
                    z.clone() if "mean_logits" in outputs else None,
                    torch.zeros((0,), dtype=torch.bool, device=hidden.device) if "is_top1" in outputs else None)
        logits = hidden.float() @ weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        logits = logits * logit_scale
        if final_logit_softcapping is not None:
            logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
        logits = logits / temperature
        log_z = torch.logsumexp(logits, dim=-1)
        target_logit = logits.gather(1, targets[:, None]).squeeze(1)
        logprobs = target_logit - log_z
        entropy = log_sum_sq = mean_logits = is_top1 = None
        if "entropy" in outputs or "log_sum_sq_probs" in outputs:
            p = torch.softmax(logits, dim=-1)
            if "entropy" in outputs:
                entropy = log_z - (p * logits).sum(-1)
            if "log_sum_sq_probs" in outputs:
                log_sum_sq = torch.log((p * p).sum(-1)).detach()
        if "mean_logits" in outputs:
            mean_logits = logits.mean(-1).detach()
        if "is_top1" in outputs:
            with torch.no_grad():
                running_max = logits.max(-1).values
                before = torch.arange(vocab, device=logits.device)[None, :] < targets[:, None]
                max_before_target = logits.masked_fill(~before, float("-inf")).max(-1).values
                is_top1 = (target_logit >= running_max) & (target_logit > max_before_target)
        return logprobs, entropy, log_sum_sq, mean_logits, is_top1


def fused_head_status(cpu_dry_run: bool) -> dict[str, Any]:
    """TRL fused-LM-head kernel availability. The shim is installed only for a CPU dry run when the kernel is missing;
    a real run without the kernel is refused (the first step would fail with `'NoneType' object has no attribute 'apply'`)."""
    import trl.trainer.utils as tu

    info = {"trl_kernel_available": tu._ChunkedLogProbFunction is not None, "shim_installed": False}
    if tu._ChunkedLogProbFunction is None:
        if not cpu_dry_run:
            raise RuntimeError("TRL's Triton fused LM head is unavailable in this environment; training would fail at the "
                               "first step. Use the GPU venv (Triton ships with the CUDA torch wheel) or run a CPU dry run.")
        tu._ChunkedLogProbFunction = TorchChunkedLogProb
        info["shim_installed"] = True
    return info


# ------------------------------------------------------------------------------------------- model and LoRA setup
def pick_device(cpu_dry_run: bool) -> str:
    """'cuda' when available, else 'cpu' only for a dry run (a real run must never silently train on CPU)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if cpu_dry_run:
        return "cpu"
    raise RuntimeError("CUDA is not available; refusing to start a real training run on CPU (use the smoke / dry-run mode)")


def checkpoint_identity(base_dir: Path) -> dict[str, Any]:
    """architectures / model_type / sha256 of config.json plus merge_provenance.json when the base is a merged dir."""
    base_dir = Path(base_dir)
    cfg_path = base_dir / "config.json"
    if not cfg_path.is_file():
        raise ConfigError(f"model directory {base_dir} has no config.json")
    cfg = json.loads(cfg_path.read_text())
    info: dict[str, Any] = {"path": str(base_dir.resolve()), "architectures": cfg.get("architectures"),
                            "model_type": cfg.get("model_type"), "config_sha256": sha256_file(cfg_path),
                            "transformers_version_in_config": cfg.get("transformers_version")}
    prov = base_dir / "merge_provenance.json"
    if prov.is_file():
        info["merge_provenance"] = json.loads(prov.read_text())
    return info


def resolve_loader(base_dir: Path, *, allow_text_only_twin: bool) -> tuple[str, str]:
    """(Auto class name, architecture) for the checkpoint. The served class Qwen3_5ForConditionalGeneration is the
    production class; the text-only Qwen3_5ForCausalLM is accepted only for dry runs on a shrunken twin
    (training on it yields the wrong adapter prefix and a merge that reloads with a random vision tower)."""
    archs = json.loads((Path(base_dir) / "config.json").read_text()).get("architectures") or []
    arch = archs[0] if archs else None
    if arch not in AUTO_CLASS_FOR_ARCH:
        raise ConfigError(f"unsupported architectures {archs}; expected one of {sorted(AUTO_CLASS_FOR_ARCH)}")
    if arch == TEXT_ONLY_ARCHITECTURE and not allow_text_only_twin:
        raise ConfigError(f"{base_dir} is the text-only {TEXT_ONLY_ARCHITECTURE}; production runs need a {SERVED_ARCHITECTURE} "
                          "checkpoint; only a dry run may use a text-only twin")
    return AUTO_CLASS_FOR_ARCH[arch], arch


def load_base_model(run: TrainRun, base_dir: Path, *, auto_class: str, dtype_name: str, device: str,
                    attn_implementation: str) -> tuple[Any, dict[str, Any]]:
    """from_pretrained with output_loading_info; the load must be clean (missing == [], unexpected keys only mtp.*).
    Returns (model, info) with the class, counts, layer pattern and load time; use_cache is turned off."""
    import torch
    import transformers

    cls = getattr(transformers, auto_class)
    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype_name), "output_loading_info": True, "attn_implementation": attn_implementation}
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    t0 = time.time()
    model, info = cls.from_pretrained(str(base_dir), **kwargs)
    missing = sorted(info.get("missing_keys", []))
    unexpected = sorted(info.get("unexpected_keys", []))
    non_mtp = [k for k in unexpected if not k.startswith(MTP_PREFIX)]
    if missing or non_mtp or info.get("mismatched_keys"):
        raise RuntimeError(f"base load not clean: missing {missing[:5]} ({len(missing)}), unexpected non-mtp {non_mtp[:5]} "
                           f"({len(non_mtp)}), mismatched {list(info.get('mismatched_keys', []))[:5]}")
    model.config.use_cache = False
    text_cfg = model.config.get_text_config()
    layer_types = Counter(getattr(text_cfg, "layer_types", None) or [])
    details = {"class": type(model).__name__, "auto_class": auto_class, "load_seconds": round(time.time() - t0, 1),
               "n_params": sum(p.numel() for p in model.parameters()), "n_unexpected_mtp": len(unexpected) - len(non_mtp),
               "weight_conversions": [str(c) for c in (getattr(model, "_weight_conversions", None) or [])],
               "attn_implementation": getattr(model.config, "_attn_implementation", None),
               "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
               "text_config": {"num_hidden_layers": text_cfg.num_hidden_layers, "layer_types": dict(layer_types),
                               "hidden_size": text_cfg.hidden_size, "intermediate_size": getattr(text_cfg, "intermediate_size", None),
                               "mamba_ssm_dtype": getattr(text_cfg, "mamba_ssm_dtype", None)}}
    run.log(f"loaded {details['class']} via {auto_class} in {details['load_seconds']} s: {details['n_params']:,d} params, "
            f"{details['dtype']} on {details['device']}, unexpected mtp.* {details['n_unexpected_mtp']}, "
            f"attn={details['attn_implementation']} layers={dict(layer_types)}")
    return model, details


def build_lora_config(lora: dict[str, Any]) -> Any:
    """peft.LoraConfig from a `lora` config block. Refuses forbidden targets (gate, conv1d, embeddings, lm_head,
    all-linear) and anything but plain LoRA on nn.Linear modules (no target_parameters, no quantization)."""
    from peft import LoraConfig

    targets = list(get_dotted(lora, "target_modules") or [])
    if not targets or not all(isinstance(t, str) for t in targets):
        raise ConfigError("lora.target_modules must be a non-empty list of module-name suffixes (PEFT has no default for qwen3_5)")
    bad = [t for t in targets if t in FORBIDDEN_TARGETS]
    if bad:
        raise ConfigError(f"lora.target_modules contains forbidden entries {bad} (configs/sft.yaml lora.never_target)")
    if get_dotted(lora, "target_parameters", None):
        raise ConfigError("lora.target_parameters is not part of the recipe: LoRA applies to nn.Linear modules only")
    dropout = float(get_dotted(lora, "dropout"))
    for key in ("r", "alpha"):
        if not isinstance(lora.get(key), int) or lora[key] <= 0:
            raise ConfigError(f"lora.{key} must be a positive integer")
    if str(lora.get("quantization", "none")).lower() not in ("none", "none (bf16 lora)"):
        raise ConfigError("lora.quantization must be 'none': the recipe is bf16 LoRA (the forward stays identical to serving)")
    return LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=dropout, bias=str(lora.get("bias", "none")),
                      task_type="CAUSAL_LM", init_lora_weights=lora.get("init_lora_weights", True), target_modules=targets)


def adapter_layout(peft_model: Any) -> dict[str, Any]:
    """Trainable counts and the invariants of the recipe on the wrapped model (vision tower untouched, LoRA only)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    lora_modules = [n for n, m in peft_model.named_modules() if isinstance(m, BaseTunerLayer)]
    first_key = next((n.replace(".default.", ".") for n, _ in peft_model.named_parameters() if "lora_A" in n), None)
    return {"trainable": trainable, "total": total, "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
            "n_lora_modules": len(lora_modules), "n_lora_modules_in_visual": sum(".visual." in n for n in lora_modules),
            "non_lora_trainable": [n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" not in n][:5],
            "visual_trainable": [n for n, p in peft_model.named_parameters() if p.requires_grad and ".visual." in n][:5],
            "adapter_param_dtypes": sorted({str(p.dtype) for p in peft_model.parameters() if p.requires_grad}),
            "first_adapter_key": first_key, "adapter_prefix_is_served_layout": bool(first_key and first_key.startswith(ADAPTER_KEY_PREFIX))}


def assert_adapter_layout(layout: dict[str, Any], trainer: Any, *, expected_trainable: int | None, served_class: bool) -> None:
    if layout["n_lora_modules"] == 0:
        raise RuntimeError("PEFT attached zero LoRA modules: target names do not match this model")
    if layout["n_lora_modules_in_visual"] or layout["visual_trainable"]:
        raise RuntimeError(f"vision tower is not frozen: {layout['n_lora_modules_in_visual']} adapters, trainable {layout['visual_trainable']}")
    if layout["non_lora_trainable"]:
        raise RuntimeError(f"non-LoRA parameters are trainable: {layout['non_lora_trainable']}")
    if served_class and not layout["adapter_prefix_is_served_layout"]:
        raise RuntimeError(f"adapter keys do not start with the served prefix: {layout['first_adapter_key']}")
    if expected_trainable is not None and layout["trainable"] != int(expected_trainable):
        raise RuntimeError(f"trainable parameters {layout['trainable']:,d} != expected {int(expected_trainable):,d} "
                           "(recipe or model changed; update the config deliberately)")
    if getattr(trainer, "aux_loss_enabled", False):
        raise RuntimeError("TRL enabled a router aux loss; this model has no router")


# ------------------------------------------------------------------------------------------- loss normalisation
LOSS_NORMALISATIONS = ("tokens", "rows")


def loss_normalisation_of(block: dict[str, Any], where: str) -> str:
    """The `loss_normalisation` of a trainer block (configs/sft.yaml optim, configs/stage2.yaml filtered_sft); tokens by
    default (the mixture is designed in loss tokens); anything else is a ConfigError."""
    mode = block.get("loss_normalisation", "tokens")
    if mode not in LOSS_NORMALISATIONS:
        raise ConfigError(f"{where}.loss_normalisation must be one of {LOSS_NORMALISATIONS}, got {mode!r}")
    return str(mode)


def apply_loss_normalisation(trainer: Any, mode: str) -> dict[str, Any]:
    """Choose how an SFT step combines its accumulated micro-batches (TRL 1.15 SFTTrainer, transformers 5.19 Trainer):

    tokens  sum of the token NLL over every trained token of the accumulation window / the window's trained tokens. Setting
            `model_accepts_loss_kwargs = True` makes Trainer.get_batch_samples count num_items_in_batch = the window's
            labels[..., 1:] != -100 (the causal-LM shift, exactly TRL's label_mask), SFTTrainer.compute_loss return
            sum(token NLL of the micro-batch) / num_items_in_batch, and Trainer.training_step skip its division by the
            accumulation steps (loss_is_scaled_for_ga). The forward never receives num_items_in_batch (TRL's compute_loss
            does not pass it), so nothing else changes. The accumulated gradient is the gradient of the window's token mean.
    rows    each micro-batch's own token mean, divided by the accumulation steps: every row weighs the same whatever its loss
            tokens. This is what Qwen3_5ForConditionalGeneration (accepts_loss_kwargs = False) gets by default, i.e. the
            behaviour before this switch, bit for bit.

    Returns the record for metadata.json. Refuses a trainer whose loss path would ignore the switch."""
    if mode not in LOSS_NORMALISATIONS:
        raise ConfigError(f"loss_normalisation must be one of {LOSS_NORMALISATIONS}, got {mode!r}")
    if getattr(trainer, "loss_is_scaled_for_ga", None) is not None or getattr(trainer, "compute_loss_func", None) is not None \
            or getattr(trainer, "label_smoother", None) is not None:
        raise ConfigError(f"loss_normalisation={mode}: {type(trainer).__name__} sets its own loss scaling (loss_is_scaled_for_ga / "
                          "compute_loss_func / label smoothing); the switch applies to TRL's plain SFT loss only")
    native = bool(trainer.model_accepts_loss_kwargs)
    trainer.model_accepts_loss_kwargs = mode == "tokens"
    return {"mode": mode, "model_accepts_loss_kwargs_native": native, "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
            "loss_shifts_labels": bool(getattr(trainer, "_loss_shifts_labels", False)),
            "definition": ("step loss = sum of token NLL over the accumulation window / its trained tokens (token-weighted)" if mode == "tokens"
                           else "step loss = mean over the window's micro-batches of each micro-batch's token-mean NLL (row-weighted)")}


# ---------------------------------------------------------------------------------- shared TrainingArguments block
def shared_trainer_kwargs(train: dict[str, Any], model_cfg: dict[str, Any], ckpt: dict[str, Any], *, out_dir: Path, device: str,
                          seed: int, data_seed: int, logging_steps: int, max_length: int) -> dict[str, Any]:
    """The TrainingArguments fields SFTConfig and DPOConfig share (REPORT.md 'Trainer settings'): micro-batch 1 on GPU,
    gradient accumulation for the effective batch, non-reentrant checkpointing, bf16 autocast on GPU (fp32 on the CPU
    twin), adamw_torch, save_strategy steps, nothing ever deleted (save_total_limit None)."""
    micro = int(require_type(train, "per_device_train_batch_size", int))
    accum = int(require_type(train, "gradient_accumulation_steps", int))
    effective = int(require_type(train, "effective_batch_size", int))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if effective != micro * accum * world:
        raise ConfigError(f"effective_batch_size {effective} != per_device_train_batch_size {micro} x gradient_accumulation_steps "
                          f"{accum} x world size {world}; keep the recorded effective batch the one actually used")
    if ckpt.get("save_total_limit") is not None:
        raise ConfigError("save_total_limit must be null: checkpoints are never deleted")
    if ckpt.get("save_strategy", "steps") not in ("steps", "epoch", "no"):
        raise ConfigError("save_strategy must be steps, epoch or no")
    warmup_ratio = float(get_dotted(train, "warmup_ratio", 0.0))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ConfigError("warmup_ratio must be in [0, 1): it is a ratio of the total optimizer steps")
    max_steps = get_dotted(train, "max_steps", None)
    return dict(
        output_dir=str(out_dir), max_steps=-1 if max_steps in (None, -1) else int(max_steps),
        num_train_epochs=float(get_dotted(train, "num_train_epochs", 1)),
        per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
        learning_rate=float(require_type(train, "learning_rate", (int, float))),
        lr_scheduler_type=str(get_dotted(train, "lr_scheduler_type", "cosine")),
        warmup_steps=warmup_ratio,   # transformers 5: a float in [0, 1) is a ratio of the total steps
        weight_decay=float(get_dotted(train, "weight_decay", 0.0)), max_grad_norm=float(get_dotted(train, "max_grad_norm", 1.0)),
        optim=str(get_dotted(train, "optim", "adamw_torch")), logging_steps=logging_steps, logging_first_step=True,
        save_strategy=str(ckpt.get("save_strategy", "steps")), save_steps=int(ckpt.get("save_steps", 0) or 0), save_total_limit=None,
        save_only_model=False, report_to="none", seed=seed, data_seed=data_seed,
        bf16=(device == "cuda" and str(get_dotted(model_cfg, "dtype")) == "bfloat16"), fp16=False, use_cpu=(device == "cpu"),
        dataloader_num_workers=0, dataloader_pin_memory=(device == "cuda"), disable_tqdm=True, remove_unused_columns=True,
        gradient_checkpointing=bool(get_dotted(model_cfg, "gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": bool(get_dotted(model_cfg, "gradient_checkpointing_use_reentrant", False))},
        max_length=max_length, padding_free=False,
    )


# ----------------------------------------------------------------------------------------- metrics and curves
BASE_TRAIN_COLUMNS = ("step", "epoch", "loss", "learning_rate", "grad_norm", "tokens_this_step", "tokens_cumulative", "tokens_per_s",
                      "step_time_s", "elapsed_s", "peak_gpu_mem_allocated_gb", "peak_gpu_mem_reserved_gb", "rss_gb")
EVAL_CURVE_COLUMNS = ("step", "epoch", "eval_loss", "eval_entropy", "eval_mean_token_accuracy", "eval_runtime_s", "elapsed_s")


class CsvAppender:
    """Append-only CSV with a fixed column list (curves/source/<run_id>_<kind>.csv); the header is written on creation.
    An existing file is refused unless `append` (a resumed run continues its own curve)."""

    def __init__(self, path: Path, columns: Sequence[str], *, append: bool = False) -> None:
        self.path, self.columns = Path(path), tuple(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not append:
            raise FileExistsError(f"curves file {self.path} already exists; never overwrite source data")
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.columns)
        self.n_rows = 0

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(["" if row.get(c) is None else row[c] for c in self.columns])
        self.n_rows += 1


class TrainMetrics:
    """TrainerCallback payload: every Trainer log (loss, lr, grad_norm, TRL's rewards/* ...) goes to metrics.jsonl with
    step timing, tokens/s (from TRL's cumulative num_tokens) and peak memory, mirrored into the curve CSVs; evaluations
    and checkpoints are recorded too. `extra_columns` adds method-specific curve columns (e.g. DPO rewards). With an
    enabled `monitor` (apex_train.monitor.TrainingMonitor) the bridge also drives it: the per-step and eval records gain the
    monitor fields in metrics.jsonl (the curve CSVs keep exactly their columns) and the monitor may ask the Trainer to save
    and stop after a CRITICAL non-finite or data alert."""

    def __init__(self, run: TrainRun, curves_dir: Path, *, device: str, extra_columns: Sequence[str] = (), resume: bool = False,
                 monitor: Any = None) -> None:
        self.run, self.device = run, device
        self.monitor = monitor if monitor is not None and getattr(monitor, "enabled", False) else None
        self.columns = (*BASE_TRAIN_COLUMNS, *extra_columns)
        self.train_csv = CsvAppender(curves_dir / f"{run.run_id}_train.csv", self.columns, append=resume)
        self.eval_csv = CsvAppender(curves_dir / f"{run.run_id}_eval.csv", EVAL_CURVE_COLUMNS, append=resume)
        self.t_train0 = self.t_step0 = time.perf_counter()
        self.step_time = 0.0
        self.prev_tokens = 0
        self.train_records: list[dict[str, Any]] = []
        self.eval_records: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.train_summary: dict[str, Any] = {}

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_train0

    def on_log(self, step: int, logs: dict[str, Any], control: Any = None) -> None:
        alloc, reserved = gpu_peak_gb()
        if "loss" in logs:
            tokens = int(logs.get("num_tokens", 0) or 0)
            tokens_this_step = max(tokens - self.prev_tokens, 0)
            self.prev_tokens = tokens
            rec = {"event": "train", "step": step, **{k: logs.get(k) for k in logs if k not in ("num_tokens",)},
                   "tokens_this_step": tokens_this_step, "tokens_cumulative": tokens,
                   "tokens_per_s": round(tokens_this_step / self.step_time, 1) if self.step_time and tokens_this_step else None,
                   "step_time_s": round(self.step_time, 3), "elapsed_s": round(self.elapsed(), 1),
                   "peak_gpu_mem_allocated_gb": alloc, "peak_gpu_mem_reserved_gb": reserved, "rss_gb": round(current_rss_gb(), 3)}
            if self.monitor is not None:
                rec.update(self.monitor.on_train_log(step, logs, rec, control))
            self.train_records.append(rec)
            self.train_csv.write(rec)
            self.run.metric(rec)
            shown = ", ".join(f"{k}={logs[k]:.5g}" for k in ("loss", "rewards/accuracies", "rewards/margins", "grad_norm", "learning_rate")
                              if isinstance(logs.get(k), (int, float)))
            self.run.log(f"step {step}: {shown}, tok/s={rec['tokens_per_s']}, step_time={rec['step_time_s']}s, rss_gb={rec['rss_gb']}"
                         + (self.monitor.log_suffix(rec) if self.monitor is not None else ""))
        elif "eval_loss" in logs:
            rec = {"event": "eval", "step": step, "epoch": logs.get("epoch"), "eval_loss": logs.get("eval_loss"),
                   "eval_entropy": logs.get("eval_entropy"), "eval_mean_token_accuracy": logs.get("eval_mean_token_accuracy"),
                   "eval_runtime_s": logs.get("eval_runtime"), "elapsed_s": round(self.elapsed(), 1)}
            if self.monitor is not None:
                rec.update(self.monitor.on_eval_log(step, logs, rec))
            self.eval_records.append(rec)
            self.eval_csv.write(rec)
            self.run.metric(rec)
            slices = " ".join(f"{k}={v:.4f}" for k, v in (rec.get("slices_tokens") or {}).items() if isinstance(v, float))
            tokens = f", token-weighted {rec['eval_loss_tokens']:.6g}" if isinstance(rec.get("eval_loss_tokens"), float) else ""
            self.run.log(f"eval @ step {step}: eval_loss={rec['eval_loss']} runtime={rec['eval_runtime_s']}s{tokens}"
                         + (f", slices (token-weighted) {slices}" if slices else ""))
        elif "train_runtime" in logs:
            self.train_summary = {k: logs[k] for k in ("train_runtime", "train_samples_per_second", "train_steps_per_second", "total_flos",
                                                      "train_loss", "epoch") if k in logs}
            self.run.metric({"event": "train_summary", "step": step, **self.train_summary})

    def on_save(self, step: int, output_dir: Path) -> None:
        ckpt = output_dir / f"checkpoint-{step}"
        rec = {"event": "checkpoint", "step": step, "path": str(ckpt), "bytes": dir_bytes(ckpt)}
        self.checkpoints.append(rec)
        self.run.metric(rec)
        self.run.log(f"checkpoint saved: {ckpt} ({rec['bytes'] / 1e6:.1f} MB)")
        if self.monitor is not None:
            self.monitor.on_save(step, ckpt)

    def callback(self, output_dir: Path) -> Any:
        """The transformers TrainerCallback bridging to this object (imported lazily to keep --help fast)."""
        from transformers import TrainerCallback

        metrics = self
        mon = self.monitor

        class _Bridge(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                metrics.t_train0 = time.perf_counter()
                if mon is not None:
                    mon.on_train_begin(args, state)

            def on_step_begin(self, args, state, control, **kw):
                if mon is not None:
                    mon.on_step_begin()
                metrics.t_step0 = time.perf_counter()

            def on_step_end(self, args, state, control, **kw):
                if metrics.device == "cuda":
                    import torch

                    torch.cuda.synchronize()
                metrics.step_time = time.perf_counter() - metrics.t_step0
                if mon is not None:
                    mon.on_step_end(state, control, metrics.step_time)

            def on_log(self, args, state, control, logs=None, **kw):
                metrics.on_log(state.global_step, dict(logs or {}), control)

            def on_evaluate(self, args, state, control, **kw):
                if mon is not None:
                    mon.on_evaluate(control)

            def on_save(self, args, state, control, **kw):
                metrics.on_save(state.global_step, output_dir)

            def on_train_end(self, args, state, control, **kw):
                if mon is not None:
                    mon.on_train_end()

        return _Bridge()

    def summary(self, keys: Sequence[str]) -> dict[str, Any]:
        """first / last / min / max / finiteness of each key over the train records (metadata.summary and smoke checks)."""
        out: dict[str, Any] = {}
        for k in keys:
            vals = [r[k] for r in self.train_records if isinstance(r.get(k), (int, float))]
            if vals:
                out[k] = {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals), "n": len(vals),
                          "all_finite": all(math.isfinite(v) for v in vals)}
        return out


# ------------------------------------------------------------------------------------------ labels == masks
def _row_hash(ids: Any) -> str:
    import numpy as np

    return hashlib.blake2b(np.asarray(ids, dtype=np.int64).tobytes(), digest_size=16).hexdigest()


def check_prepared_labels(prepared: Any, source: Any) -> dict[str, Any]:
    """TRL's prepared rows (input_ids + labels) must equal the source rows masked: labels[j] = ids[j] if mask else -100,
    same length (no truncation), same order, no row dropped. Returns the check record incl. the hash -> labels map
    consumed by FirstBatchLabelCheck and the hash -> row index map the training monitor identifies collated rows with
    (both are popped by the callers before the record goes to metadata.json)."""
    import numpy as np

    if len(prepared) != len(source):
        raise AssertionError(f"TRL kept {len(prepared)} of {len(source)} rows (fully-masked or truncated rows were dropped)")
    if "labels" not in prepared.column_names:
        raise AssertionError(f"prepared dataset has no labels column: {prepared.column_names}")
    by_hash: dict[str, Any] = {}
    index_by_hash: dict[str, int] = {}
    n_tokens = n_trained = 0
    p_np, s_np = prepared.with_format("numpy"), source.with_format("numpy")
    for i in range(len(source)):
        ids, mask = s_np[i]["input_ids"], s_np[i]["assistant_masks"]
        row = p_np[i]
        if not np.array_equal(row["input_ids"], ids):
            raise AssertionError(f"row {i}: TRL changed input_ids (len {len(row['input_ids'])} vs {len(ids)})")
        want = np.where(np.asarray(mask, dtype=bool), np.asarray(ids, dtype=np.int64), -100)
        if not np.array_equal(row["labels"], want):
            raise AssertionError(f"row {i}: labels differ from the assistant mask at token {int(np.argmax(np.asarray(row['labels']) != want))}")
        h = _row_hash(ids)
        by_hash[h] = want
        index_by_hash.setdefault(h, i)
        n_tokens += len(ids)
        n_trained += int(mask.sum())
    return {"rows_checked": len(source), "tokens": n_tokens, "trained_tokens": n_trained, "by_hash": by_hash, "index_by_hash": index_by_hash}


class FirstBatchLabelCheck:
    """Collator wrapper: on the first collated batch (the one the first forward sees: a training batch, or an eval
    batch when the run evaluates on start) assert that the padded labels equal the mask-derived labels of each row and
    that padding positions are -100; then delegate transparently. `by_hash` covers the training and the eval rows."""

    def __init__(self, inner: Callable[..., dict[str, Any]], by_hash: dict[str, Any], on_result: Callable[[dict[str, Any]], None]) -> None:
        self.inner, self.by_hash, self.on_result = inner, by_hash, on_result
        self.result: dict[str, Any] | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.inner(examples)
        if self.result is None:
            self.result = self.verify(batch)
            self.on_result(self.result)
        return batch

    def verify(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        input_ids, labels = batch["input_ids"].cpu().numpy(), batch["labels"].cpu().numpy()
        attn = batch["attention_mask"].cpu().numpy().astype(bool) if "attention_mask" in batch else np.ones_like(input_ids, dtype=bool)
        trained = 0
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][attn[i]]
            want = self.by_hash.get(_row_hash(ids))
            if want is None:
                raise AssertionError(f"first batch row {i}: input_ids match no training or eval row (collator changed the tokens)")
            if not np.array_equal(labels[i][attn[i]], want):
                raise AssertionError(f"first batch row {i}: collated labels != assistant-mask labels")
            if not np.all(labels[i][~attn[i]] == -100):
                raise AssertionError(f"first batch row {i}: padding positions are not -100 in labels")
            trained += int((want != -100).sum())
        return {"ok": True, "batch_rows": int(input_ids.shape[0]), "batch_tokens": int(attn.sum()), "batch_trained_tokens": trained,
                "padded_length": int(input_ids.shape[1])}


# -------------------------------------------------------------------------------------------------------- data
def rows_paths(value: Any) -> list[Path]:
    """A rows setting: one path or a list of paths (concatenated in order); None -> []."""
    if value is None:
        return []
    return [resolve_path(str(v)) for v in (value if isinstance(value, list) else [value])]


def load_split(paths: list[Path]) -> tuple[Any, list[int]]:
    """Rows files -> (one datasets.Dataset with apex_train.data_format's explicit features, row count of each file)."""
    from datasets import concatenate_datasets

    from apex_train.data_format import load_hf_dataset

    parts = [load_hf_dataset(p) for p in paths]
    return (parts[0] if len(parts) == 1 else concatenate_datasets(parts)), [len(p) for p in parts]


def describe_split(ds: Any, paths: list[Path], rows_per_file: list[int]) -> dict[str, Any]:
    """Data-version block: sha256, bytes and row count per file; row count, token and loss-token (assistant-mask) totals,
    modes, sources, task types and the thinking count of the whole split."""
    n_tokens, n_asst = ds["n_tokens"], ds["n_assistant_tokens"]
    return {"files": [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size, "n_rows": n}
                      for p, n in zip(paths, rows_per_file, strict=True)],
            "n_rows": len(ds), "n_tokens": int(sum(n_tokens)), "n_assistant_tokens": int(sum(n_asst)),
            "max_tokens": int(max(n_tokens, default=0)), "min_assistant_tokens": int(min(n_asst, default=0)),
            "thinking_rows": int(sum(1 for t in ds["thinking"] if t)), "rows_by_mode": dict(Counter(ds["mode"])),
            "rows_by_source": dict(Counter(str(s) for s in ds["source"])), "rows_by_task_type": dict(Counter(str(s) for s in ds["task_type"]))}


def check_split(desc: dict[str, Any], max_length: int, name: str) -> None:
    """Every row must fit max_length (so TRL never truncates) and train at least one token."""
    if desc["n_rows"] == 0:
        raise ConfigError(f"{name} rows: empty dataset")
    if desc["max_tokens"] > max_length:
        raise ConfigError(f"{name} rows: longest row has {desc['max_tokens']} tokens > max_length {max_length}; rebuild the rows with "
                          "`apex_train.data_format --max-length` (TRL truncation is never relied on)")
    if desc["min_assistant_tokens"] <= 0:
        raise ConfigError(f"{name} rows: a row trains zero tokens (TRL would drop it silently)")


def planned_optimizer_steps(n_rows: int, train: dict[str, Any]) -> dict[str, Any]:
    """Optimizer steps the Trainer will run over `n_rows` with the `train` block (transformers 5.19
    set_initial_training_values: ceil(batches / accumulation) per epoch, the last partial accumulation counts; a positive
    max_steps wins over epochs). `epochs` is the share of one pass the steps cover (1.0 for an epoch-based run)."""
    micro = int(require_type(train, "per_device_train_batch_size", int))
    accum = int(require_type(train, "gradient_accumulation_steps", int))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    batches = math.ceil(n_rows / (micro * world))
    per_epoch = max(math.ceil(batches / accum), 1)
    max_steps = get_dotted(train, "max_steps", None)
    if max_steps not in (None, -1):
        steps, source = int(max_steps), "max_steps"
    else:
        steps, source = math.ceil(float(get_dotted(train, "num_train_epochs", 1)) * per_epoch), "num_train_epochs"
    return {"rows": n_rows, "rows_per_step": micro * accum * world, "steps_per_epoch": per_epoch, "optimizer_steps": steps,
            "epochs": round(steps / per_epoch, 4), "source": source}


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def manifest_record(path: Path | None, files: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The frozen data manifest of a run (data/manifests/mixture.json for SFT, the stage-2 set manifest for stage 2) for
    metadata.json: path, sha256 and parsed content, plus, per rows file, whether its sha256 appears anywhere in the
    manifest (schema-agnostic: the manifest records the sha256 of the files it froze, so a miss means the rows on disk
    are not the frozen set). None when the path is unset or the file does not exist."""
    if path is None or not path.is_file():
        return None
    content = json.loads(path.read_text())
    listed = set(_strings(content))
    return {"path": str(path), "sha256": sha256_file(path), "content": content,
            "rows_sha256_listed": {f["path"]: f["sha256"] in listed for f in files}}


# --------------------------------------------------------------------------------------------- adapter / resume
def save_adapter(trainer: Any, adapter_dir: Path, run: TrainRun, *, base_repo: str | None, base_revision: str, base_path: Path,
                 lora_block: dict[str, Any], data_version: dict[str, Any] | None = None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """trainer.save_model + adapter_config.json base_model_name_or_path / revision set to the hub id and revision +
    a base_model.json sidecar (local path, config sha, classes, adapter sha, LoRA record, data version, `extra` fields
    such as the stage-2 lineage)."""
    trainer.save_model(str(adapter_dir))
    cfg_path = adapter_dir / "adapter_config.json"
    adapter_cfg = json.loads(cfg_path.read_text())
    repo = base_repo or str(base_path)
    rewritten = {k: {"from": adapter_cfg.get(k), "to": v} for k, v in (("base_model_name_or_path", repo), ("revision", base_revision))}
    adapter_cfg["base_model_name_or_path"], adapter_cfg["revision"] = repo, base_revision
    cfg_path.write_text(json.dumps(adapter_cfg, indent=2, sort_keys=True) + "\n")
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"adapter save incomplete: {weights} missing")
    sidecar = {"base_model_repo": base_repo, "base_model_revision": base_revision, "base_model_local_path": str(base_path),
               "base_model_config_sha256": sha256_file(base_path / "config.json"), "served_class": SERVED_ARCHITECTURE,
               "adapter_config_rewritten": rewritten, "adapter_model_sha256": sha256_file(weights), "adapter_bytes": dir_bytes(adapter_dir),
               "adapter_model_bytes": weights.stat().st_size, "quantization": "none (bf16 LoRA)", "lora": lora_block,
               "run_id": run.run_id, "created": utc_now(), "code": git_info(), "data_version": data_version, **(extra or {})}
    (adapter_dir / "base_model.json").write_text(json.dumps(sidecar, indent=2, default=str) + "\n")
    run.log(f"adapter saved: {adapter_dir} ({sidecar['adapter_model_bytes'] / 1e6:.1f} MB weights, {sidecar['adapter_bytes'] / 1e6:.1f} MB dir); "
            f"adapter_config base_model_name_or_path={repo!r} revision={base_revision!r}")
    return sidecar


def find_checkpoint(path: Path) -> Path:
    """A checkpoint directory itself, or the latest checkpoint-N under a trainer output directory."""
    if (path / "trainer_state.json").is_file():
        return path
    candidates = sorted((p for p in path.glob("checkpoint-*") if (p / "trainer_state.json").is_file()), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"--resume {path}: neither a checkpoint nor a directory with checkpoint-N/ inside")
    return candidates[-1]


def compare_steps(other_run_dir: Path, records: list[dict[str, Any]], tol: float) -> dict[str, Any]:
    """Compare loss and grad_norm of overlapping steps with another run's metrics.jsonl (resume bit-exactness)."""
    other: dict[int, dict[str, Any]] = {}
    for line in (other_run_dir / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "train" and rec.get("loss") is not None:
            other[int(rec["step"])] = rec
    mine = {int(r["step"]): r for r in records}
    common = sorted(set(other) & set(mine))
    diffs = {s: {k: abs(float(other[s][k]) - float(mine[s][k])) for k in ("loss", "grad_norm")
                 if other[s].get(k) is not None and mine[s].get(k) is not None} for s in common}
    worst = max((d for s in common for d in diffs[s].values()), default=None)
    return {"other_run": str(other_run_dir), "common_steps": common, "max_abs_diff": worst, "tol": tol,
            "per_step": {s: {"other_loss": other[s]["loss"], "loss": mine[s]["loss"], **diffs[s]} for s in common},
            "ok": bool(common) and worst is not None and worst <= tol}


def training_resource_rows(run: TrainRun, metrics: TrainMetrics, *, device: str, adapter_bytes: int | None = None,
                           trained_tokens: int | None = None) -> list[tuple[str, Any, str, str]]:
    """resources.csv rows for a training-side run on one GPU VM (or a CPU dry run), with the project's cost assumptions."""
    rates = compute_rates()
    duration = run.elapsed()
    gpus = 1 if device == "cuda" else 0
    gpu_hours = gpus * duration / 3600
    cost = gpu_hours * rates["gpu_vm_usd_per_hour"] + (0.0 if gpus else duration / 3600 * rates["cpu_vm_usd_per_hour"])
    tps = sorted(r["tokens_per_s"] for r in metrics.train_records if r.get("tokens_per_s"))
    alloc, reserved = gpu_peak_gb()
    return [
        ("duration_sec", round(duration, 1), "s", "wall clock of this run (load + train + save)"),
        ("train_runtime_sec", round(float(metrics.train_summary.get("train_runtime") or 0.0), 1), "s", "trainer.train() wall clock"),
        ("optimizer_steps", len(metrics.train_records), "count", "steps logged by this run"),
        ("tokens_processed", metrics.prev_tokens, "tokens", "attention-mask tokens seen by this run (TRL num_tokens)"),
        ("trained_tokens_dataset", trained_tokens if trained_tokens is not None else "n/a", "tokens", "assistant-mask tokens of the training rows"),
        ("tokens_per_s_mean", round(sum(tps) / len(tps), 1) if tps else "n/a", "tok/s", "mean over optimizer steps"),
        ("tokens_per_s_median", round(tps[len(tps) // 2], 1) if tps else "n/a", "tok/s", "median over optimizer steps"),
        ("peak_gpu_memory_allocated_gb", alloc if alloc is not None else "n/a", "GB", "torch.cuda.max_memory_allocated; n/a = no CUDA"),
        ("peak_gpu_memory_reserved_gb", reserved if reserved is not None else "n/a", "GB", "torch.cuda.max_memory_reserved; n/a = no CUDA"),
        ("peak_rss_gb", round(peak_rss_gb(), 2), "GB", "ru_maxrss of this process"),
        ("adapter_bytes", adapter_bytes if adapter_bytes is not None else "n/a", "B", "final adapter directory"),
        ("checkpoints_bytes", sum(c["bytes"] for c in metrics.checkpoints), "B", "all checkpoint-N directories of this run"),
        ("gpus", gpus, "count", "GPUs used by this process"),
        ("gpu_hours", round(gpu_hours, 4), "h", "duration x gpus"),
        ("cost_usd", round(cost, 3), "USD", f"gpu_hours x {rates['gpu_vm_usd_per_hour']} (CPU dry runs: cpu_hours x {rates['cpu_vm_usd_per_hour']}); configs/compute.yaml"),
    ]


def add_train_args(p: argparse.ArgumentParser, *, default_config: Path) -> None:
    """CLI flags shared by the training-side entry points."""
    p.add_argument("--config", type=Path, default=default_config, help=f"YAML config (default {default_config.relative_to(REPO_ROOT)})")
    p.add_argument("--run-id", default=None, help="logs/<run_id>/; default <prefix>_<UTC timestamp> (dryrun_ prefix for smoke runs)")
    p.add_argument("--override", action="append", default=[], metavar="KEY.PATH=VALUE", help="dotted config override, YAML-parsed; repeatable")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU dry runs)")
    p.add_argument("--note", default="", help="free text stored in metadata.json")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: write logs/, checkpoints and curves under this root")


Rows = Iterable[dict[str, Any]]
