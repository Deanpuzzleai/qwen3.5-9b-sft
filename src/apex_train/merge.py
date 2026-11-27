"""Merge a LoRA adapter into Qwen/Qwen3.5-9B, assemble the served checkpoint directory and gate its layout.

Implements REPORT.md section 2, "Merge / save / serve flow" and "Model class and adapter key prefix", of the archived 35B
training-stack study (README, repository layout notes; verified there on the previous base; the dense specifics are below). Run from
the repository root, normally through scripts/merge_adapter.sh:

    PYTHONPATH=src .venv/train/bin/python -m apex_train.merge merge --base <base dir> --adapter <adapter dir> --out <dir>
        [--snapshot configs/eval/model_snapshot] [--dtype bf16] [--merge-dtype fp32|bf16] [--device cpu|cuda]
        [--max-shard-size 5GB] [--run-id <id>] [--skip-gate]
    PYTHONPATH=src .venv/train/bin/python -m apex_train.merge gate --out <dir> --base <base dir> [--snapshot <dir>]
    PYTHONPATH=src .venv/train/bin/python -m apex_train.merge verify-adapter --adapter <dir> [--fix-adapter-config]

Subcommands
  merge           AutoModelForImageTextToText.from_pretrained(base) -> Qwen3_5ForConditionalGeneration, the served
                  class (the text-only class saves a text-only config that reloads with a random vision tower);
                  PeftModel.from_pretrained(adapter); merge_and_unload(safe_merge=True) in --merge-dtype (fp32 by
                  default: the bf16 base is upcast in process, ~38 GB RAM for the real model = 9.41 B params x 4 B;
                  bf16 halves that on a GPU at the cost of rounding W + dW in bf16); cast every tensor to the dtype
                  the base stores it in (bf16; Qwen3.5-9B keeps the 48 linear_attn.A_log / linear_attn.norm.weight
                  tensors in fp32: LoRA never touches them and the bf16 load rounded them, so they are copied back
                  from the base shards exactly); default save_pretrained(out, safe_serialization=True) (never
                  save_original_format=False: on a from_pretrained model it writes a `model.*` prefix); copy the side
                  files below from --snapshot,
                  falling back to --base; write out/merge_provenance.json; run the gate. Side files: tokenizer.json,
                  tokenizer_config.json, chat_template.jinja (required); vocab.json, merges.txt,
                  preprocessor_config.json, video_preprocessor_config.json, LICENSE (optional).
                  generation_config.json mirrors the snapshot: copied when the snapshot has one; when it has none
                  (the locked revision ships none) the file that save_pretrained derives from config.json is removed
                  again, so the merged directory resolves its generation defaults exactly like the base (vLLM falls
                  back to config.json for both, plus the lock's --override-generation-config). Nothing else is copied
                  (no mtp.* tensors, no fetch.log).
  gate            Layout gate for any checkpoint directory, reading safetensors headers only (seconds on the 19 GB
                  checkpoint): key set and shapes == --base's index minus mtp.* (excluded and counted on both sides),
                  dense 2-D mlp.{gate,up,down}_proj in every decoder layer, model.visual.*
                  present, every tensor in the base's dtype, index/shard consistency and total_size, architectures ==
                  ["Qwen3_5ForConditionalGeneration"], text_config.mamba_ssm_dtype == "float32" (read by vLLM for the
                  SSM cache dtype), config.json equal to the snapshot's except the fields in ALLOWED_CONFIG_DIFFS (a
                  diff is printed), generation_config.json present iff the snapshot has one (and then equal to it),
                  chat_template.jinja byte-identical, required side files present. Exit 1 on any violation.
  verify-adapter  adapter_config.json: LoRA, target_modules == the lock recipe (R2), r / alpha, no target_parameters,
                  base_model_name_or_path / revision == configs/eval/model.yaml (--fix-adapter-config rewrites those
                  two fields after backing the file up); adapter keys under
                  base_model.model.model.language_model.layers.* and matching the declared targets.

Inputs : --base checkpoint dir (read-only; the real one is /mnt/apex-shared/models/Qwen3.5-9B@c2022362...),
         --adapter PEFT directory, --snapshot (default configs/eval/model_snapshot), configs/eval/model.yaml (lock).
Outputs: --out (merged bf16 checkpoint; must not exist or be empty; never under configs/eval or the weights mount),
         logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv} for every subcommand.

Run bookkeeping: apex_train.common.TrainRun (train.log variant of the run-directory layout; merging is a training-side step).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from apex_eval.common import REPO_ROOT, compute_rates, git_info, model_lock, new_run_id, sha256_file, timestamp, utc_now
from apex_train.common import TrainRun, current_rss_gb, package_versions, peak_rss_gb

MODEL_LOCK = REPO_ROOT / "configs" / "eval" / "model.yaml"
DEFAULT_SNAPSHOT = REPO_ROOT / "configs" / "eval" / "model_snapshot"
WEIGHTS_MOUNT = Path("/mnt/apex-shared/models")
SERVED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
LM_LAYER_PREFIX = "model.language_model.layers."
VISUAL_PREFIX = "model.visual."
MTP_PREFIX = "mtp."
ADAPTER_KEY_PREFIX = "base_model.model." + LM_LAYER_PREFIX
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
PROVENANCE_FILE = "merge_provenance.json"
# LoRA recipe R2 in its dense form (configs/sft.yaml lora:): attention q/k/v/o (8 full-attention layers), GatedDeltaNet
# in_proj_qkv/z/b/a + out_proj (24 linear-attention layers), dense MLP gate/up/down (all 32 layers); nothing in the vision tower.
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                       "out_proj", "gate_proj", "up_proj", "down_proj")
REQUIRED_SIDE_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MIRRORED_SIDE_FILES = ("generation_config.json",)   # in out iff in the snapshot (module docstring, "merge")
OPTIONAL_SIDE_FILES = ("vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE")
SAFETENSORS_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "BOOL": 1, "I16": 2, "I32": 4,
                     "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
TORCH_DTYPE_NAMES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}   # safetensors dtype -> torch attribute

ANY = object()      # allowlist value: any value is accepted for that field
_ABSENT = object()  # the field exists on one side only
# config.json fields that may differ from the snapshot (written by transformers 4.57.0.dev0) after a transformers 5.19 save.
# Measured on 2026-10-11 for Qwen3.5-9B: AutoConfig.from_pretrained(snapshot).save_pretrained() and the config of a bf16
# model built from it write the same ten differences: transformers_version, dtype at the top level and inside
# vision_config, the nested vision model_type, the vision rotary defaults, and four text_config defaults made explicit
# (bos/pad token ids, partial_rotary_factor, tie_word_embeddings). Everything else (incl. text_config.dtype and
# text_config.mamba_ssm_dtype) must match exactly.
ALLOWED_CONFIG_DIFFS: dict[str, Any] = {
    "transformers_version": ANY,
    "dtype": "bfloat16",                                  # top-level dtype written by transformers 5
    "vision_config.dtype": "bfloat16",                    # nested dtype written by transformers 5
    "vision_config.model_type": "qwen3_5_vision",         # transformers 5 names the nested vision config
    "vision_config.rope_parameters.rope_theta": 10000.0,  # transformers 5 writes the vision rotary default explicitly
    "vision_config.rope_parameters.rope_type": "axial",
    "text_config.bos_token_id": None,                     # text_config defaults made explicit by transformers 5
    "text_config.pad_token_id": None,
    "text_config.partial_rotary_factor": 0.25,
    "text_config.tie_word_embeddings": False,
}
ALLOWED_GENERATION_CONFIG_DIFFS: dict[str, Any] = {"transformers_version": ANY}
# adapter checks whose failure aborts a merge; the other adapter checks are recipe/naming conformance (warnings in
# merge, failures in verify-adapter)
STRUCTURAL_ADAPTER_CHECKS = frozenset({"adapter_files_present", "peft_type_lora", "adapter_keys_under_language_model_layers",
                                       "adapter_keys_match_declared_targets", "no_vision_adapters", "lora_pairs"})


# ------------------------------------------------------------------------------------------------ run bookkeeping
class Stage:
    """Context manager that logs one merge stage with its duration and RSS to train.log and metrics.jsonl."""

    def __init__(self, run: TrainRun, name: str) -> None:
        self.run, self.name, self.t0 = run, name, 0.0

    def __enter__(self) -> Stage:
        self.t0 = time.time()
        self.run.log(f"stage {self.name}: start (rss {current_rss_gb():.2f} GB)")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        seconds = time.time() - self.t0
        self.run.metric({"stage": self.name, "seconds": round(seconds, 1), "rss_gb": round(current_rss_gb(), 2), "ok": exc is None})
        self.run.log(f"stage {self.name}: {'done' if exc is None else 'FAILED'} in {seconds:.1f} s (rss {current_rss_gb():.2f} GB)")


def merge_resource_rows(run: TrainRun, *, gpus: int, peak_gpu_mem_gb: float | None) -> list[tuple[str, Any, str, str]]:
    """resources.csv rows for a merge / gate / verify step (the project's cost assumptions from configs/compute.yaml)."""
    rates = compute_rates()
    duration = run.elapsed()
    gpu_hours = gpus * duration / 3600
    cost = gpu_hours * rates["gpu_vm_usd_per_hour"] + (0.0 if gpus else duration / 3600 * rates["cpu_vm_usd_per_hour"])
    return [
        ("duration_sec", round(duration, 1), "s", "wall clock of this run"),
        ("peak_rss_gb", round(peak_rss_gb(), 2), "GB", "ru_maxrss of this process"),
        ("peak_gpu_memory_gb", "n/a" if peak_gpu_mem_gb is None else round(peak_gpu_mem_gb, 2), "GB", "torch.cuda.max_memory_allocated; n/a on CPU"),
        ("gpus", gpus, "count", "GPUs used by this process (--device cuda -> 1)"),
        ("gpu_hours", round(gpu_hours, 4), "h", "duration x gpus"),
        ("cost_usd", round(cost, 2), "USD", f"gpu_hours x {rates['gpu_vm_usd_per_hour']} (CPU steps: cpu_hours x {rates['cpu_vm_usd_per_hour']})"),
    ]


# ------------------------------------------------------------------------------------------- checkpoint layout
@dataclass
class Check:
    """One gate / adapter check: name, verdict, human-readable detail, structured data for metrics.jsonl."""

    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointLayout:
    """Tensor names, dtypes and shapes of a safetensors checkpoint directory, read from the file headers only."""

    root: Path
    tensors: dict[str, tuple[str, tuple[int, ...]]]
    shard_files: list[str]
    index_total_size: int | None   # metadata.total_size of the index, None for a single-file checkpoint
    problems: list[str]            # index / shard inconsistencies

    @property
    def header_bytes(self) -> int:
        return sum(SAFETENSORS_BYTES.get(d, 0) * _numel(s) for d, s in self.tensors.values())

    def fingerprint(self) -> str:
        """sha256 over the sorted (name, dtype, shape) triples: identifies a layout independent of sharding."""
        blob = json.dumps(sorted((k, d, list(s)) for k, (d, s) in self.tensors.items())).encode()
        return hashlib.sha256(blob).hexdigest()


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _examples(items: list[str], n: int = 5) -> str:
    return f"{len(items)} e.g. {items[:n]}" if items else "0"


def read_safetensors_header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return {tensor name: (dtype, shape)} from the JSON header of a .safetensors file (8-byte length prefix + header;
    no tensor data is read). Lifted from safetensors_keys() in probe 03_tiny_e2e_training.py of the archived 35B training-stack
    study."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}


def read_checkpoint_layout(root: Path) -> CheckpointLayout:
    """Read model.safetensors.index.json plus every shard header, or the single model.safetensors header."""
    index_path = root / INDEX_FILE
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        headers: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
        problems: list[str] = []
        for shard in shards:
            if (root / shard).is_file():
                headers[shard] = read_safetensors_header(root / shard)
            else:
                problems.append(f"shard {shard} named in the index is missing")
        tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
        duplicates: list[str] = []
        for header in headers.values():
            for k, v in header.items():
                if k in tensors:
                    duplicates.append(k)
                tensors[k] = v
        misplaced = [k for k, shard in weight_map.items() if shard in headers and k not in headers[shard]]
        unindexed = sorted(set(tensors) - set(weight_map))
        for label, items in (("tensors stored in more than one shard", duplicates),
                             ("tensors not in the shard the index names", misplaced),
                             ("tensors in shards but absent from the index", unindexed)):
            if items:
                problems.append(f"{label}: {_examples(items)}")
        total = index.get("metadata", {}).get("total_size")
        return CheckpointLayout(root, tensors, shards, int(total) if total is not None else None, problems)
    single = root / SINGLE_FILE
    if single.is_file():
        return CheckpointLayout(root, read_safetensors_header(single), [SINGLE_FILE], None, [])
    raise FileNotFoundError(f"{root}: neither {INDEX_FILE} nor {SINGLE_FILE} found")


# ----------------------------------------------------------------------------------------------- config diffs
def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to {'a.b.c': leaf}; lists and empty dicts are leaves."""
    if isinstance(obj, dict) and obj:
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(flatten_json(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: obj}


def parse_allow_arg(items: list[str]) -> dict[str, Any]:
    """--allow-config-diff KEY or KEY=JSON_VALUE -> allowlist entries (KEY alone accepts any value)."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        out[key] = json.loads(value) if sep else ANY
    return out


def _print_diff(label: str, a_path: Path, b_path: Path, a_text: str, b_text: str, log: Callable[[str], None],
                max_lines: int = 300) -> None:
    lines = list(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(), fromfile=str(a_path),
                                      tofile=str(b_path), lineterm="", n=1))
    shown = lines[:max_lines] + ([f"... {len(lines) - max_lines} more diff lines"] if len(lines) > max_lines else [])
    log(f"{label}: not byte-identical to the snapshot; diff (snapshot -> out):\n" + "\n".join(shown))


def compare_json_file(name: str, snapshot_dir: Path, out_dir: Path, allowed: dict[str, Any],
                      log: Callable[[str], None]) -> Check:
    """Byte-compare out_dir/name with snapshot_dir/name; if they differ, every differing flattened field must be in
    `allowed` (ANY or the exact expected out value). The diff is printed either way."""
    a_path, b_path = snapshot_dir / name, out_dir / name
    check_name = f"{name}_matches_snapshot"
    if not a_path.is_file():
        return Check(check_name, False, f"snapshot has no {name}")
    if not b_path.is_file():
        return Check(check_name, False, f"out dir has no {name}")
    a_bytes, b_bytes = a_path.read_bytes(), b_path.read_bytes()
    if a_bytes == b_bytes:
        return Check(check_name, True, "byte-identical", {"byte_identical": True})
    a, b = flatten_json(json.loads(a_bytes)), flatten_json(json.loads(b_bytes))
    accepted, violations = [], []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, _ABSENT), b.get(key, _ABSENT)
        if va == vb:
            continue
        shown = f"{key}: {'<absent>' if va is _ABSENT else repr(va)} -> {'<absent>' if vb is _ABSENT else repr(vb)}"
        if key in allowed and (allowed[key] is ANY or allowed[key] == vb):
            accepted.append(shown)
        else:
            violations.append(shown)
    _print_diff(name, a_path, b_path, json.dumps(json.loads(a_bytes), indent=2, sort_keys=True),
                json.dumps(json.loads(b_bytes), indent=2, sort_keys=True), log)
    detail = f"allowed differences: {accepted or 'none'}; violations: {violations or 'none'}"
    return Check(check_name, not violations, detail, {"byte_identical": False, "allowed": accepted, "violations": violations})


def compare_text_file(name: str, snapshot_dir: Path, out_dir: Path, log: Callable[[str], None]) -> Check:
    a_path, b_path = snapshot_dir / name, out_dir / name
    check_name = f"{name}_matches_snapshot"
    if not a_path.is_file():
        return Check(check_name, False, f"snapshot has no {name}")
    if not b_path.is_file():
        return Check(check_name, False, f"out dir has no {name}")
    if a_path.read_bytes() == b_path.read_bytes():
        return Check(check_name, True, "byte-identical", {"byte_identical": True})
    _print_diff(name, a_path, b_path, a_path.read_text(errors="replace"), b_path.read_text(errors="replace"), log)
    return Check(check_name, False, "differs from the snapshot (diff printed above)", {"byte_identical": False})


# ------------------------------------------------------------------------------------------------------ gate
def dense_mlp_check(tensors: dict[str, tuple[str, tuple[int, ...]]]) -> Check:
    """Every decoder layer must hold the 2-D dense mlp.{gate,up,down}_proj.weight (anything else means the wrong class)."""
    layers = sorted({int(m.group(1)) for k in tensors if (m := re.match(re.escape(LM_LAYER_PREFIX) + r"(\d+)\.", k))})
    bad: list[str] = []
    for i in layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            key = f"{LM_LAYER_PREFIX}{i}.mlp.{name}.weight"
            if key not in tensors:
                bad.append(f"missing {key}")
            elif len(tensors[key][1]) != 2:
                bad.append(f"{key} has shape {tensors[key][1]}, expected 2-D")
    ok = bool(layers) and not bad
    detail = (f"{len(layers)} decoder layers; problems: {_examples(bad)}"
              + ("" if layers else "; no model.language_model.layers.* keys at all"))
    shapes = {n: list(tensors[f"{LM_LAYER_PREFIX}0.mlp.{n}.weight"][1]) for n in ("gate_proj", "up_proj", "down_proj")
              if f"{LM_LAYER_PREFIX}0.mlp.{n}.weight" in tensors}
    return Check("mlp_dense_2d", ok, detail, {"n_layers": len(layers), "layer0_mlp_shapes": shapes})


def gate_checks(out_dir: Path, base_dir: Path, snapshot_dir: Path, allowed_config_diffs: dict[str, Any],
                log: Callable[[str], None]) -> list[Check]:
    """All gate checks of out_dir against base_dir's layout and snapshot_dir's config files. Reads safetensors headers
    and small JSON/text files only, so it runs in seconds on the 19 GB checkpoint."""
    out, base = read_checkpoint_layout(out_dir), read_checkpoint_layout(base_dir)
    checks: list[Check] = []
    for label, layout in (("out", out), ("base", base)):
        checks.append(Check(f"{label}_index_consistent", not layout.problems, "; ".join(layout.problems) or "ok",
                            {"n_tensors": len(layout.tensors), "n_shards": len(layout.shard_files),
                             "header_bytes": layout.header_bytes, "fingerprint": layout.fingerprint()}))
    if out.index_total_size is not None:
        checks.append(Check("out_index_total_size_matches_headers", out.index_total_size == out.header_bytes,
                            f"index total_size {out.index_total_size:,d} B vs headers {out.header_bytes:,d} B "
                            f"({out.header_bytes / 1e9:.2f} GB)", {"index_total_size": out.index_total_size,
                                                                     "header_bytes": out.header_bytes}))
    out_keys = {k for k in out.tensors if not k.startswith(MTP_PREFIX)}
    base_keys = {k for k in base.tensors if not k.startswith(MTP_PREFIX)}
    n_mtp = {"out": len(out.tensors) - len(out_keys), "base": len(base.tensors) - len(base_keys)}
    log(f"mtp.* tensors excluded from the comparison: out {n_mtp['out']}, base {n_mtp['base']}")
    only_out, only_base = sorted(out_keys - base_keys), sorted(base_keys - out_keys)
    checks.append(Check("key_set_equal_minus_mtp", not only_out and not only_base,
                        f"{len(out_keys)} vs {len(base_keys)} keys; only in out: {_examples(only_out)}; "
                        f"only in base: {_examples(only_base)}",
                        {"n_out": len(out_keys), "n_base": len(base_keys), "n_mtp_excluded": n_mtp,
                         "only_in_out": only_out[:50], "only_in_base": only_base[:50]}))
    mismatched = [f"{k}: out {list(out.tensors[k][1])} vs base {list(base.tensors[k][1])}"
                  for k in sorted(out_keys & base_keys) if out.tensors[k][1] != base.tensors[k][1]]
    checks.append(Check("shapes_equal", not mismatched, f"shape mismatches: {_examples(mismatched)}",
                        {"n_compared": len(out_keys & base_keys), "mismatches": mismatched[:50]}))
    dtype_mismatch = [f"{k}: out {out.tensors[k][0]} vs base {base.tensors[k][0]}"
                      for k in sorted(out_keys & base_keys) if out.tensors[k][0] != base.tensors[k][0]]
    dtype_hist = dict(sorted(Counter(d for k, (d, _) in out.tensors.items() if k in out_keys).items()))
    checks.append(Check("dtypes_match_base", not dtype_mismatch, f"dtype mismatches vs base: {_examples(dtype_mismatch)}; out dtypes {dtype_hist}",
                        {"n_dtype_mismatch": len(dtype_mismatch), "mismatches": dtype_mismatch[:50], "dtypes": dtype_hist}))
    checks.append(dense_mlp_check(out.tensors))
    n_visual = sum(k.startswith(VISUAL_PREFIX) for k in out_keys)
    checks.append(Check("vision_tower_present", n_visual > 0, f"{n_visual} {VISUAL_PREFIX}* tensors", {"n_visual": n_visual}))
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    checks.append(Check("architectures_served_class", cfg.get("architectures") == [SERVED_ARCHITECTURE],
                        f"architectures={cfg.get('architectures')!r} model_type={cfg.get('model_type')!r}"
                        + ("" if cfg_path.is_file() else " (config.json missing)")))
    ssm = cfg.get("text_config", {}).get("mamba_ssm_dtype") if isinstance(cfg.get("text_config"), dict) else None
    checks.append(Check("text_config_mamba_ssm_dtype_float32", ssm == "float32", f"text_config.mamba_ssm_dtype={ssm!r}"))
    checks.append(compare_json_file("config.json", snapshot_dir, out_dir, allowed_config_diffs, log))
    gen = "generation_config.json"
    if (snapshot_dir / gen).is_file():
        checks.append(compare_json_file(gen, snapshot_dir, out_dir, ALLOWED_GENERATION_CONFIG_DIFFS, log))
    else:   # the locked revision ships none: the merged directory must not add one (it would come from config.json anyway)
        present = (out_dir / gen).is_file()
        checks.append(Check(f"{gen}_matches_snapshot", not present,
                            "absent in the snapshot and in out" if not present else
                            "the snapshot has none but out has one; remove it so the directory resolves generation defaults like the base",
                            {"snapshot_has_file": False, "out_has_file": present}))
    checks.append(compare_text_file("chat_template.jinja", snapshot_dir, out_dir, log))
    missing = [f for f in ("config.json",) + REQUIRED_SIDE_FILES if not (out_dir / f).is_file()]
    optional_missing = [f for f in OPTIONAL_SIDE_FILES if not (out_dir / f).is_file()]
    checks.append(Check("required_files_present", not missing, f"missing: {missing or 'none'}; optional missing: "
                        f"{optional_missing or 'none'}", {"missing": missing, "optional_missing": optional_missing}))
    return checks


def report_checks(checks: list[Check], run: TrainRun, kind: str) -> bool:
    """Log every check, write it to metrics.jsonl, record failures in errors.log; return True iff all passed."""
    for c in checks:
        run.log(f"[{'PASS' if c.ok else 'FAIL'}] {kind}.{c.name}: {c.detail}")
        run.metric({"kind": kind, "check": c.name, "ok": c.ok, **c.data})
        if not c.ok:
            run.error(f"{kind} check failed: {c.name}: {c.detail}")
    n_fail = sum(not c.ok for c in checks)
    run.log(f"{kind}: {len(checks) - n_fail}/{len(checks)} checks passed")
    return n_fail == 0


# --------------------------------------------------------------------------------------------- adapter checks
@dataclass
class AdapterExpectation:
    """What verify-adapter requires of adapter_config.json (defaults = lock recipe R2 and configs/eval/model.yaml)."""

    targets: tuple[str, ...]
    r: int
    alpha: int
    base_model: str
    revision: str


def adapter_checks(adapter_dir: Path, expect: AdapterExpectation, log: Callable[[str], None]) -> list[Check]:
    """Checks on adapter_config.json and on the adapter safetensors header (no tensor data is read)."""
    cfg_path, weights_path = adapter_dir / ADAPTER_CONFIG, adapter_dir / ADAPTER_WEIGHTS
    missing = [p.name for p in (cfg_path, weights_path) if not p.is_file()]
    if missing:
        return [Check("adapter_files_present", False, f"missing in {adapter_dir}: {missing}")]
    cfg = json.loads(cfg_path.read_text())
    header = read_safetensors_header(weights_path)
    keys = sorted(header)
    declared = tuple(cfg.get("target_modules") or [])
    declared_params = tuple(cfg.get("target_parameters") or [])
    checks = [Check("adapter_files_present", True, f"{len(keys)} adapter tensors"),
              Check("peft_type_lora", cfg.get("peft_type") == "LORA", f"peft_type={cfg.get('peft_type')!r}"),
              Check("target_modules_match_lock_recipe", set(declared) == set(expect.targets),
                    f"extra: {sorted(set(declared) - set(expect.targets))}; missing: {sorted(set(expect.targets) - set(declared))}",
                    {"declared": sorted(declared)}),
              Check("rank", cfg.get("r") == expect.r, f"r={cfg.get('r')!r} expected {expect.r}"),
              Check("alpha", cfg.get("lora_alpha") == expect.alpha, f"lora_alpha={cfg.get('lora_alpha')!r} expected {expect.alpha}"),
              Check("bias_none", cfg.get("bias") == "none", f"bias={cfg.get('bias')!r}"),
              Check("no_target_parameters", not declared_params,
                    f"target_parameters={list(declared_params)} (PEFT ParamWrapper adapters are not part of the dense recipe)"),
              Check("base_model_name_or_path", cfg.get("base_model_name_or_path") == expect.base_model,
                    f"{cfg.get('base_model_name_or_path')!r} expected {expect.base_model!r}"),
              Check("revision", cfg.get("revision") == expect.revision, f"{cfg.get('revision')!r} expected {expect.revision!r}")]
    bad_prefix = [k for k in keys if not k.startswith(ADAPTER_KEY_PREFIX)]
    checks.append(Check("adapter_keys_under_language_model_layers", not bad_prefix,
                        f"keys outside {ADAPTER_KEY_PREFIX}*: {_examples(bad_prefix)}"
                        + (" (keys under base_model.model.model.layers.* mean the adapter was trained on the text-only "
                           f"Qwen3_5ForCausalLM; retrain/merge on {SERVED_ARCHITECTURE})" if bad_prefix else ""),
                        {"n_keys": len(keys), "example": keys[:2]}))
    unknown = []
    for k in keys:
        module = re.sub(r"\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B)\.(weight|default)$", "", k)
        if not any(module.endswith("." + t) for t in declared):
            unknown.append(k)
    checks.append(Check("adapter_keys_match_declared_targets", not unknown,
                        f"keys whose module is not a declared target: {_examples(unknown)}", {"n_unknown": len(unknown)}))
    visual = [k for k in keys if ".visual." in k]
    checks.append(Check("no_vision_adapters", not visual, f"adapter tensors in the vision tower: {_examples(visual)}"))
    n_a, n_b = sum(".lora_A." in k for k in keys), sum(".lora_B." in k for k in keys)
    n_params = sum(_numel(s) for _, s in header.values())
    dtypes = sorted({d for d, _ in header.values()})
    checks.append(Check("lora_pairs", n_a == n_b and n_a > 0, f"lora_A {n_a}, lora_B {n_b}, {n_params:,d} adapter parameters, dtypes {dtypes}",
                        {"n_lora_A": n_a, "n_lora_B": n_b, "n_params": n_params, "dtypes": dtypes}))
    log(f"adapter {adapter_dir}: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} dropout={cfg.get('lora_dropout')} "
        f"targets={sorted(declared)} target_parameters={list(declared_params)} tensors={len(keys)} params={n_params:,d}")
    return checks


def fix_adapter_config(adapter_dir: Path, base_model: str, revision: str, log: Callable[[str], None]) -> Path:
    """Back up adapter_config.json and rewrite base_model_name_or_path / revision; returns the backup path."""
    cfg_path = adapter_dir / ADAPTER_CONFIG
    backup = adapter_dir / f"{ADAPTER_CONFIG}.orig-{timestamp()}"
    shutil.copy2(cfg_path, backup)
    cfg = json.loads(cfg_path.read_text())
    before = {k: cfg.get(k) for k in ("base_model_name_or_path", "revision")}
    cfg["base_model_name_or_path"], cfg["revision"] = base_model, revision
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    log(f"adapter_config.json fixed: {before} -> base_model_name_or_path={base_model!r} revision={revision!r}; backup {backup}")
    return backup


# ------------------------------------------------------------------------------------------------------ merge
def guard_out_dir(out: Path, base: Path) -> None:
    """Refuse output locations that would overwrite data or write under the locked configs / the weights mount."""
    for forbidden in (WEIGHTS_MOUNT, REPO_ROOT / "configs" / "eval"):
        if out == forbidden or forbidden in out.parents:
            raise PermissionError(f"--out {out} is under {forbidden}; never write there")
    if out == base or out in base.parents:
        raise ValueError(f"--out {out} must not be the base directory or one of its parents")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"--out {out} exists and is not empty; never overwrite a checkpoint")


def cast_to_base_dtypes(model: Any, base: Path, base_layout: CheckpointLayout, adapted_keys: set[str]) -> dict[str, int]:
    """Give every parameter the dtype the base stores it in. Parameters the adapter did not touch and whose base dtype
    is not the bf16 load dtype (Qwen3.5-9B: the fp32 linear_attn.A_log and linear_attn.norm.weight) are copied back
    from the base shards, because from_pretrained(dtype=bf16) rounded them; everything else is cast. Returns counts."""
    import torch
    from safetensors import safe_open

    params = dict(model.named_parameters())
    index = json.loads((base / INDEX_FILE).read_text())["weight_map"] if (base / INDEX_FILE).is_file() else {}
    counts: Counter[str] = Counter()
    restore: dict[str, list[str]] = {}
    for key, p in params.items():
        base_dtype = base_layout.tensors[key][0] if key in base_layout.tensors else "BF16"
        if base_dtype != "BF16" and key in base_layout.tensors and key not in adapted_keys:
            restore.setdefault(index.get(key, SINGLE_FILE), []).append(key)
            continue
        want = getattr(torch, TORCH_DTYPE_NAMES[base_dtype])
        if p.dtype != want:
            p.data = p.data.to(want)
        counts[f"cast_{base_dtype}"] += 1
    for shard, keys in restore.items():
        with safe_open(str(base / shard), framework="pt", device="cpu") as f:
            for key in keys:
                params[key].data = f.get_tensor(key).to(params[key].device)
                counts[f"restored_{base_layout.tensors[key][0]}"] += 1
    return dict(sorted(counts.items()))


def assemble_side_files(out: Path, snapshot: Path, base: Path, log: Callable[[str], None]) -> dict[str, dict[str, str | None]]:
    """Copy tokenizer / template / preprocessor files and LICENSE into out: from the snapshot when it has them, else
    from the base. Required files missing from both abort; optional ones are logged and skipped. generation_config.json
    mirrors the snapshot only: copied when the snapshot has one, otherwise the file save_pretrained derived from
    config.json is removed so the directory resolves its generation defaults exactly like the base."""
    copied: dict[str, dict[str, str | None]] = {}
    for name in REQUIRED_SIDE_FILES + OPTIONAL_SIDE_FILES + MIRRORED_SIDE_FILES:
        sources = (snapshot,) if name in MIRRORED_SIDE_FILES else (snapshot, base)
        src = next((d / name for d in sources if (d / name).is_file()), None)
        if src is None:
            if name in REQUIRED_SIDE_FILES:
                raise FileNotFoundError(f"required side file {name} found neither in {snapshot} nor in {base}")
            if name in MIRRORED_SIDE_FILES and (out / name).exists():
                (out / name).unlink()
                copied[name] = {"source": None, "sha256": None, "note": "not in the snapshot; the copy save_pretrained derived from config.json was removed"}
                log(f"side file {name}: not in the snapshot (the locked revision ships none); removed the file save_pretrained derived from config.json")
                continue
            log(f"side file {name}: not in snapshot or base, skipped (optional)")
            continue
        replaced = (out / name).exists()
        shutil.copyfile(src, out / name)
        copied[name] = {"source": str(src), "sha256": sha256_file(out / name)}
        log(f"side file {name}: copied from {src}" + (" (replacing the file written by save_pretrained)" if replaced else ""))
    return copied


def cmd_merge(args: argparse.Namespace, run: TrainRun) -> bool:
    """Load base (served class) -> attach adapter -> merge -> bf16 -> save -> side files -> provenance -> gate."""
    import torch
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import AutoModelForImageTextToText

    base, adapter, out, snapshot = (Path(p).resolve() for p in (args.base, args.adapter, args.out, args.snapshot))
    guard_out_dir(out, base)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    lock = model_lock()
    base_layout = read_checkpoint_layout(base)
    run.write_metadata(inputs={"base": str(base), "adapter": str(adapter), "out": str(out), "snapshot": str(snapshot)},
                       hashes={"adapter_config_sha256": sha256_file(adapter / ADAPTER_CONFIG),
                               "adapter_model_sha256": sha256_file(adapter / ADAPTER_WEIGHTS),
                               "base_index_sha256": sha256_file(base / INDEX_FILE) if (base / INDEX_FILE).is_file() else None,
                               "base_layout_fingerprint": base_layout.fingerprint()},
                       base_path_names_lock_revision=lock["revision"] in str(base))

    # pre-flight on the adapter: structural problems abort before the 19 GB base is loaded; recipe/naming mismatches warn
    expect = AdapterExpectation(tuple(args.expect_targets.split(",")), args.expect_r, args.expect_alpha,
                                args.expect_base_model or lock["repo"], args.expect_revision or lock["revision"])
    checks = adapter_checks(adapter, expect, run.log)
    for c in checks:
        run.log(f"[{'PASS' if c.ok else 'FAIL'}] adapter.{c.name}: {c.detail}")
        run.metric({"kind": "adapter", "check": c.name, "ok": c.ok, **c.data})
    fatal = [c for c in checks if not c.ok and c.name in STRUCTURAL_ADAPTER_CHECKS]
    if fatal:
        raise ValueError("adapter rejected: " + "; ".join(f"{c.name}: {c.detail}" for c in fatal))
    for c in checks:
        if not c.ok:
            run.warn(f"adapter.{c.name} failed (not fatal for merge; verify-adapter enforces it): {c.detail}")

    with Stage(run, "load_base"):
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "output_loading_info": True}
        if args.device == "cuda":
            kwargs["device_map"] = "cuda"
        model, info = AutoModelForImageTextToText.from_pretrained(str(base), **kwargs)
        if type(model).__name__ != SERVED_ARCHITECTURE:
            raise TypeError(f"base loaded as {type(model).__name__}, expected {SERVED_ARCHITECTURE}")
        missing = list(info.get("missing_keys", []))
        unexpected = list(info.get("unexpected_keys", []))
        non_mtp = [k for k in unexpected if not k.startswith(MTP_PREFIX)]
        run.log(f"loaded {type(model).__name__}: missing {len(missing)}, unexpected {len(unexpected)} "
                f"({len(unexpected) - len(non_mtp)} mtp.*), _weight_conversions={getattr(model, '_weight_conversions', None)!r}")
        if missing or non_mtp:
            raise ValueError(f"base load not clean: missing {missing[:5]} unexpected non-mtp {non_mtp[:5]}")
        if getattr(model, "_weight_conversions", None):
            run.warn("_weight_conversions is not empty; the default save may re-key tensors (the gate will catch it)")
        run.metric({"stage": "load_base", "n_missing": len(missing), "n_unexpected": len(unexpected),
                    "n_params": sum(p.numel() for p in model.parameters())})
    if args.merge_dtype == "fp32":
        with Stage(run, "upcast_fp32"):
            model.float()
    with Stage(run, "attach_adapter"):
        peft_model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        lora_modules = [n for n, m in peft_model.named_modules() if isinstance(m, BaseTunerLayer)]
        run.log(f"adapter attached: {len(lora_modules)} LoRA layers")
        if not lora_modules:
            raise ValueError("adapter attached to zero modules; target names do not match this model")
        run.metric({"stage": "attach_adapter", "n_lora_layers": len(lora_modules)})
        adapted_keys = {n.removeprefix("base_model.model.") + ".weight" for n in lora_modules}   # bias=none: only weights change
    with Stage(run, "merge"), torch.no_grad():
        merged = peft_model.merge_and_unload(safe_merge=True)
        left = [n for n, m in merged.named_modules() if isinstance(m, BaseTunerLayer)]
        if left or type(merged).__name__ != SERVED_ARCHITECTURE:
            raise RuntimeError(f"merge left {len(left)} LoRA layers / class {type(merged).__name__}")
    with Stage(run, "cast_to_base_dtypes"), torch.no_grad():
        dtype_actions = cast_to_base_dtypes(merged, base, base_layout, adapted_keys)
        expected = {n: getattr(torch, TORCH_DTYPE_NAMES[base_layout.tensors[n][0] if n in base_layout.tensors else "BF16"])
                    for n, _ in merged.named_parameters()}
        wrong = [n for n, p in merged.named_parameters() if p.dtype != expected[n]]
        if wrong:
            raise RuntimeError(f"{len(wrong)} parameters not in the base's dtype after the cast, e.g. {wrong[:3]}")
        run.log(f"dtypes: {dtype_actions} (adapted weights {len(adapted_keys)})")
        run.metric({"stage": "cast_to_base_dtypes", **dtype_actions, "n_adapted_weights": len(adapted_keys)})
    with Stage(run, "save"):
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out), safe_serialization=True, max_shard_size=args.max_shard_size)
        run.log(f"saved to {out}: {sorted(p.name for p in out.iterdir())}")
    peak_gpu = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None
    del merged, peft_model, model
    with Stage(run, "side_files"):
        copied = assemble_side_files(out, snapshot, base, run.log)
    out_layout = read_checkpoint_layout(out)
    provenance = {
        "run_id": run.run_id, "created": utc_now(), "code": git_info(), "versions": package_versions(),
        "base": {"path": str(base), "lock_repo": lock["repo"], "lock_revision": lock["revision"],
                 "layout_fingerprint": base_layout.fingerprint()},
        "adapter": {"path": str(adapter), "adapter_config_sha256": sha256_file(adapter / ADAPTER_CONFIG),
                    "adapter_model_sha256": sha256_file(adapter / ADAPTER_WEIGHTS),
                    "adapter_config": json.loads((adapter / ADAPTER_CONFIG).read_text())},
        "merge": {"served_class": SERVED_ARCHITECTURE, "merge_dtype": args.merge_dtype,
                  "output_dtype": f"{args.dtype}; tensors the base stores in another dtype keep it (restored from the base shards)",
                  "dtype_actions": dtype_actions, "n_adapted_weights": len(adapted_keys),
                  "quantization": "none", "device": args.device, "max_shard_size": args.max_shard_size,
                  "safe_merge": True, "save_original_format": "default (True)"},
        "out": {"n_tensors": len(out_layout.tensors), "bytes": out_layout.header_bytes,
                "layout_fingerprint": out_layout.fingerprint(), "side_files": copied},
    }
    (out / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    run.write_metadata(provenance_file=str(out / PROVENANCE_FILE), peak_gpu_memory_gb=peak_gpu,
                       out_index_sha256=sha256_file(out / INDEX_FILE) if (out / INDEX_FILE).is_file() else None,
                       out_layout_fingerprint=out_layout.fingerprint(), out_bytes=out_layout.header_bytes)
    run.log(f"merged checkpoint: {len(out_layout.tensors)} tensors, {out_layout.header_bytes / 1e9:.2f} GB, "
            f"{len(out_layout.shard_files)} shard(s)")
    if args.skip_gate:
        run.warn("--skip-gate given: the layout gate was NOT run; run `gate --out ... --base ...` before serving")
        return True
    with Stage(run, "gate"):
        allowed = {**ALLOWED_CONFIG_DIFFS, **parse_allow_arg(args.allow_config_diff)}
        ok = report_checks(gate_checks(out, base, snapshot, allowed, run.log), run, "gate")
    run.log(f"GATE {'PASSED' if ok else 'FAILED'} for {out}")
    return ok


def cmd_gate(args: argparse.Namespace, run: TrainRun) -> bool:
    out, base, snapshot = (Path(p).resolve() for p in (args.out, args.base, args.snapshot))
    run.write_metadata(inputs={"out": str(out), "base": str(base), "snapshot": str(snapshot)})
    if out == base:
        run.log("out == base: self-check of the base checkpoint (passes modulo the mtp.* exclusion)")
    t0 = time.time()
    allowed = {**ALLOWED_CONFIG_DIFFS, **parse_allow_arg(args.allow_config_diff)}
    checks = gate_checks(out, base, snapshot, allowed, run.log)
    ok = report_checks(checks, run, "gate")
    fingerprints = {c.name.replace("_index_consistent", "_layout_fingerprint"): c.data.get("fingerprint")
                    for c in checks if c.name.endswith("_index_consistent")}
    run.write_metadata(hashes={"out_index_sha256": sha256_file(out / INDEX_FILE) if (out / INDEX_FILE).is_file() else None,
                               "base_index_sha256": sha256_file(base / INDEX_FILE) if (base / INDEX_FILE).is_file() else None,
                               **fingerprints}, gate_seconds=round(time.time() - t0, 2))
    run.log(f"GATE {'PASSED' if ok else 'FAILED'} for {out} in {time.time() - t0:.1f} s")
    return ok


def cmd_verify_adapter(args: argparse.Namespace, run: TrainRun) -> bool:
    adapter = Path(args.adapter).resolve()
    lock = model_lock()
    expect = AdapterExpectation(tuple(args.expect_targets.split(",")), args.expect_r, args.expect_alpha,
                                args.expect_base_model or lock["repo"], args.expect_revision or lock["revision"])
    run.write_metadata(inputs={"adapter": str(adapter)}, expectation=expect.__dict__,
                       hashes={"adapter_config_sha256": sha256_file(adapter / ADAPTER_CONFIG) if (adapter / ADAPTER_CONFIG).is_file() else None,
                               "adapter_model_sha256": sha256_file(adapter / ADAPTER_WEIGHTS) if (adapter / ADAPTER_WEIGHTS).is_file() else None})
    checks = adapter_checks(adapter, expect, run.log)
    naming = {"base_model_name_or_path", "revision"}
    if args.fix_adapter_config and any(not c.ok and c.name in naming for c in checks):
        backup = fix_adapter_config(adapter, expect.base_model, expect.revision, run.log)
        run.write_metadata(adapter_config_backup=str(backup), adapter_config_sha256_after_fix=sha256_file(adapter / ADAPTER_CONFIG))
        checks = adapter_checks(adapter, expect, run.log)
    return report_checks(checks, run, "adapter")


# -------------------------------------------------------------------------------------------------------- CLI
def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", default=None, help="logs/<run_id>/; default <subcommand>_<UTC timestamp>")
    p.add_argument("--out-root", default=None, help="SMOKE TESTS ONLY: parent of logs/ instead of the repository")
    p.add_argument("--note", default="", help="free text stored in metadata.json")


def add_adapter_expectation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--expect-targets", default=",".join(LORA_TARGET_MODULES), help="comma-separated lock recipe target_modules")
    p.add_argument("--expect-r", type=int, default=32)
    p.add_argument("--expect-alpha", type=int, default=64)
    p.add_argument("--expect-base-model", default=None, help="default: model.repo from configs/eval/model.yaml")
    p.add_argument("--expect-revision", default=None, help="default: model.revision from configs/eval/model.yaml")


def add_gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT), help="pinned config/tokenizer snapshot to compare against")
    p.add_argument("--allow-config-diff", action="append", default=[], metavar="KEY[=JSON]",
                   help="extra config.json field allowed to differ from the snapshot (flattened key, optional exact value)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apex_train.merge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="merge an adapter into the base and assemble + gate the served checkpoint")
    m.add_argument("--base", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", choices=["bf16"], default="bf16",
                   help="output dtype of the base's bf16 tensors (only bf16, quantization none); its fp32 tensors stay fp32")
    m.add_argument("--merge-dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of W + dW; fp32 default (~38 GB RAM for the real model)")
    m.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    m.add_argument("--max-shard-size", default="5GB")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for the CPU merge")
    m.add_argument("--skip-gate", action="store_true")
    add_gate_args(m)
    add_adapter_expectation_args(m)
    add_common_args(m)
    g = sub.add_parser("gate", help="header-only layout gate of a checkpoint directory against the base index")
    g.add_argument("--out", required=True, help="directory to gate")
    g.add_argument("--base", required=True)
    add_gate_args(g)
    add_common_args(g)
    v = sub.add_parser("verify-adapter", help="check adapter_config.json and the adapter key layout")
    v.add_argument("--adapter", required=True)
    v.add_argument("--fix-adapter-config", action="store_true", help="rewrite base_model_name_or_path / revision to the lock")
    add_adapter_expectation_args(v)
    add_common_args(v)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or new_run_id(args.cmd.replace("-", "_"))
    with TrainRun(run_id, Path(args.out_root).resolve() if args.out_root else None) as run:
        run.start_metadata(stage="merge", subcommand=args.cmd, config={k: v for k, v in vars(args).items() if k != "cmd"},
                           seed=getattr(args, "seed", None), data_version="n/a (no training data read)", note=args.note)
        run.log(f"{args.cmd}: run_id {run.run_id} -> {run.run_dir}")
        ok = {"merge": cmd_merge, "gate": cmd_gate, "verify-adapter": cmd_verify_adapter}[args.cmd](args, run)
        gpus = 1 if getattr(args, "device", "cpu") == "cuda" else 0
        run.finish("passed" if ok else "failed", merge_resource_rows(run, gpus=gpus, peak_gpu_mem_gb=run.read_metadata().get("peak_gpu_memory_gb")))
        run.log(f"{args.cmd}: {'passed' if ok else 'FAILED'} ({run.n_errors} error(s) recorded in {run.run_dir / 'errors.log'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
