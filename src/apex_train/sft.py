"""LoRA SFT of Qwen/Qwen3.5-9B with TRL + PEFT on pre-tokenised rows (configs/sft.yaml; docs/DATA_PLAN.md sections 4, 9, 10).

What a run does
  1. resolve configs/sft.yaml (+ `--preset NAME`, e.g. the 10 percent pilot, + the `smoke:` section under `--smoke`,
     + `--max-steps` and `--override k=v`);
  2. create logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv} + config_snapshot/;
  3. load the rows (data version = sha256 + row count per file, loss tokens); every row must fit data.max_length, the
     row cap (rows above it are refused, never truncated), and train at least one token; record the cap, the planned
     optimizer steps and the frozen mixture manifest (data.mixture_manifest, when the file exists);
  4. load the base with AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration (bf16, sdpa, vision tower
     frozen); the text-only Qwen3_5ForCausalLM is accepted only for a --smoke twin; the load must be clean
     (missing == [], unexpected keys only the 15 mtp.*);
  5. PEFT LoRA from the `lora` block (attention q/k/v/o of the 8 full-attention layers, GatedDeltaNet in_proj_qkv/z/b/a
     + out_proj of the 24 linear-attention layers, dense MLP gate/up/down of all 32 layers; embeddings, LM head, conv1d,
     norms and the vision tower stay frozen), TRL SFTTrainer with the shared trainer settings and the step's loss
     normalisation (optim.loss_normalisation: tokens = the token mean over the 16-row accumulation window, the default;
     rows = the mean of the rows' token means, the behaviour before the switch; common.apply_loss_normalisation); assert the
     adapter layout and that TRL's labels equal the assistant masks (every prepared row, then the first collated batch);
  6. train (optionally `--resume <checkpoint>`), one record per optimizer step in metrics.jsonl and
     curves/source/<run_id>_train.csv, every dev evaluation (loss only; every eval.eval_steps and at the last step) in
     <run_id>_eval.csv, checkpoints every logging.save_steps steps and at the last step (adapter + optimizer + scheduler
     + RNG, resumable, never deleted) under checkpoints/<run_id>/trainer/; a multi-epoch run also saves and evaluates at
     the end of every epoch but the last (epoch_end_callback); the training monitor (apex_train.monitor, the
     `monitor:` block) adds per-row and per-slice losses, LoRA health, GPU telemetry, alerts (alerts.jsonl) and the
     status.json heartbeat, and saves a checkpoint and stops the run after a CRITICAL non-finite loss or data alert
     (metadata status `stopped`, exit 1);
  7. save the final adapter to checkpoints/<run_id>/adapter/ (adapter_config base_model_name_or_path / revision =
     the locked hub id + revision, base_model.json sidecar); record sizes, throughput, peak memory, duration, GPU
     hours and cost.

Inputs : configs/sft.yaml; rows files from `python -m apex_train.data_format` (parquet or JSONL; the frozen set is
         data/processed/sft/{train,dev}_rows.parquet and the pilot draw pilot_rows.parquet); data/manifests/mixture.json
         (optional); the base checkpoint directory (read-only); configs/eval/model.yaml (repo + revision lock);
         configs/eval/model_snapshot/chat_template.jinja.
Outputs: logs/<run_id>/ (five files + config_snapshot/sft.yaml + the monitor's alerts.jsonl, status.json, rows.jsonl),
         checkpoints/<run_id>/{trainer/checkpoint-N/, adapter/, smoke_data/ (--smoke only)}, curves/source/<run_id>_{train,eval}.csv
         and the monitor's <run_id>_{health,slices,rows,lora,gpu}.csv. Nothing is ever deleted or overwritten.

CLI (run from the repository root, normally through scripts/train_sft.sh, which loads .env and picks .venv/train):
    PYTHONPATH=src .venv/train/bin/python -m apex_train.sft [--config configs/sft.yaml] [--preset pilot] [--run-id ID]
        [--smoke] [--max-steps N] [--resume <checkpoint-N dir | trainer dir>] [--override section.key=value ...]
        [--compare-steps-with logs/<run_id>] [--compare-tol 0.0] [--allow-kernel-fallback] [--threads N] [--note TEXT]
    run_id: sft_<run.name>_<UTC timestamp> (run.name is `main`, or `pilot` under --preset pilot); dryrun_ prefix with --smoke.
    --smoke: the yaml's smoke: section (tiny twin, 8 steps, evals at 6 and 8, checkpoints at 3, 6 and 8) on top of the preset; the
             twin (checkpoints/gpu_checks/tiny_cg_twin, git-ignored) is built on demand by
             scripts/dev/gpu_checks/make_tiny_cg_twin.py and the rows are produced by the apex_train.data_format CLI from
             its synthetic samples; without CUDA the twin trains in fp32 on CPU with the pure-torch fused-head shim.
             Production runs never install the shim and refuse to train without CUDA.
    Resume check: run A, then `--resume <A>/trainer/checkpoint-6 --compare-steps-with logs/<A>`: the overlapping steps
             must reproduce A's loss and grad_norm within --compare-tol (0.0 = bit-exact, as verified on CPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from apex_eval.common import REPO_ROOT, model_lock, new_run_id, sha256_file
from apex_train.common import (SERVED_ARCHITECTURE, ConfigError, FirstBatchLabelCheck, TrainMetrics, TrainRun, add_train_args,
                               adapter_layout, apply_loss_normalisation, apply_overrides, assert_adapter_layout, build_lora_config,
                               check_prepared_labels, loss_normalisation_of,
                               check_split, compare_steps, describe_split, ensure_tiny_twin, find_checkpoint, fused_head_status,
                               get_dotted, kernel_preflight, load_base_model, load_split, load_yaml, manifest_record, parse_override,
                               pick_device, planned_optimizer_steps, resolve_loader, resolve_path, rows_paths, run_cli, save_adapter,
                               seed_everything, set_dotted, shared_trainer_kwargs, torch_env, training_resource_rows,
                               twin_revision)
from apex_train.monitor import MonitorConfig, MonitorStop, TrainingMonitor, stop_monitored_run

DEFAULT_CONFIG = REPO_ROOT / "configs" / "sft.yaml"
SNAPSHOT_TEMPLATE = REPO_ROOT / "configs" / "eval" / "model_snapshot" / "chat_template.jinja"
KERNEL_NAMES = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d"}


# ------------------------------------------------------------------------------------------------------ config
def validate_config(cfg: dict[str, Any]) -> None:
    """Reject configurations that contradict the recipe or the run record's required fields (the shared helpers check the rest)."""
    for section in ("model", "lora", "data", "optim", "eval", "logging", "run"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"sft config: section {section!r} is missing or not a mapping")
    data, ev = cfg["data"], cfg["eval"]
    if data.get("packing") or data.get("padding_free"):
        raise ConfigError("data.packing / data.padding_free must be false (unsafe without FlashAttention)")
    if not isinstance(data.get("max_length"), int) or isinstance(data["max_length"], bool) or data["max_length"] <= 0:
        raise ConfigError("data.max_length must be a positive integer (the row cap)")
    if ev.get("strategy") not in ("steps", "epoch", "no"):
        raise ConfigError("eval.strategy must be steps, epoch or no")
    loss_normalisation_of(cfg["optim"], "optim")
    MonitorConfig.from_block(cfg.get("monitor"))   # the monitor: block (unknown keys and wrong types are ConfigErrors)


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """yaml -> (+preset) -> (+smoke overrides) -> (+--max-steps, CLI overrides) -> validated config and its provenance."""
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    applied: list[dict[str, Any]] = []
    if args.preset is not None:
        presets = get_dotted(cfg, "presets", {}) or {}
        if args.preset not in presets:
            raise ConfigError(f"--preset {args.preset!r}: not in presets {sorted(presets)}")
        applied += apply_overrides(cfg, presets[args.preset], f"preset:{args.preset}")
    if args.smoke:
        applied += apply_overrides(cfg, get_dotted(cfg, "smoke.overrides", {}) or {}, "smoke")
    cli = dict(parse_override(item) for item in args.override)
    if args.max_steps is not None:
        cli["optim.max_steps"] = args.max_steps
    applied += apply_overrides(cfg, cli, "cli")
    validate_config(cfg)
    return cfg, {"config_path": str(config_path), "config_sha256": sha256_file(config_path), "preset": args.preset,
                 "smoke": bool(args.smoke), "overrides_applied": applied}


def make_run_id(cfg: dict[str, Any], prov: dict[str, Any], explicit: str | None) -> str:
    """`sft_<run.name>_<UTC ts>` (run.name: main, or pilot under --preset pilot); smoke runs are prefixed dryrun_."""
    if explicit:
        return explicit
    parts = ["dryrun"] if prov["smoke"] else []
    return new_run_id(*parts, "sft", str(get_dotted(cfg, "run.name", "run")))


def prepare_smoke_rows(run: TrainRun, cfg: dict[str, Any], tokenizer_dir: Path, work: Path) -> dict[str, Path]:
    """Synthetic rows through the apex_train.data_format CLI (own content): N samples -> episode rows <= max_length."""
    syn = get_dotted(cfg, "smoke.synthetic")
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split in ("train", "dev"):
        samples, rows, report = work / f"{split}_samples.jsonl", work / f"{split}_rows.parquet", work / f"{split}_report.json"
        run_cli(run, f"smoke data ({split} samples)", [sys.executable, "-m", "apex_train.data_format", "--write-synthetic", str(samples),
                                                      "--n", str(syn[f"n_{split}_samples"]), "--seed", str(syn[f"seed_{split}"])])
        run_cli(run, f"smoke data ({split} rows)", [sys.executable, "-m", "apex_train.data_format", "--in", str(samples), "--out", str(rows),
                                                   "--tokenizer", str(tokenizer_dir), "--mode", "episode", "--max-length",
                                                   str(get_dotted(cfg, "data.max_length")), "--report", str(report)])
        out[split] = rows
    return out


def build_sft_config(shared: dict[str, Any], cfg: dict[str, Any], *, has_dev: bool, eval_on_start: bool = False) -> Any:
    """SFTConfig = the shared trainer block + the SFT-only settings (dev evaluation, masks as labels, no packing).
    `eval_on_start` (monitor.eval_on_start): one dev evaluation before the first step, the reference of the forgetting check."""
    from trl import SFTConfig

    ev = cfg["eval"]
    return SFTConfig(**shared, eval_strategy=str(ev["strategy"]) if has_dev else "no", eval_steps=int(ev.get("eval_steps", 0) or 0) or None,
                     per_device_eval_batch_size=int(ev.get("per_device_eval_batch_size", 1)), prediction_loss_only=True,
                     eval_on_start=bool(eval_on_start and has_dev), load_best_model_at_end=False, packing=False,
                     assistant_only_loss=False, completion_only_loss=None)   # labels come from our own token-span assistant_masks column


def epoch_end_callback(run: TrainRun, *, save_steps: int, eval_steps: int | None) -> Any:
    """A multi-epoch run (optim.num_train_epochs > 1) keeps the exact end of every epoch but the last: a checkpoint and,
    with a dev set (eval_steps not None), a dev evaluation, unless that step already got one from save_steps / eval_steps.
    The last epoch ends at the last step, which the Trainer saves and evaluates anyway. Single-epoch runs are unaffected."""
    from transformers import TrainerCallback

    class _EpochEnd(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            if state.global_step >= state.max_steps or control.should_training_stop:
                return control
            step = state.global_step
            save = not (save_steps and step % save_steps == 0)
            evaluate = eval_steps is not None and step % eval_steps != 0
            control.should_save = control.should_save or save
            control.should_evaluate = control.should_evaluate or evaluate
            run.log(f"epoch {state.epoch:g} ends at step {step}: checkpoint-{step} "
                    f"({'saved now' if save else 'already saved'}), dev evaluation {'now' if evaluate else 'already done or no dev set'}")
            run.metric({"event": "epoch_end", "step": step, "epoch": state.epoch, "extra_save": save, "extra_eval": evaluate})
            return control

    return _EpochEnd()


# --------------------------------------------------------------------------------------------------- training
def run_training(args: argparse.Namespace, cfg: dict[str, Any], prov: dict[str, Any], run: TrainRun, root: Path) -> dict[str, Any]:
    """The whole run; returns the summary dict stored in metadata.json. Raises on any failed check."""
    import torch
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    smoke = prov["smoke"]
    device = pick_device(cpu_dry_run=smoke)
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype_name = str(get_dotted(cfg, "model.dtype")) if device == "cuda" else "float32"   # the CPU twin runs fp32
    head = fused_head_status(cpu_dry_run=smoke and device == "cpu")
    kernels = kernel_preflight()
    if device == "cuda" and not all(kernels.values()):
        msg = f"GDN kernels not importable: {[KERNEL_NAMES[k] for k, ok in kernels.items() if not ok]}"
        if smoke or args.allow_kernel_fallback:
            run.warn(msg + " (continuing: torch fallback, correct but slow)")
        else:
            raise RuntimeError(msg + "; install them (scripts/gpu_setup_train_env.sh) or pass --allow-kernel-fallback")
    run.log(f"device={device} dtype={dtype_name} smoke={smoke} fused_head={head} kernels={kernels}")

    # --- lock and model identity
    lock = model_lock()
    model_path = resolve_path(str(get_dotted(cfg, "model.path")))
    if smoke:
        ensure_tiny_twin(model_path, run)   # the git-ignored twin is rebuilt when missing (make_tiny_cg_twin.py)
    repo = get_dotted(cfg, "model.repo", None)
    revision = get_dotted(cfg, "model.revision", None)
    if not smoke and (repo, str(revision)) != (lock["repo"], lock["revision"]):
        raise ConfigError(f"model.repo/revision {repo}@{revision} != the lock {lock['repo']}@{lock['revision']} (configs/eval/model.yaml); "
                          "fix configs/sft.yaml")
    if revision is None:   # the smoke twin has no hub revision: identify it by its config
        revision = twin_revision(model_path)
    auto_class, arch = resolve_loader(model_path, allow_text_only_twin=smoke)
    seed = int(get_dotted(cfg, "logging.seed"))
    seed_everything(seed)

    # --- directories
    out_root = (root / "checkpoints" if args.out_root else resolve_path(str(get_dotted(cfg, "run.output_root")))) / run.run_id
    trainer_dir, adapter_dir = out_root / "trainer", out_root / "adapter"
    curves_dir = root / "curves" / "source" if args.out_root else resolve_path(str(get_dotted(cfg, "run.curves_dir")))
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} exists and is not empty; never overwrite checkpoints")
    out_root.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(run, get_dotted(cfg, "monitor", None), stage="sft", device=device, curves_dir=curves_dir, checkpoint_root=out_root,
                              smoke=smoke, resume=bool(args.resume))
    run.attach_monitor(monitor)

    # --- data
    if smoke and get_dotted(cfg, "data.train_rows", None) is None:
        paths = prepare_smoke_rows(run, cfg, model_path, out_root / "smoke_data")
        set_dotted(cfg, "data.train_rows", str(paths["train"]))
        set_dotted(cfg, "data.dev_rows", str(paths["dev"]))
    train_paths, dev_paths = rows_paths(get_dotted(cfg, "data.train_rows")), rows_paths(get_dotted(cfg, "data.dev_rows", None))
    if not train_paths:
        raise ConfigError("data.train_rows is not set")
    max_length = int(get_dotted(cfg, "data.max_length"))   # the row cap (M4.1)
    train_ds, train_counts = load_split(train_paths)
    data_version: dict[str, Any] = {"train": describe_split(train_ds, train_paths, train_counts)}
    check_split(data_version["train"], max_length, "train")
    dev_ds = None
    if dev_paths:
        dev_ds, dev_counts = load_split(dev_paths)
        data_version["dev"] = describe_split(dev_ds, dev_paths, dev_counts)
        check_split(data_version["dev"], max_length, "dev")
        dev_ds = monitor.select_eval_rows(dev_ds)   # monitor.eval_max_rows_per_slice: stratified subset; null = every dev row
        if monitor.eval_subset and monitor.eval_subset["selection"] != "all":
            data_version["dev"]["eval_subset"] = monitor.eval_subset
    data_seed = int(get_dotted(cfg, "logging.data_seed", seed))
    planned = planned_optimizer_steps(len(train_ds), cfg["optim"])
    manifest_path = get_dotted(cfg, "data.mixture_manifest", None)
    mixture = manifest_record(resolve_path(str(manifest_path)) if manifest_path else None,
                              [f for split in data_version.values() for f in split["files"]])
    if mixture is None and manifest_path:
        run.warn(f"no mixture manifest at {manifest_path}: metadata.json records no frozen mixture block")
    elif mixture is not None and not all(mixture["rows_sha256_listed"].values()):
        run.warn(f"rows files not listed by sha256 in {mixture['path']}: {[p for p, ok in mixture['rows_sha256_listed'].items() if not ok]} "
                 "(the rows on disk may not be the frozen set)")
    monitor.set_plan(planned_steps=planned["optimizer_steps"], max_length=max_length,
                     eval_every=int(get_dotted(cfg, "eval.eval_steps", 0) or 0) or None if dev_ds is not None else None)
    train_desc = data_version["train"]
    run.log(f"train rows {train_desc['n_rows']} ({[f['n_rows'] for f in train_desc['files']]} per file), tokens {train_desc['n_tokens']:,d}, "
            f"loss tokens {train_desc['n_assistant_tokens']:,d}, max row {train_desc['max_tokens']} <= row cap {max_length}; dev rows "
            f"{len(dev_ds) if dev_ds is not None else 0}; planned optimizer steps {planned['optimizer_steps']} ({planned['source']}, "
            f"{planned['rows_per_step']} rows per step)")
    run.write_metadata(config=cfg, row_cap=max_length, data_version=data_version, planned=planned, mixture_manifest=mixture)

    # --- tokenizer, template, model, trainer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    template_sha = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
    snapshot_sha = hashlib.sha256(SNAPSHOT_TEMPLATE.read_bytes()).hexdigest()
    if template_sha != snapshot_sha:
        run.warn("tokenizer chat template differs from configs/eval/model_snapshot/chat_template.jinja")
    model, model_info = load_base_model(run, model_path, auto_class=auto_class, dtype_name=dtype_name, device=device,
                                        attn_implementation=str(get_dotted(cfg, "model.attn_implementation")))
    model_info.update(path=str(model_path), architecture=arch, revision=revision, config_sha256=sha256_file(model_path / "config.json"),
                      chat_template_sha256=template_sha, chat_template_matches_snapshot=template_sha == snapshot_sha)
    lora_cfg = build_lora_config(cfg["lora"])
    shared = shared_trainer_kwargs(cfg["optim"], cfg["model"], cfg["logging"], out_dir=trainer_dir, device=device, seed=seed,
                                   data_seed=data_seed, logging_steps=int(get_dotted(cfg, "logging.logging_steps", 1)), max_length=max_length)
    sft_cfg = build_sft_config(shared, cfg, has_dev=dev_ds is not None, eval_on_start=monitor.enabled and monitor.cfg.eval_on_start)
    metrics = TrainMetrics(run, curves_dir, device=device, extra_columns=("entropy", "mean_token_accuracy"), resume=bool(args.resume),
                           monitor=monitor)
    epoch_end = epoch_end_callback(run, save_steps=int(get_dotted(cfg, "logging.save_steps", 0) or 0),
                                   eval_steps=sft_cfg.eval_steps if dev_ds is not None else None)
    t0 = time.time()
    trainer = monitor.trainer_class(SFTTrainer)(model=model, args=sft_cfg, train_dataset=train_ds, eval_dataset=dev_ds, processing_class=tokenizer,
                                                peft_config=lora_cfg, callbacks=[metrics.callback(trainer_dir), epoch_end])
    init_s = time.time() - t0
    loss_norm = apply_loss_normalisation(trainer, loss_normalisation_of(cfg["optim"], "optim"))
    run.log(f"loss normalisation {loss_norm['mode']}: {loss_norm['definition']}")
    layout = adapter_layout(trainer.model)
    assert_adapter_layout(layout, trainer, expected_trainable=get_dotted(cfg, "lora.expected_trainable_params", None),
                          served_class=arch == SERVED_ARCHITECTURE)
    run.log(f"trainer ready in {init_s:.1f} s: trainable {layout['trainable']:,d} / {layout['total']:,d} ({layout['trainable_pct']} %), "
            f"{layout['n_lora_modules']} LoRA modules, adapter dtypes {layout['adapter_param_dtypes']}, first key {layout['first_adapter_key']}")

    # --- self-check: TRL labels == assistant masks (every prepared row, then the first collated batch)
    label_check = check_prepared_labels(trainer.train_dataset, train_ds)
    by_hash = label_check.pop("by_hash")
    monitor.register_rows("train", train_ds, label_check.pop("index_by_hash"))
    run.log(f"self-check labels == assistant_masks: {label_check['rows_checked']} rows, {label_check['trained_tokens']:,d} trained tokens: OK")
    if dev_ds is not None:
        dev_check = check_prepared_labels(trainer.eval_dataset, dev_ds)
        by_hash = {**by_hash, **dev_check.pop("by_hash")}   # the first collated batch may be an eval batch (eval_on_start)
        monitor.register_rows("eval", dev_ds, dev_check.pop("index_by_hash"))
        label_check["dev"] = dev_check
    first_batch: dict[str, Any] = {}

    def on_first_batch(result: dict[str, Any]) -> None:
        first_batch.update(result)
        run.log(f"self-check first collated batch: {result}")
        run.metric({"event": "self_check_first_batch", **result})

    trainer.data_collator = FirstBatchLabelCheck(trainer.data_collator, by_hash, on_first_batch)
    monitor.attach(trainer)   # after the label check: tags collated rows, LoRA health, planned LR schedule

    resume_from = find_checkpoint(resolve_path(args.resume)) if args.resume else None
    resumed_step = json.loads((resume_from / "trainer_state.json").read_text()).get("global_step") if resume_from else None
    run.write_metadata(data_version=data_version, model=model_info, lora_config=lora_cfg.to_dict(), sft_config=json.loads(sft_cfg.to_json_string()),
                       loss_normalisation=loss_norm,
                       trainable=layout, fused_head=head, kernels_importable=kernels, self_check_labels={**label_check, "ok": True},
                       trainer_init_sec=round(init_s, 1), resumed_from=str(resume_from) if resume_from else None, resumed_from_step=resumed_step,
                       dirs={"checkpoints": str(out_root), "trainer": str(trainer_dir), "adapter": str(adapter_dir),
                             "curves_train": str(metrics.train_csv.path), "curves_eval": str(metrics.eval_csv.path)})
    if resume_from:
        run.log(f"resuming from {resume_from} (global_step {resumed_step})")

    # --- train
    t1 = time.time()
    try:
        train_out = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM (peak allocated/reserved GB {metrics.train_records[-1] if metrics.train_records else 'n/a'}): {str(exc)[:300]}") from exc
    train_s = time.time() - t1
    if not first_batch:
        raise AssertionError("the first-batch label check never ran (no batch was collated)")
    if monitor.stop_reason:
        stop_monitored_run(run, monitor, metrics, device=device, global_step=train_out.global_step, trained_tokens=data_version["train"]["n_assistant_tokens"])
    losses = [float(r["loss"]) for r in metrics.train_records]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError(f"non-finite or missing training loss: {losses[:5]}")

    # --- save, compare, account
    sidecar = save_adapter(trainer, adapter_dir, run, base_repo=repo, base_revision=revision, base_path=model_path, lora_block=cfg["lora"],
                           data_version=data_version)
    comparison = None
    if args.compare_steps_with:
        comparison = compare_steps(resolve_path(args.compare_steps_with), metrics.train_records, args.compare_tol)
        run.log(f"step comparison vs {comparison['other_run']}: steps {comparison['common_steps']} max |diff| {comparison['max_abs_diff']} "
                f"(tol {comparison['tol']}): {'OK' if comparison['ok'] else 'MISMATCH'}")
        if not comparison["ok"]:
            raise AssertionError(f"resumed steps do not reproduce the reference run: {comparison['per_step']}")
    if run.capture.fallbacks:   # expected without CUDA (no fla / causal-conv1d wheels for CPU); an error on the GPU VM
        msg = f"kernel fallbacks during training: {sorted(run.capture.fallbacks)} (correct but slow)"
        run.error(msg) if device == "cuda" else run.warn(msg)
    rows = training_resource_rows(run, metrics, device=device, adapter_bytes=sidecar["adapter_bytes"],
                                  trained_tokens=data_version["train"]["n_assistant_tokens"]) + monitor.resource_rows()
    best_eval = min(metrics.eval_records, key=lambda r: r["eval_loss"]) if metrics.eval_records else None
    summary = {"global_step": train_out.global_step, "steps_this_run": len(losses), "first_loss": losses[0], "last_loss": losses[-1],
               "train_loss_mean": train_out.training_loss, "train_seconds": round(train_s, 1), **metrics.train_summary,
               "eval": metrics.eval_records, "best_eval": best_eval, "checkpoints": metrics.checkpoints,
               "adapter_dir": str(adapter_dir), "adapter_bytes": sidecar["adapter_bytes"], "adapter_model_sha256": sidecar["adapter_model_sha256"],
               "resources": {r[0]: r[1] for r in rows}, "self_check_first_batch": first_batch, "step_comparison": comparison}
    run.finish("finished", rows, summary=summary, adapter=sidecar, monitor=monitor.summary())
    return summary



# -------------------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apex_train.sft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_train_args(p, default_config=DEFAULT_CONFIG)
    p.add_argument("--preset", default=None, metavar="NAME", help="apply presets.NAME of the yaml first (pilot: the 10 percent draw)")
    p.add_argument("--smoke", action="store_true", help="tiny-twin dry run with the yaml's smoke: section (CPU allowed)")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="stop after N optimizer steps (optim.max_steps; check runs, e.g. T7 through this entry point)")
    p.add_argument("--resume", default=None, metavar="PATH", help="checkpoint-N directory, or a trainer directory (latest checkpoint)")
    p.add_argument("--compare-steps-with", default=None, metavar="RUN_DIR",
                   help="after training, compare overlapping steps' loss/grad_norm with that run's metrics.jsonl (resume check)")
    p.add_argument("--compare-tol", type=float, default=0.0, help="max |diff| accepted by --compare-steps-with (0.0 = bit-exact)")
    p.add_argument("--allow-kernel-fallback", action="store_true", help="train even if fla / causal-conv1d are not importable")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg, prov = resolve_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    root = Path(args.out_root) if args.out_root else REPO_ROOT
    try:
        with TrainRun(make_run_id(cfg, prov, args.run_id), root) as run:
            run.snapshot_config(Path(prov["config_path"]))
            run.start_metadata(stage="sft", config=cfg, config_provenance=prov, seed=get_dotted(cfg, "logging.seed"), note=args.note,
                               torch_env=torch_env())
            run.log(f"sft: run_id {run.run_id} -> {run.run_dir} (config {prov['config_path']}, preset {prov['preset']}, smoke {prov['smoke']})")
            summary = run_training(args, cfg, prov, run, root)
            run.log(f"sft: finished, {summary['steps_this_run']} steps, adapter {summary['adapter_dir']}")
    except MonitorStop as exc:
        print(f"training stopped by the monitor: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
