"""Stage 2 for Qwen/Qwen3.5-9B: turn-level DPO on the SFT model (`dpo`, revision 5, the default method) or, as the earlier
revision-4.3 design (docs/DATA_PLAN.md section 11), continued SFT of the SFT adapter on execution-filtered rows.

`filtered_sft` (the revision-4.3 method, kept for reference; --method filtered_sft)
  The raw base (model.base, the weights configs/sft.yaml trains on) is loaded with AutoModelForImageTextToText (the served
  class Qwen3_5ForConditionalGeneration, bf16, sdpa, vision tower frozen) and the SFT adapter (model.init_adapter) is
  attached with PeftModel.from_pretrained(is_trainable=True): the same adapter keeps training, with its rank, targets and
  weights. No fresh adapter, no merge before stage 2 (the rollouts are served from the merged SFT weights; training
  continues the adapter over the raw base). Startup checks, all before training:
    - the adapter's adapter_config.json names the locked base (base_model_name_or_path / revision ==
      configs/eval/model.yaml; the smoke twin: its config identity) and its r / lora_alpha / target_modules equal the
      `lora` block;
    - every tensor of adapter_model.safetensors is loaded into the model unchanged (PEFT only warns on a key mismatch and
      would leave fresh weights in place), the trainable parameter count equals the adapter's parameter count and
      lora.expected_trainable_params, and the vision tower has no trainable parameter;
    - every row fits max_length (the SFT row cap); a longer row stops the run, because dropping it would unbalance the
      loss-token-matched arms;
    - TRL's labels equal the assistant masks on every prepared row and on the first collated batch (as in SFT).
  Arms (`--arm`, filtered_sft.arms): `positives` = verified positives (arm a) and `control` = unfiltered format-valid
  rollouts sampled to arm (a)'s rollout loss tokens (arm b). Each trains on its own rows file plus the shared replay rows
  (filtered_sft.replay_rows), one epoch at LR 5e-5, the step loss token-weighted over the accumulation window
  (filtered_sft.loss_normalisation: tokens, as in SFT; rows = the mean of the rows' token means). The control arm runs arm (a)'s optimizer steps through
  `--match-steps-of logs/<arm a run_id>`, which also asserts that both runs continue the same adapter on the same replay
  rows. Checkpoints at checkpointing.fraction (50 percent) and 100 percent of the planned steps. Rows, loss tokens and
  planned steps of the arm and the replay go to metadata.json before training, with the stage-2 set manifest
  (filtered_sft.manifest) when it exists. The continued adapter is saved as a plain PEFT adapter over the base
  (adapter_config base_model_name_or_path / revision = the lock; base_model.json records the init adapter), loadable in
  one line: PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(<base>), <adapter dir>).

`dpo` (revision 5, the default method of configs/stage2.yaml)
  TRL 1.15.0 DPOTrainer with a fresh LoRA adapter (the `lora` block) on a merged SFT checkpoint (`--base <merged dir>`,
  output of scripts/merge_adapter.sh merge) and ref_model=None: the reference log-probs come from the same weights with
  the adapter disabled, i.e. the merged SFT policy, with no second copy of the base. With an init adapter the disabled-
  adapter reference would be the raw base, so this path refuses model.init_adapter (pass --override
  model.init_adapter=null). Preference pairs are conversational: `prompt` = messages (system/user/assistant/tool,
  earlier assistant turns with `reasoning_content`, the request's `tools`), `chosen` / `rejected` = one assistant message
  each with `reasoning_content` and `content` and/or `tool_calls`. The pairs are rendered here with the shipped chat
  template (`prompt_ids = render(prompt, add_generation_prompt=True)`, `completion_ids = render(prompt +
  [turn])[len(prompt_ids):]` up to and including `<|im_end|>`; a rejected turn that is reasoning only, a loop cut at the
  sampling cap, is its trace tokens alone, with no `</think>` or `<|im_end|>`) and TRL gets pre-tokenised rows: TRL's conversational path
  stores messages in Arrow, which None-pads heterogeneous `tool_calls[].function.arguments` dicts into `<parameter=x>
  null` targets, and TRL 1.15.0 has no `max_prompt_length`, so prompt-side control (tool-output truncation, never
  completions) happens before tokenisation. rewards/chosen, rewards/rejected, rewards/margins, rewards/accuracies go to
  metrics.jsonl and the curves CSV per step.

Both paths use the shared trainer settings of apex_train.common (bf16 base, fp32 LoRA parameters, non-reentrant
checkpointing, TRL's fused LM head, checkpoints never deleted) and the training monitor of apex_train.monitor (the `monitor:`
block: per-step health, LoRA health, GPU telemetry, alerts, status.json; per-row and per-slice losses for filtered_sft). Stage 2
has no dev evaluation unless monitor.eval_rows names a rows file (e.g. the SFT dev shard): then the loss per slice is evaluated
before step 1 (eval_on_start), every eval_every_fraction of the planned steps and at the end, on a task-stratified subset
(eval_max_rows_per_slice), the forgetting check of the monitor. A CRITICAL non-finite or data alert saves a checkpoint and stops
the run (status `stopped`, exit 1, no final adapter).

Inputs : configs/stage2.yaml; the raw base directory (model.base, read-only); the SFT adapter directory
         (model.init_adapter); the arm's rows file and the replay rows file (apex_train.data_format rows, written by the
         stage-2 set builder); the stage-2 set manifest (optional); for dpo a preference-pair JSONL (dpo.data.pairs:
         `{"id", "prompt": [...], "tools": [...] | null, "chosen": {...}, "rejected": {...}, "metadata": {...}}` per line).
Outputs: logs/<run_id>/{metadata.json, train.log, metrics.jsonl, errors.log, resources.csv} (+ config_snapshot/, the monitor's
         alerts.jsonl, status.json, rows.jsonl and, for dpo, data_drops.jsonl when pairs were dropped),
         curves/source/<run_id>_{train,eval}.csv and the monitor's <run_id>_{health,slices,rows,lora,gpu}.csv,
         checkpoints/stage2/<run_id>/{trainer/checkpoint-N/, adapter/ (+ base_model.json)}.

CLI (run from the repository root, normally through scripts/train_stage2.sh):
    PYTHONPATH=src .venv/train/bin/python -m apex_train.stage2 --method filtered_sft --arm positives|control [--init-adapter <SFT adapter dir>]
        [--rows <arm rows>] [--replay-rows <replay rows>] [--match-steps-of logs/<arm a run_id>] [--max-steps N]
        [--base <raw base dir>] [--config configs/stage2.yaml] [--run-id ID] [--override key.path=value ...] [--note TEXT]
    PYTHONPATH=src .venv/train/bin/python -m apex_train.stage2 --method filtered_sft --smoke --arm positives --init-adapter <adapter of an SFT --smoke run>
        (tiny twin; synthetic arm rows (8) and replay rows (4) built through the data_format CLI; CPU allowed)
    PYTHONPATH=src .venv/train/bin/python -m apex_train.stage2 --method dpo --base <merged SFT dir> --pairs <pairs.jsonl>
        --override model.init_adapter=null          (the default method; `--smoke --method dpo`: 8 synthetic pairs on the twin)
    run_id: <arm run_id_prefix>_<UTC timestamp> (stage2_positives_..., stage2_control_...) or stage2_dpo_...; dryrun_ prefix with --smoke.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from apex_eval.common import REPO_ROOT, model_lock, new_run_id, sha256_file
from apex_train.common import (SERVED_ARCHITECTURE, TINY_TWIN_DIR, ConfigError, FirstBatchLabelCheck, TrainMetrics, TrainRun,
                               add_train_args, adapter_layout, apply_loss_normalisation, apply_overrides, assert_adapter_layout,
                               build_lora_config, loss_normalisation_of,
                               check_prepared_labels, check_split, checkpoint_identity, describe_split, ensure_tiny_twin,
                               fused_head_status, get_dotted, load_base_model, load_split, load_yaml, manifest_record, parse_override,
                               pick_device, planned_optimizer_steps, require_type, resolve_loader, resolve_path, rows_paths, run_cli,
                               save_adapter, seed_everything, shared_trainer_kwargs, torch_env, training_resource_rows, twin_revision)
from apex_train.monitor import MonitorConfig, MonitorStop, TrainingMonitor, stop_monitored_run
from apex_train.data_format import (IM_END_ID, TRUNCATION_MARKER, DataFormatError, Message, PrefixViolationError, RenderError,
                                    RowTooLongError, Sample, SampleValidationError, encode_messages, head_tail_truncate, histogram,
                                    iter_jsonl, load_tokenizer)

METHODS = ("filtered_sft", "dpo")
LOSS_TYPES = ("sigmoid", "ipo", "sft")   # dpo.loss_type: one of these or a list of them (TRL 1.15 combines a list by dpo.loss_weights)
TOO_LONG_POLICIES = ("drop", "truncate_tool_outputs")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage2.yaml"
REQUIRED_DPO_METRICS = ("rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "loss", "learning_rate", "grad_norm")
REQUIRED_SFT_METRICS = ("loss", "learning_rate", "grad_norm")
DPO_CURVE_EXTRA = ("rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "logps/chosen", "logps/rejected")
SFT_CURVE_EXTRA = ("entropy", "mean_token_accuracy")
# --smoke settings on the shrunken twin, applied before --max-steps and --override. null rows / pairs = synthetic inputs.
SMOKE_OVERRIDES: dict[str, dict[str, Any]] = {
    "filtered_sft": {"model.base": str(TINY_TWIN_DIR), "model.repo": None, "model.revision": None, "lora.expected_trainable_params": None,
                     "filtered_sft.per_device_train_batch_size": 1, "filtered_sft.gradient_accumulation_steps": 2,
                     "filtered_sft.effective_batch_size": 2, "filtered_sft.max_length": 1024, "filtered_sft.warmup_ratio": 0.0,
                     "filtered_sft.replay_rows": None, "filtered_sft.manifest": None,
                     **{f"filtered_sft.arms.{arm}.rows": None for arm in ("positives", "control")}},
    "dpo": {"model.base": str(TINY_TWIN_DIR), "model.init_adapter": None, "model.repo": None, "model.revision": None,
            "lora.expected_trainable_params": None, "dpo.max_steps": 4, "dpo.num_train_epochs": 1, "dpo.per_device_train_batch_size": 2,
            "dpo.gradient_accumulation_steps": 1, "dpo.effective_batch_size": 2, "dpo.max_length": 1024, "dpo.max_prompt_length": 768,
            "dpo.learning_rate": 5.0e-5, "dpo.lr_scheduler_type": "constant", "dpo.warmup_ratio": 0.0, "dpo.data.pairs": None},
}
# data_format's synthetic kinds cycle by 6 and two of them drop in episode mode: 12 samples -> 8 arm rows, 6 -> 4 replay rows
SMOKE_ARM_SAMPLES, SMOKE_REPLAY_SAMPLES = 12, 6


# ------------------------------------------------------------------------------------------- preference pairs
class IdenticalCompletionsError(DataFormatError):
    """chosen and rejected render to the same completion tokens (no preference signal)."""

    reason = "identical_completions"


class PromptTooLongError(DataFormatError):
    """The prompt exceeds max_prompt_length and cannot (or may not) be shortened by tool-output truncation."""

    reason = "prompt_too_long"


@dataclass
class PairOptions:
    """Rendering and length policy for preference pairs (recorded in metadata)."""

    thinking: bool = True
    max_length: int = 16384
    max_prompt_length: int | None = None
    on_too_long: str = "drop"
    min_tool_tokens: int = 64

    def __post_init__(self) -> None:
        if self.on_too_long not in TOO_LONG_POLICIES:
            raise ConfigError(f"dpo.on_too_long must be one of {TOO_LONG_POLICIES}, got {self.on_too_long!r}")
        if self.max_length < 2:
            raise ConfigError("dpo.max_length must be >= 2")
        if self.max_prompt_length is not None and not 0 < self.max_prompt_length < self.max_length:
            raise ConfigError("dpo.max_prompt_length must be positive and smaller than dpo.max_length")
        if self.min_tool_tokens < 2:
            raise ConfigError("dpo.min_tool_tokens must be >= 2")


def is_unterminated_reasoning(turn: Message) -> bool:
    """An assistant turn that is reasoning only: a non-empty trace, no content, no tool call. This is what a sample cut off
    at the sampling cap inside its thinking looks like (a reasoning loop that never reached `</think>`)."""
    return (turn.role == "assistant" and bool((turn.reasoning_content or "").strip()) and not turn.content.strip()
            and not turn.tool_calls)


@dataclass
class PreferencePair:
    """One validated pair: shared prompt (messages + tools) and two candidate assistant turns. `rejected_unterminated`:
    the rejected turn is reasoning only, cut at the sampling cap (is_unterminated_reasoning), rendered without an ending."""

    id: str
    prompt: list[Message]
    tools: list[dict[str, Any]] | None
    chosen: Message
    rejected: Message
    meta: dict[str, Any]
    rejected_unterminated: bool = False

    @classmethod
    def from_dict(cls, d: Any, *, default_id: str, thinking: bool) -> PreferencePair:
        """Parse + validate through data_format's Sample rules: prompt + chosen and prompt + rejected must each be a
        valid sample (roles, tool-call structure, dict arguments, non-empty reasoning_content on every assistant turn
        in thinking mode; no traces in non-thinking mode) and the prompt must end with a user or tool turn. One
        exception, on the rejected side only and in thinking mode only: a reasoning-only turn (is_unterminated_reasoning,
        a loop cut at the sampling cap) is accepted as an unterminated rejected turn. The chosen side stays strict."""
        if not isinstance(d, dict):
            raise SampleValidationError(f"pair must be a JSON object, got {type(d).__name__}")
        pid = str(d.get("id") or (d.get("metadata") or {}).get("id") or default_id)
        prompt = d.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise SampleValidationError(f"pair {pid}: prompt must be a non-empty list of messages")
        samples, unterminated = {}, None
        for name in ("chosen", "rejected"):
            turn = d.get(name)
            if isinstance(turn, list):
                if len(turn) != 1:
                    raise SampleValidationError(f"pair {pid}: {name} must be one assistant message (got {len(turn)})")
                turn = turn[0]
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                raise SampleValidationError(f"pair {pid}: {name} must be an assistant message")
            if name == "rejected" and thinking:
                parsed = Message.from_dict(turn, f"pair {pid}: rejected")
                if is_unterminated_reasoning(parsed):
                    unterminated = parsed
                    continue
            sample = Sample.from_dict({"messages": prompt + [turn], "tools": d.get("tools"), "metadata": {"id": pid}})
            sample.validate(enable_thinking=thinking)
            samples[name] = sample
        if samples["chosen"].messages[-2].role not in ("user", "tool"):
            raise SampleValidationError(f"pair {pid}: the prompt must end with a user or tool message")
        meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        flat = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool)) or v is None}
        return cls(id=pid, prompt=list(samples["chosen"].messages[:-1]), tools=samples["chosen"].tools,
                   chosen=samples["chosen"].messages[-1], rejected=unterminated or samples["rejected"].messages[-1], meta=flat,
                   rejected_unterminated=unterminated is not None)


def completion_end(tokenizer: Any, ids: Sequence[int]) -> int:
    """Exclusive end of a rendered completion: just after <|im_end|> (the template's glue '\\n' after it is never generated)."""
    if ids and ids[-1] == IM_END_ID:
        return len(ids)
    if len(ids) >= 2 and ids[-2] == IM_END_ID and tokenizer.decode([ids[-1]]) == "\n":
        return len(ids) - 1
    raise RenderError("completion render does not end with <|im_end|>")


def _prompt_ids(tokenizer: Any, prompt: Sequence[Message], tools: Any, thinking: bool) -> list[int]:
    return encode_messages(tokenizer, [m.to_template() for m in prompt], tools, add_generation_prompt=True, enable_thinking=thinking)


def fit_prompt(pair: PreferencePair, tokenizer: Any, opts: PairOptions) -> tuple[list[Message], list[int], int, int]:
    """Prompt tokens under opts.max_prompt_length. With on_too_long == 'truncate_tool_outputs', tool results in the prompt
    are shortened longest-first (head + tail around TRUNCATION_MARKER, floor min_tool_tokens); user turns, the system
    prompt and the candidate completions are never touched. Returns (messages, prompt_ids, n_truncated, tokens_removed)."""
    msgs = [dataclasses.replace(m) for m in pair.prompt]
    idx = [i for i, m in enumerate(msgs) if m.role == "tool"]
    original = {i: msgs[i].content for i in idx}
    length = {i: len(tokenizer.encode(original[i], add_special_tokens=False)) for i in idx}
    budget = dict(length)
    marker = len(tokenizer.encode(TRUNCATION_MARKER.format(n=10 ** 6), add_special_tokens=False))
    for _ in range(8 * len(idx) + 16):
        prompt_ids = _prompt_ids(tokenizer, msgs, pair.tools, opts.thinking)
        excess = len(prompt_ids) - opts.max_prompt_length if opts.max_prompt_length else 0
        if excess <= 0:
            return msgs, prompt_ids, sum(budget[i] < length[i] for i in idx), sum(length[i] - budget[i] for i in idx)
        if opts.on_too_long != "truncate_tool_outputs":
            raise PromptTooLongError(f"pair {pair.id}: prompt is {len(prompt_ids)} tokens > max_prompt_length {opts.max_prompt_length} (on_too_long=drop)")
        saving = {i: budget[i] - opts.min_tool_tokens - (0 if budget[i] < length[i] else marker) for i in idx}
        candidates = [(budget[i], i) for i in idx if saving[i] > 0]
        if not candidates:
            raise PromptTooLongError(f"pair {pair.id}: prompt is {len(prompt_ids)} tokens > max_prompt_length {opts.max_prompt_length} "
                                     f"after truncating every tool output to >= {opts.min_tool_tokens} tokens")
        cur, i = max(candidates)
        want = excess + (0 if budget[i] < length[i] else marker)
        budget[i] = max(opts.min_tool_tokens, cur - want)
        msgs[i].content, _ = head_tail_truncate(tokenizer, original[i], budget[i])
    raise PromptTooLongError(f"pair {pair.id}: tool-output truncation did not converge")


UNTERMINATED_TAIL_TOLERANCE = 4   # trailing trace tokens that may re-merge with the template's "\n</think>" (e.g. "." + "\n")


def unterminated_completion_ids(pair: PreferencePair, tokenizer: Any, prompt_ids: list[int], prompt_tpl: list[dict[str, Any]],
                                opts: PairOptions) -> list[int]:
    """The completion of an unterminated rejected turn: its trace tokens exactly as the model produced them after the
    prompt's `<think>\\n`, with no `</think>`, no content and no `<|im_end|>` appended (appending them would teach the policy
    to close a thought that never closed). The trace is trimmed as the template trims it. The boundary is checked against
    the template's own render of the same trace as a closed reasoning-only turn: the prompt is a token prefix and every
    trace token matches except at most the last UNTERMINATED_TAIL_TOLERANCE, which may merge with the following newline."""
    trace = (pair.rejected.reasoning_content or "").strip()
    ids = tokenizer.encode(trace, add_special_tokens=False)
    closed = dataclasses.replace(pair.rejected, reasoning_content=trace, content="", tool_calls=None)
    full = encode_messages(tokenizer, prompt_tpl + [closed.to_template()], pair.tools, enable_thinking=opts.thinking)
    head = full[len(prompt_ids):len(prompt_ids) + len(ids)]
    agree = next((j for j, (a, b) in enumerate(zip(ids, head, strict=False)) if a != b), min(len(ids), len(head)))
    if full[:len(prompt_ids)] != prompt_ids or agree < len(ids) - UNTERMINATED_TAIL_TOLERANCE or tokenizer.decode(ids) != trace:
        raise PrefixViolationError(f"pair {pair.id}: the unterminated rejected trace does not continue the generation prompt as the "
                                   f"template renders it ({agree} of {len(ids)} trace tokens agree)")
    return ids


def render_pair(pair: PreferencePair, tokenizer: Any, opts: PairOptions) -> dict[str, Any]:
    """One pair -> {prompt_ids, chosen_ids, rejected_ids, pair_id, n_prompt, n_chosen, n_rejected, rejected_unterminated,
    truncated_tool_outputs, truncation_tokens_removed}. prompt_ids is the inference prompt (generation prompt appended;
    thinking on: it ends with `<|im_start|>assistant\\n<think>\\n`), each completion is the template's render of prompt +
    [turn] after that prefix up to and including <|im_end|>; an unterminated rejected turn is its trace tokens only
    (unterminated_completion_ids). The prefix property is asserted, identical completions are rejected, and pairs whose
    prompt + longer completion exceed opts.max_length raise RowTooLongError (completions are never cut)."""
    msgs, prompt_ids, n_trunc, removed = fit_prompt(pair, tokenizer, opts)
    prompt_tpl = [m.to_template() for m in msgs]
    completions: dict[str, list[int]] = {}
    for name, turn in (("chosen", pair.chosen), ("rejected", pair.rejected)):
        if name == "rejected" and pair.rejected_unterminated:
            completions[name] = unterminated_completion_ids(pair, tokenizer, prompt_ids, prompt_tpl, opts)
            continue
        full = encode_messages(tokenizer, prompt_tpl + [turn.to_template()], pair.tools, enable_thinking=opts.thinking)
        if full[:len(prompt_ids)] != prompt_ids:
            k = next((j for j, (a, b) in enumerate(zip(prompt_ids, full, strict=False)) if a != b), min(len(prompt_ids), len(full)))
            raise PrefixViolationError(
                f"pair {pair.id}: the generation prompt is not a token prefix of render(prompt + [{name}]) (first mismatch at token {k}: "
                f"...{tokenizer.decode(prompt_ids[max(0, k - 6):k + 3])!r} vs ...{tokenizer.decode(full[max(0, k - 6):k + 3])!r}); in thinking "
                "mode every candidate turn needs reasoning_content so the completion starts right after the prompt's <think> opener")
        ids = full[len(prompt_ids):completion_end(tokenizer, full)]
        if not ids or ids[-1] != IM_END_ID:
            raise RenderError(f"pair {pair.id}: {name} completion is empty or does not end with <|im_end|>")
        completions[name] = ids
    if completions["chosen"] == completions["rejected"]:
        raise IdenticalCompletionsError(f"pair {pair.id}: chosen and rejected render identically")
    longest = max(len(completions["chosen"]), len(completions["rejected"]))
    if len(prompt_ids) + longest > opts.max_length:
        raise RowTooLongError(f"pair {pair.id}: prompt {len(prompt_ids)} + longest completion {longest} tokens > max_length {opts.max_length} "
                              "(completions are never truncated)", n_tokens=len(prompt_ids) + longest)
    return {"prompt_ids": prompt_ids, "chosen_ids": completions["chosen"], "rejected_ids": completions["rejected"], "pair_id": pair.id,
            "n_prompt": len(prompt_ids), "n_chosen": len(completions["chosen"]), "n_rejected": len(completions["rejected"]),
            "rejected_unterminated": pair.rejected_unterminated, "truncated_tool_outputs": n_trunc, "truncation_tokens_removed": removed}


@dataclass
class DropRecord:
    pair_id: str
    reason: str
    detail: str


def build_pair_rows(path: Path, tokenizer: Any, opts: PairOptions) -> tuple[list[dict[str, Any]], list[DropRecord], int]:
    """Pairs JSONL -> (rows, dropped, n_pairs_read). Every failure becomes a DropRecord with its reason."""
    rows: list[dict[str, Any]] = []
    dropped: list[DropRecord] = []
    n_read = 0
    for line_no, obj in iter_jsonl(path):
        n_read += 1
        pid = str(obj.get("id", f"line{line_no}")) if isinstance(obj, dict) else f"line{line_no}"
        try:
            rows.append(render_pair(PreferencePair.from_dict(obj, default_id=f"line{line_no}", thinking=opts.thinking), tokenizer, opts))
        except DataFormatError as exc:
            dropped.append(DropRecord(pid, exc.reason, str(exc)))
    return rows, dropped, n_read


def pair_summary(rows: Sequence[dict[str, Any]], dropped: Sequence[DropRecord], n_read: int) -> dict[str, Any]:
    n_prompt = [r["n_prompt"] for r in rows]
    total = [r["n_prompt"] + max(r["n_chosen"], r["n_rejected"]) for r in rows]
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else 0.0  # noqa: E731
    return {"n_pairs_read": n_read, "n_rows": len(rows), "n_dropped": len(dropped),
            "dropped_by_reason": dict(sorted(Counter(d.reason for d in dropped).items())),
            "prompt_tokens": {"max": max(n_prompt, default=0), "mean": mean(n_prompt), "histogram": histogram(n_prompt)},
            "chosen_tokens": {"max": max((r["n_chosen"] for r in rows), default=0), "mean": mean([r["n_chosen"] for r in rows])},
            "rejected_tokens": {"max": max((r["n_rejected"] for r in rows), default=0), "mean": mean([r["n_rejected"] for r in rows])},
            "sequence_tokens": {"max": max(total, default=0), "histogram": histogram(total)},
            "rejected_unterminated": sum(1 for r in rows if r["rejected_unterminated"]),
            "truncated_prompts": sum(1 for r in rows if r["truncated_tool_outputs"]),
            "truncation_tokens_removed": sum(r["truncation_tokens_removed"] for r in rows)}


def pairs_to_dataset(rows: Sequence[dict[str, Any]]) -> Any:
    """datasets.Dataset with exactly the three id columns TRL's DataCollatorForPreference consumes."""
    from datasets import Dataset, Features, List, Value

    feats = Features({k: List(Value("int32")) for k in ("prompt_ids", "chosen_ids", "rejected_ids")})
    return Dataset.from_list([{k: r[k] for k in feats} for r in rows], features=feats)


def boundary_audit(tokenizer: Any, rows: Sequence[dict[str, Any]], n: int = 2) -> list[dict[str, Any]]:
    """Decoded prompt tail / completion heads of the first rows (metadata evidence that the boundary sits right after
    the generation prompt's `<think>\\n` opener)."""
    return [{"pair_id": r["pair_id"], "prompt_tail": tokenizer.decode(r["prompt_ids"][-6:]), "chosen_head": tokenizer.decode(r["chosen_ids"][:10]),
             "rejected_head": tokenizer.decode(r["rejected_ids"][:10]), "chosen_tail": tokenizer.decode(r["chosen_ids"][-3:]),
             "n_prompt": r["n_prompt"], "n_chosen": r["n_chosen"], "n_rejected": r["n_rejected"]} for r in rows[:n]]


# ----------------------------------------------------------------------------------------------- TRL trainers
def pretokenized_dpo_trainer_class() -> type:
    """DPOTrainer subclass that accepts rows already holding prompt_ids / chosen_ids / rejected_ids. TRL 1.15.0's
    `_prepare_dataset` assumes conversational or plain-text rows, so a pre-tokenised dataset must bypass it;
    `DataCollatorForPreference` and the signature columns already work on the id columns."""
    from trl import DPOTrainer

    class PreTokenizedDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            if {"prompt_ids", "chosen_ids", "rejected_ids"} <= set(dataset.column_names):
                return dataset
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    return PreTokenizedDPOTrainer



def write_drops(run: TrainRun, dropped: Sequence[Any]) -> None:
    if not dropped:
        return
    path = run.run_dir / "data_drops.jsonl"
    with path.open("w") as f:
        for d in dropped:
            f.write(json.dumps(dataclasses.asdict(d)) + "\n")
    run.warn(f"{len(dropped)} input records were not trained on; reasons in {path}")



# ------------------------------------------------------------------------------------------ identity and init adapter
def base_identity(cfg: dict[str, Any], base_dir: Path, *, method: str, smoke: bool) -> dict[str, Any]:
    """The base the trained adapter names in adapter_config.json: the locked hub id and revision for filtered_sft on the
    raw base; the config identity of a merged SFT checkpoint for dpo; the twin's config identity for smoke runs."""
    if smoke:
        return {"repo": None, "revision": twin_revision(base_dir)}
    if method == "dpo":
        if not (base_dir / "merge_provenance.json").is_file():
            raise ConfigError(f"dpo trains a fresh adapter on a merged SFT checkpoint (scripts/merge_adapter.sh merge output, with "
                              f"merge_provenance.json); {base_dir} is not one")
        return {"repo": None, "revision": "merged-config-sha256-" + sha256_file(base_dir / "config.json")[:12]}
    lock = model_lock()
    repo, revision = get_dotted(cfg, "model.repo"), str(get_dotted(cfg, "model.revision"))
    if (repo, revision) != (lock["repo"], lock["revision"]):
        raise ConfigError(f"model.repo/revision {repo}@{revision} != the lock {lock['repo']}@{lock['revision']} (configs/eval/model.yaml)")
    return {"repo": repo, "revision": revision}


def adapter_param_count(weights: Path) -> tuple[int, int]:
    """(tensors, parameters) of an adapter_model.safetensors, read from the header only."""
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt") as f:
        shapes = [f.get_slice(k).get_shape() for k in f.keys()]
    return len(shapes), sum(math.prod(s) for s in shapes)


def inspect_init_adapter(adapter_dir: Path, identity: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    """Check the SFT adapter before anything is loaded: a LoRA adapter over the same base (revision, and hub id unless the
    smoke twin), the `lora` block's r / alpha / targets, no extra trainable modules. Returns its record for metadata."""
    cfg_path, weights = adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"
    if not cfg_path.is_file() or not weights.is_file():
        raise ConfigError(f"model.init_adapter {adapter_dir}: no adapter_config.json + adapter_model.safetensors; pass the SFT run's "
                          "checkpoints/<run_id>/adapter directory (smoke: the adapter of a `scripts/train_sft.sh --smoke` run)")
    ac = json.loads(cfg_path.read_text())
    problems = []
    if ac.get("peft_type") != "LORA":
        problems.append(f"peft_type {ac.get('peft_type')!r} != 'LORA'")
    if ac.get("revision") != identity["revision"]:
        problems.append(f"revision {ac.get('revision')!r} != {identity['revision']!r} (a trainer checkpoint-N adapter records none: "
                        "run `scripts/merge_adapter.sh verify-adapter --adapter <dir> --fix-adapter-config` on it first)")
    if identity["repo"] is not None and ac.get("base_model_name_or_path") != identity["repo"]:
        problems.append(f"base_model_name_or_path {ac.get('base_model_name_or_path')!r} != {identity['repo']!r}")
    if (ac.get("r"), ac.get("lora_alpha")) != (lora.get("r"), lora.get("alpha")):
        problems.append(f"r/alpha {ac.get('r')}/{ac.get('lora_alpha')} != lora block {lora.get('r')}/{lora.get('alpha')}")
    if sorted(ac.get("target_modules") or []) != sorted(lora.get("target_modules") or []):
        problems.append(f"target_modules {sorted(ac.get('target_modules') or [])} != lora block {sorted(lora.get('target_modules') or [])}")
    if ac.get("modules_to_save") or ac.get("target_parameters") or ac.get("trainable_token_indices"):
        problems.append("modules_to_save / target_parameters / trainable_token_indices are set (the recipe trains LoRA factors only)")
    if problems:
        raise ConfigError(f"init adapter {adapter_dir}: " + "; ".join(problems))
    n_tensors, n_params = adapter_param_count(weights)
    sidecar = adapter_dir / "base_model.json"
    return {"path": str(adapter_dir.resolve()), "adapter_config_sha256": sha256_file(cfg_path), "adapter_model_sha256": sha256_file(weights),
            "n_tensors": n_tensors, "n_params": n_params, "r": ac["r"], "lora_alpha": ac["lora_alpha"],
            "base_model_name_or_path": ac.get("base_model_name_or_path"), "revision": ac.get("revision"),
            "sft_run_id": json.loads(sidecar.read_text()).get("run_id") if sidecar.is_file() else None}


def attach_init_adapter(model: Any, adapter_dir: Path) -> tuple[Any, dict[str, Any]]:
    """PeftModel.from_pretrained(model, adapter_dir, is_trainable=True), then prove the continuation: the model's adapter
    state dict has exactly the file's keys and every tensor equals the saved one (an fp32 copy of a bf16 file compares
    after casting back). Returns (peft model, check record)."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(peft_model)
    missing, unexpected = sorted(set(saved) - set(loaded)), sorted(set(loaded) - set(saved))
    if missing or unexpected:
        raise RuntimeError(f"init adapter keys do not match the model: missing {missing[:3]} ({len(missing)}), unexpected {unexpected[:3]} "
                           f"({len(unexpected)})")
    differ = [k for k, t in saved.items() if not torch.equal(loaded[k].detach().to("cpu", t.dtype), t)]
    if differ:
        raise RuntimeError(f"{len(differ)} adapter tensors differ from {adapter_dir} after loading (first {differ[0]}): not a continuation")
    return peft_model, {"tensors_checked": len(saved), "params_checked": sum(t.numel() for t in saved.values()), "identical": True,
                        "file_dtypes": sorted({str(t.dtype) for t in saved.values()}),
                        "trainable_dtypes": sorted({str(p.dtype) for p in peft_model.parameters() if p.requires_grad})}


# ----------------------------------------------------------------------------------------------------- run context
@dataclass
class RunContext:
    """Everything a method runner needs besides the config."""

    run: TrainRun
    cfg: dict[str, Any]
    method: str
    arm: str | None
    device: str
    base_dir: Path
    identity: dict[str, Any]
    init_adapter: Path | None
    init_record: dict[str, Any] | None
    auto_class: str
    arch: str
    curves_dir: Path
    out_dir: Path
    seed: int
    smoke: bool
    match_steps_of: Path | None
    monitor: TrainingMonitor


def _load_policy(ctx: RunContext) -> tuple[Any, dict[str, Any], Any, dict[str, Any] | None]:
    """The base (served class, bf16 on GPU, fp32 on the CPU twin) plus either the init adapter, attached trainable
    (filtered_sft), or a fresh LoRA config for the trainer to apply (dpo). Returns (model, load info, peft_config or
    None, attach check or None)."""
    model_cfg = ctx.cfg["model"]
    dtype_name = str(model_cfg["dtype"]) if ctx.device == "cuda" else "float32"
    model, load_info = load_base_model(ctx.run, ctx.base_dir, auto_class=ctx.auto_class, dtype_name=dtype_name, device=ctx.device,
                                       attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")))
    if ctx.init_adapter is None:
        return model, load_info, build_lora_config(ctx.cfg["lora"]), None
    model, attach = attach_init_adapter(model, ctx.init_adapter)
    ctx.run.log(f"init adapter attached trainable: {ctx.init_adapter} ({attach['tensors_checked']} tensors, {attach['params_checked']:,d} params "
                f"identical to the file; trainable dtypes {attach['trainable_dtypes']})")
    return model, load_info, None, attach


def _checkpointing(ctx: RunContext, planned_steps: int) -> tuple[dict[str, Any], list[int]]:
    """Checkpoint block for shared_trainer_kwargs: save_steps = ceil(fraction x planned steps), so with fraction 0.5 the
    trainer saves at ~50 percent and at the last step (100 percent; transformers always saves the final step)."""
    fraction = float(require_type(ctx.cfg, "checkpointing.fraction", (int, float)))
    if not 0.0 < fraction <= 1.0:
        raise ConfigError("checkpointing.fraction must be in (0, 1]")
    save_steps = max(1, math.ceil(fraction * planned_steps))
    steps = sorted({save_steps * k for k in range(1, planned_steps // save_steps + 1)} | {planned_steps})
    return {"save_strategy": "steps", "save_steps": save_steps, "save_total_limit": get_dotted(ctx.cfg, "checkpointing.save_total_limit", None)}, steps


def _after_init(ctx: RunContext, trainer: Any, load_info: dict[str, Any], kwargs: dict[str, Any], **extra: Any) -> dict[str, Any]:
    layout = adapter_layout(trainer.model)
    assert_adapter_layout(layout, trainer, expected_trainable=get_dotted(ctx.cfg, "lora.expected_trainable_params", None),
                          served_class=ctx.arch == SERVED_ARCHITECTURE)
    if ctx.init_record is not None and layout["trainable"] != ctx.init_record["n_params"]:
        raise RuntimeError(f"trainable parameters {layout['trainable']:,d} != the SFT adapter's {ctx.init_record['n_params']:,d}")
    ctx.run.log(f"LoRA: {layout['trainable']:,d} trainable of {layout['total']:,d} ({layout['trainable_pct']} %), {layout['n_lora_modules']} modules, "
                f"first key {layout['first_adapter_key']}, vision tower frozen")
    recorded = {k: kwargs[k] for k in ("max_steps", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
                                       "lr_scheduler_type", "warmup_steps", "weight_decay", "max_grad_norm", "optim", "save_strategy", "save_steps",
                                       "bf16", "use_cpu", "gradient_checkpointing", "max_length")}
    adapter_cfg = trainer.model.peft_config["default"]
    ctx.run.write_metadata(model_load=load_info, lora_config=adapter_cfg.to_dict(), trainable=layout, trainer_args={**recorded, **extra})
    return layout


def _finish_method(ctx: RunContext, trainer: Any, metrics: TrainMetrics, required: Sequence[str], extra_keys: Sequence[str], train_s: float,
                   n_inputs: int, data_version: dict[str, Any] | None) -> dict[str, Any]:
    """Save the adapter (lineage in base_model.json), check the logged metrics, build the method summary."""
    lineage = {"stage": "stage2", "method": ctx.method, "arm": ctx.arm, "init_adapter": ctx.init_record}
    sidecar = save_adapter(trainer, ctx.out_dir / "adapter", ctx.run, base_repo=ctx.identity["repo"], base_revision=ctx.identity["revision"],
                           base_path=ctx.base_dir, lora_block=ctx.cfg["lora"], data_version=data_version, extra=lineage)
    step_summary = metrics.summary((*required, *extra_keys))
    missing = [k for k in required if k not in step_summary]
    if missing:
        raise RuntimeError(f"required {ctx.method} metrics were not logged: {missing}")
    if not all(v["all_finite"] for v in step_summary.values()):
        raise RuntimeError(f"non-finite training metrics: {step_summary}")
    trained_tokens = data_version["total"]["n_assistant_tokens"] if data_version else None
    return {"method": ctx.method, "arm": ctx.arm, "n_inputs": n_inputs, "trained_tokens": trained_tokens, "global_step": trainer.state.global_step,
            "train_seconds": round(train_s, 1), "metrics": step_summary, "checkpoints": metrics.checkpoints, "adapter": sidecar,
            "curves_csv": str(metrics.train_csv.path)}


# --------------------------------------------------------------------------------------------------- filtered_sft
def load_monitor_eval(ctx: RunContext, max_length: int, planned_steps: int) -> tuple[Any, dict[str, Any] | None, int | None]:
    """monitor.eval_rows (opt-in): the dev rows evaluated during stage 2, the task-stratified subset, and the eval interval
    ceil(eval_every_fraction x planned steps). Returns (dataset or None, data-version block or None, eval_steps or None)."""
    mcfg = ctx.monitor.cfg
    if not ctx.monitor.enabled or not mcfg.eval_rows:
        return None, None, None
    paths = rows_paths(mcfg.eval_rows)
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"monitor.eval_rows not found: {absent}")
    ds, counts = load_split(paths)
    desc = describe_split(ds, paths, counts)
    check_split(desc, max_length, "monitor eval")
    ds = ctx.monitor.select_eval_rows(ds)
    desc["eval_subset"] = ctx.monitor.eval_subset
    return ds, desc, max(1, math.ceil(mcfg.eval_every_fraction * planned_steps))


def load_arm_rows(ctx: RunContext, arm_paths: list[Path], replay_paths: list[Path], max_length: int) -> tuple[Any, dict[str, Any]]:
    """The arm's rows followed by the replay rows (the trainer's seeded sampler mixes them). Both must exist, fit the row
    cap and train >= 1 token per row. Returns (dataset, data version with rows and loss tokens per part)."""
    from datasets import concatenate_datasets

    parts, version = [], {}
    for name, paths in (("arm", arm_paths), ("replay", replay_paths)):
        if not paths:
            raise ConfigError(f"stage 2 needs {'the arm rows' if name == 'arm' else 'the replay rows (filtered_sft.replay_rows)'}")
        absent = [str(p) for p in paths if not p.is_file()]
        if absent:
            raise FileNotFoundError(f"{name} rows not found: {absent}")
        ds, counts = load_split(paths)
        version[name] = describe_split(ds, paths, counts)
        check_split(version[name], max_length, f"{name} ({ctx.arm})" if name == "arm" else name)
        parts.append(ds)
    arm_tokens, replay_tokens = version["arm"]["n_assistant_tokens"], version["replay"]["n_assistant_tokens"]
    version["total"] = {"n_rows": sum(len(p) for p in parts), "n_tokens": version["arm"]["n_tokens"] + version["replay"]["n_tokens"],
                        "n_assistant_tokens": arm_tokens + replay_tokens,
                        "replay_loss_token_share": round(replay_tokens / max(arm_tokens + replay_tokens, 1), 4)}
    return concatenate_datasets(parts), version


def match_steps(ctx: RunContext, data_version: dict[str, Any]) -> dict[str, Any]:
    """--match-steps-of: the reference arm's planned optimizer steps; asserts the same init adapter and replay rows."""
    meta_path = ctx.match_steps_of / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"--match-steps-of {ctx.match_steps_of}: no metadata.json")
    ref = json.loads(meta_path.read_text())
    problems = []
    if ref.get("method") != "filtered_sft" or not ref.get("planned"):
        problems.append("not a filtered_sft run with a planned block")
    if (ref.get("init_adapter") or {}).get("adapter_model_sha256") != ctx.init_record["adapter_model_sha256"]:
        problems.append("different init adapter")
    if [f["sha256"] for f in (ref.get("data_version") or {}).get("replay", {}).get("files", [])] != [f["sha256"] for f in data_version["replay"]["files"]]:
        problems.append("different replay rows")
    if ref.get("arm") == ctx.arm:
        problems.append(f"same arm {ctx.arm!r}")
    if problems:
        raise RuntimeError(f"--match-steps-of {ctx.match_steps_of}: " + "; ".join(problems) + " (the arms must differ only in their rows)")
    return {"run_id": ref.get("run_id"), "arm": ref.get("arm"), "optimizer_steps": int(ref["planned"]["optimizer_steps"]),
            "arm_loss_tokens": (ref.get("data_version") or {}).get("arm", {}).get("n_assistant_tokens")}


def run_filtered_sft(ctx: RunContext) -> tuple[dict[str, Any], TrainMetrics]:
    """Continue the SFT adapter on the arm's rows + the replay rows (one epoch, or the matched arm's steps)."""
    import torch
    from trl import SFTConfig, SFTTrainer

    scfg = ctx.cfg["filtered_sft"]
    max_length = int(require_type(scfg, "max_length", int))
    if ctx.smoke:
        arm_paths, replay_paths = smoke_rows(ctx)
    else:
        arm_paths = rows_paths(get_dotted(scfg, f"arms.{ctx.arm}.rows"))
        replay_paths = rows_paths(get_dotted(scfg, "replay_rows", None))
    ds, data_version = load_arm_rows(ctx, arm_paths, replay_paths, max_length)
    matched = None
    if ctx.match_steps_of is not None:
        matched = match_steps(ctx, data_version)
        scfg["max_steps"] = matched["optimizer_steps"]
    planned = planned_optimizer_steps(len(ds), scfg)
    ckpt, checkpoint_steps = _checkpointing(ctx, planned["optimizer_steps"])
    manifest_path = get_dotted(scfg, "manifest", None)
    manifest = manifest_record(resolve_path(str(manifest_path)) if manifest_path else None,
                               [f for part in ("arm", "replay") for f in data_version[part]["files"]])
    if manifest is None and manifest_path:
        ctx.run.warn(f"no stage-2 set manifest at {manifest_path}: metadata.json records no frozen-set block")
    elif manifest is not None and not all(manifest["rows_sha256_listed"].values()):
        ctx.run.warn(f"rows files not listed by sha256 in {manifest['path']}: "
                     f"{[p for p, ok in manifest['rows_sha256_listed'].items() if not ok]} (the rows on disk may not be the frozen set)")
    eval_ds, eval_version, eval_steps = load_monitor_eval(ctx, max_length, planned["optimizer_steps"])
    ctx.monitor.set_plan(planned_steps=planned["optimizer_steps"], max_length=max_length, eval_every=eval_steps)
    ctx.run.write_metadata(config=ctx.cfg, row_cap=max_length, data_version=data_version, planned=planned, checkpoint_steps=checkpoint_steps,
                           steps_matched_to=matched, stage2_manifest=manifest, monitor_eval=eval_version)
    total = data_version["total"]
    ctx.run.log(f"{ctx.arm}: arm rows {data_version['arm']['n_rows']} ({data_version['arm']['n_assistant_tokens']:,d} loss tokens) + replay rows "
                f"{data_version['replay']['n_rows']} ({data_version['replay']['n_assistant_tokens']:,d}; share {total['replay_loss_token_share']}); "
                f"planned optimizer steps {planned['optimizer_steps']} ({planned['source']}, {planned['epochs']} epochs), checkpoints at "
                f"{checkpoint_steps}; max row {max(data_version['arm']['max_tokens'], data_version['replay']['max_tokens'])} <= cap {max_length}")
    tokenizer = load_tokenizer(ctx.base_dir)
    model, load_info, _, attach = _load_policy(ctx)
    kwargs = shared_trainer_kwargs(scfg, ctx.cfg["model"], ckpt, out_dir=ctx.out_dir / "trainer", device=ctx.device, seed=ctx.seed,
                                   data_seed=ctx.seed, logging_steps=int(get_dotted(ctx.cfg, "run.logging_steps", 1)), max_length=max_length)
    eval_kwargs = ({"eval_strategy": "steps", "eval_steps": eval_steps, "per_device_eval_batch_size": 1, "prediction_loss_only": True,
                    "eval_on_start": ctx.monitor.cfg.eval_on_start} if eval_ds is not None else {})
    args = SFTConfig(**kwargs, packing=False, assistant_only_loss=False, completion_only_loss=None, **eval_kwargs)
    metrics = TrainMetrics(ctx.run, ctx.curves_dir, device=ctx.device, extra_columns=SFT_CURVE_EXTRA, monitor=ctx.monitor)
    trainer = ctx.monitor.trainer_class(SFTTrainer)(model=model, args=args, train_dataset=ds, eval_dataset=eval_ds, processing_class=tokenizer,
                                                    callbacks=[metrics.callback(ctx.out_dir / "trainer")])
    loss_norm = apply_loss_normalisation(trainer, loss_normalisation_of(scfg, "filtered_sft"))
    ctx.run.log(f"loss normalisation {loss_norm['mode']}: {loss_norm['definition']}")
    ctx.run.write_metadata(init_adapter_check=attach, loss_normalisation=loss_norm)
    _after_init(ctx, trainer, load_info, kwargs, packing=False, assistant_only_loss=False, eval_steps=eval_steps)
    label_check = check_prepared_labels(trainer.train_dataset, ds)
    by_hash = label_check.pop("by_hash")
    ctx.monitor.register_rows("train", ds, label_check.pop("index_by_hash"))
    if eval_ds is not None:
        eval_check = check_prepared_labels(trainer.eval_dataset, eval_ds)
        by_hash = {**by_hash, **eval_check.pop("by_hash")}   # the first collated batch may be an eval batch (eval_on_start)
        ctx.monitor.register_rows("eval", eval_ds, eval_check.pop("index_by_hash"))
        label_check["eval"] = eval_check
    first_batch: dict[str, Any] = {}

    def on_first_batch(result: dict[str, Any]) -> None:
        first_batch.update(result)
        ctx.run.log(f"self-check first collated batch: {result}")
        ctx.run.metric({"event": "self_check_first_batch", **result})

    trainer.data_collator = FirstBatchLabelCheck(trainer.data_collator, by_hash, on_first_batch)
    ctx.monitor.attach(trainer)
    ctx.run.write_metadata(self_check_labels={**label_check, "ok": True})
    ctx.run.log(f"self-check labels == assistant_masks: {label_check['rows_checked']} rows, {label_check['trained_tokens']:,d} trained tokens: OK; "
                f"training {planned['optimizer_steps']} steps at LR {args.learning_rate}")
    t1 = time.time()
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM: {str(exc)[:300]}") from exc
    if not first_batch:
        raise AssertionError("the first-batch label check never ran (no batch was collated)")
    if ctx.monitor.stop_reason:
        stop_monitored_run(ctx.run, ctx.monitor, metrics, device=ctx.device, global_step=trainer.state.global_step,
                           trained_tokens=data_version["total"]["n_assistant_tokens"])
    summary = _finish_method(ctx, trainer, metrics, REQUIRED_SFT_METRICS, SFT_CURVE_EXTRA, time.time() - t1, len(ds), data_version)
    if trainer.state.global_step != planned["optimizer_steps"]:
        raise RuntimeError(f"trained {trainer.state.global_step} optimizer steps, planned {planned['optimizer_steps']}")
    return summary, metrics


# ------------------------------------------------------------------------------------------------------------ dpo
def dpo_losses(dcfg: dict[str, Any]) -> tuple[list[str], list[float] | None]:
    """dpo.loss_type (a name or a list of names from LOSS_TYPES) and dpo.loss_weights (null, or one positive weight per
    name). ["sigmoid", "sft"] with [1.0, 0.2] is DPO plus an NLL term on the chosen turn (RPO; Llama 3's 0.2), the
    replacement for TRL's removed rpo_alpha; ["sft"] alone is SFT on the chosen turns (the stage-2 control arm)."""
    raw = dcfg.get("loss_type", "sigmoid")
    names = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else None
    if not names or any(n not in LOSS_TYPES for n in names) or len(set(names)) != len(names):
        raise ConfigError(f"dpo.loss_type must be one of {LOSS_TYPES} or a list of distinct ones, got {raw!r}")
    weights = dcfg.get("loss_weights")
    if weights is None:
        return names, None
    if (not isinstance(weights, (list, tuple)) or len(weights) != len(names)
            or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w > 0 for w in weights)):
        raise ConfigError(f"dpo.loss_weights must be null or {len(names)} positive finite numbers (one per dpo.loss_type), got {weights!r}")
    return names, [float(w) for w in weights]


def run_dpo(ctx: RunContext) -> tuple[dict[str, Any], TrainMetrics]:
    """Revision 5 stage 2: pre-rendered turn-level preference pairs -> PreTokenizedDPOTrainer with a fresh LoRA, ref_model=None."""
    from trl import DPOConfig

    dcfg = ctx.cfg["dpo"]
    loss_type, loss_weights = dpo_losses(dcfg)
    opts = PairOptions(thinking=bool(get_dotted(dcfg, "data.thinking", True)), max_length=int(require_type(dcfg, "max_length", int)),
                       max_prompt_length=dcfg.get("max_prompt_length"), on_too_long=str(dcfg.get("on_too_long", "drop")),
                       min_tool_tokens=int(dcfg.get("min_tool_tokens", 64)))
    pairs_path = smoke_pairs(ctx) if ctx.smoke else resolve_path(str(get_dotted(dcfg, "data.pairs")))
    if not pairs_path.is_file():
        raise FileNotFoundError(f"dpo.data.pairs: {pairs_path} does not exist")
    tokenizer = load_tokenizer(ctx.base_dir)
    t0 = time.time()
    rows, dropped, n_read = build_pair_rows(pairs_path, tokenizer, opts)
    summary = pair_summary(rows, dropped, n_read)
    ctx.run.log(f"pairs: {n_read} read -> {len(rows)} rows, {len(dropped)} dropped {summary['dropped_by_reason']} in {time.time() - t0:.1f} s; "
                f"prompt max {summary['prompt_tokens']['max']}, sequence max {summary['sequence_tokens']['max']}")
    write_drops(ctx.run, dropped)
    if not rows:
        raise RuntimeError("no preference pair survived validation and rendering; see data_drops.jsonl")
    planned = planned_optimizer_steps(len(rows), dcfg)
    ckpt, checkpoint_steps = _checkpointing(ctx, planned["optimizer_steps"])
    ctx.run.write_metadata(data={"pairs": str(pairs_path), "sha256": sha256_file(pairs_path)}, pair_options=dataclasses.asdict(opts),
                           data_summary=summary, boundary_audit=boundary_audit(tokenizer, rows), planned=planned, checkpoint_steps=checkpoint_steps)
    dataset = pairs_to_dataset(rows)

    model, load_info, lora, _ = _load_policy(ctx)
    kwargs = shared_trainer_kwargs(dcfg, ctx.cfg["model"], ckpt, out_dir=ctx.out_dir / "trainer", device=ctx.device, seed=ctx.seed,
                                   data_seed=ctx.seed, logging_steps=int(get_dotted(ctx.cfg, "run.logging_steps", 1)), max_length=opts.max_length)
    args = DPOConfig(**kwargs, beta=float(require_type(dcfg, "beta", (int, float))), loss_type=loss_type, loss_weights=loss_weights,
                     precompute_ref_log_probs=bool(dcfg.get("precompute_ref_log_probs", False)), disable_dropout=bool(dcfg.get("disable_dropout", True)))
    metrics = TrainMetrics(ctx.run, ctx.curves_dir, device=ctx.device, extra_columns=DPO_CURVE_EXTRA, monitor=ctx.monitor)
    ctx.monitor.set_plan(planned_steps=planned["optimizer_steps"], max_length=opts.max_length)
    trainer = pretokenized_dpo_trainer_class()(model=model, ref_model=None, args=args, train_dataset=dataset, processing_class=tokenizer,
                                               peft_config=lora, callbacks=[metrics.callback(ctx.out_dir / "trainer")])
    if trainer.ref_model is not None:
        raise RuntimeError("DPOTrainer instantiated a separate reference model; expected ref_model=None with the adapter disabled as reference")
    if len(trainer.train_dataset) != len(rows) or "prompt_ids" not in trainer.train_dataset.column_names:
        raise RuntimeError("TRL re-processed the pre-tokenised pairs; the PreTokenizedDPOTrainer bypass did not apply")
    _after_init(ctx, trainer, load_info, kwargs, beta=args.beta, loss_type=args.loss_type, loss_weights=args.loss_weights,
                precompute_ref_log_probs=args.precompute_ref_log_probs,
                disable_dropout=args.disable_dropout, reference="adapter disabled on the policy (ref_model=None)")
    ctx.run.log(f"DPO: {len(rows)} pairs, beta {args.beta}, loss {args.loss_type} weights {args.loss_weights}, planned steps {planned['optimizer_steps']}, micro "
                f"{args.per_device_train_batch_size} x accum {args.gradient_accumulation_steps}")
    ctx.monitor.attach(trainer)   # step-level health only: DPO rows are pairs, no per-row records
    t1 = time.time()
    trainer.train()
    if ctx.monitor.stop_reason:
        stop_monitored_run(ctx.run, ctx.monitor, metrics, device=ctx.device, global_step=trainer.state.global_step, trained_tokens=None)
    return _finish_method(ctx, trainer, metrics, REQUIRED_DPO_METRICS, ("logps/chosen", "logps/rejected"), time.time() - t1, len(rows), None), metrics


# ------------------------------------------------------------------------------------------------ smoke test data
def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


_SMOKE_TOOLS = [
    _tool("run_tests", "Run the test suite (or a selection) and return the summary line.", {"selection": {"type": "string"}, "timeout_s": {"type": "integer", "default": 300}}, []),
    _tool("read_file", "Read a slice of a text file with line numbers.", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _tool("apply_patch", "Apply a unified diff to the repository.", {"diff": {"type": "string"}}, ["diff"]),
    _tool("bash", "Run a shell command in the sandbox.", {"command": {"type": "string"}}, ["command"]),
]


def _asst(reasoning: str, content: str = "", calls: Sequence[tuple[str, dict[str, Any]]] = ()) -> dict[str, Any]:
    m: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning, "content": content}
    if calls:
        m["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]
    return m


def synthetic_pairs(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic own-content preference pairs for the smoke test: coding-agent prompts with tools, every assistant
    turn carrying reasoning_content; half the prompts end with a tool result, half with the user request; chosen/rejected
    mix tool calls and plain answers with heterogeneous argument schemas (the case TRL's Arrow path cannot hold); every
    fourth rejected turn is a reasoning loop cut at the sampling cap (trace only: an unterminated rejected turn)."""
    rng = random.Random(seed)
    system = {"role": "system", "content": "You are a coding agent working in a sandboxed repository."}
    pairs = []
    for i in range(n):
        module, test = f"pkg/mod_{i}.py", f"tests/test_mod_{i}.py"
        user = {"role": "user", "content": f"{test} fails with an off-by-one error in {module}. Fix it and make the tests pass."}
        tools = _SMOKE_TOOLS if i % 2 == 0 else _SMOKE_TOOLS[:2] + _SMOKE_TOOLS[3:]
        if i % 2 == 0:
            prompt = [system, user,
                      _asst("Read the implementation before editing.", "Reading the module first.", [("read_file", {"path": module, "start_line": 1, "end_line": 30})]),
                      {"role": "tool", "name": "read_file", "content": f"1 def window(xs, k):\n2     return xs[0:k - 1]\n3 # {rng.randint(100, 999)}\n"}]
            chosen = _asst("The slice drops the last element; patch it and rerun the failing test before answering.", "",
                           [("apply_patch", {"diff": f"--- a/{module}\n+++ b/{module}\n@@ -2 +2 @@\n-    return xs[0:k - 1]\n+    return xs[0:k]\n"}),
                            ("run_tests", {"selection": test, "timeout_s": 120})])
            rejected = _asst("Probably fine; just claim success.", f"Fixed the off-by-one in {module}; the tests should pass now.")
        else:
            prompt = [system, user]
            chosen = _asst("Inspect the module and the failing test before changing anything.", "Let me look at the code.",
                           [("read_file", {"path": module}), ("bash", {"command": f"python -m pytest -q {test} -x"})])
            rejected = _asst("Rewrite the whole file from memory without looking.", "", [("bash", {"command": f"echo 'def window(xs, k): return xs[:k]' > {module}"})])
        if i % 4 == 3:   # a reasoning loop cut at the sampling cap: trace only, no content, no tool call (an unterminated rejected turn)
            rejected = _asst(" ".join(["Wait, maybe the slice is right after all. Let me reconsider the bounds again."] * (4 + i)), "")
        meta = {"id": f"syn-pair-{i:03d}", "source": "synthetic", "task_type": "code_fix", "reward": "execution_pass_minus_fail",
                "positive_source": "passing_rollout", "negative_source": "failing_rollout"}
        pairs.append({"id": meta["id"], "prompt": prompt, "tools": tools, "chosen": chosen, "rejected": rejected, "metadata": meta})
    return pairs



def smoke_rows(ctx: RunContext) -> tuple[list[Path], list[Path]]:
    """--smoke filtered_sft inputs under <run_dir>/smoke_data/: arm rows (12 synthetic samples -> 8 episode rows; the seed
    differs per arm) and replay rows (6 -> 4), built through the data_format CLI with the twin's tokenizer."""
    out = ctx.run.run_dir / "smoke_data"
    out.mkdir(exist_ok=True)
    arm_seed = sorted(ctx.cfg["filtered_sft"]["arms"]).index(ctx.arm)
    paths = []
    for name, n, seed in (("arm", SMOKE_ARM_SAMPLES, ctx.seed + arm_seed), ("replay", SMOKE_REPLAY_SAMPLES, ctx.seed + 100)):
        samples, rows = out / f"{name}_samples.jsonl", out / f"{name}_rows.parquet"
        run_cli(ctx.run, f"smoke data ({name} samples)", [sys.executable, "-m", "apex_train.data_format", "--write-synthetic", str(samples),
                                                          "--n", str(n), "--seed", str(seed)])
        run_cli(ctx.run, f"smoke data ({name} rows)", [sys.executable, "-m", "apex_train.data_format", "--in", str(samples), "--out", str(rows),
                                                       "--tokenizer", str(ctx.base_dir), "--mode", "episode", "--max-length",
                                                       str(get_dotted(ctx.cfg, "filtered_sft.max_length")), "--report", str(out / f"{name}_report.json")])
        paths.append([rows])
    return paths[0], paths[1]


def smoke_pairs(ctx: RunContext) -> Path:
    """--smoke dpo inputs: 8 synthetic preference pairs under <run_dir>/smoke_data/pairs.jsonl."""
    out = ctx.run.run_dir / "smoke_data"
    out.mkdir(exist_ok=True)
    path = out / "pairs.jsonl"
    with path.open("w") as f:
        for p in synthetic_pairs(8, ctx.seed):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    ctx.run.log(f"smoke: wrote 8 synthetic preference pairs -> {path}")
    return path


# ------------------------------------------------------------------------------------------------------------ CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m apex_train.stage2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_train_args(p, default_config=DEFAULT_CONFIG)
    p.add_argument("--method", choices=METHODS, default=None, help="override the config's `method` (default dpo; filtered_sft is the earlier revision-4.3 design)")
    p.add_argument("--arm", default=None, help="filtered_sft arm: positives (arm a) or control (arm b); required for filtered_sft")
    p.add_argument("--base", type=Path, default=None, help="override model.base (the raw base; dpo: a merged SFT checkpoint)")
    p.add_argument("--init-adapter", type=Path, default=None, help="override model.init_adapter (the SFT adapter directory to continue)")
    p.add_argument("--rows", type=Path, default=None, help="override filtered_sft.arms.<arm>.rows")
    p.add_argument("--replay-rows", type=Path, default=None, help="override filtered_sft.replay_rows (shared by both arms)")
    p.add_argument("--pairs", type=Path, default=None, help="dpo: override dpo.data.pairs")
    p.add_argument("--match-steps-of", type=Path, default=None, metavar="RUN_DIR",
                   help="filtered_sft: train the optimizer steps planned by that arm's run (logs/<run_id>); asserts the same init adapter and replay rows")
    p.add_argument("--max-steps", type=int, default=None, metavar="N", help="stop after N optimizer steps (<method>.max_steps)")
    p.add_argument("--smoke", action="store_true", help="tiny twin, synthetic inputs, CPU allowed (filtered_sft needs --init-adapter of an SFT smoke)")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], str, str | None, list[dict[str, Any]]]:
    """yaml -> (+smoke overrides) -> (+CLI path flags, --max-steps, --override) -> (config, method, arm, applied overrides)."""
    cfg = load_yaml(args.config)
    method = args.method or str(get_dotted(cfg, "method"))
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}, got {method!r}")
    cfg["method"] = method
    for section in ("run", "model", "lora", method, "checkpointing"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"config section '{section}' is missing or not a mapping")
    arm = args.arm
    if method == "filtered_sft":
        arms = get_dotted(cfg, "filtered_sft.arms", {}) or {}
        if arm not in arms:
            raise ConfigError(f"--arm must be one of {sorted(arms)} for filtered_sft (got {arm!r})")
    elif arm is not None or args.rows or args.replay_rows or args.match_steps_of:
        raise ConfigError("--arm, --rows, --replay-rows and --match-steps-of apply to filtered_sft only")
    if args.max_steps is not None and args.match_steps_of is not None:
        raise ConfigError("--max-steps and --match-steps-of are exclusive")
    applied: list[dict[str, Any]] = []
    if args.smoke:
        applied += apply_overrides(cfg, SMOKE_OVERRIDES[method], "smoke")
    cli: dict[str, Any] = {}
    for flag, key in ((args.base, "model.base"), (args.init_adapter, "model.init_adapter"), (args.rows, f"filtered_sft.arms.{arm}.rows"),
                      (args.replay_rows, "filtered_sft.replay_rows"), (args.pairs, "dpo.data.pairs")):
        if flag is not None:
            cli[key] = str(flag)
    if args.max_steps is not None:
        cli[f"{method}.max_steps"] = args.max_steps
    cli.update(parse_override(item) for item in args.override)
    applied += apply_overrides(cfg, cli, "cli")
    if method == "filtered_sft":
        loss_normalisation_of(cfg["filtered_sft"], "filtered_sft")
    MonitorConfig.from_block(cfg.get("monitor"))   # the monitor: block (unknown keys and wrong types are ConfigErrors)
    return cfg, method, arm, applied


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg, method, arm, overrides = resolve_config(args)
        device = pick_device(cpu_dry_run=args.smoke)
        base_dir = resolve_path(str(get_dotted(cfg, "model.base")))
        if args.smoke:
            ensure_tiny_twin(base_dir)   # the git-ignored twin is rebuilt when missing (make_tiny_cg_twin.py)
        auto_class, arch = resolve_loader(base_dir, allow_text_only_twin=args.smoke)
        identity = base_identity(cfg, base_dir, method=method, smoke=args.smoke)
        init_value = get_dotted(cfg, "model.init_adapter", None)
        init_adapter = resolve_path(str(init_value)) if init_value else None
        if method == "filtered_sft" and init_adapter is None:
            raise ConfigError("filtered_sft continues the SFT adapter: set model.init_adapter (APEX_SFT_ADAPTER_DIR) or pass --init-adapter")
        if method == "dpo" and init_adapter is not None:
            raise ConfigError("dpo trains a fresh adapter on a merged SFT checkpoint (its reference is the adapter-disabled policy): pass "
                              "--base <merged SFT dir> --override model.init_adapter=null")
        init_record = inspect_init_adapter(init_adapter, identity, cfg["lora"]) if init_adapter is not None else None
        seed = int(require_type(cfg, "run.seed", int))
        prefix = str(get_dotted(cfg, f"filtered_sft.arms.{arm}.run_id_prefix") if method == "filtered_sft" else get_dotted(cfg, "dpo.run_id_prefix"))
        run_id = args.run_id or new_run_id(*(["dryrun"] if args.smoke else []), prefix)
        root = Path(args.out_root) if args.out_root else REPO_ROOT
        out_dir = (root / "checkpoints" / "stage2" if args.out_root else resolve_path(str(get_dotted(cfg, "run.out_root")))) / run_id
        curves_dir = root / "curves" / "source" if args.out_root else resolve_path(str(get_dotted(cfg, "run.curves_dir")))
        if out_dir.exists():
            raise ConfigError(f"output directory {out_dir} already exists; never overwrite checkpoints")
    except (ConfigError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        with TrainRun(run_id, root) as run:
            if args.threads:
                import torch

                torch.set_num_threads(max(1, args.threads))
            seed_everything(seed)
            head = fused_head_status(cpu_dry_run=args.smoke and device == "cpu")
            run.snapshot_config(resolve_path(args.config))
            run.start_metadata(stage="stage2", method=method, arm=arm, config=cfg, config_path=str(args.config), overrides=overrides,
                               smoke=args.smoke, device=device, seed=seed, base_checkpoint=checkpoint_identity(base_dir), base_identity=identity,
                               auto_class=auto_class, init_adapter=init_record, fused_lm_head=head, out_dir=str(out_dir), note=args.note,
                               torch_env=torch_env())
            run.log(f"run {run_id}: {method} arm {arm}, device {device}, class {auto_class}, base {base_dir}, init adapter "
                    f"{init_record['path'] if init_record else None} ({init_record['n_params'] if init_record else 0:,d} params), fused head {head}")
            if head["shim_installed"]:
                run.warn("TRL fused LM head: pure-torch shim installed (CPU dry run only; the GPU venv uses the Triton kernel)")
            out_dir.mkdir(parents=True)
            monitor = TrainingMonitor(run, get_dotted(cfg, "monitor", None), stage="stage2", device=device, curves_dir=curves_dir, checkpoint_root=out_dir,
                                      smoke=args.smoke, row_tracking=method == "filtered_sft", method=method, arm=arm)
            run.attach_monitor(monitor)
            ctx = RunContext(run=run, cfg=cfg, method=method, arm=arm, device=device, base_dir=base_dir, identity=identity, init_adapter=init_adapter,
                             init_record=init_record, auto_class=auto_class, arch=arch, curves_dir=curves_dir, out_dir=out_dir, seed=seed,
                             smoke=args.smoke, match_steps_of=resolve_path(args.match_steps_of) if args.match_steps_of else None, monitor=monitor)
            result, metrics = run_filtered_sft(ctx) if method == "filtered_sft" else run_dpo(ctx)
            run.finish("finished", training_resource_rows(run, metrics, device=device, adapter_bytes=result["adapter"]["adapter_bytes"],
                                                          trained_tokens=result["trained_tokens"]) + monitor.resource_rows(), summary=result,
                       monitor=monitor.summary())
            run.log(f"finished: {result['global_step']} steps, adapter {result['adapter']['adapter_bytes'] / 1e6:.1f} MB, curves {result['curves_csv']}")
    except MonitorStop as exc:
        print(f"training stopped by the monitor: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
