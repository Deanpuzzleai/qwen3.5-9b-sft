"""Chat samples -> pre-tokenised trainer rows for Qwen/Qwen3.5-9B (plan-independent core of the data pipeline).

Design source: REPORT.md section 2 ("Masking", "Thinking traces", "Tool-call wire format") of the archived 35B training-stack
study (README, repository layout notes) and logs/train_stack_check_qwen35_9b/chat_template_format/FINDINGS.md sections 3 and 5,
re-validated for the Qwen3.5-9B template (sha256 a4aee8af...) in
logs/train_stack_check_qwen35_9b/chat_template_format/FINDINGS.md. That template differs from the previous one in two
lines only: it has no `preserve_thinking` variable (think blocks are kept only for assistant turns after the last user
query; nothing can switch that off), and tool-call argument values that are neither strings nor dicts/lists render
through `str()` (`True`/`False`/`None`) instead of JSON (`true`/`false`/`null`).

* `Sample`: OpenAI-style message list (roles system/user/assistant/tool). Assistant turns carry `content`,
  `reasoning_content` (mandatory and non-empty in thinking mode: the template renders an empty `<think>` block
  otherwise, which would train the model to skip thinking) and optional `tool_calls` as
  `{type: function, function: {name, arguments: dict}}` (JSON-string `arguments` are parsed on ingest; the template
  raises `TypeError` on strings). Tool schemas are normalised to vLLM's serialised shape
  `{type, function: {name, description, parameters}}`. Metadata is flat: id, source, task_type, language,
  difficulty, license.
* `render_and_mask`: `tokenizer.apply_chat_template(tokenize=True)` on the shipped template plus assistant-only
  masks from token spans (method M3 `masks_incremental`, lifted from logs/train_stack_check_qwen35_9b/probes/02_chat_template_format.py):
  for a trained assistant turn at message index i the span is
  `tokens(render(msgs[:i+1]))[len(tokens(render(msgs[:i], add_generation_prompt=True))):]` up to and including
  `<|im_end|>` (248046). The prompt-supplied `<think>\n` opener (or the whole empty `<think>\n\n</think>\n\n` block of
  non-thinking rows) is prompt, not target. Prefix consistency is asserted (`PrefixViolationError` otherwise) and
  `sum(assistant_masks) > 0` is guarded (`EmptyMaskError`).
* Splitting: `episode` (one row, every assistant turn trained; function-calling loops whose results are
  `role=tool`) or `per_turn` (one row per assistant turn, prompt = `render(msgs[:i], add_generation_prompt=True)`;
  observation-as-user loops), with either a per-episode turn cap (the last N turns) or `turns_per_episode` (the
  latest turn whose row fits `max_length` plus turns drawn at random, seeded per sample, among the earlier turns whose
  rows fit; docs/DATA_PLAN.md section 3.3). `max_length` truncates tool outputs (longest first,
  head and tail kept around a marker) and never assistant turns; rows that still do not fit are dropped with a
  recorded reason.
* Rows are Arrow-safe (`input_ids`, `assistant_masks`, flat scalar metadata; never nested heterogeneous dicts,
  which HF datasets None-pads into the targets) and go to JSONL or Parquet; `to_hf_dataset` / `load_rows` /
  `load_hf_dataset` read them back. Feed TRL with `assistant_only_loss=False`; it builds `labels` from
  `assistant_masks` (sft_trainer.py:1406-1431).

Inputs : JSONL of samples (one object per line: `{"messages": [...], "tools": [...] | null, "metadata": {...}}`;
         a flat shape with top-level `id`/`source`/... is accepted too) and a tokenizer directory.
Outputs: `--out` rows.parquet | rows.jsonl and `--report` stats.json (metadata, token histograms, dropped rows and reasons).

CLI (run from the repository root; nothing random, nothing written outside --out/--report):
    PYTHONPATH=src .venv/train/bin/python -m apex_train.data_format --in samples.jsonl --out rows.parquet \
        --tokenizer <dir> --mode episode|per_turn [--max-length N] [--report stats.json] [--no-thinking] \
        [--turn-cap N | --turns-per-episode K [--seed S]] [--min-tool-tokens 64] [--truncate-user-observations] [--strict]
    PYTHONPATH=src .venv/train/bin/python -m apex_train.data_format --write-synthetic samples.jsonl --n 20
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from jinja2.exceptions import TemplateError

REPO_ROOT = Path(__file__).resolve().parents[2]
IM_END_TOKEN = "<|im_end|>"
IM_END_ID = 248046                      # eos; must stay inside every trained span (REPORT section 2, "Masking")
ROLES = ("system", "user", "assistant", "tool")
MODES = ("episode", "per_turn")
META_FIELDS = ("id", "source", "task_type", "language", "difficulty", "license")
TRUNCATION_MARKER = "\n[... tool output truncated: {n} tokens removed ...]\n"
TOKEN_BINS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
ROW_COLUMNS = ("input_ids", "assistant_masks", "sample_id", "source", "task_type", "language", "difficulty",
               "license", "mode", "turn_index", "n_tokens", "n_assistant_tokens", "n_assistant_turns", "thinking",
               "truncated_tool_outputs", "truncation_tokens_removed")


# ------------------------------------------------------------------------------------------------------ errors
class DataFormatError(ValueError):
    """Base class: the sample cannot be turned into a trainer row. `reason` is the key used in drop records."""

    reason = "data_format_error"


class SampleValidationError(DataFormatError):
    """The sample violates the schema (roles, structure, reasoning traces, tool calls, tool schemas)."""

    reason = "invalid_sample"


class RenderError(DataFormatError):
    """The chat template refused the messages (jinja raise_exception or a TypeError on string arguments)."""

    reason = "render_error"


class PrefixViolationError(DataFormatError):
    """render(msgs[:i], add_generation_prompt=True) is not a token prefix of render(msgs[:i+1])."""

    reason = "prefix_violation"


class EmptyMaskError(DataFormatError):
    """No token would be trained (transformers only warns and returns an all-zero mask in this situation)."""

    reason = "empty_mask"


class RowTooLongError(DataFormatError):
    """The row exceeds max_length even after tool-output truncation (`n_tokens`: its length when it was given up)."""

    reason = "too_long"

    def __init__(self, message: str, n_tokens: int | None = None) -> None:
        super().__init__(message)
        self.n_tokens = n_tokens


# ------------------------------------------------------------------------------------------------------ schema
def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _normalise_content(value: Any, where: str) -> str:
    """Message content as one string. None -> ''. A list of text parts is concatenated exactly as the template
    does; image/video parts are rejected (text-only SFT: no pixel_values are ever fed)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for j, item in enumerate(value):
            if not isinstance(item, dict) or item.get("type", "text") != "text" or not isinstance(item.get("text"), str):
                kind = item.get("type") if isinstance(item, dict) else type(item).__name__
                raise SampleValidationError(f"{where}.content[{j}]: only text parts are supported in text-only SFT "
                                            f"(got {kind!r})")
            parts.append(item["text"])
        return "".join(parts)
    raise SampleValidationError(f"{where}: content must be a string or a list of text parts, got {type(value).__name__}")


@dataclass
class ToolCall:
    """One function call of an assistant turn. `arguments` is always a dict (the template needs a mapping). The
    template renders each value as `<parameter=k>\\n{v}\\n</parameter>`: strings verbatim (multi-line allowed, no
    quoting), dicts/lists as JSON, every other scalar through `str()` (`True`/`False`/`None`, not JSON)."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None   # provenance only; the template does not render it

    @classmethod
    def from_dict(cls, d: Any, where: str) -> ToolCall:
        """Accepts `{type, function: {name, arguments}}` or the flat `{name, arguments}`; JSON-string arguments are
        parsed (OpenAI wire format), anything that is not a JSON object is a validation error."""
        if not isinstance(d, dict):
            raise SampleValidationError(f"{where}: tool call must be a dict, got {type(d).__name__}")
        if d.get("type", "function") != "function":
            raise SampleValidationError(f"{where}: tool call type must be 'function', got {d.get('type')!r}")
        fn = d.get("function", d)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise SampleValidationError(f"{where}: tool call needs function.name as a non-empty string")
        args = fn.get("arguments")
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise SampleValidationError(f"{where}: arguments is not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise SampleValidationError(f"{where}: arguments must be a JSON object, got {type(args).__name__}")
        return cls(name=fn["name"], arguments=args, id=_opt_str(d.get("id")))

    def to_template(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Message:
    """One chat message. `reasoning_content`/`tool_calls` exist on assistant turns only; `name`/`tool_call_id`
    are kept for provenance on tool messages (the template renders neither)."""

    role: str
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str) -> Message:
        if not isinstance(d, dict):
            raise SampleValidationError(f"{where}: message must be a dict, got {type(d).__name__}")
        role = d.get("role")
        if role not in ROLES:
            raise SampleValidationError(f"{where}: role must be one of {ROLES}, got {role!r}")
        content = _normalise_content(d.get("content"), where)
        reasoning = d.get("reasoning_content")
        if reasoning is None and isinstance(d.get("reasoning"), str):
            reasoning = d["reasoning"]          # vLLM 0.31 response field name (harness_request_shapes section 2)
        if reasoning is not None and not isinstance(reasoning, str):
            raise SampleValidationError(f"{where}: reasoning_content must be a string or null")
        raw_calls = d.get("tool_calls")
        tool_calls = None
        if raw_calls:
            if not isinstance(raw_calls, list):
                raise SampleValidationError(f"{where}: tool_calls must be a list")
            tool_calls = [ToolCall.from_dict(tc, f"{where}.tool_calls[{j}]") for j, tc in enumerate(raw_calls)]
        if role != "assistant" and (reasoning or tool_calls):
            raise SampleValidationError(f"{where}: only assistant turns may carry reasoning_content or tool_calls")
        if role == "assistant" and ("<think>" in content or "</think>" in content):
            raise SampleValidationError(f"{where}: assistant content contains a think tag; put the trace in reasoning_content")
        return cls(role=role, content=content, reasoning_content=reasoning if role == "assistant" else None,
                   tool_calls=tool_calls, name=_opt_str(d.get("name")), tool_call_id=_opt_str(d.get("tool_call_id")))

    def to_template(self) -> dict[str, Any]:
        """The dict handed to apply_chat_template (only keys the template reads)."""
        m: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant":
            if self.reasoning_content is not None:
                m["reasoning_content"] = self.reasoning_content
            if self.tool_calls:
                m["tool_calls"] = [tc.to_template() for tc in self.tool_calls]
        return m

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON form (all fields, nulls dropped, tool calls nested)."""
        d = self.to_template()
        if self.tool_calls:
            d["tool_calls"] = [{**tc.to_template(), **({"id": tc.id} if tc.id else {})} for tc in self.tool_calls]
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class SampleMeta:
    """Flat provenance fields; every one of them becomes a scalar row column."""

    id: str
    source: str | None = None
    task_type: str | None = None
    language: str | None = None
    difficulty: str | None = None
    license: str | None = None

    @classmethod
    def from_dict(cls, d: Any, default_id: str | None = None) -> SampleMeta:
        if not isinstance(d, dict):
            raise SampleValidationError(f"metadata must be a dict, got {type(d).__name__}")
        sid = d.get("id", default_id)
        if sid is None or sid == "":
            raise SampleValidationError("metadata.id is required (pass default_id to derive one)")
        fields = {k: d.get(k) for k in META_FIELDS if k != "id"}
        for k, v in fields.items():
            if v is not None and not isinstance(v, str):
                fields[k] = str(v)
        return cls(id=str(sid), **fields)


def normalise_tools(tools: Any, where: str) -> list[dict[str, Any]] | None:
    """Tool schemas in the exact shape vLLM hands to the template at serving time
    (`ChatCompletionToolsParam`/`FunctionDefinition` serialisers, vllm 0.31.0): `{type, function: {name, description,
    parameters}}` with `description`/`parameters` kept as null when absent and `strict`/`defer_loading` only when
    set. Accepts the nested or the flat `{name, description, parameters}` form, or a JSON string of either.
    Returns None for no tools (an empty list renders exactly like None)."""
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise SampleValidationError(f"{where}: tools is not valid JSON: {exc}") from exc
    if not isinstance(tools, list):
        raise SampleValidationError(f"{where}: tools must be a list, got {type(tools).__name__}")
    out: list[dict[str, Any]] = []
    for j, t in enumerate(tools):
        if not isinstance(t, dict):
            raise SampleValidationError(f"{where}.tools[{j}]: tool must be a dict")
        if t.get("type", "function") != "function":
            raise SampleValidationError(f"{where}.tools[{j}]: tool type must be 'function'")
        fn = t.get("function", t)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise SampleValidationError(f"{where}.tools[{j}]: function.name must be a non-empty string")
        if fn.get("description") is not None and not isinstance(fn["description"], str):
            raise SampleValidationError(f"{where}.tools[{j}]: function.description must be a string or null")
        if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
            raise SampleValidationError(f"{where}.tools[{j}]: function.parameters must be a JSON object or null")
        norm: dict[str, Any] = {"name": fn["name"], "description": fn.get("description"), "parameters": fn.get("parameters")}
        for extra in ("strict", "defer_loading"):
            if fn.get(extra) is not None:
                norm[extra] = fn[extra]
        out.append({"type": "function", "function": norm})
    return out or None


@dataclass
class Sample:
    """One conversational training sample (an episode or a single exchange)."""

    messages: list[Message]
    meta: SampleMeta
    tools: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, d: Any, default_id: str | None = None) -> Sample:
        """Parse and normalise one JSON object; raises SampleValidationError with the offending path."""
        if not isinstance(d, dict):
            raise SampleValidationError(f"sample must be a dict, got {type(d).__name__}")
        meta_src = d.get("metadata")
        if meta_src is None:
            meta_src = {k: d[k] for k in META_FIELDS if k in d}
        meta = SampleMeta.from_dict(meta_src, default_id)
        where = f"sample {meta.id}"
        if not isinstance(d.get("messages"), list) or not d["messages"]:
            raise SampleValidationError(f"{where}: messages must be a non-empty list")
        messages = [Message.from_dict(m, f"{where}: messages[{i}]") for i, m in enumerate(d["messages"])]
        sample = cls(messages=messages, meta=meta, tools=normalise_tools(d.get("tools"), where))
        sample.check_structure()
        return sample

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.messages], "tools": self.tools,
                "metadata": dataclasses.asdict(self.meta)}

    def assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m.role == "assistant"]

    def template_messages(self) -> list[dict[str, Any]]:
        return [m.to_template() for m in self.messages]

    def check_structure(self) -> None:
        """Role-sequence rules the template and the masking rely on."""
        msgs, where = self.messages, f"sample {self.meta.id}"
        if any(m.role == "system" for m in msgs[1:]):
            raise SampleValidationError(f"{where}: a system message is only allowed at messages[0]")
        first = next((m for m in msgs if m.role != "system"), None)
        if first is None or first.role != "user":
            raise SampleValidationError(f"{where}: the first non-system message must be a user message")
        if msgs[-1].role != "assistant":
            raise SampleValidationError(f"{where}: the sample must end with an assistant turn (a trailing observation "
                                        "has no target; cut it upstream)")
        for i, m in enumerate(msgs):
            if m.role == "tool":
                prev = msgs[i - 1]
                if not (prev.role == "tool" or (prev.role == "assistant" and prev.tool_calls)):
                    raise SampleValidationError(f"{where}: messages[{i}] (tool) must follow a tool-calling assistant turn")
            if m.role == "assistant" and not m.content.strip() and not m.tool_calls:
                raise SampleValidationError(f"{where}: messages[{i}] is an assistant turn with neither content nor tool calls")

    def validate(self, *, enable_thinking: bool = True) -> None:
        """Structure plus the thinking-trace rule: thinking rows need a non-empty `reasoning_content` on every
        assistant turn; non-thinking rows must carry none (the render would otherwise break the prefix property)."""
        self.check_structure()
        for i, m in enumerate(self.messages):
            if m.role != "assistant":
                continue
            has_trace = isinstance(m.reasoning_content, str) and bool(m.reasoning_content.strip())
            if enable_thinking and not has_trace:
                raise SampleValidationError(
                    f"sample {self.meta.id}: messages[{i}] has no reasoning_content; thinking-mode samples need a real "
                    "trace on every assistant turn (an empty <think> block would train the model to skip thinking)")
            if not enable_thinking and has_trace:
                raise SampleValidationError(
                    f"sample {self.meta.id}: messages[{i}] carries reasoning_content but the row is rendered with "
                    "enable_thinking=False; strip the traces upstream or render the sample in thinking mode")


# --------------------------------------------------------------------------------------------- render + mask
def _single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise DataFormatError(f"{text!r} is not a single token for this tokenizer (got {ids})")
    return ids[0]


def encode_messages(tokenizer: Any, messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None, *,
                    add_generation_prompt: bool = False, **chat_kwargs: Any) -> list[int]:
    """`apply_chat_template(tokenize=True)` as a plain list of ids; template refusals become RenderError."""
    try:
        ids = tokenizer.apply_chat_template(list(messages), tools=tools, tokenize=True, return_dict=False,
                                            add_generation_prompt=add_generation_prompt, **chat_kwargs)
    except (TypeError, TemplateError) as exc:
        raise RenderError(f"chat template refused the messages: {type(exc).__name__}: {exc}") from exc
    return list(ids)


def _turn_end(tokenizer: Any, turn_ids: list[int], im_end: int) -> int:
    """Exclusive end of the trained span: just after <|im_end|>; the glue '\\n' the template appends after it is
    never generated by the model and stays out of the mask."""
    if turn_ids and turn_ids[-1] == im_end:
        return len(turn_ids)
    if len(turn_ids) >= 2 and turn_ids[-2] == im_end and tokenizer.decode([turn_ids[-1]]) == "\n":
        return len(turn_ids) - 1
    raise RenderError("assistant turn render does not end with <|im_end|>")


def _prefix_error(tokenizer: Any, sample_id: str, i: int, what: str, shorter: list[int], longer: list[int]) -> PrefixViolationError:
    """Error for `shorter` not being a token prefix of `longer` (`what` names the two renders), with the fix."""
    k = next((j for j, (a, b) in enumerate(zip(shorter, longer, strict=False)) if a != b), min(len(shorter), len(longer)))
    lo = max(0, k - 8)
    return PrefixViolationError(
        f"sample {sample_id}: for the assistant turn at messages[{i}], {what} (first mismatch at token {k}: "
        f"...{tokenizer.decode(shorter[lo:k + 4])!r} vs ...{tokenizer.decode(longer[lo:k + 4])!r}). The template re-renders "
        "earlier assistant turns without their think block once a later user query exists (think window; this template has "
        "no switch to keep them): use mode 'per_turn' for observation-as-user loops, or cut the episode at the query that "
        "resets the window")


def render_and_mask(sample: Sample, tokenizer: Any, *, enable_thinking: bool = True,
                    train_turns: Iterable[int] | None = None) -> dict[str, Any]:
    """Render one sample with the shipped chat template and build the assistant-only mask from token spans (M3).

    Inputs : sample; tokenizer (Qwen3.5-9B, with its chat template); enable_thinking (False renders non-thinking rows
             whose empty think block is prompt); train_turns = message indices of the assistant turns to train
             (default: all of them). Earlier assistant turns are replayed exactly as the template does at inference
             (think blocks only after the last user query; the template has no knob to change that).
    Output : {input_ids, attention_mask, assistant_masks, assistant_spans [(start, end), ...], n_assistant_tokens,
             n_tokens}. Raises SampleValidationError, RenderError, PrefixViolationError or EmptyMaskError.

    Lifted from `masks_incremental()` in logs/train_stack_check_qwen35_9b/probes/02_chat_template_format.py (first written and
    fuzzed in the archived 35B training-stack study, 900 cases with 0 prefix violations; re-run on this template, 123/123 checks):
    span_i = tokens(render(msgs[:i+1]))[len(tokens(render(msgs[:i], add_generation_prompt=True))):end],
    with <|im_end|> inside and the think opener supplied by the generation prompt outside.
    """
    sample.validate(enable_thinking=enable_thinking)
    msgs, tools = sample.template_messages(), sample.tools
    kw = {"enable_thinking": bool(enable_thinking)}
    im_end = _single_token_id(tokenizer, IM_END_TOKEN)
    assistant_idx = sample.assistant_indices()
    turns = assistant_idx if train_turns is None else sorted(set(train_turns))
    if any(i not in assistant_idx for i in turns):
        raise ValueError(f"train_turns {turns} must be message indices of assistant turns {assistant_idx}")
    full = encode_messages(tokenizer, msgs, tools, **kw)
    mask = [0] * len(full)
    spans: list[tuple[int, int]] = []
    for i in turns:
        prompt = encode_messages(tokenizer, msgs[:i], tools, add_generation_prompt=True, **kw)
        turn = encode_messages(tokenizer, msgs[:i + 1], tools, **kw)
        if turn[:len(prompt)] != prompt:
            raise _prefix_error(tokenizer, sample.meta.id, i, "its generation prompt is not a token prefix of its render",
                                prompt, turn)
        if full[:len(turn)] != turn:
            raise _prefix_error(tokenizer, sample.meta.id, i, f"render(messages[:{i + 1}]) is not a token prefix of the "
                                "full render, i.e. later messages change how this turn renders", turn, full)
        end = _turn_end(tokenizer, turn, im_end)
        for j in range(len(prompt), end):
            mask[j] = 1
        spans.append((len(prompt), end))
    n_trained = sum(mask)
    if n_trained == 0:
        raise EmptyMaskError(f"sample {sample.meta.id}: no token would be trained (train_turns={turns})")
    return {"input_ids": full, "attention_mask": [1] * len(full), "assistant_masks": mask, "assistant_spans": spans,
            "n_assistant_tokens": n_trained, "n_tokens": len(full)}


def trained_text(tokenizer: Any, input_ids: Sequence[int], assistant_masks: Sequence[int]) -> str:
    """Decoded concatenation of the trained tokens (for inspection and tests)."""
    return tokenizer.decode([t for t, m in zip(input_ids, assistant_masks, strict=True) if m])


# ------------------------------------------------------------------------------------------ rows and splitting
@dataclass
class BuildOptions:
    """Knobs of `build_rows` (one instance per data-prep run; every field is recorded in the report)."""

    mode: str = "episode"
    enable_thinking: bool = True
    max_length: int | None = None
    turn_cap: int | None = None
    turns_per_episode: int | None = None
    candidate_turns: tuple[int, ...] | None = None
    priority_turns: tuple[int, ...] | None = None
    max_turn_completion_tokens: int | None = None
    seed: int = 0
    min_tool_tokens: int = 64
    truncate_roles: tuple[str, ...] = ("tool",)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if self.turn_cap is not None and self.turn_cap < 1:
            raise ValueError("turn_cap must be a positive integer")
        if self.turns_per_episode is not None:
            if self.turns_per_episode < 1:
                raise ValueError("turns_per_episode must be a positive integer")
            if self.mode != "per_turn" or self.turn_cap is not None:
                raise ValueError("turns_per_episode needs mode 'per_turn' and excludes turn_cap")
        if self.candidate_turns is not None:
            if self.turns_per_episode is None:
                raise ValueError("candidate_turns restricts the turns_per_episode selection; set turns_per_episode")
            self.candidate_turns = tuple(sorted(set(int(i) for i in self.candidate_turns)))
        if self.priority_turns is not None:
            if self.turns_per_episode is None:
                raise ValueError("priority_turns orders the turns_per_episode selection; set turns_per_episode")
            self.priority_turns = tuple(dict.fromkeys(int(i) for i in self.priority_turns))   # order kept, repeats dropped
            if any(i < 0 and i != LATEST_FITTING for i in self.priority_turns):
                raise ValueError("priority_turns holds message indices and LATEST_FITTING")
        if self.max_turn_completion_tokens is not None and self.max_turn_completion_tokens < 1:
            raise ValueError("max_turn_completion_tokens must be a positive integer")
        if self.min_tool_tokens < 2:
            raise ValueError("min_tool_tokens must be >= 2 (head and tail need at least one token each)")
        if any(r not in ("tool", "user") for r in self.truncate_roles):
            raise ValueError("truncate_roles may only contain 'tool' and 'user' (never assistant turns or the system prompt)")


def turn_span_lengths(assistant_masks: Sequence[int]) -> list[int]:
    """Lengths of the contiguous trained spans of a row, one per trained assistant turn (reasoning + content + tool calls
    + the end token)."""
    out, run = [], 0
    for m in assistant_masks:
        if m:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


TURN_TOO_LONG = "turn_too_long"
LATEST_FITTING = -1   # BuildOptions.priority_turns entry: take the latest fitting turn at this position


@dataclass
class DropRecord:
    """Why a row was not produced. turn_index is -1 for whole-sample records."""

    sample_id: str
    turn_index: int
    reason: str
    detail: str
    n_tokens: int | None = None


def _truncatable_indices(messages: Sequence[Message], roles: tuple[str, ...]) -> list[int]:
    """Messages whose content may be shortened: tool results and, when enabled, user observations after the first
    user message (the task statement is never touched; assistant turns and the system prompt never are)."""
    first_user = next(i for i, m in enumerate(messages) if m.role == "user")
    return [i for i, m in enumerate(messages) if m.role in roles and not (m.role == "user" and i == first_user)]


def head_tail_truncate(tokenizer: Any, text: str, keep_tokens: int) -> tuple[str, int]:
    """Keep the first and last `keep_tokens` tokens of `text` around TRUNCATION_MARKER; returns (text, removed)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= keep_tokens:
        return text, 0
    head = keep_tokens // 2
    tail = keep_tokens - head
    removed = len(ids) - keep_tokens
    return tokenizer.decode(ids[:head]) + TRUNCATION_MARKER.format(n=removed) + tokenizer.decode(ids[-tail:]), removed


def _fit_to_max_length(sample: Sample, rendered: dict[str, Any], tokenizer: Any, opts: BuildOptions,
                       train_turns: list[int] | None) -> tuple[Sample, dict[str, Any], int, int]:
    """Shrink tool outputs (longest first, head+tail) until the row fits opts.max_length; never touches assistant
    turns. Returns (sample, rendered, n_truncated_outputs, tokens_removed) or raises RowTooLongError."""
    assert opts.max_length is not None
    messages = [dataclasses.replace(m) for m in sample.messages]
    original = {k: messages[k].content for k in _truncatable_indices(messages, opts.truncate_roles)}
    length = {k: len(tokenizer.encode(v, add_special_tokens=False)) for k, v in original.items()}
    budget = dict(length)                       # tokens of each truncatable content still kept
    marker = len(tokenizer.encode(TRUNCATION_MARKER.format(n=10 ** 6), add_special_tokens=False))
    kw = {"enable_thinking": opts.enable_thinking}
    n_tokens_before = rendered["n_tokens"]
    for _ in range(8 * len(original) + 16):
        excess = rendered["n_tokens"] - opts.max_length
        if excess <= 0:
            removed = sum(length[k] - budget[k] for k in original)
            return sample, rendered, sum(budget[k] < length[k] for k in original), removed
        # Net saving still available per output; the marker costs tokens the first time an output is cut, so a
        # short output whose cut would not pay for the marker is never touched.
        saving = {k: budget[k] - opts.min_tool_tokens - (0 if budget[k] < length[k] else marker) for k in original}
        candidates = [(budget[k], k) for k in original if saving[k] > 0]
        if not candidates:
            raise RowTooLongError(
                f"sample {sample.meta.id}: {rendered['n_tokens']} tokens > max_length {opts.max_length} after truncating "
                f"{sum(budget[k] < length[k] for k in original)} of {len(original)} truncatable outputs to "
                f">= {opts.min_tool_tokens} tokens (row was {n_tokens_before} tokens before truncation)",
                n_tokens=rendered["n_tokens"])
        cur, k = max(candidates)
        want = excess + (0 if budget[k] < length[k] else marker)
        budget[k] = max(opts.min_tool_tokens, cur - want)
        messages[k].content, _ = head_tail_truncate(tokenizer, original[k], budget[k])
        sample = dataclasses.replace(sample, messages=messages)
        rendered = render_and_mask(sample, tokenizer, train_turns=train_turns, **kw)
    raise RowTooLongError(f"sample {sample.meta.id}: tool-output truncation did not converge to max_length {opts.max_length}",
                          n_tokens=rendered["n_tokens"])


def make_row(sample: Sample, rendered: dict[str, Any], *, mode: str, turn_index: int, n_assistant_turns: int,
             thinking: bool, truncated_tool_outputs: int, truncation_tokens_removed: int) -> dict[str, Any]:
    """Flat, Arrow-safe trainer row (ROW_COLUMNS)."""
    meta = sample.meta
    return {"input_ids": rendered["input_ids"], "assistant_masks": rendered["assistant_masks"],
            "sample_id": meta.id, "source": meta.source, "task_type": meta.task_type, "language": meta.language,
            "difficulty": meta.difficulty, "license": meta.license, "mode": mode, "turn_index": turn_index,
            "n_tokens": rendered["n_tokens"], "n_assistant_tokens": rendered["n_assistant_tokens"],
            "n_assistant_turns": n_assistant_turns, "thinking": thinking,
            "truncated_tool_outputs": truncated_tool_outputs, "truncation_tokens_removed": truncation_tokens_removed}


def _build_one(sample: Sample, messages: list[Message], train_turns: list[int] | None, turn_index: int,
               tokenizer: Any, opts: BuildOptions) -> dict[str, Any]:
    sub = dataclasses.replace(sample, messages=list(messages))
    rendered = render_and_mask(sub, tokenizer, enable_thinking=opts.enable_thinking, train_turns=train_turns)
    n_trunc, removed = 0, 0
    if opts.max_length is not None and rendered["n_tokens"] > opts.max_length:
        sub, rendered, n_trunc, removed = _fit_to_max_length(sub, rendered, tokenizer, opts, train_turns)
    n_turns = len(sub.assistant_indices()) if train_turns is None else len(train_turns)
    return make_row(sub, rendered, mode=opts.mode, turn_index=turn_index, n_assistant_turns=n_turns,
                    thinking=opts.enable_thinking, truncated_tool_outputs=n_trunc, truncation_tokens_removed=removed)


def _units(sample: Sample, opts: BuildOptions) -> tuple[list[tuple[int, list[Message], list[int] | None]], list[DropRecord]]:
    """(turn_index, messages, train_turns) per row to build, plus turn-cap drop records (per_turn keeps the last N
    assistant turns: late turns carry the most task-specific signal and the cost per row grows with i)."""
    if opts.mode == "episode":
        return [(-1, sample.messages, None)], []
    idx = sample.assistant_indices()
    dropped: list[DropRecord] = []
    if opts.turn_cap is not None and len(idx) > opts.turn_cap:
        skipped, idx = idx[:-opts.turn_cap], idx[-opts.turn_cap:]
        dropped = [DropRecord(sample.meta.id, i, "turn_cap", f"kept the last {opts.turn_cap} of {len(skipped) + len(idx)} "
                              "assistant turns") for i in skipped]
    return [(i, sample.messages[:i + 1], [i]) for i in idx], dropped


def _row_lower_bound(sample: Sample, tokenizer: Any, opts: BuildOptions) -> list[int]:
    """Per assistant turn i (in `assistant_indices` order): a lower bound on the tokens of the per_turn row over
    messages[:i+1]: every message's content (earlier assistant turns content only, as the template renders them;
    truncatable messages at their `min_tool_tokens` floor) plus turn i's reasoning, minus 2 tokens per message for
    BPE merges across boundaries. A turn whose bound exceeds max_length cannot fit and is never rendered."""
    roles = opts.truncate_roles if opts.max_length is not None else ()
    first_user = next((k for k, m in enumerate(sample.messages) if m.role == "user"), None)

    def n(text: str | None) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

    prefix, bounds = 0, []
    for k, m in enumerate(sample.messages):
        own = n(m.content)
        if m.role in roles and not (m.role == "user" and k == first_user):
            own = min(own, opts.min_tool_tokens)
        if m.role == "assistant":
            bounds.append(prefix + own + n(m.reasoning_content) - 2 * (k + 1))
        prefix += own
    return bounds


def _completion_lower_bound(sample: Sample, tokenizer: Any) -> dict[int, int]:
    """Per assistant turn (message index): a lower bound on its trained tokens (reasoning + content, minus a margin for
    BPE merges and stripped whitespace at the boundaries); a turn whose bound exceeds max_turn_completion_tokens is
    never rendered."""
    def n(text: str | None) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
    return {i: n(sample.messages[i].reasoning_content) + n(sample.messages[i].content) - 16
            for i in sample.assistant_indices()}


def _turn_cap_drop(sample_id: str, row: dict[str, Any], turn_index: int, cap: int) -> DropRecord | None:
    """A DropRecord when a trained span of `row` exceeds `cap` (max_turn_completion_tokens), else None."""
    longest = max(turn_span_lengths(row["assistant_masks"]), default=0)
    if longest <= cap:
        return None
    return DropRecord(sample_id, turn_index, TURN_TOO_LONG,
                      f"trained turn of {longest} tokens > max_turn_completion_tokens {cap}", longest)


def _build_selected_turns(sample: Sample, tokenizer: Any, opts: BuildOptions,
                          sources: dict[int, str] | None = None) -> tuple[list[dict[str, Any]], list[DropRecord]]:
    """per_turn rows for `turns_per_episode` turns: the latest assistant turn whose row fits (rendered, after the
    configured truncation), then turns drawn uniformly at random among the earlier turns whose rows fit. The RNG is
    seeded with (opts.seed, sample id), so a rebuild picks the same turns. With `candidate_turns` (message indices) only
    those assistant turns may be trained; the others are recorded as `not_candidate`. With `max_turn_completion_tokens`
    a turn whose trained tokens exceed it does not fit either (`turn_too_long`): the latest fitting turn is the latest
    within that cap and the draw runs over the earlier turns within it. Turns that do not fit are recorded with their
    failure reason, turns never selected as `not_selected`; an episode where no turn fits yields no row. With
    `priority_turns` (message indices in priority order) those turns are tried first, each skipped when it is not a
    candidate or does not fit (the same checks and drop records); a LATEST_FITTING entry takes the latest fitting turn
    at its position (without one, after them), and the random draw fills the slots left. `sources`, when given, gets
    turn_index -> the rule that chose the row: 'priority', 'latest' or 'random'."""
    all_idx = sample.assistant_indices()
    bound = dict(zip(all_idx, _row_lower_bound(sample, tokenizer, opts)))
    turn_cap = opts.max_turn_completion_tokens
    completion_bound = _completion_lower_bound(sample, tokenizer) if turn_cap is not None else {}
    allowed = set(all_idx) if opts.candidate_turns is None else set(opts.candidate_turns)
    idx = [i for i in all_idx if i in allowed]
    rows: list[dict[str, Any]] = []
    dropped: list[DropRecord] = [DropRecord(sample.meta.id, i, "not_candidate", "excluded by candidate_turns")
                                 for i in all_idx if i not in allowed]

    def attempt(i: int, how: str) -> bool:
        if opts.max_length is not None and bound[i] > opts.max_length:
            dropped.append(DropRecord(sample.meta.id, i, RowTooLongError.reason, f"lower bound {bound[i]} tokens > "
                                      f"max_length {opts.max_length} (not rendered)", bound[i]))
            return False
        if turn_cap is not None and completion_bound[i] > turn_cap:
            dropped.append(DropRecord(sample.meta.id, i, TURN_TOO_LONG, f"lower bound {completion_bound[i]} trained tokens "
                                      f"> max_turn_completion_tokens {turn_cap} (not rendered)", completion_bound[i]))
            return False
        try:
            row = _build_one(sample, sample.messages[:i + 1], [i], i, tokenizer, opts)
        except (PrefixViolationError, EmptyMaskError, RowTooLongError, RenderError) as exc:
            dropped.append(DropRecord(sample.meta.id, i, exc.reason, str(exc), getattr(exc, "n_tokens", None)))
            return False
        if turn_cap is not None and (over := _turn_cap_drop(sample.meta.id, row, i, turn_cap)) is not None:
            dropped.append(over)
            return False
        rows.append(row)
        picked[i] = how
        return True

    def latest_fitting() -> None:
        while remaining and len(rows) < opts.turns_per_episode and not attempt(remaining.pop(), "latest"):
            pass

    picked: dict[int, str] = {}
    remaining = list(idx)
    order = opts.priority_turns or ()
    for i in order:
        if len(rows) >= opts.turns_per_episode:
            break
        if i == LATEST_FITTING:
            latest_fitting()
        elif i in remaining:
            remaining.remove(i)
            attempt(i, "priority")
    if LATEST_FITTING not in order:
        latest_fitting()
    rng = random.Random(f"{opts.seed}:{sample.meta.id}")
    while remaining and len(rows) < opts.turns_per_episode:
        attempt(remaining.pop(rng.randrange(len(remaining))), "random")
    dropped += [DropRecord(sample.meta.id, i, "not_selected", f"turns_per_episode={opts.turns_per_episode}")
                for i in remaining]
    rows.sort(key=lambda r: r["turn_index"])
    if sources is not None:
        sources.update(picked)
    return rows, dropped


def build_rows(sample: Sample, tokenizer: Any, opts: BuildOptions, *,
               sources: dict[int, str] | None = None) -> tuple[list[dict[str, Any]], list[DropRecord]]:
    """One sample -> trainer rows under `opts` ('episode': one row with every assistant turn trained; 'per_turn':
    one row per assistant turn i over messages[:i+1] with only turn i trained; with `turns_per_episode`, only the
    selected turns, see `_build_selected_turns`, which fills `sources`).

    Returns (rows, dropped). Row-level failures (prefix violation, empty mask, too long, template refusal, a trained
    turn longer than `max_turn_completion_tokens`) become DropRecords; a sample that fails validation raises
    SampleValidationError (the caller decides whether to stop).
    """
    sample.validate(enable_thinking=opts.enable_thinking)
    if opts.turns_per_episode is not None:
        return _build_selected_turns(sample, tokenizer, opts, sources)
    units, dropped = _units(sample, opts)
    rows: list[dict[str, Any]] = []
    for turn_index, messages, train_turns in units:
        try:
            row = _build_one(sample, messages, train_turns, turn_index, tokenizer, opts)
        except (PrefixViolationError, EmptyMaskError, RowTooLongError, RenderError) as exc:
            dropped.append(DropRecord(sample.meta.id, turn_index, exc.reason, str(exc), getattr(exc, "n_tokens", None)))
            continue
        cap = opts.max_turn_completion_tokens
        if cap is not None and (over := _turn_cap_drop(sample.meta.id, row, turn_index, cap)) is not None:
            dropped.append(over)
            continue
        rows.append(row)
    return rows, dropped


# ------------------------------------------------------------------------------------------------ serialisation
def assert_arrow_safe(row: dict[str, Any]) -> None:
    """Rows hold only int lists and scalars (no nested dicts: HF datasets None-pads heterogeneous structs)."""
    if tuple(row) != ROW_COLUMNS:
        raise DataFormatError(f"row columns {tuple(row)} != {ROW_COLUMNS}")
    ids, mask = row["input_ids"], row["assistant_masks"]
    if len(ids) != len(mask) or not all(isinstance(t, int) for t in ids) or not set(mask) <= {0, 1}:
        raise DataFormatError("input_ids/assistant_masks must be equal-length int lists with masks in {0, 1}")
    for k in ROW_COLUMNS[2:]:
        if row[k] is not None and not isinstance(row[k], (str, int, bool)):
            raise DataFormatError(f"column {k} must be a scalar, got {type(row[k]).__name__}")


def arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema([("input_ids", pa.list_(pa.int32())), ("assistant_masks", pa.list_(pa.int8())),
                      ("sample_id", pa.string()), ("source", pa.string()), ("task_type", pa.string()),
                      ("language", pa.string()), ("difficulty", pa.string()), ("license", pa.string()),
                      ("mode", pa.string()), ("turn_index", pa.int32()), ("n_tokens", pa.int32()),
                      ("n_assistant_tokens", pa.int32()), ("n_assistant_turns", pa.int32()), ("thinking", pa.bool_()),
                      ("truncated_tool_outputs", pa.int32()), ("truncation_tokens_removed", pa.int32())])


def hf_features() -> Any:
    from datasets import Features, List, Value

    return Features({"input_ids": List(Value("int32")), "assistant_masks": List(Value("int8")),
                     **{k: Value("string") for k in ("sample_id", "source", "task_type", "language", "difficulty", "license", "mode")},
                     **{k: Value("int32") for k in ("turn_index", "n_tokens", "n_assistant_tokens", "n_assistant_turns",
                                                    "truncated_tool_outputs", "truncation_tokens_removed")},
                     "thinking": Value("bool")})


def write_rows(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    """Write rows to `path` (.parquet via pyarrow with the explicit schema, or .jsonl). Returns the path."""
    path = Path(path)
    for row in rows:
        assert_arrow_safe(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(list(rows), schema=arrow_schema()), path)
    elif path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"unsupported rows file suffix {path.suffix!r} (use .parquet or .jsonl)")
    return path


def load_rows(path: Path | str) -> list[dict[str, Any]]:
    """Read rows written by `write_rows` back into plain dicts (Python ints, no numpy)."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"unsupported rows file suffix {path.suffix!r} (use .parquet or .jsonl)")
    for row in rows:
        assert_arrow_safe(row)
    return rows


def to_hf_dataset(rows: Sequence[dict[str, Any]]) -> Any:
    """`datasets.Dataset` with the explicit features (what SFTTrainer consumes with assistant_only_loss=False)."""
    from datasets import Dataset

    for row in rows:
        assert_arrow_safe(row)
    return Dataset.from_list(list(rows), features=hf_features())


def load_hf_dataset(path: Path | str) -> Any:
    """Rows file -> `datasets.Dataset` (parquet is read by datasets directly, JSONL through load_rows)."""
    path = Path(path)
    if path.suffix == ".parquet":
        from datasets import Dataset

        return Dataset.from_parquet(str(path), features=hf_features())
    return to_hf_dataset(load_rows(path))


# -------------------------------------------------------------------------------------------------- statistics
def histogram(values: Iterable[int], bins: Sequence[int] = TOKEN_BINS) -> dict[str, int]:
    """Counts per upper-bounded bin ('<=512', ..., '>131072'); every bin is present even when empty."""
    labels = [f"<={b}" for b in bins] + [f">{bins[-1]}"]
    counts = dict.fromkeys(labels, 0)
    for v in values:
        counts[next((f"<={b}" for b in bins if v <= b), labels[-1])] += 1
    return counts


def summarise(rows: Sequence[dict[str, Any]], dropped: Sequence[DropRecord]) -> dict[str, Any]:
    """Aggregate statistics of a build: counts, token histograms, drop reasons, truncation totals."""
    n_tok = [r["n_tokens"] for r in rows]
    n_asst = [r["n_assistant_tokens"] for r in rows]
    return {
        "n_rows": len(rows), "n_dropped": len(dropped),
        "dropped_by_reason": dict(sorted(Counter(d.reason for d in dropped).items())),
        "tokens": {"total": sum(n_tok), "max": max(n_tok, default=0),
                   "mean": round(sum(n_tok) / len(n_tok), 1) if n_tok else 0.0, "histogram": histogram(n_tok)},
        "assistant_tokens": {"total": sum(n_asst), "max": max(n_asst, default=0),
                             "mean": round(sum(n_asst) / len(n_asst), 1) if n_asst else 0.0, "histogram": histogram(n_asst)},
        "rows_by_source": dict(sorted(Counter(str(r["source"]) for r in rows).items())),
        "rows_by_task_type": dict(sorted(Counter(str(r["task_type"]) for r in rows).items())),
        "truncated_rows": sum(1 for r in rows if r["truncated_tool_outputs"]),
        "truncation_tokens_removed": sum(r["truncation_tokens_removed"] for r in rows),
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable version of `summarise()` for stdout."""
    lines = [f"rows {summary['n_rows']}  dropped {summary['n_dropped']}  {summary['dropped_by_reason']}",
             f"tokens total {summary['tokens']['total']}  max {summary['tokens']['max']}  mean {summary['tokens']['mean']}; "
             f"assistant tokens total {summary['assistant_tokens']['total']}  max {summary['assistant_tokens']['max']}",
             f"truncated rows {summary['truncated_rows']}  tokens removed {summary['truncation_tokens_removed']}",
             "histogram (row tokens | assistant tokens):"]
    for label, n in summary["tokens"]["histogram"].items():
        lines.append(f"  {label:>9} {n:6d} | {summary['assistant_tokens']['histogram'][label]:6d}")
    lines.append(f"rows by source {summary['rows_by_source']}; by task_type {summary['rows_by_task_type']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- synthetic samples
def _oai_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    # Key order type -> function -> {name, description, parameters}: vLLM's serialisation order
    # (lifted from oai_tool() in logs/train_stack_check_qwen35_9b/probes/02_chat_template_format.py).
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


_SYN_TOOLS = [
    _oai_tool("run_shell", "Run a shell command in the sandbox and return stdout+stderr.",
              {"command": {"type": "string"}, "timeout_s": {"type": "integer", "default": 60}}, ["command"]),
    _oai_tool("read_file", "Read a slice of a text file with line numbers.",
              {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
]


def _syn_meta(i: int, task_type: str) -> dict[str, Any]:
    return {"id": f"syn-{i:03d}", "source": "synthetic", "task_type": task_type, "language": "en",
            "difficulty": ("easy", "medium", "hard")[i % 3], "license": "own-synthetic"}


def _syn_qa(i: int, rng: random.Random) -> dict[str, Any]:
    a, b = 11 + i, 7 + (i % 5)
    return {"messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": f"What is {a} * {b}? Answer with the number only."},
        {"role": "assistant", "reasoning_content": f"{a} * {b} = {a * b}. Reply with the number.", "content": str(a * b)}],
        "tools": None, "metadata": _syn_meta(i, "qa")}


def _syn_tool_episode(i: int, rng: random.Random) -> dict[str, Any]:
    path = f"src/module_{i}.py"
    return {"messages": [
        {"role": "system", "content": "You are a coding agent working in a sandbox."},
        {"role": "user", "content": f"The test tests/test_module_{i}.py fails. Inspect {path} and fix the bug."},
        {"role": "assistant", "reasoning_content": "Read the implementation before changing anything.",
         "content": "Let me read the file first.",
         "tool_calls": [{"type": "function", "function": {"name": "read_file",
                                                          "arguments": json.dumps({"path": path, "start_line": 1, "end_line": 20})}}]},
        {"role": "tool", "name": "read_file", "content": "1 def f(x):\n2     return x - 1\n"},
        {"role": "assistant", "reasoning_content": "The function subtracts where it should add; patch and rerun the test.",
         "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": {
             "command": f"sed -i 's/x - 1/x + 1/' {path} && python -m pytest -q tests/test_module_{i}.py", "timeout_s": 120}}}]},
        {"role": "tool", "name": "run_shell", "content": f".{' ' * rng.randint(1, 40)}[100%]\n1 passed in 0.0{rng.randint(1, 9)}s"},
        {"role": "assistant", "reasoning_content": "The test passes; summarise.",
         "content": f"Fixed: `f` subtracted 1 instead of adding; tests/test_module_{i}.py passes."}],
        "tools": _SYN_TOOLS, "metadata": _syn_meta(i, "tool_episode")}


def _syn_parallel_calls(i: int, rng: random.Random) -> dict[str, Any]:
    return {"messages": [
        {"role": "user", "content": f"Show the first lines of a.py and b{i}.py."},
        {"role": "assistant", "reasoning_content": "Both reads are independent; issue them together.", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py", "start_line": 1, "end_line": 3}}},
                        {"type": "function", "function": {"name": "read_file", "arguments": {"path": f"b{i}.py", "start_line": 1, "end_line": 3}}}]},
        {"role": "tool", "content": "1 import os\n2 \n3 X = 1\n"},
        {"role": "tool", "content": f"1 import sys\n2 \n3 Y = {i}\n"},
        {"role": "assistant", "reasoning_content": "Report both heads.",
         "content": f"a.py starts with `import os` and defines X = 1; b{i}.py starts with `import sys` and defines Y = {i}."}],
        "tools": [_SYN_TOOLS[1]], "metadata": _syn_meta(i, "parallel_calls")}


def _syn_observation_as_user(i: int, rng: random.Random) -> dict[str, Any]:
    def step(analysis: str, plan: str, commands: list[str], done: bool = False) -> str:
        return json.dumps({"analysis": analysis, "plan": plan, "commands": commands, "task_complete": done}, indent=2)
    return {"messages": [
        {"role": "user", "content": "You operate a Linux shell. Reply with one JSON object with keys analysis, plan, "
                                    f"commands, task_complete. Task: create out{i}.txt containing the number {i}."},
        {"role": "assistant", "reasoning_content": "Write the file, then verify it.",
         "content": step("Fresh shell in /app.", "Write the file then verify.", [f"echo {i} > out{i}.txt"])},
        {"role": "user", "content": f"root@box:/app# echo {i} > out{i}.txt\nroot@box:/app# "},
        {"role": "assistant", "reasoning_content": "Verify the content before finishing.",
         "content": step("File written.", "Check its content.", [f"cat out{i}.txt"])},
        {"role": "user", "content": f"root@box:/app# cat out{i}.txt\n{i}\nroot@box:/app# "},
        {"role": "assistant", "reasoning_content": "The content matches; the task is complete.",
         "content": step(f"out{i}.txt contains {i}.", "Nothing left to do.", [], True)}],
        "tools": None, "metadata": _syn_meta(i, "observation_as_user")}


def _syn_long_tool_output(i: int, rng: random.Random) -> dict[str, Any]:
    listing = "\n".join(f"{k:04d}: sample_{i}_{k} size={rng.randint(0, 99999)}" for k in range(rng.randint(250, 400)))
    return {"messages": [
        {"role": "user", "content": f"How many entries does the listing of /data/run{i} have?"},
        {"role": "assistant", "reasoning_content": "List the directory and count the lines.", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": {"command": f"ls -l /data/run{i}"}}}]},
        {"role": "tool", "name": "run_shell", "content": listing},
        {"role": "assistant", "reasoning_content": "Count the numbered lines of the listing.",
         "content": f"The listing of /data/run{i} has {listing.count(chr(10)) + 1} entries."}],
        "tools": [_SYN_TOOLS[0]], "metadata": _syn_meta(i, "long_tool_output")}


def _syn_missing_reasoning(i: int, rng: random.Random) -> dict[str, Any]:
    sample = _syn_qa(i, rng)
    sample["messages"][-1]["reasoning_content"] = ""      # a thinking-mode data error: must be dropped as invalid
    sample["metadata"]["task_type"] = "missing_reasoning"
    return sample


_SYNTHETIC_KINDS = (_syn_qa, _syn_tool_episode, _syn_parallel_calls, _syn_observation_as_user, _syn_long_tool_output,
                    _syn_missing_reasoning)


def synthetic_samples(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic synthetic conversations for smoke tests (own content, no benchmark material). Kinds cycle:
    qa, tool_episode (one call with JSON-string arguments), parallel_calls, observation_as_user (multi-query:
    episode mode must drop it with prefix_violation, per_turn must accept it), long_tool_output (exercises
    --max-length truncation) and missing_reasoning (invalid: dropped with reason invalid_sample)."""
    rng = random.Random(seed)
    return [_SYNTHETIC_KINDS[i % len(_SYNTHETIC_KINDS)](i, rng) for i in range(n)]


# --------------------------------------------------------------------------------------------------------- CLI
def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line_number, object) for every non-blank line; non-object lines are a DataFormatError."""
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataFormatError(f"{path}:{n}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise DataFormatError(f"{path}:{n}: expected a JSON object, got {type(obj).__name__}")
            yield n, obj


def load_tokenizer(path: Path | str) -> Any:
    """AutoTokenizer from a directory that ships the Qwen3.5-9B chat template (tokenizer files only, no weights)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if not tokenizer.chat_template:
        raise DataFormatError(f"tokenizer at {path} has no chat template")
    if tokenizer.encode(IM_END_TOKEN, add_special_tokens=False) != [IM_END_ID]:
        raise DataFormatError(f"tokenizer at {path} does not map {IM_END_TOKEN} to {IM_END_ID}; not the Qwen3.5 tokenizer")
    return tokenizer


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=30, check=False).stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_metadata(args: argparse.Namespace, tokenizer: Any, opts: BuildOptions) -> dict[str, Any]:
    """Provenance block of the report: command, code commit, tokenizer/template identity, library versions, host."""
    import transformers

    return {"command": sys.argv, "started": _utc_now(), "code_commit": _git_commit(),
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "tokenizer_class": type(tokenizer).__name__, "transformers": transformers.__version__,
            "python": sys.version.split()[0], "hostname": socket.gethostname(), "platform": platform.platform(),
            "input": str(Path(args.inp).resolve()), "output": str(Path(args.out).resolve()),
            "options": dataclasses.asdict(opts)}


def convert_file(inp: Path, tokenizer: Any, opts: BuildOptions, *, strict: bool = False,
                 limit: int | None = None) -> tuple[list[dict[str, Any]], list[DropRecord], int]:
    """Samples JSONL -> (rows, dropped, n_samples_read). Invalid samples are recorded (or raised with strict)."""
    rows: list[dict[str, Any]] = []
    dropped: list[DropRecord] = []
    n_samples = 0
    for line_no, obj in iter_jsonl(inp):
        if limit is not None and n_samples >= limit:
            break
        n_samples += 1
        try:
            sample = Sample.from_dict(obj, default_id=f"line{line_no}")
            new_rows, new_dropped = build_rows(sample, tokenizer, opts)
        except SampleValidationError as exc:
            if strict:
                raise
            meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else obj
            dropped.append(DropRecord(str(meta.get("id", f"line{line_no}")), -1, exc.reason, str(exc)))
            continue
        rows.extend(new_rows)
        dropped.extend(new_dropped)
    return rows, dropped, n_samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m apex_train.data_format", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, help="samples JSONL (one sample object per line)")
    p.add_argument("--out", type=Path, help="rows file to write: .parquet or .jsonl")
    p.add_argument("--tokenizer", type=Path, help="tokenizer directory (Qwen3.5-9B tokenizer + chat template)")
    p.add_argument("--mode", choices=MODES, default="episode", help="episode: one row per sample, all assistant turns "
                   "trained; per_turn: one row per assistant turn (observation-as-user loops)")
    p.add_argument("--max-length", type=int, default=None, help="row token cap: tool outputs are truncated (longest first, "
                   "head+tail) until the row fits, otherwise the row is dropped and recorded")
    p.add_argument("--report", type=Path, default=None, help="stats JSON (metadata, histograms, dropped rows with reasons)")
    p.add_argument("--no-thinking", action="store_true", help="render non-thinking rows (enable_thinking=False; samples "
                   "must carry no reasoning_content; the empty think block is prompt, never target)")
    p.add_argument("--turn-cap", type=int, default=None, help="per_turn: keep only the last N assistant turns per episode")
    p.add_argument("--turns-per-episode", type=int, default=None, help="per_turn: train K turns per episode: the latest "
                   "turn whose row fits --max-length plus K-1 random earlier fitting turns (seeded by --seed and the sample id)")
    p.add_argument("--min-tool-tokens", type=int, default=64, help="floor per truncated tool output (head+tail)")
    p.add_argument("--truncate-user-observations", action="store_true", help="also truncate user messages after the "
                   "first one (observation-as-user loops); the task statement is never touched")
    p.add_argument("--strict", action="store_true", help="stop at the first invalid sample instead of recording it")
    p.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    p.add_argument("--write-synthetic", type=Path, default=None, help="write N deterministic synthetic samples to this "
                   "JSONL and exit (smoke-test input; own content)")
    p.add_argument("--n", type=int, default=20, help="number of synthetic samples for --write-synthetic")
    p.add_argument("--seed", type=int, default=0, help="seed of the synthetic generator and of --turns-per-episode")
    args = p.parse_args(argv)
    if args.write_synthetic is None and not (args.inp and args.out and args.tokenizer):
        p.error("--in, --out and --tokenizer are required (or use --write-synthetic)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code (0 iff at least one row was written, or the synthetic file was)."""
    args = parse_args(argv)
    if args.write_synthetic is not None:
        args.write_synthetic.parent.mkdir(parents=True, exist_ok=True)
        with args.write_synthetic.open("w", encoding="utf-8") as f:
            for s in synthetic_samples(args.n, args.seed):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"wrote {args.n} synthetic samples -> {args.write_synthetic}")
        return 0
    opts = BuildOptions(mode=args.mode, enable_thinking=not args.no_thinking, max_length=args.max_length,
                        turn_cap=args.turn_cap, turns_per_episode=args.turns_per_episode, seed=args.seed,
                        min_tool_tokens=args.min_tool_tokens,
                        truncate_roles=("tool", "user") if args.truncate_user_observations else ("tool",))
    t0 = time.time()
    tokenizer = load_tokenizer(args.tokenizer)
    meta = run_metadata(args, tokenizer, opts)
    if opts.max_length is None:
        print("note: no --max-length; rows are not length-checked (set the trainer's max_length >= the reported max)")
    rows, dropped, n_samples = convert_file(args.inp, tokenizer, opts, strict=args.strict, limit=args.limit)
    write_rows(rows, args.out)
    summary = summarise(rows, dropped)
    summary["n_samples"] = n_samples
    print(f"samples {n_samples} -> {format_summary(summary)}\nrows -> {args.out}")
    for d in dropped[:20]:
        print(f"  dropped {d.sample_id} turn {d.turn_index}: {d.reason}: {d.detail[:160]}")
    if len(dropped) > 20:
        print(f"  ... {len(dropped) - 20} more drop records in the report")
    if args.report:
        meta.update(finished=_utc_now(), duration_sec=round(time.time() - t0, 1))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"metadata": meta, "summary": summary,
                                           "dropped": [dataclasses.asdict(d) for d in dropped]}, indent=2, ensure_ascii=False))
        print(f"report -> {args.report}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
