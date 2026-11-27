"""Episode-level and QA-level quality gates of docs/DATA_PLAN.md section 6 with per-gate survival counts.

Gates run in a fixed order and an episode stops at its first failure, so the survival table is sequential
(n_in / n_dropped / n_out per gate, one table per row kind). Nothing is deleted: every dropped episode is returned as a
`Dropped` record with its gate and reason, and the caller writes those records next to the kept rows.

Gates:
  verdict (r_pass == 1: upstream label, TermGrade reward, our replay, or the TerminalHorizon handshake label),
  tamper (no flags), adapt (scaffold structure, data_format schema), exit (explicit submit / the task_complete
  handshake pair / final answer, and a consistent exit_reason), format_error_turn (no harness format-error user turn:
  episodes are dropped, never spliced), steps (<= max_steps when set: the stage-2 rollout caps), finish_length (no
  finish_reason == length), reasoning (non-empty reasoning_content on every assistant turn), think_length (<=
  max_think_tokens per turn when set: the retention caps), language (no CJK in reasoning for English tasks), parse
  (swe: vLLM qwen3 reasoning + qwen3_coder tool parser round trip, ported regexes; terminal: Harbor's
  TerminusJSONPlainParser without its auto-fixes; every kind: no `</think>` inside reasoning), thrashing (swe: no three
  identical consecutive commands that all fail), repetition (retention: no 50-token window repeated >= 3 times),
  constraints (constrained retention rows: an injected deterministic checker or a constraint verdict), render
  (data_format.build_rows; terminal rows pick their turns by the row cap and, when set, max_turn_completion_tokens;
  an episode with no turn within it fails as render_turn_too_long; with terminal_turn_roles, role-chosen turns are tried
  first and each row's provenance gets turn_role), row_length (agentic rows <= max_row_tokens,
  retention <= max_qa_tokens; rows are dropped, never truncated: harness-style observation truncation already happened).

Every kept row carries its provenance (schema.PROVENANCE_FIELDS, validated) plus the bookkeeping keys selection and
the data card use. Loaders put the source-specific provenance values (generator, label_provenance, oswt_subsource,
size_stratum, category, category_group, retention_part, patch sizes, hf_repo, dataset_license) into the task record's
`extra["provenance"]`; the filter adds the row-level ones (task_id, episode_id, fits_48k, decontam_flags, dup_group).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from apex_data.render.adapters import (
    MINI_SWE_FORMAT_ERROR_PREFIX, MINI_SWE_SUBMIT_COMMAND, TERMINUS_OUTPUT_LIMIT_PREFIX, TERMINUS_PARSE_ERROR_PREFIX,
    TERMINUS_WARNING_PREFIX, Adapted, AdapterError, adapt, episode_dict, is_terminus_confirmation, row_kind,
)
from apex_data.schema import PROVENANCE_FIELDS, validate_provenance
from apex_train.data_format import TURN_TOO_LONG, DropRecord, Message, SampleValidationError, build_rows

FAMILY_BY_KIND = {"swe_episode": "swe", "terminal_turn": "terminal", "retention": "qa"}
GATES = ("verdict", "tamper", "adapt", "exit", "format_error_turn", "steps", "finish_length", "reasoning", "think_length",
         "language", "parse", "thrashing", "repetition", "constraints", "render", "row_length")
CAP_TARGET, CAP_FALLBACK = 65_536, 49_152      # docs/DATA_PLAN.md section 4: 64K target, 48K if 64K fails T8

# vllm/parser/qwen3.py:51-66 (v0.31.0), ported verbatim as in logs/train_stack_check/probes/02_chat_template_format.py.
_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>]*)>(.*)</function>", re.DOTALL)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿ｦ-ﾟ]")


@dataclass
class FilterConfig:
    max_row_tokens: int = CAP_TARGET          # agentic row cap (the T8 gate decides 64K or 48K)
    max_qa_tokens: int = 12_288               # retention rows (plan 6.3)
    max_steps: int | None = None              # stage-2 rollout caps (50 SWE, 40 terminal); teacher rows: the row cap only
    max_think_tokens: int | None = None       # retention parts set 10K (math, science) or 6K (IF, chat)
    thrashing_repeats: int = 3                # identical consecutive failing commands that drop a SWE episode
    repetition_window: int = 50
    repetition_repeats: int = 3
    require_verdict: bool = True
    max_turn_completion_tokens: int | None = None   # trained tokens per assistant turn (BuildOptions); None = no limit
    terminal_turns_per_episode: int | None = None   # trained turns per terminal episode; None = the adapter's (2)
    # Role-chosen terminal turns: messages -> [(message index or data_format.LATEST_FITTING, role)] in the order
    # BuildOptions.priority_turns tries them (render.turn_roles.priority_turns); rows then carry provenance turn_role.
    terminal_turn_roles: Callable[[list[Message]], list[tuple[int, str]]] | None = None


@dataclass
class Rendered:
    """One trainer row (data_format.ROW_COLUMNS) plus its manifest entry (PROVENANCE_FIELDS and selection keys)."""

    row: dict[str, Any]
    provenance: dict[str, Any]


@dataclass
class Dropped:
    episode_id: str
    task_id: str
    row_kind: str
    gate: str
    reason: str
    detail: str = ""


@dataclass
class Survival:
    """Sequential survival counts for one row kind plus the detailed drop reasons and side statistics."""

    counts: dict[str, dict[str, int]] = field(default_factory=lambda: {g: {"n_in": 0, "n_dropped": 0, "n_out": 0} for g in GATES})
    reasons: Counter = field(default_factory=Counter)
    stats: Counter = field(default_factory=Counter)

    def record(self, gate: str, dropped: bool) -> None:
        c = self.counts[gate]
        c["n_in"] += 1
        c["n_dropped" if dropped else "n_out"] += 1

    def table(self) -> list[dict[str, Any]]:
        return [{"gate": g, **c} for g, c in self.counts.items()]

    def to_dict(self) -> dict[str, Any]:
        return {"gates": self.table(), "reasons": dict(sorted(self.reasons.items())), "stats": dict(sorted(self.stats.items()))}


class GateFailure(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass
class Ctx:
    ep: dict[str, Any]
    verdict: dict[str, Any] | None
    task: dict[str, Any] | None
    kind: str
    cfg: FilterConfig
    tokenizer: Any
    adapted: Adapted | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_drops: list[DropRecord] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    turn_roles: dict[int, str] | None = None     # turn_index -> role when FilterConfig.terminal_turn_roles is set

    @property
    def assistant_turns(self) -> list[dict[str, Any]]:
        return [m for m in self.ep.get("messages", []) if m.get("role") == "assistant"]

    @property
    def messages(self) -> list[Message]:
        assert self.adapted is not None
        return self.adapted.sample.messages


# ---------------------------------------------------------------------------------------------- parser ports
def _trim_wrapping_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def qwen3_parse_calls(text: str) -> list[dict[str, Any]]:
    """Tool-call half of vLLM's Qwen3 engine parser (`qwen3_coder`): XML calls with string parameter values."""
    calls = []
    for body in _TOOL_CALL_RE.findall(text):
        m = _FUNC_RE.search(body)
        if m is None:
            calls.append({"name": None, "arguments": {}})
            continue
        calls.append({"name": m.group(1), "arguments": {k: _trim_wrapping_newlines(v) for k, v in _PARAM_RE.findall(m.group(2))}})
    return calls


def qwen3_roundtrip(completion: str, msg: Message) -> str | None:
    """Mismatch reason (or None) between the rendered completion of an assistant turn and the turn itself, as the
    qwen3 reasoning parser and the qwen3_coder tool parser would recover them at inference."""
    reasoning = (msg.reasoning_content or "").strip()
    if "</think>" in reasoning or "<tool_call>" in reasoning:
        return "tag_in_reasoning"
    if "<tool_call>" in msg.content or "</tool_call>" in msg.content:
        return "tag_in_content"
    head = f"{reasoning}\n</think>\n\n"
    if not completion.startswith(head):
        return "reasoning_boundary"
    rest = completion[len(head):].removesuffix("\n").removesuffix("<|im_end|>")
    calls = msg.tool_calls or []
    parsed = qwen3_parse_calls(rest)
    if len(parsed) != len(calls) or rest.count("<tool_call>") != len(calls):
        return "tool_call_count"
    content_part = rest.split("<tool_call>", 1)[0] if calls else rest
    if content_part.rstrip("\n") != msg.content.strip():
        return "content_boundary"
    for got, want in zip(parsed, calls, strict=True):
        if got["name"] != want.name or list(got["arguments"]) != list(want.arguments):
            return "tool_call_name_or_keys"
        for k, v in want.arguments.items():
            text = got["arguments"][k]
            if isinstance(v, str):
                if text != v:
                    return "string_argument"
            else:
                try:
                    if json.loads(text) != v:
                        return "typed_argument"
                except json.JSONDecodeError:
                    return "typed_argument"
    return None


@dataclass
class TerminusParse:
    error: str
    warnings: list[str]
    commands: list[tuple[str, float]]
    task_complete: bool


def _terminus_extract_json(response: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    json_start = json_end = -1
    brace_count = 0
    in_string = escape_next = False
    for i, char in enumerate(response):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
                    break
    if json_start == -1 or json_end == -1:
        return "", ["No valid JSON object found"]
    if response[:json_start].strip():
        warnings.append("Extra text detected before JSON object")
    if response[json_end:].strip():
        warnings.append("Extra text detected after JSON object")
    return response[json_start:json_end], warnings


def terminus_parse(response: str) -> TerminusParse:
    """Port of Harbor's `TerminusJSONPlainParser._try_parse_response` (the strict path: no auto-fixes, because a turn
    the harness had to auto-correct is not a target worth training)."""
    json_content, warnings = _terminus_extract_json(response)
    if not json_content:
        return TerminusParse("No valid JSON found in response", warnings, [], False)
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as exc:
        return TerminusParse(f"Invalid JSON: {exc}", warnings, [], False)
    if not isinstance(data, dict):
        return TerminusParse("Response must be a JSON object", warnings, [], False)
    missing = [f for f in ("analysis", "plan", "commands") if f not in data]
    if missing:
        return TerminusParse(f"Missing required fields: {', '.join(missing)}", warnings, [], False)
    for f in ("analysis", "plan"):
        if not isinstance(data.get(f, ""), str):
            warnings.append(f"Field '{f}' should be a string")
    commands_data = data.get("commands", [])
    if not isinstance(commands_data, list):
        return TerminusParse("Field 'commands' must be an array", warnings, [], False)
    positions = {f: m.start() for f in ("analysis", "plan", "commands") if (m := re.search(f'"({f})"\\s*:', json_content))}
    if len(positions) >= 2 and sorted(positions, key=positions.get) != list(positions):
        warnings.append("Fields appear in wrong order")
    is_complete = data.get("task_complete", False)
    if isinstance(is_complete, str):
        is_complete = is_complete.lower() in ("true", "1", "yes")
    elif not isinstance(is_complete, bool):
        warnings.append("Field 'task_complete' should be a boolean or string")
        is_complete = bool(is_complete)
    commands: list[tuple[str, float]] = []
    for i, cmd in enumerate(commands_data):
        if not isinstance(cmd, dict):
            return TerminusParse(f"Command {i + 1} must be an object", warnings, [], is_complete)
        if "keystrokes" not in cmd:
            return TerminusParse(f"Command {i + 1} missing required 'keystrokes' field", warnings, [], is_complete)
        keystrokes = cmd["keystrokes"]
        if not isinstance(keystrokes, str):
            return TerminusParse(f"Command {i + 1} 'keystrokes' must be a string", warnings, [], is_complete)
        duration = cmd.get("duration", 1.0)
        if "duration" not in cmd:
            warnings.append(f"Command {i + 1}: Missing duration field, using default 1.0")
        elif not isinstance(duration, (int, float)) or isinstance(duration, bool):
            warnings.append(f"Command {i + 1}: Invalid duration value, using default 1.0")
            duration = 1.0
        if set(cmd) - {"keystrokes", "duration"}:
            warnings.append(f"Command {i + 1}: Unknown fields: {', '.join(sorted(set(cmd) - {'keystrokes', 'duration'}))}")
        if i < len(commands_data) - 1 and keystrokes and not keystrokes.endswith("\n"):
            warnings.append(f"Command {i + 1} should end with newline when followed by another command")
        commands.append((keystrokes, float(duration)))
    return TerminusParse("", warnings, commands, bool(is_complete))


def repeated_window(ids: list[int], window: int, repeats: int) -> bool:
    """True iff some `window`-token span occurs at least `repeats` times (the plan 6.3 repetition-loop rule)."""
    if len(ids) < window:
        return False
    seen: Counter = Counter()
    for i in range(len(ids) - window + 1):
        key = tuple(ids[i:i + window])
        seen[key] += 1
        if seen[key] >= repeats:
            return True
    return False


# ------------------------------------------------------------------------------------------------------- gates
def _submit_call(m: Message) -> bool:
    return bool(m.tool_calls) and len(m.tool_calls) == 1 and m.tool_calls[0].name == "bash" and \
        str(m.tool_calls[0].arguments.get("command", "")).strip() == MINI_SWE_SUBMIT_COMMAND


def gate_verdict(c: Ctx) -> None:
    if c.verdict is None:
        if c.cfg.require_verdict:
            raise GateFailure("verdict_missing")
        return
    if c.verdict.get("r_pass") != 1:
        raise GateFailure("verdict_fail", f"r_pass={c.verdict.get('r_pass')} verifier={c.verdict.get('verifier')}")


def gate_tamper(c: Ctx) -> None:
    flags = (c.verdict or {}).get("tamper_flags") or []
    if flags:
        raise GateFailure("tamper", ",".join(map(str, flags)))


def gate_adapt(c: Ctx) -> None:
    meta = {}
    if c.task:
        meta = {"source": c.task.get("source"), "language": c.task.get("language"),
                "license": (c.task.get("licence") or {}).get("repo_spdx") or (c.task.get("licence") or {}).get("dataset")}
    try:
        c.adapted = adapt(c.ep, meta)
    except AdapterError as exc:
        raise GateFailure(f"adapt_{exc.reason}", str(exc)) from exc
    except SampleValidationError as exc:
        raise GateFailure("invalid_sample", str(exc)) from exc
    for k, v in c.adapted.notes.items():
        if isinstance(v, (int, bool)) and v:
            c.stats[f"adapt_{k}"] += int(v)


def gate_exit(c: Ctx) -> None:
    exit_reason = c.ep.get("exit_reason")
    last = c.messages[-1]
    if c.kind == "swe_episode":
        if exit_reason != "submit":
            raise GateFailure("exit_reason", str(exit_reason))
        if not _submit_call(last):
            raise GateFailure("no_submit", "last assistant turn is not the lone submit command")
        if any(_submit_call(m) for m in c.messages[:-1] if m.role == "assistant"):
            raise GateFailure("submit_not_last", "submit command issued before the final turn")
    elif c.kind == "terminal_turn":
        if exit_reason != "task_complete":
            raise GateFailure("exit_reason", str(exit_reason))
        msgs = c.messages
        idx = [i for i, m in enumerate(msgs) if m.role == "assistant"]
        if len(idx) < 2 or idx[-1] - idx[-2] != 2 or not is_terminus_confirmation(msgs[idx[-1] - 1].content):
            raise GateFailure("no_task_complete", "missing the task_complete confirmation pair")
        for i in idx[-2:]:
            p = terminus_parse(msgs[i].content)
            if p.error or not p.task_complete:
                raise GateFailure("no_task_complete", p.error or f"turn {i} does not set task_complete")
    elif exit_reason != "submit":
        raise GateFailure("exit_reason", str(exit_reason))


def gate_format_error_turn(c: Ctx) -> None:
    users = [m.content for m in c.messages[1:] if m.role == "user"]
    if c.kind == "swe_episode":
        bad = [u for u in users if u.lstrip().startswith(MINI_SWE_FORMAT_ERROR_PREFIX)]
    elif c.kind == "terminal_turn":
        bad = [u for u in users if u.startswith((TERMINUS_PARSE_ERROR_PREFIX, TERMINUS_OUTPUT_LIMIT_PREFIX))]
        c.stats["terminus_warning_turns"] += sum(u.startswith(TERMINUS_WARNING_PREFIX) for u in users)
    else:
        bad = []
    if bad:
        raise GateFailure("format_error_turn", bad[0][:120])


def gate_steps(c: Ctx) -> None:
    if c.cfg.max_steps is None:
        return
    n = len([m for m in c.messages if m.role == "assistant"])
    steps = max(int(c.ep.get("steps") or 0), n)
    if steps > c.cfg.max_steps:
        raise GateFailure("too_many_steps", f"{steps} > {c.cfg.max_steps}")


def gate_finish_length(c: Ctx) -> None:
    if c.ep.get("exit_reason") == "token_cap":
        raise GateFailure("finish_length", "exit_reason token_cap")
    reasons = [m.get("finish_reason") for m in c.assistant_turns]
    reasons += list((c.ep.get("extra") or {}).get("finish_reasons") or [])
    if any(r == "length" for r in reasons):
        raise GateFailure("finish_length")


def gate_reasoning(c: Ctx) -> None:
    for i, m in enumerate(c.messages):
        if m.role == "assistant" and not (m.reasoning_content or "").strip():
            raise GateFailure("reasoning_missing", f"messages[{i}]")


def gate_think_length(c: Ctx) -> None:
    for i, m in enumerate(c.messages):
        if m.role == "assistant":
            n = len(c.tokenizer.encode(m.reasoning_content or "", add_special_tokens=False))
            c.stats["think_tokens_total"] += n
            c.stats["think_turns"] += 1
            if c.cfg.max_think_tokens is not None and n > c.cfg.max_think_tokens:
                raise GateFailure("think_too_long", f"messages[{i}]: {n} > {c.cfg.max_think_tokens}")


def gate_language(c: Ctx) -> None:
    natural = ((c.task or {}).get("extra") or {}).get("natural_language")
    task_text = next((m.content for m in c.messages if m.role == "user"), "")
    english = (natural in (None, "en")) and not CJK_RE.search(task_text)
    if not english:
        c.stats["non_english_tasks"] += 1
        return
    for i, m in enumerate(c.messages):
        if m.role == "assistant" and CJK_RE.search(m.reasoning_content or ""):
            raise GateFailure("cjk_in_reasoning", f"messages[{i}]")


def _completions(c: Ctx) -> list[tuple[int, str]]:
    """(message index, rendered completion string) per assistant turn, from the shipped template."""
    sample = c.adapted.sample
    msgs, tools = sample.template_messages(), sample.tools
    out = []
    for i in sample.assistant_indices():
        prompt = c.tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        full = c.tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False)
        if not full.startswith(prompt):
            raise GateFailure("parse_prefix", f"messages[{i}]: generation prompt is not a prefix of the render")
        out.append((i, full[len(prompt):]))
    return out


def gate_parse(c: Ctx) -> None:
    msgs = c.messages
    for i, m in enumerate(msgs):   # vLLM's reasoning parser closes the think block at the first </think>
        if m.role == "assistant" and "</think>" in (m.reasoning_content or ""):
            raise GateFailure("parse_tag_in_reasoning", f"messages[{i}]")
    if c.kind == "swe_episode":
        for i, completion in _completions(c):
            reason = qwen3_roundtrip(completion, msgs[i])
            if reason:
                raise GateFailure(f"parse_{reason}", f"messages[{i}]")
    elif c.kind == "terminal_turn":
        for i, m in enumerate(msgs):
            if m.role != "assistant":
                continue
            p = terminus_parse(m.content)
            if p.error:
                raise GateFailure("parse_terminus", f"messages[{i}]: {p.error[:100]}")
            c.stats["terminus_parser_warnings"] += len(p.warnings)


def _returncode(observation: str) -> int | None:
    m = re.match(r'\{\s*"returncode":\s*(-?\d+)', observation or "")
    return int(m.group(1)) if m else None


def gate_thrashing(c: Ctx) -> None:
    """swe: `thrashing_repeats` consecutive turns issuing the same single command that fails every time (plan 3.2)."""
    if c.kind != "swe_episode":
        return
    msgs, run, last = c.messages, 0, None
    for i, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        cmd = json.dumps([tc.arguments for tc in m.tool_calls], sort_keys=True)
        obs = msgs[i + 1].content if i + 1 < len(msgs) and msgs[i + 1].role == "tool" else ""
        failed = (_returncode(obs) or 0) != 0
        run = run + 1 if failed and cmd == last else (1 if failed else 0)
        last = cmd if failed else None
        if run >= c.cfg.thrashing_repeats:
            raise GateFailure("thrashing", f"messages[{i}]: {run} identical failing commands in a row")


def gate_repetition(c: Ctx) -> None:
    if c.kind != "retention":
        return
    for i, m in enumerate(c.messages):
        if m.role == "assistant":
            ids = c.tokenizer.encode(f"{m.reasoning_content or ''}\n\n{m.content}", add_special_tokens=False)
            if repeated_window(ids, c.cfg.repetition_window, c.cfg.repetition_repeats):
                raise GateFailure("repetition_loop", f"messages[{i}]")


def gate_constraints(c: Ctx, checker: Callable[[list[dict[str, Any]], list[Message]], list[str]] | None = None) -> None:
    if c.kind != "retention":
        return
    constraints = (c.ep.get("extra") or {}).get("constraints") or []
    if not constraints:
        return
    c.stats["constrained_rows"] += 1
    if checker is not None:
        failed = checker(constraints, c.messages)
        if failed:
            raise GateFailure("constraint_fail", ",".join(map(str, failed))[:200])
        return
    if (c.verdict or {}).get("verifier") != "constraint":
        raise GateFailure("constraint_unverified", f"verifier={(c.verdict or {}).get('verifier')}")


def gate_render(c: Ctx) -> None:
    opts = c.adapted.options
    if opts.turns_per_episode is not None:        # the turn choice depends on what fits the cap
        opts = replace(opts, max_length=c.cfg.max_row_tokens)
    if c.cfg.max_turn_completion_tokens is not None:
        opts = replace(opts, max_turn_completion_tokens=c.cfg.max_turn_completion_tokens)
    roles = None
    if c.kind == "terminal_turn":
        if c.cfg.terminal_turns_per_episode is not None:
            opts = replace(opts, turns_per_episode=c.cfg.terminal_turns_per_episode)
        if c.cfg.terminal_turn_roles is not None:
            roles = c.cfg.terminal_turn_roles(c.messages)
            opts = replace(opts, priority_turns=tuple(i for i, _ in roles))
    sources: dict[int, str] = {}
    try:
        c.rows, c.row_drops = build_rows(c.adapted.sample, c.tokenizer, opts, sources=sources)
    except SampleValidationError as exc:
        raise GateFailure("invalid_sample", str(exc)) from exc
    if roles is not None:
        role_of = dict(roles)
        c.turn_roles = {i: role_of[i] if how == "priority" else how for i, how in sources.items()}
    for d in c.row_drops:
        c.stats[f"row_drop_{d.reason}"] += 1
    if not c.rows:
        fail = next((d for d in c.row_drops if d.reason == TURN_TOO_LONG), None) or \
            next((d for d in c.row_drops if d.reason not in ("not_selected", "not_candidate")), None)
        raise GateFailure(f"render_{fail.reason if fail else 'no_rows'}", fail.detail[:200] if fail else "")
    if opts.mode == "episode" and c.row_drops:
        raise GateFailure(f"render_{c.row_drops[0].reason}", c.row_drops[0].detail[:200])


def gate_row_length(c: Ctx) -> None:
    cap = c.cfg.max_qa_tokens if c.kind == "retention" else c.cfg.max_row_tokens
    over = [r for r in c.rows if r["n_tokens"] > cap]
    if not over:
        return
    c.stats["rows_over_cap"] += len(over)
    if c.adapted.options.mode == "episode" or len(over) == len(c.rows):
        raise GateFailure("too_long", f"{max(r['n_tokens'] for r in over)} > {cap}")
    c.rows = [r for r in c.rows if r["n_tokens"] <= cap]


# ------------------------------------------------------------------------------------------------- provenance
def _sha256_messages(messages: Iterable[Message]) -> str:
    return hashlib.sha256(json.dumps([m.to_dict() for m in messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def provenance_for(c: Ctx, row: dict[str, Any]) -> dict[str, Any]:
    """The manifest entry of a row: schema.PROVENANCE_FIELDS (validated) plus the keys selection and the data card use."""
    t = c.task or {}
    task_id = c.ep["task_id"]
    screens = t.get("screens") or {}
    given = (t.get("extra") or {}).get("provenance") or {}
    unknown = set(given) - set(PROVENANCE_FIELDS)
    if unknown:
        raise ValueError(f"task {task_id}: unknown provenance keys {sorted(unknown)}")
    fields = {k: None for k in PROVENANCE_FIELDS}
    fields.update(row_kind=c.kind, task_id=task_id, episode_id=c.ep["episode_id"], hf_revision=t.get("hf_revision"),
                  repo=t.get("repo"), repo_license=(t.get("licence") or {}).get("repo_spdx"),
                  dataset_license=(t.get("licence") or {}).get("dataset"), dup_group=screens.get("dedup_group"),
                  fits_48k=row["n_tokens"] <= CAP_FALLBACK,
                  decontam_flags=json.dumps({k: screens[k] for k in ("d1", "d2", "d3", "d4", "d5") if k in screens},
                                            sort_keys=True))
    fields.update(given)
    validate_provenance(fields)
    msgs = c.messages if row["turn_index"] < 0 else c.messages[:row["turn_index"] + 1]
    out = {**fields, "row_key": f"{row['sample_id']}:{row['turn_index']}", "sample_id": row["sample_id"],
           "turn_index": row["turn_index"], "family": t.get("family") or FAMILY_BY_KIND[c.kind],
           "language": t.get("language"), "n_tokens": row["n_tokens"], "n_assistant_tokens": row["n_assistant_tokens"],
           "n_steps": len([m for m in c.messages if m.role == "assistant"]), "sha256": _sha256_messages(msgs)}
    if c.turn_roles is not None:
        out["turn_role"] = c.turn_roles[row["turn_index"]]
    return out


# ------------------------------------------------------------------------------------------------------ driver
GATE_FUNCS: dict[str, Callable[..., None]] = {
    "verdict": gate_verdict, "tamper": gate_tamper, "adapt": gate_adapt, "exit": gate_exit,
    "format_error_turn": gate_format_error_turn, "steps": gate_steps, "finish_length": gate_finish_length,
    "reasoning": gate_reasoning, "think_length": gate_think_length, "language": gate_language, "parse": gate_parse,
    "thrashing": gate_thrashing, "repetition": gate_repetition, "constraints": gate_constraints, "render": gate_render,
    "row_length": gate_row_length,
}


@dataclass
class FilterResult:
    kept: list[Rendered]
    dropped: list[Dropped]
    survival: dict[str, Survival]


def filter_episodes(episodes: Iterable[Any], verdicts: dict[str, dict[str, Any]], tasks: dict[str, dict[str, Any]],
                    tokenizer: Any, cfg: FilterConfig | None = None, *,
                    constraint_checker: Callable[..., list[str]] | None = None, skip_gates: tuple[str, ...] = (),
                    on_progress: Callable[[int, int, int], None] | None = None) -> FilterResult:
    """Run the gates over `episodes` (Episode objects or dicts). `verdicts` and `tasks` are keyed by episode_id and
    task_id. `skip_gates` names gates to pass through unchecked (("verdict",) keeps format-valid failures for the
    stage-2 control arm)."""
    cfg = cfg or FilterConfig()
    if any(g not in GATES for g in skip_gates):
        raise ValueError(f"skip_gates must be a subset of {GATES}, got {skip_gates}")
    kept: list[Rendered] = []
    dropped: list[Dropped] = []
    survival: dict[str, Survival] = {}
    for n, episode in enumerate(episodes, 1):
        ep = episode_dict(episode)
        try:
            kind = row_kind(ep)
        except AdapterError as exc:
            dropped.append(Dropped(str(ep.get("episode_id")), str(ep.get("task_id")), "?", "adapt", exc.reason, str(exc)))
            continue
        surv = survival.setdefault(kind, Survival())
        c = Ctx(ep, verdicts.get(ep["episode_id"]), tasks.get(ep["task_id"]), kind, cfg, tokenizer)
        failure: GateFailure | None = None
        for gate in GATES:
            if gate in skip_gates:
                surv.record(gate, False)
                continue
            try:
                if gate == "constraints":
                    gate_constraints(c, constraint_checker)
                else:
                    GATE_FUNCS[gate](c)
            except GateFailure as exc:
                failure = exc
                surv.record(gate, True)
                surv.reasons[exc.reason] += 1
                dropped.append(Dropped(ep["episode_id"], ep["task_id"], kind, gate, exc.reason, exc.detail))
                break
            surv.record(gate, False)
        surv.stats.update(c.stats)
        if failure is None:
            kept.extend(Rendered(row, provenance_for(c, row)) for row in c.rows)
            surv.stats["rows_kept"] += len(c.rows)
        if on_progress:
            on_progress(n, len(kept), len(dropped))
    return FilterResult(kept, dropped, survival)
