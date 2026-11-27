"""Episode -> `data_format.Sample` per scaffold, shaped exactly as each evaluation harness presents the history.

Sources: logs/train_stack_check/harness_request_shapes/FINDINGS.md (captured request bodies of the real harnesses),
logs/train_stack_check_qwen35_9b/chat_template_format/FINDINGS.md (split rule per harness for the Qwen3.5-9B template),
docs/DATA_PLAN.md sections 2.3-2.4, 3.2-3.4.

| Row kind      | Scaffold          | Sample shape and build options                                                       |
|---------------|-------------------|--------------------------------------------------------------------------------------|
| swe_episode   | mini_swe_toolcall | system + task user turn, assistant turns (reasoning + `bash` call), `role=tool` JSON |
|               |                   | observations. One row per episode, every assistant turn trained with its reasoning  |
|               |                   | (one user message, so the template keeps every think block).                        |
| terminal_turn | terminus2         | no system message; user #1 = protocol prompt + task + screen, assistant = JSON text, |
|               |                   | later user turns = terminal screens (<= 10,000 bytes, Terminus marker). Per-turn     |
|               |                   | rows (`per_turn`, `turns_per_episode` 2): every observation is a user query, so the |
|               |                   | template drops earlier reasoning by itself, which is what Terminus-2 sends. Replies  |
|               |                   | to the task_complete confirmation prompt are not trainable except in a seeded       |
|               |                   | TERMINAL_CONFIRMATION_SHARE of episodes (see `adapt_terminus`).                      |
| retention     | chat              | optional system + one user prompt + one assistant answer with reasoning. One row.   |

Observation truncation reproduces the harness code byte for byte (`mini_swe_observation`, `terminus_limit_output`) and
is applied only to raw observations (`episode.extra["observation_form"] == "raw"`, or dict-valued tool contents);
harness-faithful transcripts are left untouched, because re-applying either rule to an already truncated observation
changes it. The adapters never set `max_length`: the caller passes the row cap (filters.FilterConfig.max_row_tokens).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from apex_data import paths
from apex_data.schema import Episode
from apex_train.data_format import BuildOptions, Sample

ROW_KIND_BY_SCAFFOLD = {"mini_swe_toolcall": "swe_episode", "terminus2": "terminal_turn", "chat": "retention"}
TERMINAL_TURNS_PER_EPISODE = 2          # docs/DATA_PLAN.md 3.3 (revision 4.3.3): latest fitting turn + one random earlier
# Share of terminal episodes whose reply to the task_complete confirmation prompt may be trained (changelog 4.3.8). The
# reply only repeats `"task_complete": true`; as the latest fitting turn it took 845 of 1,000 episodes' latest rows and
# half the terminal sequence tokens. A small seeded share keeps the double handshake of Terminus-2 in the data.
TERMINAL_CONFIRMATION_SHARE = 0.10

# mini-swe-agent 2.4.6 toolcall mode (config/mini.yaml, used by the SWE-bench Pro driver; captured verbatim in
# logs/train_stack_check/harness_request_shapes/tool_schemas.json).
MINI_SWE_SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer.\n"
MINI_SWE_TOOLS: list[dict[str, Any]] = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                   "required": ["command"]}}}]
MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_SWE_FORMAT_ERROR_PREFIX = "Tool call error:"
MINI_SWE_ELIDE_AT = 10_000          # observation_template: `output | length < 10000` keeps the whole output
MINI_SWE_KEEP = 5_000               # else output[:5000] + output[-5000:] with elided_chars

# Harbor 0.24.0 Terminus-2 (harbor/agents/terminus_2/terminus_2.py, parser_name json).
TERMINUS_MAX_BYTES = 10_000
TERMINUS_CONFIRMATION = ("Are you sure you want to mark the task as complete? This will trigger your solution to be graded "
                         'and you won\'t be able to make any further corrections. If so, include "task_complete": true '
                         "in your JSON response again.")
TERMINUS_PARSE_ERROR_PREFIX = "Previous response had parsing errors:"
TERMINUS_WARNING_PREFIX = "Previous response had warnings:"
TERMINUS_OUTPUT_LIMIT_PREFIX = "ERROR!! NONE of the actions you just requested were performed"

HARNESS_CAPTURES = paths.CONFIGS / "train_stack_check" / "harness_request_shapes"


class AdapterError(ValueError):
    """The episode cannot be rendered for its scaffold; `reason` is the drop-record key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Adapted:
    """A render-ready sample: the Sample, the build options its harness implies, the row kind and bookkeeping."""

    sample: Sample
    options: BuildOptions
    row_kind: str
    episode_id: str
    task_id: str
    notes: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------- harness truncation rules
def _jinja_tojson(value: Any) -> str:
    """Jinja2's default `tojson` (htmlsafe_json_dumps with sort_keys=True): json.dumps + HTML-safe escapes."""
    return (json.dumps(value, sort_keys=True).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("'", "\\u0027"))


def mini_swe_observation(returncode: int, output: str, exception_info: str = "") -> str:
    """Byte-exact reproduction of mini-swe-agent 2.4.6 `model.observation_template` (config/mini.yaml): the JSON
    observation the agent appends as the `tool` message; outputs of >= 10,000 characters keep head and tail 5,000."""
    exc = f', "exception_info": {_jinja_tojson(exception_info)}' if exception_info else ""
    if len(output) < MINI_SWE_ELIDE_AT:
        return f'{{\n  "returncode": {returncode},\n  "output": {_jinja_tojson(output)}{exc}\n}}'
    return (f'{{\n  "returncode": {returncode},\n  "output_head": {_jinja_tojson(output[:MINI_SWE_KEEP])},\n'
            f'  "output_tail": {_jinja_tojson(output[-MINI_SWE_KEEP:])},\n  "elided_chars": {len(output) - MINI_SWE_ELIDE_AT},\n'
            f'  "warning": "Output too long."{exc}\n}}')


def terminus_limit_output(output: str, max_bytes: int = TERMINUS_MAX_BYTES) -> str:
    """Verbatim port of `Terminus2._limit_output_length`: first and last max_bytes // 2 bytes around the marker."""
    if len(output.encode("utf-8")) <= max_bytes:
        return output
    portion_size = max_bytes // 2
    output_bytes = output.encode("utf-8")
    first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted_bytes = len(output_bytes) - len(first_portion.encode("utf-8")) - len(last_portion.encode("utf-8"))
    return (f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}")


def terminus_confirmation(terminal_output: str) -> str:
    """`Terminus2._get_completion_confirmation_message` for the json parser."""
    return f"Current terminal state:\n{terminal_output}\n\n{TERMINUS_CONFIRMATION}"


def is_terminus_confirmation(text: str) -> bool:
    return text.startswith("Current terminal state:\n") and text.rstrip().endswith(TERMINUS_CONFIRMATION)


# ------------------------------------------------------------------------------------------------------ helpers
def episode_dict(episode: Episode | dict[str, Any]) -> dict[str, Any]:
    if isinstance(episode, Episode):
        return {k: getattr(episode, k) for k in Episode.__dataclass_fields__}
    if not isinstance(episode, dict):
        raise AdapterError("invalid_episode", f"expected Episode or dict, got {type(episode).__name__}")
    return episode


def row_kind(episode: Episode | dict[str, Any]) -> str:
    """swe_episode, terminal_turn or retention, from the episode's scaffold."""
    kind = ROW_KIND_BY_SCAFFOLD.get(episode_dict(episode).get("scaffold", ""))
    if kind is None:
        raise AdapterError("unknown_scaffold", str(episode_dict(episode).get("scaffold")))
    return kind


def _text(content: Any) -> str:
    """Message content as one string (text parts concatenated as the template does); dicts are not text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return ""


def _messages(ep: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = ep.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise AdapterError("no_messages", ep.get("episode_id", "?"))
    return copy.deepcopy(msgs)


def _strip_trailing_observations(msgs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """A transcript may end with the observation of the last action (the harness records it; the model never answers
    it). Rows end with an assistant turn, so trailing non-assistant messages are cut and counted."""
    n = len(msgs)
    while msgs and msgs[-1].get("role") != "assistant":
        msgs.pop()
    if not msgs:
        raise AdapterError("no_assistant_turn")
    return msgs, n - len(msgs)


def _sample(msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None, ep: dict[str, Any], kind: str,
            meta: dict[str, Any] | None) -> Sample:
    base = {"id": ep["episode_id"], "source": ep.get("task_id", "").split(":", 1)[0] or None, "task_type": kind,
            "language": None, "difficulty": None, "license": None}
    base.update({k: v for k, v in (meta or {}).items() if k in base})
    return Sample.from_dict({"messages": msgs, "tools": tools, "metadata": base})


def _tool_names_and_params(tools: list[dict[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        props = (fn.get("parameters") or {}).get("properties") or {}
        out.append((fn.get("name"), tuple(sorted(props))))
    return out


# ------------------------------------------------------------------------------------------------------ adapters
def adapt_mini_swe(ep: dict[str, Any], meta: dict[str, Any] | None = None) -> Adapted:
    msgs = _messages(ep)
    if msgs[0].get("role") != "system":
        raise AdapterError("missing_system", "mini-swe-agent sends its system template first")
    notes: dict[str, Any] = {}
    if ep.get("tools"):
        if _tool_names_and_params(ep["tools"]) != _tool_names_and_params(MINI_SWE_TOOLS):
            raise AdapterError("tool_schema_mismatch", f"{_tool_names_and_params(ep['tools'])} != bash(command)")
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), dict):   # raw {returncode, output[, exception_info]}
            raw = m["content"]
            m["content"] = mini_swe_observation(int(raw.get("returncode", 0)), str(raw.get("output", "")),
                                                str(raw.get("exception_info") or ""))
            notes["raw_observations_rendered"] = notes.get("raw_observations_rendered", 0) + 1
    msgs, cut = _strip_trailing_observations(msgs)
    notes["trailing_messages_cut"] = cut
    sample = _sample(msgs, MINI_SWE_TOOLS, ep, "swe_episode", meta)
    return Adapted(sample, BuildOptions(mode="episode"), "swe_episode", ep["episode_id"], ep["task_id"], notes)


def confirmation_replies(msgs: list[dict[str, Any]]) -> list[int]:
    """Indices of assistant messages that answer Terminus-2's task_complete confirmation prompt."""
    return [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and i > 0 and msgs[i - 1].get("role") == "user"
            and is_terminus_confirmation(_text(msgs[i - 1].get("content")))]


def trains_confirmation(episode_id: str, share: float = TERMINAL_CONFIRMATION_SHARE) -> bool:
    """Seeded, order-independent draw: True for ~`share` of episode ids (sha256 of the id, not Python's hash)."""
    return int(hashlib.sha256(f"{paths.SEED}:{episode_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def adapt_terminus(ep: dict[str, Any], meta: dict[str, Any] | None = None,
                   turns_per_episode: int = TERMINAL_TURNS_PER_EPISODE, seed: int = paths.SEED,
                   confirmation_share: float = TERMINAL_CONFIRMATION_SHARE) -> Adapted:
    msgs = _messages(ep)
    if msgs[0].get("role") != "user":
        raise AdapterError("unexpected_system", "Terminus-2 sends no system message; the protocol prompt is user #1")
    notes: dict[str, Any] = {}
    if (ep.get("extra") or {}).get("observation_form") == "raw":
        for m in msgs[1:]:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = terminus_limit_output(m["content"])
        notes["raw_observations_limited"] = True
    for m in msgs:
        if m.get("role") not in ("user", "assistant"):
            raise AdapterError("unexpected_role", f"{m.get('role')} in a Terminus-2 transcript")
        if m.get("role") == "assistant" and m.get("tool_calls"):
            raise AdapterError("unexpected_tool_calls", "Terminus-2 is a plain-text JSON protocol")
    msgs, cut = _strip_trailing_observations(msgs)
    notes["trailing_messages_cut"] = cut
    confirmations = confirmation_replies(msgs)
    keep_confirmation = trains_confirmation(ep["episode_id"], confirmation_share)
    candidates = tuple(i for i, m in enumerate(msgs) if m.get("role") == "assistant"
                       and (keep_confirmation or i not in confirmations))
    notes.update(confirmation_replies=len(confirmations), confirmation_trainable=bool(confirmations) and keep_confirmation)
    sample = _sample(msgs, None, ep, "terminal_turn", meta)
    opts = BuildOptions(mode="per_turn", turns_per_episode=turns_per_episode, candidate_turns=candidates, seed=seed)
    return Adapted(sample, opts, "terminal_turn", ep["episode_id"], ep["task_id"], notes)


def adapt_chat(ep: dict[str, Any], meta: dict[str, Any] | None = None) -> Adapted:
    """Retention rows: one prompt, one answer with reasoning (multi-turn sources are cut to their last turn by the
    loader, so the template keeps the think block and the row trains exactly what a single query would produce)."""
    msgs = _messages(ep)
    for m in msgs:
        if m.get("role") not in ("system", "user", "assistant"):
            raise AdapterError("unexpected_role", f"{m.get('role')} in a chat transcript")
        if m.get("tool_calls"):
            raise AdapterError("unexpected_tool_calls", "retention rows carry no tools")
    msgs, cut = _strip_trailing_observations(msgs)
    if [m.get("role") for m in msgs if m.get("role") != "system"] != ["user", "assistant"]:
        raise AdapterError("chat_shape", "expected [system,] user, assistant")
    sample = _sample(msgs, None, ep, "retention", meta)
    constraints = (ep.get("extra") or {}).get("constraints") or []
    return Adapted(sample, BuildOptions(mode="episode"), "retention", ep["episode_id"], ep["task_id"],
                   {"trailing_messages_cut": cut, "constrained": bool(constraints), "n_constraints": len(constraints)})


def adapt(episode: Episode | dict[str, Any], meta: dict[str, Any] | None = None, **kwargs: Any) -> Adapted:
    """Dispatch on the scaffold. `meta` fills the Sample's flat metadata (source, language, difficulty, license);
    kwargs go to `adapt_terminus` (turns_per_episode, seed). Raises AdapterError (structure) or
    data_format.SampleValidationError (message-level schema)."""
    ep = episode_dict(episode)
    kind = row_kind(ep)
    if kind == "swe_episode":
        return adapt_mini_swe(ep, meta)
    if kind == "terminal_turn":
        return adapt_terminus(ep, meta, **kwargs)
    return adapt_chat(ep, meta)
