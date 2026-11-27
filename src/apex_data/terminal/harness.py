"""Harbor 0.24.0 Terminus-2 (parser `json`) as the terminal rows must reproduce it (docs/DATA_PLAN.md principle 3).

What our evaluation harness sends (harbor/agents/terminus_2/terminus_2.py, `run` and `_run_agent_loop`):
  * no system message; user #1 = `terminus-json-plain.txt` formatted with the task instruction and the initial terminal
    state (`_limit_output_length(get_incremental_output())`, i.e. "Current Terminal Screen:\\n..." capped at 10,000 bytes);
  * assistant turns = the model's text (reasoning is not replayed: interleaved_thinking is off);
  * later user turns = the terminal output after the turn's commands ("New Terminal Output:" or "Current Terminal
    Screen:", capped at 10,000 bytes), the task_complete confirmation ("Current terminal state:" ... "Are you sure ..."),
    "Previous response had warnings:" / "Previous response had parsing errors:" feedback, or the timeout template.

`TERMINUS_JSON_PLAIN_TEMPLATE` is a verbatim copy of the template file of Harbor 0.24.0 (sha256 below); `template_drift()`
compares it with the installed Harbor when `.venv/harbor` exists.

The corpora differ from it only in packaging, measured on the full files (build manifest, `harness_match`):
  * TermGrade `trajectories_messages` splits user #1 into a `system` message (the template text before "Task
    Description:") and a `user` message (the rest); its card states the two concatenate to the original first prompt.
    `match_first_prompt(user, system=...)` re-renders that concatenation under our template and checks byte identity.
  * TerminalHorizon-3K stores user #1 as one user message and the reasoning inside the assistant text as
    `<think>...</think>`; `split_think` moves it into `reasoning_content`.
A first prompt whose template head differs from ours but whose instruction and initial screen can be cut out is
re-rendered under our template (status `rerendered`, the difference recorded); one that cannot be parsed is dropped.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from apex_data import paths

HARBOR_VERSION = "0.24.0"
HARBOR_TEMPLATE_FILE = (paths.REPO / ".venv/harbor/lib/python3.12/site-packages/harbor/agents/terminus_2/templates/"
                        "terminus-json-plain.txt")
HARBOR_PARSER_FILE = HARBOR_TEMPLATE_FILE.parent.parent / "terminus_json_plain_parser.py"
TERMINUS_JSON_PLAIN_TEMPLATE = (
    'You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.\n'
    '\n'
    'Format your response as JSON with the following structure:\n'
    '\n'
    '{{\n'
    '  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",\n'
    '  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",\n'
    '  "commands": [\n'
    '    {{\n'
    '      "keystrokes": "ls -la\\n",\n'
    '      "duration": 0.1\n'
    '    }},\n'
    '    {{\n'
    '      "keystrokes": "cd project\\n",\n'
    '      "duration": 0.1\n'
    '    }}\n'
    '  ],\n'
    '  "task_complete": true\n'
    '}}\n'
    '\n'
    'Required fields:\n'
    '- "analysis": Your analysis of the current situation\n'
    '- "plan": Your plan for the next steps\n'
    '- "commands": Array of command objects to execute\n'
    '\n'
    'Optional fields:\n'
    '- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)\n'
    '\n'
    'Command object structure:\n'
    '- "keystrokes": String containing the exact keystrokes to send to the terminal (required)\n'
    '- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)\n'
    '\n'
    'IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:\n'
    '- You must end every command with a newline (\\n) or it will not execute.\n'
    '- For special key sequences, use tmux-style escape sequences:\n'
    '  - C-c for Ctrl+C\n'
    '  - C-d for Ctrl+D\n'
    '\n'
    'The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.\n'
    '\n'
    'It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.\n'
    '\n'
    'Important notes:\n'
    "- Each command's keystrokes are sent exactly as written to the terminal\n"
    "- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command\n"
    '- Extra text before or after the JSON will generate warnings but be tolerated\n'
    '- The JSON must be valid - use proper escaping for quotes and special characters within strings\n'
    '- Commands array can be empty if you want to wait without taking action\n'
    '\n'
    'Task Description:\n'
    '{instruction}\n'
    '\n'
    'Current terminal state:\n'
    '{terminal_state}\n'
)
TEMPLATE_SHA256 = "89a3dc3a15752b748a99fb907c9251ab31a464e13386583fb48476b078b54a34"

TASK_MARKER = "Task Description:\n"
STATE_MARKER = "\n\nCurrent terminal state:\n"
PROMPT_HEAD = TERMINUS_JSON_PLAIN_TEMPLATE.split(TASK_MARKER + "{instruction}")[0].format()   # `{{` -> `{`
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# Observation kinds (user turns after #1), by the prefix Harbor writes.
OBSERVATION_PREFIXES = (
    ("new_output", "New Terminal Output:\n"),
    ("current_screen", "Current Terminal Screen:\n"),
    ("confirmation", "Current terminal state:\n"),
    ("warnings", "Previous response had warnings:"),
    ("parse_error", "Previous response had parsing errors:"),
    ("output_limit_error", "ERROR!! NONE of the actions you just requested were performed"),
    ("timeout", "Previous command:\n"),
)


class HarnessError(ValueError):
    """The record cannot be brought into our harness shape; `reason` is the drop key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def first_prompt(instruction: str, terminal_state: str) -> str:
    """User #1 exactly as Harbor 0.24.0 Terminus-2 formats it (no MCP servers, no skills: our evaluation uses neither)."""
    return TERMINUS_JSON_PLAIN_TEMPLATE.format(instruction=instruction, terminal_state=terminal_state)


def split_first_prompt(text: str) -> tuple[str, str] | None:
    """(instruction, terminal_state) of a first prompt written with our template, else None. The state is cut at the
    last state marker, so an instruction that itself contains the marker text survives."""
    head = PROMPT_HEAD + TASK_MARKER
    if not text.startswith(head) or not text.endswith("\n"):
        return None
    body = text[len(head):-1]
    k = body.rfind(STATE_MARKER)
    if k < 0:
        return None
    return body[:k], body[k + len(STATE_MARKER):]


def _cut_task_and_state(text: str) -> tuple[str, str] | None:
    """Instruction and screen of a Terminus-style prompt whose template head is not ours (the text after the first task
    marker up to the last state marker, and the rest)."""
    i = text.find(TASK_MARKER)
    k = text.rfind(STATE_MARKER)
    if i < 0 or k < i:
        return None
    return text[i + len(TASK_MARKER):k], text[k + len(STATE_MARKER):].removesuffix("\n")


@dataclass
class FirstPromptMatch:
    status: str                           # identical | merged_system | rerendered
    prompt: str                           # user #1 under our template
    instruction: str
    terminal_state: str
    differences: list[str] = field(default_factory=list)


def match_first_prompt(user: str, system: str | None = None) -> FirstPromptMatch:
    """Bring a corpus' first prompt into our harness shape. `system` is TermGrade's split-off template head.
    Raises HarnessError("first_prompt_unrecognised") when neither our template nor the Terminus markers are found."""
    differences: list[str] = []
    text = user
    if system is not None:
        text = system + user
        differences.append("system_message_merged_into_user_1")
    parts = split_first_prompt(text)
    if parts is not None:
        prompt = first_prompt(*parts)
        if prompt != text:                # cannot happen for a well-formed split; kept as an assertion of the claim
            raise HarnessError("first_prompt_rerender_mismatch")
        return FirstPromptMatch("merged_system" if system is not None else "identical", prompt, parts[0], parts[1],
                                differences)
    cut = _cut_task_and_state(text)
    if cut is None:
        raise HarnessError("first_prompt_unrecognised", text[:80])
    if not text.startswith(PROMPT_HEAD):
        differences.append("template_head")
    if not text.endswith("\n"):
        differences.append("trailing_newline")
    return FirstPromptMatch("rerendered", first_prompt(*cut), cut[0], cut[1], differences)


def split_think(text: str) -> tuple[str, str]:
    """TerminalHorizon assistant text `<think>R</think>C` -> (R, C), both stripped as the chat template strips them.
    A turn without a think block returns ("", text) (the reasoning gate drops its episode). More than one think block,
    or a stray tag, raises HarnessError("think_tags"): splitting would leave a tag inside the trained content."""
    n_open, n_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    if n_open == 0 and n_close == 0:
        return "", text.strip()
    if n_open != 1 or n_close != 1 or not text.lstrip().startswith(THINK_OPEN) or text.find(THINK_CLOSE) < text.find(THINK_OPEN):
        raise HarnessError("think_tags", f"{n_open} <think>, {n_close} </think>")
    head, content = text.split(THINK_CLOSE, 1)
    return head.lstrip()[len(THINK_OPEN):].strip(), content.strip()


def observation_kind(text: str) -> str:
    for kind, prefix in OBSERVATION_PREFIXES:
        if text.startswith(prefix):
            return kind
    return "other"


def template_drift(path: Path = HARBOR_TEMPLATE_FILE) -> str | None:
    """None when the installed Harbor template equals our copy (or Harbor is absent); else a description."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256 or raw.decode("utf-8") != TERMINUS_JSON_PLAIN_TEMPLATE:
        return f"{path} differs from the pinned Harbor {HARBOR_VERSION} template (sha256 {TEMPLATE_SHA256[:16]})"
    return None
