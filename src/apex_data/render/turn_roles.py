"""Turn roles of a Terminus-2 transcript: which assistant turns of a terminal episode the stage-2 set trains first
(scripts/data/build_stage2_set.py --terminal-turn-roles; docs/DATA_PLAN.md section 11).

A terminal episode trains a few of its turns (per-turn rows, `turns_per_episode`). By default these are the latest
fitting turn and random earlier ones. With roles on, the turns that carry an agent skill are tried first, in priority order:

  verify_before_complete  the assistant turn right before the first turn that sets task_complete (the last check of the
                          work before handing it in)
  after_error             a turn answering an observation whose new terminal output shows an error (traceback,
                          `<prefix>: error:`, `command not found`, `No such file or directory`, failed tests, ...)
  long_job                a turn that starts or watches a long job (a trailing `&`, nohup/setsid/disown, a detached
                          screen/tmux; sleep, tail -f, watch, wait, jobs, kill -0, pgrep; an empty keystroke or any
                          command waited on for >= LONG_WAIT_SEC), or the first turn that writes (shell redirect or
                          tee) a path the task names, if it comes in the first half of the episode (writing the
                          deliverable early)
  recovery                a turn that sends a control key (C-c, C-d, C-z, C-\\) or answers a screen stuck at a
                          continuation or REPL prompt (`>`, `>>>`, `...`) or a pager (`(END)`, `--More--`)

A turn gets the first role it matches. ORDER is the selection rule `priority_turns` encodes for
`data_format.BuildOptions.priority_turns`: the verify_before_complete turn; then the latest fitting turn (the hand-in
turn, as in stage 1; role 'latest'); then a round robin over after_error, long_job and recovery (each role's latest turn
in that order, then each role's next latest, ...: every role gets a slot before any role gets a second, and late turns
come first within a role, as for the latest-fitting rule); then the seeded random draw over the remaining turns (role
'random'). Turns that are not candidates (the confirmation replies the adapter excludes) or do not fit (row cap,
max_turn_completion_tokens) are skipped and the next one is tried.

The signals are text heuristics over the transcript the model saw: Harbor's strict JSON parser port
(filters.terminus_parse) for the commands, and only the terminal output section of each observation. The section
header lines are excluded, as are echoed command lines (`user@host:dir# ...`) and heredoc continuation echoes (`> ...`).
A false positive only moves one training turn.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from apex_data.render.filters import terminus_parse
from apex_train.data_format import LATEST_FITTING

ROLES = ("verify_before_complete", "after_error", "long_job", "recovery")   # a turn gets the first role it matches
ROUND_ROBIN = ROLES[1:]
FILL_ROLES = ("latest", "random")                                            # data_format build_rows sources
ALL_ROLES = (*ROLES, *FILL_ROLES)
VERSION = "turn_roles:1"
ORDER = ("trained terminal turns per episode, in order: (1) verify_before_complete, the assistant turn right before the "
         "first task_complete turn; (2) latest, the latest fitting turn (the hand-in turn, as in stage 1); (3) a round "
         "robin over after_error, long_job, recovery (each role's latest turn in that order, then each role's next "
         "latest); (4) random, the seeded draw over the remaining turns. Turns that are not candidates (confirmation "
         "replies) or do not fit the row cap or the per-turn cap are skipped.")
LONG_WAIT_SEC = 30.0

_OUTPUT_HEADERS = ("New Terminal Output:", "Current Terminal Screen:")
_PROMPT_LINE = re.compile(r"^\S+@\S+:[^\n]*?[#$](\s|$)")
_ECHO_LINE = re.compile(r"^\s*\S?>(\s|$)")
_ERROR_LINE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|^[\w.]*(Error|Exception):"
    r"|^([^\s:]+:\s*)*(fatal error|error|fatal|ERROR|Error|FATAL)(\[\w+\])?\s*[:!]"
    r"|command not found|No such file or directory|Permission denied|Segmentation fault|core dumped"
    r"|syntax error|npm ERR!|^E: |\bpanicked\b|^panic:|undefined reference"
    r"|^FAILED\b|^--- FAIL|^FAIL\b|\b[1-9]\d* failed\b|\*\*\* .*Error \d+")
_BACKGROUND = re.compile(r"(?<![&|>])&\s*$|\bnohup\b|\bsetsid\b|\bdisown\b|\bscreen\s+-\w*d|\btmux\s+new(-session)?\b.*\s-d\b")
_POLL = re.compile(r"\bsleep\s+\d|\btail\s+-[fF]\b|\bwatch\s|^\s*(wait|jobs)\b|\bkill\s+-0\b|\bpgrep\b")
_CONTROL_KEY = re.compile(r"^C-[cdz\\]$")
_STUCK_LAST_LINE = re.compile(r"^\s*(>|>>>|\.\.\.)\s*$|^\s*\S?>\s.*|\(END\)\s*$|--More--")
_WRITE_TARGET = re.compile(r"(?:(?<![<0-9&])>>?|\btee\s+(?:-a\s+)?)\s*['\"]?(/[\w.\-/]+)")
_TASK_PATH = re.compile(r"(?<![\w.:/])/(?:[\w.\-]+/)*[\w.\-]+")


def _content(m: Any) -> str:
    c = m.get("content") if isinstance(m, dict) else m.content
    return c if isinstance(c, str) else ""


def _role(m: Any) -> str:
    return m.get("role") if isinstance(m, dict) else m.role


def terminal_output(observation: str) -> str:
    """The terminal output of a Terminus-2 observation (after its last output header). Warning and parse-error prefaces
    and the confirmation prompt are cut; an observation without a header (a parse-error reply) has none."""
    at = max((observation.rfind(h) + len(h) for h in _OUTPUT_HEADERS if h in observation), default=-1)
    if at < 0:
        return ""
    out = observation[at:]
    cut = out.find("\n\nAre you sure you want to mark the task as complete?")
    return out if cut < 0 else out[:cut]


def error_lines(output: str) -> list[str]:
    """Output lines that report an error (echoed commands and heredoc continuation lines excluded)."""
    return [ln for ln in output.splitlines()
            if not _PROMPT_LINE.match(ln) and not _ECHO_LINE.match(ln) and _ERROR_LINE.search(ln)]


def stuck_screen(output: str) -> bool:
    """True when the screen's last non-blank line is a continuation/REPL prompt or a pager, not a shell prompt."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return bool(lines) and not _PROMPT_LINE.match(lines[-1]) and bool(_STUCK_LAST_LINE.search(lines[-1]))


def task_paths(first_user: str) -> set[str]:
    """Absolute paths named in the task description of Terminus-2's first user message."""
    at = first_user.find("Task Description:")
    text = first_user[at:] if at >= 0 else first_user
    end = text.find("Current terminal state:")
    return {p.rstrip(".") for p in _TASK_PATH.findall(text if end < 0 else text[:end])}


def classify(messages: Sequence[Any]) -> dict[int, str]:
    """Message index of an assistant turn -> its role (the first of ROLES it matches); turns without one are absent.
    `messages` are data_format.Message objects or message dicts of one Terminus-2 transcript."""
    idx = [i for i, m in enumerate(messages) if _role(m) == "assistant"]
    parsed = {i: terminus_parse(_content(messages[i])) for i in idx}
    roles: dict[int, str] = {}
    first_complete = next((k for k, i in enumerate(idx) if not parsed[i].error and parsed[i].task_complete), None)
    if first_complete:
        roles[idx[first_complete - 1]] = "verify_before_complete"
    paths = task_paths(_content(messages[0])) if messages and _role(messages[0]) == "user" else set()
    first_write = next((i for i in idx if paths & {t for k, _ in parsed[i].commands for t in _WRITE_TARGET.findall(k)}), None)
    for k, i in enumerate(idx):
        if i in roles:
            continue
        prev = _content(messages[i - 1]) if k > 0 and _role(messages[i - 1]) == "user" else ""
        out = terminal_output(prev)
        cmds = parsed[i].commands
        lines = [ln for keys, _ in cmds for ln in keys.splitlines()]
        if out and error_lines(out):
            roles[i] = "after_error"
        elif (any(_BACKGROUND.search(ln) or _POLL.search(ln) for ln in lines)
              or any(d >= LONG_WAIT_SEC or (not keys.strip() and d >= 5) for keys, d in cmds)
              or (i == first_write and k < len(idx) / 2)):
            roles[i] = "long_job"
        elif any(_CONTROL_KEY.match(keys.strip()) for keys, _ in cmds) or (out and stuck_screen(out)):
            roles[i] = "recovery"
    return roles


def priority_turns(messages: Sequence[Any]) -> list[tuple[int, str]]:
    """(message index, role) in the order BuildOptions.priority_turns tries them (ORDER); the latest fitting turn is the
    entry (LATEST_FITTING, 'latest')."""
    roles = classify(messages)
    by_role = {r: sorted((i for i, x in roles.items() if x == r), reverse=True) for r in ROLES}
    out = [(i, "verify_before_complete") for i in by_role["verify_before_complete"]] + [(LATEST_FITTING, "latest")]
    for k in range(max(len(by_role[r]) for r in ROUND_ROBIN)):
        out += [(by_role[r][k], r) for r in ROUND_ROBIN if k < len(by_role[r])]
    return out
