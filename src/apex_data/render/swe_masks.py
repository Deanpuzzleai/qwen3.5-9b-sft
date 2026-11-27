"""Turn masks on rendered SWE episode rows (v6 rules S1 and S2, data/manifests/v6/DATA_CARD_v6.md): the same
episodes and token ids, fewer trained turns. A masked turn stays in the sequence as context and carries no loss.

Turn view, from the row alone. Every trained span of an episode row (a run of the assistant mask, ended by
<|im_end|>) is one assistant turn of the mini-swe-agent episode, in order. Its span starts after `<think>\\n`: the
reasoning is everything before `</think>` (its token count is the thinking of the turn), and its commands are the
`command` parameters of all its qwen3_coder tool calls (render.filters.qwen3_parse_calls); a turn may make several
calls, and the kinds and rules below look at every one of them. The observations before a turn are the
`<tool_response>` JSONs between the previous span and this one (efficiency.parse_obs: returncode, output).

Turn kinds (by priority; the shell parsing is efficiency.segments, whose `cd` segments only move the directory):
  submit  some command contains COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT;
  edit    some segment edits a file (efficiency kind `edit`: sed -i, cat >, tee, python writes, patch / git apply, ...);
  run     some segment runs tests or the program, or inline code (efficiency kinds `run`, `inline`);
  read    every segment of every call only lists, prints or searches: READ_COMMANDS (ls, cat, grep / egrep / fgrep / rg, find, head,
          tail, sed -n, nl, wc, tree, pwd, git diff / status / log / show, and echo / printf, which the teacher uses
          as separators between reads: `sed -n 1,9p a && echo --- && cat b`), pipes between them allowed, no file
          redirect, no heredoc, no find -exec / -delete (efficiency's read-only test plus this narrower list);
  other   anything else.
Rules:
  S1  a read turn stays trained when (a) it is among the episode's first READ_KEEP_FIRST assistant turns, (b) the
      observations just before it (the previous turn's tool responses) show an error (non-zero returncode, or
      Traceback / Error / FAILED / panic in an output), or (c) the next assistant turn edits a file; otherwise its span
      is masked.
  S2  the submit turn is masked when none of the SUBMIT_LOOKBACK previous assistant turns runs tests or the program
      (a `run` or `inline` segment in any of their calls, also inside an editing turn).
Rule v7 ("adaptive thinking", data/manifests/v7/DATA_CARD_v7.md): after S1 and S2, a turn is a decision point
when an observation just before it shows a real failure (FAILURE_RE or a non-zero returncode: stricter than S1's error
test, which also matches code in a listing such as `t.Errorf`) or it is the turn right before the first edit; a routine turn stays
trained only with thinking <= ROUTINE_MAX; a decision-point turn up to DECISION_MAX when its reasoning is not
repetitive (no run of REPEAT_WORDS words seen twice, and at least 80 % distinct lines among its lines of >= 20
characters) and it ends in a tool call with a command; thinking above DECISION_MAX is never trained.
After masking, n_assistant_tokens and n_assistant_turns are recomputed from the mask. Deterministic: no randomness.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from apex_data import efficiency
from apex_data.render.filters import qwen3_parse_calls

VERSION = "swe_masks:2"                          # 2: every tool call of a turn counts (1 read only the first)
THINK_END, IM_END = 248069, 248046                 # Qwen3.5-9B `</think>` and `<|im_end|>` token ids
READ_KEEP_FIRST = 3
SUBMIT_LOOKBACK = 3
READ_COMMANDS = frozenset({"ls", "cat", "grep", "egrep", "fgrep", "rg", "find", "head", "tail", "sed", "nl", "wc", "tree", "pwd",
                           "git", "echo", "printf"})
GIT_READ = frozenset({"diff", "status", "log", "show"})
ERROR_RE = re.compile(r"Traceback|Error|FAILED|panic")
FAILURE_RE = re.compile(r"Traceback \(most recent call last\)|FAILED|FAIL:|panic:|\b[1-9]\d* failed\b|timed out|command not found|"
                        r"SyntaxError|No such file|undefined:|error TS\d*|[Cc]annot find|npm ERR!")
KINDS = ("read", "edit", "run", "submit", "other")
ROUTINE_MAX, DECISION_MAX = 1000, 8192               # rule v7: thinking tokens per trained turn
REPEAT_WORDS, DISTINCT_LINES, LONG_LINE = 30, 0.8, 20
_RESPONSE_RE = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)


# ------------------------------------------------------------------------------------------------ classification
def _git_sub(seg: str) -> str | None:
    words = seg.split()
    k = 1
    while k < len(words) and words[k].startswith("-"):           # git -C <dir> / --no-pager before the subcommand
        k += 2 if words[k] == "-C" else 1
    return words[k] if k < len(words) else None


def is_read_segment(seg: efficiency.Segment) -> bool:
    if seg.kind != "read":
        return False
    word = seg.text.split(None, 1)[0] if seg.text.split() else ""
    if word not in READ_COMMANDS:
        return False
    if word == "git":
        return _git_sub(seg.text) in GIT_READ
    if word == "sed":
        return bool(re.search(r"(?:^|\s)-[a-zA-Z]*n", seg.text))
    return True


@dataclass(frozen=True)
class TurnFacts:
    """What the rules read from one command."""

    kind: str                    # KINDS
    runs: bool                   # some segment runs tests or the program, or inline code


def classify(commands: str | Sequence[str]) -> TurnFacts:
    """The kind of a turn from all its tool calls' commands (one string = one call): the kinds' priority applies over
    every call, so a turn is a read only when every segment of every call is a read, and it runs when any call runs."""
    cmds = [commands] if isinstance(commands, str) else list(commands)
    if any(efficiency.SUBMIT in c for c in cmds):
        return TurnFacts("submit", False)
    segs = [seg for c in cmds for seg in efficiency.segments(c, family="swe", root=efficiency.HARNESS_REPO_ROOT)]
    runs = any(s.kind in ("run", "inline") for s in segs)
    if any(s.kind == "edit" for s in segs):
        return TurnFacts("edit", runs)
    if runs:
        return TurnFacts("run", True)
    if segs and all(is_read_segment(s) for s in segs):
        return TurnFacts("read", False)
    return TurnFacts("other", False)


def shows_failure(observation: str | None) -> bool:
    """Rule v7's decision point: a non-zero returncode or a strong failure marker (FAILURE_RE) in the output."""
    if observation is None:
        return False
    rc, output = efficiency.parse_obs(observation)
    return (rc is not None and rc != 0) or bool(FAILURE_RE.search(output))


def shows_error(observation: str | None) -> bool:
    if observation is None:
        return False
    rc, output = efficiency.parse_obs(observation)
    return (rc is not None and rc != 0) or bool(ERROR_RE.search(output))


# ------------------------------------------------------------------------------------------------ the turn view
def trained_spans(mask: Sequence[int]) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, m in enumerate([*mask, 0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((start, i))
            start = None
    return spans


@dataclass
class Turn:
    index: int                   # assistant turn of the episode, 0-based
    start: int                   # token span [start, end)
    end: int
    commands: list[str]          # the `command` of every tool call of the turn, in order
    observations: list[str]      # the tool responses just before the turn (empty for the first turn)
    thinking: int                # reasoning tokens (position of </think> in the span; -1 when not closed)
    kind: str = "other"
    runs: bool = False
    trained: bool = True
    reason: str = ""             # why a read or submit turn is kept or masked (rule v7: also routine / decision turns)
    decision: bool = False       # rule v7: right after an error, or the turn before the first edit
    reasoning: str = ""          # the reasoning text (rule v7's repetition test)


def turns_of(input_ids: Sequence[int], assistant_masks: Sequence[int], decode: Any) -> list[Turn]:
    """The assistant turns of an episode row. `decode` maps token ids to text (the tokenizer's decode)."""
    out, prev_end = [], None
    for k, (a, b) in enumerate(trained_spans(assistant_masks)):
        ids = list(input_ids[a:b])
        text = decode(ids)
        after = text.split("</think>", 1)[1] if "</think>" in text else text
        commands = [str(c["arguments"].get("command", "")) for c in qwen3_parse_calls(after)]
        obs = _RESPONSE_RE.findall(decode(list(input_ids[prev_end:a]))) if prev_end is not None else []
        facts = classify(commands)
        reasoning = text.split("</think>", 1)[0] if "</think>" in text else ""
        out.append(Turn(k, a, b, commands, obs, ids.index(THINK_END) if THINK_END in ids else -1, facts.kind, facts.runs,
                        reasoning=reasoning))
        prev_end = b
    return out


def apply_rules(turns: list[Turn], *, read_keep_first: int = READ_KEEP_FIRST, submit_lookback: int = SUBMIT_LOOKBACK) -> list[Turn]:
    """Sets `trained` and `reason` on read and submit turns (S1, S2); other turns stay trained."""
    for k, t in enumerate(turns):
        if t.kind == "read":
            if k < read_keep_first:
                t.reason = "read_first_turns"
            elif any(shows_error(o) for o in t.observations):
                t.reason = "read_after_error"
            elif k + 1 < len(turns) and turns[k + 1].kind == "edit":
                t.reason = "read_before_edit"
            else:
                t.trained, t.reason = False, "read_masked"
        elif t.kind == "submit":
            if any(p.runs for p in turns[max(0, k - submit_lookback):k]):
                t.reason = "submit_after_run"
            else:
                t.trained, t.reason = False, "submit_unverified"
    return turns


def repetitive(text: str) -> bool:
    """Rule v7's test: some run of REPEAT_WORDS words occurs twice, or fewer than DISTINCT_LINES of the reasoning's lines
    of >= LONG_LINE characters are distinct (code fences and short interjections are left out of the line test)."""
    words = text.split()
    seen: set[tuple[str, ...]] = set()
    for i in range(len(words) - REPEAT_WORDS + 1):
        w = tuple(words[i:i + REPEAT_WORDS])
        if w in seen:
            return True
        seen.add(w)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= LONG_LINE]
    return bool(lines) and len(set(lines)) / len(lines) < DISTINCT_LINES


def apply_v7_rules(turns: list[Turn]) -> list[Turn]:
    """S1 and S2, then rule v7 on the turns still trained (module docstring). Sets `decision`, `trained`, `reason`."""
    apply_rules(turns)
    first_edit = next((i for i, t in enumerate(turns) if t.kind == "edit"), None)
    for k, t in enumerate(turns):
        t.decision = any(shows_failure(o) for o in t.observations) or (first_edit is not None and k == first_edit - 1)
        if not t.trained:
            continue
        if t.thinking < 0 or t.thinking > DECISION_MAX:
            t.trained, t.reason = False, "thinking_over_8k"
        elif t.decision:
            if not any(c.strip() for c in t.commands):
                t.trained, t.reason = False, "decision_no_tool_call"
            elif repetitive(t.reasoning):
                t.trained, t.reason = False, "decision_repetitive"
            else:
                t.reason = t.reason or "decision_trained"
        elif t.thinking > ROUTINE_MAX:
            t.trained, t.reason = False, "routine_over_1000"
    return turns


def mask_row(row: Mapping[str, Any], decode: Any, *, rule: str = "v6") -> tuple[dict[str, Any], list[Turn]]:
    """The row with the masks of `rule` applied (v6: S1/S2; v7: S1/S2 and the adaptive thinking rule): new
    assistant_masks, n_assistant_tokens, n_assistant_turns, and its turns."""
    if rule not in ("v6", "v7"):
        raise ValueError(f"unknown rule {rule!r}")
    turns = turns_of(row["input_ids"], row["assistant_masks"], decode)
    turns = apply_rules(turns) if rule == "v6" else apply_v7_rules(turns)
    mask = list(row["assistant_masks"])
    for t in turns:
        if not t.trained:
            mask[t.start:t.end] = [0] * (t.end - t.start)
    out = {**row, "assistant_masks": mask, "n_assistant_tokens": int(sum(mask)), "n_assistant_turns": sum(t.trained for t in turns)}
    if out["n_assistant_tokens"] <= 0:
        raise ValueError(f"{row.get('row_key')}: masking left no trained token")
    return out, turns


# ------------------------------------------------------------------------------------------------ reporting
def _stats(values: Iterable[int]) -> dict[str, float]:
    v = sorted(values)
    if not v:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    return {"n": len(v), "mean": round(sum(v) / len(v), 1), "p50": float(v[len(v) // 2]),
            "p90": float(v[min(len(v) - 1, int(0.9 * len(v)))]), "max": v[-1]}


@dataclass
class Report:
    """Turn kinds before and after, the reasons, loss tokens and thinking per trained turn, overall and by group."""

    rows: int = 0
    turns_before: Counter = field(default_factory=Counter)
    turns_after: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    loss_before: int = 0
    loss_after: int = 0
    thinking_before: list[int] = field(default_factory=list)
    thinking_after: list[int] = field(default_factory=list)
    masked_tokens_by_kind: Counter = field(default_factory=Counter)

    def add(self, before: Mapping[str, Any], after: Mapping[str, Any], turns: Sequence[Turn]) -> None:
        self.rows += 1
        self.loss_before += int(before["n_assistant_tokens"])
        self.loss_after += int(after["n_assistant_tokens"])
        for t in turns:
            self.turns_before[t.kind] += 1
            self.thinking_before.append(t.thinking)
            if t.reason:
                self.reasons[t.reason] += 1
            if t.trained:
                self.turns_after[t.kind] += 1
                self.thinking_after.append(t.thinking)
            else:
                self.masked_tokens_by_kind[t.kind] += t.end - t.start

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "turns_before": {k: self.turns_before[k] for k in KINDS},
                "turns_after": {k: self.turns_after[k] for k in KINDS},
                "trained_turns": {"before": sum(self.turns_before.values()), "after": sum(self.turns_after.values())},
                "reasons": dict(sorted(self.reasons.items())),
                "loss_tokens": {"before": self.loss_before, "after": self.loss_after,
                                "masked": self.loss_before - self.loss_after,
                                "masked_by_kind": dict(sorted(self.masked_tokens_by_kind.items()))},
                "thinking_per_trained_turn": {"before": _stats(self.thinking_before), "after": _stats(self.thinking_after)}}


def rules() -> dict[str, Any]:
    return {"version": VERSION, "read_commands": sorted(READ_COMMANDS), "git_read": sorted(GIT_READ),
            "read_keep_first": READ_KEEP_FIRST, "submit_lookback": SUBMIT_LOOKBACK, "error_pattern": ERROR_RE.pattern,
            "v7_failure_pattern": FAILURE_RE.pattern,
            "doc": __doc__}


def turn_record(t: Turn) -> dict[str, Any]:
    return {k: v for k, v in asdict(t).items() if k not in ("commands", "observations", "reasoning")}


def dumps_turns(turns: Sequence[Turn]) -> str:
    return json.dumps([turn_record(t) for t in turns])
