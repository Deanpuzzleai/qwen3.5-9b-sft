"""Process-quality checks of stage-2 rollout episodes (data plan revision 4.3.16, item #12), applied by
scripts/data/build_stage2_set.py identically to both arms (the verdict is the only difference between them).

Deterministic heuristics over an episode's messages (and, for SWE, its final patch); they never read verdicts or observations
beyond the episode itself.

  loop                    both families: the longest run of consecutive assistant turns that issue the same command(s) and
                          get the same observation back. A run of >= LOOP_REPEATS is a hard drop in both arms (a stuck
                          agent, not a strategy). Polling a background job survives because its observation changes.
  swe_tests_after_edit    SWE: stage 1's test-after-edit rule (apex_data.teacher.verify, the `fewest_turns_verified`
                          trajectory choice): at least one test-runner command and no edit command after the last one.
  swe_clean_diff          SWE: the final patch adds no non-test junk file: reproduction / debug / scratch scripts, backups
                          (.orig, .bak, .rej, ~), diff dumps, logs, __pycache__. Test-named scratch files are test paths (the
                          grader keeps them out of the replay) and do not count. `patch.txt` does not count either
                          (quality:2): the SFT policy writes it as its submission step (`git diff > patch.txt &&
                          cat patch.txt`, a habit of the teacher's harness), so it is in every submitted episode.
  terminal_verify         terminal: a verification command (running the deliverable, a script or tests, or inspecting
                          output files) in the first task_complete turn or the VERIFY_WINDOW assistant turns before it.

The builder turns `swe_tests_after_edit and swe_clean_diff` (SWE) and `terminal_verify` (terminal) into a hard gate of that
family when at least HARD_SHARE of its positives pass, otherwise into a preference (positives: passing episodes first in the
per-task pick; control: passing episodes first in the loss-token draw). `VERSION` names the rules.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from apex_data.render.filters import terminus_parse
from apex_data.render.turn_roles import terminal_output
from apex_data.teacher import verify

VERSION = "quality:2"
LOOP_REPEATS = 3
VERIFY_WINDOW = 3
HARD_SHARE = 0.70

# a new file whose basename looks like a throwaway artefact rather than part of the fix
_JUNK_NAME = re.compile(r"^(reproduce|repro|debug|scratch|tmp|temp|demo|check|verify|try|sandbox)([_\-.].*)?\.\w+$"
                        r"|\.(orig|bak|rej|swp|pyc|patch|diff|log)$|~$|^(diff|changes|output|out)\.txt$", re.I)
_JUNK_DIR = re.compile(r"(^|/)__pycache__/|(^|/)\.pytest_cache/")
_TEST_PATH = re.compile(r"(^|/)(tests?|testing|testdata)/|(^|/)test_[^/]*$|_test\.(py|go)$|(^|/)conftest\.py$", re.I)
_NEW_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)\n(?:(?!diff --git ).*\n)*?new file mode", re.M)

# terminal verification: run something, or read an output file without writing it in the same command
_RUN = re.compile(r"^\s*(sudo\s+)?(python3?|node|bash|sh|zsh|ruby|perl|php|java|go\s+(run|test)|cargo\s+(run|test)|make|"
                  r"npm\s+(run|test)|npx|pytest|ctest|\./|/app/|/usr/local/bin/|\S+\.sh\b)")
_INSPECT = re.compile(r"^\s*(cat|head|tail|less|more|wc|diff|cmp|md5sum|sha\d+sum|ls|stat|file|jq|grep|egrep|od|xxd|hexdump|"
                      r"test|\[|sqlite3|curl|wget)\b")
_WRITES = re.compile(r"(?<![<0-9&2])>>?\s*[^\s&|]|<<\s*['\"]?\w+|\btee\b")


def _content(m: Any) -> str:
    c = m.get("content") if isinstance(m, Mapping) else getattr(m, "content", "")
    return c if isinstance(c, str) else json.dumps(c)


def _role(m: Any) -> str:
    return m.get("role") if isinstance(m, Mapping) else getattr(m, "role", "")


def _tool_args(m: Any) -> list[Any]:
    calls = (m.get("tool_calls") if isinstance(m, Mapping) else getattr(m, "tool_calls", None)) or []
    out = []
    for tc in calls:
        fn = tc.get("function") if isinstance(tc, Mapping) else None
        args = (fn or {}).get("arguments") if fn is not None else getattr(tc, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        out.append(args)
    return out


def command_observations(messages: Sequence[Any], family: str) -> list[tuple[str, str]]:
    """(command key, observation) per assistant turn that issued commands, in order. SWE: the tool calls' arguments and
    the following tool messages; terminal: the parsed Terminus-2 keystrokes and the terminal output of the next user
    message. Turns without a command (parse errors, prose) break a run with an empty key."""
    msgs = list(messages)
    out: list[tuple[str, str]] = []
    for i, m in enumerate(msgs):
        if _role(m) != "assistant":
            continue
        if family == "swe":
            key = json.dumps(_tool_args(m), sort_keys=True) if _tool_args(m) else ""
            obs = []
            for n in msgs[i + 1:]:
                if _role(n) != "tool":
                    break
                obs.append(_content(n))
            out.append((key, "\n".join(obs)))
        else:
            p = terminus_parse(_content(m))
            key = "" if p.error else json.dumps([k for k, _ in p.commands])
            nxt = msgs[i + 1] if i + 1 < len(msgs) and _role(msgs[i + 1]) == "user" else None
            out.append((key, terminal_output(_content(nxt)) if nxt is not None else ""))
    return out


def longest_loop(pairs: Sequence[tuple[str, str]]) -> int:
    """Longest run of consecutive identical (command, observation) pairs with a non-empty command."""
    best = run = 0
    prev: tuple[str, str] | None = None
    for p in pairs:
        run = run + 1 if (p[0] and p == prev) else (1 if p[0] else 0)
        prev = p
        best = max(best, run)
    return best


def junk_files(patch: str) -> list[str]:
    """Files the patch adds that are not test paths and look like throwaway artefacts."""
    out = []
    for _, b in _NEW_FILE.findall(patch or ""):
        base = b.rsplit("/", 1)[-1]
        if _TEST_PATH.search(b):
            continue
        if _JUNK_DIR.search(b) or _JUNK_NAME.search(base):
            out.append(b)
    return sorted(set(out))


def _verify_command(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("#"):
        return False
    if _RUN.search(line):
        return True
    return bool(_INSPECT.search(line)) and not _WRITES.search(line)


def terminal_verify(messages: Sequence[Any]) -> dict[str, Any]:
    """Whether a verification command ran in the first task_complete turn or the VERIFY_WINDOW assistant turns before it."""
    turns = [(i, terminus_parse(_content(m))) for i, m in enumerate(messages) if _role(m) == "assistant"]
    first = next((k for k, (_, p) in enumerate(turns) if not p.error and p.task_complete), None)
    if first is None:
        return {"terminal_verify": False, "verify_turn": None, "reason": "no task_complete turn"}
    for k in range(first, max(-1, first - VERIFY_WINDOW - 1), -1):
        i, p = turns[k]
        if p.error:
            continue
        for keys, _ in p.commands:
            if any(_verify_command(ln) for ln in keys.splitlines()):
                return {"terminal_verify": True, "verify_turn": i, "reason": None}
    return {"terminal_verify": False, "verify_turn": None, "reason": f"no verification command in the {VERIFY_WINDOW + 1} turns "
                                                                     "ending at the first task_complete"}


def episode_quality(episode: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Every check of one episode; `pass` is the family's preference / gate condition."""
    msgs = episode.get("messages") or []
    q: dict[str, Any] = {"version": VERSION, "loop": longest_loop(command_observations(msgs, family))}
    if family == "swe":
        feats = verify.command_features(verify.commands(msgs))
        junk = junk_files(episode.get("patch") or "")
        q.update(swe_tests_after_edit=bool(feats["verified"]), swe_clean_diff=not junk, junk_files=junk)
        q["pass"] = q["swe_tests_after_edit"] and q["swe_clean_diff"]
    else:
        q.update(terminal_verify(msgs))
        q["pass"] = q["terminal_verify"]
    q["loop_drop"] = q["loop"] >= LOOP_REPEATS
    return q
