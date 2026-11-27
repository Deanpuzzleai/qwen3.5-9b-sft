"""Test-after-edit features of a teacher trajectory (the `fewest_turns_verified` trajectory choice of the SWE draw).

A deterministic heuristic over the raw Open-SWE-Traces messages: the commands are the `arguments` strings of the assistant
turns' tool calls, in order (parallel calls of one turn in their listed order). A command is a TEST when it matches
`TEST_RE` (a test runner: pytest, go test, npm/yarn/pnpm test, jest, vitest, mocha, unittest, tox, nox, make test, npx
tsc) and an EDIT when it matches `EDIT_RE` (sed -i, cat > or heredoc, tee, a redirect into a source file, a Python
heredoc, apply_patch, patch -p). A trajectory is `verified` when it ran at least one test and no edit came after its last
test. It reads commands only, never observations or verdicts: it says the teacher checked its change, not that the check
passed (the upstream `resolved` label already says the final patch is correct).

`RULE_ID` names the two patterns; cached features with another id are computed again.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

TEST_RE = re.compile(r"\b(pytest|py\.test|go test|npm (run )?test|yarn (run )?test|pnpm (run )?test|jest|vitest|mocha|"
                     r"unittest|tox\b|nox\b|make test|npx (jest|vitest|mocha|tsc))")
EDIT_RE = re.compile(r"(sed -i|cat\s*>|cat <<|tee |>\s*\S+\.(py|go|ts|js|tsx|jsx)|python3? - <<|apply_patch|patch -p)")
RULE_ID = "test_after_edit:" + hashlib.sha256(f"{TEST_RE.pattern}\n{EDIT_RE.pattern}".encode()).hexdigest()[:12]


def _messages(messages: Any) -> list[Mapping[str, Any]]:
    return json.loads(messages) if isinstance(messages, str) else list(messages or [])


def commands(messages: Any) -> list[str]:
    """The assistant tool calls' argument strings in order (a non-string argument object is JSON-encoded)."""
    out = []
    for m in _messages(messages):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments") or ""
            out.append(args if isinstance(args, str) else json.dumps(args))
    return out


def command_features(cmds: Iterable[str]) -> dict[str, Any]:
    """Counts and last positions of the TEST and EDIT commands, and `verified` (>= 1 test, none edited after the last)."""
    cmds = list(cmds)
    tests = [k for k, c in enumerate(cmds) if TEST_RE.search(c)]
    edits = [k for k, c in enumerate(cmds) if EDIT_RE.search(c)]
    last_test, last_edit = (tests[-1] if tests else None), (edits[-1] if edits else None)
    return {"n_commands": len(cmds), "n_test": len(tests), "n_edit": len(edits), "last_test": last_test, "last_edit": last_edit,
            "verified": bool(tests) and (last_edit is None or last_test > last_edit)}


def trajectory_features(raw: Mapping[str, Any]) -> dict[str, Any]:
    """One cache record of a raw Open-SWE-Traces row: trajectory id, rule id, assistant turns and the test-after-edit
    fields."""
    msgs = _messages(raw["messages"])
    return {"trajectory_id": raw["trajectory_id"], "rule_id": RULE_ID,
            "n_assistant": sum(1 for m in msgs if m.get("role") == "assistant"), **command_features(commands(msgs))}
