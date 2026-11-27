"""Structured test reporters for the SWE-smith replay grader (grade/swesmith.py): Go and Python only.

Log parsing is the fallback: when the test runner is detectable in the profile's test command, calibration first runs the
command with that runner's machine-readable reporter added, validates the structured output against the task's F2P/P2P
ids and persists the validated form in task.extra.eval_spec; every later replay executes exactly that command. Nothing
else in the command changes.

Runner    change to the command                                      output                       ids emitted
go test   `-json` inserted after every `go test`                     test2json lines in the log   TestName, TestName/subtest
pytest    `--junitxml=/apex_out/junit.xml -o junit_family=xunit1`    JUnit XML (file)             "path/test_x.py::Class::test[param]" + bare name
          appended to the last segment that runs pytest
A command without a detectable runner (make, tox, nox, custom scripts, `python -m unittest`) has no structured form and
is graded from the log by the profile's own parser.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import dataclass

STRUCTURED_DIR = "/apex_out"                      # inside the container; run_tests copies the file next to the test log
OUTPUT_FILES = {"junit:pytest": f"{STRUCTURED_DIR}/junit.xml"}
PYTEST_FLAGS = f"--junitxml={OUTPUT_FILES['junit:pytest']} -o junit_family=xunit1"
REPORTERS = ("json:go", "junit:pytest")
WRAPPERS = frozenset({"env", "exec", "timeout", "time", "xvfb-run", "coverage"})
_OPERATOR = re.compile(r"(\s*(?:&&|\|\||;|\|)\s*)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_GO_TEST = re.compile(r"\bgo test\b")
_PASS, _FAIL, _SKIP = "PASSED", "FAILED", "SKIPPED"


@dataclass(frozen=True)
class Reporter:
    name: str                 # json:go | junit:pytest
    test_cmd: str             # the profile's command with the reporter flags added
    output_file: str | None   # container path of the structured file; None when the output is in the log (go)


# --- detection ---------------------------------------------------------------------------------------------------------
def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def detect_runner(segment: str) -> str | None:
    """Test runner of one shell segment: "pytest" | "go" | None. Leading env assignments and simple wrappers are skipped;
    `coverage run -m pytest` and `python -m pytest` count as pytest."""
    toks = _tokens(segment)
    while toks and (_ENV_ASSIGN.match(toks[0]) or toks[0] in WRAPPERS or toks[0] == "run"):
        toks = toks[1:]
    if not toks:
        return None
    head = os.path.basename(toks[0])
    if head == "go":
        return "go" if len(toks) > 1 and toks[1] == "test" else None
    if head in ("pytest", "py.test"):
        return "pytest"
    if re.fullmatch(r"python(\d(\.\d+)?)?", head) and len(toks) > 2 and toks[1] == "-m" and toks[2] in ("pytest", "py.test"):
        return "pytest"
    if head == "-m" and len(toks) > 1 and toks[1] in ("pytest", "py.test"):     # `coverage run -m pytest`
        return "pytest"
    return None


def structured_command(test_cmd: str) -> Reporter | None:
    """The command with the runner's machine-readable reporter added, or None when no runner is detectable. `go test`
    gets `-json` in every segment that runs it; pytest gets the JUnit flags on the last segment that runs it."""
    parts = _OPERATOR.split(test_cmd)
    segments = parts[::2]
    runners = [detect_runner(s) for s in segments]
    if "go" in runners:
        parts = list(parts)
        for i, (seg, r) in enumerate(zip(segments, runners)):
            if r == "go" and not re.search(r"\bgo test\b[^&|;]*\s-json\b", seg):
                parts[2 * i] = _GO_TEST.sub("go test -json", seg, count=1)
        return Reporter("json:go", "".join(parts), None)
    for i in range(len(segments) - 1, -1, -1):
        if runners[i] == "pytest":
            parts = list(parts)
            parts[2 * i] = segments[i].rstrip() + " " + PYTEST_FLAGS
            return Reporter("junit:pytest", "".join(parts), OUTPUT_FILES["junit:pytest"])
    return None


# --- parsers -----------------------------------------------------------------------------------------------------------
def _add(seen: dict[str, list[str]], name: str, status: str) -> None:
    name = " ".join(str(name).split())
    if name:
        seen.setdefault(name, []).append(status)


def _merged(seen: dict[str, list[str]]) -> dict[str, str]:
    """An id reported more than once fails if any occurrence failed, else passes if any passed, else is skipped."""
    rank = {_FAIL: 3, "ERROR": 3, _PASS: 2, _SKIP: 1}
    return {k: max(v, key=lambda s: rank.get(s, 0)) for k, v in seen.items()}


def parse_go_json(log: str) -> dict[str, str]:
    """`go test -json` events in the log (other lines are ignored): Test + Action pass/fail/skip."""
    seen: dict[str, list[str]] = {}
    for line in log.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        action, test = ev.get("Action"), ev.get("Test")
        if test and action in ("pass", "fail", "skip"):
            _add(seen, test, {"pass": _PASS, "fail": _FAIL, "skip": _SKIP}[action])
    return _merged(seen)


def parse_junit_xml(text: str) -> dict[str, str]:
    """JUnit XML (pytest xunit1 family): node ids rebuilt from file + classname + name, plus the bare test name."""
    root = ET.fromstring(text)
    seen: dict[str, list[str]] = {}
    for tc in root.iter("testcase"):
        name, classname, file = tc.get("name") or "", tc.get("classname") or "", tc.get("file") or ""
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = _FAIL
        elif tc.find("skipped") is not None:
            status = _SKIP
        else:
            status = _PASS
        parts = classname.split(".") if classname else []
        module = (file[:-3] if file.endswith(".py") else file).replace("/", ".") if file else ""
        if file and classname.startswith(module):
            _add(seen, "::".join([file, *[p for p in classname[len(module):].split(".") if p], name]), status)
        elif parts:
            # `file` names where the function is defined (a mixin in tests/util.py), the classname the collecting module:
            # every module/class split of the classname, so "tests.test_x.TestK" also yields tests/test_x.py::TestK::name
            for i in range(1, len(parts) + 1):
                _add(seen, "::".join(["/".join(parts[:i]) + ".py", *parts[i:], name]), status)
            if not file:
                _add(seen, "::".join([*parts, name]), status)
        _add(seen, name, status)
    return _merged(seen)


def parse_structured(reporter: str, log: str, output_text: str | None) -> dict[str, str] | None:
    """Status map from a reporter's output, or None when the structured output is missing or unparsable (the caller then
    records zero coverage and calibration falls back to the profile's log parser). `output_text` is the copied-out file."""
    try:
        if reporter == "json:go":
            return parse_go_json(log) or None
        if reporter == "junit:pytest":
            return parse_junit_xml(output_text) if output_text else None
    except (ValueError, ET.ParseError, TypeError, AttributeError):
        return None
    return None
