"""Tamper flags (data plan sections 6 and 11): a flagged episode gets r_pass = 0 whatever the tests say.

Flags
  empty_diff           SWE: the agent's final diff has no non-test change
  no_submit            SWE: no explicit submit; terminal: no confirmed `task_complete` handshake; QA: generation did not
                       finish normally
  tests_touched        the diff edits, deletes or renames a test file that exists at the task's base commit (a path under
                       a test directory or with a test-file name); a new test-named file the agent created (a scratch
                       reproduction script such as test_issue.py) is not tampering: it is listed by
                       `scratch_test_files` and recorded in the verdict, and like every test path it is left out of the
                       replayed diff, so it cannot influence the verdict
  ci_touched           the diff touches CI configuration (.github/, .gitlab-ci.yml, Jenkinsfile, ...)
  test_config_touched  the diff touches the test runner's configuration (conftest.py, pytest.ini, tox.ini, setup.cfg,
                       pyproject.toml, noxfile.py, package.json and the JS runners' configs)
  skip_markers         added lines introduce skip / focus markers (pytest.mark.skip, t.Skip, it.skip, .only, @Ignore, ...)
  verifier_touched     terminal: a command references /tests, /logs/verifier or reward files (hidden-test probing)

`split_patch` returns the diff restricted to non-test, non-CI, non-test-config paths; the graders apply only that part,
so a test edit can never influence the verdict even before the flag is read. The diff parser is deliberately small
(unified diff headers and hunks) so this module imports without third-party packages in every venv.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TEST_DIR_NAMES = frozenset({"test", "tests", "testing", "spec", "specs", "__tests__", "testdata", "test_data", "e2e", "integration_tests"})
TEST_FILE_PATTERNS = (
    re.compile(r"^test_.*\.py$"), re.compile(r".*_tests?\.py$"), re.compile(r"^tests?\.py$"),
    re.compile(r".*_test\.go$"),
    re.compile(r".*\.(test|spec)\.(js|jsx|ts|tsx|mjs|cjs|mts|cts)$"), re.compile(r"^test.*\.(js|jsx|ts|tsx|mjs|cjs)$"),
    re.compile(r".*Tests?\.(java|kt|cs|scala)$"), re.compile(r".*_test\.(rs|rb|php|c|cc|cpp)$"), re.compile(r".*_spec\.rb$"),
)
CI_PREFIXES = (".github/", ".gitlab/", ".circleci/", ".buildkite/", ".travis/", ".azure-pipelines/", ".woodpecker/")
CI_FILES = frozenset({".gitlab-ci.yml", ".travis.yml", "jenkinsfile", "azure-pipelines.yml", "appveyor.yml", ".drone.yml",
                      "bitbucket-pipelines.yml", "codecov.yml", ".codecov.yml", "cloudbuild.yaml", ".cirrus.yml"})
TEST_CONFIG_FILES = frozenset({"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "noxfile.py", "package.json",
                               "mocha.opts", ".mocharc", ".mocharc.js", ".mocharc.cjs", ".mocharc.json", ".mocharc.yml", ".mocharc.yaml",
                               ".nycrc", "jest.config.js", "jest.config.ts", "jest.config.cjs", "jest.config.mjs", "jest.config.json",
                               "vitest.config.js", "vitest.config.ts", "vitest.config.mjs", "vitest.workspace.ts", "karma.conf.js",
                               "karma.conf.cjs", "jasmine.json", "babel.config.js", "codecov.yml"})
SKIP_MARKERS = (
    "pytest.mark.skip", "pytest.skip(", "pytest.mark.xfail", "pytest.xfail(", "unittest.skip", "@skip", ".skipTest(", "skipTest(",
    "t.Skip(", "t.SkipNow(", "t.Skipf(", "b.Skip(", "testing.Short()",
    "xit(", "xdescribe(", "xtest(", "it.skip(", "describe.skip(", "test.skip(", "it.only(", "describe.only(", "test.only(", "fit(", "fdescribe(",
    "@Ignore", "@Disabled", "#[ignore]", "pending(", ":skip", "skip: true",
)

FLAGS = ("empty_diff", "no_submit", "tests_touched", "ci_touched", "test_config_touched", "skip_markers", "verifier_touched")
VERIFIER_PATTERNS = (re.compile(r"(^|[\s'\"=:;|&(])/tests(/|\b)"), re.compile(r"/logs/verifier"), re.compile(r"reward\.(txt|json)"),
                     re.compile(r"(^|[\s/'\"])tests/test\.sh"))


@dataclass
class FileChange:
    path: str                           # new path (old path for deletions)
    old_path: str | None = None
    status: str = "modified"            # modified | added | deleted | renamed | binary
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    text: str = ""                      # the raw per-file section of the patch

    @property
    def paths(self) -> list[str]:
        return [p for p in (self.path, self.old_path) if p]


_HEADER = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def _strip_prefix(p: str) -> str:
    return p[2:] if p.startswith(("a/", "b/")) else p


def parse_diff(patch: str | None) -> list[FileChange]:
    """Per-file changes of a unified diff (git format); tolerant of odd headers, never raises."""
    if not patch or not patch.strip():
        return []
    changes: list[FileChange] = []
    sections: list[list[str]] = []
    for line in patch.splitlines():
        if line.startswith("diff --git ") or not sections:
            sections.append([])
        sections[-1].append(line)
    for lines in sections:
        head = lines[0]
        m = _HEADER.match(head)
        fc = FileChange(path=m.group(2), old_path=m.group(1)) if m else FileChange(path="")
        in_hunk = False
        body = lines[1:]
        for i, line in enumerate(body):
            # a file header, not a removed line starting with "-- ": outside a hunk, or followed by its "+++ " line
            header = not in_hunk or (line.startswith("--- ") and i + 1 < len(body) and body[i + 1].startswith("+++ "))
            if line.startswith("--- ") and header:
                in_hunk = False
                old = line[4:].split("\t")[0].strip()
                if old != "/dev/null" and not fc.old_path:
                    fc.old_path = _strip_prefix(old)
            elif line.startswith("+++ ") and not in_hunk:
                new = line[4:].split("\t")[0].strip()
                if new == "/dev/null":
                    fc.status = "deleted"
                elif not fc.path:
                    fc.path = _strip_prefix(new)
            elif line.startswith("new file mode"):
                fc.status = "added"
            elif line.startswith("deleted file mode"):
                fc.status = "deleted"
            elif line.startswith("rename to "):
                fc.status = "renamed"
                fc.path = line[len("rename to "):].strip()
            elif line.startswith("rename from "):
                fc.old_path = line[len("rename from "):].strip()
            elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                fc.status = "binary"
            elif line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                fc.added.append(line[1:])
            elif in_hunk and line.startswith("-"):
                fc.removed.append(line[1:])
        if fc.status == "deleted" and not fc.path and fc.old_path:
            fc.path = fc.old_path
        if fc.old_path == fc.path and fc.status == "modified":
            fc.old_path = None
        fc.text = "\n".join(lines) + "\n"
        if fc.path or fc.old_path:
            changes.append(fc)
    return changes


def classify_path(path: str) -> set[str]:
    """Subset of {'test', 'ci', 'test_config'} that applies to a repository path."""
    out: set[str] = set()
    norm = path.replace("\\", "/")
    while norm.startswith("./"):      # only an explicit "./" prefix: a bare lstrip("./") would eat the dot of .github/
        norm = norm[2:]
    norm = norm.lstrip("/")
    parts = norm.split("/")
    name = parts[-1]
    lower = name.lower()
    if any(p.lower() in TEST_DIR_NAMES for p in parts[:-1]) or any(pat.match(name) for pat in TEST_FILE_PATTERNS):
        out.add("test")
    if lower in TEST_CONFIG_FILES or lower.startswith((".mocharc", "jest.config", "vitest.config", "karma.conf")):
        out.add("test_config")
    if norm.startswith(CI_PREFIXES) or lower in CI_FILES:
        out.add("ci")
    return out


def skip_markers_in(changes: list[FileChange]) -> list[str]:
    """Skip/focus markers introduced by added lines (test files included: a skip added to a P2P test is tampering)."""
    found: list[str] = []
    for fc in changes:
        for line in fc.added:
            for marker in SKIP_MARKERS:
                if marker in line and marker not in found:
                    found.append(marker)
    return found


def split_patch(patch: str | None) -> tuple[str, list[dict[str, Any]]]:
    """(diff restricted to source paths, excluded sections [{path, reasons}]). Binary sections of source files are kept."""
    kept: list[str] = []
    excluded: list[dict[str, Any]] = []
    for fc in parse_diff(patch):
        reasons = sorted(set().union(*(classify_path(p) for p in fc.paths)))
        if reasons:
            excluded.append({"path": fc.path, "reasons": reasons})
        else:
            kept.append(fc.text)
    return "".join(kept), excluded


def _new_test_file(fc: FileChange) -> bool:
    """A test-named file the agent created (not a test-config or CI file): a scratch script, not a change to the tests."""
    return fc.status == "added" and classify_path(fc.path) == {"test"}


def scratch_test_files(patch: str | None) -> list[str]:
    """New test-named files of a diff (recorded in the verdict; they never force r_pass 0)."""
    return [fc.path for fc in parse_diff(patch) if _new_test_file(fc)]


def swe_flags(patch: str | None, exit_reason: str | None) -> list[str]:
    changes = parse_diff(patch)
    flags: list[str] = []
    if exit_reason != "submit":
        flags.append("no_submit")
    kinds: set[str] = set()
    for fc in changes:
        if _new_test_file(fc):
            continue
        for p in fc.paths:
            kinds |= classify_path(p)
    source_changes = [fc for fc in changes if not any(classify_path(p) for p in fc.paths)]
    if not source_changes:
        flags.append("empty_diff")
    if "test" in kinds:
        flags.append("tests_touched")
    if "ci" in kinds:
        flags.append("ci_touched")
    if "test_config" in kinds:
        flags.append("test_config_touched")
    if skip_markers_in([fc for fc in changes if not _new_test_file(fc)]):
        flags.append("skip_markers")
    return flags


def _keystrokes(messages: list[dict[str, Any]]) -> list[str]:
    """Commands a Terminus-2 transcript asked for (JSON `commands[].keystrokes`; raw text when the JSON does not parse)."""
    out: list[str] = []
    for m in messages:
        if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
            continue
        text = m["content"]
        start, end = text.find("{"), text.rfind("}")
        parsed = None
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("commands"), list):
            out.extend(str(c.get("keystrokes", "")) for c in parsed["commands"] if isinstance(c, dict))
        else:
            out.append(text)
    return out


def terminal_flags(messages: list[dict[str, Any]], exit_reason: str | None) -> list[str]:
    flags: list[str] = []
    if exit_reason != "task_complete":
        flags.append("no_submit")
    if any(pat.search(k) for k in _keystrokes(messages) for pat in VERIFIER_PATTERNS):
        flags.append("verifier_touched")
    return flags


def qa_flags(exit_reason: str | None) -> list[str]:
    """QA (hle_qa / chat) episodes have no filesystem to tamper with: the only flag is a generation that did not finish
    normally (length cap, error), which must never count as a positive."""
    return [] if exit_reason == "submit" else ["no_submit"]


def tamper_flags(episode: dict[str, Any], family: str) -> list[str]:
    """Flags for an Episode record (dict form) of the given family."""
    if family == "swe":
        return swe_flags(episode.get("patch"), episode.get("exit_reason"))
    if family == "terminal":
        return terminal_flags(episode.get("messages") or [], episode.get("exit_reason"))
    if family in ("qa", "chat"):
        return qa_flags(episode.get("exit_reason"))
    raise ValueError(f"no tamper rules for family {family!r}")
