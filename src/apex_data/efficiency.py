"""Efficiency profiler of agent trajectories.

One implementation serves the stage-1 gates, the data card and the stage-2 pair rules. Pure functions over a trajectory's
messages; the only heavy dependency is the token counter, which callers can replace.

Message shapes (both as stored upstream and as rendered by our harness templates):
  swe       mini-swe-agent tool calls: assistant {reasoning_content, content, tool_calls[{function: {arguments}}]} where
            the arguments are JSON with a `command`; each call is answered by a `tool` message holding the harness
            JSON ({returncode, output} or the elided {output_head, output_tail, ...}, optional exception_info).
            Submission is the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  terminal  Terminus-2: assistant {reasoning_content, content} where content is the strict JSON response
            (filters.terminus_parse); the next user message is the observation. Completion is task_complete.

Shell commands (ported from logs/data_plan_rev5/probes/oswt_cmds.py, unchanged where the probe defined a rule). A
command is split into segments outside quotes (&&, ||, ;, |, newline) after heredoc bodies are cut out; `cd <dir>`
segments move the working directory (start: the repository root); env assignments and timeout/time/sudo/env/nice/
stdbuf/xvfb-run/exec/command wrappers are skipped. Each segment gets the first kind it matches:
  edit    (probe EDIT) sed -i, perl -i, cat > | >> | <<, tee (not at a pipeline's end), echo/printf > <code file>,
          patch/git apply/apply_patch, python -c or python stdin-heredoc code that writes a file. Targets are
          repository-relative when under the root, None outside it; "" marks an unknown target.
  inline  (probe INLINE) python -c, python - <<heredoc, node -e|-p (not an edit).
  run     (probe RUN) test runners and program runs: pytest, python -m pytest|unittest|..., python <file>.py, go
          test|run, npm|yarn|pnpm test, jest/vitest/mocha, node <file>, make, cargo test|run, tox/nox, mvn/gradle test,
          phpunit. The terminal family also counts an interpreter given a script or code (bash, sh, zsh, ruby, perl,
          php, java, Rscript, ctest; npm|yarn|pnpm run|start) and an executable named by path that is not a read-only
          tool (./solve, /app/run.sh).
  write   (new) any other segment that redirects output into a file (not /dev/null) or starts with a file-system or
          package mutator (MUTATORS: rm, mv, cp, mkdir, touch, chmod, git checkout|reset|commit|..., pip install, ...).
  read    (new: the read-only command list READ_ONLY) inspection without side effects: ls, cat, grep, find (no
          -delete/-exec), sed without -i, git status|diff|log|show, ... and no file redirect or heredoc.
  other   anything else (including control keys and interactive input).

Detectors:
  thinking per turn   tokens of each assistant turn's reasoning_content (Qwen3.5-9B tokenizer by default).
  output tokens       tokens the model writes per turn: reasoning + visible content + tool-call commands (SWE). The
                      chat template's own tokens are not counted; the renderer (apex_train.data_format) gives exact
                      loss tokens of a rendered row.
  wasted action       an action whose normalised command (whitespace collapsed) and normalised observation both equal
                      an earlier action's (data plan 5.1; logs/data_plan_rev5/probes/dev_noprogress.py). Every tool
                      call counts as one action (SWE); a terminal turn's keystrokes are one action. Turns without a
                      command are counted as no-action turns, never as wasted.
  verified            verify before done (data plan 1.3): before the completion (SWE: the submit command; terminal: the
                      first task_complete turn, its own commands included), a verifying segment comes after the last
                      change. SWE: change = an edit touching the repository (or an unknown target); verifying = run or
                      inline, not while changes are stashed. Terminal: change = edit or write; verifying = run, inline or
                      read (inspecting the produced output, as quality.terminal_verify). With no change, one verifying
                      segment anywhere suffices.
  recovery arc        (probe arc B, data plan 3.1) scanned until the submit: an edit touching a non-test file of the
                      final patch (or an unknown target) -> a run (not under `git stash`) that fails (returncode != 0,
                      `Traceback (most recent call last)`, FAILED, FAIL:, --- FAIL, ^FAIL, panic:, "<n> failed") -> a
                      later edit touching any final-patch file -> a passing run (returncode 0, no FAIL, Traceback,
                      panic: or "<n> failed"). Within one command the edit precedes the run. SWE only; needs the patch.
  band                SWE: the teacher's solved share over attempts with a known outcome (resolved 1 or 0; -1 is
                      unknown); fewer than 2 known attempts is unbanded. 0 unsolved, <= 1/3 hard, < 1 medium, 1 easy.
                      Terminal: the six-model mean pass rate (apex_data.devset.terminal.band_of, the dev band rule).
  limits              per band (data plan 1.3): turn ceiling, window for the mean thinking per turn, steps, rendered
                      row, and for terminal the output per trajectory.

Stage-2 helpers (data plan 4.4 rules D, P, E): action_key, is_completion, unverified_completion, repeats_without_change.
Rendered rows: trained_thinking (reasoning tokens per trained turn, read from a row's token ids and mask).
`VERSION` names the rules; records written with another version are recomputed.
"""
from __future__ import annotations

import functools
import json
import posixpath
import re
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from apex_data import paths
from apex_data.devset.terminal import band_of as terminal_band  # noqa: F401 - the terminal band rule, re-exported
from apex_data.render.filters import terminus_parse
from apex_train.data_format import IM_END_ID

VERSION = "efficiency:1"
FAMILIES = ("swe", "terminal")
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TEACHER_REPO_ROOT = "/testbed"     # Open-SWE-Traces as stored
HARNESS_REPO_ROOT = "/app"         # our rendered rows and harness (teacher.render rewrites /testbed)

Counter = Callable[[Sequence[str]], list[int]]

# ------------------------------------------------------------------------------------------------ shell parsing (probe)
HEREDOC_RE = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1([^\n]*)\n(.*?)(?:\n[ \t]*\2[ \t]*(?=\n|$)|\Z)", re.S)
TOK = re.compile(r"'[^']*'?|\"(?:\\.|[^\"\\])*\"?|\\.|&&|\|\||[;|\n]|[^'\"\\;|&\n]+|&", re.S)
SEPS = {"&&", "||", ";", "|", "\n"}
WRAP_RE = re.compile(r"^(?:[({!]\s*)*(?:(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S*)\s+)|"
                     r"timeout\s+(?:-\S+\s+)*\d+(?:\.\d+)?[smhd]?\s+|time\s+|sudo\s+(?:-\S+\s+)*|"
                     r"env\s+(?:-\S+\s+)*|nice\s+(?:-n\s*\S+\s+)?|stdbuf\s+\S+\s+|xvfb-run\s+(?:-\S+\s+)*|"
                     r"exec\s+|command\s+)*")
PY = r"(?:\S*/)?python(?:3(?:\.\d+)?)?"
RUN_PATTERNS = (
    ("python -m pytest", re.compile(PY + r"\s+(?:-[A-Za-z]+\s+)*-m\s+(?:pytest|py\.test)\b")),
    ("python -m unittest", re.compile(PY + r"\s+(?:-[A-Za-z]+\s+)*-m\s+(?:unittest|nose2?|tox|doctest)\b")),
    ("python <file>.py", re.compile(PY + r"\s+(?:-[A-Za-z]+(?:\s+\S+)?\s+)*['\"]?[^\s'\"-][^\s'\"]*\.py\b")),
    ("pytest", re.compile(r"(?:\S*/)?(?:pytest|py\.test)\b")),
    ("go test", re.compile(r"go\s+test\b")),
    ("go run", re.compile(r"go\s+run\b")),
    ("npm/yarn/pnpm test", re.compile(r"(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b")),
    ("npx jest/vitest/mocha", re.compile(r"(?:npx|pnpm\s+exec|yarn)\s+(?:--?\S+\s+)*(?:jest|vitest|mocha|ava|tap)\b")),
    ("jest/vitest/mocha", re.compile(r"(?:\S*/)?(?:jest|vitest|mocha)\b")),
    ("node <file>", re.compile(r"node\s+(?:--?[\w-]+(?:=\S+)?\s+)*['\"]?[^\s'\"-]")),
    ("make", re.compile(r"make\b(?!\w)")),
    ("cargo test/run", re.compile(r"cargo\s+(?:\+\S+\s+)?(?:test|run|nextest)\b")),
    ("tox/nox", re.compile(r"(?:tox|nox)\b")),
    ("mvn/gradle test", re.compile(r"(?:\./)?(?:mvnw?|gradlew?)\b.*\btest\b")),
    ("phpunit", re.compile(r"(?:\S*/)?(?:phpunit|pest)\b|composer\s+(?:run\s+)?test\b")),
)
# terminal deliverables are scripts and programs as often as test suites (cf. the run patterns of quality.terminal_verify):
# an interpreter given a script or code, npm/yarn run|start, or an executable named by path (not a read-only tool)
TERMINAL_INTERPRETERS = re.compile(r"^(?:bash|sh|zsh|ruby|perl|php|java|Rscript|ctest|(?:npm|yarn|pnpm)\s+(?:run|start))\s+\S")
INLINE_PATTERNS = (
    ("python -c", re.compile(PY + r"\s+(?:-[A-Za-z]+\s+)*-c\b")),
    ("python - <<", re.compile(PY + r"\s+(?:-[A-Za-z]+\s+)*-?\s*<<")),
    ("node -e", re.compile(r"node\s+(?:--?[\w-]+\s+)*-(?:e|p)\b")),
)
SED_I = re.compile(r"^sed\s+(?:.*\s)?(?:-[a-zA-Z]*i[a-zA-Z.]*|--in-place)(?=\s|=|$)", re.S)
PERL_I = re.compile(r"^perl\s+(?:\S+\s+)*?-[a-zA-Z]*i", re.S)
CAT_W = re.compile(r"^cat\s*(?:-\s*)?>>?(?!&)|^cat\s+(?:-\S+\s+)*<<")
TEE = re.compile(r"tee\s")
ECHO_SRC = re.compile(r"^(?:echo|printf)\b.*?>>?\s*['\"]?[^\s'\"]+\.(?:py|go|rs|ts|tsx|js|jsx|mjs|java|php|c|cc|cpp|h|hpp|rb|"
                      r"toml|cfg|ini|yaml|yml|json)\b", re.S)
PATCH = re.compile(r"^patch\s+(?:-\S+\s+)*(?:<|-p\s*\d|-i\b)|^git\s+apply\b|^apply_patch\b")
PY_INLINE = re.compile(PY + r"\s+(?:-[A-Za-z]+\s+)*(?:-c\b|-?\s*<<)")
PY_WRITE = re.compile(r"open\([^)]*?,\s*(?:mode\s*=\s*)?['\"](?:w|a|r\+|wb|ab|w\+|a\+)['\"]|(?<!stdout)(?<!stderr)\.write\(|"
                      r"write_text\(|write_bytes\(|shutil\.(?:copy|move)|os\.replace\(|fileinput\.input\([^)]*inplace")
PATHLIT = re.compile(r"['\"]((?:/|\.{0,2}/)?[\w.\-/]+\.(?:py|go|rs|ts|tsx|js|jsx|mjs|cjs|java|php|c|cc|cpp|h|hpp|rb|toml|cfg|"
                     r"ini|yaml|yml|json|md|rst|txt|lock|html|css|scss|sh|sql|xml|proto|pyi|vue|svelte|kt|scala))['\"]")
REDIR = re.compile(r"(?<![0-9&<>])>>?\s*(['\"]?)([^\s'\";&|<>()]+)\1")
TEST_PATH = re.compile(r"(?i)(^|/)(tests?|testing|__tests__|spec|specs|testdata|fixtures)(/|$)|(^|/)test_[^/]*$|_test\.\w+$|"
                       r"\.(test|spec)\.\w+$|Tests?\.(java|php|kt|cs)$|(^|/)conftest\.py$")
STASH_PUSH = re.compile(r"\bgit\s+stash(?!\s+(?:pop|apply|list|show|drop|clear))\b")
STASH_POP = re.compile(r"\bgit\s+stash\s+(?:pop|apply)\b")
NOT_PASS = re.compile(r"FAIL|Traceback|panic:|\b[1-9]\d* failed\b")
FAIL_B = re.compile(r"Traceback \(most recent call last\)|FAILED|FAIL:|--- FAIL|^FAIL\b|panic:|\b[1-9]\d* failed\b", re.M)

# ------------------------------------------------------------------------------------- write and read-only lists (new)
MUTATORS = re.compile(r"^(?:rm|rmdir|mv|cp|mkdir|touch|chmod|chown|ln|truncate|install|unzip|dd\s.*\bof=|tar\s+(?:-\S*x|x)|"
                      r"git\s+(?:checkout|restore|reset|commit|am|clean|mv|rm|merge|rebase|cherry-pick|pull|stash|add|init|"
                      r"clone)|(?:pip3?|uv\s+pip|npm|yarn|pnpm|go|cargo|gem|composer|apt|apt-get|dnf|yum|apk|conda|brew)\s+"
                      r"(?:\S+\s+)*?(?:install|uninstall|add|remove|get|update|upgrade|ci)\b)")
READ_ONLY = frozenset((
    "ls", "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "fd", "sed", "awk",
    "gawk", "nl", "wc", "sort", "uniq", "cut", "tr", "column", "diff", "cmp", "comm", "file", "stat", "du", "df", "tree",
    "pwd", "which", "type", "whereis", "realpath", "readlink", "basename", "dirname", "echo", "printf", "env", "printenv",
    "id", "whoami", "hostname", "uname", "date", "md5sum", "sha1sum", "sha256sum", "xxd", "od", "hexdump", "strings",
    "jq", "yq", "true", "test", "[", "ps", "free", "nproc", "lsof", "git", "go", "npm", "pip", "pip3", "python",
    "python3", "node", "cargo", "rustc", "java", "gcc", "ldd", "nm", "objdump", "xargs",
))
_GIT_READ = re.compile(r"^git\s+(?:-C\s+\S+\s+|--no-pager\s+)*(?:status|diff|log|show|grep|ls-files|ls-tree|blame|rev-parse|"
                       r"describe|branch\s*$|branch\s+(?:-[avlr]+\s*)+$|remote\s+-v|config\s+--get|cat-file|shortlog)\b")
_TOOL_READ = re.compile(r"^(?:go\s+(?:list|version|env|doc|vet)|npm\s+(?:ls|list|view|outdated)|pip3?\s+(?:list|show|freeze)|"
                        r"(?:python3?|node|cargo|rustc|java|gcc)\s+(?:--version|-V|version)|cargo\s+(?:tree|metadata))\b")
_FIND_WRITE = re.compile(r"\s-(?:delete|exec|execdir|ok|okdir|fprint\w*|fls)\b")
_BACKGROUND = re.compile(r"(?<![&|>])&\s*$|\bnohup\b|\bsetsid\b|\bdisown\b|\bscreen\s+-\w*d|\btmux\s+new(?:-session)?\b.*\s-d\b",
                         re.M)


def normalise(text: Any) -> str:
    """Whitespace-collapsed text: the key of 'same command' and 'same output'."""
    return " ".join(str(text or "").split())


def strip_heredocs(cmd: str) -> tuple[str, list[str]]:
    bodies: list[str] = []

    def repl(m: re.Match) -> str:
        bodies.append(m.group(4))
        return "<<" + m.group(2) + m.group(3) + "\n"
    return HEREDOC_RE.sub(repl, cmd), bodies


def split_top(s: str) -> list[tuple[str, str]]:
    """(separator before, segment) pairs, split on &&, ||, ;, |, newline outside quotes."""
    segs, buf, sep = [], [], ""
    for m in TOK.finditer(s):
        t = m.group(0)
        if t in SEPS:
            segs.append((sep, "".join(buf)))
            buf, sep = [], t
        else:
            buf.append(t)
    segs.append((sep, "".join(buf)))
    return segs


def _norm_target(target: str, cwd: str, root: str) -> str | None:
    """Repository-relative path of a target, or None when outside the repository (e.g. /tmp)."""
    t = target.strip("'\"")
    if not t or t.startswith(("&", "$")):
        return None
    p = posixpath.normpath(t if t.startswith("/") else posixpath.join(cwd, t))
    return p[len(root) + 1:] if p.startswith(root + "/") else None


def _args(seg: str) -> list[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def _file_redirects(seg: str) -> list[str]:
    return [m.group(2) for m in REDIR.finditer(seg) if m.group(2) != "/dev/null"]


def _read_only(seg: str) -> bool:
    if _file_redirects(seg) or "<<" in seg:
        return False
    word = seg.split(None, 1)[0] if seg.split() else ""
    word = word.rsplit("/", 1)[-1] if word.startswith(("/", "./")) else word
    if word not in READ_ONLY:
        return False
    if word == "git":
        return bool(_GIT_READ.match(seg))
    if word in ("go", "npm", "pip", "pip3", "python", "python3", "node", "cargo", "rustc", "java", "gcc"):
        return bool(_TOOL_READ.match(seg))
    if word == "find":
        return not _FIND_WRITE.search(seg)
    if word in ("awk", "gawk"):
        return ">" not in seg and "system(" not in seg
    if word == "xargs":
        rest = [a for a in _args(seg)[1:] if not a.startswith("-")]
        return bool(rest) and _read_only(" ".join(rest))
    return True


def _terminal_run(seg: str) -> bool:
    word = seg.split(None, 1)[0]
    if word.startswith(("/", "./", "../")):
        return word.rsplit("/", 1)[-1] not in READ_ONLY
    return bool(TERMINAL_INTERPRETERS.match(seg))


@dataclass(frozen=True)
class Segment:
    """One shell segment: `kind` is edit, inline, run, write, read or other; `targets` are the edit's targets."""

    text: str
    kind: str
    label: str | None = None
    targets: tuple[str | None, ...] = ()


def segments(cmd: str, *, family: str = "swe", root: str = TEACHER_REPO_ROOT) -> list[Segment]:
    """The command's segments in order (cd segments move the working directory and are not returned)."""
    stripped, bodies = strip_heredocs(cmd or "")
    body_iter = iter(bodies)
    cwd, out = root, []
    for sep, raw in split_top(stripped):
        seg = WRAP_RE.sub("", raw.strip())
        if not seg:
            continue
        n_here = len(re.findall(r"<<[A-Za-z_]", seg))
        seg_bodies = [next(body_iter, "") for _ in range(n_here)]
        if seg.startswith("cd ") or seg == "cd":
            a = _args(seg)
            if len(a) > 1 and not a[1].startswith("-"):
                cwd = posixpath.normpath(a[1] if a[1].startswith("/") else posixpath.join(cwd, a[1]))
            continue
        lab, tg = None, []
        if SED_I.match(seg) or PERL_I.match(seg):
            lab = "sed -i" if seg.startswith("sed") else "perl -i"
            toks = _args(seg)[1:]
            has_e = any(t in ("-e", "--expression") or (t.startswith("-") and not t.startswith("--") and "e" in t[1:])
                        for t in toks)
            pos = [t for k, t in enumerate(toks) if not t.startswith("-")
                   and not (k and toks[k - 1] in ("-e", "--expression"))]   # scripts given with -e are not files
            tg = pos if has_e else pos[1:]
        elif CAT_W.match(seg) or (TEE.match(seg) and sep != "|") or ECHO_SRC.match(seg):
            # tee at the end of a pipeline only logs output, so it is not an edit
            tg = [m.group(2) for m in REDIR.finditer(seg) if m.group(2) != "/dev/null"]
            m = re.match(r"tee\s+((?:-\S+\s+)*)([^\s;&|<>]+)", seg)
            if m:
                tg.append(m.group(2))
            if tg:
                lab = "cat >/<<" if seg.startswith("cat") else ("tee" if seg.startswith("tee") else "echo/printf >")
        elif PATCH.match(seg):
            lab = "patch/git apply"
        elif PY_INLINE.match(seg):
            code = seg + "\n" + "\n".join(seg_bodies)
            if PY_WRITE.search(code):
                lab = "python write"
                tg = [m.group(1) for m in PATHLIT.finditer(code)]
        if lab:
            targets = [_norm_target(t, cwd, root) for t in tg if t and ("/" in t or "." in t)]
            if lab == "patch/git apply" or not tg:
                targets.append("")   # unknown target
            out.append(Segment(seg, "edit", lab, tuple(targets)))
            continue
        il = next((label for label, rx in INLINE_PATTERNS if rx.match(seg)), None)
        if il:
            out.append(Segment(seg, "inline", il))
            continue
        rl = next((label for label, rx in RUN_PATTERNS if rx.match(seg)), None)
        if rl is None and family == "terminal" and _terminal_run(seg):
            rl = "program run"
        if rl:
            out.append(Segment(seg, "run", rl))
        elif _file_redirects(seg) or MUTATORS.match(seg):
            out.append(Segment(seg, "write"))
        else:
            out.append(Segment(seg, "read" if _read_only(seg) else "other"))
    return out


def classify(cmd: str, *, family: str = "swe", root: str = TEACHER_REPO_ROOT) -> dict[str, Any]:
    """The probe's per-command summary: first edit label and all edit targets, first run and inline labels, stash."""
    segs = segments(cmd, family=family, root=root)
    edits = [s for s in segs if s.kind == "edit"]
    return {"edit": edits[0].label if edits else None,
            "targets": [t for s in edits for t in s.targets],
            "run": next((s.label for s in segs if s.kind == "run"), None),
            "inline": next((s.label for s in segs if s.kind == "inline"), None),
            "stash": bool(STASH_PUSH.search(strip_heredocs(cmd or "")[0]))}


def is_read_only(cmd: str, *, family: str = "swe") -> bool:
    """Every segment inspects without side effects (an empty command is not read-only)."""
    segs = segments(cmd, family=family)
    return bool(segs) and all(s.kind == "read" for s in segs)


def writes(cmd: str, *, family: str = "swe") -> bool:
    """The write detector: some segment edits a file, redirects into one, or mutates files or packages."""
    return any(s.kind in ("edit", "write") for s in segments(cmd, family=family))


def parse_obs(content: str | None) -> tuple[int | None, str]:
    """(returncode, output) of a mini-swe-agent observation; (None, text) when it is not the harness JSON."""
    c = content or ""
    try:
        o = json.loads(c)
    except (ValueError, TypeError):
        return None, c
    if not isinstance(o, dict):
        return None, c
    out = o.get("output")
    if out is None:
        out = (o.get("output_head") or "") + "\n" + (o.get("output_tail") or "")
    if o.get("exception_info"):
        out = (out or "") + "\n" + str(o["exception_info"])
    rc = o.get("returncode")
    try:
        rc = int(rc) if rc is not None else None
    except (TypeError, ValueError):
        rc = None
    return rc, out or ""


def stash_step(cmd: str, stashed: bool) -> tuple[bool, bool]:
    """(this command runs with the changes stashed, stashed afterwards): the probe's `git stash` tracking."""
    stripped = strip_heredocs(cmd or "")[0]
    push, pop = bool(STASH_PUSH.search(stripped)), bool(STASH_POP.search(cmd or ""))
    return push or (stashed and not pop), (push and not pop) or (stashed and not pop)


def run_failed(rc: int | None, output: str) -> bool:
    return (rc is not None and rc != 0) or bool(FAIL_B.search(output))


def run_passed(rc: int | None, output: str) -> bool:
    return rc == 0 and not NOT_PASS.search(output)


def patch_files(patch: str | None) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", patch or "", re.M))


def is_test_path(p: str) -> bool:
    return bool(TEST_PATH.search(p))


# ------------------------------------------------------------------------------------------------- trajectory views
def _get(m: Any, key: str) -> Any:
    return m.get(key) if isinstance(m, Mapping) else getattr(m, key, None)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else json.dumps(value))


def command_of(tool_call: Any) -> str:
    """The `command` argument of a bash tool call (the raw arguments text when it is not JSON with a command)."""
    fn = _get(tool_call, "function")
    args = _get(fn, "arguments") if fn is not None else _get(tool_call, "arguments")
    if isinstance(args, Mapping):
        return str(args.get("command", ""))
    try:
        a = json.loads(args or "")
        if isinstance(a, dict):
            return str(a.get("command", ""))
    except (ValueError, TypeError):
        pass
    return args or ""


def terminal_command(content: str) -> tuple[str, bool, bool]:
    """(keystrokes as one shell text, task_complete, parsed) of a Terminus-2 response."""
    p = terminus_parse(content or "")
    if p.error:
        return "", False, False
    text = "".join(k if k.endswith("\n") else k + "\n" for k, _ in p.commands if k)
    return text, p.task_complete, True


@dataclass
class Action:
    """One command and what came back. `turn` is the 0-based assistant turn."""

    turn: int
    command: str
    observation: str
    returncode: int | None = None
    output: str = ""
    completes: bool = False


def actions(messages: Sequence[Any], family: str) -> list[Action]:
    """Commands in order with their observations. SWE: one per tool call, answered by the tool messages in order
    (a call left unanswered gets an empty observation). Terminal: one per parsed turn with keystrokes or task_complete,
    observed by the next user message."""
    msgs, out, turn = list(messages), [], -1
    if family == "swe":
        pending: list[Action] = []
        for m in msgs:
            role = _get(m, "role")
            if role == "assistant":
                turn += 1
                pending = []
                for tc in _get(m, "tool_calls") or []:
                    cmd = command_of(tc)
                    a = Action(turn, cmd, "", completes=SUBMIT in cmd)
                    out.append(a)
                    pending.append(a)
            elif role == "tool" and pending:
                a = pending.pop(0)
                a.observation = _text(_get(m, "content"))
                a.returncode, a.output = parse_obs(a.observation)
        return out
    for i, m in enumerate(msgs):
        if _get(m, "role") != "assistant":
            continue
        turn += 1
        cmd, done, parsed = terminal_command(_text(_get(m, "content")))
        if not parsed or (not cmd and not done):
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) and _get(msgs[i + 1], "role") == "user" else None
        obs = _text(_get(nxt, "content")) if nxt is not None else ""
        out.append(Action(turn, cmd, obs, None, obs, completes=done))
    return out


def wasted_flags(acts: Sequence[Action]) -> list[bool]:
    """Per action: same normalised command and same normalised observation as an earlier action."""
    seen: set[tuple[str, str]] = set()
    flags = []
    for a in acts:
        key = (normalise(a.command), normalise(a.observation))
        flags.append(bool(key[0]) and key in seen)
        if key[0]:
            seen.add(key)
    return flags


def _verifying(seg: Segment, family: str) -> bool:
    return seg.kind in (("run", "inline") if family == "swe" else ("run", "inline", "read"))


def _changes(seg: Segment, family: str) -> bool:
    if family == "swe":   # an edit whose every target lies outside the repository (a /tmp scratch file) is no change
        return seg.kind == "edit" and (not seg.targets or any(t is not None for t in seg.targets))
    return seg.kind in ("edit", "write")


def verification(acts: Sequence[Action], family: str, *, root: str = TEACHER_REPO_ROOT) -> dict[str, Any]:
    """Verify before done over the actions up to the completion (data plan 1.3). `completed` says whether a completion
    was reached; `verified` whether a verifying segment followed the last change before it."""
    stashed, last_change, last_verify, k, completed = False, None, None, -1, False
    for a in acts:
        if family == "swe" and a.completes:
            completed = True
            break
        stash_run, stashed = stash_step(a.command, stashed)
        for seg in segments(a.command, family=family, root=root):
            k += 1
            if _changes(seg, family):
                last_change = (a.turn, k)
            elif _verifying(seg, family) and not (family == "swe" and stash_run):
                last_verify = (a.turn, k)
        if family == "terminal" and a.completes:
            completed = True
            break
    verified = last_verify is not None and (last_change is None or last_verify[1] > last_change[1])
    return {"completed": completed, "verified": verified,
            "last_change_turn": last_change[0] if last_change else None,
            "last_verify_turn": last_verify[0] if last_verify else None}


def recovery_arc(acts: Sequence[Action], patch: str | None, *, root: str = TEACHER_REPO_ROOT) -> list[int] | None:
    """The arc's four assistant turns (first edit, failing run, later edit, passing run), or None (probe arc B)."""
    pfiles = patch_files(patch)
    code_files = {f for f in pfiles if not is_test_path(f)}
    ev, stashed = [], False
    for a in acts:
        if a.completes:
            break
        k = classify(a.command, root=root)
        tg = k["targets"]
        unknown = "" in tg
        stash_run, stashed = stash_step(a.command, stashed)
        is_edit = k["edit"] is not None
        ev.append({"turn": a.turn,
                   "edit_code": is_edit and (unknown or any(t in code_files for t in tg if t)),
                   "edit_patch": is_edit and (unknown or any(t in pfiles for t in tg if t)),
                   "run": k["run"] is not None and not stash_run,
                   "fail": run_failed(a.returncode, a.output), "passed": run_passed(a.returncode, a.output)})
    state, idx = 0, []
    for i, e in enumerate(ev):
        if state == 0 and e["edit_code"]:
            state, idx = 1, [i]
        elif state == 2 and e["edit_patch"] and i > idx[1]:
            state = 3
            idx.append(i)
        if e["run"]:
            if state == 1 and e["fail"]:
                state = 2
                idx.append(i)
            elif state == 3 and e["passed"]:
                idx.append(i)
                return [ev[j]["turn"] for j in idx]
    return None


# ------------------------------------------------------------------------------------------------------- tokens
@functools.lru_cache(maxsize=1)
def _qwen_tokenizer() -> Any:
    from transformers import AutoTokenizer  # noqa: PLC0415 - heavy import, only when counting
    return AutoTokenizer.from_pretrained(str(paths.BASE_MODEL_DIR), local_files_only=True)


def qwen_token_counts(texts: Sequence[str]) -> list[int]:
    """Token counts with the Qwen3.5-9B tokenizer (no special tokens), batched."""
    texts = [t or "" for t in texts]
    if not texts:
        return []
    return [len(ids) for ids in _qwen_tokenizer()(texts, add_special_tokens=False)["input_ids"]]


def char_counts(texts: Sequence[str]) -> list[int]:
    """Character counts: the probes' unit (logs/data_plan_rev5/probes), for parity checks."""
    return [len(t or "") for t in texts]


char_counts.unit = "chars"   # a counter's unit (default "tokens"); band limits refuse other units


# ------------------------------------------------------------------------------------------------------- profile
@dataclass
class Profile:
    family: str
    steps: int                         # assistant turns
    thinking: list[int]                # reasoning tokens per assistant turn
    output_tokens: int                 # reasoning + content + tool-call commands, all turns
    n_actions: int
    wasted_actions: int
    wasted_turns: list[int]
    no_action_turns: int               # assistant turns without a command (SWE: no tool call; terminal: unparsed or empty)
    completed: bool
    verified: bool
    last_change_turn: int | None
    last_verify_turn: int | None
    recovery_arc: list[int] | None     # SWE with a patch only
    version: str = VERSION
    unit: str = "tokens"

    @property
    def thinking_mean(self) -> float:
        return sum(self.thinking) / len(self.thinking) if self.thinking else 0.0

    @property
    def thinking_max(self) -> int:
        return max(self.thinking, default=0)

    @property
    def wasted_share(self) -> float:
        return self.wasted_actions / self.n_actions if self.n_actions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "thinking_mean": round(self.thinking_mean, 2), "thinking_max": self.thinking_max,
                "thinking_total": sum(self.thinking), "wasted_share": round(self.wasted_share, 4)}


def profile(messages: Sequence[Any], family: str, *, patch: str | None = None, count: Counter = qwen_token_counts,
            root: str = TEACHER_REPO_ROOT) -> Profile:
    """Every detector over one trajectory. `root` is the repository root its SWE commands assume (TEACHER_REPO_ROOT
    for Open-SWE-Traces as stored, HARNESS_REPO_ROOT for our rendered rows); `count` maps texts to sizes in its
    `unit` attribute (tokens when absent). `patch` is the final model patch (SWE recovery arc)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    asst = [m for m in messages if _get(m, "role") == "assistant"]
    reasoning = [_text(_get(m, "reasoning_content")) for m in asst]
    written = [_text(_get(m, "content")) for m in asst]
    commands = [command_of(tc) for m in asst for tc in (_get(m, "tool_calls") or [])] if family == "swe" else []
    sizes = count([*reasoning, *written, *commands])
    thinking = sizes[:len(asst)]
    acts = actions(messages, family)
    flags = wasted_flags(acts)
    ver = verification(acts, family, root=root)
    acted = {a.turn for a in acts}
    return Profile(family=family, steps=len(asst), thinking=thinking, output_tokens=int(sum(sizes)),
                   n_actions=len(acts), wasted_actions=sum(flags),
                   wasted_turns=sorted({a.turn for a, f in zip(acts, flags) if f}),
                   no_action_turns=sum(1 for t in range(len(asst)) if t not in acted),
                   completed=ver["completed"], verified=ver["verified"], last_change_turn=ver["last_change_turn"],
                   last_verify_turn=ver["last_verify_turn"],
                   recovery_arc=recovery_arc(acts, patch, root=root) if family == "swe" and patch else None,
                   unit=getattr(count, "unit", "tokens"))


# ------------------------------------------------------------------------------------------------- bands and limits
def swe_band(resolved: Iterable[int]) -> str | None:
    """Band of an SWE instance from its attempts' `resolved` values (1 solved, 0 failed, -1 unknown)."""
    vals = list(resolved)
    known = [v for v in vals if v in (0, 1)]
    if len(known) < 2:
        return None
    share = sum(known) / len(known)
    if share == 0:
        return "unsolved"
    if share <= 1 / 3:
        return "hard"
    return "medium" if share < 1 else "easy"


@dataclass(frozen=True)
class BandLimits:
    """Data plan 1.3, in reasoning tokens per trained turn (Qwen3.5-9B tokenizer)."""

    turn_ceiling: int
    mean_max: int
    mean_min: int
    steps_max: int
    row_max: int
    output_max: int | None = None


SWE_LIMITS = {"easy": BandLimits(1000, 270, 0, 50, 32_768),
              "medium": BandLimits(1500, 360, 0, 65, 49_152),
              "hard": BandLimits(3000, 450, 180, 80, 65_536)}
TERMINAL_LIMITS = {"easy": BandLimits(1200, 450, 0, 15, 32_768, 21_600),
                   "medium": BandLimits(1800, 750, 0, 20, 49_152, 21_600),
                   "hard": BandLimits(2500, 1000, 300, 30, 65_536, 21_600)}
LIMITS = {"swe": SWE_LIMITS, "terminal": TERMINAL_LIMITS}


def limit_violations(p: Profile, band: str, *, row_tokens: int | None = None,
                     limits: Mapping[str, BandLimits] | None = None) -> list[str]:
    """The band limits a trajectory breaks (empty when within them). `row_tokens` is the longest rendered row;
    `limits` replaces the family's table (supply probes of other limits; the gates use the default)."""
    if p.unit != "tokens":
        raise ValueError("band limits are in tokens; profile with a token counter")
    lim = (limits or LIMITS[p.family])[band]
    out = []
    if p.thinking_max > lim.turn_ceiling:
        out.append("turn_ceiling")
    if p.thinking_mean > lim.mean_max:
        out.append("mean_above")
    if p.thinking_mean < lim.mean_min:
        out.append("mean_below")
    if p.steps > lim.steps_max:
        out.append("steps")
    if lim.output_max is not None and p.output_tokens > lim.output_max:
        out.append("output")
    if row_tokens is not None and row_tokens > lim.row_max:
        out.append("row")
    return out


def gate_failures(p: Profile, band: str, *, row_tokens: int | None = None,
                  limits: Mapping[str, BandLimits] | None = None) -> list[str]:
    """Profiler gates of data plan 5.1: band limits, zero wasted actions, verification before completion."""
    out = limit_violations(p, band, row_tokens=row_tokens, limits=limits)
    if p.wasted_actions:
        out.append("wasted_actions")
    if not p.completed:
        out.append("not_completed")
    elif not p.verified:
        out.append("unverified_completion")
    return out


# --------------------------------------------------------------------------------------------------- rendered rows
THINK_CLOSE_ID = 248069     # `</think>` in the Qwen3.5-9B tokenizer


def trained_thinking(input_ids: Sequence[int], assistant_masks: Sequence[int]) -> list[int | None]:
    """Reasoning tokens of each trained turn of a rendered row, as trained: per trained span (a run of mask 1, split
    after each <|im_end|>), the tokens before its first `</think>` (the reasoning plus the newline the template puts
    before the tag); None for a span without one (unclosed reasoning). The band limits are per trained turn (data plan
    1.3), so a rendered trajectory's authoritative gate is
    `gate_failures(dataclasses.replace(profile, thinking=<its trained turns' values>), band, row_tokens=...)`."""
    spans: list[list[int]] = []
    cur: list[int] | None = None
    for tok, m in zip(input_ids, assistant_masks):
        if not m:
            cur = None
            continue
        if cur is None:
            cur = []
            spans.append(cur)
        cur.append(tok)
        if tok == IM_END_ID:
            cur = None
    return [s.index(THINK_CLOSE_ID) if THINK_CLOSE_ID in s else None for s in spans]


# ------------------------------------------------------------------------------------------------ stage-2 helpers
def action_key(message: Any, family: str) -> str:
    """The normalised action of one assistant turn ("" = no action): SWE the commands of its tool calls; terminal the
    keystrokes and the task_complete flag of its parsed response."""
    if family == "swe":
        return normalise(" ; ".join(command_of(tc) for tc in (_get(message, "tool_calls") or [])))
    p = terminus_parse(_text(_get(message, "content")))
    if p.error or (not p.commands and not p.task_complete):
        return ""
    return json.dumps([[normalise(k), d] for k, d in p.commands] + [p.task_complete])


def is_completion(message: Any, family: str) -> bool:
    if family == "swe":
        return any(SUBMIT in command_of(tc) for tc in (_get(message, "tool_calls") or []))
    p = terminus_parse(_text(_get(message, "content")))
    return not p.error and p.task_complete


def unverified_completion(history: Sequence[Any], candidate: Any, family: str, *, root: str = HARNESS_REPO_ROOT) -> bool:
    """Rule D: the candidate completes while the last change is unverified (its own commands count first)."""
    if not is_completion(candidate, family):
        return False
    acts = actions([*history, candidate], family)
    return not verification(acts, family, root=root)["verified"]


def repeats_without_change(history: Sequence[Any], candidate: Any, family: str) -> bool:
    """Rule P: the candidate is an empty action, or a read-only command already run with only read-only commands since
    and no background job started earlier (a job's output can change on its own)."""
    if not action_key(candidate, family):
        return True
    cand = [a.command for a in actions([candidate], family)]
    if len(cand) != 1 or not is_read_only(cand[0], family=family):
        return False
    past = [a.command for a in actions(history, family)]
    if any(_BACKGROUND.search(c) for c in past):
        return False
    key = normalise(cand[0])
    for c in reversed(past):
        if normalise(c) == key:
            return True
        if not is_read_only(c, family=family):
            return False
    return False
