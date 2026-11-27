"""Rebuild one Open-SWE-Traces trajectory under our mini-swe-agent 2.4.6 harness templates as a `schema.Episode` for the
shared gates of `apex_data.render.filters` (data plan 2.3, 3.2 filters 3 and 4; README rule 7: the only renderer is
`apex_train.data_format`, reached through `filters.filter_episodes`).

Which templates. Our SWE-bench Pro harness (configs/eval/swebench_pro_v2_hard.yaml `agent.config_file`, written by the
runner into each run's config_snapshot/mini_swe_agent_config.yaml) carries the `agent.system_template`,
`agent.instance_template` and `model.observation_template` of the installed package's `config/mini.yaml` byte for byte.
This module imports `config/mini.yaml` and `BASH_TOOL` from the installed package (through the harbor interpreter; the
train venv does not ship mini-swe-agent), asserts the version, and asserts that the three templates equal a runner-written
snapshot (default: configs/train_stack_check/harness_request_shapes/captures/mini_swe_agent_config.generated.yaml; any later
snapshot can be passed as `lock_config`). It also asserts that the installed `BASH_TOOL` equals the tool schema the shared
adapter renders (`adapters.MINI_SWE_TOOLS`).

What is rebuilt and from where
* System and first user prompt: the `mini.yaml` templates rendered with jinja2 as `DefaultAgent._render_template` does
  (`{{task}}` plus the host's `platform.uname()` fields for `<system_information>`; the harness loop runs on this host).
  The task text is cut from the teacher's first user message, whose layout is the package's
  `config/benchmarks/swebench.yaml` (`<pr_description>` ... `</pr_description>`); only that file's anchors are used.
* Assistant turns verbatim: `reasoning_content`, `content`, tool calls (one `bash` call with a string `command`, nothing
  else, per call). One normalisation, counted: the teacher's layout asks for `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  && cat patch.txt` as the final command (sometimes prefixed with `cd <dir> &&`), our harness prompt for the lone
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` ("Do not combine it with any other command"); the final command is
  rewritten to the lone form (the shared exit gate requires it). Every earlier turn is kept as is.
* Observations: the stored harness JSON (`{returncode, output}`, the elided `{output_head, output_tail, elided_chars,
  warning}` form at >= 10,000 characters, optional `exception_info`) is parsed back and re-rendered through the installed
  observation template; an elided output is reinflated to its recorded length so the template re-elides it. Identical and
  differing re-renders are counted.
* Repository path (`repo_root`, the builder passes /app): the teacher worked with the repository at /testbed; our
  SWE-bench Pro harness starts the agent in the repository at /app (it runs every command in a fresh shell from there)
  and the prompt never names the path. Every command first loses a leading `cd /testbed && ` (also `;` or a newline),
  which is a no-op from that working directory; a leading `cd /testbed/<sub> && ` becomes `cd <sub> && `. Then the
  root token (ROOT_TOKEN_RE: `/testbed` at a path start and a name end) becomes `repo_root` in every message: task
  text, reasoning, content, commands, and tool outputs (on the parsed output before the observation template
  re-renders it; an elided observation keeps its recorded `elided_chars`). Other `/testbed` strings (conda paths such
  as envs/testbed, relative ../testbed, /tmp/x/testbed) stay and are counted. An episode with a command whose meaning
  the rewrite could change (`command_risks`: the bare word `testbed` outside a conda environment name, a `/testbed`
  path the boundary rule skips, `cd /testbed && cd ..`) is dropped (`repo_root_ambiguous`), never guessed at.
* Format-error user turns, missing or extra tool calls, unparseable observations and a missing final submit raise
  `RenderDrop` (episodes are dropped, never spliced; plan 6).

Reference-patch check (plan 3.2 filter 4: no reference-patch line inside the rendered prompt). The substantive added lines
(>= LEAK_MIN_CHARS non-blank characters) of the reference patch's non-test files are matched against the system and user
text (stripped-line match, or substring for lines of 40+ characters). The caller drops the row when the matched share
reaches its threshold (0 = any matched line, the plan's literal rule); the overlap count is recorded on every trajectory.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import platform
import re
import subprocess
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template

from apex_data import paths
from apex_data.grade.tamper import classify_path, parse_diff
from apex_data.render.adapters import MINI_SWE_FORMAT_ERROR_PREFIX, MINI_SWE_SUBMIT_COMMAND, MINI_SWE_TOOLS
from apex_data.schema import Episode
from apex_train.data_format import normalise_tools

REQUIRED_MINI_VERSION = "2.4.6"
HARNESS_CONFIG = "mini.yaml"                     # installed config whose templates equal the harness lock
TEACHER_CONFIG = "benchmarks/swebench.yaml"      # the layout the teacher's prompts were rendered with (anchors only)
LOCK_SNAPSHOT = paths.CONFIGS / "train_stack_check" / "harness_request_shapes" / "captures" / "mini_swe_agent_config.generated.yaml"
LOCK_FIELDS = (("agent", "system_template"), ("agent", "instance_template"), ("model", "observation_template"))
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ROUND, POLICY, SCAFFOLD = "teacher", "qwen38_27b", "mini_swe_toolcall"
LEAK_MIN_CHARS = 12
# Format-error turns start "Tool call error:" in our harness (adapters.MINI_SWE_FORMAT_ERROR_PREFIX) and "Tool call error."
# in the teacher's; both are the harness answering a malformed response.
FORMAT_ERROR_HEAD = MINI_SWE_FORMAT_ERROR_PREFIX.rstrip(":")
DEFAULT_HARBOR_PYTHON = paths.REPO / ".venv" / "harbor" / "bin" / "python"
ELIDED_KEYS = ("output_head", "output_tail", "elided_chars")
UNAME_KEYS = ("system", "node", "release", "version", "machine")
_TASK_SENTINEL = "\x00APEX_TASK\x00"
TEACHER_REPO_ROOT = "/testbed"
# The repository-root token: `/testbed` at a path start (start of text, whitespace, a quote, =, :, (, `, [, {, comma, < or >
# before it) and at a name end (not followed by a word character, `-`, or `.` plus a word character).
_ROOT_LEFT = " \t\n\r'\"`=:([{,<>"
ROOT_TOKEN_RE = re.compile(r"(?<![^\s'\"`=:(\[{,<>])/testbed(?![\w\-]|\.\w)")
ROOT_CANDIDATE_RE = re.compile(r"/testbed(?![\w\-]|\.\w)")            # the same without the left boundary
ROOT_CD_PREFIX_RE = re.compile(r"^\s*cd\s+/testbed/?\s*(?:&&|;|\n)\s*")   # a leading `cd /testbed && ` (also ; and newline)
SUB_CD_PREFIX_RE = re.compile(r"^(\s*)cd\s+/testbed/([^\s;&|'\"]+?)/?(\s*(?:&&|;|\n))")
BARE_WORD_RE = re.compile(r"(?<![\w/.\-])testbed\b")                  # the repository named as a word
CONDA_NAME_RE = re.compile(r"(?:activate|-n|--name)\s+testbed\b")       # a conda environment called testbed
_PROBE = textwrap.dedent("""
    import json
    import minisweagent
    from minisweagent.config import builtin_config_dir
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    out = {"version": minisweagent.__version__, "bash_tool": BASH_TOOL, "configs": {}}
    for name in %r:
        p = builtin_config_dir / name
        out["configs"][name] = {"path": str(p), "text": p.read_text(encoding="utf-8")}
    print(json.dumps(out))
""" % ([HARNESS_CONFIG, TEACHER_CONFIG],))


class TemplateError(RuntimeError):
    """The installed mini-swe-agent could not be imported, has the wrong version, or its templates differ from the lock."""


class RenderDrop(Exception):
    """The trajectory cannot be rebuilt in our harness grammar; `reason` is the funnel key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


@dataclass
class TemplateBundle:
    """The harness templates, tool schema and template variables, as imported from the installed package."""

    version: str
    config_path: str                      # installed mini.yaml
    config_sha256: str
    system_template: str
    instance_template: str
    observation_template: str
    bash_tool: dict[str, Any]
    teacher_prefix: str                   # literal text before the task in the teacher's prompt layout
    teacher_close: str                    # first line after the task in that layout ('</pr_description>')
    teacher_config_path: str
    template_vars: dict[str, str]         # platform.uname() fields the instance template reads
    lock_reference: dict[str, Any] = field(default_factory=dict)

    @property
    def bash_tool_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.bash_tool, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"bash_tool_sha256": self.bash_tool_sha256}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemplateBundle:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


def host_template_vars() -> dict[str, str]:
    """What the harness environment supplies to the templates: this host's platform.uname()."""
    u = platform.uname()._asdict()
    return {k: str(u.get(k, "")) for k in UNAME_KEYS}


def _lock_templates(path: Path) -> dict[tuple[str, str], str]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {(sec, key): (cfg.get(sec) or {}).get(key) for sec, key in LOCK_FIELDS}


def load_templates(harbor_python: Path | str | None = None, *, lock_config: Path | str | None = None,
                   template_vars: dict[str, str] | None = None, timeout: int = 120) -> TemplateBundle:
    """Import the installed mini-swe-agent (harbor venv), take the harness templates from its config/mini.yaml and the
    teacher-layout anchors from config/benchmarks/swebench.yaml, and assert version, lock equality and tool-schema
    equality. Raises TemplateError when anything is missing or differs."""
    py = Path(harbor_python or os.environ.get("APEX_HARBOR_PYTHON") or DEFAULT_HARBOR_PYTHON)
    if not py.exists():
        raise TemplateError(f"harbor interpreter not found at {py} (mini-swe-agent {REQUIRED_MINI_VERSION} lives there)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")} | {"MSWEA_SILENT_STARTUP": "1"}
    proc = subprocess.run([str(py), "-I", "-c", _PROBE], capture_output=True, text=True, timeout=timeout, env=env, check=False)
    if proc.returncode != 0:
        raise TemplateError(f"importing minisweagent with {py} failed: {proc.stderr.strip()[-800:]}")
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise TemplateError(f"unexpected probe output: {proc.stdout[-400:]!r}") from exc
    if info["version"] != REQUIRED_MINI_VERSION:
        raise TemplateError(f"mini-swe-agent {info['version']} installed, {REQUIRED_MINI_VERSION} required")
    if normalise_tools([info["bash_tool"]], "installed") != normalise_tools(MINI_SWE_TOOLS, "adapters"):
        raise TemplateError("installed BASH_TOOL differs from apex_data.render.adapters.MINI_SWE_TOOLS")
    harness, teacher = info["configs"][HARNESS_CONFIG], info["configs"][TEACHER_CONFIG]
    cfg = yaml.safe_load(harness["text"])
    try:
        system_t, instance_t = cfg["agent"]["system_template"], cfg["agent"]["instance_template"]
        observation_t = cfg["model"]["observation_template"]
        teacher_instance = yaml.safe_load(teacher["text"])["agent"]["instance_template"]
    except (KeyError, TypeError) as exc:
        raise TemplateError(f"installed config lacks template key {exc}") from exc
    if "{{task}}" not in instance_t.replace(" ", "") or "{{task}}" not in teacher_instance.replace(" ", ""):
        raise TemplateError("instance_template has no {{task}} placeholder")
    ref_path = Path(lock_config) if lock_config else LOCK_SNAPSHOT
    if not ref_path.exists():
        raise TemplateError(f"lock reference {ref_path} not found; pass lock_config=<runner-written mini-swe-agent config>")
    ref = _lock_templates(ref_path)
    ours = {("agent", "system_template"): system_t, ("agent", "instance_template"): instance_t,
            ("model", "observation_template"): observation_t}
    differing = [f"{sec}.{key}" for (sec, key), text in ours.items() if ref.get((sec, key)) != text]
    if differing:
        raise TemplateError(f"installed {HARNESS_CONFIG} differs from the lock reference {ref_path} in {differing}")
    tvars = dict(template_vars or host_template_vars())
    prefix, _, suffix = _render(teacher_instance, task=_TASK_SENTINEL, **tvars).partition(_TASK_SENTINEL)
    close = next((line for line in suffix.splitlines() if line.strip()), "")
    return TemplateBundle(version=info["version"], config_path=harness["path"],
                          config_sha256=hashlib.sha256(harness["text"].encode()).hexdigest(), system_template=system_t,
                          instance_template=instance_t, observation_template=observation_t, bash_tool=info["bash_tool"],
                          teacher_prefix=prefix, teacher_close=close, teacher_config_path=teacher["path"],
                          template_vars=tvars,
                          lock_reference={"path": str(ref_path), "sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                                          "matched": [f"{s}.{k}" for s, k in LOCK_FIELDS]})


# ------------------------------------------------------------------------------------------- templates
@functools.lru_cache(maxsize=32)
def _compiled(template: str) -> Template:
    return Template(template, undefined=StrictUndefined)       # as DefaultAgent._render_template, compiled once


def _render(template: str, **variables: Any) -> str:
    return _compiled(template).render(**variables)


def render_system(bundle: TemplateBundle) -> str:
    return _render(bundle.system_template, **bundle.template_vars)


def render_instance(bundle: TemplateBundle, task: str) -> str:
    return _render(bundle.instance_template, task=task, **bundle.template_vars)


def extract_task(user_text: str, bundle: TemplateBundle) -> str | None:
    """Task text of the teacher's first user message (its `<pr_description>` layout): between the layout's literal
    prefix and the last occurrence of its closing line; None when the message is not in that layout."""
    if not user_text.startswith(bundle.teacher_prefix) or not bundle.teacher_close:
        return None
    k = user_text.rfind("\n" + bundle.teacher_close)
    return user_text[len(bundle.teacher_prefix): k] if k >= len(bundle.teacher_prefix) else None


def observation_output(content: Any) -> dict[str, Any] | None:
    """The `output` mapping the observation template expects, rebuilt from the stored observation JSON."""
    if isinstance(content, dict):
        obs = content
    else:
        try:
            obs = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            return None
    if not isinstance(obs, dict) or "returncode" not in obs:
        return None
    if all(k in obs for k in ELIDED_KEYS):
        head, tail, elided = obs["output_head"], obs["output_tail"], obs["elided_chars"]
        if not (isinstance(head, str) and isinstance(tail, str) and isinstance(elided, int) and elided >= 0):
            return None
        text = head + ("\0" * elided) + tail            # same length as the original: the template re-elides it
    elif isinstance(obs.get("output"), str):
        text = obs["output"]
    else:
        return None
    return {"output": text, "returncode": obs["returncode"], "exception_info": obs.get("exception_info") or ""}


def render_observation(bundle: TemplateBundle, content: Any) -> str | None:
    out = observation_output(content)
    return None if out is None else _render(bundle.observation_template, output=out)


def _tojson(value: Any) -> str:
    """Jinja2's default `tojson` filter (json.dumps with sort_keys plus HTML-safe escapes), as the template applies it."""
    return (json.dumps(value, sort_keys=True).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("'", "\\u0027"))


def elided_observation(returncode: Any, head: str, tail: str, elided_chars: int, exception_info: str = "") -> str:
    """The elided branch of the observation template with explicit head and tail (used after a path rewrite changed
    them; byte-equal to the template when they are its own 5,000-character slices)."""
    exc = f', "exception_info": {_tojson(exception_info)}' if exception_info else ""
    return (f'{{\n  "returncode": {returncode},\n  "output_head": {_tojson(head)},\n  "output_tail": {_tojson(tail)},\n'
            f'  "elided_chars": {elided_chars},\n  "warning": "Output too long."{exc}\n}}')


def _skip_category(text: str, i: int) -> str:
    """Why the candidate `/testbed` at index i is not the repository root."""
    if text[:i].endswith("envs"):
        return "conda_env_path"
    left = text[i - 1]
    if left == ".":
        return "relative_path"
    if left == "/" or left.isalnum() or left == "_":
        return "path_segment"
    return "other_left_character"


def command_risks(command: str) -> list[str]:
    """Reasons the root rewrite could change what a command does (the episode is then dropped)."""
    risks = []
    m = ROOT_CD_PREFIX_RE.match(command)
    if m and re.match(r"cd\s+\.\.", command[m.end():]):
        risks.append("cd_parent_after_root_cd")
    if BARE_WORD_RE.search(CONDA_NAME_RE.sub("", command)):
        risks.append("bare_word_testbed")
    if any(c.start() and command[c.start() - 1] not in _ROOT_LEFT and _skip_category(command, c.start()) != "conda_env_path"
           for c in ROOT_CANDIDATE_RE.finditer(command)):
        risks.append("non_root_testbed_path")
    return risks


class RootRewriter:
    """Applies the repository-path rewrite to one episode and counts what it did (identity when root is None)."""

    def __init__(self, root: str | None) -> None:
        self.root = None if root in (None, "", TEACHER_REPO_ROOT) else root
        self.counts: Counter = Counter()

    def __call__(self, text: str | None) -> str | None:
        """The root token -> root; candidates the boundary rule skips are counted by category."""
        if self.root is None or not text or TEACHER_REPO_ROOT not in text:
            return text
        for c in ROOT_CANDIDATE_RE.finditer(text):
            if c.start() and text[c.start() - 1] not in _ROOT_LEFT:
                self.counts[f"skipped_{_skip_category(text, c.start())}"] += 1
        new, n = ROOT_TOKEN_RE.subn(self.root, text)
        self.counts["paths_rewritten"] += n
        return new

    def command(self, command: str) -> str:
        """Drop leading `cd /testbed && ` prefixes, make a leading `cd /testbed/<sub> &&` relative, then rewrite."""
        if self.root is None:
            return command
        while m := ROOT_CD_PREFIX_RE.match(command):
            command = command[m.end():]
            self.counts["cd_prefixes_deleted"] += 1
        command, n = SUB_CD_PREFIX_RE.subn(r"\1cd \2\3", command, count=1)
        self.counts["cd_sub_made_relative"] += n
        return self(command)

    def observation(self, bundle: TemplateBundle, content: Any, original: str) -> str:
        """The re-rendered observation with the rewrite applied to the parsed output (`original` = the re-render
        without rewrite)."""
        if self.root is None:
            return original
        obs = content if isinstance(content, dict) else json.loads(content)
        exc = self(obs.get("exception_info") or "")
        if all(k in obs for k in ELIDED_KEYS):
            head, tail = self(obs["output_head"]), self(obs["output_tail"])
            if (head, tail, exc) == (obs["output_head"], obs["output_tail"], obs.get("exception_info") or ""):
                return original
            return elided_observation(obs["returncode"], head, tail, obs["elided_chars"], exc)
        output = self(obs["output"])
        return _render(bundle.observation_template, output={"output": output, "returncode": obs["returncode"],
                                                            "exception_info": exc})


# ------------------------------------------------------------------------------------- leakage check
def nontest_added_lines(patch: str | None, min_chars: int = LEAK_MIN_CHARS) -> list[str]:
    """Distinct added lines (stripped) of the patch's non-test files with at least `min_chars` non-blank characters."""
    out: list[str] = []
    seen: set[str] = set()
    for fc in parse_diff(patch):
        if any("test" in classify_path(p) for p in fc.paths):
            continue
        for line in fc.added:
            body = line.strip()
            if len("".join(body.split())) >= min_chars and body not in seen:
                seen.add(body)
                out.append(body)
    return out


def reference_patch_overlap(texts: list[str], reference_patch: str | None) -> tuple[list[str], int]:
    """(matched lines, substantive added lines) of the reference patch found in `texts` (stripped-line match, or a
    substring match for lines of 40+ characters)."""
    hay = "\n".join(line.strip() for text in texts for line in text.splitlines())
    hay_lines = set(hay.splitlines())
    added = nontest_added_lines(reference_patch)
    return [line for line in added if line in hay_lines or (len(line) >= 40 and line in hay)], len(added)


# ------------------------------------------------------------------------------------------- episodes
# The teacher's submit commands: `[cd <dir> && ]echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT[ && cat patch.txt]`; mini-swe-agent
# submits when the first output line is the marker, which every variant here satisfies.
_SUBMIT_RE = re.compile(r"^(?:cd\s+\S+\s*&&\s*)*echo\s+" + SUBMIT_MARKER + r"\s*(?:(?:&&|;)\s*.*)?$", re.DOTALL)


def is_submit(command: str) -> bool:
    return bool(_SUBMIT_RE.match(command.strip()))


def _bash_calls(m: dict[str, Any], i: int, rewrite: RootRewriter) -> list[dict[str, Any]]:
    """The turn's tool calls as `{id, type, function: {name, arguments: <JSON string>}}` with exactly one string
    `command` argument each; RenderDrop otherwise (the harness would have answered with a format error)."""
    calls = []
    for j, tc in enumerate(m.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError as exc:
            raise RenderDrop("parse_invalid", f"message {i} call {j}: arguments are not JSON ({exc})") from exc
        if fn.get("name") != "bash" or not isinstance(parsed, dict) or set(parsed) != {"command"} \
                or not isinstance(parsed["command"], str):
            raise RenderDrop("parse_invalid", f"message {i} call {j}: not a bash call with exactly one string command")
        calls.append({"id": tc.get("id"), "type": "function",
                      "function": {"name": "bash", "arguments": json.dumps({"command": rewrite.command(parsed["command"])})}})
    if not calls:
        raise RenderDrop("no_tool_call", f"assistant message {i} issued no tool call (the harness answers a format error)")
    return calls


def build_episode(raw: dict[str, Any], bundle: TemplateBundle, *, task_id: str, episode_id: str,
                  normalise_submit: bool = True, repo_root: str | None = None) -> tuple[Episode, dict[str, Any]]:
    """Raw shard row -> schema.Episode (scaffold mini_swe_toolcall, exit_reason submit) in our harness grammar, plus
    rebuild statistics; `repo_root` rewrites the teacher's /testbed repository path (None keeps it). Raises
    RenderDrop(reason) for rows that cannot be rebuilt, `repo_root_ambiguous` when a command makes the rewrite unsafe."""
    msgs = list(raw.get("messages") or [])
    if len(msgs) < 3 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        raise RenderDrop("unexpected_prefix", "expected system, user, assistant ... messages")
    task = extract_task(msgs[1].get("content") or "", bundle)
    if task is None:
        raise RenderDrop("prompt_template_mismatch", "first user message is not in the teacher's <pr_description> layout")
    risks: Counter = Counter()
    if repo_root not in (None, "", TEACHER_REPO_ROOT):
        for m in msgs[2:]:
            for tc in m.get("tool_calls") or []:
                try:
                    cmd = json.loads((tc.get("function") or {}).get("arguments") or "{}").get("command")
                except (json.JSONDecodeError, AttributeError):
                    cmd = None                               # _bash_calls raises the drop below
                if isinstance(cmd, str):
                    risks.update(command_risks(cmd))
    if risks:
        raise RenderDrop("repo_root_ambiguous", json.dumps(dict(sorted(risks.items()))))
    rewrite = RootRewriter(repo_root)
    task = rewrite(task)
    out: list[dict[str, Any]] = [{"role": "system", "content": render_system(bundle)},
                                 {"role": "user", "content": render_instance(bundle, task)}]
    for i, m in enumerate(msgs[2:], 2):        # harness-injected user turns: dropped, never spliced (plan 6)
        if m.get("role") == "user":
            content = m.get("content") or ""
            reason = "format_error_turn" if content.lstrip().startswith(FORMAT_ERROR_HEAD) else "harness_user_turn"
            raise RenderDrop(reason, f"message {i} is a harness-injected user turn: {content[:60]!r}")
    stats = {"n_assistant": 0, "n_tool_calls": 0, "n_observations": 0, "n_elided_observations": 0,
             "n_exception_observations": 0, "observations_identical": 0, "observations_rerendered_differently": 0,
             "parallel_call_turns": 0, "trailing_observations_cut": 0, "task_chars": len(task), "submit_normalised": 0}
    i = 2
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") != "assistant":
            raise RenderDrop("role_sequence", f"message {i} is {m.get('role')!r} where an assistant turn was expected")
        calls = _bash_calls(m, i, rewrite)
        out.append({"role": "assistant", "content": rewrite(m.get("content") or ""),
                    "reasoning_content": rewrite(m.get("reasoning_content")), "tool_calls": calls})
        stats["n_assistant"] += 1
        stats["n_tool_calls"] += len(calls)
        stats["parallel_call_turns"] += len(calls) > 1
        j = i + 1
        obs: list[dict[str, Any]] = []
        while j < len(msgs) and msgs[j].get("role") == "tool":
            obs.append(msgs[j])
            j += 1
        if j >= len(msgs):                     # the observation of the final action is never answered by the model
            stats["trailing_observations_cut"] = len(obs)
            obs = []
        elif len(obs) != len(calls):
            raise RenderDrop("observation_count_mismatch", f"message {i}: {len(calls)} calls, {len(obs)} observations")
        for call, o in zip(calls, obs):
            content = o.get("content")
            output = observation_output(content)
            if output is None:
                raise RenderDrop("observation_unparseable", f"message {i}: observation is not the harness output dict")
            rendered = _render(bundle.observation_template, output=output)
            stats["n_elided_observations"] += isinstance(content, str) and '"output_head"' in content
            stats["n_exception_observations"] += bool(output["exception_info"])
            stats["observations_identical" if rendered == content else "observations_rerendered_differently"] += 1
            out.append({"role": "tool", "content": rewrite.observation(bundle, content, rendered), "name": "bash",
                        "tool_call_id": call["id"]})
            stats["n_observations"] += 1
        i = j
    last = out[-1]
    if last["role"] != "assistant" or len(last["tool_calls"]) != 1:
        raise RenderDrop("no_submit", "the final assistant turn is not a single submit command")
    command = json.loads(last["tool_calls"][0]["function"]["arguments"])["command"]
    if not is_submit(command):
        raise RenderDrop("no_submit", f"final command {command[:80]!r}")
    if normalise_submit and command.strip() != MINI_SWE_SUBMIT_COMMAND:
        last["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": MINI_SWE_SUBMIT_COMMAND})
        stats["submit_normalised"] = 1
    stats["repo_root"] = rewrite.root or TEACHER_REPO_ROOT
    stats["repo_root_rewrites"] = rewrite.counts["paths_rewritten"]
    stats["repo_root_counts"] = dict(rewrite.counts)
    model_patch = ((raw.get("metadata") or {}).get("model_patch") or {}).get("patch")
    sample_index = abs(int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16))
    ep = Episode(episode_id=episode_id, task_id=task_id, round=ROUND, policy=POLICY, sample_index=sample_index,
                 scaffold=SCAFFOLD, messages=out, tools=[bundle.bash_tool], exit_reason="submit", steps=stats["n_assistant"],
                 output_tokens=0, wall_sec=0.0, patch=model_patch, request_log=None, created_at="",
                 extra={"trajectory_id": raw.get("trajectory_id"), "rebuild": stats})
    ep.validate()
    return ep, stats


def prompt_texts(ep: Episode) -> list[str]:
    """System and user text of an episode (the only places a reference patch could have been planted upstream)."""
    return [m["content"] for m in ep.messages if m["role"] in ("system", "user")]


def episode_text(messages: list[dict[str, Any]]) -> str:
    """Every rendered string of an episode's messages (prompts, reasoning, content, commands, observations), for the D2
    text screen."""
    parts: list[str] = []
    for m in messages:
        parts.append(m.get("content") or "")
        parts.append(m.get("reasoning_content") or "")
        parts.extend(tc["function"]["arguments"] for tc in m.get("tool_calls") or [])
    return "\n".join(parts)
