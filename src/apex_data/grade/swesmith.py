"""Pristine-image replay grader for SWE-smith tasks (Python and Go), plus the Docker sandbox and mirror helpers the rollout
drivers share (data plan sections 6 and 11: reward definition, grading mechanism, the no-patch rule).

Grading interface (what the rollout and dev-evaluation drivers call; train venv, Docker on the host):

    from apex_data.grade.swesmith import grade_patch, GradeOptions
    verdict = grade_patch(task, patch, run_dir, second_run=True, episode_id=None, exit_reason="submit", opts=None)
        task       a schema.Task (or its dict form) from data/pools/swesmith/tasks.jsonl, calibrated (extra.eval_spec)
        patch      the agent's final `git diff` against the instance branch (any text; test paths are stripped)
        run_dir    directory for the replay logs: run1/ and run2/ with patch.diff, eval.sh, test_output.txt, report.json
        returns    schema.Verdict: r_pass in {0, 1}, runs (one record per container replay), tamper_flags, verifier
                   "swesmith", verifier_version, extra (image digest, calibration summary, eval path, r_partial, ...)
    grade_episode(episode, task, work_dir, opts) is the same for a schema.Episode dict (its patch and exit_reason).

Reward (exact): r_pass = 1 iff, on a pristine container of the task's repository image with the agent's diff restricted to
non-test paths re-applied and the harness-side tests restored, every FAIL_TO_PASS test passes and every graded
PASS_TO_PASS test still passes, in two runs (the second only after a first pass; both must pass; a timeout is a fail).
Tamper flags (grade/tamper.py) force 0; a flagged, empty or calibration-excluded episode gets no second run. Strict
semantics: a listed test that is missing, skipped or errored failed; ids are matched whitespace-insensitively and an id
reported more than once passes only if no occurrence failed.

Replay protocol (= swesmith.harness.utils.run_patch_in_container of swesmith 0.0.9 in evaluation mode, without network):
  1. container `s2swesmith_<instance>_<k>_<rand>` from the task's repository image (2 CPUs / 8 GB, --network none), with
     the local bare mirror of github.com/swesmith/<repo_name> mounted read-only (cloned once per repository under
     RAW/swesmith_mirrors/; the images ship only the clean commit);
  2. the task's instance branch is fetched from the mirror and checked out, then `git checkout HEAD~1` steps back over
     the branch's "Remove F2P Tests" commit so the harness-side tests are present again (the toolkit's step);
  3. the non-test part of the patch is applied with the toolkit's ladder (git apply, git apply --reject, patch -p1 fuzz 5);
     any change that still lands on a test / CI / test-config path is reverted;
  4. the test command runs between the toolkit's output markers with a timeout (max(2 x profile timeout, 300 s)); the
     output is parsed by the structured reporter the calibration validated (Go `go test -json`, pytest JUnit XML) or, when
     the command has no structured form, by the profile's own `log_parser` (run in .venv/swesmith, where swesmith 0.0.9
     and swebench 4.1.0 live).
Calibration (`calibrate_task`, scripts/data/calibrate_tasks.py): the gold fix, i.e. the reverse of the dataset's bug patch,
is replayed exactly like an agent diff, first with the structured reporter, then with the profile's command and parser if
the structured output does not cover every F2P/P2P id. The validated form is persisted as task.extra.eval_spec = {workdir,
test_cmd, reporter, parser, p2p_excluded, image, image_digest, timeout_sec, ...} and every later replay executes it
verbatim. P2P ids the gold fix fails under the sandbox are excluded from grading (SWE-bench's own definition of
PASS_TO_PASS: passing before and after the fix in the harness environment) and counted; a task whose gold fix does not
apply, fails an F2P test or times out is excluded and counted.

CLI (repository root; `grade` and `calibrate` run in the train venv, the toolkit subcommands in .venv/swesmith):
    python -m apex_data.grade.swesmith grade --task task.json (--episode ep.json | --patch fix.diff) --out verdict.json --work-dir DIR
    python -m apex_data.grade.swesmith calibrate --task task.json --out calibration.json --work-dir DIR
    python -m apex_data.grade.swesmith mirror --repo joho__godotenv.3a7a1902
    [toolkit venv] profile --repo R | parse --repo R --log test_output.txt [--expected-file ids.json] | test-cmd --repo R ...
  Shared options: --cpus 2 --memory 8g --network none --mirrors-dir DIR --timeout-scale 2 --run-id ID [--no-second-run].
  Test-id lists always travel through a JSON file, never on the command line (a large P2P list exceeds ARG_MAX).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from apex_data.grade import reporters
from apex_data.grade.tamper import classify_path, scratch_test_files, split_patch, tamper_flags
from apex_data.paths import RAW, REPO
from apex_data.schema import Verdict

SWESMITH_PYTHON = REPO / ".venv" / "swesmith" / "bin" / "python"
SRC_DIR = REPO / "src"
MIRROR_URL = "https://github.com/swesmith/{repo_name}"
DEFAULT_MIRRORS_DIR = RAW / "swesmith_mirrors"
CONTAINER_PREFIX = "s2swesmith_"                       # every container this module starts is named with this prefix
WORKDIR = "/testbed"                                   # swebench.harness.constants.DOCKER_WORKDIR (all SWE-smith images)
MIRROR_IN_CONTAINER = "/apex_mirror.git"
TEST_OUTPUT_START = ">>>>> Start Test Output"          # swesmith.constants
TEST_OUTPUT_END = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"          # swebench.harness.constants
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
GIT_APPLY_CMDS = ("git apply --verbose", "git apply --verbose --reject", "patch --batch --fuzz=5 -p1 -i")   # swesmith.constants
REMOVE_TESTS_COMMIT = "Remove F2P Tests"               # swesmith.harness.gather commit message
GIT_SAFE_DIRS = "git config --global --add safe.directory '*'"   # the mirror keeps the host uid; git refuses "dubious ownership" otherwise
PASS_STATUSES = frozenset({"PASSED", "XFAIL"})          # swesmith.harness.grading.test_passed
STATUS_RANK = {"FAILED": 3, "ERROR": 3, "PASSED": 2, "XFAIL": 2, "SKIPPED": 1}   # merge order for an id seen more than once
VERIFIER_VERSION = "swesmith:4;swesmith==0.0.9;swebench==4.1.0"   # 4: new scratch test files are recorded, not tampering
MIN_TEST_TIMEOUT_SEC = 300
SECRET_ENV = ("GITHUB_TOKEN", "OPENAI_API_KEY", "HF_TOKEN")


class SandboxError(RuntimeError):
    """The container could not be started or driven (environment failure; the caller may retry)."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(s: str, limit: int = 56) -> str:
    """Docker-name-safe version of an id such as 'swesmith:owner__repo.abc#1#base#0'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-.")[:limit] or "x"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --- docker sandbox ------------------------------------------------------------------------------------------------
@dataclass
class ExecResult:
    rc: int | None
    output: str
    timed_out: bool
    seconds: float


@dataclass
class DockerSandbox:
    """One throw-away container driven through the docker CLI (the same mechanism mini-swe-agent uses)."""

    image: str
    name: str
    workdir: str = WORKDIR
    cpus: float = 2.0
    memory: str = "8g"
    cgroup_parent: str | None = None
    network: str = "none"
    labels: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)   # host path -> "container_path[:ro]"
    keepalive_sec: int = 4 * 3600
    container_id: str | None = None

    def start(self) -> str:
        cmd = ["docker", "run", "-d", "--pull", "never", "--name", self.name, "-w", self.workdir, "--cpus", str(self.cpus), "--memory", self.memory,
               "--network", self.network]
        if self.cgroup_parent:
            cmd += ["--cgroup-parent", self.cgroup_parent]
        for k, v in self.labels.items():
            cmd += ["--label", f"{k}={v}"]
        for host, target in self.volumes.items():
            cmd += ["-v", f"{host}:{target}"]
        cmd += [self.image, "sleep", str(self.keepalive_sec)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        except subprocess.TimeoutExpired as e:
            raise SandboxError(f"docker run timed out for {self.image}") from e
        if r.returncode != 0:
            raise SandboxError(f"docker run failed ({r.returncode}): {r.stderr.strip()[-800:]}")
        self.container_id = r.stdout.strip()
        return self.container_id

    def exec(self, command: str, *, timeout: int, cwd: str | None = None, user: str | None = None) -> ExecResult:
        assert self.container_id, "sandbox not started"
        cmd = ["docker", "exec", "-w", cwd or self.workdir]
        if user:
            cmd += ["-u", user]
        cmd += [self.container_id, "bash", "-c", command]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return ExecResult(r.returncode, r.stdout.decode("utf-8", errors="replace"), False, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = e.output.decode("utf-8", errors="replace") if isinstance(e.output, bytes) else (e.output or "")
            return ExecResult(None, out, True, time.time() - t0)

    def cp_in(self, src: Path | str, dst: str) -> None:
        r = subprocess.run(["docker", "cp", "-q", str(src), f"{self.container_id}:{dst}"], capture_output=True, text=True, timeout=600, check=False)
        if r.returncode != 0:
            raise SandboxError(f"docker cp into container failed: {r.stderr.strip()[-400:]}")

    def cp_out(self, src: str, dst: Path | str) -> bool:
        r = subprocess.run(["docker", "cp", "-q", f"{self.container_id}:{src}", str(dst)], capture_output=True, text=True, timeout=600, check=False)
        return r.returncode == 0

    def write_file(self, path: str, text: str) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=Path(path).suffix) as f:
            f.write(text)
            tmp = f.name
        try:
            self.cp_in(tmp, path)
        finally:
            os.unlink(tmp)

    def remove(self) -> None:
        """Remove the container; an overloaded daemon that does not finish within the timeout gets a detached `docker rm -f`
        and never costs the verdict (seen 2026-10-11: `docker rm -f` past 120 s crashed graders in cleanup)."""
        if self.container_id:
            try:
                subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True, timeout=120, check=False)
            except subprocess.TimeoutExpired:
                subprocess.Popen(["docker", "rm", "-f", self.container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            self.container_id = None

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()


def image_workdir(image: str) -> str | None:
    """The image's configured WorkingDir (`docker inspect .Config.WorkingDir`); None when absent or '/'."""
    r = subprocess.run(["docker", "inspect", "--format", "{{.Config.WorkingDir}}", image], capture_output=True, text=True, timeout=60, check=False)
    wd = (r.stdout or "").strip()
    return wd if r.returncode == 0 and wd and wd != "/" else None


def image_digest(image: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image], capture_output=True, text=True, timeout=60, check=False)
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def image_present(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60, check=False).returncode == 0


# --- SWE-smith repository mirrors ----------------------------------------------------------------------------------
def repo_name_of(task: dict[str, Any]) -> str:
    """'owner__repo.commit8' from a SWE-smith task (the toolkit's profile key; instance ids are '<repo_name>.<bug id>')."""
    swesmith_repo = (task.get("extra") or {}).get("swesmith_repo")
    if swesmith_repo:
        return str(swesmith_repo).rsplit("/", 1)[-1]
    iid = str(task["upstream_id"])
    head, _, _ = iid.rpartition("__") if iid.count("__") > 1 else (iid, "", "")
    return head.rsplit(".", 1)[0]


def mirror_path(repo_name: str, mirrors_dir: Path | str = DEFAULT_MIRRORS_DIR) -> Path:
    return Path(mirrors_dir) / f"{repo_name}.git"


def ensure_mirror(repo_name: str, mirrors_dir: Path | str = DEFAULT_MIRRORS_DIR, *, timeout_sec: int = 1800) -> Path:
    """Local bare mirror of github.com/swesmith/<repo_name> (all instance branches), cloned once under a file lock."""
    dest = mirror_path(repo_name, mirrors_dir)
    if (dest / "HEAD").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = dest.parent / f".{repo_name}.lock"
    with lock.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if (dest / "HEAD").exists():
                return dest
            tmp = dest.parent / f".{repo_name}.{uuid.uuid4().hex[:6]}.partial"
            env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
            env["GIT_TERMINAL_PROMPT"] = "0"
            try:
                r = subprocess.run(["git", "clone", "--quiet", "--mirror", MIRROR_URL.format(repo_name=repo_name), str(tmp)],
                                   capture_output=True, text=True, timeout=timeout_sec, check=False, env=env)
            except subprocess.TimeoutExpired as e:
                raise SandboxError(f"git clone --mirror timed out for {repo_name}") from e
            if r.returncode != 0:
                raise SandboxError(f"git clone --mirror failed for {repo_name}: {r.stderr.strip()[-400:]}")
            tmp.rename(dest)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return dest


def checkout_instance(sb: DockerSandbox, instance_id: str, *, restore_tests: bool, mirror: str = MIRROR_IN_CONTAINER) -> dict[str, Any]:
    """Fetch the instance branch from the mirror visible at `mirror` inside the container and check it out; with
    restore_tests, step back over the 'Remove F2P Tests' commit (the toolkit's `git checkout HEAD~1`)."""
    ref = shlex.quote(instance_id)
    cmd = (f"{GIT_SAFE_DIRS} && git fetch -q {shlex.quote(mirror)} +refs/heads/{ref}:refs/heads/{ref} && "
           f"git checkout -q -f {ref} && git log -1 --format=%s")
    r = sb.exec(cmd, timeout=900)
    if r.rc != 0:
        raise SandboxError(f"instance checkout failed for {instance_id}: {r.output[-600:]}")
    head_subject = r.output.strip().splitlines()[-1] if r.output.strip() else ""
    info = {"head_subject": head_subject, "restored_tests": False}
    if restore_tests and head_subject == REMOVE_TESTS_COMMIT:
        r2 = sb.exec("git checkout -q HEAD~1", timeout=120)
        if r2.rc != 0:
            raise SandboxError(f"checkout HEAD~1 failed: {r2.output[-400:]}")
        info["restored_tests"] = True
    return info


def test_removal_diff(task: dict[str, Any], mirrors_dir: Path | str = DEFAULT_MIRRORS_DIR) -> str:
    """The first file section of the instance branch's "Remove F2P Tests" commit (the diff that deletes one harness-side
    test file), read from the host mirror: added to the gold fix it makes the smoke's tampered patch ("gold with one test
    file reverted"). Empty when the branch has no such commit."""
    mirror = ensure_mirror(repo_name_of(task), mirrors_dir)
    ref = f"refs/heads/{task['upstream_id']}"
    r = subprocess.run(["git", "-C", str(mirror), "log", "-1", "--format=%s", ref], capture_output=True, text=True, timeout=120, check=False)
    if r.returncode != 0 or r.stdout.strip() != REMOVE_TESTS_COMMIT:
        return ""
    files = subprocess.run(["git", "-C", str(mirror), "diff", "--name-only", f"{ref}~1", ref], capture_output=True, text=True, timeout=120, check=False)
    names = [n for n in files.stdout.splitlines() if n.strip()]
    if not names:
        return ""
    d = subprocess.run(["git", "-C", str(mirror), "diff", "--no-color", f"{ref}~1", ref, "--", names[0]], capture_output=True, text=True, timeout=120, check=False)
    return d.stdout if d.returncode == 0 else ""


# --- toolkit bridge (runs in .venv/swesmith) --------------------------------------------------------------------------
_PROFILE_CACHE: dict[str, dict[str, Any]] = {}


@contextlib.contextmanager
def json_argfile(obj: Any) -> Iterator[str]:
    """Path of a temporary JSON file holding `obj`, for arguments too large for the command line (test-id lists)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        path = f.name
    try:
        yield path
    finally:
        os.unlink(path)


def read_json_arg(path: str, key: str) -> list[str]:
    """`key` of a json_argfile payload (a bare list is accepted too)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(obj if isinstance(obj, list) else obj.get(key) or [])


def _toolkit(args: list[str], timeout: int = 600) -> dict[str, Any]:
    """Run a toolkit subcommand of this module in .venv/swesmith and return the JSON object it prints last."""
    if not SWESMITH_PYTHON.exists():
        raise SandboxError(f"swesmith toolkit venv missing at {SWESMITH_PYTHON}")
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env["PYTHONPATH"] = str(SRC_DIR)
    with tempfile.TemporaryDirectory() as cwd:   # not the repository root: the toolkit's load_dotenv() would read .env
        try:
            r = subprocess.run([str(SWESMITH_PYTHON), "-m", "apex_data.grade.swesmith", *args], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise SandboxError(f"toolkit call {args[:2]} timed out") from e
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        raise SandboxError(f"toolkit call {args[:2]} failed ({r.returncode}): {(r.stderr or r.stdout)[-800:]}")
    return json.loads(lines[-1])


def profile_info(repo_name: str) -> dict[str, Any]:
    """Image, test command and timeout of a SWE-smith repository profile (cached per process)."""
    if repo_name not in _PROFILE_CACHE:
        _PROFILE_CACHE[repo_name] = _toolkit(["profile", "--repo", repo_name])
    return _PROFILE_CACHE[repo_name]


def parse_test_log(repo_name: str, log_path: Path | str, expected: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """{found, status_map, n_parsed, parser_used, coverage, error} of a test log parsed in the toolkit venv by the profile's
    own `log_parser`; ids travel through a file."""
    with json_argfile({"expected": list(expected)}) as ids:
        return _toolkit(["parse", "--repo", repo_name, "--log", str(log_path), "--expected-file", ids])


def test_command_for(task: dict[str, Any], profile: dict[str, Any]) -> str:
    """The profile's test command; `min_testing` profiles narrow it to the F2P/P2P test files exactly as the toolkit does
    (`get_test_cmd`)."""
    if profile.get("min_testing"):
        f2p, p2p = test_lists(task)
        with json_argfile({"f2p": f2p, "p2p": p2p}) as lists:
            res = _toolkit(["test-cmd", "--repo", profile["repo_name"], "--instance", str(task["upstream_id"]), "--lists", lists])
        return str(res["test_cmd"])
    return str(profile["test_cmd"])


def _tk_profile(repo_name: str) -> dict[str, Any]:
    from swesmith.profiles import registry  # noqa: PLC0415 - toolkit venv only

    rp = registry.get(repo_name)
    return {"repo_name": rp.repo_name, "mirror_name": rp.mirror_name, "image_name": rp.image_name, "owner": rp.owner, "repo": rp.repo,
            "commit": rp.commit, "test_cmd": rp.test_cmd, "timeout": rp.timeout, "min_testing": rp.min_testing,
            "profile_class": type(rp).__name__, "language_profile": type(rp).__mro__[1].__name__}


def _tk_parse(repo_name: str, log_path: str, expected: list[str]) -> dict[str, Any]:
    from swesmith.profiles import registry  # noqa: PLC0415 - toolkit venv only

    rp = registry.get(repo_name)
    output, found = extract_test_output(Path(log_path).read_text(errors="replace"))
    if not found or output is None:
        return {"found": False, "status_map": {}, "n_parsed": 0, "parser_used": None, "coverage": 0.0, "error": None}
    res = run_parser(f"profile:{type(rp).__name__}", rp.log_parser, output, expected)
    return {**res, "n_parsed": len(res["status_map"])}


def _tk_test_cmd(repo_name: str, instance_id: str, f2p: list[str], p2p: list[str]) -> dict[str, Any]:
    from swesmith.profiles import registry  # noqa: PLC0415 - toolkit venv only

    rp = registry.get(repo_name)
    try:
        test_cmd, files = rp.get_test_cmd({"instance_id": instance_id, "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p})
        return {"test_cmd": test_cmd, "files": [str(f) for f in files], "narrowed": True}
    except Exception as e:  # noqa: BLE001 - a profile that needs a clone to narrow falls back to the full suite
        return {"test_cmd": rp.test_cmd, "files": [], "narrowed": False, "error": f"{type(e).__name__}: {e}"}


# --- pure helpers (importable everywhere; unit-tested) ------------------------------------------------------------------
def extract_test_output(content: str) -> tuple[str | None, bool]:
    """swesmith.harness.grading.read_test_output: (text between the markers, found); None on apply failure / timeout."""
    if APPLY_PATCH_FAIL in content or TESTS_TIMEOUT in content:
        return None, False
    if TEST_OUTPUT_START not in content or TEST_OUTPUT_END not in content:
        return content, False
    start_sep, end_sep = f"+ : '{TEST_OUTPUT_START}'", f"+ : '{TEST_OUTPUT_END}'"
    start, end = content.find(start_sep), content.find(end_sep)
    if start < 0 or end < 0 or start > end:
        return content, False
    return content[start + len(start_sep):end], True


def cpu_limit(cpus: float) -> int:
    """Whole CPUs a runtime should size itself to under a `--cpus` quota (at least one)."""
    return max(1, int(-(-float(cpus) // 1)))


def memory_bytes(spec: str) -> int:
    """Bytes of a docker memory limit such as '8g', '512m', '4096k' or a plain byte count."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*", str(spec).lower())
    if not m:
        raise ValueError(f"not a docker memory limit: {spec!r}")
    return int(float(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}[m.group(2)])


def eval_script(test_cmd: str, workdir: str = WORKDIR, cpus: int | None = None, mem_bytes: int | None = None) -> str:
    """The toolkit's eval.sh: `set -uxo pipefail` traces the marker commands, which is what the extractor keys on. With
    `cpus`, one runtime-sizing line tells the runners that size themselves from the host's CPU count (192 here, also
    inside a `--cpus 2` container) the sandbox's quota: pytest-xdist's `-n auto` (PYTEST_XDIST_AUTO_NUM_WORKERS; hips/autograd
    sets `-n auto` in its pytest addopts and 192 workers never finish inside 8 GB), Go's GOMAXPROCS (the images' Go 1.24
    is not cgroup-aware, so `go test` would build and run 192 packages at once; Go 1.25 sets this value itself) and
    modin's MODIN_CPUS / MODIN_MEMORY (a quarter of `mem_bytes`; modin otherwise sizes Ray from 192 CPUs and 755 GB)."""
    lines = ["#!/bin/bash", "set -uxo pipefail", f"cd {workdir}"]
    if cpus:
        n = int(cpus)
        lines.append(f"export APEX_CPUS={n} PYTEST_XDIST_AUTO_NUM_WORKERS={n} GOMAXPROCS={n} MODIN_CPUS={n}"
                     + (f" MODIN_MEMORY={int(mem_bytes) // 4}" if mem_bytes else ""))
    return "\n".join([*lines, f": '{TEST_OUTPUT_START}'", test_cmd, f": '{TEST_OUTPUT_END}'"]) + "\n"


def normalise_test_id(test_id: str) -> str:
    """Whitespace-insensitive form of a test id (reporters pad names; the datasets' lists do not)."""
    return " ".join(str(test_id).split())


def merge_status(old: str | None, new: str) -> str:
    """Status of an id reported more than once: any failure fails it, otherwise any pass passes it, otherwise skipped."""
    if old is None:
        return new
    return new if STATUS_RANK.get(new, 0) > STATUS_RANK.get(old, 0) else old


def normalise_status_map(status_map: dict[str, str]) -> dict[str, str]:
    """Status map keyed by normalised id with duplicates merged by `merge_status`."""
    out: dict[str, str] = {}
    for test_id, status in status_map.items():
        key = normalise_test_id(test_id)
        out[key] = merge_status(out.get(key), status)
    return out


def assemble_report(status_map: dict[str, str], f2p: list[str], p2p: list[str]) -> dict[str, Any]:
    """F2P / P2P outcome from a parsed status map. Strict: a listed test that is missing, skipped or errored failed."""
    sm = normalise_status_map(status_map)

    def bucket(ids: list[str]) -> dict[str, Any]:
        status = {t: sm.get(normalise_test_id(t)) for t in ids}
        ok = [t for t in ids if status[t] in PASS_STATUSES]
        bad = [t for t in ids if status[t] not in PASS_STATUSES]
        return {"n": len(ids), "success": len(ok), "failure": len(bad), "failed_ids": bad[:50],
                "missing": sum(1 for t in bad if status[t] is None), "skipped": sum(1 for t in bad if status[t] == "SKIPPED")}
    f, p = bucket(list(f2p)), bucket(list(p2p))
    return {"f2p": f, "p2p": p, "resolved": bool(f2p) and f["failure"] == 0 and p["failure"] == 0, "n_parsed": len(status_map)}


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def coverage_of(status_map: dict[str, str], expected: list[str]) -> float:
    """Share of the expected ids present in the status map (whitespace-normalised); with no expected ids, 1.0 for any parse."""
    wanted = {normalise_test_id(t) for t in expected}
    return round((len(wanted & set(normalise_status_map(status_map))) / len(wanted)) if wanted else float(len(status_map) > 0), 4)


def run_parser(name: str, func: Callable[[str], dict[str, str]], output: str, expected: list[str]) -> dict[str, Any]:
    """One log parser on the ANSI-stripped output; a parser that raises yields an empty map and the error text."""
    try:
        status_map = func(strip_ansi(output))
    except Exception as e:  # noqa: BLE001 - recorded; the replay then counts every listed test as failed
        return {"found": True, "status_map": {}, "parser_used": name, "coverage": 0.0, "error": f"{type(e).__name__}: {e}"[:200]}
    return {"found": True, "status_map": status_map, "parser_used": name, "coverage": coverage_of(status_map, expected), "error": None}


_INDEX = re.compile(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)(.*)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_DIFF_GIT = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def reverse_patch(patch: str) -> str:
    """The inverse of a git unified diff (SWE-smith's `patch` introduces the bug; its inverse is the gold fix): per file the
    `---`/`+++` headers swap sides (so additions become deletions and back), index hashes, hunk ranges, mode lines,
    renames and +/- lines swap. Reversing twice gives the original text back."""
    out: list[str] = []
    lines = (patch or "").splitlines()
    i = 0
    in_hunk = False
    while i < len(lines):
        line = lines[i]
        m = _DIFF_GIT.match(line)
        if m:
            out.append(f"diff --git a/{m.group(2)} b/{m.group(1)}")
            in_hunk = False
        elif (line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")
              and (not in_hunk or (line.startswith(("--- a/", "--- /dev/null")) and lines[i + 1].startswith(("+++ b/", "+++ /dev/null"))))):
            old, new = line[4:], lines[i + 1][4:]
            old_path = None if old.split("\t")[0] == "/dev/null" else old.split("\t")[0][2:] if old.startswith("a/") else old.split("\t")[0]
            new_path = None if new.split("\t")[0] == "/dev/null" else new.split("\t")[0][2:] if new.startswith("b/") else new.split("\t")[0]
            out.append("--- " + (f"a/{new_path}" if new_path else "/dev/null"))
            out.append("+++ " + (f"b/{old_path}" if old_path else "/dev/null"))
            i += 2
            in_hunk = False
            continue
        elif _INDEX.match(line) and not in_hunk:
            mi = _INDEX.match(line)
            out.append(f"index {mi.group(2)}..{mi.group(1)}{mi.group(3)}")
        elif line.startswith("@@"):
            mh = _HUNK.match(line)
            if mh:
                a, al, b, bl, rest = mh.groups()
                out.append(f"@@ -{b}{',' + bl if bl is not None else ''} +{a}{',' + al if al is not None else ''} @@{rest}")
            else:
                out.append(line)
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            out.append("-" + line[1:])
        elif in_hunk and line.startswith("-"):
            out.append("+" + line[1:])
        elif line.startswith("new file mode") and not in_hunk:
            out.append("deleted file mode" + line[len("new file mode"):])
        elif line.startswith("deleted file mode") and not in_hunk:
            out.append("new file mode" + line[len("deleted file mode"):])
        elif line.startswith("old mode ") and not in_hunk:
            out.append("new mode " + line[len("old mode "):])
        elif line.startswith("new mode ") and not in_hunk:
            out.append("old mode " + line[len("new mode "):])
        elif line.startswith("rename from ") and not in_hunk:
            out.append("rename to " + line[len("rename from "):])
        elif line.startswith("rename to ") and not in_hunk:
            out.append("rename from " + line[len("rename to "):])
        else:
            out.append(line)
        i += 1
    if any(o.startswith("rename to ") for o in out):      # git wants "rename from" before "rename to"
        for j in range(len(out) - 1):
            if out[j].startswith("rename to ") and out[j + 1].startswith("rename from "):
                out[j], out[j + 1] = out[j + 1], out[j]
    return "\n".join(out) + ("\n" if (patch or "").endswith("\n") else "")


def bug_patch_of(task: dict[str, Any]) -> str:
    """The dataset's bug-introducing patch stored in the pool record, checked against its sha256."""
    extra = task.get("extra") or {}
    bug = extra.get("bug_patch")
    if not isinstance(bug, str) or not bug.strip():
        raise ValueError(f"{task.get('task_id')}: no bug patch in extra.bug_patch")
    want = extra.get("bug_patch_sha256")
    if want and sha256_text(bug) != want:
        raise ValueError(f"{task.get('task_id')}: extra.bug_patch does not match its sha256")
    return bug


def gold_patch(task: dict[str, Any]) -> str:
    """The gold fix of a SWE-smith task: the reverse of its bug patch (checked against task.gold_patch_sha when set)."""
    gold = reverse_patch(bug_patch_of(task))
    if task.get("gold_patch_sha") and sha256_text(gold) != task["gold_patch_sha"]:
        raise ValueError(f"{task.get('task_id')}: reversed bug patch does not match gold_patch_sha")
    return gold


# --- calibration (task.extra.calibration and task.extra.eval_spec, written by scripts/data/calibrate_tasks.py) -------------
def test_lists(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(F2P, P2P) id lists; a non-list value counts as no list."""
    tests = task.get("tests") or {}
    return (list(tests["f2p"]) if isinstance(tests.get("f2p"), list) else [], list(tests["p2p"]) if isinstance(tests.get("p2p"), list) else [])


def calibration_of(task: dict[str, Any]) -> dict[str, Any]:
    cal = (task.get("extra") or {}).get("calibration")
    return cal if isinstance(cal, dict) else {}


def eval_spec_of(task: dict[str, Any]) -> dict[str, Any] | None:
    """task.extra.eval_spec validated by calibration (None when absent): executed verbatim by the replays."""
    spec = (task.get("extra") or {}).get("eval_spec")
    return spec if isinstance(spec, dict) and spec.get("test_cmd") else None


def graded_lists(task: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """(F2P, P2P graded, P2P excluded): the task's lists minus the eval spec's (else the calibration's) `p2p_excluded`."""
    f2p, p2p = test_lists(task)
    spec = eval_spec_of(task)
    excluded = set((spec or {}).get("p2p_excluded") or calibration_of(task).get("p2p_excluded") or [])
    return f2p, [t for t in p2p if t not in excluded], [t for t in p2p if t in excluded]


def calibration_summary(task: dict[str, Any]) -> dict[str, Any] | None:
    """What a verdict records about the task's calibration (None when the task was never calibrated)."""
    cal = calibration_of(task)
    if not cal:
        return None
    return {"run_id": cal.get("run_id"), "status": cal.get("status"), "gold_pass": cal.get("gold_pass"), "n_p2p_excluded": len(graded_lists(task)[2])}


def failed_buckets(f2p: list[str], p2p: list[str], n_excluded: int) -> dict[str, Any]:
    return {"f2p": {"n": len(f2p), "success": 0, "failure": len(f2p)}, "p2p": {"n": len(p2p), "success": 0, "failure": len(p2p), "excluded": n_excluded}}


def parse_run(reporter: str, log_text: str, structured_text: str | None, expected: list[str],
              text_parse: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Status map of one replay: the structured reporter's output when the run used one (json:go from the log, junit:pytest
    from the copied-out file), else the profile's log parser (`text_parse`)."""
    if reporter == "text":
        return text_parse()
    output, found = extract_test_output(log_text)
    if not found or output is None:
        return {"found": False, "status_map": {}, "parser_used": None, "coverage": 0.0, "error": None}
    status_map = reporters.parse_structured(reporter, strip_ansi(output), structured_text)
    if status_map is None:
        return {"found": True, "status_map": {}, "parser_used": reporter, "coverage": 0.0, "error": "no structured output"}
    return {"found": True, "status_map": status_map, "parser_used": reporter, "coverage": coverage_of(status_map, expected), "error": None}


def build_eval_spec(run: dict[str, Any], record: dict[str, Any], image: str) -> dict[str, Any]:
    """The validated evaluation spec persisted as task.extra.eval_spec (only for a calibration with status ok)."""
    return {"workdir": run.get("workdir"), "test_cmd": run.get("test_cmd"), "reporter": run.get("reporter") or "text", "parser": run.get("parser"),
            "p2p_excluded": list(record.get("p2p_excluded") or []), "image": image, "image_digest": image_digest(image),
            "timeout_sec": run.get("test_timeout_sec"), "cpus": run.get("cpus"), "calibrated_at": record.get("calibrated_at"),
            "run_id": record.get("run_id"), "verifier_version": record.get("verifier_version")}


def calibration_base(task: dict[str, Any], run_id: str | None, network: str) -> dict[str, Any]:
    f2p, p2p = test_lists(task)
    return {"task_id": task["task_id"], "run_id": run_id, "verifier_version": VERIFIER_VERSION, "network": network, "n_f2p": len(f2p), "n_p2p": len(p2p),
            "gold_pass": False, "gold_applied": False, "p2p_excluded": [], "f2p_missing": list(f2p), "parser": None, "reporter": None,
            "coverage": 0.0, "seconds": 0.0, "timed_out": False, "calibrated_at": utc_now()}


def finish_calibration(base: dict[str, Any], run: dict[str, Any], status_map: dict[str, str], f2p: list[str], p2p: list[str]) -> dict[str, Any]:
    """The calibration record from one gold replay (`run` = a replay_once record, `status_map` its parse):
      status ok           every F2P test passes; `p2p_excluded` = the P2P ids the gold fix fails under our sandbox
             excluded     the gold fix does not apply / fails an F2P test / times out; `reason` = "calibration:<why>"
             env_failure  the sandbox failed (retryable); nothing is known
      gold_pass           every F2P and every P2P test passed (SWE-bench's own validation criterion)."""
    sm = normalise_status_map(status_map)
    rec = {**base, "gold_pass": bool(run.get("pass")), "gold_applied": bool((run.get("apply") or {}).get("ok")),
           "f2p_missing": [t for t in f2p if sm.get(normalise_test_id(t)) not in PASS_STATUSES],
           "p2p_excluded": [t for t in p2p if sm.get(normalise_test_id(t)) not in PASS_STATUSES],
           "parser": run.get("parser"), "reporter": run.get("reporter"), "coverage": run.get("parser_coverage") or 0.0,
           "seconds": run.get("seconds"), "timed_out": bool(run.get("timed_out")), "error": run.get("error"), "log": run.get("log"),
           "test_cmd": run.get("test_cmd"), "workdir": run.get("workdir"), "n_parsed": run.get("n_parsed", 0)}
    if run.get("error") and run.get("retryable"):
        return {**rec, "status": "env_failure", "reason": f"sandbox: {run['error']}"[:400]}
    if run.get("error"):
        return {**rec, "status": "excluded", "reason": f"calibration:{run['error']}"[:400]}
    if not rec["gold_applied"]:
        return {**rec, "status": "excluded", "reason": "calibration:gold_patch_apply_failed"}
    if rec["timed_out"]:
        return {**rec, "status": "excluded", "reason": f"calibration:gold_run_timeout_{run.get('test_timeout_sec')}s"}
    if rec["f2p_missing"]:
        why = "parser_no_match" if rec["n_parsed"] == 0 else f"gold_f2p_failed_{len(rec['f2p_missing'])}_of_{len(f2p)}"
        return {**rec, "status": "excluded", "reason": f"calibration:{why}"}
    return {**rec, "status": "ok", "reason": None}


def calibrate_with(replay: Callable[[Path, int, bool], dict[str, Any]], base: dict[str, Any], f2p: list[str], p2p: list[str],
                   work_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gold replays for one task: first with the structured reporter when the runner is detectable (`replay(dir, k, True)`),
    then with the profile's command and parser when that output does not cover every id; the better-covered run (ties:
    structured) decides. Returns (calibration record with `attempts`, the chosen run)."""
    first = replay(work_dir / "gold", 1, True)
    chosen, chosen_map = first, first.pop("status_map", None) or {}
    attempts = [{"reporter": first.get("reporter", "text"), "coverage": first.get("parser_coverage"), "pass": first.get("pass"), "seconds": first.get("seconds")}]
    if (first.get("reporter", "text") != "text" and (first.get("parser_coverage") or 0.0) < 1.0 and not first.get("error")
            and (first.get("apply") or {}).get("ok", True) and not first.get("timed_out")):
        second = replay(work_dir / "gold_text", 2, False)
        second_map = second.pop("status_map", None) or {}
        attempts.append({"reporter": "text", "coverage": second.get("parser_coverage"), "pass": second.get("pass"), "seconds": second.get("seconds")})
        if (second.get("parser_coverage") or 0.0) > (first.get("parser_coverage") or 0.0):
            chosen, chosen_map = second, second_map
    record = finish_calibration(base, chosen, chosen_map, f2p, p2p)
    record.update(attempts=attempts, seconds=round(sum(float(a.get("seconds") or 0.0) for a in attempts), 1))
    return record, chosen


def verdict_from_runs(episode: dict[str, Any], flags: list[str], runs: list[dict[str, Any]], *, extra: dict[str, Any],
                      second_run_required: bool = True) -> Verdict:
    """Both runs must pass (one when the second run is disabled for debugging); `extra.r_partial` = the first run's F2P
    pass fraction when it is clean (no flag, no P2P regression, no timeout), else None."""
    passes = [bool(r.get("pass")) for r in runs]
    enough = len(runs) >= (2 if second_run_required else 1)
    r_pass = int(not flags and enough and all(passes))
    r_partial = None
    if runs and not flags:
        first = runs[0]
        if first.get("p2p", {}).get("failure", 0) == 0 and first.get("f2p", {}).get("n", 0) > 0 and not first.get("timed_out"):
            r_partial = round(first["f2p"]["success"] / first["f2p"]["n"], 4)
    v = Verdict(episode_id=episode["episode_id"], task_id=episode["task_id"], r_pass=r_pass, runs=runs, tamper_flags=flags,
                verifier="swesmith", verifier_version=VERIFIER_VERSION, graded_at=utc_now(),
                extra={**extra, "r_partial": r_partial, "second_run_required": second_run_required})
    v.validate()
    return v


# --- container steps ------------------------------------------------------------------------------------------------
def apply_patch(sb: DockerSandbox, patch: str, workdir: str, name: str = "apex_patch.diff") -> dict[str, Any]:
    """The toolkit's apply ladder; returns {ok, cmd, rc, output} (no reset between attempts, as in the toolkit)."""
    path = f"/tmp/{name}"
    sb.write_file(path, patch if patch.endswith("\n") else patch + "\n")
    last: dict[str, Any] = {"ok": False, "cmd": None, "rc": None, "output": ""}
    for cmd in GIT_APPLY_CMDS:
        r = sb.exec(f"{cmd} {path}", timeout=300, cwd=workdir)
        last = {"ok": r.rc == 0, "cmd": cmd, "rc": r.rc, "output": r.output[-1500:]}
        if r.rc == 0:
            break
    return last


def revert_test_changes(sb: DockerSandbox, workdir: str) -> list[str]:
    """Undo any change on test / CI / test-config paths left in the working tree (tracked: checkout; untracked: remove)."""
    r = sb.exec("git status --porcelain --untracked-files=all", timeout=120, cwd=workdir)
    reverted: list[str] = []
    for line in r.output.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        if not classify_path(path):
            continue
        if status.strip() == "??":
            sb.exec(f"rm -rf -- {shlex.quote(path)}", timeout=60, cwd=workdir)
        else:
            sb.exec(f"git checkout -q -- {shlex.quote(path)} || git rm -q --cached -- {shlex.quote(path)}", timeout=60, cwd=workdir)
        reverted.append(path)
    return reverted


def run_tests(sb: DockerSandbox, test_cmd: str, workdir: str, timeout_sec: int, out_file: Path, *,
              cpus: int | None = None, mem_bytes: int | None = None, structured_output: str | None = None) -> dict[str, Any]:
    """Write and run eval.sh; the output file carries the toolkit's timeout marker when the suite ran out of time. A
    structured reporter's file (`structured_output`, container path) is copied next to the log (None when not written)."""
    if structured_output:
        sb.exec(f"mkdir -p {shlex.quote(str(Path(structured_output).parent))}", timeout=60, cwd="/")
    script = eval_script(test_cmd, workdir, cpus, mem_bytes)
    sb.write_file("/eval.sh", script)
    (out_file.parent / "eval.sh").write_text(script)
    r = sb.exec("/bin/bash /eval.sh", timeout=timeout_sec, cwd=workdir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    text = r.output + (f"\n\n{TESTS_TIMEOUT}: {timeout_sec} seconds exceeded" if r.timed_out else "")
    out_file.write_text(text, errors="replace")
    copied = None
    if structured_output:
        dst = out_file.parent / Path(structured_output).name
        copied = str(dst) if sb.cp_out(structured_output, dst) and dst.exists() else None
    return {"rc": r.rc, "timed_out": r.timed_out, "test_seconds": round(r.seconds, 1), "log": str(out_file), "test_cmd": test_cmd,
            "test_timeout_sec": timeout_sec, "cpus": cpus, "structured_output": copied}


# --- the grader -------------------------------------------------------------------------------------------------------
@dataclass
class GradeOptions:
    cpus: float = 2.0
    memory: str = "8g"
    cgroup_parent: str | None = None
    network: str = "none"
    mirrors_dir: Path = DEFAULT_MIRRORS_DIR
    second_run: bool = True
    timeout_scale: float = 2.0            # test timeout = max(scale x profile timeout, MIN_TEST_TIMEOUT_SEC)
    labels: dict[str, str] = field(default_factory=dict)


def as_task_dict(task: Any) -> dict[str, Any]:
    return dataclasses.asdict(task) if dataclasses.is_dataclass(task) else dict(task)


def replay_once(task: dict[str, Any], patch: str, run_dir: Path, profile: dict[str, Any], opts: GradeOptions, k: int, *,
                keep_status_map: bool = False, prefer_structured: bool = False) -> dict[str, Any]:
    """One pristine replay; returns a `runs[]` record. With task.extra.eval_spec the spec's command, reporter and parser
    are executed verbatim (eval_path "eval_spec"); otherwise the profile's command (plus the structured reporter when
    `prefer_structured`, calibration's first attempt) and the profile's parser ("derived")."""
    run_dir.mkdir(parents=True, exist_ok=True)
    instance_id = str(task["upstream_id"])
    repo_name = repo_name_of(task)
    image = task.get("image") or profile["image_name"]
    spec = eval_spec_of(task)
    f2p, p2p, excluded = graded_lists(task)
    timeout = int((spec or {}).get("timeout_sec") or max(int(opts.timeout_scale * int(profile["timeout"])), MIN_TEST_TIMEOUT_SEC))
    cpus = cpu_limit(opts.cpus)
    t0 = time.time()
    rec: dict[str, Any] = {"pass": False, "run": k, "image": image, "eval_path": "eval_spec" if spec else "derived", "reporter": "text",
                           "workdir": WORKDIR}
    try:
        mirror = ensure_mirror(repo_name, opts.mirrors_dir)
    except SandboxError as e:
        rec.update(error=str(e), timed_out=False, **failed_buckets(f2p, p2p, len(excluded)), seconds=round(time.time() - t0, 1), retryable=True)
        return rec
    sb = DockerSandbox(image=image, name=f"{CONTAINER_PREFIX}{slug(instance_id, 80)}_{k}_{uuid.uuid4().hex[:6]}", workdir=WORKDIR, cpus=opts.cpus,
                       memory=opts.memory, cgroup_parent=opts.cgroup_parent, network=opts.network,
                       labels={"apex.role": "grade", "apex.unit": "s2swesmith", "apex.task": slug(task["task_id"], 120), **opts.labels},
                       volumes={str(mirror): f"{MIRROR_IN_CONTAINER}:ro"})
    rec["container"] = sb.name
    try:
        sb.start()
        rec["checkout"] = checkout_instance(sb, instance_id, restore_tests=True)
        (run_dir / "patch.diff").write_text(patch)
        # an empty patch (the smoke's "no fix" replay) runs the tests on the bug state; grade_episode never replays one
        rec["apply"] = apply_patch(sb, patch, WORKDIR) if patch.strip() else {"ok": True, "cmd": None, "rc": None, "output": "empty patch: nothing applied"}
        rec["reverted_test_paths"] = revert_test_changes(sb, WORKDIR)
        rec["prep_seconds"] = round(time.time() - t0, 1)
        if not rec["apply"]["ok"]:
            (run_dir / "test_output.txt").write_text(f"{APPLY_PATCH_FAIL}:\n{rec['apply']['output']}\n")
            rec.update(log=str(run_dir / "test_output.txt"), timed_out=False, **failed_buckets(f2p, p2p, len(excluded)), seconds=round(time.time() - t0, 1))
            return rec
        output_file: str | None = None
        if spec:
            test_cmd, rec["reporter"] = str(spec["test_cmd"]), str(spec.get("reporter") or "text")
            output_file = reporters.OUTPUT_FILES.get(rec["reporter"])
        else:
            test_cmd = test_command_for(task, profile)
            structured = reporters.structured_command(test_cmd) if prefer_structured else None
            if structured:
                test_cmd, rec["reporter"], output_file = structured.test_cmd, structured.name, structured.output_file
        rec.update(run_tests(sb, test_cmd, WORKDIR, timeout, run_dir / "test_output.txt", cpus=cpus, mem_bytes=memory_bytes(opts.memory),
                             structured_output=output_file))
    except SandboxError as e:
        rec.update(error=str(e), timed_out=False, **failed_buckets(f2p, p2p, len(excluded)), seconds=round(time.time() - t0, 1), retryable=True)
        return rec
    finally:
        sb.remove()
    structured_text = Path(rec["structured_output"]).read_text(errors="replace") if rec.get("structured_output") else None
    try:
        parsed = parse_run(rec["reporter"], Path(rec["log"]).read_text(errors="replace"), structured_text, f2p + p2p,
                           lambda: parse_test_log(repo_name, rec["log"], f2p + p2p))
    except SandboxError as e:          # the toolkit parser process failed: retryable, never a verdict on the patch
        rec.update(error=str(e), **failed_buckets(f2p, p2p, len(excluded)), seconds=round(time.time() - t0, 1), retryable=True)
        return rec
    report = assemble_report(parsed["status_map"], f2p, p2p)
    report["p2p"]["excluded"] = len(excluded)
    rec.update(f2p=report["f2p"], p2p=report["p2p"], n_parsed=report["n_parsed"], log_found=parsed["found"], parser=parsed.get("parser_used"),
               parser_coverage=parsed.get("coverage"), parser_error=parsed.get("error"), seconds=round(time.time() - t0, 1),
               **({"pass": True} if report["resolved"] and not rec["timed_out"] else {}))
    if keep_status_map:
        rec["status_map"] = parsed["status_map"]
    (run_dir / "report.json").write_text(json.dumps({"parser": parsed.get("parser_used"), "coverage": parsed.get("coverage"), **report}, indent=1))
    return rec


def grade_episode(episode: dict[str, Any], task: Any, work_dir: Path, opts: GradeOptions | None = None) -> Verdict:
    """Two-run pristine replay (second run only after a clean first pass) with tamper flags; never raises for a bad patch.
    Sandbox failures are recorded in the run (`error`, `retryable`) and give r_pass = 0; callers retry those episodes."""
    task = as_task_dict(task)
    opts = opts or GradeOptions()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    flags = tamper_flags(episode, "swe")
    non_test_patch, excluded_paths = split_patch(episode.get("patch"))
    repo_name = repo_name_of(task)
    cal = calibration_of(task)
    spec = eval_spec_of(task)
    extra: dict[str, Any] = {"instance_id": task["upstream_id"], "repo_name": repo_name, "image": task.get("image"),
                             "image_digest": (spec or {}).get("image_digest"), "excluded_paths": excluded_paths,
                             "scratch_test_files": scratch_test_files(episode.get("patch")),
                             "non_test_patch_sha256": sha256_text(non_test_patch), "non_test_patch_bytes": len(non_test_patch),
                             "strict_missing_as_fail": True, "network": opts.network, "cpus": cpu_limit(opts.cpus),
                             "calibration": calibration_summary(task), "eval_path": "eval_spec" if spec else "derived",
                             "reporter": (spec or {}).get("reporter", "text")}
    runs: list[dict[str, Any]] = []
    if "empty_diff" in flags or not non_test_patch.strip():
        extra["skipped_reason"] = "empty non-test diff"
        return verdict_from_runs(episode, flags, runs, extra=extra, second_run_required=opts.second_run)
    if cal.get("status") == "excluded":   # the gold fix itself cannot pass under our sandbox: no replay can be a verified positive
        extra["skipped_reason"] = f"task excluded by calibration ({cal.get('reason')})"
        return verdict_from_runs(episode, flags, runs, extra=extra, second_run_required=opts.second_run)
    try:
        profile = profile_info(repo_name)
    except SandboxError as e:
        runs.append({"pass": False, "run": 1, "error": str(e), "retryable": True, "seconds": 0.0})
        return verdict_from_runs(episode, flags, runs, extra=extra, second_run_required=opts.second_run)
    first = replay_once(task, non_test_patch, work_dir / "run1", profile, opts, 1)
    runs.append(first)
    if first["pass"] and opts.second_run and not flags:
        runs.append(replay_once(task, non_test_patch, work_dir / "run2", profile, opts, 2))
    elif first["pass"] and flags:
        extra["skipped_reason"] = "tamper flags set; second run skipped"
    extra["env_failure"] = any(r.get("retryable") for r in runs)
    return verdict_from_runs(episode, flags, runs, extra=extra, second_run_required=opts.second_run)


def grade_patch(task: Any, patch: str | None, run_dir: Path | str, second_run: bool = True, *, episode_id: str | None = None,
                exit_reason: str = "submit", opts: GradeOptions | None = None) -> Verdict:
    """Grade one patch against one calibrated task (see the module docstring). `exit_reason` other than "submit" sets the
    no_submit flag; `opts` overrides the sandbox conditions (its second_run is replaced by the argument)."""
    t = as_task_dict(task)
    opts = dataclasses.replace(opts or GradeOptions(), second_run=second_run)
    episode = {"episode_id": episode_id or f"{t['task_id']}#adhoc#patch#0", "task_id": t["task_id"], "patch": patch or "", "exit_reason": exit_reason}
    return grade_episode(episode, t, Path(run_dir), opts)


def calibrate_task(task: Any, work_dir: Path, opts: GradeOptions | None = None, run_id: str | None = None) -> dict[str, Any]:
    """One gold-fix replay of a SWE-smith task under the grading conditions: the reverse of the dataset's bug patch,
    restricted to non-test paths and applied on the instance branch with the tests restored, exactly as an agent diff
    is. Record semantics: `finish_calibration`; an ok record carries `eval_spec`."""
    task = as_task_dict(task)
    opts = opts or GradeOptions()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base = calibration_base(task, run_id, opts.network)
    probe = dict(task, extra={k: v for k, v in (task.get("extra") or {}).items() if k not in ("calibration", "eval_spec")})   # never calibrate on a calibration
    try:
        gold = gold_patch(probe)
    except ValueError as e:
        return {**base, "status": "excluded", "reason": f"calibration:gold_patch_unavailable:{e}"[:400]}
    non_test_gold, excluded_paths = split_patch(gold)
    base["gold_patch_sha256"] = sha256_text(gold)
    if excluded_paths:
        return {**base, "status": "excluded", "reason": "calibration:gold_touches_test_paths", "excluded_paths": excluded_paths}
    f2p, p2p = test_lists(task)
    if not f2p:
        return {**base, "status": "excluded", "reason": "calibration:no_f2p_list"}
    try:
        profile = profile_info(repo_name_of(task))
    except SandboxError as e:
        return {**base, "status": "env_failure", "reason": f"sandbox: {e}"[:400]}
    record, run = calibrate_with(lambda d, k, structured: replay_once(probe, non_test_gold, d, profile, opts, k, keep_status_map=True,
                                                                     prefer_structured=structured),
                                 base, f2p, p2p, work_dir)
    if record["status"] == "ok":
        record["eval_spec"] = build_eval_spec(run, record, task.get("image") or profile["image_name"])
    return record


# --- CLI ------------------------------------------------------------------------------------------------------------
def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def _opts(a: argparse.Namespace) -> GradeOptions:
    return GradeOptions(cpus=a.cpus, memory=a.memory, cgroup_parent=a.cgroup_parent, network=a.network, mirrors_dir=Path(a.mirrors_dir),
                        second_run=not getattr(a, "no_second_run", False), timeout_scale=a.timeout_scale,
                        labels={"apex.run_id": a.run_id} if a.run_id else {})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("grade", "calibrate"):
        g = sub.add_parser(name, help="grade one patch or episode (train venv)" if name == "grade" else "gold-fix replay of one task (train venv)")
        if name == "grade":
            src = g.add_mutually_exclusive_group(required=True)
            src.add_argument("--episode", help="schema.Episode record as JSON (patch and exit_reason are read from it)")
            src.add_argument("--patch", help="a diff file to grade as a submitted patch")
            g.add_argument("--no-second-run", action="store_true", help="debugging only: a verdict from one run is not a positive")
        g.add_argument("--task", required=True, help="schema.Task record as JSON (a line of data/pools/swesmith/tasks.jsonl)")
        g.add_argument("--out", required=True, help="verdict / calibration JSON to write (atomic)")
        g.add_argument("--work-dir", required=True, help="directory for run1/ run2/ (grade) or gold/ gold_text/ (calibrate) logs")
        g.add_argument("--cpus", type=float, default=2.0)
        g.add_argument("--memory", default="8g")
        g.add_argument("--cgroup-parent", default=None, help="systemd slice for the containers")
        g.add_argument("--network", default="none", help="docker network of the replay (none = hermetic)")
        g.add_argument("--mirrors-dir", default=str(DEFAULT_MIRRORS_DIR))
        g.add_argument("--timeout-scale", type=float, default=2.0)
        g.add_argument("--run-id", default=None, help="label for the containers")
    m = sub.add_parser("mirror", help="clone (once) the SWE-smith mirror of a repository")
    m.add_argument("--repo", required=True, help="repo_name, e.g. joho__godotenv.3a7a1902")
    m.add_argument("--mirrors-dir", default=str(DEFAULT_MIRRORS_DIR))
    pr = sub.add_parser("profile", help="[toolkit venv] profile facts as JSON")
    pr.add_argument("--repo", required=True)
    pa = sub.add_parser("parse", help="[toolkit venv] parse a test log with the profile's log_parser")
    pa.add_argument("--repo", required=True)
    pa.add_argument("--log", required=True)
    pa.add_argument("--expected-file", default=None, help='JSON file {"expected": [ids]} (for the coverage figure)')
    tc = sub.add_parser("test-cmd", help="[toolkit venv] the profile's test command narrowed to a task's F2P/P2P files")
    tc.add_argument("--repo", required=True)
    tc.add_argument("--instance", required=True)
    tc.add_argument("--lists", required=True, help='JSON file {"f2p": [...], "p2p": [...]}')
    a = p.parse_args(argv)
    if a.cmd == "grade":
        task = json.loads(Path(a.task).read_text())
        if a.episode:
            episode = json.loads(Path(a.episode).read_text())
        else:
            episode = {"episode_id": f"{task['task_id']}#adhoc#patch#0", "task_id": task["task_id"], "patch": Path(a.patch).read_text(), "exit_reason": "submit"}
        v = grade_episode(episode, task, Path(a.work_dir), _opts(a))
        _write_json_atomic(Path(a.out), dataclasses.asdict(v))
        print(json.dumps({"episode_id": v.episode_id, "r_pass": v.r_pass, "tamper_flags": v.tamper_flags, "n_runs": len(v.runs)}))
        return 0
    if a.cmd == "calibrate":
        res = calibrate_task(json.loads(Path(a.task).read_text()), Path(a.work_dir), _opts(a), run_id=a.run_id)
        _write_json_atomic(Path(a.out), res)
        print(json.dumps({"task_id": res["task_id"], "status": res["status"], "reason": res.get("reason"), "gold_pass": res["gold_pass"],
                          "n_p2p_excluded": len(res["p2p_excluded"]), "n_f2p_missing": len(res["f2p_missing"]), "reporter": res.get("reporter"),
                          "parser": res["parser"], "coverage": res["coverage"], "seconds": res["seconds"]}))
        return 0
    if a.cmd == "mirror":
        print(ensure_mirror(a.repo, Path(a.mirrors_dir)))
        return 0
    if a.cmd == "profile":
        print(json.dumps(_tk_profile(a.repo)))
        return 0
    if a.cmd == "parse":
        print(json.dumps(_tk_parse(a.repo, a.log, read_json_arg(a.expected_file, "expected") if a.expected_file else [])))
        return 0
    if a.cmd == "test-cmd":
        print(json.dumps(_tk_test_cmd(a.repo, a.instance, read_json_arg(a.lists, "f2p"), read_json_arg(a.lists, "p2p"))))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
