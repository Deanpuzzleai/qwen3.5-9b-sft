"""mini-swe-agent 2.4.6 rollout driver (tool-call mode) for SWE-smith tasks (docs/DATA_PLAN.md section 11).

One worker process runs one episode in the task's SWE-smith image with the agent configuration the SWE-bench Pro
evaluation uses: the lock's mini-swe-agent config file (system and instance templates, observation and format-error
templates, environment variables) with the lock's sampling (apex_eval.swebench_pro.build_agent_config), litellm
`openai/<served id>`, one `bash` tool, 60 s per command. The stage-2 caps replace the evaluation budget: 50 steps,
16,384 tokens per turn, 40,000 output tokens per episode (logging proxy), 20 minutes wall clock (the agent's own
wall_time_limit_seconds, so the patch is still collected). The per-request seed is the sample's seed.

Sandbox: container `s2roll_<episode>_<rand>` from the task image (/testbed), 1 CPU / 4 GB, `--network none`, run as root
with `bash -c` (a login shell drops /usr/local/go/bin in the Go images); every agent command sources AGENT_ENV_FILE through
BASH_ENV, which activates the image's conda env `testbed` where there is one (the Python images activate it only in
~/.bashrc, which a non-login `bash -c` never reads, so their tests would otherwise run under the base interpreter without
the package). The agent sees the repository at /app, as in SWE-bench Pro (its driver starts in `/app`, else `/testbed`):
after the preparation below the repository is moved to /app, /testbed becomes a symlink to it (scripts and absolute paths
keep working) and the editable-install path hooks of the conda envs (*.pth, __editable__ finders, *.egg-link, pip's
direct_url.json) are pointed
at /app, so pwd, os.getcwd(), pytest's rootdir and tracebacks all show /app; the agent's cwd is /app (SANDBOX_LAYOUT,
recorded per episode and in the dev protocol). Stage-2 rollouts (build_spec size_runtimes, set by run_rollouts.py) also
export the grader's runtime sizing for the sandbox quota (`runtime_env`: GOMAXPROCS, PYTEST_XDIST_AUTO_NUM_WORKERS,
MODIN_CPUS / MODIN_MEMORY, APEX_CPUS; recorded as extra.runtime_env): the images' Go 1.24, pytest-xdist's `-n auto` and
modin size themselves from the host's 192 CPUs inside the 1-CPU container, and a full `go test` of the larger Go suites
then runs past the 60 s command timeout. The dev evaluation keeps its recorded sandbox (no runtime_env). Grading is unaffected: the diff is repo-relative and is replayed on a
pristine image. Repository state: the task's instance branch (bug
applied, FAIL_TO_PASS tests removed: what the toolkit's own agents see) is fetched from the host's SWE-smith mirror
(grade/swesmith.ensure_mirror) through a thin git bundle (the branch's commits on top of the image's HEAD; the whole
mirror is copied in only if the image HEAD is unknown to the mirror), checked out, and the history is then squashed into
one commit with remotes, other branches, tags and reflog removed and the objects pruned, so neither the clean code nor
the bug commit can be recovered with git. The agent's diff is `git add -A && git diff --cached HEAD` at the end.

Episode: harness-faithful transcript (system, task user turn, assistant turns with reasoning_content and tool_calls,
`tool` observations, format-error user turns), exit_reason (submit | step_cap | time_cap | token_cap | format_errors |
error), steps, output tokens, wall time, the final diff and the request log. Sandbox failures (image, mirror, checkout)
give exit_reason error with extra.exit_status "SandboxError"; the scheduler retries those.

Worker CLI (scripts/data/run_rollouts.py launches it with .venv/harbor/bin/python, where mini-swe-agent lives):
    PYTHONPATH=src .venv/harbor/bin/python -m apex_data.rollout.mini_swe run --spec spec.json --out episode.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from apex_data.grade.swesmith import (DEFAULT_MIRRORS_DIR, GIT_SAFE_DIRS, WORKDIR, SandboxError, ensure_mirror, image_digest,
                                      memory_bytes, repo_name_of, slug)
from apex_data.paths import REPO
from apex_data.rollout.endpoint import (BUDGET_MARKER, SCAFFOLD_FOR_FAMILY, Caps, Endpoint, RequestLogProxy, caps_for,
                                        episode_id as make_episode_id, reasoning_of, sample_seed)
from apex_data.schema import Episode

HARBOR_PYTHON = REPO / ".venv" / "harbor" / "bin" / "python"
SRC_DIR = REPO / "src"
CONTAINER_PREFIX = "s2roll_"
COMMAND_TIMEOUT_SEC = 60                       # per bash command (the SWE-bench Pro host driver's setting)
INTERPRETER = ["bash", "-c"]
PATCH_MAX_BYTES = 5_000_000
MIRROR_IN_CONTAINER = "/apex_mirror.git"
AGENT_REPO_DIR = "/app"                        # where the agent sees the repository (SWE-bench Pro's layout)
AGENT_ENV_FILE = "/etc/apex_agent_env.sh"      # sourced by every agent command through BASH_ENV
AGENT_ENV_SCRIPT = ("if [ -f /opt/miniconda3/etc/profile.d/conda.sh ] && [ -d /opt/miniconda3/envs/testbed ]; then\n"
                    "  . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed\n"
                    "fi\n")
# /testbed at the start of a path only (never inside a longer name such as /opt/miniconda3/envs/testbed)
_PATH_HOOK_SED = r"s#(^|[^[:alnum:]_./-])/testbed($|/|[^[:alnum:]_.-])#\1/app\2#g"
_URL_HOOK_SED = r"s#file:///testbed($|/|[^[:alnum:]_.-])#file:///app\1#g"   # pip's direct_url.json (`pip show`)
PRESENT_AT_APP = (
    f"cd / && if [ -e {AGENT_REPO_DIR} ]; then rmdir {AGENT_REPO_DIR}; fi && mv {WORKDIR} {AGENT_REPO_DIR} && "
    f"ln -s {AGENT_REPO_DIR} {WORKDIR} && "
    "{ find /opt/miniconda3/envs -maxdepth 6 -path '*site-packages/*' \\( -name '*.pth' -o -name '__editable__*.py' -o -name '*.egg-link' "
    f"-o -name direct_url.json \\) -exec grep -l {WORKDIR} {{}} + 2>/dev/null || true; }} | xargs -r sed -i -E -e '{_PATH_HOOK_SED}' -e '{_URL_HOOK_SED}'"
)
SANDBOX_LAYOUT = (f"repository at {AGENT_REPO_DIR} (moved from {WORKDIR}, {WORKDIR} symlinked to it, editable-install hooks rewritten); "
                  f"agent cwd {AGENT_REPO_DIR}; BASH_ENV={AGENT_ENV_FILE} activates conda env testbed when present")
BUNDLE_IN_CONTAINER = "/tmp/apex_task.bundle"
EXIT_REASONS = {"Submitted": "submit", "LimitsExceeded": "step_cap", "TimeExceeded": "time_cap",
                "OutputTokenBudgetExceeded": "token_cap", "RepeatedFormatError": "format_errors"}
AGENT_KEYS = ("system_template", "instance_template", "step_limit", "cost_limit", "wall_time_limit_seconds")
SANITISE_GIT = (
    "git config user.email apex@localhost && git config user.name apex && "
    "git -c core.hooksPath=/dev/null checkout -q --orphan apex_task_state && git add -A && "
    "git -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --no-verify -m 'task state' && "
    "git for-each-ref --format='%(refname:short)' refs/heads | grep -vx apex_task_state | xargs -r git branch -q -D && "
    "git remote | xargs -r -n1 git remote remove && git tag -l | xargs -r git tag -d >/dev/null && "
    "git stash clear && git reflog expire --expire=now --all && git gc -q --prune=now"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- run-level setup (train venv) -----------------------------------------------------------------------------------------
def agent_config(out_path: Path, caps: Caps | None = None) -> dict[str, Any]:
    """The SWE-bench Pro lock's mini-swe-agent config (apex_eval.swebench_pro.build_agent_config: the lock's config file
    with the lock's sampling) with the stage-2 caps: step_limit, wall_time_limit_seconds and max_tokens per turn. The
    per-request seed is set per sample by the worker. Written to out_path (the run's config snapshot)."""
    import yaml  # noqa: PLC0415

    from apex_eval.swebench_pro import LOCK, build_agent_config  # noqa: PLC0415 - read-only reuse of the evaluation driver

    caps = caps or caps_for("swe")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_agent_config(yaml.safe_load(LOCK.read_text()), caps.max_steps, out_path)
    cfg["agent"]["step_limit"] = caps.max_steps
    cfg["agent"]["wall_time_limit_seconds"] = caps.wall_sec
    cfg["model"]["model_kwargs"]["max_tokens"] = caps.max_tokens_per_turn
    cfg["model"]["model_kwargs"]["seed"] = None   # per sample (sample_seed)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def problem_statement(task: dict[str, Any]) -> str:
    return str((task.get("extra") or {}).get("problem_statement") or task.get("problem_statement") or "")


def runtime_env(cpus: float, memory: str) -> dict[str, str]:
    """The grader's runtime sizing (grade.swesmith.eval_script) for a sandbox of `cpus` CPUs and `memory` (module
    docstring)."""
    n = max(1, int(cpus))
    return {"APEX_CPUS": str(n), "PYTEST_XDIST_AUTO_NUM_WORKERS": str(n), "GOMAXPROCS": str(n), "MODIN_CPUS": str(n),
            "MODIN_MEMORY": str(memory_bytes(memory) // 4)}


def build_spec(task: dict[str, Any], endpoint: Endpoint, *, round_: str, policy: str, sample_index: int, episode_dir: Path,
               requests_dir: Path, run_id: str, agent_config_path: Path, caps: Caps | None = None,
               docker: dict[str, Any] | None = None, mirrors_dir: Path | str = DEFAULT_MIRRORS_DIR,
               size_runtimes: bool = False) -> dict[str, Any]:
    caps = caps or caps_for("swe")
    eid = make_episode_id(task["task_id"], round_, policy, sample_index)
    d = {"cpus": 1, "memory": "4g", "network": "none", "cgroup_parent": None, **(docker or {})}
    spec = {"episode_id": eid, "task": task, "round": round_, "policy": policy, "sample_index": int(sample_index),
            "seed": sample_seed(task["task_id"], sample_index), "endpoint": endpoint.to_dict(), "caps": dataclasses.asdict(caps),
            "agent_config": str(agent_config_path), "run_id": run_id, "mirrors_dir": str(mirrors_dir), "episode_dir": str(episode_dir),
            "docker": d, "requests_path": str(Path(requests_dir) / f"{slug(eid, 120)}.jsonl.gz")}
    if size_runtimes:
        spec["runtime_env"] = runtime_env(d["cpus"], d["memory"])
    return spec


def worker_command(spec_path: Path, out_path: Path) -> list[str]:
    return [str(HARBOR_PYTHON), "-m", "apex_data.rollout.mini_swe", "run", "--spec", str(spec_path), "--out", str(out_path)]


def worker_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC_DIR), "MSWEA_COST_TRACKING": "ignore_errors", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "MSWEA_SILENT_STARTUP": "1", "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": os.environ.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "5")}


# --- pure helpers (unit-tested in the train venv) -------------------------------------------------------------------------
def exit_reason_of(exit_status: str | None, *, budget_hit: bool = False, last_finish_reason: str | None = None) -> str:
    """schema.EXIT_REASONS from mini-swe-agent's exit status. A submitted episode is `submit` whatever else happened
    (the render filters drop finish_reason == length turns on their own)."""
    if exit_status == "Submitted":
        return "submit"
    if budget_hit or exit_status == "OutputTokenBudgetExceeded":
        return "token_cap"
    if exit_status in EXIT_REASONS:
        return EXIT_REASONS[exit_status]
    if last_finish_reason == "length":
        return "token_cap"
    return "error" if exit_status else "unknown"


def transcript(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(wire-faithful messages, per-assistant-turn metadata) from mini-swe-agent's `agent.messages`: the `extra` blocks
    are dropped exactly as LitellmModel._prepare_messages_for_api drops them, reasoning is kept under reasoning_content."""
    out: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        msg: dict[str, Any] = {"role": role, "content": m.get("content")}
        extra = m.get("extra") or {}
        if role == "assistant":
            msg["reasoning_content"] = reasoning_of(m)
            calls = [{"id": tc.get("id"), "type": tc.get("type") or "function",
                      "function": {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments")}}
                     for tc in m.get("tool_calls") or []]
            msg["tool_calls"] = calls or None
            resp = extra.get("response") or {}
            choice = (resp.get("choices") or [{}])[0]
            turns.append({"finish_reason": choice.get("finish_reason"), "usage": resp.get("usage"),
                          "n_actions": len(extra.get("actions") or []), "timestamp": extra.get("timestamp")})
        elif role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id")
        elif role == "user" and extra.get("interrupt_type") == "FormatError":
            msg["format_error"] = True
        out.append(msg)
    return out, turns


# --- worker (harbor venv) ----------------------------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int, **kw: Any) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kw)
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"timed out after {timeout}s: {' '.join(cmd[:4])}") from e


class _Worker:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.task = spec["task"]
        self.caps = Caps(**spec["caps"])
        self.endpoint = Endpoint.from_dict(spec["endpoint"])
        self.episode_dir = Path(spec["episode_dir"])
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.container_name = f"{CONTAINER_PREFIX}{slug(spec['episode_id'], 80)}_{uuid.uuid4().hex[:6]}"
        self.env: Any = None
        self.prep_log: list[dict[str, Any]] = []

    def _run_args(self) -> list[str]:
        d = self.spec["docker"]
        args = ["--cpus", str(d["cpus"]), "--memory", str(d["memory"]), "--network", str(d["network"]),
                "--label", "apex.role=rollout", "--label", "apex.unit=s2roll", "--label", f"apex.run_id={self.spec['run_id']}",
                "--label", f"apex.task={slug(self.task['task_id'], 120)}"]
        if d.get("cgroup_parent"):
            args += ["--cgroup-parent", str(d["cgroup_parent"])]
        return args

    def _exec(self, cmd: str, timeout: int = 600) -> dict[str, Any]:
        t0 = time.time()
        out = self.env.execute({"command": cmd}, timeout=timeout)
        self.prep_log.append({"cmd": cmd[:160], "rc": out["returncode"], "tail": out["output"][-400:], "seconds": round(time.time() - t0, 1)})
        if out["returncode"] != 0:
            raise SandboxError(f"preparation command failed ({out['returncode']}): {cmd[:120]}: {out['output'][-400:]}")
        return out

    def _copy_in(self, src: Path, dst: str) -> None:
        r = _run(["docker", "cp", "-q", str(src), f"{self.env.container_id}:{dst}"], 900)
        if r.returncode != 0:
            raise SandboxError(f"docker cp failed: {r.stderr[-300:]}")

    def prepare_repository(self) -> dict[str, Any]:
        """Instance branch checked out from the mirror, then the history squashed (module docstring)."""
        iid = str(self.task["upstream_id"])
        ref = f"refs/heads/{iid}"
        mirror = ensure_mirror(repo_name_of(self.task), self.spec["mirrors_dir"])
        head = self._exec(f"{GIT_SAFE_DIRS} && git rev-parse HEAD", timeout=120)["output"].strip().splitlines()[-1]
        info: dict[str, Any] = {"image_head": head, "transfer": "bundle"}
        with tempfile.TemporaryDirectory(prefix="s2roll_bundle_") as tmp:
            bundle = Path(tmp) / "task.bundle"
            r = _run(["git", "-C", str(mirror), "bundle", "create", "-q", str(bundle), f"{head}..{ref}"], 600)
            if r.returncode == 0:
                self._copy_in(bundle, BUNDLE_IN_CONTAINER)
                source = BUNDLE_IN_CONTAINER
            else:   # the image HEAD is not in the mirror: copy the mirror itself (slower, same result)
                info.update(transfer="mirror", bundle_error=r.stderr.strip()[-300:])
                self._copy_in(mirror, MIRROR_IN_CONTAINER)
                source = MIRROR_IN_CONTAINER
        q = shlex.quote(iid)
        self._exec(f"git fetch -q {source} +refs/heads/{q}:refs/heads/{q} && git checkout -q -f {q} && rm -rf {source}", timeout=900)
        info["head_subject"] = self._exec("git log -1 --format=%s", timeout=60)["output"].strip()
        self._exec(SANITISE_GIT, timeout=900)
        return info

    def present_repository(self) -> dict[str, Any]:
        """After the preparation: the repository moved to /app with /testbed symlinked to it, the editable-install hooks
        pointed at /app, the agent env file written; from here on every command runs in /app with BASH_ENV set (module
        docstring). Returns what the agent's shell sees (pwd, python, conda env) for the episode record."""
        self._exec(f"cat > {AGENT_ENV_FILE} <<'APEX_ENV'\n{AGENT_ENV_SCRIPT}APEX_ENV", timeout=60)
        self._exec(PRESENT_AT_APP, timeout=600)
        self.env.config.cwd = AGENT_REPO_DIR
        self.env.config.env = {**self.env.config.env, "BASH_ENV": AGENT_ENV_FILE, **(self.spec.get("runtime_env") or {})}
        seen = self._exec("pwd; python -c 'import os, sys; print(os.getcwd()); print(sys.executable)' 2>/dev/null || true; "
                          "echo \"conda_env=${CONDA_DEFAULT_ENV:-none}\"", timeout=120)["output"].strip().splitlines()
        if not seen or seen[0] != AGENT_REPO_DIR:
            raise SandboxError(f"the repository is not presented at {AGENT_REPO_DIR}: {seen[:3]}")
        return {"layout": SANDBOX_LAYOUT, "agent_shell": seen[:4]}

    def collect_patch(self) -> tuple[str | None, dict[str, Any]]:
        out = self.env.execute({"command": "git add -A && git diff --cached HEAD --no-color --binary"}, timeout=300)
        info: dict[str, Any] = {"rc": out["returncode"], "bytes": len(out["output"])}
        if out["returncode"] != 0:
            info["error"] = out["output"][-400:]
            return None, info
        if len(out["output"]) > PATCH_MAX_BYTES:
            info["oversize"] = True
            return None, info
        return out["output"], info

    def _remove_container(self) -> None:
        cid = getattr(self.env, "container_id", None) if self.env is not None else None
        target = cid or self.container_name
        try:
            subprocess.run(["docker", "rm", "-f", target], capture_output=True, timeout=120, check=False)
        except subprocess.TimeoutExpired:
            subprocess.Popen(["docker", "rm", "-f", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        if self.env is not None:
            self.env.container_id = None

    def run(self) -> Episode:
        import yaml  # noqa: PLC0415 - worker-only imports (harbor venv)
        from minisweagent.agents.default import DefaultAgent  # noqa: PLC0415
        from minisweagent.environments.docker import DockerEnvironment  # noqa: PLC0415
        from minisweagent.exceptions import LimitsExceeded  # noqa: PLC0415
        from minisweagent.models.litellm_model import LitellmModel  # noqa: PLC0415
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # noqa: PLC0415

        spec, task, caps = self.spec, self.task, self.caps
        cfg = yaml.safe_load(Path(spec["agent_config"]).read_text())
        model_kwargs = {**cfg["model"]["model_kwargs"], "seed": int(spec["seed"]), "max_tokens": caps.max_tokens_per_turn}
        agent_kwargs = {k: cfg["agent"][k] for k in AGENT_KEYS if k in cfg["agent"]}
        agent_kwargs.update(step_limit=caps.max_steps, wall_time_limit_seconds=caps.wall_sec, cost_limit=0)
        name = self.container_name

        class RolloutDocker(DockerEnvironment):
            """DockerEnvironment whose container carries our name (s2roll_ prefix) and labels."""

            def _start_container(self_inner) -> None:   # noqa: N805
                c = self_inner.config
                # --pull never: a missing image is a sandbox failure, never an implicit pull (pulls only through the paced,
                # Terminal-Bench-aware paths: scripts/data/pull_images.py, run_rollouts.py --pull-missing)
                cmd = [c.executable, "run", "-d", "--pull", "never", "--name", name, "-w", c.cwd, *c.run_args, c.image, "sleep", c.container_timeout]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=c.pull_timeout, check=False)
                if r.returncode != 0:
                    raise SandboxError(f"docker run failed ({r.returncode}): {r.stderr.strip()[-600:]}")
                self_inner.container_id = r.stdout.strip()

        t_start, created = time.time(), utc_now()
        exit_status: str | None = None
        error_text = ""
        agent = None
        patch, patch_info, prep, prep_seconds, t_agent = None, {}, {}, None, None
        with RequestLogProxy(self.endpoint.api_base, spec["requests_path"], output_token_budget=caps.max_output_tokens) as proxy:
            os.environ.update(OPENAI_API_BASE=proxy.url, OPENAI_BASE_URL=proxy.url, OPENAI_API_KEY=self.endpoint.api_key())
            model = LitellmModel(model_name=self.endpoint.litellm_model, model_kwargs=model_kwargs, cost_tracking="ignore_errors",
                                 observation_template=cfg["model"]["observation_template"],
                                 format_error_template=cfg["model"]["format_error_template"])
            budget = caps.max_output_tokens

            class RolloutAgent(DefaultAgent):
                def query(self_inner) -> dict:   # noqa: N805 - stop before a call once the episode's budget is spent
                    if proxy.usage.completion_tokens >= budget:
                        raise LimitsExceeded({"role": "exit", "content": "OutputTokenBudgetExceeded",
                                              "extra": {"exit_status": "OutputTokenBudgetExceeded", "submission": ""}})
                    return super().query()

            try:
                self.env = RolloutDocker(image=task["image"], cwd=WORKDIR, env=(cfg.get("environment") or {}).get("env") or {},
                                         timeout=COMMAND_TIMEOUT_SEC, run_args=self._run_args(), interpreter=INTERPRETER,
                                         container_timeout=f"{caps.wall_sec + 1800}s", pull_timeout=600)
                t0 = time.time()
                prep = self.prepare_repository()
                prep["presentation"] = self.present_repository()
                prep_seconds = round(time.time() - t0, 1)
                agent = RolloutAgent(model, self.env, **agent_kwargs)
                t_agent = time.time()
                exit_status = agent.run(problem_statement(task)).get("exit_status")
                patch, patch_info = self.collect_patch()
            except SandboxError as e:
                exit_status, error_text = "SandboxError", str(e)
            except Exception as e:  # noqa: BLE001 - recorded as an error episode, never dropped
                exit_status, error_text = type(e).__name__, str(e)[:2000]
                if agent is not None and self.env is not None and self.env.container_id:
                    try:
                        patch, patch_info = self.collect_patch()
                    except Exception as e2:  # noqa: BLE001
                        patch_info = {"error": str(e2)[:200]}
            finally:
                self._remove_container()
            usage = proxy.summary()
        messages, turns = transcript(agent.messages) if agent is not None else ([], [])
        last_finish = turns[-1].get("finish_reason") if turns else None
        reason = exit_reason_of(exit_status, budget_hit=usage["budget_hits"] > 0 or BUDGET_MARKER in error_text, last_finish_reason=last_finish)
        extra = {"exit_status": exit_status, "error": error_text or None, "turns": turns, "finish_reasons": [t["finish_reason"] for t in turns],
                 "usage": usage["usage"], "budget_hits": usage["budget_hits"], "clamped_requests": usage["clamped_requests"],
                 "n_format_errors": sum(1 for m in messages if m.get("format_error")), "image": task["image"],
                 "image_digest": image_digest(task["image"]), "container": self.container_name, "prep": prep, "prep_log": self.prep_log,
                 "prep_seconds": prep_seconds, "agent_seconds": round(time.time() - t_agent, 1) if t_agent else None,
                 "patch_info": patch_info, "seed": spec["seed"], "sampling": {k: v for k, v in model_kwargs.items() if k != "drop_params"},
                 "caps": spec["caps"], "model": self.endpoint.model_name, "litellm_model": self.endpoint.litellm_model,
                 "agent_config": spec["agent_config"], "command_timeout_sec": COMMAND_TIMEOUT_SEC, "interpreter": INTERPRETER,
                 "docker": spec["docker"], "sandbox_layout": SANDBOX_LAYOUT, **({"runtime_env": spec["runtime_env"]} if spec.get("runtime_env") else {}),
                 "harness": "mini-swe-agent 2.4.6 toolcall (SWE-bench Pro lock config)"}
        ep = Episode(episode_id=spec["episode_id"], task_id=task["task_id"], round=spec["round"], policy=spec["policy"],
                     sample_index=int(spec["sample_index"]), scaffold=SCAFFOLD_FOR_FAMILY["swe"], messages=messages, tools=[BASH_TOOL],
                     exit_reason=reason, steps=agent.n_calls if agent is not None else 0,
                     output_tokens=int(usage["usage"]["completion_tokens"]), wall_sec=round(time.time() - t_start, 1), patch=patch,
                     request_log=spec["requests_path"], created_at=created, extra=extra)
        ep.validate()
        return ep


def error_episode(spec: dict[str, Any], error: str, *, interrupted: bool = False) -> Episode:
    return Episode(episode_id=spec["episode_id"], task_id=spec["task"]["task_id"], round=spec["round"], policy=spec["policy"],
                   sample_index=int(spec["sample_index"]), scaffold=SCAFFOLD_FOR_FAMILY["swe"], messages=[], tools=None,
                   exit_reason="error", steps=0, output_tokens=0, wall_sec=0.0, patch=None, request_log=spec.get("requests_path"),
                   created_at=utc_now(), extra={"error": error[:2000], "interrupted": interrupted, "seed": spec.get("seed")})


def write_episode(ep: Episode, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(ep), ensure_ascii=False))
    tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one episode (worker)")
    r.add_argument("--spec", required=True, help="episode spec JSON written by scripts/data/run_rollouts.py")
    r.add_argument("--out", required=True, help="schema.Episode JSON to write (atomic)")
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # let `finally` remove the container
    spec = json.loads(Path(a.spec).read_text())
    worker = _Worker(spec)
    try:
        ep = worker.run()
    except BaseException as e:  # noqa: BLE001 - SystemExit(143) included: record what we have, then re-raise
        worker._remove_container()
        ep = error_episode(spec, f"{type(e).__name__}: {e}", interrupted=isinstance(e, SystemExit))
        write_episode(ep, Path(a.out))
        if isinstance(e, SystemExit):
            raise
        return 1
    write_episode(ep, Path(a.out))
    print(json.dumps({"episode_id": ep.episode_id, "exit_reason": ep.exit_reason, "steps": ep.steps, "output_tokens": ep.output_tokens,
                      "wall_sec": ep.wall_sec, "patch_bytes": len(ep.patch or "")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
