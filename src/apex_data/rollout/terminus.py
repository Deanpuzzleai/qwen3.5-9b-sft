"""Terminus-2 (Harbor 0.24.0) rollout driver for TermGrade tasks (docs/DATA_PLAN.md section 11).

One worker process runs one Harbor trial for one (task, sample) pair on the prepared TermGrade dataset
(apex_data.grade.harbor_tests.PREPARED_DIR/<task>, images `tgr-<task>:local` built by scripts/data/build_terminal_envs.py)
with the Terminal-Bench lock's agent settings as the grader module builds them (harbor_tests.terminus2_job_config:
Terminus-2, json parser, summarisation on, the lock's model_info, T 1.0 / top_p 0.95 / top_k 20 / min_p 0, the lock's
max_tokens per turn, store_all_messages) plus the stage-2 caps: max_turns 40, agent timeout 30 min, 60,000 output tokens
per episode (logging proxy), and the sample's seed on every request. Container resources are each task.toml's own (no
overrides, as in the lock). Harbor runs the task's tests/test.sh right after the agent; grade/harbor_tests.grade_trial
turns the trial directory into the Verdict (scripts/data/grade_episodes.py).

The trial is started with `harbor trial start -c trial.yaml` (the single-trial form of `harbor run`: same Trial code,
same result layout) so the trial name, and with it the compose project and every container name, is ours:
`s2roll_<sha1(episode_id)[:16]>` (`_r<N>` for a re-run after an environment failure; earlier trial directories are kept).
A worker interrupted mid-trial removes exactly that project's containers.

Start (every round except the dev evaluation's round "dev", whose trials start at once as in its recorded protocol): before
the trial the worker waits a seeded jitter of 0 to START_JITTER_SEC (sha256 of the trial name), then runs
Harbor's own egress-control probe (DockerEnvironment._egress_control_kernel_support: `docker container run --rm <alpine>
<kernel-config check>`, 30 s timeout, any failure read as "cannot disable internet", which rejects every no-network task)
until it answers within PROBE_FAST_SEC, for at most PROBE_WAIT_SEC. In a launch burst (32 terminal + 64 SWE starts in ten
seconds) the probe timed out in 4 of 32 trials; spreading the starts and waiting for a responsive daemon keeps Harbor's
probe inside its timeout. The wait only moves the trial's start; both are recorded (extra.start_jitter_sec,
extra.probe_wait_sec, extra.probe) and are not part of wall_sec.

Memory (every round except "dev", like Start): Harbor limits a trial container's memory only to the task's declared
`[environment] memory_mb` / `memory` or a trial-level override, and TermGrade's task.toml declares none, so its containers
ran without a limit (an SFT dev trial's verification script reached ~560 GB RSS). A task without a declared memory gets
Harbor's `environment.override_memory_mb` = ROLLOUT_MEMORY_MB (compose `mem_limit`; the host has no swap, so the limit
caps the container's whole memory); a declared value is left to Harbor. Recorded as extra.memory_limit {mb, source}.

Transcript (messages exactly as sent, reasoning kept): Harbor's Chat history (`agent_result.metadata.all_messages`,
recorded with store_all_messages) is the sequence the model was sent: user #1 = the Terminus-2 prompt with the task, then
alternating assistant JSON responses and user observations (no system message; earlier reasoning is not replayed:
interleaved_thinking false). Each assistant turn gets its reasoning_content from the logging proxy's record of the
response with the same content, in order (fallback: the trajectory step's reasoning). A context summarisation is
recorded as extra.context_reset and the render filters drop such episodes. exit_reason: task_complete (the double
task_complete handshake), step_cap, time_cap (AgentTimeoutError), token_cap (budget refusal), error (any other
exception: environment failures are retried by the scheduler).

Worker CLI (scripts/data/run_rollouts.py launches it with .venv/harbor/bin/python):
    PYTHONPATH=src .venv/harbor/bin/python -m apex_data.rollout.terminus run --spec spec.json --out episode.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from apex_data.grade import harbor_tests
from apex_data.paths import REPO, SEED
from apex_data.rollout.endpoint import (BUDGET_MARKER, SCAFFOLD_FOR_FAMILY, Caps, Endpoint, RequestLogProxy, answered, caps_for,
                                        episode_id as make_episode_id, read_request_log, sample_seed)
from apex_data.schema import Episode

HARBOR = REPO / ".venv" / "harbor" / "bin" / "harbor"
HARBOR_PYTHON = REPO / ".venv" / "harbor" / "bin" / "python"
SRC_DIR = REPO / "src"
TRIAL_PREFIX = "s2roll_"
ENVIRONMENT_EXCEPTIONS = ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError",
                          "SandboxLikelyOutOfMemoryError", "MemoryLimitExceededError")   # == apex_eval.harbor_utils
GRACE_SEC = 1800                     # environment build/start + verifier on top of the agent timeout before a hard kill
START_JITTER_SEC = 30.0              # seeded start delay per trial (module docstring: Start)
PROBE_WAIT_SEC = 120.0               # longest wait for a responsive Docker daemon before the trial
PROBE_FAST_SEC = 10.0                # a probe answering within this is well inside Harbor's 30 s
PROBE_TIMEOUT_SEC = 30               # Harbor's own probe timeout
PROBE_RETRY_SEC = 10.0
DEV_ROUND = "dev"                    # apex_data.devset.run builds its specs with round_="dev"
ROLLOUT_MEMORY_MB = 16_384           # memory limit of a rollout trial whose task declares none (module docstring: Memory)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trial_name(episode_id: str) -> str:
    return f"{TRIAL_PREFIX}{hashlib.sha1(episode_id.encode()).hexdigest()[:16]}"


def compose_project(name: str) -> str:
    """Harbor's compose project of a trial's environment (session id `<trial>__env`, sanitised like
    harbor.environments.docker._sanitize_docker_compose_project_name)."""
    return re.sub(r"[^a-z0-9_-]", "-", f"{name}__env".lower())


# --- spec and trial config (pure) ------------------------------------------------------------------------------------------
def build_spec(task: dict[str, Any], endpoint: Endpoint, *, round_: str, policy: str, sample_index: int, episode_dir: Path,
               requests_dir: Path, run_id: str, caps: Caps | None = None, dataset_dir: Path | str = harbor_tests.PREPARED_DIR) -> dict[str, Any]:
    caps = caps or caps_for("terminal")
    eid = make_episode_id(task["task_id"], round_, policy, sample_index)
    base = trial_name(eid)
    harbor_dir = Path(episode_dir) / "harbor"
    n_prev = len(list(harbor_dir.glob(f"{base}*"))) if harbor_dir.exists() else 0   # an earlier attempt's trial is kept
    name = base if n_prev == 0 else f"{base}_r{n_prev}"
    return {"episode_id": eid, "task": task, "round": round_, "policy": policy, "sample_index": int(sample_index),
            "seed": sample_seed(task["task_id"], sample_index), "endpoint": endpoint.to_dict(), "caps": dataclasses.asdict(caps),
            "task_dir": str(Path(dataset_dir) / str(task["upstream_id"])), "trial_name": name, "run_id": run_id,
            "episode_dir": str(episode_dir), "requests_path": str(Path(requests_dir) / f"{name}.jsonl.gz")}


def declared_memory_mb(task_dir: Path | str) -> int | None:
    """The task.toml's `[environment] memory_mb`, or its deprecated `memory` size string ("4G", "512M"), in MB."""
    import tomllib  # noqa: PLC0415

    path = Path(task_dir) / "task.toml"
    if not path.exists():     # Harbor itself rejects such a task; nothing declared here
        return None
    env = tomllib.loads(path.read_text(encoding="utf-8")).get("environment") or {}
    if env.get("memory_mb") is not None:
        return int(env["memory_mb"])
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*", str(env.get("memory") or ""), re.I)
    if not m:
        return None
    return int(float(m.group(1)) * {"": 1 / (1 << 20), "k": 1 / 1024, "m": 1, "g": 1024, "t": 1 << 20}[m.group(2).lower()])


def memory_limit(spec: dict[str, Any]) -> dict[str, Any] | None:
    """{mb, source} of a rollout trial's memory limit (module docstring: Memory); None for the dev evaluation's round."""
    if spec["round"] == DEV_ROUND:
        return None
    declared = declared_memory_mb(spec["task_dir"])
    return {"mb": declared, "source": "task"} if declared else {"mb": ROLLOUT_MEMORY_MB, "source": "default"}


def trial_config(spec: dict[str, Any], api_base: str) -> dict[str, Any]:
    """Harbor TrialConfig (schema: `harbor trial schema`) for one rollout trial."""
    caps = Caps(**spec["caps"])
    endpoint = Endpoint.from_dict(spec["endpoint"])
    agent = harbor_tests.terminus2_job_config(endpoint.model_name, api_base, seed=int(spec["seed"]), agent_timeout_sec=float(caps.wall_sec),
                                              max_turns=caps.max_steps, store_all_messages=True)["agents"][0]
    if caps.max_tokens_per_turn is not None:
        agent["kwargs"]["llm_call_kwargs"]["max_tokens"] = int(caps.max_tokens_per_turn)
    environment: dict[str, Any] = {"delete": True}
    limit = memory_limit(spec)
    if limit and limit["source"] == "default":
        environment["override_memory_mb"] = limit["mb"]
    return {"task": {"path": spec["task_dir"]}, "trial_name": spec["trial_name"], "trials_dir": str(Path(spec["episode_dir"]) / "harbor"),
            "agent": agent, "environment": environment}


def worker_command(spec_path: Path, out_path: Path) -> list[str]:
    return [str(HARBOR_PYTHON), "-m", "apex_data.rollout.terminus", "run", "--spec", str(spec_path), "--out", str(out_path)]


def worker_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC_DIR), "LITELLM_LOCAL_MODEL_COST_MAP": "True"}


# --- transcript (pure) -------------------------------------------------------------------------------------------------------
def rebuild_messages(all_messages: list[dict[str, Any]] | None, responses: list[dict[str, Any]],
                     trajectory: dict[str, Any] | None = None) -> dict[str, Any]:
    """{messages, n_unmatched_reasoning, n_assistant}: Harbor's Chat history with reasoning_content attached to each
    assistant turn from the proxy's answered responses (`endpoint.answered`) with the same content, consumed in order;
    trajectory steps with the same message text are the fallback."""
    pools: dict[str, list[str | None]] = {}
    for r in responses:
        pools.setdefault(r["message"].get("content") or "", []).append(r["message"].get("reasoning_content"))
    for s in harbor_tests.agent_steps(trajectory):
        if isinstance(s.get("message"), str):
            pools.setdefault("\0traj\0" + s["message"], []).append(s.get("reasoning_content"))
    messages: list[dict[str, Any]] = []
    unmatched = n_assistant = 0
    for m in all_messages or []:
        role, content = m.get("role"), m.get("content")
        if role != "assistant":
            messages.append({"role": role, "content": content})
            continue
        n_assistant += 1
        content = content if isinstance(content, str) else ""
        pool = pools.get(content) or pools.get("\0traj\0" + content)
        if pool:
            reasoning = pool.pop(0)
        else:
            reasoning, unmatched = None, unmatched + 1
        messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
    return {"messages": messages, "n_unmatched_reasoning": unmatched, "n_assistant": n_assistant}


def exit_reason_of(outcome_exit: str, *, budget_hit: bool, harbor_rc: int | None, exception_type: str | None) -> str:
    """schema.EXIT_REASONS: the grader's reading of the trial (harbor_tests.exit_reason_of: task_complete | time_cap |
    error | step_cap | unknown) refined by what the proxy saw (token_cap) and a killed harbor process (time_cap)."""
    if outcome_exit == "task_complete":
        return "task_complete"
    if budget_hit:
        return "token_cap"
    if harbor_rc is None and not exception_type:
        return "time_cap"
    return outcome_exit


def is_environment_failure(exception_type: str | None, exception_message: str | None = None) -> bool:
    return exception_type in ENVIRONMENT_EXCEPTIONS or (exception_type == "RuntimeError" and "Docker compose command failed" in (exception_message or ""))


# --- worker ---------------------------------------------------------------------------------------------------------------------
def cleanup_trial(name: str) -> None:
    """Remove the containers and network of one trial's compose project (after an interrupted harbor process)."""
    project = compose_project(name)
    ps = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
                        capture_output=True, text=True, check=False, timeout=120)
    ids = ps.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, check=False, timeout=300)
    nets = subprocess.run(["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
                          capture_output=True, text=True, check=False, timeout=120).stdout.split()
    for net in nets:
        subprocess.run(["docker", "network", "rm", net], capture_output=True, check=False, timeout=120)


def _stop(proc: subprocess.Popen, name: str) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    cleanup_trial(name)


def start_jitter(name: str) -> float:
    """Seeded start delay in [0, START_JITTER_SEC) for a trial name (a re-run's `_r<N>` name gets its own delay)."""
    return round(int(hashlib.sha256(f"{SEED}:start:{name}".encode()).hexdigest()[:8], 16) / 0x100000000 * START_JITTER_SEC, 2)


def probe_command() -> list[str]:
    """Harbor's egress-control kernel probe, exactly as harbor.environments.docker.docker.DockerEnvironment runs it."""
    from harbor.environments.docker.docker import DockerEnvironment as D  # noqa: PLC0415 - harbor venv only

    return D._engine_cmd("container", "run", "--rm", D._EGRESS_CONTROL_KERNEL_PROBE_IMAGE, "sh", "-c", D._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT)


def wait_for_probe(cmd: list[str], *, run: Any = subprocess.run, sleep: Any = time.sleep, clock: Any = time.monotonic) -> dict[str, Any]:
    """Run the probe until it passes within PROBE_FAST_SEC or PROBE_WAIT_SEC have passed; a fast non-zero exit (the kernel
    lacks support: waiting cannot help) ends the wait. Returns {ok, attempts, last_sec, timed_out}; the trial starts either
    way and Harbor's own probe decides."""
    t_start, attempts = clock(), []
    while True:
        t = clock()
        try:
            rc = run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = None
        attempts.append({"rc": rc, "sec": round(clock() - t, 2)})
        if (rc is not None and attempts[-1]["sec"] <= PROBE_FAST_SEC) or clock() - t_start + PROBE_RETRY_SEC > PROBE_WAIT_SEC:
            break
        sleep(PROBE_RETRY_SEC)
    return {"ok": attempts[-1]["rc"] == 0 and attempts[-1]["sec"] <= PROBE_FAST_SEC, "attempts": len(attempts),
            "last_sec": attempts[-1]["sec"], "timed_out": sum(1 for x in attempts if x["rc"] is None)}


def run_episode(spec: dict[str, Any]) -> Episode:
    task, caps, endpoint = spec["task"], Caps(**spec["caps"]), Endpoint.from_dict(spec["endpoint"])
    episode_dir = Path(spec["episode_dir"])
    episode_dir.mkdir(parents=True, exist_ok=True)
    created = utc_now()
    start: dict[str, Any] = {}
    if spec["round"] != DEV_ROUND:             # module docstring: Start
        start["start_jitter_sec"] = start_jitter(spec["trial_name"])
        time.sleep(start["start_jitter_sec"])
        t_probe = time.time()
        start["probe"] = wait_for_probe(probe_command())
        start["probe_wait_sec"] = round(time.time() - t_probe, 2)
        start["memory_limit"] = memory_limit(spec)
    t0 = time.time()
    rc: int | None = None
    cfg_path = episode_dir / "trial.yaml"
    with RequestLogProxy(endpoint.api_base, spec["requests_path"], output_token_budget=caps.max_output_tokens) as proxy:
        cfg = trial_config(spec, proxy.url)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        env = {**os.environ, "OPENAI_API_KEY": endpoint.api_key(), "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        with (episode_dir / "harbor.log").open("a") as log:
            proc = subprocess.Popen([str(HARBOR), "trial", "start", "-c", str(cfg_path)], cwd=REPO, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=caps.wall_sec + GRACE_SEC)
            except subprocess.TimeoutExpired:      # harbor itself hung: kill it and record what the trial left
                _stop(proc, spec["trial_name"])
            except BaseException:                  # SIGTERM / Ctrl-C: remove the trial's containers, then propagate
                _stop(proc, spec["trial_name"])
                raise
        usage = proxy.summary()
    trial_dir = Path(cfg["trials_dir"]) / spec["trial_name"]
    result = json.loads((trial_dir / "result.json").read_text()) if (trial_dir / "result.json").exists() else {}
    traj_path = trial_dir / "agent" / "trajectory.json"
    traj = json.loads(traj_path.read_text()) if traj_path.exists() else None
    outcome = harbor_tests.read_trial(trial_dir) if result else None
    meta = ((result.get("agent_result") or {}).get("metadata")) or {}
    responses = answered(list(read_request_log(spec["requests_path"])))
    rebuilt = rebuild_messages(meta.get("all_messages"), responses, traj)
    exc = result.get("exception_info") or {}
    exc_type, exc_msg = exc.get("exception_type"), (exc.get("exception_message") or "")[:1000] or None
    if outcome is None:
        reason = "error"
    else:
        reason = exit_reason_of(outcome.exit_reason, budget_hit=usage["budget_hits"] > 0 or BUDGET_MARKER in (exc_msg or ""),
                                harbor_rc=rc, exception_type=exc_type)
    extra = {"harbor": {"trial_name": spec["trial_name"], "trial_dir": str(trial_dir) if result else None, "harbor_rc": rc,
                        "reward": outcome.reward if outcome else None, "handshake": outcome.handshake if outcome else False,
                        "exception_type": exc_type, "exception_message": exc_msg,
                        "environment_failure": (not result) or is_environment_failure(exc_type, exc_msg),
                        "n_agent_steps": outcome.n_agent_steps if outcome else 0, "agent_sec": outcome.agent_sec if outcome else None,
                        "verifier_sec": outcome.verifier_sec if outcome else None, "trial_sec": outcome.trial_sec if outcome else None,
                        "summarization_count": meta.get("summarization_count"), "n_episodes": meta.get("n_episodes")},
             "finish_reasons": [r["finish_reason"] for r in responses], "context_reset": bool(meta.get("summarization_count")),
             "n_unmatched_reasoning": rebuilt["n_unmatched_reasoning"], "usage": usage["usage"], "budget_hits": usage["budget_hits"],
             "clamped_requests": usage["clamped_requests"], "seed": spec["seed"],
             "sampling": (cfg["agent"]["kwargs"].get("llm_call_kwargs") or {}) | {"temperature": cfg["agent"]["kwargs"].get("temperature")},
             "caps": spec["caps"], "model": endpoint.model_name, "litellm_model": endpoint.litellm_model, "task_dir": spec["task_dir"],
             "trial_config": str(cfg_path), "harness": "Harbor 0.24.0 Terminus-2 (Terminal-Bench lock settings)", **start}
    ep = Episode(episode_id=spec["episode_id"], task_id=task["task_id"], round=spec["round"], policy=spec["policy"],
                 sample_index=int(spec["sample_index"]), scaffold=SCAFFOLD_FOR_FAMILY["terminal"], messages=rebuilt["messages"],
                 tools=None, exit_reason=reason, steps=rebuilt["n_assistant"], output_tokens=int(usage["usage"]["completion_tokens"]),
                 wall_sec=round(time.time() - t0, 1), patch=None, request_log=spec["requests_path"], created_at=created, extra=extra)
    ep.validate()
    return ep


def error_episode(spec: dict[str, Any], error: str, *, interrupted: bool = False) -> Episode:
    return Episode(episode_id=spec["episode_id"], task_id=spec["task"]["task_id"], round=spec["round"], policy=spec["policy"],
                   sample_index=int(spec["sample_index"]), scaffold=SCAFFOLD_FOR_FAMILY["terminal"], messages=[], tools=None,
                   exit_reason="error", steps=0, output_tokens=0, wall_sec=0.0, patch=None, request_log=spec.get("requests_path"),
                   created_at=utc_now(), extra={"error": error[:2000], "interrupted": interrupted, "seed": spec.get("seed"),
                                                "harbor": {"trial_name": spec.get("trial_name"), "trial_dir": None, "environment_failure": True}})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one episode (worker)")
    r.add_argument("--spec", required=True, help="episode spec JSON written by scripts/data/run_rollouts.py")
    r.add_argument("--out", required=True, help="schema.Episode JSON to write (atomic)")
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    spec = json.loads(Path(a.spec).read_text())
    rc = 0
    try:
        ep = run_episode(spec)
    except BaseException as e:  # noqa: BLE001 - SystemExit(143) included: record, then re-raise
        ep = error_episode(spec, f"{type(e).__name__}: {e}", interrupted=isinstance(e, SystemExit))
        rc = 1
        if isinstance(e, SystemExit):
            _write(ep, Path(a.out))
            raise
    _write(ep, Path(a.out))
    print(json.dumps({"episode_id": ep.episode_id, "exit_reason": ep.exit_reason, "steps": ep.steps, "output_tokens": ep.output_tokens,
                      "wall_sec": ep.wall_sec, "reward": (ep.extra.get("harbor") or {}).get("reward")}))
    return rc


def _write(ep: Episode, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(ep), ensure_ascii=False))
    tmp.replace(out)


if __name__ == "__main__":
    sys.exit(main())
