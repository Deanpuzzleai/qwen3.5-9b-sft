"""Terminal verdicts from Harbor trials, and the run interface for TermGrade episodes (docs/DATA_PLAN.md sections 6, 10
and 11: "terminal: Harbor's in-trial `tests/test.sh` verdict plus the `task_complete` handshake").

How a TermGrade trial runs (rollouts, U3.3; dev evaluation, U3.4)
------------------------------------------------------------------
Inputs: the prepared Harbor dataset `apex_data.pools.termgrade.PREPARED_DIR` (= /mnt/apex-shared/apex_data/work/tgr_env/
tasks, written by scripts/data/build_terminal_envs.py; each task's image `tgr-<task>:local` is already built), the
admitted task ids of `data/pools/tgr_env/tasks.jsonl` (`admitted_tasks(split)`: split "dev" or "stage2"), and an
OpenAI-compatible endpoint. One job per (policy, sample); `terminus2_job_config` writes everything the Harbor CLI cannot
express, from the Terminal-Bench lock (configs/eval/terminal_bench_2.yaml): Terminus-2, parser json, summarisation on,
the lock's model_info, temperature 1.0, top_p 0.95, top_k 20, min_p 0, max_tokens 81920 per turn, a per-request seed.

    job.yaml (yaml.safe_dump(terminus2_job_config(...))):
      agents:
      - name: terminus-2
        model_name: openai/<served-model-id>
        override_timeout_sec: 1800          # see "Agent time budget" below
        kwargs:
          api_base: http://<endpoint-host>:8000/v1
          model_info: {max_input_tokens: 262144, max_output_tokens: 81920, ...}   # the lock's block, verbatim
          parser_name: json
          enable_summarize: true
          max_turns: 40                     # rollouts (plan 11); null for the dev evaluation (lock: Terminus-2 default)
          temperature: 1.0
          llm_call_kwargs: {max_tokens: 81920, top_p: 0.95, extra_body: {top_k: 20, min_p: 0.0}, seed: <sample seed>}
          store_all_messages: true          # harness-faithful chat in result.json (agent_result.metadata.all_messages)

    OPENAI_API_KEY=<endpoint key> .venv/harbor/bin/harbor run \
        -p /mnt/apex-shared/apex_data/work/tgr_env/tasks -i task_00002_aa2b6f [-i <task> ...] \
        -c <run_dir>/config_snapshot/harbor_job.yaml -n <n_concurrent> \
        --max-retries 2 --retry-include EnvironmentStartTimeoutError --retry-include AgentSetupTimeoutError \
        --retry-include SandboxBuildFailedError --quiet --yes --job-name <job> -o <run_dir>/harbor

`harbor_run_args` builds exactly this argument list (without the executable); `apex_eval.harbor_utils
.harbor_run_resilient` runs it with the repository's probe wait and environment-failure re-runs. Checked end to end
against a mock OpenAI endpoint (logs/data_tgr_env_mockterminus_20261012_0625/): the requests carried temperature 1.0,
top_p 0.95, top_k 20, min_p 0.0, max_tokens 81920 and the seed; the handshake and turn-cap trials read as below. Trial directories land
in <run_dir>/harbor/<job>/<task>__<suffix>/ (result.json, agent/trajectory.json, verifier/{reward.txt, ctrf.json,
pytest.out, test-stdout.txt}); `apex_eval.harbor_utils.load_trials(job_dir)` maps task -> latest trial dir.

Agent time budget. TermGrade's task.toml declares no `[agent] timeout_sec` (its card: a 12 h wall-clock scoring budget),
so the lock's rule "each task.toml timeout applies" leaves Harbor without any agent timeout. The drivers must therefore
pass `agent_timeout_sec`: the rollouts use the plan's terminal cap (30 min, ROLLOUT_AGENT_TIMEOUT_SEC); the dev
evaluation's value is the dev driver's decision, recorded in its run metadata.

Verdict (`read_trial`, `grade_trial`)
-------------------------------------
Harbor uploads the task's tests/ to /tests after the agent finished, runs tests/test.sh on the container the agent left
(TermGrade's test.sh runs pytest from /opt/verifier_venv and writes /logs/verifier/reward.txt = 1 or 0, plus a CTRF
report) and stores the reward in result.json. From one trial directory:
  reward       result.json `verifier_result.rewards.reward` (None when the verifier did not produce one)
  handshake    Terminus-2 ended on its double `task_complete` confirmation: the last two agent steps of
               agent/trajectory.json both carry the `mark_task_complete` tool call (Terminus-2 returns right after the
               second one; any step in between resets the confirmation)
  verifier_consistent  reward 1 only with a CTRF report of >= 1 passed and 0 failed tests, and no reward.json (TermGrade's
               test.sh writes reward.txt only; Harbor would prefer a reward.json the agent planted)
  tamper_flags apex_data.grade.tamper terminal rules (no_submit when no handshake; verifier_touched when a command
               references /tests, /logs/verifier or reward files) plus `verifier_inconsistent`. Note: 10 of TermGrade's
               1,004 instructions name /tests/test_outputs.py, which is absent until grading; an agent that looks for
               it is flagged, so those tasks rarely yield positives (conservative by design; pool records carry
               extra.instruction_names_tests)
  resolved     reward == 1.0 (the Terminal-Bench-style success the dev slice reports; budget-exhausted trials still graded)
  r_pass       resolved and handshake and verifier_consistent and no tamper flag (the stage-2 positive label)
`grade_trial` returns that as a schema.Verdict (verifier "harbor_tests"). The task checks of build_terminal_envs.py use
the same reader on nop and oracle trials (no handshake there: those agents run no LLM).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from apex_data import paths
from apex_data.schema import Verdict, read_jsonl

VERIFIER_VERSION = "harbor_tests:2;harbor==0.24.0"
TB_LOCK = paths.CONFIGS / "eval" / "terminal_bench_2.yaml"
POOL_TASKS = paths.POOLS / "tgr_env" / "tasks.jsonl"
PREPARED_DIR = paths.WORK / "tgr_env" / "tasks"              # == apex_data.pools.termgrade.PREPARED_DIR
ROLLOUT_AGENT_TIMEOUT_SEC = 1800.0                           # plan section 11: terminal rollouts 40 turns / 60K / 30 min
ROLLOUT_MAX_TURNS = 40
TASK_COMPLETE_TOOL = "mark_task_complete"                   # Terminus-2's trajectory name for task_complete: true
RETRY_INCLUDE = ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError")   # lock's list


# --- run interface ------------------------------------------------------------------------------------------------------
def admitted_tasks(split: str, pool_tasks: Path = POOL_TASKS) -> list[str]:
    """Upstream task names that passed every task check, for split 'dev' or 'stage2', in pool-file order."""
    if split not in ("dev", "stage2"):
        raise ValueError("split must be 'dev' or 'stage2'")
    return [r["upstream_id"] for r in read_jsonl(pool_tasks)
            if r["split"] == split and ((r.get("extra") or {}).get("env_check") or {}).get("passed")]


def terminus2_job_config(model_name: str, api_base: str, *, seed: int | None, agent_timeout_sec: float | None,
                         max_turns: int | None, store_all_messages: bool = True, lock_path: Path = TB_LOCK) -> dict[str, Any]:
    """Harbor JobConfig fragment for one Terminus-2 job under the Terminal-Bench lock's agent and sampling settings
    (same construction as apex_eval.terminal_bench.job_config; `model_name` is the served id without the provider
    prefix). `agent_timeout_sec` None leaves Harbor without an agent timeout for TermGrade (see the module docstring)."""
    lock = yaml.safe_load(lock_path.read_text())
    s, agent, inf = lock["inference"]["sampling"], lock["agent"], lock["inference"]
    llm_kwargs: dict[str, Any] = {"max_tokens": int(inf["max_tokens_per_turn"]), "top_p": s["top_p"],
                                  "extra_body": {"top_k": s["top_k"], "min_p": s["min_p"]}}
    if seed is not None:
        llm_kwargs["seed"] = int(seed)
    cfg: dict[str, Any] = {
        "name": agent["name"],
        "model_name": f"openai/{model_name}",
        "kwargs": {"api_base": api_base, "model_info": agent["model_info"], "parser_name": agent["parser"],
                   "enable_summarize": bool(agent["context_summarization"]), "max_turns": max_turns,
                   "temperature": s["temperature"], "llm_call_kwargs": llm_kwargs, "store_all_messages": store_all_messages},
    }
    if agent_timeout_sec is not None:
        cfg["override_timeout_sec"] = float(agent_timeout_sec)
    return {"agents": [cfg]}


def harbor_run_args(task_ids: list[str], job_config_path: Path, *, n_concurrent: int = 16, max_retries: int = 2,
                    dataset_dir: Path = PREPARED_DIR) -> list[str]:
    """`harbor run` arguments for TermGrade trials (prepend the harbor executable; harbor_utils.harbor_run adds --quiet
    --yes --job-name and -o)."""
    args = ["-p", str(dataset_dir)]
    for t in task_ids:
        args += ["-i", t]
    args += ["-c", str(job_config_path), "-n", str(n_concurrent), "--max-retries", str(max_retries)]
    for exc in RETRY_INCLUDE:
        args += ["--retry-include", exc]
    return args


# --- reading a trial ------------------------------------------------------------------------------------------------------
def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_reward(verifier_dir: Path | str) -> float | None:
    """Harbor's precedence: reward.json {"reward": x} over reward.txt (a bare number); None when neither parses."""
    d = Path(verifier_dir)
    obj = _read_json(d / "reward.json")
    if isinstance(obj, dict) and obj:
        try:
            return float(obj.get("reward", next(iter(obj.values()))))
        except (TypeError, ValueError):
            return None
    try:
        return float((d / "reward.txt").read_text().strip())
    except (OSError, ValueError):
        return None


def ctrf_summary(verifier_dir: Path | str) -> dict[str, int] | None:
    """{tests, passed, failed, skipped, pending, other} of the CTRF report TermGrade's test.sh writes, or None."""
    obj = _read_json(Path(verifier_dir) / "ctrf.json")
    s = ((obj or {}).get("results") or {}).get("summary") if isinstance(obj, dict) else None
    if not isinstance(s, dict):
        return None
    return {k: int(s.get(k) or 0) for k in ("tests", "passed", "failed", "skipped", "pending", "other")}


def agent_steps(trajectory: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Agent steps of the main Terminus-2 trajectory (copied summarisation context excluded)."""
    steps = (trajectory or {}).get("steps") or []
    return [s for s in steps if s.get("source") == "agent" and not s.get("is_copied_context")]


def _completes(step: dict[str, Any]) -> bool:
    return any(tc.get("function_name") == TASK_COMPLETE_TOOL for tc in step.get("tool_calls") or [])


def handshake(trajectory: dict[str, Any] | None) -> bool:
    """True iff Terminus-2 stopped on the confirmed task_complete: its last two agent steps both mark completion."""
    steps = agent_steps(trajectory)
    return len(steps) >= 2 and _completes(steps[-1]) and _completes(steps[-2])


def trajectory_messages(trajectory: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Assistant turns as {role, content, reasoning_content}, with the turn's commands rendered as the Terminus-2 JSON
    action (`commands[].keystrokes`, `task_complete`), the form apex_data.grade.tamper reads."""
    out = []
    for s in agent_steps(trajectory):
        cmds = [{"keystrokes": (tc.get("arguments") or {}).get("keystrokes", "")}
                for tc in s.get("tool_calls") or [] if tc.get("function_name") == "bash_command"]
        out.append({"role": "assistant", "reasoning_content": s.get("reasoning_content"),
                    "content": json.dumps({"analysis": s.get("message") or "", "commands": cmds, "task_complete": _completes(s)})})
    return out


@dataclass
class TrialOutcome:
    trial_dir: str
    task_name: str | None
    agent: str | None
    reward: float | None
    exception_type: str | None
    exception_message: str | None
    handshake: bool
    exit_reason: str                    # schema.EXIT_REASONS
    n_agent_steps: int
    ctrf: dict[str, int] | None
    reward_json_present: bool
    verifier_consistent: bool
    tamper_flags: list[str] = field(default_factory=list)
    resolved: bool = False
    r_pass: bool = False
    agent_sec: float | None = None
    verifier_sec: float | None = None
    trial_sec: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seconds(span: dict[str, Any] | None) -> float | None:
    from datetime import datetime
    try:
        return round((datetime.fromisoformat(span["finished_at"]) - datetime.fromisoformat(span["started_at"])).total_seconds(), 1)
    except (TypeError, KeyError, ValueError):
        return None


def exit_reason_of(exc_type: str | None, shook: bool, n_steps: int, max_turns: int | None) -> str:
    if shook:
        return "task_complete"
    if exc_type == "AgentTimeoutError":
        return "time_cap"
    if exc_type:
        return "error"
    if max_turns is not None and n_steps >= max_turns:
        return "step_cap"
    return "unknown"


def read_trial(trial_dir: Path | str) -> TrialOutcome:
    """Everything a verdict needs from one Harbor trial directory (see the module docstring)."""
    d = Path(trial_dir)
    res = _read_json(d / "result.json") or {}
    vdir = d / "verifier"
    reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    reward = None if reward is None else float(reward)
    ei = res.get("exception_info") or {}
    exc_type = ei.get("exception_type")
    exc_msg = (ei.get("exception_message") or "")[:500] or None
    agent_cfg = (res.get("config") or {}).get("agent") or {}
    agent_name = agent_cfg.get("name")
    traj = _read_json(d / "agent" / "trajectory.json")
    shook = handshake(traj)
    n_steps = len(agent_steps(traj))
    max_turns = (agent_cfg.get("kwargs") or {}).get("max_turns")
    exit_reason = exit_reason_of(exc_type, shook, n_steps, max_turns)
    ctrf = ctrf_summary(vdir)
    reward_json = (vdir / "reward.json").exists()
    consistent = not reward_json and (reward != 1.0 or (ctrf is not None and ctrf["passed"] > 0 and ctrf["failed"] == 0))
    flags: list[str] = []
    if agent_name not in ("nop", "oracle"):   # the model-free check agents run no commands of their own
        from apex_data.grade.tamper import tamper_flags
        flags = list(tamper_flags({"messages": trajectory_messages(traj), "exit_reason": exit_reason}, "terminal"))
    if not consistent:
        flags.append("verifier_inconsistent")
    resolved = reward == 1.0
    ar = res.get("agent_result") or {}
    return TrialOutcome(
        trial_dir=str(d), task_name=(res.get("task_name") or "").split("/")[-1] or None, agent=agent_name, reward=reward,
        exception_type=exc_type, exception_message=exc_msg, handshake=shook, exit_reason=exit_reason, n_agent_steps=n_steps,
        ctrf=ctrf, reward_json_present=reward_json, verifier_consistent=consistent, tamper_flags=flags, resolved=resolved,
        r_pass=resolved and shook and consistent and not flags, agent_sec=_seconds(res.get("agent_execution")),
        verifier_sec=_seconds(res.get("verifier")), trial_sec=_seconds(res), input_tokens=ar.get("n_input_tokens"),
        output_tokens=ar.get("n_output_tokens"))


def grade_trial(trial_dir: Path | str, *, episode_id: str, task_id: str, graded_at: str = "") -> Verdict:
    """schema.Verdict of one Terminus-2 trial (r_pass as defined in the module docstring)."""
    o = read_trial(trial_dir)
    run = {"pass": o.resolved, "reward": o.reward, "handshake": o.handshake, "ctrf": o.ctrf,
           "log": str(Path(trial_dir) / "verifier" / "pytest.out"), "seconds": o.verifier_sec,
           "exception": o.exception_type}
    v = Verdict(episode_id=episode_id, task_id=task_id, r_pass=int(o.r_pass), runs=[run], tamper_flags=o.tamper_flags,
                verifier="harbor_tests", verifier_version=VERIFIER_VERSION, graded_at=graded_at,
                extra={"trial_dir": o.trial_dir, "exit_reason": o.exit_reason, "n_agent_steps": o.n_agent_steps,
                       "resolved": o.resolved, "agent_sec": o.agent_sec, "output_tokens": o.output_tokens})
    v.validate()
    return v
