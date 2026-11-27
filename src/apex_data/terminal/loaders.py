"""Loaders of the two terminal teacher corpora -> schema Episode, Task and Verdict dicts (docs/DATA_PLAN.md 3.3, WP4).

TermGrade (`tgr`, `ai-and/termgrade-trajectories`, one model's `trajectories_messages` parquet + `trials.parquet`):
  the teacher is a model key (`TGR_TEACHERS`: `deepseek` = DeepSeek-V4-Pro, the first build's; `kimi` = Kimi-K3, revision
  5's; `qwen` = Qwen3.6-27B, `gemma_think` = Gemma-4-31B-it with reasoning on and `glm` =
  GLM-5.2 for the revision-5 supply probe). Kept for loading: that model, `reward == 1` (equal in both files), status `scored`,
  counted toward k (not a beyond-k retry), not flagged contaminated, messages present, every assistant turn `parsed`
  (upstream flag: the harness could parse the JSON; an unparsed turn drops the episode, never spliced). user #1 = the `system` message + the first `user` message, re-rendered under our
  Harbor template (harness.match_first_prompt); `reasoning_content` is the stored reasoning. Label: `verifier`.
TerminalHorizon-3K (`th3k`, one JSONL line per episode, `conversations` of {role, content}): user #1 kept when it
  equals our template (all 2,852 do), `<think>...</think>` split into `reasoning_content` (harness.split_think).
  Episode index = the 0-based line number (the only upstream id; `decontam_exclusions.json` uses it). Label:
  `authors_filter+handshake`; the verdict's r_pass is 1 iff the double task_complete handshake closes the episode.

Both: no system message, observations as user text exactly as stored (already limited to 10,000 bytes by the
harness that produced them: `extra.observation_form = "harness"`), exit_reason `task_complete` when the handshake is
present (else `unknown`, which the exit gate drops). Loader drops are returned as records, never raised.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_data.render.adapters import is_terminus_confirmation
from apex_data.render.filters import TERMINUS_PARSE_ERROR_PREFIX, terminus_parse
from apex_data.schema import Episode, Task, task_id as make_task_id
from apex_data.terminal.harness import HarnessError, match_first_prompt, observation_kind, split_think

TH3K_SOURCE, TGR_SOURCE = "th3k", "tgr"
TH3K_REPO = "shunzou05/TerminalHorizon-3K"
TH3K_ENV_REPO = "shunzou05/TerminalHorizon-Environment"
TGR_REPO = "ai-and/termgrade-trajectories"
DATASET_LICENSE = "apache-2.0"                     # both dataset cards
TH3K_GENERATOR = "deepseek-ai/DeepSeek-V4-Pro-0813"   # TerminalHorizon-3K card: "collected with DeepSeek-V4-Pro-0813"
TGR_MODEL_KEY = "deepseek"                         # TermGrade's short model key (labels.jsonl, trials.parquet)
POLICY = "deepseek-v4-pro"


@dataclass(frozen=True)
class TgrTeacher:
    """A TermGrade model as a teacher: the episode `policy`, the generator recorded when a row has no model_id, and the
    weights licence (docs/DATA_PLAN.md 3.1)."""

    policy: str
    generator: str
    licence: str


# dataset card, "Which models these are, exactly" (gemma's non-thinking trials carry no reasoning: never a teacher)
TGR_TEACHERS = {"deepseek": TgrTeacher(POLICY, "deepseek-ai/DeepSeek-V4-Pro", "MIT"),
                "kimi": TgrTeacher("kimi-k3", "moonshotai/Kimi-K3", "Kimi K3 License (MIT-style)"),
                "qwen": TgrTeacher("qwen3.6-27b", "Qwen/Qwen3.6-27B", "Apache-2.0"),
                "gemma_think": TgrTeacher("gemma-4-31b-it-think", "google/gemma-4-31B-it", "Apache-2.0"),
                "glm": TgrTeacher("glm-5.2", "zai-org/GLM-5.2", "MIT")}
UNLINKED = "unlinked"


@dataclass
class Loaded:
    """One upstream record: the schema dicts when it loaded, else `drop` (reason) and `detail`. `info` holds the
    facts admission and selection read (handshake, parse flags, task text, sizes) for loaded and dropped records."""

    source: str
    upstream_id: str
    episode: dict[str, Any] | None = None
    task: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    drop: str | None = None
    detail: str = ""


# ------------------------------------------------------------------------------------------------- shared helpers
def handshake(messages: list[dict[str, Any]]) -> bool:
    """The double task_complete handshake closes the episode (the same rule as filters.gate_exit): the last two
    assistant turns both set task_complete and parse, with Harbor's confirmation message between them."""
    msgs = list(messages)
    while msgs and msgs[-1].get("role") != "assistant":
        msgs.pop()
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if len(idx) < 2 or idx[-1] - idx[-2] != 2 or not is_terminus_confirmation(msgs[idx[-1] - 1].get("content") or ""):
        return False
    for i in idx[-2:]:
        p = terminus_parse(msgs[i].get("content") or "")
        if p.error or not p.task_complete:
            return False
    return True


def turn_facts(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse-cleanliness, reasoning presence and observation shapes of a harness-shaped transcript."""
    asst = [m for m in messages if m.get("role") == "assistant"]
    users = [m.get("content") or "" for m in messages[1:] if m.get("role") == "user"]
    parse_errors = [terminus_parse(m.get("content") or "").error for m in asst]
    kinds = Counter(observation_kind(u) for u in users)
    return {
        "n_assistant": len(asst),
        "strict_parse_failures": sum(bool(e) for e in parse_errors),
        "parse_error_turns": sum(u.startswith(TERMINUS_PARSE_ERROR_PREFIX) for u in users),
        "turns_without_reasoning": sum(not (m.get("reasoning_content") or "").strip() for m in asst),
        "observation_kinds": dict(kinds),
        "assistant_chars": sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in asst),
        "total_chars": sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in messages),
    }


def _task(source: str, upstream_id: str, revision: str | None, provenance: dict[str, Any], **extra: Any) -> dict[str, Any]:
    t = Task(task_id=make_task_id(source, upstream_id), source=source, family="terminal", language="bash", repo=None,
             upstream_id=upstream_id, hf_revision=revision,
             licence={"dataset": DATASET_LICENSE, "repo_spdx": None, "status": "ok"},
             extra={"provenance": provenance, **extra})
    t.validate()
    return {k: getattr(t, k) for k in Task.__dataclass_fields__}


def _episode(episode_id: str, tid: str, sample_index: int, messages: list[dict[str, Any]], *, exit_reason: str,
             output_tokens: int, wall_sec: float, created_at: str, extra: dict[str, Any],
             policy: str = POLICY) -> dict[str, Any]:
    e = Episode(episode_id=episode_id, task_id=tid, round="teacher", policy=policy, sample_index=sample_index,
                scaffold="terminus2", messages=messages, tools=None, exit_reason=exit_reason,
                steps=sum(m["role"] == "assistant" for m in messages), output_tokens=output_tokens, wall_sec=wall_sec,
                patch=None, request_log=None, created_at=created_at, extra={"observation_form": "harness", **extra})
    e.validate()
    return {k: getattr(e, k) for k in Episode.__dataclass_fields__}


def _verdict(episode_id: str, tid: str, r_pass: int, verifier: str, version: str) -> dict[str, Any]:
    return {"episode_id": episode_id, "task_id": tid, "r_pass": r_pass, "runs": [], "tamper_flags": [],
            "verifier": verifier, "verifier_version": version, "label_source": "upstream"}


# ------------------------------------------------------------------------------------------------- TerminalHorizon
def th3k_episode_id(env: str, index: int) -> str:
    return f"{TH3K_SOURCE}:{env}#{index}"


def read_th3k(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line index, record) per non-empty JSONL line; the index counts every line, so it is the upstream row number."""
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def load_th3k_record(index: int, record: dict[str, Any], *, revision: str | None, env: str | None,
                     category: str | None, category_group: str | None, subcategory: str | None) -> Loaded:
    """One TerminalHorizon line -> Loaded. `env`, `category`, `category_group`, `subcategory` come from linking."""
    env_id = env or f"{UNLINKED}-{index:04d}"
    out = Loaded(TH3K_SOURCE, str(index), info={"index": index, "env": env, "category": category,
                                                "category_group": category_group, "subcategory": subcategory})
    conv = record.get("conversations") or []
    roles = [c.get("role") for c in conv]
    if not conv or roles[0] != "user" or any(r not in ("user", "assistant") for r in roles):
        out.drop, out.detail = "unexpected_roles", ",".join(map(str, roles[:4]))
        return out
    try:
        fp = match_first_prompt(conv[0].get("content") or "")
    except HarnessError as exc:
        out.drop, out.detail = exc.reason, str(exc)
        return out
    out.info.update(first_prompt_status=fp.status, first_prompt_differences=fp.differences, instruction=fp.instruction,
                    initial_state_kind=observation_kind(fp.terminal_state))
    messages: list[dict[str, Any]] = [{"role": "user", "content": fp.prompt}]
    think_errors = 0
    for c in conv[1:]:
        text = c.get("content") or ""
        if c["role"] == "user":
            messages.append({"role": "user", "content": text})
            continue
        try:
            reasoning, content = split_think(text)
        except HarnessError:
            think_errors += 1
            reasoning, content = "", text
        messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
    out.info.update(turn_facts(messages), think_tag_errors=think_errors, handshake=handshake(messages))
    if think_errors:
        out.drop, out.detail = "think_tags", f"{think_errors} assistant turns with more than one think block"
        return out
    tid = make_task_id(TH3K_SOURCE, env_id)
    eid = th3k_episode_id(env_id, index)
    hs = out.info["handshake"]
    out.episode = _episode(eid, tid, index, messages, exit_reason="task_complete" if hs else "unknown", output_tokens=0,
                           wall_sec=0.0, created_at="", extra={"upstream_index": index, "environment": env})
    out.task = _task(TH3K_SOURCE, env_id, revision, {
        "hf_repo": TH3K_REPO, "hf_revision": revision, "generator": TH3K_GENERATOR,
        "label_provenance": "authors_filter+handshake", "dataset_license": DATASET_LICENSE,
        "category": category, "category_group": category_group}, environment=env, subcategory=subcategory)
    out.verdict = _verdict(eid, tid, int(hs), "authors_filter+handshake", "TerminalHorizon-3K card (filtered for training)")
    return out


# ------------------------------------------------------------------------------------------------------ TermGrade
TGR_COLUMNS = ("task", "model", "trial", "messages", "reward", "status", "exceeded_cap", "contaminated",
               "counts_toward_k", "model_id", "output_tokens", "duration_s", "n_episodes")


def tgr_episode_id(trial: str) -> str:
    return f"{TGR_SOURCE}:{trial}"


def load_tgr_record(r: dict[str, Any], *, revision: str | None, trial_reward: float | None,
                    started_at: str | None = None, model_key: str = TGR_MODEL_KEY) -> Loaded:
    """One row of a trajectories_messages parquet -> Loaded, with `model_key` (TGR_TEACHERS) as the teacher.
    `trial_reward` is the reward of the same (task, model, trial) in trials.parquet (None when absent)."""
    teacher = TGR_TEACHERS[model_key]
    task, trial = str(r["task"]), str(r["trial"])
    out = Loaded(TGR_SOURCE, trial, info={"task": task, "trial": trial, "reward": r.get("reward"),
                                          "trial_reward": trial_reward})
    msgs = r.get("messages") or []
    if not msgs:
        out.drop = "no_messages"
        return out
    roles = [m.get("role") for m in msgs]
    if len(msgs) < 3 or roles[0] != "system" or roles[1] != "user" or any(x not in ("user", "assistant") for x in roles[2:]):
        out.drop, out.detail = "unexpected_roles", ",".join(map(str, roles[:4]))
        return out
    try:
        fp = match_first_prompt(msgs[1].get("content") or "", system=msgs[0].get("content") or "")
    except HarnessError as exc:
        out.drop, out.detail = exc.reason, str(exc)
        return out
    # the instruction is recorded for every trial (dev-task texts feed the train-vs-dev near-duplicate screen)
    out.info.update(first_prompt_status=fp.status, first_prompt_differences=fp.differences, instruction=fp.instruction,
                    initial_state_kind=observation_kind(fp.terminal_state))
    checks = (
        ("not_teacher_model", r.get("model") != model_key, str(r.get("model"))),
        ("reward_not_1", r.get("reward") != 1.0, f"reward={r.get('reward')}"),
        ("reward_mismatch_trials", trial_reward is None or trial_reward != r.get("reward"),
         f"messages {r.get('reward')} vs trials {trial_reward}"),
        ("status_not_scored", r.get("status") != "scored", str(r.get("status"))),
        ("not_counted_toward_k", r.get("counts_toward_k") is False, ""),
        ("contaminated_flag", bool(r.get("contaminated")), ""),
    )
    for reason, failed, detail in checks:
        if failed:
            out.drop, out.detail = reason, detail
            return out
    messages: list[dict[str, Any]] = [{"role": "user", "content": fp.prompt}]
    unparsed = 0
    for m in msgs[2:]:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m.get("content") or ""})
        else:
            unparsed += not m.get("parsed")
            messages.append({"role": "assistant", "content": (m.get("content") or "").strip(),
                             "reasoning_content": (m.get("reasoning_content") or "").strip()})
    out.info.update(turn_facts(messages), upstream_unparsed_turns=unparsed, handshake=handshake(messages))
    if unparsed:
        out.drop, out.detail = "upstream_unparsed_turn", f"{unparsed} assistant turns with parsed == False"
        return out
    tid = make_task_id(TGR_SOURCE, task)
    eid = tgr_episode_id(trial)
    hs = out.info["handshake"]
    out.episode = _episode(eid, tid, 0, messages, exit_reason="task_complete" if hs else "unknown",
                           output_tokens=int(r.get("output_tokens") or 0), wall_sec=float(r.get("duration_s") or 0.0),
                           created_at=started_at or "", extra={"upstream_trial": trial}, policy=teacher.policy)
    out.task = _task(TGR_SOURCE, task, revision, {
        "hf_repo": TGR_REPO, "hf_revision": revision, "generator": str(r.get("model_id") or teacher.generator),
        "label_provenance": "verifier", "dataset_license": DATASET_LICENSE})
    out.verdict = _verdict(eid, tid, 1, "harbor_tests", "termgrade-trajectories reward (tests/test.sh)")
    return out


def read_tgr(messages_path: Path, trials_path: Path) -> Iterator[tuple[dict[str, Any], float | None, str | None]]:
    """(row, trials.parquet reward, started_at) for every row of one model's messages parquet."""
    import pyarrow.parquet as pq

    trials = pq.read_table(trials_path, columns=["task", "model", "trial", "reward", "started_at"]).to_pylist()
    by_key = {(t["task"], t["model"], t["trial"]): t for t in trials}
    pf = pq.ParquetFile(messages_path)
    cols = [c for c in TGR_COLUMNS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        for r in batch.to_pylist():
            t = by_key.get((r["task"], r["model"], r["trial"]))
            yield r, (None if t is None else t["reward"]), (None if t is None else t["started_at"])


def all_six_solve(labels_path: Path, models: tuple[str, ...] = ("gemma", "gemma_think", "glm", "kimi", "qwen", "deepseek")
                  ) -> tuple[set[str], int]:
    """TermGrade tasks every labelled model solves at every counted attempt (labels.jsonl pass_rate == 1.0 for all six;
    the rule of apex_data.pools.termgrade.teacher_coverage), and the number of labelled tasks."""
    per: dict[str, dict[str, float]] = {}
    with Path(labels_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                per.setdefault(r["task"], {})[r["model"]] = float(r["pass_rate"])
    solved = {t for t, lab in per.items() if set(lab) >= set(models) and all(lab[m] == 1.0 for m in models)}
    return solved, len(per)
