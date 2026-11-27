"""Stage-2 states: held-out teacher episodes cut before one assistant turn.

Sources (never dev, never a stage-1 instance or task; the caller passes the stage-1 ids to exclude):
  swe       gate-passing Open-SWE-Traces trajectories of an SWE slice build: its evals/outcomes.jsonl (ok records) and
            evals/rows/*.parquet (`episode_json`: the episode rendered by teacher.render.build_episode under our harness,
            /testbed rewritten to /app), i.e. the stage-1 row format after every stage-1 screen and gate. Band: the
            outcome's `band`, else efficiency.swe_band over the instance's attempts in the run's index. One trajectory per
            instance; instances in a seeded order that takes the bands in turn (hard, medium, easy).
  swe (rollouts, optional) the first build's stage-2 rollouts of its SFT model on SWE-smith
            tasks in our mini-swe-agent harness (episodes.jsonl): histories with the model's own loops, for rules V, D, P.
            SWE-smith tasks have no teacher band, so these states take band `hard`, the most generous turn ceiling and no
            rule E. Dev tasks and dev repositories are excluded; one rollout per task, seeded. Their recorded turns are the
            first SFT model's, not a teacher's, so these states have no teacher fallback (`teacher` None).
  terminal  Kimi-K3 trials of ai-and/termgrade-trajectories, passing and failing (status scored, counted toward k, not
            flagged contaminated, every assistant turn parsed upstream). user #1 is re-rendered under our Harbor template
            (terminal.harness.match_first_prompt) and the turns are assembled as terminal.loaders.load_tgr_record does;
            that loader keeps passing trials only, so the assembly is repeated here without the pass filter. Screens:
            terminal.admission.screen_episode (D1, D4, identity, credentials); dev tasks excluded. Band:
            efficiency.terminal_band of the six-model mean pass rate (labels.jsonl). One trial per task, seeded, bands in
            turn.
Turns. Per episode, up to per_episode distinct turns among those whose state fits: one drawn among the turns right
after a file change (efficiency.writes on the previous turn's commands) or right after a repeated command
(efficiency.wasted_flags), when such a turn fits, and the rest drawn uniformly (`kind` records which). A terminal turn that answers Terminus-2's
task_complete confirmation prompt is a state only in the seeded 10 percent of episodes whose confirmation reply stage 1
may train (render.adapters.trains_confirmation), so confirmation states stay as rare as in stage 1. A state fits when its prompt, rendered by
apex_train.stage2.fit_prompt with the DPO settings of configs/stage2.yaml, is at most dpo.max_prompt_length tokens and
leaves dpo.max_length - prompt >= RUNAWAY_FACTOR x the band's turn ceiling + ACTION_MARGIN tokens for the turn: a
candidate cut off by that budget is then a clear runaway (rules.R), and every pair fits max_length.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from apex_data import efficiency, paths
from apex_data.pairs.rules import RUNAWAY_FACTOR
from apex_data.render.adapters import MINI_SWE_TOOLS, is_terminus_confirmation, trains_confirmation
from apex_data.rollout.endpoint import lock_inference

BANDS = ("hard", "medium", "easy")
ACTION_MARGIN = 1_024                # tokens of the action after the reasoning, inside the state's sampling budget
RETOKENIZE_MARGIN = 64               # a cut-off's text re-encoded by the trainer can be a few tokens longer than generated
PLACEHOLDER = {"role": "assistant", "reasoning_content": "placeholder", "content": "placeholder"}
STAGE2_CONFIG = paths.CONFIGS / "stage2.yaml"


def family_max_tokens(family: str) -> int:
    """Per-turn sampling cap: the family's lock max_tokens_per_turn (SWE 16,384, terminal 81,920), as in the harness, so
    reasoning loops run to the cap as they do in evaluation; states.sample_budget caps it by the pair's max_length."""
    return lock_inference(family)["max_tokens_per_turn"]


def sample_budget(state: Mapping[str, Any], max_length: int) -> int:
    """max_tokens of a state's samples: the family's lock cap, at most max_length - prompt - RETOKENIZE_MARGIN (every pair
    fits the trainer's max_length; a candidate cut there has reasoned >= RUNAWAY_FACTOR x the ceiling + ACTION_MARGIN -
    RETOKENIZE_MARGIN tokens)."""
    return min(family_max_tokens(state["family"]), max_length - state["n_prompt"] - RETOKENIZE_MARGIN)


def dpo_options(config: Path = STAGE2_CONFIG, *, on_too_long: str | None = None) -> Any:
    """apex_train.stage2.PairOptions from the config's dpo block (on_too_long overridable: states use `drop`)."""
    import yaml

    from apex_train.stage2 import PairOptions

    d = yaml.safe_load(Path(config).read_text())["dpo"]
    return PairOptions(thinking=bool(d["data"]["thinking"]), max_length=int(d["max_length"]),
                       max_prompt_length=int(d["max_prompt_length"]), on_too_long=on_too_long or d["on_too_long"],
                       min_tool_tokens=int(d["min_tool_tokens"]))


def prompt_ids(state_id: str, prompt: list[dict[str, Any]], tools: Any, tokenizer: Any, opts: Any) -> list[int]:
    """The state's token ids through stage2's own validator and fit_prompt (raises DataFormatError subclasses:
    SampleValidationError for an invalid history, PromptTooLongError above max_prompt_length with on_too_long drop)."""
    from apex_train.stage2 import PreferencePair, fit_prompt

    pair = PreferencePair.from_dict({"id": state_id, "prompt": prompt, "tools": tools, "chosen": PLACEHOLDER,
                                     "rejected": PLACEHOLDER}, default_id=state_id, thinking=opts.thinking)
    return fit_prompt(pair, tokenizer, opts)[1]


def turn_kinds(messages: Sequence[Mapping[str, Any]], family: str) -> dict[int, str]:
    """{assistant ordinal: "change" | "repeat"} for turns right after a turn that changed a file or repeated a command."""
    acts = efficiency.actions(messages, family)
    flags = efficiency.wasted_flags(acts)
    out: dict[int, str] = {}
    for a, wasted in zip(acts, flags, strict=True):
        if efficiency.writes(a.command, family=family):
            out.setdefault(a.turn + 1, "change")
        if wasted:
            out[a.turn + 1] = "repeat"
    return out


def episode_states(ep: Mapping[str, Any], family: str, band: str, tokenizer: Any, opts: Any, seed: int,
                   per_episode: int = 2, min_prompt: int = 0) -> tuple[list[dict[str, Any]], str | None]:
    """Up to per_episode states of one episode ({episode_id, task_id, messages, tools, source}), or ([], reason).
    min_prompt > 0 keeps only turns whose prompt has at least that many tokens (later turns)."""
    from apex_train.data_format import DataFormatError

    msgs = ep["messages"]
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    ceil = efficiency.LIMITS[family][band].turn_ceiling
    budget_min = RUNAWAY_FACTOR * ceil + ACTION_MARGIN
    limit = min(opts.max_prompt_length, opts.max_length - budget_min)
    cache: dict[int, int | None] = {}

    def n_prompt(t: int) -> int | None:
        if t not in cache:
            try:
                cache[t] = len(prompt_ids(f"{ep['episode_id']}@{t}", msgs[:asst[t]], ep.get("tools"), tokenizer, opts))
            except DataFormatError:
                cache[t] = None
        return cache[t]

    lo, hi = 0, len(asst) - 1                    # the prompt grows with the turn: binary search the last fitting turn
    if not asst or n_prompt(0) is None or n_prompt(0) > limit:
        return [], "no_fitting_turn"
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = n_prompt(mid)
        if n is not None and n <= limit:
            lo = mid
        else:
            hi = mid - 1
    first = 0
    if min_prompt:                               # the first turn whose prompt reaches min_prompt (binary search)
        if (n_prompt(lo) or 0) < min_prompt:
            return [], "below_min_prompt"
        a_, b_ = 0, lo
        while a_ < b_:
            mid = (a_ + b_) // 2
            if (n_prompt(mid) or 0) >= min_prompt:
                b_ = mid
            else:
                a_ = mid + 1
        first = a_

    def confirmation(t: int) -> bool:
        prev = msgs[asst[t] - 1]
        return family == "terminal" and prev.get("role") == "user" and is_terminus_confirmation(prev.get("content") or "")

    allowed = [t for t in range(first, lo + 1) if not confirmation(t) or trains_confirmation(ep["episode_id"])]
    if not allowed:
        return [], "no_fitting_turn"
    rng = random.Random(f"{seed}:states:{ep['episode_id']}")
    picks = [(rng.choice(allowed), "uniform")]
    special = [(t, k) for t, k in sorted(turn_kinds(msgs, family).items()) if t in allowed and t != picks[0][0]]
    if special and per_episode > 1:
        picks.append(rng.choice(special))
    rest = [t for t in allowed if t not in {t for t, _ in picks}]
    rng.shuffle(rest)
    picks += [(t, "uniform") for t in rest[:max(0, per_episode - len(picks))]]
    out = []
    for t, kind in sorted(picks[:per_episode]):
        n = n_prompt(t)
        if n is None or n > limit:
            continue
        out.append({"state_id": f"{ep['episode_id']}@{t}", "family": family, "band": band, "task_id": ep["task_id"],
                    "episode_id": ep["episode_id"], "turn": t, "kind": kind, "prompt": msgs[:asst[t]], "tools": ep.get("tools"),
                    "teacher": msgs[asst[t]] if ep.get("teacher_turns", True) else None, "n_prompt": n,
                    "max_tokens": min(family_max_tokens(family), opts.max_length - n),   # = sample_budget
                    "confirmation": confirmation(t),
                    "source": dict(ep.get("source") or {})})
    return out, None if out else "no_fitting_turn"


def screen_item(state: Mapping[str, Any]) -> dict[str, Any]:
    """The task of a state as a scripts/data/run_screens.py pool item (p1: D1, D2, D3, D4, dedup and lineage vs dev)."""
    src = state.get("source") or {}
    upstream = state["task_id"].split(":", 1)[1]
    if state["family"] == "swe":
        user = next((m.get("content") or "" for m in state["prompt"] if m.get("role") == "user"), "")
        m = re.search(r"<pr_description>(.*?)</pr_description>", user, re.S)
        return {"task_id": state["task_id"], "kind": "swe", "text": (m.group(1) if m else user).strip(), "repo": src.get("repo"),
                "upstream_id": upstream, "split": "train"}
    return {"task_id": state["task_id"], "kind": "terminal", "text": src.get("instruction") or state["prompt"][0]["content"],
            "upstream_id": upstream, "split": "train"}


def band_order(items: Iterable[Mapping[str, Any]], seed: int, tag: str) -> list[Mapping[str, Any]]:
    """Seeded order that takes the bands in turn (hard, medium, easy); a band that runs out is skipped."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for x in sorted(items, key=lambda x: x["key"]):
        by[x["band"]].append(x)
    for b, xs in by.items():
        random.Random(f"{seed}:{tag}:{b}").shuffle(xs)
    out, k = [], 0
    while any(by.values()):
        b = BANDS[k % len(BANDS)]
        if by.get(b):
            out.append(by[b].pop(0))
        k += 1
    return out


# ------------------------------------------------------------------------------------------------------------- SWE
def swe_candidates(run_dir: Path, exclude: set[str], seed: int, index: Path | None = None) -> list[dict[str, Any]]:
    """One gate-passing trajectory per held-out instance of an SWE slice build, in band_order:
    [{key (instance id), band, trajectory_id, language}]."""
    import pyarrow.parquet as pq

    ok: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run_dir / "evals" / "outcomes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r["instance_id"] not in exclude:
                ok[r["instance_id"]].append(r)
    bands: dict[str, str | None] = {}
    if any("band" not in rs[0] for rs in ok.values()):
        idx = pq.read_table(index or run_dir / "index.parquet", columns=["instance_id", "resolved"]).to_pylist()
        att: dict[str, list[int]] = defaultdict(list)
        for r in idx:
            att[r["instance_id"]].append(int(r["resolved"]))
        bands = {i: efficiency.swe_band(v) for i, v in att.items()}
    out = []
    for inst, rs in ok.items():
        band = rs[0].get("band") or bands.get(inst)
        if band in BANDS:
            r = random.Random(f"{seed}:swe_traj:{inst}").choice(sorted(rs, key=lambda r: r["trajectory_id"]))
            out.append({"key": inst, "band": band, "trajectory_id": r["trajectory_id"], "language": r.get("language"),
                        "repo": r.get("repo") or (r.get("provenance") or {}).get("repo")})
    return band_order(out, seed, "swe")


def swe_episodes(run_dir: Path, wanted: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """The wanted trajectories' episodes from the run's evals/rows/*.parquet (episode_json), in `wanted` order."""
    import pyarrow.parquet as pq

    pos = {w["trajectory_id"]: k for k, w in enumerate(wanted)}
    found: dict[str, dict[str, Any]] = {}
    for f in sorted((run_dir / "evals" / "rows").glob("*.parquet")):
        for r in pq.read_table(f, columns=["trajectory_id", "episode_json"]).to_pylist():
            if r["trajectory_id"] in pos and r["trajectory_id"] not in found:
                found[r["trajectory_id"]] = json.loads(r["episode_json"])
    for w in wanted:
        e = found.get(w["trajectory_id"])
        if e is None:
            continue
        tools = e.get("tools") or MINI_SWE_TOOLS
        yield {"episode_id": w["trajectory_id"], "task_id": f"oswt:{w['key']}", "messages": e["messages"], "tools": tools,
               "band": w["band"], "source": {"instance_id": w["key"], "trajectory_id": w["trajectory_id"],
                                             "language": w.get("language"), "repo": w.get("repo"), "run": run_dir.name}}


ROLLOUT_BAND = "hard"


def swesmith_repo(upstream_id: str) -> str:
    """'owner/name' (lowercase) of a SWE-smith instance id ('Owner__name.<commit>.<mutation>')."""
    return upstream_id.split(".", 1)[0].replace("__", "/").lower()


def swe_rollout_episodes(path: Path, exclude: set[str], dev_repos: set[str], seed: int
                         ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One mini-swe-agent rollout per held-out SWE-smith task (seeded), in a seeded order, and the drop counts."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            upstream = e["task_id"].split(":", 1)[1]
            if e.get("scaffold") != "mini_swe_toolcall":
                drops["not_mini_swe"] += 1
            elif upstream in exclude:
                drops["excluded_task"] += 1
            elif swesmith_repo(upstream) in dev_repos:
                drops["dev_repository"] += 1
            elif not e.get("messages"):
                drops["no_messages"] += 1
            else:
                by_task[e["task_id"]].append(e)
    out = []
    for tid in sorted(by_task):
        e = random.Random(f"{seed}:rollout:{tid}").choice(sorted(by_task[tid], key=lambda x: x["episode_id"]))
        upstream = tid.split(":", 1)[1]
        out.append({"episode_id": e["episode_id"], "task_id": tid, "messages": e["messages"], "tools": MINI_SWE_TOOLS,
                    "band": ROLLOUT_BAND, "teacher_turns": False, "source": {"kind": "swesmith_rollout", "task": upstream, "repo": swesmith_repo(upstream),
                                                     "exit_reason": e.get("exit_reason"), "run": path.parent.name}})
    random.Random(f"{seed}:rollout_order").shuffle(out)
    return out, dict(drops)


# -------------------------------------------------------------------------------------------------------- terminal
def terminal_bands(labels: Path) -> dict[str, str]:
    """{task: band} from the six-model mean pass rate (efficiency.terminal_band)."""
    rates: dict[str, list[float]] = defaultdict(list)
    for line in labels.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rates[r["task"]].append(float(r["pass_rate"]))
    return {t: efficiency.terminal_band(sum(v) / len(v)) for t, v in rates.items()}


def terminal_messages(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]] | None, str, str]:
    """(messages, instruction, drop reason) of one trial row: loaders.load_tgr_record's assembly without its pass filter."""
    from apex_data.terminal.harness import HarnessError, match_first_prompt

    msgs = row.get("messages") or []
    roles = [m.get("role") for m in msgs]
    if len(msgs) < 3 or roles[:2] != ["system", "user"] or any(x not in ("user", "assistant") for x in roles[2:]):
        return None, "", "unexpected_roles"
    if row.get("status") != "scored" or row.get("counts_toward_k") is False or row.get("contaminated"):
        return None, "", "not_scored_counted_clean"
    if any(m["role"] == "assistant" and not m.get("parsed") for m in msgs[2:]):
        return None, "", "upstream_unparsed_turn"
    try:
        fp = match_first_prompt(msgs[1].get("content") or "", system=msgs[0].get("content") or "")
    except HarnessError as exc:
        return None, "", exc.reason
    out: list[dict[str, Any]] = [{"role": "user", "content": fp.prompt}]
    for m in msgs[2:]:
        if m["role"] == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        else:
            out.append({"role": "assistant", "content": (m.get("content") or "").strip(),
                        "reasoning_content": (m.get("reasoning_content") or "").strip()})
    return out, fp.instruction, ""


def terminal_episodes(messages_path: Path, labels: Path, exclude: set[str], seed: int, model: str = "kimi",
                      trials_per_task: int = 1, exclude_episodes: set[str] | frozenset[str] = frozenset()
                      ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Up to trials_per_task screened trials per held-out banded task (seeded; trials in exclude_episodes, e.g. the ones
    an earlier state set used, are skipped), in band_order, and the drop counts."""
    import pyarrow.parquet as pq

    from apex_data.screens import d4_structural
    from apex_data.terminal import admission

    bands = terminal_bands(labels)
    pats = d4_structural.slug_patterns(d4_structural.load_tb_slugs())
    cols = ["task", "model", "trial", "messages", "reward", "status", "contaminated", "counts_toward_k"]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: dict[str, int] = defaultdict(int)
    for r in pq.read_table(messages_path, columns=cols).to_pylist():
        task = str(r["task"])
        if r["model"] != model:
            drops["other_model"] += 1
        elif task in exclude:
            drops["excluded_task"] += 1
        elif bands.get(task) not in BANDS:
            drops["no_band"] += 1
        elif f"tgr:{r['trial']}" in exclude_episodes:
            drops["excluded_trial"] += 1
        else:
            msgs, instruction, why = terminal_messages(r)
            if msgs is None:
                drops[why] += 1
                continue
            reason = admission.screen_drop_reason(admission.screen_episode(msgs, instruction, task, pats), False)
            if reason:
                drops[reason] += 1
                continue
            by_task[task].append({"episode_id": f"tgr:{r['trial']}", "task_id": f"tgr:{task}", "messages": msgs,
                                  "tools": None, "band": bands[task],
                                  "source": {"task": task, "trial": str(r["trial"]), "reward": r["reward"], "model": model,
                                             "instruction": instruction}})
    picked = []
    for t, eps in by_task.items():
        order = sorted(eps, key=lambda e: e["episode_id"])
        if trials_per_task == 1:
            order = [random.Random(f"{seed}:tgr_trial:{t}").choice(order)]
        else:
            random.Random(f"{seed}:tgr_trials:{t}").shuffle(order)
        picked += [{"key": f"{t}#{k}", "band": e["band"], "episode": e} for k, e in enumerate(order[:trials_per_task])]
    return [x["episode"] for x in band_order(picked, seed, "terminal")], dict(drops)
