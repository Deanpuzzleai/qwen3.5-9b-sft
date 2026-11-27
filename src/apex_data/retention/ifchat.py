"""Instruction-following and chat parts: nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 (docs/DATA_PLAN.md 3.4).

IF split (`data/instruction_following.jsonl`, GPT-OSS-120B per row `metadata.model`). Single-turn rows are the ones
seeded from allenai/tulu-3-sft-personas-instruction-following ([system, user, assistant]); the multi-turn synthetic rows
train only their last turn, whose prompt depends on the earlier turns, so they are not used. The rows carry no
machine-readable constraints: `prompts.constraints.extract_constraints` reads the verifiable ones from the prompt and a
row qualifies only if at least one is found and all pass `prompts.checkers` on the answer (the reasoning never counts).

Chat split (`data/chat.jsonl`, GLM-5 per row, best of several samples by a generative reward model upstream). The card
says to train only the last assistant turn; the lmsys- and WildChat-seeded rows ship their first prompt as null. Used:
lmarena-seeded rows (`metadata.seed_dataset` starts with "lmarena-ai/") with one user turn, a non-empty English prompt
(`common.is_english`), so the trained turn is the whole conversation. The topic label comes later (`topics`).

Both: non-empty reasoning and answer, no think tags inside them, reasoning short enough to possibly fit the 6K-token
cap (character pre-filter; the exact cap is the `think_length` gate). Funnels count every check in order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from apex_data.prompts.constraints import extract_constraints
from apex_data.prompts.checkers import check_all
from apex_data.retention.common import (
    CHARS_PER_TOKEN_CEILING, CONTROL_CHARS, PARTS, Funnel, candidate, has_think_tags, is_english, iter_jsonl_range,
    load_jsonl,
)

IF_STAGES = ("single_turn", "reasoning_present", "think_closed", "control_chars", "reasoning_chars", "has_constraints",
             "constraints_pass")
CHAT_STAGES = ("lmarena_seed", "single_turn", "prompt_present", "english_prompt", "reasoning_present", "think_closed",
               "control_chars", "reasoning_chars")


def _turns(msgs: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    roles = [m.get("role") for m in msgs]
    users = [m for m in msgs if m.get("role") == "user"]
    return roles, (users[0] if users else {}), (msgs[-1] if msgs else {})


def _if_chunk(path: str, start: int, end: int, revision: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    funnel = Funnel(IF_STAGES)
    out: list[dict[str, Any]] = []
    kinds: dict[str, int] = {}
    char_cap = PARTS["if"].think_cap * CHARS_PER_TOKEN_CEILING
    for offset, r in iter_jsonl_range(path, start, end):
        msgs = r.get("messages") or []
        roles, user, asst = _turns(msgs)
        prompt = user.get("content") or ""
        reasoning, answer = asst.get("reasoning_content") or "", asst.get("content") or ""
        state: dict[str, Any] = {}

        def extracted() -> bool:
            state["cs"], state["uncovered"] = extract_constraints(prompt)
            return bool(state["cs"])

        def passes() -> bool:
            ok, results = check_all(state["cs"], answer)
            state["results"] = results
            return ok

        failed = funnel.run([
            ("single_turn", lambda: [x for x in roles if x != "system"] == ["user", "assistant"] and not asst.get("tool_calls")),
            ("reasoning_present", lambda: bool(reasoning.strip()) and bool(answer.strip()) and bool(prompt.strip())),
            ("think_closed", lambda: not has_think_tags(reasoning, answer, prompt)),
            ("control_chars", lambda: not CONTROL_CHARS.search(prompt + reasoning + answer)),
            ("reasoning_chars", lambda: len(reasoning) <= char_cap),
            ("has_constraints", extracted),
            ("constraints_pass", passes),
        ])
        if "cs" in state and "results" in state:
            for c, res in zip(state["cs"], state["results"], strict=True):
                kinds[f"{c.kind}:{'pass' if res.ok else 'fail'}"] = kinds.get(f"{c.kind}:{'pass' if res.ok else 'fail'}", 0) + 1
        if failed:
            continue
        meta = r.get("metadata") or {}
        out.append(candidate("if", f"instruction_following/{r['uuid']}", prompt, answer, reasoning, hf_revision=revision,
                             generator=meta.get("model"), origin={"file": Path(path).name, "offset": offset},
                             seed_dataset=meta.get("seed_dataset"),
                             constraints=[c.as_dict() for c in state["cs"]], uncovered=state["uncovered"],
                             system_dropped=bool(roles and roles[0] == "system")))
    return out, funnel.to_dict(), kinds


def _chat_chunk(path: str, start: int, end: int, revision: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    funnel = Funnel(CHAT_STAGES)
    out: list[dict[str, Any]] = []
    seeds: dict[str, int] = {}
    char_cap = PARTS["chat"].think_cap * CHARS_PER_TOKEN_CEILING
    for offset, r in iter_jsonl_range(path, start, end):
        meta = r.get("metadata") or {}
        seed = str(meta.get("seed_dataset") or "")
        msgs = r.get("messages") or []
        roles, user, asst = _turns(msgs)
        prompt = user.get("content") or ""
        reasoning, answer = asst.get("reasoning_content") or "", asst.get("content") or ""
        failed = funnel.run([
            ("lmarena_seed", lambda: seed.startswith("lmarena-ai/")),
            ("single_turn", lambda: [x for x in roles if x != "system"] == ["user", "assistant"] and not asst.get("tool_calls")),
            ("prompt_present", lambda: bool(prompt.strip())),
            ("english_prompt", lambda: is_english(prompt)),
            ("reasoning_present", lambda: bool(reasoning.strip()) and bool(answer.strip())),
            ("think_closed", lambda: not has_think_tags(reasoning, answer, prompt)),
            ("control_chars", lambda: not CONTROL_CHARS.search(prompt + reasoning + answer)),
            ("reasoning_chars", lambda: len(reasoning) <= char_cap),
        ])
        if failed:
            continue
        seeds[seed] = seeds.get(seed, 0) + 1
        out.append(candidate("chat", f"chat/{r['uuid']}", prompt, answer, reasoning, hf_revision=revision,
                             generator=meta.get("model"), origin={"file": Path(path).name, "offset": offset},
                             seed_dataset=seed, system_dropped=bool(roles and roles[0] == "system")))
    return out, funnel.to_dict(), seeds



def load_if(raw_dir: Path, revision: str, workers: int = 16) -> tuple[list[dict[str, Any]], Funnel, dict[str, int]]:
    """IF candidates, the funnel, and per-kind pass/fail counts of the extracted constraints."""
    return load_jsonl(_if_chunk, raw_dir / "data" / "instruction_following.jsonl", IF_STAGES, revision, workers, 64)


def load_chat(raw_dir: Path, revision: str, workers: int = 16) -> tuple[list[dict[str, Any]], Funnel, dict[str, int]]:
    """Chat candidates, the funnel, and per-seed-dataset counts of the candidates."""
    return load_jsonl(_chat_chunk, raw_dir / "data" / "chat.jsonl", CHAT_STAGES, revision, workers, 256)
