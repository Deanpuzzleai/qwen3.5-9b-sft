"""One-turn candidates: sampling through vLLM's /v1/completions with token-id prompts, and parsing a completion into an
assistant turn (data plan v5, 4.4, "One rendering path" and "Existing parsers").

Sampling. The prompt is the state's token ids (apex_train.stage2.fit_prompt: the chat template with the generation prompt,
ending `<|im_start|>assistant\\n<think>\\n`), so sampling and DPO training see the same tokens, with no second
chat-template or tool-parser path. One request per state asks for n samples with the family's evaluation sampling
(rollout.endpoint.lock_inference), a seed from rollout.endpoint.sample_seed(state id, 0), max_tokens from the state, stop
at <|im_end|>, and special tokens kept in the text (so </think> and <tool_call> stay visible).

Parsing (`parse_completion`). The text is split at the first `</think>` (vLLM's qwen3 reasoning parser does the same):
  no </think>, cut by max_tokens   finished False, closed False: a runaway cut off before acting (rule R; the degenerate
                                   reasoning loops of the SFT quick check end here); message = the reasoning with empty
                                   content, a rejected side only (it needs apex_train.stage2 to accept a rejected turn
                                   without content or tool call; otherwise the pair is a counted validator drop)
  no </think>, stopped             invalid `unclosed_reasoning` (rule V); message as above, rejected side only
  </think>, cut by max_tokens      invalid `truncated_action`, no message (cut by the sampling budget: never a pair side)
  swe                              the part after </think> through render.filters.qwen3_parse_calls (vLLM's qwen3_coder
                                   parser), valid by mini-swe-agent 2.4.6's rule (parse_toolcall_actions: at least one
                                   call, every call `bash` with a `command`; it runs several calls in order, and the
                                   stage-1 SWE rows keep such turns) and every <tool_call> block parsed; content = the
                                   text before the first <tool_call> (text after the calls is dropped, as the parser does)
  terminal                         render.filters.terminus_parse (Harbor's strict Terminus-2 JSON parser): valid without
                                   a parse error (warnings are allowed, as in the stage-1 gate); content = the text
An invalid finished turn keeps the text after </think> as its content (the rejected side of rule V is what the model
wrote; empty when it wrote nothing). Invalid turns are never chosen (rules.label_turn: not valid, so not clean).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from apex_data.render.filters import qwen3_parse_calls, terminus_parse
from apex_data.rollout.endpoint import Endpoint, lock_inference, sample_seed
from apex_train.data_format import IM_END_ID

THINK_END = "</think>"


def _message(reasoning: str, content: str, commands: list[str] | None = None) -> dict[str, Any]:
    m: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning, "content": content}
    if commands:
        m["tool_calls"] = [{"type": "function", "function": {"name": "bash", "arguments": {"command": c}}} for c in commands]
    return m


def parse_completion(text: str, finish_reason: str | None, family: str) -> dict[str, Any]:
    """{reasoning, closed, finished, valid, invalid_reason, message} of one sampled completion (see the module doc)."""
    finished = finish_reason != "length"
    if THINK_END not in text:
        reasoning = text.strip()
        reason = None if not finished else "unclosed_reasoning"
        return {"reasoning": reasoning, "closed": False, "finished": finished, "valid": False, "invalid_reason": reason or "cut_off",
                "message": _message(reasoning, "") if reasoning else None}
    reasoning, rest = text.split(THINK_END, 1)
    reasoning, rest = reasoning.strip(), rest.strip()
    out = {"reasoning": reasoning, "closed": True, "finished": finished, "valid": False, "invalid_reason": None, "message": None}
    if not finished:
        out["invalid_reason"] = "truncated_action"
        return out
    if not reasoning:
        out["invalid_reason"] = "empty_reasoning"
        return out
    if family == "swe":
        calls = qwen3_parse_calls(rest)
        if not calls:
            out["invalid_reason"] = "no_tool_call"
        elif rest.count("<tool_call>") != len(calls):
            out["invalid_reason"] = "malformed_tool_call"
        elif any(c["name"] != "bash" or "command" not in c["arguments"] for c in calls):
            out["invalid_reason"] = "not_bash_command"
        else:
            content = rest.split("<tool_call>", 1)[0].strip()
            out.update(valid=True, message=_message(reasoning, content, [c["arguments"]["command"] for c in calls]))
            return out
    else:
        p = terminus_parse(rest)
        if not p.error:
            out.update(valid=True, message=_message(reasoning, rest))
            return out
        out["invalid_reason"] = "terminus_" + p.error.split(":")[0].lower().replace(" ", "_")[:40]
    out["message"] = _message(reasoning, rest)
    return out


def teacher_parsed(message: dict[str, Any], family: str) -> dict[str, Any]:
    """The teacher's turn in parse_completion's shape: SWE valid with at least one call, every call bash with a command
    (JSON-string arguments are parsed); terminal valid when its content parses. The message is normalised to the
    candidates' form."""
    reasoning = (message.get("reasoning_content") or "").strip()
    content = (message.get("content") or "").strip()
    out = {"reasoning": reasoning, "closed": True, "finished": True, "valid": False, "invalid_reason": None, "message": None}
    if not reasoning or THINK_END in reasoning:
        out["invalid_reason"] = "empty_reasoning" if not reasoning else "tag_in_reasoning"
        return out
    if family == "swe":
        commands = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if fn.get("name") != "bash" or not isinstance(args, dict) or "command" not in args:
                commands = []
                break
            commands.append(str(args["command"]))
        if not commands:
            out["invalid_reason"] = "teacher_not_bash_commands"
            return out
        out.update(valid=True, message=_message(reasoning, content, commands))
        return out
    p = terminus_parse(content)
    if p.error:
        out["invalid_reason"] = "teacher_terminus_parse"
        return out
    out.update(valid=True, message=_message(reasoning, content))
    return out


def request_body(model: str, prompt_ids: list[int], family: str, n: int, max_tokens: int, state_id: str) -> dict[str, Any]:
    """The /v1/completions request of one state."""
    return {"model": model, "prompt": prompt_ids, "n": n, "max_tokens": max_tokens, **lock_inference(family)["sampling"],
            "seed": sample_seed(state_id, 0), "stop_token_ids": [IM_END_ID], "skip_special_tokens": False,
            "spaces_between_special_tokens": False}


def complete(endpoint: Endpoint, body: dict[str, Any], *, timeout: float = 3600.0, retries: int = 3) -> dict[str, Any]:
    """POST /completions with retries on connection errors and 5xx; returns the response JSON."""
    url = endpoint.api_base.rstrip("/") + "/completions"
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {endpoint.api_key()}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator-given endpoint
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")
