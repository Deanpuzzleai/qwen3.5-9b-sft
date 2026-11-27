"""M2.2 row checks of the terminal slice (docs/EXECUTION_PLAN.md M2.2, DATA_PLAN.md WP4 acceptance test).

`check_row` decodes one rendered per-turn row and verifies, against the episode turn it trains:
  render       the row has exactly one trained span, ending with <|im_end|>; the prompt ends with the generation opener
               `<|im_start|>assistant\\n<think>\\n`; the history holds no think block (the template drops earlier
               reasoning, which is what Terminus-2 replays); no system turn; user #1 is our Harbor first prompt
  reasoning    the trained span starts with the turn's reasoning (non-empty) and `\\n</think>\\n\\n`
  round trip   splitting the trained text at the first `</think>` (vLLM's qwen3 reasoning parser) gives back the
               reasoning and a content (both in NFC: the tokenizer's normaliser applies NFC on encode) that Harbor's TerminusJSONPlainParser parses without error into the same
               commands, durations and task_complete flag as the source turn; checked with the port in
               render/filters.py and, when `.venv/harbor` is present, with Harbor's own parser module (strict path
               `_try_parse_response`, no auto-fixes)
`render_excerpt` gives the short human-readable record of the 20-row render check.
"""
from __future__ import annotations

import importlib.util
import unicodedata
from functools import lru_cache
from typing import Any

from apex_data.render.filters import terminus_parse
from apex_data.terminal.harness import HARBOR_PARSER_FILE, split_first_prompt
from apex_train.data_format import IM_END_TOKEN

GEN_OPENER = "<|im_start|>assistant\n<think>\n"
THINK_CLOSE_SEP = "\n</think>\n\n"


@lru_cache(maxsize=1)
def harbor_parser() -> Any | None:
    """Harbor 0.24.0's TerminusJSONPlainParser loaded from its file (pure standard library), or None if absent."""
    if not HARBOR_PARSER_FILE.exists():
        return None
    spec = importlib.util.spec_from_file_location("harbor_terminus_json_plain_parser", HARBOR_PARSER_FILE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.TerminusJSONPlainParser()


def _spans(mask: list[int]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def nfc_changed(turn: dict[str, Any]) -> bool:
    """The turn's text is not in NFC, so its trained tokens are the NFC form (informational count)."""
    return any(nfc(t) != t for t in ((turn.get("reasoning_content") or ""), (turn.get("content") or "")))


def parse_signature(content: str) -> tuple[str, tuple[tuple[str, float], ...], bool]:
    p = terminus_parse(content)
    return p.error, tuple(p.commands), p.task_complete


def harbor_signature(content: str) -> tuple[str, tuple[tuple[str, float], ...], bool] | None:
    parser = harbor_parser()
    if parser is None:
        return None
    r = parser._try_parse_response(content)      # strict path (parse_response would add auto-fixes)
    return r.error, tuple((c.keystrokes, float(c.duration)) for c in r.commands), bool(r.is_task_complete)


def decode_row(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str]:
    """(prompt text, trained text) of a single-span per-turn row; raises ValueError for any other mask shape."""
    ids, mask = list(row["input_ids"]), list(row["assistant_masks"])
    spans = _spans(mask)
    if len(spans) != 1:
        raise ValueError(f"expected one trained span, got {len(spans)}")
    a, b = spans[0]
    if tokenizer.decode(ids[b:]) not in ("", "\n"):          # the template's newline after <|im_end|> is prompt
        raise ValueError(f"tokens after the trained span: {tokenizer.decode(ids[b:])[:40]!r}")
    return tokenizer.decode(ids[:a]), tokenizer.decode(ids[a:b])


def check_row(tokenizer: Any, row: dict[str, Any], messages: list[dict[str, Any]], *, harbor: bool = True) -> list[str]:
    """Problems of one row (empty list = clean). `messages` = the episode's harness-shaped transcript."""
    problems: list[str] = []
    try:
        prompt, trained = decode_row(tokenizer, row)
    except ValueError:
        return ["mask_shape"]
    turn = messages[row["turn_index"]]
    if turn.get("role") != "assistant":
        return ["turn_index_not_assistant"]
    # the Qwen3.5 tokenizer normalises text to NFC on encode (tokenizer.json `normalizer`), at training and at serving
    reasoning = nfc((turn.get("reasoning_content") or "").strip())
    content = nfc((turn.get("content") or "").strip())
    if not prompt.endswith(GEN_OPENER):
        problems.append("render_no_generation_opener")
    history = prompt[:-len(GEN_OPENER)] if prompt.endswith(GEN_OPENER) else prompt
    expected_think = sum((m.get("content") or "").count("<think>") for m in messages[:row["turn_index"]])
    if history.count("<think>") != expected_think or history.count("</think>") != sum(
            (m.get("content") or "").count("</think>") for m in messages[:row["turn_index"]]):
        problems.append("render_think_block_in_history")
    if "<|im_start|>system" in history:
        problems.append("render_system_turn")
    if not history.startswith("<|im_start|>user\n") or split_first_prompt(messages[0]["content"]) is None:
        problems.append("render_user_1_not_harbor_prompt")
    if not trained.endswith(IM_END_TOKEN):
        problems.append("render_span_without_im_end")
    if not reasoning:
        problems.append("reasoning_empty")
    if not trained.startswith(reasoning + THINK_CLOSE_SEP):
        problems.append("reasoning_boundary")
    # round trip: what vLLM's reasoning parser and Harbor's parser recover from the trained text
    body = trained.removesuffix(IM_END_TOKEN)
    if "</think>" not in body:
        return problems + ["roundtrip_no_think_close"]
    got_reasoning, got_content = body.split("</think>", 1)
    if got_reasoning.strip() != reasoning:
        problems.append("roundtrip_reasoning")
    if got_content.strip() != content:
        problems.append("roundtrip_content")
    want = parse_signature(content)
    got = parse_signature(got_content.strip())
    if got[0] or got != want:
        problems.append("roundtrip_port_parse")
    if harbor:
        h_got, h_want = harbor_signature(got_content.strip()), harbor_signature(content)
        if h_got is not None and (h_got[0] or h_got != h_want or h_got[1:] != got[1:]):
            problems.append("roundtrip_harbor_parse")
    return problems


def render_excerpt(tokenizer: Any, row: dict[str, Any], messages: list[dict[str, Any]], n: int = 160) -> dict[str, Any]:
    """Short record for the 20-row render check (corpus text only)."""
    try:
        prompt, trained = decode_row(tokenizer, row)
    except ValueError as exc:
        return {"row_key": row.get("row_key"), "n_tokens": row["n_tokens"], "problems": [f"mask_shape: {exc}"],
                "history_assistant_turns": 0, "history_think_blocks": 0, "trained_reasoning_chars": 0}
    history = prompt[:-len(GEN_OPENER)] if prompt.endswith(GEN_OPENER) else prompt
    return {"row_key": row.get("row_key"), "n_tokens": row["n_tokens"], "n_assistant_tokens": row["n_assistant_tokens"],
            "turn_index": row["turn_index"],
            "history_assistant_turns": history.count("<|im_start|>assistant\n"),
            "history_think_blocks": history.count("<think>"),
            "prompt_tail": prompt[-n:], "trained_head": trained[:n], "trained_tail": trained[-n:],
            "trained_reasoning_chars": len((messages[row["turn_index"]].get("reasoning_content") or "").strip()),
            "problems": check_row(tokenizer, row, messages)}
