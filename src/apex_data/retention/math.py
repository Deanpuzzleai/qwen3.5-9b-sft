"""Math part: nvidia/Nemotron-SFT-Math-v4 AoPS chain-of-thought rows (docs/DATA_PLAN.md section 3.4).

Rows. The card describes AoPS-derived (CC-BY-4.0) and Math StackExchange (CC-BY-SA-4.0) problems, each with a
chain-of-thought (`subset == "cot"`) or tool-integrated (`"tir"`) DeepSeek-V4-Pro (High) solution whose final answer was
verified against the Nemotron-Math-v2 reference upstream. The AoPS chain-of-thought rows (`source == "AoPS"`,
`subset == "cot"`) all sit in shards 0 to 2 at the pinned revision (data/manifests/raw_sources.json), which are the only
shards fetched.

Checks (counted in order): AoPS chain-of-thought; per-row licence cc-by-4.0; conversation shape [user, assistant] with no
tools or tool calls; non-empty reasoning; no think tags inside reasoning or answer; a \\boxed{} final answer in the
answer; reasoning short enough to possibly fit the 10K-token cap (character pre-filter; the exact cap is the
`think_length` gate). The upstream verification is the label; our own normalised comparison of the boxed answer with
`expected_answer` (`ref_match`: exact, numeric or unmatched) is recorded for the data card, not used to select, because
a string or numeric matcher without a CAS would reject equivalent symbolic answers and bias the slice towards numbers.
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from apex_data.retention.common import (
    CHARS_PER_TOKEN_CEILING, PARTS, Funnel, boxed_contents, candidate, has_think_tags, parallel_map,
)

STAGES = ("aops_cot", "license_cc_by", "shape", "reasoning_present", "think_closed", "boxed_answer", "reasoning_chars")
COLUMNS = ["uuid", "problem", "expected_answer", "messages", "tools", "source", "license", "subset"]

_STRIP = [(re.compile(p), r) for p, r in [
    (r"\\\(|\\\)|\\\[|\\\]|\\displaystyle|\\[bB]ig[lr]?", ""),
    (r"\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad|\$", ""), (r"\\[dt]frac", r"\\frac"), (r"\\text\{([^{}]*)\}", r"\1"),
    (r"([_^])\{(\w)\}", r"\1\2"),
    (r"\\mathrm\{([^{}]*)\}", r"\1"), (r"\\mathbf\{([^{}]*)\}", r"\1"), (r"\^\{?\\circ\}?|°", ""), (r"\s+", ""),
]]


def normalise_answer(ans: str) -> str:
    s = (ans or "").strip()
    for rx, rep in _STRIP:
        s = rx.sub(rep, s)
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)            # "x=5" -> "5"
    return s.rstrip(".").lower()


def _number(s: str) -> Fraction | float | None:
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}|(-?\d+)/(-?\d+)", s)
    if m:
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return Fraction(int(a), int(b)) if int(b) else None
    m = re.fullmatch(r"-\\frac\{(\d+)\}\{(\d+)\}", s)
    if m and int(m.group(2)):
        return -Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return float(s.replace("{,}", "").replace(",", "")) if re.fullmatch(r"-?[\d,]*\.?\d+(?:e-?\d+)?", s) else None
    except ValueError:
        return None


def reference_match(boxed: str, expected: str) -> str:
    """exact | numeric | unmatched: our check of the final boxed answer against the upstream reference."""
    a, b = normalise_answer(boxed), normalise_answer(expected)
    if a and a == b:
        return "exact"
    x, y = _number(a), _number(b)
    if x is not None and y is not None and abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(y))):
        return "numeric"
    return "unmatched"


def _load_row_group(path: str, rg: int, revision: str, think_cap: int | None = None
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    spec = PARTS["math"]
    funnel = Funnel(STAGES)
    out: list[dict[str, Any]] = []
    char_cap = (think_cap or spec.think_cap) * CHARS_PER_TOKEN_CEILING
    table = pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS)
    for r in table.to_pylist():
        msgs = r["messages"] or []
        roles = [m.get("role") for m in msgs]
        asst = msgs[-1] if msgs else {}
        reasoning, answer = asst.get("reasoning_content") or "", asst.get("content") or ""
        boxed = boxed_contents(answer)
        failed = funnel.run([
            ("aops_cot", lambda: r["source"] == "AoPS" and (r["subset"] or "").lower() == "cot"),
            ("license_cc_by", lambda: (r["license"] or "").lower() == "cc-by-4.0"),
            ("shape", lambda: roles == ["user", "assistant"] and not r["tools"] and not asst.get("tool_calls")),
            ("reasoning_present", lambda: bool(reasoning.strip()) and bool(answer.strip())),
            ("think_closed", lambda: not has_think_tags(reasoning, answer, msgs[0].get("content"))),
            ("boxed_answer", lambda: bool(boxed) and bool(boxed[-1])),
            ("reasoning_chars", lambda: len(reasoning) <= char_cap),
        ])
        if failed:
            continue
        out.append(candidate("math", r["uuid"], msgs[0]["content"], answer, reasoning, hf_revision=revision,
                             row_license=r["license"], origin={"file": Path(path).name, "row_group": rg},
                             expected_answer=r["expected_answer"], boxed=boxed[-1],
                             ref_match=reference_match(boxed[-1], r["expected_answer"] or "")))
    return out, funnel.to_dict()


def load(raw_dir: Path, revision: str, workers: int = 16, think_cap: int | None = None
         ) -> tuple[list[dict[str, Any]], Funnel]:
    """Candidates from every parquet shard under raw_dir/data (one task per row group). `think_cap` (reasoning tokens)
    sets the character pre-filter; the default is the part's 10K cap (the convergence replay passes 24K)."""
    import pyarrow.parquet as pq

    files = sorted((raw_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Math-v4 shards under {raw_dir / 'data'}")
    args = [(str(f), rg, revision, think_cap) for f in files for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]
    funnel = Funnel(STAGES)
    out: list[dict[str, Any]] = []
    for cands, fd in parallel_map(_load_row_group, args, workers):
        out.extend(cands)
        funnel.merge(Funnel.from_dict(fd))
    return out, funnel
