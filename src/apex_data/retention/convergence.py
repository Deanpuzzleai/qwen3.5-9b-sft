"""Convergence replay for stage 2 (docs/DATA_PLAN.md section 11, "Retention replay").

Why. On our own dev set (results/dev/base/dev_base_full_20261012_1010) Base fails the AIME items mostly by budget
exhaustion (29 of 30 failures end without an answer within the 32K budget) and the STEM QA slice the same way (89 of
125 failures). The stage-2 replay therefore carries rows that reason at length on hard problems and then *commit*: a
short visible final section that states one answer, right after the reasoning. The set replaces the math and science
share of the stage-2 replay; the replay reserve's instruction-following and chat rows are kept verbatim next to it
(scripts/data/build_retention_slice.py --convergence-replay).

Parts (same sources, loaders, checks, screens and gates as the retention pool; rows render through data_format):
  math_hard    Nemotron-SFT-Math-v4 AoPS chain-of-thought, reasoning 10K < r <= 24K tokens (above the SFT pool's 10K
               math cap), the final boxed answer matching the upstream reference by our normalised check (exact or
               numeric: AIME-style integer answers), one committed answer, and a short final section
  math_medium  the same source and rules, reasoning 4K < r <= 10K tokens
  sci_mcq      Nemotron-Science-v1 MCQ, reasoning <= 10K tokens, the pool's letter checks, and a visible answer that
               *ends* with the chosen letter in an explicit final-answer form (the dev QA slice is graded by the HLE
               judge, which extracts the final answer; its response format asks for an "Answer: X" line)
Shares of the parts' loss tokens: CVG_SHARES (hard math the largest). The visible answer of every part is at most
FINAL_SECTION_CAP tokens, so the answer line comes right after </think>.

Profiles (PROFILES; build_retention_slice.py --convergence-profile): `stage2_replay` is the build above (the default);
`hard_math_v5` is the hard-math part of the revision-5 retention slice (data/manifests/v5/DATA_CARD_v5.md): math only,
reasoning 10,240 < r <= 12,288 tokens (the floor sits just above the pool's 10K math cap; the ceiling is Base's median
output on solved dev QA items), the same reference match and commitment rules, a final section of at most 600 tokens
(decision D6), and no reserve rows.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from apex_data.retention.common import CHARS_PER_TOKEN_CEILING, THINK_CAP_LONG, apportion, boxed_contents
from apex_data.retention.math import normalise_answer
from apex_data.retention.science import _FINAL

THINK_CAP_HARD = 24_576          # reasoning tokens of the hard math part (upper bound)
MEDIUM_FLOOR = 4_096             # medium math: 4K < reasoning <= 10K
FINAL_SECTION_CAP = 300          # tokens of the visible answer (content after </think>)
FINAL_TAIL_CHARS = 80            # characters allowed after the last \boxed{...} of a math answer (closing words, "$$")
ROW_CAP = 65_536                 # stage-2 row cap (64K)
REF_MATCHES = ("exact", "numeric")


@dataclass(frozen=True)
class CvgPart:
    name: str                    # replay part
    retention_part: str          # schema.RETENTION_PARTS value of the rows (loader, checks, label provenance)
    floor: int                   # reasoning tokens: floor < r
    cap: int                     # reasoning tokens: r <= cap
    share: float                 # share of the convergence parts' loss tokens


CVG_PARTS: dict[str, CvgPart] = {
    "math_hard": CvgPart("math_hard", "math", THINK_CAP_LONG, THINK_CAP_HARD, 0.60),
    "math_medium": CvgPart("math_medium", "math", MEDIUM_FLOOR, THINK_CAP_LONG, 0.15),
    "sci_mcq": CvgPart("sci_mcq", "sci_mcq", 0, THINK_CAP_LONG, 0.25),
}
CVG_ORDER = ("math_hard", "math_medium", "sci_mcq")


@dataclass(frozen=True)
class CvgProfile:
    """One convergence build: its parts (walk and interleave order), the final-section cap, the replay reserve parts
    kept verbatim next to them, and its names (output files <stem>.parquet / <stem>_manifest.json, default run-id
    prefix, manifest version)."""
    name: str
    parts: Mapping[str, CvgPart]
    final_cap: int
    reserve_parts: tuple[str, ...]
    stem: str
    run_prefix: str
    version: str

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(self.parts)

    def source_parts(self) -> tuple[str, ...]:
        """The retention parts the profile draws from (loader order: math first)."""
        return tuple(dict.fromkeys(self.parts[p].retention_part for p in self.order))


HARD_MATH_V5_CAP = 12_288        # revision-5 hard math: 10,241-12,288 exact reasoning tokens
HARD_MATH_V5_FINAL_CAP = 600     # decision D6: final section <= 600 tokens
PROFILES: dict[str, CvgProfile] = {
    "stage2_replay": CvgProfile("stage2_replay", CVG_PARTS, FINAL_SECTION_CAP, ("if", "chat"), "convergence_replay",
                                "data_retention_convergence", "retention-convergence-replay-v1"),
    "hard_math_v5": CvgProfile("hard_math_v5",
                               {"math_hard": CvgPart("math_hard", "math", THINK_CAP_LONG, HARD_MATH_V5_CAP, 1.0)},
                               HARD_MATH_V5_FINAL_CAP, (), "hard_math", "data_retention_hard_math_v5",
                               "retention-hard-math-v5"),
}
MATH_PREFILTER_STAGES = ("not_in_pool_rows", "ref_match", "single_committed_answer", "ends_with_answer",
                         "final_section_chars")
MCQ_PREFILTER_STAGES = ("not_in_pool_rows", "ends_with_letter", "final_section_chars")


def token_targets(total: int, shares: Mapping[str, float] | None = None) -> dict[str, int]:
    """Loss tokens per part for `total` convergence loss tokens (largest remainder over the shares)."""
    return apportion(total, dict(shares or {p: CVG_PARTS[p].share for p in CVG_ORDER}))


def math_band(reasoning_tokens: int, parts: Mapping[str, CvgPart] | None = None) -> str | None:
    """The math part whose reasoning band holds the count (default parts: math_hard, math_medium), or None."""
    for p, spec in (parts or CVG_PARTS).items():
        if spec.retention_part == "math" and spec.floor < reasoning_tokens <= spec.cap:
            return p
    return None


def in_band(part: str, reasoning_tokens: int | None, parts: Mapping[str, CvgPart] | None = None) -> bool:
    spec = (parts or CVG_PARTS)[part]
    return reasoning_tokens is not None and spec.floor < reasoning_tokens <= spec.cap and reasoning_tokens > 0


def final_section_ok(content_tokens: int | None, cap: int = FINAL_SECTION_CAP) -> bool:
    """The short-final-section rule on the measured visible answer."""
    return content_tokens is not None and 0 < content_tokens <= cap


def final_section_chars_ok(answer: str, cap: int = FINAL_SECTION_CAP) -> bool:
    """Cheap pre-filter: an answer longer than cap x 6 characters cannot fit the token cap."""
    return 0 < len((answer or "").strip()) <= cap * CHARS_PER_TOKEN_CEILING


def _last_boxed_end(text: str) -> int | None:
    """Index just past the closing brace of the last balanced \\boxed{...} (or \\fbox{...})."""
    end = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text or ""):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            end = i
    return end


def math_commitment(answer: str) -> dict[str, Any]:
    """Does a math answer commit to one final answer and end with it? `single`: every \\boxed{} of the answer states the
    same normalised answer; `tail_chars`: characters after the last \\boxed{...} (whitespace, "$", "\\]" and "." do not
    count); `ends`: tail_chars <= FINAL_TAIL_CHARS."""
    boxed = boxed_contents(answer)
    distinct = {normalise_answer(b) for b in boxed if b}
    end = _last_boxed_end(answer)
    tail = "" if end is None else re.sub(r"[\s$.]|\\\]|\\\)", "", answer[end:])
    return {"n_boxed": len(boxed), "single": len(distinct) == 1, "tail_chars": None if end is None else len(tail),
            "ends": end is not None and len(tail) <= FINAL_TAIL_CHARS}


_TRAILING_FORMAT = re.compile(r"(?:[\s*.)}\]$_]|\\\]|\\\)|\\end\{[a-z*]+\})*")
_FORM_NAMES = ("boxed", "double_paren", "final_answer", "best_answer", "answer_colon", "answer_is")


def mcq_ending(answer: str, letter: str | None) -> str | None:
    """The final-answer form ("boxed", "answer_colon", ...) when the visible answer ENDS with `letter` stated in one of
    the explicit final-answer forms of the MCQ split (science._FINAL: \\boxed{X}, ((X)), "Final Answer: X", "The best
    answer is X", "Answer: X", "the answer is X") and only formatting follows it; None otherwise."""
    if not letter or not answer:
        return None
    text = answer.rstrip()
    last: tuple[int, int, str, str] | None = None
    for name, pat in zip(_FORM_NAMES, _FINAL, strict=True):
        for m in pat.finditer(text):
            if last is None or m.end() > last[1] or (m.end() == last[1] and m.start() < last[0]):
                last = (m.start(), m.end(), m.group(1), name)
    if last is None or last[2] != letter:
        return None
    return last[3] if _TRAILING_FORMAT.fullmatch(text[last[1]:]) else None


def take_until(rows: Sequence[dict[str, Any]], target: int, key: str = "n_assistant_tokens",
               skip: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Rows in order (task ids in `skip` passed over) while the running sum of `key` is below `target`; the result
    reaches the target with its last row (or holds every row when the supply is short)."""
    skip = set(skip)
    out, total = [], 0
    for r in rows:
        if total >= target:
            break
        if r["task_id"] in skip:
            continue
        out.append(r)
        total += r[key]
    return out


def disjointness(new: Sequence[Mapping[str, Any]], prior: Sequence[Mapping[str, Any]],
                 keys: Sequence[str] = ("task_id", "row_key", "prompt_sha256"),
                 clusters: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Shared keys between the new rows and the prior rows, the new rows' duplicates, and the dedup clusters that hold a
    new row together with a prior row or with another new row (`clusters`: {cluster id: member task ids})."""
    out: dict[str, Any] = {}
    for k in keys:
        a = [r[k] for r in new]
        out[f"shared_{k}"] = len(set(a) & {r[k] for r in prior})
        out[f"duplicate_{k}"] = len(a) - len(set(a))
    new_ids, prior_ids = {r["task_id"] for r in new}, {r["task_id"] for r in prior}
    joined = [g for g, m in (clusters or {}).items()
              if sum(x in new_ids for x in m) > 1 or (any(x in new_ids for x in m) and any(x in prior_ids for x in m))]
    out["dedup_clusters_joining"] = len(joined)
    out["pass"] = not joined and not any(v for k, v in out.items() if k.startswith(("shared_", "duplicate_")))
    return out


def prefix_token_shares(cells: Sequence[str], tokens: Sequence[int], ks: Iterable[int]) -> list[dict[str, Any]]:
    """Loss-token share of every cell in the first k rows, for each k."""
    total = Counter()
    for c, t in zip(cells, tokens, strict=True):
        total[c] += t
    grand = sum(total.values()) or 1
    out = []
    for k in ks:
        head = Counter()
        for c, t in zip(cells[:k], tokens[:k], strict=True):
            head[c] += t
        n = sum(head.values()) or 1
        out.append({"k": min(k, len(cells)), "loss_tokens": sum(head.values()),
                    "shares": {c: round(head[c] / n, 4) for c in sorted(total)}})
    return [{"k": len(cells), "loss_tokens": grand, "shares": {c: round(total[c] / grand, 4) for c in sorted(total)}}] + out


def prefilter_checks(source_part: str, c: dict[str, Any], exclude: set[str] | frozenset[str],
                     final_cap: int = FINAL_SECTION_CAP) -> list[tuple[str, Any]]:
    """The (stage, predicate) pairs of the convergence pre-filter for one loader candidate (common.Funnel.run order):
    math (MATH_PREFILTER_STAGES) or sci_mcq (MCQ_PREFILTER_STAGES). Records `ending_form` (MCQ) or `commitment` (math)
    in c["meta"]."""
    ans = c["messages"][1]["content"] or ""
    meta = c["meta"]
    if source_part == "math":
        def committed() -> bool:
            meta["commitment"] = math_commitment(ans)
            return meta["commitment"]["single"]

        return [("not_in_pool_rows", lambda: c["task_id"] not in exclude),
                ("ref_match", lambda: meta.get("ref_match") in REF_MATCHES),
                ("single_committed_answer", committed),
                ("ends_with_answer", lambda: meta["commitment"]["ends"]),
                ("final_section_chars", lambda: final_section_chars_ok(ans, final_cap))]
    if source_part == "sci_mcq":
        def ends() -> bool:
            meta["ending_form"] = mcq_ending(ans, meta.get("letter"))
            return meta["ending_form"] is not None

        return [("not_in_pool_rows", lambda: c["task_id"] not in exclude),
                ("ends_with_letter", ends),
                ("final_section_chars", lambda: final_section_chars_ok(ans, final_cap))]
    raise ValueError(f"no convergence pre-filter for {source_part}")


def walk_part(source_part: str, reasoning_tokens: int, final_tokens: int, profile: CvgProfile | None = None
              ) -> tuple[str | None, str]:
    """(replay part, "") for a measured candidate that fits a part's reasoning band and the final-section cap of the
    profile (default stage2_replay), else (None, reason)."""
    prof = profile or PROFILES["stage2_replay"]
    part = math_band(reasoning_tokens, prof.parts) if source_part == "math" else (
        "sci_mcq" if "sci_mcq" in prof.parts and in_band("sci_mcq", reasoning_tokens, prof.parts) else None)
    if part is None:
        return None, "reasoning_out_of_band"
    if not final_section_ok(final_tokens, prof.final_cap):
        return None, "final_section_tokens"
    return part, ""
