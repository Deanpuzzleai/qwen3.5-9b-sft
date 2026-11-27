"""Science parts: nvidia/Nemotron-Science-v1 MCQ and RQA splits (docs/DATA_PLAN.md section 3.4), GPT-OSS-120B traces
(the card's statement; the rows carry no model field).

MCQ (`data/MCQ.jsonl`): synthetic GPQA-style questions with lettered options and a requested answer format (\\boxed{X},
"Answer: X", "((X))", "The Best Answer is X", ...). The split ships no reference answers, so the check is internal:
the answer states exactly one option letter in a final-answer form, the letter is one of the listed options (contiguous
A.. with at least four), and it is the letter the reasoning concludes with. The draw prefers rows with >= 5 options.

RQA (`data/RQA.jsonl`): synthetic chemistry problems, often one sub-question of a multi-part problem with a shared
context. Every prompt at the pinned revision opens with an instruction whose "\\b" was written as a backspace
("\\x08oxed{}"); the loader restores "\\boxed{}" (counted) and drops prompts with any other control character or with
literal "\\n" escapes in place of newlines. Kept: single-question prompts (at most one enumerated question item and at
most one question mark), exactly one non-empty \\boxed{} of at most 200 characters in the answer, no tool turns.
`label_provenance = format`: the split ships no reference answers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apex_data.retention.common import (
    CHARS_PER_TOKEN_CEILING, CONTROL_CHARS, PARTS, Funnel, boxed_contents, candidate, has_think_tags, iter_jsonl_range,
    load_jsonl,
)

MCQ_STAGES = ("shape", "reasoning_present", "think_closed", "control_chars", "options", "letter_present",
              "letter_consistent", "reasoning_chars")
RQA_STAGES = ("shape", "reasoning_present", "think_closed", "control_chars", "escaped_newlines", "single_question",
              "one_boxed", "reasoning_chars")

LETTERS = "ABCDEFGHIJ"
_OPTION = re.compile(r"(?m)^\s*\(?([A-J])[).:]\s+\S")
_FINAL = [re.compile(p) for p in (
    r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}",
    r"\(\(\s*([A-J])\s*\)\)",
    r"(?i:final\s+answer)\s*[:：]\s*\**\s*\(?([A-J])\b(?![\w'])",
    r"(?i:best\s+answer\s+is)\s*:?\s*\**\s*\(?([A-J])\b(?![\w'])",
    r"(?<![A-Za-z])(?i:answer)\s*[:：]\s*\**\s*\(?([A-J])\b(?![\w'])",
    r"(?i:answer\s+is)\s*:?\s*\**\s*\(?([A-J])\)?(?![\w'])",
)]
_REASONING_FINAL = _FINAL + [re.compile(r"(?i:answer)\s*(?:(?i:is|would\s+be|should\s+be)\s*)?[:：]?\s*\**\s*\(?([A-J])\b(?![\w'])")]
_BACKSPACE_BOXED = "\x08oxed"
_ESCAPED_NEWLINE = re.compile(r"\\n(?![a-z])")
_Q_VERBS = (r"(?:determine|calculate|compute|derive|estimate|find|show|explain|predict|propose|identify|write|evaluate|"
            r"obtain|express|give|state|sketch|discuss|rank|assign|draw|suggest|describe|provide|construct|formulate|"
            r"justify|compare|what|which|how|why|list|outline|design|interpret|use|using|assuming|assess|quantify|"
            r"rationali[sz]e|deduce|solve|verify|prove|account)")
_Q_ITEM = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*|#+\s*)?\s*(?:\(?[a-h]\)|\(?(?:i{1,3}|iv|vi{0,3}|v)\)|\d{1,2}[.)]|"
                     r"(?:task|part|question|step)\s*[\w.]{0,4}[.:)]?)\s*(?:\*\*)?[^\n]{0,80}?\b" + _Q_VERBS + r"\b")
_INLINE_ROMAN = re.compile(r"\((i{1,3}|iv|v|vi)\)")
_INLINE_LETTER = re.compile(r"(?<![\w(])\(([a-f])\)")


def options(prompt: str) -> list[str]:
    """Option letters listed in the prompt, in order of first appearance."""
    seen: list[str] = []
    for x in _OPTION.findall(prompt or ""):
        if x not in seen:
            seen.append(x)
    return seen


def final_letters(text: str, patterns: list[re.Pattern[str]] = _FINAL) -> list[tuple[int, str]]:
    found = sorted((m.start(), m.group(1)) for p in patterns for m in p.finditer(text or ""))
    return found


def mcq_letters(prompt: str, answer: str, reasoning: str) -> dict[str, Any]:
    """The letters the answer and the end of the reasoning state, and the listed options."""
    opts = options(prompt)
    ans = [x for _, x in final_letters(answer)]
    rsn = [x for _, x in final_letters(reasoning[-2000:], _REASONING_FINAL)]
    return {"options": opts, "answer_letters": sorted(set(ans)), "letter": ans[-1] if ans else None,
            "reasoning_letter": rsn[-1] if rsn else None}


def repair_rqa_prompt(prompt: str) -> tuple[str, bool]:
    """Restore the "\\boxed{}" whose backslash-b became a backspace character upstream."""
    if _BACKSPACE_BOXED in prompt:
        return prompt.replace(_BACKSPACE_BOXED, "\\boxed"), True
    return prompt, False


def question_items(prompt: str) -> int:
    """Enumerated question items (lines opening with a)/i)/1./Task/Part followed by a question verb) plus distinct inline
    sub-question markers beyond the first of each family; > 1 means a multi-question prompt."""
    n = len(_Q_ITEM.findall(prompt or ""))
    roman, letters = set(_INLINE_ROMAN.findall(prompt or "")), set(_INLINE_LETTER.findall(prompt or ""))
    return max(n, len(roman), len(letters))


def single_question(prompt: str) -> bool:
    return question_items(prompt) <= 1 and (prompt or "").count("?") <= 1


def _mcq_chunk(path: str, start: int, end: int, revision: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    funnel = Funnel(MCQ_STAGES)
    out: list[dict[str, Any]] = []
    n_options: dict[str, int] = {}
    char_cap = PARTS["sci_mcq"].think_cap * CHARS_PER_TOKEN_CEILING
    for offset, r in iter_jsonl_range(path, start, end):
        msgs = r.get("messages") or []
        roles = [m.get("role") for m in msgs]
        prompt = (msgs[0].get("content") or "") if msgs else ""
        asst = msgs[-1] if msgs else {}
        reasoning, answer = asst.get("reasoning_content") or "", asst.get("content") or ""
        info: dict[str, Any] = {}

        def letters() -> bool:
            info.update(mcq_letters(prompt, answer, reasoning))
            return len(info["answer_letters"]) == 1

        failed = funnel.run([
            ("shape", lambda: roles == ["user", "assistant"] and not r.get("tools") and not asst.get("tool_calls")),
            ("reasoning_present", lambda: bool(reasoning.strip()) and bool(answer.strip())),
            ("think_closed", lambda: not has_think_tags(reasoning, answer, prompt)),
            ("control_chars", lambda: not CONTROL_CHARS.search(prompt + reasoning + answer)),
            ("options", lambda: len(options(prompt)) >= 4 and options(prompt) == list(LETTERS[:len(options(prompt))])),
            ("letter_present", letters),
            ("letter_consistent", lambda: info["letter"] in info["options"] and info["letter"] == info["reasoning_letter"]),
            ("reasoning_chars", lambda: len(reasoning) <= char_cap),
        ])
        if failed:
            continue
        k = str(len(info["options"]))
        n_options[k] = n_options.get(k, 0) + 1
        out.append(candidate("sci_mcq", f"MCQ/{r['uuid']}", prompt, answer, reasoning, hf_revision=revision,
                             row_license=r.get("license"), origin={"file": Path(path).name, "offset": offset},
                             n_options=len(info["options"]), letter=info["letter"]))
    return out, funnel.to_dict(), n_options


def _rqa_chunk(path: str, start: int, end: int, revision: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    funnel = Funnel(RQA_STAGES)
    out: list[dict[str, Any]] = []
    side = {"prompt_backspace_repaired": 0}
    char_cap = PARTS["sci_rqa"].think_cap * CHARS_PER_TOKEN_CEILING
    for offset, r in iter_jsonl_range(path, start, end):
        msgs = r.get("messages") or []
        roles = [m.get("role") for m in msgs]
        raw_prompt = (msgs[0].get("content") or "") if msgs else ""
        prompt, repaired = repair_rqa_prompt(raw_prompt)
        side["prompt_backspace_repaired"] += int(repaired)
        asst = msgs[-1] if msgs else {}
        reasoning, answer = asst.get("reasoning_content") or "", asst.get("content") or ""
        boxed = boxed_contents(answer)
        failed = funnel.run([
            ("shape", lambda: roles == ["user", "assistant"] and not r.get("tools") and not asst.get("tool_calls")),
            ("reasoning_present", lambda: bool(reasoning.strip()) and bool(answer.strip())),
            ("think_closed", lambda: not has_think_tags(reasoning, answer, prompt)),
            ("control_chars", lambda: not CONTROL_CHARS.search(prompt + reasoning + answer)),
            ("escaped_newlines", lambda: not _ESCAPED_NEWLINE.search(prompt)),
            ("single_question", lambda: single_question(prompt)),
            ("one_boxed", lambda: len(boxed) == 1 and 0 < len(boxed[0]) <= 200),
            ("reasoning_chars", lambda: len(reasoning) <= char_cap),
        ])
        if failed:
            continue
        out.append(candidate("sci_rqa", f"RQA/{r['uuid']}", prompt, answer, reasoning, hf_revision=revision,
                             row_license=r.get("license"), origin={"file": Path(path).name, "offset": offset},
                             boxed=boxed[0], prompt_repaired=repaired))
    return out, funnel.to_dict(), side



def load_mcq(raw_dir: Path, revision: str, workers: int = 16) -> tuple[list[dict[str, Any]], Funnel, dict[str, int]]:
    """MCQ candidates, the funnel, and the candidates' option-count histogram."""
    return load_jsonl(_mcq_chunk, raw_dir / "data" / "MCQ.jsonl", MCQ_STAGES, revision, workers, 64)


def load_rqa(raw_dir: Path, revision: str, workers: int = 16) -> tuple[list[dict[str, Any]], Funnel, dict[str, int]]:
    """RQA candidates, the funnel, and the prompt-repair count."""
    return load_jsonl(_rqa_chunk, raw_dir / "data" / "RQA.jsonl", RQA_STAGES, revision, workers, 32)
