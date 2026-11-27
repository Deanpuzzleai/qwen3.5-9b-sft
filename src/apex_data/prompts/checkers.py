"""Deterministic checkers for the constraints in ``constraints.py`` (docs/DATA_PLAN.md sections 3.4 and 6.3): regex and
counting only; no model, no network, no randomness.

Interface (used by the retention builder's constraint gate and by later grading):
* ``check_all(constraints, response) -> (all_ok, [CheckResult])`` and ``check(constraint, response) -> CheckResult``
  accept constraint dicts (``{"kind", "params", "instruction", "id"}``, as stored in ``episode.extra["constraints"]``)
  or ``Constraint`` objects.
* ``response`` is the assistant's final message text. Both forms are accepted: the raw completion (a leading
  ``<think>...</think>`` block is removed here, so reasoning never counts toward words, bullets, phrases, ...) and the
  content that vLLM's reasoning parser already separated. An unclosed think block leaves an empty content, which
  fails every constraint.
* Every ``CheckResult`` carries the constraint's stable ``id``, ``kind``, ``ok`` (alias ``passed``) and a human-readable
  ``detail``; ``failed_ids`` lists the ids that failed; ``constraint_verdict`` builds the schema.Verdict (verifier
  "constraint", verifier_version CHECKER_VERSION, label_source "ours", the pass fraction in ``extra["r_partial"]``);
  ``filter_checker`` is the callable that ``render.filters.filter_episodes(constraint_checker=...)`` takes.

Revision 3 (restored): the thirteen template kinds. Revision 4.3 (WP5) adds the kinds the prompt extractor produces
(upper case, postscript, title, quotation, two responses, placeholders, highlights, capital-word and letter frequency),
inclusive (min, max) bounds for word, sentence, paragraph and bullet counts, the "Section N" marker for section counts,
a normalised end-phrase comparison (case, surrounding quotes and markdown emphasis ignored) and English as a response
language. Language checks are script ratios (non-Latin scripts) or function-word counts (Latin-script languages); both
are documented approximations and the only non-trivial heuristics in this file.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from apex_data.prompts.constraints import Constraint
from apex_data.schema import Verdict

CHECKER_VERSION = "wp5-checkers-v2"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S", re.M)
_SECTION_RE = re.compile(r"^\s*##(?!#)\s+\S", re.M)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.S)
_HIGHLIGHT_RE = re.compile(r"\*\*[^\n*]+?\*\*|(?<![*\w])\*(?!\s)[^\n*]+?(?<!\s)\*(?![*\w])")
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]\n]{1,80}\]")
COMMAS = (",", "，", "،")
_EDGE = " \t\n\"'“”‘’*_`>#"

SCRIPT_RANGES = {
    "russian": r"Ѐ-ӿ", "greek": r"Ͱ-Ͽἀ-῿", "arabic": r"؀-ۿݐ-ݿ",
    "hindi": r"ऀ-ॿ", "korean": r"가-힯ᄀ-ᇿ㄰-㆏",
    "japanese": r"぀-ゟ゠-ヿ一-鿿", "chinese": r"一-鿿㐀-䶿",
}
_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
STOPWORDS = {
    "english": {"the", "and", "is", "are", "to", "of", "in", "that", "it", "you", "for", "with", "this", "on", "be", "as",
                "your", "can", "will", "have", "was", "but", "not", "they", "from"},
    "french": {"le", "la", "les", "des", "une", "un", "et", "est", "pour", "dans", "que", "qui", "pas", "vous", "nous",
               "sont", "avec", "sur", "ce", "cette", "il", "elle", "au", "aux", "du", "votre", "vos", "mais", "plus"},
    "spanish": {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "es", "por", "para", "con", "se",
                "su", "al", "lo", "como", "más", "pero", "sus", "también", "esta", "este", "puede", "son"},
    "german": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu", "mit", "auf", "für", "sich", "sie", "es",
               "auch", "den", "dem", "von", "im", "wir", "ich", "dass", "oder", "wird", "einen", "einer", "kann"},
    "italian": {"il", "la", "le", "gli", "di", "che", "e", "è", "un", "una", "per", "non", "con", "sono", "del", "della",
                "si", "da", "al", "come", "anche", "più", "ma", "questo", "nel", "nella", "dei", "delle"},
    "portuguese": {"o", "os", "as", "de", "do", "da", "que", "e", "em", "um", "uma", "para", "com", "não", "se", "por",
                   "mais", "como", "são", "dos", "das", "ao", "seu", "sua", "você", "também", "está", "pode"},
}


@dataclass
class CheckResult:
    id: str          # the constraint's stable id (its kind within a set)
    kind: str
    ok: bool
    detail: str

    @property
    def passed(self) -> bool:
        return self.ok

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.ok}


# ------------------------------------------------------------------------------------------------ text helpers
def split_think(raw: str) -> tuple[bool, bool, str | None, str]:
    """(opened, closed, reasoning, content). Qwen's template supplies the ``<think>\\n`` opener in the prompt, so a raw
    completion may start without ``<think>`` and still carry ``</think>``."""
    if THINK_CLOSE in raw:
        head, _, tail = raw.partition(THINK_CLOSE)
        reasoning = head.split(THINK_OPEN, 1)[1] if THINK_OPEN in head else head
        return True, True, reasoning.strip(), tail.lstrip("\n")
    if THINK_OPEN in raw:
        return True, False, raw.split(THINK_OPEN, 1)[1].strip(), ""
    return False, False, None, raw


def strip_think(response: str) -> str:
    """The content part of a raw completion: text after the last ``</think>``; an opened-but-unclosed block yields ""."""
    text = response or ""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].lstrip("\n")
    opened, closed, _, content = split_think(text)
    return "" if opened and not closed else content


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split((text or "").strip()) if _WORD_RE.search(s)]


def count_paragraphs(text: str, separator: str | None = None) -> int:
    """Blank-line separated blocks, or blocks between `separator` lines (`***`, `---`) when one is given."""
    if separator:
        parts = re.split(rf"(?m)^\s*{re.escape(separator)}\s*$", (text or "").strip())
        return len([p for p in parts if p.strip()])
    return len([p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()])


def count_bullets(text: str) -> int:
    return len(_BULLET_RE.findall(text or ""))


def count_sections(text: str, marker: str | None = None) -> int:
    """`## ` headings, or lines that open with `<marker> <number>` (e.g. "Section 2", "**SECTION 2:**")."""
    if marker:
        return len(re.findall(rf"(?im)^[\s#*_>]*{re.escape(marker)}\s+\d+\b", text or ""))
    return len(_SECTION_RE.findall(text or ""))


def count_word(text: str, word: str) -> int:
    return len(re.findall(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", text or "", re.I))


def count_capital_words(text: str) -> int:
    """Words of two or more letters written entirely in capitals."""
    return sum(1 for w in _WORD_RE.findall(text or "") if sum(ch.isalpha() for ch in w) >= 2 and w.isupper())


def strip_fence(text: str) -> str:
    s = (text or "").strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "\u202f": " ", "\u00a0": " "})


def normalise_edge(text: str) -> str:
    """Lower-cased text without surrounding whitespace, quotes and markdown emphasis, typographic quotes and narrow
    spaces folded to ASCII (for end-phrase comparison)."""
    return re.sub(r"\s+", " ", (text or "").translate(_APOSTROPHES).strip(_EDGE)).lower()


def language_ok(text: str, language: str) -> tuple[bool, str]:
    """Script-ratio check for non-Latin languages, function-word count for Latin-script ones."""
    lang = language.lower()
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return False, "no letters"
    if lang in SCRIPT_RANGES:
        rx = re.compile(f"[{SCRIPT_RANGES[lang]}]")
        ratio = sum(1 for c in letters if rx.match(c)) / len(letters)
        if lang == "japanese" and not _KANA.search(text):
            return False, f"japanese needs kana; script ratio {ratio:.2f}"
        if lang == "chinese" and _KANA.search(text):
            return False, "kana present in a chinese response"
        return ratio >= 0.6, f"{lang} script ratio {ratio:.2f} (>= 0.60 required)"
    if lang not in STOPWORDS:
        return False, f"unsupported language {language}"
    latin_ratio = sum(1 for c in letters if _LATIN.match(c)) / len(letters)
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    target = sum(1 for t in toks if t in STOPWORDS[lang])
    if lang == "english":
        other = max(sum(1 for t in toks if t in STOPWORDS[o] and t not in STOPWORDS["english"]) for o in STOPWORDS if o != lang)
        short = len(toks) < 25 and other == 0            # a slogan or a list of names carries few function words
        ok = latin_ratio >= 0.9 and ((target >= 3 and target > 2 * other) or short)
        return ok, f"english function words {target}, best other-language-only {other}, latin ratio {latin_ratio:.2f}"
    english = sum(1 for t in toks if t in STOPWORDS["english"])
    ok = latin_ratio >= 0.6 and target >= 3 and target > 2 * english
    return ok, f"{lang} function words {target}, english {english}, latin ratio {latin_ratio:.2f}"


def _bounds(n: int, lo: int | None, hi: int | None) -> bool:
    return (lo is None or n >= lo) and (hi is None or n <= hi)


# ------------------------------------------------------------------------------------------------ the checkers
def _check_content(c: Constraint, r: str) -> tuple[bool, str]:
    k, p = c.kind, c.params
    if k == "word_count":
        n = count_words(r)
        lo, hi = p.get("min"), p.get("max")
        return _bounds(n, lo, hi), f"{n} words; min {lo} max {hi}"
    if k == "sentence_count":
        n = len(split_sentences(r))
        if p.get("relation") == "range":
            return _bounds(n, p.get("min"), p.get("max")), f"{n} sentences; min {p.get('min')} max {p.get('max')}"
        return (n <= p["count"] if p["relation"] == "at most" else n >= p["count"]), f"{n} sentences; {p['relation']} {p['count']}"
    if k == "paragraph_count":
        n = count_paragraphs(r, p.get("separator"))
        lo, hi = p.get("min", p["count"]), p.get("max", p["count"])
        return _bounds(n, lo, hi), f"{n} paragraphs (separator {p.get('separator')!r}); min {lo} max {hi}"
    if k == "bullet_count":
        n = count_bullets(r)
        lo, hi = p.get("min", p["count"]), p.get("max", p["count"])
        return _bounds(n, lo, hi), f"{n} bullets; min {lo} max {hi}"
    if k == "section_count":
        n = count_sections(r, p.get("marker"))
        return n == p["count"], f"{n} sections (marker {p.get('marker') or '## '!r}); exactly {p['count']}"
    if k == "required_keywords":
        counts = {w: count_word(r, w) for w in p["keywords"]}
        return all(counts[w] >= need for w, need in p["keywords"].items()), f"counts {counts}; required {p['keywords']}"
    if k == "forbidden_keywords":
        counts = {w: count_word(r, w) for w in p["keywords"]}
        return not any(counts.values()), f"counts {counts}; all must be 0"
    if k == "opening_phrase":
        ok = r.lstrip().startswith(p["phrase"])
        return ok, f"starts with {p['phrase']!r}: {ok}"
    if k == "closing_phrase":
        if p.get("normalise"):
            got, want = normalise_edge(r), normalise_edge(p["phrase"])
            ok = bool(want) and (got.endswith(want) or got.rstrip(".!?").endswith(want.rstrip(".!?")))
        else:
            ok = r.rstrip().endswith(p["phrase"])
        return ok, f"ends with {p['phrase']!r}: {ok}"
    if k == "all_lowercase":
        uppers = sum(1 for ch in r if ch.isupper())
        return uppers == 0 and any(ch.isalpha() for ch in r), f"{uppers} uppercase letters"
    if k == "all_uppercase":
        lowers = sum(1 for ch in r if ch.islower())
        return lowers == 0 and any(ch.isalpha() for ch in r), f"{lowers} lowercase letters"
    if k == "no_commas":
        n = sum(r.count(ch) for ch in COMMAS)
        return n == 0, f"{n} commas"
    if k == "json_only":
        try:
            obj = json.loads(strip_fence(r))
        except (json.JSONDecodeError, TypeError) as exc:
            return False, f"invalid JSON: {str(exc)[:80]}"
        allowed = (dict, list) if p.get("allow_array") else (dict,)
        return isinstance(obj, allowed), f"parsed {type(obj).__name__}; {'object or array' if p.get('allow_array') else 'object'} required"
    if k == "response_language":
        return language_ok(r, p["language"])
    if k == "postscript":
        marker = p.get("marker", "P.S.")
        rx = r"p\.\s?p\.\s?s\." if marker.upper().startswith("P.P") else r"p\.\s?(?:p\.\s?)?s\."
        ok = bool(re.search(rf"(?i)(?<![\w.]){rx}", r))           # IFEval's rule: the marker anywhere in the response
        return ok, f"{marker} marker present: {ok}"
    if k == "title":
        rx = r"<<[^<>\n]+>>" if p.get("brackets", "<<>>") == "<<>>" else r"\[\[[^\[\]\n]+\]\]"
        ok = bool(re.search(rx, r))
        return ok, f"title in {p.get('brackets', '<<>>')}: {ok}"
    if k == "quotation":
        s = r.strip()
        ok = len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "“" and s[-1] == "”"))
        return ok, f"wrapped in double quotes: {ok}"
    if k == "two_responses":
        parts = [x for x in r.split(p.get("separator", "******")) if x.strip()]
        ok = len(parts) == 2 and parts[0].strip() != parts[1].strip()
        return ok, f"{len(parts)} non-empty parts around {p.get('separator', '******')!r}; two different required"
    if k == "placeholders":
        n = len(_PLACEHOLDER_RE.findall(r))
        return _bounds(n, p.get("min"), p.get("max")), f"{n} [placeholders]; min {p.get('min')} max {p.get('max')}"
    if k == "highlights":
        n = len(_HIGHLIGHT_RE.findall(r))
        return _bounds(n, p.get("min"), p.get("max")), f"{n} highlighted spans; min {p.get('min')}"
    if k == "capital_words":
        n = count_capital_words(r)
        return _bounds(n, p.get("min"), p.get("max")), f"{n} all-capital words; min {p.get('min')} max {p.get('max')}"
    if k == "letter_frequency":
        n = r.lower().count(p["letter"].lower())
        return _bounds(n, p.get("min"), p.get("max")), f"letter {p['letter']!r} x{n}; min {p.get('min')} max {p.get('max')}"
    raise ValueError(f"unknown constraint kind {k}")


def check(constraint: Constraint | dict[str, Any], response: str) -> CheckResult:
    """Verdict for one constraint on the assistant's final text (think block stripped here)."""
    c = constraint if isinstance(constraint, Constraint) else Constraint.from_dict(constraint)
    content = strip_think(response)
    if not content.strip():
        return CheckResult(c.id, c.kind, False, "empty content (no answer after the think block)")
    ok, detail = _check_content(c, content)
    return CheckResult(c.id, c.kind, bool(ok), detail)


def check_all(constraints: list[Constraint | dict[str, Any]], response: str) -> tuple[bool, list[CheckResult]]:
    """Every constraint must pass; an empty list passes (natural chat rows carry format checks only)."""
    results = [check(c, response) for c in constraints]
    return all(x.ok for x in results), results


def failed_ids(constraints: list[Constraint | dict[str, Any]], response: str) -> list[str]:
    """Ids of the constraints the response violates."""
    return [r.id for r in check_all(constraints, response)[1] if not r.ok]


def filter_checker(constraints: list[dict[str, Any]], messages: Sequence[Any]) -> list[str]:
    """`render.filters.gate_constraints` hook: failed constraint ids on the last assistant turn's content (the
    reasoning never counts). `messages` are data_format.Message objects or plain dicts."""
    last = next((m for m in reversed(list(messages)) if (getattr(m, "role", None) or m.get("role")) == "assistant"), None)
    if last is None:
        return [str(c.get("id") or c.get("kind")) for c in constraints]
    content = getattr(last, "content", None) if not isinstance(last, dict) else last.get("content")
    return failed_ids(constraints, content or "")


def constraint_verdict(episode_id: str, task_id: str, constraints: list[Constraint | dict[str, Any]], response: str) -> Verdict:
    """schema.Verdict for a constrained chat episode: r_pass = 1 iff every constraint passes; extra.r_partial = pass
    fraction."""
    ok, results = check_all(constraints, response)
    n = len(results)
    v = Verdict(episode_id=episode_id, task_id=task_id, r_pass=int(ok),
                runs=[{"pass": ok, "results": [r.as_dict() for r in results]}], tamper_flags=[], verifier="constraint",
                verifier_version=CHECKER_VERSION, label_source="ours",
                graded_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                extra={"n_constraints": n, "failed_ids": [r.id for r in results if not r.ok],
                       "r_partial": (sum(1 for r in results if r.ok) / n) if n else 1.0})
    v.validate()
    return v
