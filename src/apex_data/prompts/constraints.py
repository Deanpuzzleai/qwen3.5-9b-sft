"""Programmatic instruction-following constraints (docs/DATA_PLAN.md sections 3.4 and 6.3).

A constraint is a (kind, params) pair with a natural-language instruction; ``checkers.py`` verifies it with regex and
counting alone. Two producers:

* The revision-3 template set (restored unchanged): our own templates, IFEval-style in *type* only; ``sample_constraints``
  draws 1-3 mutually compatible constraints for a prompt from a seeded ``random.Random`` and ``attach_to_prompt`` appends
  the numbered requirement block to a user turn.
* ``extract_constraints`` (revision 4.3, WP5): the IF split of nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 states
  its constraints only in the prompt text (the rows carry no machine-readable constraint list), so the retention builder
  reads the verifiable ones out of the prompt with the regular expressions below. Only unambiguous, whole-response
  constraints are extracted: a count or keyword rule scoped to a part ("each paragraph", "per section", "in the title")
  is skipped, and phrasings the extractor does not recognise are reported as uncovered, never guessed. A row qualifies for
  the IF part only if at least one constraint is extracted and every extracted constraint passes its checker.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

TEMPLATE_VERSION = "wp3-constraints-v1"
EXTRACTOR_VERSION = "wp5-extract-v1"
TEMPLATE_KINDS = ("word_count", "sentence_count", "paragraph_count", "bullet_count", "section_count", "required_keywords",
                  "forbidden_keywords", "opening_phrase", "closing_phrase", "all_lowercase", "no_commas", "json_only",
                  "response_language")
# Kinds that only the extractor produces (IFEval-type rules that the template set never draws).
EXTRACTED_KINDS = ("all_uppercase", "postscript", "title", "quotation", "two_responses", "placeholders", "highlights",
                   "capital_words", "letter_frequency")
KINDS = TEMPLATE_KINDS + EXTRACTED_KINDS
KIND_WEIGHTS = {"word_count": 2.0, "forbidden_keywords": 2.0, "required_keywords": 2.0, "bullet_count": 1.5,
                "paragraph_count": 1.5, "all_lowercase": 1.0, "no_commas": 1.0, "opening_phrase": 1.0,
                "closing_phrase": 1.0, "section_count": 1.0, "sentence_count": 1.0, "json_only": 0.7,
                "response_language": 0.7}
N_WEIGHTS = {1: 0.4, 2: 0.35, 3: 0.25}

_EXCLUSIVE_ONE_WAY = {
    "json_only": {"paragraph_count", "bullet_count", "section_count", "opening_phrase", "closing_phrase", "no_commas",
                  "sentence_count", "response_language", "all_lowercase"},
    "response_language": {"required_keywords", "forbidden_keywords", "opening_phrase", "closing_phrase", "all_lowercase",
                          "word_count", "sentence_count"},
    "sentence_count": {"word_count", "bullet_count", "paragraph_count", "section_count"},
    "paragraph_count": {"bullet_count", "section_count"},
}
EXCLUSIVE: dict[str, frozenset[str]] = {k: frozenset() for k in KINDS}
for _k, _vs in _EXCLUSIVE_ONE_WAY.items():
    for _v in _vs:
        EXCLUSIVE[_k] = EXCLUSIVE[_k] | {_v}
        EXCLUSIVE[_v] = EXCLUSIVE[_v] | {_k}

WORD_AT_LEAST = (80, 120, 150, 200, 300)
WORD_AT_MOST = (60, 100, 150, 200)
WORD_BETWEEN = ((50, 100), (100, 200), (150, 250), (200, 300))
SENTENCE_CHOICES = (("at most", 3), ("at most", 5), ("at least", 6), ("at least", 10))
PARAGRAPH_CHOICES = (2, 3, 4, 5)
BULLET_CHOICES = (3, 4, 5, 6)
SECTION_CHOICES = (2, 3, 4)
KEYWORD_REPEATS = (1, 2, 3)
OPENING_PHRASES = ("Here is my response.", "Let me address this directly.", "Thank you for the question.",
                   "To put it simply:", "Short version first.")
CLOSING_PHRASES = ("Is there anything else I can help with?", "That covers everything I wanted to say.",
                   "Let me know if you need more detail.", "I hope this helps.", "End of response.")
LANGUAGES = ("French", "Spanish", "German", "Italian", "Portuguese", "Russian", "Greek", "Arabic", "Hindi", "Chinese",
             "Japanese", "Korean")
# words never used as keywords (common function words of five letters or more plus chat boilerplate)
STOPLIST = frozenset("""about above after again against almost already also although always among another anyone
anything around because become before being below between beyond could doing during either enough every everything
following further having hello however itself little maybe might never nothing often other others ought please really
rather seems several should since still such than thank thanks their theirs there these they thing things think those
through under until using which while whole whose within without would write would you're yours could would should
describe explain provide question answer response respond following include detail details example examples
additional context postscript section sections paragraph paragraphs sentence sentences words letter letters format
ensure exactly least entire title bullet point points summary summarize summarise rewrite version style tone audience
english language brief briefly short shorter longer concise specific mention mentions including instead above""".split())


@dataclass(frozen=True)
class Constraint:
    """One constraint as stored in a pool record (tests["reference"][i]) and in episode.extra["constraints"]: the dict
    form {"kind", "params", "instruction", "id"} is JSON round-trip safe. `id` is stable within a constraint set (the
    kind, which `sample_constraints` never repeats) and is what checkers report."""
    kind: str
    params: dict[str, Any]
    instruction: str
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.kind)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Constraint":
        if d["kind"] not in KINDS:
            raise ValueError(f"unknown constraint kind {d['kind']}")
        return Constraint(d["kind"], dict(d["params"]), d["instruction"], d.get("id") or d["kind"])


_WORD = re.compile(r"[A-Za-z]{5,}")


def content_words(text: str, max_words: int = 40) -> list[str]:
    """Distinct lowercase alphabetic words of >= 5 letters from the prompt, in order of first appearance, minus STOPLIST."""
    out: list[str] = []
    for w in _WORD.findall(text or ""):
        lw = w.lower()
        if lw not in STOPLIST and lw not in out:
            out.append(lw)
            if len(out) >= max_words:
                break
    return out


def _quote_list(words: list[str]) -> str:
    q = [f'"{w}"' for w in words]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + " and " + q[-1]


def render_instruction(kind: str, params: dict[str, Any]) -> str:
    """The deterministic instruction text for a (kind, params) pair."""
    if kind == "word_count":
        rel = params["relation"]
        if rel == "between":
            return f"Your response must contain between {params['min']} and {params['max']} words."
        return f"Your response must contain {rel} {params['min'] if rel == 'at least' else params['max']} words."
    if kind == "sentence_count":
        return (f"Use {params['relation']} {params['count']} sentences in total "
                "(a sentence ends with '.', '!' or '?').")
    if kind == "paragraph_count":
        return f"Write exactly {params['count']} paragraphs, separated from each other by one blank line."
    if kind == "bullet_count":
        return (f"Include exactly {params['count']} markdown bullet points, each on its own line starting with '- ', "
                "and no other bullet points.")
    if kind == "section_count":
        return (f"Organise the response into exactly {params['count']} sections, each introduced by a markdown heading "
                "line that starts with '## '.")
    if kind == "required_keywords":
        parts = [f'the word "{w}" at least {k} time{"s" if k > 1 else ""}' for w, k in params["keywords"].items()]
        return "Use " + (parts[0] if len(parts) == 1 else " and ".join(parts)) + " (any letter case, as a whole word)."
    if kind == "forbidden_keywords":
        ws = params["keywords"]
        return f"Do not use the word{'s' if len(ws) > 1 else ''} {_quote_list(ws)} anywhere in your response."
    if kind == "opening_phrase":
        return f'Begin your response with the exact phrase "{params["phrase"]}" as the very first characters.'
    if kind == "closing_phrase":
        return f'End your response with the exact phrase "{params["phrase"]}" and write nothing after it.'
    if kind == "all_lowercase":
        return "Write your entire response in lowercase letters only; do not use any capital letters."
    if kind == "no_commas":
        return "Do not use any commas in your response."
    if kind == "json_only":
        return ("Format your entire response as a single valid JSON object. Do not write anything outside the JSON "
                "(a ```json code fence around it is allowed).")
    if kind == "response_language":
        return f"Write your entire response in {params['language']}; do not use any other language."
    raise ValueError(f"unknown constraint kind {kind}")


def make_constraint(kind: str, rng: random.Random, words: list[str], used_words: set[str]) -> Constraint | None:
    """Draw the parameters of one constraint; None if the prompt cannot support the kind (no usable keywords)."""
    params: dict[str, Any]
    if kind == "word_count":
        rel = rng.choice(("at least", "at most", "between"))
        if rel == "at least":
            params = {"relation": rel, "min": rng.choice(WORD_AT_LEAST), "max": None}
        elif rel == "at most":
            params = {"relation": rel, "min": None, "max": rng.choice(WORD_AT_MOST)}
        else:
            lo, hi = rng.choice(WORD_BETWEEN)
            params = {"relation": rel, "min": lo, "max": hi}
    elif kind == "sentence_count":
        rel, n = rng.choice(SENTENCE_CHOICES)
        params = {"relation": rel, "count": n}
    elif kind == "paragraph_count":
        params = {"count": rng.choice(PARAGRAPH_CHOICES)}
    elif kind == "bullet_count":
        params = {"count": rng.choice(BULLET_CHOICES)}
    elif kind == "section_count":
        params = {"count": rng.choice(SECTION_CHOICES)}
    elif kind in ("required_keywords", "forbidden_keywords"):
        avail = [w for w in words if w not in used_words]
        n = min(len(avail), rng.choice((1, 2)) if kind == "required_keywords" else rng.choice((1, 2, 3)))
        if n == 0:
            return None
        chosen = rng.sample(avail, n)
        used_words.update(chosen)
        if kind == "required_keywords":
            params = {"keywords": {w: rng.choice(KEYWORD_REPEATS) for w in chosen}}
        else:
            params = {"keywords": chosen}
    elif kind == "opening_phrase":
        params = {"phrase": rng.choice(OPENING_PHRASES)}
    elif kind == "closing_phrase":
        params = {"phrase": rng.choice(CLOSING_PHRASES)}
    elif kind in ("all_lowercase", "no_commas", "json_only"):
        params = {}
    elif kind == "response_language":
        params = {"language": rng.choice(LANGUAGES)}
    else:
        raise ValueError(kind)
    return Constraint(kind, params, render_instruction(kind, params))


def sample_constraints(prompt_text: str, rng: random.Random, n: int | None = None) -> list[Constraint]:
    """1-3 compatible constraints for `prompt_text`, deterministic for a given RNG state."""
    if n is None:
        n = rng.choices(list(N_WEIGHTS), weights=list(N_WEIGHTS.values()))[0]
    if not 1 <= n <= 3:
        raise ValueError("n must be 1..3")
    words = content_words(prompt_text)
    pool = dict(KIND_WEIGHTS)
    if len(words) < 2:
        pool.pop("required_keywords", None)
        pool.pop("forbidden_keywords", None)
    chosen: list[Constraint] = []
    used_words: set[str] = set()
    while len(chosen) < n and pool:
        kinds = list(pool)
        kind = rng.choices(kinds, weights=[pool[k] for k in kinds])[0]
        pool.pop(kind)
        for k in EXCLUSIVE[kind]:
            pool.pop(k, None)
        c = make_constraint(kind, rng, words, used_words)
        if c is not None:
            chosen.append(c)
    if any(c.kind == "all_lowercase" for c in chosen):          # phrases must themselves be lowercase then
        chosen = [Constraint(c.kind, {**c.params, "phrase": c.params["phrase"].lower()},
                             render_instruction(c.kind, {**c.params, "phrase": c.params["phrase"].lower()}), c.id)
                  if c.kind in ("opening_phrase", "closing_phrase") else c for c in chosen]
    return chosen


def render_block(constraints: list[Constraint]) -> str:
    """The numbered requirement block appended to a user turn."""
    if not constraints:
        return ""
    lines = [f"{i + 1}. {c.instruction}" for i, c in enumerate(constraints)]
    return "Requirements for your response:\n" + "\n".join(lines)


def attach_to_prompt(prompt_text: str, constraints: list[Constraint]) -> str:
    block = render_block(constraints)
    return prompt_text.rstrip() + "\n\n" + block if block else prompt_text


def compatible(constraints: list[Constraint]) -> bool:
    kinds = [c.kind for c in constraints]
    if len(set(kinds)) != len(kinds):
        return False
    return not any(b in EXCLUSIVE[a] for i, a in enumerate(kinds) for b in kinds[i + 1:])


@dataclass
class ConstraintSet:
    """What a pool record stores under tests["reference"] for kind == "constraints"."""
    constraints: list[dict[str, Any]] = field(default_factory=list)
    template_version: str = TEMPLATE_VERSION


# ============================================================================================ extraction from prompts
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
                 "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                 "hundred": 100, "once": 1, "twice": 2, "thrice": 3, "a single": 1, "single": 1}
_NUM = r"(\d{1,5}|" + "|".join(sorted((k for k in _NUMBER_WORDS if k not in ("once", "twice", "thrice")), key=len,
                                       reverse=True)) + r")"
_Q_OPEN, _Q_CLOSE = "\"'“‘", "\"'”’"
_QUOTED = rf"[{_Q_OPEN}]([^{_Q_OPEN}{_Q_CLOSE}\n]{{1,60}}?)[{_Q_CLOSE}]"
_QUOTED_LIST = rf"((?:{_QUOTED}(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|nor)\s+)?)+)"
# A rule is scoped to a part of the response when its clause names a part ("each paragraph", "per section", ...).
_SCOPE_BEFORE = re.compile(r"\b(?:each|every|per|any\s+single|individual|first|last|second|third|final|opening|closing)\b",
                           re.I)
_SCOPE_AFTER = re.compile(r"^\W{0,3}(?:each|every|per|in\s+(?:each|every|the\s+(?:title|first|last|heading|subject|"
                          r"opening|closing|introduction|conclusion|summary|section|paragraph|list|table|bullet|caption|"
                          r"chorus|verse|headline|tagline|name))|within\s+(?:each|the\s+text\s+of)|for\s+each)\b", re.I)
_CLAUSE_START = re.compile(r"[.;:!?\n]|,\s+(?=and\b)|\(")
LANGUAGE_NAMES = ("english", "french", "spanish", "german", "italian", "portuguese", "russian", "greek", "arabic", "hindi",
                  "chinese", "japanese", "korean")
UNCOVERED_MARKERS = {   # IFEval-type rules the extractor does not check (counted, never guessed)
    "repeat_prompt": re.compile(r"\brepeat\s+(?:the|this|your)\s+(?:prompt|request|question|instruction)|"
                                r"\brepeat\s+the\s+(?:request|prompt)\s+above", re.I),
    "choose_option": re.compile(r"choose\s+one\s+from|choose\s+from\s+the\s+following|choose\s+one\s+of\s+the\s+following",
                                re.I),
    "nth_paragraph_first_word": re.compile(r"(?:first|second|third|\w+th)\s+paragraph\s+(?:must|should)\s+start\s+with",
                                           re.I),
    "exact_word_count": re.compile(rf"\bexactly\s+{_NUM}\s+words\b", re.I),
}


def parse_number(token: str) -> int | None:
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    return _NUMBER_WORDS.get(t)


def _clause_before(text: str, start: int) -> str:
    """The text of the current clause up to `start` (back to the last sentence or list boundary)."""
    head = text[max(0, start - 160):start]
    cut = [m.end() for m in _CLAUSE_START.finditer(head)]
    return head[cut[-1]:] if cut else head


def _scoped(text: str, m: re.Match[str]) -> bool:
    """True when the matched rule applies to a part of the response, not to the whole of it."""
    return bool(_SCOPE_BEFORE.search(_clause_before(text, m.start()))) or bool(_SCOPE_AFTER.search(text[m.end():m.end() + 50]))


def _quoted_items(blob: str) -> list[str]:
    items = []
    for m in re.finditer(_QUOTED, blob):
        w = m.group(1).strip().strip(".,;:!?").strip()
        if w and w.lower() not in [x.lower() for x in items]:
            items.append(w)
    return items


_REL = (r"(at\s+least|no\s+less\s+than|not\s+less\s+than|a\s+minimum\s+of|minimum\s+of|more\s+than|over|"
        r"no\s+more\s+than|not\s+more\s+than|at\s+most|a\s+maximum\s+of|maximum\s+of|up\s+to|fewer\s+than|less\s+than|"
        r"under|not\s+exceed(?:ing)?|exactly|only)")
_LOWER = re.compile(r"\b(?:in\s+(?:english\s+and\s+)?(?:all\s+|entirely\s+(?:in\s+)?|only\s+)?|all\s+|entirely\s+(?:in\s+)?|"
                    r"only\s+(?:in\s+)?|using\s+(?:only\s+)?(?:all\s+)?|use\s+(?:only\s+)?(?:all\s+)?)lower\s*-?\s*case"
                    r"(?:\s+letters?)?(?:\s+only)?\b|\bno\s+capital\s+letters\s+(?:are\s+)?allowed\b", re.I)
_UPPER = re.compile(r"\b(?:in\s+(?:english\s+and\s+)?(?:all\s+|entirely\s+in\s+)?|all\s+|entirely\s+in\s+|use\s+(?:only\s+)?all\s+|"
                    r"using\s+(?:only\s+)?all\s+|written\s+in\s+all\s+|only\s+)(?:capital(?:s|\s+letters|ized)?|upper\s*-?\s*case"
                    r"(?:\s+letters)?|caps)\b(?!\s+(?:letters\s+)?(?:at\s+least|for|to|on|when))", re.I)
_CAPITAL_WORDS = re.compile(rf"{_REL}\s+{_NUM}\s+(?:\w+\s+)?words?\s+(?:that\s+are\s+|should\s+be\s+|written\s+|appearing\s+)?"
                            r"(?:in\s+)?(?:all\s+)?(?:capital(?:s|\s+letters|ized)?|caps|upper\s*-?\s*case)", re.I)
_CAPITAL_WORDS_2 = re.compile(rf"(?:words?\s+(?:in|with)\s+all\s+(?:capital\s+letters|caps)|capitali[sz]ed\s+words|"
                              rf"all-caps\s+words)\s+(?:should\s+)?(?:appear\s+)?{_REL}\s+{_NUM}\s+times", re.I)
_NO_COMMAS = re.compile(r"\b(?:use\s+no\s+commas?|(?:with\s+)?no\s+commas?|without\s+(?:using\s+|the\s+use\s+of\s+)?(?:any\s+)?"
                        r"commas?|(?:do\s+not|don't|must\s+not|should\s+not)\s+(?:use|include|contain)\s+(?:any\s+)?commas?|"
                        r"avoid\s+(?:using\s+)?(?:any\s+)?commas?|refrain\s+from\s+using\s+(?:any\s+)?commas?|"
                        r"commas?\s+(?:are|is)\s+not\s+allowed)\b", re.I)
_POSTSCRIPT = re.compile(r"\b(?:post\s*-?\s*script|p\.\s?p?\.?\s?s\.)", re.I)
_PPS = re.compile(r"\bp\.\s?p\.\s?s\b", re.I)
_TITLE_ANGLE = re.compile(r"<<\s*[^<>\n]{1,40}\s*>>|double\s+angular\s+brackets", re.I)
_TITLE_SQUARE = re.compile(r"\[\[\s*title\s*\]\]|double\s+square\s+brackets", re.I)
_QUOTATION = re.compile(r"\b(?:wrap|enclose|surround|put)\s+(?:your\s+|the\s+)?(?:entire|whole|full)\s+(?:response|answer|reply|"
                        r"output)\s+(?:with|in|within|inside|between)\s+double\s+quot(?:ation|e)s?\s*(?:marks)?", re.I)
_TWO_RESPONSES = re.compile(r"\*{6}")
_PLACEHOLDERS = re.compile(rf"{_REL}?\s*{_NUM}\s+placeholders?\b", re.I)
_HIGHLIGHTS = re.compile(rf"(?:highlight(?:ed)?\s+{_REL}?\s*{_NUM}\s+(?:\w+\s+){{0,3}}?\w+|{_REL}?\s*{_NUM}\s+(?:highlighted|"
                         rf"bold(?:ed)?(?:\s+text)?)\s+(?:sections?|parts?|phrases?|points?|words?|items?|examples?|"
                         rf"strategies|tips|ideas))", re.I)
_LETTER_FREQ = re.compile(rf"(?:the\s+)?letter\s+[{_Q_OPEN}]?([a-z])[{_Q_CLOSE}]?\s+(?:should\s+|must\s+)?(?:appear(?:s)?\s+|"
                          rf"be\s+used\s+)?{_REL}\s+{_NUM}\s+times", re.I)
_WORDS = re.compile(rf"{_REL}\s+{_NUM}\s+words\b|\bbetween\s+{_NUM}\s+and\s+{_NUM}\s+words\b|"
                    rf"(?:should|must|does|do|shall)\s+not\s+(?:exceed|be\s+(?:longer|more)\s+than)\s+{_NUM}\s+words\b", re.I)
_SENTENCES = re.compile(rf"(?:{_REL}\s+|\bin\s+|\busing\s+|\bof\s+|\bwith\s+|\bto\s+){_NUM}\s+sentences?\b|"
                        rf"\b{_NUM}\s+sentences?\s+long\b|"
                        rf"(?:should|must)\s+not\s+exceed\s+{_NUM}\s+sentences\b", re.I)
_PARAGRAPHS = re.compile(rf"(?:{_REL}\s+|\bin\s+|\binto\s+|\bof\s+|\bwith\s+|\bhave\s+|\bcontain\s+|\bwrite\s+)?\b{_NUM}"
                         r"(?:\s+|-)(?:distinct\s+|separate\s+|short\s+|concise\s+|detailed\s+)?paragraphs?\b", re.I)
_PARA_SEP = re.compile(r"separated\s+(?:from\s+each\s+other\s+)?(?:by|with|using)\s+(?:the\s+)?(?:markdown\s+divider\s*:?\s*)?"
                       r"[\"'“]?(\*\*\*|---)", re.I)
_BULLETS = re.compile(rf"(?:{_REL}\s+)?\b{_NUM}(?:\s+|-)(?:markdown\s+)?bullet(?:ed)?(?:\s+|-)points?\b", re.I)
_SECTIONS = re.compile(rf"\b{_NUM}\s+(?:distinct\s+|separate\s+|highlighted\s+)?sections?,?\s+(?:each\s+|and\s+each\s+)?"
                       r"(?:should\s+be\s+|must\s+be\s+)?(?:marked|labell?ed|beginning|starting|titled|denoted|introduced)"
                       r"\s+(?:with|by|as)\s+[\"'“]?(section|part|chapter|step|day|paragraph)\s+(?:n|x|\d+|\[\w+\])\b", re.I)
_KEYWORDS = re.compile(rf"\b(?:include|includes|including|use|using|incorporate|incorporating|mention|mentioning|contain|"
                       rf"containing|feature|featuring)\s+(?:all\s+of\s+)?(?:the\s+)?(?:following\s+)?(?:specific\s+)?"
                       rf"(?:key\s*words?|terms?|words?|phrases?)\s*:?\s*{_QUOTED_LIST}", re.I)
_KEYWORD_FREQ = re.compile(rf"\b(?:the\s+)?(?:key\s*)?(?:word|term|phrase|keyword)\s+{_QUOTED}\s+(?:should\s+|must\s+)?"
                           rf"(?:appear(?:s)?\s+|be\s+(?:used|mentioned|included)\s+|is\s+(?:used|mentioned)\s+)?"
                           rf"(?:{_REL}\s+)?({_NUM[1:-1]}|once|twice|thrice)(?:\s+times)?\b", re.I)
_KEYWORD_FREQ_2 = re.compile(rf"\b(?:include|use|mention)\s+(?:the\s+)?(?:key\s*)?(?:word|term|phrase|keyword)\s+{_QUOTED}\s+"
                             rf"(?:{_REL}\s+)?({_NUM[1:-1]}|once|twice|thrice)(?:\s+times)?\b", re.I)
_FORBIDDEN = re.compile(rf"\b(?:exclude|excluding|without\s+(?:using|mentioning|including)|(?:do\s+not|don't|must\s+not|"
                        rf"should\s+not|never|not\s+to)\s+(?:use|include|mention|contain)|avoid(?:\s+using|\s+mentioning)?|"
                        rf"refrain\s+from\s+using)\s+(?:any\s+(?:mention\s+of\s+)?|the\s+use\s+of\s+)?(?:the\s+)?"
                        rf"(?:following\s+)?(?:key\s*)?(?:words?|terms?|phrases?)?\s*:?\s*{_QUOTED_LIST}", re.I)
_END = re.compile(rf"\b(?:end|ends|ending|conclude|concludes|concluding|finish|close|wrap\s+up)\s+(?:(?:your|the|this|it|each)\s+"
                  rf"(?:\w+\s+){{0,3}}?)?with\s+(?:the\s+|this\s+)?(?:exact\s+|specific\s+|following\s+|precise\s+)*"
                  rf"(?:sentence|phrase|line|words?|quote|statement|question|exclamation)?\s*:?\s*"
                  rf"[\"“]([^\"“”\n]{{2,200}})[\"”]", re.I)
_END_SINGLE = re.compile(rf"\b(?:end|ends|ending|conclude|concludes|concluding|finish|close)\s+(?:(?:your|the|this|it)\s+"
                         rf"(?:\w+\s+){{0,3}}?)?with\s+(?:the\s+|this\s+)?(?:exact\s+|specific\s+|following\s+)*"
                         rf"(?:sentence|phrase|line|words?|statement)?\s*:?\s*'([^'\n]{{2,200}})'(?=[\s.,;:!?)]|$)", re.I)
_JSON = re.compile(r"\b(?:in|as|using|use|into|valid|entire\s+(?:response|output)\s+(?:should|must)\s+be\s+(?:in|wrapped\s+in))"
                   r"\s+(?:a\s+|the\s+)?(?:valid\s+)?json(?:\s+format(?:ted)?|\s+object|\s+structure|\s+array)?\b", re.I)
_LANGUAGE = re.compile(r"\b(?:written|be|is|are|respond|response|reply|answer|write|output|entirely|only|completely|"
                       r"exclusively)\s+(?:\w+\s+){0,2}?in\s+(" + "|".join(LANGUAGE_NAMES) + r")(?:\s+language)?\b", re.I)


_NEGATED = re.compile(r"\b(?:not|n't|never)\b(?:\s+\w+){0,2}\s*$", re.I)
_NEGATED_VERB = re.compile(r"\b(?:without|not|n't|never|avoid|avoiding|exclude|excluding|refrain\s+from)\b(?:\s+\w+){0,1}\s*$",
                           re.I)
_PART_WORDS = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|last|final|each|every|paragraph|section|sentence|"
                         r"bullet|stanza|verse|chorus|chapter|title|heading|part|list|introduction|conclusion)\b", re.I)
_TARGETED = re.compile(r"\b(?:word|words|phrase|term|keyword|title|name|heading|headline|letter|placeholders?|"
                       r"acronyms?)\b", re.I)


def _negated(text: str, m: re.Match[str]) -> bool:
    return bool(_NEGATED.search(text[max(0, m.start() - 30):m.start()]))


def _rel_bounds(rel: str | None, n: int, negated: bool = False) -> tuple[int | None, int | None]:
    """(min, max) inclusive for a relation phrase and a number; `negated` reads "not more than", "must not contain
    more than", ... as the complementary bound."""
    r = re.sub(r"\s+", " ", (rel or "exactly").lower())
    if negated:
        if r in ("more than", "over"):
            return None, n
        if r in ("fewer than", "less than", "under"):
            return n, None
        if r in ("at least", "a minimum of", "minimum of"):
            return None, n - 1
    if r in ("at least", "no less than", "not less than", "a minimum of", "minimum of"):
        return n, None
    if r in ("more than", "over"):
        return n + 1, None
    if r in ("no more than", "not more than", "at most", "a maximum of", "maximum of", "up to", "not exceed",
             "not exceeding"):
        return None, n
    if r in ("fewer than", "less than", "under"):
        return None, n - 1
    return n, n                                   # exactly / only / in N / N ... long


def _add(found: dict[str, Constraint], kind: str, params: dict[str, Any], span: str) -> None:
    """Keep one constraint per kind and scope; a second rule of the same kind gets a numbered id."""
    cid, k = kind, 2
    while cid in found:
        if found[cid].params == params:
            return
        cid, k = f"{kind}#{k}", k + 1
    found[cid] = Constraint(kind, params, span.strip(), cid)


def extract_constraints(prompt: str) -> tuple[list[Constraint], list[str]]:
    """(verifiable whole-response constraints stated in `prompt`, uncovered rule markers). Deterministic, regex only."""
    text = prompt or ""
    found: dict[str, Constraint] = {}
    for m in _CAPITAL_WORDS.finditer(text):
        n = parse_number(m.group(2))
        if n is not None and not _scoped(text, m):
            lo, hi = _rel_bounds(m.group(1), n, _negated(text, m))
            _add(found, "capital_words", {"min": lo, "max": hi}, m.group(0))
    for m in _CAPITAL_WORDS_2.finditer(text):
        n = parse_number(m.group(2))
        if n is not None and not _scoped(text, m):
            lo, hi = _rel_bounds(m.group(1), n)
            _add(found, "capital_words", {"min": lo, "max": hi}, m.group(0))
    capital_spans = [m.span() for m in (*_CAPITAL_WORDS.finditer(text), *_CAPITAL_WORDS_2.finditer(text))]
    for m in _UPPER.finditer(text):
        if any(a <= m.start() < b for a, b in capital_spans) or _scoped(text, m):
            continue
        if re.search(r"words?\s+(?:in\s+)?$", text[max(0, m.start() - 12):m.start()], re.I):
            continue
        if _TARGETED.search(_clause_before(text, m.start())):          # "the word X ... in uppercase"
            continue
        _add(found, "all_uppercase", {}, m.group(0))
    for m in _LOWER.finditer(text):
        if not _scoped(text, m) and not _TARGETED.search(_clause_before(text, m.start())):
            _add(found, "all_lowercase", {}, m.group(0))
    for m in _NO_COMMAS.finditer(text):
        if not _scoped(text, m):
            _add(found, "no_commas", {}, m.group(0))
    for m in _POSTSCRIPT.finditer(text):
        window = text[max(0, m.start() - 60):m.end() + 60]
        if re.search(r"\b(?:include|add|end|with|append|conclude|finish|postscript|followed)\b", window, re.I):
            marker = "P.P.S." if _PPS.search(window) else "P.S."
            _add(found, "postscript", {"marker": marker}, m.group(0))
            break
    if _TITLE_ANGLE.search(text):
        _add(found, "title", {"brackets": "<<>>"}, _TITLE_ANGLE.search(text).group(0))
    elif _TITLE_SQUARE.search(text):
        _add(found, "title", {"brackets": "[[]]"}, _TITLE_SQUARE.search(text).group(0))
    m = _QUOTATION.search(text)
    if m:
        _add(found, "quotation", {}, m.group(0))
    m = _TWO_RESPONSES.search(text)
    if m:
        sentence = text[max(0, m.start() - 200):m.end() + 80]
        if re.search(r"\b(?:two|2)\b", sentence, re.I) and not re.search(r"\b(?:three|four|five|3|4|5)\b\s+\w+\s+(?:like|as)",
                                                                         sentence, re.I):
            _add(found, "two_responses", {"separator": "******"}, m.group(0))
    for m in _PLACEHOLDERS.finditer(text):
        n = parse_number(m.group(2))
        if n is not None and not _scoped(text, m):
            lo, hi = _rel_bounds(m.group(1) or "at least", n)
            _add(found, "placeholders", {"min": lo, "max": hi}, m.group(0))
    for m in _HIGHLIGHTS.finditer(text):
        rel, num = (m.group(1), m.group(2)) if m.group(2) else (m.group(3), m.group(4))
        n = parse_number(num or "")
        if n is not None and not _scoped(text, m):
            lo, _ = _rel_bounds(rel or "at least", n)
            _add(found, "highlights", {"min": lo}, m.group(0))
    for m in _LETTER_FREQ.finditer(text):
        n = parse_number(m.group(3))
        if n is not None and not _scoped(text, m):
            lo, hi = _rel_bounds(m.group(2), n)
            _add(found, "letter_frequency", {"letter": m.group(1).lower(), "min": lo, "max": hi}, m.group(0))
    for m in _WORDS.finditer(text):
        if _scoped(text, m) or re.search(r"capital|caps\b|upper\s*-?\s*case|lower\s*-?\s*case|bold|italic|highlight",
                                         text[m.end():m.end() + 40], re.I):
            continue
        if m.group(1):
            n = parse_number(m.group(2))
            if n is None or m.group(1).lower() in ("exactly", "only"):
                continue
            lo, hi = _rel_bounds(m.group(1), n, _negated(text, m))
        elif m.group(3):
            lo, hi = parse_number(m.group(3)), parse_number(m.group(4))
        else:
            lo, hi = None, parse_number(m.group(5))
        if lo is None and hi is None:
            continue
        _add(found, "word_count", {"relation": "range", "min": lo, "max": hi}, m.group(0))
    for m in _SENTENCES.finditer(text):
        num = m.group(2) or m.group(3) or m.group(4)
        n = parse_number(num or "")
        if n is None or _scoped(text, m):
            continue
        rel = m.group(1) if m.group(2) else ("at most" if m.group(4) else "exactly")
        lo, hi = _rel_bounds(rel, n, _negated(text, m))
        _add(found, "sentence_count", {"relation": "range", "min": lo, "max": hi}, m.group(0))
    for m in _PARAGRAPHS.finditer(text):
        n = parse_number(m.group(2))
        if n is None or _scoped(text, m):
            continue
        lo, hi = _rel_bounds(m.group(1), n)
        sep = _PARA_SEP.search(text[m.end():m.end() + 80])
        _add(found, "paragraph_count", {"count": n, "min": lo, "max": hi, "separator": sep.group(1) if sep else None},
             m.group(0))
    for m in _BULLETS.finditer(text):
        n = parse_number(m.group(2))
        if n is None or _scoped(text, m):
            continue
        lo, hi = _rel_bounds(m.group(1), n)
        _add(found, "bullet_count", {"count": n, "min": lo, "max": hi}, m.group(0))
    for m in _SECTIONS.finditer(text):
        n = parse_number(m.group(1))
        if n is not None:
            _add(found, "section_count", {"count": n, "marker": m.group(2).capitalize()}, m.group(0))
    keywords: dict[str, int] = {}
    kw_spans: list[str] = []
    for rx in (_KEYWORD_FREQ, _KEYWORD_FREQ_2):
        for m in rx.finditer(text):
            n = parse_number(m.group(m.lastindex))
            word = m.group(1).strip().strip(".,;:!?").strip()
            if n is None or not word or _scoped(text, m) or _NEGATED_VERB.search(text[max(0, m.start() - 30):m.start()]):
                continue
            lo, _ = _rel_bounds(m.group(2) if m.lastindex and m.lastindex >= 3 and m.group(2) else "at least", n)
            keywords[word] = max(keywords.get(word, 0), lo or 1)
            kw_spans.append(m.group(0))
    for m in _KEYWORDS.finditer(text):
        if _scoped(text, m) or _NEGATED_VERB.search(text[max(0, m.start() - 30):m.start()]):
            continue
        tail = text[m.end():m.end() + 60]
        freq = re.match(rf"\W*(?:each\s+)?(?:(?:used|appearing|mentioned|appear)\s+)?(?:{_REL}\s+)?({_NUM[1:-1]}|once|twice|thrice)"
                        r"(?:\s+times)?\b", tail, re.I)
        k = 1
        if freq:
            n = parse_number(freq.group(2))
            k = (_rel_bounds(freq.group(1) or "at least", n)[0] or 1) if n else 1
        for w in _quoted_items(m.group(1)):
            keywords[w] = max(keywords.get(w, 0), k)
        kw_spans.append(m.group(0))
    if keywords:
        _add(found, "required_keywords", {"keywords": keywords}, " | ".join(kw_spans))
    forbidden: list[str] = []
    for m in _FORBIDDEN.finditer(text):
        if _scoped(text, m):
            continue
        forbidden += [w for w in _quoted_items(m.group(1)) if w.lower() not in [x.lower() for x in forbidden]]
    if forbidden:
        _add(found, "forbidden_keywords", {"keywords": forbidden}, "exclude " + ", ".join(forbidden))
    for rx in (_END, _END_SINGLE):
        m = rx.search(text)
        target = m.group(0)[:m.group(0).lower().find("with")] if m else ""
        if m and not re.search(r"\b(?:each|every)\b", _clause_before(text, m.start()), re.I) and not _PART_WORDS.search(target):
            phrase = m.group(1).strip()
            if phrase:
                _add(found, "closing_phrase", {"phrase": phrase, "normalise": True}, m.group(0))
                break
    m = _JSON.search(text)
    if m and not re.search(r"\b(?:each|every)\b", _clause_before(text, m.start()), re.I):
        _add(found, "json_only", {"allow_array": True}, m.group(0))
    for m in _LANGUAGE.finditer(text):
        lang = m.group(1).lower()
        ctx_after = text[m.end():m.end() + 40]
        if _scoped(text, m) or re.search(r"\b(?:as\s+well|also|too|and\s+(?:then\s+)?(?:in|translate))\b", ctx_after, re.I):
            continue
        if re.search(r"\b(?:repeat|translat|summary|summari[sz]e|version|each|both)\w*", _clause_before(text, m.start()), re.I):
            continue
        _add(found, "response_language", {"language": lang.capitalize()}, m.group(0))
    langs = {c.params["language"] for c in found.values() if c.kind == "response_language"}
    if len(langs) > 1:                                         # contradictory or bilingual instructions: no language rule
        found = {k: c for k, c in found.items() if c.kind != "response_language"}
    uncovered = sorted(k for k, rx in UNCOVERED_MARKERS.items() if rx.search(text))
    return list(found.values()), uncovered
