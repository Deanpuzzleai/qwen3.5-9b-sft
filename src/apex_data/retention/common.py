"""Shared pieces of the retention loaders (docs/DATA_PLAN.md sections 3.4, 6.3 and 7).

A loader turns raw source rows into *candidates*: plain dicts with the single-turn conversation that will be trained
(`messages` = [user, assistant]), the prompt text, the part, the upstream id and the provenance values, after the cheap
structural checks of its part. Every check is counted in a `Funnel` (n_in / n_dropped / n_out, in order). The builder
then measures exact token lengths, draws, screens and runs `render.filters.filter_episodes` on the candidates, through the
Episode, task and Verdict records built here.

System messages are not kept: the math and science rows have none, the chat rows' system messages are empty, and the IF
rows carry one generic sentence ("You are not allowed to use any tools.") that contradicts the agentic harnesses the
model is trained for; a retention row is therefore one user prompt and one answer with reasoning in every part.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex_data.schema import Episode, Verdict, task_id as make_task_id

MAX_WORKERS = 16                 # host shared with the benchmark evaluations (coordinator's cap)
CHARS_PER_TOKEN_CEILING = 6.0    # cheap pre-filter: a reasoning longer than cap x 6 characters cannot fit the token cap
QA_ROW_CAP = 12_288              # plan 6.3: rendered retention rows <= 12K tokens (FilterConfig.max_qa_tokens)
THINK_CAP_LONG, THINK_CAP_SHORT = 10_240, 6_144   # plan 3.4: math and science <= 10K, IF and chat <= 6K reasoning tokens


@dataclass(frozen=True)
class PartSpec:
    part: str                    # schema.RETENTION_PARTS
    source: str                  # schema.SOURCES code
    hf_repo: str
    split: str                   # the dataset file or split the rows come from
    n_sft: int                   # SFT rows (plan 3.4, revision 4.3.3)
    think_cap: int               # reasoning tokens
    generator: str               # teacher named on the dataset card (rows that name their model override it)
    label_provenance: str        # schema.LABEL_PROVENANCE
    verifier: str                # Verdict.verifier of the row's label
    label_source: str            # "upstream" (the dataset's own check) or "ours"
    dataset_license: str
    family: str                  # schema.FAMILIES
    check: str                   # the M2.3 check this part's rows pass (for the manifest)


PARTS: dict[str, PartSpec] = {
    "math": PartSpec("math", "nm_math", "nvidia/Nemotron-SFT-Math-v4", "train (source AoPS, subset cot)", 630,
                     THINK_CAP_LONG, "deepseek-ai/DeepSeek-V4-Pro (High)", "reference", "reference_match", "upstream",
                     "CC-BY-4.0", "qa", "final answer verified against the Nemotron-Math-v2 reference upstream; a "
                     "\\boxed{} final answer present; our normalised match recorded"),
    "if": PartSpec("if", "nm_ifchat", "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "instruction_following",
                   435, THINK_CAP_SHORT, "openai/GPT-OSS-120B", "reference", "constraint", "ours",
                   "CC-BY-4.0 AND ODC-BY-1.0", "chat", "every constraint the extractor reads from the prompt passes "
                   "apex_data.prompts.checkers"),
    "chat": PartSpec("chat", "nm_ifchat", "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "chat", 285,
                     THINK_CAP_SHORT, "zai-org/GLM-5", "format", "format", "upstream", "CC-BY-4.0 AND ODC-BY-1.0", "chat",
                     "reward-model selected upstream; non-empty English lmarena prompt, closed reasoning, non-empty answer"),
    "sci_mcq": PartSpec("sci_mcq", "nm_sci", "nvidia/Nemotron-Science-v1", "MCQ", 275, THINK_CAP_LONG,
                        "openai/GPT-OSS-120B", "format", "format", "ours", "CC-BY-4.0", "qa",
                        "answer letter present, one of the listed options, and the same letter the reasoning concludes "
                        "(the split ships no reference answers)"),
    "sci_rqa": PartSpec("sci_rqa", "nm_sci", "nvidia/Nemotron-Science-v1", "RQA", 125, THINK_CAP_LONG,
                        "openai/GPT-OSS-120B", "format", "format", "ours", "CC-BY-4.0", "qa",
                        "single-question prompt, exactly one non-empty \\boxed{} final answer (format only: the split "
                        "ships no reference answers)"),
}
PART_ORDER = ("math", "if", "chat", "sci_mcq", "sci_rqa")
RESERVE_TOTAL = 600              # stage-2 replay reserve (plan 11), same part proportions as the SFT rows


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` over `weights` (ties broken by key order)."""
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), list(weights).index(k)))[:total - sum(out.values())]:
        out[k] += 1
    return out


def reserve_quota(total: int = RESERVE_TOTAL) -> dict[str, int]:
    return apportion(total, {p: PARTS[p].n_sft for p in PART_ORDER})


# ------------------------------------------------------------------------------------------------------- funnel
class Funnel:
    """Ordered survival counts: every stage sees the rows that passed the stages before it."""

    def __init__(self, stages: Iterable[str]) -> None:
        self.stages = list(stages)
        self.n_in: Counter = Counter()
        self.n_out: Counter = Counter()

    def run(self, checks: Iterable[tuple[str, Callable[[], bool]]]) -> str | None:
        """Evaluate (stage, predicate) pairs in order; returns the first failing stage or None."""
        for stage, pred in checks:
            self.n_in[stage] += 1
            if not pred():
                return stage
            self.n_out[stage] += 1
        return None

    def merge(self, other: Funnel) -> None:
        self.n_in.update(other.n_in)
        self.n_out.update(other.n_out)

    def table(self) -> list[dict[str, Any]]:
        return [{"stage": s, "n_in": self.n_in[s], "n_dropped": self.n_in[s] - self.n_out[s], "n_out": self.n_out[s]}
                for s in self.stages]

    def to_dict(self) -> dict[str, Any]:
        return {"stages": self.stages, "n_in": dict(self.n_in), "n_out": dict(self.n_out)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Funnel:
        f = cls(d["stages"])
        f.n_in.update(d["n_in"])
        f.n_out.update(d["n_out"])
        return f


# ------------------------------------------------------------------------------------------------------ text checks
_THINK_TAG = re.compile(r"</?think>")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ASCII_WORD = re.compile(r"[A-Za-z]+")
_ENGLISH = frozenset("the a an and or of to in is are was be it that this for with on as at by from what how why "
                     "which who when where can could would should do does did i you we my your me please write "
                     "explain tell give make about if not no have has".split())
_OTHER = frozenset("el la los las y que de del en es por para con una le les des et est pour dans qui une der die das "
                   "und ist nicht ein eine mit auf il di che non per sono o os da em um uma não se como het een naar "
                   "deze dit voor niet zijn wat hoe ik je moi mon ma tu".split())


def has_think_tags(*texts: str | None) -> bool:
    return any(_THINK_TAG.search(t or "") for t in texts)


def is_english(text: str) -> bool:
    """Prompt-language heuristic: >= 97 percent of the letters ASCII, and more English than other-language function
    words (prompts of fewer than four words need ASCII letters only), over the whole prompt and over its first 300
    characters (the instruction usually opens the prompt; pasted code or quoted English text follows it)."""
    return _english(text) and _english(text[:300])


def _english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.97:
        return False
    words = [w.lower() for w in _ASCII_WORD.findall(text)]
    if len(words) < 4:
        return True
    en = sum(w in _ENGLISH for w in words)
    other = sum(w in _OTHER for w in words)
    return en >= 1 and en > other


def boxed_contents(text: str) -> list[str]:
    """Contents of every \\boxed{...} (and \\fbox{...}) with balanced braces, in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text or ""):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append(text[m.end():i - 1].strip())
    return out


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


# --------------------------------------------------------------------------------------------- parallel file reading
def jsonl_chunks(path: Path, n_chunks: int) -> list[tuple[str, int, int]]:
    """Byte ranges that split a JSONL file into ~n_chunks pieces; a reader owns the lines that *start* in its range."""
    size = path.stat().st_size
    step = max(1, math.ceil(size / n_chunks))
    return [(str(path), a, min(size, a + step)) for a in range(0, size, step)]


def iter_jsonl_range(path: str, start: int, end: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """(byte offset, record) for every line that starts in [start, end)."""
    with open(path, "rb") as f:
        if start:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        pos = f.tell()
        while pos < end:
            line = f.readline()
            if not line:
                break
            if line.strip():
                yield pos, json.loads(line)
            pos += len(line)


def parallel_map(fn: Callable[..., Any], args: list[Any], workers: int = MAX_WORKERS) -> list[Any]:
    """`fn(*a)` for every a, in order, on at most `workers` processes (MAX_WORKERS caps it)."""
    workers = max(1, min(workers, MAX_WORKERS, len(args)))
    if workers == 1:
        return [fn(*a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, *zip(*args)))


def load_jsonl(chunk_fn: Callable[..., tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]], path: Path,
               stages: tuple[str, ...], revision: str, workers: int, n_chunks: int
               ) -> tuple[list[dict[str, Any]], Funnel, dict[str, int]]:
    """Run a loader's chunk function over byte ranges of a JSONL file; merge candidates, funnels and side counts."""
    if not path.exists():
        raise FileNotFoundError(path)
    funnel = Funnel(stages)
    out: list[dict[str, Any]] = []
    side: Counter = Counter()
    for cands, fd, extra in parallel_map(chunk_fn, [(*c, revision) for c in jsonl_chunks(path, n_chunks)], workers):
        out.extend(cands)
        funnel.merge(Funnel.from_dict(fd))
        side.update(extra)
    return out, funnel, dict(sorted(side.items()))


# ------------------------------------------------------------------------------------------------ candidate records
def candidate(part: str, upstream_id: str, prompt: str, answer: str, reasoning: str, *, hf_revision: str,
              generator: str | None = None, row_license: str | None = None, origin: dict[str, Any] | None = None,
              **meta: Any) -> dict[str, Any]:
    """The candidate dict every loader emits (JSON-serialisable)."""
    spec = PARTS[part]
    return {"part": part, "task_id": make_task_id(spec.source, upstream_id), "upstream_id": upstream_id,
            "source": spec.source, "hf_repo": spec.hf_repo, "hf_revision": hf_revision,
            "generator": generator or spec.generator, "row_license": row_license, "prompt": prompt,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "reasoning_content": reasoning, "content": answer}],
            "reasoning_chars": len(reasoning), "origin": origin or {}, "meta": meta, "split": "train", "screens": {}}


def episode_record(c: dict[str, Any], created_at: str) -> dict[str, Any]:
    """schema.Episode (as a dict) of a candidate: scaffold "chat", one assistant turn, exit "submit"."""
    ep = Episode(episode_id=c["task_id"], task_id=c["task_id"], round="teacher", policy=c["generator"], sample_index=0,
                 scaffold="chat", messages=c["messages"], tools=None, exit_reason="submit", steps=1, output_tokens=0,
                 wall_sec=0.0, patch=None, request_log=None, created_at=created_at,
                 extra={"constraints": c["meta"].get("constraints") or [], "retention_part": c["part"]})
    ep.validate()
    return {k: getattr(ep, k) for k in Episode.__dataclass_fields__}


def task_record(c: dict[str, Any]) -> dict[str, Any]:
    """The task dict `filter_episodes` reads: licence, screens (dedup group, D1, D5') and the row provenance."""
    spec = PARTS[c["part"]]
    screens = c.get("screens") or {}
    prov = {"hf_repo": spec.hf_repo, "generator": c["generator"], "label_provenance": spec.label_provenance,
            "retention_part": c["part"], "category": c["meta"].get("topic")}
    return {"task_id": c["task_id"], "source": spec.source, "family": spec.family, "language": None, "repo": None,
            "upstream_id": c["upstream_id"], "hf_revision": c["hf_revision"],
            "licence": {"dataset": spec.dataset_license, "row": c.get("row_license"), "status": "ok"},
            "screens": {k: v for k, v in screens.items() if k in ("dedup_group", "d1", "d4", "d5")}, "split": c["split"],
            "extra": {"provenance": prov}}


def verdict_record(c: dict[str, Any], graded_at: str, checker_version: str) -> dict[str, Any]:
    """The label of a candidate as a schema.Verdict: upstream reference verification (math), our constraint checker
    (IF), or the format checks of the part (chat, MCQ, RQA)."""
    spec = PARTS[c["part"]]
    v = Verdict(episode_id=c["task_id"], task_id=c["task_id"], r_pass=1,
                runs=[{"pass": True, "check": spec.check, **({"ref_match": c["meta"]["ref_match"]}
                                                              if "ref_match" in c["meta"] else {})}],
                tamper_flags=[], verifier=spec.verifier, verifier_version=checker_version, label_source=spec.label_source,
                graded_at=graded_at, extra={"retention_part": c["part"]})
    v.validate()
    return {k: getattr(v, k) for k in Verdict.__dataclass_fields__}
