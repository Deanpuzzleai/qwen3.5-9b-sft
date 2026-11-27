"""Dev-set evaluation (docs/DATA_PLAN.md sections 1 and 10; docs/EXECUTION_PLAN.md U3.4): the item plan, the three slice
runners, per-item records, the executor, run summaries and the do-no-harm guards. `scripts/data/run_dev_eval.py` is the
command line; this module holds everything it runs so the tests can drive it with fake runners.

Plan. Items come from the frozen manifests `data/manifests/dev_{swe,terminal,qa}.json` (devset/common.py sample plan):
every item carries its full-set samples `{index, seed, reduced}`. Sizes:
  full     every sample: SWE 100 x 4, terminal 40 x 3, QA 228 x 2 (+ AIME 2025 30 x 2, report only, slice `qa_aime`)
  reduced  the samples flagged `reduced` = the first samples of the full set with the same ids, indices and seeds
           (SWE 100 x 2, terminal 40 x 2, QA 228 x 1; no AIME), so a reduced run pairs item for item with a full run
  smoke    the pilot's smoke: the first 20 SWE / 10 terminal / 50 QA items, sample index 0 only
Item order inside a slice is the manifest order (a seeded permutation, so any prefix is a random subsample); the QA
selection items interleave GPQA Diamond and reasoning-gym by position, so any QA prefix keeps their 198 : 30 shares.
Execution order is sample index first, so an interrupted full run has finished the reduced samples before the rest.

Slice harnesses (each identical for every model version):
  swe       mini-swe-agent through the U3.3 rollout driver (`apex_data.rollout.mini_swe`) with the SWE-bench Pro lock's
            agent config (the lock's config file and sampling via the driver's agent_config; step_limit 250, 16,384
            tokens per turn, wall_time_limit_seconds 2940, no total token cap) and the manifest seed per sample; graded
            by the SWE-smith pristine-replay grader (`apex_data.grade.swesmith.grade_patch`, two runs). As in the
            SWE-bench Pro protocol the final diff is graded however the episode ended (a step- or time-capped episode is
            `budget_exhausted` and still graded), so the grader's no_submit flag is not applied; every other tamper flag
            is. pass@1 per sample.
  terminal  Terminus-2 through the U3.3 rollout driver (`apex_data.rollout.terminus`: one `harbor trial start` per
            (task, sample), since seeds are per sample) on the TermGrade task pool, with the Terminal-Bench lock's agent,
            parser, summarisation, model_info, sampling and 81,920 tokens per turn (grade/harbor_tests.py
            terminus2_job_config), no turn cap (Terminus-2's default), no output cap, task.toml resources. TermGrade
            declares no agent timeout, so the dev driver sets one (--terminal-agent-timeout-sec, default 1800 s, recorded
            in metadata). Success = verifier reward 1 and the confirmed task_complete handshake
            (`harbor_tests.read_trial`); the reward-only outcome and the strict stage-2 label are kept in extra.
  qa        one completion per sample with the HLE prediction prompt (the official SYSTEM_PROMPT, system turn + a single
            text part, as the HLE script sends it; no sampling parameters, so the serving defaults apply as in the HLE
            protocol; the manifest seed per request; 32,768 completion tokens), judged by the HLE judge (the official
            JUDGE_PROMPT, the lock's judge model, structured output with the lock's fields, 4,096 completion tokens).
            Prompt and judge text are read at run time from the pinned HLE checkout (commit checked against
            configs/eval/hle.yaml); no HLE data is read. Metrics: accuracy, empty-content rate, think length and HLE
            grammar compliance; a deterministic cross-check (MCQ letter, reasoning-gym scorer, AIME integer) is reported
            next to the judge's verdict. Judge tokens and estimated spend are recorded per item and per run.

Records (one JSON line per item in results/dev/<version>/<run_id>/per_item.jsonl; the last line of a key wins): the
shared evaluation schema of apex_eval.common.record plus top-level `slice`, `sample_index`, `seed` (the pairing keys
scripts/summarize.py dev reads) and `turn_metrics` {n_turns, n_format_failures, n_truncated, think_tokens, think_chars}.
`benchmark` is `dev_<slice>`; `resolved` is the verdict; `run_status` uses the shared vocabulary (environment failures,
budget exhaustion, agent errors, verifier and scoring failures are counted separately; nothing is dropped).

Turn metrics. A turn is one model completion. Format failure = a turn the scaffold could not use: SWE, a mini-swe-agent
format-error reply (no or malformed tool call, which includes turns cut at the token limit); terminal, a Terminus-2
"parsing errors" or "exceeded the output limit" reply; QA, a completion without the parts of the HLE grammar the
protocol consumes (an Answer line and a Confidence value) or cut at the token limit; the Explanation label is reported
separately. Think length = tokens of the turn's reasoning text under the base model's tokenizer (characters when the
tokenizer is unavailable; the unit is recorded).

Guards (docs/DATA_PLAN.md section 1), against a baseline run on the shared (slice, task_id, sample_index) items:
  scores        per scored slice (swe, terminal, qa): point difference >= -3 pp and the lower bound of the paired 95 percent
                interval (task-clustered bootstrap, computed by scripts/summarize.py dev) >= -resolution, with resolutions
                SWE 7 / terminal 12 / QA 6 pp when both runs are full sets and 10 / 15 / 8 pp otherwise
  format        pooled format-failure rate over all turns of the scored slices <= 2x the baseline's
  think length  per scored slice, mean think tokens per turn: QA <= 2x the baseline's; SWE <= 761 and terminal <= 4,248
                tokens per turn, 2x the frozen SFT training rows' mean (THINK_MAX_TOKENS_PER_TURN, revision 4.3.20)
Reported flags, not guards (docs/DATA_PLAN_CHANGELOG.md 4.3.18): a pooled format-failure rate above 1 percent of turns, a
slice's mean think below 0.5x the baseline's (collected under `flags`), and the limit-hit comparison (revision 4.3.9).
The opencode guards come from scripts/dev/opencode_smoke.py, not from here. A smoke-size run reports guards for
information only (`decision_grade: false`).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from apex_data import paths
from apex_data.devset.common import MANIFEST_NAMES
from apex_eval.common import (AGENT_ERROR, BUDGET_EXHAUSTED, ENVIRONMENT_FAILURE, MISSING, OK, RUN_FAILURE_STATUSES,
                              SCORING_FAILURE, VERIFIER_FAILURE, failure_summary, git_head, record)

SLICES = ("swe", "terminal", "qa")                     # scored slices (selection rule, macro-average, guards)
SIZES = ("full", "reduced", "smoke")
VERSIONS = ("base", "sft", "stage2_a", "stage2_b", "pilot")   # stage-2 arms (a) filtered, (b) control; pilot = the SFT pilot adapter
SMOKE_ITEMS = {"swe": 20, "terminal": 10, "qa": 50}    # docs/EXECUTION_PLAN.md U4.2
RESOLUTIONS_PP = {"full": {"swe": 7.0, "terminal": 12.0, "qa": 6.0}, "reduced": {"swe": 10.0, "terminal": 15.0, "qa": 8.0}}
GUARD_POINT_PP = 3.0
FORMAT_FAILURE_REL = 2.0               # guard: pooled format-failure rate <= 2x the baseline's
THINK_RATIO_MAX = 2.0                  # guard (QA): the slice's mean think per turn <= 2x the baseline's
# guard (agent slices, revision 4.3.20): mean think tokens per turn <= 2x the frozen SFT train rows' mean reasoning tokens per
# trained assistant turn (f9ffb40: SWE 380.4, terminal 2,124; training files only, set before any SFT dev number was used)
THINK_MAX_TOKENS_PER_TURN = {"swe": 761, "terminal": 4248}
FORMAT_FAILURE_FLAG = 0.01             # reported flag, not a guard (revision 4.3.18): pooled rate above 1 percent of turns
THINK_RATIO_FLAG_BELOW = 0.5           # reported flag, not a guard (revision 4.3.18): mean think below 0.5x the baseline's
QA_SELECTION = ("gpqa_diamond", "reasoning_gym")
QA_REPORT = ("aime25",)
# The fixed dev protocol, identical for every dev run (base, sft, stage2_a, stage2_b, pilot; full and reduced) so that runs
# pair cleanly: 24 + 12 + 16 = 52 concurrent sequences stay under the locked vLLM max_num_seqs 64 (above it requests queue,
# turns slow down and TermGrade trials hit their timeouts, which would distort the comparison). A run with other values is
# not reportable, and --baseline refuses a pair whose protocols differ.
DEFAULT_CONCURRENCY = {"swe": 24, "terminal": 12, "qa": 16}
TERMINAL_AGENT_TIMEOUT_SEC = 1800.0   # TermGrade task.toml declares no agent timeout
QA_REQUEST_TIMEOUT_SEC = 2400.0
OPENAI_URL = "https://api.openai.com/v1"
JUDGE_BASE_URL_ENV = "APEX_DEV_JUDGE_BASE_URL"         # SMOKE TESTS ONLY: point the judge at a mock server
# o3-mini list prices (USD per million tokens) used for the spend estimate; overridable on the command line
JUDGE_USD_PER_MTOK = {"input": 1.10, "output": 4.40}
SWE_PARSE_ERROR_PREFIX = "Tool call error"             # mini-swe-agent toolcall format-error template, first line
TERMINAL_PARSE_ERROR_PREFIX = "Previous response had parsing errors"
TERMINAL_OUTPUT_LIMIT_PREFIX = "ERROR!! NONE of the actions you just requested were performed"
LABEL_RUN = "apex.devev_run"


class SliceBlocked(RuntimeError):
    """A slice cannot run (a dependency, pool, checkout or credential is missing); raised by a runner's preflight."""


def slug(s: str, limit: int = 80) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_.")
    if len(out) > limit:
        out = out[: limit - 9] + "_" + hashlib.sha256(s.encode()).hexdigest()[:8]
    return out or "x"


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


# ==== plan ===========================================================================================================
@dataclass(frozen=True)
class PlanItem:
    slice: str                 # swe | terminal | qa | qa_aime
    task_id: str               # swesmith:<instance> | tgr_env:<task> | <dev_id> (gpqa:<id>, rg:..., aime25:<id>)
    upstream_id: str
    sample_index: int
    seed: int
    order: int                 # position of the item inside its slice (manifest order)
    group: str | None = None   # repository | difficulty band | QA sub-slice
    meta: tuple[tuple[str, Any], ...] = ()

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.slice, self.task_id, self.sample_index)

    @property
    def priority(self) -> tuple[int, int, int]:
        return (self.sample_index, ("swe", "terminal", "qa", "qa_aime").index(self.slice), self.order)

    def meta_dict(self) -> dict[str, Any]:
        return dict(self.meta)


def _manifest(manifests_dir: Path, slice_name: str) -> dict[str, Any]:
    p = Path(manifests_dir) / MANIFEST_NAMES[slice_name]
    if not p.exists():
        raise SliceBlocked(f"dev manifest {p} does not exist (WP6, M1.4)")
    m = json.loads(p.read_text())
    if not m.get("frozen"):
        raise SliceBlocked(f"dev manifest {p} is not frozen")
    return m


def interleave(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge ordered groups so every prefix keeps their shares: an entry at position i of a group of n sorts at
    (i + 0.5) / n; ties keep the group order given."""
    keyed = [((i + 0.5) / len(items), gi, i, it) for gi, items in enumerate(groups.values()) for i, it in enumerate(items)]
    return [it for *_, it in sorted(keyed, key=lambda t: t[:3])]


def slice_entries(manifests_dir: Path, slice_name: str) -> dict[str, list[dict[str, Any]]]:
    """{plan slice: ordered manifest entries} for one manifest (`qa` also yields `qa_aime`). Each entry is normalised to
    {task_id, upstream_id, group, samples, meta}."""
    m = _manifest(manifests_dir, slice_name)
    if slice_name == "swe":
        return {"swe": [{"task_id": it["task_id"], "upstream_id": it["upstream_id"], "group": it.get("repo"),
                         "samples": it["samples"], "meta": {"language": it.get("language"), "repo": it.get("repo")}}
                        for it in m["items"]]}
    if slice_name == "terminal":
        return {"terminal": [{"task_id": it["task_id"], "upstream_id": it["upstream_id"], "group": it.get("band"),
                              "samples": it["samples"], "meta": {"band": it.get("band")}} for it in m["items"]]}
    sl = m["slices"]

    def entries(name: str) -> list[dict[str, Any]]:
        return [{"task_id": e["dev_id"], "upstream_id": e["dev_id"], "group": name, "samples": e["samples"],
                 "meta": {"qa_slice": name}} for e in sl.get(name, {}).get("items", [])]

    return {"qa": interleave({name: entries(name) for name in QA_SELECTION}),
            "qa_aime": [e for name in QA_REPORT for e in entries(name)]}


def parse_limits(spec: str | None) -> dict[str, int]:
    """'2' -> every slice 2; 'swe=2,terminal=2,qa=3' -> per slice (qa also limits qa_aime)."""
    if not spec:
        return {}
    spec = str(spec).strip()
    if spec.isdigit():
        return {s: int(spec) for s in SLICES}
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, _, n = part.partition("=")
        if name.strip() not in SLICES or not n.strip().isdigit():
            raise ValueError(f"bad --limit-per-slice entry {part!r} (want N or swe=N,terminal=N,qa=N)")
        out[name.strip()] = int(n)
    return out


def build_plan(manifests_dir: Path, size: str, slices: Iterable[str] = SLICES, limits: dict[str, int] | None = None,
               task_ids: Iterable[str] | None = None) -> list[PlanItem]:
    """Plan items for `size` (module docstring), in execution order (sample index first)."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    limits, wanted = limits or {}, set(task_ids or ())
    items: list[PlanItem] = []
    for name in slices:
        for plan_slice, entries in slice_entries(manifests_dir, name).items():
            if plan_slice == "qa_aime" and size != "full":
                continue
            if size == "smoke":
                entries = entries[: SMOKE_ITEMS[name]]
            if limits.get(name):
                entries = entries[: limits[name]]
            if wanted:
                entries = [e for e in entries if e["task_id"] in wanted or e["upstream_id"] in wanted]
            for order, e in enumerate(entries):
                for s in e["samples"]:
                    keep = True if size == "full" else bool(s.get("reduced")) if size == "reduced" else int(s["index"]) == 0
                    if keep:
                        items.append(PlanItem(plan_slice, e["task_id"], e["upstream_id"], int(s["index"]), int(s["seed"]), order,
                                              e.get("group"), tuple(sorted((e.get("meta") or {}).items()))))
    return sorted(items, key=lambda it: it.priority)


def plan_counts(items: list[PlanItem]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for it in items:
        c = out.setdefault(it.slice, {"items": 0, "tasks": 0})
        c["items"] += 1
    for sl in out:
        out[sl]["tasks"] = len({it.task_id for it in items if it.slice == sl})
    return out


# ==== records ========================================================================================================
def make_record(item: PlanItem, *, run_status: str, resolved: bool | None, turn_metrics: dict[str, Any] | None = None,
                harness_status: str | None = None, error_reason: str | None = None, time_sec: float | None = None,
                input_tokens: int | None = None, output_tokens: int | None = None, reasoning_tokens: int | None = None,
                artifacts: str | None = None, **extra: Any) -> dict[str, Any]:
    """One per-item record: apex_eval.common.record plus the pairing keys and turn metrics (module docstring)."""
    rec = record(f"dev_{item.slice}", item.task_id, run_status=run_status, sample=str(item.sample_index), group=item.group,
                 resolved=resolved, score=None if resolved is None else float(bool(resolved)), max_score=1.0,
                 harness_status=harness_status, error_reason=error_reason, time_sec=time_sec, input_tokens=input_tokens,
                 output_tokens=output_tokens, reasoning_tokens=reasoning_tokens, artifacts=artifacts,
                 upstream_id=item.upstream_id, **item.meta_dict(), **extra)
    return {"slice": item.slice, "task_id": item.task_id, "sample_index": item.sample_index, "seed": item.seed,
            **{k: v for k, v in rec.items() if k != "task_id"},
            "turn_metrics": turn_metrics or empty_turn_metrics()}


def failure_record(item: PlanItem, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return make_record(item, run_status=status, resolved=False, error_reason=reason[:1000], **extra)


def record_key(r: dict[str, Any]) -> tuple[str, str, int]:
    return (str(r["slice"]), str(r["task_id"]), int(r["sample_index"]))


def read_records(path: Path) -> tuple[dict[tuple[str, str, int], dict[str, Any]], int]:
    """(key -> last record, number of superseded lines); a truncated last line from a crash is ignored."""
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    n_superseded = 0
    if not Path(path).exists():
        return out, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = record_key(r)
        n_superseded += k in out
        out[k] = r
    return out, n_superseded


class RecordWriter:
    """Append-only per_item.jsonl (fsynced per line) with the METRIC_KEYS mirror into the run's metrics.jsonl."""

    def __init__(self, path: Path, metric: Callable[[dict[str, Any]], None] | None = None):
        self.path, self.metric = Path(path), metric
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, rec: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self.metric:
            self.metric({"event": "item", "slice": rec["slice"], "task_id": rec["task_id"], "sample_index": rec["sample_index"],
                         "resolved": rec["resolved"], "run_status": rec["run_status"], "time_sec": rec["time_sec"],
                         "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"]})


# ==== turn metrics ===================================================================================================
def empty_turn_metrics() -> dict[str, Any]:
    return {"n_turns": 0, "n_format_failures": 0, "n_truncated": 0, "think_tokens": 0, "think_chars": 0, "think_unit": None}


class ThinkCounter:
    """Reasoning length in tokens of the base model's tokenizer (loaded lazily from paths.BASE_MODEL_DIR, local files
    only); None tokens when it cannot be loaded, in which case summaries fall back to characters."""

    def __init__(self, model_dir: Path | None = paths.BASE_MODEL_DIR):
        self.model_dir = model_dir
        self._tok: Any = None
        self._failed = model_dir is None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._tok is None and not self._failed:
                try:
                    from transformers import AutoTokenizer  # noqa: PLC0415 - heavy import, only when counting
                    self._tok = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
                except Exception:  # noqa: BLE001 - recorded as unit "chars"
                    self._failed = True
        return self._tok

    @property
    def unit(self) -> str:
        return "tokens" if self._load() is not None else "chars"

    def count(self, text: str | None) -> int | None:
        if not text:
            return 0
        tok = self._load()
        return None if tok is None else len(tok.encode(text, add_special_tokens=False))


def _think(texts: Iterable[str | None], counter: ThinkCounter) -> dict[str, Any]:
    texts = [t or "" for t in texts]
    counts = [counter.count(t) for t in texts]
    tokens = None if any(c is None for c in counts) else sum(c for c in counts if c is not None)
    return {"think_tokens": tokens, "think_chars": sum(len(t) for t in texts),
            "think_unit": "tokens" if tokens is not None else "chars"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):   # OpenAI content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def swe_turn_metrics(episode: dict[str, Any], counter: ThinkCounter) -> dict[str, Any]:
    """Turns, format failures (mini-swe-agent format-error replies), truncated turns and reasoning length of an Episode."""
    msgs = episode.get("messages") or []
    assistant = [m for m in msgs if m.get("role") == "assistant"]
    extra = episode.get("extra") or {}
    n_format = sum(1 for m in msgs if m.get("role") == "user" and (m.get("format_error")
                   or _text(m.get("content")).lstrip().startswith(SWE_PARSE_ERROR_PREFIX)))
    if isinstance(extra.get("n_format_errors"), int):
        n_format = max(n_format, int(extra["n_format_errors"]))
    finish = extra.get("finish_reasons") or [t.get("finish_reason") for t in extra.get("turns") or []]
    return {"n_turns": len(assistant), "n_format_failures": n_format, "n_truncated": sum(1 for f in finish if f == "length"),
            **_think((m.get("reasoning_content") for m in assistant), counter)}


def terminal_turn_metrics(result: dict[str, Any] | None, trajectory: dict[str, Any] | None, counter: ThinkCounter) -> dict[str, Any]:
    """From a Harbor trial: turns and format failures from Terminus-2's chat history (result.json
    agent_result.metadata.all_messages, which also holds the output-limit turns the trajectory omits), else from the
    trajectory's agent steps; reasoning from the trajectory's agent steps."""
    steps = [s for s in (trajectory or {}).get("steps") or [] if s.get("source") == "agent" and not s.get("is_copied_context")]
    history = (((result or {}).get("agent_result") or {}).get("metadata") or {}).get("all_messages")
    if isinstance(history, list) and history:
        users = [_text(m.get("content")).lstrip() for m in history if m.get("role") == "user"]
        n_turns = sum(1 for m in history if m.get("role") == "assistant")
        n_parse = sum(1 for u in users if u.startswith(TERMINAL_PARSE_ERROR_PREFIX))
        n_trunc = sum(1 for u in users if u.startswith(TERMINAL_OUTPUT_LIMIT_PREFIX))
        source = "all_messages"
    else:
        n_turns, n_trunc, source = len(steps), 0, "trajectory"
        n_parse = sum(1 for s in steps for r in ((s.get("observation") or {}).get("results") or [])
                      if _text(r.get("content")).lstrip().startswith(TERMINAL_PARSE_ERROR_PREFIX))
    return {"n_turns": n_turns, "n_format_failures": n_parse + n_trunc, "n_truncated": n_trunc, "source": source,
            **_think((s.get("reasoning_content") for s in steps), counter)}


_LABEL = r"^[ \t>#*_-]*(?:\*\*|__)?[ \t]*{name}[ \t]*(?:\*\*|__)?[ \t]*[:：][ \t]*(?:\*\*|__)?[ \t]*(?P<rest>.*)$"
_RE_EXPL = re.compile(_LABEL.format(name="Explanation"), re.I | re.M)
_RE_ANS = re.compile(_LABEL.format(name=r"(?:Exact |Final )?Answer"), re.I | re.M)
_RE_CONF = re.compile(_LABEL.format(name="Confidence"), re.I | re.M)
_RE_NUM = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%?")


def hle_grammar(content: str | None) -> dict[str, Any]:
    """The HLE response grammar (Explanation / Answer / Confidence lines): which parts are present, the answer text
    (the Answer line, or the next non-empty line when it is empty) and the confidence value. `compliant` = the parts
    the HLE protocol consumes are there (an Answer line with text and a Confidence line with a 0-100 value);
    `all_parts` also requires the Explanation label (reported separately)."""
    text = content or ""
    m_ans = _RE_ANS.search(text)
    answer = None
    if m_ans:
        answer = m_ans.group("rest").strip().strip("*_").strip()
        if not answer:
            nxt = [ln.strip() for ln in text[m_ans.end():].splitlines() if ln.strip()]
            answer = nxt[0].strip("*_ ") if nxt and not _RE_CONF.match(nxt[0]) else ""
    m_conf = _RE_CONF.search(text)
    conf = None
    if m_conf:
        num = _RE_NUM.search(m_conf.group("rest"))
        conf = float(num.group(1)) if num else None
    has_expl = bool(_RE_EXPL.search(text))
    compliant = bool(answer) and conf is not None and 0 <= conf <= 100
    return {"explanation": has_expl, "answer": answer or None, "confidence": conf, "compliant": compliant,
            "all_parts": compliant and has_expl}


def mcq_letter(answer: str | None, n_options: int = 4) -> str | None:
    """The chosen letter of an MCQ answer line ('C', 'C.', '(C)', '**C**', 'C) text', 'Option C')."""
    if not answer:
        return None
    letters = "ABCDEFGHIJ"[:n_options]
    s = answer.strip().strip("*_`").strip()
    m = re.match(rf"^\(?([{letters}])\)?(?:[.):\s]|$)", s) or re.search(rf"\b(?:option|choice)\s*\(?([{letters}])\b", s, re.I)
    if m:
        return m.group(1).upper()
    found = re.findall(rf"\b([{letters}])\b", s)
    return found[0] if len(set(found)) == 1 else None


# ==== HLE prompt and judge ===========================================================================================
@dataclass(frozen=True)
class HLEAssets:
    system_prompt: str
    judge_prompt: str
    judge_model: str
    max_completion_tokens: int
    judge_max_completion_tokens: int
    code_commit: str

    def fingerprint(self) -> dict[str, Any]:
        """What metadata records (hashes, never the text)."""
        return {"hle_code_commit": self.code_commit, "judge_model": self.judge_model,
                "max_completion_tokens": self.max_completion_tokens, "judge_max_completion_tokens": self.judge_max_completion_tokens,
                "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode()).hexdigest(),
                "judge_prompt_sha256": hashlib.sha256(self.judge_prompt.encode()).hexdigest()}


def module_constant(path: Path, name: str) -> str:
    """A module-level string constant of a Python file, read with `ast` (the file is never imported or executed)."""
    with warnings.catch_warnings():   # the official files carry invalid escape sequences; the value is what Python evaluates
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(Path(path).read_text(), filename=str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            raise SliceBlocked(f"{path.name}: {name} is not a plain string literal")
    raise SliceBlocked(f"{path.name}: no module-level {name}")


def load_hle_assets(code_dir: Path | None = None, lock_path: Path | None = None, check_commit: bool = True) -> HLEAssets:
    """The HLE prediction prompt and judge prompt from the pinned official scripts, plus the lock's judge settings."""
    from apex_eval import hle as hle_eval  # noqa: PLC0415 - paths and lock of the HLE runner

    code_dir = Path(code_dir or hle_eval.HLE_CODE)
    lock = yaml.safe_load(Path(lock_path or hle_eval.LOCK).read_text())
    commit = lock["source"]["evaluation_code_commit"]
    if check_commit:
        head = git_head(code_dir.parent)
        if head != commit:
            raise SliceBlocked(f"HLE checkout at {head!r}, lock says {commit} (scripts/setup_eval_env.sh)")
    proto = lock["protocol"]
    judge_tokens = re.search(r"max_completion_tokens (\d+)", str(proto.get("judge", "")))
    return HLEAssets(system_prompt=module_constant(code_dir / "run_model_predictions.py", "SYSTEM_PROMPT"),
                     judge_prompt=module_constant(code_dir / "run_judge_results.py", "JUDGE_PROMPT"),
                     judge_model=str(proto["judge_model"]), max_completion_tokens=int(proto["max_completion_tokens"]),
                     judge_max_completion_tokens=int(judge_tokens.group(1)) if judge_tokens else 4096, code_commit=commit)


def hle_messages(system_prompt: str, question: str) -> list[dict[str, Any]]:
    """The HLE prediction script's message format for a text question: system turn + one text content part."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": question}]}]


# structured output of the HLE judge (configs/eval/hle.yaml protocol.judge: extracted_final_answer, reasoning, correct,
# confidence)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"extracted_final_answer": {"type": "string"}, "reasoning": {"type": "string"},
                   "correct": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "integer"}},
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence"],
    "additionalProperties": False,
}


def secret_from_env(name: str, env_file: Path = paths.REPO / ".env") -> str | None:
    """A secret from the environment or the repository .env (never printed or stored)."""
    if os.environ.get(name):
        return os.environ[name]
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            line = line[len("export "):] if line.startswith("export ") else line
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class EndpointError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def post_json(url: str, body: dict[str, Any], api_key: str, timeout: float, attempts: int = 3,
              stop: threading.Event | None = None) -> dict[str, Any]:
    """POST JSON with retries on transient failures (connection errors, 408/429/5xx); EndpointError otherwise."""
    data = json.dumps(body).encode()
    last: EndpointError | None = None
    for attempt in range(1, attempts + 1):
        if stop is not None and stop.is_set():
            raise EndpointError("run aborted")
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed endpoint URLs
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode(errors="replace")
            last = EndpointError(f"HTTP {e.code}: {detail}", e.code)
            if e.code not in RETRYABLE_HTTP:
                raise last from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = EndpointError(f"{type(e).__name__}: {e}"[:400])
        if attempt < attempts:
            time.sleep(min(60, 5 * 2 ** attempt))
    raise last or EndpointError("request failed")


def message_parts(resp: dict[str, Any]) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    return {"content": _text(msg.get("content")), "reasoning": reasoning or "", "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage") or {}, "refusal": msg.get("refusal")}


class HLEJudge:
    """The HLE judge: JUDGE_PROMPT formatted with (question, response, correct_answer), one user turn, structured output,
    the lock's judge model and completion budget. Usage of every call is returned for the spend record."""

    def __init__(self, assets: HLEAssets, api_key: str, base_url: str = OPENAI_URL, timeout: float = 300.0, attempts: int = 4,
                 usd_per_mtok: dict[str, float] | None = None):
        self.assets, self.api_key, self.base_url = assets, api_key, base_url.rstrip("/")
        self.timeout, self.attempts = timeout, attempts
        self.usd = usd_per_mtok or JUDGE_USD_PER_MTOK

    def cost(self, usage: dict[str, Any]) -> float:
        return round((int(usage.get("prompt_tokens") or 0) * self.usd["input"]
                      + int(usage.get("completion_tokens") or 0) * self.usd["output"]) / 1e6, 6)

    def __call__(self, question: str, response: str, correct_answer: str, stop: threading.Event | None = None) -> dict[str, Any]:
        prompt = self.assets.judge_prompt.format(question=question, response=response, correct_answer=correct_answer)
        body = {"model": self.assets.judge_model, "max_completion_tokens": self.assets.judge_max_completion_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "ExtractedAnswer", "schema": JUDGE_SCHEMA, "strict": True}}}
        t0 = time.time()
        resp = post_json(f"{self.base_url}/chat/completions", body, self.api_key, self.timeout, self.attempts, stop)
        parts = message_parts(resp)
        usage = parts["usage"]
        try:
            parsed = json.loads(parts["content"])
            correct = str(parsed["correct"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise EndpointError(f"judge returned no parseable verdict ({type(e).__name__}; refusal={parts['refusal']!r})") from None
        return {"correct": correct, "model_answer": parsed.get("extracted_final_answer"), "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning"), "model": resp.get("model") or self.assets.judge_model,
                "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                          "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
                "cost_usd": self.cost(usage), "seconds": round(time.time() - t0, 2)}


# ==== runtime context ================================================================================================
class Aborted(RuntimeError):
    pass


class RunContext:
    """What every runner shares: identity, endpoint, directories, the think counter, and the process registry that
    lets an interrupted run terminate its children and remove its containers (labels apex.run_id / apex.devev_run)."""

    def __init__(self, *, run_id: str, version: str, model_name: str, api_base: str, api_key: str, work_dir: Path,
                 results_dir: Path, log: Callable[[str], None], error: Callable[[str], None],
                 counter: ThinkCounter | None = None):
        self.run_id, self.version, self.model_name, self.api_base, self.api_key = run_id, version, model_name, api_base, api_key
        self.work_dir, self.results_dir = Path(work_dir), Path(results_dir)
        self.log, self.error = log, error
        self.counter = counter or ThinkCounter()
        self.policy = f"{version}@{run_id}"     # episode-id policy: unique per run, so concurrent runs never share trial names
        self.fresh: set[tuple[str, str, int]] = set()   # re-run items whose cached rollout must not be reused (--rerun-statuses)
        self.regrade: set[tuple[str, str, int]] = set()  # recorded items graded again from their cached rollout (--regrade-slices)
        self.no_rollouts = False                          # a re-grade without a served model: cached rollouts are graded as they are
        self.prior_records: dict[tuple[str, str, int], dict[str, Any]] = {}   # latest record per item before this invocation
        self.stop = threading.Event()
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def item_dir(self, item: PlanItem) -> Path:
        d = self.work_dir / item.slice / slug(item.task_id) / f"s{item.sample_index}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_item(self, item: PlanItem, d: Path, cached: Iterable[str]) -> int:
        """Attempts already made for this item (attempt numbers continue, so nothing is overwritten); for an item in
        `fresh` the cached outputs are renamed aside (kept, never deleted) so the item runs again from the start."""
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        names = list(cached) if item.key in self.fresh else ["verdict.json"] if item.key in self.regrade else []
        for name in names:
            p = d / name
            if p.exists():
                os.replace(p, p.with_name(f"{p.stem}.superseded_{stamp}{p.suffix}"))
        return len(list(d.glob("spec_a*.json")))

    def results_item_dir(self, item: PlanItem) -> Path:
        return self.results_dir / "items" / item.slice / slug(item.task_id) / f"s{item.sample_index}"

    def run_process(self, cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path, timeout: float | None) -> int | None:
        """Run a child in its own session with output to log_path; terminate it after `timeout` (returns None then)."""
        if self.stop.is_set():
            raise Aborted("run aborted")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            with self._lock:
                self._procs.add(proc)
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate(proc)
                return None
            finally:
                with self._lock:
                    self._procs.discard(proc)

    def abort(self) -> None:
        """Stop launching, terminate every running child (drivers remove their own containers on SIGTERM), then remove
        any container still carrying this run's labels."""
        self.stop.set()
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        deadline = time.time() + 90
        for p in procs:
            try:
                p.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        remove_labelled_containers(self.run_id)


def _terminate(proc: subprocess.Popen, grace: float = 60.0) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def remove_labelled_containers(run_id: str) -> list[str]:
    """Remove containers labelled as this run's (only ever this run's: the labels carry its run id)."""
    ids: set[str] = set()
    for label in (f"{LABEL_RUN}={run_id}", f"apex.run_id={run_id}"):
        r = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={label}"], capture_output=True, text=True, check=False)
        ids.update(r.stdout.split())
    if ids:
        subprocess.run(["docker", "rm", "-f", *sorted(ids)], capture_output=True, check=False)
    return sorted(ids)


# ==== QA runner ======================================================================================================
class QARunner:
    """Dev STEM QA (slices qa and qa_aime): generation through the endpoint, HLE judge, deterministic cross-check."""

    slices = ("qa", "qa_aime")

    def __init__(self, ctx: RunContext, items: dict[str, Any], assets: HLEAssets, judge: HLEJudge,
                 max_tokens: int | None = None, request_timeout: float = QA_REQUEST_TIMEOUT_SEC,
                 scorer: Callable[[Any, str], float] | None = None):
        self.ctx, self.items, self.assets, self.judge = ctx, items, assets, judge
        self.max_tokens = int(max_tokens or assets.max_completion_tokens)
        self.request_timeout = request_timeout
        self.scorer = scorer

    def settings(self) -> dict[str, Any]:
        return {"harness": "HLE prediction prompt + HLE judge (dev only)", **self.assets.fingerprint(),
                "max_completion_tokens": self.max_tokens, "sampling": "serving defaults (no sampling parameters, as the HLE script)",
                "seed": "per sample (dev manifest)", "request_timeout_sec": self.request_timeout,
                "judge_usd_per_mtok": self.judge.usd}

    def generate(self, item: PlanItem, dev: Any, path: Path) -> dict[str, Any]:
        cached = read_json(path)
        if isinstance(cached, dict) and "response" in cached:
            return {**cached, "cached": True}
        body = {"model": self.ctx.model_name, "messages": hle_messages(self.assets.system_prompt, dev.text),
                "max_completion_tokens": self.max_tokens, "seed": item.seed, "stream": False}
        t0 = time.time()
        resp = post_json(f"{self.ctx.api_base.rstrip('/')}/chat/completions", body, self.ctx.api_key, self.request_timeout,
                         attempts=3, stop=self.ctx.stop)
        parts = message_parts(resp)
        g = {"response": parts["content"], "reasoning": parts["reasoning"], "finish_reason": parts["finish_reason"],
             "usage": parts["usage"], "seconds": round(time.time() - t0, 2), "seed": item.seed,
             "request": {"model": self.ctx.model_name, "max_completion_tokens": self.max_tokens, "seed": item.seed}}
        write_json_atomic(path, g)
        return g

    def crosscheck(self, dev: Any, grammar: dict[str, Any]) -> dict[str, Any]:
        ans = grammar["answer"]
        if dev.kind == "mcq":
            got = mcq_letter(ans, len((dev.extra or {}).get("options") or "ABCD"))
            return {"method": "mcq_letter", "value": got == dev.reference}
        if dev.slice == "reasoning_gym" and self.scorer is not None and ans:
            try:
                return {"method": "reasoning_gym_scorer", "value": float(self.scorer(dev, ans)) >= 1.0}
            except Exception as e:  # noqa: BLE001 - informational only
                return {"method": "reasoning_gym_scorer", "value": None, "error": type(e).__name__}
        if dev.slice == "aime25" and ans:
            m = re.search(r"-?\d+", ans.replace(",", ""))
            return {"method": "integer", "value": bool(m) and int(m.group(0)) == int(dev.reference)}
        return {"method": "none", "value": False if not ans else None}

    def run(self, item: PlanItem) -> dict[str, Any]:
        dev = self.items.get(item.task_id)
        if dev is None:
            return failure_record(item, ENVIRONMENT_FAILURE, "dev QA item text not loadable (devset.qa.load_dev_items)")
        d = self.ctx.item_dir(item)
        try:
            g = self.generate(item, dev, d / "generation.json")
        except EndpointError as e:
            self.ctx.error(f"{item.key}: generation failed: {e}")
            return failure_record(item, AGENT_ERROR, f"generation: {e}", harness_status="generation_failed", artifacts=str(d))
        content, reasoning = g["response"] or "", g.get("reasoning") or ""
        usage = g.get("usage") or {}
        grammar = hle_grammar(content)
        empty = not content.strip()
        truncated = g.get("finish_reason") == "length"
        status, error, harness = (BUDGET_EXHAUSTED if truncated else OK), None, "judged"
        judge_rec: dict[str, Any] | None = None
        if empty:
            resolved, harness = False, "empty_content"
        else:
            jpath = d / "judge.json"
            judge_rec = read_json(jpath)
            if not isinstance(judge_rec, dict) or "correct" not in judge_rec:
                try:
                    judge_rec = self.judge(dev.text, content, dev.reference, stop=self.ctx.stop)
                    write_json_atomic(jpath, judge_rec)
                except EndpointError as e:
                    self.ctx.error(f"{item.key}: judge failed: {e}")
                    judge_rec, status, error, harness = None, SCORING_FAILURE, f"judge: {e}", "judge_failed"
            resolved = bool(judge_rec) and judge_rec.get("correct") == "yes"
        think = _think([reasoning], self.ctx.counter)
        metrics = {"n_turns": 1, "n_format_failures": int(not (grammar["compliant"] and not truncated)), "n_truncated": int(truncated), **think}
        cross = self.crosscheck(dev, grammar)
        judge_public = None if judge_rec is None else {
            "model": judge_rec.get("model"), "correct": judge_rec.get("correct"), "confidence": judge_rec.get("confidence"),
            "usage": judge_rec.get("usage"), "cost_usd": judge_rec.get("cost_usd"), "seconds": judge_rec.get("seconds")}
        return make_record(item, run_status=status, resolved=resolved, turn_metrics=metrics, harness_status=harness,
                           error_reason=error, time_sec=g.get("seconds"), input_tokens=usage.get("prompt_tokens"),
                           output_tokens=usage.get("completion_tokens"),
                           reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or think["think_tokens"],
                           artifacts=str(d), kind=dev.kind, finish_reason=g.get("finish_reason"), empty_content=empty,
                           format_compliant=grammar["compliant"], grammar_all_parts=grammar["all_parts"], extracted_answer=(grammar["answer"] or "")[:200] or None,
                           stated_confidence=grammar["confidence"], judge=judge_public, crosscheck=cross,
                           generation_cached=bool(g.get("cached")))


# ==== terminal runner ================================================================================================
TERMINUS_DEFAULT_MAX_TURNS = 1_000_000                 # Terminus-2's own default (the lock's max_turns: null)
NO_TOTAL_CAP = 10 ** 12                                # the locks set no per-episode output cap; the proxy budget never binds


def terminal_message_metrics(messages: list[dict[str, Any]], finish_reasons: list[str | None], counter: ThinkCounter) -> dict[str, Any]:
    """Turn metrics of a rollout Episode's Terminus-2 transcript (Harbor's chat history with reasoning attached)."""
    users = [_text(m.get("content")).lstrip() for m in messages if m.get("role") == "user"]
    assistant = [m for m in messages if m.get("role") == "assistant"]
    n_parse = sum(1 for u in users if u.startswith(TERMINAL_PARSE_ERROR_PREFIX))
    n_limit = sum(1 for u in users if u.startswith(TERMINAL_OUTPUT_LIMIT_PREFIX))
    return {"n_turns": len(assistant), "n_format_failures": n_parse + n_limit,
            "n_truncated": max(n_limit, sum(1 for f in finish_reasons if f == "length")), "source": "episode",
            **_think((m.get("reasoning_content") for m in assistant), counter)}


def _terminal_env_failure(ep: dict[str, Any] | None) -> bool:
    return not isinstance(ep, dict) or bool(((ep.get("extra") or {}).get("harbor") or {}).get("environment_failure"))


class TerminalRunner:
    """TermGrade dev tasks: one Terminus-2 trial per (task, sample) through the U3.3 rollout driver
    (`apex_data.rollout.terminus`) with the Terminal-Bench lock's agent settings and no turn or output cap, then the
    trial's verdict from `apex_data.grade.harbor_tests.read_trial`."""

    slices = ("terminal",)

    def __init__(self, ctx: RunContext, pool: dict[str, dict[str, Any]], admitted: set[str], *, driver: Any, endpoint_mod: Any,
                 dataset_dir: Path, agent_timeout_sec: float = TERMINAL_AGENT_TIMEOUT_SEC, env_retries: int = 1):
        self.ctx, self.pool, self.admitted, self.driver, self.endpoint_mod = ctx, pool, admitted, driver, endpoint_mod
        self.dataset_dir, self.agent_timeout_sec, self.env_retries = Path(dataset_dir), float(agent_timeout_sec), env_retries

    def caps(self) -> Any:
        return self.endpoint_mod.Caps(max_steps=TERMINUS_DEFAULT_MAX_TURNS, max_tokens_per_turn=None, max_output_tokens=NO_TOTAL_CAP,
                                      wall_sec=int(self.agent_timeout_sec))

    def settings(self) -> dict[str, Any]:
        from apex_data.grade import harbor_tests  # noqa: PLC0415

        agent = harbor_tests.terminus2_job_config(self.ctx.model_name, self.ctx.api_base, seed=None, agent_timeout_sec=self.agent_timeout_sec,
                                                  max_turns=TERMINUS_DEFAULT_MAX_TURNS)["agents"][0]
        return {"harness": "apex_data.rollout.terminus (harbor trial start, Terminus-2, configs/eval/terminal_bench_2.yaml agent and sampling)",
                "agent": agent, "agent_timeout_sec": self.agent_timeout_sec,
                "max_turns": f"{TERMINUS_DEFAULT_MAX_TURNS} (= Terminus-2's default; the lock sets none)",
                "output_token_cap": "none (lock)", "seed": "per sample (dev manifest)", "resources": "task.toml (no override)",
                "dataset_dir": str(self.dataset_dir), "n_admitted_dev_tasks": len(self.admitted), "env_retries": self.env_retries,
                "success": "verifier reward 1 and the confirmed task_complete handshake",
                "containers": "named by the driver (trial s2roll_<sha1(episode id)>); policy in the episode id = <version>@<run_id>"}

    def run(self, item: PlanItem) -> dict[str, Any]:
        from apex_data.grade import harbor_tests  # noqa: PLC0415
        from apex_eval.harbor_utils import classify  # noqa: PLC0415

        name = item.upstream_id
        if name not in self.admitted:
            return failure_record(item, ENVIRONMENT_FAILURE, "task not admitted by the TermGrade task checks (data/pools/tgr_env: image "
                                  "built, nop fails, oracle passes); recorded, never dropped", harness_status="not_admitted")
        task = self.pool.get(item.task_id) or {"task_id": item.task_id, "upstream_id": name}
        d = self.ctx.item_dir(item)
        done = self.ctx.start_item(item, d, ("episode.json",))
        ep_path = d / "episode.json"
        ep = read_json(ep_path)
        attempts = 0
        t0 = time.time()
        for attempt in range(done + 1, done + self.env_retries + 2):
            if not _terminal_env_failure(ep):
                break
            attempts += 1
            endpoint = self.endpoint_mod.Endpoint(api_base=self.ctx.api_base, model_name=self.ctx.model_name, api_key_env="EVAL_API_KEY")
            spec = self.driver.build_spec(task, endpoint, round_="dev", policy=self.ctx.policy, sample_index=item.sample_index,
                                          episode_dir=d / f"attempt{attempt}", requests_dir=d / "requests", run_id=self.ctx.run_id,
                                          caps=self.caps(), dataset_dir=self.dataset_dir)
            spec["seed"] = item.seed
            spec_path = d / f"spec_a{attempt}.json"
            write_json_atomic(spec_path, spec)
            out = d / f"episode_a{attempt}.json"
            env = {**self.driver.worker_env(), "EVAL_API_KEY": self.ctx.api_key}
            rc = self.ctx.run_process(self.driver.worker_command(spec_path, out), env, paths.REPO, d / f"worker_a{attempt}.log",
                                      timeout=self.agent_timeout_sec + getattr(self.driver, "GRACE_SEC", 1800) + 900)
            ep = read_json(out)
            if isinstance(ep, dict):
                write_json_atomic(ep_path, ep)
            if rc != 0 or _terminal_env_failure(ep):
                h = ((ep or {}).get("extra") or {}).get("harbor") or {}
                self.ctx.error(f"{item.key}: terminal attempt {attempt}: worker exit {rc}; "
                               f"{h.get('exception_type')}: {str(h.get('exception_message') or (ep or {}).get('extra', {}).get('error'))[:300]}")
        wall = round(time.time() - t0, 1)
        extra = (ep or {}).get("extra") or {}
        h = extra.get("harbor") or {}
        trial_dir = h.get("trial_dir")
        if not trial_dir or not Path(trial_dir).exists():
            status = ENVIRONMENT_FAILURE if _terminal_env_failure(ep) else MISSING
            return failure_record(item, status, f"no Harbor trial result after {attempts} attempt(s): "
                                  f"{h.get('exception_type') or extra.get('error') or 'unknown'}"[:600], harness_status="no_trial",
                                  artifacts=str(d), attempts=attempts, wall_sec=wall)
        o = harbor_tests.read_trial(trial_dir)
        status = classify(o.exception_type, o.exception_message)
        if status == OK and h.get("harbor_rc") is None:   # harbor itself was killed at the hard timeout
            status = BUDGET_EXHAUSTED
        messages = (ep or {}).get("messages") or []
        metrics = (terminal_message_metrics(messages, extra.get("finish_reasons") or [], self.ctx.counter) if messages else
                   terminal_turn_metrics(read_json(Path(trial_dir) / "result.json", {}), read_json(Path(trial_dir) / "agent" / "trajectory.json", {}),
                                         self.ctx.counter))
        error = f"{o.exception_type}: {o.exception_message}" if o.exception_type else None
        usage = extra.get("usage") or {}
        return make_record(item, run_status=status, resolved=bool(o.resolved and o.handshake), turn_metrics=metrics,
                           harness_status=o.exception_type, error_reason=error, time_sec=o.agent_sec,
                           input_tokens=usage.get("prompt_tokens") if usage else o.input_tokens,
                           output_tokens=usage.get("completion_tokens") if usage else o.output_tokens,
                           reasoning_tokens=metrics.get("think_tokens"), artifacts=str(trial_dir), reward=o.reward,
                           handshake=o.handshake, reward_only_resolved=o.resolved, r_pass_strict=o.r_pass,
                           tamper_flags=o.tamper_flags, exit_reason=(ep or {}).get("exit_reason"), verifier_consistent=o.verifier_consistent,
                           n_agent_steps=o.n_agent_steps, verifier_sec=o.verifier_sec, trial_sec=o.trial_sec, attempts=attempts,
                           wall_sec=wall, context_reset=extra.get("context_reset"), n_unmatched_reasoning=extra.get("n_unmatched_reasoning"))


# ==== SWE runner =====================================================================================================
SWE_LOCK = paths.CONFIGS / "eval" / "swebench_pro_v2_hard.yaml"
SWE_BUDGET = {"step_cap": BUDGET_EXHAUSTED, "time_cap": BUDGET_EXHAUSTED, "token_cap": BUDGET_EXHAUSTED}
_ENV_ERROR = re.compile(r"SandboxError|docker|container|No such image|pull access", re.I)


def swe_lock_settings(lock_path: Path = SWE_LOCK) -> dict[str, Any]:
    """The SWE-bench Pro lock's agent budget the dev SWE slice uses (sampling and templates come from the lock's config
    through the driver's agent_config)."""
    lock = yaml.safe_load(Path(lock_path).read_text())
    inf, agent = lock["inference"], lock["agent"]
    return {"sampling": dict(inf["sampling"]), "step_limit": int(agent["step_limit"]), "max_tokens_per_turn": int(inf["max_tokens_per_turn"]),
            "wall_time_limit_seconds": int(agent["wall_time_limit_seconds"]), "agent_framework": agent["framework"],
            "agent_config_file": agent["config_file"], "total_token_cap": None}


def swe_episode_status(ep: dict[str, Any] | None) -> tuple[str, str | None]:
    """(run_status, error_reason) of a rollout Episode: submit -> ok; step/time/token cap -> budget_exhausted (still
    graded); repeated format errors -> agent_error (a model failure, still graded); a sandbox or Docker error ->
    environment_failure; any other error -> agent_error."""
    if not ep:
        return ENVIRONMENT_FAILURE, "the rollout worker wrote no episode"
    reason = ep.get("exit_reason")
    extra = ep.get("extra") or {}
    if reason == "submit":
        return OK, None
    if reason in SWE_BUDGET:
        return SWE_BUDGET[reason], f"episode ended at {reason}"
    if reason == "format_errors":
        return AGENT_ERROR, "repeated format errors"
    if reason == "error":
        what = f"{extra.get('exit_status') or ''}: {extra.get('error') or ''}".strip(": ")
        return (ENVIRONMENT_FAILURE if _ENV_ERROR.search(what) else AGENT_ERROR), (what or "episode error")[:800]
    return AGENT_ERROR, f"episode exit_reason {reason!r}"


class SWERunner:
    """SWE-smith dev tasks: one mini-swe-agent episode per (task, sample) through the U3.3 rollout driver with the
    SWE-bench Pro lock's agent config and budget, then the pristine-replay grader (two runs)."""

    slices = ("swe",)

    def __init__(self, ctx: RunContext, pool: dict[str, dict[str, Any]], *, driver: Any, endpoint_mod: Any, grader: Any,
                 lock: dict[str, Any] | None = None, env_retries: int = 1):
        self.ctx, self.pool, self.driver, self.endpoint_mod, self.grader = ctx, pool, driver, endpoint_mod, grader
        self.lock = lock or swe_lock_settings()
        self.env_retries = env_retries
        self.agent_config_path = ctx.work_dir / "swe_agent_config.yaml"
        self.agent_config = driver.agent_config(self.agent_config_path, self.caps())

    def caps(self) -> Any:
        lk = self.lock
        return self.endpoint_mod.Caps(max_steps=lk["step_limit"], max_tokens_per_turn=lk["max_tokens_per_turn"], max_output_tokens=NO_TOTAL_CAP,
                                      wall_sec=lk["wall_time_limit_seconds"])

    def settings(self) -> dict[str, Any]:
        cfg = self.agent_config or {}
        return {"harness": "apex_data.rollout.mini_swe (mini-swe-agent toolcall, the SWE-bench Pro lock's config and sampling)", **self.lock,
                "agent_config": str(self.agent_config_path),
                "model_kwargs": {k: v for k, v in ((cfg.get("model") or {}).get("model_kwargs") or {}).items()},
                "agent": {k: v for k, v in (cfg.get("agent") or {}).items() if k in ("step_limit", "wall_time_limit_seconds", "cost_limit")},
                "seed": "per sample (dev manifest)", "grader": getattr(self.grader, "VERIFIER_VERSION", None),
                "grading": "pristine replay of the final non-test diff, two runs; graded however the episode ended (no_submit not applied)",
                "n_pool_tasks": len(self.pool), "env_retries": self.env_retries,
                "containers": "agent: s2roll_<episode>_<rand> (driver), label apex.run_id=<run_id>; grader: s2swesmith_..., "
                              f"label {LABEL_RUN}=<run_id>"}

    def episode(self, item: PlanItem, task: dict[str, Any], d: Path) -> tuple[dict[str, Any] | None, int]:
        done = self.ctx.start_item(item, d, ("episode.json", "verdict.json"))
        ep_path = d / "episode.json"
        ep = read_json(ep_path)
        attempts = 0
        for attempt in range(done + 1, done + self.env_retries + 2):
            if (isinstance(ep, dict) and swe_episode_status(ep)[0] != ENVIRONMENT_FAILURE) or self.ctx.no_rollouts or item.key in self.ctx.regrade:
                break   # a re-grade never rolls out: it grades the cached rollout, whatever its status
            attempts += 1
            endpoint = self.endpoint_mod.Endpoint(api_base=self.ctx.api_base, model_name=self.ctx.model_name, api_key_env="EVAL_API_KEY")
            spec = self.driver.build_spec(task, endpoint, round_="dev", policy=self.ctx.policy, sample_index=item.sample_index,
                                          episode_dir=d / f"attempt{attempt}", requests_dir=d / "requests", run_id=self.ctx.run_id,
                                          agent_config_path=self.agent_config_path, caps=self.caps())
            spec["seed"] = item.seed
            spec_path = d / f"spec_a{attempt}.json"
            write_json_atomic(spec_path, spec)
            out = d / f"episode_a{attempt}.json"
            env = {**self.driver.worker_env(), "EVAL_API_KEY": self.ctx.api_key}
            rc = self.ctx.run_process(self.driver.worker_command(spec_path, out), env, paths.REPO, d / f"worker_a{attempt}.log",
                                      timeout=self.lock["wall_time_limit_seconds"] + 1800)
            ep = read_json(out)
            if isinstance(ep, dict):
                write_json_atomic(ep_path, ep)
            if rc != 0 or not isinstance(ep, dict) or swe_episode_status(ep)[0] == ENVIRONMENT_FAILURE:
                self.ctx.error(f"{item.key}: rollout attempt {attempt}: worker exit {rc}; {swe_episode_status(ep)[1]}")
        return (ep if isinstance(ep, dict) else None), attempts

    def grade(self, item: PlanItem, task: dict[str, Any], ep: dict[str, Any], d: Path) -> tuple[dict[str, Any] | None, int]:
        vpath = d / "verdict.json"
        verdict = read_json(vpath)
        attempts = 0
        done = len(list(d.glob("grade_a*")))
        for attempt in range(done + 1, done + self.env_retries + 2):
            if isinstance(verdict, dict) and not (verdict.get("extra") or {}).get("env_failure"):
                break
            if self.ctx.stop.is_set():
                raise Aborted("run aborted")
            attempts += 1
            opts = self.grader.GradeOptions(labels={LABEL_RUN: self.ctx.run_id, "apex.unit": "devev"})
            v = self.grader.grade_patch(task, ep.get("patch") or "", d / f"grade_a{attempt}", True,
                                        episode_id=ep.get("episode_id"), exit_reason="submit", opts=opts)
            verdict = dict(v.__dict__) if hasattr(v, "__dict__") else dict(v)
            write_json_atomic(vpath, verdict)
        return (verdict if isinstance(verdict, dict) else None), attempts

    def run(self, item: PlanItem) -> dict[str, Any]:
        task = self.pool.get(item.task_id)
        if task is None:
            return failure_record(item, ENVIRONMENT_FAILURE, "task not in the SWE-smith pool file (data/pools/swesmith); recorded, never dropped",
                                  harness_status="not_in_pool")
        cal = self.grader.calibration_of(task) or {}
        if cal.get("status") != "ok":
            return failure_record(item, ENVIRONMENT_FAILURE, f"task not admitted by gold-replay calibration ({cal.get('status')}: "
                                  f"{cal.get('reason')}); recorded, never dropped", harness_status="not_calibrated")
        d = self.ctx.item_dir(item)
        t0 = time.time()
        ep, attempts = self.episode(item, task, d)
        status, error = swe_episode_status(ep)
        prior = self.ctx.prior_records.get(item.key) if item.key in self.ctx.regrade else None
        if prior is not None and (ep is None or status == ENVIRONMENT_FAILURE):   # nothing to grade: the record stands
            return regrade_kept(prior, getattr(self.grader, "VERIFIER_VERSION", None),
                                "no cached rollout to grade" if ep is None else "cached rollout is an environment failure")
        if ep is None or status == ENVIRONMENT_FAILURE:
            return failure_record(item, ENVIRONMENT_FAILURE, error or "environment failure", harness_status="rollout",
                                  artifacts=str(d), rollout_attempts=attempts)
        agent_wall = round(time.time() - t0, 1)
        if prior is not None:   # a re-grade keeps the rollout's own agent time and attempts
            agent_wall = (prior.get("extra") or {}).get("agent_wall_sec", agent_wall)
            attempts = (prior.get("extra") or {}).get("rollout_attempts", attempts)
        verdict, g_attempts = self.grade(item, task, ep, d)
        v_extra = (verdict or {}).get("extra") or {}
        if verdict is None or v_extra.get("env_failure"):
            msg = "; ".join(str(r.get("error")) for r in (verdict or {}).get("runs") or [] if r.get("error"))[:600] or "no verdict"
            self.ctx.error(f"{item.key}: grading sandbox failure: {msg}")
            if status in (OK, BUDGET_EXHAUSTED):
                status, error = VERIFIER_FAILURE, f"grading sandbox failure: {msg}"
        resolved = bool(verdict) and int(verdict.get("r_pass", 0)) == 1 and not v_extra.get("env_failure")
        patch = ep.get("patch") or ""
        patch_file = None
        if patch:
            out_dir = self.ctx.results_item_dir(item)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "model.patch").write_text(patch)
            patch_file = str((out_dir / "model.patch").relative_to(self.ctx.results_dir))
        ep_extra = ep.get("extra") or {}
        usage = ep_extra.get("usage") or {}
        metrics = swe_turn_metrics(ep, self.ctx.counter)
        grade_sec = round(time.time() - t0 - (0.0 if prior is not None else agent_wall), 1)
        return make_record(item, run_status=status, resolved=resolved, turn_metrics=metrics, harness_status=ep.get("exit_reason"),
                           error_reason=error, time_sec=ep.get("wall_sec"), input_tokens=usage.get("prompt_tokens"),
                           output_tokens=ep.get("output_tokens"), reasoning_tokens=metrics.get("think_tokens"), artifacts=str(d),
                           exit_reason=ep.get("exit_reason"), steps=ep.get("steps"), patch_bytes=len(patch), patch_file=patch_file,
                           regraded=prior is not None,
                           tamper_flags=(verdict or {}).get("tamper_flags"), r_partial=v_extra.get("r_partial"),
                           grader_version=(verdict or {}).get("verifier_version"), scratch_test_files=v_extra.get("scratch_test_files"),
                           grade_runs=len((verdict or {}).get("runs") or []), grade_skipped=v_extra.get("skipped_reason"),
                           rollout_attempts=attempts, grade_attempts=g_attempts, agent_wall_sec=agent_wall, grade_sec=grade_sec)


def regrade_kept(prior: dict[str, Any], grader_version: str | None, reason: str) -> dict[str, Any]:
    """A re-graded item with nothing to grade keeps its record: the same status, verdict and metrics, under the current
    grader version, with the reason (the superseded line stays in per_item.jsonl)."""
    rec = json.loads(json.dumps({k: v for k, v in prior.items() if k != "supersedes"}))
    rec.setdefault("extra", {}).update(grader_version=grader_version, regraded=True, regrade_kept=reason)
    return rec


# ==== executor =======================================================================================================
def crash_record(item: PlanItem, exc: BaseException) -> dict[str, Any]:
    return failure_record(item, MISSING, f"driver exception {type(exc).__name__}: {exc}", harness_status="driver_exception")


def execute(items: list[PlanItem], runners: dict[str, Any], concurrency: dict[str, int], on_record: Callable[[dict[str, Any]], None],
            ctx: RunContext, progress_every: float = 300.0) -> dict[str, int]:
    """Run every item on its slice's runner with per-slice worker threads (daemon threads; slices run concurrently),
    calling on_record in the calling thread as items finish. A runner exception becomes a `missing` record with the
    traceback in errors.log. On interruption the context aborts (children terminated, labelled containers removed)
    and the exception propagates."""
    results: queue.Queue = queue.Queue()
    by_runner: dict[int, list[PlanItem]] = {}
    for it in items:
        runner = runners[it.slice]
        by_runner.setdefault(id(runner), []).append(it)
    workers: list[threading.Thread] = []
    for its in by_runner.values():
        runner = runners[its[0].slice]
        q: queue.Queue = queue.Queue()
        for it in its:
            q.put(it)
        base = "qa" if its[0].slice.startswith("qa") else its[0].slice
        n = max(1, min(len(its), int(concurrency.get(base, 1))))
        for _ in range(n):
            q.put(None)

        def work(q: queue.Queue = q, runner: Any = runner) -> None:
            while not ctx.stop.is_set():
                it = q.get()
                if it is None:
                    return
                try:
                    rec = runner.run(it)
                except Aborted:
                    return
                except Exception as e:  # noqa: BLE001 - a driver bug never drops an item
                    ctx.error(f"{it.key}: driver exception\n{traceback.format_exc()}")
                    rec = crash_record(it, e)
                results.put(rec)

        for i in range(n):
            t = threading.Thread(target=work, name=f"dev_{its[0].slice}_{i}", daemon=True)
            t.start()
            workers.append(t)
    done: Counter = Counter()
    last = time.time()
    try:
        while sum(done.values()) < len(items):
            try:
                rec = results.get(timeout=1.0)
            except queue.Empty:
                if not any(t.is_alive() for t in workers) and results.empty():
                    break
                if time.time() - last > progress_every:
                    ctx.log(f"progress: {sum(done.values())}/{len(items)} items done ({dict(done)})")
                    last = time.time()
                continue
            on_record(rec)
            done[rec["slice"]] += 1
    except BaseException:
        ctx.log("interrupted: terminating running workers and removing this run's containers")
        ctx.abort()
        raise
    return dict(done)


# ==== summaries and guards ===========================================================================================
def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 2) if den else None


def _quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "p90": None, "max": None, "sum": 0.0}
    s = sorted(xs)
    return {"mean": round(statistics.fmean(s), 1), "median": round(statistics.median(s), 1),
            "p90": round(s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)], 1), "max": round(s[-1], 1), "sum": round(sum(s), 1)}


def turn_totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recs = list(records)
    tm = [r.get("turn_metrics") or {} for r in recs]
    n_turns = sum(int(m.get("n_turns") or 0) for m in tm)
    tokens_known = all(m.get("think_tokens") is not None for m in tm if int(m.get("n_turns") or 0))
    think = sum(int((m.get("think_tokens") if tokens_known else m.get("think_chars")) or 0) for m in tm)
    fails = sum(int(m.get("n_format_failures") or 0) for m in tm)
    return {"n_turns": n_turns, "n_format_failures": fails, "format_failure_rate": round(fails / n_turns, 5) if n_turns else None,
            "n_truncated": sum(int(m.get("n_truncated") or 0) for m in tm),
            "think_unit": "tokens" if tokens_known else "chars", "think_total": think,
            "mean_think_per_turn": round(think / n_turns, 1) if n_turns else None}


def item_wall_sec(r: dict[str, Any]) -> float | None:
    """Wall clock one item occupied its slot: SWE rollout + grading, the terminal worker, QA generation + judge."""
    e = r.get("extra") or {}
    if e.get("agent_wall_sec") is not None:
        return float(e["agent_wall_sec"]) + float(e.get("grade_sec") or 0.0)
    if e.get("wall_sec") is not None:
        return float(e["wall_sec"])
    if r.get("time_sec") is None:
        return None
    return float(r["time_sec"]) + float((e.get("judge") or {}).get("seconds") or 0.0)


# Process and budget metrics (timeout risk after training): which per-episode limit an item ran into,
# turn counts, tokens and seconds per turn, exit reasons. A limit-hit rate above Base's by more than the slice's design
# resolution is flagged in the guard report (limit_guard), next to the score guards.
LIMIT_KINDS = {"qa": ("answer_budget",), "swe": ("step_cap", "time_cap", "token_cap", "context_overflow"),
               "terminal": ("timeout", "context_overflow")}
_CONTEXT_OVERFLOW = re.compile(r"ContextWindowExceeded|context.?length|maximum context|ContextLengthExceeded|context_length_exceeded|"
                               r"max_model_len|prompt is too long", re.I)


def base_slice(slice_name: str) -> str:
    return "qa" if slice_name.startswith("qa") else slice_name


def limit_hits(r: dict[str, Any]) -> list[str]:
    """The per-episode limits one item ran into: QA the 32K answer budget (finish_reason length); SWE the step limit
    (250), the wall-clock limit, the output-token cap; terminal the agent timeout (1,800 s); SWE and terminal a context
    overflow (the request exceeded the served context)."""
    sl, e = base_slice(r["slice"]), r.get("extra") or {}
    hits: list[str] = []
    if sl == "qa":
        if e.get("finish_reason") == "length" or r.get("run_status") == BUDGET_EXHAUSTED:
            hits.append("answer_budget")
        return hits
    if sl == "swe" and e.get("exit_reason") in ("step_cap", "time_cap", "token_cap"):
        hits.append(str(e["exit_reason"]))
    if sl == "terminal" and r.get("run_status") == BUDGET_EXHAUSTED:
        hits.append("timeout")
    if _CONTEXT_OVERFLOW.search(f"{r.get('error_reason') or ''} {r.get('harness_status') or ''}"):
        hits.append("context_overflow")
    return hits


def process_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Limit hits (rates over every item) and per-episode distributions (over the items that made at least one model
    call): turns, output tokens per turn and per episode, input tokens per episode, seconds per turn (agent time),
    think tokens per item; exit reasons (QA: finish reasons)."""
    if not records:
        return {}
    sl, n = base_slice(records[0]["slice"]), len(records)
    hits = [limit_hits(r) for r in records]
    kinds = Counter(k for h in hits for k in h)
    ran = [r for r in records if int((r.get("turn_metrics") or {}).get("n_turns") or 0) > 0]

    def turns(r: dict[str, Any]) -> int:
        return int(r["turn_metrics"]["n_turns"])

    def agent_sec(r: dict[str, Any]) -> float | None:
        e = r.get("extra") or {}
        v = e.get("agent_wall_sec") if e.get("agent_wall_sec") is not None else r.get("time_sec")
        return None if v is None else float(v)

    exit_key = "finish_reason" if sl == "qa" else "exit_reason"
    out: dict[str, Any] = {
        "n_items": n, "n_items_with_turns": len(ran),
        "limit_hit_rate_pct": _pct(sum(1 for h in hits if h), n),
        "limit_hits": {k: kinds[k] for k in LIMIT_KINDS[sl]}, "limit_hit_rates_pct": {k: _pct(kinds[k], n) for k in LIMIT_KINDS[sl]},
        "exit_reasons": dict(Counter(str((r.get("extra") or {}).get(exit_key)) for r in records)),
        "turns_per_episode": _quantiles([turns(r) for r in ran]),
        "output_tokens_per_turn": _quantiles([int(r.get("output_tokens") or 0) / turns(r) for r in ran]),
        "output_tokens_per_episode": _quantiles([int(r.get("output_tokens") or 0) for r in ran]),
        "input_tokens_per_episode": _quantiles([int(r.get("input_tokens") or 0) for r in ran]),
        "seconds_per_turn": _quantiles([s / turns(r) for r in ran if (s := agent_sec(r)) is not None]),
        "think_tokens_per_item": _quantiles([int((r.get("turn_metrics") or {}).get("think_tokens") or 0) for r in ran]),
    }
    if sl == "swe":
        out["steps"] = _quantiles([int((r.get("extra") or {}).get("steps") or 0) for r in ran])
    if sl == "terminal":
        out["handshake_rate_pct"] = _pct(sum(1 for r in records if (r.get("extra") or {}).get("handshake")), n)
        out["n_context_resets"] = sum(1 for r in records if (r.get("extra") or {}).get("context_reset"))
    return out


def limit_comparison(run: list[dict[str, Any]], base: list[dict[str, Any]], margin_pp: float) -> dict[str, Any]:
    """Process metrics of one slice on the shared items, run vs baseline, and the limit guard: the run's limit-hit rate
    is at most the baseline's plus `margin_pp` (the slice's design resolution)."""
    r, b = process_stats(run), process_stats(base)
    if not r or not b:
        return {}
    ok = r["limit_hit_rate_pct"] <= b["limit_hit_rate_pct"] + margin_pp

    def means(key: str) -> dict[str, Any]:
        return {"run": r[key]["mean"], "baseline": b[key]["mean"]}

    return {"limit_hit_rate_pct": {"run": r["limit_hit_rate_pct"], "baseline": b["limit_hit_rate_pct"],
                                   "diff_pp": round(r["limit_hit_rate_pct"] - b["limit_hit_rate_pct"], 2)},
            "limit_hit_rates_pct": {"run": r["limit_hit_rates_pct"], "baseline": b["limit_hit_rates_pct"]},
            "exit_reasons": {"run": r["exit_reasons"], "baseline": b["exit_reasons"]},
            "means": {k: means(k) for k in ("turns_per_episode", "output_tokens_per_turn", "output_tokens_per_episode", "seconds_per_turn",
                                             "think_tokens_per_item")},
            "limit_guard": {"ok": ok, "margin_pp": margin_pp, "rule": f"limit-hit rate <= baseline's + {margin_pp:g} pp"}}


def slice_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score, failure accounting, time, tokens and turn metrics of one slice (no intervals: scripts/summarize.py).
    `time_sec` is the model-facing time (agent or generation); `item_wall_sec` the slot time used for planning."""
    n = len(records)
    n_res = sum(1 for r in records if r.get("resolved"))
    times = [float(r["time_sec"]) for r in records if r.get("time_sec") is not None]
    walls = [w for w in (item_wall_sec(r) for r in records) if w is not None]
    out = {"n_items": n, "n_tasks": len({r["task_id"] for r in records}), "n_resolved": n_res, "score_pct": _pct(n_res, n),
           "failures": {k: v for k, v in failure_summary(records).items() if k != "failed"},
           "time_sec": _quantiles(times), "item_wall_sec": _quantiles(walls), "turns": turn_totals(records),
           "tokens": {"input": sum(int(r.get("input_tokens") or 0) for r in records),
                      "output": sum(int(r.get("output_tokens") or 0) for r in records),
                      "output_mean": round(statistics.fmean([int(r.get("output_tokens") or 0) for r in records]), 1) if records else None},
           "process": process_stats(records)}
    if records and records[0]["slice"].startswith("qa"):
        ex = [r.get("extra") or {} for r in records]
        judged = [e["judge"] for e in ex if e.get("judge")]
        cross = [(r.get("resolved"), (e.get("crosscheck") or {}).get("value")) for r, e in zip(records, ex)]
        cross = [(a, b) for a, b in cross if b is not None]
        out["qa"] = {"empty_content_rate_pct": _pct(sum(1 for e in ex if e.get("empty_content")), n),
                     "format_compliance_pct": _pct(sum(1 for e in ex if e.get("format_compliant")), n),
                     "grammar_all_parts_pct": _pct(sum(1 for e in ex if e.get("grammar_all_parts")), n),
                     "truncated_rate_pct": _pct(sum(1 for e in ex if e.get("finish_reason") == "length"), n),
                     "judge_calls": len(judged),
                     "judge_input_tokens": sum(int((j.get("usage") or {}).get("prompt_tokens") or 0) for j in judged),
                     "judge_output_tokens": sum(int((j.get("usage") or {}).get("completion_tokens") or 0) for j in judged),
                     "judge_cost_usd": round(sum(float(j.get("cost_usd") or 0) for j in judged), 4),
                     "crosscheck_agreement_pct": _pct(sum(1 for a, b in cross if bool(a) == bool(b)), len(cross)),
                     "crosscheck_n": len(cross)}
        if any(e.get("qa_slice") for e in ex):
            out["by_qa_slice"] = {s: {"n": sum(1 for e in ex if e.get("qa_slice") == s),
                                      "score_pct": _pct(sum(1 for r, e in zip(records, ex) if e.get("qa_slice") == s and r.get("resolved")),
                                                        sum(1 for e in ex if e.get("qa_slice") == s))}
                                  for s in sorted({e.get("qa_slice") for e in ex if e.get("qa_slice")})}
    return out


def run_stats(records: dict[tuple[str, str, int], dict[str, Any]], plan_keys: Iterable[tuple[str, str, int]] | None = None) -> dict[str, Any]:
    recs = list(records.values())
    by_slice = {sl: [r for r in recs if r["slice"] == sl] for sl in sorted({r["slice"] for r in recs})}
    slices = {sl: slice_stats(rs) for sl, rs in by_slice.items()}
    scored = [s for s in SLICES if s in slices]
    macro = round(statistics.fmean(slices[s]["score_pct"] for s in scored), 2) if len(scored) == len(SLICES) else None
    planned = set(plan_keys) if plan_keys is not None else set(records)
    return {"slices": slices, "macro_average_pct": macro, "macro_slices": list(SLICES),
            "macro_note": None if macro is not None else f"macro needs all of {SLICES}; present: {scored}",
            "pooled_turns": turn_totals(r for r in recs if r["slice"] in SLICES),
            "n_planned": len(planned), "n_recorded": len(planned & set(records)),
            "complete": planned <= set(records), "not_yet_run": len(planned - set(records)),
            "judge_cost_usd": round(sum((s.get("qa") or {}).get("judge_cost_usd", 0.0) for s in slices.values()), 4)}


def pairing(run: dict[tuple[str, str, int], dict[str, Any]], base: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    """Shared keys and the seed check: a shared (slice, task_id, sample_index) must carry the same seed in both runs."""
    shared = sorted(set(run) & set(base))
    mismatched = [k for k in shared if run[k].get("seed") != base[k].get("seed")]
    return {"shared": shared, "n_shared": len(shared), "n_only_run": len(set(run) - set(base)),
            "n_only_baseline": len(set(base) - set(run)), "seed_mismatches": [list(k) for k in mismatched[:20]],
            "n_seed_mismatches": len(mismatched)}


def resolution_set(run_size: str | None, base_size: str | None) -> str:
    return "full" if run_size == "full" and base_size == "full" else "reduced"


def guard_report(run: dict[tuple[str, str, int], dict[str, Any]], base: dict[tuple[str, str, int], dict[str, Any]],
                 run_size: str, base_size: str | None, paired_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Guard verdicts of docs/DATA_PLAN.md section 1 on the shared items (module docstring). `paired_summary` is the
    output of scripts/summarize.py dev (its per-slice paired difference and interval)."""
    pr = pairing(run, base)
    rset = resolution_set(run_size, base_size)
    res = RESOLUTIONS_PP[rset]
    shared = set(pr["shared"])
    r_sh = {k: v for k, v in run.items() if k in shared}
    b_sh = {k: v for k, v in base.items() if k in shared}
    out: dict[str, Any] = {"resolution_set": rset, "resolutions_pp": res, "point_drop_max_pp": GUARD_POINT_PP,
                           "pairing": {k: v for k, v in pr.items() if k != "shared"},
                           "decision_grade": run_size != "smoke" and pr["n_seed_mismatches"] == 0, "slices": {}}
    if pr["n_seed_mismatches"]:
        out["error"] = "seed mismatch on shared items: the runs do not pair; no guard verdicts"
        out["all_pass"] = None
        return out
    passes: list[bool] = []
    ps = (paired_summary or {}).get("slices") or {}
    for sl in SLICES:
        rk = [k for k in r_sh if k[0] == sl]
        if not rk:
            continue
        r_score = _pct(sum(1 for k in rk if r_sh[k].get("resolved")), len(rk))
        b_score = _pct(sum(1 for k in rk if b_sh[k].get("resolved")), len(rk))
        p = (ps.get(sl) or {}).get("paired") or {}
        ci = p.get("ci")
        diff = round((r_score or 0.0) - (b_score or 0.0), 2)
        point_ok = diff >= -GUARD_POINT_PP
        lb_ok = None if not ci else ci[0] >= -res[sl]
        r_tt, b_tt = turn_totals(r_sh[k] for k in rk), turn_totals(b_sh[k] for k in rk)
        ratio = (round(r_tt["mean_think_per_turn"] / b_tt["mean_think_per_turn"], 3)
                 if r_tt["mean_think_per_turn"] is not None and b_tt["mean_think_per_turn"] else None)
        comparable = ratio is not None and r_tt["think_unit"] == b_tt["think_unit"]
        cap = THINK_MAX_TOKENS_PER_TURN.get(sl)
        if cap is not None:      # an absolute bound in tokens; unknown when the think length could only be counted in characters
            measured = r_tt["mean_think_per_turn"] is not None and r_tt["think_unit"] == "tokens"
            think_ok = r_tt["mean_think_per_turn"] <= cap if measured else None
        else:
            think_ok = ratio <= THINK_RATIO_MAX if comparable else None
        think_low = ratio < THINK_RATIO_FLAG_BELOW if comparable else None
        entry = {"n_shared_items": len(rk), "run_pct": r_score, "baseline_pct": b_score, "diff_pp": diff,
                 "ci_95": ci, "ci_method": p.get("ci_method"), "mcnemar": p.get("mcnemar"),
                 "point_ok": point_ok, "lower_bound_ok": lb_ok, "resolution_pp": res[sl],
                 "think": {"run_mean": r_tt["mean_think_per_turn"], "baseline_mean": b_tt["mean_think_per_turn"], "unit": r_tt["think_unit"],
                           "ratio": ratio, "ok": think_ok,
                           **({"max_tokens_per_turn": cap} if cap is not None else {"max_ratio": THINK_RATIO_MAX}),
                           "below_flag": think_low, "flag_below_ratio": THINK_RATIO_FLAG_BELOW},
                 "format_failure_rate": {"run": r_tt["format_failure_rate"], "baseline": b_tt["format_failure_rate"]}}
        entry["pass"] = bool(point_ok and lb_ok and think_ok)
        entry["process"] = limit_comparison([r_sh[k] for k in rk], [b_sh[k] for k in rk], res[sl])
        out["slices"][sl] = entry
        passes.append(entry["pass"])
    r_ff = turn_totals(v for k, v in r_sh.items() if k[0] in SLICES)
    b_ff = turn_totals(v for k, v in b_sh.items() if k[0] in SLICES)
    rate, brate = r_ff["format_failure_rate"], b_ff["format_failure_rate"]
    rel_ok = rate is not None and brate is not None and rate <= FORMAT_FAILURE_REL * brate
    out["format_failure"] = {"run_rate": rate, "baseline_rate": brate, "run_turns": r_ff["n_turns"], "baseline_turns": b_ff["n_turns"],
                             "rel_ok": rel_ok, "pass": bool(rel_ok), "above_flag": rate is not None and rate > FORMAT_FAILURE_FLAG,
                             "flag_above_rate": FORMAT_FAILURE_FLAG,
                             "rule": f"pooled over the scored slices: <= {FORMAT_FAILURE_REL:g}x baseline"}
    passes.append(out["format_failure"]["pass"])
    out["flags"] = {"think_below": sorted(sl for sl, e in out["slices"].items() if e["think"]["below_flag"]),
                    "format_failure_above": out["format_failure"]["above_flag"],
                    "note": f"reported, not guards (revision 4.3.18): a slice's mean think below {THINK_RATIO_FLAG_BELOW:g}x the "
                            f"baseline's; a pooled format-failure rate above {FORMAT_FAILURE_FLAG:.0%} of turns"}
    lg = {sl: (e.get("process") or {}).get("limit_guard", {}).get("ok") for sl, e in out["slices"].items()}
    out["limit_guard"] = {"slices": lg, "all_ok": bool(lg) and all(v is True for v in lg.values()),
                          "note": "limit-hit rate <= baseline's + the slice's design resolution; a reported flag, not part of all_pass; "
                                  "used only as a tie-breaker between checkpoints that pass every guard with scores within one design "
                                  "resolution (docs/DATA_PLAN_CHANGELOG.md revision 4.3.9)"}
    out["macro"] = (ps and (paired_summary or {}).get("macro")) or None
    out["all_pass"] = bool(passes) and all(passes) and len(out["slices"]) == len(SLICES)
    if len(out["slices"]) != len(SLICES):
        out["note"] = f"guards need all scored slices; shared slices: {sorted(out['slices'])}"
    return out


def run_summarize_py(run_dir: Path, base_dir: Path, rset: str, out_path: Path, run_id: str, log_root: Path | None,
                     labels: tuple[str, str] = ("run", "baseline"), timeout: float = 1800.0,
                     admitted: Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """scripts/summarize.py dev on the two record files (the only source of intervals), restricted to the frozen
    admitted dev tasks when `admitted` is given; (its JSON, error)."""
    res = RESOLUTIONS_PP[rset]
    cmd = [sys.executable, str(paths.REPO / "scripts" / "summarize.py"), "dev", "--run", str(run_dir), "--baseline", str(base_dir),
           "--guard-resolutions", ",".join(f"{k}={v:g}" for k, v in res.items()), "--guard-point-pp", f"{GUARD_POINT_PP:g}",
           "--macro-slices", ",".join(SLICES), "--label", labels[0], "--baseline-label", labels[1], "--out", str(out_path),
           "--run-id", f"summarize_dev_{run_id}_{time.strftime('%H%M%S', time.gmtime())}"]
    if log_root is not None:
        cmd += ["--log-root", str(log_root)]
    if admitted is not None:
        cmd += ["--admitted", str(admitted)]
    r = subprocess.run(cmd, cwd=paths.REPO, capture_output=True, text=True, timeout=timeout, check=False)
    if r.returncode != 0:
        return None, f"scripts/summarize.py dev exited {r.returncode}: {(r.stderr or r.stdout)[-800:]}"
    return read_json(out_path), None


def resolve_baseline(spec: str | Path, repo: Path = paths.REPO) -> Path:
    """A baseline given as a per_item.jsonl, a results dir, a logs/<run_id> dir (metadata.results_dir) or a run id."""
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir() and (p / "per_item.jsonl").exists():
        return p / "per_item.jsonl"
    if p.is_dir() and (p / "metadata.json").exists():
        rd = (read_json(p / "metadata.json", {}) or {}).get("results_dir")
        if rd and (Path(rd) / "per_item.jsonl").exists():
            return Path(rd) / "per_item.jsonl"
    hits = sorted((repo / "results" / "dev").glob(f"*/{p.name}/per_item.jsonl"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"baseline {spec}: no per_item.jsonl found (give a results dir, logs/<run_id> or a run id)")


PHASE_ORDER = ("swe+qa", "terminal")   # terminal trials never share the endpoint with SWE / QA load


def phase_of(slice_name: str) -> str:
    return "terminal" if slice_name == "terminal" else "swe+qa"


def endpoint_of(phase: dict[str, Any]) -> str | None:
    return (phase.get("endpoint") or {}).get("api_base")


def phase_deviations(phases: list[dict[str, Any]]) -> list[str]:
    """Protocol deviations of a run's recorded phases (metadata `phases`): a terminal phase whose interval overlaps a
    swe+qa phase on the same endpoint (or where either endpoint is unrecorded), or a phase that never recorded its end.
    Phases on two different endpoints may run at the same time (terminal trials then never share a server with SWE / QA
    load)."""
    out = [f"phase {p.get('phase')} started {p.get('start')} has no end time (still running or killed)" for p in phases if not p.get("end")]
    term = [p for p in phases if p.get("phase") == "terminal" and p.get("end")]
    other = [p for p in phases if p.get("phase") != "terminal" and p.get("end")]
    for t in term:
        for o in other:
            separate = endpoint_of(t) is not None and endpoint_of(o) is not None and endpoint_of(t) != endpoint_of(o)
            if t["start"] < o["end"] and o["start"] < t["end"] and not separate:
                out.append(f"terminal phase {t['start']}..{t['end']} overlaps swe+qa phase {o['start']}..{o['end']} on "
                           f"{endpoint_of(t) or 'an unrecorded endpoint'}")
    return out


def host_load_window(csv_path: Path | None, start: str, end: str) -> dict[str, Any] | None:
    """apex-cpu load over [start, end] from scripts/dev/host_load_sampler.sh's host_load.csv (None without samples)."""
    if csv_path is None or not Path(csv_path).exists():
        return None
    import csv  # noqa: PLC0415

    rows = [r for r in csv.DictReader(Path(csv_path).open()) if start <= r.get("ts_utc", "") <= end]
    if not rows:
        return {"source": str(csv_path), "n_samples": 0}

    def col(name: str) -> list[float]:
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    stats = {name: {"mean": round(statistics.fmean(v), 1), "max": round(max(v), 1)} for name in ("load1", "cpu_busy_pct", "cpu_iowait_pct", "containers")
             if (v := col(name))}
    return {"source": str(csv_path), "n_samples": len(rows), **stats}


def protocol_of(concurrency: dict[str, int], terminal_agent_timeout_sec: float, admitted_sha256: str | None = None,
                gated: bool = False) -> dict[str, Any]:
    """The run's protocol record (metadata.json config.protocol and summary.json protocol): the per-slice concurrency, the
    terminal agent timeout, the SWE sandbox layout (rollout.mini_swe.SANDBOX_LAYOUT: repository at /app, conda env
    activated), the phase order, the sha256 of the frozen admitted dev task lists (devset.admission; None for a run
    without SWE or terminal items) and the deviations from the fixed protocol. `gated`: the run has SWE or terminal
    items, so it needs the frozen lists."""
    from apex_data.rollout.mini_swe import SANDBOX_LAYOUT  # noqa: PLC0415

    conc = {s: int(concurrency[s]) for s in DEFAULT_CONCURRENCY}
    deviations = [f"{s} concurrency {conc[s]} (protocol {n})" for s, n in DEFAULT_CONCURRENCY.items() if conc[s] != n]
    if float(terminal_agent_timeout_sec) != TERMINAL_AGENT_TIMEOUT_SEC:
        deviations.append(f"terminal agent timeout {terminal_agent_timeout_sec:g} s (protocol {TERMINAL_AGENT_TIMEOUT_SEC:g} s)")
    if gated and admitted_sha256 is None:
        deviations.append("admitted dev task lists not frozen (data/manifests/dev_admitted.json; scripts/data/freeze_dev_admission.py)")
    return {"concurrency": conc, "terminal_agent_timeout_sec": float(terminal_agent_timeout_sec), "swe_sandbox": SANDBOX_LAYOUT,
            "phase_order": list(PHASE_ORDER), "admitted_sha256": admitted_sha256, "deviations": deviations}


def with_swe_grader_version(protocol: dict[str, Any], records: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    """The protocol with `swe_grader_version`: the SWE-smith grader version of the latest record of every SWE item (None
    without SWE items). Records graded under another version than the current grader's, or under several versions, make
    a deviation: such a run is re-graded (--resume --regrade-slices swe) before it is reported or paired."""
    from apex_data.grade.swesmith import VERIFIER_VERSION  # noqa: PLC0415

    versions = Counter((r.get("extra") or {}).get("grader_version") for k, r in records.items() if k[0] == "swe")
    if not versions:
        return {**protocol, "swe_grader_version": None}
    out = {**protocol, "swe_grader_version": next(iter(versions)) if len(versions) == 1 else "mixed"}
    if set(versions) != {VERIFIER_VERSION}:
        out["deviations"] = [*protocol["deviations"],
                             f"SWE items graded under {dict(versions)}, not only the current grader {VERIFIER_VERSION} (re-grade with "
                             "--resume --regrade-slices swe)"]
    return out


def protocol_mismatch(own: dict[str, Any] | None, baseline: dict[str, Any] | None) -> str | None:
    """Why two runs' protocols do not pair (None when concurrency, terminal agent timeout, SWE sandbox, phase order and
    admitted dev task lists are equal)."""
    if not baseline:
        return "the baseline records no protocol (it predates the fixed dev protocol)"
    keys = ("concurrency", "terminal_agent_timeout_sec", "swe_sandbox", "phase_order")
    # compared when both runs carry them: a run without SWE / terminal items has no admitted list and no SWE grader version
    when_both = ("admitted_sha256", "swe_grader_version")
    differs = own is None or any(own.get(k) != baseline.get(k) for k in keys) or any(
        own.get(k) is not None and baseline.get(k) is not None and own.get(k) != baseline.get(k) for k in when_both)
    if differs:
        shown = keys + when_both
        return (f"protocol differs: this run {({k: (own or {}).get(k) for k in shown})}, baseline {({k: baseline.get(k) for k in shown})}")
    return None


def baseline_size(per_item: Path) -> str | None:
    s = read_json(per_item.parent / "summary.json", {}) or {}
    return s.get("size")


def baseline_refusal(per_item: Path, protocol: dict[str, Any] | None = None) -> str | None:
    """Why a run cannot serve as a baseline (None when it can): it must be a finished dev run whose summary.json says
    `"smoke": false` and `"reportable": true` (full or reduced size, the frozen dev set with the real pools, all three
    slices, every planned item recorded, the fixed protocol) and, given this run's `protocol`, the same concurrency and
    terminal agent timeout."""
    s = read_json(per_item.parent / "summary.json")
    if not isinstance(s, dict):
        return f"{per_item.parent}: no summary.json (not a finished dev run)"
    if s.get("smoke") is not False or s.get("reportable") is not True:
        return (f"{per_item.parent}: smoke={s.get('smoke')!r} reportable={s.get('reportable')!r}; smoke, dry and non-reportable runs "
                "never serve as a baseline")
    if protocol is not None:
        mismatch = protocol_mismatch(protocol, s.get("protocol"))
        if mismatch:
            return f"{per_item.parent}: {mismatch}"
    return None


def smoke_reasons(*, size: str, limits: dict[str, int] | None, task_ids: Iterable[str] | None, default_inputs: bool | None,
                  slices: Iterable[str], blocked: dict[str, str] | None, qa_max_tokens: int | None) -> list[str]:
    """Why a run is a smoke or dry run rather than a dev evaluation (empty when it is a real one)."""
    out = []
    if size == "smoke":
        out.append("size smoke")
    if limits:
        out.append(f"item limits {limits}")
    if task_ids:
        out.append("task-id filter")
    if default_inputs is None:
        out.append("inputs not recorded (run predates the smoke flag)")
    elif not default_inputs:
        out.append("non-default manifests, pools or output root (fixtures)")
    if set(slices) != set(SLICES):
        out.append(f"slices {sorted(slices)}")
    if blocked:
        out.append(f"blocked slices {sorted(blocked)}")
    if qa_max_tokens not in (None, 32768):
        out.append(f"QA budget {qa_max_tokens}")
    return out


def failure_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    c = Counter(r["run_status"] for r in records)
    return {"environment_failure": c.get(ENVIRONMENT_FAILURE, 0), "budget_exhausted": c.get(BUDGET_EXHAUSTED, 0),
            "agent_error": c.get(AGENT_ERROR, 0), "verifier_failure": c.get(VERIFIER_FAILURE, 0),
            "scoring_failure": c.get(SCORING_FAILURE, 0), "missing": c.get(MISSING, 0), "ok": c.get(OK, 0),
            "run_failures": sum(v for k, v in c.items() if k in RUN_FAILURE_STATUSES)}

