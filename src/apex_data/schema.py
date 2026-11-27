"""Record schemas shared by every work package (src/apex_data/README.md, "Schemas"). JSONL on disk, one record per
line, UTF-8; rendered training rows are Parquet (apex_train.data_format rows plus the provenance columns below).

Every record carries the fields below; modules may add keys under `extra` only. Validators raise SchemaError. Text that
must never be stored in the repository (benchmark inputs) has no field here by design.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Source codes of docs/DATA_PLAN.md revision 4.3 (sections 3 and 10). Every task_id, episode and row names one.
SOURCES = {
    "oswt": "nvidia/Open-SWE-Traces v1.2, mini-swe-agent, Qwen3.8-27B thinking (SWE teacher rows; subsource "
            "swe-rebench-v2 or scale-swe)",
    "th3k": "shunzou05/TerminalHorizon-3K (DeepSeek-V4-Pro Terminus-2 episodes; authors' filter + handshake, unverified)",
    "tgr": "ai-and/termgrade-trajectories (DeepSeek-V4-Pro Terminus-2 episodes with verifier reward)",
    "tgr_env": "ai-and/termgrade-environments (Harbor task directories: terminal dev set and stage-2 pool)",
    "swesmith": "SWE-bench/SWE-smith-py, SWE-bench/SWE-smith-go (task instances: SWE dev set and stage-2 pool)",
    "nm_math": "nvidia/Nemotron-SFT-Math-v4 (AoPS chain-of-thought rows, retention)",
    "nm_ifchat": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 (IF and chat rows, retention)",
    "nm_sci": "nvidia/Nemotron-Science-v1 (MCQ and RQA rows, retention)",
    "devqa": "dev STEM QA: GPQA Diamond (ids only), reasoning-gym items, AIME 2025 (report only)",
}
OSWT_SUBSOURCES = ("swe-rebench-v2", "scale-swe")
FAMILIES = ("swe", "terminal", "qa", "chat")
SPLITS = ("train", "dev", "pilot", "stage2", "replay_reserve", "excluded")
SCAFFOLDS = ("mini_swe_toolcall", "terminus2", "opencode", "hle_qa", "chat")
EXIT_REASONS = ("submit", "task_complete", "step_cap", "token_cap", "time_cap", "format_errors", "error", "unknown")
ROW_KINDS = ("swe_episode", "terminal_turn", "retention")          # data_format modes: episode, per_turn, episode
RETENTION_PARTS = ("math", "if", "chat", "sci_mcq", "sci_rqa")
LABEL_SOURCES = ("ours", "upstream")
# How a row's success label was established (plan 2.1): an execution verifier (upstream `resolved`, TermGrade `reward`,
# our stage-2 replay) or, for TerminalHorizon only, the corpus authors' filter plus the double task_complete handshake.
# Retention rows carry `reference` (answer or constraint checked) or `format` (Science RQA ships no reference answers).
LABEL_PROVENANCE = ("verifier", "authors_filter+handshake", "reference", "format")
SIZE_STRATA = ("big", "feature", "random")                        # plan 3.2: >=3 files & >=100 lines / >=3 files or >=50 / random
# Terminal-Bench 2.0 category groups of the TerminalHorizon quota (plan 3.3; quotas live in the WP4 builder)
CATEGORY_GROUPS = ("software_engineering", "security", "system_administration", "scientific_computing",
                   "machine_learning", "data_processing", "rest")

# Columns every rendered training row carries next to apex_train.data_format.ROW_COLUMNS (flat scalars; Arrow-safe).
# `None` is allowed where a field does not apply (e.g. oswt_subsource on terminal rows).
PROVENANCE_FIELDS = (
    "row_kind",            # ROW_KINDS
    "task_id",             # "<source>:<upstream_id>"; the split and dedup unit
    "episode_id",          # upstream trajectory id (teacher rows), our episode id (rollouts), the source row id (retention)
    "hf_repo",             # dataset id the row came from
    "hf_revision",         # pinned dataset revision (data/manifests/raw_sources.json)
    "repo",                # owner/name of the code repository (SWE rows)
    "repo_license",        # SPDX id of the code repository (SWE rows)
    "dataset_license",     # licence of the dataset row
    "generator",           # teacher model that produced the loss-bearing text
    "label_provenance",    # LABEL_PROVENANCE
    "oswt_subsource",      # OSWT_SUBSOURCES on SWE rows
    "size_stratum",        # SIZE_STRATA on SWE rows
    "patch_files",         # non-test files changed by the reference patch (SWE rows)
    "patch_added_lines",   # added lines of the reference patch without test files (SWE rows)
    "category",            # TerminalHorizon category (terminal rows) or retention topic label
    "category_group",      # CATEGORY_GROUPS on TerminalHorizon rows
    "retention_part",      # RETENTION_PARTS on retention rows
    "fits_48k",            # the row also fits the 48K fallback cap
    "dup_group",           # dedup cluster id (screens/dedup.py)
    "decontam_flags",      # JSON string of screen flags (empty "{}" when clean)
)


class SchemaError(ValueError):
    pass


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def task_id(source: str, upstream_id: str) -> str:
    if source not in SOURCES:
        raise SchemaError(f"unknown source {source!r}")
    return f"{source}:{upstream_id}"


@dataclass
class Task:
    task_id: str                       # unique across sources: f"{source}:{upstream_id}"
    source: str                        # SOURCES key
    family: str                        # swe | terminal | qa | chat
    language: str | None               # go | py | js | ts | bash | null
    repo: str | None                   # owner/name
    upstream_id: str                   # instance_id / task slug / prompt id in the source
    hf_revision: str | None            # source snapshot revision
    base_commit: str | None = None
    pr: str | None = None              # PR number or commit of the fix, for lineage dedup
    image: str | None = None           # container image reference or local build tag
    problem_statement_sha: str | None = None
    tests: dict[str, Any] = field(default_factory=dict)   # f2p, p2p, test_cmd (ids only)
    gold_patch_sha: str | None = None  # the gold patch itself stays in the pool file, never in rendered text
    licence: dict[str, Any] = field(default_factory=dict) # {"dataset": ..., "repo_spdx": ..., "status": ok|pending|excluded}
    screens: dict[str, Any] = field(default_factory=dict) # {"d1": .., "d2": .., "d3": .., "d4": .., "dedup_group": .., "lineage_key": ..}
    split: str = "train"
    stratum: dict[str, Any] = field(default_factory=dict) # language, difficulty band, category ...
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.source not in SOURCES: raise SchemaError(f"unknown source {self.source}")
        if self.family not in FAMILIES: raise SchemaError(f"unknown family {self.family}")
        if self.split not in SPLITS: raise SchemaError(f"unknown split {self.split}")
        if not self.task_id.startswith(self.source + ":"): raise SchemaError("task_id must be '<source>:<upstream_id>'")


@dataclass
class Episode:
    episode_id: str                    # f"{task_id}#{round}#{policy}#{sample_index}"
    task_id: str
    round: str                         # "teacher" | "pilot" | "dev" | "stage2"
    policy: str                        # base | sft | stage2 | <teacher name>
    sample_index: int
    scaffold: str
    messages: list[dict[str, Any]]     # harness-faithful transcript: role, content, reasoning_content, tool_calls, tool_call_id
    tools: list[dict[str, Any]] | None
    exit_reason: str
    steps: int
    output_tokens: int
    wall_sec: float
    patch: str | None                  # final diff for SWE; null for terminal (final state judged in place)
    request_log: str | None            # path to the logged request bodies (model, seed, sampling)
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.scaffold not in SCAFFOLDS: raise SchemaError(f"unknown scaffold {self.scaffold}")
        if self.exit_reason not in EXIT_REASONS: raise SchemaError(f"unknown exit_reason {self.exit_reason}")


@dataclass
class Verdict:
    episode_id: str
    task_id: str
    r_pass: int                        # 1 iff all F2P pass and all P2P still pass on a pristine image, in two runs
    runs: list[dict[str, Any]]         # [{"pass": bool, "f2p": {...}, "p2p": {...}, "log": path, "seconds": float}]
    tamper_flags: list[str]            # tests_touched | ci_touched | skip_markers | empty_diff | no_submit | ...
    verifier: str                      # swesmith | harbor_tests | reference_match | constraint
    verifier_version: str
    label_source: str = "ours"
    graded_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.r_pass not in (0, 1): raise SchemaError("r_pass must be 0 or 1")
        if self.label_source not in LABEL_SOURCES: raise SchemaError("label_source")


_ENUMS = {"row_kind": ROW_KINDS, "label_provenance": LABEL_PROVENANCE, "oswt_subsource": OSWT_SUBSOURCES,
          "size_stratum": SIZE_STRATA, "category_group": CATEGORY_GROUPS, "retention_part": RETENTION_PARTS}
_REQUIRED = {  # provenance fields that must be non-null per row kind (plus _REQUIRED_OSWT on Open-SWE-Traces rows)
    "swe_episode": ("task_id", "episode_id", "hf_repo", "hf_revision", "repo", "repo_license", "generator",
                    "label_provenance", "fits_48k"),
    "terminal_turn": ("task_id", "episode_id", "hf_repo", "hf_revision", "generator", "label_provenance", "fits_48k"),
    "retention": ("task_id", "hf_repo", "hf_revision", "generator", "label_provenance", "retention_part"),
}
_REQUIRED_OSWT = ("oswt_subsource", "size_stratum", "patch_files", "patch_added_lines")   # SWE teacher rows only (WP3)


def validate_provenance(row: dict[str, Any]) -> None:
    """Raise SchemaError unless `row` carries every provenance column with legal values for its row kind."""
    missing = [k for k in PROVENANCE_FIELDS if k not in row]
    if missing:
        raise SchemaError(f"row lacks provenance fields {missing}")
    kind = row["row_kind"]
    if kind not in ROW_KINDS:
        raise SchemaError(f"unknown row_kind {kind!r}")
    for k, allowed in _ENUMS.items():
        if row[k] is not None and row[k] not in allowed:
            raise SchemaError(f"{k}={row[k]!r} not in {allowed}")
    src = str(row["task_id"]).split(":", 1)[0]
    if src not in SOURCES:
        raise SchemaError(f"task_id {row['task_id']!r} does not start with a source code")
    required = _REQUIRED[kind] + (_REQUIRED_OSWT if src == "oswt" else ())
    null = [k for k in required if row[k] is None]
    if null:
        raise SchemaError(f"{kind} row {row.get('task_id')!r} has null {null}")
    if kind == "terminal_turn" and src == "th3k" and row["label_provenance"] != "authors_filter+handshake":
        raise SchemaError("TerminalHorizon rows must be labelled authors_filter+handshake")
    if kind == "terminal_turn" and src == "tgr" and row["label_provenance"] != "verifier":
        raise SchemaError("TermGrade rows must be labelled verifier")
    try:
        json.loads(row["decontam_flags"] or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"decontam_flags must be a JSON string: {exc}") from exc


def write_jsonl(path: Path | str, records: Iterable[Any]) -> int:
    """Atomic JSONL write (tmp file + rename); dataclasses are converted with asdict. Returns the record count."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            d = asdict(r) if hasattr(r, "__dataclass_fields__") else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n"); n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def task_from_dict(d: dict[str, Any]) -> Task:
    t = Task(**{k: v for k, v in d.items() if k in Task.__dataclass_fields__}); t.validate(); return t
