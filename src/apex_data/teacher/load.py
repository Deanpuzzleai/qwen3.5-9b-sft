"""Open-SWE-Traces v1.2 mini-swe-agent Qwen3.8-27B shards (source `oswt`, data plan 3.2): shard access, per-row
integrity assertions, the trajectory index with reference-patch sizes, and the side inputs of the slice (Scale-SWE-Verified
instance ids, SWE-rebench-V2 task rows, the pinned revisions).

Inputs (all read-only, under paths.RAW unless overridden):
  oswt/data/minisweagent/qwen38_27b/{swe-rebench-v2,scale-swe}/*.parquet   both folders share one schema: instance_id,
      repo, license, language, trajectory_id, messages, tools, resolved, metadata{category, teacher_model,
      reference_patch, model_patch}, hf_dataset_name
  swe_rebench_v2/data/*.parquet        nebius/SWE-rebench-V2 task rows (reference `patch`, licence, language, PR url)
  scale_swe_verified/data/*.parquet    PrimeIntellect/Scale-SWE-Verified: the Scale-SWE instances that pass gold validation
  data/manifests/raw_sources.json      pinned revisions (scripts/data/fetch_raw.py)
Outputs: in-memory records only; the build script owns every file.

Integrity: every row must carry metadata.teacher_model.name == "Qwen3.8-27B", enable_thinking == true, and the
hf_dataset_name of its folder; a violation raises TeacherDataError (nothing is skipped silently). The index pass reads
the messages column with pyarrow compute so the ~10 GB of text is never materialised as Python objects; full rows are
read later, per shard, only for the trajectories the draw reaches.

Patch sizes (plan 3.2 item 8, "size is the reference patch counted without test files"): files changed and added lines
of the reference patch over the files that `apex_data.grade.tamper.classify_path` does not class as tests. The upstream
task row's patch (SWE-rebench-V2 `patch`, Scale-SWE-Verified `patch`) is the reference; the trajectory's own
metadata.reference_patch is used where no upstream row exists, and the agreement of the two is reported.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from apex_data import paths
from apex_data.grade.tamper import classify_path, parse_diff

SOURCE = "oswt"
HF_REPO = "nvidia/Open-SWE-Traces"
DATASET_LICENSE = "CC-BY-4.0"
GENERATOR = "Qwen3.8-27B"
SHARD_ROOT = Path("data/minisweagent/qwen38_27b")
SUBSOURCES = {"swe-rebench-v2": "nebius/SWE-rebench-V2", "scale-swe": "AweAI-Team/Scale-SWE"}   # folder -> hf_dataset_name
RAW_KEYS = {"oswt": "oswt", "swe_rebench_v2": "swe_rebench_v2", "scale_swe_verified": "scale_swe_verified"}
LANGUAGE_CODES = {"go": "go", "python": "py", "js": "js", "ts": "ts"}   # dataset value -> plan / schema code

LIGHT_COLUMNS = ("instance_id", "repo", "license", "language", "trajectory_id", "resolved", "hf_dataset_name", "tools")
META_LEAVES = ("metadata.teacher_model.name", "metadata.teacher_model.enable_thinking",
               "metadata.teacher_model.reasoning_effort", "metadata.category", "metadata.model_patch.patch",
               "metadata.reference_patch.patch")


class TeacherDataError(ValueError):
    """A raw row violates an invariant asserted per row (teacher identity, thinking mode, upstream dataset)."""


def oswt_root(root: Path | None = None) -> Path:
    return Path(root) if root else paths.RAW / RAW_KEYS["oswt"]


def list_shards(root: Path | None = None, subsources: Iterable[str] = tuple(SUBSOURCES),
                limit: int | None = None) -> list[tuple[str, Path]]:
    """(subsource, shard path) for both folders, sorted; `limit` keeps the first N shards of each folder (smoke runs)."""
    out: list[tuple[str, Path]] = []
    for sub in subsources:
        if sub not in SUBSOURCES:
            raise ValueError(f"unknown subsource {sub!r}")
        d = oswt_root(root) / SHARD_ROOT / sub
        shards = sorted(d.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no parquet shards under {d} (scripts/data/fetch_raw.py --only oswt)")
        out += [(sub, s) for s in (shards[:limit] if limit else shards)]
    return out


def raw_revisions(manifest: Path | None = None) -> dict[str, dict[str, Any]]:
    """{key: {repo, revision, dest, n_files, bytes}} for the three inputs of this slice from data/manifests/raw_sources.json."""
    path = Path(manifest) if manifest else paths.RAW_SOURCES
    obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    src = obj.get("sources", obj) if isinstance(obj, dict) else {}
    out = {}
    for key in RAW_KEYS.values():
        e = src.get(key) or {}
        out[key] = {k: e.get(k) for k in ("repo", "revision", "dest", "n_files", "bytes", "fetch_run")}
    return out


# ------------------------------------------------------------------------------------------------- patches
@dataclass(frozen=True)
class PatchSize:
    files: int              # non-test files changed
    added: int              # added lines in non-test files
    files_all: int          # every file of the patch
    added_all: int

    @property
    def stratum(self) -> str:
        return size_stratum(self.files, self.added)


def size_stratum(files: int, added: int) -> str:
    """Plan 3.2 item 8: big = >= 3 non-test files AND >= 100 added lines; feature = >= 3 files OR >= 50 added lines;
    random = the rest."""
    if files >= 3 and added >= 100:
        return "big"
    if files >= 3 or added >= 50:
        return "feature"
    return "random"


def is_test_path(path: str) -> bool:
    return "test" in classify_path(path)


def patch_size(patch: str | None) -> PatchSize:
    changes = parse_diff(patch)
    nontest = [fc for fc in changes if not any(is_test_path(p) for p in fc.paths)]
    return PatchSize(files=len(nontest), added=sum(len(fc.added) for fc in nontest), files_all=len(changes),
                     added_all=sum(len(fc.added) for fc in changes))


def patch_paths(patch: str | None) -> list[str]:
    """Every path a diff touches (old and new side), order preserved, duplicates removed."""
    out: list[str] = []
    for fc in parse_diff(patch):
        for p in fc.paths:
            if p not in out:
                out.append(p)
    return out


def sha256_text(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------- the index
@dataclass
class IndexRow:
    """One trajectory: light columns, message statistics and patch facts (no message text)."""

    subsource: str
    shard: str
    row: int
    instance_id: str
    trajectory_id: str
    repo: str
    license: str
    language: str                      # dataset value (go | python | js | ts | rust | ...)
    resolved: int                      # 1 resolved, 0 not resolved, -1 unknown
    category: str
    reasoning_effort: str
    n_messages: int
    n_assistant: int
    n_tool: int
    n_user: int
    n_chars: int                       # content + reasoning_content of every message + tool-call arguments
    n_assistant_without_reasoning: int
    last_role: str
    tools_json: str                    # the `tools` column as a JSON list of its JSON strings
    model_patch_paths: list[str]
    model_patch_empty: bool
    ref_files: int                     # metadata.reference_patch without test files
    ref_added: int
    ref_files_all: int
    ref_added_all: int
    ref_patch_sha256: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.subsource, self.shard, self.row)

    @property
    def language_code(self) -> str | None:
        return LANGUAGE_CODES.get(self.language)


def _leaf(table: pa.Table, path: str) -> pa.Array:
    """Column for a dotted leaf path whether pyarrow returned the pruned parent struct or a flattened leaf column."""
    parts = path.split(".")
    if parts[0] in table.column_names and len(parts) > 1:
        col: Any = table.column(parts[0]).combine_chunks()
        for p in parts[1:]:
            col = col.field(p)
        return col
    name = parts[-1]
    if name not in table.column_names:
        raise TeacherDataError(f"column {path} not found in shard (have {table.column_names})")
    return table.column(name).combine_chunks()


def message_stats(messages: pa.ListArray) -> dict[str, np.ndarray]:
    """Per-row counts over the nested `messages` column without converting it to Python objects."""
    n_rows = len(messages)
    flat = messages.flatten()
    parent = pc.list_parent_indices(messages).to_numpy()
    role = flat.field("role")
    content_len = pc.fill_null(pc.utf8_length(flat.field("content")), 0).to_numpy(zero_copy_only=False)
    reasoning = flat.field("reasoning_content")
    reasoning_len = pc.fill_null(pc.utf8_length(reasoning), 0).to_numpy(zero_copy_only=False)
    reasoning_blank = pc.fill_null(pc.utf8_length(pc.utf8_trim_whitespace(pc.fill_null(reasoning, ""))), 0)\
        .to_numpy(zero_copy_only=False) == 0
    is_assistant = pc.equal(role, "assistant").to_numpy(zero_copy_only=False)
    is_tool = pc.equal(role, "tool").to_numpy(zero_copy_only=False)
    is_user = pc.equal(role, "user").to_numpy(zero_copy_only=False)
    calls = flat.field("tool_calls")
    call_msg = pc.list_parent_indices(calls).to_numpy()
    call_args = pc.list_flatten(calls).field("function").field("arguments")
    call_len = pc.fill_null(pc.utf8_length(call_args), 0).to_numpy(zero_copy_only=False)
    chars = np.bincount(parent, weights=content_len + reasoning_len, minlength=n_rows)
    if len(call_msg):
        chars += np.bincount(parent[call_msg], weights=call_len, minlength=n_rows)
    offsets = messages.offsets.to_numpy()
    n_messages = (offsets[1:] - offsets[:-1]).astype(np.int64)
    last_role = role.take(pa.array(np.maximum(offsets[1:] - 1, 0))).to_pylist()
    return {
        "n_messages": n_messages,
        "n_assistant": np.bincount(parent, weights=is_assistant, minlength=n_rows).astype(np.int64),
        "n_tool": np.bincount(parent, weights=is_tool, minlength=n_rows).astype(np.int64),
        "n_user": np.bincount(parent, weights=is_user, minlength=n_rows).astype(np.int64),
        "n_chars": chars.astype(np.int64),
        "n_assistant_without_reasoning": np.bincount(parent, weights=is_assistant & reasoning_blank, minlength=n_rows).astype(np.int64),
        "last_role": np.array([r if (n > 0 and r is not None) else "" for r, n in zip(last_role, n_messages)], dtype=object),
    }


def scan_shard(subsource: str, path: Path) -> list[IndexRow]:
    """Index every row of one shard; asserts the per-row invariants (TeacherDataError names the shard and row)."""
    pf = pq.ParquetFile(path)
    light = pf.read(columns=[*LIGHT_COLUMNS, "metadata"])
    cols = {c: light.column(c).to_pylist() for c in LIGHT_COLUMNS}
    leaves = {p: _leaf(light, p).to_pylist() for p in META_LEAVES}
    stats = message_stats(pf.read(columns=["messages"]).column("messages").combine_chunks())
    want_hf = SUBSOURCES[subsource]
    out: list[IndexRow] = []
    for i in range(light.num_rows):
        teacher, thinking, hf = (leaves["metadata.teacher_model.name"][i], leaves["metadata.teacher_model.enable_thinking"][i],
                                 cols["hf_dataset_name"][i])
        if teacher != GENERATOR or thinking is not True or hf != want_hf:
            raise TeacherDataError(f"{subsource}/{path.name} row {i}: teacher_model.name={teacher!r} enable_thinking={thinking!r} "
                                   f"hf_dataset_name={hf!r}; expected {GENERATOR!r} / True / {want_hf!r}")
        mpatch, rpatch = leaves["metadata.model_patch.patch"][i], leaves["metadata.reference_patch.patch"][i]
        size = patch_size(rpatch)
        out.append(IndexRow(
            subsource=subsource, shard=path.name, row=i, instance_id=cols["instance_id"][i],
            trajectory_id=cols["trajectory_id"][i], repo=cols["repo"][i] or "", license=cols["license"][i] or "",
            language=cols["language"][i] or "", resolved=int(cols["resolved"][i]) if cols["resolved"][i] is not None else -1,
            category=leaves["metadata.category"][i] or "", reasoning_effort=leaves["metadata.teacher_model.reasoning_effort"][i] or "",
            n_messages=int(stats["n_messages"][i]), n_assistant=int(stats["n_assistant"][i]), n_tool=int(stats["n_tool"][i]),
            n_user=int(stats["n_user"][i]), n_chars=int(stats["n_chars"][i]),
            n_assistant_without_reasoning=int(stats["n_assistant_without_reasoning"][i]), last_role=str(stats["last_role"][i]),
            tools_json=json.dumps(list(cols["tools"][i] or [])), model_patch_paths=patch_paths(mpatch),
            model_patch_empty=not (mpatch or "").strip(), ref_files=size.files, ref_added=size.added,
            ref_files_all=size.files_all, ref_added_all=size.added_all, ref_patch_sha256=sha256_text(rpatch)))
    return out


_INDEX_SCHEMA = pa.schema([
    ("subsource", pa.string()), ("shard", pa.string()), ("row", pa.int64()), ("instance_id", pa.string()),
    ("trajectory_id", pa.string()), ("repo", pa.string()), ("license", pa.string()), ("language", pa.string()),
    ("resolved", pa.int64()), ("category", pa.string()), ("reasoning_effort", pa.string()), ("n_messages", pa.int64()),
    ("n_assistant", pa.int64()), ("n_tool", pa.int64()), ("n_user", pa.int64()), ("n_chars", pa.int64()),
    ("n_assistant_without_reasoning", pa.int64()), ("last_role", pa.string()), ("tools_json", pa.string()),
    ("model_patch_paths", pa.list_(pa.string())), ("model_patch_empty", pa.bool_()), ("ref_files", pa.int64()),
    ("ref_added", pa.int64()), ("ref_files_all", pa.int64()), ("ref_added_all", pa.int64()), ("ref_patch_sha256", pa.string())])
assert tuple(_INDEX_SCHEMA.names) == tuple(f.name for f in fields(IndexRow))


def index_to_table(index: list[IndexRow]) -> pa.Table:
    return pa.Table.from_pylist([asdict(r) for r in index], schema=_INDEX_SCHEMA)


def index_from_table(table: pa.Table) -> list[IndexRow]:
    return [IndexRow(**{k: (list(v) if k == "model_patch_paths" else v) for k, v in d.items()}) for d in table.to_pylist()]


def read_rows(subsource: str, shard_path: Path, rows: list[int],
              columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Raw rows of one shard, every column or only `columns` (one read of the shard; the order of `rows` is kept)."""
    table = pq.read_table(shard_path, columns=None if columns is None else list(dict.fromkeys([*columns, "hf_dataset_name"])))
    if table.column("hf_dataset_name")[0].as_py() != SUBSOURCES[subsource]:
        raise TeacherDataError(f"{shard_path} is not a {subsource} shard")
    return table.take(pa.array(rows, type=pa.int64())).to_pylist()


# -------------------------------------------------------------------------------------------- side inputs
def scale_swe_verified_ids(root: Path | None = None) -> set[str]:
    """Instance ids of PrimeIntellect/Scale-SWE-Verified (rows that passed the gold-patch validation)."""
    d = Path(root) if root else paths.RAW / RAW_KEYS["scale_swe_verified"]
    files = sorted((d / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Scale-SWE-Verified parquet under {d}/data (scripts/data/fetch_raw.py --only scale_swe_verified)")
    return {i for f in files for i in pq.read_table(f, columns=["instance_id"]).column("instance_id").to_pylist()}


@dataclass(frozen=True)
class UpstreamTask:
    instance_id: str
    repo: str
    language: str
    license: str | None
    pr: str | None
    patch_sha256: str
    size: PatchSize


def pr_number(instance_id: str, pr_url: str | None = None) -> str | None:
    """PR number from a PR url, else from the instance id suffix (`owner__repo-<pr>` or `owner_repo_pr<pr>`)."""
    if pr_url and pr_url.rstrip("/").rsplit("/", 1)[-1].isdigit():
        return pr_url.rstrip("/").rsplit("/", 1)[-1]
    for sep in ("_pr", "-"):
        tail = instance_id.rsplit(sep, 1)[-1]
        if tail.isdigit() and tail != instance_id:
            return tail
    return None


def load_upstream(root_rebench: Path | None = None, root_scale: Path | None = None) -> dict[tuple[str, str], UpstreamTask]:
    """{(subsource, instance_id): UpstreamTask} from SWE-rebench-V2 and Scale-SWE-Verified (reference patch sizes without
    test files, licence, language, PR). Test patches are never read."""
    out: dict[tuple[str, str], UpstreamTask] = {}
    rb = Path(root_rebench) if root_rebench else paths.RAW / RAW_KEYS["swe_rebench_v2"]
    for f in sorted((rb / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["instance_id", "repo", "language", "license", "patch", "meta"])
        for rec in t.to_pylist():
            meta = rec.get("meta") or {}
            out[("swe-rebench-v2", rec["instance_id"])] = UpstreamTask(
                rec["instance_id"], rec.get("repo") or "", rec.get("language") or "", rec.get("license"),
                pr_number(rec["instance_id"], meta.get("pr_url")), sha256_text(rec.get("patch")), patch_size(rec.get("patch")))
    sc = Path(root_scale) if root_scale else paths.RAW / RAW_KEYS["scale_swe_verified"]
    for f in sorted((sc / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["instance_id", "repo", "language", "patch", "github_url"])
        for rec in t.to_pylist():
            out[("scale-swe", rec["instance_id"])] = UpstreamTask(
                rec["instance_id"], rec.get("repo") or "", rec.get("language") or "", None, pr_number(rec["instance_id"]),
                sha256_text(rec.get("patch")), patch_size(rec.get("patch")))
    return out


def group_by_instance(rows: Iterable[IndexRow]) -> dict[tuple[str, str], list[IndexRow]]:
    per: dict[tuple[str, str], list[IndexRow]] = defaultdict(list)
    for r in rows:
        per[(r.subsource, r.instance_id)].append(r)
    return dict(per)
