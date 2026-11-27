"""Candidate units of the three SFT pools at one row cap, screen drops and same-stratum replacements (docs/DATA_PLAN.md
revision 4.3 sections 3.2 to 3.4, 4 and 7; WP7, unit U2.4).

The pools are built by WP3 to WP5; this module only reads them:
- SWE (`data/pools/swe_teacher/`): `rows.parquet` holds every row drawn at either cap (`selected_64k`, `selected_48k`),
  and `selection_<cap>.json` lists each cap's draw in draw order with its quota part. One unit = one episode row.
- Terminal (`data/pools/terminal_teacher/rows_<cap>.parquet`): two per-turn rows per episode. One unit = one episode;
  its rows travel together through every selection, split and trim.
- Retention (`data/pools/retention/rows.parquet`): one unit = one row. `replay_reserve.parquet` is stage-2 data and is
  never read here.

Screens never rewrite pool files (src/apex_data/README.md): the screens passes list dropped ids in
`data/manifests/screens_<pass>.json["dropped"]`. `apply_drops` removes every unit of a task dropped by the p1_pools pass
(task text) and the unit of every row dropped by the p2_rows pass (rendered rows). A dropped SWE
unit is replaced by a spare row of the pool, a row rendered for the other cap that also fits this cap, from the same
quota part and size stratum (else the same part, recorded as borrowed), respecting one row per instance and at most 5
rows per repository. The terminal and retention pools hold no rendered spares (the terminal pool already uses every
admissible TermGrade episode; the retention replay reserve belongs to stage 2), so their drops are reported as
shortfalls. Also here: the p1 screen items (task text decoded from the rendered rows) that `run_screens.py` reads.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_data import paths

SLICES = ("swe", "terminal", "retention")
SLICE_OF_KIND = {"swe_episode": "swe", "terminal_turn": "terminal", "retention": "retention"}
CAPS = {"64k": 65536, "48k": 49152}
SWE_PARTS = ("go", "ts", "js", "py_rebench", "py_scale")
MAX_ROWS_PER_REPOSITORY = 5
TOKEN_COLUMNS = ("input_ids", "assistant_masks")
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    """What moves as a whole: an SWE episode row, a terminal episode (its trained turns), a retention row."""

    unit_id: str                 # episode_id (SWE, terminal) or row_key (retention)
    task_id: str                 # split and dedup unit
    slice: str                   # SLICES
    cell: str                    # SWE quota part, terminal source code, retention part
    stratum: str | None          # SWE size stratum, TerminalHorizon category group, None elsewhere
    row_keys: tuple[str, ...]
    seq_tokens: int
    loss_tokens: int
    order: int                   # draw order inside the pool's selection (spares: seeded rank)
    repo: str | None = None


@dataclass
class PoolFiles:
    """Where the pools live (defaults: data/pools/)."""

    swe_rows: Path
    swe_selection: dict[str, Path]
    swe_manifest: Path
    terminal_rows: dict[str, Path]
    terminal_manifest: Path
    retention_rows: Path
    retention_manifest: Path

    @classmethod
    def under(cls, pools_dir: Path | str = paths.POOLS) -> PoolFiles:
        d = Path(pools_dir)
        return cls(swe_rows=d / "swe_teacher" / "rows.parquet",
                   swe_selection={c: d / "swe_teacher" / f"selection_{c}.json" for c in CAPS},
                   swe_manifest=d / "swe_teacher" / "manifest.json",
                   terminal_rows={c: d / "terminal_teacher" / f"rows_{c}.parquet" for c in CAPS},
                   terminal_manifest=d / "terminal_teacher" / "manifest.json",
                   retention_rows=d / "retention" / "rows.parquet",
                   retention_manifest=d / "retention" / "manifest.json")

    def row_files(self, cap: str) -> dict[str, Path]:
        return {"swe": self.swe_rows, "terminal": self.terminal_rows[cap], "retention": self.retention_rows}

    def all_files(self) -> list[Path]:
        return [self.swe_rows, *self.swe_selection.values(), self.swe_manifest, *self.terminal_rows.values(),
                self.terminal_manifest, self.retention_rows, self.retention_manifest]


@dataclass
class CapCandidates:
    """One cap's candidates: each pool's own selection (draw order) plus SWE spares, and the metadata of every row."""

    cap: str
    selected: dict[str, list[Unit]]
    spares: dict[str, list[Unit]]
    rows: dict[str, dict[str, Any]]              # row_key -> metadata and provenance columns (no token lists)
    files: dict[str, Path]                       # slice -> rows file holding the token lists

    def all_units(self) -> list[Unit]:
        return [u for s in SLICES for u in (*self.selected.get(s, []), *self.spares.get(s, []))]


# --- reading ---------------------------------------------------------------------------------------------------------
def read_meta(path: Path | str) -> list[dict[str, Any]]:
    """Every column of a rows file except the token lists, as dicts in file order."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c not in TOKEN_COLUMNS]
    return pq.read_table(path, columns=cols).to_pylist()


def swe_part(row: Mapping[str, Any]) -> str:
    """Quota part of an SWE row (plan 3.2): Go / TS / JS, Python split by task source."""
    if row.get("oswt_subsource") == "scale-swe":
        return "py_scale"
    lang = str(row.get("language") or "")
    return "py_rebench" if lang == "py" else lang


def rank(seed: int, *labels: Any) -> float:
    """A seeded, order-free rank for `labels` (stable across runs and Python versions)."""
    return random.Random(":".join(map(str, (seed, *labels)))).random()


def unit_id_of(row: Mapping[str, Any]) -> str:
    """The unit a rendered row belongs to: its episode (SWE, terminal) or the row itself (retention)."""
    return str(row["row_key"] if row["row_kind"] == "retention" else row["episode_id"])


def _row_unit(row: Mapping[str, Any], *, cell: str, stratum: str | None, order: int) -> Unit:
    return Unit(unit_id=unit_id_of(row),
                task_id=str(row["task_id"]), slice=SLICE_OF_KIND[row["row_kind"]], cell=cell, stratum=stratum,
                row_keys=(str(row["row_key"]),), seq_tokens=int(row["n_tokens"]), loss_tokens=int(row["n_assistant_tokens"]),
                order=order, repo=row.get("repo"))


def episode_units(rows: Iterable[Mapping[str, Any]], *, cell_key: str = "source", stratum_key: str = "category_group") -> list[Unit]:
    """Group per-turn rows into episode units, in order of first appearance."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["episode_id"]), []).append(r)
    out = []
    for i, (ep, rs) in enumerate(groups.items()):
        tasks = {str(r["task_id"]) for r in rs}
        if len(tasks) != 1:
            raise SelectionError(f"episode {ep} spans tasks {sorted(tasks)}")
        first = rs[0]
        out.append(Unit(unit_id=ep, task_id=str(first["task_id"]), slice=SLICE_OF_KIND[first["row_kind"]],
                        cell=str(first.get(cell_key) or str(first["task_id"]).split(":", 1)[0]), stratum=first.get(stratum_key),
                        row_keys=tuple(str(r["row_key"]) for r in rs), seq_tokens=sum(int(r["n_tokens"]) for r in rs),
                        loss_tokens=sum(int(r["n_assistant_tokens"]) for r in rs), order=i, repo=None))
    return out


def swe_candidates(meta: list[dict[str, Any]], selection: Mapping[str, Any], cap: str, seed: int = paths.SEED) -> tuple[list[Unit], list[Unit]]:
    """(the cap's draw in draw order, spare rows that fit the cap in seeded order). A spare is a pool row not drawn at
    this cap whose instance is not drawn at this cap either."""
    by_key = {str(r["row_key"]): r for r in meta}
    flag = f"selected_{cap}"
    selected: list[Unit] = []
    for i, s in enumerate(selection["rows"]):
        row = by_key.get(str(s["row_key"]))
        if row is None:
            raise SelectionError(f"selection_{cap}.json row {s['row_key']} is not in the SWE rows file")
        if flag in row and not row[flag]:
            raise SelectionError(f"row {s['row_key']} is in selection_{cap}.json but {flag} is false")
        part = swe_part(row)
        if s.get("part") and s["part"] != part:
            raise SelectionError(f"row {s['row_key']}: selection part {s['part']} != {part} from its provenance")
        selected.append(_row_unit(row, cell=part, stratum=row.get("size_stratum"), order=i))
    if len({u.task_id for u in selected}) != len(selected):
        raise SelectionError(f"selection_{cap}.json holds more than one row per instance")
    chosen_tasks = {u.task_id for u in selected}
    chosen_keys = {k for u in selected for k in u.row_keys}
    limit = CAPS[cap]
    spare_rows = [r for r in meta if str(r["row_key"]) not in chosen_keys and str(r["task_id"]) not in chosen_tasks
                  and int(r["n_tokens"]) <= limit and (cap != "48k" or bool(r.get("fits_48k")))]
    spare_rows.sort(key=lambda r: (rank(seed, "swe_spare", cap, r["row_key"]), str(r["row_key"])))
    spares = [_row_unit(r, cell=swe_part(r), stratum=r.get("size_stratum"), order=i) for i, r in enumerate(spare_rows)]
    return selected, spares


def load_candidates(files: PoolFiles, cap: str, seed: int = paths.SEED) -> CapCandidates:
    if cap not in CAPS:
        raise SelectionError(f"unknown cap {cap!r} (one of {list(CAPS)})")
    rows: dict[str, dict[str, Any]] = {}
    selected: dict[str, list[Unit]] = {}
    spares: dict[str, list[Unit]] = {}

    swe_meta = read_meta(files.swe_rows)
    selection = json.loads(files.swe_selection[cap].read_text(encoding="utf-8"))
    selected["swe"], spares["swe"] = swe_candidates(swe_meta, selection, cap, seed)
    term_meta = read_meta(files.terminal_rows[cap])
    selected["terminal"], spares["terminal"] = episode_units(term_meta), []
    ret_meta = read_meta(files.retention_rows)
    selected["retention"] = [_row_unit(r, cell=str(r["retention_part"]), stratum=None, order=i) for i, r in enumerate(ret_meta)]
    spares["retention"] = []
    for r in (*swe_meta, *term_meta, *ret_meta):
        key = str(r["row_key"])
        if key in rows:
            raise SelectionError(f"row_key {key} appears twice across the pools")
        rows[key] = r
    for s in SLICES:
        for u in selected[s]:
            for k in u.row_keys:
                if int(rows[k]["n_tokens"]) > CAPS[cap]:
                    raise SelectionError(f"{s} row {k} has {rows[k]['n_tokens']} tokens > the {cap} cap")
    return CapCandidates(cap, selected, spares, rows, files.row_files(cap))


# --- screen drops ----------------------------------------------------------------------------------------------------
def screen_drops(manifest: Mapping[str, Any] | None, pass_name: str) -> dict[str, list[str]]:
    """{item id: ["<pass>:<rule>", ...]} from a screens manifest (`dropped`), empty when the manifest is absent."""
    if not manifest:
        return {}
    return {str(i): [f"{pass_name}:{r}" for r in rules] for i, rules in (manifest.get("dropped") or {}).items()}


def row_drops_to_units(dropped: Mapping[str, list[str]], rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """p2_rows ids (`row:<sample_id>`; one id per episode for per-turn rows) -> {unit_id: rules}: a hit on a rendered
    row drops its unit (the episode with both trained turns, or the row). Other units of the same task stay: they are
    in the screened files too, so a hit caused by the shared task text drops them in the same pass."""
    unit_of_sample = {str(r["sample_id"]): unit_id_of(r) for r in rows}
    out: dict[str, list[str]] = defaultdict(list)
    unknown = []
    for item, rules in dropped.items():
        sample = item.split(":", 1)[1] if item.startswith("row:") else item
        u = unit_of_sample.get(sample)
        if u is None:
            unknown.append(item)
            continue
        out[u].extend(r for r in rules if r not in out[u])
    if unknown:
        raise SelectionError(f"p2_rows drops name rows that no pool holds: {unknown[:5]}")
    return dict(out)


@dataclass
class SliceResult:
    slice: str
    kept: list[Unit]
    dropped: list[tuple[Unit, list[str]]] = field(default_factory=list)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    shortfall: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"units_kept": len(self.kept), "rows_kept": sum(len(u.row_keys) for u in self.kept),
                "units_dropped": len(self.dropped),
                "dropped": [{"unit_id": u.unit_id, "task_id": u.task_id, "cell": u.cell, "stratum": u.stratum, "rules": rules}
                            for u, rules in self.dropped],
                "replacements": self.replacements, "shortfall_by_cell": self.shortfall,
                "units_by_cell": dict(sorted(Counter(u.cell for u in self.kept).items()))}


def apply_drops(cands: CapCandidates, drops: Mapping[str, list[str]], unit_drops: Mapping[str, list[str]] | None = None, *,
                max_per_repo: int = MAX_ROWS_PER_REPOSITORY) -> dict[str, SliceResult]:
    """Remove every unit whose task is in `drops` (p1: task-level items) or whose unit id is in `unit_drops` (p2: rendered
    rows); replace dropped SWE units from the spares (same part and size stratum, else the same part as a recorded
    borrow); report the rest as shortfalls per cell."""
    unit_drops = unit_drops or {}

    def rules_of(u: Unit) -> list[str] | None:
        if u.task_id not in drops and u.unit_id not in unit_drops:
            return None
        return sorted({*drops.get(u.task_id, []), *unit_drops.get(u.unit_id, [])})

    out: dict[str, SliceResult] = {}
    for s in SLICES:
        sel = cands.selected.get(s, [])
        res = SliceResult(s, [u for u in sel if rules_of(u) is None])
        res.dropped = [(u, rules_of(u)) for u in sel if rules_of(u) is not None]
        spares = [u for u in cands.spares.get(s, []) if rules_of(u) is None]
        used_tasks = {u.task_id for u in res.kept}
        per_repo = Counter(u.repo for u in res.kept if u.repo)
        shortfall: Counter[str] = Counter()
        for u, _ in res.dropped:
            pick, how = None, None
            for want_stratum in (True, False):
                for sp in spares:
                    if sp.task_id in used_tasks or sp.cell != u.cell or (want_stratum and sp.stratum != u.stratum):
                        continue
                    if sp.repo and per_repo[sp.repo] >= max_per_repo:
                        continue
                    pick, how = sp, ("same_stratum" if want_stratum else "same_part_borrowed")
                    break
                if pick:
                    break
            if pick is None:
                shortfall[u.cell] += 1
                continue
            res.kept.append(pick)
            used_tasks.add(pick.task_id)
            if pick.repo:
                per_repo[pick.repo] += 1
            res.replacements.append({"dropped": u.unit_id, "replacement": pick.unit_id, "cell": u.cell,
                                     "stratum_dropped": u.stratum, "stratum_replacement": pick.stratum, "how": how})
        res.shortfall = dict(sorted(shortfall.items()))
        out[s] = res
    return out


# --- p1 screen items -------------------------------------------------------------------------------------------------
def message_spans(text: str) -> list[tuple[str, str]]:
    """(role, content) of every chat-template message in a decoded row."""
    out = []
    for chunk in text.split(IM_START)[1:]:
        role, _, rest = chunk.partition("\n")
        out.append((role.strip(), rest.split(IM_END, 1)[0]))
    return out


def first_user_message(text: str) -> str:
    return next((c for r, c in message_spans(text) if r == "user"), "")


def strip_instance_template(user_text: str, prefix: str, suffix: str) -> tuple[str, bool]:
    """The task inside a mini-swe-agent instance message (prefix + task + suffix); (whole text, False) otherwise."""
    if prefix and suffix and user_text.startswith(prefix) and user_text.endswith(suffix) and len(user_text) >= len(prefix) + len(suffix):
        return user_text[len(prefix): len(user_text) - len(suffix)], True
    return user_text, False


def instance_template_parts(swe_manifest: Mapping[str, Any]) -> tuple[str, str]:
    """(prefix, suffix) of the installed mini-swe-agent instance template around the task, rendered with the template
    variables the SWE pool recorded; ("", "") when the installed config is not readable."""
    import yaml
    from jinja2 import StrictUndefined, Template

    tpl = swe_manifest.get("templates") or {}
    cfg_path = Path(str(tpl.get("config_path") or ""))
    if not cfg_path.is_file():
        return "", ""
    instance = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("agent", {}).get("instance_template")
    if not instance:
        return "", ""
    sentinel = "\x00APEX_TASK_SENTINEL\x00"
    rendered = Template(instance, undefined=StrictUndefined).render(task=sentinel, **(tpl.get("template_vars") or {}))
    prefix, _, suffix = rendered.partition(sentinel)
    return prefix, suffix


def iter_decoded(path: Path, tokenizer: Any, keep: Iterable[str] | None = None, columns: Iterable[str] = ()) -> Iterator[tuple[dict[str, Any], str]]:
    """(metadata, decoded text) per row of a rows file; only rows whose row_key is in `keep` when given."""
    import pyarrow.parquet as pq

    keep_set = set(keep) if keep is not None else None
    cols = ["input_ids", "row_key", *columns]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=[c for c in dict.fromkeys(cols) if c in pf.schema_arrow.names]):
        for row in batch.to_pylist():
            if keep_set is not None and row["row_key"] not in keep_set:
                continue
            ids = row.pop("input_ids")
            yield row, tokenizer.decode(ids)


def pool_screen_items(files: PoolFiles, tokenizer: Any, *, swe_upstream: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """One p1 screen item per pool row (run_screens.py merges rows of one task): task_id, kind, text (the task text the
    row's model sees: SWE issue inside the instance template, the first Terminus-2 user message, the retention prompt),
    repo / upstream_id / pr / base_commit / patch where known. Both terminal caps are read so that every rendered row is
    covered. Returns (items per pool, extraction counts)."""
    swe_manifest = json.loads(files.swe_manifest.read_text(encoding="utf-8")) if files.swe_manifest.exists() else {}
    prefix, suffix = instance_template_parts(swe_manifest)
    stats: dict[str, Any] = {"swe_template_found": bool(prefix), "swe_task_extracted": 0, "swe_task_whole_message": 0}
    items: dict[str, list[dict[str, Any]]] = {"swe_teacher": [], "terminal_teacher": [], "retention": []}
    seen_tasks: set[str] = set()
    cols = ("task_id", "repo", "episode_id", "oswt_subsource", "row_kind")
    for meta, text in iter_decoded(files.swe_rows, tokenizer, columns=cols):
        task, found = strip_instance_template(first_user_message(text), prefix, suffix)
        stats["swe_task_extracted" if found else "swe_task_whole_message"] += 1
        upstream_id = str(meta["task_id"]).split(":", 1)[1]
        up = (swe_upstream or {}).get((str(meta.get("oswt_subsource")), upstream_id), {})
        items["swe_teacher"].append({"task_id": meta["task_id"], "kind": "swe", "text": task, "repo": meta.get("repo"),
                                     "upstream_id": upstream_id, "base_commit": up.get("base_commit"), "patch": up.get("patch"),
                                     "split": "train", "stratum": {"subsource": meta.get("oswt_subsource")}})
    for cap in CAPS:
        for meta, text in iter_decoded(files.terminal_rows[cap], tokenizer, columns=("task_id", "episode_id", "source")):
            if meta["task_id"] in seen_tasks:
                continue
            seen_tasks.add(meta["task_id"])
            items["terminal_teacher"].append({"task_id": meta["task_id"], "kind": "terminal", "text": first_user_message(text),
                                              "upstream_id": str(meta["task_id"]).split(":", 1)[1], "split": "train",
                                              "stratum": {"source": meta.get("source")}})
    for meta, text in iter_decoded(files.retention_rows, tokenizer, columns=("task_id", "retention_part")):
        items["retention"].append({"task_id": meta["task_id"], "kind": "prompt", "text": first_user_message(text),
                                   "upstream_id": str(meta["task_id"]).split(":", 1)[1], "split": "train",
                                   "stratum": {"part": meta.get("retention_part")}})
    stats["items"] = {k: len(v) for k, v in items.items()}
    stats["empty_text"] = {k: sum(1 for it in v if not str(it["text"]).strip()) for k, v in items.items()}
    return items, stats


def swe_upstream_fields(keys: Iterable[tuple[str, str]], raw_root: Path = paths.RAW) -> dict[tuple[str, str], dict[str, Any]]:
    """{(subsource, instance_id): {base_commit, patch}} from the pinned SWE-rebench-V2 and Scale-SWE-Verified task rows
    (the reference patch for the exact-patch and fix-family keys; test patches are never read)."""
    import pyarrow.parquet as pq

    want = defaultdict(set)
    for sub, iid in keys:
        want[sub].add(iid)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for sub, folder, commit_col in (("swe-rebench-v2", "swe_rebench_v2", "base_commit"), ("scale-swe", "scale_swe_verified", "parent_commit")):
        if not want[sub]:
            continue
        for f in sorted((raw_root / folder / "data").glob("*.parquet")):
            for rec in pq.read_table(f, columns=["instance_id", commit_col, "patch"]).to_pylist():
                if rec["instance_id"] in want[sub]:
                    out[(sub, rec["instance_id"])] = {"base_commit": rec.get(commit_col), "patch": rec.get("patch")}
    return out


def dev_screen_items(manifests_dir: Path = paths.MANIFESTS, raw_root: Path = paths.RAW, *, qa: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The frozen dev items with their text, as split=dev screen items, so that run_screens.py checks train-vs-dev
    overlap on text (its dev manifests carry ids only): the terminal dev tasks' instruction.md, the SWE dev problem
    statements and patches, and the STEM QA items (apex_data.devset.qa, offline cache)."""
    import pyarrow.parquet as pq

    items: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    term = json.loads((manifests_dir / "dev_terminal.json").read_text(encoding="utf-8"))
    n = 0
    for it in term.get("items", []):
        p = raw_root / "tgr_env" / "environments" / str(it["upstream_id"]) / "instruction.md"
        if p.is_file():
            items.append({"task_id": it["task_id"], "kind": "terminal", "text": p.read_text(encoding="utf-8"),
                          "upstream_id": it["upstream_id"], "split": "dev"})
            n += 1
    stats["terminal"] = {"manifest": len(term.get("items", [])), "with_text": n}
    swe = json.loads((manifests_dir / "dev_swe.json").read_text(encoding="utf-8"))
    want = {str(it["upstream_id"]): it for it in swe.get("items", [])}
    found = 0
    for folder in ("swesmith_py", "swesmith_go"):
        for f in sorted((raw_root / folder / "data").glob("*.parquet")):
            for rec in pq.read_table(f, columns=["instance_id", "repo", "patch", "problem_statement"]).to_pylist():
                it = want.get(rec["instance_id"])
                if it is not None:
                    items.append({"task_id": it["task_id"], "kind": "swe", "text": rec.get("problem_statement") or "",
                                  "repo": it.get("repo"), "upstream_id": rec["instance_id"], "patch": rec.get("patch"), "split": "dev"})
                    found += 1
    stats["swe"] = {"manifest": len(want), "with_text": found}
    if qa:
        try:
            from apex_data.devset.qa import load_dev_items

            dev_qa, _ = load_dev_items(manifests_dir / "dev_qa.json")
            for it in dev_qa:
                items.append({"task_id": f"devqa:{it.dev_id}", "kind": "prompt", "text": it.text, "split": "dev"})
            stats["qa"] = {"with_text": len(dev_qa)}
        except Exception as exc:   # the QA cache is optional here; the retention pool ran D5' against it already
            stats["qa"] = {"with_text": 0, "skipped": f"{type(exc).__name__}: {exc}"[:300]}
    return items, stats
