#!/usr/bin/env python
"""The revision-5 SFT mixture (data plan 1.3, 3.4, 3.5, 5, 7 step G2): screen items over the three v5 slices, the
mixture with its validation shard, the gate report and data card, and the freeze.

Run from the repository root, in this order:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture_v5.py items
    OMP_NUM_THREADS=24 PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name v5_p1_pools ...
        (the exact command is printed by `items`)
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture_v5.py build
        (exit 3 when a screen dropped a drawn SWE instance: `build` writes the exclusions and prints the SWE redraw,
         scripts/data/build_swe_slice.py --resume <run> --exclude-task-ids data/manifests/v5/swe_exclusions.json;
         then items, p1 and build again)
    OMP_NUM_THREADS=24 PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name v5_p2_rows --pools '' \\
        --workers 24 --rows data/processed/v5/sft/train_rows.parquet --rows data/processed/v5/sft/dev_rows.parquet
        (when p2 dropped rows: build again, the drops are sticky; then p2 again until it drops nothing)
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture_v5.py report     (gate report + data card)
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture_v5.py freeze     (writes data_frozen.json)

Slices (inputs, read-only; each written by its builder and announced by a marker in logs/coordination/v5/):
  SWE        data/pools/swe_v5/{rows.parquet, manifest.json, selection.json}   (scripts/data/build_swe_slice.py)
  terminal   data/processed/v5/terminal/{rows.parquet, episodes.jsonl, tasks.jsonl, manifest.json}
             (scripts/data/build_terminal_slice_v5.py)
  retention  data/pools/retention_v5/{rows.parquet, manifest.json}: rows ranked by v5_rank so that every prefix keeps
             the part shares (scripts/data/build_retention_slice.py --compose-v5)
Also read: data/manifests/screens_v5_{p1_pools,p2_rows}.json, data/manifests/dev_*.json, the RAW SWE task rows (items:
reference patches and base commits), the Qwen3.5-9B tokenizer (items: decoding).

Rules:
  units      an SWE episode row; a terminal episode (all its per-turn rows); a retention row. Screens drop whole units:
             a p1_pools drop (task text) removes every unit of the task, a p2_rows drop (rendered row) the row's unit.
             SWE drops go back to the SWE draw as exclusions (it refills from its admitted supply); terminal drops are
             shortfalls; retention drops are replaced by extending the ranked prefix.
  mixture    SWE and terminal as drawn; retention = the shortest prefix of the ranked rows (screen drops removed) whose
             loss tokens reach RETENTION_SHARE of the total (plan 3.3: 25 %). Loss shares are reported against
             TARGET_SHARES (plan 3.5: ~64 / 11 / 25) with TOLERANCE.
  split      the task-grouped validation shard (apex_data.render.split, --val-fraction 5 %), stratified by cell (SWE
             band x language group, terminal band, retention part); tasks never straddle train and validation.
Outputs:
  data/processed/v5/sft/{train,dev}_rows.parquet         apex_train.data_format.ROW_COLUMNS (git-ignored)
  data/processed/v5/sft/{train,dev}_provenance.parquet   row_key + schema.PROVENANCE_FIELDS, aligned row by row
  data/manifests/v5/{selection,split,mixture}.json       units, drops, split arithmetic, shares, files with sha256
  data/manifests/v5/swe_exclusions.json                  SWE task ids the screens dropped (input of the SWE redraw)
  data/manifests/v5/gate_report.json, data/README_v5.md  (report) every data plan 1.3 gate; the v5 data card
  logs/coordination/v5/data_frozen.json                  (freeze) the marker with every artifact's sha256
  logs/<run_id>/                                         metadata.json, train.log, metrics.jsonl, errors.log, resources.csv
Seeded by apex_data.paths.SEED (--seed).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import efficiency, paths, schema  # noqa: E402
from apex_data.render import mixture as mix  # noqa: E402
from apex_data.render import select as sel  # noqa: E402
from apex_data.render import split as spl  # noqa: E402
from apex_data.teacher import select_v5 as swe_sel  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import ROW_COLUMNS, arrow_schema  # noqa: E402

TOLERANCE = 0.02
RETENTION_SHARE = 0.25
MIX_GATE_PP = 5.0                                      # plan 1.3: mix within 5 points of target, or the reported supply limit
TERMINAL_V6_OVERALL_MEAN = 400                         # v6 plan 2: terminal thinking per trained turn, slice mean, at most


@dataclass(frozen=True)
class Layout:
    """Where one data version reads and writes, its screen pass names and marker, its targets and band-limit rule.
    band_rule `episode` (revision 5, data plan 1.3): ceiling and mean window per episode over its trained turns, plus
    the terminal step and output caps. `slice` (v6 plan section 2; gates are for safety, not selection): every
    trained turn within its band's ceiling, and the mean thinking per trained turn checked per band over the slice
    (SWE easy <= 270, medium <= 360, hard 180-450; terminal <= 450 / 750 / 1,000 by band and <= 400 overall), terminal
    steps <= 20, no output cap."""

    version: str
    plan: str
    swe_dir: Path
    terminal_dir: Path
    retention_dir: Path
    out_dir: Path
    man_dir: Path
    card: Path
    coord: Path
    p1: str
    p2: str
    marker: str
    target_shares: dict[str, float]
    band_rule: str
    swe_redraw: bool             # screen drops of drawn SWE instances go back to the SWE draw (v5); else shortfalls (v6)


LAYOUTS = {
    "v5": Layout("v5", "revision-5 data plan 1.3, 3.4, 3.5, 5, 7 (G2)", paths.POOLS / "swe_v5",
                 paths.PROCESSED / "v5" / "terminal", paths.POOLS / "retention_v5", paths.PROCESSED / "v5" / "sft",
                 paths.MANIFESTS / "v5", paths.REPO / "data" / "README_v5.md", paths.LOGS / "coordination" / "v5",
                 "v5_p1_pools", "v5_p2_rows", "data_frozen.json", {"swe": 0.64, "terminal": 0.11, "retention": 0.25}, "episode", True),
    "v6": Layout("v6", "revision-5 data plan with the v6 per-turn safety gates", paths.POOLS / "swe_v6",
                 paths.PROCESSED / "v6" / "terminal", paths.POOLS / "retention_v5", paths.PROCESSED / "v6" / "sft",
                 paths.MANIFESTS / "v6", paths.REPO / "data" / "README_v6.md", paths.LOGS / "coordination" / "v6",
                 "v6_p1_pools", "v6_p2_rows", "data_frozen_v6.json", {"swe": 0.45, "terminal": 0.30, "retention": 0.25}, "slice", False),
    "v7": Layout("v7", "revision-5 data plan with the v7 adaptive thinking masks", paths.POOLS / "swe_v7",
                 paths.PROCESSED / "v7" / "terminal", paths.POOLS / "retention_v5", paths.PROCESSED / "v7" / "sft",
                 paths.MANIFESTS / "v7", paths.REPO / "data" / "README_v7.md", paths.LOGS / "coordination" / "v7",
                 "v7_p1_pools", "v7_p2_rows", "data_frozen_v7.json", {"swe": 0.38, "terminal": 0.34, "retention": 0.28}, "v7", False),
}
THREAD_CAP = "OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false"
PROVENANCE_INT = ("patch_files", "patch_added_lines")
PROVENANCE_BOOL = ("fits_48k",)
FILE_KINDS = ("train", "dev")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def slice_files(a: argparse.Namespace) -> dict[str, Path]:
    return {"swe_rows": a.swe_dir / "rows.parquet", "swe_manifest": a.swe_dir / "manifest.json",
            "swe_selection": a.swe_dir / "selection.json", "terminal_rows": a.terminal_dir / "rows.parquet",
            "terminal_episodes": a.terminal_dir / "episodes.jsonl", "terminal_tasks": a.terminal_dir / "tasks.jsonl",
            "terminal_manifest": a.terminal_dir / "manifest.json", "retention_rows": a.retention_dir / "rows.parquet",
            "retention_manifest": a.retention_dir / "manifest.json"}


def file_shas(files: dict[str, Path]) -> dict[str, str | None]:
    return {k: common.sha256_file(p) if p.exists() else None for k, p in files.items()}


def resources(run: common.RunDir, t0: float) -> None:
    run.write_resources([("duration_sec", round(time.time() - t0, 1), "s", "wall clock on the CPU VM"),
                         ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
                         ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                         ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only")])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


# --- items ------------------------------------------------------------------------------------------------------------
def cmd_items(a: argparse.Namespace) -> int:
    L = a.layout
    from apex_train.data_format import load_tokenizer

    run_id = a.run_id or common.new_run_id(f"data_mixture_{L.version}_items")
    out_dir = a.out_dir or (paths.WORK / run_id / "screen_items")
    files = slice_files(a)
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        shas = file_shas(files)
        missing = [k for k, v in shas.items() if v is None and k != "terminal_tasks"]
        if missing:
            raise SystemExit(f"slice files missing: {missing} (wait for the slice markers)")
        run.write_metadata(run_id=run_id, step="build_mixture_v5 items", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running",
                           data_version={"slice_files": {rel(files[k]): v for k, v in shas.items()}},
                           model_revision=common.model_revision())
        tok = load_tokenizer(paths.BASE_MODEL_DIR)
        swe_k, term_k, ret_k = f"swe_{L.version}", f"terminal_{L.version}", f"retention_{L.version}"     # screen pool names
        items: dict[str, list[dict[str, Any]]] = {swe_k: [], term_k: [], ret_k: []}
        stats: dict[str, Any] = {}
        # SWE: the issue inside the instance template, with the reference patch and base commit for the patch keys
        swe_manifest = json.loads(files["swe_manifest"].read_text(encoding="utf-8"))
        prefix, suffix = sel.instance_template_parts(swe_manifest)
        swe_meta = sel.read_meta(files["swe_rows"])
        upstream = sel.swe_upstream_fields({(str(r["oswt_subsource"]), str(r["task_id"]).split(":", 1)[1]) for r in swe_meta})
        found = Counter()
        for meta, text in sel.iter_decoded(files["swe_rows"], tok, columns=("task_id", "repo", "oswt_subsource")):
            task, ok = sel.strip_instance_template(sel.first_user_message(text), prefix, suffix)
            found[ok] += 1
            upstream_id = str(meta["task_id"]).split(":", 1)[1]
            up = upstream.get((str(meta["oswt_subsource"]), upstream_id), {})
            items[swe_k].append({"task_id": meta["task_id"], "kind": "swe", "text": task, "repo": meta.get("repo"),
                                    "upstream_id": upstream_id, "base_commit": up.get("base_commit"), "patch": up.get("patch"),
                                    "split": "train", "stratum": {"subsource": meta.get("oswt_subsource")}})
        stats["swe"] = {"template_found": bool(prefix), "task_extracted": found[True], "whole_message": found[False],
                        "reference_patches": len(upstream)}
        # terminal: the task instruction of each drawn task (the terminal builder's screen input, one item per task);
        # without it, the first Terminus-2 user message decoded from the rows (run_screens cuts the harness template)
        seen: set[str] = set()
        if files["terminal_tasks"].exists():
            for rec in read_jsonl(files["terminal_tasks"]):
                if rec["task_id"] not in seen:
                    seen.add(rec["task_id"])
                    items[term_k].append({**rec, "split": "train"})
        else:
            for meta, text in sel.iter_decoded(files["terminal_rows"], tok, columns=("task_id", "source")):
                if meta["task_id"] not in seen:
                    seen.add(meta["task_id"])
                    items[term_k].append({"task_id": meta["task_id"], "kind": "terminal", "text": sel.first_user_message(text),
                                                 "upstream_id": str(meta["task_id"]).split(":", 1)[1], "split": "train",
                                                 "stratum": {"source": meta.get("source")}})
        term_tasks = {str(r["task_id"]) for r in sel.read_meta(files["terminal_rows"])}
        if term_tasks - seen:
            raise SystemExit(f"{len(term_tasks - seen)} terminal tasks of the rows have no screen item, e.g. {sorted(term_tasks - seen)[:3]}")
        # retention: every ranked row (so that a prefix can be re-cut after drops)
        for meta, text in sel.iter_decoded(files["retention_rows"], tok, columns=("task_id", "v5_part")):
            items[ret_k].append({"task_id": meta["task_id"], "kind": "prompt", "text": sel.first_user_message(text),
                                          "upstream_id": str(meta["task_id"]).split(":", 1)[1], "split": "train",
                                          "stratum": {"part": meta.get("v5_part")}})
        if not a.no_dev:
            items["dev"], stats["dev"] = sel.dev_screen_items(paths.MANIFESTS)
        stats["items"] = {k: len(v) for k, v in items.items()}
        stats["empty_text"] = {k: sum(1 for it in v if not str(it["text"]).strip()) for k, v in items.items()}
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, recs in items.items():
            p = out_dir / f"{name}.jsonl"
            schema.write_jsonl(p, recs)
            written[name] = {"path": str(p), "records": len(recs), "sha256": common.sha256_file(p)}
        common.write_json(out_dir / "items_manifest.json", {"run_id": run_id, "created": common.utc_now(),
                                                            "slice_files": {k: v for k, v in shas.items()}, "items": written,
                                                            "stats": stats})
        run.log(f"items: {json.dumps(stats)}")
        cmd = (f"{THREAD_CAP} PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name {L.p1} --pools '' "
               f"--workers 24 " + " ".join(f"--pool {n}={w['path']}" for n, w in written.items()))
        run.log(f"next: {cmd}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), status="finished", summary={"items": stats["items"], "next_command": cmd})
        run.finalize()
    print(cmd)
    return 0


# --- units ------------------------------------------------------------------------------------------------------------
def swe_units(meta: list[dict[str, Any]]) -> list[sel.Unit]:
    return [sel.Unit(unit_id=str(r["episode_id"]), task_id=str(r["task_id"]), slice="swe",
                     cell=f"{r['band']}/{r['language_group']}", stratum=str(r["band"]), row_keys=(str(r["row_key"]),),
                     seq_tokens=int(r["n_tokens"]), loss_tokens=int(r["n_assistant_tokens"]), order=int(r["draw_order"]),
                     repo=r.get("repo")) for r in meta]


def terminal_units(meta: list[dict[str, Any]]) -> list[sel.Unit]:
    return sel.episode_units(meta, cell_key="difficulty", stratum_key="difficulty")


def retention_units(meta: list[dict[str, Any]]) -> list[sel.Unit]:
    ranked = sorted(meta, key=lambda r: int(r["v5_rank"]))
    return [sel.Unit(unit_id=str(r["row_key"]), task_id=str(r["task_id"]), slice="retention", cell=str(r["v5_part"]),
                     stratum=None, row_keys=(str(r["row_key"]),), seq_tokens=int(r["n_tokens"]),
                     loss_tokens=int(r["n_assistant_tokens"]), order=int(r["v5_rank"])) for r in ranked]


def p1_status(p1: dict[str, Any] | None, files: dict[str, Path], pass_name: str) -> dict[str, Any]:
    """Is the p1 pass current for these slice files? (via the items manifest next to its input files)"""
    if not p1:
        return {"present": False, "current": False, "reason": f"no screens_{pass_name}.json"}
    inputs = [Path(f["path"]) for inp in (p1.get("inputs") or {}).values() for f in inp.get("files", [])]
    im = next((p.parent / "items_manifest.json" for p in inputs if (p.parent / "items_manifest.json").exists()), None)
    if im is None:
        return {"present": True, "current": False, "reason": "the p1 inputs carry no items_manifest.json"}
    recorded = json.loads(im.read_text(encoding="utf-8")).get("slice_files", {})
    now = file_shas(files)
    stale = sorted(k for k in now if recorded.get(k) != now[k])
    for inp in (p1.get("inputs") or {}).values():
        for f in inp.get("files", []):
            if Path(f["path"]).exists() and common.sha256_file(Path(f["path"])) != f["sha256"]:
                stale.append(f["path"])
    return {"present": True, "current": not stale, "stale": stale, "items_manifest": str(im), "run_id": p1.get("run_id"),
            "reason": "slice files changed since the items were written" if stale else "ok"}


def write_rows(units: list[sel.Unit], sources: dict[str, Path], path: Path) -> tuple[list[str], dict[str, int]]:
    """Rows of `units` in unit order (each unit's rows in their file order), ROW_COLUMNS only, written atomically."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    order = [k for u in units for k in u.row_keys]
    parts, where = [], {}
    for s, f in sources.items():
        keys = [k for u in units if u.slice == s for k in u.row_keys]
        if not keys:
            continue
        t = pq.read_table(f, columns=[*ROW_COLUMNS, "row_key"])
        t = t.filter(pc.is_in(t["row_key"], value_set=pa.array(keys)))
        base = sum(len(p) for p in parts)
        for i, k in enumerate(t["row_key"].to_pylist()):
            if k in where:
                raise sel.SelectionError(f"row_key {k} appears twice in {f}")
            where[k] = base + i
        parts.append(t.select(list(ROW_COLUMNS)).cast(arrow_schema()))
    missing = set(order) - set(where)
    if missing:
        raise sel.SelectionError(f"{len(missing)} rows not found in the slice files, e.g. {sorted(missing)[:3]}")
    table = pa.concat_tables(parts).take(pa.array([where[k] for k in order], type=pa.int64()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return order, {"rows": table.num_rows, "sequence_tokens": int(pc.sum(table["n_tokens"]).as_py() or 0),
                   "loss_tokens": int(pc.sum(table["n_assistant_tokens"]).as_py() or 0)}


def write_provenance(order: list[str], rows: dict[str, dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [("row_key", pa.string())] + [(f, pa.int64() if f in PROVENANCE_INT else pa.bool_() if f in PROVENANCE_BOOL
                                            else pa.string()) for f in schema.PROVENANCE_FIELDS]
    recs = []
    for k in order:
        r = rows[k]
        rec = {"row_key": k, **{f: r.get(f) for f in schema.PROVENANCE_FIELDS}}
        schema.validate_provenance(rec)
        recs.append(rec)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(recs, schema=pa.schema(fields)), tmp)
    os.replace(tmp, path)


def swe_draw_flags(swe_manifest: dict[str, Any]) -> str:
    """The build_swe_slice.py draw flags that reproduce the SWE slice's draw settings (its manifest records them)."""
    import shlex

    ds = (swe_manifest.get("rules") or {}).get("draw_settings") or {}
    flags = []
    if ds.get("keep_all_bands"):
        flags.append(f"--keep-all-bands {','.join(ds['keep_all_bands'])}")
        flags += [f"--band-share {b}={v}" for b, v in ds["band_shares"].items() if b not in ds["keep_all_bands"]]
    elif ds.get("target_loss_tokens"):
        flags.append(f"--target-loss-tokens {ds['target_loss_tokens']}")
    flags += [f"--repo-cap {g}={n}" for g, n in (ds.get("repo_caps") or {}).items()]
    if ds.get("second_attempt_bands"):
        flags.append(f"--second-attempt-bands {','.join(ds['second_attempt_bands'])} --max-second-share {ds['max_second_share']}")
    if ds.get("note"):
        flags.append(f"--note {shlex.quote(ds['note'])}")
    return " ".join(flags)


def proportional_swe_cap(swe: list[sel.Unit], terminal_loss: int, max_share: float,
                         retention_share: float = RETENTION_SHARE) -> tuple[list[sel.Unit], dict[str, Any]]:
    """The D1 rule: cap SWE at `max_share` of all loss tokens (retention taking `retention_share` of the total) by keeping
    the same fraction of each band's loss tokens, dropping episodes from the end of each band's draw order (a second
    attempt is drawn after its first, so it goes first). Bands keep their shares; languages and task types roughly
    theirs. Returns (kept units in their order, report)."""
    bound = max_share / (1 - retention_share)             # SWE <= bound x (SWE + terminal)
    total = sum(u.loss_tokens for u in swe)
    allowed = bound / (1 - bound) * terminal_loss
    frac = min(1.0, allowed / total) if total else 1.0
    gone: set[str] = set()
    per_band: dict[str, dict[str, int]] = {}
    for band in ("hard", "medium", "easy"):
        units = sorted((u for u in swe if u.stratum == band), key=lambda u: -u.order)
        loss = sum(u.loss_tokens for u in units)
        keep_max = frac * loss
        removed = 0
        for u in units:                                   # last drawn first
            if loss - removed <= keep_max:
                break
            gone.add(u.unit_id)
            removed += u.loss_tokens
        per_band[band] = {"loss_before": loss, "loss_after": loss - removed, "episodes_removed": sum(1 for u in units if u.unit_id in gone)}
    kept = [u for u in swe if u.unit_id not in gone]
    after = sum(u.loss_tokens for u in kept)
    return kept, {"max_share": max_share, "rule": "proportional by band: each band keeps the same fraction of its loss tokens, "
                  "episodes dropped from the end of the band's draw order (the D1 rule)",
                  "kept_fraction": round(frac, 4), "allowed_loss_tokens": round(allowed), "episodes_removed": len(gone),
                  "loss_tokens_removed": total - after, "reached": after <= allowed + 1, "per_band": per_band, "units": sorted(gone)}


def targeted_swe_trim(swe: list[sel.Unit], meta: Mapping[str, Mapping[str, Any]], keep_fraction: float,
                      lang_tolerance_pp: float = 1.0) -> tuple[list[sel.Unit], dict[str, Any]]:
    """The v6r2 trim: SWE to `keep_fraction` of its loss tokens, deterministic.
    (1) every band (hard, medium, easy) keeps the same fraction of its loss tokens; (2) within a band, the episodes with
    an unverified submit go first (`submit_masked`: no test, program or inline-code run in the 3 turns before the
    submit), last drawn first; (3) then episodes from the end of the band's draw order, skipping one whose removal would
    lower its language group's episode share by more than `lang_tolerance_pp` points below its share before the trim.
    Bands are trimmed hard, medium, easy; the language counts are global. `meta`: episode_id -> its SWE row."""
    lang = {u.unit_id: str(meta[u.unit_id]["language_group"]) for u in swe}
    unverified = {u.unit_id for u in swe if meta[u.unit_id].get("submit_masked")}
    counts = Counter(lang.values())
    n = len(swe)
    share0 = {g: c / n for g, c in counts.items()}
    gone: set[str] = set()
    per_band: dict[str, Any] = {}
    skipped: Counter = Counter()
    for band in ("hard", "medium", "easy"):
        units = sorted((u for u in swe if u.stratum == band), key=lambda u: -u.order)    # last drawn first
        loss = sum(u.loss_tokens for u in units)
        target, cur, by_step = keep_fraction * loss, loss, Counter()
        for step in ("unverified_submit", "draw_order"):
            progress = True
            while progress and cur > target:           # passes from the end: a protected episode may be freed by later drops
                progress = False
                for u in units:
                    if cur <= target:
                        break
                    if u.unit_id in gone or (step == "unverified_submit" and u.unit_id not in unverified):
                        continue
                    g = lang[u.unit_id]
                    if step == "draw_order" and share0[g] - (counts[g] - 1) / (n - 1) > lang_tolerance_pp / 100:
                        skipped[g] += 1
                        continue
                    gone.add(u.unit_id)
                    cur -= u.loss_tokens
                    counts[g] -= 1
                    n -= 1
                    by_step[step] += 1
                    progress = True
        per_band[band] = {"loss_before": loss, "target": round(target), "loss_after": cur, "removed_unverified_submit":
                          by_step["unverified_submit"], "removed_draw_order": by_step["draw_order"], "reached": cur <= target}
    kept = [u for u in swe if u.unit_id not in gone]
    return kept, {"keep_fraction": keep_fraction, "lang_tolerance_pp": lang_tolerance_pp,
                  "rule": "v6r2 targeted trim: per band the same fraction; unverified-submit episodes first, then the end "
                          "of the band's draw order with the language shares protected",
                  "episodes_removed": len(gone), "loss_tokens_removed": sum(u.loss_tokens for u in swe) - sum(u.loss_tokens for u in kept),
                  "unverified_submit_episodes": len(unverified), "language_skips": dict(skipped), "per_band": per_band,
                  "units": sorted(gone)}


def retention_by_part(units: list[sel.Unit], reference: list[sel.Unit], target: float,
                      keep_whole: Sequence[str]) -> tuple[list[sel.Unit], dict[str, Any]]:
    """The v6r2 retention trim: from the `reference` rows (the current retention selection, rank order), the parts in
    `keep_whole` (hard math) are kept entirely; every other part gets the rest of `target` in proportion to its loss
    share in the reference, as the shortest prefix of its rank order reaching it (so trimmed from the end). `units` is
    unused beyond the reference's membership check (the reference is a prefix of it)."""
    if not {u.unit_id for u in reference} <= {u.unit_id for u in units}:
        raise sel.SelectionError("the retention reference is not part of the candidate rows")
    by_part: dict[str, list[sel.Unit]] = defaultdict(list)
    for u in sorted(reference, key=lambda u: u.order):
        by_part[u.cell].append(u)
    ref_loss = {p: sum(u.loss_tokens for u in us) for p, us in by_part.items()}
    whole = [p for p in by_part if p in keep_whole]
    rest = target - sum(ref_loss[p] for p in whole)
    others = {p: v for p, v in ref_loss.items() if p not in keep_whole}
    out: list[sel.Unit] = []
    parts: dict[str, Any] = {}
    for p, us in sorted(by_part.items()):
        want = ref_loss[p] if p in keep_whole else max(0.0, rest) * others[p] / sum(others.values())
        took = us if p in keep_whole else retention_prefix(us, want)
        out += took
        parts[p] = {"reference_loss": ref_loss[p], "target": round(want), "loss": sum(u.loss_tokens for u in took),
                    "rows": len(took), "reference_rows": len(us)}
    total = sum(v["loss"] for v in parts.values())
    for v in parts.values():
        v["share"] = round(v["loss"] / total, 4) if total else 0.0
    ref_total = sum(ref_loss.values())
    for p, v in parts.items():
        v["reference_share"] = round(ref_loss[p] / ref_total, 4) if ref_total else 0.0
    return sorted(out, key=lambda u: u.order), {"target_loss_tokens": round(target), "loss_tokens": total, "rows": len(out),
                                                "keep_whole": list(keep_whole), "parts": parts}


def retention_prefix(units: list[sel.Unit], target: float) -> list[sel.Unit]:
    out, total = [], 0
    for u in units:
        if total >= target:
            break
        out.append(u)
        total += u.loss_tokens
    return out


# --- build ------------------------------------------------------------------------------------------------------------
def cmd_build(a: argparse.Namespace) -> int:
    L = a.layout
    run_id = a.run_id or common.new_run_id(f"data_mixture_{L.version}")
    files = slice_files(a)
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        shas = file_shas(files)
        run.write_metadata(run_id=run_id, step="build_mixture_v5 build", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running",
                           config={k: str(v) for k, v in vars(a).items()},
                           data_version={"slice_files": {rel(files[k]): v for k, v in shas.items()}},
                           model_revision=common.model_revision())
        p1 = common.read_json(paths.MANIFESTS / f"screens_{L.p1}.json")
        p1_info = p1_status(p1, files, L.p1)
        run.log(f"{L.p1}: {json.dumps({k: v for k, v in p1_info.items() if k != 'stale'})}")
        if not p1_info["current"]:
            run.error(f"{L.p1} not current: {p1_info}")
            raise SystemExit(f"{L.p1} missing or stale ({p1_info['reason']}): run `items` and the p1 screens first")
        drops_p1 = sel.screen_drops(p1, L.p1)
        previous = common.read_json(L.man_dir / "selection.json") or {}
        swe_meta = sel.read_meta(files["swe_rows"])
        term_meta = sel.read_meta(files["terminal_rows"])
        ret_meta = sel.read_meta(files["retention_rows"])
        rows: dict[str, dict[str, Any]] = {}
        for r in (*swe_meta, *term_meta, *ret_meta):
            if str(r["row_key"]) in rows:
                raise sel.SelectionError(f"row_key {r['row_key']} appears twice across the slices")
            rows[str(r["row_key"])] = r
        units = {"swe": swe_units(swe_meta), "terminal": terminal_units(term_meta), "retention": retention_units(ret_meta)}
        # p2 drops: the current manifest's (rows -> units), plus every earlier build's (sticky)
        p2 = common.read_json(paths.MANIFESTS / f"screens_{L.p2}.json")
        samples = {str(r["sample_id"]) for r in rows.values()}
        p2_ids = sel.screen_drops(p2, L.p2)
        p2_stale = sorted(i for i in p2_ids if i.split(":", 1)[1] not in samples)   # rows a rebuilt slice no longer holds
        p2_now = sel.row_drops_to_units({i: r for i, r in p2_ids.items() if i not in p2_stale}, rows.values())
        if p2_stale:
            run.log(f"{len(p2_stale)} {L.p2} drops name rows the current slices no longer hold (rebuilt since): {p2_stale[:5]}")
        sticky = (previous.get("screens") or {}).get(L.p2, {}).get("dropped_units", {})
        drops_p2 = {u: sorted(set(sticky.get(u, [])) | set(p2_now.get(u, []))) for u in set(sticky) | set(p2_now)}
        dropped: dict[str, list[sel.Unit]] = {}
        kept: dict[str, list[sel.Unit]] = {}
        for s, us in units.items():
            kept[s] = [u for u in us if u.task_id not in drops_p1 and u.unit_id not in drops_p2]
            dropped[s] = [u for u in us if u.task_id in drops_p1 or u.unit_id in drops_p2]
        run.log("screen drops: " + json.dumps({s: len(v) for s, v in dropped.items()}) + f" (p1 ids {len(drops_p1)}, p2 units {len(drops_p2)})")
        if dropped["swe"] and L.swe_redraw:
            excl_path = L.man_dir / "swe_exclusions.json"
            old = common.read_json(excl_path) or {}
            by_task = dict(old.get("by_task", {}))
            for u in dropped["swe"]:
                by_task[u.task_id] = sorted(set(drops_p1.get(u.task_id, [])) | set(drops_p2.get(u.unit_id, [])))
            L.man_dir.mkdir(parents=True, exist_ok=True)
            common.write_json(excl_path, {"updated": common.utc_now(), "task_ids": sorted(by_task), "by_task": by_task})
            swe_man = json.loads(files["swe_manifest"].read_text(encoding="utf-8"))
            cmd = (f"PYTHONPATH=src .venv/train/bin/python scripts/data/build_swe_slice.py --resume {swe_man['run_id']} "
                   f"--exclude-task-ids {rel(excl_path)} --out-dir {rel(a.swe_dir)} {swe_draw_flags(swe_man)}")
            run.log(f"{len(dropped['swe'])} drawn SWE instances dropped by the screens -> {excl_path}; redraw: {cmd}")
            run.write_metadata(end_time=common.utc_now(), status="swe_redraw_needed", next_command=cmd)
            run.finalize()
            print(cmd)
            return 3
        # the current retention selection at the plan's 25 %, before any SWE trim: the reference of a per-part trim
        reference = retention_prefix(kept["retention"], RETENTION_SHARE / (1 - RETENTION_SHARE)
                                     * sum(u.loss_tokens for s in ("swe", "terminal") for u in kept[s]))
        swe_trim = None
        if a.swe_trim_fraction is not None:
            kept["swe"], swe_trim = targeted_swe_trim(kept["swe"], {str(r["episode_id"]): r for r in swe_meta}, a.swe_trim_fraction,
                                                      a.swe_lang_protect_pp)
            run.log(f"SWE targeted trim to {a.swe_trim_fraction}: {swe_trim['episodes_removed']} episodes removed "
                    f"({swe_trim['loss_tokens_removed']:,} loss tokens); per band {json.dumps({b: v['reached'] for b, v in swe_trim['per_band'].items()})}; "
                    f"language skips {swe_trim['language_skips']}")
        swe_cap = None
        if a.swe_max_share is not None:
            kept["swe"], swe_cap = proportional_swe_cap(kept["swe"], sum(u.loss_tokens for u in kept["terminal"]), a.swe_max_share,
                                                        a.retention_share)
            run.log(f"SWE cap {a.swe_max_share}: {swe_cap['episodes_removed']} episodes removed ({swe_cap['loss_tokens_removed']:,} "
                    f"loss tokens), kept fraction {swe_cap['kept_fraction']}; reached {swe_cap['reached']}")
        loss_fixed = sum(u.loss_tokens for s in ("swe", "terminal") for u in kept[s])
        ret_target = a.retention_share / (1 - a.retention_share) * loss_fixed
        ret_trim = None
        if a.retention_rows_from:
            want = set((common.read_json(a.retention_rows_from) or {}).get("units", {}).get("retention", []))
            ret_units = [u for u in kept["retention"] if u.unit_id in want]
            ret_trim = {"rows_from": rel(a.retention_rows_from), "rows_wanted": len(want), "rows": len(ret_units),
                        "missing_or_screened_out": len(want) - len(ret_units), "loss_tokens": sum(u.loss_tokens for u in ret_units),
                        "rule": "the retention rows of an earlier selection, unchanged (v7: v6r2's)"}
            run.log(f"retention rows from {a.retention_rows_from}: {len(ret_units)} of {len(want)}, {ret_trim['loss_tokens']:,} loss tokens")
        elif a.retention_keep_whole:
            ret_units, ret_trim = retention_by_part(kept["retention"], reference, ret_target, a.retention_keep_whole)
            run.log(f"retention by part to {a.retention_share}: {ret_trim['rows']} rows, {ret_trim['loss_tokens']:,} loss tokens; "
                    + json.dumps({p: v["share"] for p, v in ret_trim["parts"].items()}))
        else:
            ret_units = retention_prefix(kept["retention"], ret_target)
        if sum(u.loss_tokens for u in ret_units) < ret_target:
            run.error(f"retention supply {sum(u.loss_tokens for u in ret_units)} below the target {ret_target:.0f}")
        all_units = [*kept["swe"], *kept["terminal"], *ret_units]
        acct = mix.account(all_units, rows, targets=L.target_shares, tolerance=TOLERANCE)
        run.log(f"mixture: rows {acct['totals']['rows']}, seq {acct['totals']['sequence_tokens']:,}, loss {acct['totals']['loss_tokens']:,}; "
                f"loss shares {acct['shares_by_loss_tokens']} (within +-{100 * TOLERANCE:.0f} pp: {acct['within_tolerance']}); "
                f"retention prefix {len(ret_units)} rows for a target of {ret_target:,.0f}")
        val = spl.task_grouped_draw(all_units, a.val_fraction, seed=a.seed, label="validation")
        straddle = spl.straddling_tasks(val.rest, val.drawn)
        if straddle:
            raise sel.SelectionError(f"tasks straddle train and validation: {straddle[:5]}")
        order = lambda us, lab: sorted(us, key=lambda u: (sel.rank(a.seed, lab, u.unit_id), u.unit_id))  # noqa: E731
        groups = {"train": order(val.rest, "train_order"), "dev": order(val.drawn, "dev_order")}
        sources = {"swe": files["swe_rows"], "terminal": files["terminal_rows"], "retention": files["retention_rows"]}
        a.out_dir.mkdir(parents=True, exist_ok=True)
        out = {f"{k}_{t}": a.out_dir / f"{k}_{t}.parquet" for k in FILE_KINDS for t in ("rows", "provenance")}
        orders, stats = {}, {}
        for kind, us in groups.items():
            orders[kind], stats[kind] = write_rows(us, sources, out[f"{kind}_rows"])
            write_provenance(orders[kind], rows, out[f"{kind}_provenance"])
            run.log(f"{kind}: {stats[kind]} -> {out[f'{kind}_rows']}")
        file_block = {}
        for k, p in out.items():
            kind = k.split("_")[0]
            file_block[k] = {"path": rel(p), "sha256": common.sha256_file(p), "bytes": p.stat().st_size, "rows": stats[kind]["rows"],
                             **({"sequence_tokens": stats[kind]["sequence_tokens"], "loss_tokens": stats[kind]["loss_tokens"]}
                                if k.endswith("_rows") else {"columns": ["row_key", *schema.PROVENANCE_FIELDS], "aligned_with": f"{kind}_rows"})}
        p1_block = {**p1_info, "manifest": rel(paths.MANIFESTS / f"screens_{L.p1}.json"),
                    "sha256": common.sha256_file(paths.MANIFESTS / f"screens_{L.p1}.json"), "dropped_ids": len(drops_p1)}
        p2_block = {"manifest": rel(paths.MANIFESTS / f"screens_{L.p2}.json") if p2 else None, "run_id": (p2 or {}).get("run_id"),
                    "current_dropped_units": len(p2_now), "stale_row_ids": p2_stale, "dropped_units": drops_p2}
        ret_parts = Counter()
        for u in ret_units:
            ret_parts[u.cell] += u.loss_tokens
        ret_total = sum(ret_parts.values())
        L.man_dir.mkdir(parents=True, exist_ok=True)
        selection = {"run_id": run_id, "created": common.utc_now(), "plan": L.plan, "seed": a.seed, "slice_files": shas,
                     "screens": {L.p1: p1_block, L.p2: p2_block},
                     "dropped": {s: [{"unit_id": u.unit_id, "task_id": u.task_id,
                                      "rules": sorted(set(drops_p1.get(u.task_id, [])) | set(drops_p2.get(u.unit_id, [])))}
                                     for u in us] for s, us in dropped.items()},
                     "swe_cap": swe_cap, "swe_trim": swe_trim, "retention_trim": ret_trim,
                     "retention_prefix": {"target_loss_tokens": round(ret_target), "rows": len(ret_units), "loss_tokens": ret_total,
                                          "last_rank": ret_units[-1].order if ret_units else None,
                                          "part_shares": {k: round(v / ret_total, 4) for k, v in sorted(ret_parts.items())} if ret_total else {}},
                     "units": {s: [u.unit_id for u in us] for s, us in (("swe", kept["swe"]), ("terminal", kept["terminal"]),
                                                                        ("retention", ret_units))}}
        common.write_json(L.man_dir / "selection.json", selection)
        split_m = {"run_id": run_id, "created": common.utc_now(), "plan": L.plan, "seed": a.seed,
                   "validation": {**val.report, "tasks": sorted({u.task_id for u in val.drawn}), "row_keys": orders["dev"]},
                   "invariants": {"straddling_tasks": straddle, "unit": "an SWE row, a terminal episode with all its rows, a retention row"}}
        common.write_json(L.man_dir / "split.json", split_m)
        acct_parts = {k: mix.account(us, rows, targets=L.target_shares, tolerance=TOLERANCE, provenance=(k == "all"))
                      for k, us in (("all", all_units), ("train", val.rest), ("dev", val.drawn))}
        mixture = {"status": "CANDIDATE", "frozen": False, "run_id": run_id, "created": common.utc_now(), "plan": L.plan,
                   "seed": a.seed, "code": common.git_info(), "max_length": 65_536,
                   "slices": {"files": {rel(files[k]): v for k, v in shas.items()},
                              "markers": {m.name: common.read_json(m) for m in sorted(L.coord.glob("*slice*.ready"))}},
                   "screens": {L.p1: {k: v for k, v in p1_block.items() if k != "stale"},
                               L.p2: {k: v for k, v in p2_block.items() if k != "dropped_units"}},
                   "files": file_block, "mixture": acct_parts["all"], "train": acct_parts["train"], "dev": acct_parts["dev"],
                   "targets": {"loss_shares": L.target_shares, "tolerance_pp": 100 * TOLERANCE, "retention_share": a.retention_share},
                   "retention_prefix": selection["retention_prefix"],
                   "drops": {s: len(us) for s, us in dropped.items()}, "swe_cap": swe_cap, "swe_trim": swe_trim,
                   "retention_trim": ret_trim, "revision": a.revision,
                   "sources": mix.source_table(all_units, rows), "normalisation": mix.NORMALISATION_NOTE,
                   "split": {"manifest": rel(L.man_dir / "split.json"), "validation_rows": len(orders["dev"]),
                             "validation_tasks": val.report["tasks_drawn"], "train_rows": len(orders["train"])},
                   "selection": rel(L.man_dir / "selection.json"),
                   "freeze": {"command": f"PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture_v5.py freeze --version {L.version}",
                              "requires": f"{L.p1} current, {L.p2} over exactly these train/dev files with zero drops, the gate report passing"}}
        common.write_json(L.man_dir / "mixture.json", mixture)
        cmd = (f"{THREAD_CAP} PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name {L.p2} --pools '' "
               f"--workers 24 --rows {rel(out['train_rows'])} --rows {rel(out['dev_rows'])}")
        run.log(f"wrote {L.man_dir / 'mixture.json'} (CANDIDATE); next: {cmd}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), status="finished", next_command=cmd,
                           mixture={"shares": acct["shares_by_loss_tokens"], "totals": acct["totals"],
                                    "files": {k: v["sha256"] for k, v in file_block.items()}})
        run.finalize()
    print(cmd)
    return 0


# --- report: gates and the data card -----------------------------------------------------------------------------------
def _pp(x: float) -> float:
    return round(100 * x, 2)


THINK_END, IM_END = 248069, 248046                 # Qwen3.5-9B `</think>` and `<|im_end|>` token ids


def trained_thinking(path: Path, keep: set[str]) -> dict[str, list[int]]:
    """{episode_id: reasoning tokens per trained turn} from the rows' token ids: in every trained span (a run of the
    assistant mask, ended by <|im_end|>), the position of </think> (the span starts after `<think>\n`). -1 marks a
    trained turn without a closed reasoning block. This is the data plan 1.3 unit (per trained turn), measured on
    exactly what the trainer sees."""
    import pyarrow.parquet as pq

    out: dict[str, list[int]] = defaultdict(list)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=["input_ids", "assistant_masks", "episode_id"]):
        for r in batch.to_pylist():
            if str(r["episode_id"]) not in keep:
                continue
            cur: list[int] = []
            for x, m in zip(r["input_ids"], r["assistant_masks"]):
                if m:
                    cur.append(x)
                if cur and ((m and x == IM_END) or not m):
                    out[str(r["episode_id"])].append(cur.index(THINK_END) if THINK_END in cur else -1)
                    cur = []
            if cur:
                out[str(r["episode_id"])].append(cur.index(THINK_END) if THINK_END in cur else -1)
    return dict(out)


TERMINAL_V6_STEPS = 20                             # v6 plan 2: terminal admission keeps steps <= 20
TERMINAL_V7_MAX = 4000                             # v7 terminal: thinking per trained turn, at most


def ceiling_check(thinking: list[int], lim: Any) -> list[str]:
    """v6 safety limit: closed reasoning on every trained turn, each within the band's turn ceiling."""
    if not thinking or min(thinking) < 0:
        return ["unclosed_or_missing_reasoning"]
    return ["turn_ceiling"] if max(thinking) > lim.turn_ceiling else []


def limit_check(thinking: list[int], lim: Any) -> list[str]:
    """Band limits on the trained turns: closed reasoning, the turn ceiling, the window for the mean."""
    if not thinking or min(thinking) < 0:
        return ["unclosed_or_missing_reasoning"]
    out = []
    if max(thinking) > lim.turn_ceiling:
        out.append("turn_ceiling")
    mean = sum(thinking) / len(thinking)
    if not lim.mean_min <= mean <= lim.mean_max:
        out.append("mean_window")
    return out


def v7_agent_gates(files: dict[str, Path], swe: list[dict[str, Any]], term_eps: list[dict[str, Any]], term_rows: list[dict[str, Any]],
                   swe_man: dict[str, Any], term_man: dict[str, Any], ret_man: dict[str, Any], gate: Any) -> None:
    """The v7 agent-turn gates (no band-mean windows): from the rows' token ids, every trained turn has closed reasoning
    and ends in a parsed action (SWE: a qwen3_coder tool call with a command; terminal: Terminus-2 JSON with keystrokes
    or task_complete); SWE turns at most 8K, routine ones (not a decision point, by the row's turns_json) at most 1,000,
    no submit trained after 3 turns without a run, and turns above 2K at most 10 % of the SWE loss tokens; terminal
    turns at most TERMINAL_V7_MAX (for the half DeepSeek / half Kimi slice this per-turn guard replaces the terminal
    10 % cap; its > 2K share is reported); verified outcomes; the builders' checks."""
    from transformers import AutoTokenizer

    from apex_data.render.filters import qwen3_parse_calls, terminus_parse
    from apex_data.render import swe_masks as sm

    tok = AutoTokenizer.from_pretrained(str(paths.BASE_MODEL_DIR), local_files_only=True)

    def spans(path: Path, keep: set[str]) -> dict[str, list[tuple[int, str, int]]]:
        """{episode_id: [(thinking, text after </think>, trained tokens) per trained span]} from the rows' token ids."""
        import pyarrow.parquet as pq

        out: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["input_ids", "assistant_masks", "episode_id"]):
            for r in batch.to_pylist():
                if str(r["episode_id"]) not in keep:
                    continue
                for a0, b0 in sm.trained_spans(r["assistant_masks"]):
                    ids = r["input_ids"][a0:b0]
                    th = ids.index(THINK_END) if THINK_END in ids else -1
                    out[str(r["episode_id"])].append((th, tok.decode(ids[th + 1:]) if th >= 0 else "", b0 - a0))
        return dict(out)

    swe_sp = spans(files["swe_rows"], {str(r["episode_id"]) for r in swe})
    term_sp = spans(files["terminal_rows"], {str(e["episode_id"]) for e in term_eps})
    viol: dict[str, list[str]] = defaultdict(list)
    swe_turns, routine = [], []
    for r in swe:
        ep = str(r["episode_id"])
        flags = [t for t in json.loads(r["turns_json"]) if t["trained"]]
        got = swe_sp.get(ep, [])
        if len(flags) != len(got) or any(f["thinking"] != th for f, (th, _, _) in zip(flags, got)):
            viol[ep].append("turn_flags_mismatch")
            continue
        for f, (th, text, _) in zip(flags, got):
            swe_turns.append(th)
            if not f["decision"]:
                routine.append(th)
            if th < 0 or th > sm.DECISION_MAX:
                viol[ep].append("thinking_over_8k_or_open")
            if not f["decision"] and th > sm.ROUTINE_MAX:
                viol[ep].append("routine_over_1000")
            calls = qwen3_parse_calls(text)
            if not calls or not any(str(c["arguments"].get("command", "")).strip() for c in calls):
                viol[ep].append("no_tool_call")
            if f["kind"] == "submit" and f.get("reason") != "submit_after_run":
                viol[ep].append("unverified_submit_trained")
    term_turns = []
    for e in term_eps:
        ep = str(e["episode_id"])
        for th, text, _ in term_sp.get(ep, []):
            term_turns.append(th)
            if th < 0 or th > TERMINAL_V7_MAX:
                viol[ep].append("thinking_over_4k_or_open")
            parsed = terminus_parse(text.removesuffix("<|im_end|>"))
            if parsed.error or (not parsed.commands and not parsed.task_complete):
                viol[ep].append("no_action")
    def big_share(sp: dict[str, list[tuple[int, str, int]]]) -> tuple[int, int]:
        return sum(n for v in sp.values() for th, _, n in v if th > 2048), sum(n for v in sp.values() for _, _, n in v)
    (swe_big, swe_loss), (term_big, term_loss) = big_share(swe_sp), big_share(term_sp)
    gate("thinking_rule", not viol,
         rule="v7: every trained turn has closed reasoning and ends in a parsed action; SWE <= 8K, routine turns (not a "
              "decision point) <= 1,000; terminal <= 4,000; thinking = the position of </think> in each "
              "trained span of the rows; no band-mean windows",
         swe_trained_turns=len(swe_turns), terminal_trained_turns=len(term_turns), violations=dict(list(viol.items())[:20]),
         thinking_per_trained_turn={"swe": per_turn_stats(swe_turns), "swe_routine": per_turn_stats(routine),
                                    "terminal": per_turn_stats(term_turns)})
    gate("tail_cap", swe_big <= 0.10 * swe_loss + 1,
         rule="SWE trained turns above 2K thinking: at most 10 % of the SWE loss tokens (terminal: the per-turn 4,000 guard "
              "instead; its share is reported)",
         swe_share=round(swe_big / swe_loss, 4) if swe_loss else 0.0,
         terminal_share_reported=round(term_big / term_loss, 4) if term_loss else 0.0)
    gate("verified_outcomes", all(r["label_provenance"] == "verifier" for r in swe) and all(
        r["label_provenance"] == "verifier" for r in term_rows), rule="SWE resolved == 1, terminal reward == 1")
    swe_checks, term_checks = swe_man.get("checks") or {}, term_man.get("checks") or {}
    ret_checks = ret_man.get("checks")
    ret_ok = all(c.get("pass") for c in ret_checks) if isinstance(ret_checks, list) else bool((ret_checks or {}).get("pass"))
    gate("builder_checks", bool(swe_checks.get("pass")) and bool(term_checks.get("all_pass", term_checks.get("pass", False))) and ret_ok,
         rule="each slice builder's own checks pass (SWE: the v7 rule on every row, the tail cap; terminal and retention: theirs)",
         swe=swe_checks.get("pass"), terminal=term_checks.get("all_pass", term_checks.get("pass")), retention=ret_ok)


def gate_report(a: argparse.Namespace) -> dict[str, Any]:
    """Every data plan 1.3 gate on the built mixture, from the rows, the slice manifests and the screens."""
    L = a.layout
    files = slice_files(a)
    m = common.read_json(L.man_dir / "mixture.json")
    selection = common.read_json(L.man_dir / "selection.json")
    if not m or not selection:
        raise SystemExit("no v5 mixture: run `build` first")
    in_mix = {k for s in ("swe", "terminal", "retention") for k in selection["units"][s]}
    swe = [r for r in sel.read_meta(files["swe_rows"]) if str(r["episode_id"]) in in_mix]
    term_eps = [e for e in read_jsonl(files["terminal_episodes"]) if str(e.get("episode_id")) in in_mix]
    term_rows = [r for r in sel.read_meta(files["terminal_rows"]) if str(r["episode_id"]) in in_mix]
    swe_man = common.read_json(files["swe_manifest"]) or {}
    term_man = common.read_json(files["terminal_manifest"]) or {}
    ret_man = common.read_json(files["retention_manifest"]) or {}
    p2 = common.read_json(paths.MANIFESTS / f"screens_{L.p2}.json") or {}
    gates: dict[str, Any] = {}

    def gate(name: str, ok: bool, **detail: Any) -> None:
        gates[name] = {"pass": bool(ok), **detail}

    if L.band_rule == "v7":
        v7_agent_gates(files, swe, term_eps, term_rows, swe_man, term_man, ret_man, gate)
    else:
        # band limits: thinking per trained turn from the token ids (ceiling, mean window); steps, the terminal output per
        # trajectory, the row cap and the 32K loss cap per row from the rows and the builders' profiles
        swe_think = trained_thinking(files["swe_rows"], {str(r["episode_id"]) for r in swe})
        term_think = trained_thinking(files["terminal_rows"], {str(e["episode_id"]) for e in term_eps})
        per_episode = L.band_rule == "episode"
        swe_viol = {}
        for r in swe:
            lim = efficiency.SWE_LIMITS[r["band"]]
            why = (limit_check if per_episode else ceiling_check)(swe_think.get(str(r["episode_id"]), []), lim)
            why += ["steps"] if r["steps"] > lim.steps_max else []
            why += ["row"] if r["n_tokens"] > lim.row_max else []
            why += ["loss_cap"] if r["n_assistant_tokens"] > swe_sel.LOSS_CAP_PER_ROW else []
            if why:
                swe_viol[r["row_key"]] = why
        term_viol = {}
        for e in term_eps:
            lim = efficiency.TERMINAL_LIMITS[e["band"]]
            prof = e.get("profile") or {}
            why = (limit_check if per_episode else ceiling_check)(term_think.get(str(e["episode_id"]), []), lim)
            why += ["steps"] if prof.get("steps", 10 ** 9) > (lim.steps_max if per_episode else TERMINAL_V6_STEPS) else []
            why += ["output"] if per_episode and prof.get("output_tokens", 10 ** 9) > lim.output_max else []
            if why:
                term_viol[e["episode_id"]] = why
        term_row_viol = [r["row_key"] for r in term_rows if r["n_tokens"] > efficiency.TERMINAL_LIMITS[r["difficulty"]].row_max
                         or r["n_assistant_tokens"] > swe_sel.LOSS_CAP_PER_ROW]
        swe_turns = [x for v in swe_think.values() for x in v]
        term_turns = [x for v in term_think.values() for x in v]
        slice_means = {}
        if not per_episode:          # v6: the mean per band over the slice's trained turns
            for fam, think, band_of, limits in (("swe", swe_think, {str(r["episode_id"]): r["band"] for r in swe}, efficiency.SWE_LIMITS),
                                                ("terminal", term_think, {str(e["episode_id"]): e["band"] for e in term_eps},
                                                 efficiency.TERMINAL_LIMITS)):
                for band in ("easy", "medium", "hard"):
                    xs = [x for ep, v in think.items() if band_of.get(ep) == band for x in v]
                    lim = limits[band]
                    lo = lim.mean_min if fam == "swe" else 0
                    mean = sum(xs) / len(xs) if xs else 0.0
                    slice_means[f"{fam}/{band}"] = {"mean": round(mean, 1), "window": [lo, lim.mean_max], "turns": len(xs),
                                                    "pass": not xs or lo <= mean <= lim.mean_max}
            overall = sum(term_turns) / len(term_turns) if term_turns else 0.0
            slice_means["terminal/overall"] = {"mean": round(overall, 1), "window": [0, TERMINAL_V6_OVERALL_MEAN],
                                               "turns": len(term_turns), "pass": overall <= TERMINAL_V6_OVERALL_MEAN}
        gate("band_limits", not swe_viol and not term_viol and not term_row_viol and all(v["pass"] for v in slice_means.values()),
             rule=("thinking per trained turn = the position of </think> in each trained span of the rows (ceiling and mean "
                   "window per episode); steps and terminal output per trajectory; band row caps; <= 32K loss tokens per row"
                   if per_episode else
                   "v6 (gates are for safety, not selection): every trained turn within its "
                   "band's ceiling; the mean thinking per trained turn checked per band over the slice (SWE easy <= 270, medium "
                   "<= 360, hard 180-450; terminal <= 450 / 750 / 1,000 and <= 400 overall), not per episode; SWE steps per "
                   "band, terminal steps <= 20; band row caps; <= 32K loss tokens per row. Thinking = the position of </think> "
                   "in each trained span of the rows"),
             slice_means=slice_means,
             swe_rows=len(swe), swe_trained_turns=len(swe_turns), swe_violations=dict(list(swe_viol.items())[:20]),
             terminal_episodes=len(term_eps), terminal_trained_turns=len(term_turns),
             terminal_violations=dict(list(term_viol.items())[:20]), terminal_row_violations=term_row_viol[:20],
             thinking_per_trained_turn={"swe": per_turn_stats(swe_turns), "terminal": per_turn_stats(term_turns)})
        # verified outcomes, zero wasted actions, verify before completion, parser round trip (builder checks, every row)
        swe_checks, term_checks = swe_man.get("checks") or {}, term_man.get("checks") or {}
        gate("verified_outcomes", all(r["label_provenance"] == "verifier" for r in swe) and all(
            r["label_provenance"] == "verifier" for r in term_rows), rule="SWE resolved == 1, terminal reward == 1 (both builders admit nothing else)")
        term_wasted = [e.get("episode_id") for e in term_eps if (e.get("profile") or {}).get("wasted_actions", 1) != 0]
        gate("zero_wasted_actions", bool(swe_checks.get("zero_wasted_actions")) and not term_wasted, terminal_violations=term_wasted[:20])
        term_unver = [e.get("episode_id") for e in term_eps if not ((e.get("profile") or {}).get("completed") and (e.get("profile") or {}).get("verified"))]
        gate("verify_before_completion", bool(swe_checks.get("verified_completion")) and not term_unver, terminal_violations=term_unver[:20])
        ret_checks = ret_man.get("checks")                 # the retention builder writes a list of {check, pass, detail}
        ret_ok = all(c.get("pass") for c in ret_checks) if isinstance(ret_checks, list) else bool((ret_checks or {}).get("pass"))
        gate("parser_round_trip", bool((swe_checks.get("roundtrip") or {}).get("pass")) and bool(term_checks.get("all_pass", False)) and ret_ok,
             rule="every SWE turn passed the qwen3 parse gate and every terminal row the Terminus-2 parse; the builders' "
                  "re-render checks pass", swe_roundtrip=(swe_checks.get("roundtrip") or {}).get("rows_checked"),
             terminal_checks_pass=term_checks.get("all_pass"), retention_checks_pass=ret_ok)
    # contamination, duplicates, dev overlap: every p1 drop applied, p2 over these files dropped nothing
    p1_ok = (m["screens"][L.p1].get("current") is True)
    screened = {f["sha256"] for inp in (p2.get("inputs") or {}).values() for f in inp.get("files", [])}
    p2_ok = bool(p2) and all(m["files"][k]["sha256"] in screened for k in ("train_rows", "dev_rows")) and not p2.get("dropped")
    d3_ok = "skipped" not in ((p2.get("counts") or {}).get("d3") or {})
    dev = common.read_json(paths.DEV_EXCLUSIONS) or {}
    dev_ids = {str(x) for x in (dev.get("swe_task_ids") or []) + (dev.get("terminal_task_ids") or [])}
    mix_tasks = {str(rows_task) for rows_task in (r["task_id"] for r in swe + term_rows)}
    dev_hits = sorted(t for t in mix_tasks if t in dev_ids or t.split(":", 1)[-1] in {d.split(":", 1)[-1] for d in dev_ids})
    gate("contamination_duplicates_dev", p1_ok and p2_ok and d3_ok and not dev_hits, p1_current=p1_ok, p2_screened_no_drops=p2_ok,
         d3_applied=d3_ok, dev_overlap=dev_hits[:20], p2_counts={k: (p2.get("counts") or {}).get(k) for k in ("d1", "d2", "d3")})
    # mix within 5 points: SWE language (episodes), band (loss tokens), recovery; or the reported supply limit
    draw = (common.read_json(files["swe_selection"]) or {}).get("report") or {}
    n = len(swe)
    lang = {g: sum(r["language_group"] == g for r in swe) / n if n else 0.0 for g in swe_sel.GROUPS}
    loss = sum(r["n_assistant_tokens"] for r in swe)
    band = {b: sum(r["n_assistant_tokens"] for r in swe if r["band"] == b) / loss if loss else 0.0 for b in swe_sel.BANDS}
    rec = sum(bool(r["recovery_arc"]) for r in swe) / n if n else 0.0
    exhausted = {b: draw.get("bands", {}).get(b, {}).get("exhausted_groups", []) for b in swe_sel.BANDS}
    supply_short = {b: draw.get("bands", {}).get(b, {}).get("supply_exhausted") for b in swe_sel.BANDS}
    rec_short = sum(draw.get("bands", {}).get(b, {}).get("recovery_supply_short_picks", 0) for b in swe_sel.BANDS)
    lang_target = ((swe_man.get("rules") or {}).get("draw_settings") or {}).get("language_shares") or swe_sel.LANGUAGE_SHARES
    lang_dev = {g: round(100 * (lang[g] - lang_target[g]), 2) for g in swe_sel.GROUPS}
    settings = (swe_man.get("rules") or {}).get("draw_settings") or {}
    keep_all = settings.get("keep_all_bands") or []
    band_target = {b: (None if b in keep_all else (settings.get("band_shares") or swe_sel.BAND_LOSS_SHARES)[b]) for b in swe_sel.BANDS}
    band_dev = {b: round(100 * (band[b] - band_target[b]), 2) if band_target[b] is not None else None for b in swe_sel.BANDS}
    ran_out = {g for b in swe_sel.BANDS for g in exhausted[b]}
    # under target: acceptable when the group itself ran out; over target: when another group ran out (its gap moved here)
    lang_ok = all(abs(d) <= MIX_GATE_PP or (d < 0 and g in ran_out) or (d > 0 and bool(ran_out - {g})) for g, d in lang_dev.items())
    # keep-all bands must have drawn their whole admissible supply; a share band below its share is
    # acceptable only when its supply ran out, above it only when a harder band's did (the supply rule of plan 1.3)
    band_ok = all(supply_short[b] if d is None else (abs(d) <= MIX_GATE_PP or (d < 0 and supply_short[b])
                                                      or (d > 0 and any(supply_short[h] for h in swe_sel.BANDS[:swe_sel.BANDS.index(b)])))
                  for b, d in band_dev.items())
    rec_ok = rec >= swe_sel.RECOVERY_MIN - MIX_GATE_PP / 100 or rec_short > 0
    gate("mix_within_5_points", lang_ok and band_ok and rec_ok,
         swe_language_shares={g: _pp(v) for g, v in lang.items()}, language_deviation_pp=lang_dev, exhausted_groups=exhausted,
         swe_band_loss_shares={b: _pp(v) for b, v in band.items()},
         swe_band_targets={b: ("all admitted" if t is None else _pp(t)) for b, t in band_target.items()},
         band_deviation_pp=band_dev, band_supply_exhausted=supply_short, draw_settings=settings, swe_cap=m.get("swe_cap"),
         swe_recovery_share=_pp(rec), recovery_supply_short_picks=rec_short,
         terminal_bands=dict(Counter(e["band"] for e in term_eps)))
    # plan 3.5's slice shares are targets, not a 1.3 gate (and later slice decisions moved them): reported
    info = {"slice_loss_shares_pct": {s: _pp(v) for s, v in m["mixture"]["shares_by_loss_tokens"].items()},
            "slice_targets_pct": {s: _pp(v) for s, v in L.target_shares.items()}, "deviation_pp": m["mixture"]["deviation_pp"]}
    return {"created": common.utc_now(), "plan": L.plan, "mixture_run_id": m["run_id"], "mixture_sha256": common.sha256_file(L.man_dir / "mixture.json"),
            "pass": all(g["pass"] for g in gates.values()), "gates": gates, "informational": info,
            "decisions": list(a.decision),
            "dropped_nice_to_haves": {"swe": draw.get("dropped_nice_to_haves", [])}}


def per_turn_stats(values: list[int]) -> dict[str, float]:
    v = sorted(values)
    if not v:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    return {"mean": round(sum(v) / len(v), 1), "p50": float(v[len(v) // 2]), "p90": float(v[min(len(v) - 1, int(0.9 * len(v)))]),
            "max": v[-1]}


def card_text(a: argparse.Namespace, report: dict[str, Any]) -> str:
    L = a.layout
    files = slice_files(a)
    m = common.read_json(L.man_dir / "mixture.json")
    selection = common.read_json(L.man_dir / "selection.json")
    in_mix = {k for s in ("swe", "terminal", "retention") for k in selection["units"][s]}
    swe = [r for r in sel.read_meta(files["swe_rows"]) if str(r["episode_id"]) in in_mix]
    swe_sel_doc = common.read_json(files["swe_selection"]) or {}
    swe_man = common.read_json(files["swe_manifest"]) or {}
    term_eps = [e for e in read_jsonl(files["terminal_episodes"]) if str(e.get("episode_id")) in in_mix]
    g = report["gates"]
    mx = m["mixture"]
    lines = ["# Data card, revision 5 (stage-1 SFT mixture)", "",
             f"Generated by `scripts/data/build_mixture_v5.py report --version {L.version}` from `{rel(L.man_dir / 'mixture.json')}` (run {m['run_id']}); "
             f"plan: {L.plan}. Every number below comes from a manifest or the rows; nothing is hand-entered.", "",
             "## Sizes and shares (loss tokens carry the gradient; the loss is token-averaged)", "",
             "| Slice | Rows | Units | Sequence tokens | Loss tokens | Loss share | Target |", "|---|---|---|---|---|---|---|"]
    for s in ("swe", "terminal", "retention"):
        d = mx["slices"][s]
        lines.append(f"| {s} | {d['rows']:,} | {d['units']:,} | {d['sequence_tokens']:,} | {d['loss_tokens']:,} | "
                     f"{_pp(d['loss_share'])} % | {_pp(L.target_shares[s])} % |")
    t = mx["totals"]
    lines += [f"| **total** | {t['rows']:,} | {t['units']:,} | {t['sequence_tokens']:,} | {t['loss_tokens']:,} | | |", "",
              f"Validation shard: {m['split']['validation_rows']} rows ({m['split']['validation_tasks']} tasks, task-grouped, "
              f"stratified by cell); training rows: {m['split']['train_rows']}.", ""]
    rep = swe_sel_doc.get("report") or {}
    lines += ["## SWE slice (nvidia/Open-SWE-Traces v1.2, Qwen3.8-27B, mini-swe-agent)", "",
              "Bands from the teacher's attempts (hard 1 of 3, medium 2 of 3 or 1 of 2, easy 3 of 3 or 2 of 2); limits per band "
              "(data plan 1.3); per instance the gate-passing attempt with the lowest mean thinking per turn. Draw settings: "
              + json.dumps((swe_man.get("rules") or {}).get("draw_settings") or {}) + ".", "",
              "| Band | Episodes | Loss tokens | Target | Share | Supply exhausted | Recovery share |", "|---|---|---|---|---|---|---|"]
    for b in swe_sel.BANDS:
        rows_b = [r for r in swe if r["band"] == b]
        br = (rep.get("bands") or {}).get(b, {})
        loss_b = sum(r["n_assistant_tokens"] for r in rows_b)
        target = br.get("target_loss_tokens")
        lines.append(f"| {b} | {len(rows_b)} | {loss_b:,} | {f'{target:,}' if target is not None else 'all admitted'} | "
                     f"{g['mix_within_5_points']['swe_band_loss_shares'][b]} % | {br.get('supply_exhausted')} | "
                     f"{_pp(sum(bool(r['recovery_arc']) for r in rows_b) / len(rows_b)) if rows_b else 0.0} % |")
    lines += ["", "Language groups (episodes): " + ", ".join(f"{k} {v} %" for k, v in g["mix_within_5_points"]["swe_language_shares"].items())
              + "; targets " + ", ".join(f"{k} {_pp(v)} %" for k, v in swe_sel.LANGUAGE_SHARES.items()) + ".",
              "Categories (episodes): " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["category"] for r in swe).items())) + ".",
              f"Episodes {len(swe)} from {len({r['task_id'] for r in swe})} instances ({sum(r.get('attempt', 1) == 2 for r in swe)} "
              f"second attempts) in {len({r['repo'] for r in swe})} repositories.",
              f"Recovery share (arc B): {g['mix_within_5_points']['swe_recovery_share']} % (target >= 20 %; picks short of "
              f"recovery supply: {g['mix_within_5_points']['recovery_supply_short_picks']}).",
              "Thinking per turn of the drawn episodes (Qwen3.5-9B tokens): "
              + json.dumps(swe_man.get("selected_thinking_per_turn") or {}) + "; steps per episode: mean "
              + f"{round(sum(r['steps'] for r in swe) / len(swe), 1) if swe else 0}.",
              "Admitted supply per (language group, band), instances: " + json.dumps(
                  {k: v["instances"] for k, v in ((swe_man.get("supply") or {}).get("by_group_band") or {}).items()}) + ".",
              f"Nice-to-haves dropped (execution plan 0.2): {rep.get('dropped_nice_to_haves') or 'none'}.", ""]
    if ((swe_man.get("rules") or {}).get("masks") or {}).get("rule") == "v7":
        tt = swe_man.get("trained_turns") or {}
        lines += ["### v7 adaptive thinking masks (one random passing attempt per task, recovery trials first)", "",
                  "After the v6 read and unverified-submit masks: routine turns stay trained only with at most 1,000 thinking "
                  "tokens; decision-point turns (right after an error or a failing run, or before the first edit) up to 8K when "
                  "their reasoning does not repeat itself and they end in a tool call; thinking above 8K is never trained; trained "
                  "turns above 2K hold at most 10 % of the SWE loss tokens. Rules: `src/apex_data/render/swe_masks.py` (rule v7).", "",
                  f"Trained SWE turns {tt.get('trained_turns', 0):,}: thinking p50 {tt.get('p50')} / p90 {tt.get('p90')} / p99 "
                  f"{tt.get('p99')} / max {tt.get('max')}; routine p90 {tt.get('routine_p90')}; decision p90 {tt.get('decision_p90')}; "
                  f"turns above 2K {100 * tt.get('share_turns_over_2k', 0):.2f} % of trained turns and {100 * tt.get('share_loss_over_2k', 0):.2f} % "
                  f"of SWE loss tokens. Mask reasons: {json.dumps(swe_man.get('mask_reasons'))}.", ""]
    masks = swe_man.get("v6_masks")
    if masks:
        mr = masks["report"]
        lines += ["### v6 turn masks on the SWE rows (same episodes and token ids)", "",
                  "S1 masks read-only turns (they list, print or search) unless among the first 3 turns, right after an error, or "
                  "right before an edit; S2 masks the submit turn when none of the 3 previous turns ran tests, the program or "
                  "inline code. Masked turns stay in the sequence as context. Rules: `src/apex_data/render/swe_masks.py`.", "",
                  f"Read turns trained {mr['turns_before']['read']:,} -> {mr['turns_after']['read']:,}; submit turns masked "
                  f"{mr['reasons'].get('submit_unverified', 0)}; trained turns {mr['trained_turns']['before']:,} -> "
                  f"{mr['trained_turns']['after']:,}; SWE loss tokens {mr['loss_tokens']['before']:,} -> {mr['loss_tokens']['after']:,}; "
                  f"thinking per trained turn {json.dumps(mr['thinking_per_trained_turn']['after'])}.", ""]
    if m.get("swe_trim"):
        t = m["swe_trim"]
        lines += [f"**SWE trim ({m.get('revision') or 'targeted'}):** kept {t['keep_fraction']} of each band's SWE loss tokens; "
                  f"{t['episodes_removed']} episodes removed ({t['loss_tokens_removed']:,} loss tokens): first the episodes with an "
                  f"unverified submit ({t['unverified_submit_episodes']} flagged), then the end of each band's draw order with the "
                  f"language shares protected ({t['lang_tolerance_pp']} point; skips {json.dumps(t['language_skips'])}). Per band: "
                  + json.dumps({b: {k: v[k] for k in ('loss_before', 'loss_after', 'removed_unverified_submit', 'removed_draw_order')}
                                for b, v in t["per_band"].items()}) + ".", ""]
    if (m.get("retention_trim") or {}).get("rows_from"):
        r = m["retention_trim"]
        lines += [f"**Retention:** the {r['rows']} retention rows of `{r['rows_from']}` unchanged ({r['loss_tokens']:,} loss tokens).", ""]
    elif m.get("retention_trim"):
        r = m["retention_trim"]
        lines += [f"**Retention by part:** {r['rows']} rows, {r['loss_tokens']:,} loss tokens; kept whole: {', '.join(r['keep_whole'])}; "
                  "the other parts trimmed from the end of their rank order in proportion to their current shares. Shares (now / "
                  "before): " + ", ".join(f"{p} {100 * v['share']:.1f} / {100 * v['reference_share']:.1f} %" for p, v in r["parts"].items())
                  + ".", ""]
    if m.get("swe_cap"):
        c = m["swe_cap"]
        lines += [f"SWE cap {c['max_share']} (the D1 rule): every band keeps {c['kept_fraction']} of its loss tokens; "
                  f"{c['episodes_removed']} episodes removed from the end of each band's draw order ({c['loss_tokens_removed']:,} "
                  "loss tokens).", ""]
    if L.band_rule == "slice":
        lines += ["**Gate change in v6 (gates are for safety, not selection):** the thinking limits are checked as the per-turn "
                  "band ceilings on every trained turn and as the mean per trained turn per band over the slice (SWE easy <= 270, "
                  "medium <= 360, hard 180-450; terminal <= 450 / 750 / 1,000 and <= 400 overall), not per episode. Masking the "
                  "short-thinking read turns moved some SWE episodes' own means outside their windows; they are kept.", "",
                  "Slice means: " + json.dumps(g["band_limits"].get("slice_means")) + ".", ""]
    thinking = [x for e in term_eps for x in (e.get("profile") or {}).get("thinking", [])]
    lines += ["## Terminal slice (ai-and/termgrade-trajectories, Kimi-K3, Terminus-2 per-turn rows)", "",
              f"Episodes {len(term_eps)} by band {dict(Counter(e['band'] for e in term_eps))}; thinking per turn "
              f"{json.dumps(per_turn_stats(thinking))}; turns per episode mean "
              f"{round(sum((e.get('profile') or {}).get('steps', 0) for e in term_eps) / len(term_eps), 1) if term_eps else 0}.", "",
              "## Retention slice (Nemotron math, science, instruction-following, chat; ~70 hard-math rows)", "",
              f"Ranked prefix of {selection['retention_prefix']['rows']} rows, {selection['retention_prefix']['loss_tokens']:,} loss "
              f"tokens (target {selection['retention_prefix']['target_loss_tokens']:,} = 25 % of the total); part shares "
              + json.dumps(selection["retention_prefix"]["part_shares"]) + ".", "",
              "## Screens and decontamination", "",
              f"p1 ({L.p1}) dropped {m['screens'][L.p1]['dropped_ids']} ids before the build; drops per slice {json.dumps(m['drops'])}. "
              f"p2 ({L.p2}) over the final files: {json.dumps(g['contamination_duplicates_dev']['p2_counts'])[:600]}.", "",
              "D2 repository exclusion with the GitHub fork graph: the SWE builder looked up "
              f"{((swe_man.get('d2') or {}).get('fork_graph_lookup') or {}).get('requested')} repositories (cached lookups, token present); "
              f"p1 fork-graph states {json.dumps(((common.read_json(paths.MANIFESTS / f'screens_{L.p1}.json') or {}).get('counts') or {}).get('d2', {}).get('fork_graph'))} "
              "(not_run = items without a code repository: terminal, retention, dev QA); the p2 row pass checks repository names "
              "and signatures only, so the fork graph is not run there.", "",
              "## Acceptance gates (data plan 1.3)", "", "| Gate | Pass |", "|---|---|"]
    lines += [f"| {k} | {v['pass']} |" for k, v in g.items()]
    lines += ["", f"Overall: **{'PASS' if report['pass'] else 'FAIL'}** (details: `{rel(L.man_dir / 'gate_report.json')}`).", ""]
    if report.get("decisions"):
        lines += ["## Decisions recorded at the freeze", "", *[f"- {d}" for d in report["decisions"]], ""]
    lines += ["## Sources and licences", "", "| Slice | Dataset | Revision | Licence | Generator | Label | Rows | Loss tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for src in m["sources"]:
        lines.append(f"| {src['slice']} | {src['hf_repo']} | {str(src['hf_revision'])[:12]} | {src['dataset_license']} | "
                     f"{src['generator']} | {src['label_provenance']} | {src['rows']:,} | {src['loss_tokens']:,} |")
    return "\n".join(lines) + "\n"


def cmd_report(a: argparse.Namespace) -> int:
    L = a.layout
    run_id = a.run_id or common.new_run_id(f"data_mixture_{L.version}_report")
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="build_mixture_v5 report", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running")
        report = gate_report(a)
        common.write_json(L.man_dir / "gate_report.json", report)
        L.card.write_text(card_text(a, report), encoding="utf-8")
        run.log(f"gates: {json.dumps({k: v['pass'] for k, v in report['gates'].items()})}; overall {report['pass']}")
        for k, v in report["gates"].items():
            if not v["pass"]:
                run.error(f"gate {k} fails: {json.dumps(v)[:1500]}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), status="finished", gates_pass=report["pass"],
                           outputs={"gate_report": rel(L.man_dir / "gate_report.json"), "card": rel(L.card)})
        run.finalize()
    print(json.dumps({k: v["pass"] for k, v in report["gates"].items()}))
    return 0 if report["pass"] else 1


# --- freeze -----------------------------------------------------------------------------------------------------------
def cmd_freeze(a: argparse.Namespace) -> int:
    L = a.layout
    run_id = a.run_id or common.new_run_id(f"data_mixture_{L.version}_freeze")
    path = L.man_dir / "mixture.json"
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="build_mixture_v5 freeze", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running")
        m = common.read_json(path)
        if not m:
            raise SystemExit(f"no mixture manifest at {path}")
        if m.get("frozen"):
            raise SystemExit(f"{path} is already frozen (run {m.get('freeze', {}).get('run_id')}); a frozen set never changes")
        problems = []
        for k, f in m["files"].items():
            p = ROOT / f["path"]
            if not p.exists() or common.sha256_file(p) != f["sha256"]:
                problems.append(f"{k}: {f['path']} missing or changed since the build")
        st = p1_status(common.read_json(paths.MANIFESTS / f"screens_{L.p1}.json"), slice_files(a), L.p1)
        if not st["current"]:
            problems.append(f"{L.p1} not current: {st['reason']}")
        elif common.sha256_file(paths.MANIFESTS / f"screens_{L.p1}.json") != m["screens"][L.p1]["sha256"]:
            problems.append(f"screens_{L.p1}.json changed since the build (run build again)")
        report = gate_report(a)
        common.write_json(L.man_dir / "gate_report.json", report)
        problems += [f"gate {k} fails" for k, v in report["gates"].items() if not v["pass"]]
        if problems:
            for pr in problems:
                run.error(pr)
            run.write_metadata(end_time=common.utc_now(), status="refused", problems=problems)
            run.finalize()
            print("freeze refused:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        p2_path = paths.MANIFESTS / f"screens_{L.p2}.json"
        m.update({"status": "FROZEN", "frozen": True,
                  "freeze": {**m.get("freeze", {}), "run_id": run_id, "frozen_at": common.utc_now(),
                             "gate_report_sha256": common.sha256_file(L.man_dir / "gate_report.json"),
                             L.p2: {"run_id": common.read_json(p2_path).get("run_id"), "sha256": common.sha256_file(p2_path)}}})
        common.write_json(path, m)
        if a.no_marker:
            run.log(f"{path} frozen; marker not written (--no-marker): run `marker --version {L.version}` when the consumers may start")
        else:
            write_marker(a, m)
            run.log(f"{path} frozen; marker {L.coord / L.marker}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), status="finished", marker_written=not a.no_marker)
        run.finalize()
    return 0


def write_marker(a: argparse.Namespace, m: dict[str, Any]) -> dict[str, Any]:
    """The freeze marker (logs/coordination/<version>/<marker>) with the sha256 of every frozen file."""
    L = a.layout
    path = L.man_dir / "mixture.json"
    artifacts = {f["path"]: f["sha256"] for f in m["files"].values()}
    for p in (path, L.man_dir / "selection.json", L.man_dir / "split.json", L.man_dir / "gate_report.json", L.card):
        artifacts[rel(p)] = common.sha256_file(p)
    sh = m["mixture"]["shares_by_loss_tokens"]
    marker = {"producer": "scripts/data/build_mixture_v5.py freeze", "utc": common.utc_now(),
              **({"revision": a.revision or m.get("revision")} if (a.revision or m.get("revision")) else {}), "artifacts": artifacts,
              "notes": (f"frozen {L.version} SFT mixture, run {m['run_id']}: train {m['split']['train_rows']} rows, dev "
                        f"{m['split']['validation_rows']} rows, {m['mixture']['totals']['loss_tokens']:,} loss / "
                        f"{m['mixture']['totals']['sequence_tokens']:,} sequence tokens; loss shares SWE {_pp(sh['swe'])} / "
                        f"terminal {_pp(sh['terminal'])} / retention {_pp(sh['retention'])} %. For configs/sft.yaml: "
                        f"APEX_SFT_TRAIN_ROWS={m['files']['train_rows']['path']} APEX_SFT_DEV_ROWS={m['files']['dev_rows']['path']}, "
                        f"data.mixture_manifest={rel(path)}. Gate report: {rel(L.man_dir / 'gate_report.json')} (all pass).")}
    common.write_json(L.coord / L.marker, marker)
    return marker


def cmd_marker(a: argparse.Namespace) -> int:
    """Write the marker of an already frozen mixture after checking that every frozen file still has its sha256."""
    L = a.layout
    m = common.read_json(L.man_dir / "mixture.json")
    if not m or not m.get("frozen"):
        raise SystemExit(f"{L.man_dir / 'mixture.json'} is not frozen: run `freeze` first")
    changed = [k for k, f in m["files"].items() if common.sha256_file(ROOT / f["path"]) != f["sha256"]]
    if changed:
        raise SystemExit(f"frozen files changed since the freeze: {changed}")
    marker = write_marker(a, m)
    print(f"{L.coord / L.marker} written ({len(marker['artifacts'])} artifacts)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("items", "build", "report", "freeze", "marker"))
    p.add_argument("--no-marker", action="store_true", help="freeze: do not write the marker yet (`marker` writes it later)")
    p.add_argument("--version", choices=tuple(LAYOUTS), default="v5", help="the data version (paths, screen passes, marker, rules)")
    p.add_argument("--swe-dir", type=Path, default=None, help="default: the version's SWE pool")
    p.add_argument("--terminal-dir", type=Path, default=None, help="default: the version's terminal slice")
    p.add_argument("--retention-dir", type=Path, default=None, help="default: the version's retention slice")
    p.add_argument("--swe-max-share", type=float, default=None,
                   help="build: cap SWE at this share of all loss tokens, every band keeping the same fraction, episodes "
                        "dropped from the end of each band's draw order (v6 plan, the D1 rule: 0.35)")
    p.add_argument("--out-dir", type=Path, default=None, help="items: WORK/<run_id>/screen_items; build: data/processed/v5/sft")
    p.add_argument("--swe-trim-fraction", type=float, default=None, help="build: keep this fraction of the SWE loss tokens "
                   "with the targeted trim (per band; unverified submits first; then the draw order with language protection)")
    p.add_argument("--swe-lang-protect-pp", type=float, default=1.0, help="build: the targeted trim's language-share protection")
    p.add_argument("--retention-share", type=float, default=RETENTION_SHARE, help="build: retention's share of all loss tokens")
    p.add_argument("--retention-keep-whole", action="append", default=[], metavar="PART",
                   help="build: keep this retention part whole and trim the others by part from the current selection "
                        "(the 25 %% rank prefix), in proportion to their shares (repeatable; v6r2: hard_math)")
    p.add_argument("--retention-rows-from", type=Path, default=None, metavar="SELECTION_JSON",
                   help="build: take exactly the retention rows of this earlier selection.json (v7: data/manifests/v6/selection.json)")
    p.add_argument("--revision", default=None, help="build / freeze: a revision label recorded in the manifest and the marker")
    p.add_argument("--val-fraction", type=float, default=spl.VAL_FRACTION)
    p.add_argument("--decision", action="append", default=[], help="report / freeze: a decision to record in the gate report and "
                   "the data card (repeatable)")
    p.add_argument("--no-dev", action="store_true", help="items: skip the dev items")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    a = p.parse_args(argv)
    a.layout = LAYOUTS[a.version]
    a.swe_dir = a.swe_dir or a.layout.swe_dir
    a.terminal_dir = a.terminal_dir or a.layout.terminal_dir
    a.retention_dir = a.retention_dir or a.layout.retention_dir
    if a.command == "build" and a.out_dir is None:
        a.out_dir = a.layout.out_dir
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return {"items": cmd_items, "build": cmd_build, "report": cmd_report, "freeze": cmd_freeze, "marker": cmd_marker}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
