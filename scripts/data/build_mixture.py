#!/usr/bin/env python
"""The candidate SFT mixture, the validation shard, the pilot draw and their manifests (WP7, unit U2.4; docs/DATA_PLAN.md
revision 4.3 sections 4, 7 and 9; docs/EXECUTION_PLAN.md M2.4 and M2.5).

Run from the repository root, in this order:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture.py items
    OMP_NUM_THREADS=24 PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name p1_pools --pools '' \\
        --workers 24 --pool swe_teacher=<items dir>/swe_teacher.jsonl --pool terminal_teacher=<items dir>/terminal_teacher.jsonl \\
        --pool retention=<items dir>/retention.jsonl --pool dev=<items dir>/dev.jsonl        (printed by `items`)
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture.py build [--cap 64k]
    OMP_NUM_THREADS=24 PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name p2_rows --pools '' \\
        --workers 24 --rows data/processed/sft/train_rows.parquet --rows data/processed/sft/dev_rows.parquet
    (if p2_rows dropped any row: `build` again, then p2_rows again, until p2_rows drops nothing)
    PYTHONPATH=src .venv/train/bin/python scripts/data/write_data_card.py
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture.py freeze      (M2.5, after review)

Subcommands
  items   p1 screen items from the pools: one record per pool row (task text decoded from the rendered row, the SWE issue
          cut out of the mini-swe-agent instance template; repo, PR and the SWE reference patch from the pinned task rows)
          plus the frozen dev items with their text (split=dev, so run_screens checks train-vs-dev overlap on text).
          Writes WORK/<run_id>/screen_items/{swe_teacher,terminal_teacher,retention,dev}.jsonl and items_manifest.json
          (sha256 of every pool file read, so `build` can tell a stale p1 pass).
  build   For each cap: the pool's own selection minus every unit the screens dropped (a p1_pools drop removes the whole
          task, a p2_rows drop the hit row's episode or row; p2 drops are sticky through selection.json); dropped SWE
          rows replaced from the pool's spare rows (same quota part and size stratum, <= 5 rows per repository),
          shortfalls reported; the budget rule (mixture.py) at --throughput and --max-hours; loss-token shares against
          SWE 67 / terminal 18 / retention 15 +-2 pp. At --cap only: the task-grouped validation shard (--val-fraction
          of the rows, stratified by cell), the pilot (--pilot-fraction of the training rows, task-stratified), and the
          rows files.
  freeze  M2.5: checks that the files on disk are the ones mixture.json lists, that p1_pools is current and that
          p2_rows screened exactly these train/dev files and dropped nothing, then sets "frozen": true.

Inputs   data/pools/{swe_teacher,terminal_teacher,retention}/ (rows, selections, manifests; never written here),
         data/manifests/screens_{p1_pools,p2_rows}.json, data/manifests/dev_*.json, RAW task rows (items only),
         the base-model tokenizer (items only).
Outputs  data/processed/sft/{train,dev,pilot}_rows.parquet (apex_train.data_format.ROW_COLUMNS only) and
         {train,dev,pilot}_provenance.parquet (row_key + schema.PROVENANCE_FIELDS, aligned row by row);
         data/manifests/selection.json (drops, replacements, shortfalls, budget trims, per cap; the row keys of --cap),
         data/manifests/split.json (validation and pilot tasks, per-cell arithmetic, invariants),
         data/manifests/mixture.json (per file sha256 / rows / tokens; per-slice and per-provenance counts and shares at
         --cap and at every other cap for the record; the budget rule; "frozen": false until `freeze`);
         logs/<run_id>/ (metadata.json, train.log, metrics.jsonl, errors.log, resources.csv).
Seeded by apex_data.paths.SEED (--seed).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths, schema  # noqa: E402
from apex_data.render import mixture as mix  # noqa: E402
from apex_data.render import select as sel  # noqa: E402
from apex_data.render import split as spl  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import ROW_COLUMNS, arrow_schema, load_rows  # noqa: E402

PLAN = "docs/DATA_PLAN.md revision 4.3.7, sections 4, 7 and 9"
THREAD_CAP = "OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false"   # host-load cap for the screens' embedder
FILE_KINDS = ("train", "dev", "pilot")
PROVENANCE_INT = ("patch_files", "patch_added_lines")
PROVENANCE_BOOL = ("fits_48k",)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def resources(run: common.RunDir, t0: float) -> None:
    duration = time.time() - t0
    run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock on the CPU VM"),
                         ("orchestrator_cpus", os.cpu_count(), "vcpu", "apex-cpu, one process"),
                         ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
                         ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                         ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)")])


# --- items ------------------------------------------------------------------------------------------------------------
def cmd_items(a: argparse.Namespace) -> int:
    from apex_train.data_format import load_tokenizer

    run_id = a.run_id or common.new_run_id("data_mixture_items")
    out_dir = a.out_dir or (paths.WORK / run_id / "screen_items")
    files = sel.PoolFiles.under(a.pools_dir)
    t0 = time.time()
    with common.RunDir(run_id, a.log_root, log_name="train.log") as run:
        pool_sha = {rel(p): common.sha256_file(p) for p in files.all_files() if p.exists()}
        run.write_metadata(run_id=run_id, step="build_mixture items", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running",
                           config={"pools_dir": str(a.pools_dir), "out_dir": str(out_dir), "dev_items": not a.no_dev},
                           data_version={"pool_files": pool_sha}, model_revision=common.model_revision())
        tok = load_tokenizer(paths.BASE_MODEL_DIR)
        swe_meta = sel.read_meta(files.swe_rows)
        upstream = sel.swe_upstream_fields({(str(r["oswt_subsource"]), str(r["task_id"]).split(":", 1)[1]) for r in swe_meta})
        run.log(f"SWE reference patches found for {len(upstream)} of {len({r['task_id'] for r in swe_meta})} instances")
        items, stats = sel.pool_screen_items(files, tok, swe_upstream=upstream)
        if not a.no_dev:
            items["dev"], stats["dev"] = sel.dev_screen_items(a.manifests_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, recs in items.items():
            p = out_dir / f"{name}.jsonl"
            schema.write_jsonl(p, recs)
            written[name] = {"path": str(p), "records": len(recs), "sha256": common.sha256_file(p)}
        manifest = {"run_id": run_id, "created": common.utc_now(), "pool_files": pool_sha, "items": written, "stats": stats,
                    "tokenizer": str(paths.BASE_MODEL_DIR)}
        common.write_json(out_dir / "items_manifest.json", manifest)
        run.log(f"items: {json.dumps(stats)}")
        run.metric({"step": "items", **{k: v["records"] for k, v in written.items()}})
        cmd = (f"{THREAD_CAP} PYTHONPATH=src .venv/train/bin/python scripts/data/run_screens.py --pass-name p1_pools --pools '' --workers 24 "
               + " ".join(f"--pool {n}={w['path']}" for n, w in written.items()))
        run.log(f"next: {cmd}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(time.time() - t0, 1), status="finished",
                           summary={"items": {k: v["records"] for k, v in written.items()}, "stats": stats, "out_dir": str(out_dir),
                                    "next_command": cmd})
        run.finalize()
    print(cmd)
    return 0


# --- build ------------------------------------------------------------------------------------------------------------
def p1_status(p1: dict[str, Any] | None, files: sel.PoolFiles) -> dict[str, Any]:
    """Is the p1_pools pass current for these pool files? (via the items manifest next to its input files)"""
    if not p1:
        return {"present": False, "current": False, "reason": "no screens_p1_pools.json"}
    item_paths = [Path(f["path"]) for inp in (p1.get("inputs") or {}).values() for f in inp.get("files", [])]
    im = next((p.parent / "items_manifest.json" for p in item_paths if (p.parent / "items_manifest.json").exists()), None)
    if im is None:
        return {"present": True, "current": False, "reason": "the p1 inputs carry no items_manifest.json"}
    recorded = json.loads(im.read_text(encoding="utf-8")).get("pool_files", {})
    now = {rel(p): common.sha256_file(p) for p in files.all_files() if p.exists()}
    stale = sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k) and not k.endswith("manifest.json"))
    for name, inp in (p1.get("inputs") or {}).items():
        for f in inp.get("files", []):
            if Path(f["path"]).exists() and common.sha256_file(Path(f["path"])) != f["sha256"]:
                stale.append(f["path"])
    return {"present": True, "current": not stale, "stale_files": stale, "items_manifest": str(im), "run_id": p1.get("run_id"),
            "reason": "pool files changed since the items were written" if stale else "ok"}


def write_rows_file(units: list[sel.Unit], cands: sel.CapCandidates, path: Path) -> tuple[list[str], dict[str, Any]]:
    """Rows of `units` in unit order (each unit's rows in their pool order), ROW_COLUMNS only, written atomically."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    order = [k for u in units for k in u.row_keys]
    wanted = set(order)
    parts, where = [], {}
    for s, f in cands.files.items():
        keys_here = [k for k in order if sel.SLICE_OF_KIND[cands.rows[k]["row_kind"]] == s]
        if not keys_here:
            continue
        t = pq.read_table(f, columns=[*ROW_COLUMNS, "row_key"])
        t = t.filter(pc.is_in(t["row_key"], value_set=pa.array(keys_here)))
        base = sum(len(p) for p in parts)
        for i, k in enumerate(t["row_key"].to_pylist()):
            if k in where:
                raise sel.SelectionError(f"row_key {k} appears twice in {f}")
            where[k] = base + i
        parts.append(t.select(list(ROW_COLUMNS)).cast(arrow_schema()))
    missing = wanted - set(where)
    if missing:
        raise sel.SelectionError(f"{len(missing)} selected rows not found in the pool files, e.g. {sorted(missing)[:3]}")
    table = pa.concat_tables(parts).take(pa.array([where[k] for k in order], type=pa.int64()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return order, {"rows": table.num_rows, "sequence_tokens": int(pc.sum(table["n_tokens"]).as_py() or 0),
                   "loss_tokens": int(pc.sum(table["n_assistant_tokens"]).as_py() or 0)}


def provenance_schema() -> Any:
    import pyarrow as pa

    fields = [("row_key", pa.string())]
    for f in schema.PROVENANCE_FIELDS:
        fields.append((f, pa.int64() if f in PROVENANCE_INT else pa.bool_() if f in PROVENANCE_BOOL else pa.string()))
    return pa.schema(fields)


def write_provenance(order: list[str], cands: sel.CapCandidates, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = []
    for k in order:
        r = cands.rows[k]
        rec = {"row_key": k, **{f: r.get(f) for f in schema.PROVENANCE_FIELDS}}
        schema.validate_provenance(rec)
        recs.append(rec)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(recs, schema=provenance_schema()), tmp)
    os.replace(tmp, path)


def verify_files(out: dict[str, Path], orders: dict[str, list[str]]) -> dict[str, Any]:
    """Every rows file loads with data_format.load_rows (column and type checks) and apex_train.common.load_split; the
    sidecar is aligned (row_key == sample_id:turn_index, row by row) and validates; the pilot is a subset of train."""
    import pyarrow.parquet as pq

    from apex_train.common import load_split

    report: dict[str, Any] = {}
    for kind in FILE_KINDS:
        rows = load_rows(out[f"{kind}_rows"])
        if tuple(pq.ParquetFile(out[f"{kind}_rows"]).schema_arrow.names) != ROW_COLUMNS:
            raise sel.SelectionError(f"{kind}_rows columns differ from ROW_COLUMNS")
        prov = pq.read_table(out[f"{kind}_provenance"]).to_pylist()
        if len(prov) != len(rows):
            raise sel.SelectionError(f"{kind}: {len(rows)} rows but {len(prov)} provenance records")
        for r, p, k in zip(rows, prov, orders[kind], strict=True):
            if p["row_key"] != k or f"{r['sample_id']}:{r['turn_index']}" != k:
                raise sel.SelectionError(f"{kind}: provenance not aligned at {k}")
            schema.validate_provenance(p)
        ds, counts = load_split([out[f"{kind}_rows"]])
        report[kind] = {"load_rows": len(rows), "load_split": counts, "provenance_aligned": True,
                        "max_tokens": max((r["n_tokens"] for r in rows), default=0),
                        "min_loss_tokens": min((r["n_assistant_tokens"] for r in rows), default=0)}
        del rows, ds
    if not set(orders["pilot"]) <= set(orders["train"]):
        raise sel.SelectionError("pilot rows are not a subset of the training rows")
    report["pilot_subset_of_train"] = True
    side = {kind: pq.read_table(out[f"{kind}_provenance"], columns=["task_id", "episode_id"]).to_pylist() for kind in ("train", "dev")}
    for key in ("task_id", "episode_id"):
        both = {r[key] for r in side["train"]} & {r[key] for r in side["dev"]}
        if both:
            raise sel.SelectionError(f"{len(both)} {key}s on both sides of the split, e.g. {sorted(both)[:3]}")
    report["no_task_or_episode_on_both_sides"] = True
    return report


def p2_drops(a: argparse.Namespace, files: sel.PoolFiles, previous: dict[str, Any] | None) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Unit-level drops of the p2_rows pass: the current manifest's plus every earlier build's (sticky, so a later pass
    over the rebuilt files, which no longer hold the dropped rows, cannot bring them back)."""
    p2 = common.read_json(a.manifests_dir / "screens_p2_rows.json")
    rows = [*sel.read_meta(files.swe_rows), *(r for c in sel.CAPS for r in sel.read_meta(files.terminal_rows[c])),
            *sel.read_meta(files.retention_rows)]
    current = sel.row_drops_to_units(sel.screen_drops(p2, "p2_rows"), rows) if p2 else {}
    prev = (previous or {}).get("screens", {}).get("p2_rows", {})
    sticky = dict(prev.get("dropped_units", {}))
    merged = {u: sorted(set(sticky.get(u, [])) | set(current.get(u, []))) for u in set(sticky) | set(current)}
    history = dict(prev.get("history", {}))
    if p2 and p2.get("run_id"):
        c = p2.get("counts", {})
        history[p2["run_id"]] = {"inputs": {k: [f.get("sha256") for f in v.get("files", [])] for k, v in (p2.get("inputs") or {}).items()},
                                 "survival": p2.get("survival"), "units_dropped": len(current),
                                 "d1_items_dropped": (c.get("d1") or {}).get("n_items_dropped"),
                                 "d2_items_dropped": sum(1 for r in (p2.get("dropped") or {}).values() if any(x.startswith("d2_") for x in r)),
                                 "d3": {k: (c.get("d3") or {}).get(k) for k in ("items_screened", "items_hit", "hits", "by_rule", "skipped")}}
    info = {"manifest": rel(a.manifests_dir / "screens_p2_rows.json") if p2 else None, "run_id": (p2 or {}).get("run_id"),
            "history": dict(sorted(history.items())), "current_dropped_units": len(current), "sticky_from_previous_build": len(sticky),
            "dropped_units": merged}
    return merged, info


def terminal_profile(units: list[sel.Unit], labels_path: Path) -> dict[str, Any]:
    """Terminal tasks and episodes by TermGrade difficulty band (mean pass rate of the six labelled models in the
    dataset's labels.jsonl; the dev set's bands), episodes per task."""
    from apex_data.devset import terminal as dterm

    term = [u for u in units if u.slice == "terminal"]
    eps_per_task = Counter(u.task_id for u in term)
    out: dict[str, Any] = {"episodes": len(term), "tasks": len(eps_per_task), "rows": sum(len(u.row_keys) for u in term),
                           "episodes_per_task": dict(sorted(Counter(eps_per_task.values()).items())),
                           "by_source": dict(sorted(Counter(u.cell for u in term).items()))}
    if labels_path.exists():
        labels = dterm.task_labels(dterm.read_jsonl(labels_path))
        band = {t: dterm.band_of(dterm.mean_pass_rate(labels[t.split(':', 1)[1]])) if t.split(":", 1)[1] in labels else "unlabelled"
                for t in eps_per_task}
        out["bands"] = {"definition": "mean pass rate of the six labelled models: easy >= 2/3, medium >= 1/3, hard > 0, unsolved 0",
                        "tasks": dict(sorted(Counter(band.values()).items())),
                        "episodes": dict(sorted(Counter(band[u.task_id] for u in term).items())),
                        "labels_sha256": common.sha256_file(labels_path)}
    return out


def cmd_build(a: argparse.Namespace) -> int:
    run_id = a.run_id or common.new_run_id("data_mixture")
    files = sel.PoolFiles.under(a.pools_dir)
    out_dir, man_out = a.out_dir, a.manifests_out or a.manifests_dir
    caps = [a.cap, *(c for c in sel.CAPS if c != a.cap)]
    t0 = time.time()
    with common.RunDir(run_id, a.log_root, log_name="train.log") as run:
        pool_sha = {rel(p): common.sha256_file(p) for p in files.all_files() if p.exists()}
        pool_runs = {name: (common.read_json(p) or {}).get("run_id") for name, p in
                     (("swe_teacher", files.swe_manifest), ("terminal_teacher", files.terminal_manifest), ("retention", files.retention_manifest))}
        config = {"cap": a.cap, "caps_reported": caps, "throughput_tokens_per_s": a.throughput, "throughput_source": a.throughput_source,
                  "max_hours": a.max_hours,
                  "val_fraction": a.val_fraction, "pilot_fraction": a.pilot_fraction, "retention_floor": a.retention_floor,
                  "pools_dir": str(a.pools_dir), "out_dir": str(out_dir), "manifests_out": str(man_out),
                  "allow_unscreened": a.allow_unscreened}
        run.write_metadata(run_id=run_id, step="build_mixture build", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running",
                           config=config, data_version={"pool_files": pool_sha, "pool_runs": pool_runs},
                           model_revision=common.model_revision())
        p1 = common.read_json(a.manifests_dir / "screens_p1_pools.json")
        p1_info = p1_status(p1, files)
        run.log(f"p1_pools: {json.dumps({k: v for k, v in p1_info.items() if k != 'stale_files'})}")
        if not p1_info["current"] and not a.allow_unscreened:
            run.error(f"p1_pools not current: {p1_info}")
            raise SystemExit(f"p1_pools pass missing or stale ({p1_info['reason']}): run `items` and run_screens p1_pools first "
                             "(or --allow-unscreened for a development build outside data/)")
        previous = common.read_json(man_out / "selection.json")
        drops_p1 = sel.screen_drops(p1, "p1_pools") if p1_info["present"] else {}
        drops_p2, p2_info = p2_drops(a, files, previous)
        run.log(f"screen drops: p1 {len(drops_p1)} ids, p2 {len(drops_p2)} units")

        per_cap: dict[str, Any] = {}
        main: dict[str, Any] = {}
        for cap in caps:
            cands = sel.load_candidates(files, cap, a.seed)
            pool_ids = {x for u in cands.all_units() for x in (u.task_id, u.unit_id)}
            res = sel.apply_drops(cands, drops_p1, drops_p2)
            kept = [u for s in sel.SLICES for u in res[s].kept]
            budget = mix.apply_budget_rule(kept, throughput=a.throughput, max_hours=a.max_hours,
                                           retention_floor=a.retention_floor, seed=a.seed)
            acct = mix.account(budget.kept, cands.rows)
            per_cap[cap] = {"selection": {s: r.to_dict() for s, r in res.items()}, "budget": budget.report, "mixture": acct,
                            "swe_profile": mix.swe_profile(budget.kept, cands.rows),
                            "dropped_ids_not_in_pools": sorted((set(drops_p1) | set(drops_p2)) - pool_ids)}
            run.log(f"{cap}: rows {acct['totals']['rows']}, seq {acct['totals']['sequence_tokens']:,}, loss {acct['totals']['loss_tokens']:,}; "
                    f"loss shares {acct['shares_by_loss_tokens']} (within +-2 pp: {acct['within_tolerance']}); "
                    f"budget {budget.report['predicted_hours_after']} h of {a.max_hours} h, trimmed {budget.report['trimmed']}; "
                    f"drops {[res[s].to_dict()['units_dropped'] for s in sel.SLICES]}, replacements "
                    f"{[len(res[s].replacements) for s in sel.SLICES]}, shortfalls {[res[s].shortfall for s in sel.SLICES]}")
            run.metric({"step": "mixture", "cap": cap, **acct["totals"], "shares": acct["shares_by_loss_tokens"],
                        "within_tolerance": acct["within_tolerance"], "predicted_hours": budget.report["predicted_hours_after"]})
            if cap == a.cap:
                main = {"cands": cands, "units": budget.kept}

        cands, units = main["cands"], main["units"]
        val = spl.task_grouped_draw(units, a.val_fraction, seed=a.seed, label="validation")
        train_units, dev_units = val.rest, val.drawn
        pilot = spl.task_grouped_draw(train_units, a.pilot_fraction, seed=a.seed, label="pilot")
        straddle = spl.straddling_tasks(train_units, dev_units)
        if straddle:
            raise sel.SelectionError(f"tasks straddle train and validation: {straddle[:5]}")
        order = lambda us, lab: sorted(us, key=lambda u: (sel.rank(a.seed, lab, u.unit_id), u.unit_id))  # noqa: E731
        groups = {"train": order(train_units, "train_order"), "dev": order(dev_units, "dev_order"), "pilot": order(pilot.drawn, "pilot_order")}
        out = {f"{k}_{t}": out_dir / f"{k}_{t}.parquet" for k in FILE_KINDS for t in ("rows", "provenance")}
        orders, file_stats = {}, {}
        for kind, us in groups.items():
            orders[kind], file_stats[kind] = write_rows_file(us, cands, out[f"{kind}_rows"])
            write_provenance(orders[kind], cands, out[f"{kind}_provenance"])
            run.log(f"{kind}: {file_stats[kind]} -> {out[f'{kind}_rows']}")
        checks = verify_files(out, orders)
        run.log(f"file checks: {json.dumps(checks)}")
        dev_seq = file_stats["dev"]["sequence_tokens"]
        acct_parts = {k: mix.account(us, cands.rows, provenance=(k != "pilot")) for k, us in
                      (("all", units), ("train", train_units), ("dev", dev_units), ("pilot", pilot.drawn))}
        evals = mix.eval_overhead(len(orders["train"]), dev_seq, a.throughput)
        budget_main = per_cap[a.cap]["budget"]
        hours = {"training_and_validation_tokens": budget_main["predicted_hours_after"],
                 "training_tokens_only": mix.predicted_hours(file_stats["train"]["sequence_tokens"], a.throughput),
                 "eval_loss_passes_estimate": evals["hours"],
                 "pilot_training_tokens": mix.predicted_hours(file_stats["pilot"]["sequence_tokens"], a.throughput)}
        file_block = {}
        for k, p in out.items():
            kind = k.split("_")[0]
            file_block[k] = {"path": rel(p), "sha256": common.sha256_file(p), "bytes": p.stat().st_size,
                             "rows": file_stats[kind]["rows"],
                             **({"sequence_tokens": file_stats[kind]["sequence_tokens"], "loss_tokens": file_stats[kind]["loss_tokens"]}
                                if k.endswith("_rows") else {"columns": ["row_key", *schema.PROVENANCE_FIELDS], "aligned_with": f"{kind}_rows"})}
        screens = {"p1_pools": {**p1_info,
                                "manifest": rel(a.manifests_dir / "screens_p1_pools.json") if p1_info["present"] else None,
                                "sha256": common.sha256_file(a.manifests_dir / "screens_p1_pools.json") if p1_info["present"] else None,
                                "dropped_ids": len(drops_p1)},
                   "p2_rows": p2_info}
        man_out.mkdir(parents=True, exist_ok=True)
        selection = {"run_id": run_id, "created": common.utc_now(), "plan": PLAN, "seed": a.seed, "cap": a.cap,
                     "pool_runs": pool_runs, "pool_files": pool_sha, "screens": screens,
                     "rules": {"unit": "SWE episode row / terminal episode (its two trained turns) / retention row; a p1_pools drop (task text) removes every unit of the task, a p2_rows drop (rendered row) removes the row's unit",
                               "replacement": "SWE only: a spare pool row (rendered for the other cap, fits this cap) of the same quota part and size stratum, else the same part (borrowed); one row per instance; <= 5 rows per repository",
                               "no_spares": "terminal (every admissible TermGrade episode is already in the pool) and retention (the replay reserve is stage-2 data): drops are shortfalls"},
                     "caps": {c: {"selection": v["selection"], "budget": v["budget"], "dropped_ids_not_in_pools": v["dropped_ids_not_in_pools"]}
                              for c, v in per_cap.items()},
                     "row_keys": {a.cap: sorted(k for u in units for k in u.row_keys)}}
        common.write_json(man_out / "selection.json", selection)
        split_m = {"run_id": run_id, "created": common.utc_now(), "plan": PLAN, "seed": a.seed, "cap": a.cap,
                   "validation": {**val.report, "tasks": sorted({u.task_id for u in dev_units}), "row_keys": orders["dev"]},
                   "pilot": {**pilot.report, "of": "the training rows", "tasks": sorted({u.task_id for u in pilot.drawn})},
                   "invariants": {"straddling_tasks": straddle, "pilot_subset_of_train": checks["pilot_subset_of_train"],
                                  "no_task_or_episode_on_both_sides": checks["no_task_or_episode_on_both_sides"],
                                  "note": "units (SWE row, terminal episode with both trained turns, retention row) never split; tasks never straddle"},
                   "shares": {k: acct_parts[k]["shares_by_loss_tokens"] for k in ("all", "train", "dev", "pilot")}}
        common.write_json(man_out / "split.json", split_m)
        mixture = {"status": "CANDIDATE", "frozen": False, "run_id": run_id, "created": common.utc_now(), "plan": PLAN,
                   "seed": a.seed, "code": common.git_info(),
                   "cap": {"name": a.cap, "max_length": sel.CAPS[a.cap], "source": "M4.1 (T8 memory gate on the B200): 64K passes"},
                   "pools": {"runs": pool_runs, "files": pool_sha}, "screens": screens, "files": file_block,
                   "mixture": acct_parts["all"], "train": acct_parts["train"], "dev": acct_parts["dev"], "pilot": acct_parts["pilot"],
                   "other_caps": {c: {"mixture": {k: v["mixture"][k] for k in ("slices", "totals", "shares_by_loss_tokens",
                                                                              "shares_by_sequence_tokens", "targets", "tolerance_pp",
                                                                              "deviation_pp", "within_tolerance")},
                                      "budget": v["budget"], "swe_profile": v["swe_profile"]} for c, v in per_cap.items() if c != a.cap},
                   "sources": mix.source_table(units, cands.rows), "swe_profile": per_cap[a.cap]["swe_profile"],
                   "terminal_profile": terminal_profile(units, paths.RAW / "tgr_env" / "labels.jsonl"),
                   "budget_rule": {**budget_main, "throughput_source": a.throughput_source,
                                   "rule": mix.__doc__.split("The budget rule: ", 1)[1].strip()},
                   "predicted_hours": hours, "eval_overhead": evals, "file_checks": checks,
                   "split": {"manifest": rel(man_out / "split.json"), "validation_rows": len(orders["dev"]),
                             "validation_tasks": val.report["tasks_drawn"], "pilot_rows": len(orders["pilot"]),
                             "pilot_tasks": pilot.report["tasks_drawn"], "train_rows": len(orders["train"])},
                   "selection": rel(man_out / "selection.json"), "normalisation": mix.NORMALISATION_NOTE,
                   "freeze": {"command": "PYTHONPATH=src .venv/train/bin/python scripts/data/build_mixture.py freeze",
                              "requires": "p1_pools current, p2_rows over exactly these train/dev files with zero drops, a manual review"}}
        common.write_json(man_out / "mixture.json", mixture)
        run.log(f"wrote {man_out / 'mixture.json'} (CANDIDATE), selection.json, split.json; train {len(orders['train'])} / "
                f"dev {len(orders['dev'])} / pilot {len(orders['pilot'])} rows; predicted {hours}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(time.time() - t0, 1), status="finished",
                           mixture={"cap": a.cap, "shares": acct_parts["all"]["shares_by_loss_tokens"], "totals": acct_parts["all"]["totals"],
                                    "within_tolerance": acct_parts["all"]["within_tolerance"], "files": {k: v["sha256"] for k, v in file_block.items()}},
                           summary={"predicted_hours": hours, "train_rows": len(orders["train"]), "dev_rows": len(orders["dev"]),
                                    "pilot_rows": len(orders["pilot"])})
        run.finalize()
    return 0


# --- freeze -----------------------------------------------------------------------------------------------------------
def cmd_freeze(a: argparse.Namespace) -> int:
    run_id = a.run_id or common.new_run_id("data_mixture_freeze")
    man_out = a.manifests_out or a.manifests_dir
    path = man_out / "mixture.json"
    t0 = time.time()
    with common.RunDir(run_id, a.log_root, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="build_mixture freeze", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=a.seed, start_time=common.utc_now(), status="running",
                           config={"mixture": str(path)})
        m = common.read_json(path)
        problems = []
        if not m:
            raise SystemExit(f"no mixture manifest at {path}")
        if m.get("frozen"):
            raise SystemExit(f"{path} is already frozen (run {m.get('freeze', {}).get('run_id')}); a frozen set never changes")
        for k, f in m["files"].items():
            p = ROOT / f["path"] if not Path(f["path"]).is_absolute() else Path(f["path"])
            if not p.exists() or common.sha256_file(p) != f["sha256"]:
                problems.append(f"{k}: {f['path']} missing or changed since the build")
        files = sel.PoolFiles.under(a.pools_dir)
        p1 = common.read_json(a.manifests_dir / "screens_p1_pools.json")
        st = p1_status(p1, files)
        if not st["current"]:
            problems.append(f"p1_pools not current: {st['reason']}")
        elif common.sha256_file(a.manifests_dir / "screens_p1_pools.json") != m["screens"]["p1_pools"].get("sha256"):
            problems.append("screens_p1_pools.json changed since the build (re-run build)")
        p2 = common.read_json(a.manifests_dir / "screens_p2_rows.json")
        if not p2:
            problems.append("no screens_p2_rows.json: run the p2_rows pass on the train and dev files")
        else:
            screened = {f["sha256"] for inp in p2.get("inputs", {}).values() for f in inp.get("files", [])}
            for k in ("train_rows", "dev_rows"):
                if m["files"][k]["sha256"] not in screened:
                    problems.append(f"p2_rows did not screen the current {k} (sha256 not among its inputs)")
            if p2.get("dropped"):
                problems.append(f"p2_rows dropped {len(p2['dropped'])} rows: re-run build, then p2_rows")
            if "skipped" in (p2.get("counts", {}).get("d3") or {}):
                problems.append("p2_rows ran without the D3 sealed screen")
        if "skipped" in ((p1 or {}).get("counts", {}).get("d3") or {}):
            problems.append("p1_pools ran without the D3 sealed screen")
        if problems:
            for pr in problems:
                run.error(pr)
            run.write_metadata(end_time=common.utc_now(), status="refused", problems=problems)
            run.finalize()
            print("freeze refused:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        m.update({"status": "FROZEN", "frozen": True,
                  "freeze": {**m.get("freeze", {}), "run_id": run_id, "frozen_at": common.utc_now(),
                             "p2_rows": {"run_id": p2.get("run_id"), "sha256": common.sha256_file(a.manifests_dir / "screens_p2_rows.json")}}})
        common.write_json(path, m)
        run.log(f"{path} frozen: {json.dumps({k: v['sha256'] for k, v in m['files'].items()})}")
        resources(run, t0)
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(time.time() - t0, 1), status="finished",
                           mixture={k: v["sha256"] for k, v in m["files"].items()})
        run.finalize()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("items", "build", "freeze"))
    p.add_argument("--pools-dir", type=Path, default=paths.POOLS)
    p.add_argument("--manifests-dir", type=Path, default=paths.MANIFESTS, help="screens_*.json and dev_*.json are read here")
    p.add_argument("--manifests-out", type=Path, default=None, help="selection/split/mixture.json go here (default --manifests-dir)")
    p.add_argument("--out-dir", type=Path, default=None, help="items: WORK/<run_id>/screen_items; build: data/processed/sft")
    p.add_argument("--cap", choices=tuple(sel.CAPS), default="64k", help="the row cap of the files (M4.1: 64k)")
    p.add_argument("--throughput", type=float, default=mix.THROUGHPUT_TOKENS_PER_S, help="training tokens/s at the cap")
    p.add_argument("--throughput-source", default="measured on real 64K rows through apex_train.sft on GPU-T (M4.1's synthetic T7: 8.8K tokens/s)",
                   help="where --throughput comes from (recorded in mixture.json)")
    p.add_argument("--max-hours", type=float, default=mix.MAX_SFT_HOURS, help="SFT wall-clock budget of the budget rule")
    p.add_argument("--val-fraction", type=float, default=spl.VAL_FRACTION)
    p.add_argument("--pilot-fraction", type=float, default=spl.PILOT_FRACTION)
    p.add_argument("--retention-floor", type=float, default=mix.RETENTION_FLOOR)
    p.add_argument("--no-dev", action="store_true", help="items: skip the dev items")
    p.add_argument("--allow-unscreened", action="store_true", help="DEVELOPMENT ONLY: build without a current p1_pools pass")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-root", type=Path, default=None, help="TESTS ONLY: write logs/<run_id>/ under this root")
    a = p.parse_args(argv)
    if a.command == "build" and a.out_dir is None:
        a.out_dir = paths.PROCESSED / "sft"
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return {"items": cmd_items, "build": cmd_build, "freeze": cmd_freeze}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
