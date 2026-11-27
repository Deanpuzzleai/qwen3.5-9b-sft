#!/usr/bin/env python
"""Merge a prefix of the retention top-up rows into a copy of the retention pool, so that the mixture's retention slice
grows while the pool's own rows and its replay reserve stay as they are (docs/DATA_PLAN.md sections 3.4 and 4; the
top-up rows come from `scripts/data/build_retention_slice.py --topup`).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/merge_retention_topup.py --topup-dir DIR --k K \\
        [--exclude DROPS.json ...] --out-dir DIR [--pool-dir data/pools/retention] [--run-id ID]
then point scripts/data/build_mixture.py (items, build) at a pools directory whose retention/ is --out-dir.

Inputs
  --pool-dir    the retention pool: rows.parquet, replay_reserve.parquet, manifest.json (read only; their sha256 must
                equal what the pool manifest and the top-up manifest record)
  --topup-dir   topup_rows.parquet (the pool's columns + topup_rank, sorted by it) and topup_manifest.json
  --k           take the top-up rows with topup_rank < K; the top-up's interleaved order keeps the part and stratum
                proportions of every prefix (topup_manifest.json "prefix")
  --exclude     JSON file of screen drops on top-up rows (repeatable): `dropped_topup_task_ids` {task_id: rules} or a
                run_screens manifest's `dropped` {task_id or row:<sample_id>: rules}. Matching rows inside the prefix are
                left out (the prefix is not extended to make up for them); ids naming no top-up row are counted and ignored
Outputs (--out-dir, never the pool directory)
  rows.parquet            the pool's rows in their order, then the kept top-up rows in topup_rank order; the pool's
                          columns and Arrow schema (topup_rank is recorded in the manifest, not in the rows)
  replay_reserve.parquet  the pool's stage-2 reserve, copied byte for byte
  manifest.json           the pool manifest with counts.sft, tokens.sft, files and each part's funnel updated, plus
                          `topup_merge`: run id, K, the top-up run and file sha256s, the kept rows (row_key, task_id, part,
                          topup_rank, row sha256), the excluded rows with their rules, per-part counts, the merge checks
  logs/<run_id>/          metadata.json, train.log, metrics.jsonl, errors.log, resources.csv
Checks (exit 1 when one fails; nothing is written then): the pool files are the ones both manifests record; the top-up
file is the one its manifest records; the top-up columns equal the pool's (+ topup_rank); no task_id, row_key or
prompt_sha256 shared between the kept top-up rows and the pool's rows or reserve, none repeated among the kept rows;
schema.validate_provenance on every merged row; n_assistant_tokens equals the assistant-mask sum of every added row.
Deterministic: the same inputs give byte-identical rows files.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths, schema  # noqa: E402
from apex_eval import common  # noqa: E402

PARTS = ("math", "if", "chat", "sci_mcq", "sci_rqa")
RANK = "topup_rank"


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def excluded_ids(files: list[Path]) -> dict[str, list[str]]:
    """{task or sample id: rules} from screen-drop files (see --exclude)."""
    out: dict[str, set[str]] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        drops = d.get("dropped_topup_task_ids") if "dropped_topup_task_ids" in d else d.get("dropped")
        if not isinstance(drops, dict):
            raise SystemExit(f"{f}: neither `dropped_topup_task_ids` nor `dropped` holds an {{id: rules}} map")
        for i, rules in drops.items():
            out.setdefault(i.split(":", 1)[1] if i.startswith("row:") else i, set()).update(rules)
    return {i: sorted(r) for i, r in sorted(out.items())}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The pool manifest's `tokens.sft` block for `rows`."""
    reasoning = sorted(int(r["reasoning_tokens"]) for r in rows)
    n = len(reasoning)
    median = (reasoning[n // 2] if n % 2 else (reasoning[n // 2 - 1] + reasoning[n // 2]) / 2) if n else 0
    seq = sum(int(r["n_tokens"]) for r in rows)
    return {"rows": len(rows), "sequence_tokens": seq, "loss_tokens": sum(int(r["n_assistant_tokens"]) for r in rows),
            "mean_sequence_tokens": round(seq / len(rows), 1) if rows else 0.0, "median_reasoning_tokens": float(median),
            "max_reasoning_tokens": max(reasoning, default=0), "max_row_tokens": max((int(r["n_tokens"]) for r in rows), default=0)}


def merge(a: argparse.Namespace, run: common.RunDir) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pool_files = {"rows": a.pool_dir / "rows.parquet", "replay_reserve": a.pool_dir / "replay_reserve.parquet",
                  "manifest": a.pool_dir / "manifest.json"}
    topup_rows_path, topup_manifest_path = a.topup_dir / "topup_rows.parquet", a.topup_dir / "topup_manifest.json"
    pm = json.loads(pool_files["manifest"].read_text(encoding="utf-8"))
    tm = json.loads(topup_manifest_path.read_text(encoding="utf-8"))
    sha = {k: common.sha256_file(p) for k, p in pool_files.items()}
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        run.log(f"[check] {'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail, default=str)[:400]}")

    check("pool_files_as_recorded", sha["rows"] == pm["files"]["rows"]["sha256"]
          and sha["replay_reserve"] == pm["files"]["replay_reserve"]["sha256"] and sha == tm["exclude_pool"]["sha256"],
          {"now": sha, "pool_manifest": {k: pm["files"][k]["sha256"] for k in ("rows", "replay_reserve")},
           "topup_manifest": tm["exclude_pool"]["sha256"]})
    topup_sha = common.sha256_file(topup_rows_path)
    check("topup_file_as_recorded", topup_sha == tm["files"]["topup_rows"]["sha256"], {"now": topup_sha})

    pool = pq.read_table(pool_files["rows"])
    topup = pq.read_table(topup_rows_path)
    check("columns_match", topup.schema.names == [*pool.schema.names, RANK]
          and all(topup.schema.field(n).type == pool.schema.field(n).type for n in pool.schema.names),
          {"extra": [n for n in topup.schema.names if n not in pool.schema.names],
           "missing": [n for n in pool.schema.names if n not in topup.schema.names]})
    ranks = topup[RANK].to_pylist()
    check("topup_sorted_by_rank", ranks == list(range(len(ranks))), {"rows": len(ranks)})
    if not all(c["pass"] for c in checks):
        return {"checks": checks}

    if not 0 < a.k <= topup.num_rows:
        raise SystemExit(f"--k must be in 1..{topup.num_rows}")
    prefix = topup.filter(pc.less(topup[RANK], a.k))
    meta_cols = [n for n in prefix.schema.names if n not in ("input_ids", "assistant_masks")]
    meta = prefix.select(meta_cols).to_pylist()
    drops = excluded_ids(a.exclude)
    hit = {i for i in drops if any(i in (m["task_id"], m["sample_id"]) for m in meta)}
    all_topup_ids = {x for m in topup.select(["task_id", "sample_id"]).to_pylist() for x in m.values()}
    keep_mask = [m["task_id"] not in drops and m["sample_id"] not in drops for m in meta]
    kept = prefix.filter(pa.array(keep_mask))
    kept_meta = [m for m, k in zip(meta, keep_mask) if k]
    excluded = [{"row_key": m["row_key"], "task_id": m["task_id"], "part": m["retention_part"], RANK: m[RANK],
                 "rules": drops.get(m["task_id"]) or drops.get(m["sample_id"])} for m, k in zip(meta, keep_mask) if not k]
    run.log(f"prefix rank < {a.k}: {prefix.num_rows} rows; excluded {len(excluded)} (screen drops inside the prefix); "
            f"drop ids outside the prefix {len((set(drops) & all_topup_ids) - hit)}, naming no top-up row {len(set(drops) - all_topup_ids)}")

    reserve = pq.read_table(pool_files["replay_reserve"], columns=["task_id", "row_key", "prompt_sha256"]).to_pylist()
    pool_meta = pool.select([n for n in pool.schema.names if n not in ("input_ids", "assistant_masks")]).to_pylist()
    shared = {key: len({str(m[key]) for m in kept_meta} & {str(r[key]) for r in (*pool_meta, *reserve)})
              for key in ("task_id", "row_key", "prompt_sha256")}
    repeated = {key: len(kept_meta) - len({str(m[key]) for m in kept_meta}) for key in ("task_id", "row_key", "prompt_sha256")}
    check("disjoint_from_pool_and_reserve", not any(shared.values()) and not any(repeated.values()),
          {"shared": shared, "repeated_among_added": repeated})
    bad_prov = []
    for r in (*pool_meta, *kept_meta):
        try:
            schema.validate_provenance(r)
        except schema.SchemaError as exc:
            bad_prov.append(f"{r['row_key']}: {exc}")
    check("provenance_valid", not bad_prov, bad_prov[:5])
    sums = [sum(m) for m in kept["assistant_masks"].to_pylist()]
    bad_loss = [m["row_key"] for m, s in zip(kept_meta, sums) if int(m["n_assistant_tokens"]) != s]
    check("loss_tokens_equal_mask_sums", not bad_loss, bad_loss[:5])
    if not all(c["pass"] for c in checks):
        return {"checks": checks}

    merged = pa.concat_tables([pool, kept.drop_columns([RANK]).cast(pool.schema)])
    merged_meta = [*pool_meta, *[{k: v for k, v in m.items() if k != RANK} for m in kept_meta]]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"rows": a.out_dir / "rows.parquet", "replay_reserve": a.out_dir / "replay_reserve.parquet", "manifest": a.out_dir / "manifest.json"}
    tmp = out["rows"].with_suffix(".tmp.parquet")
    pq.write_table(merged, tmp)
    os.replace(tmp, out["rows"])
    tmp = out["replay_reserve"].with_suffix(".tmp.parquet")
    shutil.copyfile(pool_files["replay_reserve"], tmp)
    os.replace(tmp, out["replay_reserve"])

    pool_counts = Counter(str(m["retention_part"]) for m in pool_meta)
    added = Counter(str(m["retention_part"]) for m in kept_meta)
    m = json.loads(json.dumps(pm))
    m["counts"]["sft"] = {p: pool_counts[p] + added[p] for p in PARTS}
    m["counts"]["sft_total"] = len(merged_meta)
    m["tokens"]["sft"] = summary(merged_meta)
    for p in PARTS:
        part = m.get("parts", {}).get(p)
        if part is not None and isinstance(part.get("funnel"), list):
            part["funnel"].append({"stage": "topup_merge", "n_in": pool_counts[p], "n_added": added[p], "n_out": pool_counts[p] + added[p],
                                   "note": f"top-up rows with topup_rank < {a.k} of run {tm['run_id']}, screen drops excluded"})
    m["files"]["rows"] = {**pm["files"]["rows"], "path": rel(out["rows"]), "rows": merged.num_rows, "bytes": out["rows"].stat().st_size,
                          "sha256": common.sha256_file(out["rows"])}
    m["files"]["replay_reserve"] = {**pm["files"]["replay_reserve"], "path": rel(out["replay_reserve"])}
    m["topup_merge"] = {
        "run_id": run.run_id, "created": common.utc_now(), "code": common.git_info(), "k": a.k,
        "rule": "pool rows (unchanged, in their order), then the top-up rows with topup_rank < k in rank order, minus the screen drops "
                "listed in `excluded`; the prefix keeps the top-up's part and stratum proportions",
        "pool": {"dir": rel(a.pool_dir), "run_id": pm.get("run_id"), "version": pm.get("version"), "rows": len(pool_meta),
                 "sha256": sha, "counts_sft": dict(pm["counts"]["sft"]), "tokens_sft": pm["tokens"]["sft"]},
        "topup": {"dir": rel(a.topup_dir), "run_id": tm["run_id"], "version": tm.get("version"), "rows": topup.num_rows,
                  "rows_sha256": topup_sha, "manifest_sha256": common.sha256_file(topup_manifest_path)},
        "exclude_files": {rel(f): common.sha256_file(f) for f in a.exclude},
        "prefix_rows": prefix.num_rows, "added_rows": len(kept_meta), "added_by_part": {p: added[p] for p in PARTS},
        "added_tokens": summary(kept_meta) if kept_meta else {},
        "added": [{"row_key": r["row_key"], "task_id": r["task_id"], "part": r["retention_part"], RANK: r[RANK], "sha256": r["sha256"]}
                  for r in kept_meta],
        "excluded": excluded,
        "drop_ids_outside_prefix": len((set(drops) & all_topup_ids) - hit), "drop_ids_naming_no_topup_row": len(set(drops) - all_topup_ids),
        "checks": checks}
    common.write_json(out["manifest"], m)
    return {"checks": checks, "rows": merged.num_rows, "added": len(kept_meta), "excluded": len(excluded),
            "added_by_part": {p: added[p] for p in PARTS}, "sha256": {k: common.sha256_file(p) for k, p in out.items()},
            "tokens_sft": m["tokens"]["sft"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool-dir", type=Path, default=paths.POOLS / "retention", help="the retention pool (read only)")
    p.add_argument("--topup-dir", type=Path, required=True, help="build_retention_slice.py --topup output directory")
    p.add_argument("--k", type=int, required=True, help="take the top-up rows with topup_rank < K")
    p.add_argument("--exclude", type=Path, action="append", default=[], help="screen-drop JSON on top-up rows (repeatable)")
    p.add_argument("--out-dir", type=Path, required=True, help="the merged pool directory (not --pool-dir)")
    p.add_argument("--run-id", default=None, help="default data_retention_merge_<UTC>")
    p.add_argument("--log-root", type=Path, default=None, help="TESTS ONLY: write logs/<run_id>/ under this root")
    a = p.parse_args(argv)
    if a.out_dir.resolve() == a.pool_dir.resolve():
        p.error("--out-dir must not be the pool directory (the pool's own files are never rewritten)")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or common.new_run_id("data_retention_merge")
    t0 = time.time()
    with common.RunDir(run_id, a.log_root, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="merge_retention_topup", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=paths.SEED, start_time=common.utc_now(), status="running",
                           config={"pool_dir": str(a.pool_dir), "topup_dir": str(a.topup_dir), "k": a.k,
                                   "exclude": [str(f) for f in a.exclude], "out_dir": str(a.out_dir)})
        res = merge(a, run)
        failed = [c["check"] for c in res["checks"] if not c["pass"]]
        for c in failed:
            run.error(f"check failed: {c}")
        run.metric({"step": "merge", **{k: v for k, v in res.items() if k in ("rows", "added", "excluded")}, "checks_failed": len(failed)})
        duration = time.time() - t0
        run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock on the CPU VM"),
                             ("orchestrator_cpus", os.cpu_count(), "vcpu", "apex-cpu, one process"),
                             ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
                             ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)")])
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(duration, 1), status="failed" if failed else "finished",
                           summary={k: v for k, v in res.items() if k != "checks"}, checks_failed=failed)
        run.log(f"done: {json.dumps({k: v for k, v in res.items() if k != 'checks'})}; checks failed: {failed or 'none'}")
        run.finalize()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
