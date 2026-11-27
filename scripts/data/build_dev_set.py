#!/usr/bin/env python
"""Build and freeze the dev set (WP6; docs/DATA_PLAN.md section 10): SWE, terminal and STEM QA manifests, the exclusion
file every training and stage-2 pool builder reads, and the freeze record.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_dev_set.py [--slices swe,terminal,qa] [--freeze]

Inputs
  RAW/swesmith_py, RAW/swesmith_go      SWE-bench/SWE-smith-{py,go} parquet files (scripts/data/fetch_raw.py)
  RAW/tgr_env/{tasks,labels}.jsonl      ai-and/termgrade-environments task index and per-model pass rates
  RAW/tgr/data/trials.parquet           ai-and/termgrade-trajectories trial table (teacher passes lost to the dev set)
  data/manifests/raw_sources.json       pinned revisions of the above (recorded in the manifests)
  data/manifests/decontam_exclusions.json, configs/eval/swebench_pro_v2_hard.yaml (repository names; read-only)
  GitHub API (GITHUB_TOKEN from the environment or .env) for the licence and fork parent of the SWE repositories
      considered; answers cached under WORK/licence_cache/
  Idavidrein/gpqa gpqa_diamond.csv (gated; HF_TOKEN from .env) and math-ai/aime25 test.jsonl at pinned revisions,
      downloaded into WORK/dev_qa_cache/ (never into the repository); the reasoning-gym package (.venv/train)
Outputs (in --out-dir, default data/manifests/)
  dev_swe.json, dev_terminal.json, dev_qa.json   items, per-item sample indices and seeds (full and reduced set),
                                                 inputs with sha256, survival tables, draw reports; frozen: true
  dev_exclusions.json                            SWE dev repositories and task ids, terminal task ids, QA ids (--freeze)
  dev_freeze.json                                sha256 and counts of the four files above, date, run id (--freeze)
  logs/<run_id>/                                 metadata.json, train.log, metrics.jsonl, errors.log, resources.csv;
                                                 also the D1/D4 screen (apex_data.screens, WP2) of the 40 terminal dev
                                                 task directories when both exist (a hit blocks the freeze)
Freeze rule. Once dev_freeze.json exists with frozen: true, nothing in --out-dir is ever written again: every run
rebuilds the requested slices in memory and verifies that they are byte-identical to the frozen files (exit 0), or
stops with an error naming the file that differs (exit 1). Without a freeze, the built manifests are written; --freeze
(all three slices in the same run) also writes the exclusion file and the freeze record.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.devset import common, qa, swe, terminal  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

SLICES = ("swe", "terminal", "qa")
PINNED = ("swesmith_py", "swesmith_go", "tgr_env", "tgr")     # raw_sources.json keys recorded in the manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slices", default=",".join(SLICES), help="comma-separated subset of swe,terminal,qa (default: all)")
    p.add_argument("--freeze", action="store_true", help="write dev_exclusions.json and dev_freeze.json (needs all slices)")
    p.add_argument("--seed", type=int, default=paths.SEED, help=f"seed of every draw and sample (default {paths.SEED})")
    p.add_argument("--out-dir", type=Path, default=paths.MANIFESTS, help="manifest directory (default data/manifests)")
    p.add_argument("--raw-root", type=Path, default=paths.RAW, help="root of the SWE-smith raw folders (default RAW)")
    p.add_argument("--tgr-env-dir", type=Path, default=terminal.DEFAULT_ENV_DIR,
                   help="folder holding tasks.jsonl and labels.jsonl (default RAW/tgr_env)")
    p.add_argument("--tgr-trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet",
                   help="TermGrade trials.parquet (default RAW/tgr/data/trials.parquet)")
    p.add_argument("--qa-cache", type=Path, default=qa.CACHE_DIR,
                   help="GPQA / AIME download cache (default WORK/dev_qa_cache)")
    p.add_argument("--licence-cache", type=Path, default=swe.LICENCE_CACHE,
                   help="GitHub metadata cache (default WORK/licence_cache)")
    p.add_argument("--offline", action="store_true", help="no network: QA files and GitHub answers from the caches only")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES,
                   help="pinned revisions (default data/manifests/raw_sources.json)")
    p.add_argument("--log-root", type=Path, default=None,
                   help="write logs/<run_id>/ under this root (tests; default the repository)")
    p.add_argument("--run-id", default=None, help="logs/<run_id>/ (default data_build_dev_set_<UTC timestamp>)")
    a = p.parse_args(argv)
    a.slices = [s.strip() for s in a.slices.split(",") if s.strip()]
    bad = [s for s in a.slices if s not in SLICES]
    if bad:
        p.error(f"unknown slices {bad}")
    if a.freeze and set(a.slices) != set(SLICES):
        p.error("--freeze needs all three slices in the same run")
    return a


def raw_revisions(path: Path) -> dict[str, str | None]:
    """Source folder name -> pinned revision from raw_sources.json (empty when the file is absent)."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    d = d.get("sources", d)
    return {k: (v.get("revision") if isinstance(v, dict) else None) for k, v in d.items()}


def build_slice(name: str, a: argparse.Namespace, revisions: dict[str, str | None], run: RunDir) -> dict[str, Any]:
    if name == "swe":
        meta = swe.RepoMeta(a.licence_cache, token=None if a.offline else swe.github_token(), offline=a.offline)
        m = swe.build(a.raw_root, a.seed, meta, revisions, log=run.log)
        run.log(f"dev_swe: GitHub API calls {meta.api_calls} (the rest from {a.licence_cache})")
        return m
    if name == "terminal":
        return terminal.build(a.tgr_env_dir, a.tgr_trials, a.seed, revisions, log=run.log)
    manifest, _items = qa.build(None if a.offline else qa.hf_token(), a.seed, a.qa_cache, a.offline, log=run.log)
    return manifest


def screen_terminal_dev(m: dict[str, Any], env_dir: Path, run: RunDir) -> dict[str, Any] | None:
    """D1 and D4 of the WP2 screens (apex_data.screens.d4_structural) over the drawn task directories. The result goes to
    the run's metrics and metadata, never into the manifest; None when the screens or the task directories are absent."""
    try:
        from apex_data.screens import d4_structural as d4
    except ImportError as exc:
        run.log(f"D1/D4 over the terminal dev tasks not run: screens not importable ({exc})")
        return None
    dirs = [(it["upstream_id"], env_dir / it["task_dir"]) for it in m["items"]]
    if not all(d.is_dir() for _, d in dirs):
        run.log(f"D1/D4 over the terminal dev tasks not run: task directories missing under {env_dir}")
        return None
    patterns = d4.slug_patterns(d4.load_tb_slugs())
    hits = {}
    for name, d in dirs:
        r = d4.screen_task_dir(d, patterns, name)
        if r.d1_hit or r.d4_hit:
            hits[name] = r.reasons()
    out = {"tasks_screened": len(dirs), "tasks_with_hits": len(hits), "hits": hits}
    run.metric({"screen": "d1_d4_terminal_dev", **out})
    run.log(f"D1/D4 over the {len(dirs)} terminal dev task directories: {len(hits)} with hits {hits or ''}")
    return out


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_build_dev_set")
    out_dir = a.out_dir.resolve()
    freeze = common.read_freeze(out_dir)
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log") as run:
        revisions = raw_revisions(a.raw_sources)
        unpinned = [k for k in PINNED if not revisions.get(k)]
        if a.freeze and unpinned:
            raise SystemExit(f"cannot freeze: no pinned revision for {unpinned} in {a.raw_sources}")
        run.write_metadata(run_id=run_id, step="data_build_dev_set", command=command_line(), git=git_info(),
                           code_commit=git_info()["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={k: revisions.get(k) for k in PINNED},
                           model_revision=None, hardware=hardware_info(), seed=a.seed, start_time=utc_now(),
                           status="running", frozen_before_run=bool(freeze))
        built: dict[str, dict[str, Any]] = {}
        texts: dict[str, str] = {}
        for name in a.slices:
            built[name] = build_slice(name, a, revisions, run)
            texts[common.MANIFEST_NAMES[name]] = common.dumps(built[name])
            run.metric({"slice": name, **built[name]["counts"]})
        screen = screen_terminal_dev(built["terminal"], a.tgr_env_dir, run) if "terminal" in built else None
        if screen and screen["tasks_with_hits"]:
            run.error(f"terminal dev tasks hit D1/D4: {screen['hits']}; the dev set cannot be frozen with them")
            rc = 1
        if set(built) == set(SLICES):
            texts[common.EXCLUSIONS_NAME] = common.dumps(
                common.exclusions_record(built["swe"], built["terminal"], built["qa"]))
        if freeze:
            outcome = "verified identical to the frozen dev set; nothing written"
            for fname, text in texts.items():
                try:
                    sha = common.check_frozen(out_dir, fname, text, freeze)
                    run.log(f"{fname}: rebuilt byte-identical to the frozen file (sha256 {sha[:16]})")
                except common.FrozenManifestError as exc:
                    run.error(str(exc))
                    run.log(f"VERIFY FAILED: {exc}")
                    outcome, rc = "rebuild differs from the frozen dev set; nothing written", 1
        else:
            for fname, text in texts.items():
                if fname == common.EXCLUSIONS_NAME and not a.freeze:
                    continue
                common.write_atomic(out_dir / fname, text)
                run.log(f"wrote {out_dir / fname} (sha256 {common.sha256_bytes(text.encode())[:16]})")
            outcome = "manifests written (not frozen)"
            if a.freeze and rc:
                outcome = "not frozen: a terminal dev task hit D1/D4 (errors.log)"
            elif a.freeze:
                record = {
                    "frozen": True, "frozen_at": utc_now(), "run_id": run_id, "code": git_info(), "seed": a.seed,
                    "files": {f: {"sha256": common.sha256_bytes(t.encode()), "bytes": len(t.encode())}
                              for f, t in texts.items()},
                    "counts": {n: built[n]["counts"] for n in SLICES},
                    "sample_plan": {"full_k": common.FULL_K, "reduced_k": common.REDUCED_K,
                                    "reduced_rule": "the reduced set is the first reduced_k samples (index < reduced_k) "
                                                    "of every scored item, with the full set's seeds; AIME is not in it"},
                    "verify": "PYTHONPATH=src .venv/train/bin/python scripts/data/build_dev_set.py  (rebuilds and checks "
                              "byte identity; never rewrites a frozen file)",
                    "note": "Frozen before any training row is drawn (docs/DATA_PLAN.md section 10). Frozen files are "
                            "never edited; a change needs a new dev-set version recorded in docs/DATA_PLAN_CHANGELOG.md.",
                }
                common.write_atomic(out_dir / common.FREEZE_NAME, common.dumps(record))
                run.log(f"frozen: wrote {out_dir / common.FREEZE_NAME}")
                outcome = "dev set frozen"
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1),
                           status="finished" if rc == 0 else "failed", outcome=outcome,
                           summary={n: built[n]["counts"] for n in built}, terminal_dev_screen=screen,
                           sha256={f: common.sha256_bytes(t.encode()) for f, t in texts.items()})
        run.finalize()
        run.log(f"done: {outcome}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
