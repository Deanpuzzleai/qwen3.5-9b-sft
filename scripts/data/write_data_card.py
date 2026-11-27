#!/usr/bin/env python
"""Generate the data card `data/README.md` from the manifests (WP7, unit U2.4; docs/DATA_PLAN.md revision 4.3 sections 8
and 14).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/write_data_card.py [--manifests-dir data/manifests] \\
        [--pools-dir data/pools] [--out data/README.md] [--run-id ID]

Inputs   <manifests-dir>/{mixture,selection,split,screens_p1_pools,screens_p2_rows,raw_sources,decontam_exclusions}.json,
         <pools-dir>/{swe_teacher,terminal_teacher,retention}/manifest.json, <pools-dir>/terminal_teacher/admission_report.json,
         configs/eval/swebench_pro_v2_hard.yaml (the public language profile of the 51 HARD tasks). Missing inputs are
         stated in the card.
Outputs  the Markdown card at --out (sources, licences and attribution; mixture at the cap and at 48K; budget rule;
         per-provenance counts; per-slice funnels; SWE strata, languages and HARD-51's profile; terminal coverage; the
         TerminalHorizon admission failure and fallback; dedup and decontamination counts; validation split and pilot;
         exclusions; limitations; reproduction), logs/<run_id>/ (metadata.json, train.log, metrics.jsonl, errors.log,
         resources.csv). Deterministic: the same manifests give the same card.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import cards, paths  # noqa: E402
from apex_eval import common  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests-dir", type=Path, default=paths.MANIFESTS)
    p.add_argument("--pools-dir", type=Path, default=paths.POOLS)
    p.add_argument("--out", type=Path, default=paths.DATA / "README.md")
    p.add_argument("--run-id", default=None, help="default data_card_<UTC timestamp>")
    p.add_argument("--log-root", type=Path, default=None, help="TESTS ONLY: write logs/<run_id>/ under this root")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or common.new_run_id("data_card")
    t0 = time.time()
    with common.RunDir(run_id, a.log_root, log_name="train.log") as run:
        inputs = {p.name: common.sha256_file(p) for p in sorted(a.manifests_dir.glob("*.json"))}
        inputs.update({f"pools/{n}/manifest.json": common.sha256_file(a.pools_dir / n / "manifest.json")
                       for n in ("swe_teacher", "terminal_teacher", "retention") if (a.pools_dir / n / "manifest.json").exists()})
        run.write_metadata(run_id=run_id, step="write_data_card", command=common.command_line(), code=common.git_info(),
                           hardware=common.hardware_info(), seed=paths.SEED, start_time=common.utc_now(), status="running",
                           config={"manifests_dir": str(a.manifests_dir), "pools_dir": str(a.pools_dir), "out": str(a.out)},
                           data_version=inputs)
        out = cards.write_card(a.out, a.manifests_dir, a.pools_dir)
        text = out.read_text(encoding="utf-8")
        run.log(f"data card -> {out} ({len(text.splitlines())} lines, sha256 {common.sha256_file(out)[:16]})")
        run.metric({"step": "card", "lines": len(text.splitlines()), "bytes": out.stat().st_size})
        duration = time.time() - t0
        run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock"), ("orchestrator_cpus", os.cpu_count(), "vcpu", "CPU only"),
                             ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)")])
        run.write_metadata(end_time=common.utc_now(), duration_sec=round(duration, 1), status="finished",
                           summary={"card": str(out), "sha256": common.sha256_file(out)})
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
