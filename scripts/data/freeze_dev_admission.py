#!/usr/bin/env python3
"""Freeze the admitted SWE and terminal dev task lists (src/apex_data/devset/admission.py) once, before Base's full dev run.

Usage (from the repository root):
  PYTHONPATH=src .venv/train/bin/python scripts/data/freeze_dev_admission.py            # write data/manifests/dev_admitted.json
  PYTHONPATH=src .venv/train/bin/python scripts/data/freeze_dev_admission.py --check    # compare the pools with the frozen file

Inputs : data/manifests/dev_{swe,terminal}.json (the frozen dev set), data/pools/swesmith/tasks.jsonl (gold-replay
         calibration per task) and data/pools/tgr_env/tasks.jsonl (nop / oracle / image-scan checks per task).
Outputs: data/manifests/dev_admitted.json (admitted task ids per slice in manifest order, excluded tasks with their
         reasons, admitted_sha256, input sha256s), written once: the script refuses to overwrite it and refuses while any
         dev task has no decision yet; logs/<run_id>/ with the five run files (run_id data_dev_admission_<UTC time>).
         --check writes nothing but the logs and exits 1 when the pools disagree with the frozen file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from apex_data import paths  # noqa: E402
from apex_data.devset import admission  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, sha256_file, utc_now  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests-dir", type=Path, default=paths.MANIFESTS)
    p.add_argument("--swe-pool", type=Path, default=admission.DEFAULT_POOLS["swe"])
    p.add_argument("--terminal-pool", type=Path, default=admission.DEFAULT_POOLS["terminal"])
    p.add_argument("--out", type=Path, default=None, help="default <manifests-dir>/dev_admitted.json")
    p.add_argument("--check", action="store_true", help="verify the pools against the frozen file; write nothing")
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-root", type=Path, default=REPO, help="SMOKE TESTS ONLY: write logs/<run_id>/ under this root")
    a = p.parse_args(argv)
    out = a.out or a.manifests_dir / admission.ADMITTED_FILE
    run_id = a.run_id or new_run_id("data_dev_admission")
    pools = {"swe": a.swe_pool, "terminal": a.terminal_pool}
    inputs = {str(f): sha256_file(f) for f in (a.manifests_dir / "dev_swe.json", a.manifests_dir / "dev_terminal.json", *pools.values())
              if f.exists()}
    with RunDir(run_id, a.log_root, log_name="eval.log") as run:
        run.write_metadata(run_id=run_id, script="scripts/data/freeze_dev_admission.py", mode="check" if a.check else "freeze",
                           code=git_info(), hardware=hardware_info(), command=command_line(), config={"out": str(out), "pools": {k: str(v) for k, v in pools.items()}},
                           data_version={"inputs": inputs}, seed=None, start_time=utc_now(), status="running")
        decisions = admission.decide(a.manifests_dir, pools)
        for sl, d in decisions.items():
            run.log(f"{sl}: {d['n_manifest']} dev tasks: {len(d['admitted'])} admitted, {len(d['excluded'])} excluded, {len(d['pending'])} pending")
            for e in d["excluded"]:
                run.log(f"  excluded {e['task_id']}: {e['reason']}")
        rc = 0
        if a.check:
            frozen = admission.load(out)
            problems = ["no frozen file at " + str(out)] if frozen is None else admission.verify(frozen, decisions)
            for msg in problems:
                run.error(msg)
            run.log(f"check against {out}: {'OK' if not problems else 'MISMATCH'}")
            rc = 1 if problems else 0
            result = {"check": "ok" if not problems else "mismatch", "problems": problems}
        else:
            if out.exists():
                run.error(f"{out} exists; the admitted lists are frozen once and never rewritten (use --check)")
                raise SystemExit(f"{out} exists; the admitted lists are frozen once and never rewritten (use --check)")
            try:
                record = admission.freeze_record(decisions, inputs, utc_now())
            except ValueError as e:
                run.error(str(e))
                raise SystemExit(str(e)) from None
            record["run_id"] = run_id
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=1) + "\n")
            run.log(f"froze {out}: admitted_sha256 {record['admitted_sha256']}; "
                    + ", ".join(f"{sl} {d['n_admitted']}/{d['n_manifest']}" for sl, d in record["slices"].items()))
            result = {"out": str(out), "sha256": sha256_file(out), "admitted_sha256": record["admitted_sha256"]}
        elapsed = run.elapsed()
        run.write_resources([("duration_sec", round(elapsed, 1), "s", "wall clock (CPU only)"), ("gpu_hours", 0, "h", "no GPU used"),
                             ("peak_gpu_memory_gb", "n/a", "GB", "no GPU used"), ("cost_usd", 0.0, "USD", "GPU cost only; CPU-only step, the always-on CPU VM is not charged")])
        run.write_metadata(result=result, end_time=utc_now(), duration_sec=round(elapsed, 1), status="finished")
        run.finalize()
    return rc


if __name__ == "__main__":
    sys.exit(main())
