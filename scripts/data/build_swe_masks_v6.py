#!/usr/bin/env python
"""The v6 SWE slice: the frozen v5 SWE episode rows with the v6 turn masks (data/manifests/v6/DATA_CARD_v6.md;
S1 read masking, S2 unverified submits; apex_data.render.swe_masks). Same episodes and token ids, no re-selection.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_swe_masks_v6.py [--in-dir data/pools/swe_v5]
        [--out-dir data/pools/swe_v6] [--split data/manifests/v5/split.json] [--run-id ID]

Inputs (read-only): <in-dir>/{rows.parquet, manifest.json, selection.json, supply.json} (scripts/data/build_swe_slice.py),
  --split (the v5 validation row keys: the report gives train and dev separately), the Qwen3.5-9B tokenizer.
Outputs:
  <out-dir>/rows.parquet      the v5 rows, columns unchanged, with assistant_masks / n_assistant_tokens /
                              n_assistant_turns after masking, plus turns_read_masked, submit_masked, turn_kinds
                              (one letter per assistant turn: r read, e edit, u run, s submit, o other; upper case = masked)
  <out-dir>/mask_report.json  turn kinds before and after, the reasons, loss tokens, thinking per trained turn
                              (all, train, dev, per band), the band limits over the trained turns, the checks
  <out-dir>/manifest.json     the v5 manifest plus `v6_masks` (rules, report, input and output sha256)
  <out-dir>/{selection,supply}.json   copied from <in-dir> (the episodes are the same)
  logs/<run_id>/              metadata.json, train.log, metrics.jsonl, errors.log, resources.csv
Checks (exit 1 on failure): token ids identical; the new mask a subset of the old, by whole trained spans; every row
keeps a trained turn; every trained span ends with <|im_end|>.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from apex_data import efficiency, paths  # noqa: E402
from apex_data.render import swe_masks as sm  # noqa: E402
from apex_eval import common  # noqa: E402
from apex_train.data_format import load_tokenizer  # noqa: E402

LETTER = {"read": "r", "edit": "e", "run": "u", "submit": "s", "other": "o"}
EXTRA = (("turns_read_masked", pa.int32()), ("submit_masked", pa.bool_()), ("turn_kinds", pa.string()))


def limit_failures(thinking: list[int], band: str) -> list[str]:
    """SWE band limits over the trained turns (data plan 1.3 per trained turn)."""
    lim = efficiency.SWE_LIMITS[band]
    if not thinking or min(thinking) < 0:
        return ["unclosed_or_missing_reasoning"]
    mean = sum(thinking) / len(thinking)
    return (["turn_ceiling"] if max(thinking) > lim.turn_ceiling else []) + ([] if lim.mean_min <= mean <= lim.mean_max else ["mean_window"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=paths.POOLS / "swe_v5")
    p.add_argument("--out-dir", type=Path, default=paths.POOLS / "swe_v6")
    p.add_argument("--split", type=Path, default=paths.MANIFESTS / "v5" / "split.json")
    p.add_argument("--run-id", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or common.new_run_id("data_swe_masks_v6")
    t0 = time.time()
    with common.RunDir(run_id, log_name="train.log") as run:
        src = a.in_dir / "rows.parquet"
        run.write_metadata(run_id=run_id, step="data_swe_masks_v6", command=common.command_line(), git=common.git_info(),
                           hardware=common.hardware_info(), start_time=common.utc_now(), status="running", seed=None,
                           config={k: str(v) for k, v in vars(a).items()}, rules={k: v for k, v in sm.rules().items() if k != "doc"},
                           data_versions={"rows": {"path": str(src), "sha256": common.sha256_file(src)}},
                           model_revision=common.model_revision())
        tok = load_tokenizer(paths.BASE_MODEL_DIR)
        table = pq.read_table(src)
        dev = set(json.loads(a.split.read_text(encoding="utf-8"))["validation"]["row_keys"]) if a.split.exists() else set()
        reports: dict[str, sm.Report] = defaultdict(sm.Report)
        out_rows, problems = [], []
        limits: dict[str, list[str]] = {}
        thinking_after: dict[str, list[int]] = defaultdict(list)
        for row in table.to_pylist():
            new, turns = sm.mask_row(row, tok.decode)
            groups = ("all", "dev" if row["row_key"] in dev else "train", f"band:{row['band']}")
            for g in groups:
                reports[g].add(row, new, turns)
            if new["input_ids"] != row["input_ids"]:
                problems.append(f"{row['row_key']}: token ids changed")
            if any(n and not o for n, o in zip(new["assistant_masks"], row["assistant_masks"])):
                problems.append(f"{row['row_key']}: the mask grew")
            if any(row["input_ids"][t.end - 1] != sm.IM_END for t in turns if t.trained):
                problems.append(f"{row['row_key']}: a trained span does not end with <|im_end|>")
            trained = [t.thinking for t in turns if t.trained]
            thinking_after[row["band"]] += trained
            fails = limit_failures(trained, row["band"])
            if fails:
                limits[row["row_key"]] = fails
            out_rows.append({**new, "turns_read_masked": sum(t.kind == "read" and not t.trained for t in turns),
                             "submit_masked": any(t.kind == "submit" and not t.trained for t in turns),
                             "turn_kinds": "".join(LETTER[t.kind] if t.trained else LETTER[t.kind].upper() for t in turns)})
        a.out_dir.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([*table.schema, *EXTRA])
        rows_path = a.out_dir / "rows.parquet"
        tmp = rows_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), tmp)
        tmp.replace(rows_path)
        for name in ("selection.json", "supply.json"):
            shutil.copyfile(a.in_dir / name, a.out_dir / name)
        report = {"run_id": run_id, "created": common.utc_now(), "rules": {k: v for k, v in sm.rules().items() if k != "doc"},
                  "input": {"rows": str(src), "sha256": common.sha256_file(src), "split": str(a.split)},
                  "groups": {g: r.to_dict() for g, r in sorted(reports.items())},
                  "band_limits_over_trained_turns": {"violations": limits, "n_rows": len(out_rows),
                                                     "thinking_by_band": {b: sm._stats(v) for b, v in sorted(thinking_after.items())}},
                  "checks": {"problems": problems[:50], "pass": not problems}}
        common.write_json(a.out_dir / "mask_report.json", report)
        manifest = json.loads((a.in_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["v6_masks"] = {"run_id": run_id, "rules": report["rules"], "report": report["groups"]["all"],
                                "derived_from": report["input"], "rows_sha256": common.sha256_file(rows_path),
                                "band_limit_violations": len(limits), "doc": sm.__doc__}
        manifest["selected_thinking_per_turn"] = {k: report["groups"]["all"]["thinking_per_trained_turn"]["after"][k] for k in ("mean", "p90", "max")}
        common.write_json(a.out_dir / "manifest.json", manifest)
        al = report["groups"]["all"]
        run.log(f"{len(out_rows)} rows; trained turns {al['trained_turns']}; reads {al['turns_before']['read']} -> "
                f"{al['turns_after']['read']}; submits masked {al['reasons'].get('submit_unverified', 0)}; loss tokens "
                f"{al['loss_tokens']['before']:,} -> {al['loss_tokens']['after']:,}; band-limit violations over trained turns "
                f"{len(limits)}; problems {len(problems)}")
        for g in ("train", "dev"):
            r = report["groups"].get(g)
            if r:
                run.log(f"{g}: {json.dumps({k: r[k] for k in ('rows', 'turns_before', 'turns_after', 'reasons', 'loss_tokens')})}")
        if problems:
            run.error(f"checks failed: {problems[:5]}")
        run.write_resources([("duration_sec", round(time.time() - t0, 1), "s", "wall clock on the CPU host"),
                             ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "this process"),
                             ("peak_gpu_memory_gb", 0, "GB", "CPU only"), ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only")])
        run.write_metadata(end_time=common.utc_now(), status="ok" if not problems else "checks_failed",
                           outputs={"rows": str(rows_path), "rows_sha256": manifest["v6_masks"]["rows_sha256"],
                                    "report": str(a.out_dir / "mask_report.json")})
        run.finalize()
        return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
