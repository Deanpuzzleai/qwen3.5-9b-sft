#!/usr/bin/env python3
"""Freeze the stage-2 pairs: copy a final label run's pairs to data/processed/v5/stage2/pairs.jsonl
(refuses to overwrite) and write the marker logs/coordination/v5/pairs_frozen.json.

Run from the repository root:
    python3 scripts/data/freeze_stage2_pairs.py <final label run id of build_stage2_pairs.py> <audit json>
        [--out data/processed/v5/stage2/pairs.jsonl] [--marker logs/coordination/v5/pairs_frozen.json] [--note TEXT]
Inputs : WORK/<run id>/{pairs.jsonl, report.json, examples.md}; the audit record (per rule: audited, wrong, decision).
Outputs: data/processed/v5/stage2/pairs.jsonl (git-ignored), logs/<run id>/{report.json, examples.md} (copies),
         logs/coordination/v5/pairs_frozen.json (sha256s, counts by family x rule, R chosen kinds, loop shares, mean tokens,
         audit, the DPO command)."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_id")
ap.add_argument("audit", type=Path)
ap.add_argument("--out", type=Path, default=Path("data/processed/v5/stage2/pairs.jsonl"))
ap.add_argument("--marker", type=Path, default=Path("logs/coordination/v5/pairs_frozen.json"))
ap.add_argument("--note", default="SFT pick epoch 1 (sft_v5_20261013_072959 step 239)")
args = ap.parse_args()
run_id, audit_path = args.run_id, args.audit
work = Path("/mnt/apex-shared/apex_data/work") / run_id
dst = args.out
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists():
    raise SystemExit(f"{dst} exists; refusing to overwrite")
shutil.copyfile(work / "pairs.jsonl", dst)
for name in ("report.json", "examples.md"):
    shutil.copyfile(work / name, Path("logs") / run_id / name)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


rep = json.loads((work / "report.json").read_text())
fam_rule: dict[str, int] = {}
for line in dst.open():
    m = json.loads(line)["metadata"]
    k = f"{m['family']}/{m['rule']}"
    fam_rule[k] = fam_rule.get(k, 0) + 1
marker = {
    "producer": "scripts/data/freeze_stage2_pairs.py (v5 stage 2: retention and pairs)",
    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "artifacts": {str(dst): sha(dst), f"logs/{run_id}/report.json": sha(Path("logs") / run_id / "report.json"),
                  f"logs/{run_id}/examples.md": sha(Path("logs") / run_id / "examples.md")},
    "pairs": rep["pairs"], "by_rule": rep["by_rule"], "by_family": rep["by_family"], "by_family_rule": dict(sorted(fam_rule.items())),
    "R_by_family_and_chosen": rep["R_by_family_and_chosen"], "loops_in_V_R": rep["loops_in_V_R"],
    "tokens_by_family_rule": rep["tokens_by_family_rule"], "validator_drops": rep["validator_drops"],
    "audit": json.loads(audit_path.read_text()),
    "label_run": run_id, "sampling_run": rep.get("sampling_run") or run_id,
    "for_dpo": f"python -m apex_train.stage2 --method dpo --pairs {dst} (or --override dpo.data.pairs={dst}); arm B = the same "
               "pairs with loss [\"sft\"]",
    "notes": f"{args.note}; rules V and R (R up to 3 pairs per state); D, E and P dropped after their audits; states never "
             "dev or stage-1 (data_frozen.json ids), p1 screens incl. D3.",
}
args.marker.write_text(json.dumps(marker, indent=1) + "\n")
print(json.dumps({k: marker[k] for k in ("pairs", "by_family_rule", "R_by_family_and_chosen")}, indent=1))
