"""The frozen admitted dev task lists (docs/EXECUTION_PLAN.md M3.1 and M3.2; docs/DATA_PLAN.md sections 6 and 10).

The SWE and terminal dev slices contain only tasks their task checks admitted: SWE-smith tasks whose gold replay passed
(pool record `extra.calibration.status == "ok"`, scripts/data/calibrate_tasks.py) and TermGrade tasks that passed nop /
oracle / image scan (`extra.env_check.passed`, scripts/data/build_terminal_envs.py). The lists are frozen once, before
Base's full dev run, in data/manifests/dev_admitted.json (scripts/data/freeze_dev_admission.py), and every later dev run
(base, sft, stage2_a, stage2_b; full and reduced) uses exactly that list: the dev driver plans only admitted items,
refuses to run when the pools no longer agree with the frozen list, and records `admitted_sha256` in its protocol so
that --baseline refuses a pair evaluated on different lists. Excluded tasks are recorded with their check's reason and
never replaced (no top-up from training or stage-2 pools); the QA slice has no admission step.

dev_admitted.json:
  {"frozen": true, "frozen_at": ..., "admitted_sha256": <sha256 of canonical_ids()>, "slices": {"swe": {"n_manifest",
   "admitted": [{"task_id", "upstream_id"}, ...] (manifest order), "excluded": [{"task_id", "upstream_id", "reason"}, ...]},
   "terminal": {...}}, "inputs": {path: sha256}, "rule": ...}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from apex_data import paths
from apex_data.schema import read_jsonl

ADMITTED_FILE = "dev_admitted.json"
GATED_SLICES = ("swe", "terminal")
DEFAULT_POOLS = {"swe": paths.POOLS / "swesmith" / "tasks.jsonl", "terminal": paths.POOLS / "tgr_env" / "tasks.jsonl"}
RULE = ("SWE: gold-replay calibration status ok; terminal: env_check passed (build, nop fails, oracle passes, image scan); "
        "excluded tasks are recorded with their reason and never replaced; frozen before Base's full dev run")


def _decision(slice_name: str, record: dict[str, Any] | None) -> tuple[str, str | None]:
    """('admitted' | 'excluded' | 'pending', reason) of one dev task from its pool record."""
    if record is None:
        return "pending", "not in the pool file"
    extra = record.get("extra") or {}
    if slice_name == "swe":
        cal = extra.get("calibration") or {}
        status = cal.get("status")
        if status == "ok":
            return "admitted", None
        if status == "excluded":
            return "excluded", str(cal.get("reason") or "calibration: excluded")
        return "pending", f"calibration status {status!r}"
    check = extra.get("env_check")
    if not isinstance(check, dict) or check.get("passed") is None:
        return "pending", "env_check not run"
    if check.get("passed") is True:
        return "admitted", None
    reasons = check.get("fail_reasons") or [(extra.get("exclusion") or {}).get("detail") or "env_check failed"]
    return "excluded", "env_check: " + ",".join(map(str, reasons))


def decide(manifests_dir: Path = paths.MANIFESTS, pools: dict[str, Path] | None = None) -> dict[str, dict[str, Any]]:
    """Per gated slice: the manifest's tasks split into admitted / excluded / pending by the current pool records."""
    from apex_data.devset.run import slice_entries  # noqa: PLC0415 - the driver's own reading of the dev manifests

    pools = {**DEFAULT_POOLS, **(pools or {})}
    out: dict[str, dict[str, Any]] = {}
    for sl in GATED_SLICES:
        by_id = {r["task_id"]: r for r in read_jsonl(pools[sl])} if Path(pools[sl]).exists() else {}
        entries = slice_entries(manifests_dir, sl)[sl]
        d: dict[str, Any] = {"n_manifest": len(entries), "admitted": [], "excluded": [], "pending": []}
        for e in entries:
            state, reason = _decision(sl, by_id.get(e["task_id"]))
            row = {"task_id": e["task_id"], "upstream_id": e["upstream_id"]}
            d[state].append(row if state == "admitted" else {**row, "reason": reason})
        out[sl] = d
    return out


def canonical_ids(slices: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {sl: sorted(r["task_id"] for r in slices[sl]["admitted"]) for sl in sorted(slices)}


def admitted_sha256(slices: dict[str, dict[str, Any]]) -> str:
    """sha256 of the canonical JSON of the sorted admitted task ids per slice (what --baseline compares)."""
    return hashlib.sha256(json.dumps(canonical_ids(slices), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_record(decisions: dict[str, dict[str, Any]], inputs: dict[str, str], frozen_at: str) -> dict[str, Any]:
    pending = {sl: d["pending"] for sl, d in decisions.items() if d["pending"]}
    if pending:
        raise ValueError(f"dev tasks without a decision (their checks have not finished): {json.dumps(pending)[:600]}")
    slices = {sl: {"n_manifest": d["n_manifest"], "n_admitted": len(d["admitted"]), "n_excluded": len(d["excluded"]),
                   "admitted": d["admitted"], "excluded": d["excluded"]} for sl, d in decisions.items()}
    return {"frozen": True, "frozen_at": frozen_at, "admitted_sha256": admitted_sha256(slices), "rule": RULE, "slices": slices,
            "inputs": inputs}


def load(path: Path) -> dict[str, Any] | None:
    """The frozen file, or None when it does not exist; raises when its recorded hash does not match its lists."""
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    if not d.get("frozen") or admitted_sha256(d["slices"]) != d.get("admitted_sha256"):
        raise ValueError(f"{path}: not frozen or admitted_sha256 does not match its lists")
    return d


def admitted_set(frozen: dict[str, Any]) -> dict[str, set[str]]:
    return {sl: {r["task_id"] for r in d["admitted"]} for sl, d in frozen["slices"].items()}


def verify(frozen: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> list[str]:
    """Differences between the frozen lists and what the pools decide now (empty when they agree)."""
    problems: list[str] = []
    want = admitted_set(frozen)
    for sl, d in decisions.items():
        now = {r["task_id"] for r in d["admitted"]}
        lost, gained = sorted(want.get(sl, set()) - now), sorted(now - want.get(sl, set()))
        if lost:
            problems.append(f"{sl}: {len(lost)} frozen-admitted tasks are no longer admitted by the pool (e.g. {lost[0]})")
        if gained:
            problems.append(f"{sl}: {len(gained)} tasks admitted by the pool are not in the frozen list (e.g. {gained[0]})")
    return problems
