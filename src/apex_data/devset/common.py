"""Shared pieces of the dev-set manifests (docs/DATA_PLAN.md section 10): the per-item sample plan, deterministic JSON,
the write-once freeze and the exclusion file the pool builders read.

Sample plan. Every scored item carries `samples`: one entry `{"index": i, "seed": s, "reduced": bool}` per sample of
the full set (SWE 4, terminal 3, QA 2). The reduced set is the first `REDUCED_K` samples of the full set (SWE 2,
terminal 2, QA 1): same item ids, sample indices and seeds, so a reduced-set result pairs item for item with the first
samples of a full run. A sample's seed is a pure function of (SEED, slice, item id, index); a re-run reproduces it.

Freeze. A build writes the three slice manifests, `dev_exclusions.json` and `dev_freeze.json` (sha256 of the other
four, counts, date). Once `dev_freeze.json` exists with `frozen: true`, a re-run never writes: it rebuilds in memory
and checks that every file is byte-identical to the frozen one, and fails loudly when it is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from apex_data import paths

FULL_K = {"swe": 4, "terminal": 3, "qa": 2}
REDUCED_K = {"swe": 2, "terminal": 2, "qa": 1}
MANIFEST_NAMES = {"swe": "dev_swe.json", "terminal": "dev_terminal.json", "qa": "dev_qa.json"}
EXCLUSIONS_NAME = paths.DEV_EXCLUSIONS.name          # dev_exclusions.json
FREEZE_NAME = "dev_freeze.json"
_LINE_WIDTH = 160


class FrozenManifestError(RuntimeError):
    """A rebuild differs from a frozen dev manifest; frozen files are never overwritten."""


def rng_for(seed: int, *labels: str) -> random.Random:
    """Independent seeded stream per draw step, e.g. rng_for(42, "swesmith", "dev", "py")."""
    return random.Random(":".join([str(seed), *labels]))


def sample_seed(seed: int, slice_name: str, item_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:dev:{slice_name}:{item_id}:{index}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def sample_plan(slice_name: str, item_id: str, seed: int = paths.SEED, *, in_reduced: bool = True) -> list[dict[str, Any]]:
    """Full-set samples of one item; the first REDUCED_K are flagged `reduced` (none for report-only items)."""
    k_reduced = REDUCED_K[slice_name] if in_reduced else 0
    return [{"index": i, "seed": sample_seed(seed, slice_name, item_id, i), "reduced": i < k_reduced}
            for i in range(FULL_K[slice_name])]


def sample_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {"items": len(items), "full_samples": sum(len(it["samples"]) for it in items),
            "reduced_items": sum(any(s["reduced"] for s in it["samples"]) for it in items),
            "reduced_samples": sum(s["reduced"] for it in items for s in it["samples"])}


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split `total` by `weights` (largest remainder; ties broken by key)."""
    if total <= 0 or not weights or sum(weights.values()) <= 0:
        return {k: 0 for k in weights}
    s = sum(weights.values())
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


# --- deterministic JSON ---------------------------------------------------------------------------------------------
def _flat(v: Any) -> bool:
    if isinstance(v, dict):
        return all(not isinstance(x, (dict, list)) for x in v.values())
    if isinstance(v, list):
        return all(not isinstance(x, (dict, list)) for x in v)
    return True


def _encode(obj: Any, level: int) -> str:
    one_line = json.dumps(obj, ensure_ascii=False)
    if not isinstance(obj, (dict, list)) or not obj or (_flat(obj) and len(one_line) + 2 * level <= _LINE_WIDTH):
        return one_line
    pad, inner = "  " * level, "  " * (level + 1)
    if isinstance(obj, dict):
        body = ",\n".join(f"{inner}{json.dumps(k, ensure_ascii=False)}: {_encode(v, level + 1)}" for k, v in obj.items())
        return "{\n" + body + "\n" + pad + "}"
    body = ",\n".join(inner + _encode(v, level + 1) for v in obj)
    return "[\n" + body + "\n" + pad + "]"


def dumps(obj: Any) -> str:
    """Stable, diff-friendly JSON: short scalar-only objects and lists on one line, everything else indented."""
    return _encode(obj, 0) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --- freeze ---------------------------------------------------------------------------------------------------------
def read_freeze(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / FREEZE_NAME
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    return rec if rec.get("frozen") else None


def check_frozen(out_dir: Path, name: str, text: str, freeze: dict[str, Any]) -> str:
    """Compare a rebuilt file with its frozen version; return its sha256 or raise FrozenManifestError."""
    new = sha256_bytes(text.encode("utf-8"))
    want = (freeze.get("files") or {}).get(name, {}).get("sha256")
    p = out_dir / name
    on_disk = sha256_file(p) if p.exists() else None
    if want is None or new != want or on_disk != want:
        raise FrozenManifestError(f"{name}: rebuilt sha256 {new[:16]} vs frozen {str(want)[:16]} (file on disk "
                                  f"{str(on_disk)[:16]}); frozen dev manifests are never overwritten")
    return new


def exclusions_record(swe: dict[str, Any], terminal: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    """The file every training and stage-2 pool builder reads (paths.DEV_EXCLUSIONS)."""
    return {
        "note": ("Dev-set exclusions (docs/DATA_PLAN.md sections 7 and 10). Ids are upstream ids without a source prefix: "
                 "SWE-smith instance ids, TermGrade task names (the same name in ai-and/termgrade-trajectories and "
                 "ai-and/termgrade-environments) and dev QA ids. Training rows (WP3, WP4, WP5, WP7) and stage-2 pools "
                 "(WP8) drop every task, episode or prompt on these ids, and every SWE task or teacher row on these "
                 "repositories; train-vs-dev near-duplicates are dropped on the training side (section 7)."),
        "frozen": True,
        "swe_repositories": sorted(swe["repositories"]),
        "swe_task_ids": sorted(it["upstream_id"] for it in swe["items"]),
        "terminal_task_ids": sorted(it["upstream_id"] for it in terminal["items"]),
        "qa_ids": sorted(it["dev_id"] for sl in qa["slices"].values() for it in sl["items"]),
        "applies_to": {
            "swe_repositories": "every SWE training row (oswt, both subsources) and the SWE-smith stage-2 pool (WP3, WP8)",
            "swe_task_ids": "the SWE-smith stage-2 pool (WP8); implied by swe_repositories",
            "terminal_task_ids": "TermGrade trajectories in the terminal slice (WP4) and the TermGrade stage-2 pool (WP8)",
            "qa_ids": "the D5' dev-validity screen of the retention slice (WP5); items are loaded by "
                      "apex_data.devset.qa.load_dev_items at run time",
        },
    }
