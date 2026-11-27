"""Dev STEM QA slice (docs/DATA_PLAN.md section 10): GPQA Diamond (198, gated, ids only), 30 verifier-checked
reasoning-gym items, and AIME 2025 (30, ids only, reported but never used for selection).

Where the text lives. GPQA and AIME files are downloaded at pinned revisions into a cache under WORK (never into the
repository); the manifest keeps ids, the file sha256 and our seeded GPQA option order only. Reasoning-gym items are
generated here (Apache-2.0); their question and answer are kept in the manifest and re-generated on load to prove
determinism. `load_dev_items` rebuilds every item in memory for the dev evaluation and the D5' screen.

GPQA option order is a seeded permutation per record id (`option_order[j]` = source option shown at letter j, where 0
is the correct answer and 1..3 the incorrect answers in column order), so the rendered letter is reproducible. The
reasoning-gym families below are reserved for the dev set; no training prompt comes from reasoning-gym.
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from apex_data import paths
from apex_data.devset.common import MANIFEST_NAMES, sample_counts, sample_plan, sha256_file

MANIFEST_VERSION = "dev_qa-v2"
CACHE_DIR = paths.WORK / "dev_qa_cache"
MANIFEST_PATH = paths.MANIFESTS / MANIFEST_NAMES["qa"]
GPQA = {"repo": "Idavidrein/gpqa", "revision": "83022cefff930aea54f654c0b282e74b9eeda5c6", "file": "gpqa_diamond.csv",
        "n": 198, "licence": "CC BY 4.0, gated (the HF_TOKEN account must accept the terms); ids only in the repository"}
AIME = {"repo": "math-ai/aime25", "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b", "file": "test.jsonl", "n": 30,
        "licence": "HF mirror tagged Apache-2.0; the MAA problems are evaluation-only; ids only in the repository"}
REASONING_GYM_FAMILIES = ("simple_equations", "polynomial_equations", "prime_factorization", "fraction_simplification",
                          "complex_arithmetic", "base_conversion")
REASONING_GYM_PER_FAMILY = 5
GPQA_COLUMNS = {"id": "Record ID", "question": "Question", "correct": "Correct Answer",
                "incorrect": ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
                "domain": "High-level domain", "subdomain": "Subdomain"}
LETTERS = "ABCDEFGHIJ"
SELECTION_SLICES = ("gpqa_diamond", "reasoning_gym")
REPORT_ONLY_SLICES = ("aime25",)


@dataclass
class DevItem:
    dev_id: str
    slice: str                 # gpqa_diamond | aime25 | reasoning_gym
    kind: str                  # mcq | math
    text: str                  # question as sent to the model (MCQ with lettered options)
    reference: str             # answer letter (MCQ) or answer string (math)
    samples: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def hf_token() -> str | None:
    """HF_TOKEN from the environment or the repository .env (never printed)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = paths.REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            line = line[len("export "):] if line.startswith("export ") else line
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def render_mcq_question(stem: str, options: list[str]) -> str:
    """Question stem plus lettered options ("Answer Choices:" then "A. ..." lines)."""
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"MCQ needs 2..{len(LETTERS)} options, got {len(options)}")
    return stem.strip() + "\n\nAnswer Choices:\n" + "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))


# --- downloads ------------------------------------------------------------------------------------------------------
def cached_path(spec: dict[str, Any], cache: Path = CACHE_DIR) -> Path | None:
    p = cache / f"datasets--{spec['repo'].replace('/', '--')}" / "snapshots" / spec["revision"] / spec["file"]
    return p if p.exists() else None


def fetch(spec: dict[str, Any], token: str | None, cache: Path = CACHE_DIR, offline: bool = False) -> Path:
    """Cached file, else a pinned download. Raises RuntimeError with a readable reason (gated access, offline)."""
    p = cached_path(spec, cache)
    if p is not None:
        return p
    if offline:
        raise RuntimeError(f"{spec['repo']}: not cached under {cache} and --offline was given")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        return Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"],
                                    token=token, cache_dir=str(cache)))
    except GatedRepoError as exc:
        raise RuntimeError(f"gated: the HF_TOKEN account is not authorised for {spec['repo']}; accept the terms at "
                           f"https://huggingface.co/datasets/{spec['repo']} and re-run") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"download failed for {spec['repo']}: {type(exc).__name__}") from exc


# --- slices ---------------------------------------------------------------------------------------------------------
def gpqa_option_order(record_id: str, seed: int = paths.SEED) -> list[int]:
    order = [0, 1, 2, 3]
    random.Random(f"{seed}:gpqa:{record_id}").shuffle(order)
    return order


def gpqa_items(csv_path: Path, seed: int = paths.SEED) -> list[DevItem]:
    """In-memory items (text never leaves memory); manifest entries come from `gpqa_entry`."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = (GPQA_COLUMNS["id"], GPQA_COLUMNS["question"], GPQA_COLUMNS["correct"], *GPQA_COLUMNS["incorrect"])
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"GPQA csv lacks columns {missing}")
        rows = list(reader)
    items = []
    for r in rows:
        rid = r[GPQA_COLUMNS["id"]].strip()
        order = gpqa_option_order(rid, seed)
        answers = [r[GPQA_COLUMNS["correct"]].strip()] + [r[c].strip() for c in GPQA_COLUMNS["incorrect"]]
        options = [answers[k] for k in order]
        items.append(DevItem(f"gpqa:{rid}", "gpqa_diamond", "mcq", render_mcq_question(r[GPQA_COLUMNS["question"]], options),
                             LETTERS[order.index(0)],
                             extra={"record_id": rid, "option_order": order, "options": options,
                                    "correct_answer": answers[0], "domain": r.get(GPQA_COLUMNS["domain"], "").strip(),
                                    "subdomain": r.get(GPQA_COLUMNS["subdomain"], "").strip()}))
    return items


def aime_items(jsonl_path: Path) -> list[DevItem]:
    items = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                items.append(DevItem(f"aime25:{r['id']}", "aime25", "math", str(r["problem"]).strip(),
                                     str(r["answer"]).strip(), extra={"upstream_id": str(r["id"])}))
    return items


def reasoning_gym_items(seed: int = paths.SEED) -> tuple[list[DevItem], str]:
    """Procedural items whose reference answers score 1.0 under the family's own scorer (verifier-checked)."""
    import reasoning_gym

    items = []
    for fi, fam in enumerate(REASONING_GYM_FAMILIES):
        fam_seed = seed * 1000 + fi
        ds = reasoning_gym.create_dataset(fam, size=REASONING_GYM_PER_FAMILY, seed=fam_seed)
        for i in range(REASONING_GYM_PER_FAMILY):
            e = ds[i]
            score = float(ds.score_answer(answer=e["answer"], entry=e))
            if score < 1.0:
                raise RuntimeError(f"reasoning-gym {fam}[{i}] reference scores {score}, not 1.0")
            meta = {k: v for k, v in (e.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool))}
            items.append(DevItem(f"rg:{fam}:{fam_seed}:{i}", "reasoning_gym", "math", e["question"], str(e["answer"]),
                                 extra={"family": fam, "seed": fam_seed, "index": i, "metadata": meta}))
    return items, reasoning_gym.__version__


def score_reasoning_gym(item: DevItem, answer: str) -> float:
    """The family's own scorer, for the dev evaluation."""
    import reasoning_gym

    fam, fam_seed, i = item.extra["family"], item.extra["seed"], item.extra["index"]
    ds = reasoning_gym.create_dataset(fam, size=REASONING_GYM_PER_FAMILY, seed=fam_seed)
    return float(ds.score_answer(answer=answer, entry=ds[i]))


# --- manifest -------------------------------------------------------------------------------------------------------
def _gpqa_entry(it: DevItem, seed: int) -> dict[str, Any]:
    return {"dev_id": it.dev_id, "record_id": it.extra["record_id"], "domain": it.extra["domain"],
            "subdomain": it.extra["subdomain"], "option_order": it.extra["option_order"],
            "samples": sample_plan("qa", it.dev_id, seed)}


def build(token: str | None, seed: int = paths.SEED, cache: Path = CACHE_DIR, offline: bool = False,
          log: Callable[[str], Any] = print) -> tuple[dict[str, Any], list[DevItem]]:
    """(manifest, in-memory items). Raises when a slice cannot be built: the dev set is never frozen partially."""
    cache.mkdir(parents=True, exist_ok=True)
    gpqa_path, aime_path = fetch(GPQA, token, cache, offline), fetch(AIME, token, cache, offline)
    gpqa, aime = gpqa_items(gpqa_path, seed), aime_items(aime_path)
    rg, rg_version = reasoning_gym_items(seed)
    for name, got, spec in (("gpqa_diamond", gpqa, GPQA), ("aime25", aime, AIME)):
        if len(got) != spec["n"] or len({i.dev_id for i in got}) != spec["n"]:
            raise RuntimeError(f"{name}: {len(got)} items ({len({i.dev_id for i in got})} distinct), expected {spec['n']}")
    slices: dict[str, Any] = {
        "gpqa_diamond": {"repo": GPQA["repo"], "revision": GPQA["revision"], "file": GPQA["file"],
                         "file_sha256": sha256_file(gpqa_path), "licence": GPQA["licence"], "kind": "mcq",
                         "use": "selection", "text_in_manifest": False, "n": len(gpqa),
                         "items": [_gpqa_entry(it, seed) for it in gpqa]},
        "reasoning_gym": {"generator": "reasoning-gym", "version": rg_version, "families": list(REASONING_GYM_FAMILIES),
                          "per_family": REASONING_GYM_PER_FAMILY, "family_seed_rule": "seed * 1000 + family index",
                          "licence": "Apache-2.0 (generated here)", "kind": "math", "use": "selection",
                          "text_in_manifest": True, "scorer": "reasoning_gym.<family>.score_answer", "n": len(rg),
                          "disjointness": "families reserved for the dev set; no training prompt comes from reasoning-gym",
                          "items": [{"dev_id": it.dev_id, "family": it.extra["family"], "seed": it.extra["seed"],
                                     "index": it.extra["index"], "question": it.text, "answer": it.reference,
                                     "metadata": it.extra["metadata"], "samples": sample_plan("qa", it.dev_id, seed)}
                                    for it in rg]},
        "aime25": {"repo": AIME["repo"], "revision": AIME["revision"], "file": AIME["file"],
                   "file_sha256": sha256_file(aime_path), "licence": AIME["licence"], "kind": "math",
                   "use": "report_only (full set only; not in the reduced set or the selection metric)",
                   "text_in_manifest": False, "n": len(aime),
                   "items": [{"dev_id": it.dev_id, "upstream_id": it.extra["upstream_id"],
                              "samples": sample_plan("qa", it.dev_id, seed, in_reduced=False)} for it in aime]},
    }
    for it in gpqa + rg:
        it.samples = sample_plan("qa", it.dev_id, seed)
    for it in aime:
        it.samples = sample_plan("qa", it.dev_id, seed, in_reduced=False)
    selection = [e for name in SELECTION_SLICES for e in slices[name]["items"]]
    every = selection + slices["aime25"]["items"]
    manifest = {
        "slice": "qa", "version": MANIFEST_VERSION, "frozen": True, "seed": seed,
        "plan": "docs/DATA_PLAN.md section 10: GPQA Diamond 198 + 30 reasoning-gym items scored (228 x 2), AIME 2025 30 "
                "reported only; reduced set = 228 x 1 (sample index 0)",
        "selection_slices": list(SELECTION_SLICES), "report_only_slices": list(REPORT_ONLY_SLICES),
        "counts": {"selection": sample_counts(selection), "all": sample_counts(every),
                   "per_slice": {k: v["n"] for k, v in slices.items()}},
        "render": {"prompt": "the HLE prompt and judge of the dev evaluation (dev only), 32K budget",
                   "mcq": "stem + 'Answer Choices:' + lettered options in option_order; reference = letter of option 0",
                   "math": "question text as is; reference = the answer string"},
        "loader": "apex_data.devset.qa.load_dev_items (GPQA and AIME text from the WORK cache, checked against file_sha256)",
        "slices": slices,
    }
    log(f"dev_qa: gpqa_diamond {len(gpqa)}, reasoning_gym {len(rg)} (v{rg_version}), aime25 {len(aime)}")
    return manifest, gpqa + rg + aime


def load_dev_items(manifest_path: Path = MANIFEST_PATH, cache: Path = CACHE_DIR, token: str | None = None,
                   offline: bool = True) -> tuple[list[DevItem], dict[str, Any]]:
    """Every manifest item rebuilt in memory, in manifest order, with a check report. Raises when a cached file's sha256
    differs from the manifest, an id is missing, a GPQA letter does not point at the correct answer, or a reasoning-gym
    item no longer regenerates identically. With offline=False a missing cache file is downloaded first."""
    m = json.loads(Path(manifest_path).read_text())
    seed, sl = m["seed"], m["slices"]
    report: dict[str, Any] = {}
    out: list[DevItem] = []
    for name, spec, loader in (("gpqa_diamond", GPQA, lambda p: gpqa_items(p, seed)), ("aime25", AIME, aime_items)):
        p = fetch(spec, token, cache, offline)
        if sha256_file(p) != sl[name]["file_sha256"]:
            raise RuntimeError(f"{name}: cached file sha256 differs from the manifest ({p})")
        by_id = {it.dev_id: it for it in loader(p)}
        missing = [e["dev_id"] for e in sl[name]["items"] if e["dev_id"] not in by_id]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} manifest ids not in the source file, e.g. {missing[:3]}")
        resolved = 0
        for e in sl[name]["items"]:
            it = by_id[e["dev_id"]]
            it.samples = e["samples"]
            if name == "gpqa_diamond":
                if it.extra["option_order"] != e["option_order"]:
                    raise RuntimeError(f"{it.dev_id}: option order differs from the manifest")
                letter = it.reference
                ok = it.extra["options"][LETTERS.index(letter)] == it.extra["correct_answer"] and \
                    f"\n{letter}. {it.extra['correct_answer']}" in it.text
            else:
                ok = it.reference.isdigit() and 0 <= int(it.reference) <= 999
            if not ok:
                raise RuntimeError(f"{it.dev_id}: reference answer does not resolve")
            resolved += 1
            out.append(it)
        report[name] = {"n": len(sl[name]["items"]), "answers_resolved": resolved, "file_sha256_ok": True}
    regen, version = reasoning_gym_items(seed)
    regen_by_id = {it.dev_id: it for it in regen}
    for e in sl["reasoning_gym"]["items"]:
        it = regen_by_id.get(e["dev_id"])
        if it is None or it.text != e["question"] or it.reference != e["answer"]:
            raise RuntimeError(f"{e['dev_id']}: reasoning-gym {version} no longer regenerates the frozen item")
        it.samples = e["samples"]
        out.append(it)
    n_rg = len(sl["reasoning_gym"]["items"])
    report["reasoning_gym"] = {"n": n_rg, "answers_resolved": n_rg, "regenerated_identically": True, "version": version}
    manifest_order = [e["dev_id"] for s in (*SELECTION_SLICES, *REPORT_ONLY_SLICES) for e in sl[s]["items"]]
    order = {dev_id: k for k, dev_id in enumerate(manifest_order)}
    out.sort(key=lambda it: order[it.dev_id])
    report["n_items"] = len(out)
    report["n_selection_items"] = sum(it.slice in SELECTION_SLICES for it in out)
    return out, report
