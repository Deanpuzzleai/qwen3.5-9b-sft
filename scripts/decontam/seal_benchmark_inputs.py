#!/usr/bin/env python
"""Seal the benchmark INPUTS for the D3 / D5 decontamination screens (docs/DATA_PLAN.md section 7).

RUN THIS ONCE, BY HAND, FROM A TERMINAL. It is the only code in the repository that opens benchmark files, it opens
exactly the allowlisted fields below, and it writes no text anywhere: hashed n-grams, embedding vectors, item ids and
counts only. The pipeline never calls it; the pipeline reads data/decontam/sealed/ through apex_data.screens.d3_match.

Run from the repository root:
    .venv/train/bin/python scripts/decontam/seal_benchmark_inputs.py
        [--benchmarks terminal_bench,asibench,hle,swebench_pro] [--force]

Inputs (hard-coded allowlist, one entry per benchmark; any other field or path is refused with RefusedInput):
  terminal_bench  <tb-dir>/<task>/instruction.md for the 89 ids in configs/eval/terminal_bench_task_ids.txt
                  (default third_party/terminal-bench/terminal-bench; each file's sha256 must equal the locked value in
                  configs/eval/terminal_bench_manifest.tsv). Never task.toml, tests/, solution/ or environment/.
  asibench        <asi-dir>/<instance>/prompt_<level>.md for the prompt levels of the lock (configs/eval/asibench.yaml
                  task_set.prompt_levels: b1, so 60 items for the 60 seed42 instances; --asi-levels overrides), default
                  third_party/asibench_instances/seed42. Never data/, reference/, scorers or task_eval files: a path with a
                  `reference` component is refused outright and no directory is globbed, the prompt paths are constructed.
  hle             --hle-parquet (default third_party/hle_data/data/test-00000-of-00001.parquet; sha256 must equal
                  configs/eval/hle_parquet.sha256): columns id, question, image only, rows whose image is empty (the
                  text-only set, 2,158 rows per configs/eval/hle.yaml). Never answer, rationale or any other column.
  swebench_pro    <swebench-pro-tasks-dir>/<instance_id>/instruction.md for the 51 ids in configs/eval/swebench_pro_hard51_ids.txt
                  (default third_party/SWE-bench_Pro-os/v2/tasks, the locked harness checkout; that file is exactly the problem
                  statement the harness shows the model). Paths are constructed, never globbed; task.toml, tests/, solution/,
                  environment/ and every other file in those directories are refused. Alternative, only when given explicitly:
                  --swebench-pro-parquet, the ScaleAI/SWE-bench_Pro Hugging Face mirror (config hard, split test), columns
                  instance_id and problem_statement only, filtered to the same ids (pyarrow column projection). The
                  manifest records which source was used (`source`: tasks_dir | parquet) and every input file's sha256.
Outputs (nothing else under --sealed-dir is read or written; a target that holds anything this script did not write is
refused): data/decontam/sealed/<benchmark>/{ids.json, ngram8.npy, ngram8_offsets.npy, ngram13.npy, ngram13_offsets.npy,
         embeddings.npy} for the four benchmarks and data/decontam/sealed/manifest.json (writer, item counts, n-gram counts,
         input files with sha256, the embedding model, the n-gram spec, code commit); logs/<run_id>/ (data_seal_<UTC>) with
         counts only. Item counts are checked against the locks: 89 Terminal-Bench, 60 ASI-Bench seed42 B1 prompts, 2,158 HLE
         text-only, 51 SWE-bench Pro HARD. A benchmark whose input sha256s match the manifest and whose files exist is
         skipped (resume) unless --force.
Embeddings: BAAI/bge-small-en-v1.5 @ 5c38ec7c (mean pooled, L2-normalised, float16 on disk) from
         /mnt/apex-shared/models/bge-small-en-v1.5@<revision> or $APEX_BGE_MODEL_DIR.
Fixture mode (--fixture; tests only, on synthetic files in a temporary directory): the lock-count and lock-sha checks are
         relaxed and the manifest is marked `fixture: true`, which SealedIndex.load refuses unless told it is a test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from apex_data.paths import SEALED, SEED  # noqa: E402
from apex_data.screens import d3_match as dm  # noqa: E402
from apex_eval.common import RunDir, command_line, compute_rates, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

CONFIGS = REPO / "configs" / "eval"
THIRD_PARTY = REPO / "third_party"
ASI_LEVEL_RE = re.compile(r"^b[1-4]$")
ASI_PROMPT_RE = re.compile(r"^prompt_b[1-4]\.md$")
ASI_FORBIDDEN = frozenset({"reference", "references", "data", "scorer", "scorers", "tests", "solution", "solutions"})
TASK_DIR_FORBIDDEN = frozenset({"tests", "test", "solution", "solutions", "environment", "reference", "references"})
ALLOWLIST: dict[str, dict[str, Any]] = {
    "terminal_bench": {"file": "instruction.md", "ids": CONFIGS / "terminal_bench_task_ids.txt",
                       "lock_manifest": CONFIGS / "terminal_bench_manifest.tsv", "expected_items": 89},
    "asibench": {"lock": CONFIGS / "asibench.yaml", "default_levels": ("b1",), "expected_instances": 60},
    "hle": {"columns": ("id", "question", "image"), "sha256": CONFIGS / "hle_parquet.sha256", "lock": CONFIGS / "hle.yaml"},
    "swebench_pro": {"file": "instruction.md", "columns": ("instance_id", "problem_statement"),
                     "ids": CONFIGS / "swebench_pro_hard51_ids.txt", "expected_items": 51},
}
DEFAULTS = {"tb_dir": THIRD_PARTY / "terminal-bench" / "terminal-bench", "asi_dir": THIRD_PARTY / "asibench_instances" / "seed42",
            "hle_parquet": THIRD_PARTY / "hle_data" / "data" / "test-00000-of-00001.parquet",
            "swebench_pro_tasks_dir": THIRD_PARTY / "SWE-bench_Pro-os" / "v2" / "tasks"}


class RefusedInput(Exception):
    """A path or field outside the allowlist."""


def assert_allowed(benchmark: str, path: Path) -> None:
    """Refuse anything that is not the one allowlisted file kind of the benchmark, and any `reference` path at all."""
    path = Path(path)
    parts = [p.lower() for p in path.parts]
    if any(p.startswith("reference") for p in parts):
        raise RefusedInput(f"{benchmark}: refusing a reference path ({path})")
    if benchmark == "terminal_bench":
        if path.name != ALLOWLIST["terminal_bench"]["file"] or path.parent.name.lower() in TASK_DIR_FORBIDDEN:
            raise RefusedInput(f"terminal_bench: only <task>/instruction.md is allowed, not {path}")
    elif benchmark == "asibench":   # the prompt file sits directly in the instance directory, never under data/, reference/, ...
        if not ASI_PROMPT_RE.match(path.name) or path.parent.name.lower() in ASI_FORBIDDEN:
            raise RefusedInput(f"asibench: only <instance>/prompt_b1..4.md are allowed, not {path}")
    elif benchmark == "hle":
        if path.suffix != ".parquet":
            raise RefusedInput(f"hle: only the parquet file is allowed, not {path}")
    elif benchmark == "swebench_pro":   # <instance_id>/instruction.md of the harness checkout, or the HF-mirror parquet
        is_task_file = path.name == ALLOWLIST["swebench_pro"]["file"] and path.parent.name.lower() not in TASK_DIR_FORBIDDEN
        if not is_task_file and path.suffix != ".parquet":
            raise RefusedInput(f"swebench_pro: only <instance_id>/instruction.md or the hard/test parquet is allowed, not {path}")
    else:
        raise RefusedInput(f"unknown benchmark {benchmark!r}")


def assert_columns(benchmark: str, columns: tuple[str, ...]) -> None:
    allowed = ALLOWLIST[benchmark].get("columns")
    if allowed is None or tuple(columns) != tuple(allowed):
        raise RefusedInput(f"{benchmark}: columns {columns} are not the allowlisted {allowed}")


def read_text(benchmark: str, path: Path) -> str:
    assert_allowed(benchmark, path)
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def read_columns(benchmark: str, parquet: Path, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Only the allowlisted columns are ever materialised (pyarrow column projection)."""
    import pyarrow.parquet as pq

    assert_allowed(benchmark, parquet)
    assert_columns(benchmark, columns)
    names = pq.ParquetFile(parquet).schema_arrow.names
    missing = [c for c in columns if c not in names]
    if missing:
        raise RefusedInput(f"{benchmark}: parquet lacks the allowlisted columns {missing}")
    return pq.read_table(parquet, columns=list(columns)).to_pylist()


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def lock_asi_levels(lock: Path = ALLOWLIST["asibench"]["lock"]) -> tuple[str, ...]:
    """Prompt levels the ASI-Bench lock evaluates (task_set.prompt_levels; b1 since 2026-10-12)."""
    if not lock.exists():
        return ALLOWLIST["asibench"]["default_levels"]
    levels = ((yaml.safe_load(lock.read_text()) or {}).get("task_set") or {}).get("prompt_levels")
    return tuple(str(x) for x in levels) if levels else ALLOWLIST["asibench"]["default_levels"]


# --- collectors: (items [(id, text)], inputs [{path, sha256}], checks {}) ------------------------------------------------
Collected = tuple[list[tuple[str, str]], list[dict[str, Any]], dict[str, Any]]


def collect_terminal_bench(tb_dir: Path, ids_file: Path, lock_manifest: Path | None, fixture: bool) -> Collected:
    ids = read_ids(ids_file)
    locked: dict[str, str] = {}
    if lock_manifest and lock_manifest.exists():
        for line in lock_manifest.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                cols = line.split("\t")
                if len(cols) >= 4:
                    locked[cols[0]] = cols[3]          # column 4: sha256(instruction.md)
    items, inputs, mismatches = [], [], []
    for tid in ids:
        p = tb_dir / tid / ALLOWLIST["terminal_bench"]["file"]
        assert_allowed("terminal_bench", p)
        if not p.exists():
            raise FileNotFoundError(f"terminal_bench: {p} missing (run scripts/setup_eval_env.sh)")
        sha = dm.sha256_file(p)
        if locked.get(tid) and locked[tid] != sha:
            mismatches.append(tid)
        inputs.append({"path": _rel(p), "sha256": sha})
        items.append((tid, read_text("terminal_bench", p)))
    if mismatches and not fixture:
        raise ValueError(f"terminal_bench: {len(mismatches)} instruction.md files differ from configs/eval/terminal_bench_manifest.tsv")
    return items, inputs, {"n_ids": len(ids), "n_lock_sha_checked": sum(1 for t in ids if t in locked),
                           "n_lock_sha_mismatch": len(mismatches)}


def collect_asibench(asi_dir: Path, levels: tuple[str, ...], fixture: bool) -> Collected:
    bad = [lv for lv in levels if not ASI_LEVEL_RE.match(lv)]
    if bad or not levels:
        raise RefusedInput(f"asibench: prompt levels must be among b1..b4, not {list(levels)}")
    instances = sorted(d for d in asi_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not instances:
        raise FileNotFoundError(f"asibench: no instance directories under {asi_dir}")
    items, inputs = [], []
    for inst in instances:
        for level in levels:
            p = inst / f"prompt_{level}.md"         # constructed, never globbed: nothing else in the instance is looked at
            assert_allowed("asibench", p)
            if not p.exists():
                raise FileNotFoundError(f"asibench: {inst.name} lacks prompt_{level}.md")
            inputs.append({"path": _rel(p), "sha256": dm.sha256_file(p)})
            items.append((f"{inst.name}#{level}", read_text("asibench", p)))
    expected = ALLOWLIST["asibench"]["expected_instances"]
    if len(instances) != expected and not fixture:
        raise ValueError(f"asibench: {len(instances)} instances, the lock expects {expected}")
    marker = asi_dir / ".revision"
    return items, inputs, {"n_instances": len(instances), "levels": list(levels),
                           "instances_revision": marker.read_text().strip() if marker.exists() else None}


def collect_hle(parquet: Path, fixture: bool) -> Collected:
    sha = dm.sha256_file(parquet)
    sha_file = ALLOWLIST["hle"]["sha256"]
    locked = sha_file.read_text().split()[0] if sha_file.exists() else None
    if locked and sha != locked and not fixture:
        raise ValueError("hle: parquet sha256 differs from configs/eval/hle_parquet.sha256")
    rows = read_columns("hle", parquet, ALLOWLIST["hle"]["columns"])
    items = [(str(r["id"]), str(r["question"] or "")) for r in rows if not r.get("image")]
    return items, [{"path": _rel(parquet), "sha256": sha}], {"n_rows": len(rows), "n_text_only": len(items),
                                                              "n_with_image": len(rows) - len(items),
                                                              "sha256_matches_lock": sha == locked if locked else None}


def collect_swebench_pro_tasks(tasks_dir: Path, ids_file: Path) -> Collected:
    """Primary source: <tasks_dir>/<instance_id>/instruction.md for each HARD-51 id (constructed paths; the other task
    directories and every other file are never touched)."""
    ids = read_ids(ids_file)
    items, inputs = [], []
    for tid in ids:
        p = tasks_dir / tid / ALLOWLIST["swebench_pro"]["file"]
        assert_allowed("swebench_pro", p)
        if not p.exists():
            raise FileNotFoundError(f"swebench_pro: {p} missing (run scripts/setup_eval_env.sh)")
        inputs.append({"path": _rel(p), "sha256": dm.sha256_file(p)})
        items.append((tid, read_text("swebench_pro", p)))
    return items, inputs, {"source": "tasks_dir", "n_ids": len(ids)}


def collect_swebench_pro(parquet: Path, ids_file: Path, fixture: bool) -> Collected:
    """Alternative source (explicit --swebench-pro-parquet): the HF mirror's instance_id / problem_statement columns."""
    ids = read_ids(ids_file)
    wanted = set(ids)
    rows = read_columns("swebench_pro", parquet, ALLOWLIST["swebench_pro"]["columns"])
    by_id = {str(r["instance_id"]): str(r["problem_statement"] or "") for r in rows if str(r["instance_id"]) in wanted}
    missing = [i for i in ids if i not in by_id]
    if missing and not fixture:
        raise ValueError(f"swebench_pro: {len(missing)} HARD-51 ids are not in the parquet")
    items = [(i, by_id[i]) for i in ids if i in by_id]
    return items, [{"path": _rel(parquet), "sha256": dm.sha256_file(parquet)}], {
        "source": "parquet", "n_rows_in_file": len(rows), "n_ids": len(ids), "n_missing": len(missing)}


def check_expected_count(name: str, n_items: int, asi_levels: tuple[str, ...], fixture: bool) -> int | None:
    """Item count the locks imply (89 / 60 x levels / 2,158 / 51); a mismatch outside fixture mode is an error."""
    if name == "asibench":
        expected = ALLOWLIST["asibench"]["expected_instances"] * len(asi_levels)
    elif name == "hle":
        lock = ALLOWLIST["hle"]["lock"]
        expected = int(yaml.safe_load(lock.read_text())["dataset_facts"]["text_only"]) if lock.exists() else None
    else:
        expected = ALLOWLIST[name]["expected_items"]
    if expected is not None and n_items != expected and not fixture:
        raise ValueError(f"{name}: {n_items} items sealed, the locks imply {expected}")
    return expected


def combined_sha(inputs: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(i["sha256"] for i in inputs)).encode()).hexdigest()


def seal_benchmark(name: str, items: list[tuple[str, str]], embedder: dm.Embedder, sealed_dir: Path) -> dict[str, Any]:
    """Hash and embed; the texts stay in memory and are discarded."""
    ids = [i for i, _ in items]
    g8, g13 = [], []
    for _, text in items:
        a, b = dm.item_ngrams(text)
        g8.append(a)
        g13.append(b)
    vectors = embedder.embed([t for _, t in items])
    return dm.write_sealed_benchmark(sealed_dir / name, ids, g8, g13, vectors)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", default=",".join(dm.BENCHMARKS), help="comma-separated subset of " + ",".join(dm.BENCHMARKS))
    p.add_argument("--tb-dir", type=Path, default=DEFAULTS["tb_dir"], help="Terminal-Bench 2.0 task directories")
    p.add_argument("--tb-ids", type=Path, default=ALLOWLIST["terminal_bench"]["ids"], help="task id list (locked)")
    p.add_argument("--asi-dir", type=Path, default=DEFAULTS["asi_dir"], help="ASI-Bench seed42 instances directory")
    p.add_argument("--asi-levels", default=None, help="comma-separated prompt levels (default: the lock's task_set.prompt_levels)")
    p.add_argument("--hle-parquet", type=Path, default=DEFAULTS["hle_parquet"], help="HLE test parquet (locked sha256)")
    p.add_argument("--swebench-pro-tasks-dir", type=Path, default=DEFAULTS["swebench_pro_tasks_dir"],
                   help="SWE-bench Pro v2 task directories (locked harness checkout)")
    p.add_argument("--swebench-pro-parquet", type=Path, default=None,
                   help="alternative source: ScaleAI/SWE-bench_Pro hard/test parquet (used only when given)")
    p.add_argument("--hard51-ids", type=Path, default=ALLOWLIST["swebench_pro"]["ids"], help="HARD-51 id list (locked)")
    p.add_argument("--sealed-dir", type=Path, default=SEALED, help="output directory (git-ignored)")
    p.add_argument("--model-dir", type=Path, default=dm.BGE_MODEL_DIR, help="bge-small-en-v1.5 directory")
    p.add_argument("--force", action="store_true", help="re-seal benchmarks whose inputs are unchanged")
    p.add_argument("--fixture", action="store_true", help="TESTS ONLY: synthetic fixtures; relaxes lock checks, marks the manifest")
    p.add_argument("--run-root", type=Path, default=None, help="TESTS ONLY: write logs/ under this root instead of the repository")
    a = p.parse_args(argv)
    a.names = [n for n in a.benchmarks.split(",") if n]
    unknown = [n for n in a.names if n not in dm.BENCHMARKS]
    if unknown:
        p.error(f"unknown benchmarks {unknown}")
    a.levels = tuple(x for x in a.asi_levels.split(",") if x) if a.asi_levels else lock_asi_levels()
    return a


def collect(name: str, a: argparse.Namespace) -> Collected:
    if name == "terminal_bench":
        return collect_terminal_bench(a.tb_dir, a.tb_ids, ALLOWLIST["terminal_bench"]["lock_manifest"], a.fixture)
    if name == "asibench":
        return collect_asibench(a.asi_dir, a.levels, a.fixture)
    if name == "hle":
        return collect_hle(a.hle_parquet, a.fixture)
    if a.swebench_pro_parquet is not None:
        return collect_swebench_pro(a.swebench_pro_parquet, a.hard51_ids, a.fixture)
    return collect_swebench_pro_tasks(a.swebench_pro_tasks_dir, a.hard51_ids)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    sealed_dir = a.sealed_dir
    dm.check_sealed_writable(sealed_dir, a.names)          # before anything is read: never write over foreign files
    mpath = sealed_dir / dm.MANIFEST_NAME
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "writer": dm.MANIFEST_WRITER, "created_at": utc_now(), "benchmarks": {}, "run_ids": []}
    if bool(manifest.get("fixture")) != a.fixture and manifest.get("benchmarks"):
        sys.exit(f"{mpath} is {'a fixture' if manifest.get('fixture') else 'a real'} manifest; use a different --sealed-dir")
    run_id = new_run_id("data_seal")
    rates = compute_rates()
    t_start = time.time()
    with RunDir(run_id, root=a.run_root, log_name="train.log") as run:
        run.write_metadata(run_id=run_id, step="seal_benchmark_inputs", command=command_line(), code=git_info(), hardware=hardware_info(),
                           config={"benchmarks": a.names, "sealed_dir": str(sealed_dir), "fixture": a.fixture, "force": a.force,
                                   "asi_levels": list(a.levels), "ngram_sizes": [dm.NGRAM_SHORT, dm.NGRAM_LONG],
                                   "ngram_spec": dm.NGRAM_SPEC, "model_dir": str(a.model_dir)},
                           seed=SEED, start_time=utc_now(), status="running")
        run.log(f"run_id={run_id} benchmarks={a.names} sealed_dir={sealed_dir} fixture={a.fixture} asi_levels={list(a.levels)}")
        embedder = dm.Embedder(a.model_dir)
        run.log(f"embedding model {embedder.info['name']} weights sha256 {embedder.info['weights_sha256'][:16]}")
        sealed_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}
        for name in a.names:
            t0 = time.time()
            items, inputs, checks = collect(name, a)
            expected = check_expected_count(name, len(items), a.levels, a.fixture)
            digest = combined_sha(inputs)
            prev = manifest["benchmarks"].get(name) or {}
            outputs_exist = all((sealed_dir / name / f).exists() for f in dm.ARTEFACT_FILES)
            if (prev.get("input_sha256_combined") == digest and outputs_exist and not a.force
                    and prev.get("embedding_model_sha256") == embedder.info["weights_sha256"]):
                run.log(f"{name}: unchanged inputs ({len(items)} items), kept the existing artefacts")
                summary[name] = {"status": "unchanged", "n_items": prev.get("n_items")}
                continue
            counts = seal_benchmark(name, items, embedder, sealed_dir)
            del items                                      # the texts are never kept beyond hashing and embedding
            entry = {**counts, **checks, "expected_items": expected, "inputs": inputs, "input_sha256_combined": digest,
                     "n_input_files": len(inputs), "embedding_model_sha256": embedder.info["weights_sha256"],
                     "sealed_at": utc_now(), "run_id": run_id, "seconds": round(time.time() - t0, 1)}
            manifest["benchmarks"][name] = entry
            summary[name] = {"status": "sealed", **{k: v for k, v in entry.items() if k != "inputs"}}
            run.metric({"benchmark": name, **{k: v for k, v in entry.items() if k != "inputs"}})
            run.log(f"{name}: sealed {entry['n_items']} items (expected {expected}), {entry['n_8grams']} 8-grams, "
                    f"{entry['n_13grams']} 13-grams, {entry['n_input_files']} input files in {entry['seconds']}s")
            manifest.update({"writer": dm.MANIFEST_WRITER, "updated_at": utc_now(), "fixture": a.fixture,
                             "ngram_spec": dm.NGRAM_SPEC, "ngram_sizes": [dm.NGRAM_SHORT, dm.NGRAM_LONG],
                             "embedding_model": embedder.info, "code_commit": git_info()["commit"],
                             "run_ids": sorted({*manifest.get("run_ids", []), run_id})})
            dm.write_json_atomic(mpath, manifest)      # after every benchmark, so a crash keeps the finished ones
        duration = time.time() - t_start
        peak_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        run.write_resources([("duration_sec", round(duration, 1), "s", "wall clock on the CPU VM"),
                             ("orchestrator_cpus", os.cpu_count(), "vcpu", "apex-cpu"), ("peak_rss_gb", peak_gb, "GB", "this process"),
                             ("gpu_hours", 0, "h", "CPU only"),
                             ("cost_usd", round(duration / 3600 * rates["cpu_vm_usd_per_hour"], 4), "USD",
                              f"cpu_hours x {rates['cpu_vm_usd_per_hour']} (configs/compute.yaml)")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(duration, 1), status="finished", summary=summary,
                           sealed_manifest=str(mpath))
        run.finalize()
    print(json.dumps({"run_id": run_id, "sealed_dir": str(sealed_dir), "benchmarks": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
