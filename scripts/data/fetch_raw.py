#!/usr/bin/env python
"""Fetch the pinned raw sources of docs/DATA_PLAN.md (revision 4.3) and record every file in
data/manifests/raw_sources.json.

Run from the repository root with the train venv:
    PYTHONPATH=src .venv/train/bin/python scripts/data/fetch_raw.py \
        [--only KEY,...] [--seed DIR ...] [--jobs N] [--workers N] [--manifest PATH]

Inputs
  SOURCES (below; listed under --help): per source a Hugging Face repo, a full commit sha and the file patterns to fetch.
  The keys are the directory names under paths.RAW: the source codes of src/apex_data/README.md, split where one code
  has several downloads, plus the auxiliary downloads; `bge_small` goes to paths.BGE_SMALL_DIR.
  HF_TOKEN comes from the environment or .env and is never printed (every source is public; the token only raises
  rate limits).
  --seed DIR (repeatable): earlier local copies. A file under DIR (HF .cache trees skipped) is copied instead of
  downloaded when its content hashes to the Hub's id for the pinned revision (LFS sha256, or git blob sha1 for small
  files); the file name does not matter.
Outputs
  paths.RAW/<key>/<repo path> (APEX_DATA_RAW overrides the root): written once by atomic rename, never modified after.
  data/manifests/raw_sources.json: per source the repo, revision, destination, patterns, purpose and every file with
  bytes and sha256, plus `fetch_run` (the run that last changed the entry). Each entry is merged in under a lock as soon
  as its source completes; the file is rewritten only when an entry changes, so a run that changes nothing leaves it
  untouched.
  logs/data_fetch_<UTC>/ (apex_eval.common.RunDir): metadata.json, train.log, metrics.jsonl (one record per file and per
  source), errors.log ("none" if empty), resources.csv.
Behaviour
  Every file of a source is verified (already at its destination: re-hashed against the pin), copied (from --seed) or
  downloaded; copies and downloads land in paths.RAW/.staging/<key>/ (so no HF .cache tree ever sits in a RAW
  directory) and are renamed into place only after their hash matches. A destination file whose hash differs from the
  pin is reported and left alone, never overwritten. A re-run therefore re-hashes everything and transfers nothing.
  Exit status 0 when every selected source is complete, 1 otherwise (the failed sources' entries are not written).
  No benchmark material is fetched.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Iterable

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import disable_progress_bars

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from apex_data import paths  # noqa: E402
from apex_eval.common import (RunDir, command_line, git_info, hardware_info, new_run_id,  # noqa: E402
                              utc_now)


@dataclass(frozen=True)
class Source:
    """One pinned download: fnmatch `patterns` over repo paths (None = every file); `.gitattributes` is never fetched."""
    repo: str
    revision: str                       # full commit sha, never a branch name
    patterns: tuple[str, ...] | None
    purpose: str
    repo_type: str = "dataset"
    dest: Path | None = None            # default paths.RAW / <key>


# Fetch order = priority order: the small sources other work packages wait on first, the large retention sets last.
SOURCES: dict[str, Source] = {
    "swesmith_py": Source(
        "SWE-bench/SWE-smith-py", "77cab9055d42ab4a5c25c89a8f937096db13558e", ("README.md", "data/*"),
        "SWE-smith Python task instances: SWE dev tasks (WP6) and the stage-2 SWE pool (WP8)"),
    "swesmith_go": Source(
        "SWE-bench/SWE-smith-go", "d2f6ecc2cf8ae1af00fc790801bbacfa2913e015", ("README.md", "data/*"),
        "SWE-smith Go task instances: SWE dev tasks (WP6) and the stage-2 SWE pool (WP8)"),
    "tgr": Source(
        "ai-and/termgrade-trajectories", "7d584d63dd598d7e76d444ca23e1527fb5ebd6d9",
        ("README.md", "data/messages/deepseek.parquet", "data/messages/kimi.parquet", "data/messages/qwen.parquet",
         "data/messages/gemma_think.parquet", "data/messages/glm.parquet", "data/trials.parquet"),
        "TermGrade episodes (messages format) and the per-trial verifier rewards: DeepSeek-V4-Pro (MIT), the first "
        "build's terminal teacher (WP4); Kimi-K3 (moonshotai/Kimi-K3, Kimi K3 License, MIT-style; served from the open "
        "weights by the dataset authors), the revision-5 terminal teacher (data plan v5, 3.2); with Qwen3.6-27B "
        "(Apache-2.0), Gemma-4-31B-it with reasoning on (gemma_think, Apache-2.0) and GLM-5.2 (MIT), the terminal "
        "revision-3 teachers in that priority order after Kimi-K3"),
    "th3k_env_meta": Source(
        "shunzou05/TerminalHorizon-Environment", "e9ba9fdc39694612cd792d3f1fd26c241c8c9305",
        ("README.md", "metadata/tasks.parquet"),
        "TerminalHorizon environment metadata (the `category` of each environment) for the WP4 category quota; the "
        "6.9 GB task tarballs are not fetched"),
    "scale_swe_verified": Source(
        "PrimeIntellect/Scale-SWE-Verified", "2c29b67e0ea4d7819f60d8a10d7475a7ad89ccd6", None,
        "Scale-SWE instances that pass gold validation: the admission list of the Scale-SWE rows (WP3)"),
    "swe_rebench_v2": Source(
        "nebius/SWE-rebench-V2", "10483de0f50fe5da545942705a76c6150171af7f", None,
        "SWE-rebench-V2 task rows with reference patches: the size strata of the SWE draw (WP3)"),
    "bge_small": Source(
        "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        ("README.md", "*.json", "vocab.txt", "model.safetensors"),
        "embedding model of the D3/D5' screens and the chat topic classifier: the transformers AutoModel/AutoTokenizer "
        "files plus the sentence-transformers pooling configs (no ONNX, no .bin weights)",
        repo_type="model", dest=paths.BGE_SMALL_DIR),
    "tgr_env": Source(
        "ai-and/termgrade-environments", "3a34ba29ddcd4cbbcbf8840c64f67e853f9b8b0d", None,
        "TermGrade's 1,004 task directories, base image, task, label and task-source tables: terminal dev tasks (WP6) "
        "and the stage-2 terminal pool (WP8)"),
    "th3k": Source(
        "shunzou05/TerminalHorizon-3K", "6c809a44cb608ed1fd4428b3779e79529a1df872",
        ("README.md", "data/TerminalHorizon-3K-SFT.jsonl"),
        "TerminalHorizon-3K DeepSeek-V4-Pro trajectories: the unverified terminal teacher rows (WP4). The JSONL is "
        "byte-identical to revision 1d5db7175868d6abe901616874f86a9f8447d7d5, which the plan measured; this revision's "
        "card adds the paper link and citation"),
    "oswt": Source(
        "nvidia/Open-SWE-Traces", "e192b26a43b8d09f156eeee781aef8e6bd5108ba",
        ("README.md", "data_distribution.md", "data/minisweagent/qwen38_27b/swe-rebench-v2/*",
         "data/minisweagent/qwen38_27b/scale-swe/*"),
        "Open-SWE-Traces v1.2 mini-swe-agent Qwen3.8-27B trajectories, both task-source folders: the SWE teacher rows "
        "(WP3)"),
    "nm_math": Source(
        "nvidia/Nemotron-SFT-Math-v4", "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66",
        ("README.md", "data/train-00000-of-00012.parquet", "data/train-00001-of-00012.parquet",
         "data/train-00002-of-00012.parquet"),
        "retention math (WP5). The AoPS chain-of-thought rows (source == 'AoPS', subset == 'cot'; 51,961 rows: 23,605 + "
        "23,654 + 4,702) all sit in shards 0 to 2 at this revision; the shards are sorted by subset, then source "
        "(counted from their source and subset columns). Shards 3 to 11 hold the Math StackExchange (CC-BY-SA) "
        "chain-of-thought rows and the tool-integrated (TIR) rows"),
    "nm_ifchat": Source(
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402",
        ("README.md", "prepare_chat_prompts.py", "data/instruction_following.jsonl", "data/chat.jsonl"),
        "retention instruction following and chat (WP5). The chat split ships the lmsys- and WildChat-seeded prompts as "
        "null (the lmarena-seeded ones are present); prepare_chat_prompts.py, which would restore them, is fetched, not "
        "run"),
    "nm_sci": Source(
        "nvidia/Nemotron-Science-v1", "82e1af468197076b4f0f392c239274eac032adc7",
        ("README.md", "data/MCQ.jsonl", "data/RQA.jsonl"),
        "retention science (WP5): the MCQ split and the RQA split (free-form chemistry questions)"),
}

DESCRIPTION = ("Pinned raw downloads of docs/DATA_PLAN.md, written by scripts/data/fetch_raw.py: per source the Hugging "
               "Face repo, full revision, destination and every file with bytes and sha256. Files under `dest` are "
               "never modified after the fetch.")
CHUNK = 8 << 20          # hashing / copy buffer
LOG_BYTES = 64 << 20     # files at least this large get a line in train.log (every file gets a metrics.jsonl record)
RETRIES = 3              # download attempts per file; hf_hub_download resumes a partial file
_manifest_lock = threading.Lock()


def hf_token() -> str | None:
    """HF_TOKEN from the environment, else from the repository's .env; None for anonymous access."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def digest(src: Path, git_blob: bool, copy_to: Path | None = None) -> tuple[str, str | None]:
    """sha256 of `src` and, if `git_blob`, its git blob sha1 (the id the Hub lists for a non-LFS file); one read, which
    also writes the bytes to `copy_to` when given."""
    h256 = hashlib.sha256()
    h1 = hashlib.sha1(b"blob %d\0" % src.stat().st_size) if git_blob else None
    with src.open("rb") as f, (copy_to.open("wb") if copy_to else contextlib.nullcontext()) as out:
        while chunk := f.read(CHUNK):
            h256.update(chunk)
            if h1:
                h1.update(chunk)
            if out:
                out.write(chunk)
    return h256.hexdigest(), h1.hexdigest() if h1 else None


def matches(rf: RepoFile, sha256: str, sha1: str | None) -> bool:
    return sha256 == rf.lfs.sha256 if rf.lfs else sha1 == rf.blob_id


class SeedIndex:
    """Regular files under the --seed directories, indexed by size; matched by content, never by name."""

    def __init__(self, dirs: Iterable[Path]):
        self.by_size: dict[int, list[Path]] = defaultdict(list)
        for d in dirs:
            for root, subdirs, files in os.walk(d):
                subdirs[:] = [s for s in subdirs if s != ".cache"]
                for name in files:
                    p = Path(root, name)
                    if p.is_file():
                        self.by_size[p.stat().st_size].append(p)

    def candidates(self, size: int) -> list[Path]:
        return self.by_size.get(size, [])


def thread_map(workers: int, fn: Callable[[Any], Any], items: list[Any]) -> list[tuple[Any, Any, BaseException | None]]:
    """(item, result, exception) for fn over items in a thread pool; work not yet started is cancelled if the caller is
    interrupted (SIGTERM, Ctrl-C), work in flight finishes its current file."""
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [(item, pool.submit(fn, item)) for item in items]
        out = []
        for item, fut in futures:
            try:
                out.append((item, fut.result(), None))
            except Exception as exc:  # noqa: BLE001 - collected and reported per item by the caller
                out.append((item, None, exc))
        return out
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def render(obj: Any, depth: int = 0) -> str:
    """Indented JSON with one line per element of a list of objects (keeps 7,000-file entries readable and diffable)."""
    pad, end = "  " * (depth + 1), "  " * depth
    if isinstance(obj, dict) and obj:
        items = (f"{pad}{json.dumps(k)}: {render(v, depth + 1)}" for k, v in obj.items())
        return "{\n" + ",\n".join(items) + f"\n{end}}}"
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return "[\n" + ",\n".join(pad + json.dumps(x) for x in obj) + f"\n{end}]"
    return json.dumps(obj)


class HubWarnings(logging.Handler):
    """Copies the Hub client's warnings (the HTTP errors it retries, rate-limit waits) into the run's errors.log."""

    def __init__(self, run: RunDir):
        super().__init__(logging.WARNING)
        self.run = run

    def emit(self, record: logging.LogRecord) -> None:
        self.run.error(f"huggingface_hub: {record.getMessage()}")


def update_manifest(path: Path, key: str, entry: dict[str, Any], run_id: str) -> bool:
    """Merge one source entry into the manifest (threads: a lock; processes: flock on the manifest directory) and
    write it atomically, only if something changed. Entries of other sources are kept as they are."""
    with _manifest_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            old = json.loads(path.read_text()) if path.exists() else {}
            sources = dict(old.get("sources", {}))
            prev = sources.get(key)
            if prev is not None and {k: v for k, v in prev.items() if k != "fetch_run"} == entry:
                entry = prev
            else:
                entry = {**entry, "fetch_run": run_id}
            sources[key] = entry
            order = list(SOURCES)
            sources = dict(sorted(sources.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
            new = {"description": DESCRIPTION, "sources": sources}
            if new == old:
                return False
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(render(new) + "\n")
            os.replace(tmp, path)
            return True
        finally:
            os.close(fd)  # releases the flock


@dataclass
class Fetch:
    """Everything one source's file workers share."""
    key: str
    src: Source
    dest: Path
    staging: Path
    token: str | None
    seeds: SeedIndex
    run: RunDir

    def download(self, rf: RepoFile) -> Path:
        for attempt in range(1, RETRIES + 1):
            try:
                return Path(hf_hub_download(self.src.repo, rf.path, repo_type=self.src.repo_type,
                                            revision=self.src.revision, local_dir=self.staging, token=self.token))
            except Exception as exc:  # noqa: BLE001 - network or Hub error: logged, retried, re-raised at the end
                self.run.error(f"{self.key}: {rf.path}: download attempt {attempt}/{RETRIES}: "
                               f"{type(exc).__name__}: {exc}")
                if attempt == RETRIES:
                    raise
                time.sleep(30 * attempt)
        raise AssertionError("unreachable")

    def file(self, rf: RepoFile) -> tuple[dict[str, Any], str]:
        """Verify, copy or download one file; returns its manifest record and the action taken."""
        t0, final, git_blob = time.time(), self.dest / rf.path, rf.lfs is None
        if final.exists():
            sha256, sha1 = digest(final, git_blob) if final.stat().st_size == rf.size else ("", None)
            if not matches(rf, sha256, sha1):
                raise RuntimeError(f"{final} differs from {self.src.repo}@{self.src.revision}:{rf.path}; it is never "
                                   "overwritten: move it away by hand to re-fetch")
            action = "verified"
        else:
            action, staged = "", self.staging / f"{rf.path}.seed"
            for cand in self.seeds.candidates(rf.size):
                staged.parent.mkdir(parents=True, exist_ok=True)
                sha256, sha1 = digest(cand, git_blob, copy_to=staged)
                if matches(rf, sha256, sha1):
                    action = "copied"
                    break
                staged.unlink()
            if not action:
                staged = self.download(rf)
                sha256, sha1 = digest(staged, git_blob)
                if staged.stat().st_size != rf.size or not matches(rf, sha256, sha1):
                    raise RuntimeError(f"downloaded {rf.path} does not match the Hub's hash at {self.src.revision} "
                                       f"(left in {staged})")
                action = "downloaded"
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
        seconds = round(time.time() - t0, 2)
        self.run.metric({"source": self.key, "path": rf.path, "action": action, "bytes": rf.size, "seconds": seconds})
        if rf.size >= LOG_BYTES:
            self.run.log(f"{self.key}: {action} {rf.path} ({rf.size / 1e9:.2f} GB, {seconds:.0f}s)")
        return {"path": rf.path, "bytes": rf.size, "sha256": sha256}, action


def upstream_files(api: HfApi, src: Source) -> list[RepoFile]:
    """The files `src` selects at its pinned revision, sorted by path; a pattern that selects nothing is an error."""
    files = [f for f in api.list_repo_tree(src.repo, repo_type=src.repo_type, revision=src.revision, recursive=True)
             if isinstance(f, RepoFile) and f.path != ".gitattributes"]
    if src.patterns is not None:
        unmatched = [p for p in src.patterns if not any(fnmatch(f.path, p) for f in files)]
        if unmatched:
            raise RuntimeError(f"{src.repo}@{src.revision}: no file matches {unmatched}")
        files = [f for f in files if any(fnmatch(f.path, p) for p in src.patterns)]
    return sorted(files, key=lambda f: f.path)


def fetch_source(key: str, *, api: HfApi, token: str | None, seeds: SeedIndex, run: RunDir, workers: int,
                 manifest: Path) -> dict[str, Any]:
    """Fetch one source completely, then merge its manifest entry; raises if any file failed (entry not written)."""
    t0, src = time.time(), SOURCES[key]
    ctx = Fetch(key, src, src.dest or paths.RAW / key, paths.RAW / ".staging" / key, token, seeds, run)
    files = upstream_files(api, src)
    total = sum(f.size for f in files)
    run.log(f"{key}: {src.repo}@{src.revision[:8]}: {len(files)} files, {total / 1e9:.3f} GB -> {ctx.dest}")
    results = thread_map(workers, ctx.file, files)
    failed = [(rf, exc) for rf, _, exc in results if exc is not None]
    for rf, exc in failed:
        run.error(f"{key}: {rf.path}: {type(exc).__name__}: {exc}")
    if failed:
        raise RuntimeError(f"{key}: {len(failed)} of {len(files)} files failed (see errors.log)")

    expected = {f.path for f in files}
    extra = sorted(p.relative_to(ctx.dest).as_posix() for p in ctx.dest.rglob("*")
                   if p.is_file() and p.relative_to(ctx.dest).as_posix() not in expected)
    if extra:
        run.error(f"{key}: {len(extra)} files in {ctx.dest} are outside the pinned selection (left in place, not in "
                  f"the manifest): {extra[:5]}")
    if ctx.staging.exists():
        unplaced = [p for p in ctx.staging.rglob("*")
                    if p.is_file() and ".cache" not in p.relative_to(ctx.staging).parts]
        if unplaced:
            run.error(f"{key}: {ctx.staging} keeps {len(unplaced)} unplaced files (not removed)")
        else:
            shutil.rmtree(ctx.staging)  # only HF download bookkeeping (.cache locks and metadata) is left

    records = [rec for _, (rec, _), _ in results]
    actions = [action for _, (_, action), _ in results]
    entry = {"repo": src.repo, "repo_type": src.repo_type, "revision": src.revision, "dest": str(ctx.dest),
             "patterns": list(src.patterns) if src.patterns is not None else None, "purpose": src.purpose,
             "n_files": len(records), "bytes": total, "files": records}
    changed = update_manifest(manifest, key, entry, run.run_id)
    summary = {"files": len(files), "bytes": total, "extra_files": len(extra), "manifest_changed": changed,
               "seconds": round(time.time() - t0, 1)}
    for action in ("verified", "copied", "downloaded"):
        summary[action] = actions.count(action)
        summary[f"bytes_{action}"] = sum(f.size for f, a in zip(files, actions) if a == action)
    run.metric({"source": key, "summary": summary})
    run.log(f"{key}: done: {summary['verified']} verified, {summary['copied']} copied, "
            f"{summary['downloaded']} downloaded ({summary['bytes_downloaded'] / 1e9:.3f} GB); "
            f"manifest {'updated' if changed else 'unchanged'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    epilog = "Sources, in fetch order:\n" + "\n".join(
        f"  {k:<19} {s.repo}@{s.revision[:8]}  {', '.join(s.patterns) if s.patterns else 'all files'}"
        for k, s in SOURCES.items())
    p = argparse.ArgumentParser(description=__doc__, epilog=epilog,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated source keys (default: all, in fetch order)")
    p.add_argument("--seed", type=Path, action="append", default=[],
                   help="directory of earlier local copies (repeatable)")
    p.add_argument("--jobs", type=int, default=4, help="sources fetched at the same time (default 4)")
    p.add_argument("--workers", type=int, default=8, help="files fetched at the same time within a source (default 8)")
    p.add_argument("--manifest", type=Path, default=paths.RAW_SOURCES, help="manifest to update (default %(default)s)")
    a = p.parse_args(argv)
    keys = [k for k in a.only.split(",") if k] or list(SOURCES)
    unknown = sorted(set(keys) - set(SOURCES))
    if unknown:
        p.error(f"unknown source keys {unknown}; known: {', '.join(SOURCES)}")
    keys = [k for k in SOURCES if k in keys]

    disable_progress_bars()
    token = hf_token()
    api = HfApi(token=token)
    with RunDir(new_run_id("data_fetch"), log_name="train.log") as run:
        run.write_metadata(
            run_id=run.run_id, command=command_line(), code=git_info(), hardware=hardware_info(), start_time=utc_now(),
            status="running", seed=None,  # nothing is drawn at random
            config={"sources": {k: {"repo": SOURCES[k].repo, "revision": SOURCES[k].revision,
                                    "patterns": SOURCES[k].patterns, "dest": str(SOURCES[k].dest or paths.RAW / k)}
                                for k in keys},
                    "seed_dirs": [str(d) for d in a.seed], "jobs": a.jobs, "workers": a.workers,
                    "manifest": str(a.manifest), "raw_root": str(paths.RAW)},
            data_versions={k: f"{SOURCES[k].repo}@{SOURCES[k].revision}" for k in keys},
            versions={"huggingface_hub": huggingface_hub.__version__}, hf_token="set" if token else "not set")
        logging.getLogger("huggingface_hub").addHandler(HubWarnings(run))
        seeds = SeedIndex(a.seed)
        run.log(f"fetching {len(keys)} sources ({', '.join(keys)}); {sum(map(len, seeds.by_size.values()))} seed files")
        results = thread_map(a.jobs, lambda k: fetch_source(k, api=api, token=token, seeds=seeds, run=run,
                                                            workers=a.workers, manifest=a.manifest), keys)
        summary = {k: s for k, s, exc in results if exc is None}
        failed = {k: f"{type(exc).__name__}: {exc}" for k, _, exc in results if exc is not None}
        for k, reason in failed.items():
            run.error(f"{k}: source failed: {reason}")
        with contextlib.suppress(OSError):
            (paths.RAW / ".staging").rmdir()  # removed only when every source's staging is gone

        totals = {f"bytes_{x}": sum(s[f"bytes_{x}"] for s in summary.values())
                  for x in ("verified", "copied", "downloaded")}
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on apex-cpu"),
            ("bytes_downloaded", totals["bytes_downloaded"], "B", "fetched from the Hugging Face Hub this run"),
            ("bytes_copied", totals["bytes_copied"], "B", "copied from --seed directories after a hash match"),
            ("bytes_verified", totals["bytes_verified"], "B", "already in place; re-hashed against the pin"),
            ("disk_bytes", sum(s["bytes"] for s in summary.values()), "B", "the completed sources on disk"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU host is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1),
                           status="failed" if failed else "finished", summary=summary, failed=failed, totals=totals)
        run.finalize()
        run.log(f"{'FAILED: ' + ', '.join(failed) if failed else 'all sources complete'}; "
                f"downloaded {totals['bytes_downloaded'] / 1e9:.3f} GB, copied {totals['bytes_copied'] / 1e9:.3f} GB, "
                f"verified {totals['bytes_verified'] / 1e9:.3f} GB; logs in {run.run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
