#!/usr/bin/env python
"""Pull the container images of a task pool, in parallel and resumably, with a record of digests and sizes.

Run from the repository root:
    .venv/train/bin/python scripts/data/pull_images.py --images-file data/pools/swesmith/tasks.jsonl [--splits dev,stage2] \
        [--workers 8] [--retries 3] [--max-per-hour 0] [--run-id ID] [--dry-run]

Inputs:  --images-file: a text file with one image reference per line, or a JSONL of schema.Task records with an "image"
         field (duplicates collapsed; --splits keeps only records of those splits, e.g. to skip excluded tasks).
         Registry credentials come from the Docker config written by `docker login`; nothing is read from .env.
Outputs: logs/<run_id>/ (metadata.json, train.log, metrics.jsonl, errors.log, resources.csv) and logs/<run_id>/pulls.jsonl
         with one record per image: ref, status (pulled | present | failed | would_pull), digest, size bytes, seconds,
         error. Images already present are not pulled again; `--run-id` of an earlier run resumes it (its pulled and present
         records are kept). `--max-per-hour N` paces pulls for a rate-limited account (0 = unlimited). Exit status 1 when
         any pull failed. Nothing is ever removed from Docker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from apex_data.coordination import wait_for_docker_window  # noqa: E402
from apex_eval.common import REPO_ROOT, RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402


def load_refs(path: Path, splits: set[str] | None = None) -> list[str]:
    """Image references from a text file or a JSONL of task records (only the given splits when `splits` is set)."""
    refs: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            rec = json.loads(line)
            if splits is not None and rec.get("split") not in splits:
                continue
            if rec.get("image"):
                refs.append(rec["image"])
        else:
            refs.append(line)
    seen: set[str] = set()
    return [r for r in refs if not (r in seen or seen.add(r))]


def inspect(ref: str) -> tuple[str | None, int | None]:
    r = subprocess.run(["docker", "image", "inspect", ref, "--format", "{{index .RepoDigests 0}} {{.Size}}"],
                       capture_output=True, text=True, timeout=120, check=False)
    if r.returncode != 0:
        return None, None
    parts = r.stdout.split()
    return (parts[0] if parts else None), (int(parts[1]) if len(parts) > 1 else None)


class Pacer:
    """At most `per_hour` pull starts in any rolling hour (0 = no limit)."""

    def __init__(self, per_hour: int):
        self.per_hour, self.lock, self.stamps = per_hour, threading.Lock(), []

    def wait(self) -> None:
        if self.per_hour <= 0:
            return
        while True:
            with self.lock:
                now = time.time()
                self.stamps = [s for s in self.stamps if now - s < 3600]
                if len(self.stamps) < self.per_hour:
                    self.stamps.append(now)
                    return
                sleep_for = 3600 - (now - self.stamps[0]) + 1
            time.sleep(min(sleep_for, 60))


def pull_one(ref: str, pacer: Pacer, retries: int, dry_run: bool) -> dict:
    digest, size = inspect(ref)
    if digest:
        return {"ref": ref, "status": "present", "digest": digest, "size": size, "seconds": 0.0, "error": None}
    if dry_run:
        return {"ref": ref, "status": "would_pull", "digest": None, "size": None, "seconds": 0.0, "error": None}
    err = None
    for attempt in range(1, retries + 1):
        wait_for_docker_window(f"docker pull {ref}")   # no pull while a reportable Terminal-Bench run is active
        pacer.wait()
        t0 = time.time()
        try:
            r = subprocess.run(["docker", "pull", "-q", ref], capture_output=True, text=True, timeout=3600, check=False)
            rc, out = r.returncode, (r.stderr or r.stdout)
        except subprocess.TimeoutExpired:
            rc, out = -1, "docker pull timed out after 3600 s"
        if rc == 0:
            digest, size = inspect(ref)
            return {"ref": ref, "status": "pulled", "digest": digest, "size": size, "seconds": round(time.time() - t0, 1), "error": None}
        err = out.strip()[-300:]
        if "toomanyrequests" in err.lower() or "rate limit" in err.lower():
            time.sleep(min(300 * attempt, 900))
        elif "not found" in err.lower() or "manifest unknown" in err.lower():
            break                                   # a missing repository or tag will not appear on retry
        else:
            time.sleep(10 * attempt)
    return {"ref": ref, "status": "failed", "digest": None, "size": None, "seconds": 0.0, "error": err}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-file", required=True, type=Path, help="text file of image refs, or a JSONL of task records")
    p.add_argument("--splits", default=None, help="comma-separated splits to keep from a task JSONL (default: all records)")
    p.add_argument("--workers", type=int, default=2, help="concurrent pulls (the shared host allows 2)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-per-hour", type=int, default=0, help="0 = unlimited")
    p.add_argument("--dry-run", action="store_true", help="report which images would be pulled; pull nothing")
    p.add_argument("--run-id", default=None, help="default data_pull_images_<UTC timestamp>; an existing run id resumes it")
    a = p.parse_args(argv)
    splits = set(a.splits.split(",")) if a.splits else None
    refs = load_refs(a.images_file, splits)
    run_id = a.run_id or new_run_id("data_pull_images")
    resume = bool(a.run_id) and (REPO_ROOT / "logs" / run_id / "metadata.json").exists()
    with RunDir(run_id, log_name="train.log", resume=resume) as run:
        run.write_metadata(run_id=run_id, kind="data_pull_images", command=command_line(), code=git_info(), hardware=hardware_info(),
                           config={"images_file": str(a.images_file), "splits": sorted(splits) if splits else None, "workers": a.workers,
                                   "retries": a.retries, "max_per_hour": a.max_per_hour, "dry_run": a.dry_run},
                           data_version={"images_file_n_refs": len(refs)}, seed=None, start_time=utc_now(), status="running")
        out = run.run_dir / "pulls.jsonl"
        done: dict[str, dict] = {}
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d["status"] in ("pulled", "present"):
                        done[d["ref"]] = d
        todo = [r for r in refs if r not in done]
        run.log(f"{len(refs)} images, {len(done)} already recorded, {len(todo)} to process with {a.workers} workers")
        pacer = Pacer(a.max_per_hour)
        counts = {"pulled": 0, "present": 0, "failed": 0, "would_pull": 0}
        total_bytes = sum(d.get("size") or 0 for d in done.values())
        pulled_bytes = 0
        ex = ThreadPoolExecutor(max_workers=max(1, a.workers))
        f = out.open("a")
        try:
            futs = {ex.submit(pull_one, r, pacer, a.retries, a.dry_run): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                total_bytes += rec.get("size") or 0
                pulled_bytes += (rec.get("size") or 0) if rec["status"] == "pulled" else 0
                if rec["status"] == "failed":
                    run.error(f"pull failed: {rec['ref']}: {rec['error']}")
                if i % 10 == 0 or i == len(todo):
                    run.metric({"step": i, **counts, "total_bytes": total_bytes})
                    run.log(f"{i}/{len(todo)} {counts}")
        except BaseException:      # SIGTERM (RunDir raises SystemExit): queued pulls are cancelled, the in-flight ones finish
            ex.shutdown(wait=True, cancel_futures=True)
            f.close()
            raise
        ex.shutdown(wait=True)
        f.close()
        disk = shutil.disk_usage("/var/lib/docker" if os.path.exists("/var/lib/docker") else "/")
        counts_all = {**counts, "recorded_before": len(done)}
        run.write_metadata(counts=counts_all, total_bytes=total_bytes, pulled_bytes=pulled_bytes, end_time=utc_now(),
                           duration_sec=round(run.elapsed(), 1), status="finished")
        run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "wall clock"),
                             ("images", len(refs), "count", "distinct refs in the input"),
                             ("images_failed", counts["failed"], "count", "see errors.log"),
                             ("image_bytes", total_bytes, "B", "uncompressed size of every image recorded (pulled + present)"),
                             ("pulled_bytes", pulled_bytes, "B", "uncompressed size of the images pulled by this invocation"),
                             ("disk_free_bytes_after", disk.free, "B", "free space on the Docker root's filesystem"),
                             ("gpu_hours", 0, "h", "CPU-only step"), ("cost_usd", 0.0, "USD", "GPU cost only; CPU-only step, the always-on CPU VM is not charged")])
        run.finalize()
        run.log(f"done: {counts}, {total_bytes / 1e9:.1f} GB for the recorded images ({pulled_bytes / 1e9:.1f} GB pulled now)")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
