#!/usr/bin/env python
"""Build the TermGrade terminal task pool (WP8 / U3.2): base image once, per-task images, and the four task checks of
docs/DATA_PLAN.md section 6 on the 40 frozen dev tasks and the seeded stage-2 draw.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_terminal_envs.py
        [--n-stage2 220] [--min-stage2 180] [--tasks a,b,c --pool-dir DIR] [--parallel-builds 2] [--n-concurrent 1]
        [--parallel-scans 2] [--chunk-size 20] [--build-timeout-sec 600] [--check-timeout-sec 1800] [--rebuild]
        [--recheck | --recheck-errors | --recheck-tasks a,b] [--select-only] [--run-id ID --resume]

Inputs:
    /mnt/apex-shared/apex_data/raw/tgr_env/       ai-and/termgrade-environments as fetched (scripts/data/fetch_raw.py;
                                                  revision from data/manifests/raw_sources.json key "tgr_env"):
                                                  base_image/Dockerfile, environments/<task>/, tasks.jsonl, labels.jsonl.
                                                  Never modified; every processed task directory is checked file by file
                                                  against the sha256 list of the raw_sources.json entry.
    data/manifests/dev_terminal.json              the 40 frozen dev tasks (WP6); dev_exclusions.json must agree
    .venv/harbor/bin/harbor (0.24.0), docker      Harbor runs the nop and oracle trials; docker builds and scans images
Outputs:
    dsbase-venv312:local                          the base image, built once from base_image/ (label apex.dockerfile_sha256)
    tgr-<task>:local                              one image per processed task (label apex.environment_sha256)
    /mnt/apex-shared/apex_data/work/tgr_env/
        tasks/<task>/                             prepared copy = the Harbor dataset every trial runs from: the upstream
                                                  directory with only task.toml `[environment].docker_image` pointed at
                                                  tgr-<task>:local (upstream names Harbor's own build tag hb__<task>)
        builds/<task>.log, status/<task>.json     docker build output; per-task status (the unit of resumption)
    logs/<run_id>/                                train.log, metadata.json, metrics.jsonl, errors.log, resources.csv;
                                                  harbor/{nop,oracle}*/ (raw Harbor jobs), status_table.tsv, summary.json
    data/pools/tgr_env/tasks.jsonl                one schema.Task per processed task: split dev | stage2 | excluded
                                                  (failed a check: extra.exclusion, extra.split_before_env_check), with
                                                  extra.env_check (results, seconds) and extra.teacher (stage-1 overlap)
    data/pools/tgr_env/manifest.json              source revision, base image, draw, funnel per split, exclusion reasons,
                                                  dev failures, teacher overlap, timing, sha256 of tasks.jsonl
Selection: dev = the frozen 40; stage 2 = the first --n-stage2 tasks of a seeded uniform permutation of every other task
(apex_data.pools.termgrade.stage2_order, seed 42); when fewer than --min-stage2 pass, the next tasks of the same
permutation are processed (top-up rounds, recorded). --tasks runs an explicit list (smoke tests; use a scratch
--pool-dir). Teacher overlap is recorded, never a reason to exclude (plan section 11).
The four checks (a task is admitted only if all hold; nothing is patched to make a task pass):
    build        `docker build` of environment/ within --build-timeout-sec (plan: 10 min); one retry on failure or timeout
    nop          Harbor `nop` agent (no command), then the task's tests/test.sh: reward must be 0
    oracle       Harbor `oracle` agent (solution/solve.sh in the container), then tests/test.sh: reward 1 with a
                 consistent verifier report (apex_data.grade.harbor_tests.read_trial), within --check-timeout-sec (the
                 rollout agent budget; TermGrade task.toml declares no agent timeout)
    image scan   a root shell in a fresh container of the task image (what the agent sees): no file under /solution; no
                 test file (tests/ names, *.py) under /tests; no file named solve.sh, test_outputs.py, ds_setup.sh,
                 run-tests.sh, solution.yaml; no file whose sha256 equals a tests/ or solution/ file (files also shipped
                 under environment/ are fixtures and allowed); no file containing a test-function line of tests/*.py or a
                 long solve.sh line absent from instruction.md, environment/ and tests/ (fingerprints that already
                 occur in the base image are dropped once per run; NAME hits in the base image likewise). Positive
                 control (2026-10-12): an image with planted copies of test_outputs.py, solve.sh, an edited test file
                 and a /solution file was flagged on all four.
Operational failures (Harbor environment errors) are retried by Harbor and re-run up to three passes
(apex_eval.harbor_utils.harbor_run_resilient), then recorded as nop_error / oracle_error and excluded. A later run with
--recheck-errors re-checks every task whose saved status carries an operational reason (build_failed, build_timeout,
nop_error, oracle_error, image_scan_unavailable) once more; --recheck-tasks re-checks a named list (e.g. trials that
were in flight while the run was suspended). Both keep every other task's saved status and rewrite the whole pool.
Shared docker host: tasks go through the four checks in chunks (dev tasks first), the phases one after another, each
with at most its configured concurrency: 2 builds, 2 scan containers, 1 Harbor trial (a TermGrade trial is two
containers: the task container and Harbor's egress-control sidecar of network_mode "no-network"). While <work>/PAUSE
exists no new build, Harbor job or scan round starts; while <work>/PAUSE_BUILDS exists no new image build starts (cached
images still proceed). In-flight work always finishes; `touch` a file to yield the host, remove it to continue. The pool
files are rewritten after every chunk (manifest "status": "in_progress" until the run completes).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from apex_data import paths  # noqa: E402
from apex_data.coordination import wait_for_docker_window  # noqa: E402
from apex_data.grade.harbor_tests import read_trial  # noqa: E402
from apex_data.pools import termgrade as tg  # noqa: E402
from apex_data.schema import write_jsonl  # noqa: E402
from apex_eval.common import RunDir, git_info, hardware_info, new_run_id, sha256_file, utc_now, write_json  # noqa: E402
from apex_eval.harbor_utils import DEFAULT_RETRY_INCLUDE, HARBOR, harbor_run_resilient, job_dirs, load_trials, retry_args  # noqa: E402

SCAN_NAMES = ("solve.sh", "test_outputs.py", "ds_setup.sh", "run-tests.sh", "solution.yaml")
MIN_SECRET_BYTES = 32
MAX_FINGERPRINTS = 8
TOPUP_ROUNDS = 3
RECHECKABLE = ("build_failed", "build_timeout", "nop_error", "oracle_error", "image_scan_unavailable")   # operational reasons
SCAN_SCRIPT = r"""
set -u
for p in /solution /tests; do
  if [ -d "$p" ]; then find "$p" -xdev -type f 2>/dev/null | head -200 | sed 's/^/FILE:/'; elif [ -e "$p" ]; then echo "FILE:$p"; fi
done
find / -xdev -type f \( @NAMES@ \) 2>/dev/null | head -50 | sed 's/^/NAME:/'
if [ -n "${APEX_SIZES:-}" ]; then
  eval "find / -xdev -type f \( $APEX_SIZES \) -print0" 2>/dev/null | xargs -0 -r sha256sum 2>/dev/null | sed 's/^/HASH:/'
fi
if [ -n "${APEX_FP:-}" ]; then
  printf '%s\n' "$APEX_FP" > /tmp/.apex_fp
  find / -xdev -type f -size -20M ! -path /tmp/.apex_fp -print0 2>/dev/null | xargs -0 -r grep -HoIF -f /tmp/.apex_fp 2>/dev/null | head -200 | sed 's/^/GREP:/'
fi
echo SCAN_DONE
""".replace("@NAMES@", " -o ".join(f"-name {n}" for n in SCAN_NAMES))
TABLE_COLUMNS = ("task_id", "split", "draw_rank", "built", "build_seconds", "nop", "nop_reward", "nop_sec", "oracle", "oracle_reward",
                 "oracle_sec", "image_scan", "scan_hits", "scan_sec", "passed", "fail_reasons", "teacher_passes", "stage1_supply")


# --- CLI ------------------------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-stage2", type=int, default=220, help="stage-2 candidates: the first N of the seeded permutation")
    p.add_argument("--min-stage2", type=int, default=180, help="admitted stage-2 tasks wanted; top-up rounds draw more if short")
    p.add_argument("--tasks", default=None, help="comma-separated task names instead of the draw (smoke tests)")
    p.add_argument("--no-dev", action="store_true", help="skip the dev tasks (stage-2 draw only)")
    p.add_argument("--raw-dir", type=Path, default=tg.RAW_DIR)
    p.add_argument("--work-dir", type=Path, default=tg.WORK_DIR)
    p.add_argument("--pool-dir", type=Path, default=tg.POOL_DIR)
    p.add_argument("--dev-manifest", type=Path, default=paths.MANIFESTS / "dev_terminal.json")
    p.add_argument("--parallel-builds", type=int, default=2, help="concurrent docker builds")
    p.add_argument("--n-concurrent", type=int, default=1, help="concurrent Harbor trials (2 containers each: task + egress sidecar)")
    p.add_argument("--parallel-scans", type=int, default=2, help="concurrent image-scan containers")
    p.add_argument("--chunk-size", type=int, default=20,
                   help="tasks per build/nop/oracle/scan round; the PAUSE file is honoured between rounds and before each build")
    p.add_argument("--build-timeout-sec", type=int, default=600, help="per-task docker build limit (plan: 10 min)")
    p.add_argument("--check-timeout-sec", type=float, default=1800.0,
                   help="agent timeout of the nop/oracle trials (TermGrade declares none; = the rollout agent budget)")
    p.add_argument("--rebuild", action="store_true", help="rebuild task images even when a matching image exists")
    p.add_argument("--recheck", action="store_true", help="redo the checks even when a complete status exists")
    p.add_argument("--recheck-errors", action="store_true",
                   help="redo the checks of tasks whose saved status has an operational failure reason (see RECHECKABLE)")
    p.add_argument("--recheck-tasks", default=None, help="comma-separated task names whose checks are redone")
    p.add_argument("--select-only", action="store_true", help="write the selection (no builds, no checks) and stop")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if max(a.parallel_builds, a.n_concurrent, a.parallel_scans) > 16:
        p.error("more than 16 concurrent containers is not allowed on the shared docker host")
    return a


# --- source, base image, preparation ------------------------------------------------------------------------------------
def source_revision() -> dict[str, Any]:
    man = json.loads(paths.RAW_SOURCES.read_text()) if paths.RAW_SOURCES.exists() else {}
    entry = (man.get("sources") or man).get(tg.SOURCE) if isinstance(man, dict) else None
    return entry if isinstance(entry, dict) else {}


def _docker(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)


def image_label(tag: str, key: str) -> str | None:
    r = _docker(["image", "inspect", "--format", f'{{{{index .Config.Labels "{key}"}}}}', tag])
    out = r.stdout.strip()
    return out if r.returncode == 0 and out and out != "<no value>" else None


def image_info(tag: str) -> dict[str, Any] | None:
    r = _docker(["image", "inspect", "--format", "{{.Id}} {{.Size}}", tag])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    iid, size = r.stdout.split()
    return {"id": iid, "bytes": int(size)}


def docker_build(context: Path, tag: str, labels: dict[str, str], log_path: Path, timeout: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wait_for_docker_window(f"docker build {tag}")   # no build while a reportable Terminal-Bench run is active
    cmd = ["docker", "build", "--progress=plain", "--tag", tag, *[x for k, v in labels.items() for x in ("--label", f"{k}={v}")], str(context)]
    t0 = time.time()
    with log_path.open("w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        try:
            rc: int | None = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                            env={**os.environ, "DOCKER_BUILDKIT": "1"}, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = None
    secs = round(time.time() - t0, 1)
    if rc is None:
        return {"status": "timeout", "seconds": secs, "log": str(log_path), "error": f"build exceeded {timeout}s"}
    if rc != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-8:]
        return {"status": "failed", "seconds": secs, "log": str(log_path), "error": " | ".join(x.strip() for x in tail)[-600:]}
    return {"status": "built", "seconds": secs, "cached": False, "log": str(log_path), **(image_info(tag) or {})}


def ensure_base_image(run: RunDir, raw: Path, work: Path, timeout: int) -> dict[str, Any]:
    """Build dsbase-venv312:local from base_image/ once; an existing image is reused when its label names the same
    Dockerfile sha256 (the card's instruction: build the base image once)."""
    dockerfile = raw / "base_image" / "Dockerfile"
    sha = sha256_file(dockerfile)
    if image_label(tg.BASE_IMAGE, "apex.dockerfile_sha256") == sha:
        info = {"status": "present", "dockerfile_sha256": sha, **(image_info(tg.BASE_IMAGE) or {})}
    else:
        run.log(f"building {tg.BASE_IMAGE} from {dockerfile}")
        b = docker_build(raw / "base_image", tg.BASE_IMAGE, {"apex.dockerfile_sha256": sha}, work / "builds" / "_base_image.log", max(timeout, 1800))
        info = {**b, "dockerfile_sha256": sha}
        if b["status"] != "built":
            run.error(f"base image build {b['status']}: {b.get('error')}")
            raise SystemExit(f"base image {tg.BASE_IMAGE} did not build; see {b['log']}")
    run.log(f"base image {tg.BASE_IMAGE}: {info['status']} id={info.get('id')} dockerfile sha256={sha}")
    return info


def rewrite_docker_image(text: str, image: str) -> str:
    """task.toml with `[environment].docker_image` set to `image` (the only change to a prepared copy)."""
    import tomllib
    line = f'docker_image = "{image}"'
    if re.search(r"^\s*docker_image\s*=", text, re.M):
        out = re.sub(r"^(\s*)docker_image\s*=.*$", lambda m: m.group(1) + line, text, count=1, flags=re.M)
    elif re.search(r"^\[environment\]\s*$", text, re.M):
        out = re.sub(r"^(\[environment\]\s*)$", lambda m: m.group(1) + "\n" + line, text, count=1, flags=re.M)
    else:
        out = text.rstrip("\n") + f"\n\n[environment]\n{line}\n"
    before, after = tomllib.loads(text), tomllib.loads(out)
    after_env = dict(after.get("environment") or {})
    assert after_env.pop("docker_image") == image
    before_env = dict(before.get("environment") or {})
    before_env.pop("docker_image", None)
    assert after_env == before_env and {k: v for k, v in after.items() if k != "environment"} == {k: v for k, v in before.items() if k != "environment"}
    return out


def prepare_task(src: Path, dst: Path, image: str, tree_sha: str) -> dict[str, Any]:
    """Copy the upstream task directory to the prepared dataset (once per content hash) and point docker_image at the
    locally built tag. Returns the marker {tree_sha256, image, upstream_docker_image, prepared_at}."""
    marker = dst / ".prepared.json"
    prev = json.loads(marker.read_text()) if marker.exists() else None
    if prev and prev.get("tree_sha256") == tree_sha and prev.get("image") == image:
        return prev
    tmp = dst.parent.parent / "staging" / dst.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=False)
    toml = (tmp / "task.toml").read_text(encoding="utf-8")
    upstream_image = re.search(r'^\s*docker_image\s*=\s*"([^"]*)"', toml, re.M)
    (tmp / "task.toml").write_text(rewrite_docker_image(toml, image), encoding="utf-8")
    info = {"tree_sha256": tree_sha, "image": image, "upstream_docker_image": upstream_image.group(1) if upstream_image else None,
            "source_dir": str(src), "prepared_at": utc_now()}
    (tmp / ".prepared.json").write_text(json.dumps(info))
    if dst.exists():   # an outdated copy of our own prepared data (never RAW): kept, outside the dataset directory
        old = dst.parent.parent / "superseded" / f"{dst.name}.{int(time.time())}"
        old.parent.mkdir(parents=True, exist_ok=True)
        dst.rename(old)
    tmp.rename(dst)
    return info


def build_task_image(prepared: Path, image: str, log_path: Path, timeout: int, rebuild: bool) -> dict[str, Any]:
    """Build (or reuse, when the label matches the environment/ content) one task image. A failed or timed-out build is
    retried once (network or host-load blips are operational, plan section 11); both attempts are recorded."""
    env_dir = prepared / "environment"
    env_sha = tg.tree_sha256(env_dir)
    if not rebuild and image_label(image, "apex.environment_sha256") == env_sha:
        return {"status": "built", "seconds": 0.0, "cached": True, "log": str(log_path), "environment_sha256": env_sha, **(image_info(image) or {})}
    b = docker_build(env_dir, image, {"apex.environment_sha256": env_sha}, log_path, timeout)
    if b["status"] != "built":
        first = {k: b.get(k) for k in ("status", "seconds", "error")}
        b = {**docker_build(env_dir, image, {"apex.environment_sha256": env_sha}, log_path.with_suffix(".retry.log"), timeout), "first_attempt": first}
    return {**b, "environment_sha256": env_sha}


# --- image scan --------------------------------------------------------------------------------------------------------
def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


def secret_files(task_dir: Path) -> dict[str, int]:
    """sha256 -> size of the tests/ and solution/ files (>= 32 bytes) that must not be reachable from the image; content
    also shipped under environment/ is a fixture and allowed."""
    allowed = {hashlib.sha256(p.read_bytes()).hexdigest() for p in _files(task_dir / "environment")}
    out = {}
    for sub in ("tests", "solution"):
        for p in _files(task_dir / sub):
            if p.stat().st_size >= MIN_SECRET_BYTES:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                if h not in allowed:
                    out[h] = p.stat().st_size
    return out


def fingerprints(task_dir: Path) -> list[str]:
    """Lines whose presence anywhere in the image means a copy of the hidden tests or the solution: the five longest
    `def test_...(...):` lines of tests/*.py (functions or methods, >= 24 characters) and the three longest solution lines
    (>= 40 characters, not comments) that occur in no agent-visible task file (instruction.md, environment/, tests/)."""
    def lines(p: Path) -> list[str]:
        return [x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines()]
    tests = [x for p in _files(task_dir / "tests") if p.suffix == ".py" for x in lines(p) if re.match(r"^def test_\w+\(.*\):$", x) and len(x) >= 24]
    visible = {x for p in [task_dir / "instruction.md", *_files(task_dir / "environment"), *_files(task_dir / "tests")] for x in lines(p)}
    sol = [x for p in _files(task_dir / "solution") for x in lines(p) if len(x) >= 40 and x not in visible and not x.startswith("#")]
    pick = sorted(set(tests), key=lambda x: (-len(x), x))[:5] + sorted(set(sol), key=lambda x: (-len(x), x))[:3]
    return pick[:MAX_FINGERPRINTS]


def run_scan(image: str, sizes: set[int], fps: list[str], timeout: int = 900) -> tuple[str | None, str]:
    size_expr = " -o ".join(f"-size {s}c" for s in sorted(sizes))
    cmd = ["docker", "run", "--rm", "--network", "none", "--user", "0:0", "--entrypoint", "sh",
           "-e", f"APEX_SIZES={size_expr}", "-e", f"APEX_FP={chr(10).join(fps)}", image, "-c", SCAN_SCRIPT]
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, "scan timeout"
    return (r.stdout if "SCAN_DONE" in r.stdout else None), (r.stderr.strip() or f"exit {r.returncode}")[-300:]


def parse_scan(out: str, secret: dict[str, int], test_names: set[str], fps: list[str], base_names: set[str],
               base_fps: set[str]) -> dict[str, Any]:
    hits, fixtures = [], []
    for line in out.splitlines():
        kind, _, rest = line.partition(":")
        rest = rest.strip()
        if kind == "FILE":
            if rest.startswith("/solution"):
                hits.append(f"solution_file:{rest}")
            elif Path(rest).name in test_names or rest.endswith(".py"):
                hits.append(f"tests_file:{rest}")
            else:
                fixtures.append(rest)
        elif kind == "NAME" and rest not in base_names:
            hits.append(f"name:{rest}")
        elif kind == "HASH":
            parts = rest.split(None, 1)
            if len(parts) == 2 and parts[0] in secret:
                hits.append(f"hash:{parts[1].strip()}")
        elif kind == "GREP":
            path, _, fp = rest.partition(":")   # grep -Ho prints <path>:<matched fixed string> (the string may hold colons)
            if fp in fps and fp not in base_fps and path != "/tmp/.apex_fp":
                hits.append(f"fingerprint:{path}")
    return {"hits": sorted(set(hits)), "tests_dir_fixtures": fixtures[:50]}


def scan_task(image: str, task_dir: Path, base_names: set[str], base_fps: set[str]) -> dict[str, Any]:
    t0 = time.time()
    secret = secret_files(task_dir)
    fps = [f for f in fingerprints(task_dir) if f not in base_fps]
    out, err = run_scan(image, set(secret.values()), fps)
    secs = round(time.time() - t0, 1)
    if out is None:
        return {"status": "unavailable", "hits": [], "seconds": secs, "error": err}
    test_names = {p.name for p in _files(task_dir / "tests")}
    parsed = parse_scan(out, secret, test_names, fps, base_names, base_fps)
    return {"status": "hit" if parsed["hits"] else "clean", "seconds": secs, "secret_files": len(secret), "fingerprints": len(fps), **parsed}


class BaseCalibration:
    """NAME hits and fingerprints already present in the bare base image are not task-specific and are ignored in task
    scans. One scan of the base image per batch, for the fingerprints not seen before (cached across batches)."""

    def __init__(self, run: RunDir):
        self.run, self.names, self.seen, self.in_base = run, set(), set(), set()

    def update(self, task_dirs: list[Path]) -> None:
        fps = sorted({f for d in task_dirs for f in fingerprints(d)} - self.seen)
        if not fps and self.seen:
            return
        out, err = run_scan(tg.BASE_IMAGE, set(), fps)
        if out is None:
            raise SystemExit(f"base image scan failed: {err}")
        self.names |= {line[5:].strip() for line in out.splitlines() if line.startswith("NAME:")}
        hit = {line[5:].strip().partition(":")[2] for line in out.splitlines() if line.startswith("GREP:")} & set(fps)
        self.seen |= set(fps)
        self.in_base |= hit
        self.run.log(f"base calibration: {len(fps)} new fingerprints, {len(hit)} occur in {tg.BASE_IMAGE}; {len(self.names)} base NAME hits")


# --- Harbor checks -----------------------------------------------------------------------------------------------------
def harbor_check(run: RunDir, agent: str, job: str, dataset: Path, slugs: list[str], n_concurrent: int, timeout_sec: float) -> dict[str, dict[str, Any]]:
    """One Harbor job of the nop or oracle agent over the prepared dataset; slug -> check result."""
    import yaml
    if not slugs:
        return {}
    cfg = run.run_dir / "config_snapshot" / f"harbor_{job}.yaml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(yaml.safe_dump({"agents": [{"name": agent, "override_timeout_sec": float(timeout_sec)}]}, sort_keys=False))
    env = {**os.environ, "PATH": f"{HARBOR.parent}:{os.environ.get('PATH', '')}"}
    base = ["-c", str(cfg), "-n", str(n_concurrent), *retry_args(1, DEFAULT_RETRY_INCLUDE)]
    harbor_run_resilient(run, job, lambda ids: base + ["-p", str(dataset), *[x for i in ids for x in ("-i", i)]], env, paths.REPO, slugs, max_passes=3)
    trials = load_trials(*job_dirs(run, job))
    out: dict[str, dict[str, Any]] = {}
    for slug in slugs:
        if slug not in trials:
            out[slug] = {"status": "error", "reward": None, "exception": "no trial result"}
            continue
        tdir, _res, n = trials[slug]
        o = read_trial(tdir)
        if o.reward is None:
            status = "error"
        elif agent == "nop":
            status = "fail" if o.reward <= 0 else "pass"
        else:
            status = "pass" if o.reward >= 1.0 and o.verifier_consistent else ("inconsistent" if o.reward >= 1.0 else "fail")
        out[slug] = {"status": status, "reward": o.reward, "exception": f"{o.exception_type}: {o.exception_message}" if o.exception_type else None,
                     "ctrf": o.ctrf, "trial_dir": str(tdir), "attempts": n, "trial_sec": o.trial_sec, "agent_sec": o.agent_sec,
                     "verifier_sec": o.verifier_sec}
    return out


# --- status ------------------------------------------------------------------------------------------------------------
def fail_reasons(build: dict[str, Any], nop: dict[str, Any] | None, oracle: dict[str, Any] | None, scan: dict[str, Any] | None) -> list[str]:
    if build.get("status") != "built":
        return [f"build_{build.get('status', 'not_run')}"]
    r = []
    r += {"pass": ["nop_passed"], "error": ["nop_error"]}.get((nop or {}).get("status"), [] if nop else ["nop_not_run"])
    r += {"fail": ["oracle_failed"], "inconsistent": ["oracle_verifier_inconsistent"], "error": ["oracle_error"]}.get(
        (oracle or {}).get("status"), [] if oracle else ["oracle_not_run"])
    r += {"hit": ["image_scan_hit"], "unavailable": ["image_scan_unavailable"]}.get((scan or {}).get("status"), [] if scan else ["image_scan_not_run"])
    return r


def complete(st: dict[str, Any]) -> bool:
    return st["build"].get("status") != "built" or all(st["checks"].get(c, {}).get("status") not in (None, "not_run") for c in ("nop", "oracle", "image_scan"))


def table_row(rec: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
    c, ex = st["checks"], rec["extra"]
    return {"task_id": rec["task_id"], "split": rec["split"], "draw_rank": ex.get("draw_rank"), "built": st["build"].get("status") == "built",
            "build_seconds": st["build"].get("seconds"), "nop": c["nop"].get("status"), "nop_reward": c["nop"].get("reward"),
            "nop_sec": c["nop"].get("trial_sec"), "oracle": c["oracle"].get("status"), "oracle_reward": c["oracle"].get("reward"),
            "oracle_sec": c["oracle"].get("trial_sec"), "image_scan": c["image_scan"].get("status"),
            "scan_hits": ";".join(c["image_scan"].get("hits", [])), "scan_sec": c["image_scan"].get("seconds"), "passed": st["passed"],
            "fail_reasons": ",".join(st["fail_reasons"]), "teacher_passes": ex["teacher"]["teacher_passes"],
            "stage1_supply": ex["teacher"]["stage1_supply"]}


def _stats(xs: list[float]) -> dict[str, Any]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(statistics.median(xs), 1), "p90": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 1),
            "max": round(xs[-1], 1), "mean": round(statistics.fmean(xs), 1)}


# --- pool outputs -------------------------------------------------------------------------------------------------------
def split0(r: dict[str, Any]) -> str:
    return r["extra"].get("split_before_env_check", r["split"])


def pool_records(records: dict[str, dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool records of every task with a status, env_check attached; a failing task becomes `excluded` with the reasons."""
    out = []
    for slug, rec0 in records.items():
        st = statuses.get(slug)
        if st is None:
            continue
        rec = json.loads(json.dumps({k: v for k, v in rec0.items() if not k.startswith("_")}))
        c = st["checks"]
        rec["extra"]["env_check"] = {
            "passed": st["passed"], "fail_reasons": st["fail_reasons"], "run_id": st["run_id"], "build_status": st["build"].get("status"),
            "build_seconds": st["build"].get("seconds"), "image_id": st["build"].get("id"), "image_bytes": st["build"].get("bytes"),
            "nop_reward": c["nop"].get("reward"), "oracle_reward": c["oracle"].get("reward"), "oracle_ctrf": c["oracle"].get("ctrf"),
            "nop_trial_sec": c["nop"].get("trial_sec"), "oracle_trial_sec": c["oracle"].get("trial_sec"),
            "image_scan": c["image_scan"].get("status"), "scan_hits": c["image_scan"].get("hits", []),
            "tests_dir_fixtures": len(c["image_scan"].get("tests_dir_fixtures", [])), "updated_at": st["updated_at"]}
        if not st["passed"]:
            rec["extra"]["split_before_env_check"] = rec["split"]
            rec["extra"]["exclusion"] = {"rule": "env_check", "detail": ",".join(st["fail_reasons"])}
            rec["split"] = "excluded"
        out.append(rec)
    out.sort(key=lambda r: (split0(r) != "dev", r["extra"]["draw_rank"] if r["extra"]["draw_rank"] is not None else -1, r["task_id"]))
    return out


def funnel_of(recs: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> tuple[dict, dict, dict]:
    funnel, reasons, timing = {}, {}, {}
    for sp in ("dev", "stage2"):
        sts = [statuses[r["upstream_id"]] for r in recs if split0(r) == sp]
        built = [st for st in sts if st["build"].get("status") == "built"]
        funnel[sp] = {"selected": len(sts), "built": len(built),
                      "nop_fails": sum(st["checks"]["nop"].get("status") == "fail" for st in built),
                      "oracle_passes": sum(st["checks"]["oracle"].get("status") == "pass" for st in built),
                      "image_clean": sum(st["checks"]["image_scan"].get("status") == "clean" for st in built),
                      "admitted": sum(st["passed"] for st in sts)}
        reasons[sp] = dict(Counter(x for st in sts for x in st["fail_reasons"]))
        timing[sp] = {"build_sec_fresh": _stats([st["build"].get("seconds") for st in built if not st["build"].get("cached")]),
                      "nop_trial_sec": _stats([st["checks"]["nop"].get("trial_sec") for st in built]),
                      "oracle_trial_sec": _stats([st["checks"]["oracle"].get("trial_sec") for st in built]),
                      "oracle_agent_sec": _stats([st["checks"]["oracle"].get("agent_sec") for st in built]),
                      "scan_sec": _stats([st["checks"]["image_scan"].get("seconds") for st in built])}
    return funnel, reasons, timing


# --- main -----------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    raw, work = a.raw_dir, a.work_dir
    for need in (raw / "tasks.jsonl", raw / "labels.jsonl", raw / "base_image" / "Dockerfile", raw / "environments"):
        if not need.exists():
            sys.exit(f"{need} missing: fetch ai-and/termgrade-environments first (scripts/data/fetch_raw.py, key tgr_env)")
    if not HARBOR.exists():
        sys.exit(f"{HARBOR} missing: run scripts/setup_eval_env.sh first")
    run = RunDir(a.run_id or new_run_id("data_tgr_env"), log_name="train.log", resume=a.resume)
    with run:
        t0 = time.time()
        src = source_revision()
        revision = src.get("revision")
        if not a.tasks and not (revision and re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise SystemExit("raw_sources.json has no pinned 40-hex revision for tgr_env: every pool record needs hf_revision")
        index = tg.load_index(raw)
        dev = [] if a.no_dev else tg.dev_slugs(a.dev_manifest)
        order = tg.stage2_order(index, dev, a.seed)
        rank = {s: i for i, s in enumerate(order)}
        if a.tasks:
            wanted = [tg.check_slug(s.strip()) for s in a.tasks.split(",") if s.strip()]
            selected = [(s, "dev" if s in dev else "stage2") for s in wanted]
        else:
            selected = [(s, "dev") for s in dev] + [(s, "stage2") for s in order[: a.n_stage2]]
        missing = [s for s, _ in selected if s not in index or not (raw / "environments" / s).is_dir()]
        if missing:
            raise SystemExit(f"{len(missing)} selected task(s) missing from the raw snapshot: {missing[:5]}")
        pause_file, pause_builds = work / "PAUSE", work / "PAUSE_BUILDS"
        recheck_names = {tg.check_slug(x.strip()) for x in (a.recheck_tasks or "").split(",") if x.strip()}
        run.write_metadata(command=" ".join(sys.argv), code=git_info(), hardware=hardware_info(), seed=a.seed, start_time=utc_now(),
                           status="running", data_version={"repo": tg.HF_REPO, "revision": revision, "raw_dir": str(raw)},
                           model_revision="n/a (no model: task checks only)", pause_files=[str(pause_file), str(pause_builds)],
                           config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
        run.log(f"tgr_env revision {revision}; {len(index)} tasks upstream; dev {len(dev)}; stage-2 order {len(order)}; "
                f"selected {len(selected)} ({Counter(sp for _, sp in selected)}); pause file {pause_file}")
        if a.select_only:
            write_json(run.run_dir / "summary.json", {"selected": [f"{tg.SOURCE}:{s}" for s, _ in selected]})
            run.write_metadata(end_time=utc_now(), status="completed (select only)")
            run.write_resources([("duration_sec", round(time.time() - t0, 1), "s", "selection only"), ("gpu_hours", 0, "h", "no GPU")])
            run.finalize()
            return 0

        paused_sec = [0.0]

        def wait_if_paused(what: str, *, build: bool = False) -> None:
            """Block while <work>/PAUSE (or, before an image build, <work>/PAUSE_BUILDS) exists: another session needs the
            shared docker host. In-flight work is never touched."""
            files = (pause_file, pause_builds) if build else (pause_file,)
            if not any(f.exists() for f in files):
                return
            run.log(f"paused before {what}: {[str(f) for f in files if f.exists()]} exists")
            t = time.time()
            while any(f.exists() for f in files):
                time.sleep(30)
            paused_sec[0] += time.time() - t
            run.log(f"resumed after {time.time() - t:.0f}s paused")

        wait_if_paused("the base image", build=True)
        base = ensure_base_image(run, raw, work, a.build_timeout_sec)
        status_dir = work / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, dict[str, Any]] = {}
        statuses: dict[str, dict[str, Any]] = {}
        base_cal = BaseCalibration(run)
        topups: list[dict[str, Any]] = []
        harbor_version = subprocess.run([str(HARBOR), "--version"], capture_output=True, text=True, check=False).stdout.strip()

        def write_outputs(final: bool) -> dict[str, Any]:
            recs = pool_records(records, statuses)
            pool_dir = a.pool_dir
            pool_dir.mkdir(parents=True, exist_ok=True)
            n_written = write_jsonl(pool_dir / "tasks.jsonl", recs)
            funnel, reasons, timing = funnel_of(recs, statuses)
            admitted = {sp: [r["task_id"] for r in recs if r["split"] == sp] for sp in ("dev", "stage2")}
            teach = [r["extra"]["teacher"] for r in recs if r["split"] == "stage2"]
            manifest = {
                "pool": tg.SOURCE, "status": "complete" if final else "in_progress", "written_at": utc_now(), "run_id": run.run_id,
                "smoke": bool(a.tasks),
                "source": {"repo": tg.HF_REPO, "revision": revision, "raw_dir": str(raw), "license": tg.DATASET_LICENSE,
                           "n_upstream_tasks": len(index), "raw_sources_entry": src or None},
                "base_image": {"tag": tg.BASE_IMAGE, "id": base.get("id"), "bytes": base.get("bytes"), "dockerfile_sha256": base["dockerfile_sha256"],
                               "built_by_this_run": base.get("status") == "built"},
                "prepared_dataset_dir": str(work / "tasks"),
                "draw": {"seed": a.seed, "rng": f"random.Random('{a.seed}:{tg.SOURCE}:{tg.DRAW_LABEL}').shuffle(sorted non-dev task names)",
                         "n_order": len(order), "n_stage2_candidates": a.n_stage2, "min_stage2": a.min_stage2, "topups": topups,
                         "explicit_tasks": a.tasks.split(",") if a.tasks else None,
                         "dev_manifest": None if a.no_dev else {"path": str(a.dev_manifest), "sha256": sha256_file(a.dev_manifest)}},
                "checks": {"build_timeout_sec": a.build_timeout_sec, "check_agent_timeout_sec": a.check_timeout_sec, "harbor": harbor_version,
                           "n_concurrent": a.n_concurrent, "parallel_builds": a.parallel_builds, "parallel_scans": a.parallel_scans,
                           "harbor_retries": {"max_retries": 1, "retry_include": list(DEFAULT_RETRY_INCLUDE), "rerun_passes": 3},
                           "rules": "build <= timeout; nop reward 0; oracle reward 1 with a consistent CTRF report; image scan clean "
                                    "(scripts/data/build_terminal_envs.py docstring)"},
                "funnel": funnel, "exclusion_reasons": reasons,
                "dev_failures": [{"task_id": r["task_id"], "reasons": r["extra"]["env_check"]["fail_reasons"]} for r in recs
                                 if split0(r) == "dev" and r["split"] == "excluded"],
                "admitted": {sp: len(v) for sp, v in admitted.items()}, "admitted_ids": admitted,
                "teacher_overlap": {"rule": "labels.jsonl DeepSeek-V4-Pro trials on the task (pass_rate x n_counted); stage1_supply = "
                                            ">= 1 teacher pass and not solved by all six models (WP4's admissible supply); allowed, recorded",
                                    "stage2_admitted_with_teacher_passes": sum(t["teacher_passes"] > 0 for t in teach),
                                    "stage2_admitted_in_stage1_supply": sum(t["stage1_supply"] for t in teach),
                                    "stage2_admitted_all_six_solve": sum(t["all_six_solve"] for t in teach)},
                "upstream_split_of_admitted_stage2": dict(Counter(r["extra"]["upstream_split"] for r in recs if r["split"] == "stage2")),
                "admitted_instruction_names_tests": {sp: sum(r["extra"]["instruction_names_tests"] for r in recs if r["split"] == sp)
                                                     for sp in ("dev", "stage2")},
                "timing": timing, "paused_sec": round(paused_sec[0]),
                "tasks_jsonl": {"path": str(pool_dir / "tasks.jsonl"), "records": n_written, "sha256": sha256_file(pool_dir / "tasks.jsonl")},
            }
            write_json(pool_dir / "manifest.json", manifest)
            with (run.run_dir / "status_table.tsv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS, delimiter="\t", lineterminator="\n")
                w.writeheader()
                for r in recs:
                    w.writerow(table_row(r, statuses[r["upstream_id"]]))
            write_json(run.run_dir / "summary.json", manifest)
            return manifest

        def check_chunk(chunk: list[str], tag: str) -> None:
            """Build, nop, oracle and scan one chunk (each phase at most its configured concurrency; phases in sequence)."""
            def build(s: str) -> dict[str, Any]:
                env_sha = tg.tree_sha256(work / "tasks" / s / "environment")
                if a.rebuild or image_label(records[s]["image"], "apex.environment_sha256") != env_sha:
                    wait_if_paused(f"building {s}", build=True)
                return build_task_image(work / "tasks" / s, records[s]["image"], work / "builds" / f"{s}.log", a.build_timeout_sec, a.rebuild)
            builds: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=a.parallel_builds) as ex:
                for s, b in zip(chunk, ex.map(build, chunk)):
                    builds[s] = b
                    run.metric({"step": "build", "task_id": records[s]["task_id"], **{k: v for k, v in b.items() if k != "log"}})
                    if b["status"] != "built":
                        run.error(f"{s}: build {b['status']}: {b.get('error', '')}")
            built = [s for s in chunk if builds[s]["status"] == "built"]
            run.log(f"[{tag}] built {len(built)}/{len(chunk)}")
            checks: dict[str, dict[str, dict[str, Any]]] = {"nop": {}, "oracle": {}, "image_scan": {}}
            for agent in ("nop", "oracle"):
                wait_if_paused(f"{agent} trials of {tag}")
                checks[agent] = harbor_check(run, agent, f"{agent}_{tag}", work / "tasks", built, a.n_concurrent, a.check_timeout_sec)
                run.metric({"step": agent, "batch": tag, **Counter(v["status"] for v in checks[agent].values())})
            wait_if_paused(f"image scans of {tag}")
            base_cal.update([raw / "environments" / s for s in built])
            with ThreadPoolExecutor(max_workers=a.parallel_scans) as ex:
                for s, r in zip(built, ex.map(lambda s: scan_task(records[s]["image"], raw / "environments" / s, base_cal.names, base_cal.in_base), built)):
                    checks["image_scan"][s] = r
            run.metric({"step": "image_scan", "batch": tag, **Counter(v["status"] for v in checks["image_scan"].values())})
            for s in chunk:
                c = {k: checks[k].get(s) or {"status": "not_run"} for k in checks}
                reasons = fail_reasons(builds[s], checks["nop"].get(s), checks["oracle"].get(s), checks["image_scan"].get(s))
                st = {"task_id": records[s]["task_id"], "slug": s, "image": records[s]["image"], "tree_sha256": records[s]["extra"]["tree_sha256"],
                      "prepared": records[s]["_prepared"], "build": builds[s], "checks": c, "passed": not reasons, "fail_reasons": reasons,
                      "run_id": run.run_id, "updated_at": utc_now()}
                write_json(status_dir / f"{s}.json", st)
                statuses[s] = st
                if reasons:
                    run.error(f"{records[s]['task_id']} ({records[s]['split']}): excluded: {','.join(reasons)}")
            m = write_outputs(final=False)
            run.log(f"[{tag}] done: admitted {m['admitted']}; funnel {json.dumps(m['funnel'])}")

        def process(batch: list[tuple[str, str]], tag: str) -> None:
            if src.get("dest") and Path(src["dest"]).resolve() == raw.resolve():
                bad = tg.verify_pinned(raw, [s for s, _ in batch], src)
                if bad:
                    raise SystemExit(f"{len(bad)} raw file(s) differ from the pinned snapshot (raw_sources.json): {bad[:5]}")
                run.log(f"[{tag}] {len(batch)} task directories match the pinned snapshot {revision}")
            else:
                run.log(f"[{tag}] raw dir {raw} is not the pinned fetch destination: snapshot not verified (smoke only)")
            todo = []
            for slug, split in batch:
                src_dir = raw / "environments" / slug
                meta = tg.task_meta(src_dir)
                rec = asdict(tg.pool_record(slug, split, revision, index[slug], meta, rank.get(slug)))
                rec["_prepared"] = prepare_task(src_dir, work / "tasks" / slug, rec["image"], meta["tree_sha256"])
                records[slug] = rec
                sp = status_dir / f"{slug}.json"
                prev = json.loads(sp.read_text()) if sp.exists() else None
                redo = a.rebuild or a.recheck or slug in recheck_names or (
                    a.recheck_errors and prev is not None and any(r in RECHECKABLE for r in prev.get("fail_reasons", [])))
                if prev and not redo and prev.get("tree_sha256") == meta["tree_sha256"] and complete(prev):
                    statuses[slug] = prev
                else:
                    todo.append(slug)
            run.log(f"[{tag}] {len(batch)} task(s): {len(batch) - len(todo)} with a complete status, {len(todo)} to build and check "
                    f"in chunks of {a.chunk_size}")
            for i in range(0, len(todo), a.chunk_size):
                check_chunk(todo[i: i + a.chunk_size], f"{tag}_c{i // a.chunk_size + 1:02d}")

        process(selected, "main")
        n_pass = lambda: sum(1 for s, st in statuses.items() if st["passed"] and records[s]["split"] == "stage2")  # noqa: E731
        next_pos = a.n_stage2
        while not a.tasks and n_pass() < a.min_stage2 and next_pos < len(order) and len(topups) < TOPUP_ROUNDS:
            tried = sum(1 for s in statuses if records[s]["split"] == "stage2")
            rate = max(n_pass() / tried, 0.5) if tried else 1.0
            k = min(len(order) - next_pos, math.ceil((a.min_stage2 - n_pass()) / rate) + 5)
            batch = [(s, "stage2") for s in order[next_pos: next_pos + k]]
            topups.append({"round": len(topups) + 1, "from_rank": next_pos, "n": k, "admitted_before": n_pass()})
            run.log(f"top-up {len(topups)}: {n_pass()} < {a.min_stage2} stage-2 tasks admitted; drawing ranks {next_pos}..{next_pos + k - 1}")
            next_pos += k
            process(batch, f"topup{len(topups)}")

        for rec in records.values():
            rec.pop("_prepared", None)
        manifest = write_outputs(final=True)
        run.log(f"funnel {json.dumps(manifest['funnel'])}; exclusions {json.dumps(manifest['exclusion_reasons'])}; "
                f"dev failures {len(manifest['dev_failures'])}; pool -> {a.pool_dir}")
        dur = time.time() - t0
        run.write_resources([("duration_sec", round(dur, 1), "s", "wall clock on apex-cpu"), ("paused_sec", round(paused_sec[0]), "s", "waiting on the PAUSE file"),
                             ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"), ("gpu_hours", 0, "h", "no GPU"),
                             ("images_built", sum(st["build"].get("status") == "built" for st in statuses.values()), "count", "task images present"),
                             ("cost_usd", 0.0, "USD", "GPU cost only; CPU-only step, the always-on CPU VM is not charged")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(dur, 1), status="completed", summary={"funnel": manifest["funnel"], "admitted": manifest["admitted"]})
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
