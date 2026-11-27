#!/usr/bin/env python
"""Gold calibration and the three-replay smoke of the SWE-smith pool (data plan sections 6 and 11; milestone M3.1).

Why: the grader replays agent patches under --network none, 2 CPUs and 8 GB. Tests that need the network or are unstable
in that sandbox fail for every patch, the gold one included. SWE-bench defines PASS_TO_PASS as the tests that pass before
and after the gold patch in the harness environment; this script establishes that set for our environment by replaying
each task's gold fix (the reverse of the dataset's bug patch) once through the grader's own path
(grade.swesmith.calibrate_task: same image, instance branch, tests restored, apply ladder, structured reporter or profile
parser, sandbox limits), then
  * writes task.extra.calibration = {status, reason, gold_pass, p2p_excluded, f2p_missing, parser, reporter, coverage,
    seconds, run_id, ...} and, for admitted tasks, task.extra.eval_spec (the validated command, reporter, parser and P2P
    exclusions every later replay executes verbatim) into the pool file (atomic rewrite, every record kept);
  * excludes tasks whose image is missing, whose gold fix does not apply, fails an F2P test or times out: split
    "excluded", the previous split under extra.split_before_calibration, extra.exclusion = {rule: "calibration", detail};
    a later calibration that passes restores the split. Sandbox failures are retried twice, then counted (env_failure) and
    left untouched for the next run;
  * with --smoke, runs the three-replay smoke per repository on an admitted task: the gold fix graded through
    grade_patch (one run; must give r_pass 1); the empty diff replayed on the bug state (the F2P tests must fail)
    and graded (r_pass 0, flag empty_diff); the gold fix plus one harness-side test file reverted (the first file of the
    instance branch's "Remove F2P Tests" commit) graded (r_pass 0, flag tests_touched). A task whose gold does not
    reproduce or whose F2P tests pass without the fix is excluded (rule "smoke") and the next admitted task of the
    repository is tried (three attempts); a repository without a passing smoke has all its tasks excluded;
  * records the counts in data/pools/swesmith/manifest.json ("calibration" and "smoke" blocks).
New containers wait while the 1-minute load average exceeds --load-pause. All containers are named s2swesmith_*.

Usage (repository root):
    .venv/train/bin/python scripts/data/calibrate_tasks.py [--pool data/pools/swesmith/tasks.jsonl] [--splits dev,stage2] \
        [--slots 2] [--cpus 2] [--memory 8g] [--network none] [--repos a/b,c/d] [--task-ids id,id] [--limit N] [--defer-repos a/b] \
        [--smoke | --smoke-only] [--load-pause 400] [--run-id ID --resume] [--work-dir DIR] [--dry-run]
Inputs : the pool file (schema.Task JSONL written by apex_data.pools.swe_smith); the task images (pulled by
         scripts/data/pull_images.py); the SWE-smith mirrors (cloned on first use under RAW/swesmith_mirrors); .venv/swesmith.
Outputs: logs/<run_id>/ (metadata.json, train.log, metrics.jsonl one line per task and per smoke, errors.log, resources.csv,
         calibration_summary.json); WORK/<run_id>/<task slug>/{calibration.json, gold/..., gold_text/...};
         WORK/<run_id>/smoke/<repo>/<attempt>/{gold,empty,tampered}/...; the pool file rewritten in place; the manifest blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.grade import swesmith  # noqa: E402
from apex_data.grade.swesmith import (DEFAULT_MIRRORS_DIR, VERIFIER_VERSION, GradeOptions, SandboxError, image_present,  # noqa: E402
                                      repo_name_of, slug, test_lists)
from apex_data.pools.common import repo_key  # noqa: E402
from apex_data.schema import read_jsonl, write_jsonl  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, sha256_file, utc_now  # noqa: E402

DEFAULT_POOL = paths.POOLS / "swesmith" / "tasks.jsonl"
CALIBRATION_KEYS = ("status", "reason", "gold_pass", "gold_applied", "p2p_excluded", "f2p_missing", "parser", "reporter", "coverage", "seconds",
                    "timed_out", "run_id", "verifier_version", "calibrated_at", "gold_patch_sha256", "test_cmd", "attempts", "n_f2p", "n_p2p")
FINAL_STATUSES = ("ok", "excluded")
MAX_ENV_RETRIES = 2
SMOKE_ATTEMPTS = 3
WORST_N = 40
STOP = threading.Event()              # set on SIGTERM / a crash of the main thread: no new container starts, in-flight ones finish


def run_pool(fn: Callable[[Any], Any], items: list[Any], slots: int) -> list[Any]:
    """`fn` over `items` on `slots` threads; when the main thread is interrupted (RunDir turns SIGTERM into SystemExit), the
    queued items are cancelled and the in-flight ones finish cleanly (their containers are removed) before re-raising."""
    ex = ThreadPoolExecutor(max_workers=max(1, slots))
    try:
        out = list(ex.map(fn, items))
    except BaseException:
        STOP.set()
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return out


@dataclass
class Job:
    task: dict[str, Any]
    job_dir: Path
    skip: str | None = None           # already_excluded (counted, never run)
    result: dict[str, Any] | None = None
    attempts: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="schema.Task JSONL to calibrate and rewrite")
    p.add_argument("--splits", default="dev,stage2", help="splits to calibrate (calibration-excluded tasks of these splits are re-checked)")
    p.add_argument("--repos", default=None, help="only tasks of these repositories (owner/name, comma-separated)")
    p.add_argument("--task-ids", default=None, help="only these task ids (comma-separated)")
    p.add_argument("--limit", type=int, default=None, help="only the first N selected tasks (smoke runs)")
    p.add_argument("--defer-repos", default=None, help="calibrate these repositories (owner/name, comma-separated) after all others "
                                                         "(heavy suites while the host is shared)")
    p.add_argument("--slots", type=int, default=2, help="concurrent replay containers (the shared host allows 2 while the benchmark runs load it)")
    p.add_argument("--cpus", type=float, default=2.0, help="per container (the grader's replay conditions)")
    p.add_argument("--memory", default="8g")
    p.add_argument("--cgroup-parent", default=None)
    p.add_argument("--network", default="none", help="docker network of the replay; the grader's is none")
    p.add_argument("--mirrors-dir", type=Path, default=DEFAULT_MIRRORS_DIR)
    p.add_argument("--load-pause", type=float, default=400.0, help="wait before a container while the 1-min load exceeds this")
    p.add_argument("--smoke", action="store_true", help="after calibration, run the three-replay smoke per repository")
    p.add_argument("--smoke-only", action="store_true", help="skip calibration; smoke the repositories of the selected, admitted tasks")
    p.add_argument("--run-id", default=None, help="default data_calibrate_swesmith_<UTC timestamp>")
    p.add_argument("--resume", action="store_true", help="continue --run-id: finished per-task results are reused")
    p.add_argument("--work-dir", type=Path, default=None, help="default: WORK/<run_id>")
    p.add_argument("--manifest", type=Path, default=None, help="pool manifest to update (default: manifest.json next to --pool)")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: logs/ under this root")
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true", help="validate inputs, check images and print the plan")
    return p.parse_args(argv)


# --- jobs -------------------------------------------------------------------------------------------------------------
def selected(rec: dict[str, Any], a: argparse.Namespace) -> bool:
    split = rec.get("split")
    if split == "excluded":
        split = (rec.get("extra") or {}).get("split_before_calibration") or split
    if split not in set(a.splits.split(",")):
        return False
    if a.repos and repo_key(rec.get("repo")) not in {repo_key(r) for r in a.repos.split(",")}:
        return False
    return not a.task_ids or rec["task_id"] in set(a.task_ids.split(","))


def skip_reason(rec: dict[str, Any]) -> str | None:
    """already_excluded: dropped by a rule other than calibration (a calibration exclusion is re-checked)."""
    rule = ((rec.get("extra") or {}).get("exclusion") or {}).get("rule")
    if rec.get("split") == "excluded" and rule != "calibration":
        return "already_excluded"
    if not test_lists(rec)[0]:
        return "no_f2p_list"
    return None


def load_jobs(records: list[dict[str, Any]], a: argparse.Namespace, work_dir: Path) -> list[Job]:
    jobs = [Job(task=r, job_dir=work_dir / slug(r["task_id"], 120), skip=skip_reason(r)) for r in records if selected(r, a)]
    # dev first, then the stage-2 candidates in draw order: a partial run covers the dev set and a uniform prefix of the draw
    deferred = {repo_key(r) for r in (a.defer_repos or "").split(",") if r}
    jobs.sort(key=lambda j: (original_split(j.task) != "dev", repo_key(j.task.get("repo")) in deferred,
                             (j.task.get("extra") or {}).get("draw_rank", 0), j.task["task_id"]))
    return jobs[:a.limit] if a.limit is not None else jobs


def image_missing_result(job: Job, run_id: str) -> dict[str, Any]:
    return {**swesmith.calibration_base(job.task, run_id, "none"), "status": "excluded", "reason": "calibration:image_missing"}


def check_images(jobs: list[Job], present: Callable[[str], bool], run_id: str, log: Callable[[str], None]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for j in jobs:
        if j.skip:
            continue
        image = j.task.get("image")
        if image not in status:
            status[image] = bool(image) and present(image)
        if not status[image]:
            j.result = image_missing_result(j, run_id)       # in memory only: a later run re-checks the image
    log(f"images: {sum(status.values())} present, {sum(1 for v in status.values() if not v)} missing (of {len(status)} distinct)")
    return status


def prepare_mirrors(jobs: list[Job], mirrors_dir: Path, run: RunDir, workers: int = 4) -> dict[str, str | None]:
    """Clone the mirror of every repository with work to do (4 at a time); {repo_name: error or None}."""
    names = sorted({repo_name_of(j.task) for j in jobs if not j.skip and j.result is None})

    def one(name: str) -> tuple[str, str | None]:
        try:
            swesmith.ensure_mirror(name, mirrors_dir)
            return name, None
        except SandboxError as e:
            return name, str(e)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = dict(ex.map(one, names))
    bad = {k: v for k, v in out.items() if v}
    run.log(f"mirrors: {len(names) - len(bad)} ready, {len(bad)} failed")
    for k, v in bad.items():
        run.error(f"mirror {k}: {v}")
    return out


def wait_for_load(load_pause: float | None) -> None:
    while load_pause is not None and os.getloadavg()[0] > load_pause:
        time.sleep(15)


def run_jobs(jobs: list[Job], opts: GradeOptions, run_id: str, *, slots: int, run: RunDir, resume: bool, load_pause: float | None,
             calibrate: Callable[..., dict[str, Any]] = swesmith.calibrate_task) -> None:
    """Calibrate every runnable job on `slots` threads (one container each); env failures retried MAX_ENV_RETRIES times;
    with `resume`, a finished calibration.json under the job dir is reused."""
    def one(job: Job) -> None:
        if STOP.is_set():
            return
        prior = job.job_dir / "calibration.json"
        if resume and prior.exists():
            res = json.loads(prior.read_text(encoding="utf-8"))
            if res.get("status") in FINAL_STATUSES:
                job.result = res
                return
        while True:
            wait_for_load(load_pause)
            if STOP.is_set():
                return
            job.attempts += 1
            try:
                res = calibrate(job.task, job.job_dir / "work", opts, run_id=run_id)
            except Exception as e:  # noqa: BLE001 - one task's crash is an environment failure, never a run crash
                res = {**swesmith.calibration_base(job.task, run_id, opts.network), "status": "env_failure", "reason": f"crash: {type(e).__name__}: {e}"[:400]}
            if res.get("status") != "env_failure" or job.attempts > MAX_ENV_RETRIES:
                break
            run.error(f"{job.task['task_id']}: env failure (attempt {job.attempts}): {res.get('reason')}; retrying")
        job.result = res
        job.job_dir.mkdir(parents=True, exist_ok=True)
        if res.get("status") in FINAL_STATUSES:
            tmp = prior.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            tmp.replace(prior)
        else:
            run.error(f"{job.task['task_id']}: env failure after {job.attempts} attempts: {res.get('reason')}")
        f2p, p2p = test_lists(job.task)
        run.metric({"kind": "calibration", "task_id": job.task["task_id"], "split": job.task.get("split"), "language": job.task.get("language"),
                    "repo": job.task.get("repo"), "status": res.get("status"), "reason": res.get("reason"), "reporter": res.get("reporter"),
                    "gold_pass": res.get("gold_pass"), "n_p2p_excluded": len(res.get("p2p_excluded") or []), "n_p2p": len(p2p),
                    "n_f2p_missing": len(res.get("f2p_missing") or []), "n_f2p": len(f2p), "parser": res.get("parser"),
                    "coverage": res.get("coverage"), "seconds": res.get("seconds"), "attempts": job.attempts})
        run.log(f"{job.task['task_id']}: {res.get('status')} reason={res.get('reason')} gold_pass={res.get('gold_pass')} "
                f"p2p_excluded={len(res.get('p2p_excluded') or [])}/{len(p2p)} f2p_missing={len(res.get('f2p_missing') or [])}/{len(f2p)} "
                f"reporter={res.get('reporter')} parser={res.get('parser')} coverage={res.get('coverage')} {res.get('seconds')}s")

    run_pool(one, [j for j in jobs if not j.skip and j.result is None], slots)


# --- results -> pool records -------------------------------------------------------------------------------------------
def calibration_entry(res: dict[str, Any]) -> dict[str, Any]:
    return {k: res.get(k) for k in CALIBRATION_KEYS}


def mark_excluded(rec: dict[str, Any], rule: str, detail: str | None) -> None:
    extra = rec.setdefault("extra", {})
    if rec.get("split") != "excluded":
        extra["split_before_calibration"] = rec.get("split")
        rec["split"] = "excluded"
    extra["exclusion"] = {"rule": rule, "detail": detail}


def apply_to_record(rec: dict[str, Any], res: dict[str, Any]) -> None:
    """extra.calibration on a pool record; the split for excluded tasks (previous split kept; a later pass restores it)."""
    extra = rec.setdefault("extra", {})
    extra["calibration"] = calibration_entry(res)
    if res["status"] == "ok" and res.get("eval_spec"):
        extra["eval_spec"] = res["eval_spec"]
    else:
        extra.pop("eval_spec", None)        # a stale spec must never outlive the calibration that validated it
    excl = extra.get("exclusion") or {}
    if res["status"] == "excluded":
        mark_excluded(rec, "calibration", res.get("reason"))
    elif res["status"] == "ok" and rec.get("split") == "excluded" and excl.get("rule") == "calibration":   # a smoke exclusion stands
        rec["split"] = extra.pop("split_before_calibration", "stage2")
        extra.pop("exclusion", None)


def original_split(rec: dict[str, Any]) -> str:
    return rec.get("split") if rec.get("split") != "excluded" else (rec.get("extra") or {}).get("split_before_calibration") or "excluded"


# --- the three-replay smoke --------------------------------------------------------------------------------------------
def synthetic_test_diff(task: dict[str, Any]) -> str:
    """Fallback tampered section when the instance branch has no test-removal commit: a skip marker added to a test file
    of the repository (an edit of an existing test path, which the tamper rules flag; a new scratch test file would only be
    recorded). The section is excluded from the replayed diff like every test path, so its context lines need not match."""
    if task.get("language") == "go":
        first = ((task.get("extra") or {}).get("touched_files") or ["x.go"])[0]
        path = first[: -len(".go")] + "_test.go" if first.endswith(".go") else f"{first}_test.go"
        context, added = "func TestMain(m *testing.M) {", "\tt.Skip(\"x\")"
    else:
        path, context, added = "tests/__init__.py", "", "import pytest; pytest.skip('x', allow_module_level=True)"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n {context}\n+{added}\n"


def smoke_task(task: dict[str, Any], opts: GradeOptions, out_dir: Path) -> dict[str, Any]:
    """The three replays on one admitted task; status pass | gold_not_reproduced | f2p_pass_without_fix | empty_not_zero |
    tamper_not_flagged | env_failure."""
    tid = task["task_id"]
    t0 = time.time()
    res: dict[str, Any] = {"task_id": tid}
    gold = swesmith.gold_patch(task)
    # one run: with the calibration's gold replay it makes the two runs of the reward; the second-run path is unit-tested
    v = swesmith.grade_patch(task, gold, out_dir / "gold", second_run=False, episode_id=f"{tid}#smoke#gold#0", opts=opts)
    res["gold"] = {"r_pass": v.r_pass, "n_runs": len(v.runs), "flags": v.tamper_flags, "seconds": [r.get("seconds") for r in v.runs]}
    if v.r_pass != 1:
        res["status"] = "env_failure" if any(r.get("retryable") for r in v.runs) else "gold_not_reproduced"
        res["seconds"] = round(time.time() - t0, 1)
        return res
    profile = swesmith.profile_info(repo_name_of(task))
    empty_run = swesmith.replay_once(task, "", out_dir / "empty" / "run1", profile, opts, 1)
    v_empty = swesmith.grade_patch(task, "", out_dir / "empty", second_run=True, episode_id=f"{tid}#smoke#empty#0", opts=opts)
    res["empty"] = {"tests_pass": bool(empty_run.get("pass")), "f2p_failure": (empty_run.get("f2p") or {}).get("failure"),
                    "f2p_n": (empty_run.get("f2p") or {}).get("n"), "error": empty_run.get("error"), "r_pass": v_empty.r_pass,
                    "flags": v_empty.tamper_flags, "seconds": empty_run.get("seconds")}
    if empty_run.get("retryable"):
        res["status"], res["seconds"] = "env_failure", round(time.time() - t0, 1)
        return res
    try:
        tamper = swesmith.test_removal_diff(task, opts.mirrors_dir)
    except SandboxError:
        tamper = ""
    kind = "remove_f2p_test_file" if tamper else "synthetic_skip_test"
    tampered = gold + (tamper or synthetic_test_diff(task))
    v_t = swesmith.grade_patch(task, tampered, out_dir / "tampered", second_run=True, episode_id=f"{tid}#smoke#tampered#0", opts=opts)
    res["tampered"] = {"kind": kind, "r_pass": v_t.r_pass, "flags": v_t.tamper_flags, "n_runs": len(v_t.runs),
                       "first_run_tests_pass": bool(v_t.runs and v_t.runs[0].get("pass")), "excluded_paths": v_t.extra.get("excluded_paths"),
                       "seconds": [r.get("seconds") for r in v_t.runs]}
    if any(r.get("retryable") for r in v_t.runs):
        res["status"] = "env_failure"
    elif empty_run.get("pass") or not (empty_run.get("f2p") or {}).get("failure"):
        res["status"] = "f2p_pass_without_fix"
    elif v_empty.r_pass != 0 or "empty_diff" not in v_empty.tamper_flags:
        res["status"] = "empty_not_zero"
    elif v_t.r_pass != 0 or "tests_touched" not in v_t.tamper_flags:
        res["status"] = "tamper_not_flagged"
    else:
        res["status"] = "pass"
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_smoke(records: list[dict[str, Any]], repos: list[str], opts: GradeOptions, work_dir: Path, run: RunDir, *, slots: int,
              load_pause: float | None, seed: int = paths.SEED, smoke_fn: Callable[..., dict[str, Any]] = smoke_task) -> dict[str, Any]:
    """Per repository, up to SMOKE_ATTEMPTS admitted tasks in seeded order until one passes all three replays. Excludes the
    tasks that failed for a task reason (rule "smoke") and every task of a repository whose attempts all failed for a task
    reason; a repository with an environment failure and no pass is reported as env_failure and left for the next run."""
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("split") in ("dev", "stage2") and (r.get("extra") or {}).get("eval_spec") and repo_key(r.get("repo")) in repos:
            by_repo.setdefault(repo_key(r["repo"]), []).append(r)

    def one(repo: str) -> tuple[str, dict[str, Any]]:
        cands = sorted(by_repo.get(repo, []), key=lambda r: r["task_id"])
        random.Random(f"{seed}:smoke:{repo}").shuffle(cands)
        attempts: list[dict[str, Any]] = []
        for i, task in enumerate(cands[:SMOKE_ATTEMPTS]):
            if STOP.is_set():
                break
            wait_for_load(load_pause)
            try:
                res = smoke_fn(task, opts, work_dir / "smoke" / slug(repo, 80) / str(i))
            except Exception as e:  # noqa: BLE001 - recorded as an environment failure of this attempt
                res = {"task_id": task["task_id"], "status": "env_failure", "error": f"{type(e).__name__}: {e}"[:300]}
            attempts.append(res)
            run.metric({"kind": "smoke", "repo": repo, **{k: v for k, v in res.items() if k in ("task_id", "status", "seconds")}})
            if res["status"] == "pass":
                break
        states = [a["status"] for a in attempts]
        # a repository fails only on task reasons; an environment failure leaves it for the next run (nothing excluded)
        status = ("no_admitted_task" if not attempts else "pass" if "pass" in states else "env_failure" if "env_failure" in states else "fail")
        run.log(f"smoke {repo}: {status} after {len(attempts)} attempt(s): {[a['status'] for a in attempts]}")
        return repo, {"status": status, "attempts": attempts}
    results = dict(run_pool(one, sorted(repos), slots))
    by_id = {r["task_id"]: r for r in records}
    task_excluded = 0
    repo_excluded = 0
    for repo, rr in results.items():
        for att in rr["attempts"]:
            if att["status"] in ("gold_not_reproduced", "f2p_pass_without_fix"):
                mark_excluded(by_id[att["task_id"]], "smoke", f"smoke:{att['status']}")
                task_excluded += 1
        if rr["status"] == "fail":
            for r in by_repo.get(repo, []):
                if r.get("split") != "excluded":
                    mark_excluded(r, "smoke", "smoke:repository_failed")
                    repo_excluded += 1
    return {"run_at": utc_now(), "repositories": len(results), "status_counts": dict(Counter(r["status"] for r in results.values())),
            "attempt_status_counts": dict(Counter(a["status"] for r in results.values() for a in r["attempts"])),
            "tasks_excluded_by_attempt": task_excluded, "tasks_excluded_by_repository_failure": repo_excluded,
            "tamper_kinds": dict(Counter((a.get("tampered") or {}).get("kind") for r in results.values() for a in r["attempts"] if a.get("tampered"))),
            "per_repository": {k: {"status": v["status"], "attempts": [{x: a.get(x) for x in ("task_id", "status", "gold", "empty", "tampered", "seconds")}
                                                                       for a in v["attempts"]]} for k, v in sorted(results.items())}}


# --- manifest ---------------------------------------------------------------------------------------------------------
def calibration_block(records: list[dict[str, Any]], jobs: list[Job], run_id: str, a: argparse.Namespace, pool: Path,
                      present: Callable[[str], bool] = image_present) -> dict[str, Any]:
    """Counts over the whole pool file (every dev and stage2 record, by its original split), from the records' calibration."""
    per_split: dict[str, Counter] = {}
    reasons: dict[str, Counter] = {}
    by_lang: dict[str, Counter] = {}
    secs: list[float] = []
    reporters: Counter = Counter()
    p2p_tasks = p2p_ids = p2p_total = 0
    for r in records:
        split = original_split(r)
        if split not in ("dev", "stage2"):
            continue
        extra = r.get("extra") or {}
        cal = extra.get("calibration") or {}
        excl = extra.get("exclusion") or {}
        if r.get("split") == "excluded":
            state = "excluded"
            reasons.setdefault(split, Counter())[excl.get("detail") if excl.get("rule") in ("calibration", "smoke") else f"{excl.get('rule')}:{excl.get('detail')}"] += 1
        elif cal.get("status") == "ok":
            state = "admitted"
            reporters[(extra.get("eval_spec") or {}).get("reporter") or "text"] += 1
            if cal.get("p2p_excluded"):
                p2p_tasks += 1
                p2p_ids += len(cal["p2p_excluded"])
            p2p_total += len(test_lists(r)[1])
        else:
            state = "not_calibrated"
        per_split.setdefault(split, Counter())[state] += 1
        by_lang.setdefault(f"{split}:{r.get('language')}", Counter())[state] += 1
        if cal.get("seconds"):
            secs.append(float(cal["seconds"]))
    env = Counter(original_split(j.task) for j in jobs if (j.result or {}).get("status") == "env_failure")
    secs.sort()
    images = sorted({r.get("image") for r in records if original_split(r) in ("dev", "stage2") and r.get("image")})
    n_present = sum(1 for i in images if present(i))
    return {"run_id": run_id, "images": {"distinct": len(images), "present": n_present, "missing": len(images) - n_present}, "generated_at": utc_now(), "pool_sha256": sha256_file(pool), "verifier_version": VERIFIER_VERSION,
            "conditions": {"network": a.network, "cpus": a.cpus, "memory": a.memory, "timeout": "max(2 x profile timeout, 300 s)"},
            "per_split": {k: dict(sorted(v.items())) for k, v in sorted(per_split.items())},
            "exclusion_reasons": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], str(kv[0])))) for k, v in sorted(reasons.items())},
            "env_failures_this_run": dict(env),
            "by_language": {k: dict(sorted(v.items())) for k, v in sorted(by_lang.items())},
            "reporters_admitted": dict(sorted(reporters.items())),
            "p2p_exclusions": {"tasks_with_exclusions": p2p_tasks, "ids_excluded": p2p_ids, "ids_total_admitted": p2p_total},
            "gold_replay_seconds": {"n": len(secs), "median": round(median(secs), 1) if secs else None,
                                    "p90": secs[int(0.9 * (len(secs) - 1))] if secs else None, "max": secs[-1] if secs else None,
                                    "sum_container_minutes": round(sum(secs) / 60, 1)}}


def worst_cases(jobs: list[Job]) -> list[dict[str, Any]]:
    bad = [j for j in jobs if j.result and j.result.get("status") != "ok"]
    return [{"task_id": j.task["task_id"], "repo": j.task.get("repo"), "status": j.result.get("status"), "reason": j.result.get("reason"),
             "n_f2p_missing": len(j.result.get("f2p_missing") or []), "n_p2p_excluded": len(j.result.get("p2p_excluded") or []),
             "parser": j.result.get("parser"), "coverage": j.result.get("coverage"), "seconds": j.result.get("seconds")}
            for j in sorted(bad, key=lambda j: j.task["task_id"])[:WORST_N]]


def merge_smoke(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Smoke block over every repository smoked so far: a later run's result for a repository replaces the earlier one;
    the counts are recomputed over the merged table (exclusion counts add up across runs)."""
    if not old or not old.get("per_repository"):
        return new
    per = {**old["per_repository"], **new["per_repository"]}
    return {"run_at": new["run_at"], "repositories": len(per), "status_counts": dict(Counter(v["status"] for v in per.values())),
            "attempt_status_counts": dict(Counter(a["status"] for v in per.values() for a in v["attempts"])),
            "tasks_excluded_by_attempt": int(old.get("tasks_excluded_by_attempt") or 0) + new["tasks_excluded_by_attempt"],
            "tasks_excluded_by_repository_failure": int(old.get("tasks_excluded_by_repository_failure") or 0) + new["tasks_excluded_by_repository_failure"],
            "tamper_kinds": dict(Counter((a.get("tampered") or {}).get("kind") for v in per.values() for a in v["attempts"] if a.get("tampered"))),
            "per_repository": dict(sorted(per.items()))}


def update_manifest(path: Path, **blocks: Any) -> None:
    man = json.loads(path.read_text()) if path.exists() else {}
    if "smoke" in blocks:
        blocks["smoke"] = merge_smoke(man.get("smoke"), blocks["smoke"])
    man.update(blocks)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    tmp.replace(path)


# --- main -------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None, *, image_check: Callable[[str], bool] = image_present,
         calibrate: Callable[..., dict[str, Any]] = swesmith.calibrate_task, smoke_fn: Callable[..., dict[str, Any]] = smoke_task,
         mirrors: Callable[..., dict[str, str | None]] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_calibrate_swesmith")
    work_dir = a.work_dir or (paths.WORK / run_id)
    manifest_path = a.manifest or (a.pool.parent / "manifest.json")
    records = list(read_jsonl(a.pool))
    jobs = load_jobs(records, a, work_dir)
    notes: list[str] = []
    if not a.smoke_only:
        check_images(jobs, image_check, run_id, notes.append)
    plan = {"run_id": run_id, "work_dir": str(work_dir), "pool": str(a.pool), "n_records": len(records), "n_selected": len(jobs),
            "to_run": sum(1 for j in jobs if not j.skip and j.result is None) if not a.smoke_only else 0,
            "skipped": dict(Counter(j.skip for j in jobs if j.skip)), "image_missing": sum(1 for j in jobs if j.result),
            "splits": a.splits, "slots": a.slots, "cpus": a.cpus, "memory": a.memory, "network": a.network, "load_pause": a.load_pause,
            "mirrors_dir": str(a.mirrors_dir), "smoke": a.smoke or a.smoke_only}
    if a.dry_run:
        print(json.dumps({**plan, "notes": notes}, indent=1))
        return 0
    opts = GradeOptions(cpus=a.cpus, memory=a.memory, cgroup_parent=a.cgroup_parent, network=a.network, mirrors_dir=Path(a.mirrors_dir),
                        labels={"apex.run_id": run_id})
    with RunDir(run_id, root=a.out_root, log_name="train.log", resume=a.resume) as run:
        for n in notes:
            run.log(n)
        run.log(f"plan: {json.dumps(plan)}")
        run.write_metadata(run_id=run_id, kind="data_calibrate_swesmith", code=git_info(), hardware=hardware_info(), seed=paths.SEED,
                           command=command_line(), note=a.note, config={k: plan[k] for k in ("splits", "slots", "cpus", "memory", "network", "load_pause", "mirrors_dir", "smoke")},
                           data_version={"pool": str(a.pool), "pool_sha256_before": sha256_file(a.pool), "n_records": len(records),
                                         "verifier_version": VERIFIER_VERSION}, start_time=utc_now(), status="running")
        t0 = time.time()
        if not a.smoke_only:
            mirror_state = (mirrors or prepare_mirrors)(jobs, Path(a.mirrors_dir), run)
            for j in jobs:
                err = mirror_state.get(repo_name_of(j.task)) if not j.skip and j.result is None else None
                if err:
                    j.result = {**swesmith.calibration_base(j.task, run_id, a.network), "status": "env_failure", "reason": f"mirror: {err}"[:400]}
            run_jobs(jobs, opts, run_id, slots=a.slots, run=run, resume=a.resume, load_pause=a.load_pause, calibrate=calibrate)
            for j in jobs:
                if j.result and j.result.get("status") in FINAL_STATUSES:
                    apply_to_record(j.task, j.result)
            write_jsonl(a.pool, records)
        calib_sec = round(time.time() - t0, 1)
        smoke = None
        if a.smoke or a.smoke_only:
            repos = sorted({repo_key(j.task.get("repo")) for j in jobs if not j.skip})
            smoke = run_smoke(records, repos, opts, work_dir, run, slots=a.slots, load_pause=a.load_pause, smoke_fn=smoke_fn)
            write_jsonl(a.pool, records)
            run.log(f"smoke: {smoke['status_counts']} attempts={smoke['attempt_status_counts']} excluded tasks: "
                    f"{smoke['tasks_excluded_by_attempt']} by attempt, {smoke['tasks_excluded_by_repository_failure']} by repository")
        block = calibration_block(records, jobs, run_id, a, a.pool, image_check)
        summary = {"calibration": block, "this_run": {"status_counts": dict(Counter((j.result or {}).get("status") or j.skip or "not_run" for j in jobs)),
                                                       "worst": worst_cases(jobs)}}
        if smoke is not None:
            summary["smoke"] = smoke
        (run.run_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        if a.out_root is None or a.manifest is not None:
            update_manifest(manifest_path, calibration=block, **({"smoke": smoke} if smoke is not None else {}))
        run.log(f"calibration: {json.dumps(block['per_split'])} reasons={json.dumps(block['exclusion_reasons'])} "
                f"gold replay seconds={block['gold_replay_seconds']}")
        container_min = round(sum(float((j.result or {}).get("seconds") or 0) for j in jobs) / 60, 1)
        run.write_resources([("duration_sec", round(run.elapsed(), 1), "s", "wall clock"),
                             ("calibration_wall_sec", calib_sec, "s", "time in the calibration worker pool"),
                             ("container_minutes", container_min, "min", "sum of this run's gold-replay wall clocks"),
                             ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"), ("gpu_hours", 0, "h", "CPU-only step"),
                             ("cost_usd", 0.0, "USD", "GPU cost only; CPU-only step, the always-on CPU VM is not charged")])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished",
                           data_version={"pool": str(a.pool), "pool_sha256_after": sha256_file(a.pool), "n_records": len(records),
                                         "verifier_version": VERIFIER_VERSION},
                           summary={"per_split": block["per_split"], "smoke": (smoke or {}).get("status_counts")})
        run.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
