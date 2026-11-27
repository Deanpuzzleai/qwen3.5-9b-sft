#!/usr/bin/env python3
r"""Dev-set evaluation driver (docs/DATA_PLAN.md sections 1 and 10; docs/EXECUTION_PLAN.md U3.4, M3.3, U4.2, U4.3, M5.3, M6.4).

Usage (repository root, train venv; the endpoint serves the model version under test):
  PYTHONPATH=src .venv/train/bin/python scripts/data/run_dev_eval.py --version base --api-base http://HOST:8000/v1 \
      --model-name qwen35-base --size full [--slices swe,terminal,qa] [--baseline <run>] [--run-id ID --resume]
  ... --version stage2_a --size reduced --baseline results/dev/base/<base full run_id>   (stage-2 arms a/b, M6.4)
  ... --size smoke                                                         (pilot adapter, U4.2: 20 SWE / 10 terminal / 50 QA)
  ... --summarize-only --run-id ID [--baseline <run>]                      (recompute summary.json, e.g. once Base exists)

Inputs : data/manifests/dev_{swe,terminal,qa}.json (frozen dev set: items, sample indices, seeds); for SWE the SWE-smith pool
         data/pools/swesmith/tasks.jsonl (calibrated tasks), the rollout driver apex_data.rollout.mini_swe (U3.3), the
         grader apex_data.grade.swesmith and its images; for terminal the TermGrade pool (data/pools/tgr_env/tasks.jsonl,
         the prepared Harbor tasks under WORK/tgr_env/tasks) and .venv/harbor; for QA apex_data.devset.qa.load_dev_items
         (GPQA/AIME text from the WORK cache), the pinned HLE checkout third_party/hle (prompt and judge text, commit checked
         against configs/eval/hle.yaml) and OPENAI_API_KEY (environment or .env) for the judge; locks configs/eval/
         {swebench_pro_v2_hard,terminal_bench_2,hle}.yaml (read-only); an OpenAI-compatible endpoint at --api-base.
Outputs: logs/<run_id>/              metadata.json, eval.log, metrics.jsonl (one line per item, plus per-slice lines),
                                     errors.log ("none" if none), resources.csv (duration, GPU hours, cost incl. judge spend),
                                     config_snapshot/ (locks and manifests used)
         results/dev/<version>/<run_id>/   per_item.jsonl (one record per (slice, task_id, sample_index); the format
                                     scripts/summarize.py dev reads), summary.json (per-slice scores, failure accounting,
                                     time and tokens, turn metrics, guards; `smoke` and `reportable`: only a full or reduced
                                     run of the frozen dev set with the real pools, all three slices and every item
                                     recorded is reportable, and only a reportable run is accepted as --baseline),
                                     paired_summary.json (scripts/summarize.py dev against --baseline: paired differences
                                     and intervals), items/swe/.../model.patch
         <work-root>/<run_id>/       raw episodes, grader logs, Harbor trials, QA generations and judge outputs (outside
                                     the repository: GPQA text is gated and the judge output embeds reference answers)
Protocol: every dev run uses the same per-slice concurrency, SWE 24 / terminal 12 / QA 16 (52 sequences, under the locked
vLLM max_num_seqs 64), the 1,800 s terminal agent timeout and the SWE sandbox layout (repository at /app as in SWE-bench Pro,
conda env activated; apex_data.rollout.mini_swe.SANDBOX_LAYOUT); all are recorded as `protocol` in metadata.json (config)
and summary.json. A run with other values is not reportable, and --baseline refuses a pair whose protocols differ.
Phases: SWE + QA, then terminal (phase_order); a run that has both executes them one after the other. Terminal trials never
share a server with SWE / QA load: a terminal phase may overlap a swe+qa phase only on a different endpoint. Two
endpoints serving the same checkpoint can therefore run the phases at the same time as two part runs, e.g.
`--slices swe,qa --api-base <server 1> --run-id R_p1` and `--slices terminal --api-base <server 2> --run-id R_p2`, then
`--combine R_p1 R_p2 --run-id R` builds the reportable run R (union of the records, each tagged with its part, checked
for equal version, size and protocol and for disjoint slices that cover all three). Every phase records its endpoint
and the apex-cpu load over its window (--host-load-csv, scripts/dev/host_load_sampler.sh). One run id is never run by
two processes at once: --resume refuses while the metadata says `running`.
Admission: only the frozen admitted SWE / terminal dev tasks (data/manifests/dev_admitted.json) are planned and scored.
Run ids: dev_<version>[_<label>]_<size>_<UTC timestamp>. Resume (--resume) skips every item that has a record, reuses
cached episodes, trials, generations and verdicts, and re-runs items whose last record has a status in --rerun-statuses.
Nothing is dropped: every planned item ends with a record; environment failures are retried --env-retries times first.
See src/apex_data/devset/run.py for the harness, record and guard definitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from apex_data import paths  # noqa: E402
from apex_data.devset import admission  # noqa: E402
from apex_data.devset import run as dev  # noqa: E402
from apex_eval.common import (RunDir, command_line, compute_rates, endpoint_info, git_info, hardware_info, model_lock,  # noqa: E402
                              new_run_id, read_json, resource_rows, sha256_file, utc_now, write_json)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    rates = compute_rates()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True, choices=dev.VERSIONS,
                   help="model version under test: base, sft, stage2_a (filtered arm), stage2_b (control arm), pilot (SFT pilot adapter)")
    p.add_argument("--label", default="", help="extra run label, e.g. arm_a, arm_b, ckpt_50 (goes into the run id and summary)")
    p.add_argument("--api-base", default=None, help="OpenAI-compatible base URL of the served model, e.g. http://HOST:8000/v1")
    p.add_argument("--model-name", default=None, help="served model id at --api-base")
    p.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "EMPTY"), help="endpoint key (dummy for vLLM; prefer EVAL_API_KEY)")
    p.add_argument("--size", choices=dev.SIZES, default="full", help="full | reduced (first samples of full) | smoke (20/10/50 items, sample 0)")
    p.add_argument("--slices", default=",".join(dev.SLICES), help="comma list of swe, terminal, qa")
    p.add_argument("--limit-per-slice", default=None, help="SMOKE TESTS: first N items per slice, or swe=N,terminal=N,qa=N")
    p.add_argument("--task-ids", default=None, help="SMOKE TESTS: comma list of task ids / upstream ids to keep")
    p.add_argument("--baseline", default=None, help="baseline run for the paired comparison and guards: results dir, per_item.jsonl, "
                                                    "logs/<run_id> or a run id (e.g. Base's full dev run). Refused unless its "
                                                    "summary.json says smoke: false and reportable: true")
    p.add_argument("--run-id", default=None, help="default dev_<version>[_<label>]_<size>_<UTC timestamp>")
    p.add_argument("--resume", action="store_true", help="continue --run-id: items with a record are kept, the rest are run")
    p.add_argument("--rerun-statuses", default="", help="with --resume: re-run items whose last record has one of these run_status values "
                                                        "(e.g. environment_failure,missing). For infrastructure failures only: re-running "
                                                        "model failures biases the score. The old line stays; the new one names it "
                                                        "under `supersedes`; agent_error / missing items restart from scratch")
    p.add_argument("--regrade-slices", default="", help="with --resume: grade the recorded items of these slices again from their "
                                                        "cached rollouts (swe), e.g. after a grader rule change; the old verdict is "
                                                        "kept aside and the old record line stays, the new one names it under `supersedes`")
    p.add_argument("--summarize-only", action="store_true", help="recompute summary.json (and the paired comparison) of --run-id; runs nothing")
    p.add_argument("--skip-blocked", action="store_true", help="run the slices whose dependencies are ready and report the blocked ones "
                                                               "(default: fail when any requested slice is blocked)")
    p.add_argument("--swe-concurrency", type=int, default=dev.DEFAULT_CONCURRENCY["swe"],
                   help=f"concurrent SWE episodes (protocol {dev.DEFAULT_CONCURRENCY['swe']}; another value makes the run non-reportable)")
    p.add_argument("--terminal-concurrency", type=int, default=dev.DEFAULT_CONCURRENCY["terminal"],
                   help=f"concurrent Harbor trials (protocol {dev.DEFAULT_CONCURRENCY['terminal']}; another value makes the run non-reportable)")
    p.add_argument("--qa-concurrency", type=int, default=dev.DEFAULT_CONCURRENCY["qa"],
                   help=f"concurrent QA requests (protocol {dev.DEFAULT_CONCURRENCY['qa']}; another value makes the run non-reportable)")
    p.add_argument("--env-retries", type=int, default=1, help="re-runs of an item after an environment failure (rollout, Harbor, grading)")
    p.add_argument("--terminal-agent-timeout-sec", type=float, default=dev.TERMINAL_AGENT_TIMEOUT_SEC,
                   help=f"Terminus-2 agent timeout (TermGrade task.toml declares none; protocol {dev.TERMINAL_AGENT_TIMEOUT_SEC:g} s; "
                        "another value makes the run non-reportable)")
    p.add_argument("--qa-max-tokens", type=int, default=None, help="QA completion budget (default: the HLE lock's 32,768; another value is not decision grade)")
    p.add_argument("--judge-usd-per-mtok-in", type=float, default=dev.JUDGE_USD_PER_MTOK["input"], help="judge input price for the spend estimate")
    p.add_argument("--judge-usd-per-mtok-out", type=float, default=dev.JUDGE_USD_PER_MTOK["output"], help="judge output price for the spend estimate")
    p.add_argument("--manifests-dir", type=Path, default=paths.MANIFESTS, help="dev manifests (default data/manifests)")
    p.add_argument("--admitted", type=Path, default=None, help="frozen admitted SWE / terminal dev task lists (default "
                                                               "<manifests-dir>/dev_admitted.json; scripts/data/freeze_dev_admission.py)")
    p.add_argument("--host-load-csv", type=Path, default=None, help="apex-cpu load samples (scripts/dev/host_load_sampler.sh; default "
                                                                     "logs/host_load_<UTC date>/host_load.csv): each phase records its window")
    p.add_argument("--combine", nargs="+", default=None, metavar="PART_RUN_ID",
                   help="build the run --run-id from part runs of the same version and size that together cover swe, terminal and qa "
                        "(e.g. swe+qa on one endpoint and terminal on another, at the same time); runs nothing")
    p.add_argument("--resume-running", action="store_true", help="with --resume: the run's metadata says `running`, and you have checked "
                                                                 "that no other invocation of it is still running (a killed process)")
    p.add_argument("--swe-pool-tasks", type=Path, default=paths.POOLS / "swesmith" / "tasks.jsonl", help="calibrated SWE-smith tasks")
    p.add_argument("--terminal-pool-tasks", type=Path, default=paths.POOLS / "tgr_env" / "tasks.jsonl", help="TermGrade pool (admitted tasks)")
    p.add_argument("--terminal-dataset-dir", type=Path, default=paths.WORK / "tgr_env" / "tasks", help="prepared Harbor TermGrade tasks")
    p.add_argument("--work-root", type=Path, default=paths.WORK / "dev_eval", help="raw outputs: <work-root>/<run_id>/")
    p.add_argument("--serving-info", type=Path, default=None, help="metadata.json written by scripts/serve_vllm.sh (embedded in metadata)")
    p.add_argument("--serving-gpus", type=int, default=rates["default_serving_gpus"], help="GPUs serving the model during this run")
    p.add_argument("--gpu-usd-per-hour", type=float, default=rates["gpu_vm_usd_per_hour"])
    p.add_argument("--peak-gpu-mem-gb", type=float, default=None, help="peak GPU memory on the serving VM, if measured")
    p.add_argument("--note", default="", help="free text stored in metadata.json")
    p.add_argument("--out-root", type=Path, default=None, help="SMOKE TESTS ONLY: write logs/ and results/ under this root")
    a = p.parse_args(argv)
    a.slices = [s.strip() for s in a.slices.split(",") if s.strip()]
    bad = [s for s in a.slices if s not in dev.SLICES]
    if bad:
        p.error(f"unknown slices {bad}; choose from {dev.SLICES}")
    a.limits = dev.parse_limits(a.limit_per_slice)
    a.task_id_list = [t.strip() for t in (a.task_ids or "").split(",") if t.strip()]
    a.rerun = {s.strip() for s in a.rerun_statuses.split(",") if s.strip()}
    a.regrade = {s.strip() for s in a.regrade_slices.split(",") if s.strip()}
    if a.regrade - {"swe"}:
        p.error("--regrade-slices supports swe (terminal and QA verdicts come from the trial and the judge)")
    if a.regrade and not a.resume:
        p.error("--regrade-slices needs --resume and --run-id")
    if a.regrade and a.rerun:
        p.error("--regrade-slices grades cached rollouts only and never re-runs an item; run --rerun-statuses separately")
    if not (a.summarize_only or a.combine) and (not a.api_base or not a.model_name):
        p.error("--api-base and --model-name are required (except with --summarize-only and --combine)")
    if (a.resume or a.summarize_only or a.combine) and not a.run_id:
        p.error("--resume, --summarize-only and --combine need --run-id")
    return a


def default_run_id(a: argparse.Namespace) -> str:
    return new_run_id(*[x for x in ("dev", a.version, dev.slug(a.label, 30) if a.label else "", a.size) if x])


# --- runners ---------------------------------------------------------------------------------------------------------
def build_runners(a: argparse.Namespace, ctx: dev.RunContext, plan: list[dev.PlanItem], log: Any) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """(slice -> runner, slice -> blocked reason, slice -> settings) for the requested slices."""
    runners: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    settings: dict[str, Any] = {}
    need_docker = any(s in a.slices for s in ("swe", "terminal"))
    docker_ok = not need_docker or (shutil.which("docker") is not None and
                                    subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0)
    for sl in a.slices:
        try:
            if sl in ("swe", "terminal") and not docker_ok:
                raise dev.SliceBlocked("docker daemon not reachable")
            r = {"swe": build_swe, "terminal": build_terminal, "qa": build_qa}[sl](a, ctx, plan)
        except dev.SliceBlocked as e:
            blocked[sl] = str(e)
            log.error(f"slice {sl} blocked: {e}")
            continue
        except ImportError as e:
            blocked[sl] = f"dependency not importable: {e}"
            log.error(f"slice {sl} blocked: {blocked[sl]}")
            continue
        settings[sl] = r.settings()
        for name in r.slices:
            runners[name] = r
    return runners, blocked, settings


def build_qa(a: argparse.Namespace, ctx: dev.RunContext, plan: list[dev.PlanItem]) -> Any:
    from apex_data.devset import qa as devqa  # noqa: PLC0415

    key = dev.secret_from_env("OPENAI_API_KEY")
    if not key:
        raise dev.SliceBlocked("OPENAI_API_KEY is not set (environment or .env); the HLE judge needs it")
    assets = dev.load_hle_assets()
    try:
        items, report = devqa.load_dev_items(a.manifests_dir / devqa.MANIFEST_NAMES["qa"], offline=True)
    except Exception as e:  # noqa: BLE001 - surfaced as a blocked slice
        raise dev.SliceBlocked(f"dev QA items not loadable: {type(e).__name__}: {e}") from None
    ctx.log(f"qa: {report.get('n_items')} dev items loaded and checked")
    judge = dev.HLEJudge(assets, key, base_url=os.environ.get(dev.JUDGE_BASE_URL_ENV) or dev.OPENAI_URL,
                         usd_per_mtok={"input": a.judge_usd_per_mtok_in, "output": a.judge_usd_per_mtok_out})
    return dev.QARunner(ctx, {it.dev_id: it for it in items}, assets, judge, max_tokens=a.qa_max_tokens, scorer=devqa.score_reasoning_gym)


def build_terminal(a: argparse.Namespace, ctx: dev.RunContext, plan: list[dev.PlanItem]) -> Any:
    from apex_data.grade import harbor_tests  # noqa: PLC0415
    from apex_data.schema import read_jsonl  # noqa: PLC0415
    from apex_eval.harbor_utils import HARBOR  # noqa: PLC0415

    try:
        from apex_data.rollout import endpoint as endpoint_mod  # noqa: PLC0415
        from apex_data.rollout import terminus as driver  # noqa: PLC0415
    except ImportError as e:
        raise dev.SliceBlocked(f"rollout driver apex_data.rollout.terminus not available (U3.3): {e}") from None
    if not HARBOR.exists():
        raise dev.SliceBlocked(f"{HARBOR} missing (scripts/setup_eval_env.sh)")
    if not a.terminal_pool_tasks.exists():
        raise dev.SliceBlocked(f"TermGrade pool {a.terminal_pool_tasks} does not exist (U3.2, M3.2)")
    if not a.terminal_dataset_dir.exists():
        raise dev.SliceBlocked(f"prepared Harbor tasks {a.terminal_dataset_dir} do not exist (scripts/data/build_terminal_envs.py)")
    pool = {t["task_id"]: t for t in read_jsonl(a.terminal_pool_tasks)}
    admitted = set(harbor_tests.admitted_tasks("dev", a.terminal_pool_tasks))
    wanted = {it.upstream_id for it in plan if it.slice == "terminal"}
    ctx.log(f"terminal: {len(admitted)} admitted dev tasks in the pool; {len(wanted & admitted)} of {len(wanted)} planned tasks admitted")
    return dev.TerminalRunner(ctx, pool, admitted, driver=driver, endpoint_mod=endpoint_mod, dataset_dir=a.terminal_dataset_dir,
                              agent_timeout_sec=a.terminal_agent_timeout_sec, env_retries=a.env_retries)


def build_swe(a: argparse.Namespace, ctx: dev.RunContext, plan: list[dev.PlanItem]) -> Any:
    from apex_data.grade import swesmith as grader  # noqa: PLC0415
    from apex_data.schema import read_jsonl  # noqa: PLC0415

    try:
        from apex_data.rollout import endpoint as endpoint_mod  # noqa: PLC0415
        from apex_data.rollout import mini_swe as driver  # noqa: PLC0415
    except ImportError as e:
        raise dev.SliceBlocked(f"rollout driver apex_data.rollout.mini_swe not available (U3.3): {e}") from None
    if not a.swe_pool_tasks.exists():
        raise dev.SliceBlocked(f"SWE-smith pool {a.swe_pool_tasks} does not exist (U3.1, M3.1)")
    pool = {t["task_id"]: t for t in read_jsonl(a.swe_pool_tasks)}
    wanted = {it.task_id for it in plan if it.slice == "swe"}
    ok = sum(1 for t in wanted if t in pool and (grader.calibration_of(pool[t]) or {}).get("status") == "ok")
    ctx.log(f"swe: {len(pool)} pool tasks; {ok} of {len(wanted)} planned tasks calibrated")
    return dev.SWERunner(ctx, pool, driver=driver, endpoint_mod=endpoint_mod, grader=grader, env_retries=a.env_retries)


# --- summary ---------------------------------------------------------------------------------------------------------
def summarize(a: argparse.Namespace, run: RunDir, results_dir: Path, plan_keys: list[tuple[str, str, int]] | None,
              blocked: dict[str, str], run_id: str) -> dict[str, Any]:
    all_records, n_superseded = dev.read_records(results_dir / "per_item.jsonl")
    meta = run.read_metadata()
    cfg = meta.get("config") or {}
    adm = cfg.get("admission")
    keep = {sl: set(ids) for sl, ids in (adm or {}).get("admitted", {}).items()}
    records = {k: r for k, r in all_records.items() if k[0] not in keep or k[1] in keep[k[0]]}
    not_admitted = Counter(k[0] for k in all_records if k not in records)
    if plan_keys is not None:
        plan_keys = [k for k in plan_keys if k[0] not in keep or k[1] in keep[k[0]]]
    stats = dev.run_stats(records, plan_keys)
    reasons = meta.get("smoke_reasons") if "default_inputs" in cfg else None
    if reasons is None:   # runs recorded before the flag existed: derive it from what their metadata holds
        reasons = dev.smoke_reasons(size=meta.get("size", a.size), limits=cfg.get("limits"), task_ids=cfg.get("task_ids"),
                                    default_inputs=cfg.get("default_inputs"), slices=meta.get("slices") or [],
                                    blocked=meta.get("blocked_slices"), qa_max_tokens=cfg.get("qa_max_tokens"))
    protocol = cfg.get("protocol") or {"concurrency": cfg.get("concurrency"), "terminal_agent_timeout_sec": None,
                                       "deviations": ["protocol not recorded (the run predates the fixed dev protocol)"]}
    protocol = {**protocol, "deviations": [*protocol["deviations"], *dev.phase_deviations(meta.get("phases") or [])]}
    protocol = dev.with_swe_grader_version(protocol, records)
    reportable = not reasons and not protocol["deviations"] and bool(meta.get("decision_grade")) and stats["complete"]
    out: dict[str, Any] = {"run_id": run_id, "version": meta.get("version", a.version), "label": meta.get("label", a.label),
                           "size": meta.get("size", a.size), "model_name": meta.get("model_name", a.model_name),
                           "smoke": bool(reasons), "reportable": reportable, "smoke_reasons": reasons, "protocol": protocol,
                           "decision_grade": meta.get("decision_grade"), "blocked_slices": meta.get("blocked_slices", blocked),
                           "n_superseded_record_lines": n_superseded, "phases": meta.get("phases") or [],
                           "admission": None if adm is None else {
                               "path": adm.get("path"), "admitted_sha256": adm.get("admitted_sha256"),
                               "n_admitted": {sl: len(ids) for sl, ids in keep.items()},
                               "excluded": adm.get("excluded"), "n_items_not_admitted": dict(not_admitted),
                               "rule": "only items of admitted tasks are scored; items already run for other tasks stay in per_item.jsonl"},
                           **stats,
                           "failure_counts": {sl: dev.failure_counts([r for r in records.values() if r["slice"] == sl])
                                              for sl in sorted({r["slice"] for r in records.values()})}}
    for sl, s in stats["slices"].items():
        run.metric({"event": "slice", "slice": sl, "score_pct": s["score_pct"], "n_items": s["n_items"],
                    "run_failures": s["failures"]["n_run_failures"], "format_failure_rate": s["turns"]["format_failure_rate"],
                    "mean_think_per_turn": s["turns"]["mean_think_per_turn"]})
    if a.baseline:
        try:
            base_file = dev.resolve_baseline(a.baseline, REPO)
        except FileNotFoundError as e:
            run.error(str(e))
            out["guards"] = {"error": str(e)}
            return out
        refusal = dev.baseline_refusal(base_file, protocol)
        if refusal:
            run.error(f"baseline refused: {refusal}")
            out["guards"] = {"error": f"baseline refused: {refusal}"}
            return out
        base_records, _ = dev.read_records(base_file)
        base_records = {k: r for k, r in base_records.items() if k[0] not in keep or k[1] in keep[k[0]]}
        bsize = dev.baseline_size(base_file)
        pr = dev.pairing(records, base_records)
        paired, err = (None, "seed mismatch: not paired") if pr["n_seed_mismatches"] else dev.run_summarize_py(
            results_dir / "per_item.jsonl", base_file, dev.resolution_set(out["size"], bsize), results_dir / "paired_summary.json",
            run_id, a.out_root, labels=(f"{out['version']}{('_' + out['label']) if out['label'] else ''}", "baseline"),
            admitted=Path(adm["path"]) if adm else None)
        if err:
            run.error(err)
        guards = dev.guard_report(records, base_records, out["size"], bsize, paired)
        guards["decision_grade"] = bool(guards.get("decision_grade") and reportable)
        guards["baseline"] = {"per_item": str(base_file), "size": bsize, "sha256": sha256_file(base_file)}
        guards["paired_summary"] = str(results_dir / "paired_summary.json") if paired else None
        if err:
            guards["summarize_error"] = err
        out["guards"] = guards
        run.log(f"guards vs {base_file}: all_pass={guards.get('all_pass')} (resolution set {guards.get('resolution_set')})")
    else:
        pooled = stats["pooled_turns"]["format_failure_rate"]
        out["guards"] = {"baseline": None, "note": "no --baseline: no guard verdicts; the pooled format-failure rate and its flag are reported",
                         "format_failure": {"run_rate": pooled, "above_flag": pooled is not None and pooled > dev.FORMAT_FAILURE_FLAG,
                                            "flag_above_rate": dev.FORMAT_FAILURE_FLAG}}
    return out


def write_resources(a: argparse.Namespace, run: RunDir, summary: dict[str, Any], elapsed: float) -> None:
    rows = resource_rows(elapsed, serving_gpus=a.serving_gpus, gpu_usd_per_hour=a.gpu_usd_per_hour,
                         peak_gpu_mem_gb=a.peak_gpu_mem_gb, orchestrator_cpus=os.cpu_count())
    judge = summary.get("judge_cost_usd") or 0.0
    qa = [s.get("qa") or {} for s in summary.get("slices", {}).values()]
    rows += [("judge_input_tokens", sum(q.get("judge_input_tokens", 0) for q in qa), "tokens", "HLE judge (OpenAI) prompt tokens"),
             ("judge_output_tokens", sum(q.get("judge_output_tokens", 0) for q in qa), "tokens", "HLE judge completion tokens incl. reasoning"),
             ("judge_cost_usd", round(judge, 4), "USD", f"estimated at {a.judge_usd_per_mtok_in} / {a.judge_usd_per_mtok_out} USD per M tokens"),
             ("total_cost_usd", round(next(r[1] for r in rows if r[0] == "cost_usd") + judge, 2), "USD", "cost_usd + judge_cost_usd")]
    run.write_resources(rows)


def fresh_keys(todo: list[Any], records: dict[tuple[str, str, int], dict[str, Any]], rerun: set[str]) -> set[tuple[str, str, int]]:
    """Items to run again from scratch (their cached rollout set aside): only recorded agent_error / missing items that
    --rerun-statuses names. A re-grade (--regrade-slices) never re-runs an item, whatever its status."""
    return {it.key for it in todo if it.key in records and records[it.key].get("run_status") in rerun
            and records[it.key].get("run_status") in ("agent_error", "missing")}


# --- admission and phases ----------------------------------------------------------------------------------------------
def resolve_admission(a: argparse.Namespace, slices: list[str]) -> dict[str, Any] | None:
    """The frozen admitted SWE / terminal dev task lists for a run with these slices: None when the run has neither slice or
    when no list is frozen (then a protocol deviation); refuses when the pools no longer agree with the frozen lists."""
    if not any(s in admission.GATED_SLICES for s in slices):
        return None
    path = a.admitted or a.manifests_dir / admission.ADMITTED_FILE
    try:
        frozen = admission.load(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if frozen is None:
        return None
    problems = admission.verify(frozen, admission.decide(a.manifests_dir, {"swe": a.swe_pool_tasks, "terminal": a.terminal_pool_tasks}))
    if problems:
        raise SystemExit(f"the pools disagree with the frozen admitted dev task lists {path}: " + "; ".join(problems))
    return {"path": str(path), "file_sha256": sha256_file(path), "admitted_sha256": frozen["admitted_sha256"],
            "admitted": {sl: sorted(ids) for sl, ids in admission.admitted_set(frozen).items()},
            "excluded": {sl: d["excluded"] for sl, d in frozen["slices"].items()}}


def prior_phases(prior: dict[str, Any], resume_running: bool) -> list[dict[str, Any]]:
    """Phases of the run's earlier invocations (metadata `phases`; reconstructed from the start and end times of runs
    recorded before phases existed). A phase left without an end by a killed process gets the resume time as its end
    (`end_inferred`) when --resume-running says the process is gone."""
    if prior.get("phases") is not None:
        phases = [dict(ph) for ph in prior["phases"]]
    elif prior.get("start_time") and prior.get("slices"):
        phases = [{"phase": ph, "slices": [s for s in prior["slices"] if dev.phase_of(s) == ph], "start": prior["start_time"],
                   "end": prior.get("end_time"), "n_items": None, "reconstructed": True,
                   "endpoint": {k: (prior.get("endpoint") or {}).get(k) for k in ("api_base", "models")} | {"model_name": prior.get("model_name")}}
                  for ph in dev.PHASE_ORDER if any(dev.phase_of(s) == ph for s in prior["slices"])]
    else:
        phases = []
    if resume_running:
        for ph in phases:
            if not ph.get("end"):
                ph.update(end=utc_now(), end_inferred=True)
    return phases


def host_load_csv(a: argparse.Namespace, start: str) -> Path | None:
    if a.host_load_csv is not None:
        return a.host_load_csv
    f = REPO / "logs" / f"host_load_{start[:10].replace('-', '')}" / "host_load.csv"
    return f if f.exists() else None


PART_KEYS = ("concurrency", "terminal_agent_timeout_sec", "swe_sandbox", "phase_order", "admitted_sha256")


def combine(a: argparse.Namespace, root: Path, results_dir: Path) -> int:
    """--combine: the run --run-id made of finished part runs (module docstring, Protocol). Refuses parts that differ in
    version, size, protocol or admitted lists, overlap in slices, miss a slice, or are smoke runs for any reason other
    than covering only some slices."""
    if results_dir.exists():
        raise SystemExit(f"{results_dir} exists; never overwrite results")
    parts = []
    for pid in a.combine:
        meta = read_json(root / "logs" / pid / "metadata.json", None)
        if not isinstance(meta, dict):
            raise SystemExit(f"--combine: no metadata for part {pid}")
        pdir = Path(meta.get("results_dir") or root / "results" / "dev" / a.version / pid)
        parts.append((pid, meta, pdir))
    problems = []
    for pid, meta, pdir in parts:
        if meta.get("status") != "finished":
            problems.append(f"{pid}: status {meta.get('status')!r}, not finished")
        if (meta.get("version"), meta.get("size")) != (a.version, parts[0][1].get("size")):
            problems.append(f"{pid}: version/size {meta.get('version')}/{meta.get('size')} differ")
        if not (pdir / "per_item.jsonl").exists():
            problems.append(f"{pid}: no per_item.jsonl in {pdir}")
        other = [r for r in meta.get("smoke_reasons") or [] if not r.startswith("slices ")]
        if other:
            problems.append(f"{pid}: smoke for {other}")
        proto = ((meta.get("config") or {}).get("protocol") or {})
        ref = ((parts[0][1].get("config") or {}).get("protocol") or {})
        if proto.get("deviations") or any(proto.get(k) != ref.get(k) for k in PART_KEYS):
            problems.append(f"{pid}: protocol differs from {parts[0][0]} or has deviations {proto.get('deviations')}")
    seen: dict[str, str] = {}
    for pid, meta, _ in parts:
        for sl in meta.get("slices") or []:
            if sl in seen:
                problems.append(f"slice {sl} in both {seen[sl]} and {pid}")
            seen[sl] = pid
    if set(seen) != set(dev.SLICES):
        problems.append(f"the parts cover {sorted(seen)}, not {list(dev.SLICES)}")
    if problems:
        raise SystemExit("--combine refused: " + "; ".join(problems))
    first = parts[0][1]
    with RunDir(a.run_id, root, log_name="eval.log") as run:
        results_dir.mkdir(parents=True)
        with (results_dir / "per_item.jsonl").open("w") as out:
            for pid, _, pdir in parts:
                for line in (pdir / "per_item.jsonl").read_text().splitlines():
                    if line.strip():
                        out.write(json.dumps({**json.loads(line), "part_run_id": pid}) + "\n")
        cfg = {k: v for k, v in (first.get("config") or {}).items() if k != "settings"}
        cfg["settings"] = {pid: (meta.get("config") or {}).get("settings") for pid, meta, _ in parts}
        plan_keys = [k for _, meta, _ in parts for k in meta.get("plan_keys") or []]
        phases = [{**ph, "part_run_id": pid} for pid, meta, _ in parts for ph in meta.get("phases") or []]
        run.write_metadata(
            run_id=a.run_id, script="scripts/data/run_dev_eval.py", mode="combine", parts=[pid for pid, _, _ in parts],
            version=a.version, label=first.get("label"), size=first.get("size"), slices=list(dev.SLICES), model_name=first.get("model_name"),
            endpoint={pid: meta.get("endpoint") for pid, meta, _ in parts}, code=git_info(), hardware=hardware_info(), command=command_line(),
            note=a.note, decision_grade=first.get("size") in ("full", "reduced"), smoke=False, smoke_reasons=[], config=cfg,
            data_version={pid: meta.get("data_version") for pid, meta, _ in parts}, model_revision=first.get("model_revision"),
            seed=first.get("seed"), blocked_slices={}, plan_keys=plan_keys, results_dir=str(results_dir), phases=phases,
            start_time=min(meta.get("start_time") for _, meta, _ in parts), status="running")
        run.log(f"combined {[pid for pid, _, _ in parts]} into {results_dir}: slices {seen}")
        summary = summarize(a, run, results_dir, [tuple(k) for k in plan_keys], {}, a.run_id)
        write_json(results_dir / "summary.json", summary)
        elapsed = run.elapsed()
        write_resources(argparse.Namespace(**{**vars(a), "serving_gpus": 0}), run, summary, elapsed)   # the combine step itself: CPU only
        part_costs = {pid: next((r["value"] for r in csv.DictReader((root / "logs" / pid / "resources.csv").open()) if r["metric"] == "total_cost_usd"), None)
                      for pid, _, _ in parts if (root / "logs" / pid / "resources.csv").exists()}
        run.write_metadata(end_time=max(meta.get("end_time") or "" for _, meta, _ in parts), combined_at=utc_now(), parts_total_cost_usd=part_costs,
                           status="finished" if summary["complete"] else "finished_incomplete", summary=_short(summary),
                           reportable=summary["reportable"])
        run.finalize()
        run.log(f"done: macro {summary['macro_average_pct']}; reportable {summary['reportable']}; guards all_pass="
                f"{summary.get('guards', {}).get('all_pass')}")
    return 0


# --- main ------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or default_run_id(a)
    root = Path(a.out_root).resolve() if a.out_root else REPO
    results_dir = root / "results" / "dev" / a.version / run_id
    if a.combine:
        return combine(a, root, results_dir)
    if results_dir.exists() and not (a.resume or a.summarize_only):
        raise SystemExit(f"{results_dir} exists; never overwrite results (use --resume)")
    prior = read_json(root / "logs" / run_id / "metadata.json", {}) or {}
    if a.resume and prior.get("status") == "running" and not a.resume_running:
        raise SystemExit(f"{run_id}: its metadata says it is still running (another invocation?); wait until it has finished "
                         "(phases never overlap), or pass --resume-running if that process was killed")
    order = list(dev.SLICES)
    all_slices = sorted(set(prior.get("slices") or []) | set(a.slices), key=order.index) if a.resume else list(a.slices)
    adm = None if a.summarize_only else resolve_admission(a, all_slices)
    protocol = dev.protocol_of({"swe": a.swe_concurrency, "terminal": a.terminal_concurrency, "qa": a.qa_concurrency},
                               a.terminal_agent_timeout_sec, adm["admitted_sha256"] if adm else None,
                               gated=any(s in admission.GATED_SLICES for s in all_slices))
    if a.baseline:   # fail before any work: smoke, dry and non-reportable runs and other protocols never serve as a baseline
        try:
            refusal = dev.baseline_refusal(dev.resolve_baseline(a.baseline, REPO), None if a.summarize_only else protocol)
        except FileNotFoundError as e:
            refusal = str(e)
        if refusal:
            raise SystemExit(f"--baseline refused: {refusal}")
    with RunDir(run_id, root, log_name="eval.log", resume=a.resume or a.summarize_only) as run:
        if a.summarize_only:
            meta = prior
            summary = summarize(a, run, results_dir, [tuple(k) for k in meta.get("plan_keys") or []] or None, meta.get("blocked_slices") or {}, run_id)
            write_json(results_dir / "summary.json", summary)
            run.write_metadata(summary_recomputed_at=utc_now(), summary=_short(summary), status=meta.get("status") or "finished",
                               smoke=summary["smoke"], smoke_reasons=summary["smoke_reasons"], reportable=summary["reportable"])
            run.finalize()
            run.log(f"summary recomputed: {results_dir / 'summary.json'}")
            return 0
        results_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(a.work_root) / run_id
        plan = dev.build_plan(a.manifests_dir, a.size, a.slices, a.limits, a.task_id_list)
        if adm is not None:   # only admitted SWE / terminal tasks are planned (src/apex_data/devset/admission.py)
            keep = {sl: set(ids) for sl, ids in adm["admitted"].items()}
            n_planned = len(plan)
            plan = [it for it in plan if it.slice not in keep or it.task_id in keep[it.slice]]
            run.log(f"admitted dev tasks {adm['path']} (sha256 {adm['admitted_sha256'][:12]}): "
                    f"{', '.join(f'{sl} {len(ids)}' for sl, ids in adm['admitted'].items())}; {n_planned - len(plan)} items of excluded tasks not planned")
        plan_keys = list(dict.fromkeys([*(tuple(k) for k in (prior.get("plan_keys") or []) if a.resume), *(it.key for it in plan)]))
        run.log(f"run_id={run_id} version={a.version} label={a.label or '-'} size={a.size} slices={a.slices} (run: {all_slices}) "
                f"model={a.model_name} items={len(plan)} {dev.plan_counts(plan)}")
        ep = endpoint_info(a.api_base)
        served = isinstance(ep.get("models"), list) and a.model_name in ep["models"]
        if not served and not a.regrade:
            run.error(f"preflight: model {a.model_name!r} not served at {a.api_base} ({ep.get('models')})")
            raise SystemExit("preflight failed: model not served at --api-base")
        ctx = dev.RunContext(run_id=run_id, version=a.version, model_name=a.model_name, api_base=a.api_base, api_key=a.api_key,
                             work_dir=work_dir, results_dir=results_dir, log=run.log, error=run.error)
        runners, blocked, settings = build_runners(a, ctx, plan, run)
        if blocked and not a.skip_blocked:
            raise SystemExit(f"blocked slices {blocked}; fix them or pass --skip-blocked")
        runnable = [it for it in plan if it.slice in runners]
        records, _ = dev.read_records(results_dir / "per_item.jsonl")
        todo = [it for it in runnable if it.key not in records or records[it.key].get("run_status") in a.rerun or it.slice in a.regrade]
        ctx.regrade = {it.key for it in todo if it.key in records and it.slice in a.regrade}
        if not served:   # a re-grade needs no model: only recorded items, graded from their cached rollouts as they are
            if a.rerun or any(it.key not in ctx.regrade for it in todo):
                run.error(f"preflight: model {a.model_name!r} not served at {a.api_base}, and items other than re-grades are due")
                raise SystemExit("preflight failed: model not served at --api-base (only a pure --regrade-slices run works without it)")
            ctx.no_rollouts = True
            run.log("model not served: re-grade only; cached rollouts are graded as they are (environment failures stay as recorded)")
        # a re-run agent_error / missing item starts from scratch (its cached rollout is what failed); other re-run statuses
        # reuse what is cached (e.g. a scoring_failure re-judges the cached generation, a verifier_failure re-grades)
        ctx.fresh = fresh_keys(todo, records, a.rerun)
        ctx.prior_records = {k: records[k] for k in ctx.regrade}
        for f in [*(a.manifests_dir / n for n in ("dev_swe.json", "dev_terminal.json", "dev_qa.json", "dev_freeze.json")),
                  dev.SWE_LOCK, paths.CONFIGS / "eval" / "terminal_bench_2.yaml", paths.CONFIGS / "eval" / "hle.yaml",
                  work_dir / "swe_agent_config.yaml"]:
            if f.exists():
                run.snapshot_config(f)
        default_inputs = (a.out_root is None and a.manifests_dir.resolve() == paths.MANIFESTS.resolve()
                          and a.swe_pool_tasks.resolve() == (paths.POOLS / "swesmith" / "tasks.jsonl").resolve()
                          and a.terminal_pool_tasks.resolve() == (paths.POOLS / "tgr_env" / "tasks.jsonl").resolve()
                          and a.terminal_dataset_dir.resolve() == (paths.WORK / "tgr_env" / "tasks").resolve())
        smoke_reasons = dev.smoke_reasons(size=a.size, limits=a.limits, task_ids=a.task_id_list, default_inputs=default_inputs,
                                          slices=all_slices, blocked=blocked, qa_max_tokens=a.qa_max_tokens)
        decision_grade = a.size in ("full", "reduced") and not smoke_reasons and not protocol["deviations"]
        if protocol["deviations"]:
            run.error(f"protocol deviations (the run is not reportable): {protocol['deviations']}")
        manifests = {n: sha256_file(a.manifests_dir / n) for n in ("dev_swe.json", "dev_terminal.json", "dev_qa.json", "dev_freeze.json")
                     if (a.manifests_dir / n).exists()}
        run.write_metadata(
            run_id=run_id, script="scripts/data/run_dev_eval.py", version=a.version, label=a.label, size=a.size, slices=all_slices,
            model_name=a.model_name, endpoint=ep, serving=read_json(a.serving_info, f"unreadable: {a.serving_info}") if a.serving_info else None,
            code=git_info(), hardware=hardware_info(), command=command_line(), note=a.note, decision_grade=decision_grade,
            smoke=bool(smoke_reasons), smoke_reasons=smoke_reasons,
            config={"settings": settings, "concurrency": {"swe": a.swe_concurrency, "terminal": a.terminal_concurrency, "qa": a.qa_concurrency},
                    "protocol": protocol, "admission": adm,
                    "env_retries": a.env_retries, "limits": a.limits, "task_ids": a.task_id_list, "rerun_statuses": sorted(a.rerun),
                    "regrade_slices": sorted(a.regrade),
                    "default_inputs": default_inputs, "qa_max_tokens": a.qa_max_tokens,
                    "inputs": {"manifests_dir": str(a.manifests_dir), "swe_pool_tasks": str(a.swe_pool_tasks),
                               "terminal_pool_tasks": str(a.terminal_pool_tasks), "terminal_dataset_dir": str(a.terminal_dataset_dir)},
                    "work_dir": str(work_dir), "guards": {"resolutions_pp": dev.RESOLUTIONS_PP, "point_drop_pp": dev.GUARD_POINT_PP,
                                                         "format_failure_rel": dev.FORMAT_FAILURE_REL, "think_ratio_max": dev.THINK_RATIO_MAX,
                                                         "think_max_tokens_per_turn": dev.THINK_MAX_TOKENS_PER_TURN,
                                                         "flags": {"format_failure_above": dev.FORMAT_FAILURE_FLAG,
                                                                   "think_ratio_below": dev.THINK_RATIO_FLAG_BELOW}}},
            data_version={"manifests_sha256": manifests, "swe_pool_sha256": sha256_file(a.swe_pool_tasks) if a.swe_pool_tasks.exists() else None,
                          "terminal_pool_sha256": sha256_file(a.terminal_pool_tasks) if a.terminal_pool_tasks.exists() else None,
                          "plan_counts": dev.plan_counts(plan)},
            model_revision={"base": model_lock(), "version": a.version, "label": a.label, "served_model_name": a.model_name,
                            "note": "adapter repository and revision of non-base versions: see the serving metadata (--serving-info)"},
            seed="per (item, sample) from the frozen dev manifests (devset/common.py sample_seed)", blocked_slices=blocked,
            plan_keys=[list(k) for k in plan_keys], results_dir=str(results_dir), start_time=prior.get("start_time") or utc_now(),
            phases=prior_phases(prior, a.resume_running) if a.resume else [], status="running")
        n_rerun = sum(1 for it in todo if it.key in records)
        run.log(f"{len(todo)} items to run ({len(todo) - n_rerun} new, {n_rerun} re-run by status, {len(ctx.fresh)} of them from scratch; "
                f"{len(runnable) - len(todo)} already recorded); blocked: {blocked or 'none'}")
        writer = dev.RecordWriter(results_dir / "per_item.jsonl", run.metric)

        def on_record(rec: dict[str, Any]) -> None:
            prev = records.get(dev.record_key(rec))
            if prev is not None:   # the superseded record stays in per_item.jsonl; this line names it
                rec["supersedes"] = {k: prev.get(k) for k in ("run_status", "error_reason", "resolved", "harness_status")}
            writer.append(rec)
            if rec["run_status"] != "ok":   # errors.log lists every failure and timeout (budget_exhausted) with its reason
                run.error(f"{rec['slice']} {rec['task_id']} s{rec['sample_index']}: {rec['run_status']}: {rec['error_reason']}")
            run.log(f"{rec['slice']} {rec['task_id']} s{rec['sample_index']}: resolved={rec['resolved']} status={rec['run_status']} "
                    f"time={rec['time_sec']}s out_tokens={rec['output_tokens']}")

        concurrency = {"swe": a.swe_concurrency, "terminal": a.terminal_concurrency, "qa": a.qa_concurrency, "qa_aime": a.qa_concurrency}
        phases = run.read_metadata().get("phases") or []
        done: Counter = Counter()
        for ph in dev.PHASE_ORDER:   # swe + qa together, then terminal alone (protocol phase_order)
            group = [it for it in todo if dev.phase_of(it.slice) == ph]
            if not group:
                continue
            entry: dict[str, Any] = {"phase": ph, "slices": sorted({it.slice for it in group}), "start": utc_now(), "end": None,
                                     "n_items": len(group), "endpoint": {"api_base": a.api_base, "model_name": a.model_name,
                                                                         "models": ep.get("models"), "version": ep.get("version"),
                                                                         "serving": read_json(a.serving_info, None) if a.serving_info else None}}
            phases.append(entry)
            run.write_metadata(phases=phases)
            run.log(f"phase {ph}: {len(group)} items of {entry['slices']}")
            try:
                done.update(dev.execute(group, runners, concurrency, on_record, ctx))
            finally:   # an interrupted phase still records when it stopped, and the host load it ran under
                entry["end"] = utc_now()
                entry["host_load"] = dev.host_load_window(host_load_csv(a, entry["start"]), entry["start"], entry["end"])
                run.write_metadata(phases=phases)
        run.log(f"executed: {dict(done)}")
        summary = summarize(a, run, results_dir, plan_keys, blocked, run_id)
        write_json(results_dir / "summary.json", summary)
        elapsed = run.elapsed()
        write_resources(a, run, summary, elapsed)
        run.write_metadata(end_time=utc_now(), duration_sec=round(elapsed, 1), status="finished" if summary["complete"] else "finished_incomplete",
                           summary=_short(summary), reportable=summary["reportable"])
        run.finalize()
        for sl, s in summary["slices"].items():
            run.log(f"{sl}: {s['n_resolved']}/{s['n_items']} = {s['score_pct']}% | run failures {s['failures']['n_run_failures']} "
                    f"| format failures {s['turns']['n_format_failures']}/{s['turns']['n_turns']} turns | mean think "
                    f"{s['turns']['mean_think_per_turn']} {s['turns']['think_unit']}/turn")
        run.log(f"done: macro {summary['macro_average_pct']}; guards all_pass={summary.get('guards', {}).get('all_pass')}; results in {results_dir}")
    return 0


def _short(summary: dict[str, Any]) -> dict[str, Any]:
    return {"macro_average_pct": summary.get("macro_average_pct"), "complete": summary.get("complete"), "smoke": summary.get("smoke"),
            "reportable": summary.get("reportable"), "protocol_deviations": (summary.get("protocol") or {}).get("deviations"),
            "slices": {sl: {"score_pct": s["score_pct"], "n_items": s["n_items"]} for sl, s in summary.get("slices", {}).items()},
            "guards_all_pass": (summary.get("guards") or {}).get("all_pass"), "judge_cost_usd": summary.get("judge_cost_usd")}


if __name__ == "__main__":
    sys.exit(main())
