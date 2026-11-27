"""Run HLE (Humanity's Last Exam) with the official scripts against an OpenAI-compatible endpoint, then the official judge.

Usage (from the repository root; or via scripts/evaluate.sh hle ...):
  PYTHONPATH=src .venv/hle/bin/python -m apex_eval.hle --version base --model-name <served model id> \
      --api-base http://<vllm-host>:8000/v1 [--max-samples N]

Inputs : configs/eval/hle.yaml (lock; the split is fixed there); third_party/hle (official hle_eval scripts at the pinned commit);
         third_party/hle_data (the gated dataset at the locked revision, downloaded by scripts/setup_eval_env.sh with HF_TOKEN);
         OPENAI_API_KEY in the environment for the judge (loaded from .env by scripts/evaluate.sh); a running endpoint at --api-base.
Outputs: logs/<run_id>/            metadata.json, eval.log, metrics.jsonl (one line per question), errors.log, resources.csv,
                                   config_snapshot/, hle/ (predictions file and the stdout of each prediction pass)
         results/<version>/hle/<run_id>/   per_task.jsonl (shared schema: verdict, confidence, tokens; no answer key),
                                   predictions.json (the model's raw responses), summary.json
         third_party/.hle_judge_outputs/<run_id>/   (git-ignored) the judge's output file and logs, which embed the dataset's
                                   correct answers; the output file's sha256 is recorded in metadata.json
Protocol: the official run_model_predictions.py (its system prompt, message format, one attempt per question, resume on rerun)
          and run_judge_results.py (JUDGE_PROMPT, structured output, judge model) are called unmodified. The prediction script
          sends no sampling parameters, so sampling is the serving defaults, identical for all three model versions. A further
          prediction pass runs only after a pass that logged API errors (decision_2026-10-12_hle_retries): a response with no
          final answer (reasoning past max_completion_tokens) is not re-asked and is recorded as budget_exhausted; a question
          missing after API errors is a run failure. Both count as incorrect in the official metric (the judge divides by the
          full question count); so does a missing judge verdict. Nothing is dropped. --resume continues a run directory: the
          prediction passes already there count, so a run stopped before a retry pass is judged from its existing predictions.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from apex_eval.common import (BUDGET_EXHAUSTED, MISSING, OK, REPO_ROOT, SCORING_FAILURE, SKIPPED, RunPaths, add_common_args, checkout_matches,
                              command_line, failure_summary, git_info, hardware_info, lock_deviations, model_revision, preflight,
                              read_json, record, run_logged, serving_info, sha256_file, utc_now, write_json)

LOCK = REPO_ROOT / "configs" / "eval" / "hle.yaml"
PARQUET_SHA = REPO_ROOT / "configs" / "eval" / "hle_parquet.sha256"
HLE_SRC = REPO_ROOT / "third_party" / "hle"
HLE_CODE = HLE_SRC / "hle_eval"
HLE_DATA = REPO_ROOT / "third_party" / "hle_data"
HLE_DATA_TEXT = REPO_ROOT / "third_party" / "hle_data_textonly"
JUDGE_STORE = REPO_ROOT / "third_party" / ".hle_judge_outputs"
PARQUET = Path("data") / "test-00000-of-00001.parquet"
PY = Path(sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    lock = yaml.safe_load(LOCK.read_text())
    proto = lock["protocol"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--split", default=proto["split"], choices=["full", "text-only"], help="lock value; override for smoke tests only")
    p.add_argument("--max-completion-tokens", type=int, default=int(proto["max_completion_tokens"]), help="lock value")
    p.add_argument("--num-workers", type=int, default=int(proto["num_workers"]), help="prediction workers (throughput only)")
    p.add_argument("--judge", default=proto["judge_model"], help="lock value; any other judge makes the run not reportable")
    p.add_argument("--judge-workers", type=int, default=20)
    p.add_argument("--skip-judge", action="store_true", help="debugging only; predictions without the official judge")
    p.add_argument("--max-samples", type=int, default=None, help="smoke tests only: first N questions (marks the run not reportable)")
    a = p.parse_args(argv)
    a.lock = lock
    a.prediction_passes = int(proto["retry_passes"]["prediction"])
    a.judge_passes = int(proto["retry_passes"]["judge"])
    return a


def dataset_dir(a: argparse.Namespace, paths: RunPaths) -> tuple[Path, str, str]:
    """(dataset dir for the split, source parquet sha256, evaluated parquet sha256). The source parquet is checked against
    the lock; the text-only copy (rows whose `image` is empty) is materialized once, atomically, and its row count is checked."""
    source_sha = sha256_file(HLE_DATA / PARQUET)
    locked_sha = PARQUET_SHA.read_text().split()[0]
    if source_sha != locked_sha:
        paths.error(f"HLE parquet sha256 {source_sha} does not match the lock {locked_sha}")
        raise SystemExit("HLE dataset does not match configs/eval/hle_parquet.sha256; re-run scripts/setup_eval_env.sh")
    if a.split == "full":
        return HLE_DATA, source_sha, source_sha
    out = HLE_DATA_TEXT / PARQUET
    expected = int(a.lock["dataset_facts"]["text_only"])
    if not out.exists():
        t = pq.read_table(HLE_DATA / PARQUET)
        filtered = t.filter(pc.equal(pc.utf8_length(pc.fill_null(t.column("image"), "")), 0))
        if filtered.num_rows != expected:
            raise SystemExit(f"text-only split has {filtered.num_rows} rows, lock says {expected}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pq.write_table(filtered, tmp)
        os.replace(tmp, out)
        paths.log(f"materialized text-only dataset: {filtered.num_rows} of {t.num_rows} rows -> {out}")
    n = pq.read_metadata(out).num_rows
    if n != expected:
        raise SystemExit(f"{out} has {n} rows, lock says {expected}; delete it and re-run")
    return HLE_DATA_TEXT, source_sha, sha256_file(out)


def question_index(ds_dir: Path, max_samples: int | None) -> list[dict[str, Any]]:
    """id / category / answer_type / has_image for every question in the evaluated set (no question or answer text)."""
    t = pq.read_table(ds_dir / PARQUET, columns=["id", "category", "answer_type", "image"])
    has_image = pc.greater(pc.utf8_length(pc.fill_null(t.column("image"), "")), 0).to_pylist()
    rows = [{"id": i, "category": c, "answer_type": at, "has_image": im}
            for i, c, at, im in zip(t.column("id").to_pylist(), t.column("category").to_pylist(), t.column("answer_type").to_pylist(), has_image, strict=True)]
    return rows[:max_samples] if max_samples else rows


def api_errors(log_path: Path) -> int:
    """API failures a prediction pass logged: the official script prints "Error: <exception>" for each failed call and returns
    no prediction; an empty response (no final answer content, e.g. truncated at max_completion_tokens) prints nothing."""
    return sum(1 for ln in log_path.read_text(errors="replace").splitlines() if ln.startswith("Error:")) if log_path.exists() else 0


def predict(paths: RunPaths, a: argparse.Namespace, ds_dir: Path, n: int) -> tuple[Path, dict[str, Any]]:
    """Run the official prediction script once; re-run it (it resumes, re-asking every question still without a prediction)
    only while the previous pass logged API errors, up to the lock's prediction passes (decision_2026-10-12_hle_retries:
    a question whose response had no final answer is not re-asked and counts as incorrect). On --resume, the passes already
    in the run directory count, so a run stopped before a retry pass is finished without new predictions."""
    work = paths.run_dir / "hle"
    work.mkdir(exist_ok=True)
    pred_file = work / f"hle_{os.path.basename(a.model_name)}.json"   # the official script's output name
    env = {**os.environ, "OPENAI_BASE_URL": a.api_base, "OPENAI_API_KEY": a.api_key, "HF_HUB_OFFLINE": "1", "TQDM_DISABLE": "1"}
    cmd = [str(PY), str(HLE_CODE / "run_model_predictions.py"), "--dataset", str(ds_dir), "--model", a.model_name,
           "--max_completion_tokens", str(a.max_completion_tokens), "--num_workers", str(a.num_workers)]
    if a.max_samples:
        cmd += ["--max_samples", str(a.max_samples)]
    done = sorted(int(f.stem.removeprefix("predict_pass")) for f in work.glob("predict_pass*.log") if f.stem.removeprefix("predict_pass").isdigit())
    a.prediction_api_errors = {f"pass{i}": api_errors(work / f"predict_pass{i}.log") for i in done}
    if done and pred_file.exists() and a.prediction_api_errors[f"pass{done[-1]}"] == 0:
        paths.log(f"resume: pass {done[-1]} already ran and logged no API errors; no new prediction pass "
                  f"({len(read_json(pred_file, {}))}/{n} predictions)")
        return pred_file, read_json(pred_file, {})
    prev = -1
    for i in range((done[-1] + 1) if done else 1, a.prediction_passes + 1):
        log_path = work / f"predict_pass{i}.log"
        rc = run_logged(paths, f"predict_pass{i}", cmd, env, work, log_path=log_path)
        have, errs = len(read_json(pred_file, {})), api_errors(log_path)
        a.prediction_api_errors[f"pass{i}"] = errs
        paths.log(f"predictions after pass {i}: {have}/{n}; API errors logged in this pass: {errs}")
        if rc != 0:
            paths.error(f"prediction pass {i} exited {rc}")
        if have >= n or have == prev or errs == 0:
            break
        prev = have
    return pred_file, read_json(pred_file, {})


def judge(paths: RunPaths, a: argparse.Namespace, ds_dir: Path, pred_file: Path, n_preds: int) -> tuple[Path, dict[str, Any], str]:
    """Run the official judge with the git-ignored store as its working directory, so its output (which embeds the answer
    key) and logs never land under logs/. Returns (judged file, verdicts, last judge stdout)."""
    store = JUDGE_STORE / paths.run_id
    store.mkdir(parents=True, exist_ok=True)
    judged_file = store / f"judged_{pred_file.name}.json"   # the official script writes judged_<predictions name>.json in its cwd
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_BASE_URL", "OPENAI_API_BASE")}   # the judge goes to OpenAI itself
    env.update(HF_HUB_OFFLINE="1", TQDM_DISABLE="1")
    cmd = [str(PY), str(HLE_CODE / "run_judge_results.py"), "--dataset", str(ds_dir), "--predictions", str(pred_file),
           "--judge", a.judge, "--num_workers", str(a.judge_workers)]
    judged: dict[str, Any] = {}
    log_text = ""
    for i in range(1, a.judge_passes + 1):
        log_path = store / f"judge_pass{i}.log"
        rc = run_logged(paths, f"judge_pass{i}", cmd, env, store, log_path=log_path)
        log_text = log_path.read_text(errors="replace")
        for line in [ln for ln in log_text.splitlines() if ("Error" in ln or "Traceback" in ln)
                     and not ln.startswith("Calibration Error") and "[correct_answer]" not in ln and "[question]" not in ln][:3]:
            paths.error(f"judge pass {i}: {line[:200]}")
        judged = read_json(judged_file, {})
        n_j = sum(1 for v in judged.values() if "judge_response" in v)
        paths.log(f"judged after pass {i}: {n_j}/{n_preds} predictions")
        if rc != 0 and n_j >= n_preds and "calib_err" in log_text:
            paths.log(f"judge pass {i}: all verdicts written; the official script's calibration routine failed (it needs >= 100 samples), "
                      "which only happens on small smoke runs")
        elif rc != 0:
            paths.error(f"judge pass {i} exited {rc}")
        if n_j >= n_preds:
            break
    return judged_file, judged, log_text


def question_records(paths: RunPaths, a: argparse.Namespace, questions: list[dict[str, Any]], preds: dict[str, Any],
                     judged: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-question records: prediction/judge status, verdict, confidence and tokens; no question text, no correct answers."""
    records = []
    for q in questions:
        base = dict(answer_type=q["answer_type"], has_image=q["has_image"])
        p = preds.get(q["id"])
        if p is None:
            last_errs = list(a.prediction_api_errors.values())[-1] if a.prediction_api_errors else 0
            if last_errs == 0:   # the last pass had no API failure: the response carried no final answer (budget, not a run failure)
                err = ("no final answer: the response had no content (most likely the reasoning ran past max_completion_tokens); "
                       "not re-asked (decision_2026-10-12_hle_retries); counts as incorrect")
                records.append(record("hle", q["id"], group=q["category"], resolved=False, run_status=BUDGET_EXHAUSTED,
                                      harness_status="no_final_answer", error_reason=err, **base))
            else:
                err = (f"no prediction; the last pass logged {last_errs} API errors, so this is an API failure or an empty "
                       "response; counts as incorrect")
                paths.error(f"{q['id']}: {err}")
                records.append(record("hle", q["id"], group=q["category"], resolved=False, run_status=MISSING,
                                      harness_status="missing_prediction", error_reason=err, **base))
            continue
        usage = p.get("usage") or {}
        tokens = dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                      reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        j = (judged.get(q["id"]) or {}).get("judge_response")
        if a.skip_judge:
            records.append(record("hle", q["id"], group=q["category"], run_status=SKIPPED, harness_status="unjudged",
                                  error_reason="judge skipped by request", response_chars=len(p.get("response") or ""), **tokens, **base))
        elif j is None:
            err = f"no judge verdict after {a.judge_passes} passes; counts as incorrect"
            paths.error(f"{q['id']}: {err}")
            records.append(record("hle", q["id"], group=q["category"], resolved=False, run_status=SCORING_FAILURE,
                                  harness_status="missing_judge", error_reason=err, response_chars=len(p.get("response") or ""), **tokens, **base))
        else:
            records.append(record("hle", q["id"], group=q["category"], resolved="yes" in str(j.get("correct", "")), run_status=OK,
                                  harness_status="judged", confidence=j.get("confidence"),
                                  extracted_answer_empty=str(j.get("model_answer", "")).strip().lower() in ("", "none"),
                                  response_chars=len(p.get("response") or ""), **tokens, **base))
    return records


def summarize(records: list[dict[str, Any]], judge_log: str, subset: bool) -> dict[str, Any]:
    n = len(records)
    n_correct = sum(1 for r in records if r["resolved"])
    m_acc = re.search(r"Accuracy: ([\d.]+)% \+/- ([\d.]+)% \| n = (\d+)", judge_log)
    m_cal = re.search(r"Calibration Error: ([\d.]+)", judge_log)

    def rate(sel) -> dict[str, Any]:
        xs = [r for r in records if sel(r)]
        return {"n": len(xs), "accuracy_pct": round(100 * sum(1 for r in xs if r["resolved"]) / len(xs), 2) if xs else None}

    official = None if subset else {   # the official script divides by the whole split, so it is meaningless on a subset
        "accuracy_pct": float(m_acc.group(1)) if m_acc else None, "wald_95_half_width_pct": float(m_acc.group(2)) if m_acc else None,
        "n": int(m_acc.group(3)) if m_acc else None, "calibration_error": float(m_cal.group(1)) if m_cal else None}
    out = {
        "reported_metric": "accuracy (%) over all questions in the split; questions without a prediction or verdict count as incorrect",
        "n_questions": n, "n_predicted": sum(1 for r in records if r["harness_status"] != "missing_prediction"),
        "n_judged": sum(1 for r in records if r["run_status"] == OK), "n_correct": n_correct,
        "accuracy_pct": round(100 * n_correct / n, 2) if n else None,
        "official_script_metrics": official,
        "by_category": {c: rate(lambda r, c=c: r["group"] == c) for c in sorted({r["group"] for r in records})},
        "by_answer_type": {t: rate(lambda r, t=t: r["extra"]["answer_type"] == t) for t in sorted({r["extra"]["answer_type"] for r in records})},
        **failure_summary(records),
    }
    if any(r["extra"]["has_image"] for r in records):
        out["by_modality"] = {"image": rate(lambda r: r["extra"]["has_image"]), "text": rate(lambda r: not r["extra"]["has_image"])}
    return out


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if not (HLE_CODE / "run_model_predictions.py").exists() or not (HLE_DATA / PARQUET).exists():
        sys.exit("run scripts/setup_eval_env.sh with HF_TOKEN in .env first (HLE code or dataset missing)")
    with RunPaths("hle", a.version, a.run_id, root=a.out_root, resume=a.resume) as paths:
        ep = preflight(paths, a, env_vars=() if a.skip_judge else ("OPENAI_API_KEY",))
        hle_commit = checkout_matches(paths, "HLE scripts", HLE_SRC, a.lock["source"]["evaluation_code_commit"])
        ds_dir, source_sha, eval_sha = dataset_dir(a, paths)
        questions = question_index(ds_dir, a.max_samples)
        n = len(questions)
        paths.log(f"run_id={paths.run_id} version={a.version} model={a.model_name} split={a.split} questions={n} judge={a.judge}")
        lock_copy = paths.snapshot_config(LOCK)
        proto = a.lock["protocol"]
        deviations = lock_deviations({"split": a.split, "max_completion_tokens": a.max_completion_tokens, "judge_model": a.judge},
                                     {"split": proto["split"], "max_completion_tokens": proto["max_completion_tokens"], "judge_model": proto["judge_model"]})
        reportable = not deviations and not a.max_samples and not a.skip_judge
        if not reportable:
            paths.log(f"NOT REPORTABLE: deviations={deviations} max_samples={a.max_samples} skip_judge={a.skip_judge}")
        paths.write_metadata(
            run_id=paths.run_id, benchmark="hle", benchmark_version="cais/hle @ " + a.lock["source"]["dataset_sha"], version=a.version,
            model_name=a.model_name, endpoint=ep, serving=serving_info(a.serving_info), code=git_info(), hardware=hardware_info(),
            config={"lock_file": str(lock_copy.relative_to(paths.run_dir)), "split": a.split, "n_questions": n,
                    "max_completion_tokens": a.max_completion_tokens, "num_workers": a.num_workers, "judge_model": a.judge,
                    "retry_passes": {"prediction": a.prediction_passes, "judge": a.judge_passes},
                    "prediction_retry_rule": "re-run only after a pass that logged API errors (decision_2026-10-12_hle_retries)",
                    "hle_code_commit": hle_commit,
                    "sampling": "serving defaults (the official prediction script sends none); see configs/eval/model.yaml serving.flags"},
            data_version={"dataset_dir": str(ds_dir), "source_parquet_sha256": source_sha, "evaluated_parquet_sha256": eval_sha, "max_samples": a.max_samples},
            model_revision=model_revision(), seed="none (no seed is sent; sampling is the serving defaults)", command=command_line(),
            note=a.note, lock_deviations=deviations, reportable=reportable, status="running",
            **({} if paths.resumed else {"start_time": utc_now()}),
        )
        pred_file, preds = predict(paths, a, ds_dir, n)
        judged: dict[str, Any] = {}
        judge_log = ""
        judge_store: dict[str, Any] = {"dir": None, "judged_sha256": None}
        if not a.skip_judge and preds:
            judged_file, judged, judge_log = judge(paths, a, ds_dir, pred_file, len(preds))
            judge_store = {"dir": str(judged_file.parent), "judged_sha256": sha256_file(judged_file) if judged_file.exists() else None,
                           "note": "the judge's output and logs embed the dataset's correct answers, so they live outside logs/ in a git-ignored store"}
        records = question_records(paths, a, questions, preds, judged)
        paths.write_records(records)
        write_json(paths.results_dir / "predictions.json", preds)
        summary = {"run_id": paths.run_id, "benchmark": "hle", "version": a.version, "model_name": a.model_name, "split": a.split,
                   "judge_model": a.judge, "reportable": reportable, "lock_deviations": deviations,
                   "prediction_api_errors_per_pass": getattr(a, "prediction_api_errors", {}),
                   **summarize(records, judge_log, subset=bool(a.max_samples)), "judge_output_store": judge_store}
        paths.finish(a, summary, judge_store=judge_store)
        paths.log(f"done: accuracy {summary['accuracy_pct']}% ({summary['n_correct']}/{n}); {summary['n_run_failures']} run failures; "
                  f"results in {paths.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
