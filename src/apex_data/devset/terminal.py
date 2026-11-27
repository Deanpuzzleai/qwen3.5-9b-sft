"""Terminal dev slice (docs/DATA_PLAN.md sections 8 and 10): 40 TermGrade environments drawn across difficulty bands.

Inputs: `tasks.jsonl` and `labels.jsonl` of `ai-and/termgrade-environments` (source `tgr_env`), and `data/trials.parquet`
of `ai-and/termgrade-trajectories` (source `tgr`) for the count of teacher passes the dev exclusion removes.

Eligibility (survival table in the manifest):
  1. `split == "unseen"`. The card defines it: 202 of the 1,004 tasks (`train151`, `heldout120`) were used by the
     dataset authors to train or validate their own model; the 802 `unseen` tasks never were. We take the dev tasks from
     the subset no published checkpoint was tuned on.
  2. `gradable` and `oracle_verified` (the reference solution passes the task's own verifier in the built image).
  3. Labelled by all six models with every counted trial present (`n_counted == k`).

Difficulty. A task's difficulty is the unweighted mean of the six models' pass rates in `labels.jsonl` (each model
counts once, whatever its k). The per-model `band` field of `labels.jsonl` measures distance from a 50 percent pass rate
(it puts 0.125 and 0.875 in the same band), so it does not order tasks by difficulty and is not used. Bands:
easy >= 2/3 > medium >= 1/3 > hard > 0 = unsolved (no counted trial of any model passed).

Draw. 40 tasks, at least 2 per band, the rest proportional to each band's share of the eligible tasks (largest
remainder; a band lifted to the floor keeps it and the remaining seats are re-split). Inside a band, task names are
sorted and shuffled with a seeded stream per band; the first n are taken. The manifest lists the 40 in a seeded
permutation, so any prefix is a random subsample (the pilot's reduced smoke takes a prefix).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from apex_data import paths
from apex_data.devset.common import largest_remainder, rng_for, sample_counts, sample_plan, sha256_file

SOURCE = "tgr_env"
HF_REPO = "ai-and/termgrade-environments"
TRIALS_REPO = "ai-and/termgrade-trajectories"
N_TASKS = 40
SPLIT = "unseen"
MODELS = ("gemma", "gemma_think", "glm", "kimi", "qwen", "deepseek")
TEACHER_MODEL = "deepseek"
BANDS = ("easy", "medium", "hard", "unsolved")
MIN_PER_BAND = 2
DEFAULT_ENV_DIR = paths.RAW / "tgr_env"


def band_of(mean_pass_rate: float) -> str:
    if mean_pass_rate >= 2 / 3:
        return "easy"
    if mean_pass_rate >= 1 / 3:
        return "medium"
    if mean_pass_rate > 0:
        return "hard"
    return "unsolved"


def allocate_with_floor(supply: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Proportional to supply (largest remainder) with at least `floor` per band (or the band's whole supply)."""
    if sum(supply.values()) < n:
        raise ValueError(f"only {sum(supply.values())} eligible tasks for {n} seats")
    fixed: dict[str, int] = {}
    while True:
        free = {b: s for b, s in supply.items() if b not in fixed and s > 0}
        alloc = largest_remainder(n - sum(fixed.values()), free)
        low = {b for b, k in alloc.items() if k < min(floor, supply[b])}
        if not low:
            break
        fixed.update({b: min(floor, supply[b]) for b in low})
    out = {b: fixed.get(b, alloc.get(b, 0)) for b in supply}
    over = {b: k for b, k in out.items() if k > supply[b]}
    if over:
        raise ValueError(f"allocation exceeds supply in {over}")
    return out


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_labels(labels: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in labels:
        if r["model"] in out.setdefault(r["task"], {}):
            raise ValueError(f"duplicate label for {r['task']} / {r['model']}")
        out[r["task"]][r["model"]] = r
    return out


def mean_pass_rate(per_model: dict[str, dict[str, Any]]) -> float:
    return sum(float(per_model[m]["pass_rate"]) for m in MODELS) / len(MODELS)


def eligible(tasks: list[dict[str, Any]], labels: dict[str, dict[str, dict[str, Any]]]
             ) -> tuple[list[str], list[dict[str, Any]]]:
    """Task names passing the three rules, and the survival table."""
    rules: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        (f"split_{SPLIT}", lambda t: t.get("split") == SPLIT),
        ("gradable_and_oracle_verified", lambda t: bool(t.get("gradable")) and bool(t.get("oracle_verified"))),
        ("labelled_by_all_six_models_complete",
         lambda t: set(labels.get(t["task"], {})) >= set(MODELS)
         and all(int(labels[t["task"]][m]["n_counted"]) == int(labels[t["task"]][m]["k"]) for m in MODELS)),
    ]
    kept, table = list(tasks), []
    for name, ok in rules:
        nxt = [t for t in kept if ok(t)]
        table.append({"rule": name, "n_in": len(kept), "n_dropped": len(kept) - len(nxt), "n_out": len(nxt)})
        kept = nxt
    names = [t["task"] for t in kept]
    if len(set(names)) != len(names):
        raise ValueError("duplicate task names in tasks.jsonl")
    return sorted(names), table


def draw(names: list[str], labels: dict[str, dict[str, dict[str, Any]]], seed: int = paths.SEED, n: int = N_TASKS
         ) -> tuple[list[str], dict[str, Any]]:
    """(the n dev task names in manifest order, the draw report)."""
    by_band: dict[str, list[str]] = {b: [] for b in BANDS}
    for name in names:
        by_band[band_of(mean_pass_rate(labels[name]))].append(name)
    supply = {b: len(v) for b, v in by_band.items()}
    quota = allocate_with_floor(supply, n, MIN_PER_BAND)
    picked: list[str] = []
    for b in BANDS:
        pool = sorted(by_band[b])
        rng_for(seed, SOURCE, "dev", b).shuffle(pool)
        picked += pool[: quota[b]]
    order = sorted(picked)
    rng_for(seed, SOURCE, "dev", "order").shuffle(order)
    return order, {"supply": supply, "quota": quota, "min_per_band": MIN_PER_BAND}


def teacher_passes(trials_path: Path, dev_names: set[str], all_six_solve: set[str]) -> dict[str, Any]:
    """Teacher (DeepSeek-V4-Pro) passing trials in trials.parquet, and how many sit on the dev tasks."""
    import pyarrow.parquet as pq

    t = pq.read_table(trials_path, columns=["task", "model", "reward"]).to_pylist()
    passes = [r["task"] for r in t if r["model"] == TEACHER_MODEL and r["reward"] is not None and float(r["reward"]) == 1.0]
    on_dev = Counter(x for x in passes if x in dev_names)
    return {"teacher_model": TEACHER_MODEL, "rule": "trials.parquet rows with model == 'deepseek' and reward == 1",
            "teacher_passes_total": len(passes), "teacher_pass_tasks_total": len(set(passes)),
            "teacher_passes_on_dev_tasks": sum(on_dev.values()), "dev_tasks_with_teacher_passes": len(on_dev),
            "teacher_passes_on_dev_tasks_already_skipped_all_six_solve":
                sum(k for x, k in on_dev.items() if x in all_six_solve),
            "note": "WP4 drops these episodes (dev exclusion); the all-six-solve ones would be skipped by WP4 anyway"}


def build(env_dir: Path = DEFAULT_ENV_DIR, trials_path: Path | None = None, seed: int = paths.SEED,
          revisions: dict[str, str | None] | None = None, log: Callable[[str], Any] = print) -> dict[str, Any]:
    tasks_path, labels_path = Path(env_dir) / "tasks.jsonl", Path(env_dir) / "labels.jsonl"
    tasks, raw_labels = read_jsonl(tasks_path), read_jsonl(labels_path)
    labels = task_labels(raw_labels)
    by_name = {t["task"]: t for t in tasks}
    names, survival = eligible(tasks, labels)
    order, report = draw(names, labels, seed)
    all_six = {x for x, per in labels.items() if all(float(per[m]["pass_rate"]) == 1.0 for m in MODELS if m in per)}
    items = []
    for name in order:
        per = labels[name]
        mean = mean_pass_rate(per)
        items.append({"task_id": f"{SOURCE}:{name}", "upstream_id": name, "task_dir": f"environments/{name}",
                      "checksum": by_name[name]["checksum"],
                      "band": band_of(mean), "mean_pass_rate": round(mean, 4),
                      "pass_rates": {m: per[m]["pass_rate"] for m in MODELS},
                      "all_six_solve": name in all_six, "samples": sample_plan("terminal", f"{SOURCE}:{name}", seed)})
    realised = Counter(it["band"] for it in items)
    if dict(realised) != {b: k for b, k in report["quota"].items() if k}:
        raise RuntimeError(f"realised bands {dict(realised)} differ from the quota {report['quota']}")
    revisions = revisions or {}
    inputs = {"tasks.jsonl": {"repo": HF_REPO, "revision": revisions.get(SOURCE), "sha256": sha256_file(tasks_path)},
              "labels.jsonl": {"repo": HF_REPO, "revision": revisions.get(SOURCE), "sha256": sha256_file(labels_path)}}
    training_impact = None
    if trials_path is not None:
        inputs["data/trials.parquet"] = {"repo": TRIALS_REPO, "revision": revisions.get("tgr"),
                                         "sha256": sha256_file(trials_path)}
        training_impact = teacher_passes(Path(trials_path), set(order), all_six)
    bands_of_eligible = report["supply"]
    log(f"dev_terminal: {len(names)} eligible ({bands_of_eligible}); quota {report['quota']}")
    return {
        "slice": "terminal", "source": SOURCE, "frozen": True, "seed": seed,
        "plan": "docs/DATA_PLAN.md section 10: 40 TermGrade tasks x 3 samples (Harbor Terminus-2 under the Terminal-Bench "
                "lock's settings); reduced set = 40 x 2 (sample indices 0 and 1)",
        "inputs": inputs,
        "split_used": {"value": SPLIT,
                       "meaning": "card of ai-and/termgrade-environments: the 802 tasks the authors never used to train "
                                  "or validate their own model (train151 and heldout120 were)"},
        "difficulty": {"measure": "unweighted mean of the six models' pass_rate in labels.jsonl",
                       "models": list(MODELS),
                       "bands": {"easy": ">= 2/3", "medium": ">= 1/3 and < 2/3", "hard": "> 0 and < 1/3",
                                 "unsolved": "== 0 (no counted trial of any model passed)"},
                       "why_not_the_band_field": "labels.jsonl's per-model band measures distance from a 50 percent "
                                                 "pass rate, not difficulty"},
        "draw": {"rule": f"{N_TASKS} tasks; >= {MIN_PER_BAND} per band, rest proportional to the band's share of eligible "
                         "tasks (largest remainder); seeded shuffle of sorted names per band, first n taken; manifest "
                         "order is a seeded permutation",
                 "rng_labels": f"'{seed}:{SOURCE}:dev:<band>' and '{seed}:{SOURCE}:dev:order'",
                 "eligible_by_band": bands_of_eligible, "quota": report["quota"],
                 "realised": {b: realised.get(b, 0) for b in BANDS}},
        "survival": survival,
        "pending_task_checks": "the task-level checks of plan section 6 (base image, task image builds in <= 10 min, "
                               "nop fails, oracle passes, agent-visible image scan) run in the TermGrade task pipeline "
                               "(WP8) before the first dev evaluation; a task that fails them stays in the set and its "
                               "samples are recorded as environment failures",
        "training_impact": training_impact,
        "counts": sample_counts(items),
        "items": items,
    }
