"""TermGrade environments (source `tgr_env`): task index, the seeded stage-2 draw and the pool records of the terminal
task pool (docs/DATA_PLAN.md sections 6, 10 and 11; WP8 / unit U3.2).

Upstream (`ai-and/termgrade-environments`, Apache-2.0, card read 2026-10-12): 1,004 Harbor task directories
`environments/<task>/{task.toml, instruction.md, environment/{Dockerfile,setup.sh}, tests/{test.sh,test_outputs.py},
solution/solve.sh}`; every task Dockerfile is `FROM dsbase-venv312:local`, built once from `base_image/Dockerfile`, whose
`/opt/verifier_venv` runs every `tests/test.sh` offline. `tasks.jsonl` gives per task `checksum`, `gradable`,
`oracle_verified` and the authors' own `split`; `labels.jsonl` gives per (task, model) the measured pass rate of six
models (DeepSeek-V4-Pro, our stage-1 teacher, among them, k = 2).

Pool rules
  dev       the 40 tasks frozen by WP6 (`data/manifests/dev_terminal.json`, ids also in `dev_exclusions.json`); this pool
            only checks that they build and grade, it never changes the dev draw.
  stage2    a seeded uniform permutation of every other task (`stage2_order`); the first N are the candidates, later
            positions are the top-up order used only when fewer than the target pass the task checks.
  excluded  a dev or stage-2 candidate that failed a task check (build, nop, oracle, image scan); the record stays with
            `extra.env_check` and `extra.exclusion` (nothing is patched, nothing deleted).
Teacher overlap (plan section 11: allowed and recorded): `extra.teacher` carries the number of DeepSeek-V4-Pro trials
and passes on the task (labels.jsonl) and whether the task is in WP4's admissible supply (>= 1 teacher pass and not
solved by all six labelled models), i.e. whether stage-1 rows can come from the same environment.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from apex_data import paths
from apex_data.schema import Task, read_jsonl, sha256_text, task_id

SOURCE = "tgr_env"
HF_REPO = "ai-and/termgrade-environments"
DATASET_LICENSE = "apache-2.0"
RAW_DIR = paths.RAW / SOURCE                                # fetch_raw.py destination (raw_sources.json key "tgr_env")
WORK_DIR = paths.WORK / SOURCE                              # stable: prepared task copies, build logs, per-task status
PREPARED_DIR = WORK_DIR / "tasks"                           # the Harbor dataset directory every trial runs from (`-p`)
POOL_DIR = paths.POOLS / SOURCE                             # tasks.jsonl + manifest.json (tracked)
BASE_IMAGE = "dsbase-venv312:local"                         # the tag every task Dockerfile names in its FROM line
IMAGE_PREFIX = "tgr-"
LABEL_MODELS = ("gemma", "gemma_think", "glm", "kimi", "qwen", "deepseek")
TEACHER_MODEL = "deepseek"                                  # labels.jsonl key of DeepSeek-V4-Pro (plan 3.3)
DRAW_LABEL = "stage2"                                       # rng stream: random.Random(f"{seed}:tgr_env:stage2")
_SLUG = re.compile(r"^task_[0-9]{5}_[0-9a-f]{6}$")
# An instruction that names /tests (the hidden suite's mount point, absent until grading): an agent following it trips
# the terminal tamper rule `verifier_touched`, so such tasks rarely yield stage-2 positives; recorded per task.
_TESTS_PATH = re.compile(r"(^|[\s'\"=:;|&(`])/tests(/|\b)")


# --- upstream index -------------------------------------------------------------------------------------------------
def check_slug(slug: str) -> str:
    """Upstream task names are `task_<5 digits>_<6 hex>`; anything else is refused (it becomes a docker tag and a path)."""
    if not _SLUG.match(slug):
        raise ValueError(f"unexpected TermGrade task name {slug!r}")
    return slug


def image_tag(slug: str) -> str:
    """Local image of one task: `tgr-<task>:local` (valid docker reference; the prefix keeps our images apart)."""
    return f"{IMAGE_PREFIX}{check_slug(slug)}:local"


def load_index(raw: Path = RAW_DIR) -> dict[str, dict[str, Any]]:
    """slug -> {checksum, gradable, oracle_verified, upstream_split, labels: {model: {pass_rate, k, n_counted}}}."""
    index: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(raw / "tasks.jsonl"):
        slug = check_slug(r["task"])
        if slug in index:
            raise ValueError(f"duplicate task {slug} in tasks.jsonl")
        index[slug] = {"checksum": r.get("checksum"), "gradable": bool(r.get("gradable")),
                       "oracle_verified": bool(r.get("oracle_verified")), "upstream_split": r.get("split"), "labels": {}}
    for r in read_jsonl(raw / "labels.jsonl"):
        per = index.get(r["task"])
        if per is None:
            raise ValueError(f"labels.jsonl names a task missing from tasks.jsonl: {r['task']}")
        if r["model"] in per["labels"]:
            raise ValueError(f"duplicate label {r['task']} / {r['model']}")
        per["labels"][r["model"]] = {"pass_rate": float(r["pass_rate"]), "k": int(r["k"]), "n_counted": int(r["n_counted"])}
    return index


def teacher_overlap(info: dict[str, Any]) -> dict[str, Any]:
    """DeepSeek-V4-Pro coverage of one task in termgrade-trajectories, from its labels (pass_rate x n_counted trials)."""
    lab = info["labels"]
    t = lab.get(TEACHER_MODEL)
    trials = t["n_counted"] if t else 0
    passes = round(t["pass_rate"] * t["n_counted"]) if t else 0
    all_six = set(lab) >= set(LABEL_MODELS) and all(lab[m]["pass_rate"] == 1.0 for m in LABEL_MODELS)
    return {"teacher_trials": trials, "teacher_passes": passes, "all_six_solve": all_six,
            "stage1_supply": passes > 0 and not all_six}


def mean_pass_rate(info: dict[str, Any]) -> float | None:
    lab = info["labels"]
    if not set(lab) >= set(LABEL_MODELS):
        return None
    return sum(lab[m]["pass_rate"] for m in LABEL_MODELS) / len(LABEL_MODELS)


# --- dev tasks and the stage-2 draw ---------------------------------------------------------------------------------
def dev_slugs(dev_terminal: Path = paths.MANIFESTS / "dev_terminal.json",
              dev_exclusions: Path = paths.DEV_EXCLUSIONS) -> list[str]:
    """The frozen terminal dev tasks in manifest order. `terminal_task_ids` of dev_exclusions.json (upstream ids, a
    `tgr_env:` prefix tolerated) must name the same set: two views of one freeze; a disagreement is an error here."""
    man = json.loads(dev_terminal.read_text())
    order = [check_slug(it["upstream_id"]) for it in man["items"]]
    if len(set(order)) != len(order):
        raise ValueError("duplicate task in dev_terminal.json")
    if dev_exclusions.exists():
        ids = json.loads(dev_exclusions.read_text()).get("terminal_task_ids")
        if ids is not None:
            excl = {check_slug(i.split(":", 1)[1] if i.startswith(SOURCE + ":") else i) for i in ids}
            if excl != set(order):
                raise ValueError(f"dev_exclusions.json terminal_task_ids differ from dev_terminal.json: {sorted(excl ^ set(order))[:5]}")
    return order


def verify_pinned(raw: Path, slugs: Iterable[str], entry: dict[str, Any]) -> list[str]:
    """Files of the given task directories whose sha256 differs from the fetch manifest (raw_sources.json entry `files`),
    or that the manifest does not list / that are missing on disk. Empty list = the snapshot is the pinned one."""
    pinned = {f["path"]: f["sha256"] for f in entry.get("files") or []}
    bad = []
    for slug in slugs:
        prefix = f"environments/{slug}/"
        want = {k: v for k, v in pinned.items() if k.startswith(prefix)}
        have = {p.relative_to(raw).as_posix(): p for p in (raw / "environments" / slug).rglob("*") if p.is_file()}
        for rel in sorted(set(want) | set(have)):
            if rel not in have or rel not in want or hashlib.sha256(have[rel].read_bytes()).hexdigest() != want[rel]:
                bad.append(rel)
    return bad


def stage2_order(all_slugs: Iterable[str], dev: Iterable[str], seed: int = paths.SEED) -> list[str]:
    """Every non-dev task in one seeded uniform permutation (sorted names shuffled by a dedicated stream). The first N are
    the stage-2 candidates; the rest is the top-up order. Deterministic for (task set, dev set, seed)."""
    dev_set = set(dev)
    pool = sorted(s for s in set(all_slugs) if s not in dev_set)
    random.Random(f"{seed}:{SOURCE}:{DRAW_LABEL}").shuffle(pool)
    return pool


# --- task directories -------------------------------------------------------------------------------------------------
def tree_sha256(root: Path) -> str:
    """sha256 over (relative path, bytes) of every file under root, sorted: identifies the exact task content built."""
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def task_meta(task_dir: Path) -> dict[str, Any]:
    """What a pool record needs from one upstream task directory (task.toml metadata and limits, content hashes)."""
    cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    env, agent, ver = cfg.get("environment") or {}, cfg.get("agent") or {}, cfg.get("verifier") or {}
    meta = cfg.get("metadata") or {}
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    return {"category": meta.get("category"), "difficulty": meta.get("difficulty"),
            "network_mode": env.get("network_mode") or ("no-network" if env.get("allow_internet") is False else "public"),
            "upstream_docker_image": env.get("docker_image"), "agent_timeout_sec": agent.get("timeout_sec"),
            "verifier_timeout_sec": ver.get("timeout_sec"), "cpus": env.get("cpus"), "memory_mb": env.get("memory_mb"),
            "instruction_sha256": sha256_text(instruction),
            "instruction_names_tests": bool(_TESTS_PATH.search(instruction)),
            "tree_sha256": tree_sha256(task_dir)}


def pool_record(slug: str, split: str, revision: str | None, info: dict[str, Any], meta: dict[str, Any],
                draw_rank: int | None) -> Task:
    """One schema.Task for the pool file. `draw_rank` = position in stage2_order (None for dev tasks). `extra.task_dir`
    is relative to paths.RAW (the pinned upstream copy, the task source); `extra.harbor_task_dir` is the prepared copy
    Harbor runs (identical except task.toml's docker_image)."""
    lab = info["labels"]
    mean = mean_pass_rate(info)
    t = Task(task_id=task_id(SOURCE, slug), source=SOURCE, family="terminal", language="bash", repo=None, upstream_id=slug,
             hf_revision=revision, image=image_tag(slug), problem_statement_sha=meta["instruction_sha256"],
             tests={"test_cmd": "bash /tests/test.sh", "verifier": "harbor in-trial tests/test.sh -> /logs/verifier/reward.txt"},
             licence={"dataset": DATASET_LICENSE, "status": "ok"}, split=split,
             stratum={"category": meta["category"], "difficulty": meta["difficulty"],
                      "mean_pass_rate": None if mean is None else round(mean, 4),
                      "pass_rates": {m: lab[m]["pass_rate"] for m in LABEL_MODELS if m in lab}},
             extra={"task_dir": f"{SOURCE}/environments/{slug}", "raw_task_dir": str(RAW_DIR / "environments" / slug),
                    "harbor_task_dir": str(PREPARED_DIR / slug), "draw_rank": draw_rank,
                    "upstream_split": info["upstream_split"], "checksum": info["checksum"],
                    "oracle_verified_upstream": info["oracle_verified"], "teacher": teacher_overlap(info),
                    "network_mode": meta["network_mode"], "tree_sha256": meta["tree_sha256"],
                    "instruction_names_tests": meta["instruction_names_tests"]})
    t.validate()
    return t
