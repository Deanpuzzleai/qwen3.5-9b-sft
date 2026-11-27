#!/usr/bin/env python
"""Build the terminal teacher slice (WP4, unit U2.2; docs/DATA_PLAN.md section 3.3, docs/EXECUTION_PLAN.md M2.2):
1,000 DeepSeek-V4-Pro Terminus-2 episodes, two trained turns each, rendered at the 64K target cap and the 48K fallback.

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_terminal_slice.py [--limit 50] [--workers 24]

Inputs (defaults under RAW = /mnt/apex-shared/apex_data/raw; revisions from data/manifests/raw_sources.json)
  --th3k           RAW/th3k/data/TerminalHorizon-3K-SFT.jsonl            shunzou05/TerminalHorizon-3K episodes
  --th3k-env-meta  RAW/th3k_env_meta/metadata/tasks.parquet             TerminalHorizon-Environment categories
  --tgr-messages   RAW/tgr/data/messages/*deepseek*.parquet             TermGrade DeepSeek-V4-Pro trajectories_messages
  --tgr-trials     RAW/tgr/data/trials.parquet                          TermGrade per-trial rewards
  --tgr-labels     RAW/tgr_env/labels.jsonl                             per-model pass rates (all-six-solve skip)
  data/manifests/decontam_exclusions.json (th3k_episode_index), data/manifests/dev_exclusions.json
  (terminal_task_ids), configs/eval/terminal_bench_task_ids.txt (the 89 public task names, D4 slug screen only),
  the Qwen3.5-9B tokenizer (paths.BASE_MODEL_DIR), Harbor 0.24.0's Terminus-2 parser file in .venv/harbor (round trip).
Outputs
  --out-dir (default data/pools/terminal_teacher; with --limit: WORK/<run_id>/pool_preview)
    rows_64k.parquet, rows_48k.parquet   ROW_COLUMNS + PROVENANCE_FIELDS + bookkeeping (apex_data.terminal.rows);
                                         git-ignored when > 10 MB, sha256 in the manifest
    episodes.jsonl                       the selected episodes: source, label, environment, category group, turns and
                                         tokens of their rows at both caps
    manifest.json                        inputs, funnel per source, harness match, per-provenance counts, the category
                                         quota table, the TermGrade skip numbers, selection checks, row checks
    admission_report.json                admission pilot numbers and the pass / fail decision per source
  WORK/<run_id>/render_<cap>/            render shards (all candidates at both caps; reused by --resume)
  logs/<run_id>/                         metadata.json, train.log, metrics.jsonl, errors.log, resources.csv,
                                         render_check.jsonl (the 20-row render check), dropped.jsonl (every drop)

Steps: load both corpora (harness match: user #1 under our Harbor template, reasoning split off) -> link TerminalHorizon
episodes to environments -> exclusions (episode 2481, dev tasks) -> screens (D1, D4, identity, credentials) -> per-task
dedup (Terminus-2 task text; across sources and against the dev tasks) -> episode gates and per-turn rendering at 64K and
48K (render/filters.filter_episodes, two turns per episode) -> admission decision -> selection (category quota, one
episode per environment; TermGrade skip, <= 2 per task) -> row files -> row checks (render, parser round trip on every row).
Seeded by --seed (apex_data.paths.SEED); --resume reuses the render shards of the same --run-id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths  # noqa: E402
from apex_data.render.adapters import TERMINAL_CONFIRMATION_SHARE, is_terminus_confirmation  # noqa: E402
from apex_data.render.filters import CAP_FALLBACK, CAP_TARGET, GATES, FilterConfig, filter_episodes  # noqa: E402
from apex_data.schema import validate_provenance  # noqa: E402
from apex_data.screens import d1_canary, d4_structural, dedup  # noqa: E402
from apex_data.terminal import admission, checks, harness, linking, loaders, rows as rowsio, select  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402
from apex_train.data_format import load_tokenizer  # noqa: E402

CAPS = {"64k": CAP_TARGET, "48k": CAP_FALLBACK}
SOURCES = ("th3k", "tgr")
PINNED = ("th3k", "th3k_env_meta", "tgr", "tgr_env")
CHUNK = 6
RENDER_CHECK_ROWS = 20
GIT_LIMIT_BYTES = 10 * 1024 * 1024


# --------------------------------------------------------------------------------------------------------- inputs
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--th3k", type=Path, default=paths.RAW / "th3k" / "data" / "TerminalHorizon-3K-SFT.jsonl")
    p.add_argument("--th3k-env-meta", type=Path, default=paths.RAW / "th3k_env_meta" / "metadata" / "tasks.parquet")
    p.add_argument("--tgr-messages", type=Path, default=None,
                   help="DeepSeek-V4-Pro messages parquet (default: the one RAW/tgr/data/messages/*deepseek*.parquet)")
    p.add_argument("--tgr-trials", type=Path, default=paths.RAW / "tgr" / "data" / "trials.parquet")
    p.add_argument("--tgr-labels", type=Path, default=paths.RAW / "tgr_env" / "labels.jsonl")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES)
    p.add_argument("--decontam-exclusions", type=Path, default=paths.DECONTAM_EXCLUSIONS)
    p.add_argument("--dev-exclusions", type=Path, default=paths.DEV_EXCLUSIONS)
    p.add_argument("--allow-missing-dev", action="store_true",
                   help="development only: run without dev_exclusions.json (recorded; the outputs are not final)")
    p.add_argument("--allow-unpinned", action="store_true",
                   help="development only: run on files without a pinned revision in raw_sources.json (recorded)")
    p.add_argument("--th3k-admission", choices=("auto", "force", "deny"), default="auto",
                   help="auto: the admission pilot decides (default); force / deny override it for a comparison build "
                        "(recorded in the manifest; never the frozen pool without an explicit decision)")
    p.add_argument("--max-turn-tokens", type=int, default=None, metavar="N",
                   help="a turn whose trained tokens (reasoning + JSON action + end token) exceed N is not a candidate: the "
                        "latest fitting turn is the latest within N, the random turn is drawn among the earlier ones within "
                        "N; an episode with none fails as render_turn_too_long (default: no limit)")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None, help="default data/pools/terminal_teacher "
                   "(WORK/<run_id>/pool_preview with --limit)")
    p.add_argument("--work-dir", type=Path, default=None, help="default WORK/<run_id>")
    p.add_argument("--workers", type=int, default=24, help="processes for loading, rendering and checks (default 24)")
    p.add_argument("--limit", type=int, default=None, help="first N records of each corpus (small end-to-end runs)")
    p.add_argument("--seed", type=int, default=paths.SEED)
    p.add_argument("--run-id", default=None, help="logs/<run_id>/ (default data_terminal_slice_<UTC timestamp>)")
    p.add_argument("--resume", action="store_true", help="continue --run-id, reusing its render shards")
    p.add_argument("--log-root", type=Path, default=None, help="write logs/<run_id>/ under this root (tests)")
    return p.parse_args(argv)


def raw_sources(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    d = d.get("sources", d)
    return {k: v for k, v in d.items() if isinstance(v, dict)}


def verify_inputs(inputs: dict[str, Path], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """sha256 of every input file against its raw_sources.json entry (the file under the source's `dest`)."""
    out: dict[str, Any] = {}
    for name, path in inputs.items():
        path = Path(path).resolve()
        rec: dict[str, Any] = {"path": str(path), "sha256": rowsio.sha256_file(path), "pinned_sha256": None, "source": None}
        for key, src in sources.items():
            dest = Path(src.get("dest") or "/nonexistent").resolve()
            if dest in path.parents:
                rel = str(path.relative_to(dest))
                hit = next((f for f in src.get("files") or [] if f.get("path") == rel), None)
                rec.update(source=key, revision=src.get("revision"), pinned_sha256=hit and hit.get("sha256"))
        rec["matches"] = rec["sha256"] == rec["pinned_sha256"]
        out[name] = rec
    return out


def find_tgr_messages(a: argparse.Namespace) -> Path:
    if a.tgr_messages:
        return a.tgr_messages
    cands = sorted((paths.RAW / "tgr" / "data" / "messages").glob("*deepseek*.parquet"))
    if len(cands) != 1:
        raise SystemExit(f"cannot pick the DeepSeek-V4-Pro messages parquet under RAW/tgr/data/messages: {cands}; "
                         "pass --tgr-messages")
    return cands[0]


def read_exclusions(a: argparse.Namespace) -> tuple[set[int], set[str] | None]:
    dec = json.loads(a.decontam_exclusions.read_text())
    th_idx = {int(x) for x in dec["exclude"].get("th3k_episode_index", [])}
    if not a.dev_exclusions.exists():
        if not a.allow_missing_dev:
            raise SystemExit(f"{a.dev_exclusions} is missing (WP6 writes it at the dev freeze); "
                             "--allow-missing-dev runs without it for development only")
        return th_idx, None
    dev = json.loads(a.dev_exclusions.read_text())
    return th_idx, set(dev.get("terminal_task_ids") or [])


# -------------------------------------------------------------------------------------------- per-record state
@dataclass
class Rec:
    source: str
    loaded: loaders.Loaded
    screens: dict[str, Any] | None = None
    link: dict[str, Any] | None = None
    stage: str = "candidate"            # the funnel stage that dropped it, or candidate / eligible / selected
    reason: str = ""
    detail: str = ""
    render: dict[str, dict[str, Any]] = field(default_factory=dict)    # cap -> worker meta of this episode
    unit: str | None = None             # dedup unit (TerminalHorizon: environment + near-duplicate cluster; TermGrade: task)

    @property
    def eid(self) -> str | None:
        return self.loaded.episode["episode_id"] if self.loaded.episode else None

    def drop(self, stage: str, reason: str, detail: str = "") -> None:
        if self.stage == "candidate":
            self.stage, self.reason, self.detail = stage, reason, detail


# ---------------------------------------------------------------------------------------- load (worker pool)
_G: dict[str, Any] = {}     # fork-inherited worker globals


def _load_worker(item: tuple[str, Any]) -> tuple[loaders.Loaded, dict[str, Any] | None, dict[str, Any] | None]:
    kind, payload = item
    if kind == "th3k":
        index, line = payload
        record = json.loads(line)
        conv = record.get("conversations") or []
        first = (conv[0].get("content") or "") if conv else ""
        try:
            instruction = harness.match_first_prompt(first).instruction
        except harness.HarnessError:
            instruction = dedup.terminal_task_text(first)
        lk = _G["env_index"].link(instruction)
        meta = _G["env_index"].meta(lk.env)
        group = select.group_of(meta["category"])
        loaded = loaders.load_th3k_record(index, record, revision=_G["revisions"].get("th3k"), env=lk.env,
                                          category=meta["category"], category_group=group,
                                          subcategory=meta["subcategory"])
        link = {"env": lk.env, "score": lk.score, "runner_up": lk.runner_up, "reason": lk.reason}
        name = lk.env
    else:
        row, trial_reward, started = payload
        loaded = loaders.load_tgr_record(row, revision=_G["revisions"].get("tgr"), trial_reward=trial_reward,
                                         started_at=started)
        link, name = None, str(row["task"])
    screens = None
    if loaded.episode is not None:
        screens = admission.screen_episode(loaded.episode["messages"], loaded.info.get("instruction") or "", name,
                                           _G["slug_patterns"])
    return loaded, link, screens


def load_all(a: argparse.Namespace, run: RunDir, revisions: dict[str, str | None], tgr_messages: Path,
             envs: list[dict[str, Any]]) -> list[Rec]:
    _G.update(env_index=linking.EnvIndex(envs), revisions=revisions,
              slug_patterns=d4_structural.slug_patterns(d4_structural.load_tb_slugs()))

    def items():
        for n, (i, line) in enumerate(_th3k_lines(a.th3k)):
            if a.limit is not None and n >= a.limit:
                break
            yield "th3k", (i, line)
        for n, (row, rew, started) in enumerate(loaders.read_tgr(tgr_messages, a.tgr_trials)):
            if a.limit is not None and n >= a.limit:
                break
            yield "tgr", (row, rew, started)

    recs: list[Rec] = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, (loaded, link, screens) in enumerate(pool.imap(_load_worker, items(), chunksize=2), 1):
            recs.append(Rec(loaded.source, loaded, screens=screens, link=link))
            if n % 500 == 0:
                run.log(f"loaded {n} records")
    return recs


def _th3k_lines(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, line


# ------------------------------------------------------------------------------------------ render (worker pool)
def _render_init(tokenizer_dir: str, max_turn_tokens: int | None = None) -> None:
    _G["tok"] = load_tokenizer(tokenizer_dir)
    _G["max_turn_tokens"] = max_turn_tokens


def _render_worker(job: tuple[str, int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    shard, cap, items = job
    shard_path, meta_path = Path(shard + ".parquet"), Path(shard + ".json")
    eps = [it["episode"] for it in items]
    verdicts = {it["episode"]["episode_id"]: it["verdict"] for it in items}
    tasks = {it["task"]["task_id"]: it["task"] for it in items}
    sub = {it["episode"]["episode_id"]: (it["task"]["extra"] or {}).get("subcategory") for it in items}
    res = filter_episodes(eps, verdicts, tasks, _G["tok"], FilterConfig(max_row_tokens=cap,
                                                                          max_turn_completion_tokens=_G["max_turn_tokens"]))
    records = []
    for r in res.kept:
        validate_provenance(r.provenance)
        records.append(rowsio.flat_row(r.row, r.provenance, sub[r.provenance["episode_id"]]))
    rowsio.write_table(records, shard_path)
    by_ep: dict[str, dict[str, Any]] = {e["episode_id"]: {"episode_id": e["episode_id"], "kept": False, "rows": []}
                                        for e in eps}
    for rec in records:
        m = by_ep[rec["episode_id"]]
        m["kept"] = True
        m["rows"].append({k: rec[k] for k in ("row_key", "turn_index", "n_tokens", "n_assistant_tokens", "fits_48k")})
    for d in res.dropped:
        by_ep[d.episode_id].update(gate=d.gate, reason=d.reason, detail=d.detail[:200])
    surv = {k: v.to_dict() for k, v in res.survival.items()}
    out = [{**m, "survival": surv if i == 0 else None} for i, m in enumerate(by_ep.values())]
    tmp = meta_path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(out))
    tmp.replace(meta_path)
    return out


def render_all(cands: list[Rec], cap_name: str, a: argparse.Namespace, work: Path,
               run: RunDir) -> tuple[dict[str, Any], list[Path]]:
    """Render every candidate at one cap; shards are keyed by their episode ids, so --resume skips finished ones."""
    cap = CAPS[cap_name]
    out_dir = work / f"render_{cap_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cands, key=lambda r: r.eid)
    jobs, done, shards = [], [], []
    for i in range(0, len(ordered), CHUNK):
        part = ordered[i:i + CHUNK]
        key = hashlib.sha256("\n".join(r.eid for r in part).encode()).hexdigest()[:12]
        shard = str(out_dir / f"chunk_{i // CHUNK:05d}_{key}")
        shards.append(Path(shard + ".parquet"))
        if Path(shard + ".json").exists() and Path(shard + ".parquet").exists():
            done.append(json.loads(Path(shard + ".json").read_text()))
            continue
        jobs.append((shard, cap, [{"episode": r.loaded.episode, "verdict": r.loaded.verdict, "task": r.loaded.task}
                                  for r in part]))
    run.log(f"render {cap_name}: {len(jobs)} shards to build, {len(done)} reused ({len(ordered)} episodes)")
    jobs.sort(key=lambda j: -sum(len(json.dumps(it["episode"]["messages"])) for it in j[2]))   # long shards first
    with mp.get_context("spawn").Pool(a.workers, initializer=_render_init,
                                      initargs=(str(a.tokenizer), a.max_turn_tokens)) as pool:
        for n, meta in enumerate(pool.imap_unordered(_render_worker, jobs), 1):
            done.append(meta)
            if n % 50 == 0 or n == len(jobs):
                run.log(f"render {cap_name}: {n}/{len(jobs)} shards")
    by_eid = {m["episode_id"]: m for shard in done for m in shard}
    for r in cands:
        r.render[cap_name] = by_eid[r.eid]
    return merge_survival([m["survival"] for shard in done for m in shard if m.get("survival")]), sorted(shards)


def merge_survival(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        for kind, s in part.items():
            o = out.setdefault(kind, {"gates": {g: {"n_in": 0, "n_dropped": 0, "n_out": 0} for g in GATES},
                                      "reasons": Counter(), "stats": Counter()})
            for row in s["gates"]:
                for k in ("n_in", "n_dropped", "n_out"):
                    o["gates"][row["gate"]][k] += row[k]
            o["reasons"].update(s["reasons"])
            o["stats"].update(s["stats"])
    return {k: {"gates": [{"gate": g, **c} for g, c in v["gates"].items()], "reasons": dict(sorted(v["reasons"].items())),
                "stats": dict(sorted(v["stats"].items()))} for k, v in out.items()}


# --------------------------------------------------------------------------------------------- funnel steps
def apply_exclusions(recs: list[Rec], th_idx: set[int], dev_tasks: set[str] | None) -> None:
    for r in recs:
        if r.loaded.drop:
            r.drop("load", r.loaded.drop, r.loaded.detail)
            continue
        if r.source == "th3k" and r.loaded.info["index"] in th_idx:
            r.drop("exclusions", "decontam_exclusions_th3k_index")
        elif r.source == "tgr" and dev_tasks is not None and r.loaded.info["task"] in dev_tasks:
            r.drop("exclusions", "dev_task")


def apply_d1_file_hits(recs: list[Rec], hits: dict[str, set[str]]) -> None:
    for r in recs:
        key = f"row={r.loaded.info['index']}" if r.source == "th3k" else r.loaded.info["trial"]
        r.loaded.info["d1_file_hit"] = key in hits[r.source]


def apply_screens(recs: list[Rec]) -> None:
    for r in recs:
        if r.stage != "candidate" or r.screens is None:
            continue
        reason = admission.screen_drop_reason(r.screens, r.loaded.info.get("d1_file_hit", False))
        if reason:
            detail = {"d1_canary": ",".join(r.screens["d1_reasons"]) or "whole-file scan",
                      "d4_structural": ",".join(r.screens["d4_reasons"]),
                      "identity_self_mention": ",".join(r.screens["identity"]),
                      "credential_pattern": ",".join(r.screens["credentials"])}[reason]
            r.drop("screens", reason, detail)


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def apply_dedup(recs: list[Rec], dev_tasks: set[str] | None) -> dict[str, Any]:
    """Terminus-2 task-text clusters (exact key + MinHash Jaccard >= 0.8 on word 5-grams, screens.dedup) over every
    record with an instruction; dev-task trials enter as the dev side. A training episode clustered with a dev task is
    dropped; a TerminalHorizon episode clustered with a TermGrade task is dropped (the verified side stays); within a
    source, clusters become dedup units (TerminalHorizon: joined with the environment; TermGrade: joined with the task)."""
    items, owner = [], {}
    for k, r in enumerate(recs):
        text = r.loaded.info.get("instruction")
        if not text:
            continue
        is_dev = r.source == "tgr" and dev_tasks is not None and r.loaded.info["task"] in dev_tasks
        iid = f"{k}"
        items.append(dedup.DedupItem(iid, text, split="dev" if is_dev else "train"))
        owner[iid] = (r, is_dev)
    cl = dedup.terminal_task_clusters(items)
    members: dict[str, list[str]] = defaultdict(list)
    for iid in owner:
        members[cl.group_of.get(iid, iid)].append(iid)
    uf = _UF()
    stats = Counter()
    for rep, ids in members.items():
        recs_in = [owner[i] for i in ids]
        has_dev = any(d for _, d in recs_in)
        sources = {r.source for r, _ in recs_in}
        th_envs = {r.loaded.info.get("env") or r.eid for r, _ in recs_in if r.source == "th3k"}
        tg_tasks = {r.loaded.info["task"] for r, _ in recs_in if r.source == "tgr"}
        stats["clusters"] += 1
        stats["clusters_multi_environment"] += len(th_envs) > 1
        stats["clusters_multi_task_tgr"] += len(tg_tasks) > 1
        stats["clusters_cross_source"] += len(sources) > 1
        stats["clusters_with_dev"] += has_dev
        for r, d in recs_in:
            if d:
                continue
            if has_dev:
                r.drop("dedup", "near_duplicate_of_dev_task", rep)
            elif r.source == "th3k" and "tgr" in sources:
                r.drop("dedup", "near_duplicate_of_termgrade_task", rep)
            unit_key = f"{r.source}:~cluster:{rep}"        # '~' sorts after the names, so a unit is named by its smallest env / task
            base = f"th3k:env:{r.loaded.info.get('env') or r.eid}" if r.source == "th3k" else f"tgr:task:{r.loaded.info['task']}"
            uf.union(base, unit_key)
            r.unit = base
    for r in recs:
        if r.unit:
            r.unit = uf.find(r.unit)
    return {"method": "screens.dedup.terminal_task_clusters (exact normalised task text + MinHash-LSH 25x10, Jaccard >= "
                      f"{dedup.TERMINAL_THRESHOLD} on word {dedup.TERMINAL_NGRAM}-grams)", "items": len(items),
            **dict(stats), **{f"minhash_{k}": v for k, v in cl.counts().items()}}


def apply_render_outcome(recs: list[Rec]) -> None:
    for r in recs:
        if r.stage != "candidate":
            continue
        for cap in CAPS:
            m = r.render.get(cap)
            if m is None or not m["kept"]:
                r.drop(f"gates_{cap}", (m or {}).get("reason", "not_rendered"), (m or {}).get("detail", ""))
                break
        if r.stage == "candidate":
            if r.source == "th3k" and not r.loaded.info.get("category_group"):
                r.drop("link", "unlinked_or_unknown_category", (r.link or {}).get("reason", ""))
            else:
                r.stage = "eligible"


# --------------------------------------------------------------------------------------------------- reports
def harness_report(recs: list[Rec], src: str) -> dict[str, Any]:
    rs = [r for r in recs if r.source == src and r.loaded.info.get("first_prompt_status")]
    status = Counter(r.loaded.info["first_prompt_status"] for r in rs)
    diffs = Counter(d for r in rs for d in r.loaded.info.get("first_prompt_differences") or [])
    init = Counter(r.loaded.info.get("initial_state_kind") for r in rs)
    obs = Counter()
    for r in rs:
        obs.update(r.loaded.info.get("observation_kinds") or {})
    return {"records_with_first_prompt": len(rs), "first_prompt_status": dict(status), "differences": dict(diffs),
            "initial_terminal_state_kind": dict(init), "observation_kinds": dict(obs),
            "system_message_in_output": 0, "template": f"Harbor {harness.HARBOR_VERSION} terminus-json-plain.txt "
                                                      f"sha256 {harness.TEMPLATE_SHA256}"}


def admission_stats(recs: list[Rec], src: str, d1_file: dict[str, Any]) -> dict[str, Any]:
    """Admission numbers over every record with a transcript (TermGrade: the reward == 1 DeepSeek-V4-Pro episodes)."""
    if src == "tgr":
        pool = [r for r in recs if r.source == "tgr" and r.loaded.info.get("reward") == 1.0 and "n_assistant" in r.loaded.info]
    else:
        pool = [r for r in recs if r.source == "th3k" and "n_assistant" in r.loaded.info]
    info = [r.loaded.info for r in pool]
    parse_clean = sum(i["strict_parse_failures"] == 0 and i["parse_error_turns"] == 0 and not i.get("upstream_unparsed_turns")
                      for i in info)
    every_turn_reasoning = sum(i["turns_without_reasoning"] == 0 for i in info)
    eligible = [r for r in recs if r.source == src and r.stage in ("eligible", "selected")]
    if src == "th3k":
        families = len({r.loaded.info.get("subcategory") for r in eligible if r.loaded.info.get("subcategory")})
        fam_def = "distinct TerminalHorizon environment subcategories among the eligible episodes"
    else:
        families = len({r.unit for r in eligible})
        fam_def = "distinct task clusters (screens.dedup terminal task text) among the eligible episodes"
    screened = [r.screens for r in pool if r.screens is not None]
    d1_review = sum(bool(r.screens and r.screens["d1"]) or r.loaded.info.get("d1_file_hit", False) for r in pool)
    stats = {
        "episodes": len(pool), "parse_clean": parse_clean, "parse_clean_share": round(parse_clean / len(pool), 4) if pool else 0,
        "handshake": sum(bool(i.get("handshake")) for i in info),
        "every_assistant_turn_has_reasoning": every_turn_reasoning,
        "assistant_turns": sum(i["n_assistant"] for i in info),
        "assistant_turns_failing_strict_parser": sum(i["strict_parse_failures"] for i in info),
        "parse_error_user_turns": sum(i["parse_error_turns"] for i in info),
        "upstream_unparsed_turns": sum(i.get("upstream_unparsed_turns", 0) for i in info),
        "d1_whole_file": d1_file, "screens": admission.screen_counts(screened),
        "d3": "pending (sealed screen at M2.4 on the pool and the rows)",
        "eligible_episodes": len(eligible), "families": families,
    }
    stats["decision"] = admission.decide(src, n_episodes=len(pool), parse_clean=parse_clean, handshake=stats["handshake"],
                                         d1_hits_after_review=d1_review, d3="pending", families=families,
                                         family_definition=fam_def, gated=src == "th3k")
    return stats


def funnel(recs: list[Rec], src: str, survival: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rs = [r for r in recs if r.source == src]
    stages = ["load", "exclusions", "screens", "dedup", "gates_64k", "gates_48k", "link", "selection"]
    n = len(rs)
    table = []
    for st in stages:
        dropped = [r for r in rs if r.stage == st]
        table.append({"step": st, "n_in": n, "n_dropped": len(dropped), "n_out": n - len(dropped),
                      "reasons": dict(Counter(r.reason for r in dropped).most_common())})
        n -= len(dropped)
    return {"records": len(rs), "steps": table, "selected": sum(r.stage == "selected" for r in rs),
            "gate_survival": {cap: survival[cap].get("terminal_turn") for cap in CAPS}}


# ----------------------------------------------------------------------------------------------- selection
def select_episodes(recs: list[Rec], a: argparse.Namespace, th_admitted: bool, skip_tasks: set[str],
                    dev_tasks: set[str] | None) -> dict[str, Any]:
    th = [r for r in recs if r.source == "th3k" and r.stage == "eligible"]
    units: dict[str, list[str]] = defaultdict(list)
    by_eid = {r.eid: r for r in recs if r.eid}
    for r in th:
        units[r.unit].append(r.eid)
    chosen = select.one_per_environment(units, a.seed)                     # unit -> episode
    supply: dict[str, list[str]] = defaultdict(list)
    for unit, eid in chosen.items():
        supply[by_eid[eid].loaded.info["category_group"]].append(unit)
    quota = select.quota_fill(supply, select.QUOTAS, a.seed)
    th_sel = [chosen[u] for u in quota.environments] if th_admitted else []
    for eid in th_sel:
        by_eid[eid].stage = "selected"
    tg = {r.eid: r.unit for r in recs if r.source == "tgr" and r.stage == "eligible"}
    unit_tasks: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        if r.source == "tgr" and r.unit:
            unit_tasks[r.unit].add(r.loaded.info["task"])
    skip_units = {u for u, ts in unit_tasks.items() if ts and ts <= skip_tasks}
    dev_units = {u for u, ts in unit_tasks.items() if dev_tasks and ts & dev_tasks}
    tg_target = select.TOTAL_EPISODES - len(th_sel)
    tsel = select.select_termgrade(tg, target=tg_target, skip_tasks=skip_units, dev_tasks=dev_units, seed=a.seed)
    for eid in tsel.selected:
        by_eid[eid].stage = "selected"
    for r in recs:                       # eligible but not drawn: counted as the selection step's drop
        if r.stage == "eligible":
            r.stage, r.reason = "selection", "not_drawn"
    sel_th = [by_eid[e] for e in th_sel]
    envs = [r.loaded.info["env"] for r in sel_th]
    groups_sel = Counter(r.loaded.info["category_group"] for r in sel_th)
    cats_sel = Counter(r.loaded.info["category"] for r in sel_th)
    th_turns_sel = sorted(r.loaded.info["n_assistant"] for r in sel_th)
    th_turns_all = sorted(r.loaded.info["n_assistant"] for r in th)
    return {
        "th3k": {"admitted": th_admitted, "admission_mode": a.th3k_admission,
                 "quota_table_applied": th_admitted, "eligible_episodes": len(th), "eligible_units": len(units),
                 "eligible_environments": len({r.loaded.info['env'] for r in th}), "selected": len(th_sel),
                 "quota_table": quota.table, "unfilled": quota.unfilled if th_admitted else select.TH3K_MAX,
                 "selected_by_group": dict(groups_sel), "selected_by_category": dict(cats_sel.most_common()),
                 "one_episode_per_environment": len(envs) == len(set(envs)),
                 "excluded_index_2481_absent": all(r.loaded.info["index"] != 2481 for r in sel_th),
                 "length_check_turns_median": {"selected": _median(th_turns_sel), "eligible": _median(th_turns_all)}},
        "tgr": {**tsel.counts, "dev_tasks_absent": not any(dev_tasks and by_eid[e].loaded.info["task"] in dev_tasks
                                                              for e in tsel.selected),
                "selected_on_all_six_solve_tasks": sum(by_eid[e].loaded.info["task"] in skip_tasks for e in tsel.selected),
                "max_episodes_per_task": max(Counter(by_eid[e].loaded.info["task"] for e in tsel.selected).values(), default=0),
                "length_check_turns_median": {"selected": _median(sorted(by_eid[e].loaded.info["n_assistant"] for e in tsel.selected)),
                                              "eligible": _median(sorted(by_eid[e].loaded.info["n_assistant"] for e in tg))}},
        "total_selected": len(th_sel) + len(tsel.selected),
    }


def _median(xs: list[int]) -> float | None:
    return xs[len(xs) // 2] if xs else None


def tgr_skip_numbers(recs: list[Rec], skip_tasks: set[str], n_labelled: int) -> dict[str, Any]:
    passing = [r for r in recs if r.source == "tgr" and r.loaded.info.get("reward") == 1.0]
    on_skip = [r for r in passing if r.loaded.info["task"] in skip_tasks]
    left = [r for r in passing if r.loaded.info["task"] not in skip_tasks]
    return {"labelled_tasks": n_labelled, "all_six_solve_tasks": len(skip_tasks),
            "passing_episodes": len(passing), "passing_tasks": len({r.loaded.info["task"] for r in passing}),
            "passing_episodes_on_all_six_solve_tasks": len(on_skip),
            "passing_episodes_left_after_skip": len(left), "tasks_left_after_skip": len({r.loaded.info["task"] for r in left})}


# ------------------------------------------------------------------------------------------- outputs / checks
def collect_rows(recs: list[Rec], shards: list[Path]) -> list[dict[str, Any]]:
    """The rows of the selected episodes from this run's render shards (stale shards of other runs are never read)."""
    unit = {r.eid: r.unit for r in recs if r.stage == "selected"}
    out: list[dict[str, Any]] = []
    for shard in shards:
        out.extend(rowsio.read_table(shard, episode_ids=set(unit)))
    for row in out:                       # dup_group = the dedup unit (task-text cluster joined with its env / task)
        row["dup_group"] = unit[row["episode_id"]]
    order = {"th3k": 0, "tgr": 1}
    out.sort(key=lambda x: (order[x["task_id"].split(":", 1)[0]], x["episode_id"], x["turn_index"]))
    return out


def _check_worker(job: tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]) -> list[tuple[str, list[str], bool]]:
    rows, msgs = job
    return [(r["row_key"], checks.check_row(_G["tok"], r, msgs[r["episode_id"]]),
             checks.nfc_changed(msgs[r["episode_id"]][r["turn_index"]])) for r in rows]


def check_rows(rows_by_cap: dict[str, list[dict[str, Any]]], recs: list[Rec], a: argparse.Namespace,
               run: RunDir) -> dict[str, Any]:
    msgs = {r.eid: r.loaded.episode["messages"] for r in recs if r.stage == "selected"}
    out: dict[str, Any] = {"harbor_parser": str(harness.HARBOR_PARSER_FILE) if checks.harbor_parser() else "absent"}
    with mp.get_context("spawn").Pool(a.workers, initializer=_render_init, initargs=(str(a.tokenizer),)) as pool:
        for cap, rows in rows_by_cap.items():
            jobs = []
            for i in range(0, len(rows), 8):
                part = rows[i:i + 8]
                jobs.append((part, {r["episode_id"]: msgs[r["episode_id"]] for r in part}))
            problems = Counter()
            bad: list[dict[str, Any]] = []
            n = n_nfc = 0
            for res in pool.imap_unordered(_check_worker, jobs):
                for key, probs, changed in res:
                    n += 1
                    n_nfc += changed
                    problems.update(probs)
                    if probs:
                        bad.append({"row_key": key, "problems": probs})
            out[cap] = {"rows_checked": n, "rows_clean": n - len(bad), "problems": dict(problems), "examples": bad[:10],
                        "trained_turns_not_in_nfc": n_nfc,
                        "nfc_note": "the tokenizer normalises text to NFC on encode (training and serving alike); "
                                    "such turns are compared in NFC"}
            run.log(f"row checks {cap}: {n - len(bad)}/{n} clean {dict(problems)}")
    return out


def render_check(rows: list[dict[str, Any]], recs: list[Rec], a: argparse.Namespace, path: Path) -> dict[str, Any]:
    msgs = {r.eid: r.loaded.episode["messages"] for r in recs if r.stage == "selected"}
    rng = random.Random(f"{a.seed}:render_check")
    picks: list[dict[str, Any]] = []
    by_src = {src: [r for r in rows if r["task_id"].startswith(src + ":")] for src in SOURCES}
    for k, src in enumerate(SOURCES):          # half per source; a source without rows leaves its half to the other
        want = (RENDER_CHECK_ROWS - len(picks)) if k == len(SOURCES) - 1 else RENDER_CHECK_ROWS // 2
        if k == 0 and not by_src[SOURCES[1]]:
            want = RENDER_CHECK_ROWS
        picks += rng.sample(by_src[src], min(want, len(by_src[src])))
    tok = load_tokenizer(a.tokenizer)
    ex = [checks.render_excerpt(tok, r, msgs[r["episode_id"]]) for r in picks]
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ex))
    return {"rows": len(ex), "history_think_blocks_total": sum(e["history_think_blocks"] for e in ex),
            "rows_with_history": sum(e["history_assistant_turns"] > 0 for e in ex),
            "trained_turn_with_reasoning": sum(e["trained_reasoning_chars"] > 0 for e in ex),
            "rows_with_problems": sum(bool(e["problems"]) for e in ex), "file": str(path)}


def _third(k: int, n: int) -> str:
    return "early" if 3 * k < n else "middle" if 3 * k < 2 * n else "late"


def turn_report(rows: list[dict[str, Any]], recs: list[Rec]) -> dict[str, Any]:
    """Which turns the rows train: confirmation replies (the answer to Terminus-2's task_complete prompt), the position
    of the latest-fitting and the random turn in thirds of the episode's assistant turns, and episodes short of rows."""
    msgs = {r.eid: r.loaded.episode["messages"] for r in recs if r.stage == "selected"}
    by_ep: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_ep[row["episode_id"]].append(row["turn_index"])
    pos = {"latest_fitting": Counter(), "random": Counter()}
    confirmation = Counter()
    for eid, turns in by_ep.items():
        m = msgs[eid]
        asst = [i for i, x in enumerate(m) if x["role"] == "assistant"]
        for j, t in enumerate(sorted(turns)):
            kind = "latest_fitting" if j == len(turns) - 1 else "random"
            pos[kind][_third(asst.index(t), len(asst))] += 1
            if t > 0 and m[t - 1]["role"] == "user" and is_terminus_confirmation(m[t - 1]["content"] or ""):
                confirmation[kind] += 1
    return {"confirmation_reply_rows": sum(confirmation.values()), "confirmation_reply_rows_by_kind": dict(confirmation),
            "episodes_training_a_confirmation_reply": sum(
                any(t > 0 and is_terminus_confirmation(msgs[e][t - 1]["content"] or "") for t in ts) for e, ts in by_ep.items()),
            "position_thirds": {k: dict(v) for k, v in pos.items()},
            "episodes_by_row_count": dict(Counter(len(v) for v in by_ep.values())),
            "selected_episodes_without_rows": len(msgs) - len(by_ep)}


def episode_records(recs: list[Rec], rows_by_cap: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    per: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cap, rows in rows_by_cap.items():
        for r in rows:
            per[r["episode_id"]][cap].append({"turn_index": r["turn_index"], "n_tokens": r["n_tokens"],
                                              "n_assistant_tokens": r["n_assistant_tokens"]})
    out = []
    for r in recs:
        if r.stage != "selected":
            continue
        i = r.loaded.info
        prov = r.loaded.task["extra"]["provenance"]
        out.append({"episode_id": r.eid, "task_id": r.loaded.task["task_id"], "source": r.source,
                    "label_provenance": prov["label_provenance"], "generator": prov["generator"],
                    "upstream": i.get("index") if r.source == "th3k" else i.get("trial"),
                    "environment": i.get("env"), "category": i.get("category"), "category_group": i.get("category_group"),
                    "subcategory": i.get("subcategory"), "dedup_unit": r.unit, "n_assistant_turns": i["n_assistant"],
                    "rows": {cap: per[r.eid][cap] for cap in CAPS}})
    out.sort(key=lambda x: (x["source"] != "th3k", x["episode_id"]))
    return out


def write_json(path: Path, obj: Any) -> str:
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_id = a.run_id or new_run_id("data_terminal_slice")
    work = (a.work_dir or paths.WORK / run_id).resolve()
    out_dir = (a.out_dir or (work / "pool_preview" if a.limit else paths.POOLS / "terminal_teacher")).resolve()
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        sources = raw_sources(a.raw_sources)
        revisions = {k: (sources.get(k) or {}).get("revision") for k in PINNED}
        unpinned = [k for k in PINNED if not revisions.get(k)]
        if unpinned and not a.allow_unpinned:
            raise SystemExit(f"no pinned revision for {unpinned} in {a.raw_sources}; --allow-unpinned for development")
        revisions = {k: v or "unpinned" for k, v in revisions.items()}
        tgr_messages = find_tgr_messages(a)
        th_idx, dev_tasks = read_exclusions(a)
        drift = harness.template_drift()
        if drift:
            raise SystemExit(drift)
        inputs = {"th3k": a.th3k, "th3k_env_meta": a.th3k_env_meta, "tgr_messages": tgr_messages,
                  "tgr_trials": a.tgr_trials, "tgr_labels": a.tgr_labels}
        input_check = verify_inputs(inputs, sources)
        mismatched = [k for k, v in input_check.items() if not v["matches"]]
        if mismatched and not a.allow_unpinned:
            raise SystemExit(f"inputs not matching raw_sources.json sha256: {mismatched}")
        run.write_metadata(run_id=run_id, step="data_terminal_slice", command=command_line(), git=git_info(),
                           code_commit=git_info()["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions=revisions, inputs={k: str(v) for k, v in inputs.items()},
                           model_revision=str(a.tokenizer), hardware=hardware_info(), seed=a.seed,
                           start_time=utc_now(), status="running", work_dir=str(work), out_dir=str(out_dir),
                           dev_exclusions=str(a.dev_exclusions) if dev_tasks is not None else "missing (development run)")
        work.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        import pyarrow.parquet as pq
        envs = pq.read_table(a.th3k_env_meta, columns=["task_id", "category", "subcategory", "instruction"]).to_pylist()
        unknown_cats = sorted({e["category"] for e in envs} - set(select.CATEGORY_GROUP))
        if unknown_cats:
            raise SystemExit(f"environment categories outside the quota table: {unknown_cats}")
        skip_tasks, n_labelled = loaders.all_six_solve(a.tgr_labels)
        run.log(f"inputs: {len(envs)} environments, {len(skip_tasks)} all-six-solve TermGrade tasks of {n_labelled}; "
                f"dev tasks {'missing' if dev_tasks is None else len(dev_tasks)}; th3k exclusions {sorted(th_idx)}")

        # 1. load + link + per-episode screens
        recs = load_all(a, run, revisions, tgr_messages, envs)
        run.log(f"loaded {len(recs)} records: {dict(Counter(r.source for r in recs))}")
        links = [linking.Link(r.link["env"], r.link["score"], 0.0, r.link["runner_up"], r.link["reason"])
                 for r in recs if r.source == "th3k"]
        link_summary = linking.link_summary(links)
        run.metric({"step": "link", **{k: v for k, v in link_summary.items() if not isinstance(v, dict)}})

        # 2. whole-file D1 scans
        d1_th = d1_canary.scan_jsonl(a.th3k)
        d1_tg = d1_canary.scan_parquet(tgr_messages, id_keys=("trial",))
        d1_file = {"th3k": d1_th.counts(), "tgr": d1_tg.counts()}
        apply_d1_file_hits(recs, {"th3k": {h.record for h in d1_th.hits}, "tgr": {h.record for h in d1_tg.hits}})
        run.log(f"D1 whole-file: th3k {d1_th.counts()['n_hits']} hits, tgr {d1_tg.counts()['n_hits']} hits")

        # 3. exclusions, screens, dedup
        apply_exclusions(recs, th_idx, dev_tasks)
        apply_screens(recs)
        dedup_report = apply_dedup(recs, dev_tasks)
        run.metric({"step": "dedup", **{k: v for k, v in dedup_report.items() if isinstance(v, int)}})
        cands = [r for r in recs if r.stage == "candidate"]
        run.log(f"candidates for rendering: {dict(Counter(r.source for r in cands))}")

        # 4. gates and rendering at both caps
        survival, shards = {}, {}
        for cap in CAPS:
            survival[cap], shards[cap] = render_all(cands, cap, a, work, run)
            run.metric({"step": f"render_{cap}", "kept": sum(r.render[cap]["kept"] for r in cands), "episodes": len(cands)})
        apply_render_outcome(recs)

        # 5. admission
        adm = {src: admission_stats(recs, src, d1_file[src]) for src in SOURCES}
        th_admitted = {"auto": adm["th3k"]["decision"]["admitted"], "force": True, "deny": False}[a.th3k_admission]
        if a.th3k_admission != "auto":
            run.log(f"th3k admission overridden: {a.th3k_admission} (pilot decision: {adm['th3k']['decision']['decision']})")
        run.log(f"admission: th3k {adm['th3k']['decision']['decision']}, tgr {adm['tgr']['decision']['decision']}")

        # 6. selection
        sel = select_episodes(recs, a, th_admitted, skip_tasks, dev_tasks)
        run.log(f"selected: th3k {sel['th3k']['selected']}, tgr {sel['tgr']['selected']}, total {sel['total_selected']}")

        # 7. rows
        rows_by_cap = {cap: collect_rows(recs, shards[cap]) for cap in CAPS}
        files = {}
        for cap, rows in rows_by_cap.items():
            path = out_dir / f"rows_{cap}.parquet"
            sha = rowsio.write_table(rows, path)
            files[path.name] = {"sha256": sha, "bytes": path.stat().st_size, "rows": len(rows),
                                "git": "keep out of git (> 10 MB; sha256 here)" if path.stat().st_size > GIT_LIMIT_BYTES
                                else "small enough to track"}
            run.log(f"wrote {path} ({len(rows)} rows, {path.stat().st_size / 1e6:.1f} MB)")
        light = {cap: [{k: r[k] for k in ("episode_id", "n_tokens", "n_assistant_tokens", "label_provenance", "fits_48k")}
                       for r in rows] for cap, rows in rows_by_cap.items()}
        tokens = {cap: rowsio.token_stats(v) for cap, v in light.items()}
        eps = episode_records(recs, rows_by_cap)
        ep_path = out_dir / "episodes.jsonl"
        ep_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))
        files[ep_path.name] = {"sha256": rowsio.sha256_file(ep_path), "bytes": ep_path.stat().st_size, "rows": len(eps)}

        # 8. checks
        row_checks = check_rows(rows_by_cap, recs, a, run)
        rc = render_check(rows_by_cap["64k"], recs, a, run.run_dir / "render_check.jsonl")
        final_d1 = d1_canary.scan_records(({"episode_id": r.eid, "messages": r.loaded.episode["messages"]}
                                           for r in recs if r.stage == "selected"), source="selected_episodes").counts()
        missing_turns = {cap: sum(len(e["rows"][cap]) != 2 for e in eps) for cap in CAPS}

        # 9. drops, reports, manifest
        with (run.run_dir / "dropped.jsonl").open("w") as f:
            for r in recs:
                if r.stage not in ("selected",):
                    f.write(json.dumps({"source": r.source, "upstream": r.loaded.upstream_id, "episode_id": r.eid,
                                        "stage": r.stage, "reason": r.reason, "detail": r.detail[:200]}) + "\n")
        funnels = {src: funnel(recs, src, survival) for src in SOURCES}
        prov_counts = {cap: {p: {"episodes": b["episodes"], "rows": b["rows"], "sequence_tokens": b["sequence_tokens"],
                                 "loss_tokens": b["loss_tokens"]} for p, b in tokens[cap]["by_label_provenance"].items()}
                       for cap in CAPS}
        admission_report = {
            "plan": "docs/DATA_PLAN.md 3.3 (admission pilot) and 6", "run_id": run_id, "created": utc_now(),
            "thresholds": {"parse_clean_share_min": admission.PARSE_CLEAN_MIN, "d1_hits_after_review": admission.D1_MAX,
                           "d3_hits_after_review": admission.D3_MAX, "task_families_min": admission.FAMILIES_MIN},
            "sources": adm,
            "th3k_admission_mode": a.th3k_admission,
            "fallback_applied": None if th_admitted else (
                "TerminalHorizon not admitted: TermGrade supplies all 1,000 episodes (plan 3.3); the all-six-solve tasks "
                "are re-admitted only as far as the supply after the dev exclusion falls short, and counted"),
            "review_notes": {"d1": "every D1 hit (whole-file and task text) is listed in logs/<run_id>/dropped.jsonl with "
                                   "stage screens; zero hits means nothing to review",
                             "d3": "pending until M2.4 (sealed screen on the pool and the rows, WP7)"},
        }
        write_json(out_dir / "admission_report.json", admission_report)
        final = dev_tasks is not None and not unpinned and a.limit is None and a.th3k_admission == "auto"
        manifest = {
            "pool": "terminal_teacher", "unit": "U2.2 (WP4)", "plan": "docs/DATA_PLAN.md rev 4.3.6 section 3.3",
            "run_id": run_id, "created": utc_now(), "code_commit": git_info()["commit"], "seed": a.seed,
            "final": final, "development_flags": {"limit": a.limit, "dev_exclusions_missing": dev_tasks is None,
                                                  "unpinned_sources": unpinned, "th3k_admission": a.th3k_admission},
            "inputs": input_check,
            "exclusions": {"th3k_episode_index": sorted(th_idx),
                           "decontam_exclusions_sha256": rowsio.sha256_file(a.decontam_exclusions),
                           "dev_terminal_task_ids": None if dev_tasks is None else len(dev_tasks),
                           "dev_exclusions_sha256": rowsio.sha256_file(a.dev_exclusions) if dev_tasks is not None else None},
            "files": files, "harness_match": {src: harness_report(recs, src) for src in SOURCES},
            "linking": link_summary, "dedup": dedup_report, "funnel": funnels,
            "termgrade_skip": tgr_skip_numbers(recs, skip_tasks, n_labelled), "selection": sel,
            "per_provenance": prov_counts, "tokens": tokens,
            "trained_turns": {"confirmation_reply_rule": "replies to the task_complete confirmation prompt are trainable "
                                                         f"only in a seeded {TERMINAL_CONFIRMATION_SHARE:.0%} of episodes "
                                                         "(render/adapters.trains_confirmation)",
                              **{cap: turn_report(rows, recs) for cap, rows in rows_by_cap.items()}},
            "checks": {"rows_per_episode_not_2": missing_turns, "row_checks": row_checks, "render_check_20": rc,
                       "d1_selected_episodes": final_d1,
                       "dev_tasks_absent": sel["tgr"]["dev_tasks_absent"],
                       "one_episode_per_environment": sel["th3k"]["one_episode_per_environment"],
                       "episode_2481_absent": sel["th3k"]["excluded_index_2481_absent"],
                       "selected_th3k_labels": "authors_filter+handshake", "selected_tgr_labels": "verifier"},
            "admission": {src: adm[src]["decision"] for src in SOURCES},
        }
        if a.max_turn_tokens is not None:
            manifest["max_turn_completion_tokens"] = {
                "cap": a.max_turn_tokens,
                "rule": "a turn whose trained tokens (its contiguous assistant_masks span: reasoning, JSON action, end token) "
                        "exceed the cap is not a candidate for the two trained turns; an episode with no candidate fails "
                        "the render gate as render_turn_too_long (funnel gates_64k / gates_48k) and the selection fills "
                        "from the remaining supply",
                "episodes_dropped": {cap: sum(1 for r in recs if r.stage == f"gates_{cap}" and r.reason == "render_turn_too_long")
                                     for cap in CAPS},
                "episodes_with_one_row": {cap: sum(len(e["rows"][cap]) == 1 for e in eps) for cap in CAPS},
                "max_trained_tokens_per_row": {cap: max((r["n_assistant_tokens"] for r in rows), default=0)
                                               for cap, rows in rows_by_cap.items()}}
        write_json(out_dir / "manifest.json", manifest)
        run.metric({"step": "done", "selected": sel["total_selected"], **{f"rows_{c}": len(v) for c, v in rows_by_cap.items()}})
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("workers", a.workers, "processes", "loading, rendering and checks"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        ok = all(row_checks[c]["rows_checked"] == row_checks[c]["rows_clean"] for c in CAPS)
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if ok else "failed",
                           outcome="terminal slice built" + ("" if final else " (development run, not final)"),
                           summary={"selected": sel["total_selected"], "th3k": sel["th3k"]["selected"],
                                    "tgr": sel["tgr"]["selected"], "rows": {c: len(v) for c, v in rows_by_cap.items()},
                                    "admission": {s: adm[s]["decision"]["decision"] for s in SOURCES}},
                           outputs={k: v for k, v in files.items()})
        if not ok:
            run.error(f"row checks failed: {json.dumps({c: row_checks[c]['problems'] for c in CAPS})}")
        run.finalize()
        run.log(f"done in {run.elapsed():.0f} s; outputs in {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
