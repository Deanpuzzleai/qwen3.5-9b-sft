#!/usr/bin/env python
"""Build the retention pool and the stage-2 replay reserve (WP5, U2.3; docs/DATA_PLAN.md sections 3.4, 6.3, 7, 11).

Run from the repository root:
    PYTHONPATH=src .venv/train/bin/python scripts/data/build_retention_slice.py [--small 20] [--run-id ID --resume]

Parts and rows (SFT / reserve): math 630 / 216 (Nemotron-SFT-Math-v4 AoPS chain-of-thought, reasoning <= 10K tokens,
length bands 55/25/20 percent), IF 435 / 149 and chat 285 / 98 (Nemotron-SFT-Instruction-Following-Chat-v3; reasoning
<= 6K; chat >= 40 percent humanities / social science / knowledge by the bge-small prototype classifier), science MCQ
275 / 94 and RQA 125 / 43 (Nemotron-Science-v1; reasoning <= 10K). The reserve is ~600 rows in the same proportions,
disjoint from the SFT rows by source row and by dedup cluster.

Steps (each writes into the work directory, so --resume skips what is done):
  1 load      apex_data.retention.{math,ifchat,science} loaders over RAW (structural and answer / constraint / format
              checks, counted per stage)                                        -> WORK/<run_id>/candidates_<part>.jsonl
  2 measure   exact reasoning tokens (Qwen3.5-9B tokenizer) and, for chat, the topic (bge-small prototypes) along each
              part's seeded shuffle until every stratum holds 1.5 x its SFT + reserve rows under the think cap
  3 screens   on the drawn rows of all parts: D1 canary, D4 task-name screen (the public Terminal-Bench task names of
              configs/eval/terminal_bench_task_ids.txt as whole tokens in the prompt), prompt dedup (exact + MinHash,
              screens.dedup, one row per cluster), D5' against the dev QA items (cosine >= 0.85 drops the prompt)
  4 gates     render.filters.filter_episodes (QA-level gates: closed reasoning, think caps, no repetition loop,
              constraints, render through data_format episode mode, <= 12,288 tokens) on <= 16 processes
  5 draw      per part and stratum, SFT rows then reserve rows from the same seeded shuffle of the survivors
  6 write     outputs below, then the M2.3 checks recorded in the manifest (exit 1 if any fails)

Inputs
  RAW/nm_math/data/train-0000[0-2]-of-00012.parquet, RAW/nm_ifchat/data/{instruction_following,chat}.jsonl,
  RAW/nm_sci/data/{MCQ,RQA}.jsonl                     pinned in data/manifests/raw_sources.json (nm_math, nm_ifchat, nm_sci)
  paths.BASE_MODEL_DIR (tokenizer and chat template), paths.BGE_SMALL_DIR (bge-small-en-v1.5)
  data/manifests/dev_qa.json + WORK/dev_qa_cache/     dev QA items, rebuilt in memory for the D5' screen only
  src/apex_data/retention/topic_check_labels.json     hand labels of the classifier check
Outputs
  data/pools/retention/rows.parquet            SFT rows: data_format.ROW_COLUMNS + schema.PROVENANCE_FIELDS + selection keys
  data/pools/retention/replay_reserve.parquet  the stage-2 replay reserve, same columns
  data/pools/retention/manifest.json           sources (revisions, licences), per-part funnels, counts, math band shares,
                                               chat topic share and classifier check, D1 / dedup / D5' counts, tokens,
                                               reserve disjointness, file sha256, the M2.3 checks
  WORK/<run_id>/                               candidates, drawn rows, dropped records with reasons, selected episodes
  logs/<run_id>/                               metadata.json, train.log, metrics.jsonl, errors.log, resources.csv
--small N builds N SFT rows and N // 2 reserve rows per part into WORK/<run_id>/small/ (a smoke run; nothing under
data/pools/ is written). --candidates-from RUN_ID reuses step 1 of an earlier run (WORK/RUN_ID/candidates_<part>.jsonl
and funnel_<part>.json, the loaders' output for the same pinned files) instead of reading RAW.

Top-up mode (--topup N [--exclude-pool DIR] --out-dir DIR): N extra SFT rows, in the SFT part proportions (largest
remainder over math 630 / IF 435 / chat 285 / MCQ 275 / RQA 125) and under the same per-part strata rules, that the
mixture can add without touching the pool's rows. The pool's rows.parquet, replay_reserve.parquet and manifest.json are
read, never written; the top-up rows are disjoint from both by source row, prompt and dedup cluster.
  1 load      the candidates of the pool's own run (manifest run_id; --candidates-from overrides)
  2 measure   the same seeded measurement walk, continued until every stratum also holds 1.5 x its top-up rows (+10)
              under the think cap; the first rows per stratum must be exactly the pool run's drawn rows (else exit 1)
  3 screens   D1, D4, prompt dedup and D5' on the new rows; the dedup clusters the new rows together with every row the
              pool run drew, and a new row in a cluster with any of those rows is dropped (the earlier rows win)
  4 gates     the same render.filters gates on the pool run's surviving rows and on the new rows
  5 draw      the SFT and reserve draw is redone and must reproduce rows.parquet and replay_reserve.parquet byte for
              byte (else exit 1); per stratum the top-up takes the survivors that follow the SFT and reserve rows in the
              same seeded shuffle, then the new survivors in their own seeded shuffle (draw.draw_topup); an original
              survivor that a new row links to an SFT or reserve row (or to a lower-id survivor) is skipped
  6 order     draw.interleave over the (part, stratum) cells: the rows are written in the Webster divisor order, so the
              first K rows (topup_rank < K) keep the part and stratum proportions up to rounding for every K
              (the per-cell worst-case deviation over all prefixes is recorded in the manifest)
  Outputs     <out-dir>/topup_rows.parquet (the pool's row columns + topup_rank, sorted by it),
              <out-dir>/topup_manifest.json (counts, tokens, funnels, screens, draw, prefix checks, sha256s),
              WORK/<run_id>/topup_screen_items.jsonl (p1 screen items of the top-up rows for scripts/data/run_screens.py;
              the commands for the p1 and p2 sealed screens are written to the log and the manifest)

Convergence mode (--convergence-replay N_TOKENS [--exclude-pool DIR] [--topup-rows FILE]): the stage-2 replay with the
math and science share replaced by rows that reason at length and then commit to one answer in a short final section
(apex_data.retention.convergence: hard math 10K-24K reasoning 60 percent, medium math 4K-10K 15 percent, MCQ science
ending with the chosen letter 25 percent of the N_TOKENS loss tokens; visible answer <= 300 tokens; row cap 64K), combined
with the replay reserve's IF and chat rows verbatim. Disjoint by source row and dedup cluster from rows.parquet (incl.
the merged top-up rows), replay_reserve.parquet and the top-up rows file (the pool manifest's topup_merge record,
sha256-checked); none of them is written.
  1 load      math from RAW at the 24K think cap (the pool's math loader); MCQ = the pool run's loader output; then the
              counted pre-filters (convergence.prefilter_checks: not a pool row, reference match, one committed answer,
              answer at the end, final-section characters / letter ending)
  2 walk      per source a seeded shuffle; exact reasoning and answer tokens in batches; a row joins the part whose band
              it fits while the part holds < 2 x its loss-token target
  3 screens   D1, D4, prompt dedup (exact + MinHash; the pool, reserve and top-up rows join the clusters and win them), D5'
  4 gates     render.filters gates with the part's reasoning cap and the 64K row cap
  5 sealed    scripts/data/run_screens.py p1 (prompts) and p2 (rendered rows) on every survivor (subprocesses, logged)
  6 select    per part the survivors in walk order until its loss-token target, minus the sealed-screen drops; then
              draw.interleave over the cells if/all, chat/hss, chat/other, math_hard, math_medium, sci_mcq
  Outputs     <out-dir>/convergence_replay.parquet (the pool's row columns + replay_part, final_section_tokens,
              replay_rank; sorted by replay_rank), <out-dir>/convergence_replay_manifest.json (sources, filters, funnels,
              screen drop counts, tokens per part, length distributions, prefix checks, checks, sha256),
              WORK/<run_id>/ (candidates, drawn rows, drops, screen items, sealed-screen manifests and logs)
  --convergence-profile hard_math_v5 (revision-5 hard math) runs the same steps for one part only: math_hard, reasoning
  10,241-12,288 tokens, visible answer <= 600 tokens, no reserve rows; outputs <out-dir>/hard_math.parquet and
  <out-dir>/hard_math_manifest.json (apex_data.retention.convergence.PROFILES)

Compose mode (--compose-v5 N_TOKENS --out-dir DIR [--hard-math DIR] [--slice-tokens 3500000]): the revision-5 retention
slice (apex_data.retention.v5) from the --exclude-pool rows (rows.parquet, read only) and the
hard-math rows of --convergence-profile hard_math_v5 (sha256 and checks of its manifest verified). Loss-token shares
hard math 23 / other math 25 / science 15 / IF 25 / chat 12 of N_TOKENS; within a part the first build's strata keep their
row proportions; the parts are merged by loss tokens, so every prefix keeps the shares (P1e scales the slice by a prefix).
  Outputs     <out-dir>/rows.parquet (the pool's row columns + v5_part, v5_cell, final_section_tokens, v5_rank; sorted
              by v5_rank), <out-dir>/manifest.json (sources and sha256s, targets, the --slice-tokens prefix with its
              hard-math share, prefix shares, strata, supply, checks)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apex_data import paths, schema  # noqa: E402
from apex_data.prompts import checkers, constraints  # noqa: E402
from apex_data.render import filters  # noqa: E402
from apex_data.retention import common, draw, ifchat, math as math_part, science, topics  # noqa: E402
from apex_data.retention.common import MAX_WORKERS, PART_ORDER, PARTS  # noqa: E402
from apex_eval.common import RunDir, command_line, git_info, hardware_info, new_run_id, utc_now  # noqa: E402

POOL_VERSION = "retention-v1"
RAW_KEYS = {"math": "nm_math", "if": "nm_ifchat", "chat": "nm_ifchat", "sci_mcq": "nm_sci", "sci_rqa": "nm_sci"}
ROW_EXTRA = ("row_key", "family", "n_steps", "sha256", "split", "reasoning_tokens", "length_band", "topic", "topic_group",
             "topic_cosine", "n_options", "answer_letter", "constraint_kinds", "ref_match", "boxed_answer", "check",
             "prompt_sha256")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=paths.RAW, help="RAW root (default %(default)s)")
    p.add_argument("--out-dir", type=Path, default=paths.POOLS / "retention", help="pool directory (default %(default)s)")
    p.add_argument("--raw-sources", type=Path, default=paths.RAW_SOURCES, help="pinned revisions (default %(default)s)")
    p.add_argument("--tokenizer", type=Path, default=paths.BASE_MODEL_DIR, help="Qwen3.5-9B tokenizer dir")
    p.add_argument("--seed", type=int, default=paths.SEED, help="seed of every draw (default %(default)s)")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"processes (capped at {MAX_WORKERS})")
    p.add_argument("--reserve-total", type=int, default=common.RESERVE_TOTAL, help="replay reserve rows (default 600)")
    p.add_argument("--small", type=int, default=0, help="smoke run: N SFT rows per part, outputs under WORK only")
    p.add_argument("--parts", default=",".join(PART_ORDER), help="comma-separated parts (smoke runs only)")
    p.add_argument("--run-id", default=None, help="logs/<run_id>/ and WORK/<run_id>/ (default data_retention_<UTC>)")
    p.add_argument("--resume", action="store_true", help="continue an existing run id, reusing finished steps")
    p.add_argument("--log-root", type=Path, default=None, help="write logs/<run_id>/ under this root (tests)")
    p.add_argument("--candidates-from", default=None, metavar="RUN_ID",
                   help="reuse WORK/RUN_ID/candidates_<part>.jsonl and funnel_<part>.json instead of reading RAW")
    p.add_argument("--topup", type=int, default=0, metavar="N",
                   help="top-up mode: N extra SFT rows after the --exclude-pool rows (see above); 0 = the normal build")
    p.add_argument("--exclude-pool", type=Path, default=paths.POOLS / "retention",
                   help="top-up, convergence and compose modes: the existing pool (rows, reserve, manifest; read only) "
                        "(default %(default)s)")
    p.add_argument("--convergence-replay", type=int, default=0, metavar="N_TOKENS",
                   help="convergence mode: N_TOKENS loss tokens of hard math, medium math and MCQ science rows, combined "
                        "with the reserve's IF and chat rows (see above); 0 = off")
    p.add_argument("--convergence-profile", default="stage2_replay", choices=["stage2_replay", "hard_math_v5"],
                   help="convergence mode: stage2_replay (the parts above; default) or hard_math_v5 (math only, reasoning "
                        "10,241-12,288 tokens, final section <= 600, no reserve rows; output <out-dir>/hard_math.parquet)")
    p.add_argument("--compose-v5", type=int, default=0, metavar="N_TOKENS",
                   help="compose mode: the revision-5 retention slice of N_TOKENS loss tokens from the --exclude-pool rows and "
                        "the --hard-math rows, ranked so that every prefix keeps the part shares (see above); 0 = off")
    p.add_argument("--hard-math", type=Path, default=paths.POOLS / "retention_v5",
                   help="compose mode: directory of hard_math.parquet and hard_math_manifest.json (default %(default)s)")
    p.add_argument("--slice-tokens", type=int, default=3_500_000,
                   help="compose mode: the slice's planned loss tokens, reported as a prefix (default %(default)s)")
    p.add_argument("--topup-rows", type=Path, default=None,
                   help="convergence mode: the top-up rows file to stay disjoint from (default: the pool manifest's "
                        "topup_merge.topup.dir, sha256-checked)")
    a = p.parse_args(argv)
    a.workers = max(1, min(a.workers, MAX_WORKERS))
    a.parts = [x.strip() for x in a.parts.split(",") if x.strip()]
    if any(x not in PARTS for x in a.parts):
        p.error(f"unknown part in {a.parts}")
    if not a.small and a.parts != list(PART_ORDER):
        p.error("--parts is for smoke runs (--small) only")
    if a.topup < 0:
        p.error("--topup must be >= 0")
    if a.topup and a.small:
        p.error("--topup builds on a full pool; it does not combine with --small")
    if a.topup and a.out_dir.resolve() == a.exclude_pool.resolve():
        p.error("--topup never writes into the --exclude-pool directory: pass a separate --out-dir")
    if a.convergence_replay < 0:
        p.error("--convergence-replay must be >= 0")
    if a.convergence_replay and (a.topup or a.small):
        p.error("--convergence-replay does not combine with --topup or --small")
    if a.compose_v5 < 0 or (a.compose_v5 and not 0 < a.slice_tokens <= a.compose_v5):
        p.error("--compose-v5 must be >= 0, and with it 0 < --slice-tokens <= --compose-v5")
    if a.compose_v5 and (a.topup or a.small or a.convergence_replay):
        p.error("--compose-v5 does not combine with --topup, --small or --convergence-replay")
    if a.compose_v5 and a.out_dir.resolve() == a.exclude_pool.resolve():
        p.error("--compose-v5 never writes into the --exclude-pool directory: pass a separate --out-dir")
    return a


# ------------------------------------------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_sources(path: Path) -> dict[str, dict[str, Any]]:
    d = json.loads(path.read_text())
    return d.get("sources", d)


_TOK: Any = None


def _tokenizer(path: str) -> Any:
    global _TOK
    if _TOK is None:
        from apex_train.data_format import load_tokenizer
        _TOK = load_tokenizer(path)
    return _TOK


def _count_tokens(tok_path: str, texts: list[str]) -> list[int]:
    tok = _tokenizer(tok_path)
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


class TokenCounter:
    """Exact token counts with the Qwen3.5-9B tokenizer on a persistent pool of <= MAX_WORKERS processes."""

    def __init__(self, tok_path: Path, workers: int) -> None:
        self.tok_path, self.workers = str(tok_path), max(1, min(workers, MAX_WORKERS))
        self.pool = ProcessPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def __call__(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        if self.pool is None:
            return _count_tokens(self.tok_path, texts)
        size = max(1, -(-len(texts) // (self.workers * 4)))
        batches = [texts[i:i + size] for i in range(0, len(texts), size)]
        return [n for part in self.pool.map(_count_tokens, [self.tok_path] * len(batches), batches) for n in part]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


def _gate_chunk(tok_path: str, cfg: dict[str, Any], episodes: list[dict[str, Any]], tasks: dict[str, Any],
                verdicts: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    res = filters.filter_episodes(episodes, verdicts, tasks, _tokenizer(tok_path), filters.FilterConfig(**cfg),
                                  constraint_checker=checkers.filter_checker)
    surv = {k: {"counts": v.counts, "reasons": dict(v.reasons), "stats": dict(v.stats)} for k, v in res.survival.items()}
    return ([(r.row, r.provenance) for r in res.kept], [d.__dict__ for d in res.dropped], surv)


def merge_survival(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"gates": {g: {"n_in": 0, "n_dropped": 0, "n_out": 0} for g in filters.GATES}, "reasons": Counter(),
           "stats": Counter()}
    for s in parts:
        for v in s.values():
            for g, c in v["counts"].items():
                for k in ("n_in", "n_dropped", "n_out"):
                    out["gates"][g][k] += c[k]
            out["reasons"].update(v["reasons"])
            out["stats"].update(v["stats"])
    return {"gates": [{"gate": g, **c} for g, c in out["gates"].items()], "reasons": dict(sorted(out["reasons"].items())),
            "stats": dict(sorted(out["stats"].items()))}


def stage(name: str, n_in: int, n_out: int, note: str = "") -> dict[str, Any]:
    return {"stage": name, "n_in": n_in, "n_dropped": n_in - n_out, "n_out": n_out, **({"note": note} if note else {})}


# ------------------------------------------------------------------------------------------------------ step 1
def load_part(part: str, a: argparse.Namespace, revisions: dict[str, str], run: RunDir, work: Path
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cpath, fpath = work / f"candidates_{part}.jsonl", work / f"funnel_{part}.json"
    if a.resume and cpath.exists() and fpath.exists():
        run.log(f"[load] {part}: reusing {cpath}")
        return read_jsonl(cpath), json.loads(fpath.read_text())
    if a.candidates_from:
        src = paths.WORK / a.candidates_from
        cpath, fpath = src / f"candidates_{part}.jsonl", src / f"funnel_{part}.json"
        run.log(f"[load] {part}: reusing the candidates of run {a.candidates_from} ({cpath})")
        cands, info = read_jsonl(cpath), json.loads(fpath.read_text())
        if len(cands) != info["n_candidates"]:
            raise ValueError(f"{cpath}: {len(cands)} candidates, its funnel says {info['n_candidates']}")
        return cands, info
    raw = a.raw_root / RAW_KEYS[part]
    rev = revisions[RAW_KEYS[part]]
    run.log(f"[load] {part}: reading {raw} @ {rev[:8]} on {a.workers} processes")
    side: dict[str, Any] = {}
    if part == "math":
        cands, funnel = math_part.load(raw, rev, a.workers)
        side = {"ref_match": dict(Counter(c["meta"]["ref_match"] for c in cands))}
    elif part == "if":
        cands, funnel, side = ifchat.load_if(raw, rev, a.workers)
        side = {"constraint_checks": side}
    elif part == "chat":
        cands, funnel, side = ifchat.load_chat(raw, rev, a.workers)
        side = {"seed_datasets": side}
    elif part == "sci_mcq":
        cands, funnel, side = science.load_mcq(raw, rev, a.workers)
        side = {"n_options": side}
    else:
        cands, funnel, side = science.load_rqa(raw, rev, a.workers)
    info = {"funnel": funnel.table(), "side": side, "n_candidates": len(cands)}
    write_jsonl(cpath, cands)
    write_json(fpath, info)
    run.log(f"[load] {part}: {len(cands)} candidates; funnel {[(r['stage'], r['n_out']) for r in funnel.table()]}")
    return cands, info


# ------------------------------------------------------------------------------------------------------ step 2
def measure_and_draw(part: str, cands: list[dict[str, Any]], a: argparse.Namespace, run: RunDir, classifier: Any,
                     counter: TokenCounter, n_sft: int, n_res: int, n_top: int = 0
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Seeded shuffle; topics (chat, all candidates); exact reasoning tokens in batches until each stratum holds its
    oversample target of rows under the think cap. With n_top > 0 (top-up mode) the walk goes on until each stratum
    also holds the top-up's oversample target; per stratum the first rows under the cap are still the normal draw.
    Returns the drawn rows, the top-up rows drawn beyond them (empty without n_top) and the measurement report."""
    spec = PARTS[part]
    report: dict[str, Any] = {}
    order = draw.shuffled(cands, a.seed, part, "measure")
    n_ge5 = {"ge5": sum(c["meta"].get("n_options", 0) >= 5 for c in cands)}
    q_total = draw.quotas(part, n_sft + n_res, n_ge5)
    targets = {s: -(-v * 3 // 2) + (10 if not a.small else 2) for s, v in q_total.items()}
    top_targets = {s: -(-v * 3 // 2) + 10 for s, v in draw.quotas(part, n_top, n_ge5).items()} if n_top else {}
    drawn: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    need = dict(targets)
    need_top = dict(top_targets)
    measured = over_cap = 0
    tokens_all: list[int] = []
    batch = 4000 if not a.small else 200
    i = 0
    while i < len(order) and any(v > 0 for v in (*need.values(), *need_top.values())):
        chunk = order[i:i + batch]
        i += batch
        if part == "chat":                      # topics on the measured prefix of the shuffle (a random sample)
            for c, p in zip(chunk, classifier.classify([c["prompt"] for c in chunk]), strict=True):
                c["meta"].update(topic=p["topic"], topic_group=p["group"], topic_cosine=p["cosine"])
        counts = counter([c["messages"][1]["reasoning_content"] for c in chunk])
        for c, n in zip(chunk, counts, strict=True):
            measured += 1
            tokens_all.append(n)
            c["meta"]["reasoning_tokens"] = n
            if n > spec.think_cap:
                over_cap += 1
                continue
            if part == "math":
                c["meta"]["length_band"] = draw.math_band(n)
            s = draw.stratum_of(part, c)
            if need.get(s, 0) > 0:
                drawn.append(c)
                need[s] -= 1
            elif need_top.get(s, 0) > 0:
                extra.append(c)
                need_top[s] -= 1
    under = [n for n in tokens_all if n <= spec.think_cap]
    report.update({"n_candidates": len(cands), "n_measured": measured, "n_over_think_cap": over_cap,
                   "think_cap": spec.think_cap, "oversample_targets": targets, "unfilled_targets": {k: v for k, v in need.items() if v > 0},
                   "n_drawn": len(drawn), "drawn_strata": dict(Counter(draw.stratum_of(part, c) for c in drawn)),
                   "reasoning_tokens_measured": {"median": statistics.median(tokens_all) if tokens_all else None,
                                                 "share_under_cap": round(len(under) / max(1, measured), 4)}})
    if part == "chat":
        seen = [c["meta"] for c in order[:measured]]
        report["classifier_measured_candidates"] = {
            "n": len(seen), "hss_share": round(sum(m["topic_group"] == topics.HSS for m in seen) / max(1, len(seen)), 4),
            "topics": dict(Counter(m["topic"] for m in seen).most_common())}
    if part == "math":
        bands = Counter(draw.math_band(n) for n in under)
        report["natural_band_shares_under_cap"] = {b: round(bands[b] / max(1, len(under)), 4) for b, _ in draw.MATH_BANDS}
        report["natural_band_counts_under_cap"] = {b: bands[b] for b, _ in draw.MATH_BANDS}
    if n_top:
        report.update({"topup_oversample_targets": top_targets,
                       "topup_unfilled_targets": {k: v for k, v in need_top.items() if v > 0}, "n_topup_drawn": len(extra),
                       "topup_drawn_strata": dict(Counter(draw.stratum_of(part, c) for c in extra))})
    run.log(f"[measure] {part}: measured {measured}, over cap {over_cap}, drawn {len(drawn)} {report['drawn_strata']}"
            + (f", unfilled {report['unfilled_targets']}" if report["unfilled_targets"] else "")
            + (f"; top-up drawn {len(extra)} {report['topup_drawn_strata']}" if n_top else ""))
    return drawn, extra, report


# ------------------------------------------------------------------------------------------------------ step 3
def prior_wins_drops(cl: Any, prior_ids: set[str]) -> dict[str, str]:
    """{dropped_id: kept_id} when earlier rows take precedence: in every cluster each row not in `prior_ids` is dropped,
    except the smallest such id of a cluster that holds no prior row; prior rows are never dropped here (their own
    screen already decided them). The kept id named is the cluster's smallest prior id, else its smallest new id."""
    drops: dict[str, str] = {}
    for members in cl.groups.values():
        old = sorted(m for m in members if m in prior_ids)
        new = sorted(m for m in members if m not in prior_ids)
        keep = old[0] if old else new[0]
        drops.update({m: keep for m in new if m != keep})
    return drops


def run_screens(drawn: list[dict[str, Any]], run: RunDir, a: argparse.Namespace,
                prior: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """D1, D4, prompt dedup and D5' on the drawn rows (marks them in place; returns counts). `prior` (top-up mode): rows
    an earlier run drew and screened; they join the dedup clusters, are never re-marked, and win every cluster."""
    from apex_data.devset import qa
    from apex_data.screens import d1_canary, d4_structural, d5_prime, dedup

    rep: dict[str, Any] = {}
    # D1 canary over the full candidate records (prompt, reasoning, answer)
    d1 = d1_canary.scan_records(drawn, source="retention_drawn")
    hit_ids = {h.record for h in d1.hits}
    for c in drawn:
        c["screens"]["d1"] = "hit" if c["task_id"] in hit_ids else "clean"
        if c["task_id"] in hit_ids:
            c["split"], c["screens"]["exclusion_reason"] = "excluded", "d1_canary"
    rep["d1"] = {**d1.counts(), "n_dropped": len(hit_ids)}
    # D4 name screen: the public Terminal-Bench task names (configs/eval, read-only) as whole tokens in the prompt
    slug_pats = d4_structural.slug_patterns(d4_structural.load_tb_slugs())
    slug_hits = Counter()
    for c in drawn:
        if c["split"] != "train":
            continue
        hits = d4_structural.slug_scan(c["prompt"], slug_pats)
        c["screens"]["d4"] = "hit" if hits else "clean"
        if hits:
            slug_hits[c["part"]] += 1
            c["split"], c["screens"]["exclusion_reason"] = "excluded", "d4_task_name"
    rep["d4_task_names"] = {"n_names": len(slug_pats), "n_scanned": sum(c["screens"].get("d4") is not None for c in drawn),
                            "n_dropped": sum(slug_hits.values()), "dropped_by_part": dict(slug_hits)}
    # prompt dedup: exact normalised prompt + MinHash-LSH (QA-prompt settings of screens.dedup), across all parts
    live = [c for c in drawn if c["split"] == "train"]
    prior = prior or []
    cl = dedup.cluster([dedup.DedupItem(c["task_id"], c["prompt"]) for c in prior + live], dedup.PROMPT_NGRAM,
                       dedup.PROMPT_THRESHOLD, min_tokens=dedup.PROMPT_EXACT_ONLY_UNDER)
    prior_ids = {c["task_id"] for c in prior}
    drops = prior_wins_drops(cl, prior_ids) if prior else cl.drops()
    for c in live:
        c["screens"]["dedup_group"] = cl.group_of.get(c["task_id"], c["task_id"])
        if c["task_id"] in drops:
            c["split"], c["screens"]["exclusion_reason"] = "excluded", f"dedup:{drops[c['task_id']]}"
    part_of = {c["task_id"]: c["part"] for c in prior + live}
    rep["dedup"] = {**cl.counts(), "ngram": dedup.PROMPT_NGRAM, "threshold": dedup.PROMPT_THRESHOLD,
                    "exact_only_under_words": dedup.PROMPT_EXACT_ONLY_UNDER,
                    "cross_part_clusters": sum(len({part_of[m] for m in g}) > 1 for g in cl.groups.values()),
                    "dropped_by_part": dict(Counter(c["part"] for c in live if c["task_id"] in drops))}
    if prior:   # the clusters that hold a new row, for the top-up's disjointness rules and checks (ids only)
        mixed = {rep_id: g for rep_id, g in cl.groups.items() if any(m not in prior_ids for m in g)}
        rep["dedup"].update({"n_prior_items": len(prior), "n_new_items": len(live), "n_dropped": len(drops),
                             "n_dropped_into_prior_clusters": sum(v in prior_ids for v in drops.values()),
                             "note": "n_dropped counts new rows only (prior rows are never dropped here)",
                             "clusters_with_new_rows": mixed})
    # D5': bge-small cosine against the dev QA items; a hit drops the training prompt
    items, dev_report = qa.load_dev_items()
    dev = [{"dev_id": it.dev_id, "slice": it.slice, "text": it.text} for it in items]
    live = [c for c in drawn if c["split"] == "train"]
    embedder = d5_prime.load_embedder()
    d5 = d5_prime.apply(live, dev, embedder, text_key="prompt", log=run.log)
    for c in live:
        info = c["screens"].pop("d5_prime")
        c["screens"]["d5"] = {"max_cosine": info["max_cosine"], "hit": info["hit"]}
    rep["d5_prime"] = {**d5, "dropped_by_part": dict(Counter(c["part"] for c in live if c["split"] == "excluded")),
                       "dev_items_check": dev_report}
    run.log(f"[screens] D1 hits {len(hit_ids)}; D4 task-name hits {sum(slug_hits.values())}; dedup dropped {len(drops)} ({cl.counts()['n_clusters']} clusters); "
            f"D5' dropped {d5['n_dropped_from_train']}")
    return rep


# ------------------------------------------------------------------------------------------------------ step 4
def run_gates(part: str, rows: list[dict[str, Any]], a: argparse.Namespace, run: RunDir, created: str,
              think_cap: int | None = None, row_cap: int | None = None
              ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """render.filters gates on <= a.workers processes; the caps default to the part's think cap and the 12K QA row cap
    (the convergence replay passes its own)."""
    spec = PARTS[part]
    cfg = {"max_qa_tokens": row_cap or common.QA_ROW_CAP, "max_think_tokens": think_cap or spec.think_cap}
    eps = [common.episode_record(c, created) for c in rows]
    tasks = {c["task_id"]: common.task_record(c) for c in rows}
    verdicts = {c["task_id"]: common.verdict_record(c, created, f"{checkers.CHECKER_VERSION}+{constraints.EXTRACTOR_VERSION}")
                for c in rows}
    n = max(1, min(a.workers, len(eps)))
    size = -(-len(eps) // n)
    args = []
    for k in range(0, len(eps), size):
        chunk = eps[k:k + size]
        ids = {e["task_id"] for e in chunk}
        args.append((str(a.tokenizer), cfg, chunk, {t: tasks[t] for t in ids}, {t: verdicts[t] for t in ids}))
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []
    survs = []
    if len(args) == 1:
        results = [_gate_chunk(*args[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(args)) as ex:
            results = list(ex.map(_gate_chunk, *zip(*args)))
    for k_, d_, s_ in results:
        kept += k_
        dropped += d_
        survs.append(s_)
    surv = merge_survival(survs)
    run.log(f"[gates] {part}: {len(kept)} kept, {len(dropped)} dropped {dict(Counter(d['reason'] for d in dropped))}")
    return kept, dropped, surv


# ------------------------------------------------------------------------------------------------------ step 6
def row_record(row: dict[str, Any], prov: dict[str, Any], c: dict[str, Any], split: str) -> dict[str, Any]:
    m = c["meta"]
    extra = {"split": split, "reasoning_tokens": m.get("reasoning_tokens"), "length_band": m.get("length_band"),
             "topic": m.get("topic"), "topic_group": m.get("topic_group"), "topic_cosine": m.get("topic_cosine"),
             "n_options": m.get("n_options"), "answer_letter": m.get("letter"),
             "constraint_kinds": json.dumps(sorted({x["kind"] for x in m.get("constraints") or []})) if c["part"] == "if" else None,
             "ref_match": m.get("ref_match"), "boxed_answer": m.get("boxed"), "check": PARTS[c["part"]].check,
             "prompt_sha256": schema.sha256_text(c["prompt"])}
    out = {**row, **{k: v for k, v in prov.items() if k not in row}, **extra}
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from apex_train.data_format import ROW_COLUMNS, arrow_schema

    base = arrow_schema()
    rest = [k for k in rows[0] if k not in ROW_COLUMNS]
    types = {"fits_48k": pa.bool_(), "patch_files": pa.int32(), "patch_added_lines": pa.int32(), "turn_index": pa.int32(),
             "n_steps": pa.int32(), "reasoning_tokens": pa.int32(), "n_options": pa.int32(), "topic_cosine": pa.float32(),
             "topup_rank": pa.int32(), "replay_rank": pa.int32(), "final_section_tokens": pa.int32(), "v5_rank": pa.int32()}
    sch = pa.schema(list(base) + [pa.field(k, types.get(k, pa.string())) for k in rest])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=sch), tmp, compression="zstd")
    tmp.replace(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "rows": len(rows),
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "git_ignored_over_10mb": path.stat().st_size > 10 * 2 ** 20}


def estimated_true_share(pred_share: float, check: dict[str, Any]) -> float | None:
    """Hand-label-corrected hss share: P(hss | pred hss) x pred share + P(hss | pred other) x (1 - pred share)."""
    conf = (check or {}).get("group_confusion")
    if not conf:
        return None
    pred_h = conf["label_hss__pred_hss"] + conf["label_other__pred_hss"]
    pred_o = conf["label_hss__pred_other"] + conf["label_other__pred_other"]
    p_h = conf["label_hss__pred_hss"] / pred_h if pred_h else 0.0
    p_o = conf["label_hss__pred_other"] / pred_o if pred_o else 0.0
    return round(pred_share * p_h + (1 - pred_share) * p_o, 4)


def token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seq = [r["n_tokens"] for r in rows]
    loss = [r["n_assistant_tokens"] for r in rows]
    rsn = [r["reasoning_tokens"] for r in rows]
    return {"rows": len(rows), "sequence_tokens": sum(seq), "loss_tokens": sum(loss),
            "mean_sequence_tokens": round(statistics.fmean(seq), 1) if seq else None,
            "median_reasoning_tokens": statistics.median(rsn) if rsn else None,
            "max_reasoning_tokens": max(rsn) if rsn else None, "max_row_tokens": max(seq) if seq else None}


def m23_checks(sft: dict[str, list[dict[str, Any]]], res: dict[str, list[dict[str, Any]]], cands: dict[str, dict[str, Any]],
               targets: dict[str, int], res_targets: dict[str, int], tok: Any, small: bool) -> list[dict[str, Any]]:
    """The milestone checks of docs/EXECUTION_PLAN.md M2.3, recomputed from the written rows."""
    out = []

    def add(name: str, ok: bool, detail: Any) -> None:
        out.append({"check": name, "pass": bool(ok), "detail": detail})

    every = [(p, r) for p in sft for r in sft[p] + res[p]]
    add("row_counts", all(len(sft[p]) == targets[p] for p in sft), {p: [len(sft[p]), targets[p]] for p in sft})
    add("reserve_counts", all(len(res[p]) == res_targets[p] for p in res), {p: [len(res[p]), res_targets[p]] for p in res})
    sft_ids = {r["task_id"] for p in sft for r in sft[p]}
    res_ids = {r["task_id"] for p in res for r in res[p]}
    sft_groups = {r["dup_group"] or r["task_id"] for p in sft for r in sft[p]}
    res_groups = {r["dup_group"] or r["task_id"] for p in res for r in res[p]}
    sft_prompts = {r["prompt_sha256"] for p in sft for r in sft[p]}
    res_prompts = {r["prompt_sha256"] for p in res for r in res[p]}
    add("reserve_disjoint", not (sft_ids & res_ids) and not (sft_groups & res_groups) and not (sft_prompts & res_prompts),
        {"shared_task_ids": len(sft_ids & res_ids), "shared_dedup_groups": len(sft_groups & res_groups),
         "shared_prompt_sha256": len(sft_prompts & res_prompts)})
    caps = {p: PARTS[p].think_cap for p in sft}
    over = [r["task_id"] for p, r in every if r["reasoning_tokens"] is None or r["reasoning_tokens"] > caps[p]
            or r["reasoning_tokens"] <= 0]
    add("reasoning_present_and_capped", not over, {"caps": caps, "violations": over[:5]})
    add("row_cap_12288", all(r["n_tokens"] <= common.QA_ROW_CAP for _, r in every), {"max": max(r["n_tokens"] for _, r in every)})
    bad_think = []
    for p, r in every[:: max(1, len(every) // 200)]:          # decode a spread sample of rows: reasoning in the trained span
        trained = tok.decode([t for t, m in zip(r["input_ids"], r["assistant_masks"], strict=True) if m])
        if "</think>" not in trained or not trained.endswith("<|im_end|>") or len(trained.split("</think>")[0].strip()) == 0:
            bad_think.append(r["task_id"])
    add("trained_span_has_reasoning", not bad_think, {"sampled": len(every[:: max(1, len(every) // 200)]), "violations": bad_think[:5]})
    label_fail = []
    for p, r in every:
        c = cands[r["task_id"]]
        ans = c["messages"][1]["content"]
        if p == "if":
            ok = bool(c["meta"]["constraints"]) and checkers.check_all(c["meta"]["constraints"], ans)[0]
        elif p == "sci_mcq":
            info = science.mcq_letters(c["prompt"], ans, c["messages"][1]["reasoning_content"])
            ok = len(info["answer_letters"]) == 1 and info["letter"] in info["options"] and info["letter"] == info["reasoning_letter"]
        elif p == "sci_rqa":
            b = common.boxed_contents(ans)
            ok = len(b) == 1 and 0 < len(b[0]) <= 200 and science.single_question(c["prompt"])
        elif p == "math":
            ok = bool(common.boxed_contents(ans))
        else:
            ok = bool(ans.strip()) and common.is_english(c["prompt"])
        if not ok:
            label_fail.append(r["task_id"])
    add("answer_or_constraint_check", not label_fail, {"violations": label_fail[:5]})
    for name, rows in (("math_band_shares_sft", sft.get("math", [])), ("math_band_shares_reserve", res.get("math", []))):
        if rows:
            got = Counter(r["length_band"] for r in rows)
            shares = {b: round(got[b] / len(rows), 4) for b, _ in draw.MATH_BANDS}
            add(name, all(abs(shares[b] - draw.MATH_BAND_SHARES[b]) <= (0.02 if not small else 0.2) for b in shares),
                {"shares": shares, "target": draw.MATH_BAND_SHARES})
    for name, rows in (("chat_hss_share_sft", sft.get("chat", [])), ("chat_hss_share_reserve", res.get("chat", []))):
        if rows:
            share = sum(r["topic_group"] == topics.HSS for r in rows) / len(rows)
            add(name, share >= draw.CHAT_HSS_FLOOR, {"classifier_hss_share": round(share, 4), "floor": draw.CHAT_HSS_FLOOR})
    add("provenance_valid", all(_valid(r) for _, r in every), {})
    return out


def _valid(r: dict[str, Any]) -> bool:
    try:
        schema.validate_provenance({k: r[k] for k in schema.PROVENANCE_FIELDS})
        return r["row_kind"] == "retention" and r["label_provenance"] == PARTS[r["retention_part"]].label_provenance
    except (schema.SchemaError, KeyError):
        return False


# ------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.topup:
        return main_topup(a)
    if a.convergence_replay:
        return main_convergence(a)
    if a.compose_v5:
        return main_compose_v5(a)
    run_id = a.run_id or new_run_id("data_retention" + ("_small" if a.small else ""))
    work = paths.WORK / run_id
    out_dir = (work / "small") if a.small else a.out_dir
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        srcs = raw_sources(a.raw_sources)
        revisions = {k: srcs[k]["revision"] for k in ("nm_math", "nm_ifchat", "nm_sci")}
        bge_rev = (srcs.get("bge_small") or {}).get("revision")
        created = utc_now()
        n_sft = {p: (a.small if a.small else PARTS[p].n_sft) for p in a.parts}
        n_res = {p: v for p, v in common.reserve_quota(a.reserve_total).items() if p in a.parts}
        if a.small:
            n_res = {p: max(1, a.small // 2) for p in a.parts}
        run.write_metadata(run_id=run_id, step="data_build_retention_slice", command=command_line(), git=git_info(),
                           code_commit=git_info()["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={**revisions, "bge_small": bge_rev}, model_revision=str(a.tokenizer.name),
                           hardware=hardware_info(), seed=a.seed, start_time=created, status="running",
                           targets={"sft": n_sft, "reserve": n_res}, work_dir=str(work))
        work.mkdir(parents=True, exist_ok=True)
        # 1 load
        cands: dict[str, list[dict[str, Any]]] = {}
        load_info: dict[str, Any] = {}
        for p in a.parts:
            cands[p], load_info[p] = load_part(p, a, revisions, run, work)
            run.metric({"step": "load", "part": p, "n_candidates": len(cands[p])})
        # 2 measure and draw the oversample
        tag = "_small" if a.small else ""
        drawn_path, screens_path, measure_path = work / f"drawn{tag}.jsonl", work / f"screens{tag}.json", work / f"measure{tag}.json"
        if a.resume and drawn_path.exists() and screens_path.exists() and measure_path.exists():
            drawn = read_jsonl(drawn_path)
            screen_rep = json.loads(screens_path.read_text())
            measure = json.loads(measure_path.read_text())
            run.log(f"[measure+screens] reusing {drawn_path}")
        else:
            from apex_data.screens import d5_prime
            classifier = topics.TopicClassifier(d5_prime.load_embedder()) if "chat" in a.parts else None
            drawn, measure = [], {}
            counter = TokenCounter(a.tokenizer, a.workers)
            try:
                for p in a.parts:
                    d, _, measure[p] = measure_and_draw(p, cands[p], a, run, classifier, counter, n_sft[p], n_res[p])
                    drawn += d
            finally:
                counter.close()
            if classifier is not None:
                lab = topics.load_labels()
                by = {c["task_id"]: c for c in cands["chat"]}
                ids = [k for k in lab if k in by]
                pred = dict(zip(ids, classifier.classify([by[k]["prompt"] for k in ids])))
                measure["classifier_check"] = topics.check_report(pred, lab)
                run.log(f"[topics] hand-label check: {measure['classifier_check']}")
            screen_rep = run_screens(drawn, run, a)
            write_jsonl(drawn_path, drawn)
            write_json(screens_path, screen_rep)
            write_json(measure_path, measure)
        cand_by_id = {c["task_id"]: c for c in drawn}
        # 4 gates
        kept: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        gate_rep: dict[str, Any] = {}
        all_dropped: list[dict[str, Any]] = []
        for p in a.parts:
            live = [c for c in drawn if c["part"] == p and c["split"] == "train"]
            kept[p], dropped, gate_rep[p] = run_gates(p, live, a, run, created)
            all_dropped += [{**d, "part": p} for d in dropped]
            run.metric({"step": "gates", "part": p, "n_in": len(live), "n_kept": len(kept[p])})
        excluded = [{"task_id": c["task_id"], "part": c["part"], "gate": "screens", "reason": c["screens"].get("exclusion_reason")}
                    for c in drawn if c["split"] == "excluded"]
        write_jsonl(work / f"dropped{tag}.jsonl", excluded + all_dropped)
        # 5 draw SFT and reserve rows from the survivors
        sft_rows: dict[str, list[dict[str, Any]]] = {}
        res_rows: dict[str, list[dict[str, Any]]] = {}
        draw_rep: dict[str, Any] = {}
        for p in a.parts:
            surv = []
            for row, prov in kept[p]:
                c = cand_by_id[prov["task_id"]]
                surv.append({**c, "_row": row, "_prov": prov})
            s, r, draw_rep[p] = draw.draw_sft_and_reserve(p, surv, n_sft[p], n_res[p], a.seed)
            sft_rows[p] = [row_record(x["_row"], x["_prov"], x, "train") for x in s]
            res_rows[p] = [row_record(x["_row"], x["_prov"], x, "replay_reserve") for x in r]
            run.log(f"[draw] {p}: SFT {len(s)} {draw_rep[p]['sft_strata']}, reserve {len(r)} {draw_rep[p]['reserve_strata']}"
                    + (f", reserve shortfall {draw_rep[p]['reserve_shortfall']}" if draw_rep[p]["reserve_shortfall"] else ""))
        # 6 write
        from apex_train.data_format import load_tokenizer
        tok = load_tokenizer(a.tokenizer)
        sft_all = [r for p in a.parts for r in sft_rows[p]]
        res_all = [r for p in a.parts for r in res_rows[p]]
        files = {"rows": write_rows(out_dir / "rows.parquet", sft_all),
                 "replay_reserve": write_rows(out_dir / "replay_reserve.parquet", res_all)}
        selected = [cand_by_id[r["task_id"]] for r in sft_all + res_all]
        write_jsonl(work / f"selected_episodes{tag}.jsonl",
                    [{"task_id": c["task_id"], "part": c["part"], "messages": c["messages"], "meta": c["meta"],
                      "split": "replay_reserve" if c["task_id"] in {r["task_id"] for r in res_all} else "train"} for c in selected])
        from apex_data.screens import d1_canary
        d1_final = d1_canary.scan_records([{"task_id": c["task_id"], "messages": c["messages"]} for c in selected],
                                          source="retention_selected").counts()
        checks = m23_checks(sft_rows, res_rows, cand_by_id, n_sft, n_res, tok, bool(a.small))
        checks.append({"check": "d1_canary_selected_rows", "pass": not d1_final.get("n_hits"), "detail": d1_final})
        checks.append({"check": "d5_prime_screen_applied", "pass": screen_rep["d5_prime"]["n_dev_items"] > 0,
                       "detail": {k: screen_rep["d5_prime"][k] for k in ("n_dev_items", "n_hits", "n_dropped_from_train")}})
        # manifest
        parts_rep: dict[str, Any] = {}
        for p in a.parts:
            spec = PARTS[p]
            m = measure[p]
            dr = [c for c in drawn if c["part"] == p]
            n_d1 = sum(c["screens"].get("exclusion_reason") == "d1_canary" for c in dr)
            n_d4 = sum(c["screens"].get("exclusion_reason") == "d4_task_name" for c in dr)
            n_dd = sum(str(c["screens"].get("exclusion_reason") or "").startswith("dedup") for c in dr)
            n_d5 = sum(c["screens"].get("exclusion_reason") == "d5_prime" for c in dr)
            n_drawn = len(dr)
            funnel = list(load_info[p]["funnel"])
            funnel.append(stage("think_tokens", m["n_measured"], m["n_measured"] - m["n_over_think_cap"],
                                f"exact reasoning tokens <= {spec.think_cap}; measured on the seeded shuffle until the "
                                f"oversample targets were met ({m['n_candidates'] - m['n_measured']} candidates not measured)"))
            funnel.append(stage("oversample_draw", m["n_measured"] - m["n_over_think_cap"], n_drawn,
                                "rows beyond the per-stratum oversample targets are not drawn (not a quality drop)"))
            funnel.append(stage("d1_canary", n_drawn, n_drawn - n_d1))
            funnel.append(stage("d4_task_names", n_drawn - n_d1, n_drawn - n_d1 - n_d4))
            left = n_drawn - n_d1 - n_d4
            funnel.append(stage("prompt_dedup", left, left - n_dd))
            funnel.append(stage("d5_prime", left - n_dd, left - n_dd - n_d5))
            for g in gate_rep[p]["gates"]:
                if g["n_dropped"] or g["gate"] in ("think_length", "repetition", "constraints", "render", "row_length"):
                    funnel.append(stage(f"gate_{g['gate']}", g["n_in"], g["n_out"]))
            funnel.append(stage("sft_draw", len(kept[p]), len(sft_rows[p]), "survivors beyond the stratum quotas stay unused"))
            parts_rep[p] = {
                "source": {"hf_repo": spec.hf_repo, "split": spec.split, "revision": revisions[RAW_KEYS[p]],
                           "licence": spec.dataset_license, "generator": sorted({r["generator"] for r in sft_rows[p] + res_rows[p]}),
                           "label_provenance": spec.label_provenance, "check": spec.check},
                "targets": {"sft": n_sft[p], "reserve": n_res[p]}, "counts": {"sft": len(sft_rows[p]), "reserve": len(res_rows[p])},
                "funnel": funnel, "loader_side_counts": load_info[p]["side"], "measure": m,
                "gates": gate_rep[p], "draw": draw_rep[p],
                "tokens": {"sft": token_stats(sft_rows[p]), "reserve": token_stats(res_rows[p])}}
            if p == "math":
                for split, rows in (("sft", sft_rows[p]), ("reserve", res_rows[p])):
                    got = Counter(r["length_band"] for r in rows)
                    parts_rep[p][f"band_shares_{split}"] = {b: round(got[b] / max(1, len(rows)), 4) for b, _ in draw.MATH_BANDS}
                parts_rep[p]["band_shares_target"] = draw.MATH_BAND_SHARES
                parts_rep[p]["ref_match_sft"] = dict(Counter(r["ref_match"] for r in sft_rows[p]))
            if p == "chat":
                cc = measure.get("classifier_check") or {}
                for split, rows in (("sft", sft_rows[p]), ("reserve", res_rows[p])):
                    share = sum(r["topic_group"] == topics.HSS for r in rows) / max(1, len(rows))
                    parts_rep[p][f"topic_share_{split}"] = {
                        "classifier_hss_share": round(share, 4), "topics": dict(Counter(r["topic"] for r in rows).most_common()),
                        "estimated_true_hss_share": estimated_true_share(share, cc)}
                parts_rep[p]["classifier"] = {"version": topics.CLASSIFIER_VERSION, "model": "BAAI/bge-small-en-v1.5",
                                              "revision": bge_rev, "target_hss_share": draw.CHAT_HSS_SHARE,
                                              "floor": draw.CHAT_HSS_FLOOR, "hand_label_check": cc,
                                              "prototypes": {t: {"group": g, "descriptions": d} for t, (g, d) in topics.PROTOTYPES.items()}}
            if p == "if":
                parts_rep[p]["constraint_kinds_sft"] = dict(Counter(k for r in sft_rows[p] for k in json.loads(r["constraint_kinds"])).most_common())
                parts_rep[p]["extractor"] = {"version": constraints.EXTRACTOR_VERSION, "checker": checkers.CHECKER_VERSION}
        manifest = {
            "pool": "retention", "version": POOL_VERSION, "run_id": run_id, "created": utc_now(), "seed": a.seed,
            "small_run": bool(a.small), "code": git_info(), "plan": "docs/DATA_PLAN.md rev 4.3.6 sections 3.4, 6.3, 7, 11",
            "sources": {k: {kk: srcs[k].get(kk) for kk in ("repo", "revision", "purpose", "files")} for k in ("nm_math", "nm_ifchat", "nm_sci")},
            "source_table": [{"part": p, **parts_rep[p]["source"], "rows_sft": len(sft_rows[p]), "rows_reserve": len(res_rows[p])} for p in a.parts],
            "embedding_model": {"repo": "BAAI/bge-small-en-v1.5", "revision": bge_rev},
            "tokenizer": str(a.tokenizer),
            "counts": {"sft": {p: len(sft_rows[p]) for p in a.parts}, "reserve": {p: len(res_rows[p]) for p in a.parts},
                       "sft_total": len(sft_all), "reserve_total": len(res_all)},
            "tokens": {"sft": token_stats(sft_all), "reserve": token_stats(res_all)},
            "screens": screen_rep, "parts": parts_rep, "files": files,
            "system_messages": "not kept (see apex_data/retention/common.py)",
            "checks": checks,
        }
        write_json(out_dir / "manifest.json", manifest)
        failed = [c["check"] for c in checks if not c["pass"]]
        for c in checks:
            run.log(f"[check] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {json.dumps(c['detail'], default=str)[:300]}")
            run.metric({"step": "check", "check": c["check"], "pass": c["pass"]})
        if failed:
            run.error(f"M2.3 checks failed: {failed}")
            rc = 1
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if rc == 0 else "failed",
                           outputs={k: v for k, v in files.items()}, counts=manifest["counts"], checks_failed=failed)
        run.finalize()
        run.log(f"done: SFT {len(sft_all)} rows, reserve {len(res_all)} rows -> {out_dir}; checks failed: {failed or 'none'}")
    return rc


# ------------------------------------------------------------------------------------------------------ top-up
TOPUP_VERSION = "retention-topup-v1"
PREFIX_TOLERANCE = 2.0           # rows: worst |count - k x share| over every prefix k, per (part, stratum) cell and per part
PREFIX_CHECKPOINTS = (100, 350, 700, 1050, 1400, 1500)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_pool(pool_dir: Path) -> dict[str, Any]:
    """The existing pool (read only): its manifest, the sha256 of its files (the rows files checked against the
    manifest) and the identity and length columns of its SFT and reserve rows."""
    import pyarrow.parquet as pq

    manifest = json.loads((pool_dir / "manifest.json").read_text())
    files = {"rows": pool_dir / "rows.parquet", "replay_reserve": pool_dir / "replay_reserve.parquet",
             "manifest": pool_dir / "manifest.json"}
    sha = {k: sha256_file(f) for k, f in files.items()}
    bad = [k for k in ("rows", "replay_reserve") if sha[k] != manifest["files"][k]["sha256"]]
    if bad:
        raise ValueError(f"{pool_dir}: {bad} differ from the sha256 its manifest records")
    cols = ["task_id", "row_key", "prompt_sha256", "dup_group", "retention_part", "reasoning_tokens", "n_tokens",
            "n_assistant_tokens"]
    rows = {k: pq.read_table(files[k], columns=cols).to_pylist() for k in ("rows", "replay_reserve")}
    return {"manifest": manifest, "files": files, "sha256": sha, "rows": rows}


def walk_matches(drawn: list[dict[str, Any]], src: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the (continued) walk's normal draw equal the pool run's drawn rows: same ids in the same order, same
    measured reasoning tokens, length band, topic group and option count?"""
    keys = ("reasoning_tokens", "length_band", "topic_group", "n_options")
    same_ids = [c["task_id"] for c in drawn] == [c["task_id"] for c in src]
    same_meta = same_ids and all(c["meta"].get(k) == s["meta"].get(k) for c, s in zip(drawn, src) for k in keys)
    return {"n_walk": len(drawn), "n_pool_run": len(src), "same_ids_in_order": same_ids, "same_measured_meta": same_meta,
            "pass": same_ids and same_meta}


def bridged_survivors(clusters: dict[str, list[str]], taken: set[str], nxt: set[str]) -> set[str]:
    """Next survivors that a new row joins (in one dedup cluster) to an SFT or reserve row, or to a lower-id next
    survivor. The earlier rows win, so the top-up skips these."""
    out = set()
    for members in clusters.values():
        for m in members:
            if m in nxt and any(o in taken or (o in nxt and o < m) for o in members if o != m):
                out.add(m)
    return out


def length_summary(values: list[int]) -> dict[str, Any]:
    v = sorted(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": statistics.median(v), "p90": statistics.quantiles(v, n=10, method="inclusive")[8],
            "max": v[-1]}


def topup_token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    st = token_stats(rows)
    st["loss_tokens_from_masks"] = sum(sum(r["assistant_masks"]) for r in rows)
    st["mean_loss_tokens"] = round(st["loss_tokens"] / len(rows), 1) if rows else None
    st["reasoning_tokens"] = length_summary([r["reasoning_tokens"] for r in rows])
    return st


def prefix_summary(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Composition of the first k top-up rows: part counts and shares, math band shares, chat hss share, tokens."""
    head = rows[:k]
    parts = Counter(r["retention_part"] for r in head)
    math_rows = [r for r in head if r["retention_part"] == "math"]
    chat_rows = [r for r in head if r["retention_part"] == "chat"]
    bands = Counter(r["length_band"] for r in math_rows)
    return {"k": len(head), "parts": {p: parts[p] for p in PART_ORDER},
            "part_shares": {p: round(parts[p] / max(1, len(head)), 4) for p in PART_ORDER},
            "math_band_shares": {b: round(bands[b] / max(1, len(math_rows)), 4) for b, _ in draw.MATH_BANDS},
            "chat_hss_share": round(sum(r["topic_group"] == topics.HSS for r in chat_rows) / max(1, len(chat_rows)), 4),
            "sequence_tokens": sum(r["n_tokens"] for r in head), "loss_tokens": sum(r["n_assistant_tokens"] for r in head)}


def main_topup(a: argparse.Namespace) -> int:
    """Top-up mode (see the module docstring)."""
    import shutil

    import pyarrow.parquet as pq

    pool = read_pool(a.exclude_pool)
    pm = pool["manifest"]
    src_work = paths.WORK / pm["run_id"]
    a.candidates_from = a.candidates_from or pm["run_id"]
    run_id = a.run_id or new_run_id("data_retention_topup")
    work = paths.WORK / run_id
    out_dir = a.out_dir
    n_sft = {p: PARTS[p].n_sft for p in PART_ORDER}
    n_res = common.reserve_quota(a.reserve_total)
    if pm["counts"]["sft"] != n_sft or pm["counts"]["reserve"] != n_res or pm["seed"] != a.seed:
        raise ValueError(f"{a.exclude_pool} was not built with these SFT / reserve targets and seed {a.seed}")
    n_top = common.apportion(a.topup, n_sft)
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        srcs = raw_sources(a.raw_sources)
        revisions = {k: srcs[k]["revision"] for k in ("nm_math", "nm_ifchat", "nm_sci")}
        bge_rev = (srcs.get("bge_small") or {}).get("revision")
        created = utc_now()
        run.write_metadata(run_id=run_id, step="data_build_retention_slice --topup", command=command_line(), git=git_info(),
                           code_commit=git_info()["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={**revisions, "bge_small": bge_rev,
                                          "exclude_pool": {"run_id": pm["run_id"], **pool["sha256"]}},
                           model_revision=str(a.tokenizer.name), hardware=hardware_info(), seed=a.seed, start_time=created,
                           status="running", targets={"topup": n_top}, work_dir=str(work), source_work_dir=str(src_work))
        work.mkdir(parents=True, exist_ok=True)
        run.log(f"[topup] {a.topup} rows {n_top} after the pool {rel(a.exclude_pool)} (run {pm['run_id']}) -> {out_dir}")
        # 1 load: the pool run's candidates; its drawn rows carry its screen marks
        cands: dict[str, list[dict[str, Any]]] = {}
        load_info: dict[str, Any] = {}
        for p in PART_ORDER:
            cands[p], load_info[p] = load_part(p, a, revisions, run, work)
        src_drawn = read_jsonl(src_work / "drawn.jsonl")
        # 2 measure (the pool run's walk, continued) and 3 screens of the new rows
        new_path, screens_path, measure_path = work / "topup_drawn.jsonl", work / "topup_screens.json", work / "topup_measure.json"
        if a.resume and new_path.exists() and screens_path.exists() and measure_path.exists():
            new = read_jsonl(new_path)
            screen_rep = json.loads(screens_path.read_text())
            measure = json.loads(measure_path.read_text())
            run.log(f"[measure+screens] reusing {new_path}")
        else:
            from apex_data.screens import d5_prime
            classifier = topics.TopicClassifier(d5_prime.load_embedder())
            new, measure = [], {}
            counter = TokenCounter(a.tokenizer, a.workers)
            try:
                for p in PART_ORDER:
                    d, extra, measure[p] = measure_and_draw(p, cands[p], a, run, classifier, counter, n_sft[p], n_res[p], n_top[p])
                    measure[p]["reproduces_pool_run"] = walk_matches(d, [c for c in src_drawn if c["part"] == p])
                    new += extra
            finally:
                counter.close()
            bad = {p: measure[p]["reproduces_pool_run"] for p in PART_ORDER if not measure[p]["reproduces_pool_run"]["pass"]}
            if bad:
                raise ValueError(f"the measurement walk does not reproduce the pool run's drawn rows: {bad}")
            screen_rep = run_screens(new, run, a, prior=src_drawn)
            write_jsonl(new_path, new)
            write_json(screens_path, screen_rep)
            write_json(measure_path, measure)
        del cands
        cand_by_id = {c["task_id"]: c for c in src_drawn + new}
        # 4 gates: the pool run's screened rows (to redo its draw) and the new rows
        kept_old: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        kept_new: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        gate_rep: dict[str, Any] = {}
        gate_dropped: list[dict[str, Any]] = []
        for p in PART_ORDER:
            run.log(f"[gates] {p}: the pool run's screened rows, then the new rows")
            kept_old[p], _, _ = run_gates(p, [c for c in src_drawn if c["part"] == p and c["split"] == "train"], a, run, created)
            kept_new[p], dropped, gate_rep[p] = run_gates(p, [c for c in new if c["part"] == p and c["split"] == "train"],
                                                          a, run, created)
            gate_dropped += [{**d, "part": p} for d in dropped]
        excluded = [{"task_id": c["task_id"], "part": c["part"], "gate": "screens", "reason": c["screens"].get("exclusion_reason")}
                    for c in new if c["split"] == "excluded"]
        write_jsonl(work / "topup_dropped.jsonl", excluded + gate_dropped)

        # 5 draw: redo the SFT and reserve draw (must reproduce the pool's files), then the top-up
        def survivors(kept: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
            return [{**cand_by_id[prov["task_id"]], "_row": row, "_prov": prov} for row, prov in kept]

        old_surv = {p: survivors(kept_old[p]) for p in PART_ORDER}
        new_surv = {p: survivors(kept_new[p]) for p in PART_ORDER}
        sft_all: list[dict[str, Any]] = []
        res_all: list[dict[str, Any]] = []
        for p in PART_ORDER:
            s, r, _ = draw.draw_sft_and_reserve(p, old_surv[p], n_sft[p], n_res[p], a.seed)
            sft_all += [row_record(x["_row"], x["_prov"], x, "train") for x in s]
            res_all += [row_record(x["_row"], x["_prov"], x, "replay_reserve") for x in r]
        repro: dict[str, Any] = {}
        for k, rows in (("rows", sft_all), ("replay_reserve", res_all)):
            path = work / "reproduced" / pool["files"][k].name
            info = write_rows(path, rows)
            same_bytes = info["sha256"] == pool["sha256"][k]
            repro[k] = {"path": info["path"], "rows": info["rows"], "sha256": info["sha256"], "pool_sha256": pool["sha256"][k],
                        "identical_bytes": same_bytes,
                        "identical_content": same_bytes or pq.read_table(path).equals(pq.read_table(pool["files"][k]))}
        run.log(f"[reproduce] {json.dumps({k: {kk: v[kk] for kk in ('identical_bytes', 'identical_content')} for k, v in repro.items()})}")
        taken = {r["task_id"] for r in sft_all + res_all}
        nxt = {x["task_id"] for p in PART_ORDER for x in old_surv[p]} - taken
        bridged = bridged_survivors(screen_rep["dedup"]["clusters_with_new_rows"], taken, nxt)
        top: dict[str, list[dict[str, Any]]] = {}
        draw_rep: dict[str, Any] = {}
        for p in PART_ORDER:
            top[p], draw_rep[p] = draw.draw_topup(p, old_surv[p], n_sft[p], n_res[p], new_surv[p], n_top[p], a.seed, exclude=bridged)
            run.log(f"[topup] {p}: {len(top[p])} rows {draw_rep[p]['strata']} (next survivors {draw_rep[p]['from_next_survivors']}, "
                    f"new {draw_rep[p]['from_new']})")
        # 6 order across (part, stratum) cells and write
        cells: dict[str, list[dict[str, Any]]] = {}
        for p in PART_ORDER:
            for x in top[p]:
                cells.setdefault(f"{p}/{draw.stratum_of(p, x)}", []).append(x)
        ordered = draw.interleave(cells, a.seed)
        topup_rows = [{**row_record(x["_row"], x["_prov"], x, "train"), "topup_rank": k} for k, (_, x) in enumerate(ordered)]
        cell_seq = [cell for cell, _ in ordered]
        files = {"topup_rows": write_rows(out_dir / "topup_rows.parquet", topup_rows)}
        from apex_data.render.select import first_user_message
        from apex_train.data_format import load_tokenizer
        tok = load_tokenizer(a.tokenizer)
        items_path = work / "topup_screen_items.jsonl"
        write_jsonl(items_path, [{"task_id": r["task_id"], "kind": "prompt", "text": first_user_message(tok.decode(r["input_ids"])),
                                  "upstream_id": r["task_id"].split(":", 1)[1], "split": "train",
                                  "stratum": {"part": r["retention_part"]}} for r in topup_rows])
        screens_dir = out_dir / "screens"           # run_screens reads dev_*.json from --manifests-dir and writes there
        screens_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(paths.MANIFESTS.glob("dev_*.json")):
            shutil.copy2(f, screens_dir / f.name)
        p1_path = paths.MANIFESTS / "screens_p1_pools.json"
        p1_inputs = {}
        if p1_path.exists():
            p1_inputs = {n: inp["files"][0]["path"] for n, inp in json.loads(p1_path.read_text())["inputs"].items()
                         if n in ("retention", "dev") and inp.get("files")}
        env = f"OMP_NUM_THREADS={a.workers} MKL_NUM_THREADS={a.workers} TOKENIZERS_PARALLELISM=false PYTHONPATH=src"
        base = (f"{env} .venv/train/bin/python scripts/data/run_screens.py --pools '' --no-github --workers {a.workers} "
                f"--manifests-dir {screens_dir} --report {screens_dir / 'decontam_report.json'}")
        commands = {"p1_topup": f"{base} --pass-name p1_topup --pool retention_topup={items_path} "
                                + " ".join(f"--pool {n}={pth}" for n, pth in p1_inputs.items()),
                    "p2_topup": f"{base} --pass-name p2_topup --rows {out_dir / 'topup_rows.parquet'}"}
        for name, cmd in commands.items():
            run.log(f"next ({name}): {cmd}")
        # checks
        by_part = {p: [r for r in topup_rows if r["retention_part"] == p] for p in PART_ORDER}
        checks = [{**c, "check": c["check"].replace("_sft", "_topup").replace("row_counts", "topup_row_counts")}
                  for c in m23_checks(by_part, {p: [] for p in PART_ORDER}, cand_by_id, n_top, {p: 0 for p in PART_ORDER}, tok, False)
                  if c["check"] not in ("reserve_counts", "reserve_disjoint")]

        def add(name: str, ok: bool, detail: Any) -> None:
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        def key_of(r: dict[str, Any], k: str) -> str:
            return (r[k] or r["task_id"]) if k == "dup_group" else r[k]

        pool_rows = pool["rows"]["rows"] + pool["rows"]["replay_reserve"]
        keys = ("task_id", "row_key", "prompt_sha256", "dup_group")
        ids = {k: [key_of(r, k) for r in topup_rows] for k in keys}
        pool_ids = {k: {key_of(r, k) for r in pool_rows} for k in keys}
        top_ids, pool_tids = set(ids["task_id"]), pool_ids["task_id"]
        joined = [g for g, m in screen_rep["dedup"]["clusters_with_new_rows"].items()
                  if sum(x in top_ids for x in m) > 1 or (any(x in top_ids for x in m) and any(x in pool_tids for x in m))]
        add("disjoint_from_pool_rows", not joined and not any(set(v) & pool_ids[k] for k, v in ids.items()),
            {**{f"shared_{k}": len(set(v) & pool_ids[k]) for k, v in ids.items()}, "dedup_clusters_joining_topup_and_pool_rows": len(joined),
             "pool_rows": len(pool_rows), "bridged_next_survivors_skipped": len(bridged)})
        add("topup_rows_unique", all(len(set(v)) == len(v) for v in ids.values()), {k: len(v) - len(set(v)) for k, v in ids.items()})
        add("stratum_quotas_met", all(draw_rep[p]["strata"] == {s: v for s, v in draw_rep[p]["quota"].items() if v} for p in PART_ORDER),
            {p: {"quota": draw_rep[p]["quota"], "got": draw_rep[p]["strata"]} for p in PART_ORDER})
        mask_ok = all(sum(r["assistant_masks"]) == r["n_assistant_tokens"] for r in topup_rows)
        add("loss_tokens_equal_mask_sums", mask_ok, {})
        dev_cell = draw.prefix_deviation(cell_seq)
        dev_part = draw.prefix_deviation([c.split("/")[0] for c in cell_seq])
        add("prefix_proportional", max(dev_cell.values()) <= PREFIX_TOLERANCE and max(dev_part.values()) <= PREFIX_TOLERANCE,
            {"tolerance_rows": PREFIX_TOLERANCE, "max_deviation_by_cell": dev_cell, "max_deviation_by_part": dev_part})
        from apex_data.screens import d1_canary
        d1_final = d1_canary.scan_records([{"task_id": r["task_id"], "messages": cand_by_id[r["task_id"]]["messages"]} for r in topup_rows],
                                          source="retention_topup_selected").counts()
        add("d1_canary_selected_rows", not d1_final.get("n_hits"), d1_final)
        add("d5_prime_screen_applied", screen_rep["d5_prime"]["n_dev_items"] > 0,
            {k: screen_rep["d5_prime"][k] for k in ("n_dev_items", "n_hits", "n_dropped_from_train")})
        add("measurement_walk_reproduces_pool_run", all(measure[p]["reproduces_pool_run"]["pass"] for p in PART_ORDER),
            {p: measure[p]["reproduces_pool_run"] for p in PART_ORDER})
        add("pool_selection_reproduced", all(v["identical_content"] for v in repro.values()),
            {k: {kk: v[kk] for kk in ("sha256", "pool_sha256", "identical_bytes", "identical_content")} for k, v in repro.items()})
        after = {k: sha256_file(f) for k, f in pool["files"].items()}
        add("pool_files_unchanged", after == pool["sha256"], {"before": pool["sha256"], "after": after})
        # manifest
        parts_rep: dict[str, Any] = {}
        for p in PART_ORDER:
            dr = [c for c in new if c["part"] == p]
            reasons = Counter(str(c["screens"].get("exclusion_reason") or "").split(":")[0] for c in dr if c["split"] == "excluded")
            parts_rep[p] = {
                "target": n_top[p], "count": len(top[p]), "draw": draw_rep[p], "measure": measure[p],
                "new_rows_funnel": [{"stage": "drawn_beyond_pool_run", "n_out": len(dr)},
                                    {"stage": "screens", "n_in": len(dr), "n_dropped": sum(reasons.values()),
                                     "by_reason": dict(sorted(reasons.items())), "n_out": len(dr) - sum(reasons.values())},
                                    {"stage": "gates", "n_in": len(dr) - sum(reasons.values()), "n_out": len(kept_new[p]),
                                     "by_reason": gate_rep[p]["reasons"]},
                                    {"stage": "topup_draw", "n_in": len(kept_new[p]), "n_out": sum(draw_rep[p]["from_new"].values())}],
                "gates_new_rows": gate_rep[p], "tokens": topup_token_stats(by_part[p]),
                "reasoning_tokens_pool_sft": length_summary([r["reasoning_tokens"] for r in pool["rows"]["rows"] if r["retention_part"] == p])}
        manifest = {
            "pool": "retention", "kind": "topup", "version": TOPUP_VERSION, "run_id": run_id, "created": utc_now(), "seed": a.seed,
            "code": git_info(), "plan": "docs/DATA_PLAN.md rev 4.3.6 sections 3.4, 6.3, 7, 11 (top-up of the retention SFT rows)",
            "n_topup": a.topup, "targets": n_top,
            "exclude_pool": {"dir": rel(a.exclude_pool), "run_id": pm["run_id"], "version": pm.get("version"), "sha256": pool["sha256"]},
            "candidates_from": a.candidates_from, "source_work_dir": str(src_work), "tokenizer": str(a.tokenizer),
            "embedding_model": {"repo": "BAAI/bge-small-en-v1.5", "revision": bge_rev},
            "order": ("rows sorted by topup_rank: draw.interleave over the (part, stratum) cells (Webster divisor order, seeded "
                      "tie-break); within a cell the draw order (next survivors of the SFT / reserve shuffle, then new rows in "
                      "their own seeded shuffle). Take the first K rows (topup_rank < K) for a proportional subset."),
            "counts": {"topup": {p: len(top[p]) for p in PART_ORDER}, "topup_total": len(topup_rows),
                       "strata": {p: draw_rep[p]["strata"] for p in PART_ORDER},
                       "from_next_survivors": {p: draw_rep[p]["from_next_survivors"] for p in PART_ORDER},
                       "from_new": {p: draw_rep[p]["from_new"] for p in PART_ORDER}},
            "tokens": topup_token_stats(topup_rows),
            "prefix": {"max_deviation_by_cell": dev_cell, "max_deviation_by_part": dev_part,
                       "checkpoints": [prefix_summary(topup_rows, k) for k in (*PREFIX_CHECKPOINTS, len(topup_rows)) if k <= len(topup_rows)],
                       "cumulative_loss_tokens_every_100": [sum(r["n_assistant_tokens"] for r in topup_rows[:k])
                                                            for k in range(100, len(topup_rows) + 1, 100)]},
            "screens": screen_rep, "parts": parts_rep, "pool_selection_reproduced": repro, "files": files,
            "screen_commands": commands, "screen_items": str(items_path), "checks": checks,
        }
        write_json(out_dir / "topup_manifest.json", manifest)
        failed = [c["check"] for c in checks if not c["pass"]]
        for c in checks:
            run.log(f"[check] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {json.dumps(c['detail'], default=str)[:300]}")
            run.metric({"step": "check", "check": c["check"], "pass": c["pass"]})
        if failed:
            run.error(f"top-up checks failed: {failed}")
            rc = 1
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if rc == 0 else "failed",
                           outputs=files, counts=manifest["counts"], checks_failed=failed)
        run.finalize()
        run.log(f"done: top-up {len(topup_rows)} rows -> {out_dir}; checks failed: {failed or 'none'}")
    return rc


# ------------------------------------------------------------------------------------------------------ convergence
CVG_OVERSAMPLE = 2.0             # the walk stops when each part holds 2 x its loss-token target (screens and gates drop)
CVG_LOSS_OVERHEAD = 6            # estimated loss tokens beyond reasoning + answer during the walk (</think>, <|im_end|>)
CVG_PREFIX_TOLERANCE = 2.0       # rows: worst |count - k x share| over every prefix, per cell and per replay part
CVG_PRIOR_COLUMNS = ["task_id", "row_key", "prompt_sha256", "dup_group", "retention_part", "reasoning_tokens", "n_tokens",
                     "n_assistant_tokens"]


CVG_TEXT: dict[str, dict[str, str]] = {      # manifest text per convergence profile
    "stage2_replay": {
        "plan": "docs/DATA_PLAN.md rev 4.3.6 section 11 (stage-2 retention replay), decision 2026-10-12: the math and "
                "science share of the replay becomes the convergence parts; the reserve's IF and chat rows stay",
        "rationale": "our dev set (results/dev/base/dev_base_full_20261012_1010): Base's AIME failures are 29 of 30 budget "
                     "exhaustion (no answer within 32K) and its STEM QA failures 89 of 125; the replay teaches committing "
                     "to one answer in a short final section after long reasoning",
        "order": "rows sorted by replay_rank: draw.interleave over the cells if/all, chat/hss, chat/other (reserve file "
                 "order) and math_hard, math_medium, sci_mcq (walk order); the first K rows keep every cell's row share "
                 "up to rounding"},
    "hard_math_v5": {
        "plan": "hard math in the revision-5 retention slice (data/manifests/v5/DATA_CARD_v5.md, retention slice)",
        "rationale": "harder math than anything in the pool (reasoning above its 10,240-token cap) that still asks for no "
                     "more thinking than Base spends on solved dev QA items (median output 12.1K): long reasoning, then one "
                     "committed answer in a short final section",
        "order": "rows sorted by replay_rank: the survivors in walk order (one cell, math_hard)"},
}


def _stage_rows(name: str, n_in: int, n_out: int, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "n_in": n_in, "n_dropped": n_in - n_out, "n_out": n_out, **extra}


def convergence_walk(source_part: str, cands: list[dict[str, Any]], targets: dict[str, int], a: argparse.Namespace,
                     run: RunDir, counter: TokenCounter, prof: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seeded shuffle of the pre-filtered candidates; exact reasoning and answer tokens in batches; a row joins the replay
    part whose reasoning band it fits (convergence.walk_part) while that part's estimated loss tokens are below
    CVG_OVERSAMPLE x its target. Returns the drawn rows (walk order) and the walk report."""
    from apex_data.retention import convergence as cvg

    order = draw.shuffled(cands, a.seed, "convergence", source_part)
    parts = [p for p in prof.order if prof.parts[p].retention_part == source_part]
    need = {p: CVG_OVERSAMPLE * targets[p] for p in parts}
    drawn: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    measured = 0
    i = 0
    while i < len(order) and any(v > 0 for v in need.values()):
        chunk = order[i:i + 2000]
        i += len(chunk)
        rt = counter([c["messages"][1]["reasoning_content"] for c in chunk])
        ft = counter([c["messages"][1]["content"] for c in chunk])
        for c, r, f in zip(chunk, rt, ft, strict=True):
            measured += 1
            c["meta"].update(reasoning_tokens=r, final_section_tokens=f, walk_index=measured - 1)
            p, why = cvg.walk_part(source_part, r, f, prof)
            if p is None:
                reasons[why] += 1
                continue
            if need[p] <= 0:
                reasons[f"beyond_target:{p}"] += 1
                continue
            c["meta"]["replay_part"] = p
            spec = prof.parts[p]
            c["meta"]["length_band"] = f"{spec.floor // 1024}k_{spec.cap // 1024}k" if spec.retention_part == "math" else None
            drawn.append(c)
            need[p] -= r + f + CVG_LOSS_OVERHEAD
    rep = {"n_candidates": len(cands), "n_measured": measured, "dropped_or_unused": dict(sorted(reasons.items())),
           "oversample": CVG_OVERSAMPLE, "estimated_loss_tokens_unfilled": {p: round(v) for p, v in need.items() if v > 0},
           "n_drawn": dict(Counter(c["meta"]["replay_part"] for c in drawn))}
    run.log(f"[walk] {source_part}: measured {measured} of {len(cands)}, drawn {rep['n_drawn']}, {rep['dropped_or_unused']}")
    return drawn, rep


def read_sealed_drops(paths_: list[Path], ids: set[str]) -> dict[str, list[str]]:
    """{task_id: rules} from run_screens manifests' `dropped` maps (keys: task id or row:<task id>), restricted to ids."""
    out: dict[str, set[str]] = {}
    for f in paths_:
        for k, rules in json.loads(f.read_text()).get("dropped", {}).items():
            tid = k.split(":", 1)[1] if k.startswith("row:") else k
            if tid in ids:
                out.setdefault(tid, set()).update(f"{f.stem.removeprefix('screens_')}:{r}" for r in rules)
    return {k: sorted(v) for k, v in sorted(out.items())}


def main_convergence(a: argparse.Namespace) -> int:
    """Convergence mode (see the module docstring)."""
    import shutil
    import subprocess

    import pyarrow.parquet as pq

    from apex_data.render.select import first_user_message
    from apex_data.retention import convergence as cvg
    from apex_data.screens import d1_canary
    from apex_train.data_format import load_tokenizer

    pool = read_pool(a.exclude_pool)
    pm = pool["manifest"]
    tm = (pm.get("topup_merge") or {}).get("topup") or {}
    topup_path = a.topup_rows or (Path(tm["dir"]) / "topup_rows.parquet" if tm.get("dir") else None)
    if topup_path is None or not topup_path.exists():
        raise FileNotFoundError(f"top-up rows not found ({topup_path}); pass --topup-rows")
    topup_sha = sha256_file(topup_path)
    if tm.get("rows_sha256") and topup_sha != tm["rows_sha256"]:
        raise ValueError(f"{topup_path}: sha256 {topup_sha} differs from the pool manifest's topup_merge record")
    prof = cvg.PROFILES[a.convergence_profile]
    targets = cvg.token_targets(a.convergence_replay, {p: prof.parts[p].share for p in prof.order})
    sources = prof.source_parts()
    math_cap = max(prof.parts[p].cap for p in prof.order if prof.parts[p].retention_part == "math")
    run_id = a.run_id or new_run_id(prof.run_prefix)
    work = paths.WORK / run_id
    out_dir = a.out_dir
    out_path, man_path = out_dir / f"{prof.stem}.parquet", out_dir / f"{prof.stem}_manifest.json"
    protected = {**pool["files"], "topup_rows": topup_path}
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        srcs = raw_sources(a.raw_sources)
        revisions = {k: srcs[k]["revision"] for k in ("nm_math", "nm_sci") if k == "nm_math" or "sci_mcq" in sources}
        bge_rev = (srcs.get("bge_small") or {}).get("revision")
        created = utc_now()
        run.write_metadata(run_id=run_id, step="data_build_retention_slice --convergence-replay", command=command_line(),
                           git=git_info(), code_commit=git_info()["commit"], config={k: str(v) for k, v in vars(a).items()},
                           data_versions={**revisions, "bge_small": bge_rev,
                                          "exclude_pool": {"run_id": pm["run_id"], **pool["sha256"]},
                                          "topup_rows": {"path": str(topup_path), "sha256": topup_sha}},
                           model_revision=str(a.tokenizer.name), hardware=hardware_info(), seed=a.seed, start_time=created,
                           status="running", targets=targets, work_dir=str(work))
        work.mkdir(parents=True, exist_ok=True)
        run.log(f"[convergence] profile {prof.name}: {a.convergence_replay} loss tokens {targets} + the reserve's "
                f"{list(prof.reserve_parts) or 'no'} rows -> {out_path}")
        # 0 the rows to stay disjoint from (SFT rows incl. the merged top-up, the reserve, the whole top-up file)
        topup_rows = pq.read_table(topup_path, columns=CVG_PRIOR_COLUMNS).to_pylist()
        prior_rows = pool["rows"]["rows"] + pool["rows"]["replay_reserve"] + topup_rows
        prior_ids = {r["task_id"] for r in prior_rows}
        prior_cands: dict[str, dict[str, Any]] = {}
        drawn_files = [paths.WORK / pm["run_id"] / "drawn.jsonl"]
        if tm.get("run_id"):
            drawn_files.append(paths.WORK / tm["run_id"] / "topup_drawn.jsonl")
        for f in drawn_files:
            for c in read_jsonl(f):
                if c["task_id"] in prior_ids:
                    prior_cands[c["task_id"]] = {"task_id": c["task_id"], "prompt": c["prompt"], "part": c["part"]}
        if prior_ids - prior_cands.keys():
            raise ValueError(f"{len(prior_ids - prior_cands.keys())} prior rows have no candidate record (prompt)")
        run.log(f"[prior] {len(prior_rows)} rows ({len(prior_ids)} task ids): SFT {len(pool['rows']['rows'])}, "
                f"reserve {len(pool['rows']['replay_reserve'])}, top-up file {len(topup_rows)}")
        # 1 load: math from RAW at the profile's math think cap; MCQ = the pool run's loader output (same pinned file and
        # checks)
        load_info: dict[str, Any] = {}
        mpath, mfpath = work / f"candidates_math_{math_cap // 1024}k.jsonl", work / f"funnel_math_{math_cap // 1024}k.json"
        if a.resume and mpath.exists() and mfpath.exists():
            math_cands, load_info["math"] = read_jsonl(mpath), json.loads(mfpath.read_text())
            run.log(f"[load] math: reusing {mpath}")
        else:
            run.log(f"[load] math: reading {a.raw_root / 'nm_math'} @ {revisions['nm_math'][:8]} at the "
                    f"{math_cap}-token think cap")
            math_cands, funnel = math_part.load(a.raw_root / "nm_math", revisions["nm_math"], a.workers,
                                                think_cap=math_cap)
            load_info["math"] = {"funnel": funnel.table(), "n_candidates": len(math_cands),
                                 "side": {"ref_match": dict(Counter(c["meta"]["ref_match"] for c in math_cands))}}
            write_jsonl(mpath, math_cands)
            write_json(mfpath, load_info["math"])
        src_mcq = paths.WORK / pm["run_id"]
        if "sci_mcq" in sources:
            load_info["sci_mcq"] = json.loads((src_mcq / "funnel_sci_mcq.json").read_text())
            load_info["sci_mcq"]["from_run"] = pm["run_id"]
        # 1b pre-filters (counted), streaming the MCQ candidates
        pre: dict[str, list[dict[str, Any]]] = {sp: [] for sp in sources}
        stages = {"math": cvg.MATH_PREFILTER_STAGES, "sci_mcq": cvg.MCQ_PREFILTER_STAGES}
        pre_funnel = {sp: common.Funnel(stages[sp]) for sp in sources}
        for c in math_cands:
            if pre_funnel["math"].run(cvg.prefilter_checks("math", c, prior_ids, prof.final_cap)) is None:
                pre["math"].append(c)
        del math_cands
        if "sci_mcq" in sources:
            n_mcq = 0
            with (src_mcq / "candidates_sci_mcq.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n_mcq += 1
                        c = json.loads(line)
                        if pre_funnel["sci_mcq"].run(cvg.prefilter_checks("sci_mcq", c, prior_ids, prof.final_cap)) is None:
                            pre["sci_mcq"].append(c)
            if n_mcq != load_info["sci_mcq"]["n_candidates"]:
                raise ValueError(f"MCQ candidates: read {n_mcq}, the pool run's funnel says "
                                 f"{load_info['sci_mcq']['n_candidates']}")
        for sp in pre:
            run.log(f"[prefilter] {sp}: {[(r['stage'], r['n_out']) for r in pre_funnel[sp].table()]}")
        # 2 walk, 3 screens (D1, D4, prompt dedup with the prior rows winning, D5')
        drawn_path, screens_path, walk_path = work / "drawn.jsonl", work / "screens.json", work / "walk.json"
        if a.resume and drawn_path.exists() and screens_path.exists() and walk_path.exists():
            drawn = read_jsonl(drawn_path)
            screen_rep = json.loads(screens_path.read_text())
            walk = json.loads(walk_path.read_text())
            run.log(f"[walk+screens] reusing {drawn_path}")
        else:
            counter = TokenCounter(a.tokenizer, a.workers)
            drawn, walk = [], {}
            try:
                for sp in sources:
                    d, walk[sp] = convergence_walk(sp, pre[sp], targets, a, run, counter, prof)
                    drawn += d
            finally:
                counter.close()
            screen_rep = run_screens(drawn, run, a, prior=list(prior_cands.values()))
            write_jsonl(drawn_path, drawn)
            write_json(screens_path, screen_rep)
            write_json(walk_path, walk)
        del pre
        cand_by_id = {c["task_id"]: c for c in drawn}
        # 4 gates per replay part (row cap 64K, the part's reasoning cap)
        surv: dict[str, list[dict[str, Any]]] = {}
        gate_rep: dict[str, Any] = {}
        gate_dropped: list[dict[str, Any]] = []
        for p in prof.order:
            spec = prof.parts[p]
            live = [c for c in drawn if c["meta"]["replay_part"] == p and c["split"] == "train"]
            kept, dropped, gate_rep[p] = run_gates(spec.retention_part, live, a, run, created, think_cap=spec.cap,
                                                   row_cap=cvg.ROW_CAP)
            gate_dropped += [{**d, "part": p} for d in dropped]
            rows = []
            for row, prov in kept:
                c = cand_by_id[prov["task_id"]]
                rows.append({**row_record(row, prov, c, prof.stem), "replay_part": p,
                             "final_section_tokens": c["meta"]["final_section_tokens"]})
            surv[p] = sorted(rows, key=lambda r: cand_by_id[r["task_id"]]["meta"]["walk_index"])
        excluded = [{"task_id": c["task_id"], "part": c["meta"]["replay_part"], "gate": "screens",
                     "reason": c["screens"].get("exclusion_reason")} for c in drawn if c["split"] == "excluded"]
        write_jsonl(work / "dropped.jsonl", excluded + gate_dropped)
        # 5 sealed screens (scripts/data/run_screens.py, p1 on the prompts and p2 on the rendered rows) on every survivor
        tok = load_tokenizer(a.tokenizer)
        all_surv = [r for p in prof.order for r in surv[p]]
        cand_rows_path = work / "convergence_candidates.parquet"
        write_rows(cand_rows_path, all_surv)
        items_path = work / "convergence_screen_items.jsonl"
        write_jsonl(items_path, [{"task_id": r["task_id"], "kind": "prompt", "text": first_user_message(tok.decode(r["input_ids"])),
                                  "upstream_id": r["task_id"].split(":", 1)[1], "split": "train",
                                  "stratum": {"part": r["retention_part"]}} for r in all_surv])
        screens_dir = work / "screens"
        screens_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(paths.MANIFESTS.glob("dev_*.json")):
            shutil.copy2(f, screens_dir / f.name)
        p1_path = paths.MANIFESTS / "screens_p1_pools.json"
        p1_inputs = {}
        if p1_path.exists():
            p1_inputs = {n: inp["files"][0]["path"] for n, inp in json.loads(p1_path.read_text())["inputs"].items()
                         if n in ("retention", "dev") and inp.get("files")}
        env = {**os.environ, "OMP_NUM_THREADS": str(a.workers), "MKL_NUM_THREADS": str(a.workers),
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(ROOT / "src")}
        base = [sys.executable, str(ROOT / "scripts/data/run_screens.py"), "--pools", "", "--no-github", "--workers",
                str(a.workers), "--manifests-dir", str(screens_dir), "--report", str(screens_dir / "decontam_report.json")]
        commands = {"p1_convergence": base + ["--pass-name", "p1_convergence", "--pool", f"retention_convergence={items_path}",
                                              *[x for n, pth in p1_inputs.items() for x in ("--pool", f"{n}={pth}")]],
                    "p2_convergence": base + ["--pass-name", "p2_convergence", "--rows", str(cand_rows_path)]}
        sealed_files = []
        for name, cmd in commands.items():
            out_json = screens_dir / f"screens_{name}.json"
            if not (a.resume and out_json.exists()):
                run.log(f"[sealed] {name}: {' '.join(cmd)}")
                with (screens_dir / f"{name}.log").open("w") as lf:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
                if proc.returncode != 0 or not out_json.exists():
                    raise RuntimeError(f"run_screens {name} failed (exit {proc.returncode}); see {screens_dir / f'{name}.log'}")
            sealed_files.append(out_json)
        sealed = read_sealed_drops(sealed_files, {r["task_id"] for r in all_surv})
        sealed_counts = {"n_screened": len(all_surv), "n_dropped": len(sealed),
                         "by_rule": dict(Counter(x for v in sealed.values() for x in v)),
                         "by_part": dict(Counter(next(r["replay_part"] for r in all_surv if r["task_id"] == k) for k in sealed))}
        run.log(f"[sealed] {sealed_counts}")
        # 6 select per part (walk order, token target), combine with the reserve's IF and chat rows, interleave, write
        sel = {p: cvg.take_until(surv[p], targets[p], skip=sealed) for p in prof.order}
        reserve_all = pq.read_table(pool["files"]["replay_reserve"]).to_pylist()
        reserve_keep = [{**r, "replay_part": r["retention_part"], "final_section_tokens": None}
                        for r in reserve_all if r["retention_part"] in prof.reserve_parts]
        cells: dict[str, list[dict[str, Any]]] = {}
        for r in reserve_keep:
            cells.setdefault(f"{r['retention_part']}/{r['topic_group'] or 'all'}", []).append(r)
        for p in prof.order:
            cells[p] = sel[p]
        keysets = {frozenset(r) for v in cells.values() for r in v}
        if len(keysets) != 1:
            raise ValueError("reserve rows and convergence rows do not share one column set")
        ordered = draw.interleave(cells, a.seed, tag=f"{prof.stem}_interleave")
        out_rows = [{**x, "replay_rank": k} for k, (_, x) in enumerate(ordered)]
        cell_seq = [cell for cell, _ in ordered]
        files = {prof.stem: write_rows(out_path, out_rows)}
        run.log(f"[write] {len(out_rows)} rows -> {out_path} ({files[prof.stem]['sha256'][:12]})")
        # checks
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: Any) -> None:
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        cvg_rows = [r for r in out_rows if r["replay_part"] in prof.parts]
        res_rows = [r for r in out_rows if r["replay_part"] in prof.reserve_parts]
        loss = {p: sum(r["n_assistant_tokens"] for r in sel[p]) for p in prof.order}
        add("token_targets_met", all(loss[p] >= targets[p] for p in prof.order),
            {p: {"target": targets[p], "loss_tokens": loss[p], "rows": len(sel[p])} for p in prof.order})
        band_bad = [r["task_id"] for r in cvg_rows if not cvg.in_band(r["replay_part"], r["reasoning_tokens"], prof.parts)]
        add("reasoning_bands", not band_bad, {"bands": {p: [prof.parts[p].floor, prof.parts[p].cap] for p in prof.order},
                                              "violations": band_bad[:5]})
        fs_bad = [r["task_id"] for r in cvg_rows if not cvg.final_section_ok(r["final_section_tokens"], prof.final_cap)]
        add("final_section_cap", not fs_bad, {"cap": prof.final_cap, "violations": fs_bad[:5]})
        ans_bad = []
        for r in cvg_rows:
            c = cand_by_id[r["task_id"]]
            ans = c["messages"][1]["content"]
            if r["retention_part"] == "math":
                com = cvg.math_commitment(ans)
                ok = r["ref_match"] in cvg.REF_MATCHES and com["single"] and com["ends"]
            else:
                info = science.mcq_letters(c["prompt"], ans, c["messages"][1]["reasoning_content"])
                ok = (len(info["answer_letters"]) == 1 and info["letter"] in info["options"]
                      and info["letter"] == info["reasoning_letter"] and cvg.mcq_ending(ans, info["letter"]) is not None)
            if not ok:
                ans_bad.append(r["task_id"])
        add("committed_answer_checks", not ans_bad, {"violations": ans_bad[:5]})
        add("row_cap_64k", all(r["n_tokens"] <= cvg.ROW_CAP for r in out_rows), {"max": max(r["n_tokens"] for r in out_rows)})
        disj = cvg.disjointness(cvg_rows, prior_rows, clusters=screen_rep["dedup"]["clusters_with_new_rows"])
        add("disjoint_from_pool_reserve_topup", disj["pass"], disj)
        add("sealed_screen_drops_applied", not ({r["task_id"] for r in cvg_rows} & set(sealed)), sealed_counts)
        d1_final = d1_canary.scan_records([{"task_id": r["task_id"], "messages": cand_by_id[r["task_id"]]["messages"]}
                                           for r in cvg_rows], source="convergence_replay_selected").counts()
        add("d1_canary_selected_rows", not d1_final.get("n_hits"), d1_final)
        add("d5_prime_screen_applied", screen_rep["d5_prime"]["n_dev_items"] > 0,
            {k: screen_rep["d5_prime"][k] for k in ("n_dev_items", "n_hits", "n_dropped_from_train")})
        by_id = {r["task_id"]: r for r in reserve_all}
        if prof.reserve_parts:
            written = [r for r in pq.read_table(out_path).to_pylist() if r["replay_part"] in prof.reserve_parts]
            verbatim = all({k: v for k, v in r.items() if k not in ("replay_part", "final_section_tokens", "replay_rank")}
                           == by_id[r["task_id"]] for r in written)
            add("reserve_if_chat_rows_verbatim", verbatim and len(written) == len(res_rows) == len(reserve_keep),
                {"rows": len(res_rows), "by_part": dict(Counter(r["retention_part"] for r in res_rows))})
        dev_cell = draw.prefix_deviation(cell_seq)
        part_seq = [c.split("/")[0] for c in cell_seq]
        dev_part = draw.prefix_deviation(part_seq)
        add("prefix_proportional", max(dev_cell.values()) <= CVG_PREFIX_TOLERANCE and max(dev_part.values()) <= CVG_PREFIX_TOLERANCE,
            {"tolerance_rows": CVG_PREFIX_TOLERANCE, "max_deviation_by_cell": dev_cell, "max_deviation_by_part": dev_part})
        add("loss_tokens_equal_mask_sums", all(sum(r["assistant_masks"]) == r["n_assistant_tokens"] for r in out_rows), {})
        sample = out_rows[:: max(1, len(out_rows) // 100)]
        bad_think = []
        for r in sample:
            trained = tok.decode([t for t, m in zip(r["input_ids"], r["assistant_masks"], strict=True) if m])
            if "</think>" not in trained or not trained.endswith("<|im_end|>") or not trained.split("</think>")[0].strip():
                bad_think.append(r["task_id"])
        add("trained_span_has_reasoning", not bad_think, {"sampled": len(sample), "violations": bad_think[:5]})
        add("provenance_valid", all(_valid(r) for r in out_rows), {})
        after = {k: sha256_file(f) for k, f in protected.items()}
        add("pool_files_unchanged", after == {**pool["sha256"], "topup_rows": topup_sha},
            {"before": {**pool["sha256"], "topup_rows": topup_sha}, "after": after})
        # manifest
        parts_rep: dict[str, Any] = {}
        for p in prof.order:
            spec = prof.parts[p]
            dr = [c for c in drawn if c["meta"]["replay_part"] == p]
            reasons = Counter(str(c["screens"].get("exclusion_reason") or "").split(":")[0] for c in dr if c["split"] == "excluded")
            n_scr = len(dr) - sum(reasons.values())
            n_sealed = sum(r["task_id"] in sealed for r in surv[p])
            parts_rep[p] = {
                "retention_part": spec.retention_part, "reasoning_band_tokens": [spec.floor, spec.cap],
                "loss_token_share_target": spec.share, "loss_token_target": targets[p],
                "funnel": [_stage_rows("walk_drawn", len(dr), len(dr)),
                           _stage_rows("screens_d1_d4_dedup_d5", len(dr), n_scr, by_reason=dict(sorted(reasons.items()))),
                           _stage_rows("gates", n_scr, len(surv[p]), by_reason=gate_rep[p]["reasons"]),
                           _stage_rows("sealed_screens", len(surv[p]), len(surv[p]) - n_sealed),
                           _stage_rows("token_target_selection", len(surv[p]) - n_sealed, len(sel[p]),
                                       note="survivors in walk order until the part's loss-token target")],
                "gates": gate_rep[p], "tokens": topup_token_stats(sel[p]),
                "reasoning_tokens": length_summary([r["reasoning_tokens"] for r in sel[p]]),
                "final_section_tokens": length_summary([r["final_section_tokens"] for r in sel[p]])}
            if spec.retention_part == "math":
                parts_rep[p]["ref_match"] = dict(Counter(r["ref_match"] for r in sel[p]))
            else:
                parts_rep[p]["ending_form"] = dict(Counter(cand_by_id[r["task_id"]]["meta"]["ending_form"] for r in sel[p]))
                parts_rep[p]["n_options"] = dict(Counter(str(r["n_options"]) for r in sel[p]))
        for p in prof.reserve_parts:
            rows = [r for r in res_rows if r["retention_part"] == p]
            parts_rep[p] = {"source": "data/pools/retention/replay_reserve.parquet (verbatim; screened with the pool)",
                            "tokens": topup_token_stats(rows),
                            "reasoning_tokens": length_summary([r["reasoning_tokens"] for r in rows])}
        tot_loss = sum(r["n_assistant_tokens"] for r in out_rows)
        shares = {p: round(sum(r["n_assistant_tokens"] for r in out_rows if r["replay_part"] == p) / tot_loss, 4)
                  for p in (*prof.order, *prof.reserve_parts)}
        ks = [k for k in (25, 50, 100, 200, 300, 400, 500) if k < len(out_rows)]
        sources_rep: dict[str, Any] = {
            "nm_math": {"hf_repo": PARTS["math"].hf_repo, "split": PARTS["math"].split, "revision": revisions["nm_math"],
                        "licence": PARTS["math"].dataset_license, "generator": PARTS["math"].generator}}
        math_bands = {p: [prof.parts[p].floor, prof.parts[p].cap] for p in prof.order if prof.parts[p].retention_part == "math"}
        filters_rep: dict[str, Any] = {
            "math": {"loader": f"apex_data.retention.math at the {math_cap // 1024}K think cap (character pre-filter "
                               f"{math_cap:,} x {common.CHARS_PER_TOKEN_CEILING:g})",
                     "ref_match": list(cvg.REF_MATCHES), "single_committed_answer": "every \\boxed{} states the same normalised answer",
                     "ends_with_answer": f"<= {cvg.FINAL_TAIL_CHARS} characters after the last \\boxed{{}}",
                     "bands": math_bands}}
        if "sci_mcq" in sources:
            sources_rep["nm_sci"] = {"hf_repo": PARTS["sci_mcq"].hf_repo, "split": PARTS["sci_mcq"].split,
                                     "revision": revisions["nm_sci"], "licence": PARTS["sci_mcq"].dataset_license,
                                     "generator": PARTS["sci_mcq"].generator}
            filters_rep["sci_mcq"] = {
                "loader": "apex_data.retention.science MCQ (letter present, listed, equal to the reasoning's letter), "
                          f"candidates of run {pm['run_id']}",
                "ends_with_letter": "the answer ends with the chosen letter in an explicit final-answer form; only "
                                    "formatting follows", "band": [0, prof.parts["sci_mcq"].cap]}
        if prof.reserve_parts:
            sources_rep["replay_reserve"] = {"path": rel(pool["files"]["replay_reserve"]),
                                             "sha256": pool["sha256"]["replay_reserve"], "parts": list(prof.reserve_parts)}
        manifest = {
            "pool": "retention", "kind": "convergence_replay", "profile": prof.name, "version": prof.version,
            "run_id": run_id, "created": utc_now(), "seed": a.seed, "code": git_info(),
            "plan": CVG_TEXT[prof.name]["plan"], "rationale": CVG_TEXT[prof.name]["rationale"],
            "reproduce": command_line(),
            "sources": sources_rep,
            "filters": {
                **filters_rep,
                "final_section_tokens_max": prof.final_cap, "row_cap_tokens": cvg.ROW_CAP,
                "gates": "render.filters.filter_episodes (closed reasoning, think cap of the part, repetition, render, row cap)",
                "screens": "D1 canary, D4 task names, prompt dedup (exact + MinHash, the prior rows win), D5' vs dev QA, sealed "
                           "p1/p2 screens (scripts/data/run_screens.py)"},
            "disjoint_from": {"rows": {"path": rel(pool["files"]["rows"]), "sha256": pool["sha256"]["rows"], "n": len(pool["rows"]["rows"])},
                              "replay_reserve": {"path": rel(pool["files"]["replay_reserve"]), "sha256": pool["sha256"]["replay_reserve"],
                                                 "n": len(pool["rows"]["replay_reserve"])},
                              "topup_rows": {"run_id": tm.get("run_id"), "sha256": topup_sha, "n": len(topup_rows)}},
            "targets": {"convergence_loss_tokens": a.convergence_replay, "by_part": targets},
            "counts": {"rows": len(out_rows), "by_part": dict(Counter(r["replay_part"] for r in out_rows))},
            "tokens": {**topup_token_stats(out_rows), "loss_token_shares": shares},
            "order": CVG_TEXT[prof.name]["order"],
            "prefix": {"max_deviation_by_cell": dev_cell, "max_deviation_by_part": dev_part,
                       "loss_token_shares_by_part": cvg.prefix_token_shares(part_seq, [r["n_assistant_tokens"] for r in out_rows], ks)},
            "load": {sp: load_info[sp] for sp in sources},
            "prefilter": {sp: f.table() for sp, f in pre_funnel.items()}, "walk": walk,
            "screens": {k: v for k, v in screen_rep.items() if k != "dedup"} | {"dedup": {k: v for k, v in screen_rep["dedup"].items()
                                                                                          if k != "clusters_with_new_rows"}},
            "sealed_screens": sealed_counts, "parts": parts_rep, "files": files, "checks": checks,
        }
        write_json(man_path, manifest)
        failed = [c["check"] for c in checks if not c["pass"]]
        for c in checks:
            run.log(f"[check] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {json.dumps(c['detail'], default=str)[:300]}")
            run.metric({"step": "check", "check": c["check"], "pass": c["pass"]})
        if failed:
            run.error(f"convergence replay checks failed: {failed}")
            rc = 1
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if rc == 0 else "failed",
                           outputs=files, counts=manifest["counts"], checks_failed=failed)
        run.finalize()
        run.log(f"done: convergence replay {len(out_rows)} rows -> {out_path}; checks failed: {failed or 'none'}")
    return rc

# ------------------------------------------------------------------------------------------------------ compose v5
V5_VERSION = "retention-v5"
V5_PREFIX_TOLERANCE = 0.015      # loss-token share: |prefix share - target| at --slice-tokens and over the whole file
V5_STRATA_TOLERANCE = 0.03       # row share of a stratum within its part vs the pool's row share of that stratum
V5_DROP_COLUMNS = ("replay_part", "replay_rank", "final_section_tokens")


def main_compose_v5(a: argparse.Namespace) -> int:
    """Compose mode (see the module docstring and apex_data.retention.v5)."""
    import pyarrow.parquet as pq

    from apex_data.retention import convergence as cvg, v5

    pool = read_pool(a.exclude_pool)
    hm_path, hm_man_path = a.hard_math / "hard_math.parquet", a.hard_math / "hard_math_manifest.json"
    hm_man = json.loads(hm_man_path.read_text())
    hm_sha = sha256_file(hm_path)
    if hm_sha != hm_man["files"]["hard_math"]["sha256"]:
        raise ValueError(f"{hm_path}: sha256 {hm_sha} differs from its manifest")
    if hm_man.get("profile") != "hard_math_v5" or [c["check"] for c in hm_man["checks"] if not c["pass"]]:
        raise ValueError(f"{hm_man_path}: not a passing hard_math_v5 build")
    hm_prof = cvg.PROFILES["hard_math_v5"]
    targets = common.apportion(a.compose_v5, v5.V5_SHARES)
    run_id = a.run_id or new_run_id("data_retention_v5")
    out_path, man_path = a.out_dir / "rows.parquet", a.out_dir / "manifest.json"
    protected = {**pool["files"], "hard_math": hm_path}
    before = {k: sha256_file(f) for k, f in protected.items()}
    rc = 0
    with RunDir(run_id, root=a.log_root, log_name="train.log", resume=a.resume) as run:
        created = utc_now()
        run.write_metadata(run_id=run_id, step="data_build_retention_slice --compose-v5", command=command_line(),
                           code=git_info(), config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                           data_version={"pool": pool["manifest"].get("version"), "pool_rows_sha256": pool["sha256"]["rows"],
                                         "hard_math_sha256": hm_sha, "hard_math_run_id": hm_man["run_id"]},
                           model_revision=str(a.tokenizer.name), hardware=hardware_info(), seed=a.seed, start_time=created,
                           status="running", targets=targets)
        run.log(f"[compose] {a.compose_v5} loss tokens {targets} (slice {a.slice_tokens}) -> {out_path}")
        by_part: dict[str, list[dict[str, Any]]] = {p: [] for p in v5.V5_ORDER}
        for r in pq.read_table(pool["files"]["rows"]).to_pylist() + pq.read_table(hm_path).to_pylist():
            by_part[v5.v5_part(r)].append(r)
        supply = {p: {"rows": len(v), "loss_tokens": sum(r["n_assistant_tokens"] for r in v)} for p, v in by_part.items()}
        run.log(f"[supply] {supply}")
        n_tok = lambda r: r["n_assistant_tokens"]  # noqa: E731
        sel = {p: v5.take_until(v5.part_order(rows, a.seed, keep_order=p == "hard_math"), targets[p], n_tok)
               for p, rows in by_part.items()}
        merged = v5.token_interleave(sel, n_tok)
        out_rows = [{**{k: v for k, v in r.items() if k not in V5_DROP_COLUMNS}, "v5_part": p, "v5_cell": v5.v5_cell(r),
                     "final_section_tokens": r.get("final_section_tokens"), "v5_rank": k} for k, (p, r) in enumerate(merged)]
        if len({frozenset(r) for r in out_rows}) != 1:
            raise ValueError("pool rows and hard-math rows do not share one column set")
        files = {"rows": write_rows(out_path, out_rows)}
        run.log(f"[write] {len(out_rows)} rows -> {out_path} ({files['rows']['sha256'][:12]})")
        # checks
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: Any) -> None:
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        loss = {p: sum(map(n_tok, sel[p])) for p in v5.V5_ORDER}
        add("part_targets_met", all(loss[p] >= targets[p] for p in v5.V5_ORDER),
            {p: {"target": targets[p], "loss_tokens": loss[p], "rows": len(sel[p]), "supply": supply[p]} for p in v5.V5_ORDER})
        labels, toks = [r["v5_part"] for r in out_rows], [r["n_assistant_tokens"] for r in out_rows]
        ats = sorted({t for t in (1_000_000, 2_000_000, a.slice_tokens) if t < a.compose_v5})
        prefixes = v5.prefix_shares(labels, toks, ats)
        worst = {f"{x['target_loss_tokens']}": max(abs(x["shares"].get(p, 0.0) - v5.V5_SHARES[p]) for p in v5.V5_ORDER)
                 for x in prefixes}
        checked = [x for x in prefixes if x["target_loss_tokens"] in (a.slice_tokens, sum(toks))]
        add("prefix_part_shares", all(worst[f"{x['target_loss_tokens']}"] <= V5_PREFIX_TOLERANCE for x in checked),
            {"tolerance": V5_PREFIX_TOLERANCE, "worst_abs_deviation": worst})
        strata = {}
        for p in ("math", "chat", "science"):
            pool_cells = Counter(v5.v5_cell(r) for r in by_part[p])
            got = Counter(v5.v5_cell(r) for r in sel[p])
            strata[p] = {c: {"pool": round(pool_cells[c] / len(by_part[p]), 4), "slice": round(got[c] / len(sel[p]), 4)}
                         for c in sorted(pool_cells)}
        add("strata_row_shares_kept", all(abs(v["pool"] - v["slice"]) <= V5_STRATA_TOLERANCE for x in strata.values() for v in x.values()),
            {"tolerance": V5_STRATA_TOLERANCE, "by_part": strata})
        hm_rows = [r for r in out_rows if r["v5_part"] == "hard_math"]
        hm_bad = [r["task_id"] for r in hm_rows if not (cvg.in_band("math_hard", r["reasoning_tokens"], hm_prof.parts)
                                                       and cvg.final_section_ok(r["final_section_tokens"], hm_prof.final_cap))]
        add("hard_math_band_and_final_section", not hm_bad,
            {"reasoning_band": [hm_prof.parts["math_hard"].floor, hm_prof.parts["math_hard"].cap], "final_cap": hm_prof.final_cap,
             "violations": hm_bad[:5]})
        ids = [r["task_id"] for r in out_rows]
        shas = [r["prompt_sha256"] for r in out_rows]
        add("unique_task_ids_and_prompts", len(set(ids)) == len(ids) and len(set(shas)) == len(shas),
            {"rows": len(ids), "task_ids": len(set(ids)), "prompt_sha256": len(set(shas))})
        add("hard_math_build_disjoint_from_pool", next(c for c in hm_man["checks"] if c["check"] == "disjoint_from_pool_reserve_topup")["pass"],
            {"hard_math_manifest": rel(hm_man_path)})
        add("row_cap_64k", all(r["n_tokens"] <= cvg.ROW_CAP for r in out_rows), {"max": max(r["n_tokens"] for r in out_rows)})
        add("loss_tokens_equal_mask_sums", all(sum(r["assistant_masks"]) == r["n_assistant_tokens"] for r in out_rows), {})
        add("provenance_valid", all(_valid(r) for r in out_rows), {})
        after = {k: sha256_file(f) for k, f in protected.items()}
        add("inputs_unchanged", after == before, {"before": before, "after": after})
        slice_prefix = next(x for x in prefixes if x["target_loss_tokens"] == a.slice_tokens)
        manifest = {
            "pool": "retention", "kind": "retention_v5", "version": V5_VERSION, "run_id": run_id, "created": utc_now(),
            "seed": a.seed, "code": git_info(), "reproduce": command_line(),
            "plan": "data/manifests/v5/DATA_CARD_v5.md (retention slice): the first build's retention pool plus hard math, "
                    "25 percent of all loss tokens, split hard math 23 / other math 25 / science 15 / instruction-following "
                    "25 / chat 12",
            "sources": {"pool": {"manifest": rel(pool["files"]["manifest"]), "rows": rel(pool["files"]["rows"]),
                                 "sha256": pool["sha256"]["rows"], "run_id": pool["manifest"].get("run_id")},
                        "hard_math": {"manifest": rel(hm_man_path), "rows": rel(hm_path), "sha256": hm_sha,
                                      "run_id": hm_man["run_id"], "profile": hm_man["profile"]}},
            "shares": v5.V5_SHARES, "targets": {"loss_tokens": a.compose_v5, "slice_loss_tokens": a.slice_tokens, "by_part": targets},
            "order": ("rows sorted by v5_rank: per part, cells (math length bands, chat topic groups, science MCQ / RQA) "
                      "as seeded shuffles interleaved by row share (hard math in its walk order), taken until the part's "
                      "target; parts merged by loss tokens (apex_data.retention.v5.token_interleave), so every prefix keeps "
                      "the part shares up to about one row: take the prefix that reaches the wanted loss tokens"),
            "slice": {"loss_tokens": a.slice_tokens, "rows": slice_prefix["rows"], "shares": slice_prefix["shares"],
                      "hard_math_rows": sum(1 for r in out_rows[:slice_prefix["rows"]] if r["v5_part"] == "hard_math"),
                      "hard_math_loss_token_share": slice_prefix["shares"].get("hard_math")},
            "counts": {"rows": len(out_rows), "by_part": dict(Counter(labels)), "by_cell": dict(sorted(Counter(r["v5_cell"] for r in out_rows).items()))},
            "tokens": {**topup_token_stats(out_rows), "by_part": {p: topup_token_stats(sel[p]) for p in v5.V5_ORDER}},
            "prefix_shares": prefixes, "strata": strata, "supply": supply, "files": files, "checks": checks,
        }
        write_json(man_path, manifest)
        failed = [c["check"] for c in checks if not c["pass"]]
        for c in checks:
            run.log(f"[check] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {json.dumps(c['detail'], default=str)[:300]}")
            run.metric({"step": "check", "check": c["check"], "pass": c["pass"]})
        if failed:
            run.error(f"retention v5 checks failed: {failed}")
            rc = 1
        run.write_resources([
            ("duration_sec", round(run.elapsed(), 1), "s", "wall clock on the CPU VM"),
            ("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), "GB", "main process"),
            ("peak_gpu_memory_gb", "n/a", "GB", "CPU-only step"),
            ("gpu_hours", 0, "h", "CPU-only step"),
            ("cost_usd", 0, "USD", "CPU-only step; costs count GPU hours only (the CPU VM is not charged)"),
        ])
        run.write_metadata(end_time=utc_now(), duration_sec=round(run.elapsed(), 1), status="finished" if rc == 0 else "failed",
                           outputs=files, counts=manifest["counts"], slice=manifest["slice"], checks_failed=failed)
        run.finalize()
        run.log(f"done: retention v5 {len(out_rows)} rows -> {out_path}; slice {manifest['slice']}; checks failed: {failed or 'none'}")
    return rc


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    torch.set_num_threads(MAX_WORKERS)
    sys.exit(main())
