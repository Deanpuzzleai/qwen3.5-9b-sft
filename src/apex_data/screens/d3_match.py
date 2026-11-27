"""D3 / D5 pipeline-side matcher (plan 7, the sealed inputs-only screen): screens training items against the sealed
benchmark artefacts written by scripts/decontam/seal_benchmark_inputs.py (the only script that opens benchmark inputs).
This module never sees benchmark text; it also owns the artefact format and the embedder so the sealed side and the
pipeline side cannot drift.

Sealed layout (`data/decontam/sealed/`, git-ignored, written by the seal run; other content of that directory is never
read or written here: only `manifest.json` and the four `<benchmark>/` directories below belong to this format):
  manifest.json                 `writer` = MANIFEST_WRITER; per benchmark: item count, n-gram counts, input files with
                                sha256, embedding model, n-gram spec; `fixture: true` marks synthetic test artefacts, which
                                `SealedIndex.load` refuses by default
  <benchmark>/ids.json          benchmark item ids (public identifiers: task names, instance ids, question ids)
  <benchmark>/ngram8.npy + ngram8_offsets.npy, ngram13.npy + ngram13_offsets.npy
                                per-item sorted unique uint64 xxhash64 arrays of word n-grams, concatenated (offsets: n_items + 1)
  <benchmark>/embeddings.npy    float16 (n_items, 384): BAAI/bge-small-en-v1.5, mean pooled over the attention mask, L2-normalised

Normalisation of text on both sides: `dedup.tokenize` (NFKC, casefold, `\\w+` words, i.e. punctuation stripped), then
xxhash64 of the space-joined n-gram (NGRAM_SPEC). Embeddings take the raw text truncated to 512 tokens.

Rules (a training item is a hit when any applies):
  ngram13       shares any 13-gram with a benchmark item
  ngram8_50pct  >= 50 % of its 8-grams (minimum 10) fall in one benchmark item's set (Tulu 3 / OLMo-decon style)
  cosine085     cosine >= 0.85 to a benchmark item's embedding; applied to task statements and prompts (`Item.kind`
                "task" / "prompt"), not to full rendered rows ("row"): the first 512 tokens of a 64K-token row are mostly
                system prompt, not task text, so rows are screened by the n-gram rules
A `Decision` carries (item id, benchmark, rule, benchmark item id, score) and never text. `apply_drops` turns decisions into
(benchmark, rule, dropped id, replacement id) logs through the replace-from-same-stratum hook that WP7 supplies
(`ReplacementHook`); `same_stratum_replacer` is the default implementation. `write_report` merges counts only into
data/decontam/report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from apex_data.paths import BGE_SMALL_DIR, DECONTAM, SEALED, SEED
from apex_data.screens.dedup import shingle_hashes, tokenize

BENCHMARKS = ("terminal_bench", "asibench", "hle", "swebench_pro")
NGRAM_SHORT, NGRAM_LONG = 8, 13
MIN_SHORT_NGRAMS, SHORT_FRACTION, COSINE_THRESHOLD = 10, 0.5, 0.85
NGRAM_SPEC = "nfkc-casefold-word-xxh64-v1"
RULES = ("ngram13", "ngram8_50pct", "cosine085")
BGE_NAME = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_MODEL_DIR = Path(os.environ.get("APEX_BGE_MODEL_DIR", str(BGE_SMALL_DIR)))
EMBED_DIM, EMBED_MAX_LENGTH = 384, 512
REPORT_PATH = DECONTAM / "report.json"
MANIFEST_NAME = "manifest.json"
MANIFEST_WRITER = "scripts/decontam/seal_benchmark_inputs.py"
ARTEFACT_FILES = frozenset({"ids.json", "ngram8.npy", "ngram8_offsets.npy", "ngram13.npy", "ngram13_offsets.npy",
                            "embeddings.npy"})


class ForeignFileError(RuntimeError):
    """A path the seal run would write already holds something this format did not write."""


def check_sealed_writable(sealed_dir: Path, names: Sequence[str]) -> None:
    """Refuse to write unless every target is ours: manifest.json absent or carrying `writer` = MANIFEST_WRITER, and each
    <benchmark>/ directory absent or holding only artefact files (and their `.tmp` leftovers). Other entries of the
    sealed directory are never looked into."""
    sealed_dir = Path(sealed_dir)
    mpath = sealed_dir / MANIFEST_NAME
    if mpath.exists():
        try:
            writer = json.loads(mpath.read_text(encoding="utf-8")).get("writer")
        except (json.JSONDecodeError, AttributeError):
            writer = None
        if writer != MANIFEST_WRITER:
            raise ForeignFileError(f"{mpath} was not written by {MANIFEST_WRITER}; refusing to overwrite it")
    for name in names:
        if name not in BENCHMARKS:
            raise ForeignFileError(f"{name!r} is not a sealed benchmark directory")
        d = sealed_dir / name
        if d.is_symlink() or (d.exists() and not d.is_dir()):
            raise ForeignFileError(f"{d} exists and is not a plain directory; refusing to write into it")
        if d.is_dir():
            foreign = sorted(p.name for p in d.iterdir() if not p.is_file() or p.is_symlink() or not _is_artefact(p.name))
            if foreign:
                raise ForeignFileError(f"{d} holds {len(foreign)} entries that are not seal artefacts; refusing to write into it")


def _is_artefact(name: str) -> bool:
    """An artefact file name, or the temporary name it is written under (`<name>.tmp.npy`, `<name>.tmp`)."""
    for suffix in (".tmp.npy", ".tmp"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name in ARTEFACT_FILES


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def item_ngrams(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(8-gram hashes, 13-gram hashes) of one text, sorted unique uint64."""
    words = tokenize(text)
    return shingle_hashes(words, NGRAM_SHORT), shingle_hashes(words, NGRAM_LONG)


# --- embeddings ---------------------------------------------------------------------------------------------------------
class Embedder:
    """bge-small-en-v1.5 through plain transformers on CPU: mean pooling over the attention mask, L2-normalised float32."""

    def __init__(self, model_dir: Path | str = BGE_MODEL_DIR, *, max_length: int = EMBED_MAX_LENGTH, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"embedding model not found at {model_dir} (download {BGE_NAME}@{BGE_REVISION} there or set APEX_BGE_MODEL_DIR)")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.info: dict[str, Any] = {"name": BGE_NAME, "revision": BGE_REVISION if BGE_REVISION in str(model_dir) else "unknown",
                                     "path": str(model_dir), "weights_sha256": sha256_file(model_dir / "model.safetensors"),
                                     "pooling": "mean", "normalised": True, "max_length": max_length, "dim": EMBED_DIM}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(self.torch.nn.functional.normalize(vec, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, EMBED_DIM), dtype=np.float32)


# --- sealed artefacts -----------------------------------------------------------------------------------------------------
def _concat(sets: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([s.size for s in sets])
    data = np.concatenate(sets).astype(np.uint64) if sets else np.empty(0, dtype=np.uint64)
    return data, offsets


def write_sealed_benchmark(bench_dir: Path, ids: Sequence[str], ngram8: Sequence[np.ndarray], ngram13: Sequence[np.ndarray],
                           embeddings: np.ndarray | None) -> dict[str, Any]:
    """Write one benchmark's artefacts atomically (ids, hashes, vectors; no text) and return its counts. Refuses a
    directory that holds anything but seal artefacts (`check_sealed_writable`)."""
    bench_dir = Path(bench_dir)
    check_sealed_writable(bench_dir.parent, [bench_dir.name])
    bench_dir.mkdir(parents=True, exist_ok=True)
    if not (len(ids) == len(ngram8) == len(ngram13)):
        raise ValueError("ids, ngram8 and ngram13 must align")
    d8, o8 = _concat(list(ngram8))
    d13, o13 = _concat(list(ngram13))
    arrays = {"ngram8.npy": d8, "ngram8_offsets.npy": o8, "ngram13.npy": d13, "ngram13_offsets.npy": o13}
    if embeddings is not None:
        if embeddings.shape != (len(ids), EMBED_DIM):
            raise ValueError(f"embeddings shape {embeddings.shape} != ({len(ids)}, {EMBED_DIM})")
        arrays["embeddings.npy"] = embeddings.astype(np.float16)
    for name, arr in arrays.items():
        tmp = bench_dir / (name + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, bench_dir / name)
    write_json_atomic(bench_dir / "ids.json", list(ids))
    return {"n_items": len(ids), "n_8grams": int(d8.size), "n_13grams": int(d13.size),
            "n_items_without_8grams": int(np.sum(np.diff(o8) == 0)), "n_items_without_13grams": int(np.sum(np.diff(o13) == 0)),
            "embeddings": embeddings is not None}


@dataclass
class SealedBenchmark:
    name: str
    ids: list[str]
    h8: np.ndarray       # all 8-gram hashes, sorted
    i8: np.ndarray       # benchmark item index of each entry of h8
    h13: np.ndarray
    i13: np.ndarray
    embeddings: np.ndarray | None   # float32 (n_items, 384), rows L2-normalised

    @classmethod
    def load(cls, bench_dir: Path, name: str) -> SealedBenchmark:
        bench_dir = Path(bench_dir)
        ids = json.loads((bench_dir / "ids.json").read_text(encoding="utf-8"))

        def index(n: int) -> tuple[np.ndarray, np.ndarray]:
            data = np.load(bench_dir / f"ngram{n}.npy")
            offsets = np.load(bench_dir / f"ngram{n}_offsets.npy")
            item = np.repeat(np.arange(len(ids), dtype=np.int32), np.diff(offsets))
            order = np.argsort(data, kind="stable")
            return data[order], item[order]

        h8, i8 = index(NGRAM_SHORT)
        h13, i13 = index(NGRAM_LONG)
        emb_path = bench_dir / "embeddings.npy"
        emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else None
        return cls(name, ids, h8, i8, h13, i13, emb)

    @staticmethod
    def _lookup(index: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For sorted `queries`: (lo, hi) ranges inside `index` holding equal hashes (hi > lo marks a match)."""
        lo = np.searchsorted(index, queries, side="left")
        hi = np.searchsorted(index, queries, side="right")
        return lo, hi

    def match_ngrams(self, g8: np.ndarray, g13: np.ndarray, *, min_short: int = MIN_SHORT_NGRAMS,
                     fraction: float = SHORT_FRACTION) -> list[tuple[str, int, float]]:
        """[(rule, benchmark item index, score)]: ngram13 score = shared 13-grams; ngram8_50pct score = fraction."""
        out: list[tuple[str, int, float]] = []
        if g13.size and self.h13.size:
            lo, hi = self._lookup(self.h13, g13)
            hit = hi > lo
            if hit.any():
                items = np.concatenate([self.i13[a:b] for a, b in zip(lo[hit], hi[hit], strict=True)])
                counts = np.bincount(items, minlength=len(self.ids))
                best = int(counts.argmax())
                out.append(("ngram13", best, float(counts[best])))
        if g8.size >= min_short and self.h8.size:
            lo, hi = self._lookup(self.h8, g8)
            hit = hi > lo
            if hit.sum() >= fraction * g8.size:           # necessary condition before the per-item count
                items = np.concatenate([self.i8[a:b] for a, b in zip(lo[hit], hi[hit], strict=True)])
                counts = np.bincount(items, minlength=len(self.ids))
                best = int(counts.argmax())
                frac = counts[best] / g8.size
                if frac >= fraction:
                    out.append(("ngram8_50pct", best, round(float(frac), 4)))
        return out

    def match_embeddings(self, vectors: np.ndarray, threshold: float = COSINE_THRESHOLD) -> list[tuple[int, int, float]]:
        """[(query row, benchmark item index, cosine)] for the best benchmark item of each query at or above threshold."""
        if self.embeddings is None or vectors.size == 0:
            return []
        sims = vectors @ self.embeddings.T
        best = sims.argmax(axis=1)
        out = []
        for q, b in enumerate(best):
            s = float(sims[q, b])
            if s >= threshold:
                out.append((q, int(b), round(s, 4)))
        return out


@dataclass
class SealedIndex:
    benchmarks: dict[str, SealedBenchmark]
    manifest: dict[str, Any]
    manifest_sha256: str
    path: Path

    @classmethod
    def load(cls, sealed_dir: Path | str = SEALED, *, allow_fixture: bool = False, names: Sequence[str] | None = None) -> SealedIndex:
        sealed_dir = Path(sealed_dir)
        mpath = sealed_dir / MANIFEST_NAME
        if not mpath.exists():
            raise FileNotFoundError(f"no sealed manifest at {mpath}; run scripts/decontam/seal_benchmark_inputs.py first")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("writer") != MANIFEST_WRITER:
            raise ValueError(f"{mpath} was not written by {MANIFEST_WRITER}")
        if manifest.get("fixture") and not allow_fixture:
            raise ValueError(f"{mpath} holds synthetic fixture artefacts; refusing to screen real data against them")
        if manifest.get("ngram_spec") != NGRAM_SPEC:
            raise ValueError(f"sealed n-gram spec {manifest.get('ngram_spec')!r} != {NGRAM_SPEC!r}; re-run the sealing script")
        benches = {}
        for name in names or manifest.get("benchmarks", {}):
            if (sealed_dir / name / "ids.json").exists():
                benches[name] = SealedBenchmark.load(sealed_dir / name, name)
        if not benches:
            raise FileNotFoundError(f"no benchmark artefacts under {sealed_dir}")
        return cls(benches, manifest, sha256_file(mpath), sealed_dir)

    @staticmethod
    def exists(sealed_dir: Path | str = SEALED) -> bool:
        return (Path(sealed_dir) / MANIFEST_NAME).exists()

    def embedding_model(self) -> dict[str, Any]:
        return dict(self.manifest.get("embedding_model") or {})

    def check_embedder(self, embedder: Embedder) -> None:
        sealed = self.embedding_model()
        if sealed and sealed.get("weights_sha256") and sealed["weights_sha256"] != embedder.info["weights_sha256"]:
            raise ValueError("the sealed embeddings were made with different model weights than the pipeline embedder; re-seal or fix APEX_BGE_MODEL_DIR")


# --- matching -----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    id: str
    text: str
    kind: str = "task"                      # task | prompt | row
    stratum: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Decision:
    item_id: str
    benchmark: str
    rule: str
    benchmark_item: str
    score: float


class Matcher:
    """Screens items against every loaded benchmark; embeddings are computed in batches for task / prompt items only."""

    def __init__(self, index: SealedIndex, embedder: Embedder | None = None, *, cosine: float = COSINE_THRESHOLD,
                 fraction: float = SHORT_FRACTION, min_short: int = MIN_SHORT_NGRAMS, batch_size: int = 256):
        self.index, self.embedder = index, embedder
        self.cosine, self.fraction, self.min_short, self.batch_size = cosine, fraction, min_short, batch_size
        if embedder is not None:
            index.check_embedder(embedder)

    def screen(self, items: Iterable[Item]) -> list[Decision]:
        decisions: list[Decision] = []
        pending: list[Item] = []
        for it in items:
            g8, g13 = item_ngrams(it.text)
            for name, bench in self.index.benchmarks.items():
                for rule, b_idx, score in bench.match_ngrams(g8, g13, min_short=self.min_short, fraction=self.fraction):
                    decisions.append(Decision(it.id, name, rule, bench.ids[b_idx], score))
            if self.embedder is not None and it.kind != "row":
                pending.append(it)
                if len(pending) >= self.batch_size:
                    decisions.extend(self._embed_batch(pending))
                    pending = []
        if pending:
            decisions.extend(self._embed_batch(pending))
        return decisions

    def _embed_batch(self, batch: Sequence[Item]) -> list[Decision]:
        vectors = self.embedder.embed([it.text for it in batch])
        out = []
        for name, bench in self.index.benchmarks.items():
            for q, b_idx, score in bench.match_embeddings(vectors, self.cosine):
                out.append(Decision(batch[q].id, name, "cosine085", bench.ids[b_idx], score))
        return out


def summarise(decisions: Sequence[Decision], n_items: int) -> dict[str, Any]:
    """Counts only: hits per benchmark and rule, distinct items hit (an item may hit several rules / benchmarks)."""
    by_bench: dict[str, Counter[str]] = defaultdict(Counter)
    for d in decisions:
        by_bench[d.benchmark][d.rule] += 1
    return {"items_screened": n_items, "items_hit": len({d.item_id for d in decisions}), "n_decisions": len(decisions),
            "hits": {b: dict(c) for b, c in sorted(by_bench.items())}, "by_rule": dict(Counter(d.rule for d in decisions))}


# --- drop and replace ------------------------------------------------------------------------------------------------------------
class ReplacementHook(Protocol):
    """WP7 supplies this: given the dropped item and the decision, return the id of a replacement from the same language
    and difficulty stratum of the already-screened pool, or None when the stratum is exhausted (shortfall published)."""

    def __call__(self, dropped: Item, decision: Decision) -> str | None: ...


def stratum_key(stratum: Mapping[str, Any] | None, keys: Sequence[str] = ("language", "difficulty")) -> tuple[Any, ...]:
    return tuple((stratum or {}).get(k) for k in keys)


def same_stratum_replacer(candidates: Iterable[Item], *, keys: Sequence[str] = ("language", "difficulty"),
                          used: set[str] | None = None, seed: int = SEED) -> ReplacementHook:
    """Default hook: a seeded random, never-reused candidate with the same stratum key; None when none is left."""
    rng = np.random.default_rng(seed)
    pools: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for c in candidates:
        pools[stratum_key(c.stratum, keys)].append(c.id)
    for ids in pools.values():
        ids.sort()
        rng.shuffle(ids)
    used = used if used is not None else set()

    def hook(dropped: Item, decision: Decision) -> str | None:   # noqa: ARG001 - the signature is the contract
        for cand in pools.get(stratum_key(dropped.stratum, keys), []):
            if cand not in used and cand != dropped.id:
                used.add(cand)
                return cand
        return None

    return hook


@dataclass(frozen=True)
class DropLog:
    benchmark: str
    rule: str
    dropped_id: str
    replacement_id: str | None


def apply_drops(decisions: Sequence[Decision], items_by_id: Mapping[str, Item], hook: ReplacementHook | None = None) -> list[DropLog]:
    """One log per dropped item (first decision in rule priority ngram13 > ngram8_50pct > cosine085); ids only."""
    order = {r: i for i, r in enumerate(RULES)}
    first: dict[str, Decision] = {}
    for d in sorted(decisions, key=lambda d: (d.item_id, order.get(d.rule, 99), d.benchmark)):
        first.setdefault(d.item_id, d)
    logs = []
    for item_id, d in sorted(first.items()):
        item = items_by_id.get(item_id) or Item(item_id, "")
        logs.append(DropLog(d.benchmark, d.rule, item_id, hook(item, d) if hook else None))
    return logs


# --- report ----------------------------------------------------------------------------------------------------------------------------
def write_report(pass_name: str, counts: Mapping[str, Any], path: Path | str = REPORT_PATH) -> dict[str, Any]:
    """Merge one pass's counts into the tracked report; an earlier version of the same pass moves to `history`."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"passes": {}, "history": []}
    if pass_name in report["passes"]:
        report.setdefault("history", []).append({"pass": pass_name, "replaced_at": utc_now(), "counts": report["passes"][pass_name]})
    report["passes"][pass_name] = {**dict(counts), "written_at": utc_now()}
    report["updated_at"] = utc_now()
    write_json_atomic(path, report)
    return report


def decisions_to_dicts(decisions: Sequence[Decision]) -> list[dict[str, Any]]:
    return [asdict(d) for d in decisions]
