"""D5' dev-validity paraphrase screen (plan 7): BAAI/bge-small-en-v1.5 mean-pooled embeddings (the D3 embedder,
`d3_match.Embedder`, from the local snapshot at `paths.BGE_SMALL_DIR`) on CPU; any training retention prompt with
cosine >= 0.85 to any dev QA item (GPQA Diamond, AIME 2025, reasoning-gym) is dropped from training (marked
``split = "excluded"`` with the reason; nothing is deleted). Counts go to the pool manifests; no dev text is written anywhere.

Interface for WP5: `embedder = load_embedder()`, then `apply(records, dev_items, embedder, text_key="prompt")` with
records as dicts (`split`, `screens`, the prompt under `text_key`) and dev items as {"dev_id", "slice", "text"}.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apex_data.screens import d3_match

MODEL = d3_match.BGE_NAME
REVISION = d3_match.BGE_REVISION
THRESHOLD = 0.85
POOLING = "mean"
SCREEN_VERSION = f"d5_prime-{MODEL.split('/')[-1]}-{REVISION[:8]}-{POOLING}"


def load_embedder(model_dir: Path | str = d3_match.BGE_MODEL_DIR, batch_size: int = 64) -> d3_match.Embedder:
    return d3_match.Embedder(model_dir, batch_size=batch_size)


def nearest(train: np.ndarray, dev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(max cosine, argmax dev index) per training row."""
    if len(train) == 0 or len(dev) == 0:
        return np.zeros(len(train), dtype=np.float32), np.full(len(train), -1)
    sims = train @ dev.T
    return sims.max(axis=1), sims.argmax(axis=1)


def screen_texts(train_texts: Sequence[str], dev_texts: Sequence[str], embedder: d3_match.Embedder,
                 threshold: float = THRESHOLD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(max cosine per training text, nearest dev index, boolean hit mask)."""
    best, arg = nearest(embedder.embed(list(train_texts)), embedder.embed(list(dev_texts)))
    return best, arg, best >= threshold


def apply(records: list[dict[str, Any]], dev_items: list[dict[str, Any]], embedder: d3_match.Embedder, *,
          threshold: float = THRESHOLD, text_key: str = "prompt", log: Callable[[str], None] = print) -> dict[str, Any]:
    """Annotate records in place: screens["d5_prime"] = {max_cosine, nearest_dev_id, nearest_dev_slice, hit}; hits among
    training rows become split="excluded" with the reason. Returns the counts-only report."""
    train_texts = [r.get(text_key) or "" for r in records]
    if dev_items and records:
        best, arg, hits = screen_texts(train_texts, [d["text"] for d in dev_items], embedder, threshold)
    else:
        best, arg, hits = (np.zeros(len(records), dtype=np.float32), np.full(len(records), -1),
                           np.zeros(len(records), dtype=bool))
    by_slice: Counter = Counter()
    n_dropped = 0
    for r, b, a, h in zip(records, best, arg, hits, strict=True):
        d = dev_items[int(a)] if int(a) >= 0 else None
        info = {"version": SCREEN_VERSION, "threshold": threshold, "max_cosine": round(float(b), 4),
                "nearest_dev_id": d["dev_id"] if d else None, "nearest_dev_slice": d["slice"] if d else None, "hit": bool(h)}
        r.setdefault("screens", {})["d5_prime"] = info
        if h:
            by_slice[d["slice"]] += 1
            if r.get("split") == "train":
                r["split"] = "excluded"
                r["screens"]["d5_prime"]["dropped"] = True
                r["screens"]["exclusion_reason"] = "d5_prime"
                n_dropped += 1
    quantiles = {q: round(float(np.quantile(best, float(q))), 4) for q in ("0.5", "0.9", "0.99", "1.0")} if len(best) else {}
    report = {"version": SCREEN_VERSION, "model": MODEL, "model_revision": REVISION, "pooling": POOLING, "threshold": threshold,
              "embedding_weights_sha256": embedder.info.get("weights_sha256"), "n_train_screened": len(records),
              "n_dev_items": len(dev_items), "dev_slices": dict(Counter(d["slice"] for d in dev_items)),
              "n_hits": int(hits.sum()), "n_dropped_from_train": n_dropped, "hits_by_dev_slice": dict(by_slice),
              "max_cosine_quantiles": quantiles}
    log(f"d5_prime: {len(records)} prompts vs {len(dev_items)} dev items: {int(hits.sum())} hits >= {threshold}, "
        f"{n_dropped} dropped from train; max cosine {quantiles}")
    return report
