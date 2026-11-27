"""Revision-5 retention slice: the first build's pool rows plus the hard-math rows, mixed by loss tokens
(scripts/data/build_retention_slice.py --compose-v5).

Parts and their share of the slice's loss tokens (V5_SHARES): hard math 23 (hard_math.parquet of
--convergence-profile hard_math_v5), other math 25, science 15 (MCQ and free-form), instruction-following 25, chat 12.
Within a part the first build's strata keep their row proportions (V5_CELLS): math length bands le4k / 4k_7k / 7k_10k,
chat hss / other, science MCQ / RQA. Each cell is a seeded shuffle of its rows (hard math keeps its walk order); the cells
of a part are interleaved by row share (draw.interleave), and the part takes rows in that order until its loss-token
target (take_until). The parts are then merged by `token_interleave`, so every prefix of the ranked file holds each
part's loss-token share up to about one row: P1e can scale the slice by taking a prefix.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from apex_data.retention import draw

V5_SHARES: dict[str, float] = {"hard_math": 0.23, "math": 0.25, "science": 0.15, "if": 0.25, "chat": 0.12}
V5_ORDER = tuple(V5_SHARES)


def v5_part(row: Mapping[str, Any]) -> str:
    """The v5 part of a pool row (retention_part) or of a hard-math row (replay_part math_hard)."""
    if row.get("replay_part") == "math_hard":
        return "hard_math"
    part = row["retention_part"]
    return "science" if part in ("sci_mcq", "sci_rqa") else part


def v5_cell(row: Mapping[str, Any]) -> str:
    """The stratum cell of a row within its v5 part: math/<length band>, chat/<topic group>, science/<mcq|rqa>."""
    part = v5_part(row)
    if part == "math":
        return f"math/{row['length_band']}"
    if part == "chat":
        return f"chat/{row['topic_group']}"
    if part == "science":
        return f"science/{row['retention_part'].removeprefix('sci_')}"
    return part


def part_order(rows: Sequence[dict[str, Any]], seed: int, *, keep_order: bool = False) -> list[dict[str, Any]]:
    """One part's rows in draw order: each cell a seeded shuffle (draw.shuffled; keep_order keeps the given order
    instead), the cells interleaved by row share."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault(v5_cell(r), []).append(r)
    cells = dict(sorted(cells.items()))
    if not keep_order:
        cells = {c: draw.shuffled(v, seed, "retention_v5", c) for c, v in cells.items()}
    return [r for _, r in draw.interleave(cells, seed, tag="retention_v5_cells")]


def take_until(rows: Sequence[dict[str, Any]], target: int, tokens: Callable[[dict[str, Any]], int]) -> list[dict[str, Any]]:
    """The shortest prefix whose tokens reach target (all rows if they never do)."""
    out, total = [], 0
    for r in rows:
        if total >= target:
            break
        out.append(r)
        total += tokens(r)
    return out


def token_interleave(parts: Mapping[str, Sequence[Any]], tokens: Callable[[Any], int]) -> list[tuple[str, Any]]:
    """[(part, item)] merged in loss-token order: the j-th item of a part gets the key (tokens before it + half its own
    tokens) / the part's total tokens (the Webster rule in token space), ties broken by the parts' order. Each part keeps
    its own order, and the first items up to any point hold each part's token share of the whole up to about one item."""
    keyed = []
    for rank, (part, items) in enumerate(parts.items()):
        total = sum(tokens(x) for x in items)
        before = 0
        for j, x in enumerate(items):
            t = tokens(x)
            keyed.append(((before + t / 2) / total, rank, j, part, x))
            before += t
    keyed.sort(key=lambda k: k[:3])
    return [(part, x) for _, _, _, part, x in keyed]


def prefix_shares(labels: Sequence[str], tokens: Sequence[int], at: Sequence[int]) -> list[dict[str, Any]]:
    """Loss-token shares by label of the shortest prefix reaching each token count in `at` (and of the whole)."""
    out = []
    for target in [*at, sum(tokens)]:
        acc: dict[str, int] = {}
        total = k = 0
        while k < len(labels) and total < target:
            acc[labels[k]] = acc.get(labels[k], 0) + tokens[k]
            total += tokens[k]
            k += 1
        out.append({"target_loss_tokens": target, "rows": k, "loss_tokens": total,
                    "shares": {p: round(v / total, 4) for p, v in sorted(acc.items())} if total else {}})
    return out
