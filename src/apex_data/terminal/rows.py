"""Pool row files of the terminal slice: `data_format.ROW_COLUMNS` + `schema.PROVENANCE_FIELDS` + bookkeeping columns,
one flat Parquet table (src/apex_data/README.md, "Schemas"). A trainer reads the first block with
`pq.read_table(path, columns=list(ROW_COLUMNS))`; the mixture reads the rest.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from apex_data.schema import PROVENANCE_FIELDS
from apex_train.data_format import ROW_COLUMNS, arrow_schema, histogram

BOOKKEEPING = ("row_key", "family", "n_steps", "sha256", "subcategory")
_INT_PROVENANCE = {"patch_files", "patch_added_lines"}


def rows_schema() -> Any:
    import pyarrow as pa

    fields = list(arrow_schema())
    for k in PROVENANCE_FIELDS:
        if k in ROW_COLUMNS:
            continue
        typ = pa.bool_() if k == "fits_48k" else pa.int32() if k in _INT_PROVENANCE else pa.string()
        fields.append(pa.field(k, typ))
    fields += [pa.field("row_key", pa.string()), pa.field("family", pa.string()), pa.field("n_steps", pa.int32()),
               pa.field("sha256", pa.string()), pa.field("subcategory", pa.string())]
    return pa.schema(fields)


def flat_row(row: dict[str, Any], provenance: dict[str, Any], subcategory: str | None = None) -> dict[str, Any]:
    """One output record from a `filters.Rendered` (row + provenance)."""
    out = {k: row[k] for k in ROW_COLUMNS}
    out.update({k: provenance[k] for k in PROVENANCE_FIELDS if k not in ROW_COLUMNS})
    out.update(row_key=provenance["row_key"], family=provenance["family"], n_steps=provenance["n_steps"],
               sha256=provenance["sha256"], subcategory=subcategory)
    return out


def write_table(records: Sequence[dict[str, Any]], path: Path) -> str:
    """Atomic Parquet write with the explicit schema; returns the file's sha256."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(list(records), schema=rows_schema()), tmp, compression="zstd")
    tmp.replace(path)
    return sha256_file(path)


def read_table(path: Path, columns: Sequence[str] | None = None, episode_ids: set[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=list(columns) if columns else None)
    if episode_ids is not None:
        t = t.filter(pc.is_in(t["episode_id"], value_set=pa.array(sorted(episode_ids), pa.string())))
    return t.to_pylist()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rows, sequence and loss tokens, overall and per label provenance (light records with n_tokens etc.)."""
    rows = list(rows)

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        seq = [r["n_tokens"] for r in rs]
        loss = [r["n_assistant_tokens"] for r in rs]
        return {"rows": len(rs), "episodes": len({r["episode_id"] for r in rs}),
                "sequence_tokens": sum(seq), "loss_tokens": sum(loss),
                "sequence_tokens_mean": round(sum(seq) / len(seq), 1) if seq else 0.0,
                "loss_tokens_mean": round(sum(loss) / len(loss), 1) if loss else 0.0,
                "sequence_tokens_max": max(seq, default=0), "sequence_histogram": histogram(seq),
                "loss_histogram": histogram(loss),
                "fits_48k": sum(bool(r.get("fits_48k")) for r in rs)}

    by = Counter(r["label_provenance"] for r in rows)
    return {"all": block(rows), "by_label_provenance": {p: block([r for r in rows if r["label_provenance"] == p])
                                                         for p in sorted(by)}}
