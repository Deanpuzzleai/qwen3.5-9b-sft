"""Filesystem layout shared by every data-pipeline script (src/apex_data/README.md, "Layout").

Big artefacts live on the shared filesystem; the repository keeps manifests, reports and small pool files only.
Environment overrides: APEX_DATA_RAW, APEX_DATA_WORK, APEX_BASE_MODEL_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("APEX_DATA_RAW", "/mnt/apex-shared/apex_data/raw"))      # pinned downloads (fetch_raw.py), read-only after fetch
WORK = Path(os.environ.get("APEX_DATA_WORK", "/mnt/apex-shared/apex_data/work"))   # large intermediates: work/<run_id>/...
DATA = REPO / "data"                       # tracked: manifests/, pools/ (small files), decontam/report.json, README.md (data card)
MANIFESTS = DATA / "manifests"
POOLS = DATA / "pools"
PROCESSED = DATA / "processed"             # git-ignored: data/processed/sft/{train,dev}_rows.parquet, data/processed/stage2/
DECONTAM = DATA / "decontam"
SEALED = DECONTAM / "sealed"               # git-ignored output of scripts/decontam/seal_benchmark_inputs.py (run manually)
LOGS = REPO / "logs"
CONFIGS = REPO / "configs"
MODELS = Path("/mnt/apex-shared/models")
BASE_MODEL_DIR = Path(os.environ.get("APEX_BASE_MODEL_DIR",
                                     str(MODELS / "Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a")))  # configs/eval/model.yaml
BGE_SMALL_DIR = MODELS / "bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"   # D3/D5' embeddings, chat topic classifier

# Manifests every builder reads (written by WP6 and by manual decisions; never edited by a pool builder)
DECONTAM_EXCLUSIONS = MANIFESTS / "decontam_exclusions.json"   # repository and episode exclusions (plan 3.2, 3.3, 7)
DEV_EXCLUSIONS = MANIFESTS / "dev_exclusions.json"             # SWE dev repositories and terminal dev task ids (WP6)
RAW_SOURCES = MANIFESTS / "raw_sources.json"                   # pinned revisions and sha256 of every download (WP1)

SEED = 42


def ensure_dirs() -> None:
    for p in (RAW, WORK, MANIFESTS, POOLS, DECONTAM, SEALED):
        p.mkdir(parents=True, exist_ok=True)
