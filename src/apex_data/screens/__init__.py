"""Screens of docs/DATA_PLAN.md section 7 (WP2). Each module marks and counts; none deletes anything or stores text.

  licence        repository licence allowlist; GitHub repository lookup (cache, GITHUB_TOKEN) shared with D2
  d2_repos       D2: SWE-bench Pro repositories (metadata, predecessors, basenames, identifiers, module-path signatures,
                 fork graph) and the repository-level exclusions of data/manifests/decontam_exclusions.json
  d1_canary      D1: canary GUID / sentence / phrase over files, trees, zips, JSONL, parquet and in-memory records
  d4_structural  D1 + D4 over Harbor task directories and terminal task text (Terminal-Bench slugs, registry references)
  dedup          exact keys, MinHash-LSH, the Terminus-2 task-text key, cross-split rule (plus fix families, episodes)
  lineage        (repo, PR) / (repo, base_commit) keys across pools and dev manifests
  d3_match       D3 / D5 sealed inputs-only screen: matcher over the sealed artefacts, artefact format, embedder
  d5_prime       D5': retention prompts vs dev QA items (bge-small cosine >= 0.85)

scripts/data/run_screens.py runs them over pool items (pass p1_pools) and rendered rows (pass p2_rows).
"""
