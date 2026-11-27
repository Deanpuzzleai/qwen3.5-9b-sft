# apex_data: contract for the data-construction pipeline

This package builds the training and dev data of `docs/DATA_PLAN.md` revision 4.3.6; `docs/EXECUTION_PLAN.md` gives the
order of work. This file fixes what every module must agree on: rules, layout, source codes, schemas, provenance fields,
shared parameters and file ownership. A module that needs something not written here asks for a change to this file
first.

## Rules that bind every module

1. Benchmark material never enters training data, prompts, dev items or checkpoint selection. Benchmark *inputs* are read
   by exactly one script, `scripts/decontam/seal_benchmark_inputs.py`, run by hand; it writes hashed n-grams,
   embeddings and counts under the git-ignored `data/decontam/sealed/`. No other module opens `third_party/`, benchmark
   results or a benchmark dataset; the screens match against the sealed artefacts only.
2. Nothing is deleted: filters mark and count, dropped records stay on disk with their reason.
3. Every script runs from the repository root with `.venv/train/bin/python`, documents its inputs and outputs in
   `--help`, is seeded (`apex_data.paths.SEED`), writes atomically, and resumes from existing outputs.
4. Every run writes a run record through `apex_eval.common.RunDir` (metadata.json with command, git commit, config,
   data versions, start and end, seed; the log; metrics.jsonl; errors.log, "none" if none; resources.csv). Run ids come
   from `apex_eval.common.new_run_id("data_<step>")`.
5. Secrets (`HF_TOKEN`, `GITHUB_TOKEN`, `OPENAI_API_KEY`) come from the environment or `.env`; never printed or stored.
6. No LLM judge and no reward model touches training data. Labels are execution verdicts, reference answers or
   deterministic checkers.
7. The only renderer is `src/apex_train/data_format.py` (`Sample` -> `build_rows`); training renders equal serving
   renders byte for byte.
8. Every filter writes n_in / n_dropped / n_out; every selection is seeded and its strata are reported.

## Layout (`apex_data/paths.py`)

| Path | Content | Tracked |
|---|---|---|
| `RAW` = `/mnt/apex-shared/apex_data/raw/<source_dir>/` | pinned downloads (`scripts/data/fetch_raw.py`), never modified after the fetch | no |
| `WORK` = `/mnt/apex-shared/apex_data/work/<run_id>/` | large intermediates: loaded episodes, rendered rows before selection, rollouts, verdicts | no |
| `data/pools/<pool>/` | pool outputs: task files, manifests and reports; rows Parquet files are git-ignored (the manifest keeps their sha256 and counts) | yes (except `*.parquet`) |
| `data/manifests/*.json` | ids, revisions, sha256, funnel counts, selections, the dev freeze, exclusions | yes |
| `data/processed/sft/{train,dev,pilot}_rows.parquet` | the frozen SFT set (M2.5): the trainer's row columns only (`apex_train.data_format.ROW_COLUMNS`); provenance in `*_provenance.parquet` sidecars aligned row by row | no (sha256 in the mixture manifest) |
| `data/decontam/report.json` | screen counts (no text) | yes |
| `data/decontam/sealed/` | benchmark-derived screen inputs, written by the seal run | never |
| `data/README.md` | the data card, generated from the manifests | yes |

Manifests every builder reads and never writes: `data/manifests/decontam_exclusions.json` (repository and episode
exclusions), `data/manifests/dev_exclusions.json` (SWE dev repositories and terminal dev task ids, written by WP6 at the
dev freeze), `data/manifests/raw_sources.json` (written by WP1).

## Source codes (`schema.SOURCES`)

| Code | Dataset | Role |
|---|---|---|
| `oswt` | `nvidia/Open-SWE-Traces` v1.2 `data/minisweagent/qwen38_27b/{swe-rebench-v2,scale-swe}/` @ `e192b26a` | SWE teacher rows (WP3); `oswt_subsource` says which folder |
| `th3k` | `shunzou05/TerminalHorizon-3K` (+ `shunzou05/TerminalHorizon-Environment` `metadata/tasks.parquet` @ `e9ba9fdc` for categories) | terminal teacher rows, unverified half (WP4) |
| `tgr` | `ai-and/termgrade-trajectories` (DeepSeek-V4-Pro messages, `trials.parquet`) | terminal teacher rows, verified half (WP4) |
| `tgr_env` | `ai-and/termgrade-environments` | terminal dev tasks (WP6) and the stage-2 terminal pool (WP8) |
| `swesmith` | `SWE-bench/SWE-smith-py`, `SWE-bench/SWE-smith-go` | SWE dev tasks (WP6) and the stage-2 SWE pool (WP8) |
| `nm_math` | `nvidia/Nemotron-SFT-Math-v4` | retention math (WP5) |
| `nm_ifchat` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | retention IF and chat (WP5) |
| `nm_sci` | `nvidia/Nemotron-Science-v1` (`MCQ`, `RQA`) | retention science (WP5) |
| `devqa` | GPQA Diamond (ids only), reasoning-gym items, AIME 2025 | dev STEM QA (WP6) |

Auxiliary downloads that are not row sources: `nebius/SWE-rebench-V2` (task metadata and reference patches for the
size strata), `PrimeIntellect/Scale-SWE-Verified` (instance ids that pass gold validation), `BAAI/bge-small-en-v1.5` @
`5c38ec7c` under `/mnt/apex-shared/models/` (D3/D5' embeddings and the chat topic classifier).

## Schemas (`apex_data/schema.py`)

- `Task`, `Episode`, `Verdict` dataclasses with `validate()`; JSONL IO through `write_jsonl` / `read_jsonl`.
- IDs: `task_id = "<source>:<upstream_id>"` (`schema.task_id`), `episode_id = "<task_id>#<round>#<policy>#<sample_index>"`
  for our rollouts; teacher rows keep the upstream trajectory id in `episode_id`.
- Rendered rows: the `apex_train.data_format.ROW_COLUMNS` plus every column of `schema.PROVENANCE_FIELDS` (flat scalars,
  `None` where a field does not apply); `schema.validate_provenance(row)` must pass on every row a pool writes (Open-SWE-Traces
  rows must also carry `oswt_subsource`, `size_stratum`, `patch_files`, `patch_added_lines`; stage-2 SWE-smith rows carry
  none of them). The fields
  added for revision 4.3: `label_provenance` (`verifier` | `authors_filter+handshake` | `reference` | `format`),
  `oswt_subsource` (`swe-rebench-v2` | `scale-swe`), `size_stratum` (`big` | `feature` | `random`), `category_group`
  (the TerminalHorizon quota group), `fits_48k` (the row also fits the fallback cap). Retention labels: math and IF rows
  are `reference` (upstream-verified answer; our constraint checker), Science MCQ, Science RQA and chat rows are `format`
  (no reference answers ship; MCQ letter consistency, RQA single `\boxed{}`, chat = the dataset's reward-model selection).

## Row kinds and the shared parameters (plan sections 3 and 4)

| Row kind | data_format mode | Loss | Cap |
|---|---|---|---|
| `swe_episode` | `episode`: one row per trajectory, every assistant turn trained with its reasoning | reasoning + content + tool calls + `<|im_end|>` | 64K target, 48K fallback (T8 decides); never truncated, dropped and counted when too long |
| `terminal_turn` | `per_turn` with `turns_per_episode = 2`: the latest turn whose prompt fits the cap, plus one random earlier fitting turn | the trained turn's reasoning and JSON action | same cap |
| `retention` | `episode` on a single-turn conversation | reasoning + answer | reasoning <= 10K tokens (math, science), <= 6K (IF, chat) |

- Quotas: SWE Go 470 / Py 320 (120 SWE-rebench-V2 + 200 Scale-SWE) / TS 260 / JS 120, size-stratified (65 percent large
  first, 35 percent random), <= 5 rows per repository; terminal 1,000 episodes (TerminalHorizon <= 500 by category
  quota, one per environment; TermGrade >= 500, skip all-six-solve tasks, <= 2 per task); retention math 630, IF 435,
  chat 285, science MCQ 275, RQA 125, plus a ~600-row stage-2 replay reserve disjoint from the SFT rows.
- Both caps are prepared: every agentic row is built at 64K and records `fits_48k`; the SWE size-stratified draw runs
  separately at each cap. The freeze (M2.5) picks one prepared selection and does not rebuild.
- Stage-2 rollout caps and sampling presets: SWE 50 steps / 16K tokens per turn / 40K output / 20 min; terminal 40
  turns / 60K output / 30 min; sampling = the evaluation locks (`configs/eval/*.yaml`).
- Dedup: exact keys (normalised issue text, canonical patch, (repo, base_commit, touched files), (repo, PR), normalised
  prompt, normalised Terminus-2 task text) and MinHash-LSH on word 5-grams (250 permutations, 25 x 10 bands; Jaccard
  >= 0.7 for issue text and patches, >= 0.8 for QA prompts and terminal task text).
- Decontamination: D1 canary, D2 repository exclusion (the 11 SWE-bench Pro repositories and forks, plus the
  repositories of `decontam_exclusions.json`), D4 source exclusions and Terminal-Bench slug screen, D3 sealed inputs-only
  n-gram and embedding screen, D5' dev-validity screen (plan section 7). D2's fork-graph step queries the GitHub API with
  `GITHUB_TOKEN` from `.env`; without a token the step is recorded as `not_run` in the screen report and the data card
  says so (the name, basename, identifier and signature rules still run). Screens never rewrite pool files: dropped
  ids go to `data/manifests/screens_<pass>.json["dropped"]` and the mixture step reads them from there.

## Module map

| Area | Files |
|---|---|
| contract | `apex_data/{__init__,schema,paths}.py`, this file |
| fetch | `scripts/data/fetch_raw.py`, `data/manifests/raw_sources.json` |
| screens | `apex_data/screens/*`, `apex_data/pools/common.py`, `scripts/data/run_screens.py`, `scripts/decontam/seal_benchmark_inputs.py` |
| dev set | `apex_data/devset/*`, `scripts/data/build_dev_*.py`, `data/manifests/dev_{swe,terminal,qa}.json`, `dev_freeze.json`, `dev_exclusions.json` |
| SWE slice | `apex_data/teacher/*`, `scripts/data/build_teacher_leg.py`, `data/pools/swe_teacher/` |
| terminal slice | `apex_data/terminal/*`, `scripts/data/build_terminal_slice.py`, `data/pools/terminal_teacher/`; the `turns_per_episode` option in `apex_train/data_format.py` |
| retention slice | `apex_data/retention/*` (incl. `topic_check_labels.json`), `apex_data/prompts/{__init__,checkers,constraints}.py`, `scripts/data/{build_retention_slice,merge_retention_topup}.py`, `data/pools/retention/` |
| mixture | `apex_data/render/{select,mixture,split}.py`, `apex_data/cards.py`, `scripts/data/{build_mixture,write_data_card}.py`, `data/processed/sft/`, `data/README.md` |
| shared rendering | `apex_data/render/{adapters,filters}.py` |
