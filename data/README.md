# Data

Stage-1 SFT data of the four trained data versions (v1, v5, v6, v7) and the stage-2 data that was built but not
trained on. No data rows are distributed here: the rows files (Parquet) are rebuilt by `scripts/prepare_data.sh` from
the pinned public sources, and every manifest under `manifests/` records ids, revisions, counts and sha256.

| Version | Used by | Frozen (UTC) | Train rows | Validation rows | Loss tokens (train + val) | Loss share SWE / terminal / retention |
|---|---|---|---|---|---|---|
| v1 | `sft_main_20261012_154313` (v1 model) and the pilots | 2026-10-12 15:42 | 5,598 | 298 | 35.03M | 61.2 / 13.9 / 24.9 % |
| v5 | `sft_v5_20261013_072959` (v5 model) and diagnostics D1-D3, no-DeltaNet | 2026-10-13 07:29 | 3,822 | 206 | 12.99M | 56.3 / 18.7 / 25.0 % |
| v6 (revision v6r2) | `sft_v6_20261013_174750` (v6 model) and D4 | 2026-10-13 17:46 | 5,159 | 279 | 9.96M | 37.5 / 34.5 / 28.1 % |
| v7 (revision v7r1) | `sft_v7_20261013_221813`, `sft_v7lr5e5_20261013_234355` (v7 models) | 2026-10-13 22:17 | 3,566 | 200 | 9.59M | 37.4 / 33.4 / 29.1 % |

Each version's generated data card is in its manifest folder (`manifests/v1/DATA_CARD_v1.md`,
`manifests/v5/DATA_CARD_v5.md`, `manifests/v6/DATA_CARD_v6.md` with the addendum `DATA_CARD_v6_notes.md`,
`manifests/v7/DATA_CARD_v7.md`); every number below comes from those cards and the manifests next to them.
The builders write their manifests at the top level of `manifests/` (for example `mixture.json`, `split.json`,
`screens_*.json`); the frozen copies of each version are kept in `manifests/v1/`, `v5/`, `v6/`, `v7/` and `stage2/`.

## 1. Sources and licences

Every loss-bearing token comes from an open-weight teacher model, and every dataset licence permits training and
release. Per row, the provenance sidecars record `hf_repo`, `hf_revision`, `dataset_license`, `generator` and
`label_provenance` (`verifier` = upstream execution label, `reference` = checked against a reference answer or a
constraint checker, `format` = format-checked only).

| Slice | Dataset (revision) | Licence | Teachers (generator) | Label | Versions |
|---|---|---|---|---|---|
| SWE agent episodes (mini-swe-agent) | `nvidia/Open-SWE-Traces` v1.2 (`e192b26a43b8`); tasks from SWE-rebench-V2 and Scale-SWE | CC-BY-4.0 (each row keeps its code repository's SPDX licence: MIT, Apache-2.0, BSD-2/3 only) | Qwen3.8-27B, thinking mode (Apache-2.0) | verifier | all |
| Terminal agent turns (Harbor Terminus-2) | `ai-and/termgrade-trajectories` (`7d584d63dd59`) on the synthetic TermGrade tasks (`ai-and/termgrade-environments`, Apache-2.0) | Apache-2.0 | v1: DeepSeek-V4-Pro (MIT); v5/v6: Kimi-K3 (Kimi K3 License, MIT-style) first, then Qwen3.6-27B and Gemma-4 (Apache-2.0); v7: half DeepSeek-V4-Pro, half the v6-style body (mostly Kimi-K3, with Qwen3.6-27B and Gemma-4) | verifier (reward = 1) | all |
| Retention (single turn) | `nvidia/Nemotron-SFT-Math-v4` (`84d42ad0cb96`), `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` (`be3b3e04ef60`), `nvidia/Nemotron-Science-v1` (`82e1af468197`) | CC-BY-4.0; IF-Chat CC-BY-4.0 AND ODC-BY-1.0 | DeepSeek-V4-Pro (High), GPT-OSS-120B, GLM-5 (as released by NVIDIA) | reference / format | all |

Auxiliary inputs (not row sources): `PrimeIntellect/Scale-SWE-Verified` (admission list), `nebius/SWE-rebench-V2`
(task metadata for the size strata), `BAAI/bge-small-en-v1.5` (embedding screens). Every file's revision and sha256:
`manifests/raw_sources.json`.

Why these sources (from public dataset cards and papers): Open-SWE-Traces v1.2 is the public trajectory set in exactly
our SWE harness (mini-swe-agent) from a same-family thinking teacher, with execution labels and per-row repository
licences; termgrade-trajectories gives verifier-labelled Terminus-2 trials of several open-weight teachers on synthetic
tasks with per-task difficulty labels (six-model pass rates); the Nemotron sets are licence-clean reasoning data for
keeping QA and instruction-following behaviour. Excluded by rule: trajectories of closed-model teachers (provider
terms), other agent formats (harness mismatch), datasets without a resolvable licence, and anything derived from the
four evaluation benchmarks (data card v1, section 11).

## 2. Development set (shared by every version)

Frozen 2026-10-12 06:36 UTC, before any training row was drawn (`manifests/dev_freeze.json`); admitted task lists
frozen 10:16 UTC (`manifests/dev_admitted.json`). SWE: 100 SWE-smith tasks from 8 held-out repositories (MIT); terminal:
40 tasks of TermGrade's `unseen` split; QA: GPQA Diamond (ids only) and 30 reasoning-gym items. Dev repositories and task
ids are excluded from every training and stage-2 source (`manifests/dev_exclusions.json`). Checkpoint selection and all
data decisions used this dev set only.

## 3. Per-version summary

### v1: first build (`manifests/v1/`, data card v1)

- SWE: 1,170 episodes from 736 repositories (Go 470, Python 320, TypeScript 260, JavaScript 120), size-stratified toward
  multi-file changes; 21.45M loss tokens.
- Terminal: 951 DeepSeek-V4-Pro episodes on 648 TermGrade tasks, two trained turns per episode (1,902 rows); 4.86M.
- Retention: 2,824 rows (math 1,014, instruction following 704, chat 460, science MCQ 443, science free-form 203); 8.72M.
- Row cap 65,536 tokens (memory gate on the B200); rows are never truncated, a row that does not fit is dropped and
  counted. Validation shard 5 % grouped by task; pilot file: 10 % of the training rows (561), task-stratified.
- The card's header says "CANDIDATE": it was generated three minutes before the freeze of the same files; the freeze
  record is the `freeze` block of `manifests/v1/mixture.json`.

### v5: efficiency-gated build (`manifests/v5/`, data card v5)

What changed and why: on the dev set the v1 model copied the teachers' length rather than their efficiency: its SWE
turns averaged 670 thinking tokens (3.7x Base), its terminal turns 1,539 (5.2x Base) with 55-59 turns per task against
the teacher's 27, and 10.5 % of its SWE turns had no tool call (Base 1.3 %). On TermGrade, DeepSeek-V4-Pro used a median
99.9K output tokens per passing trial against Kimi-K3's 5.6K at a statistically tied pass rate. v5 kept the teachers'
decisions but not their cost: SWE difficulty bands from the teacher's own attempts (hard 1 of 3, medium 2 of 3, easy
3 of 3), per-turn thinking limits per band anchored to Base and fixed before the build, among gate-passing attempts the
one with the least thinking, zero wasted actions, verified completion, a recovery preference; terminal teacher Kimi-K3;
retention 25 % including ~70 hard-math rows (v1's 20-25 % retention had kept QA on the dev set: pilots at ~15 % hurt QA,
45 % raised agent-format parse failures).

- SWE 820 episodes (793 instances, 553 repositories), 7.31M loss tokens; terminal 516 episodes, 2,159 per-turn rows,
  2.43M; retention 1,049 rows, 3.25M. Total 4,028 rows, 39.34M sequence tokens.

### v6 (revision v6r2): masks on the frozen rows (`manifests/v6/`, data card v6 + notes)

What changed and why: v5 looped on the SWE dev items (responses cut at the 16K per-turn limit: 9.9 % at checkpoint 125
against Base's 1.1 %). Diagnostics on v5's rows (D1 without SWE rows, D2 at LR 5e-5, D3 without terminal rows) showed
that the SWE rows drove about two thirds of the excess loops and the update size the rest (the decision rules were
fixed before the diagnostics were read; see the top-level README). v6 therefore changed masks, not sources:

- SWE: the same v5 episodes with read-only turns masked (kept only among the first 3 turns, right after an error, or
  right before an edit) and submit turns masked when none of the 3 previous turns ran tests or the program; then trimmed
  to 60 % of each band's loss tokens (episodes with an unverified submit first, language shares protected within 1
  point). 505 episodes, 3.73M loss tokens.
- Terminal rebuilt with the existing per-turn builder: up to 2 trials per task, only turns within the per-turn band
  thinking ceiling trained, at most 6 trained turns per episode, never the confirmation turn. 1,109 episodes, 4,072 rows,
  3.44M.
- Retention 28 % (pilots on v1 data: ~15 % lowered QA, 45 % broke the agent tool-call format, 20-25 % kept both): 861
  rows, 2.79M.

### v7 (revision v7r1): teacher-only, mixed short and long (`manifests/v7/`, data card v7)

What changed and why: v6 looped less but got stuck repeating whole turns, and after long thinking it rarely went back to
acting (6 % against Base's 25 %). The gates had kept only the teacher's shortest 7 % of attempts, so the model never saw
long thinking that ends in an action. v7 keeps v6's size and changes which teacher turns are trained:

- SWE: one random passing attempt per task (recovery trials first: 38.6 % of episodes), v6's masks plus adaptive
  thinking masks (routine turns trained only with <= 1,000 thinking tokens; decision-point turns, right after a real
  failure or before the first edit, up to 8K when not repetitive and ending in a tool call; > 8K never; turns above 2K
  capped at 10 % of SWE loss tokens). 321 episodes, 3.59M loss tokens.
- Terminal: half DeepSeek-V4-Pro passing trials (debugging-arc trials first, late long-context turns trained), half the
  Kimi-K3 body; per-turn guards (thinking <= 4K, parsed command, non-repetitive, no confirmation turn, no empty keystrokes
  at a stuck prompt). 584 episodes, 2,584 rows, 3.21M. Per-teacher rows and loss tokens: the sources table of the v7 card
  and `manifests/v7/terminal_slice.json` (the card's one-line terminal summary above that table still names only Kimi-K3
  and prints zero thinking statistics; it is a generator artefact of the two-half build).
- Retention: v6's 861 rows unchanged, 2.79M.

## 4. Cleaning, deduplication and decontamination (every version)

- Cleaning (pool builders, each filter writes n_in / n_dropped / n_out): resolved / reward = 1 only; licence allowlist;
  bash tool schema; clean non-empty model patch; reasoning present and closed on every trained turn; parser round trip;
  tamper screen (test-file edits); identity and credential screens; language and repetition gates; row cap. Funnels:
  `manifests/*/pools/*/manifest.json` and the data cards.
- Deduplication: exact (normalised text, canonical patch, repository + base commit + files) and near (MinHash-LSH on word
  5-grams, Jaccard >= 0.7 SWE, >= 0.8 terminal and prompts); lineage check train vs dev on (repository, PR) and
  (repository, commit).
- Decontamination, two passes per version (`scripts/data/run_screens.py`; counts only, no text): `p1_pools` over one item
  per pool task plus the dev items, `p2_rows` over the rendered training and validation rows, repeated until it drops
  nothing. Screens: D1 canary strings (GUID and canary sentence, byte-level and NFKC); D2 repository exclusion (the
  SWE-bench Pro repositories named in the public task list, their forks via the GitHub fork graph, and `bottlepy/bottle`,
  the code base of a public Terminal-Bench task; `manifests/decontam_exclusions.json`); D4 public Terminal-Bench task
  names; D3 n-gram (any shared 13-gram, or >= 50 % of an item's 8-grams) and embedding (bge-small cosine >= 0.85)
  overlap against a decontamination index held outside the repository. Results: `manifests/*/screens_*.json`; every
  frozen version's final `p2_rows` pass dropped 0 rows. The screens catch verbatim, near-verbatim and close-paraphrase
  overlap; they do not prove the absence of contamination.

## 5. Splits

Validation shard: 5 % of the rows, grouped by task (no task on both sides; an episode's per-turn rows stay together),
stratified by cell (SWE band x language, terminal band, retention part), seed 42 (`manifests/*/split.json`). It is used
for the validation-loss curve only. Each version has its own shard, so validation losses are comparable within a
version, not across versions.

## 6. Pipeline and scripts

`scripts/prepare_data.sh` runs the v7 build end to end and lists the earlier versions' commands; every builder documents
its inputs and outputs in its docstring.

| Step | Script(s) | Output |
|---|---|---|
| raw sources | `scripts/data/fetch_raw.py` | pinned downloads, `manifests/raw_sources.json` |
| dev set | `scripts/data/build_dev_set.py`, `freeze_dev_admission.py` | `manifests/dev_*.json` |
| SWE slice | v1 `build_teacher_leg.py`; v5 `build_swe_slice.py`; v6 `build_swe_masks_v6.py`; v7 `build_swe_slice_v7.py` | pool rows + `pools/swe_*/manifest.json` |
| terminal slice | v1 `build_terminal_slice.py`; v5/v6/v7 `build_terminal_slice_v5.py` / `_v6.py` / `_v7.py` | per-turn rows + manifest |
| retention | `build_retention_slice.py` (+ `merge_retention_topup.py` for v1; `--compose-v5` for v5-v7) | `pools/retention*/manifest.json` |
| screens | `run_screens.py` (p1 and p2 passes) | `screens_*.json` |
| mixture, split, gates, card, freeze | v1 `build_mixture.py` + `write_data_card.py`; v5-v7 `build_mixture_v5.py --version vN` | rows files, `mixture.json`, `selection.json`, `split.json`, `gate_report.json`, data card, freeze marker |
| rows format | `src/apex_train/data_format.py` (the only renderer; training render equals serving render) | `ROW_COLUMNS` Parquet |
| stage-2 pairs (not trained) | `build_stage2_pairs.py`, `audit_stage2_pairs.py`, `freeze_stage2_pairs.py` | `manifests/stage2/` |

## 7. Freeze records (sha256)

| Version | Freeze record (in this package) | sha256 of the record | Rows files it pins (train / validation sha256) |
|---|---|---|---|
| v1 | `manifests/v1/mixture.json` (`status: FROZEN`, `freeze` block: run `data_mixture_freeze_20261012_154250`, 2026-10-12T15:42:51Z) | `31c827180d879fd4cdfc5f67d3a2236e557f3c2d423283019ef5d79147d88d7f` (equals the `mixture_manifest` sha256 in the v1 training run's metadata) | `a154244532e6c1deff72d9b24a54e8520986fcf42affec479f32bf75af24b926` / `340c4aed85602b63c9851395b33153383aff93ef04708ae6c640970ef0e31a68` |
| v5 | `manifests/v5/data_frozen_v5.json` (2026-10-13T07:29:11Z) | `bf9fa3b9623e3af691086e3603ddbcfe80b81e7bd98e117bb2eb796a802dc1e2` | `a2a88b1d574569222df41b7dd0b564ae82ff5afbba2f7ba4cc4a6da9bcdcf80b` / `1d2bd5f1eaf0f2b9e56c501cb476e2955c1b69cf6abe49762766b62caa8d2ffa` |
| v6r2 | `manifests/v6/data_frozen_v6.json` (2026-10-13T17:46:13Z; supersedes v6r1, whose record `data_frozen_v6.superseded_v6r1_20261013_174432.json` has sha256 `b24b7acfe1457adcd69bc8addcb038fae0f3f10a461d35761744bb1a04daa50e`) | `90d2d01de2e09b1cfe74293dec04659496d8e66e312f4e8be08d184ff628e12a` | `a758e4132a3cb826c0424a27b98e0880a84d9176951cf9321e0fbcb04001a21c` / `56a8478161da4a21ed2e9afcdcfc25087e18ef5fdf2715267d95a8f017f6b3a4` |
| v7r1 | `manifests/v7/data_frozen_v7.json` (2026-10-13T22:17:07Z) | `9a38d6091fd476163c9ca694efd688bb727e896b6867f57c71590223d406417c` | `15e731d3de1cee94d0abe833fad70400b753caea619962645558b449512d71ae` / `50bbeb64c62050961428d30edff396dec84eed9a2f22ff4b5219a41fa92a9d2e` |
| stage 2, v5 pairs | `manifests/stage2/pairs_frozen_v5.json` (2026-10-13T13:32:54Z; 1,379 pairs, rules V 451 / R 928) | `1fe97617f4a6c8104ed6d396a2409074a185f00bc6572f506f569b91ecdf0e0b` | pairs file `0ba77a762ba3585b58c84e9472a76c9c0b5e2a7aa3438875c62688fcb4884fcd` |
| stage 2, filtered-SFT round | `manifests/stage2/stage2_mixture.json` (2026-10-13T00:16:25Z; 1,183 graded rollouts of the v1 model) | `cd402861cb04c664e4e4fa44f04079a98411d8633b11e54d34729ba42ebdc2fe` | in the file's `files` block |

The header of every training run's resolved config, `configs/sft_runs/<run_id>.yaml`, records the sha256 of the exact
rows files it read.

## 8. Manifest layout

```
manifests/
  raw_sources.json, decontam_exclusions.json, dev_*.json     shared by every version
  v1/  mixture, selection, split, screens_p1_pools, screens_p2_rows, DATA_CARD_v1.md,
       pools/{swe_teacher, terminal_teacher, retention}/
  v5/  mixture, selection, split, gate_report, swe_exclusions, screens_v5_*, data_frozen_v5, DATA_CARD_v5.md,
       pools/{swe_v5, retention_v5}/
  v6/  the same for v6r2 (+ the superseded v6r1 files), DATA_CARD_v6.md, DATA_CARD_v6_notes.md, pools/swe_v6/
  v7/  the same for v7r1 + terminal_slice.json, DATA_CARD_v7.md, pools/swe_v7/
  stage2/  stage2_mixture, screens_p2_stage2_rows, stage2_p2_exclusions.txt, stage2_exclude_{swe,terminal} (v5),
           pairs_frozen_v5, pairs_v5_report, pools/{swesmith, tgr_env}/
```

The builders read and write these manifests at the repository's original paths (`data/manifests/*.json` for v1,
`data/manifests/vN/` for later versions); v1's files sit in `manifests/v1/` here.
