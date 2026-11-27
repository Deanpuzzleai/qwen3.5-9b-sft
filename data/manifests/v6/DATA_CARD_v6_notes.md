# Addendum to the v6 data card (revision v6r2)

The frozen card (`DATA_CARD_v6.md` in this folder) is listed with its sha256 in `data_frozen_v6.json`, so it is
not edited after the freeze. This addendum records what was decided after it was written.

## Known side effect of the targeted SWE trim (accepted 2026-10-13 ~14:00 ET)

The SWE trim drops the episodes with an unverified submit first (no test, program or inline-code run in the 3 turns
before the submit), and only its second step protects the language shares (at most 1 point below the share before the
trim). Unverified submits are not spread evenly across languages: 57 % of the Rust / Java / PHP episodes had one,
against 6 % of the Python episodes. The first step therefore removed 34 of them, and their share fell from 7.3 % to
4.4 % of SWE episodes, while Python rose from 41.5 % to 46.1 %. Go (29.1 -> 28.3 %), TypeScript (17.0 -> 16.2 %) and
JavaScript (5.1 -> 5.0 %) stayed within about 1 point.

Kept as frozen (orchestrator, on Dean's behalf): the removed episodes were mostly hand-ins without testing, which is
what the first step is for, and the main languages of the SWE evaluation stay within about 1 point.
