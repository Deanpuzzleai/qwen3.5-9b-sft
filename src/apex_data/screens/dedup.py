"""Deduplication library (plan 7): exact keys, MinHash-LSH near-duplicates, the Terminus-2 task-text key, fix families,
episode step sequences, boilerplate windows and the train-vs-dev rule. numpy + xxhash only; every random choice is seeded
with `paths.SEED`.

Representation. `tokenize(text)` = NFKC, casefold, `\\w+` runs (punctuation stripped). Shingles are xxhash64 of word
n-grams (5-grams for issue text, terminal task text, patches and QA prompts; docs/DATA_PLAN.md section 7), kept as sorted unique uint64
arrays; MinHash uses 250 seeded permutations, each the splitmix64 bijection of (hash XOR seed_i), see `MinHasher`.
`BoilerplateIndex` removes hashed 50-token rolling windows shared by >= 10 rows from the dedup representation only: callers
keep stored text untouched and read `boilerplate_fraction` to flag rows that are > 50 % boilerplate.

LSH. `bands x rows` is a parameter: the plan's setting is 25 x 10; 50 x 5 is the higher-recall alternative.
`candidate_probability(s, bands, rows)` = 1 - (1 - s^rows)^bands is the theoretical chance that a pair of Jaccard s becomes
a candidate: at s = 0.7 it is 0.51 for 25 x 10 and > 0.999 for 50 x 5, at s = 0.8 it is 0.94 for 25 x 10; candidates are
always verified with the exact Jaccard, so the setting trades recall against the number of pairs verified, never precision.
`print_recall_curve()` prints the curve for both settings.

Rules implemented here (thresholds from plan 7):
  exact            sha256 of the normalised text / canonical patch (index and hunk-header lines removed) /
                   (repo, base_commit, touched-file set) / normalised prompt / normalised Terminus-2 task text
                   -> `exact_text_key`, `patch_key`, `repo_commit_files_key`, `terminal_task_key`
  near-duplicate   Jaccard >= 0.7 (issue text, patches; 5-grams), >= 0.8 (terminal task text; 5-grams), >= 0.8 (QA
                   prompts; 5-grams; exact match only under 20 words)
  terminal task    `terminal_task_text` cuts the task instruction out of a Terminus-2 user message (the text between
                   "Task Description:" and "Current terminal state:"), so the harness template and the per-run terminal
                   screen never count as shared text; `terminal_task_clusters` joins the exact key and MinHash >= 0.8
  fix_family       same-repo patch pairs with Jaccard >= 0.5 -> families, cap 2 kept per family (seeded choice)
  episode          (tool, normalised command) step sets, > 90 % shared -> keep one; identical sequences are exact duplicates
  boilerplate      windows shared by >= 10 rows removed from the representation; fraction > 0.5 flagged
  cross-split      before the dev freeze a dev near-duplicate of a train item leaves dev; after it the train item leaves
Results name ids only. Nothing here deletes anything: callers mark and count.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import xxhash

from apex_data.paths import SEED

NUM_PERM = 250
DEFAULT_BANDS, DEFAULT_ROWS = 25, 10      # plan 7
RECALL_FIX_BANDS, RECALL_FIX_ROWS = 50, 5  # higher-recall alternative; a parameter, never silently applied
ISSUE_NGRAM, ISSUE_THRESHOLD = 5, 0.7
TERMINAL_NGRAM, TERMINAL_THRESHOLD = 5, 0.8
PROMPT_NGRAM, PROMPT_THRESHOLD, PROMPT_EXACT_ONLY_UNDER = 5, 0.8, 20
TASK_DESCRIPTION_MARKER = "Task Description:"            # Harbor Terminus-2 prompt template (terminus-json-plain.txt)
TERMINAL_STATE_MARKER = "Current terminal state:"
FIX_FAMILY_THRESHOLD, FIX_FAMILY_CAP = 0.5, 2
EPISODE_THRESHOLD = 0.9
BOILERPLATE_WINDOW, BOILERPLATE_MIN_ROWS, BOILERPLATE_FLAG_FRACTION = 50, 10, 0.5

_WORD = re.compile(r"\w+")
_PATCH_META = ("index ", "similarity index", "dissimilarity index", "new file mode", "deleted file mode", "old mode", "new mode",
               "rename from", "rename to", "copy from", "copy to", "\\ No newline at end of file")
_HEX7 = re.compile(r"\b[0-9a-f]{7,}\b")
_NUM3 = re.compile(r"\b\d{3,}\b")
_WS = re.compile(r"\s+")


# --- normalisation and keys -----------------------------------------------------------------------------------------
def tokenize(text: str | None) -> list[str]:
    """NFKC, casefold, `\\w+` runs: the shared word tokenizer of the dedup and D3 representations."""
    if not text:
        return []
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def exact_text_key(text: str | None) -> str | None:
    """sha256 of the normalised text (issue text, task statement or prompt); None when there are no words."""
    toks = tokenize(text)
    return hashlib.sha256(" ".join(toks).encode("utf-8")).hexdigest() if toks else None


def terminal_task_text(text: str | None) -> str:
    """The task instruction of a Terminus-2 user message: the text after the first "Task Description:" up to the next
    "Current terminal state:" (TermGrade puts the template in the system message and starts the first user message with
    the marker; TerminalHorizon puts template and task in the first user message). Text without the marker is returned
    whole (a bare instruction, e.g. a Harbor instruction.md)."""
    if not text:
        return ""
    start = text.find(TASK_DESCRIPTION_MARKER)
    if start < 0:
        return text
    body = text[start + len(TASK_DESCRIPTION_MARKER):]
    end = body.find(TERMINAL_STATE_MARKER)
    return body[:end] if end >= 0 else body


def terminal_task_key(text: str | None) -> str | None:
    """Exact key of the normalised Terminus-2 task text (`exact_text_key` of `terminal_task_text`)."""
    return exact_text_key(terminal_task_text(text))


def canonical_patch(diff: str | None) -> str:
    """The diff without hunk headers, index and mode / rename metadata lines, trailing whitespace stripped per line."""
    if not diff:
        return ""
    out = []
    for line in diff.splitlines():
        if line.startswith("@@") or line.startswith(_PATCH_META):
            continue
        out.append(line.rstrip())
    return "\n".join(out).strip()


def patch_key(diff: str | None) -> str | None:
    canon = canonical_patch(diff)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest() if canon else None


def touched_files(diff: str | None) -> list[str]:
    """Paths from `diff --git a/<p> b/<p>` lines (fallback `+++ b/<p>`), sorted and unique."""
    if not diff:
        return []
    files = set(re.findall(r"^diff --git a/(\S+) b/", diff, flags=re.M))
    if not files:
        files = {m for m in re.findall(r"^\+\+\+ b/(\S+)", diff, flags=re.M)}
    return sorted(files)


def repo_commit_files_key(repo: str | None, base_commit: str | None, files: Sequence[str]) -> str | None:
    if not repo or not base_commit or not files:
        return None
    payload = json.dumps([repo.strip().lower(), base_commit.strip().lower(), sorted(set(files))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_groups(keys: Mapping[str, str | None]) -> dict[str, list[str]]:
    """{key: [ids...]} for keys shared by >= 2 ids (ids sorted; the first is the kept representative)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id, key in keys.items():
        if key is not None:
            groups[key].append(item_id)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


# --- shingles and MinHash ----------------------------------------------------------------------------------------------
def shingle_hashes(tokens: Sequence[str], n: int) -> np.ndarray:
    """Sorted unique xxhash64 values of the word n-grams of `tokens` (empty when there are fewer than n tokens)."""
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    hashes = {xxhash.xxh64_intdigest(" ".join(tokens[i:i + n]).encode("utf-8")) for i in range(len(tokens) - n + 1)}
    return np.sort(np.fromiter(hashes, dtype=np.uint64, count=len(hashes)))


def set_hashes(parts: Iterable[str]) -> np.ndarray:
    """Sorted unique xxhash64 values of arbitrary strings (episode steps)."""
    hashes = {xxhash.xxh64_intdigest(p.encode("utf-8")) for p in parts}
    return np.array(sorted(hashes), dtype=np.uint64) if hashes else np.empty(0, dtype=np.uint64)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard of two sorted unique hash arrays."""
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


class MinHasher:
    """250 seeded permutations. Permutation i maps a shingle hash x to `mix64(x XOR seed_i)`, the splitmix64 finaliser,
    a bijection of the 64-bit universe; the signature is the per-permutation minimum. (A linear (a*x + b) mod p family
    on 32-bit values under-estimates agreement by ~3 pp on 300-shingle documents, which costs candidates at the band
    threshold; the mixer is unbiased within 0.3 pp on the synthetic calibration pairs and faster.)"""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64, endpoint=True)

    @staticmethod
    def mix64(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x

    def signature(self, hashes: np.ndarray, block: int = 4096) -> np.ndarray:
        if hashes.size == 0:
            raise ValueError("MinHash of an empty set is undefined; exclude items without shingles")
        h = np.asarray(hashes, dtype=np.uint64)
        sig = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, h.size, block):
            vals = self.mix64(h[None, start:start + block] ^ self.seeds[:, None])
            np.minimum(sig, vals.min(axis=1), out=sig)
        return sig

    def signatures(self, hash_sets: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.signature(h) for h in hash_sets]) if hash_sets else np.empty((0, self.num_perm), dtype=np.uint64)


def _band_keys(band: np.ndarray) -> np.ndarray:
    """Fold the `rows` values of a band into one uint64 key (polynomial under uint64 wrap-around)."""
    keys = np.zeros(band.shape[0], dtype=np.uint64)
    mult = np.uint64(1099511628211)
    for j in range(band.shape[1]):
        keys = keys * mult + band[:, j]
    return keys


def lsh_candidate_pairs(sigs: np.ndarray, bands: int = DEFAULT_BANDS, rows: int = DEFAULT_ROWS, *,
                        blocks: Sequence[Any] | None = None, max_bucket: int = 5000) -> tuple[set[tuple[int, int]], int]:
    """Candidate index pairs (i < j) that agree on at least one band; `blocks` restricts pairs to equal block labels (same
    repo for fix families). Returns (pairs, n_oversized_buckets): a bucket above `max_bucket` members is skipped and
    counted rather than expanded into millions of pairs."""
    if bands * rows > sigs.shape[1]:
        raise ValueError(f"bands x rows = {bands * rows} exceeds the {sigs.shape[1]} permutations")
    pairs: set[tuple[int, int]] = set()
    oversized = 0
    for b in range(bands):
        keys = _band_keys(sigs[:, b * rows:(b + 1) * rows])
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
        for run in np.split(order, bounds):
            if run.size < 2:
                continue
            if run.size > max_bucket:
                oversized += 1
                continue
            for i, j in combinations(sorted(run.tolist()), 2):
                if blocks is None or blocks[i] == blocks[j]:
                    pairs.add((i, j))
    return pairs, oversized


def candidate_probability(s: float, bands: int, rows: int) -> float:
    """P(pair with Jaccard s shares at least one band) = 1 - (1 - s^rows)^bands."""
    return 1.0 - (1.0 - s ** rows) ** bands


def recall_curve(bands: int, rows: int, points: Sequence[float] = tuple(x / 100 for x in range(40, 101, 5))) -> list[tuple[float, float]]:
    return [(s, candidate_probability(s, bands, rows)) for s in points]


def print_recall_curve(settings: Sequence[tuple[int, int]] = ((DEFAULT_BANDS, DEFAULT_ROWS), (RECALL_FIX_BANDS, RECALL_FIX_ROWS)),
                       points: Sequence[float] = tuple(x / 100 for x in range(40, 101, 5))) -> str:
    """Theoretical candidate recall per Jaccard level for each (bands, rows) setting, printed and returned."""
    header = "jaccard  " + "  ".join(f"{b}x{r:<6}" for b, r in settings)
    lines = [header]
    for s in points:
        lines.append(f"{s:7.2f}  " + "  ".join(f"{candidate_probability(s, b, r):8.4f}" for b, r in settings))
    out = "\n".join(lines)
    print(out)
    return out


# --- union-find and groups ----------------------------------------------------------------------------------------------
class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return {k: sorted(v) for k, v in out.items() if len(v) > 1}


@dataclass
class NearDupResult:
    """Verified pairs and their connected components; the kept representative of a group is its smallest id."""
    pairs: list[tuple[str, str, float]] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    n_items: int = 0
    n_hashed: int = 0
    n_candidates: int = 0
    n_oversized_buckets: int = 0

    def drops(self) -> dict[str, str]:
        """{dropped_id: kept_id} for every non-representative member."""
        return {m: rep for rep, members in self.groups.items() for m in members if m != rep}

    def counts(self) -> dict[str, Any]:
        return {"n_items": self.n_items, "n_hashed": self.n_hashed, "n_candidates": self.n_candidates,
                "n_verified_pairs": len(self.pairs), "n_groups": len(self.groups), "n_dropped": len(self.drops()),
                "n_oversized_buckets": self.n_oversized_buckets}


# --- items, representations, boilerplate ------------------------------------------------------------------------------------
@dataclass
class DedupItem:
    id: str
    text: str
    block: str | None = None     # repo for fix families; split for cross-split checks is passed separately
    split: str = "train"


class BoilerplateIndex:
    """Rolling 50-token windows (polynomial hash of per-token xxhash64 values) shared by >= `min_rows` rows."""

    def __init__(self, shared: np.ndarray, window: int = BOILERPLATE_WINDOW, min_rows: int = BOILERPLATE_MIN_ROWS, n_rows: int = 0):
        self.shared = np.sort(np.asarray(shared, dtype=np.uint64))
        self.window, self.min_rows, self.n_rows = window, min_rows, n_rows
        self._pows = self._powers(window)

    @staticmethod
    def _powers(window: int) -> np.ndarray:
        """base^(window-1-j) mod 2^64 for j = 0..window-1 (Python ints, so the intended wrap-around raises no warning)."""
        base, mask = 1099511628211, (1 << 64) - 1
        pows, acc = [0] * window, 1
        for j in range(window - 1, -1, -1):
            pows[j] = acc
            acc = (acc * base) & mask
        return np.array(pows, dtype=np.uint64)

    @staticmethod
    def token_hashes(tokens: Sequence[str]) -> np.ndarray:
        return np.fromiter((xxhash.xxh64_intdigest(t.encode("utf-8")) for t in tokens), dtype=np.uint64, count=len(tokens))

    @classmethod
    def window_hashes(cls, tokens: Sequence[str], window: int = BOILERPLATE_WINDOW, pows: np.ndarray | None = None) -> np.ndarray:
        n = len(tokens)
        if n < window:
            return np.empty(0, dtype=np.uint64)
        th = cls.token_hashes(tokens)
        pows = cls._powers(window) if pows is None else pows
        m = n - window + 1
        acc = np.zeros(m, dtype=np.uint64)
        for j in range(window):
            acc += th[j:j + m] * pows[j]
        return acc

    @classmethod
    def build(cls, token_lists: Iterable[Sequence[str]], window: int = BOILERPLATE_WINDOW, min_rows: int = BOILERPLATE_MIN_ROWS) -> BoilerplateIndex:
        pows = cls._powers(window)
        parts, n_rows = [], 0
        for toks in token_lists:
            n_rows += 1
            wh = cls.window_hashes(toks, window, pows)
            if wh.size:
                parts.append(np.unique(wh))           # one vote per row
        if not parts:
            return cls(np.empty(0, dtype=np.uint64), window, min_rows, n_rows)
        allh = np.concatenate(parts)
        uniq, counts = np.unique(allh, return_counts=True)
        return cls(uniq[counts >= min_rows], window, min_rows, n_rows)

    def strip(self, tokens: Sequence[str]) -> tuple[list[str], float]:
        """(tokens outside shared windows, fraction of tokens covered by shared windows). Stored text is never touched."""
        n = len(tokens)
        if n < self.window or self.shared.size == 0:
            return list(tokens), 0.0
        wh = self.window_hashes(tokens, self.window, self._pows)
        pos = np.searchsorted(self.shared, wh)
        pos[pos >= self.shared.size] = 0
        starts = np.flatnonzero(self.shared[pos] == wh)
        if starts.size == 0:
            return list(tokens), 0.0
        diff = np.zeros(n + 1, dtype=np.int64)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + self.window, -1)
        covered = np.cumsum(diff)[:n] > 0
        return [t for t, c in zip(tokens, covered, strict=True) if not c], float(covered.mean())

    def counts(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "window": self.window, "min_rows": self.min_rows, "n_shared_windows": int(self.shared.size)}


@dataclass
class Representation:
    id: str
    n_tokens: int
    shingles: np.ndarray
    boilerplate_fraction: float = 0.0
    block: str | None = None
    split: str = "train"

    @property
    def boilerplate_flag(self) -> bool:
        return self.boilerplate_fraction > BOILERPLATE_FLAG_FRACTION


def build_representations(items: Sequence[DedupItem], ngram: int, boilerplate: BoilerplateIndex | None = None) -> list[Representation]:
    """Tokenise, optionally strip shared boilerplate windows, shingle. `n_tokens` counts the tokens before stripping."""
    reps = []
    for it in items:
        toks = tokenize(it.text)
        n_tokens = len(toks)
        frac = 0.0
        if boilerplate is not None:
            toks, frac = boilerplate.strip(toks)
        reps.append(Representation(it.id, n_tokens, shingle_hashes(toks, ngram), frac, it.block, it.split))
    return reps


def near_duplicates(reps: Sequence[Representation], threshold: float, *, bands: int = DEFAULT_BANDS, rows: int = DEFAULT_ROWS,
                    hasher: MinHasher | None = None, min_tokens: int = 0, same_block_only: bool = False) -> NearDupResult:
    """MinHash-LSH candidates verified with the exact Jaccard at `threshold`. Items with no shingles or fewer than
    `min_tokens` tokens (prompts under 20 words) are left to the exact pass."""
    hasher = hasher or MinHasher()
    idx = [i for i, r in enumerate(reps) if r.shingles.size > 0 and r.n_tokens >= min_tokens]
    res = NearDupResult(n_items=len(reps), n_hashed=len(idx))
    if len(idx) < 2:
        return res
    sigs = hasher.signatures([reps[i].shingles for i in idx])
    blocks = [reps[i].block for i in idx] if same_block_only else None
    cands, res.n_oversized_buckets = lsh_candidate_pairs(sigs, bands, rows, blocks=blocks)
    res.n_candidates = len(cands)
    uf = _UnionFind()
    for a, b in sorted(cands):
        ra, rb = reps[idx[a]], reps[idx[b]]
        j = jaccard(ra.shingles, rb.shingles)
        if j >= threshold:
            res.pairs.append((ra.id, rb.id, round(j, 4)))
            uf.union(ra.id, rb.id)
    res.groups = uf.groups()
    return res


@dataclass
class Clusters:
    """Exact-key groups and verified near-duplicate pairs joined into clusters; the representative of a cluster is its
    smallest id. `group_of` names the representative of every clustered id (the `dup_group` provenance value)."""
    group_of: dict[str, str] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    n_exact_groups: int = 0
    near: NearDupResult = field(default_factory=NearDupResult)

    def drops(self) -> dict[str, str]:
        """{dropped_id: kept_id} for every non-representative member."""
        return {m: rep for m, rep in self.group_of.items() if m != rep}

    def counts(self) -> dict[str, Any]:
        return {"n_items": self.near.n_items, "n_exact_groups": self.n_exact_groups, **{f"near_{k}": v for k, v in self.near.counts().items()
                if k != "n_items"}, "n_clusters": len(self.groups), "n_clustered": len(self.group_of), "n_dropped": len(self.drops())}


def cluster(items: Sequence[DedupItem], ngram: int, threshold: float, *, key=exact_text_key, bands: int = DEFAULT_BANDS,
            rows: int = DEFAULT_ROWS, hasher: MinHasher | None = None, min_tokens: int = 0,
            boilerplate: BoilerplateIndex | None = None) -> Clusters:
    """Exact key (`key(text)`) plus MinHash-LSH at `threshold` over word `ngram`-grams, joined with union-find."""
    keys = {it.id: key(it.text) for it in items}
    exact = exact_groups(keys)
    near = near_duplicates(build_representations(items, ngram, boilerplate), threshold, bands=bands, rows=rows,
                           hasher=hasher, min_tokens=min_tokens)
    uf = _UnionFind()
    for members in exact.values():
        for m in members[1:]:
            uf.union(members[0], m)
    for a, b, _ in near.pairs:
        uf.union(a, b)
    groups = uf.groups()
    return Clusters({m: rep for rep, members in groups.items() for m in members}, groups, len(exact), near)


def terminal_task_clusters(items: Sequence[DedupItem], *, bands: int = DEFAULT_BANDS, rows: int = DEFAULT_ROWS,
                           hasher: MinHasher | None = None) -> Clusters:
    """Terminal tasks that are the same task (plan 7): exact normalised Terminus-2 task text, or Jaccard >= 0.8 on word
    5-grams of it. `items[i].text` may be the raw first user message; the template and terminal screen are cut away."""
    task_items = [DedupItem(it.id, terminal_task_text(it.text), it.block, it.split) for it in items]
    return cluster(task_items, TERMINAL_NGRAM, TERMINAL_THRESHOLD, bands=bands, rows=rows, hasher=hasher)


def brute_force_pairs(reps: Sequence[Representation], threshold: float) -> list[tuple[str, str, float]]:
    """Exact all-pairs Jaccard (small blocks and calibration samples)."""
    out = []
    for a, b in combinations(reps, 2):
        j = jaccard(a.shingles, b.shingles)
        if j >= threshold:
            out.append((a.id, b.id, round(j, 4)))
    return out


# --- fix families -------------------------------------------------------------------------------------------------------
@dataclass
class FixFamilyResult:
    family_of: dict[str, str] = field(default_factory=dict)     # id -> family representative (smallest id)
    over_cap: list[str] = field(default_factory=list)           # ids beyond the cap in their family
    pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        fams = set(self.family_of.values())
        return {"n_families": len(fams), "n_in_families": len(self.family_of), "n_over_cap": len(self.over_cap), "n_pairs": len(self.pairs)}


def fix_families(reps: Sequence[Representation], threshold: float = FIX_FAMILY_THRESHOLD, cap: int = FIX_FAMILY_CAP, *,
                 bands: int = RECALL_FIX_BANDS, rows: int = RECALL_FIX_ROWS, brute_force_max: int = 300,
                 hasher: MinHasher | None = None, seed: int = SEED) -> FixFamilyResult:
    """Same-repo (block) patch pairs at Jaccard >= 0.5 form families; `cap` members per family are kept (seeded choice
    among sorted ids), the rest are `over_cap`. Blocks up to `brute_force_max` patches are compared exhaustively; larger
    blocks go through LSH at 50 x 5, whose candidate recall at s = 0.5 is 0.80 (25 x 10 would give 0.02)."""
    by_block: dict[str | None, list[Representation]] = defaultdict(list)
    for r in reps:
        if r.shingles.size:
            by_block[r.block].append(r)
    uf = _UnionFind()
    pairs: list[tuple[str, str, float]] = []
    large: list[Representation] = []
    for block, members in by_block.items():
        if block is None or len(members) < 2:
            continue
        if len(members) <= brute_force_max:
            pairs.extend(brute_force_pairs(members, threshold))
        else:
            large.extend(members)
    if len(large) >= 2:
        res = near_duplicates(large, threshold, bands=bands, rows=rows, hasher=hasher, same_block_only=True)
        pairs.extend(res.pairs)
    for a, b, _ in pairs:
        uf.union(a, b)
    out = FixFamilyResult(pairs=sorted(pairs))
    rng = np.random.default_rng(seed)
    for rep, members in sorted(uf.groups().items()):
        for m in members:
            out.family_of[m] = rep
        keep = set(np.array(members)[rng.permutation(len(members))[:cap]].tolist())
        out.over_cap.extend(m for m in members if m not in keep)
    out.over_cap.sort()
    return out


# --- episodes -----------------------------------------------------------------------------------------------------------
def normalise_command(cmd: str) -> str:
    """Lowercase, whitespace-collapsed, hex ids >= 7 chars and numbers >= 3 digits masked, cut at 200 characters."""
    s = _WS.sub(" ", cmd.strip().lower())
    s = _NUM3.sub("<n>", s)          # before the hex rule: a digit run is a number, not a hash
    s = _HEX7.sub("<hex>", s)
    return s[:200]


def _arguments(call: Mapping[str, Any]) -> Any:
    fn = call.get("function") or {}
    args = fn.get("arguments", call.get("arguments"))
    if isinstance(args, str):
        try:
            return json.loads(args)
        except ValueError:
            return args
    return args


def episode_steps(episode: Mapping[str, Any]) -> list[str]:
    """(tool, normalised command) per assistant action: function tool calls (`command` / `cmd` argument, else the
    sorted-key JSON of the arguments) and Terminus-style JSON content with `commands: [{keystrokes}]`."""
    steps: list[str] = []
    for msg in episode.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        for call in calls:
            name = (call.get("function") or {}).get("name") or call.get("name") or "tool"
            args = _arguments(call)
            if isinstance(args, Mapping):
                cmd = args.get("command") or args.get("cmd") or json.dumps(args, sort_keys=True, ensure_ascii=False)
            else:
                cmd = str(args)
            steps.append(f"{name}\x1f{normalise_command(str(cmd))}")
        if calls:
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.lstrip().startswith("{"):
            try:
                obj = json.loads(content)
            except ValueError:
                continue
            for c in obj.get("commands") or []:
                if isinstance(c, Mapping) and c.get("keystrokes") is not None:
                    steps.append(f"keystrokes\x1f{normalise_command(str(c['keystrokes']))}")
    return steps


def episode_sequence_key(steps: Sequence[str]) -> str | None:
    return hashlib.sha256("\n".join(steps).encode("utf-8")).hexdigest() if steps else None


def episode_duplicates(episodes: Sequence[Mapping[str, Any]], threshold: float = EPISODE_THRESHOLD, *,
                       bands: int = DEFAULT_BANDS, rows: int = DEFAULT_ROWS, hasher: MinHasher | None = None,
                       same_task_only: bool = True) -> NearDupResult:
    """Episodes whose step sets share more than `threshold` (Jaccard) collapse to one, and identical ordered sequences
    are exact duplicates even when the step set is tiny. By default only within a task (`task_id`), which is where
    near-identical solutions over-weight one answer."""
    steps_by_id: dict[str, list[str]] = {}
    reps: list[Representation] = []
    for ep in episodes:
        eid, steps = str(ep.get("episode_id")), episode_steps(ep)
        steps_by_id[eid] = steps
        block = str(ep.get("task_id")) if same_task_only else None
        reps.append(Representation(eid, len(steps), set_hashes(steps), 0.0, block, "train"))
    res = near_duplicates(reps, threshold, bands=bands, rows=rows, hasher=hasher, same_block_only=same_task_only)
    uf = _UnionFind()
    for a, b, _ in res.pairs:
        uf.union(a, b)
    block_of = {r.id: r.block for r in reps}
    for members in exact_groups({eid: episode_sequence_key(st) for eid, st in steps_by_id.items()}).values():
        by_block: dict[str | None, list[str]] = defaultdict(list)
        for m in members:
            by_block[block_of[m]].append(m)
        for group in by_block.values():
            for m in group[1:]:
                uf.union(group[0], m)
    res.groups = uf.groups()
    return res


# --- cross-split rule ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossSplitDrop:
    id: str
    split: str          # the split the item leaves
    matched_id: str     # the item on the other side it duplicates
    rule: str           # exact | near


def resolve_cross_split(groups: Iterable[Sequence[str]], split_of: Mapping[str, str], dev_frozen: bool, rule: str) -> list[CrossSplitDrop]:
    """For every group that spans train and dev: before the freeze the dev members leave dev; after it the train members
    leave train (the dev set is immutable). Groups inside one split are not this rule's business."""
    drops: list[CrossSplitDrop] = []
    for members in groups:
        train = sorted(m for m in members if split_of.get(m, "train") != "dev")
        dev = sorted(m for m in members if split_of.get(m, "train") == "dev")
        if not train or not dev:
            continue
        if dev_frozen:
            drops.extend(CrossSplitDrop(m, "train", dev[0], rule) for m in train)
        else:
            drops.extend(CrossSplitDrop(m, "dev", train[0], rule) for m in dev)
    return drops


# --- synthetic calibration ------------------------------------------------------------------------------------------------------
def synthetic_pairs(n_pairs: int, target_jaccard: tuple[float, float], *, n_tokens: int = 300, ngram: int = ISSUE_NGRAM,
                    vocab: int = 5000, seed: int = SEED) -> tuple[list[DedupItem], list[tuple[str, str]]]:
    """Pairs of random-word documents whose exact shingle Jaccard falls inside `target_jaccard` (resampled until it does),
    for measuring candidate recall at a band setting. Word replacement at random positions gives the spread."""
    rng = np.random.default_rng(seed)
    items, pairs = [], []
    lo, hi = target_jaccard
    k = 0
    while len(pairs) < n_pairs:
        base = [f"w{v}" for v in rng.integers(0, vocab, size=n_tokens)]
        # replacing a fraction f of the words removes about ngram*f of the shingles on each side
        frac = rng.uniform(0.0, 0.4)
        other = list(base)
        for pos in rng.choice(n_tokens, size=int(frac * n_tokens), replace=False):
            other[pos] = f"x{rng.integers(0, vocab)}"
        j = jaccard(shingle_hashes(base, ngram), shingle_hashes(other, ngram))
        if not (lo <= j < hi):
            continue
        a, b = f"p{k}a", f"p{k}b"
        items += [DedupItem(a, " ".join(base)), DedupItem(b, " ".join(other))]
        pairs.append((a, b))
        k += 1
    return items, pairs


def measure_candidate_recall(items: Sequence[DedupItem], pairs: Sequence[tuple[str, str]], bands: int, rows: int, *,
                             ngram: int = ISSUE_NGRAM, hasher: MinHasher | None = None) -> float:
    """Fraction of the given pairs that LSH proposes as candidates (before Jaccard verification)."""
    hasher = hasher or MinHasher()
    reps = build_representations(items, ngram)
    pos = {r.id: i for i, r in enumerate(reps)}
    sigs = hasher.signatures([r.shingles for r in reps])
    cands, _ = lsh_candidate_pairs(sigs, bands, rows)
    found = sum(1 for a, b in pairs if (min(pos[a], pos[b]), max(pos[a], pos[b])) in cands)
    return found / len(pairs) if pairs else float("nan")
