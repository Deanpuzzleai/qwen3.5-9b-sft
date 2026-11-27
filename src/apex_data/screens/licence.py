"""Per-repository licence screen (plan 6 task level, plan 7 allowlist) and the GitHub repository lookup shared with D2.

Allowlist: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, MIT-0, CC0-1.0, Unlicense, MPL-2.0. Anything else
(NOASSERTION / custom, GPL / AGPL / LGPL, BSL, null) is excluded. Open-SWE-Traces rows carry the per-row SPDX in their
`license` column and need no lookup (`classify(row["license"])`, or `annotate_licence(..., from_column=True)` on Tasks);
SWE-smith repositories are resolved with one call per distinct repository to GET https://api.github.com/repos/<owner>/<name>,
whose answer also carries the fork / parent / source fields used by the D2 fork-graph check. The token comes from
GITHUB_TOKEN in the environment or in .env and is never printed or stored; without it only the cache is consulted,
unresolved repositories are marked "pending" and the build continues after one notice line. Cache: one JSON file per
repository under WORK/licence_cache/ (definitive answers only: found / not found).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from apex_data import paths
from apex_data.pools.common import Task, exclude, repo_key
from apex_eval.common import utc_now

ALLOWLIST = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT-0", "CC0-1.0", "Unlicense",
                       "MPL-2.0"})
# SWE-smith repositories whose licences failed the allowlist when checked through the GitHub API on 2026-10-11; consulted
# only when neither the cache nor the API answered, so an offline build still excludes them.
OFFLINE_KNOWN = {"gitbookio/gitbook": "GPL-3.0", "directus/directus": "NOASSERTION", "strapi/strapi": "NOASSERTION"}
DEFAULT_CACHE_DIR = paths.WORK / "licence_cache"
API_URL = "https://api.github.com/repos/{repo}"
API_VERSION = "2022-11-28"
NOTICE_NO_TOKEN = ("NOTICE: GITHUB_TOKEN is not set (environment or .env): licence and fork-graph lookups use the cache "
                   "only; unresolved repositories are recorded as 'pending' and kept")

Fetcher = Callable[[str, dict[str, str]], tuple[int, Any, dict[str, str]]]


def github_token() -> str | None:
    """GITHUB_TOKEN from the environment, else from the repository's .env; never logged."""
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    env = paths.REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            line = line[len("export "):] if line.startswith("export ") else line
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def classify(spdx: str | None) -> str:
    return "ok" if spdx in ALLOWLIST else "excluded"


@dataclass
class RepoInfo:
    repo: str                      # lowercase owner/name as queried
    status: str                    # found | not_found | error | rate_limited
    full_name: str | None = None   # canonical name from the API (differs after a rename or transfer)
    spdx: str | None = None
    fork: bool | None = None
    parent: str | None = None
    source: str | None = None
    http_status: int | None = None
    fetched_at: str = ""
    detail: str | None = None

    @property
    def definitive(self) -> bool:
        return self.status in ("found", "not_found")


def urllib_fetch(url: str, headers: dict[str, str], timeout: float = 30.0) -> tuple[int, Any, dict[str, str]]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            body = None
        return e.code, body, {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return 0, {"message": str(e)}, {}


class RepoLookup:
    """Cache-first repository metadata: `lookup` answers from WORK/licence_cache/ and queries the API only for the rest,
    only when a token is present and the API is enabled. A rate-limit answer stops further calls for this run."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, token: str | None = None, fetcher: Fetcher | None = None,
                 workers: int = 4, enabled: bool = True, retries: int = 3, max_rps: float = 8.0):
        self.cache_dir = Path(cache_dir)
        self.token = token
        self.fetcher = fetcher or urllib_fetch
        self.workers = max(1, workers)
        self.enabled = enabled and bool(token)
        self.retries = retries
        self._stop = threading.Event()
        self._min_interval = 1.0 / max_rps if max_rps > 0 else 0.0   # GitHub's secondary limit is 900 points/min
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            time.sleep(wait)

    def _path(self, repo: str) -> Path:
        return self.cache_dir / (repo.replace("/", "__") + ".json")

    def cached(self, repo: str) -> RepoInfo | None:
        p = self._path(repo)
        if not p.exists():
            return None
        try:
            return RepoInfo(**json.loads(p.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    def _store(self, info: RepoInfo) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(info.repo).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(info), indent=1))
        os.replace(tmp, self._path(info.repo))

    def fetch(self, repo: str) -> RepoInfo:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": API_VERSION,
                   "User-Agent": "apex-data-screens", "Authorization": f"Bearer {self.token}"}
        for attempt in range(self.retries):
            if self._stop.is_set():
                return RepoInfo(repo, "rate_limited", detail="not queried: the primary rate limit was exhausted earlier in this run")
            self._throttle()
            status, body, hdr = self.fetcher(API_URL.format(repo=repo), headers)
            now = utc_now()
            body = body if isinstance(body, dict) else {}
            if status == 200:
                lic = body.get("license") or {}
                return RepoInfo(repo, "found", full_name=body.get("full_name"), spdx=lic.get("spdx_id"),
                                fork=body.get("fork"), parent=(body.get("parent") or {}).get("full_name"),
                                source=(body.get("source") or {}).get("full_name"), http_status=200, fetched_at=now)
            if status == 404:
                return RepoInfo(repo, "not_found", http_status=404, fetched_at=now)
            message = str(body.get("message", ""))
            if status in (403, 429) and hdr.get("x-ratelimit-remaining") == "0":      # primary quota: stop this run
                self._stop.set()
                return RepoInfo(repo, "rate_limited", http_status=status, fetched_at=now,
                                detail=f"primary rate limit exhausted; reset={hdr.get('x-ratelimit-reset')}")
            if status in (403, 429) and (hdr.get("retry-after") or "rate limit" in message.lower()):   # secondary: back off
                if attempt + 1 < self.retries:
                    time.sleep(min(float(hdr.get("retry-after") or 60), 120))
                    continue
                return RepoInfo(repo, "rate_limited", http_status=status, fetched_at=now,
                                detail="secondary rate limit persisted after retries")
            if status in (0, 500, 502, 503, 504) and attempt + 1 < self.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return RepoInfo(repo, "error", http_status=status, fetched_at=now, detail=message[:200])
        return RepoInfo(repo, "error", detail="retries exhausted")

    def lookup(self, repos: Iterable[str | None], log: Callable[[str], Any] | None = None
               ) -> tuple[dict[str, RepoInfo], dict[str, Any]]:
        repos = sorted({r for r in repos if r})
        infos: dict[str, RepoInfo] = {}
        missing: list[str] = []
        for r in repos:
            c = self.cached(r)
            if c is not None:
                infos[r] = c
            else:
                missing.append(r)
        summary: dict[str, Any] = {"requested": len(repos), "from_cache": len(infos), "fetched": 0, "found": 0,
                                   "not_found": 0, "errors": 0, "rate_limited": 0, "unresolved": len(missing),
                                   "token_present": bool(self.token), "api_enabled": self.enabled,
                                   "cache_dir": str(self.cache_dir)}
        if missing and self.enabled:
            if log:
                log(f"repository lookup: {len(missing)} of {len(repos)} repositories not cached; querying the GitHub API "
                    f"with {self.workers} workers")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                for info in ex.map(self.fetch, missing):
                    if info.definitive:
                        self._store(info)
                        infos[info.repo] = info
                        summary["fetched"] += 1
                        summary[info.status] += 1
                    elif info.status == "rate_limited":
                        summary["rate_limited"] += 1
                    else:
                        summary["errors"] += 1
            summary["unresolved"] = len(repos) - len(infos)
            if log:
                log(f"repository lookup: fetched {summary['fetched']} (not found {summary['not_found']}), errors "
                    f"{summary['errors']}, rate-limited {summary['rate_limited']}, unresolved {summary['unresolved']}")
        elif missing and log:
            log(f"repository lookup: {len(missing)} repositories not cached and the API is "
                f"{'disabled' if self.token else 'unavailable without a token'}; they stay pending")
        return infos, summary


def annotate_licence(tasks: Iterable[Task], infos: dict[str, RepoInfo] | None = None, *,
                     from_column: bool = False) -> Counter:
    """Set task.licence = {dataset, repo_spdx, status ok|pending|excluded, method} on every task; returns status counts.
    `from_column`: the dataset's own per-row SPDX (Open-SWE-Traces `license`). Otherwise: the GitHub answer, a not-found
    repository (cannot be verified: excluded), the offline table, else pending. Excluded tasks are marked by the caller."""
    counts: Counter = Counter()
    for t in tasks:
        if from_column:
            spdx = t.licence.get("repo_spdx")
            status, method = classify(spdx), "dataset_column"
        else:
            key = repo_key(t.repo) or ""
            info = (infos or {}).get(key)
            if info is not None and info.status == "found":
                spdx, status, method = info.spdx, classify(info.spdx), "github_api"
            elif info is not None and info.status == "not_found":
                spdx, status, method = None, "excluded", "github_api:not_found"
            elif key in OFFLINE_KNOWN:
                spdx, status, method = OFFLINE_KNOWN[key], classify(OFFLINE_KNOWN[key]), "plan_table"
            else:
                spdx, status, method = None, "pending", None
        t.licence.update({"repo_spdx": spdx, "status": status, "method": method})
        counts[status] += 1
    return counts


def licence_reason(t: Task) -> str | None:
    """Survival-rule predicate: excluded tasks name their method and SPDX; ok and pending tasks pass."""
    if t.licence.get("status") != "excluded":
        return None
    return f"{t.licence.get('method')}:{t.licence.get('repo_spdx')}"


__all__ = ["ALLOWLIST", "OFFLINE_KNOWN", "DEFAULT_CACHE_DIR", "NOTICE_NO_TOKEN", "RepoInfo", "RepoLookup",
           "annotate_licence", "classify", "exclude", "github_token", "licence_reason", "urllib_fetch"]
