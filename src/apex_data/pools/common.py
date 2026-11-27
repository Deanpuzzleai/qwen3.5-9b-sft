"""Helpers shared by the pool builders and the screens (revision 4.3 keeps what `screens.licence` and
`screens.d2_repos` import; the revision-3 loaders and draws for dropped pools are not restored).

Nothing here touches the network or deletes anything: `exclude` marks a task and records the rule (plan 2.7).
"""
from __future__ import annotations

import re

from apex_data.schema import Task, sha256_text

__all__ = ["Task", "exclude", "repo_key", "sha256_hex"]

_GITHUB_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/")


def repo_key(repo: str | None) -> str | None:
    """Canonical lowercase 'owner/name' (GitHub names are case-insensitive); None when the value is not of that form."""
    if not repo:
        return None
    r = _GITHUB_PREFIX.sub("", repo.strip()).strip("/")
    r = r[:-4] if r.endswith(".git") else r
    return r.lower() if r.count("/") == 1 and all(r.split("/")) else None


def sha256_hex(text: str) -> str:
    return sha256_text(text)


def exclude(task: Task, rule: str, detail: str | None = None) -> None:
    """Mark a task as excluded with the rule and detail; the record stays where it is."""
    task.split = "excluded"
    task.extra["exclusion"] = {"rule": rule, "detail": detail}
