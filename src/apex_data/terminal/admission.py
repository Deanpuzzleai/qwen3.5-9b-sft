"""Per-episode screens and the admission pilot of the terminal corpora (docs/DATA_PLAN.md 3.3, 6 and 7).

Screens (each returns ids and counts only):
  d1          the canary GUID / sentence / phrase: whole-file scan of the raw dump (screens.d1_canary) plus the episode's
              task text (screens.d4_structural.screen_text); any hit drops the episode
  d4          the 89 public Terminal-Bench 2.0 task names as whole tokens over the task name and task text, and the
              Terminal-Bench / Harbor / laude registry identifiers (screens.d4_structural); a hit drops the episode
  identity    teacher self-mention: an assistant turn (reasoning or content) names an AI model or vendor that no user
              turn of the episode names ("as ChatGPT, we ...", "I'm Claude"); a hit drops the episode
  credentials secret-shaped strings in any message: private-key blocks, AWS / GitHub / OpenAI / Anthropic / Hugging Face /
              Slack / Google / Stripe key formats; documented placeholder values (AWS's AKIAIOSFODNN7EXAMPLE) are not
              hits; a hit drops the episode. E-mail addresses are counted, not dropped (fixture data of the synthetic
              environments).
  d3          the sealed inputs-only screen runs later on the pool and the rows (WP7, M2.4): recorded as "pending".

Admission thresholds (plan 3.3): >= 90 percent of the episodes parse-clean (every assistant turn parses with Harbor's
strict JSON parser and no user turn is a parse-error message), zero D1 and D3 hits after review, >= 30 distinct task
families among the kept episodes (TerminalHorizon: environment subcategories; TermGrade, which has no category
metadata: near-duplicate task clusters of screens.dedup). The handshake share is reported next to them.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from apex_data.screens import d4_structural

PARSE_CLEAN_MIN = 0.90
FAMILIES_MIN = 30
D1_MAX = 0
D3_MAX = 0

IDENTITY_NAMES = re.compile(r"(?<![a-z0-9])(chatgpt|openai|claude|anthropic|deepseek|gpt-?[345](?:\.\d)?(?![a-z0-9])|"
                            r"gemini|kimi|qwen|llama|mistral)", re.I)
CREDENTIAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
    "aws_access_key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"),
    "github_token": re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})"),
    "openai_key": re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "hf_token": re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34,}(?![A-Za-z0-9])"),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "stripe_live_key": re.compile(r"(?:sk|rk)_live_[0-9A-Za-z]{20,}"),
}
PLACEHOLDER_SECRETS = frozenset({"AKIAIOSFODNN7EXAMPLE"})     # AWS documentation example key
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _texts(messages: Iterable[dict[str, Any]], role: str | None = None) -> Iterable[str]:
    for m in messages:
        if role is None or m.get("role") == role:
            yield m.get("content") or ""
            if m.get("reasoning_content"):
                yield m["reasoning_content"]


def identity_hits(messages: list[dict[str, Any]]) -> list[str]:
    """Model or vendor names that an assistant turn uses and no user turn mentions (lowercased, sorted)."""
    said = {m.group(1).lower() for t in _texts(messages, "assistant") for m in IDENTITY_NAMES.finditer(t)}
    if not said:
        return []
    given = {m.group(1).lower() for t in _texts(messages, "user") for m in IDENTITY_NAMES.finditer(t)}
    return sorted(said - given)


def credential_hits(messages: list[dict[str, Any]]) -> list[str]:
    """Credential patterns found anywhere in the episode (pattern names, sorted; placeholders ignored)."""
    found = set()
    for t in _texts(messages):
        for name, pat in CREDENTIAL_PATTERNS.items():
            if any(m.group(0) not in PLACEHOLDER_SECRETS for m in pat.finditer(t)):
                found.add(name)
    return sorted(found)


def email_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(EMAIL.findall(t)) for t in _texts(messages))


def screen_episode(messages: list[dict[str, Any]], task_text: str, task_name: str | None,
                   slug_patterns: list[tuple[str, re.Pattern[str]]]) -> dict[str, Any]:
    """D1 (task text), D4 (task name and text), identity and credential screens of one loaded episode."""
    r = d4_structural.screen_text(task_text, slug_patterns, task_name=task_name, label="task_text")
    return {"d1": r.d1_hit, "d4": r.d4_hit, "d4_reasons": [x for x in r.reasons() if x.startswith("d4:")],
            "d1_reasons": [x for x in r.reasons() if x.startswith("d1:")],
            "tb_name_mention": bool(r.tb_name_mention_files), "identity": identity_hits(messages),
            "credentials": credential_hits(messages), "emails": email_count(messages)}


def screen_drop_reason(s: dict[str, Any], whole_file_d1: bool) -> str | None:
    if s["d1"] or whole_file_d1:
        return "d1_canary"
    if s["d4"]:
        return "d4_structural"
    if s["identity"]:
        return "identity_self_mention"
    if s["credentials"]:
        return "credential_pattern"
    return None


def decide(source: str, *, n_episodes: int, parse_clean: int, handshake: int, d1_hits_after_review: int,
           d3: str, families: int, family_definition: str, gated: bool = True) -> dict[str, Any]:
    """Threshold table and the pass / fail decision. D3 is "pending" until M2.4; the decision states it is
    conditional on D3 then. `gated=False` (TermGrade: the verified source, whose episodes with an unparsed turn are
    dropped individually) reports the same table without gating the source."""
    share = parse_clean / n_episodes if n_episodes else 0.0
    checks = {
        "parse_clean_share": {"value": round(share, 4), "threshold": f">= {PARSE_CLEAN_MIN}", "pass": share >= PARSE_CLEAN_MIN},
        "d1_hits_after_review": {"value": d1_hits_after_review, "threshold": f"== {D1_MAX}", "pass": d1_hits_after_review <= D1_MAX},
        "d3_hits_after_review": {"value": d3, "threshold": f"== {D3_MAX}", "pass": None if d3 == "pending" else d3 == 0},
        "task_families_kept": {"value": families, "threshold": f">= {FAMILIES_MIN}", "pass": families >= FAMILIES_MIN,
                               "definition": family_definition},
    }
    hard = [c["pass"] for c in checks.values() if c["pass"] is not None]
    passed = all(hard)
    verdict = ("pass" if passed else "fail") + (" (conditional on D3 at M2.4)" if checks["d3_hits_after_review"]["pass"] is None else "")
    out = {"source": source, "checks": checks, "thresholds_met": passed,
           "handshake_share": round(handshake / n_episodes, 4) if n_episodes else 0.0, "gated": gated}
    if gated:
        return {**out, "decision": verdict, "admitted": passed}
    return {**out, "decision": f"not gated (verified source; thresholds {verdict}; episodes failing a check are dropped "
                               "one by one)", "admitted": True}


def screen_counts(screens: Iterable[dict[str, Any]]) -> dict[str, Any]:
    c: Counter = Counter()
    ident: Counter = Counter()
    cred: Counter = Counter()
    for s in screens:
        c["episodes"] += 1
        c["d1"] += bool(s["d1"])
        c["d4"] += bool(s["d4"])
        c["tb_name_mention_soft_flag"] += bool(s["tb_name_mention"])
        c["identity"] += bool(s["identity"])
        c["credentials"] += bool(s["credentials"])
        c["episodes_with_emails"] += bool(s["emails"])
        ident.update(s["identity"])
        cred.update(s["credentials"])
    return {**dict(c), "identity_names": dict(ident.most_common()), "credential_patterns": dict(cred.most_common())}
