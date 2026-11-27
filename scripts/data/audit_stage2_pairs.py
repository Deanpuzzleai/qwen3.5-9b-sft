#!/usr/bin/env python3
"""Hand-audit support for stage-2 pairs (v6 audit form; data plan v5, 4.4).

Run from the repository root:
    python3 scripts/data/audit_stage2_pairs.py sample --pairs WORK/<run>/pairs.jsonl --rules V,R,D,P --seed 61013 \
        --tag s1 --out-dir logs/<run>/audit [--n 30]
    python3 scripts/data/audit_stage2_pairs.py record --sample logs/<run>/audit/audit_sample_s1.json \
        --verdicts V.json --verdicts R.json ... --out logs/<run>/audit_v6.jsonl

sample  per rule, n pairs drawn at random with the fixed seed, stratified by family (largest remainder over the family
        counts); every pair when a rule has fewer than n. Rule P is first split by chosen source: P-teacher pairs (the
        teacher fallback) get max(P_TEACHER_MIN, their proportional share) of the n when available, the on-policy P pairs
        the rest (either side short: the other fills), each part stratified by family. Writes audit_sample_<tag>.json (seed, tag, ids per rule and
        family) and one packet per rule, audit_<rule>_<tag>.md: the rule's definition, the two questions, and per pair the
        last observation of the prompt (head and tail), the last 2 actions, the full chosen turn and the full rejected
        turn (reasoning and action). Never hand-pick; a fix needs a new seed and tag, recorded as such.
record  the auditors' verdict files (JSON lists of {"id", "verdict": "correct" | "wrong", "reason"}) for a sample; every
        sampled id must have exactly one verdict. Appends one JSON line per pair (pair_id, rule, family, verdict, reason,
        seed, tag, utc) to --out and prints the gate per rule (> 10 percent wrong drops the rule).
review  writes the review note (--out, e.g. logs/<run>/pairs_review_v6.md): pairs by
        family x rule, R chosen kinds (same / different action) and the degenerate-rejected share, mean token lengths by
        family x rule, the audit result per rule and sample from the audit JSONL, and the paths.
    python3 scripts/data/audit_stage2_pairs.py review --pairs WORK/<run>/pairs.jsonl --report WORK/<run>/report.json \
        --audit logs/<run>/audit_v6.jsonl --out logs/<run>/pairs_review_v6.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GATE = 0.10
P_TEACHER_MIN = 15                     # P audit: at least this many P-teacher pairs of the n, when available
QUESTIONS = ("(a) Is the rejected turn really faulty under the rule's definition?\n"
             "(b) Is the chosen turn at least as good a next step as the rejected one at this state?\n"
             "The label is WRONG if (a) or (b) is no.")
DEFINITIONS = {
    "V": ("V (valid): the rejected turn cannot be executed by the harness. SWE: no tool call, a non-bash tool, a call "
          "without `command`, a malformed <tool_call> block, or reasoning that never closed. Terminal: the response is not "
          "valid Terminus-2 JSON under Harbor's strict parser, or the reasoning never closed. Chosen: a valid sample within "
          "the band's ceiling that is not a premature completion or a repeat."),
    "R": ("R (no runaway thinking): the rejected turn's reasoning exceeds the band's turn ceiling (SWE easy 1,000 / medium "
          "1,500 / hard 3,000; terminal easy 1,200 / medium 1,800 / hard 2,500 tokens), or it was cut off by the token cap "
          "before acting. Chosen: a clean sample with the SAME action (detail same_action); a DIFFERENT action only when the "
          "rejected turn is degenerate (cut off, reasoning never closed, or repetitive) and above 2x the ceiling (detail "
          "clear_runaway)."),
    "D": ("D-SWE (verify before submit, SWE only): at this state the last assistant action edited code and nothing was "
          "run since. Rejected: the submit command (echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT). Chosen: a clean sample "
          "that runs tests or the program (inline python and heredocs count)."),
    "P": ("P (progress): the rejected turn is a read-only command identical (after whitespace normalisation) to an earlier "
          "command of the episode, with only read-only commands since (no edit, write or run). Chosen: a clean sample "
          "whose action is not such a repeat (a new read, an edit, a test or a run)."),
}


def action_text(m: dict) -> str:
    calls = m.get("tool_calls") or []
    if calls:
        out = []
        for tc in calls:
            args = tc["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            out.append(str(args.get("command", args)))
        return "\n;;\n".join(out)
    return m.get("content") or ""


def apportion(n: int, counts: dict[str, int]) -> dict[str, int]:
    total = sum(counts.values())
    raw = {k: n * v / total for k, v in counts.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:n - sum(out.values())]:
        out[k] += 1
    return out


def clip(text: str, head: int = 3000, tail: int = 1000) -> str:
    return text if len(text) <= head + tail else f"{text[:head]}\n[... {len(text) - head - tail} characters ...]\n{text[-tail:]}"


def packet(rule: str, pairs: list[dict], tag: str, seed: int) -> str:
    lines = [f"# Audit packet: rule {rule}, sample {tag} (seed {seed}, {len(pairs)} pairs)", "", "## Rule definition",
             DEFINITIONS[rule], "", "## Questions per pair", QUESTIONS, ""]
    for p in pairs:
        m, prompt = p["metadata"], p["prompt"]
        last_obs = next((x for x in reversed(prompt) if x.get("role") in ("tool", "user")), {})
        acts = [x for x in prompt if x.get("role") == "assistant"][-2:]
        lines += [f"---\n## `{p['id']}`", f"{m['family']}/{m['band']}, turn {m['turn']}, detail {m.get('detail')}, "
                  f"chosen from {m.get('chosen_source')}, thinking chosen {m['chosen_thinking']} vs rejected {m['rejected_thinking']}",
                  "", "### Last 2 actions before the state"]
        for k, a in enumerate(acts):
            lines += [f"Action -{len(acts) - k}:", "```", clip(action_text(a), 1500, 500), "```"]
        lines += ["### Last observation", "```", clip(str(last_obs.get("content") or "")), "```"]
        for side in ("chosen", "rejected"):
            t = p[side]
            lines += [f"### {side.upper()} turn: reasoning", "```", t.get("reasoning_content") or "", "```",
                      f"### {side.upper()} turn: action", "```", action_text(t), "```"]
    return "\n".join(lines) + "\n"


def cmd_sample(a: argparse.Namespace) -> int:
    by_rule: dict[str, list[dict]] = defaultdict(list)
    with Path(a.pairs).open() as f:
        for line in f:
            p = json.loads(line)
            by_rule[p["metadata"]["rule"]].append(p)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = {"seed": a.seed, "tag": a.tag, "n": a.n, "pairs": str(a.pairs), "rules": {}}
    for rule in a.rules.split(","):
        pool = sorted(by_rule.get(rule, []), key=lambda p: p["id"])
        fam = Counter(p["metadata"]["family"] for p in pool)
        def draw(cells_pool: list[dict], k: int, label: str) -> list[dict]:
            if len(cells_pool) <= k:
                return list(cells_pool)
            fam_ = Counter(p["metadata"]["family"] for p in cells_pool)
            quota = apportion(k, dict(fam_))
            out = []
            for f in sorted(fam_):
                cell = [p for p in cells_pool if p["metadata"]["family"] == f]
                out += random.Random(f"{a.seed}:{a.tag}:{rule}:{label}:{f}").sample(cell, quota[f])
            return out

        if rule == "P":
            teach = [p for p in pool if p["metadata"].get("chosen_source") == "teacher"]
            onp = [p for p in pool if p["metadata"].get("chosen_source") != "teacher"]
            k_t = min(len(teach), max(P_TEACHER_MIN, round(a.n * len(teach) / max(1, len(pool)))))
            k_o = min(len(onp), a.n - k_t)
            k_t = min(len(teach), a.n - k_o)
            picked = draw(teach, k_t, "teacher") + draw(onp, k_o, "on_policy")
        else:
            picked = draw(pool, a.n, "all")
        sample["rules"][rule] = {"available": dict(fam), "ids": [p["id"] for p in picked],
                                 "family": {p["id"]: p["metadata"]["family"] for p in picked},
                                 "chosen_source": {p["id"]: p["metadata"].get("chosen_source") for p in picked}}
        (out_dir / f"audit_{rule}_{a.tag}.md").write_text(packet(rule, picked, a.tag, a.seed))
        print(f"{rule}: {len(picked)} of {len(pool)} pairs ({dict(Counter(p['metadata']['family'] for p in picked))})")
    (out_dir / f"audit_sample_{a.tag}.json").write_text(json.dumps(sample, indent=1) + "\n")
    return 0


def cmd_record(a: argparse.Namespace) -> int:
    sample = json.loads(Path(a.sample).read_text())
    verdicts: dict[str, dict] = {}
    for f in a.verdicts:
        for v in json.loads(Path(f).read_text()):
            if v["id"] in verdicts:
                raise SystemExit(f"two verdicts for {v['id']}")
            if v["verdict"] not in ("correct", "wrong"):
                raise SystemExit(f"bad verdict {v}")
            verdicts[v["id"]] = v
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines, gate = [], {}
    for rule, info in sample["rules"].items():
        missing = [i for i in info["ids"] if i not in verdicts]
        if missing:
            raise SystemExit(f"rule {rule}: no verdict for {missing[:3]}")
        wrong = 0
        for i in info["ids"]:
            v = verdicts[i]
            wrong += v["verdict"] == "wrong"
            lines.append({"pair_id": i, "rule": rule, "family": info["family"][i],
                          "chosen_source": (info.get("chosen_source") or {}).get(i), "verdict": v["verdict"],
                          "reason": v.get("reason", ""), "seed": sample["seed"], "tag": sample["tag"], "utc": utc})
        n = len(info["ids"])
        gate[rule] = {"audited": n, "wrong": wrong, "share": round(wrong / n, 3) if n else None,
                      "decision": "kept" if n and wrong / n <= GATE else ("dropped" if n else "no pairs")}
    extra = set(verdicts) - {i for info in sample["rules"].values() for i in info["ids"]}
    if extra:
        raise SystemExit(f"verdicts for pairs outside the sample: {sorted(extra)[:3]}")
    with Path(a.out).open("a") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(json.dumps(gate, indent=1))
    return 0


def cmd_review(a: argparse.Namespace) -> int:
    pairs = [json.loads(line)["metadata"] for line in Path(a.pairs).open()]
    rep = json.loads(Path(a.report).read_text())
    audit = [json.loads(line) for line in Path(a.audit).open()] if Path(a.audit).exists() else []
    fr = Counter(f"{m['family']}/{m['rule']}" for m in pairs)
    lines = [f"# Stage-2 pairs review (v6), {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "",
             f"Pairs: `{a.pairs}` ({len(pairs)}); report `{a.report}`; audit `{a.audit}`. NOT frozen: waits for an "
             "explicit \"freeze\".", "", "## Pairs by family x rule", "", "| cell | pairs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(fr.items())]
    r = [m for m in pairs if m["rule"] == "R"]
    kinds = Counter(f"{m['family']}/{m['detail']}" for m in r)
    degen = sum(1 for m in r if not m.get("rejected_closed", True) or m.get("rejected_repetitive"))
    lines += ["", "## R: chosen kinds and degenerate rejected turns", "", "| family/kind | pairs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(kinds.items())]
    lines += ["", f"Degenerate rejected (unclosed or repetitive): {degen} of {len(r)} R pairs; unclosed "
              f"{sum(1 for m in r if not m.get('rejected_closed', True))}, loops (unclosed and repetitive) "
              f"{sum(1 for m in r if m.get('rejected_loop'))}."]
    lines += ["", "## Mean tokens per pair (rendered)", "", "| cell | pairs | prompt | chosen | rejected | prompt + longer side |",
              "|---|---|---|---|---|---|"]
    for k, t in sorted((rep.get("tokens_by_family_rule") or {}).items()):
        lines.append(f"| {k} | {t['pairs']} | {t['mean_prompt']} | {t['mean_chosen']} | {t['mean_rejected']} | {t['mean_row']} |")
    lines += ["", "## Audit (v6 form: seeded, family-stratified, 30 per rule or all; > 10 % wrong drops the rule)", "",
              "| rule | sample (seed/tag) | audited | wrong | share | decision |", "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for x in audit:
        by[(x["rule"], x["seed"], x["tag"])].append(x)
    for (rule, seed, tag), xs in sorted(by.items()):
        w = sum(x["verdict"] == "wrong" for x in xs)
        lines.append(f"| {rule} | {seed}/{tag} | {len(xs)} | {w} | {w / len(xs):.1%} | {'kept' if w / len(xs) <= GATE else 'dropped'} |")
    p_audit = [x for x in audit if x["rule"] == "P"]
    if p_audit:
        lines += ["", "P wrong-rates by chosen source (the gate applies to P as a whole):"]
        for src in ("teacher", "candidate"):
            xs = [x for x in p_audit if x.get("chosen_source") == src]
            if xs:
                w = sum(x["verdict"] == "wrong" for x in xs)
                lines.append(f"- P-{'teacher' if src == 'teacher' else 'on-policy'}: {w} wrong of {len(xs)} ({w / len(xs):.1%})")
    pt = sum(1 for m in pairs if m["rule"] == "P" and m.get("chosen_source") == "teacher")
    lines += ["", f"P-teacher pairs: {pt} of {len(pairs)} ({pt / max(1, len(pairs)):.1%}; cap 20 %, cut by the cap: "
              f"{(rep.get('balance') or {}).get('p_teacher_cut')})."]
    lines += ["", f"Validator drops: {rep.get('validator_drops')}. Labels: {rep.get('labels')}"[:2000]]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--pairs", required=True)
    s.add_argument("--rules", default="V,R,D,P")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--out-dir", required=True)
    r = sub.add_parser("record")
    r.add_argument("--sample", required=True)
    r.add_argument("--verdicts", action="append", required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("review")
    v.add_argument("--pairs", required=True)
    v.add_argument("--report", required=True)
    v.add_argument("--audit", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "record": cmd_record, "review": cmd_review}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
