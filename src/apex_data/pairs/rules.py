"""Labels of one-turn candidates and the pair rules V, R, D, P, E (data plan v5, section 4.4).

Labels (`label_turn`, the same for a sampled candidate and for the teacher's turn; detectors from apex_data.efficiency,
the code of the stage-1 gates):
  valid        the turn parses (candidates.parse_completion: closed reasoning, bash calls with a `command` as the harness
               requires / a strict Terminus-2 JSON response) and its reasoning is non-empty
  thinking     reasoning tokens (Qwen3.5-9B tokenizer)
  action       efficiency.action_key: the normalised action ("" = none)
  premature    rule D: efficiency.unverified_completion (completes while the last change is unverified); never on a
               Terminus-2 confirmation state (the harness asks to confirm task_complete; that reply is not premature)
  repeat       rule P: efficiency.repeats_without_change (empty action, or a read-only repeat with no change since)
  verifies     a verifying segment in the turn's own commands (efficiency.verification: run / inline, terminal also read)
  clean        valid, thinking within the band's turn ceiling (efficiency.LIMITS), not premature, not a repeat

Rules in priority order; each rule makes at most one pair per state, except R (up to R_MAX_PAIRS_PER_STATE pairs from
distinct runaway samples, each with a distinct chosen side); the other rules add a pair only while the state has fewer
than MAX_PAIRS_PER_STATE pairs; no candidate is rejected twice. Chosen sides are always clean; when no candidate qualifies, the teacher's turn at the state is
the chosen side if it qualifies under the same checks (the expert-vs-student fallback).
  V  rejected: a finished candidate that is not valid (no tool call, unparsable, not a bash command, reasoning never
     closed);
     chosen: a clean candidate
  R  rejected: a valid candidate whose thinking is above the ceiling, or one cut off by max_tokens before closing its
     reasoning (the state's budget is at least RUNAWAY_FACTOR x the ceiling, so a cut is a clear runaway); chosen: a clean
     candidate with the same action; if none, any clean candidate, but only when the rejected one is a clear runaway
     (above RUNAWAY_FACTOR x the ceiling, or cut off). With `degenerate_fallback_only` (v6) that fallback
     to a different action also needs a degenerate rejected turn: reasoning never closed (cut at the cap or stopped), or
     `repetitive` (labels set by the caller: the stage-1 repetition gate on the reasoning); a long but meaningful
     rejected turn then pairs only with a same-action chosen side
  D  rejected: a valid candidate within the ceiling that completes prematurely; chosen: a clean candidate that verifies
  P  rejected: a valid candidate within the ceiling, not premature, that repeats without change; chosen: a clean candidate
  E  easy and medium states only; rejected and chosen share the action and are both clean: chosen has the fewest thinking
     tokens of its action group (>= E_MIN_CHOSEN), rejected >= E_RATIO x and >= E_MIN_DIFF more thinking tokens
     (Kimi k1.5's length pairs); no teacher fallback (the teacher's action differs)
v6 rules (`rules_v6`, both audit-gated), priority V > R > D-SWE > P:
  D-SWE  SWE states only, where the last assistant turn edits code and does not run anything (render.swe_masks.classify,
         the S2 classifier: kind `edit`, runs False). Rejected: a valid submit within the ceiling (classify kind
         `submit`); chosen: a clean sample that runs tests or the program (classify runs True). The terminal D detector
         stays off. No teacher fallback.
  P      rejected: a valid read-only command within the ceiling, identical after normalisation to an earlier command of
         the episode, with only read-only commands since and no background job (efficiency.repeats_without_change, its
         empty-action case excluded: a terminal wait is not a repeat); chosen: a clean sample (so not such a repeat).
         With `p_teacher_fallback`: when EVERY valid sample is such a repeat, the teacher's
         recorded turn at the state is the chosen side (teacher states only: rollout states carry no teacher turn), if it
         is clean (valid, closed non-empty reasoning within the ceiling, not premature, not a repeat), not itself a P
         repeat and not a completion (submit / task_complete); otherwise no pair. Set level: P-teacher pairs at most
         P_TEACHER_MAX_SHARE of all pairs (`balance`).
Set level (`balance`): rule E at most E_MAX_SHARE of the pairs; the family cut is optional (see balance).
Random choices use a per-state seeded generator, so a rebuild picks the same pairs.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from apex_data import efficiency
from apex_data.render import swe_masks

RULES = ("V", "R", "D", "P", "E")
E_BANDS = ("easy", "medium")
E_RATIO, E_MIN_DIFF, E_MIN_CHOSEN = 1.5, 200, 20
RUNAWAY_FACTOR = 2
MAX_PAIRS_PER_STATE = 2
R_MAX_PAIRS_PER_STATE = 3             # rule R alone may add up to 3 pairs (P4a), each with distinct sides
E_MAX_SHARE = 0.40                    # rule E: at most 40 percent of the pairs
P_TEACHER_MAX_SHARE = 0.20            # v6: P pairs with the teacher's turn as chosen, at most 20 percent of all pairs
FAMILY_SHARES = {"swe": 0.60, "terminal": 0.40}


def ceiling(state: Mapping[str, Any]) -> int:
    """The state's band turn ceiling in reasoning tokens (data plan 1.3)."""
    return efficiency.LIMITS[state["family"]][state["band"]].turn_ceiling


def label_turn(state: Mapping[str, Any], parsed: Mapping[str, Any], count: Callable[[list[str]], list[int]]) -> dict[str, Any]:
    """Labels of one parsed turn (candidates.parse_completion output, or the teacher's turn) at the state."""
    fam, msg = state["family"], parsed.get("message")
    reasoning = parsed.get("reasoning") or ""
    out = {"valid": bool(parsed.get("valid")) and bool(reasoning.strip()), "invalid_reason": parsed.get("invalid_reason"),
           "finished": bool(parsed.get("finished", True)), "closed": bool(parsed.get("closed", True)),
           "thinking": count([reasoning.strip()])[0] if reasoning.strip() else 0,
           "action": "", "completion": False, "premature": False, "repeat": False, "verifies": False}
    if out["valid"] and msg is not None:
        history = state["prompt"]
        out["action"] = efficiency.action_key(msg, fam)
        out["completion"] = efficiency.is_completion(msg, fam)
        out["premature"] = (not state.get("confirmation")
                            and efficiency.unverified_completion(history, msg, fam, root=efficiency.HARNESS_REPO_ROOT))
        out["repeat"] = efficiency.repeats_without_change(history, msg, fam)
        acts = efficiency.actions([msg], fam)
        out["verifies"] = efficiency.verification(acts, fam, root=efficiency.HARNESS_REPO_ROOT)["last_verify_turn"] is not None
    if out["valid"] and msg is not None:
        out["p_repeat"] = bool(out["action"]) and out["repeat"]       # the v6 P rule: a repeat that is not an empty action
        if fam == "swe":
            facts = swe_masks.classify([efficiency.command_of(tc) for tc in msg.get("tool_calls") or []])
            out["swe_kind"], out["swe_runs"] = facts.kind, facts.runs
    out["clean"] = out["valid"] and out["thinking"] <= ceiling(state) and not out["premature"] and not out["repeat"]
    return out


def last_turn_unverified_edit(prompt: Sequence[Mapping[str, Any]]) -> bool:
    """D-SWE state: the last assistant turn of the history edits code and runs nothing (swe_masks.classify)."""
    last = next((m for m in reversed(prompt) if m.get("role") == "assistant"), None)
    cmds = [efficiency.command_of(tc) for tc in (last or {}).get("tool_calls") or []]
    if not cmds:
        return False
    facts = swe_masks.classify(cmds)
    return facts.kind == "edit" and not facts.runs


def _runaway(lab: Mapping[str, Any], ceil: int) -> tuple[bool, bool]:
    """(runaway, clear runaway) of a labelled turn: valid above the ceiling, or cut off before closing its reasoning."""
    cut = not lab["finished"] and not lab["closed"]
    over = lab["valid"] and lab["thinking"] > ceil
    return cut or over, cut or (over and lab["thinking"] > RUNAWAY_FACTOR * ceil)


def pick_pairs(state: Mapping[str, Any], cands: Sequence[Mapping[str, Any]], teacher: Mapping[str, Any] | None,
               seed: int, skip: frozenset[str] | set[str] = frozenset(), degenerate_fallback_only: bool = False,
               rules_v6: bool = False, p_teacher_fallback: bool = False) -> list[dict[str, Any]]:
    """Pairs at one state: [{rule, chosen: candidate index | "teacher", rejected: candidate index, detail}]. Each element
    of `cands` and `teacher` carries `labels` (label_turn) and a usable `message` when its labels allow a pair side.
    Rules in `skip` (dropped after their audit) are not evaluated, so their slots go to the next rules."""
    rng = random.Random(f"{seed}:{state['state_id']}")
    ceil = ceiling(state)
    lab = [c["labels"] for c in cands]
    clean = [i for i, x in enumerate(lab) if x["clean"]]
    t_lab = teacher["labels"] if teacher is not None else None
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []

    def chosen(pool: list[int], teacher_ok: bool) -> int | str | None:
        if pool:
            return rng.choice(pool)
        return "teacher" if teacher_ok and t_lab is not None and t_lab["clean"] else None

    def add(rule: str, rej: int, ch: int | str | None, detail: str = "") -> bool:
        if ch is None:
            return False
        pairs.append({"rule": rule, "chosen": ch, "rejected": rej, "detail": detail})
        used.add(rej)
        return True

    def free(idx: list[int]) -> list[int]:
        return [i for i in idx if i not in used and cands[i].get("message") is not None]

    # V: malformed finished candidates
    bad = free([i for i, x in enumerate(lab) if x["finished"] and not x["valid"] and x["invalid_reason"] != "empty_reasoning"])
    if bad and "V" not in skip:
        i = rng.choice(bad)
        add("V", i, chosen(clean, True), lab[i]["invalid_reason"] or "")
    # R: runaway thinking, largest first; up to R_MAX_PAIRS_PER_STATE pairs, no chosen side used twice
    r_chosen: set[int | str] = set()
    for i in sorted(free([i for i, x in enumerate(lab) if "R" not in skip and _runaway(x, ceil)[0]]),
                    key=lambda i: -lab[i]["thinking"]):
        if len(r_chosen) >= R_MAX_PAIRS_PER_STATE:
            break
        clear = _runaway(lab[i], ceil)[1]
        if degenerate_fallback_only:
            clear = clear and (not lab[i]["closed"] or bool(lab[i].get("repetitive")))
        unused = [j for j in clean if j not in r_chosen]
        same = [j for j in unused if lab[i]["valid"] and lab[j]["action"] == lab[i]["action"]]
        teacher_ok = "teacher" not in r_chosen
        teacher_same = teacher_ok and lab[i]["valid"] and t_lab is not None and t_lab["action"] == lab[i]["action"]
        if same:
            ch, detail = rng.choice(same), "same_action"
        elif clear:
            ch, detail = chosen(unused, teacher_ok), "clear_runaway"
        else:
            ch, detail = chosen([], teacher_same), "same_action_teacher"
        if add("R", i, ch, detail):
            r_chosen.add(ch)
    # D-SWE (v6): a submit right after an unverified edit
    if rules_v6:
        if (len(pairs) < MAX_PAIRS_PER_STATE and "D" not in skip and state["family"] == "swe"
                and last_turn_unverified_edit(state["prompt"])):
            sub = free([i for i, x in enumerate(lab) if x["valid"] and x["thinking"] <= ceil and x.get("swe_kind") == "submit"])
            runs = [j for j in clean if lab[j].get("swe_runs")]
            if sub and runs:
                add("D", rng.choice(sub), rng.choice(runs), "swe_submit_after_edit")
        if len(pairs) < MAX_PAIRS_PER_STATE and "P" not in skip:
            rep_ = free([i for i, x in enumerate(lab) if x["valid"] and x["thinking"] <= ceil and x.get("p_repeat")])
            if rep_ and clean:
                add("P", rng.choice(rep_), rng.choice(clean), "read_repeat_no_change")
            elif (rep_ and p_teacher_fallback and t_lab is not None and all(x.get("p_repeat") for x in lab if x["valid"])
                  and t_lab["clean"] and not t_lab.get("p_repeat") and not t_lab["completion"]):
                add("P", rng.choice(rep_), "teacher", "read_repeat_teacher")
        return pairs
    # D: premature completion
    if len(pairs) < MAX_PAIRS_PER_STATE and "D" not in skip:
        prem = free([i for i, x in enumerate(lab) if x["valid"] and x["thinking"] <= ceil and x["premature"]])
        if prem:
            add("D", rng.choice(prem), chosen([j for j in clean if lab[j]["verifies"]], bool(t_lab and t_lab["verifies"])))
    # P: repeat without change
    if len(pairs) < MAX_PAIRS_PER_STATE and "P" not in skip:
        rep = free([i for i, x in enumerate(lab) if x["valid"] and x["thinking"] <= ceil and not x["premature"] and x["repeat"]])
        if rep:
            add("P", rng.choice(rep), chosen(clean, True))
    # E: same action, fewer thinking tokens (easy and medium bands)
    if len(pairs) < MAX_PAIRS_PER_STATE and state["band"] in E_BANDS and "E" not in skip:
        best = None
        for act in sorted({lab[j]["action"] for j in clean}):
            group = [j for j in clean if lab[j]["action"] == act]
            short = [j for j in group if lab[j]["thinking"] >= E_MIN_CHOSEN]
            if not short:
                continue
            ch = min(short, key=lambda j: (lab[j]["thinking"], j))
            t = lab[ch]["thinking"]
            longer = [j for j in free(group) if lab[j]["thinking"] >= E_RATIO * t and lab[j]["thinking"] - t >= E_MIN_DIFF]
            if longer:
                rej = max(longer, key=lambda j: (lab[j]["thinking"], -j))
                gap = lab[rej]["thinking"] - t
                if best is None or gap > best[0]:
                    best = (gap, ch, rej)
        if best:
            add("E", best[2], best[1], f"thinking {lab[best[1]]['thinking']} vs {lab[best[2]]['thinking']}")
    return pairs


def balance(pairs: Sequence[dict[str, Any]], seed: int, family_shares: Mapping[str, float] | None = None
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The set-level constraints: rule E at most E_MAX_SHARE of the pairs, then, when family_shares is given (the plan's
    FAMILY_SHARES), the larger family cut to the ratio. P4a does not use the family cut (after the Base probe: SWE
    yields ~5x fewer pairs per state, so the cut would discard most terminal pairs); the natural split is reported. Cuts are seeded random and recorded; pair records carry `family` and `rule`."""
    rng = random.Random(f"{seed}:balance")
    kept = list(pairs)
    report: dict[str, Any] = {"n_in": len(kept)}
    e = [p for p in kept if p["rule"] == "E"]
    rest = len(kept) - len(e)
    e_max = int(E_MAX_SHARE / (1 - E_MAX_SHARE) * rest)
    if len(e) > e_max:
        drop = {id(p) for p in rng.sample(e, len(e) - e_max)}
        kept = [p for p in kept if id(p) not in drop]
    report["e_cut"] = max(0, len(e) - e_max)
    pt = [p for p in kept if p["rule"] == "P" and (p.get("metadata") or {}).get("chosen_source") == "teacher"]
    pt_max = int(P_TEACHER_MAX_SHARE / (1 - P_TEACHER_MAX_SHARE) * (len(kept) - len(pt)))
    if len(pt) > pt_max:
        drop = {id(p) for p in rng.sample(pt, len(pt) - pt_max)}
        kept = [p for p in kept if id(p) not in drop]
    report["p_teacher_cut"] = max(0, len(pt) - pt_max)
    n = {f: sum(p["family"] == f for p in kept) for f in (family_shares or {})}
    if family_shares and all(n.values()):
        # the largest total whose family counts fit the supply at the target shares
        total = min(int(n[f] / s) for f, s in family_shares.items())
        want = {f: round(total * s) for f, s in family_shares.items()}
        cut = {f: max(0, n[f] - want[f]) for f in family_shares}
        for f, k in cut.items():
            if k:
                fam = [p for p in kept if p["family"] == f]
                drop = {id(p) for p in rng.sample(fam, k)}
                kept = [p for p in kept if id(p) not in drop]
        report["family_cut"] = cut
    report["n_out"] = len(kept)
    return kept, report
