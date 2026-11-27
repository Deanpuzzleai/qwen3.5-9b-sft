"""Terminal teacher slice. First build (WP4, unit U2.2; docs/DATA_PLAN.md section 3.3): DeepSeek-V4-Pro Terminus-2
episodes from TermGrade (`tgr`, verifier-labelled) and TerminalHorizon-3K (`th3k`, authors' filter plus the completion
handshake). Revision 5 (data plan v5, 3.2): one Kimi-K3 TermGrade trial per task, every turn trained.

  harness    Harbor 0.24.0 Terminus-2 first prompt (template copy + sha256), the first-prompt match and re-render, the
             `<think>` split of TerminalHorizon's assistant text, observation shapes
  loaders    both corpora -> schema Episode / Task / Verdict dicts (`Loaded`), loader drops with reasons; the TermGrade
             teacher is a model key (`TGR_TEACHERS`: deepseek, kimi)
  linking    TerminalHorizon episode -> environment by instruction word 6-gram shingles (category for the quota)
  admission  per-episode screens (D1 canary, D4 slugs and identifiers, identity, credentials), the admission pilot
             thresholds and decision
  select     the category quota with rollover, one episode per environment, the TermGrade all-six-solve skip and
             the <= 2 episodes per task draw
  checks     M2.2 row checks: the 20-row render check and the parser round trip of every trained turn
  select_v5  revision 5: the lowest-thinking gate-passing trial per task, hard once, the rest medium 55 / easy 45 by loss
             tokens with the lower half by thinking first

`scripts/data/build_terminal_slice.py` builds the first build's `data/pools/terminal_teacher/`;
`scripts/data/build_terminal_slice_v5.py` builds revision 5's `data/processed/v5/terminal/`.
"""
