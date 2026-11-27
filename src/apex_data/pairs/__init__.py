"""Stage-2 turn-level preference pairs (scripts/data/build_stage2_pairs.py).

  states      held-out teacher episodes cut before one assistant turn: the prompt (messages + tools, rendered to token ids
              by apex_train.stage2's own fit_prompt), the teacher's turn there, the band and the sampling budget
  candidates  one-turn samples of the stage-1 model through vLLM's /v1/completions with the token-id prompt, split at
              </think> and parsed by the stage-1 gates' parsers (render.filters.qwen3_parse_calls / terminus_parse)
  rules       per-candidate labels from the profiler's detectors (apex_data.efficiency) and the pair rules V, R, D, P, E
              with the teacher-turn fallback; every pair passes apex_train.stage2.PreferencePair.from_dict and render_pair
"""
