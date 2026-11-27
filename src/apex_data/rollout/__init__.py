"""Stage-2 rollout drivers (docs/DATA_PLAN.md section 11; unit U3.3 of docs/EXECUTION_PLAN.md).

  endpoint   the served policy (OpenAI-compatible), the stage-2 caps, the lock sampling presets, per-sample seeds and the
             request-logging proxy every harness talks through
  mini_swe   mini-swe-agent 2.4.6 tool-call episodes on SWE-smith task images (the SWE-bench Pro lock's agent config)
  terminus   Terminus-2 trials through Harbor on the TermGrade task directories (the Terminal-Bench lock's agent settings)
  scheduler  the continuous per-family scheduler shared by scripts/data/{run_rollouts,grade_episodes}.py

Verdicts come from apex_data.grade.{swesmith,harbor_tests}; scripts/data/build_stage2_set.py renders the two arms.
"""
