"""Retention slice (WP5, docs/DATA_PLAN.md section 3.4): ready-made reasoning rows that keep math, science, instruction
following and chat where they are while the agentic slices carry ~85 percent of the loss.

Modules:
  common   part specifications (row counts, think caps, teachers, licences), the funnel counter, text checks shared by
           the loaders, and the Episode / task / Verdict records `render.filters.filter_episodes` consumes
  math     nvidia/Nemotron-SFT-Math-v4 AoPS chain-of-thought rows
  ifchat   nvidia/Nemotron-SFT-Instruction-Following-Chat-v3: IF rows (constraints extracted and checked) and chat rows
  science  nvidia/Nemotron-Science-v1 MCQ and RQA rows
  topics   bge-small prototype classifier for the chat topic quota
  draw     seeded stratified draws (math length bands, chat topic groups, MCQ option counts) and the SFT / replay
           reserve split
  convergence  the stage-2 convergence replay's parts and filters (reasoning bands, short final section, MCQ letter
           ending, token-target selection, disjointness)
The builder is scripts/data/build_retention_slice.py.
"""
