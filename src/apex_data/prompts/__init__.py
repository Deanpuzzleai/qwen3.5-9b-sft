"""Instruction-following constraints and their deterministic checkers (docs/DATA_PLAN.md sections 3.4 and 6.3).

`constraints` holds the constraint record (kind, params, instruction, id), the revision-3 template set and the extractor
that reads verifiable constraints out of a natural-language prompt; `checkers` verifies them on a response with regex and
counting only (no model, no network, no randomness).
"""
