"""Findings validation and normalization (published rules).

Follows the organisers' matcher conventions (published 2026-11-20):

1. Structure is checked FIRST, on the raw submission, against
   ``schemas/findings.schema.json``: <= 30 findings, <= 10 evidence_ids per finding
   (counted before de-duplication), explanation <= 1000 Unicode code points,
   ``claimed_amount_minor`` a JSON integer (``true`` / ``"10000"`` / ``1.5`` are rejected),
   every required field present with the right JSON type, no unknown keys.  Any
   violation -> the whole submission is INVALID (reward -1, logged by the grader).
   ``episode_id`` must equal the episode's id exactly.
2. Only then normalization:
   - ``type``: exact string, no case folding or synonyms.  An unknown type string is valid
     structure but never matches (FP).
   - IDs: exact strings (case and whitespace are significant, no prefix stripping).
     ``affected_ids`` and ``evidence_ids`` are SETS: order and repeats inside one list
     do not matter.
   - ``currency``: exact string, no case folding / trimming / conversion.
   - ``explanation`` is ignored: reward never depends on free text.
   - Repeated findings are NOT merged: each one is a separate vertex in the matching,
     so a repeat of an already matched finding is a false positive.
"""

from __future__ import annotations

import json

from .. import schema

CANONICAL_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")


def validate_and_normalize(raw, episode_id: str) -> tuple[list[dict] | None, str | None]:
    """Returns (normalized_findings, None) or (None, error_code)."""
    try:
        raw = json.loads(json.dumps(raw, allow_nan=False))  # must be plain JSON data
    except (TypeError, ValueError):
        return None, "NOT_JSON"
    if schema.validate(raw, "findings"):
        return None, "SCHEMA"
    if raw["episode_id"] != episode_id:
        return None, "EPISODE_MISMATCH"
    out = []
    for f in raw["findings"]:
        out.append({
            "type": f["type"],
            "affected_ids": sorted(set(f["affected_ids"])),
            "claimed_amount_minor": f["claimed_amount_minor"],
            "currency": f["currency"],
            "evidence_ids": sorted(set(f["evidence_ids"])),
        })
    return out, None
