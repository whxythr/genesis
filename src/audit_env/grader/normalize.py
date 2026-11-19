"""Findings validation and normalization (published rules).

- Structure must match ``schemas/findings.schema.json`` and ``episode_id`` must equal the
  episode's id; otherwise the submission is INVALID (reward -1).
- type: strip, lower-case, '-' and ' ' -> '_'.  Unknown types are allowed but never match.
- IDs: strip whitespace, upper-case; affected_ids / evidence_ids become sorted unique sets.
- currency: strip, upper-case.
- explanation is ignored by the grader (reward never depends on free text).
"""

from __future__ import annotations

import json

from .. import schema

CANONICAL_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")


def norm_type(t: str) -> str:
    return t.strip().lower().replace("-", "_").replace(" ", "_")


def norm_id(i: str) -> str:
    return i.strip().upper()


def validate_and_normalize(raw, episode_id: str) -> tuple[list[dict] | None, str | None]:
    """Returns (normalized_findings, None) or (None, error_code)."""
    try:
        raw = json.loads(json.dumps(raw))  # must be plain JSON data
    except (TypeError, ValueError):
        return None, "NOT_JSON"
    errs = schema.validate(raw, "findings")
    if errs:
        return None, "SCHEMA"
    if raw["episode_id"].strip() != episode_id:
        return None, "EPISODE_MISMATCH"
    out = []
    for f in raw["findings"]:
        affected = sorted({norm_id(x) for x in f["affected_ids"]})
        if "" in affected:
            return None, "SCHEMA"
        out.append({
            "type": norm_type(f["type"]),
            "affected_ids": affected,
            "claimed_amount_minor": f["claimed_amount_minor"],
            "currency": f["currency"].strip().upper(),
            "evidence_ids": sorted({norm_id(x) for x in f["evidence_ids"]} - {""}),
        })
    return out, None
