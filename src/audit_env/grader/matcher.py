"""One-to-one matching of findings to truth issues (published code).

A finding f can match issue t iff ALL hold:
  1. f.type == t.type
  2. set(f.affected_ids) == set(t.affected_ids)
  3. f.currency == t.currency
  4. |f.claimed_amount_minor - t.expected_amount_minor| <= tolerance(t.expected_amount_minor)
     where tolerance(x) = max(1, x // 1000)   (0.1 %, at least one minor unit)
  5. every evidence id exists in the episode, and the evidence (after alias
     resolution) contains at least one complete acceptable evidence set of t.

Evidence alias resolution: invoice_id / payment_id -> its doc_id; row_id -> its txn_id;
doc_id, txn_id, fx_id, entity_id stand for themselves.

Matching is maximum bipartite matching (Kuhn's augmenting paths), findings visited in
submission order and issues in issue_id order — deterministic. Only the matching
*size* affects metrics, and the maximum size is unique, so the tie-break never
changes the reward.
"""

from __future__ import annotations


def tolerance(expected: int) -> int:
    return max(1, expected // 1000)


def build_alias_map(bundle: dict) -> dict[str, str]:
    alias: dict[str, str] = {}
    for inv in bundle["invoices"]:
        alias[inv["doc_id"]] = inv["doc_id"]
        alias[inv["invoice_id"]] = inv["doc_id"]
        alias[inv["entity_id"]] = inv["entity_id"]
    for p in bundle["payments"]:
        alias[p["doc_id"]] = p["doc_id"]
        alias[p["payment_id"]] = p["doc_id"]
        alias[p["entity_id"]] = p["entity_id"]
    for r in bundle["ledger"]:
        alias[r["row_id"]] = r["txn_id"]
        alias[r["txn_id"]] = r["txn_id"]
    for r in bundle["fx"]:
        alias[r["fx_id"]] = r["fx_id"]
    return alias


def resolve_evidence(evidence_ids: list[str], alias: dict[str, str]) -> set[str] | None:
    """Resolved evidence set, or None if any id does not exist."""
    out = set()
    for e in evidence_ids:
        if e not in alias:
            return None
        out.add(alias[e])
    return out


def is_candidate(finding: dict, issue: dict, alias: dict[str, str]) -> bool:
    if finding["type"] != issue["type"]:
        return False
    if set(finding["affected_ids"]) != set(issue["affected_ids"]):
        return False
    if finding["currency"] != issue["currency"]:
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > tolerance(issue["expected_amount_minor"]):
        return False
    ev = resolve_evidence(finding["evidence_ids"], alias)
    if ev is None:
        return False
    return any(set(s) <= ev for s in issue["acceptable_evidence_sets"])


def match(findings: list[dict], issues: list[dict], alias: dict[str, str]) -> list[tuple[int, str]]:
    """Returns sorted list of (finding_index, issue_id) pairs of a maximum matching."""
    issues = sorted(issues, key=lambda t: t["issue_id"])
    adj = [[j for j, t in enumerate(issues) if is_candidate(f, t, alias)] for f in findings]
    owner: list[int | None] = [None] * len(issues)

    def augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] is None or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(findings)):
        augment(i, set())
    return sorted((i, issues[j]["issue_id"]) for j, i in enumerate(owner) if i is not None)
