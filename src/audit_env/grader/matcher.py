"""One-to-one matching of findings to truth issues (published code).

A finding f has an EDGE to issue t iff ALL hold:
  1. f.type == t.type                                   (exact string)
  2. set(f.affected_ids) == set(t.affected_ids)          (exact strings, canonical set)
  3. f.currency == t.currency                            (exact string)
  4. |f.claimed_amount_minor - t.expected_amount_minor| <= TOLERANCE_MINOR[t.type]
  5. every evidence id exists in the open records of the episode, and the evidence covers
     at least one complete acceptable evidence set of t.  One unknown evidence id -> no
     edge (FP), even if a full set is also present.

Amount tolerance tau: 0 minor units for every type (exact integer comparison).  All
expected amounts are produced by integer arithmetic with the published ROUND_HALF_UP
rule (policy.json), so an agent that follows the policy obtains them exactly.

Existing evidence ids: any open identifier - doc_id, invoice_id, payment_id, entity_id,
row_id, txn_id, fx_id, or the fx key "date:source_currency:target_currency".

Coverage of a required id r of an acceptable set (literal subset semantics, plus one
documented convenience):
  - r is covered if r itself is among the submitted evidence ids;
  - if r is a CANONICAL id - doc_id of a document, txn_id of a ledger transaction or
    fx_id of a rate row, the forms our truth uses - it is also covered by another open
    identifier of the same object: invoice_id/payment_id for a doc_id, any row_id of
    the transaction for a txn_id, the "date:source:target" key for an fx_id.
  A required row_id, invoice_id, payment_id, entity_id or fx key is covered only by
  itself, so {row A, row B} is never covered by row A alone.  Identifiers are resolved
  per namespace: when one string is an id in two namespaces it stands for both.

Matching: maximum-cardinality bipartite matching (left = every submitted finding,
repeats included; right = every truth issue).  TP = matching size, FP = findings - TP,
FN = issues - TP.  Tie-break among maximum matchings (affects only WHICH finding is
reported as matched, never TP/FP/FN or the reward): issue_ids sorted by Unicode code
points; the lexicographically smallest sorted sequence of (issue_id, finding_index)
pairs is chosen.
"""

from __future__ import annotations

TOLERANCE_MINOR = {
    "duplicate_invoice": 0,
    "unapplied_payment": 0,
    "fx_error": 0,
    "period_cutoff": 0,
}


def tolerance(issue_type: str) -> int:
    return TOLERANCE_MINOR.get(issue_type, 0)


def build_alias_map(bundle: dict) -> dict[str, frozenset]:
    alias: dict[str, set] = {}

    def add(key, target) -> None:
        if isinstance(key, str):
            alias.setdefault(key, set()).add(target)

    for inv in bundle["invoices"]:
        tgt = ("doc", inv["doc_id"])
        add(inv["doc_id"], tgt)
        add(inv["invoice_id"], tgt)
        add(inv.get("entity_id"), ("entity", inv.get("entity_id")))
    for p in bundle["payments"]:
        tgt = ("doc", p["doc_id"])
        add(p["doc_id"], tgt)
        add(p["payment_id"], tgt)
        add(p.get("entity_id"), ("entity", p.get("entity_id")))
    for r in bundle["ledger"]:
        tgt = ("txn", r["txn_id"])
        add(r["row_id"], tgt)
        add(r["txn_id"], tgt)
    for r in bundle["fx"]:
        tgt = ("fx", r["fx_id"])
        add(r["fx_id"], tgt)
        add(f"{r['date']}:{r['source_currency']}:{r['target_currency']}", tgt)
    return {k: frozenset(v) for k, v in alias.items()}


def resolve_evidence(evidence_ids: list[str], alias: dict[str, frozenset]) -> set | None:
    """Union of resolved targets, or None if any id does not exist in the open records."""
    out: set = set()
    for e in evidence_ids:
        if e not in alias:
            return None
        out |= alias[e]
    return out


def _covered(r: str, submitted: set[str], have: set, alias: dict[str, frozenset]) -> bool:
    if r in submitted:
        return True
    canonical = {t for t in alias.get(r, ()) if t[0] in ("doc", "txn", "fx") and t[1] == r}
    return bool(canonical & have)


def is_candidate(finding: dict, issue: dict, alias: dict[str, frozenset]) -> bool:
    if finding["type"] != issue["type"]:
        return False
    if set(finding["affected_ids"]) != set(issue["affected_ids"]):
        return False
    if finding["currency"] != issue["currency"]:
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > tolerance(issue["type"]):
        return False
    have = resolve_evidence(finding["evidence_ids"], alias)
    if have is None:
        return False
    submitted = set(finding["evidence_ids"])
    return any(all(_covered(r, submitted, have, alias) for r in s) for s in issue["acceptable_evidence_sets"])


def _max_matching(adj: list[list[int]], n_right: int, fixed: dict[int, int], banned_right: set[int]) -> int:
    """Size of a maximum matching that contains every pair in ``fixed`` (right -> left)
    and leaves ``banned_right`` unmatched.  Kuhn's augmenting paths."""
    owner: list[int | None] = [None] * n_right
    used_left = set()
    for j, i in fixed.items():
        owner[j] = i
        used_left.add(i)
    locked = set(fixed)

    def augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen or j in locked or j in banned_right:
                continue
            seen.add(j)
            if owner[j] is None or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    size = len(fixed)
    for i in range(len(adj)):
        if i not in used_left and augment(i, set()):
            size += 1
    return size


def match(findings: list[dict], issues: list[dict], alias: dict[str, frozenset]) -> list[tuple[int, str]]:
    """Sorted list of (finding_index, issue_id) pairs of the published maximum matching."""
    issues = sorted(issues, key=lambda t: t["issue_id"])
    adj = [[j for j, t in enumerate(issues) if is_candidate(f, t, alias)] for f in findings]
    best = _max_matching(adj, len(issues), {}, set())
    fixed: dict[int, int] = {}
    banned: set[int] = set()
    for j in range(len(issues)):
        chosen = None
        for i in range(len(findings)):
            if j in adj[i] and i not in fixed.values():
                if _max_matching(adj, len(issues), {**fixed, j: i}, banned) == best:
                    chosen = i
                    break
        if chosen is None:
            banned.add(j)
        else:
            fixed[j] = chosen
    return sorted((i, issues[j]["issue_id"]) for j, i in fixed.items())
