"""Phase 2: every error type, clean books, distractors, independent validation on 100 seeds."""

import copy

import pytest

from audit_env import money
from audit_env.generator import evidence, invariants
from audit_env.generator.episode import generate_episode
from audit_env.timeutil import local_date, parse_ts

TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")


@pytest.fixture(scope="module")
def hundred():
    return [generate_episode(seed) for seed in range(100)]


def test_validator_confirms_every_error_on_100_seeds(hundred):
    counts = {t: 0 for t in TYPES}
    for ep in hundred:
        assert invariants.compare_with_truth(ep.bundle, ep.truth) == []
        assert evidence.check_sufficiency(ep.bundle, ep.truth) == []
        for i in ep.truth["issues"]:
            counts[i["type"]] += 1
    assert all(n >= 20 for n in counts.values()), counts
    assert sum(1 for ep in hundred if not ep.truth["issues"]) >= 15


@pytest.mark.parametrize("etype", TYPES)
def test_each_type_alone(etype):
    """Each type can be switched on alone and produces only that type."""
    cfg = {"clean_probability": 0.0,
           "errors": {t: {"enabled": t == etype, "count": [1, 2]} for t in TYPES}}
    seen = 0
    for seed in range(12):
        ep = generate_episode(seed, cfg)
        assert {i["type"] for i in ep.truth["issues"]} <= {etype}
        seen += len(ep.truth["issues"])
    assert seen >= 8


def test_all_types_disabled_gives_clean_books():
    cfg = {"errors": {t: {"enabled": False} for t in TYPES}}
    for seed in range(5):
        ep = generate_episode(seed, cfg)
        assert ep.truth["issues"] == []
        assert invariants.check_clean(ep.bundle) == []


def test_fx_error_amount_is_posted_minus_policy_conversion(hundred):
    n = 0
    for ep in hundred:
        b = ep.bundle
        base, tz = b["manifest"]["base_currency"], b["manifest"]["timezone"]
        rates = {(r["source_currency"], r["date"]): r["rate"] for r in b["fx"]}
        docs = {i["invoice_id"]: ("invoice", i) for i in b["invoices"]}
        docs.update({p["payment_id"]: ("payment", p) for p in b["payments"]})
        for issue in ep.truth["issues"]:
            if issue["type"] != "fx_error":
                continue
            kind, d = docs[issue["affected_ids"][0]]
            ts = parse_ts(d["issued_at"] if kind == "invoice" else d["timestamp"])
            amount = d["total_minor"] if kind == "invoice" else d["amount_minor"]
            correct = money.convert(amount, d["currency"], base, rates[(d["currency"], local_date(ts, tz).isoformat())])
            key = "invoice_id" if kind == "invoice" else "payment_id"
            acct = "AR" if kind == "invoice" else "CASH"
            posted = [r["debit_minor"] for r in b["ledger"] if r[key] == d[key] and r["account"] == acct
                      and (kind == "payment" or r["payment_id"] is None)]
            assert abs(posted[0] - correct) == issue["expected_amount_minor"] > 0
            assert d["currency"] != base
            n += 1
    assert n


def test_period_cutoff_document_outside_posting_inside(hundred):
    variants = set()
    for ep in hundred:
        b = ep.bundle
        start, end = parse_ts(b["manifest"]["period_start"]), parse_ts(b["manifest"]["period_end"])
        docs = {i["invoice_id"]: i["issued_at"] for i in b["invoices"]}
        docs.update({p["payment_id"]: p["timestamp"] for p in b["payments"]})
        for issue in ep.truth["issues"]:
            if issue["type"] != "period_cutoff":
                continue
            aid = issue["affected_ids"][0]
            ts = parse_ts(docs[aid])
            assert not start <= ts < end
            rows = [r for r in b["ledger"] if aid in (r["invoice_id"], r["payment_id"])]
            assert rows and all(start <= parse_ts(r["timestamp"]) < end for r in rows)
            variants.add("reads_in_period" if docs[aid][:10] < end.date().isoformat() else "reads_next_period")
    assert variants == {"reads_in_period", "reads_next_period"}  # the timezone trap is exercised


def test_no_other_document_outside_period(hundred):
    for ep in hundred:
        b = ep.bundle
        start, end = parse_ts(b["manifest"]["period_start"]), parse_ts(b["manifest"]["period_end"])
        cut = {i["affected_ids"][0] for i in ep.truth["issues"] if i["type"] == "period_cutoff"}
        for d in b["invoices"]:
            assert (start <= parse_ts(d["issued_at"]) < end) or d["invoice_id"] in cut
        for d in b["payments"]:
            assert (start <= parse_ts(d["timestamp"]) < end) or d["payment_id"] in cut


def test_correct_fx_conversions_exist_as_distractors(hundred):
    """Foreign documents posted correctly must be common (so 'foreign' is no signal)."""
    foreign_ok = 0
    for ep in hundred:
        base = ep.bundle["manifest"]["base_currency"]
        fx_ids = {i["affected_ids"][0] for i in ep.truth["issues"] if i["type"] == "fx_error"}
        foreign_ok += sum(1 for i in ep.bundle["invoices"] if i["currency"] != base and i["invoice_id"] not in fx_ids)
    assert foreign_ok > 200


def test_weakened_evidence_is_insufficient(hundred):
    checked = 0
    for ep in hundred:
        for issue in ep.truth["issues"]:
            ev = issue["acceptable_evidence_sets"][0]
            for k in range(len(ev)):
                weaker = ev[:k] + ev[k + 1:]
                with pytest.raises(evidence.Insufficient):
                    evidence.derive(issue["type"], weaker, ep.bundle)
                checked += 1
    assert checked > 100


def test_provenance_records_every_change(hundred):
    for ep in hundred:
        changes = ep.provenance["changes"]
        assert len(changes) == len(ep.truth["issues"])
        assert {c["issue_id"] for c in changes} == {i["issue_id"] for i in ep.truth["issues"]}
        for c in changes:
            assert c["reason"] and "before" in c and "after" in c
        if not ep.truth["issues"]:
            d = ep.provenance["ledger_diff"]
            assert d == {"added": [], "removed": [], "changed": []}


def test_tampered_truth_is_caught(hundred):
    ep = next(e for e in hundred if e.truth["issues"])
    t = copy.deepcopy(ep.truth)
    t["issues"][0]["expected_amount_minor"] += 7
    assert invariants.compare_with_truth(ep.bundle, t)
    t = copy.deepcopy(ep.truth)
    t["issues"].pop()
    assert invariants.compare_with_truth(ep.bundle, t)
