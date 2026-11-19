"""The two interacting-error scenarios: expected issues and why the amounts are what they are."""

import pytest

from audit_env import money
from audit_env.generator import evidence
from audit_env.generator.episode import generate_episode
from audit_env.timeutil import local_date, parse_ts


def _find(name, n=6):
    out = []
    for s in range(400):
        ep = generate_episode(s)
        if any(c.get("scenario") == name for c in ep.provenance["changes"]):
            out.append(ep)
        if len(out) >= n:
            break
    assert out, name
    return out


@pytest.fixture(scope="module")
def scen_a():
    return _find("partial_fx_payment_at_cutoff")


@pytest.fixture(scope="module")
def scen_b():
    return _find("partially_paid_duplicate")


def _rate(b, cur, day):
    return next(r["rate"] for r in b["fx"] if r["source_currency"] == cur and r["date"] == day)


def test_scenario_a_three_consistent_issues(scen_a):
    for ep in scen_a:
        b = ep.bundle
        base, tz = b["manifest"]["base_currency"], b["manifest"]["timezone"]
        pid = next(c["affected_ids"][0] for c in ep.provenance["changes"] if c.get("scenario") == "partial_fx_payment_at_cutoff")
        issues = {i["type"]: i for i in ep.truth["issues"] if i["affected_ids"] == [pid]}
        assert set(issues) == {"period_cutoff", "fx_error", "unapplied_payment"}
        p = next(x for x in b["payments"] if x["payment_id"] == pid)
        ts = parse_ts(p["timestamp"])
        assert ts >= parse_ts(b["manifest"]["period_end"])           # really next period
        right = _rate(b, p["currency"], local_date(ts, tz).isoformat())
        cash_rows = [r for r in b["ledger"] if r["payment_id"] == pid and r["account"] == "CASH"]
        posted = cash_rows[0]["debit_minor"]
        booking_day = local_date(parse_ts(cash_rows[0]["timestamp"]), tz).isoformat()
        wrong = _rate(b, p["currency"], booking_day)
        assert posted == money.convert(p["amount_minor"], p["currency"], base, wrong)  # booked at booking-day rate
        correct = money.convert(p["amount_minor"], p["currency"], base, right)
        applied = sum(r["amount_minor"] for r in p["invoice_refs"] or [])
        assert issues["period_cutoff"]["expected_amount_minor"] == correct
        assert issues["fx_error"]["expected_amount_minor"] == abs(posted - correct)
        assert issues["unapplied_payment"]["expected_amount_minor"] == money.convert(
            p["amount_minor"] - applied, p["currency"], base, right)
        # the UNAPPLIED_CASH line uses the wrong rate, so evidence must carry the fx row
        line = sum(r["credit_minor"] for r in b["ledger"] if r["payment_id"] == pid and r["account"] == "UNAPPLIED_CASH")
        if line != issues["unapplied_payment"]["expected_amount_minor"]:
            ev = [e for e in issues["unapplied_payment"]["acceptable_evidence_sets"][0] if not e.startswith("FX-")]
            got = None
            try:
                got = evidence.derive("unapplied_payment", ev, b)
            except evidence.Insufficient:
                pass
            assert got is None or got["amount"] != issues["unapplied_payment"]["expected_amount_minor"]


def test_scenario_b_duplicate_and_unapplied(scen_b):
    for ep in scen_b:
        b = ep.bundle
        base, tz = b["manifest"]["base_currency"], b["manifest"]["timezone"]
        ch = [c for c in ep.provenance["changes"] if c.get("scenario") == "partially_paid_duplicate"]
        dup_id = next(c["affected_ids"][0] for c in ch if c["type"] == "duplicate_invoice")
        pid = next(c["affected_ids"][0] for c in ch if c["type"] == "unapplied_payment")
        p = next(x for x in b["payments"] if x["payment_id"] == pid)
        dup = next(i for i in b["invoices"] if i["invoice_id"] == dup_id)
        assert [r["invoice_id"] for r in p["invoice_refs"]] == [dup_id]
        assert p["amount_minor"] < dup["total_minor"]  # partially paid
        un = next(i for i in ep.truth["issues"] if i["type"] == "unapplied_payment" and i["affected_ids"] == [pid])
        rate = "1" if p["currency"] == base else _rate(b, p["currency"], local_date(parse_ts(p["timestamp"]), tz).isoformat())
        assert un["expected_amount_minor"] == money.convert(p["amount_minor"], p["currency"], base, rate)
        d = next(i for i in ep.truth["issues"] if i["type"] == "duplicate_invoice" and i["affected_ids"] == [dup_id])
        assert d["expected_amount_minor"] > 0
        # proof of the unapplied part requires BOTH invoices (to show which one is the duplicate)
        assert len([e for e in un["acceptable_evidence_sets"][0] if e.startswith("D-")]) == 3


def test_scenarios_respect_type_switches():
    cfg = {"clean_probability": 0.0, "errors": {"fx_error": {"enabled": False}},
           "scenarios": {"partial_fx_payment_at_cutoff": {"probability": 1.0},
                         "partially_paid_duplicate": {"probability": 1.0}}}
    for s in range(8):
        ep = generate_episode(s, cfg)
        assert "partial_fx_payment_at_cutoff" not in ep.provenance["generation"]["plan"]["scenarios"]
        assert all(i["type"] != "fx_error" for i in ep.truth["issues"])
