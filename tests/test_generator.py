import copy
import re

from audit_env import jsonutil, money, schema
from audit_env.generator import invariants
from audit_env.generator.episode import generate_episode
from audit_env.timeutil import parse_ts


def test_sizes_and_schemas(episodes):
    for ep in episodes.values():
        m = ep.bundle["manifest"]
        assert not schema.validate(m, "manifest")
        assert not schema.validate(ep.truth, "truth")
        assert 200 <= m["record_counts"]["ledger"] <= 2000
        assert 20 <= m["document_count"] <= 200
        assert len(ep.truth["issues"]) <= 12


def test_injected_episodes_match_truth_exactly(episodes):
    for ep in episodes.values():
        assert invariants.compare_with_truth(ep.bundle, ep.truth) == []


def test_mix_of_clean_and_errored(clean, errored):
    assert clean, "need clean companies"
    assert errored, "need companies with errors"


def test_each_enabled_type_is_generated(errored):
    types = {i["type"] for ep in errored for i in ep.truth["issues"]}
    assert {"duplicate_invoice", "unapplied_payment"} <= types


def test_clean_history_has_no_violations_by_construction():
    for seed in range(15):
        ep = generate_episode(seed, {"clean_probability": 1.0})
        assert ep.truth["issues"] == []
        assert invariants.check_clean(ep.bundle) == []


def test_distractors_present(episodes):
    tags = set()
    for ep in episodes.values():
        tags |= set(ep.provenance["generation"]["distractor_tags"])
    assert {"subscription", "split", "combined", "refund", "boundary_timezone"} <= tags


def test_boundary_timezone_distractor_in_period_but_reads_next_month(episodes):
    found = 0
    for ep in episodes.values():
        docs = {p["doc_id"]: p for p in ep.bundle["payments"]}
        end = parse_ts(ep.bundle["manifest"]["period_end"])
        for doc_id in ep.provenance["generation"]["distractor_tags"].get("boundary_timezone", []):
            p = docs[doc_id]
            assert parse_ts(p["timestamp"]) < end
            assert p["timestamp"][:10] >= end.date().isoformat()  # calendar text says next month
            found += 1
    assert found


def test_detector_flags_tampering(clean):
    ep = clean[0]
    b = copy.deepcopy(ep.bundle)
    b["ledger"][0]["debit_minor"] = (b["ledger"][0]["debit_minor"] or 0) + 1
    assert invariants.check_clean(b), "tampering must be detected"
    b = copy.deepcopy(ep.bundle)
    b["invoices"].append(dict(b["invoices"][0], doc_id="D-ZZZZZZZZ", invoice_id="INV-ZZZZZZZZ"))
    codes = {v["code"] for v in invariants.check_clean(b)}
    assert "duplicate_invoice" in codes


def test_duplicate_is_later_and_identical(errored):
    n = 0
    for ep in errored:
        invs = {i["invoice_id"]: i for i in ep.bundle["invoices"]}
        for issue in ep.truth["issues"]:
            if issue["type"] != "duplicate_invoice":
                continue
            dup = invs[issue["affected_ids"][0]]
            twins = [i for i in invs.values() if i is not dup and i["line_items"] == dup["line_items"]
                     and i["entity_id"] == dup["entity_id"] and i["total_minor"] == dup["total_minor"]]
            assert len(twins) == 1
            assert parse_ts(twins[0]["issued_at"]) < parse_ts(dup["issued_at"])
            n += 1
    assert n


def test_unapplied_payment_amount(errored):
    n = 0
    for ep in errored:
        pays = {p["payment_id"]: p for p in ep.bundle["payments"]}
        base = ep.bundle["manifest"]["base_currency"]
        for issue in ep.truth["issues"]:
            if issue["type"] != "unapplied_payment":
                continue
            p = pays[issue["affected_ids"][0]]
            dups = {i["affected_ids"][0] for i in ep.truth["issues"] if i["type"] == "duplicate_invoice"}
            applied_valid = sum(r["amount_minor"] for r in p["invoice_refs"] or [] if r["invoice_id"] not in dups)
            assert applied_valid < p["amount_minor"]
            if p["currency"] == base:
                assert issue["expected_amount_minor"] == p["amount_minor"] - applied_valid
            n += 1
    assert n


def test_ids_are_opaque(episodes):
    pat = re.compile(r"^(E|INV|PAY|D|T|R|FX)-[2-9A-HJ-NP-Z]{8}$")
    for ep in list(episodes.values())[:10]:
        b = ep.bundle
        ids = [r["row_id"] for r in b["ledger"]] + [r["txn_id"] for r in b["ledger"]]
        ids += [i["invoice_id"] for i in b["invoices"]] + [i["doc_id"] for i in b["invoices"]]
        ids += [p["payment_id"] for p in b["payments"]] + [r["fx_id"] for r in b["fx"]]
        for i in ids:
            assert pat.match(i), i


def test_injected_documents_not_at_end(errored):
    """Injected invoices must not be systematically last in file order."""
    positions = []
    for ep in errored:
        order = [i["invoice_id"] for i in ep.bundle["invoices"]]
        for issue in ep.truth["issues"]:
            if issue["type"] == "duplicate_invoice":
                positions.append(order.index(issue["affected_ids"][0]) / max(1, len(order) - 1))
    assert positions
    assert 0.2 < sum(positions) / len(positions) < 0.8


def test_generation_is_deterministic():
    a = generate_episode(7)
    b = generate_episode(7)
    assert jsonutil.dumps(a.bundle) == jsonutil.dumps(b.bundle)
    assert jsonutil.dumps(a.truth) == jsonutil.dumps(b.truth)
    assert jsonutil.dumps(a.provenance) == jsonutil.dumps(b.provenance)


def test_ledger_is_base_currency_and_balanced(episodes):
    for ep in list(episodes.values())[:10]:
        base = ep.bundle["manifest"]["base_currency"]
        assert {r["currency"] for r in ep.bundle["ledger"]} == {base}
        assert sum(r["debit_minor"] for r in ep.bundle["ledger"]) == sum(r["credit_minor"] for r in ep.bundle["ledger"])


def test_money_never_float(episodes):
    for ep in list(episodes.values())[:10]:
        for inv in ep.bundle["invoices"]:
            assert isinstance(inv["total_minor"], int)
        for r in ep.bundle["fx"]:
            money.parse_rate(r["rate"])
