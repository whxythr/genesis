"""Matcher: normalization, tolerance, evidence, one-to-one matching."""

import copy

from audit_env.grader import matcher
from audit_env.grader.core import grade
from audit_env.grader.normalize import validate_and_normalize

from .helpers import oracle_findings


def _ep(errored):
    return next(ep for ep in errored if len(ep.truth["issues"]) >= 2)


def _grade(ep, findings, steps=0):
    return grade({"episode_id": ep.episode_id, "findings": findings}, ep.truth, ep.bundle, steps, "submitted")


def test_oracle_all_tp(errored):
    for ep in errored:
        res = _grade(ep, oracle_findings(ep.truth))
        assert res["metrics"]["fn"] == 0 and res["metrics"]["fp"] == 0 and res["reward"] == 1.0


def test_normalization_of_ids_types_currency(errored):
    ep = _ep(errored)
    f = oracle_findings(ep.truth)
    for x in f:
        x["type"] = " " + x["type"].upper().replace("_", "-") + " "
        x["affected_ids"] = [" " + a.lower() + " " for a in x["affected_ids"]] * 2
        x["currency"] = x["currency"].lower()
        x["evidence_ids"] = [e.lower() for e in x["evidence_ids"]]
    assert _grade(ep, f)["metrics"]["fp"] == 0


def test_tolerance_boundary(errored):
    ep = _ep(errored)
    t = ep.truth["issues"][0]
    tol = matcher.tolerance(t["expected_amount_minor"])
    f = oracle_findings({"issues": [t]})
    for delta, ok in [(tol, True), (-tol, True), (tol + 1, False), (-(tol + 1), False)]:
        g = copy.deepcopy(f)
        g[0]["claimed_amount_minor"] = max(0, t["expected_amount_minor"] + delta)
        assert (_grade(ep, g)["metrics"]["tp"] == 1) == ok


def test_tolerance_formula():
    assert matcher.tolerance(1) == 1
    assert matcher.tolerance(999) == 1
    assert matcher.tolerance(5000) == 5
    assert matcher.tolerance(1_000_000) == 1000


def test_wrong_type_ids_currency_fail(errored):
    ep = _ep(errored)
    base = oracle_findings({"issues": [ep.truth["issues"][0]]})
    mutations = [
        lambda f: f.update(type="fx_error" if f["type"] != "fx_error" else "period_cutoff"),
        lambda f: f.update(affected_ids=f["affected_ids"] + ["INV-22222222"]),
        lambda f: f.update(currency="XXX"),
        lambda f: f.update(evidence_ids=[]),
        lambda f: f.update(evidence_ids=f["evidence_ids"][:-1]),
        lambda f: f.update(evidence_ids=f["evidence_ids"] + ["D-NOTEXIST"]),
    ]
    for m in mutations:
        g = copy.deepcopy(base)
        m(g[0])
        assert _grade(ep, g)["metrics"]["tp"] == 0


def test_evidence_aliases(errored):
    """invoice_id/payment_id stand for their document, row_id for its transaction."""
    ep = _ep(errored)
    docs = {i["doc_id"]: i["invoice_id"] for i in ep.bundle["invoices"]}
    docs.update({p["doc_id"]: p["payment_id"] for p in ep.bundle["payments"]})
    row_of_txn = {}
    for r in ep.bundle["ledger"]:
        row_of_txn.setdefault(r["txn_id"], r["row_id"])
    f = oracle_findings(ep.truth)
    for x in f:
        x["evidence_ids"] = [docs.get(e) or row_of_txn.get(e) or e for e in x["evidence_ids"]]
    assert _grade(ep, f)["metrics"]["fp"] == 0


def test_one_to_one(errored):
    ep = _ep(errored)
    f = oracle_findings(ep.truth)
    res = _grade(ep, f + f)
    n = len(ep.truth["issues"])
    assert res["metrics"]["tp"] == n and res["metrics"]["fp"] == n


def test_finding_order_irrelevant(errored):
    ep = _ep(errored)
    f = oracle_findings(ep.truth)
    assert _grade(ep, f)["reward"] == _grade(ep, list(reversed(f)))["reward"]


def test_unknown_type_is_fp_not_invalid(errored):
    ep = _ep(errored)
    f = oracle_findings(ep.truth) + [{"type": "embezzlement", "affected_ids": ["X"], "claimed_amount_minor": 1,
                                      "currency": "USD", "evidence_ids": []}]
    res = _grade(ep, f)
    assert res["invalid_reason"] is None and res["metrics"]["fp"] == 1


def test_validate_rejects_non_json():
    out, err = validate_and_normalize({"episode_id": "EP-2222222222", "findings": [object()]}, "EP-2222222222")
    assert out is None and err
