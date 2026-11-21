"""The organisers' Block 3 (20 matcher/reward tests) and open examples M01-M12,
reproduced literally (checks definitions published 2026-11-20).

Fixture: one truth issue A of a supported type T, affected record R1, 10000 minor KZT,
one acceptable evidence set {D1, D2}; tolerance 0; inspection_steps = 10 by default.
"""

from __future__ import annotations

import copy

import pytest

from audit_env.grader import matcher
from audit_env.grader.core import grade

EP = "EP-FIXTURE01"
T = "duplicate_invoice"
TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")
TOL = 1e-9


def _bundle(n_records: int = 5) -> dict:
    invoices = [{"doc_id": f"D{i}", "invoice_id": f"R{i}", "entity_id": "E1", "currency": "KZT",
                 "issued_at": "2026-09-01T10:00:00+05:00", "due_at": "2026-10-01T10:00:00+05:00",
                 "total_minor": 10000, "line_items": []} for i in range(1, n_records + 1)]
    return {"manifest": {"episode_id": EP}, "invoices": invoices, "payments": [], "ledger": [], "fx": []}


def _issue(iid="A", affected=("R1",), evidence=(("D1", "D2"),)) -> dict:
    return {"issue_id": iid, "type": T, "affected_ids": list(affected), "expected_amount_minor": 10000,
            "currency": "KZT", "acceptable_evidence_sets": [list(s) for s in evidence]}


def _f(**kw) -> dict:
    f = {"type": T, "affected_ids": ["R1"], "claimed_amount_minor": 10000, "currency": "KZT",
         "evidence_ids": ["D1", "D2"], "explanation": "same entity, total and lines as D1"}
    f.update(kw)
    return f


def _run(findings, issues, steps=10, bundle=None):
    truth = {"issues": issues}
    return grade({"episode_id": EP, "findings": findings}, truth, bundle or _bundle(), steps, "submitted")


def _check(res, tp, fp, fn, reward, p=None, r=None, f1=None):
    m = res["metrics"]
    assert res["invalid_reason"] is None
    assert (m["tp"], m["fp"], m["fn"]) == (tp, fp, fn)
    assert abs(res["reward"] - reward) <= TOL
    for got, want in ((m["precision"], p), (m["recall"], r), (m["f1"], f1)):
        if want is not None:
            assert abs(got - want) <= TOL


A = _issue()


def test_01_one_correct():
    _check(_run([_f()], [A]), 1, 0, 0, 0.98, 1, 1, 1)


def test_02_silence_on_errored_books():
    _check(_run([], [A]), 0, 0, 1, -1.0, 0, 0, 0)


def test_03_silence_on_clean_books():
    _check(_run([], []), 0, 0, 0, 0.98, 0, 0, 1)


def test_04_false_accusation_on_clean_books():
    _check(_run([_f()], []), 0, 1, 0, -1.0, f1=0)


def test_05_duplicate_findings():
    res = _run([_f(), _f()], [A])
    _check(res, 1, 1, 0, 47 / 150, 0.5, 1, 2 / 3)
    assert res["reward"] <= _run([_f()], [A])["reward"]


@pytest.mark.parametrize("n,mut", [
    (6, {"type": "fx_error"}),
    (7, {"affected_ids": ["R2"]}),
    (8, {"claimed_amount_minor": 10001}),
    (9, {"currency": "USD"}),
    (10, {"evidence_ids": ["D1"]}),
    (11, {"evidence_ids": ["D1", "U-NOT-THERE"]}),
])
def test_06_to_11_single_wrong_field(n, mut):
    _check(_run([_f(**mut)], [A]), 0, 1, 1, -1.0, f1=0)


def test_12_flag_everything():
    spray = [_f(type=t, affected_ids=[f"R{i}"]) for i in range(1, 6) for t in TYPES]
    assert len(spray) == 20
    _check(_run(spray, [A]), 1, 19, 0, -871 / 1050, 1 / 20, 1, 2 / 21)


B = {"issue_id": "B", "type": "unapplied_payment", "affected_ids": ["R3"], "expected_amount_minor": 500,
     "currency": "KZT", "acceptable_evidence_sets": [["D3"]]}


def test_13_two_issues_one_found():
    _check(_run([_f()], [A, B]), 1, 0, 1, 47 / 150, 1, 0.5, 2 / 3)


def test_14_two_issues_one_found_one_false():
    _check(_run([_f(), _f(affected_ids=["R4"])], [A, B]), 1, 1, 1, -0.02, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("steps,reward", [(0, 1.0), (50, 0.90), (100, 0.80)])
def test_15_to_17_step_cost(steps, reward):
    _check(_run([_f()], [A], steps=steps), 1, 0, 0, reward)
    assert abs(_run([_f()], [A], steps=10)["reward"] - _run([_f()], [A], steps=50)["reward"] - 0.08) <= TOL


@pytest.mark.parametrize("findings", [
    [_f(affected_ids=[f"R{i % 5 + 1}"]) for i in range(31)],                      # 18: 31 findings
    [_f(evidence_ids=["D1", "D2"] + [f"X{i}" for i in range(9)])],                 # 19: 11 evidence ids
    [_f(explanation="x" * 1001)],                                                  # 20: 1001 ASCII chars
])
def test_18_to_20_structure_limits(findings):
    res = _run(findings, [A])
    assert res["invalid_reason"] is not None and res["reward"] == -1.0


def test_limits_are_inclusive():
    assert _run([_f(explanation="x" * 1000)], [A])["invalid_reason"] is None
    assert _run([_f(explanation="я" * 1000)], [A])["invalid_reason"] is None  # code points, not bytes
    assert _run([_f(evidence_ids=["D1", "D2"] + ["D3"] * 8)], [A])["invalid_reason"] is None
    thirty = [_f(affected_ids=[f"R{i % 5 + 1}"]) for i in range(30)]
    assert _run(thirty, [A])["invalid_reason"] is None


# ---------------------------------------------------------------- open examples M01-M12
AM = _issue(affected=("R1", "R2"))


def test_m01_affected_repeats_are_a_set():
    _check(_run([_f(affected_ids=["R2", "R1", "R1"])], [AM]), 1, 0, 0, 0.98)


def test_m02_extra_affected_record():
    _check(_run([_f(affected_ids=["R1", "R2", "R3"])], [AM]), 0, 1, 1, -1.0)


def test_m03_extra_existing_evidence_ok():
    _check(_run([_f(affected_ids=["R1", "R2"], evidence_ids=["D2", "D1", "D3"])], [AM]), 1, 0, 0, 0.98)


def test_m04_unknown_evidence_breaks_match():
    _check(_run([_f(affected_ids=["R1", "R2"], evidence_ids=["D1", "D2", "U"])], [AM]), 0, 1, 1, -1.0)


def test_m05_evidence_limit_before_dedup():
    res = _run([_f(affected_ids=["R1", "R2"], evidence_ids=["D2"] + ["D1"] * 10)], [AM])
    assert res["invalid_reason"] is not None and res["reward"] == -1.0


@pytest.mark.parametrize("cur", ["kzt", " KZT", "KZT "])
def test_m06_currency_exact(cur):
    _check(_run([_f(affected_ids=["R1", "R2"], currency=cur)], [AM]), 0, 1, 1, -1.0)


def test_m07_amount_off_by_one():
    _check(_run([_f(affected_ids=["R1", "R2"], claimed_amount_minor=10001)], [AM]), 0, 1, 1, -1.0)


def test_m08_published_tolerance_is_inclusive(monkeypatch):
    monkeypatch.setitem(matcher.TOLERANCE_MINOR, T, 1)
    ok = _run([_f(affected_ids=["R1", "R2"], claimed_amount_minor=10001)], [AM])
    bad = _run([_f(affected_ids=["R1", "R2"], claimed_amount_minor=10002)], [AM])
    assert ok["metrics"]["tp"] == 1 and bad["metrics"]["tp"] == 0


@pytest.mark.parametrize("amount", ["10000", True, 10000.0, 1e4, None])
def test_m09_amount_must_be_json_integer(amount):
    res = _run([_f(affected_ids=["R1", "R2"], claimed_amount_minor=amount)], [AM])
    assert res["invalid_reason"] is not None and res["reward"] == -1.0


def _edge_bundle_and_issues():
    # A needs evidence {D1}; B needs evidence {D2}. F0 carries D1+D2 (fits both), F1 carries D1 (fits A).
    a = {"issue_id": "A", "type": T, "affected_ids": ["R1"], "expected_amount_minor": 10000,
         "currency": "KZT", "acceptable_evidence_sets": [["D1"]]}
    b = dict(a, issue_id="B", acceptable_evidence_sets=[["D2"]])
    return a, b


def test_m10_maximum_not_greedy_matching():
    a, b = _edge_bundle_and_issues()
    f0, f1 = _f(evidence_ids=["D1", "D2"]), _f(evidence_ids=["D1"])
    pairs = matcher.match([f0, f1], [a, b], matcher.build_alias_map(_bundle()))
    assert pairs == [(0, "B"), (1, "A")]
    _check(_run([f0, f1], [a, b]), 2, 0, 0, 0.98)


def test_m11_tie_break_and_order_invariance():
    a, b = _edge_bundle_and_issues()
    f = _f(evidence_ids=["D1", "D2"])
    alias = matcher.build_alias_map(_bundle())
    assert matcher.match([f, copy.deepcopy(f)], [b, a], alias) == [(0, "A"), (1, "B")]
    assert _run([f, f], [a, b])["reward"] == _run([f, f], [b, a])["reward"]


def test_m12_identical_findings_not_merged():
    _check(_run([_f(), _f()], [A]), 1, 1, 0, 47 / 150)


def test_evidence_is_literal_subset_for_rows_of_one_transaction():
    """Test 10 with D1, D2 = two ledger rows of the SAME transaction: one row is not the set."""
    b = _bundle()
    b["ledger"] = [{"row_id": "L1", "txn_id": "T1"}, {"row_id": "L2", "txn_id": "T1"}]
    a = _issue(evidence=(("L1", "L2"),))
    assert _run([_f(evidence_ids=["L1"])], [a], bundle=b)["metrics"]["tp"] == 0
    assert _run([_f(evidence_ids=["L2", "L1"])], [a], bundle=b)["metrics"]["tp"] == 1


def test_canonical_ids_accept_documented_aliases():
    """Our truth names documents by doc_id and transactions by txn_id; the published
    aliases (invoice_id for its doc_id, any row of the transaction for its txn_id) count."""
    b = _bundle()
    b["ledger"] = [{"row_id": "L1", "txn_id": "T1"}, {"row_id": "L2", "txn_id": "T1"}]
    a = _issue(evidence=(("D1", "T1"),))
    assert _run([_f(evidence_ids=["R1", "L2"])], [a], bundle=b)["metrics"]["tp"] == 1
    assert _run([_f(evidence_ids=["R2", "L2"])], [a], bundle=b)["metrics"]["tp"] == 0  # R2 is another invoice


def test_fx_composite_key_is_valid_evidence():
    b = _bundle()
    b["fx"] = [{"fx_id": "FX1", "date": "2026-09-01", "source_currency": "USD", "target_currency": "KZT"}]
    a = _issue(evidence=(("D1", "FX1"),))
    assert _run([_f(evidence_ids=["D1", "2026-09-01:USD:KZT"])], [a], bundle=b)["metrics"]["tp"] == 1


def test_wrong_but_wellformed_values_are_fp_not_crash():
    res = _run([_f(type="totally_new_type"), _f(affected_ids=[]), _f(claimed_amount_minor=-5)], [A])
    _check(res, 0, 3, 1, -1.0)
