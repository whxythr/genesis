"""Accounting invariants and an independent detector that works on OPEN data only.

``analyze(bundle)`` re-derives every expected posting from the documents, fx.csv and
policy, and reports violations. It never looks at truth/provenance and does not reuse
the posting code, so it acts as an independent validator:

- on a clean history it must report **no** violations;
- after injection, the *detectable* violations (duplicate_invoice, unapplied_payment,
  fx_error, period_cutoff) must equal the truth issues exactly, and nothing else may
  be violated.

Amount convention ("independent amounts"): each issue's amount is computed with the
policy-correct FX rate, as if every other issue were already fixed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from .. import jsonutil, money
from ..timeutil import local_date, parse_ts, tz_from_offset

DETECTABLE = {
    "DUPLICATE_INVOICE": "duplicate_invoice",
    "UNAPPLIED_PAYMENT": "unapplied_payment",
    "FX_MISMATCH": "fx_error",
    "DOC_OUT_OF_PERIOD": "period_cutoff",
}
DOC_ACCOUNTS = {"AR", "REVENUE", "CASH", "UNAPPLIED_CASH", "FX_GAIN_LOSS", "REFUNDS"}
BACKGROUND_ACCOUNTS = {"OPEX", "PAYROLL", "BANK_FEES", "INTEREST_INCOME", "CASH"}


class _Ctx:
    def __init__(self, bundle: dict) -> None:
        m = bundle["manifest"]
        self.base = m["base_currency"]
        self.offset = m["timezone"]
        self.start = parse_ts(m["period_start"])
        self.end = parse_ts(m["period_end"])
        self.violations: list[dict] = []
        self.rates: dict[tuple[str, str], tuple[Decimal, str]] = {}
        for row in bundle["fx"]:
            key = (row["source_currency"], row["date"])
            if key in self.rates or row["target_currency"] != self.base:
                self.add("FX_TABLE", [row["fx_id"]])
                continue
            try:
                self.rates[key] = (money.parse_rate(row["rate"]), row["fx_id"])
            except ValueError:
                self.add("FX_TABLE", [row["fx_id"]])

    def add(self, code: str, ids: list[str], amount: int | None = None, evidence: list[str] | None = None) -> None:
        self.violations.append({"code": code, "ids": sorted(ids), "amount": amount,
                                "evidence": sorted(evidence or [])})

    def in_period(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def rate(self, currency: str, ts: datetime) -> tuple[Decimal, str | None] | None:
        """Policy rate and fx_id for an operation at ``ts`` (None if missing)."""
        if currency == self.base:
            return Decimal(1), None
        return self.rates.get((currency, local_date(ts, self.offset).isoformat()))


def _line_signature(inv: dict) -> str:
    lines = [{k: li.get(k) for k in ("description", "quantity", "unit_price_minor", "amount_minor")}
             for li in inv["line_items"]]
    return jsonutil.dumps([inv["entity_id"], inv["currency"], inv["total_minor"], lines])


def analyze(bundle: dict) -> dict:
    c = _Ctx(bundle)
    invoices = {i["invoice_id"]: i for i in bundle["invoices"]}
    payments = {p["payment_id"]: p for p in bundle["payments"]}

    # -- id uniqueness -----------------------------------------------------
    seen: dict[str, int] = defaultdict(int)
    for i in bundle["invoices"]:
        seen[i["doc_id"]] += 1
        seen[i["invoice_id"]] += 1
    for p in bundle["payments"]:
        seen[p["doc_id"]] += 1
        seen[p["payment_id"]] += 1
    for r in bundle["ledger"]:
        seen[r["row_id"]] += 1
    for r in bundle["fx"]:
        seen[r["fx_id"]] += 1
    for k in sorted(seen):
        if seen[k] > 1:
            c.add("DUP_ID", [k])
    entities = {i["entity_id"] for i in bundle["invoices"]} | {p["entity_id"] for p in bundle["payments"]}

    # -- rows and transactions ----------------------------------------------
    txns: dict[str, list[dict]] = defaultdict(list)
    for r in bundle["ledger"]:
        txns[r["txn_id"]].append(r)
        d, cr = r["debit_minor"], r["credit_minor"]
        if not (isinstance(d, int) and isinstance(cr, int) and d >= 0 and cr >= 0 and (d > 0) != (cr > 0)):
            c.add("ROW_SIDES", [r["row_id"]])
        if r["currency"] != c.base:
            c.add("ROW_CURRENCY", [r["row_id"]])
        try:
            ts = parse_ts(r["timestamp"])
            if not c.in_period(ts):
                c.add("POSTING_OUT_OF_PERIOD", [r["row_id"]])
        except ValueError:
            c.add("ROW_TIMESTAMP", [r["row_id"]])
        for col, pool in (("invoice_id", invoices), ("payment_id", payments)):
            if r[col] is not None and r[col] not in pool:
                c.add("REF_UNRESOLVED", [r["row_id"]])
        if r["entity_id"] is not None and r["entity_id"] not in entities:
            c.add("REF_UNRESOLVED", [r["row_id"]])

    invoice_txn: dict[str, list[str]] = defaultdict(list)
    payment_txn: dict[str, list[str]] = defaultdict(list)
    for txn_id in sorted(txns):
        rows = txns[txn_id]
        if sum(r["debit_minor"] or 0 for r in rows) != sum(r["credit_minor"] or 0 for r in rows):
            c.add("TXN_UNBALANCED", [txn_id])
        if len({r["timestamp"] for r in rows}) != 1:
            c.add("TXN_TIMESTAMPS", [txn_id])
        pids = {r["payment_id"] for r in rows if r["payment_id"]}
        iids = {r["invoice_id"] for r in rows if r["invoice_id"]}
        if len(pids) == 1:
            payment_txn[next(iter(pids))].append(txn_id)
            if any(r["payment_id"] is None for r in rows):
                c.add("TXN_LINKS", [txn_id])
        elif len(pids) > 1:
            c.add("TXN_LINKS", [txn_id])
        elif len(iids) == 1:
            if any(r["invoice_id"] is None for r in rows):
                c.add("TXN_LINKS", [txn_id])
            invoice_txn[next(iter(iids))].append(txn_id)
        elif iids:
            c.add("TXN_LINKS", [txn_id])
        else:
            accts = [r["account"] for r in rows]
            if len(rows) != 2 or any(a not in BACKGROUND_ACCOUNTS for a in accts) or any(r["entity_id"] for r in rows):
                c.add("BACKGROUND_TXN", [txn_id])

    # -- invoices -----------------------------------------------------------
    booked: dict[str, tuple[Decimal, int]] = {}
    for inv_id in sorted(invoices):
        inv = invoices[inv_id]
        ok_lines = all(li["amount_minor"] == li["quantity"] * li["unit_price_minor"] for li in inv["line_items"])
        if not ok_lines or inv["total_minor"] != sum(li["amount_minor"] for li in inv["line_items"]):
            c.add("INVOICE_TOTAL", [inv_id])
        issued = parse_ts(inv["issued_at"])
        if parse_ts(inv["due_at"]) < issued:
            c.add("INVOICE_DUE", [inv_id])
        t_ids = invoice_txn.get(inv_id, [])
        if len(t_ids) != 1:
            c.add("INVOICE_POSTING_COUNT", [inv_id])
            continue
        rows = txns[t_ids[0]]
        rr = c.rate(inv["currency"], issued)
        if rr is None:
            c.add("FX_RATE_MISSING", [inv_id])
            continue
        rate, fx_id = rr
        correct = money.convert(inv["total_minor"], inv["currency"], c.base, rate)
        booked[inv_id] = (rate, correct)
        ar = [r for r in rows if r["account"] == "AR"]
        rev = [r for r in rows if r["account"] == "REVENUE"]
        if len(rows) != 2 or len(ar) != 1 or len(rev) != 1 or ar[0]["debit_minor"] != rev[0]["credit_minor"]:
            c.add("INVOICE_POSTING_SHAPE", [inv_id])
            continue
        if any(r["entity_id"] != inv["entity_id"] for r in rows):
            c.add("ENTITY_MISMATCH", [inv_id])
        posted = ar[0]["debit_minor"]
        if posted != correct:
            if inv["currency"] == c.base:
                c.add("AMOUNT_MISMATCH", [inv_id])
            else:
                c.add("FX_MISMATCH", [inv_id], abs(posted - correct), [inv["doc_id"], fx_id, t_ids[0]])
        posted_ts = parse_ts(rows[0]["timestamp"])
        if not c.in_period(issued):
            c.add("DOC_OUT_OF_PERIOD", [inv_id], correct, [inv["doc_id"], t_ids[0]])
        elif posted_ts != issued:
            c.add("POSTING_TS_MISMATCH", [inv_id])

    # -- duplicates ---------------------------------------------------------
    groups: dict[str, list[dict]] = defaultdict(list)
    for inv in bundle["invoices"]:
        groups[_line_signature(inv)].append(inv)
    dup_orig: dict[str, str] = {}  # duplicate invoice_id -> original doc_id
    for sig in sorted(groups):
        group = sorted(groups[sig], key=lambda i: (parse_ts(i["issued_at"]), i["doc_id"]))
        for dup in group[1:]:
            dup_orig[dup["invoice_id"]] = group[0]["doc_id"]
            t_ids = invoice_txn.get(dup["invoice_id"], [])
            amount = booked.get(dup["invoice_id"], (None, None))[1]
            c.add("DUPLICATE_INVOICE", [dup["invoice_id"]], amount,
                  [group[0]["doc_id"], dup["doc_id"]] + t_ids[:1])

    # -- payments -----------------------------------------------------------
    applied: dict[str, int] = defaultdict(int)
    refunded: dict[str, int] = defaultdict(int)
    cum: dict[str, int] = defaultdict(int)
    ordered = sorted(payments.values(), key=lambda p: (parse_ts(p["timestamp"]), p["payment_id"]))
    for pay in ordered:
        pid = pay["payment_id"]
        ts = parse_ts(pay["timestamp"])
        refs = pay["invoice_refs"]
        bad_ref = False
        for ref in refs or []:
            inv = invoices.get(ref["invoice_id"])
            if inv is None or inv["entity_id"] != pay["entity_id"] or inv["currency"] != pay["currency"] or ref["amount_minor"] <= 0:
                c.add("REF_UNRESOLVED", [pid])
                bad_ref = True
        t_ids = payment_txn.get(pid, [])
        if len(t_ids) != 1:
            c.add("PAYMENT_POSTING_COUNT", [pid])
            continue
        if bad_ref:
            continue
        txn_id = t_ids[0]
        rows = txns[txn_id]
        if any(r["entity_id"] != pay["entity_id"] for r in rows):
            c.add("ENTITY_MISMATCH", [pid])
        rr = c.rate(pay["currency"], ts)
        if rr is None:
            c.add("FX_RATE_MISSING", [pid])
            continue
        rate, fx_id = rr
        correct_cash = money.convert(pay["amount_minor"], pay["currency"], c.base, rate)
        if not c.in_period(ts):
            c.add("DOC_OUT_OF_PERIOD", [pid], correct_cash, [pay["doc_id"], txn_id] + ([fx_id] if fx_id else []))
        elif parse_ts(rows[0]["timestamp"]) != ts:
            c.add("POSTING_TS_MISMATCH", [pid])

        cash_side = "debit_minor" if pay["kind"] == "receipt" else "credit_minor"
        cash_rows = [r for r in rows if r["account"] == "CASH"]
        if len(cash_rows) != 1 or not cash_rows[0][cash_side]:
            c.add("PAYMENT_POSTING_SHAPE", [pid])
            continue
        posted_cash = cash_rows[0][cash_side]
        used_rates = [rate]
        if posted_cash != correct_cash:
            if pay["currency"] == c.base:
                c.add("AMOUNT_MISMATCH", [pid])
            else:
                c.add("FX_MISMATCH", [pid], abs(posted_cash - correct_cash), [pay["doc_id"], fx_id, txn_id])
                used_rates = [r_ for (cur, _d), (r_, _f) in sorted(c.rates.items())
                              if cur == pay["currency"] and money.convert(pay["amount_minor"], cur, c.base, r_) == posted_cash]

        if pay["kind"] == "refund":
            if not refs or len(refs) != 1 or refs[0]["amount_minor"] != pay["amount_minor"]:
                c.add("REFUND_REFS", [pid])
                continue
            inv_id = refs[0]["invoice_id"]
            refunded[inv_id] += pay["amount_minor"]
            if refunded[inv_id] > applied[inv_id]:
                c.add("REFUND_EXCEEDS", [pid])
            ref_rows = [r for r in rows if r["account"] == "REFUNDS"]
            if len(rows) != 2 or len(ref_rows) != 1 or ref_rows[0]["debit_minor"] != posted_cash or ref_rows[0]["invoice_id"] != inv_id:
                c.add("PAYMENT_POSTING_SHAPE", [pid])
            continue

        if pay["kind"] != "receipt":
            c.add("PAYMENT_KIND", [pid])
            continue
        total_applied = sum(r["amount_minor"] for r in refs or [])
        if total_applied > pay["amount_minor"]:
            c.add("REFS_EXCEED_AMOUNT", [pid])
            continue
        # expected AR relief per invoice ref (cumulative, booking rate)
        expected_ar: dict[str, int] = defaultdict(int)
        for ref in refs or []:
            inv_id, a = ref["invoice_id"], ref["amount_minor"]
            inv = invoices[inv_id]
            applied[inv_id] += a
            if applied[inv_id] > inv["total_minor"]:
                c.add("OVERAPPLIED", [inv_id])
            if inv_id not in booked:
                continue
            brate = booked[inv_id][0]
            before = cum[inv_id]
            expected_ar[inv_id] += (money.convert(before + a, inv["currency"], c.base, brate)
                                    - money.convert(before, inv["currency"], c.base, brate))
            cum[inv_id] = before + a
        actual_ar: dict[str, int] = defaultdict(int)
        for r in rows:
            if r["account"] == "AR":
                actual_ar[r["invoice_id"]] += (r["credit_minor"] or 0) - (r["debit_minor"] or 0)
        for inv_id in sorted(set(expected_ar) | set(actual_ar)):
            if expected_ar.get(inv_id, 0) != actual_ar.get(inv_id, 0):
                c.add("AR_RELIEF_MISMATCH", [pid, inv_id])
        unapplied = pay["amount_minor"] - total_applied
        posted_unapplied = sum((r["credit_minor"] or 0) - (r["debit_minor"] or 0) for r in rows if r["account"] == "UNAPPLIED_CASH")
        if not any(posted_unapplied == (money.convert(unapplied, pay["currency"], c.base, r_) if unapplied > 0 else 0)
                   for r_ in used_rates):
            c.add("UNAPPLIED_LINE_MISMATCH", [pid])
        # cash applied to a duplicate invoice (which must be voided) is effectively unapplied
        to_dup = [ref for ref in refs or [] if ref["invoice_id"] in dup_orig]
        effective = unapplied + sum(ref["amount_minor"] for ref in to_dup)
        if effective > 0:
            amount = money.convert(effective, pay["currency"], c.base, rate)
            ev = [pay["doc_id"], txn_id]
            for ref in to_dup:
                ev += [dup_orig[ref["invoice_id"]], invoices[ref["invoice_id"]]["doc_id"]]
            if fx_id and (to_dup or posted_cash != correct_cash):
                ev.append(fx_id)
            c.add("UNAPPLIED_PAYMENT", [pid], amount, ev)
        if any(r["account"] not in DOC_ACCOUNTS for r in rows):
            c.add("PAYMENT_POSTING_SHAPE", [pid])

    issues = []
    other = []
    for v in c.violations:
        if v["code"] in DETECTABLE:
            issues.append({"type": DETECTABLE[v["code"]], "affected_ids": v["ids"],
                           "expected_amount_minor": v["amount"], "currency": c.base,
                           "evidence": v["evidence"]})
        else:
            other.append(v)
    issues.sort(key=lambda i: (i["type"], i["affected_ids"]))
    return {"issues": issues, "violations": other}


def check_clean(bundle: dict) -> list[dict]:
    """All violations (detectable or not) — must be empty for a clean history."""
    res = analyze(bundle)
    return res["violations"] + [{"code": i["type"], "ids": i["affected_ids"]} for i in res["issues"]]


def compare_with_truth(bundle: dict, truth: dict) -> list[str]:
    """Problems found when checking that open data shows exactly the declared issues."""
    res = analyze(bundle)
    problems = [f"unexplained violation {v['code']} {v['ids']}" for v in res["violations"]]
    detected = {(i["type"], tuple(i["affected_ids"])): i for i in res["issues"]}
    declared = {(t["type"], tuple(sorted(t["affected_ids"]))): t for t in truth["issues"]}
    for key in sorted(set(detected) - set(declared)):
        problems.append(f"undeclared issue {key}")
    for key in sorted(set(declared) - set(detected)):
        problems.append(f"declared issue not visible {key}")
    for key in sorted(set(detected) & set(declared)):
        d, t = detected[key], declared[key]
        if d["expected_amount_minor"] != t["expected_amount_minor"] or d["currency"] != t["currency"]:
            problems.append(f"amount mismatch {key}: open={d['expected_amount_minor']} truth={t['expected_amount_minor']}")
        if not any(set(s) <= set(d["evidence"]) or set(d["evidence"]) <= set(s) for s in t["acceptable_evidence_sets"]):
            problems.append(f"evidence mismatch {key}")
    return problems
