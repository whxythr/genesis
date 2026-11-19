"""Two scenarios of interacting errors.

A. ``partial_fx_payment_at_cutoff`` — a partial receipt in a foreign currency that really
   happened just AFTER the period end (often still reading as the last day in the bank's
   offset) is booked in this period (period_cutoff), converted with the rate of the
   booking day instead of the operation day (fx_error), and only partly applied
   (unapplied_payment). Three issues on the same payment_id, each with an independent
   amount at the policy-correct rate:
     period_cutoff     = convert(P, rate(doc day))
     fx_error          = |convert(P, rate(booking day)) - convert(P, rate(doc day))|
     unapplied_payment = convert(P - applied, rate(doc day))
   The UNAPPLIED_CASH line in the ledger is posted at the WRONG rate, so the unapplied
   amount cannot be copied from the ledger: the evidence must include the fx row.

B. ``partially_paid_duplicate`` — a legitimate partial receipt for an invoice exists in
   the clean history; the invoice is then duplicated and the receipt is applied to the
   duplicate instead of the original. Two issues:
     duplicate_invoice = convert(total, rate(duplicate day))
     unapplied_payment = convert(amount applied to the duplicate, rate(payment day)),
   because cash applied to an invoice that must be voided is effectively unapplied
   (published rule). Evidence must show both invoices (to prove the duplicate) and
   the payment with its posting.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from .. import jsonutil, money
from ..ids import IdFactory
from ..timeutil import fmt_ts, local_date, offset_minutes
from .company import Company


def _payments_touching(company: Company) -> set[str]:
    return {r["invoice_id"] for p in company.payments for r in (p["invoice_refs"] or [])}


def setup(company: Company, ids: IdFactory, rng: random.Random, plan: dict) -> dict:
    """Legitimate pre-conditions that belong to the CLEAN history (checked by invariants)."""
    state: dict = {}
    if "partially_paid_duplicate" in plan["scenarios"]:
        touched = _payments_touching(company)
        cands = [i for i in sorted(company.invoices, key=lambda i: i["invoice_id"])
                 if i["invoice_id"] not in touched and i["total_minor"] >= 1000]
        rng.shuffle(cands)
        used = {int(datetime.fromisoformat(p["timestamp"]).timestamp()) for p in company.payments}
        for inv in cands:
            issued = datetime.fromisoformat(inv["issued_at"])
            delta = timedelta(seconds=rng.randint(3600, 48 * 3600))
            earliest = issued + delta + timedelta(hours=2)
            latest = company.period_end - timedelta(minutes=10)
            if earliest >= latest:
                continue
            ts = earliest + timedelta(seconds=rng.randrange(int((latest - earliest).total_seconds())))
            while int(ts.timestamp()) in used:
                ts += timedelta(seconds=1)
            if ts >= latest:
                continue
            amount = max(1, inv["total_minor"] * rng.randint(20, 80) // 100)
            ent = next(e for e in company.entities if e["entity_id"] == inv["entity_id"])
            pay = {
                "doc_id": ids.new("doc"),
                "payment_id": ids.new("payment"),
                "kind": "receipt",
                "entity_id": inv["entity_id"],
                "timestamp": fmt_ts(ts, ent["bank_offset"]),
                "currency": inv["currency"],
                "amount_minor": amount,
                "invoice_refs": [{"invoice_id": inv["invoice_id"], "amount_minor": amount}],
            }
            company.payments.append(pay)
            company.txn_of[pay["doc_id"]] = ids.new("txn")
            company.tags[pay["doc_id"]] = ["scenario_setup", "split"]
            state["partially_paid_duplicate"] = {"invoice_id": inv["invoice_id"], "payment_id": pay["payment_id"],
                                                 "delta_s": int(delta.total_seconds())}
            break
    return state


def apply(inj, plan: dict, state: dict) -> None:
    if "partial_fx_payment_at_cutoff" in plan["scenarios"]:
        _partial_fx_payment_at_cutoff(inj)
    if "partially_paid_duplicate" in plan["scenarios"] and "partially_paid_duplicate" in state:
        _partially_paid_duplicate(inj, state["partially_paid_duplicate"])


def _partial_fx_payment_at_cutoff(inj) -> bool:
    c, rng, ids = inj.c, inj.rng, inj.ids
    touched = _payments_touching(c)
    fx_id = {(r["source_currency"], r["date"]): r["fx_id"] for r in c.fx_rows}
    cands = [i for i in sorted(c.invoices, key=lambda i: i["invoice_id"])
             if i["currency"] != c.base_currency and i["invoice_id"] not in touched | inj.inj.locked
             and i["total_minor"] >= 1000]
    rng.shuffle(cands)
    for inv in cands:
        cur = inv["currency"]
        ent = next(e for e in c.entities if e["entity_id"] == inv["entity_id"])
        display = ent["bank_offset"]
        gap = (offset_minutes(c.offset) - offset_minutes(display)) * 60
        if gap > 600 and rng.random() < 0.6:
            ts = c.period_end + timedelta(seconds=rng.randint(60, gap - 60))
        else:
            ts = c.period_end + timedelta(seconds=rng.randint(60, 30 * 3600))
        ts = inj.unique_payment_ts(ts)
        posting = c.period_end - timedelta(seconds=rng.randint(1, 6 * 3600))
        right_day, wrong_day = local_date(ts, c.offset), local_date(posting, c.offset)
        if (cur, wrong_day.isoformat()) not in c.rates or (cur, right_day.isoformat()) not in c.rates:
            continue
        amount = max(2, inv["total_minor"] * rng.randint(30, 80) // 100)
        if rng.random() < 0.35:
            refs = None
            applied = 0
        else:
            applied = max(1, amount * rng.randint(20, 80) // 100)
            refs = [{"invoice_id": inv["invoice_id"], "amount_minor": applied}]
        right_rate, wrong_rate = c.rate(cur, right_day), c.rate(cur, wrong_day)
        cash_right = money.convert(amount, cur, c.base_currency, right_rate)
        cash_wrong = money.convert(amount, cur, c.base_currency, wrong_rate)
        unapplied = money.convert(amount - applied, cur, c.base_currency, right_rate)
        if cash_right == cash_wrong or unapplied <= 0:
            continue
        pay = {
            "doc_id": ids.new("doc"),
            "payment_id": ids.new("payment"),
            "kind": "receipt",
            "entity_id": inv["entity_id"],
            "timestamp": fmt_ts(ts, display),
            "currency": cur,
            "amount_minor": amount,
            "invoice_refs": refs,
        }
        c.payments.append(pay)
        txn = ids.new("txn")
        c.txn_of[pay["doc_id"]] = txn
        c.tags[pay["doc_id"]] = ["injected", "scenario_A"]
        inj.inj.overrides[pay["doc_id"]] = {"posting_ts": fmt_ts(posting, c.offset), "rate_date": wrong_day.isoformat()}
        inj.inj.locked |= {pay["payment_id"], inv["invoice_id"]}
        fx_right = fx_id[(cur, right_day.isoformat())]
        pid, base = pay["payment_id"], c.base_currency
        scenario = "partial_fx_payment_at_cutoff"
        inj.inj.issues += [
            {"type": "period_cutoff", "affected_ids": [pid], "expected_amount_minor": cash_right, "currency": base,
             "acceptable_evidence_sets": [sorted([pay["doc_id"], txn, fx_right])]},
            {"type": "fx_error", "affected_ids": [pid], "expected_amount_minor": abs(cash_wrong - cash_right),
             "currency": base, "acceptable_evidence_sets": [sorted([pay["doc_id"], txn, fx_right])]},
            {"type": "unapplied_payment", "affected_ids": [pid], "expected_amount_minor": unapplied, "currency": base,
             "acceptable_evidence_sets": [sorted([pay["doc_id"], txn, fx_right])]},
        ]
        common = {"affected_ids": [pid], "scenario": scenario, "target_doc_id": pay["doc_id"]}
        inj.inj.changes += [
            dict(common, type="period_cutoff", action="insert_document+override_posting", field="posting_ts",
                 before=None, after=jsonutil.deep_copy(pay), posting_ts=fmt_ts(posting, c.offset),
                 reason="receipt dated after the period end was booked on the last day of the period"),
            dict(common, type="fx_error", action="override_posting", field="rate_date",
                 before=right_day.isoformat(), after=wrong_day.isoformat(),
                 posted_minor={"before": cash_right, "after": cash_wrong},
                 reason="converted with the booking-day rate instead of the operation-day rate"),
            dict(common, type="unapplied_payment", action="document_state", field="invoice_refs",
                 before=[{"invoice_id": inv["invoice_id"], "amount_minor": amount}], after=refs,
                 reason="only part of the receipt was applied to the invoice it pays"),
        ]
        return True
    return False


def _partially_paid_duplicate(inj, st: dict) -> bool:
    c = inj.c
    invoices = c.invoice_by_id()
    orig = invoices[st["invoice_id"]]
    pay = next(p for p in c.payments if p["payment_id"] == st["payment_id"])
    if not inj.duplicate_invoice(forced=(orig, timedelta(seconds=st["delta_s"]))):
        return False
    dup = inj.last_duplicate
    before = jsonutil.deep_copy(pay["invoice_refs"])
    pay["invoice_refs"] = [{"invoice_id": dup["invoice_id"], "amount_minor": r["amount_minor"]} for r in before]
    inj.inj.locked.add(pay["payment_id"])
    applied_to_dup = sum(r["amount_minor"] for r in pay["invoice_refs"])
    ts = datetime.fromisoformat(pay["timestamp"])
    evidence = [pay["doc_id"], c.txn_of[pay["doc_id"]], orig["doc_id"], dup["doc_id"]]
    if pay["currency"] != c.base_currency:
        day = local_date(ts, c.offset).isoformat()
        evidence.append(next(r["fx_id"] for r in c.fx_rows if r["source_currency"] == pay["currency"] and r["date"] == day))
    inj.inj.issues.append({
        "type": "unapplied_payment",
        "affected_ids": [pay["payment_id"]],
        "expected_amount_minor": inj._base_amount(applied_to_dup, pay["currency"], ts),
        "currency": c.base_currency,
        "acceptable_evidence_sets": [sorted(evidence)],
    })
    inj.inj.changes.append({
        "type": "unapplied_payment",
        "affected_ids": [pay["payment_id"]],
        "scenario": "partially_paid_duplicate",
        "action": "modify_document",
        "target_doc_id": pay["doc_id"],
        "field": "invoice_refs",
        "before": before,
        "after": jsonutil.deep_copy(pay["invoice_refs"]),
        "reason": "partial receipt applied to the duplicate invoice instead of the original; cash on a "
                  "duplicate that must be voided is effectively unapplied",
    })
    for ch in inj.inj.changes:
        if ch["type"] == "duplicate_invoice" and ch["affected_ids"] == [dup["invoice_id"]]:
            ch["scenario"] = "partially_paid_duplicate"
    return True
