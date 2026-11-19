"""Derive double-entry ledger rows from documents (all rows in base currency).

Posting rules (published in policy.json and docs/DESIGN.md):
- Invoice:  Dr AR / Cr REVENUE, convert(total, rate(issue date)).
- Receipt:  Dr CASH convert(amount, rate(payment date));
            Cr AR per invoice_ref, relief = convert(cum_after) - convert(cum_before) at the
            invoice's booking rate (cumulative over receipts in timestamp order);
            Cr UNAPPLIED_CASH convert(unapplied, rate(payment date)) if anything is unapplied;
            FX_GAIN_LOSS takes the balancing difference.
- Refund:   Dr REFUNDS / Cr CASH, convert(amount, rate(refund date)).
- Background cash transactions: base currency, no document links.

``overrides`` lets error injection change *how* a document was posted (wrong rate date,
wrong posting timestamp) without touching the document itself.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from .. import money
from ..ids import IdFactory
from ..timeutil import fmt_ts, local_date
from .company import Company


def _row(ids: IdFactory, txn_id: str, key: str, order: int, ts: datetime, offset: str, account: str,
         debit: int, credit: int, currency: str, entity_id, invoice_id, payment_id) -> dict:
    return {
        "row_id": ids.derived("row", f"{txn_id}|{key}"),
        "txn_id": txn_id,
        "timestamp": fmt_ts(ts, offset),
        "account": account,
        "debit_minor": debit,
        "credit_minor": credit,
        "currency": currency,
        "entity_id": entity_id,
        "invoice_id": invoice_id,
        "payment_id": payment_id,
        "_sort": (int(ts.timestamp()), txn_id, order),
    }


def _rate(company: Company, currency: str, ts: datetime, override: dict) -> Decimal:
    if currency == company.base_currency:
        return Decimal(1)
    if "rate" in override:
        return money.parse_rate(override["rate"])
    day = date.fromisoformat(override["rate_date"]) if "rate_date" in override else local_date(ts, company.offset)
    return company.rate(currency, day)


def post_ledger(company: Company, ids: IdFactory, overrides: dict | None = None) -> list[dict]:
    overrides = overrides or {}
    base, off = company.base_currency, company.offset
    rows: list[dict] = []
    booking_rate: dict[str, Decimal] = {}

    for inv in sorted(company.invoices, key=lambda i: i["doc_id"]):
        ov = overrides.get(inv["doc_id"], {})
        issued = datetime.fromisoformat(inv["issued_at"])
        ts = datetime.fromisoformat(ov["posting_ts"]) if "posting_ts" in ov else issued
        rate = _rate(company, inv["currency"], issued, ov)
        booking_rate[inv["invoice_id"]] = rate
        amount = money.convert(inv["total_minor"], inv["currency"], base, rate)
        txn = company.txn_of[inv["doc_id"]]
        common = (base, inv["entity_id"], inv["invoice_id"], None)
        rows.append(_row(ids, txn, "AR", 0, ts, off, "AR", amount, 0, *common))
        rows.append(_row(ids, txn, "REVENUE", 1, ts, off, "REVENUE", 0, amount, *common))

    cum: dict[str, int] = {}
    invoices = company.invoice_by_id()
    ordered = sorted(company.payments, key=lambda p: (int(datetime.fromisoformat(p["timestamp"]).timestamp()), p["payment_id"]))
    for pay in ordered:
        ov = overrides.get(pay["doc_id"], {})
        real_ts = datetime.fromisoformat(pay["timestamp"])
        ts = datetime.fromisoformat(ov["posting_ts"]) if "posting_ts" in ov else real_ts
        cur = pay["currency"]
        rate = _rate(company, cur, real_ts, ov)
        txn = company.txn_of[pay["doc_id"]]
        eid, pid = pay["entity_id"], pay["payment_id"]
        if pay["kind"] == "refund":
            amount = money.convert(pay["amount_minor"], cur, base, rate)
            inv_id = pay["invoice_refs"][0]["invoice_id"]
            rows.append(_row(ids, txn, "REFUNDS", 0, ts, off, "REFUNDS", amount, 0, base, eid, inv_id, pid))
            rows.append(_row(ids, txn, "CASH", 1, ts, off, "CASH", 0, amount, base, eid, None, pid))
            continue
        cash = money.convert(pay["amount_minor"], cur, base, rate)
        rows.append(_row(ids, txn, "CASH", 0, ts, off, "CASH", cash, 0, base, eid, None, pid))
        applied = 0
        relief_total = 0
        for n, ref in enumerate(pay["invoice_refs"] or [], start=1):
            inv_id, a = ref["invoice_id"], ref["amount_minor"]
            inv = invoices[inv_id]
            r = booking_rate[inv_id]
            before = cum.get(inv_id, 0)
            relief = (money.convert(before + a, inv["currency"], base, r)
                      - money.convert(before, inv["currency"], base, r))
            cum[inv_id] = before + a
            applied += a
            relief_total += relief
            if relief:
                rows.append(_row(ids, txn, f"AR|{inv_id}", n, ts, off, "AR", 0, relief, base, eid, inv_id, pid))
        unapplied = pay["amount_minor"] - applied
        unapplied_base = money.convert(unapplied, cur, base, rate) if unapplied > 0 else 0
        if unapplied_base:
            rows.append(_row(ids, txn, "UNAPPLIED", 90, ts, off, "UNAPPLIED_CASH", 0, unapplied_base, base, eid, None, pid))
        diff = cash - relief_total - unapplied_base
        if diff > 0:
            rows.append(_row(ids, txn, "FXGL", 91, ts, off, "FX_GAIN_LOSS", 0, diff, base, eid, None, pid))
        elif diff < 0:
            rows.append(_row(ids, txn, "FXGL", 91, ts, off, "FX_GAIN_LOSS", -diff, 0, base, eid, None, pid))

    for bg in company.background:
        ts = datetime.fromisoformat(bg["timestamp"])
        txn = bg["txn_id"]
        rows.append(_row(ids, txn, "DR", 0, ts, off, bg["debit_account"], bg["amount_minor"], 0, base, None, None, None))
        rows.append(_row(ids, txn, "CR", 1, ts, off, bg["credit_account"], 0, bg["amount_minor"], base, None, None, None))

    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        del r["_sort"]
    return rows
