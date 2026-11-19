"""Evidence sufficiency: re-derive each truth issue from ONLY the records in its
evidence set (plus the always-visible manifest and policy).

For every acceptable evidence set the derivation must reproduce the issue's type,
affected_ids and expected amount exactly; otherwise the evidence is not sufficient.
All records named by an evidence set are reachable through the agent's tools
(get_document, list_records(filter={"txn_id": ...}), get_record("fx", fx_id)).
"""

from __future__ import annotations

from .. import jsonutil, money
from ..timeutil import local_date, parse_ts


class Insufficient(ValueError):
    pass


def _view(bundle: dict, ids: list[str]) -> dict:
    want = set(ids)
    docs = [("invoice", d) for d in bundle["invoices"] if d["doc_id"] in want]
    docs += [("payment", d) for d in bundle["payments"] if d["doc_id"] in want]
    rows = [r for r in bundle["ledger"] if r["txn_id"] in want]
    fx = [r for r in bundle["fx"] if r["fx_id"] in want]
    found = {d["doc_id"] for _, d in docs} | {r["txn_id"] for r in rows} | {r["fx_id"] for r in fx}
    if found != want:
        raise Insufficient("evidence id not found")
    return {"docs": docs, "rows": rows, "fx": fx}


def _sig(inv: dict) -> str:
    return jsonutil.dumps([inv["entity_id"], inv["currency"], inv["total_minor"], inv["line_items"]])


def _doc_ts(kind: str, doc: dict):
    return parse_ts(doc["issued_at"] if kind == "invoice" else doc["timestamp"])


def _posted(kind: str, doc: dict, rows: list[dict]) -> int:
    if kind == "invoice":
        mine = [r for r in rows if r["invoice_id"] == doc["invoice_id"] and r["payment_id"] is None and r["account"] == "AR"]
        if len(mine) != 1:
            raise Insufficient("invoice posting not in evidence")
        return mine[0]["debit_minor"]
    mine = [r for r in rows if r["payment_id"] == doc["payment_id"] and r["account"] == "CASH"]
    if len(mine) != 1:
        raise Insufficient("payment posting not in evidence")
    return mine[0]["debit_minor"] or mine[0]["credit_minor"]


def derive(issue_type: str, evidence: list[str], bundle: dict) -> dict:
    v = _view(bundle, evidence)
    m = bundle["manifest"]
    start, end, tz = parse_ts(m["period_start"]), parse_ts(m["period_end"]), m["timezone"]
    docs, rows = v["docs"], v["rows"]

    if issue_type == "duplicate_invoice":
        invs = [d for k, d in docs if k == "invoice"]
        if len(invs) != 2 or _sig(invs[0]) != _sig(invs[1]):
            raise Insufficient("need two identical invoices")
        orig, dup = sorted(invs, key=lambda d: (parse_ts(d["issued_at"]), d["doc_id"]))
        return {"affected_ids": [dup["invoice_id"]], "amount": _posted("invoice", dup, rows)}

    if issue_type == "unapplied_payment":
        pays = [d for k, d in docs if k == "payment"]
        if len(pays) != 1 or pays[0]["kind"] != "receipt":
            raise Insufficient("need one receipt")
        p = pays[0]
        applied = sum(r["amount_minor"] for r in p["invoice_refs"] or [])
        if applied >= p["amount_minor"]:
            raise Insufficient("payment fully applied")
        line = sum((r["credit_minor"] or 0) for r in rows if r["payment_id"] == p["payment_id"] and r["account"] == "UNAPPLIED_CASH")
        if not line:
            raise Insufficient("unapplied line not in evidence")
        return {"affected_ids": [p["payment_id"]], "amount": line}

    if issue_type == "fx_error":
        if len(docs) != 1 or len(v["fx"]) != 1:
            raise Insufficient("need one document and one fx row")
        kind, d = docs[0]
        fx = v["fx"][0]
        ts = _doc_ts(kind, d)
        if fx["source_currency"] != d["currency"] or fx["date"] != local_date(ts, tz).isoformat():
            raise Insufficient("fx row is not the policy rate")
        amount = d["total_minor"] if kind == "invoice" else d["amount_minor"]
        correct = money.convert(amount, d["currency"], m["base_currency"], fx["rate"])
        diff = abs(_posted(kind, d, rows) - correct)
        if diff == 0:
            raise Insufficient("conversion is correct")
        return {"affected_ids": [d["invoice_id" if kind == "invoice" else "payment_id"]], "amount": diff}

    if issue_type == "period_cutoff":
        if len(docs) != 1:
            raise Insufficient("need one document")
        kind, d = docs[0]
        ts = _doc_ts(kind, d)
        if start <= ts < end:
            raise Insufficient("document inside period")
        posted_rows = [r for r in rows if (r["invoice_id"] if kind == "invoice" else r["payment_id"]) == d[kind + "_id"]]
        if not posted_rows or not all(start <= parse_ts(r["timestamp"]) < end for r in posted_rows):
            raise Insufficient("posting not inside period")
        fx_rows = v["fx"]
        if d["currency"] != m["base_currency"] and fx_rows:
            fx = fx_rows[0]
            if fx["source_currency"] != d["currency"] or fx["date"] != local_date(ts, tz).isoformat():
                raise Insufficient("fx row is not the policy rate")
            amount = d["total_minor"] if kind == "invoice" else d["amount_minor"]
            value = money.convert(amount, d["currency"], m["base_currency"], fx["rate"])
        else:
            value = _posted(kind, d, rows)
        return {"affected_ids": [d[kind + "_id"]], "amount": value}

    raise Insufficient("unknown type")


def check_sufficiency(bundle: dict, truth: dict) -> list[str]:
    problems = []
    for issue in truth["issues"]:
        for ev in issue["acceptable_evidence_sets"]:
            try:
                got = derive(issue["type"], ev, bundle)
            except Insufficient as e:
                problems.append(f"{issue['issue_id']}: evidence insufficient ({e})")
                continue
            if got["affected_ids"] != sorted(issue["affected_ids"]) or got["amount"] != issue["expected_amount_minor"]:
                problems.append(f"{issue['issue_id']}: evidence derives {got}, truth says "
                                f"{issue['affected_ids']} {issue['expected_amount_minor']}")
    return problems
