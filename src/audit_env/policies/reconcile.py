"""Reconcile: a simple rule-based auditor that uses ONLY the environment tools.

1. Page through invoices, payments, fx and the ledger (budget-aware).
2. Duplicates: invoices with equal (entity, currency, total) -> fetch documents and
   compare line items; the later one is the duplicate.
3. Unapplied: receipts whose invoice_refs are null or sum to less than the amount.
4. FX: re-convert every foreign document at the policy rate and compare with the
   ledger posting (AR debit for invoices, CASH for receipts).
5. Cutoff: documents dated outside the period that are posted in the ledger.

``ledger_amounts=True`` copies unapplied / cutoff amounts from the ledger posting
instead of recomputing them at the policy rate (and ignores the duplicate-application
rule) — the mistake the interacting scenarios are built to expose.
``naive_dates=True`` reproduces a common mistake (reads the calendar date straight
from the timestamp text instead of converting to the company timezone). It is used
in the failure analysis, not as an official baseline.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .. import money
from ..timeutil import local_date, parse_ts
from .base import Policy, submit


class ReconcilePolicy(Policy):
    name = "reconcile"

    def __init__(self, naive_dates: bool = False, ledger_amounts: bool = False) -> None:
        self.naive_dates = naive_dates
        self.ledger_amounts = ledger_amounts
        if naive_dates:
            self.name = "reconcile_naive"
        elif ledger_amounts:
            self.name = "reconcile_ledger_amounts"

    def reset(self, obs: dict, truth: dict | None = None) -> None:
        super().reset(obs)
        self.manifest = obs["manifest"]
        self.budget = obs["steps_remaining"]
        self.used = 0
        self._gen = self._run()
        self._started = False

    def act(self, obs: dict) -> dict:
        if not self._started:
            self._started = True
            action = next(self._gen)
        else:
            action = self._gen.send(obs)
        self.used += 1
        return action

    # -- helpers ---------------------------------------------------------------
    def _left(self) -> int:
        return self.budget - self.used - 1  # keep one step for submit

    def _fetch_all(self, table: str):
        out, cursor = [], None
        while self._left() > 0:
            obs = yield {"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}}
            if not obs.get("ok"):
                break
            out += obs["result"]["records"]
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        return out

    def _day(self, ts_text: str) -> str:
        if self.naive_dates:
            return ts_text[:10]
        return local_date(parse_ts(ts_text), self.manifest["timezone"]).isoformat()

    def _inside(self, ts_text: str) -> bool:
        if self.naive_dates:
            day = ts_text[:10]
            return self.manifest["period_start"][:10] <= day < self.manifest["period_end"][:10]
        return parse_ts(self.manifest["period_start"]) <= parse_ts(ts_text) < parse_ts(self.manifest["period_end"])

    # -- main loop -------------------------------------------------------------
    def _run(self):
        base = self.manifest["base_currency"]
        invoices = yield from self._fetch_all("invoices")
        payments = yield from self._fetch_all("payments")
        fx = yield from self._fetch_all("fx")
        ledger = yield from self._fetch_all("ledger")

        rates: dict[tuple[str, str], tuple[Decimal, str]] = {
            (r["source_currency"], r["date"]): (money.parse_rate(r["rate"]), r["fx_id"]) for r in fx}
        inv_txn: dict[str, list[dict]] = defaultdict(list)
        pay_txn: dict[str, list[dict]] = defaultdict(list)
        for r in ledger:
            if r["payment_id"]:
                pay_txn[r["payment_id"]].append(r)
            elif r["invoice_id"]:
                inv_txn[r["invoice_id"]].append(r)

        def rate(cur: str, ts_text: str):
            if cur == base:
                return Decimal(1), None
            return rates.get((cur, self._day(ts_text)))

        findings: list[dict] = []

        def add(t, aid, amount, evidence):
            findings.append({"type": t, "affected_ids": [aid], "claimed_amount_minor": int(amount),
                             "currency": base, "evidence_ids": [e for e in evidence if e][:10]})

        # duplicates
        dup_orig: dict[str, str] = {}  # duplicate invoice_id -> original doc_id
        dup_doc: dict[str, str] = {}
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for inv in invoices:
            groups[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(inv)
        for key in sorted(groups):
            group = groups[key]
            if len(group) < 2:
                continue
            docs = {}
            for inv in sorted(group, key=lambda i: i["doc_id"]):
                if self._left() <= 0:
                    break
                obs = yield {"tool": "get_document", "args": {"doc_id": inv["doc_id"]}}
                if obs.get("ok"):
                    docs[inv["invoice_id"]] = obs["result"]["document"]
            by_lines: dict[str, list[dict]] = defaultdict(list)
            for d in docs.values():
                by_lines[repr(d["line_items"])].append(d)
            for same in by_lines.values():
                same.sort(key=lambda d: (parse_ts(d["issued_at"]), d["doc_id"]))
                for dup in same[1:]:
                    dup_orig[dup["invoice_id"]] = same[0]["doc_id"]
                    dup_doc[dup["invoice_id"]] = dup["doc_id"]
                    rr = rate(dup["currency"], dup["issued_at"])
                    if rr is None:
                        continue
                    txn = inv_txn.get(dup["invoice_id"], [{}])[0].get("txn_id")
                    add("duplicate_invoice", dup["invoice_id"],
                        money.convert(dup["total_minor"], dup["currency"], base, rr[0]),
                        [same[0]["doc_id"], dup["doc_id"], txn])

        # unapplied payments
        for p in payments:
            if p["kind"] != "receipt":
                continue
            refs = p["invoice_refs"] or []
            to_dup = [] if self.ledger_amounts else [r for r in refs if r["invoice_id"] in dup_orig]
            effective = p["amount_minor"] - sum(r["amount_minor"] for r in refs) + sum(r["amount_minor"] for r in to_dup)
            if effective > 0:
                rr = rate(p["currency"], p["timestamp"])
                if rr is None:
                    continue
                txn = pay_txn.get(p["payment_id"], [{}])[0].get("txn_id")
                ev = [p["doc_id"], txn, rr[1]]
                for r in to_dup:
                    ev += [dup_orig[r["invoice_id"]], dup_doc[r["invoice_id"]]]
                amount = money.convert(effective, p["currency"], base, rr[0])
                if self.ledger_amounts:
                    amount = sum(r["credit_minor"] or 0 for r in pay_txn.get(p["payment_id"], [])
                                 if r["account"] == "UNAPPLIED_CASH")
                add("unapplied_payment", p["payment_id"], amount, ev)

        # fx errors and cutoff
        docs_all = [("invoice", i, i["issued_at"], i["total_minor"]) for i in invoices]
        docs_all += [("payment", p, p["timestamp"], p["amount_minor"]) for p in payments]
        for kind, d, ts, amount in docs_all:
            rows = inv_txn.get(d["invoice_id"], []) if kind == "invoice" else pay_txn.get(d["payment_id"], [])
            if not rows:
                continue
            aid = d["invoice_id"] if kind == "invoice" else d["payment_id"]
            txn = rows[0]["txn_id"]
            rr = rate(d["currency"], ts)
            if rr is None:
                continue
            correct = money.convert(amount, d["currency"], base, rr[0])
            if d["currency"] != base:
                if kind == "invoice":
                    posted = sum(r["debit_minor"] or 0 for r in rows if r["account"] == "AR")
                else:
                    posted = sum((r["debit_minor"] or 0) + (r["credit_minor"] or 0) for r in rows if r["account"] == "CASH")
                if posted != correct:
                    add("fx_error", aid, abs(posted - correct), [d["doc_id"], rr[1], txn])
            if not self._inside(ts):
                amount = correct
                if self.ledger_amounts:
                    acct = "AR" if kind == "invoice" else "CASH"
                    amount = sum(r["debit_minor"] or 0 for r in rows if r["account"] == acct)
                add("period_cutoff", aid, amount, [d["doc_id"], txn, rr[1]])

        yield submit(self.episode_id, findings[:30])
