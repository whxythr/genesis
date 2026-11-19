"""Error injection: each error is a minimal causal change recorded for provenance.

Documents are mutated (or inserted) and/or a posting override is registered; the
ledger is then re-posted from the documents, so every downstream row change is a
consequence of the recorded cause.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .. import jsonutil, money
from ..config import ERROR_TYPES
from ..ids import IdFactory
from ..timeutil import fmt_ts
from .company import Company


@dataclass
class Injection:
    issues: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    locked: set = field(default_factory=set)


def plan_counts(rng: random.Random, cfg: dict) -> dict[str, int]:
    enabled = [t for t in ERROR_TYPES if cfg["errors"][t]["enabled"]]
    counts = {t: 0 for t in ERROR_TYPES}
    if not enabled or cfg["max_errors"] == 0 or rng.random() < cfg["clean_probability"]:
        return counts
    for t in enabled:
        counts[t] = rng.randint(*cfg["errors"][t]["count"])
    if sum(counts.values()) == 0:
        counts[rng.choice(enabled)] = 1
    while sum(counts.values()) > cfg["max_errors"]:
        biggest = max(enabled, key=lambda t: (counts[t], t))
        counts[biggest] -= 1
    return counts


class _Injector:
    def __init__(self, company: Company, ids: IdFactory, rng: random.Random, inj: Injection) -> None:
        self.c = company
        self.ids = ids
        self.rng = rng
        self.inj = inj

    def _payments_by_invoice(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for p in self.c.payments:
            for ref in p["invoice_refs"] or []:
                out.setdefault(ref["invoice_id"], []).append(p)
        return out

    def _base_amount(self, amount: int, currency: str, ts: datetime) -> int:
        return money.convert(amount, currency, self.c.base_currency, self.c.policy_rate(currency, ts))

    # ---- duplicate_invoice ------------------------------------------------
    def duplicate_invoice(self) -> bool:
        c, rng = self.c, self.rng
        by_inv = self._payments_by_invoice()
        cands = [i for i in sorted(c.invoices, key=lambda i: i["invoice_id"]) if i["invoice_id"] not in self.inj.locked]
        rng.shuffle(cands)
        for orig in cands:
            issued = datetime.fromisoformat(orig["issued_at"])
            delta = timedelta(seconds=rng.randint(3600, 72 * 3600))
            dup_issued = issued + delta
            if dup_issued >= c.period_end - timedelta(hours=1):
                continue
            dup = jsonutil.deep_copy(orig)
            dup["doc_id"] = self.ids.new("doc")
            dup["invoice_id"] = self.ids.new("invoice")
            dup["issued_at"] = fmt_ts(dup_issued, c.offset)
            dup["due_at"] = fmt_ts(datetime.fromisoformat(orig["due_at"]) + delta, c.offset)
            c.invoices.append(dup)
            c.txn_of[dup["doc_id"]] = self.ids.new("txn")
            c.tags[dup["doc_id"]] = ["injected"]
            self.inj.locked |= {orig["invoice_id"], dup["invoice_id"]}
            self.inj.locked |= {p["payment_id"] for p in by_inv.get(orig["invoice_id"], [])}
            amount = self._base_amount(dup["total_minor"], dup["currency"], dup_issued)
            self.inj.issues.append({
                "type": "duplicate_invoice",
                "affected_ids": [dup["invoice_id"]],
                "expected_amount_minor": amount,
                "currency": c.base_currency,
                "acceptable_evidence_sets": [sorted([orig["doc_id"], dup["doc_id"], c.txn_of[dup["doc_id"]]])],
            })
            self.inj.changes.append({
                "type": "duplicate_invoice",
                "affected_ids": [dup["invoice_id"]],
                "action": "insert_document",
                "target_doc_id": dup["doc_id"],
                "copied_from_doc_id": orig["doc_id"],
                "before": None,
                "after": jsonutil.deep_copy(dup),
                "reason": "invoice re-issued with new ids but identical entity, currency, total and "
                          "line items, and posted to AR/REVENUE a second time",
            })
            return True
        return False

    # ---- unapplied_payment ------------------------------------------------
    def unapplied_payment(self) -> bool:
        c, rng = self.c, self.rng
        refunded = {p["invoice_refs"][0]["invoice_id"] for p in c.payments if p["kind"] == "refund"}
        cands = []
        for p in sorted(c.payments, key=lambda p: p["payment_id"]):
            if p["kind"] != "receipt" or p["payment_id"] in self.inj.locked or not p["invoice_refs"]:
                continue
            inv_ids = {r["invoice_id"] for r in p["invoice_refs"]}
            if inv_ids & (self.inj.locked | refunded):
                continue
            cands.append(p)
        rng.shuffle(cands)
        for pay in cands:
            old_refs = jsonutil.deep_copy(pay["invoice_refs"])
            variants = ["null", "partial"] + (["drop"] if len(old_refs) >= 2 else [])
            variant = rng.choice(variants)
            if variant == "null":
                new_refs = None
            elif variant == "drop":
                new_refs = jsonutil.deep_copy(old_refs)
                del new_refs[rng.randrange(len(new_refs))]
            else:
                new_refs = jsonutil.deep_copy(old_refs)
                k = rng.randrange(len(new_refs))
                a = new_refs[k]["amount_minor"]
                if a < 2:
                    continue
                new_refs[k]["amount_minor"] = max(1, a * rng.randint(10, 90) // 100)
            applied = sum(r["amount_minor"] for r in new_refs or [])
            unapplied = pay["amount_minor"] - applied
            ts = datetime.fromisoformat(pay["timestamp"])
            amount = self._base_amount(unapplied, pay["currency"], ts)
            if unapplied <= 0 or amount <= 0:
                continue
            pay["invoice_refs"] = new_refs
            self.inj.locked.add(pay["payment_id"])
            self.inj.locked |= {r["invoice_id"] for r in old_refs}
            self.inj.issues.append({
                "type": "unapplied_payment",
                "affected_ids": [pay["payment_id"]],
                "expected_amount_minor": amount,
                "currency": c.base_currency,
                "acceptable_evidence_sets": [sorted([pay["doc_id"], c.txn_of[pay["doc_id"]]])],
            })
            self.inj.changes.append({
                "type": "unapplied_payment",
                "affected_ids": [pay["payment_id"]],
                "action": "modify_document",
                "variant": variant,
                "target_doc_id": pay["doc_id"],
                "field": "invoice_refs",
                "before": old_refs,
                "after": jsonutil.deep_copy(new_refs),
                "reason": "receipt not (fully) applied to the invoices it settles: the unapplied "
                          "remainder sits on UNAPPLIED_CASH while the invoices stay open in AR",
            })
            return True
        return False


def inject(company: Company, ids: IdFactory, rng: random.Random, cfg: dict) -> Injection:
    inj = Injection()
    counts = plan_counts(rng, cfg)
    injector = _Injector(company, ids, rng, inj)
    for etype in ERROR_TYPES:
        fn = getattr(injector, etype, None)
        for _ in range(counts[etype]):
            if fn is None:
                raise NotImplementedError(f"error type {etype} not implemented yet")
            fn()  # may fail to find a target; then that error is simply not injected
    inj.issues.sort(key=lambda i: (i["type"], i["affected_ids"]))
    key_to_id = {}
    for n, issue in enumerate(inj.issues, start=1):
        issue["issue_id"] = f"ISS-{n:02d}"
        key_to_id[(issue["type"], tuple(issue["affected_ids"]))] = issue["issue_id"]
    for ch in inj.changes:
        ch["issue_id"] = key_to_id.get((ch["type"], tuple(ch["affected_ids"])))
    inj.changes.sort(key=lambda ch: (ch["issue_id"] or "", ch["action"]))
    return inj
