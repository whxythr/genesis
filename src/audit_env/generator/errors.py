"""Error injection: each error is a minimal causal change recorded for provenance.

Documents are mutated (or inserted) and/or a posting override is registered; the
ledger is then re-posted from the documents, so every downstream row change is a
consequence of the recorded cause.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .. import jsonutil, money
from ..config import ERROR_TYPES, SCENARIOS
from ..ids import IdFactory
from ..timeutil import fmt_ts, local_date, offset_minutes
from .company import Company


@dataclass
class Injection:
    issues: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    locked: set = field(default_factory=set)


SCENARIO_TYPES = {
    "partial_fx_payment_at_cutoff": ("period_cutoff", "fx_error", "unapplied_payment"),
    "partially_paid_duplicate": ("duplicate_invoice", "unapplied_payment"),
}
SCENARIO_SIZE = {k: len(v) for k, v in SCENARIO_TYPES.items()}


def make_plan(rng: random.Random, cfg: dict) -> dict:
    """Decide clean/errored, interacting scenarios and per-type counts (before generation)."""
    enabled = [t for t in ERROR_TYPES if cfg["errors"][t]["enabled"]]
    plan = {"clean": False, "counts": {t: 0 for t in ERROR_TYPES}, "scenarios": []}
    if cfg["max_errors"] == 0 or rng.random() < cfg["clean_probability"]:
        plan["clean"] = True
        return plan
    budget = cfg["max_errors"]
    for name in SCENARIOS:
        sc = cfg["scenarios"][name]
        types_on = all(t in enabled for t in SCENARIO_TYPES[name])
        if sc["enabled"] and rng.random() < sc["probability"] and types_on and SCENARIO_SIZE[name] <= budget:
            plan["scenarios"].append(name)
            budget -= SCENARIO_SIZE[name]
    counts = plan["counts"]
    for t in enabled:
        counts[t] = rng.randint(*cfg["errors"][t]["count"])
    if sum(counts.values()) == 0 and not plan["scenarios"] and enabled:
        counts[rng.choice(enabled)] = 1
    while sum(counts.values()) > budget:
        biggest = max(enabled, key=lambda t: (counts[t], t))
        counts[biggest] -= 1
    if not enabled and not plan["scenarios"]:
        plan["clean"] = True
    return plan


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

    def unique_payment_ts(self, ts: datetime) -> datetime:
        used = {int(datetime.fromisoformat(p["timestamp"]).timestamp()) for p in self.c.payments}
        while int(ts.timestamp()) in used:
            ts += timedelta(seconds=1)
        return ts

    def _base_amount(self, amount: int, currency: str, ts: datetime) -> int:
        return money.convert(amount, currency, self.c.base_currency, self.c.policy_rate(currency, ts))

    # ---- duplicate_invoice ------------------------------------------------
    def duplicate_invoice(self, forced: tuple[dict, timedelta] | None = None) -> bool:
        c, rng = self.c, self.rng
        by_inv = self._payments_by_invoice()
        if forced is None:
            cands = [i for i in sorted(c.invoices, key=lambda i: i["invoice_id"]) if i["invoice_id"] not in self.inj.locked]
            rng.shuffle(cands)
        else:
            cands = [forced[0]]
        for orig in cands:
            issued = datetime.fromisoformat(orig["issued_at"])
            delta = forced[1] if forced else timedelta(seconds=rng.randint(3600, 72 * 3600))
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
            self.last_duplicate = dup
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


    # ---- fx_error ---------------------------------------------------------
    def fx_error(self) -> bool:
        """Post a foreign-currency document with the rate of a wrong date."""
        c, rng = self.c, self.rng
        touched = set(self._payments_by_invoice())
        fx_id = {(r["source_currency"], r["date"]): r["fx_id"] for r in c.fx_rows}
        cands: list[tuple[str, dict]] = []
        for inv in sorted(c.invoices, key=lambda i: i["invoice_id"]):
            if inv["currency"] != c.base_currency and inv["invoice_id"] not in self.inj.locked | touched:
                cands.append(("invoice", inv))
        for pay in sorted(c.payments, key=lambda p: p["payment_id"]):
            refs = {r["invoice_id"] for r in pay["invoice_refs"] or []}
            if (pay["kind"] == "receipt" and pay["currency"] != c.base_currency and refs
                    and pay["payment_id"] not in self.inj.locked and not refs & self.inj.locked):
                cands.append(("payment", pay))
        rng.shuffle(cands)
        for kind, doc in cands:
            ts = datetime.fromisoformat(doc["issued_at"] if kind == "invoice" else doc["timestamp"])
            amount_doc = doc["total_minor"] if kind == "invoice" else doc["amount_minor"]
            cur = doc["currency"]
            day = local_date(ts, c.offset)
            variants = []
            utc_day = ts.astimezone(timezone.utc).date()
            if utc_day != day:
                variants.append(("utc_date_instead_of_company_date", utc_day))
            variants += [("previous_day", day - timedelta(days=1)), ("next_day", day + timedelta(days=1)),
                         ("week_old_rate", day - timedelta(days=7))]
            rng.shuffle(variants)
            for variant, wrong_day in variants:
                if (cur, wrong_day.isoformat()) not in c.rates:
                    continue
                right = money.convert(amount_doc, cur, c.base_currency, c.rate(cur, day))
                wrong = money.convert(amount_doc, cur, c.base_currency, c.rate(cur, wrong_day))
                if wrong == right:
                    continue
                doc_key = "invoice_id" if kind == "invoice" else "payment_id"
                self.inj.overrides[doc["doc_id"]] = {"rate_date": wrong_day.isoformat()}
                self.inj.locked.add(doc[doc_key])
                if kind == "payment":
                    self.inj.locked |= {r["invoice_id"] for r in doc["invoice_refs"]}
                self.inj.issues.append({
                    "type": "fx_error",
                    "affected_ids": [doc[doc_key]],
                    "expected_amount_minor": abs(wrong - right),
                    "currency": c.base_currency,
                    "acceptable_evidence_sets": [sorted([doc["doc_id"], fx_id[(cur, day.isoformat())],
                                                         c.txn_of[doc["doc_id"]]])],
                })
                self.inj.changes.append({
                    "type": "fx_error",
                    "affected_ids": [doc[doc_key]],
                    "action": "override_posting",
                    "variant": variant,
                    "target_doc_id": doc["doc_id"],
                    "field": "rate_date",
                    "before": day.isoformat(),
                    "after": wrong_day.isoformat(),
                    "posted_minor": {"before": right, "after": wrong},
                    "reason": "document converted with the fx.csv rate of a date other than the "
                              "operation date in the company timezone",
                })
                return True
        return False

    # ---- period_cutoff ----------------------------------------------------
    def period_cutoff(self) -> bool:
        """A next-period operation (document dated after period end) posted in this period."""
        c, rng = self.c, self.rng
        by_inv: dict[str, list[dict]] = {}
        for p in c.payments:
            for ref in p["invoice_refs"] or []:
                by_inv.setdefault(ref["invoice_id"], []).append(p)
        cands: list[tuple[str, dict]] = []
        for inv in sorted(c.invoices, key=lambda i: i["invoice_id"]):
            if inv["invoice_id"] not in self.inj.locked and inv["invoice_id"] not in by_inv:
                cands.append(("invoice", inv))
        for pay in sorted(c.payments, key=lambda p: p["payment_id"]):
            refs = {r["invoice_id"] for r in pay["invoice_refs"] or []}
            if pay["kind"] != "receipt" or not refs or pay["payment_id"] in self.inj.locked or refs & self.inj.locked:
                continue
            related = [q for i in sorted(refs) for q in by_inv[i]]
            ts = datetime.fromisoformat(pay["timestamp"])
            if any(q["kind"] == "refund" for q in related):
                continue
            if any(datetime.fromisoformat(q["timestamp"]) > ts for q in related):
                continue  # must stay the latest settlement of its invoices
            cands.append(("payment", pay))
        rng.shuffle(cands)
        for kind, doc in cands:
            field_name = "issued_at" if kind == "invoice" else "timestamp"
            old_text = doc[field_name]
            display_offset = old_text[-6:]
            new_ts = c.period_end + timedelta(seconds=rng.randint(60, 30 * 3600))
            gap = (offset_minutes(c.offset) - offset_minutes(display_offset)) * 60
            if gap > 600 and rng.random() < 0.6:
                # timezone trap: in the document's own offset the date still reads as the
                # last day of the period, but the instant is after the period end
                new_ts = c.period_end + timedelta(seconds=rng.randint(60, gap - 60))
            if kind == "payment":
                new_ts = self.unique_payment_ts(new_ts)
            posting_ts = c.period_end - timedelta(seconds=rng.randint(1, 6 * 3600))
            before = jsonutil.deep_copy(doc)
            doc[field_name] = fmt_ts(new_ts, display_offset)
            if kind == "invoice":
                terms = datetime.fromisoformat(before["due_at"]) - datetime.fromisoformat(old_text)
                doc["due_at"] = fmt_ts(new_ts + terms, c.offset)
            doc_key = "invoice_id" if kind == "invoice" else "payment_id"
            amount_doc = doc["total_minor"] if kind == "invoice" else doc["amount_minor"]
            self.inj.overrides[doc["doc_id"]] = {"posting_ts": fmt_ts(posting_ts, c.offset)}
            self.inj.locked.add(doc[doc_key])
            if kind == "payment":
                self.inj.locked |= {r["invoice_id"] for r in doc["invoice_refs"]}
            reads = "reads_in_period" if doc[field_name][:10] < c.period_end.date().isoformat() else "reads_next_period"
            self.inj.issues.append({
                "type": "period_cutoff",
                "affected_ids": [doc[doc_key]],
                "expected_amount_minor": self._base_amount(amount_doc, doc["currency"], new_ts),
                "currency": c.base_currency,
                "acceptable_evidence_sets": [sorted([doc["doc_id"], c.txn_of[doc["doc_id"]]])],
            })
            self.inj.changes.append({
                "type": "period_cutoff",
                "affected_ids": [doc[doc_key]],
                "action": "modify_document+override_posting",
                "variant": reads,
                "target_doc_id": doc["doc_id"],
                "field": field_name,
                "before": before,
                "after": jsonutil.deep_copy(doc),
                "posting_ts": fmt_ts(posting_ts, c.offset),
                "reason": "operation dated after the period end (instant comparison, company "
                          "timezone) but posted inside the period",
            })
            return True
        return False


def inject(company: Company, ids: IdFactory, rng: random.Random, plan: dict, setup: dict) -> Injection:
    from . import scenarios

    inj = Injection()
    counts = plan["counts"]
    injector = _Injector(company, ids, rng, inj)
    scenarios.apply(injector, plan, setup)
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
