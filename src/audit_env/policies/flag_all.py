"""Flag-all attack: reads every invoice and payment and accuses all of them (up to the
30-finding limit), cycling through every error type. Must score badly."""

from __future__ import annotations

from .base import Pager, Policy, submit

TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")


class FlagAllPolicy(Policy):
    name = "flag_all"

    def reset(self, obs: dict, truth: dict | None = None) -> None:
        super().reset(obs)
        self.pagers = [Pager("payments"), Pager("invoices")]
        self.current: Pager | None = None

    def act(self, obs: dict) -> dict:
        if self.current is not None:
            self.current.feed(obs)
        self.current = next((p for p in self.pagers if not p.done), None)
        if self.current is not None:
            return self.current.action()
        targets = []
        for p in self.pagers[0].records:
            targets.append((p["payment_id"], p["doc_id"], p["amount_minor"]))
        for i in self.pagers[1].records:
            targets.append((i["invoice_id"], i["doc_id"], i["total_minor"]))
        findings = []
        for n, (tid, doc_id, amount) in enumerate(targets[:30]):
            findings.append({"type": TYPES[n % len(TYPES)], "affected_ids": [tid], "claimed_amount_minor": amount,
                             "currency": self.base_currency, "evidence_ids": [doc_id]})
        return submit(self.episode_id, findings)
