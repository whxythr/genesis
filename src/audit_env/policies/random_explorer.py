"""Random explorer: random tool calls with a fixed seed, then a random (mostly wrong)
submission built from the IDs it happened to see. Deterministic per (seed, episode)."""

from __future__ import annotations

import random

from .base import Policy, submit

TABLES = ("ledger", "invoices", "payments", "fx")
TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")
ID_FIELDS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}


class RandomExplorerPolicy(Policy):
    name = "random_explorer"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def reset(self, obs: dict, truth: dict | None = None) -> None:
        super().reset(obs)
        self.rng = random.Random(f"{self.seed}|{self.episode_id}")
        self.n_explore = self.rng.randint(3, 25)
        self.steps = 0
        self.seen: dict[str, list[str]] = {t: [] for t in TABLES}
        self.docs: list[str] = []
        self.amounts: list[int] = []
        self.cursors: dict[str, str] = {}

    def _absorb(self, obs: dict) -> None:
        if not obs.get("ok") or not isinstance(obs.get("result"), dict):
            return
        res = obs["result"]
        recs = res.get("records") or ([res["record"]] if "record" in res else [])
        table = res.get("table")
        for r in recs:
            if table in ID_FIELDS:
                self.seen[table].append(r[ID_FIELDS[table]])
            if "doc_id" in r:
                self.docs.append(r["doc_id"])
            for k in ("total_minor", "amount_minor", "debit_minor", "credit_minor"):
                if isinstance(r.get(k), int) and r[k] > 0:
                    self.amounts.append(r[k])
        if table and res.get("next_cursor"):
            self.cursors[table] = res["next_cursor"]

    def act(self, obs: dict) -> dict:
        if "step" in obs:
            self._absorb(obs)
        rng = self.rng
        self.steps += 1
        if self.steps <= self.n_explore:
            choice = rng.randrange(4)
            if choice == 0 or not any(self.seen.values()):
                table = rng.choice(TABLES)
                cursor = self.cursors.get(table) if rng.random() < 0.5 else None
                return {"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": rng.randint(1, 50)}}
            if choice == 1:
                table = rng.choice([t for t in TABLES if self.seen[t]])
                return {"tool": "get_record", "args": {"table": table, "id": rng.choice(self.seen[table])}}
            if choice == 2 and self.docs:
                return {"tool": "get_document", "args": {"doc_id": rng.choice(self.docs)}}
            a, b = rng.randint(1, 10**6), rng.randint(1, 10**4)
            return {"tool": "calculator", "args": {"expression": f"{a} {rng.choice('+-*/')} {b}"}}
        findings = []
        targets = self.seen["invoices"] + self.seen["payments"]
        evidence_pool = self.docs + self.seen["ledger"] + self.seen["fx"]
        for _ in range(rng.randint(0, 3) if targets else 0):
            findings.append({
                "type": rng.choice(TYPES),
                "affected_ids": [rng.choice(targets)],
                "claimed_amount_minor": rng.choice(self.amounts) if self.amounts else 1,
                "currency": self.base_currency,
                "evidence_ids": rng.sample(evidence_pool, k=min(len(evidence_pool), rng.randint(1, 3))),
            })
        return submit(self.episode_id, findings)
