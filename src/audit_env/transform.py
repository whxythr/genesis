"""transform(episode, seed): permute row order and deterministically rename every ID.

Used for the invariance checks: the oracle's answer and the reward (after mapping IDs)
must not change, and the independent validator must still agree with the mapped truth.
The ID mapping is stored on the closed (grader) side only.
"""

from __future__ import annotations

import random

from . import jsonutil
from .generator.episode import Episode
from .ids import ALPHABET, PREFIXES, IdFactory

_KIND_BY_PREFIX = {v: k for k, v in PREFIXES.items()}


class _Renamer:
    def __init__(self, rng: random.Random, enabled: bool) -> None:
        self.rng = rng
        self.ids = IdFactory(rng)
        self.enabled = enabled
        self.mapping: dict[str, str] = {}

    def __call__(self, old):
        if old is None or not self.enabled:
            return old
        if old not in self.mapping:
            prefix = old.split("-", 1)[0] + "-"
            kind = _KIND_BY_PREFIX[prefix]
            if kind == "episode":
                new = "EP-" + "".join(self.rng.choice(ALPHABET) for _ in range(10))
            else:
                new = self.ids.new(kind)
            self.mapping[old] = new
        return self.mapping[old]


def transform(ep: Episode, seed: int, permute: bool = True, rename: bool = True) -> tuple[Episode, dict[str, str]]:
    rng = random.Random(f"transform|{seed}|{ep.episode_id}")
    m = _Renamer(random.Random(f"transform-ids|{seed}|{ep.episode_id}"), rename)
    b = jsonutil.deep_copy(ep.bundle)

    b["manifest"]["episode_id"] = m(b["manifest"]["episode_id"])
    for r in b["ledger"]:
        for k in ("row_id", "txn_id", "entity_id", "invoice_id", "payment_id"):
            r[k] = m(r[k])
    for d in b["invoices"]:
        for k in ("doc_id", "invoice_id", "entity_id"):
            d[k] = m(d[k])
    for d in b["payments"]:
        for k in ("doc_id", "payment_id", "entity_id"):
            d[k] = m(d[k])
        for ref in d["invoice_refs"] or []:
            ref["invoice_id"] = m(ref["invoice_id"])
    for r in b["fx"]:
        r["fx_id"] = m(r["fx_id"])
    if permute:
        for table in ("ledger", "invoices", "payments", "fx"):
            rng.shuffle(b[table])

    t = jsonutil.deep_copy(ep.truth)
    t["episode_id"] = b["manifest"]["episode_id"]
    for issue in t["issues"]:
        issue["affected_ids"] = sorted(m(x) for x in issue["affected_ids"])
        issue["acceptable_evidence_sets"] = [sorted(m(x) for x in s) for s in issue["acceptable_evidence_sets"]]

    prov = {
        "schema_version": ep.provenance["schema_version"],
        "episode_id": b["manifest"]["episode_id"],
        "config_id": ep.provenance["config_id"],
        "version": ep.provenance["version"],
        "transform": {"seed": seed, "permute": permute, "rename": rename,
                      "source_episode_id": ep.episode_id, "id_mapping": dict(sorted(m.mapping.items()))},
        "source_provenance": ep.provenance,
    }
    return Episode(bundle=b, truth=t, provenance=prov), dict(m.mapping)


def map_findings(findings: list[dict], mapping: dict[str, str]) -> list[dict]:
    out = []
    for f in findings:
        g = dict(f)
        g["affected_ids"] = [mapping.get(x, x) for x in f["affected_ids"]]
        g["evidence_ids"] = [mapping.get(x, x) for x in f["evidence_ids"]]
        out.append(g)
    return out
