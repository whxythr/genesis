"""End-to-end episode generation: clean history -> invariants -> injection -> re-check."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .. import SCHEMA_VERSION, __version__, jsonutil
from ..config import config_id, resolve
from ..ids import ALPHABET, IdFactory
from . import evidence, invariants, scenarios, writer
from .company import build_company
from .errors import SCENARIO_SIZE, inject, make_plan

MAX_ATTEMPTS = 25
from .posting import post_ledger


class GenerationError(RuntimeError):
    pass


@dataclass
class Episode:
    bundle: dict
    truth: dict
    provenance: dict

    @property
    def episode_id(self) -> str:
        return self.bundle["manifest"]["episode_id"]


def _episode_id(seed: int, cid: str) -> str:
    rng = random.Random(f"{seed}|episode|{cid}")
    return "EP-" + "".join(rng.choice(ALPHABET) for _ in range(10))


def _ledger_diff(before: list[dict], after: list[dict]) -> dict:
    b = {r["row_id"]: r for r in before}
    a = {r["row_id"]: r for r in after}
    return {
        "added": [a[k] for k in sorted(set(a) - set(b))],
        "removed": [b[k] for k in sorted(set(b) - set(a))],
        "changed": [{"before": b[k], "after": a[k]} for k in sorted(set(a) & set(b)) if a[k] != b[k]],
    }


def generate_episode(seed: int, config: dict | None = None) -> Episode:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    cfg = resolve(config)
    cid = config_id(cfg)
    episode_id = _episode_id(seed, cid)
    plan = make_plan(random.Random(f"{seed}|plan|{cid}"), cfg)
    planned = sum(plan["counts"].values()) + sum(SCENARIO_SIZE[s] for s in plan["scenarios"])

    # A requested error may find no valid target in a particular company (e.g. no foreign
    # receipt for fx_error). Instead of silently emitting fewer errors than configured, the
    # company is rebuilt with a deterministic attempt salt (attempt 0 = the original salts).
    for attempt in range(MAX_ATTEMPTS):
        salt = cid if attempt == 0 else f"{cid}|retry{attempt}"
        ids = IdFactory(random.Random(f"{seed}|ids|{salt}"))
        company = build_company(random.Random(f"{seed}|company|{salt}"), cfg, ids, episode_id)
        scenario_state = scenarios.setup(company, ids, random.Random(f"{seed}|scenario|{salt}"), plan)

        clean_ledger = post_ledger(company, ids)
        clean_bundle = writer.build_bundle(company, clean_ledger)
        problems = invariants.check_clean(clean_bundle)
        if problems:
            raise GenerationError(f"clean history violates invariants (seed={seed}): {problems[:5]}")

        inj = inject(company, ids, random.Random(f"{seed}|inject|{salt}"), plan, scenario_state)
        if len(inj.issues) >= planned:
            break
    else:
        raise GenerationError(f"could not place the {planned} configured errors (seed={seed}, "
                              f"{MAX_ATTEMPTS} attempts); relax the config")
    ledger = post_ledger(company, ids, inj.overrides)
    bundle = writer.build_bundle(company, ledger)

    truth = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": inj.issues}
    problems = invariants.compare_with_truth(bundle, truth) + evidence.check_sufficiency(bundle, truth)
    if problems:
        raise GenerationError(f"injected data does not match truth (seed={seed}): {problems[:5]}")
    counts = bundle["manifest"]["record_counts"]
    if not (200 <= counts["ledger"] <= 2000 and 20 <= bundle["manifest"]["document_count"] <= 200):
        raise GenerationError(f"episode size out of range (seed={seed}): {counts}")

    tags: dict[str, list[str]] = {}
    for doc_id in sorted(company.tags):
        for tag in company.tags[doc_id]:
            tags.setdefault(tag, []).append(doc_id)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "seed": seed,
        "version": __version__,
        "config_id": cid,
        "config": cfg,
        "generation": {
            "steps": ["build_company", "post_ledger(clean)", "invariants.check_clean",
                      "inject", "post_ledger(injected)", "invariants.compare_with_truth",
                      "evidence.check_sufficiency"],
            "base_currency": company.base_currency,
            "timezone": company.offset,
            "period_start": bundle["manifest"]["period_start"],
            "period_end": bundle["manifest"]["period_end"],
            "clean_counts": {"ledger": len(clean_ledger), "invoices": len(clean_bundle["invoices"]),
                             "payments": len(clean_bundle["payments"])},
            "distractor_tags": tags,
            "overrides": jsonutil.deep_copy(inj.overrides),
            "company_attempt": attempt,
            "plan": plan,
        },
        "changes": inj.changes,
        "ledger_diff": _ledger_diff(clean_ledger, ledger),
    }
    return Episode(bundle=bundle, truth=jsonutil.deep_copy(truth), provenance=jsonutil.deep_copy(provenance))


def write_episode(ep: Episode, episodes_root: Path, grader_root: Path, split: str) -> None:
    writer.write_open(ep.bundle, episodes_root / split / ep.episode_id)
    writer.write_closed(ep.truth, ep.provenance, grader_root / split / ep.episode_id)
