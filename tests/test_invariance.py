"""Row permutation and ID renaming must not change the answer or the reward."""

import pytest

from audit_env import jsonutil
from audit_env.env.core import AuditEnv
from audit_env.generator import evidence, invariants
from audit_env.generator.episode import generate_episode, write_episode
from audit_env.grader.core import grade
from audit_env.grader.normalize import validate_and_normalize
from audit_env.policies.flag_all import FlagAllPolicy
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.reconcile import ReconcilePolicy
from audit_env.runner import run_episode
from audit_env.transform import map_findings, transform

from .helpers import oracle_findings

SEEDS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def _scenario_seeds(n=4):
    out = []
    for s in range(300):
        ep = generate_episode(s)
        if ep.provenance["generation"]["plan"]["scenarios"] and ep.truth["issues"]:
            out.append(s)
        if len(out) >= n:
            break
    return out


@pytest.fixture(scope="module")
def pairs():
    seeds = SEEDS + _scenario_seeds()
    out = []
    for s in seeds:
        ep = generate_episode(s)
        t, mapping = transform(ep, seed=1000 + s)
        out.append((s, ep, t, mapping))
    return out


def _canon(findings, episode_id):
    norm, err = validate_and_normalize({"episode_id": episode_id, "findings": findings}, episode_id)
    assert err is None
    return sorted(jsonutil.dumps(x) for x in norm)


def _grade(ep, findings):
    return grade({"episode_id": ep.episode_id, "findings": findings}, ep.truth, ep.bundle, 7, "submitted")


def test_transform_changes_ids_and_order(pairs):
    for _, ep, t, mapping in pairs:
        assert t.episode_id != ep.episode_id
        old_ids = {r["row_id"] for r in ep.bundle["ledger"]}
        new_ids = {r["row_id"] for r in t.bundle["ledger"]}
        assert not old_ids & new_ids
        assert [mapping[r["row_id"]] for r in ep.bundle["ledger"]] != [r["row_id"] for r in t.bundle["ledger"]]
        assert len(set(mapping.values())) == len(mapping)  # bijective


def test_transform_is_deterministic():
    ep = generate_episode(4)
    a, _ = transform(ep, 9)
    b, _ = transform(ep, 9)
    c, _ = transform(ep, 10)
    assert jsonutil.dumps(a.bundle) == jsonutil.dumps(b.bundle)
    assert jsonutil.dumps(a.bundle) != jsonutil.dumps(c.bundle)


def test_validator_agrees_after_transform(pairs):
    for _, _, t, _ in pairs:
        assert invariants.compare_with_truth(t.bundle, t.truth) == []
        assert evidence.check_sufficiency(t.bundle, t.truth) == []


@pytest.mark.parametrize("permute,rename", [(True, False), (False, True), (True, True)])
def test_oracle_and_reward_invariant(pairs, permute, rename):
    for s, ep, _, _ in pairs:
        t, mapping = transform(ep, seed=77 + s, permute=permute, rename=rename)
        f = oracle_findings(ep.truth)
        assert _grade(t, map_findings(f, mapping)) == dict(_grade(ep, f), episode_id=t.episode_id)
        # the oracle computed directly from the transformed truth is the mapped oracle answer
        assert _canon(oracle_findings(t.truth), t.episode_id) == _canon(map_findings(f, mapping), t.episode_id)


def test_partial_and_wrong_answers_invariant(pairs):
    for _, ep, t, mapping in pairs:
        f = oracle_findings(ep.truth)
        for x in f[::2]:
            x["claimed_amount_minor"] += 10_000
        f += [{"type": "fx_error", "affected_ids": [ep.bundle["invoices"][0]["invoice_id"]], "claimed_amount_minor": 5,
               "currency": ep.bundle["manifest"]["base_currency"], "evidence_ids": [ep.bundle["invoices"][0]["doc_id"]]}]
        assert _grade(t, map_findings(f, mapping))["reward"] == _grade(ep, f)["reward"]


def test_policies_through_env_are_invariant(pairs, tmp_path):
    """Reconcile/flag-all/random run on the original and transformed files give the same reward,
    and reconcile's findings are the mapped originals."""
    eroot, groot = tmp_path / "e", tmp_path / "g"
    for _, ep, t, _ in pairs:
        write_episode(ep, eroot, groot, "orig")
        write_episode(t, eroot, groot, "trans")
    env = AuditEnv(episodes_root=eroot, grader_root=groot)
    try:
        for _, ep, t, mapping in pairs:
            for pol in (ReconcilePolicy(), FlagAllPolicy()):
                ra, sa = run_episode(env, pol, config={"episode": {"split": "orig", "episode_id": ep.episode_id}})
                rb, sb = run_episode(env, pol, config={"episode": {"split": "trans", "episode_id": t.episode_id}})
                assert sa["reward"] == sb["reward"], (pol.name, sa, sb)
                if pol.name == "reconcile":
                    fa = map_findings(ra[-1]["action"]["args"]["findings"]["findings"], mapping)
                    fb = rb[-1]["action"]["args"]["findings"]["findings"]
                    assert _canon(fa, t.episode_id) == _canon(fb, t.episode_id)
            # random explorer: not order-invariant by design, but must still be valid and bounded
            _, sr = run_episode(env, RandomExplorerPolicy(1), config={"episode": {"split": "trans", "episode_id": t.episode_id}})
            assert -1.0 <= sr["reward"] <= 1.0
    finally:
        env.close()


def test_same_transform_seed_gives_distinct_episodes():
    """Regression: transforming a whole split with one seed must not collide episode ids."""
    ids = {transform(generate_episode(s), seed=7)[0].episode_id for s in range(6)}
    assert len(ids) == 6
