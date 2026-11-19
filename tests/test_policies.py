"""Baseline policies: reconcile solves the books through tools only; attacks score badly."""

import pytest

from audit_env.env.core import AuditEnv
from audit_env.policies.flag_all import FlagAllPolicy
from audit_env.policies.oracle import OraclePolicy
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.reconcile import ReconcilePolicy
from audit_env.runner import run_episode

SEEDS = range(200, 215)


@pytest.fixture(scope="module")
def results():
    env = AuditEnv()
    out = {}
    try:
        for pol in (OraclePolicy(), FlagAllPolicy(), RandomExplorerPolicy(0), ReconcilePolicy(),
                    ReconcilePolicy(naive_dates=True)):
            out[pol.name] = [run_episode(env, pol, seed=s) for s in SEEDS]
    finally:
        env.close()
    return out


def _mean(rs, key):
    return sum(s[key] for _, s in rs) / len(rs)


def test_oracle_max(results):
    assert all(s["reward"] == 1.0 for _, s in results["oracle"])


def test_reconcile_through_tools_only_is_strong(results):
    assert _mean(results["reconcile"], "f1") >= 0.95
    assert all(s["steps"] <= 100 for _, s in results["reconcile"])
    assert all(r["action"] is None or r["action"]["tool"] in
               {"list_records", "get_document", "get_record", "calculator", "submit"}
               for rows, _ in results["reconcile"] for r in rows)


def test_ordering_of_baselines(results):
    rew = {k: _mean(v, "reward") for k, v in results.items()}
    assert rew["oracle"] >= rew["reconcile"] > rew["reconcile_naive"] > rew["random_explorer"]
    assert rew["reconcile"] > rew["flag_all"]
    assert rew["flag_all"] <= -0.9


def test_random_explorer_is_deterministic():
    env = AuditEnv()
    try:
        a = run_episode(env, RandomExplorerPolicy(3), seed=5)[0]
        b = run_episode(env, RandomExplorerPolicy(3), seed=5)[0]
    finally:
        env.close()
    assert [r["action"] for r in a] == [r["action"] for r in b]
    assert [r["observation"] for r in a] == [r["observation"] for r in b]
