"""Same (seed, config, version, actions) -> byte-identical records, transitions and rewards."""

import subprocess
import sys
from pathlib import Path

from audit_env import jsonutil
from audit_env.env.core import AuditEnv
from audit_env.generator.episode import generate_episode, write_episode
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.reconcile import ReconcilePolicy
from audit_env.runner import run_episode

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_files_byte_identical(tmp_path):
    for seed in (0, 17, 99):
        a, b = generate_episode(seed), generate_episode(seed)
        write_episode(a, tmp_path / "a" / "e", tmp_path / "a" / "g", "s")
        write_episode(b, tmp_path / "b" / "e", tmp_path / "b" / "g", "s")
    files_a = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert files_a == files_b and files_a
    for rel in files_a:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_independent_of_hash_randomization():
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from audit_env.generator.episode import generate_episode; from audit_env import jsonutil;"
            "ep = generate_episode(23); print(jsonutil.digest([ep.bundle, ep.truth, ep.provenance], 32))")
    digests = set()
    for hashseed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code, SRC], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": hashseed, "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                             check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_config_changes_the_episode():
    a = generate_episode(5)
    b = generate_episode(5, {"n_invoices": [30, 40]})
    assert a.episode_id != b.episode_id
    assert a.provenance["config_id"] != b.provenance["config_id"]


def test_replaying_actions_gives_identical_transitions():
    env = AuditEnv()
    try:
        for pol in (ReconcilePolicy(), RandomExplorerPolicy(5)):
            rows1, _ = run_episode(env, pol, seed=31)
            # replay the recorded actions on a fresh reset
            obs, _ = env.reset(seed=31)
            replay = [obs]
            rewards = []
            for r in rows1[1:]:
                o, rew, term, trunc, info = env.step(r["action"])
                replay.append(o)
                rewards.append((rew, term, trunc))
            assert jsonutil.dumps([r["observation"] for r in rows1]) == jsonutil.dumps(replay)
            assert [(r["reward"], r["terminated"], r["truncated"]) for r in rows1[1:]] == rewards
    finally:
        env.close()


def test_trajectories_byte_identical_across_envs():
    dumps = []
    for _ in range(2):
        env = AuditEnv()
        try:
            rows, _ = run_episode(env, ReconcilePolicy(), seed=8)
        finally:
            env.close()
        dumps.append("\n".join(jsonutil.dumps(r) for r in rows))
    assert dumps[0] == dumps[1]
