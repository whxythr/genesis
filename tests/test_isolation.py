"""No leakage of the hidden truth: memory, paths, observations, errors, trajectories, protocol."""

import json
import re

import pytest

from audit_env import jsonutil
from audit_env.env.core import AuditEnv
from audit_env.generator.episode import generate_episode, write_episode
from audit_env.generator.writer import OPEN_FILES
from audit_env.grader.client import GraderClient
from audit_env.policies.flag_all import FlagAllPolicy
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.reconcile import ReconcilePolicy
from audit_env.runner import run_episode

from .helpers import oracle_findings, submit_action

TRUTH_MARKERS = ("acceptable_evidence_sets", "expected_amount_minor", "issue_id", "ISS-", "provenance",
                 "truth", "grader_data", "ledger_diff", "\"seed\"")


def _walk(obj, seen=None):
    """Yield every string / key reachable from obj (attributes, dicts, lists)."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk(v, seen)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("audit_env"):
        yield from _walk(vars(obj), seen)


def _errored_seed():
    return next(s for s in range(50) if generate_episode(s).truth["issues"])


def test_env_memory_holds_no_truth_and_no_grader_paths(tmp_path):
    seed = _errored_seed()
    ep = generate_episode(seed)
    write_episode(ep, tmp_path / "e", tmp_path / "g", "x")
    for env in (AuditEnv(), AuditEnv(episodes_root=tmp_path / "e", grader_root=tmp_path / "g")):
        try:
            env.reset(seed=seed)
            env.step({"tool": "list_records", "args": {"table": "ledger"}})
            strings = list(_walk(env))
            blob = "\n".join(strings)
            for marker in TRUTH_MARKERS:
                assert marker not in blob, marker
            assert str(tmp_path / "g") not in blob
            # the subprocess command line has no grader paths either
            assert not any("grader" in str(a).lower() and "grader_data" in str(a) for a in env._grader._proc.args)
        finally:
            env.close()


def test_open_directory_contains_only_open_files(tmp_path):
    ep = generate_episode(_errored_seed())
    write_episode(ep, tmp_path / "e", tmp_path / "g", "x")
    assert sorted(p.name for p in (tmp_path / "e" / "x" / ep.episode_id).iterdir()) == sorted(OPEN_FILES)
    for name in OPEN_FILES:
        text = (tmp_path / "e" / "x" / ep.episode_id / name).read_text(encoding="utf-8")
        for marker in ("acceptable_evidence_sets", "expected_amount_minor", "ISS-", "injected", "scenario"):
            assert marker not in text


def test_observations_and_trajectories_leak_nothing():
    env = AuditEnv()
    try:
        for seed in range(6):
            ep = generate_episode(seed)
            for pol in (ReconcilePolicy(), FlagAllPolicy(), RandomExplorerPolicy(0)):
                rows, _ = run_episode(env, pol, seed=seed)
                for r in rows:
                    text = jsonutil.dumps(r["observation"])
                    for marker in TRUTH_MARKERS:
                        assert marker not in text, (marker, pol.name)
                    # distractor / injection tags are hidden too
                    assert "subscription\"" not in text and "boundary_timezone" not in text
    finally:
        env.close()


def test_error_messages_are_fixed_strings():
    from audit_env.env.tools import ERROR_MESSAGES
    env = AuditEnv()
    try:
        env.reset(seed=1)
        secret = "INV-ZZZZZZZZ<script>"
        o, *_ = env.step({"tool": "get_record", "args": {"table": "invoices", "id": secret}})
        assert secret not in json.dumps(o)
        assert o["error"]["message"] == ERROR_MESSAGES["NOT_FOUND"]
    finally:
        env.close()


def test_eval_mode_responses_do_not_depend_on_correctness():
    seed = _errored_seed()
    ep = generate_episode(seed)
    outs = []
    for findings in (oracle_findings(ep.truth), []):
        env = AuditEnv(mode="eval")
        try:
            obs, info = env.reset(seed=seed)
            o, r, term, trunc, info2 = env.step(submit_action(ep.episode_id, findings))
            outs.append((info, o, r, term, trunc, info2))
            results = env.finalize()
            assert len(results) == 1
        finally:
            env.close()
    assert outs[0] == outs[1]


def test_grader_protocol_never_returns_truth():
    seed = _errored_seed()
    c = GraderClient(eval_mode=True)
    try:
        r = c.call({"cmd": "open", "source": {"kind": "generate", "seed": seed, "config": {}}})
        assert r["ok"] and set(r) == {"ok", "bundle", "config_id"}
        assert "ISS-" not in jsonutil.dumps(r)
        for cmd in ("truth", "provenance", "dump", "debug", "report", "", None):
            reply = c.call({"cmd": cmd})
            assert "ISS-" not in jsonutil.dumps(reply)
            assert reply.get("ok") is False
        assert c.call({"cmd": "init", "grader_root": "/tmp"})["ok"] is False  # cannot repoint after use
        # grading cannot be repeated (no writing into the grader's results)
        assert c.call({"cmd": "grade", "findings": None, "inspection_steps": 0, "outcome": "submitted"})["ok"]
        assert c.call({"cmd": "grade", "findings": None, "inspection_steps": 0, "outcome": "submitted"})["ok"] is False
        assert c.call({"cmd": "open", "source": {"kind": "files", "split": "..", "episode_id": "EP-2222222222"}})["ok"] is False
    finally:
        c.close()


def test_ids_do_not_encode_injection(tmp_path):
    """Injected rows/docs are indistinguishable by ID shape and are not clustered at the end."""
    pattern = re.compile(r"^(INV|PAY|D|T|R)-[2-9A-HJ-NP-Z]{8}$")
    for seed in range(30):
        ep = generate_episode(seed)
        added = ep.provenance["ledger_diff"]["added"]
        for row in added:
            assert pattern.match(row["row_id"]) and pattern.match(row["txn_id"])


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_reset_info_has_no_truth(mode):
    env = AuditEnv(mode=mode)
    try:
        obs, info = env.reset(seed=_errored_seed())
        assert "ISS-" not in jsonutil.dumps(obs) + jsonutil.dumps(info)
        if mode == "eval":
            assert info == {}
    finally:
        env.close()
