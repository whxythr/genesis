"""reset/step contract, schemas, limits, invalid input, termination."""

import pytest

from audit_env import schema
from audit_env.env.core import AuditEnv
from audit_env.generator.episode import generate_episode

from .helpers import oracle_findings, submit_action

SEED = 3


@pytest.fixture(scope="module")
def env():
    e = AuditEnv(mode="train")
    yield e
    e.close()


def _ok(obs):
    assert not schema.validate(obs, "observation"), schema.validate(obs, "observation")


def test_reset_observation(env):
    obs, info = env.reset(seed=SEED)
    _ok(obs)
    assert obs["steps_remaining"] == 100
    assert set(obs) == {"task", "manifest", "policy", "tools", "steps_remaining"}
    assert not schema.validate(obs["manifest"], "manifest")
    assert info["episode_id"] == obs["manifest"]["episode_id"]


def test_each_tool_works(env):
    obs, _ = env.reset(seed=SEED)
    o, r, term, trunc, info = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 5}})
    _ok(o)
    assert o["ok"] and len(o["result"]["records"]) == 5 and o["result"]["next_cursor"]
    assert (r, term, trunc, info) == (0.0, False, False, {})
    row = o["result"]["records"][0]
    o, *_ = env.step({"tool": "get_record", "args": {"table": "ledger", "id": row["row_id"]}})
    assert o["ok"] and o["result"]["record"] == row
    o, *_ = env.step({"tool": "list_records", "args": {"table": "invoices"}})
    inv = o["result"]["records"][0]
    assert "line_items" not in inv and "line_item_count" in inv
    o, *_ = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})
    assert o["ok"] and o["result"]["doc_type"] == "invoice" and o["result"]["document"]["line_items"]
    o, *_ = env.step({"tool": "calculator", "args": {"expression": "(10 + 5) / 4"}})
    assert o["result"] == {"value": "3.75"}
    assert o["steps_remaining"] == 95


def test_pagination_covers_table_once(env):
    obs, _ = env.reset(seed=SEED)
    n = obs["manifest"]["record_counts"]["ledger"]
    cursor, seen = None, []
    while True:
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}})
        seen += [r["row_id"] for r in o["result"]["records"]]
        assert len(o["result"]["records"]) <= 50
        cursor = o["result"]["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == n


def test_cursor_is_stable_and_opaque(env):
    env.reset(seed=SEED)
    a, *_ = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    env.reset(seed=SEED)
    b, *_ = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert a["result"]["next_cursor"] == b["result"]["next_cursor"]
    c = a["result"]["next_cursor"]
    for bad in [c[:-1] + ("A" if c[-1] != "A" else "B"), "x.y", "garbage", "a" * 150]:
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": bad}})
        assert o["error"]["code"] == "BAD_CURSOR"
    o, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "cursor": c}})
    assert o["error"]["code"] == "BAD_CURSOR"  # cursor bound to its table


def test_filter(env):
    env.reset(seed=SEED)
    o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "filter": {"account": "AR"}}})
    assert o["ok"] and all(r["account"] == "AR" for r in o["result"]["records"])
    o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "filter": {"debit_minor": "1"}}})
    assert o["error"]["code"] == "BAD_ARGS"


@pytest.mark.parametrize("action,code", [
    ({"tool": "rm_rf"}, "UNKNOWN_ACTION"),
    ({"tool": "list_records", "args": {"table": "secrets"}}, "UNKNOWN_TABLE"),
    ({"tool": "list_records", "args": {"table": "ledger", "limit": 51}}, "BAD_ARGS"),
    ({"tool": "list_records", "args": {"table": "ledger", "limit": 0}}, "BAD_ARGS"),
    ({"tool": "list_records", "args": {"table": "ledger", "limit": True}}, "BAD_ARGS"),
    ({"tool": "list_records", "args": {"table": "ledger", "bogus": 1}}, "BAD_ARGS"),
    ({"tool": "get_record", "args": {"table": "ledger", "id": "R-NOPE2345"}}, "NOT_FOUND"),
    ({"tool": "get_record", "args": {"table": "ledger"}}, "BAD_ARGS"),
    ({"tool": "get_document", "args": {"doc_id": "../../grader_data"}}, "NOT_FOUND"),
    ({"tool": "get_document", "args": {"doc_id": 5}}, "BAD_ARGS"),
    ({"tool": "calculator", "args": {"expression": "__import__('os').system('x')"}}, "CALC_ERROR"),
    ({"tool": "calculator", "args": {"expression": "1/0"}}, "CALC_ERROR"),
    ({"args": {}}, "BAD_ARGS"),
    ({"tool": "list_records", "args": [1]}, "BAD_ARGS"),
    ({"tool": "list_records", "args": {"table": "ledger"}, "extra": 1}, "BAD_ARGS"),
    ("not a dict", "BAD_ARGS"),
    (None, "BAD_ARGS"),
    ({"tool": "calculator", "args": {"expression": "1" * 200_000}}, "BAD_ARGS"),
])
def test_invalid_input_costs_a_step_and_leaks_nothing(env, action, code):
    env.reset(seed=SEED)
    o, r, term, trunc, info = env.step(action)
    _ok(o)
    assert o["ok"] is False and o["error"]["code"] == code
    assert o["steps_remaining"] == 99
    assert (r, term, trunc, info) == (0.0, False, False, {})
    text = str(o)
    for leak in ("Traceback", "grader", "truth", ".py", "\\", "/home", "C:"):
        assert leak not in text


def test_step_limit_truncates_with_empty_answer(env):
    obs, _ = env.reset(seed=SEED)
    ep = generate_episode(SEED)
    for i in range(1, 101):
        o, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
        if i < 100:
            assert not trunc and o["steps_remaining"] == 100 - i
    assert trunc and not term and o["steps_remaining"] == 0
    assert info["outcome"] == "truncated" and info["inspection_steps"] == 100
    expected_f1 = 1.0 if not ep.truth["issues"] else 0.0
    assert info["metrics"]["f1"] == expected_f1
    o, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1"}})
    assert o["error"]["code"] == "EPISODE_DONE" and r == 0.0 and trunc


def test_time_limit_truncates():
    e = AuditEnv(mode="train", time_limit_s=0.0)
    try:
        e.reset(seed=SEED)
        o, r, term, trunc, info = e.step({"tool": "calculator", "args": {"expression": "1"}})
        assert o["error"]["code"] == "TIMEOUT" and trunc and not term
        assert info["outcome"] == "truncated"
    finally:
        e.close()


def test_submit_terminates_and_second_submit_is_ignored(env):
    obs, _ = env.reset(seed=SEED)
    ep = generate_episode(SEED)
    eid = obs["manifest"]["episode_id"]
    o, r, term, trunc, info = env.step(submit_action(eid, oracle_findings(ep.truth)))
    _ok(o)
    assert term and not trunc and r == 1.0 and info["metrics"]["f1"] == 1.0
    o, r2, term2, trunc2, info2 = env.step(submit_action(eid, oracle_findings(ep.truth)))
    assert o["error"]["code"] == "EPISODE_DONE" and r2 == 0.0 and info2 == {}


@pytest.mark.parametrize("findings", [
    None,
    "nonsense",
    {"episode_id": "EP-2222222222", "findings": []},          # wrong episode
    {"findings": []},                                        # missing episode_id
    {"episode_id": "{eid}", "findings": [{"type": "fx_error"}]},
    {"episode_id": "{eid}", "findings": [{"type": "fx_error", "affected_ids": ["X"], "claimed_amount_minor": 1.5,
                                          "currency": "USD", "evidence_ids": []}]},
    {"episode_id": "{eid}", "findings": [{"type": "fx_error", "affected_ids": ["X"], "claimed_amount_minor": True,
                                          "currency": "USD", "evidence_ids": []}]},
    {"episode_id": "{eid}", "findings": [{"type": "fx_error", "affected_ids": ["X"], "claimed_amount_minor": 1,
                                          "currency": "USD", "evidence_ids": [], "explanation": "x" * 1001}]},
    {"episode_id": "{eid}", "findings": [{"type": "fx_error", "affected_ids": ["X"], "claimed_amount_minor": 1,
                                          "currency": "USD", "evidence_ids": ["E"] * 11}]},
    {"episode_id": "{eid}", "findings": [{"type": "fx_error", "affected_ids": ["X"], "claimed_amount_minor": 1,
                                          "currency": "USD", "evidence_ids": []}] * 31},
    {"episode_id": "{eid}", "findings": [], "extra": True},
])
def test_invalid_structure_gives_minus_one(env, findings):
    obs, _ = env.reset(seed=SEED)
    eid = obs["manifest"]["episode_id"]
    if isinstance(findings, dict) and findings.get("episode_id") == "{eid}":
        findings = dict(findings, episode_id=eid)
    o, r, term, trunc, info = env.step({"tool": "submit", "args": {"findings": findings}})
    assert term and r == -1.0 and info["invalid_reason"]


def test_reset_isolates_episodes(env):
    obs1, _ = env.reset(seed=1)
    env.step({"tool": "list_records", "args": {"table": "ledger"}})
    obs2, _ = env.reset(seed=2)
    assert obs2["steps_remaining"] == 100
    o, *_ = env.step({"tool": "calculator", "args": {"expression": "1"}})
    assert o["step"] == 1
    obs3, _ = env.reset(seed=1)
    assert obs3 == obs1


def test_eval_mode_hides_reward():
    e = AuditEnv(mode="eval")
    try:
        obs, info = e.reset(seed=SEED)
        assert info == {}
        ep = generate_episode(SEED)
        o, r, term, trunc, info = e.step(submit_action(obs["manifest"]["episode_id"], oracle_findings(ep.truth)))
        assert term and r == 0.0 and info == {}
        assert o["result"] == {"submitted": True}
        results = e.finalize()
        assert len(results) == 1 and results[0]["reward"] == 1.0
        with pytest.raises(Exception):
            e.reset(seed=SEED)  # finalized session cannot open more episodes
    finally:
        e.close()


def test_reset_from_files(tmp_path):
    from audit_env.generator.episode import write_episode
    ep = generate_episode(11)
    write_episode(ep, tmp_path / "episodes", tmp_path / "grader_data", "train")
    e = AuditEnv(episodes_root=tmp_path / "episodes", grader_root=tmp_path / "grader_data")
    try:
        obs, info = e.reset(config={"episode": {"split": "train", "episode_id": ep.episode_id}})
        assert obs["manifest"] == ep.bundle["manifest"]
        o, r, *_ = e.step(submit_action(ep.episode_id, oracle_findings(ep.truth)))
        assert r == 1.0
        with pytest.raises(Exception):
            e.reset(config={"episode": {"split": "../x", "episode_id": ep.episode_id}})
    finally:
        e.close()
