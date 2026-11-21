"""The organisers' checks run through judge_adapter.py exactly as the jury calls it.

Block 1 (contract C01-C20), Block 2 matrix (D01-D20 generate as documented and the
independent validator accepts them) and Block 4 (paired checks: a judge-side copy with
permuted records and/or consistently renamed ids, built here by an INDEPENDENT transform
that does not use audit_env.transform).
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge_adapter as ja  # noqa: E402

from audit_env.env.core import AuditEnv  # noqa: E402

MATRIX = json.loads((ROOT / "configs" / "judge_matrix.json").read_text(encoding="utf-8"))["scenarios"]
CLOSED_WORDS = ("acceptable_evidence_sets", "expected_amount_minor", "issue_id", "provenance", "truth",
                "ISS-", "ledger_diff", "\"seed\"")


def _gen(tmp: Path, name: str, seed: int, d: str) -> Path:
    out = tmp / name
    ja.generate_episode(seed, {"team_config": MATRIX[d]["team_config"]}, str(out))
    return out


@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("judge")
    return {"C": _gen(tmp, "C", 101, "D01"), "E": _gen(tmp, "E", 102, "D06"), "tmp": tmp}


def _read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _truth(d: Path) -> dict:
    return json.loads((d / "truth.json").read_text(encoding="utf-8"))


def _oracle(d: Path) -> dict:
    t = _truth(d)
    eid = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["episode_id"]
    return {"episode_id": eid, "findings": [
        {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
         "currency": i["currency"], "evidence_ids": i["acceptable_evidence_sets"][0], "explanation": "oracle"}
        for i in t["issues"]]}


def _closed(d: Path) -> dict:
    return {"mode": "closed", "episode_dir": str(d), "team_config": {}}


def _a(name, **args):
    return {"name": name, "args": args}


# ------------------------------------------------------------------ Block 1: C01-C20
def test_c01_c02_c03_reset_and_manifest(eps):
    env = ja.make_env()
    try:
        obs, info = env.reset(None, _closed(eps["C"]))
        json.dumps(obs), json.dumps(info)
        assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
        assert isinstance(obs["task"], str) and info == {}
        text = json.dumps(obs)
        assert not any(w in text for w in CLOSED_WORDS)
        m = obs["manifest"]
        for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                  "record_counts", "document_count", "rounding_rules"):
            assert k in m
        d = eps["C"]
        counts = {"ledger": len(_read_csv(d / "ledger.csv")), "invoices": len(_jsonl(d / "invoices.jsonl")),
                  "payments": len(_jsonl(d / "payments.jsonl")), "fx": len(_read_csv(d / "fx.csv"))}
        assert m["record_counts"] == counts
        assert m["document_count"] == counts["invoices"] + counts["payments"]
    finally:
        env.close()


def test_c04_c05_c06_paging(eps):
    env = ja.make_env()
    try:
        env.reset(None, _closed(eps["C"]))
        want = [r["row_id"] for r in _read_csv(eps["C"] / "ledger.csv")]
        assert len(want) >= 101
        seen, cursor, first_cursor_page = [], None, None
        while True:
            obs, r, term, trunc, info = env.step(_a("list_records", table="ledger", cursor=cursor, limit=50))
            assert isinstance(term, bool) and isinstance(trunc, bool) and r == 0 and obs["error"] is None
            json.dumps(obs)
            page = obs["result"]["records"]
            assert set(obs["result"]) == {"records", "next_cursor"} and len(page) <= 50
            seen += [x["row_id"] for x in page]
            if cursor is not None and first_cursor_page is None:
                first_cursor_page = (cursor, page)
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        assert seen == want  # every row reachable, no gaps, no repeats
        c, page = first_cursor_page
        again, *_ = env.step(_a("list_records", table="ledger", cursor=c, limit=50))
        assert again["result"]["records"] == page
    finally:
        env.close()


def test_c07_c08_c09_c10(eps):
    env = ja.make_env()
    try:
        env.reset(None, _closed(eps["C"]))
        row = _read_csv(eps["C"] / "ledger.csv")[0]
        obs, *_ = env.step(_a("get_record", table="ledger", id=row["row_id"]))
        assert obs["result"]["row_id"] == row["row_id"] and obs["result"]["account"] == row["account"]
        doc = _jsonl(eps["C"] / "invoices.jsonl")[0]
        obs, *_ = env.step(_a("get_document", doc_id=doc["doc_id"]))
        assert obs["result"] == doc
        obs, *_ = env.step(_a("calculator", expression="0.1+0.2"))
        assert obs["error"] is None and obs["result"] == "0.3"
        for bad in ("open('/etc/passwd').read()", "__import__('os').system('id')", "../../truth.json", "1;2"):
            obs, *_ = env.step(_a("calculator", expression=bad))
            assert obs["result"] is None and obs["error"]["code"] == "CALC_ERROR"
    finally:
        env.close()


def test_c11_to_c14_bounded_errors_consume_steps(eps):
    env = ja.make_env()
    try:
        env.reset(None, _closed(eps["C"]))
        ids = {r["row_id"] for r in _read_csv(eps["C"] / "ledger.csv")}
        ids |= {d["doc_id"] for d in _jsonl(eps["C"] / "invoices.jsonl") + _jsonl(eps["C"] / "payments.jsonl")}
        u = "R-UNKNOWN0"
        assert u not in ids
        budget = 100
        for action in (_a("get_record", table="ledger", id=u), _a("get_document", doc_id=u),
                       _a("no_such_tool", x=1), {"name": "get_record"}, "not-an-object",
                       _a("get_record", table="secrets", id=u)):
            obs, r, term, trunc, info = env.step(action)
            budget -= 1
            assert obs["result"] is None and set(obs["error"]) == {"code", "message"}
            assert obs["remaining_steps"] == budget and not term and not trunc and info == {}
            assert str(eps["C"]) not in json.dumps(obs)
        row = sorted(ids)[0]
        obs, *_ = env.step(_a("list_records", table="ledger", cursor=None, limit=1))
        budget -= 1
        assert obs["error"] is None and obs["remaining_steps"] == budget
    finally:
        env.close()


def test_c15_submit_is_final(eps):
    env = ja.make_env()
    try:
        env.reset(None, {"mode": "train", "episode_dir": str(eps["E"])})
        env.step(_a("list_records", table="invoices", cursor=None, limit=5))
        obs, r, term, trunc, info = env.step(_a("submit", findings=_oracle(eps["E"])))
        assert term and not trunc and obs["result"] == {"accepted": True}
        assert abs(r - 0.998) < 1e-9 and info["metrics"]["tp"] == len(_truth(eps["E"])["issues"])
        obs2, r2, term2, trunc2, info2 = env.step(_a("submit", findings={"episode_id": "x", "findings": []}))
        assert obs2["error"] is not None and r2 == 0 and term2 and info2 == {}
    finally:
        env.close()


def test_c16_submit_as_action_100_and_c17_truncation(eps):
    env = ja.make_env()
    try:
        env.reset(None, _closed(eps["C"]))
        for _ in range(99):
            obs, r, term, trunc, _i = env.step(_a("calculator", expression="1+1"))
            assert not term and not trunc
        obs, r, term, trunc, _i = env.step(_a("submit", findings=_oracle(eps["C"])))
        assert term and not trunc and obs["result"] == {"accepted": True} and obs["remaining_steps"] == 0

        env.reset(None, _closed(eps["C"]))
        for i in range(100):
            obs, r, term, trunc, _i = env.step(_a("calculator", expression="1+1"))
        assert trunc and not term and r == 0
        obs, r, term, trunc, _i = env.step(_a("submit", findings=_oracle(eps["C"])))
        assert obs["error"] is not None and trunc  # nothing added after truncation
    finally:
        env.close()


def test_c18_time_limit_truncates(eps):
    env = AuditEnv(mode="train", time_limit_s=0.0)
    try:
        env.reset(None, {"episode_dir": str(eps["E"])})
        obs, r, term, trunc, info = env.step({"tool": "calculator", "args": {"expression": "1"}})
        assert trunc and not term and obs["error"]["code"] == "TIMEOUT"
        assert info["metrics"]["tp"] == 0 and r == -1.0  # empty findings on errored books
    finally:
        env.close()


def test_c19_closed_mode_hides_reward(eps):
    env = ja.make_env()
    try:
        obs, info = env.reset(None, _closed(eps["E"]))
        assert info == {}
        for _ in range(3):
            o, r, term, trunc, info = env.step(_a("list_records", table="payments", cursor=None, limit=50))
            assert r == 0 and info == {}
        o, r, term, trunc, info = env.step(_a("submit", findings=_oracle(eps["E"])))
        assert term and r == 0 and isinstance(r, float) and info == {} and o["result"] == {"accepted": True}
        assert not any(w in json.dumps(o) for w in CLOSED_WORDS)
        # the true reward exists only on the grader side
        res = ja.score_episode(str(eps["E"]), _oracle(eps["E"]), 3)
        assert res["fn"] == 0 and res["fp"] == 0 and abs(res["return"] - 0.994) < 1e-9
    finally:
        env.close()


def test_c20_no_leak_anywhere(eps):
    """Walk every record and document of C and E; nothing closed appears in any channel."""
    secrets: set[str] = set()
    for d in (eps["C"], eps["E"]):
        for i in _truth(d)["issues"]:
            secrets.add(i["issue_id"])
    env = ja.make_env()
    try:
        for d in (eps["C"], eps["E"]):
            docs = [x["doc_id"] for x in _jsonl(d / "invoices.jsonl") + _jsonl(d / "payments.jsonl")]
            queue = [_a("list_records", table=t, cursor=None, limit=50) for t in ("ledger", "invoices", "payments", "fx")]
            queue += [_a("get_document", doc_id=x) for x in docs]
            queue += [_a("get_record", table="fx", id="../../../truth.json"), _a("get_document", doc_id="truth.json")]
            transcript = []
            obs, info = env.reset(None, _closed(d))
            transcript.append((obs, info))
            while queue:
                act = queue.pop(0)
                obs, r, term, trunc, info = env.step(act)
                transcript.append((obs, info, r))
                if obs["error"] is None and act["name"] == "list_records" and obs["result"]["next_cursor"]:
                    queue.insert(0, _a("list_records", table=act["args"]["table"],
                                       cursor=obs["result"]["next_cursor"], limit=50))
                if trunc or term:
                    env.reset(None, _closed(d))
            text = json.dumps(transcript)
            assert not any(w in text for w in CLOSED_WORDS)
            assert not any(s in text for s in secrets)
            assert str(d) not in text and str(d).replace("\\", "/") not in text
    finally:
        env.close()


def test_adapter_env_object_holds_no_truth(eps):
    env = ja.make_env()
    try:
        env.reset(None, _closed(eps["E"]))
        inner = env._env
        dump = json.dumps({k: str(v) for k, v in vars(inner).items()}) + json.dumps(inner._store.__dict__, default=str)
        assert "acceptable_evidence_sets" not in dump and str(eps["E"]) not in dump
    finally:
        env.close()


def test_train_mode_generates_with_seed():
    env = ja.make_env()
    try:
        obs1, _ = env.reset(5, {"mode": "train", "team_config": {}})
        obs2, _ = env.reset(5, {"mode": "train", "team_config": {}})
        assert obs1 == obs2
    finally:
        env.close()


def test_score_episode_invalid_structure(eps):
    bad = _oracle(eps["E"])
    bad["findings"] = bad["findings"] * 31
    res = ja.score_episode(str(eps["E"]), bad, 5)
    assert res["return"] == -1 and res["tp"] is None and res["f1"] is None


# ------------------------------------------------------------------ Block 2: D01-D20 matrix
@pytest.mark.parametrize("d", sorted(MATRIX))
def test_d_matrix_generates_declared_scenario(d, tmp_path):
    for seed in (1, 2):
        out = _gen(tmp_path, f"{d}-{seed}", seed, d)
        issues = _truth(out)["issues"]
        prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        n = int(d[1:])
        types = sorted(i["type"] for i in issues)
        if n <= 5:
            assert issues == [] and prov["changes"] == []
        elif n <= 8:
            assert set(types) == {"duplicate_invoice"}
        elif n <= 11:
            assert types == ["unapplied_payment"]
            want = {9: {"null"}, 10: {"partial"}, 11: {"partial", "drop"}}[n]
            assert {c["variant"] for c in prov["changes"]} <= want
        elif n <= 13:
            assert types == ["fx_error"]
        elif n <= 15:
            assert set(types) == {"period_cutoff"}
        elif n <= 18:
            assert prov["generation"]["plan"]["scenarios"] == ["partial_fx_payment_at_cutoff"]
            assert {"fx_error", "period_cutoff", "unapplied_payment"} <= set(types)
        else:
            assert prov["generation"]["plan"]["scenarios"] == ["partially_paid_duplicate"]
            assert {"duplicate_invoice", "unapplied_payment"} <= set(types)


# ------------------------------------------------------------------ Block 4: paired checks
ID_FIELDS = {  # namespace of every id-bearing field (documented extra fields included)
    "row_id": "row", "txn_id": "txn", "doc_id": "doc", "invoice_id": "inv", "payment_id": "pay",
    "entity_id": "ent", "fx_id": "fx",
}


def _judge_transform(src: Path, dst: Path, seed: int, permute: bool, rename: bool, rename_extra: bool) -> dict:
    """Independent re-implementation of the jury's copy transform. Returns new->old id map."""
    rng = random.Random(seed)
    led, fx = _read_csv(src / "ledger.csv"), _read_csv(src / "fx.csv")
    inv, pay = _jsonl(src / "invoices.jsonl"), _jsonl(src / "payments.jsonl")
    fields = [f for f in ID_FIELDS if rename_extra or f not in ("txn_id", "fx_id")]
    olds: dict[str, set] = {ID_FIELDS[f]: set() for f in fields}
    for rows in (led, fx, inv, pay):
        for r in rows:
            for f in fields:
                if r.get(f):
                    olds[ID_FIELDS[f]].add(r[f])
    mapping: dict[str, str] = {}
    if rename:
        for ns in sorted(olds):
            vals = sorted(olds[ns])
            rng.shuffle(vals)
            for n, v in enumerate(vals):
                mapping[v] = f"{ns.upper()}_{n:05d}"  # namespaces kept apart by prefix
    m = lambda v: mapping.get(v, v) if isinstance(v, str) else v  # noqa: E731

    def deep(x):
        if isinstance(x, dict):
            return {k: deep(v) for k, v in x.items()}
        if isinstance(x, list):
            return [deep(v) for v in x]
        return m(x)

    led = [{k: (m(v) if k in fields else v) for k, v in r.items()} for r in led]
    fx = [{k: (m(v) if k in fields else v) for k, v in r.items()} for r in fx]
    inv, pay = deep(inv), deep(pay)
    if permute:
        for rows in (led, fx, inv, pay):
            rng.shuffle(rows)
    dst.mkdir(parents=True)
    for name, rows in (("ledger.csv", led), ("fx.csv", fx)):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        (dst / name).write_text(buf.getvalue(), encoding="utf-8")
    for name, rows in (("invoices.jsonl", inv), ("payments.jsonl", pay)):
        (dst / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for name in ("manifest.json", "policy.json", "truth.json", "provenance.json"):
        (dst / name).write_text(json.dumps(deep(json.loads((src / name).read_text(encoding="utf-8")))),
                                encoding="utf-8")
    return {v: k for k, v in mapping.items()}


def _unmap(findings: dict, back: dict) -> dict:
    out = json.loads(json.dumps(findings))
    for f in out["findings"]:
        f["affected_ids"] = sorted(back.get(x, x) for x in f["affected_ids"])
        f["evidence_ids"] = sorted(back.get(x, x) for x in f["evidence_ids"])
    return out


def _run_oracle(d: Path, steps: int) -> tuple[dict, float]:
    env = ja.make_env()
    try:
        env.reset(None, _closed(d))
        for _ in range(steps):
            env.step(_a("list_records", table="ledger", cursor=None, limit=50))
        f = _oracle(d)
        o, r, term, *_ = env.step(_a("submit", findings=f))
        assert term and r == 0 and o["result"] == {"accepted": True}
        return f, r
    finally:
        env.close()


PAIRS = [(f"P{n:02d}", f"D{n:02d}", n <= 5 or n > 10, n > 5) for n in range(1, 21)]


@pytest.mark.parametrize("pid,d,permute,rename", PAIRS)
def test_paired_check(pid, d, permute, rename, tmp_path):
    src = _gen(tmp_path, "orig", 300 + int(d[1:]), d)
    back = _judge_transform(src, tmp_path / "copy", int(pid[1:]), permute, rename, rename_extra=(int(pid[1:]) % 2 == 0))
    dst = tmp_path / "copy"
    steps = 7
    f_src, _ = _run_oracle(src, steps)
    f_dst, _ = _run_oracle(dst, steps)
    assert _unmap(f_dst, back)["findings"] == _unmap(f_src, {})["findings"]
    a = ja.score_episode(str(src), f_src, steps)
    b = ja.score_episode(str(dst), f_dst, steps)
    assert a == b and a["fn"] == 0 and a["fp"] == 0
    # repeat on the original reproduces exactly
    assert ja.score_episode(str(src), f_src, steps) == a
    # a broken evidence set must not keep the reward (pair rule 4)
    if f_dst["findings"]:
        broken = json.loads(json.dumps(f_dst))
        broken["findings"][0]["evidence_ids"] = ["NOT_AN_ID"]
        assert ja.score_episode(str(dst), broken, steps)["tp"] == a["tp"] - 1
