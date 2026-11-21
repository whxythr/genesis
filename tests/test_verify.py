"""Independent file-level verifier (Block 2 pass conditions): it accepts every generated
episode and REJECTS each kind of corruption - otherwise it would prove nothing."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge_adapter as ja  # noqa: E402

from audit_env.verify import rebuild_clean, verify_dir  # noqa: E402
from audit_env.generator.writer import read_closed, read_open  # noqa: E402

MATRIX = json.loads((ROOT / "configs" / "judge_matrix.json").read_text(encoding="utf-8"))["scenarios"]


def _gen(tmp: Path, d: str, seed: int) -> Path:
    out = tmp / f"{d}-{seed}"
    ja.generate_episode(seed, {"team_config": MATRIX[d]["team_config"]}, str(out))
    return out


@pytest.mark.parametrize("d", ["D01", "D03", "D07", "D09", "D10", "D12", "D13", "D15", "D17", "D20"])
def test_generated_episodes_verify_clean(d, tmp_path):
    assert verify_dir(_gen(tmp_path, d, 11)) == []


def test_pre_injection_rebuild_has_clean_size_and_no_issues(tmp_path):
    out = _gen(tmp_path, "D17", 3)
    bundle = read_open(out)
    truth, prov = read_closed(out)
    clean = rebuild_clean(bundle, prov)
    assert len(clean["ledger"]) == prov["generation"]["clean_counts"]["ledger"]
    assert len(clean["invoices"]) < len(bundle["invoices"])  # the duplicate is gone
    assert truth["issues"]


# ---------------------------------------------------------------- corruption must be caught
def _rw_jsonl(path: Path, fn) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _rw_csv(path: Path, fn) -> None:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0])
    rows = fn(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _rw_json(path: Path, fn) -> None:
    path.write_text(json.dumps(fn(json.loads(path.read_text(encoding="utf-8")))), encoding="utf-8")


def _first_bg_row(rows):  # a background (non-document) cash row
    return next(r for r in rows if r["account"] == "CASH" and not r["payment_id"])


def _bump_debit(rows):
    r = _first_bg_row(rows)
    r["debit_minor"] = str(int(r["debit_minor"] or 0) + 1) if r["debit_minor"] != "0" else r["debit_minor"]
    if r["debit_minor"] == "0":
        r["credit_minor"] = str(int(r["credit_minor"]) + 1)
    return rows


CORRUPTIONS = {
    "unbalanced ledger row": ("ledger.csv", _rw_csv, _bump_debit),
    "timestamp without offset": ("payments.jsonl", _rw_jsonl,
                                 lambda rows: [dict(r, timestamp=r["timestamp"][:19]) if i == 0 else r
                                               for i, r in enumerate(rows)]),
    "fx rate as binary float": ("fx.csv", _rw_csv,
                                lambda rows: [dict(r, rate=r["rate"] + "e0") if i == 0 else r for i, r in enumerate(rows)]),
    "amount as string": ("invoices.jsonl", _rw_jsonl,
                         lambda rows: [dict(r, total_minor=str(r["total_minor"])) if i == 0 else r
                                       for i, r in enumerate(rows)]),
    "implicit null (missing key)": ("payments.jsonl", _rw_jsonl,
                                    lambda rows: [{k: v for k, v in r.items() if k != "invoice_refs"} if i == 0 else r
                                                  for i, r in enumerate(rows)]),
    "undeclared error (invoice total edited)": ("invoices.jsonl", _rw_jsonl,
                                                lambda rows: [dict(r, total_minor=r["total_minor"] + 7,
                                                                   line_items=r["line_items"]) if i == 0 else r
                                                              for i, r in enumerate(rows)]),
    "document dropped": ("payments.jsonl", _rw_jsonl, lambda rows: rows[1:]),
    "wrong truth amount": ("truth.json", _rw_json,
                           lambda t: dict(t, issues=[dict(i, expected_amount_minor=i["expected_amount_minor"] + 1)
                                                     for i in t["issues"]])),
    "extra truth issue": ("truth.json", _rw_json,
                          lambda t: dict(t, issues=t["issues"] + [dict(t["issues"][0], issue_id="ISS-99",
                                                                       type="fx_error")])),
    "truth issue removed": ("truth.json", _rw_json, lambda t: dict(t, issues=t["issues"][1:])),
    "change record removed": ("provenance.json", _rw_json, lambda p: dict(p, changes=p["changes"][1:])),
    "clean_counts falsified": ("provenance.json", _rw_json,
                               lambda p: dict(p, generation=dict(p["generation"], clean_counts=dict(
                                   p["generation"]["clean_counts"], ledger=p["generation"]["clean_counts"]["ledger"] + 2)))),
    "ledger_diff dropped": ("provenance.json", _rw_json,
                            lambda p: dict(p, ledger_diff={"added": [], "removed": [], "changed": []})),
    "evidence id that does not exist": ("truth.json", _rw_json,
                                        lambda t: dict(t, issues=[dict(i, acceptable_evidence_sets=[
                                            s + ["D-NOTHERE1"] for s in i["acceptable_evidence_sets"]])
                                            for i in t["issues"]])),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_is_detected(name, tmp_path):
    out = _gen(tmp_path, "D17", 5)  # duplicate + interaction scenario: every change kind present
    assert verify_dir(out) == []
    fname, rw, fn = CORRUPTIONS[name]
    rw(out / fname, fn)
    assert verify_dir(out), f"verifier missed: {name}"
