"""Open bundle (what the agent may see) and closed grader data: build, write, read.

Open files:   episodes/<split>/<episode_id>/{manifest,policy}.json, ledger.csv,
              invoices.jsonl, payments.jsonl, fx.csv
Closed files: grader_data/<split>/<episode_id>/{truth,provenance}.json

CSV null encoding: an empty field means JSON null.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .. import SCHEMA_VERSION, jsonutil, money
from .company import Company

LEDGER_COLUMNS = ["row_id", "txn_id", "timestamp", "account", "debit_minor", "credit_minor",
                  "currency", "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["fx_id", "date", "source_currency", "target_currency", "rate"]
OPEN_FILES = ["manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"]
CLOSED_FILES = ["truth.json", "provenance.json"]
LEDGER_INT = ("debit_minor", "credit_minor")

ACCOUNTS = {
    "AR": "accounts receivable (per invoice_id)",
    "REVENUE": "sales revenue",
    "CASH": "bank account",
    "UNAPPLIED_CASH": "cash received but not applied to any invoice",
    "FX_GAIN_LOSS": "realised FX difference on settlement and rounding",
    "REFUNDS": "refunds to customers (contra revenue)",
    "OPEX": "operating expenses",
    "PAYROLL": "salaries",
    "BANK_FEES": "bank charges",
    "INTEREST_INCOME": "interest received",
}


def build_policy(company: Company) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "period": {
            "start": company.period_start.isoformat(timespec="seconds"),
            "end": company.period_end.isoformat(timespec="seconds"),
            "timezone": company.offset,
            "rule": "an operation belongs to the period iff start <= timestamp < end, comparing instants; "
                    "calendar dates are taken in the company timezone",
        },
        "fx": {
            "base_currency": company.base_currency,
            "rate_semantics": "rate = units of target_currency per 1 unit of source_currency",
            "rate_date_rule": "use the fx.csv row whose date equals the calendar date of the document "
                              "timestamp (invoice issued_at / payment timestamp) in the company timezone",
            "same_currency": "no conversion, rate 1",
            "settlement": "AR is relieved at the invoice booking rate on cumulative amounts: "
                          "relief = convert(applied_after) - convert(applied_before); the remaining "
                          "difference to CASH is posted to FX_GAIN_LOSS",
        },
        "rounding": {
            "mode": money.ROUNDING_NAME,
            "unit": "one minor unit of the target currency",
            "currency_exponents": dict(sorted(money.EXPONENTS.items())),
        },
        "ledger": {
            "currency": company.base_currency,
            "accounts": ACCOUNTS,
            "csv_null": "empty field",
            "posting": "invoice: Dr AR / Cr REVENUE; receipt: Dr CASH / Cr AR per invoice_ref "
                       "(+ Cr UNAPPLIED_CASH for any unapplied part, FX_GAIN_LOSS balances); "
                       "refund: Dr REFUNDS / Cr CASH",
        },
        "documents": {
            "invoice_refs": "list of {invoice_id, amount_minor} in payment currency, or null if none",
            "duplicate_rule": "two invoices with the same entity_id, currency, total_minor and identical "
                              "line_items are duplicates; the one issued later is the duplicate",
            "unapplied_rule": "a receipt's unapplied amount = amount_minor - sum(invoice_refs) + amounts "
                              "applied to duplicate invoices (cash on an invoice that must be voided is unapplied)",
            "issue_amounts": "every issue amount is in base currency at the policy rate of the document's own "
                             "date, computed as if all other issues were already corrected",
        },
    }


def build_bundle(company: Company, ledger: list[dict]) -> dict:
    invoices = sorted(company.invoices, key=lambda d: d["doc_id"])
    payments = sorted(company.payments, key=lambda d: d["doc_id"])
    fx = sorted(company.fx_rows, key=lambda r: (r["source_currency"], r["date"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": company.episode_id,
        "base_currency": company.base_currency,
        "period_start": company.period_start.isoformat(timespec="seconds"),
        "period_end": company.period_end.isoformat(timespec="seconds"),
        "timezone": company.offset,
        "record_counts": {"ledger": len(ledger), "invoices": len(invoices), "payments": len(payments), "fx": len(fx)},
        "document_count": len(invoices) + len(payments),
        "rounding_rules": {"mode": money.ROUNDING_NAME, "fx_rate_format": "decimal string"},
        "currency_exponents": dict(sorted(money.EXPONENTS.items())),
    }
    return jsonutil.deep_copy({
        "manifest": manifest,
        "policy": build_policy(company),
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx": fx,
    })


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in columns])
    return buf.getvalue()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_open(bundle: dict, directory: Path) -> None:
    _write(directory / "manifest.json", jsonutil.pretty(bundle["manifest"]))
    _write(directory / "policy.json", jsonutil.pretty(bundle["policy"]))
    _write(directory / "ledger.csv", _csv_text(bundle["ledger"], LEDGER_COLUMNS))
    _write(directory / "fx.csv", _csv_text(bundle["fx"], FX_COLUMNS))
    _write(directory / "invoices.jsonl", "".join(jsonutil.dumps(d) + "\n" for d in bundle["invoices"]))
    _write(directory / "payments.jsonl", "".join(jsonutil.dumps(d) + "\n" for d in bundle["payments"]))


def write_closed(truth: dict, provenance: dict, directory: Path) -> None:
    _write(directory / "truth.json", jsonutil.pretty(truth))
    _write(directory / "provenance.json", jsonutil.pretty(provenance))


def _read_csv(path: Path, int_cols: tuple[str, ...]) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if v == "":
                    row[k] = None
                elif k in int_cols:
                    row[k] = int(v)
                else:
                    row[k] = v
            rows.append(row)
        return rows


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_open(directory: Path) -> dict:
    with open(directory / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(directory / "policy.json", encoding="utf-8") as fh:
        policy = json.load(fh)
    return {
        "manifest": manifest,
        "policy": policy,
        "ledger": _read_csv(directory / "ledger.csv", LEDGER_INT),
        "invoices": _read_jsonl(directory / "invoices.jsonl"),
        "payments": _read_jsonl(directory / "payments.jsonl"),
        "fx": _read_csv(directory / "fx.csv", ()),
    }


def read_closed(directory: Path) -> tuple[dict, dict]:
    with open(directory / "truth.json", encoding="utf-8") as fh:
        truth = json.load(fh)
    with open(directory / "provenance.json", encoding="utf-8") as fh:
        provenance = json.load(fh)
    return truth, provenance


def closed_dir_of(episode_dir: Path) -> Path:
    """Closed files live in ``<dir>/closed/`` if present, else next to the open files."""
    return episode_dir / "closed" if (episode_dir / "closed" / "truth.json").is_file() else episode_dir


def read_truth_for_grading(closed_dir: Path) -> tuple[dict, dict]:
    """Truth (+ provenance if present) for scoring an episode directory.

    Grading needs only truth.json; provenance.json is optional here (a hand-made grader
    fixture may omit it). Accepted truth shapes: ``{"issues": [...]}`` (ours) or a bare
    list of issues; an issue without ``issue_id`` gets ``ISS-<index>`` in file order.
    """
    with open(closed_dir / "truth.json", encoding="utf-8") as fh:
        truth = json.load(fh)
    if isinstance(truth, list):
        truth = {"issues": truth}
    issues = []
    for n, issue in enumerate(truth.get("issues") or [], start=1):
        issue = dict(issue)
        issue.setdefault("issue_id", f"ISS-{n:02d}")
        issues.append(issue)
    truth = dict(truth, issues=issues)
    provenance: dict = {}
    if (closed_dir / "provenance.json").is_file():
        with open(closed_dir / "provenance.json", encoding="utf-8") as fh:
            provenance = json.load(fh)
    return truth, provenance
