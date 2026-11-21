"""Independent verifier of ONE episode directory - the six pass conditions of Block 2.

Works on files only (open files + truth.json + provenance.json), never calls the
generator, so it also runs on a jury copy with permuted rows and renamed ids:

    python -m audit_env verify <episode_dir> [<episode_dir> ...]
    python -m audit_env verify --split dev            (episodes/ + grader_data/)

1. pre-injection books are consistent: the clean state is rebuilt from provenance
   (every change undone, ledger_diff undone) and must show NO violation at all
   (independent detector ``generator.invariants.analyze``, which re-derives every posting
   from documents, fx.csv and policy and does not reuse the posting code); its size
   must equal ``provenance.generation.clean_counts``; re-applying the changes must give
   back exactly the open documents;
2. injection produced exactly the declared discrepancies: on the open data the detector
   finds exactly the truth issues (type, affected_ids, amount, currency) and nothing else;
3. truth agrees with the recorded changes: every issue has a change with its issue_id,
   the same type and affected_ids, and every change belongs to an issue; a clean
   episode has neither;
4. evidence: every acceptable evidence set re-derives its issue from ONLY those records
   (``generator.evidence``) and every id of it is reachable through the agent's tools
   (checked against the same read-only Store the environment uses);
5. formats of section 03: ISO-8601 timestamps with offset, integer minor units (not
   bool), decimal-string FX rates, explicit nulls (JSON null / empty CSV field), known
   currency exponents, manifest counts equal to the files;
6. no contradictory data: ids unique per namespace, every reference resolves or is null.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import jsonutil, schema
from .env.store import Store, StoreError
from .generator import evidence, invariants
from .generator.writer import closed_dir_of, read_closed, read_open

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?[+-]\d{2}:\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RATE_RE = re.compile(r"^\d+(\.\d+)?$")
MANIFEST_KEYS = ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                 "record_counts", "document_count", "rounding_rules")


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ----------------------------------------------------------------------------- formats
def check_formats(bundle: dict) -> list[str]:
    p: list[str] = []
    m = bundle["manifest"]
    for k in MANIFEST_KEYS:
        if k not in m:
            p.append(f"manifest: missing {k}")
    exps = m.get("currency_exponents") or bundle["policy"].get("rounding", {}).get("currency_exponents", {})
    counts = {"ledger": len(bundle["ledger"]), "invoices": len(bundle["invoices"]),
              "payments": len(bundle["payments"]), "fx": len(bundle["fx"])}
    if m.get("record_counts") != counts:
        p.append(f"manifest: record_counts {m.get('record_counts')} != files {counts}")
    if m.get("document_count") != counts["invoices"] + counts["payments"]:
        p.append("manifest: document_count != invoices + payments")
    for k in ("period_start", "period_end"):
        if not TS_RE.match(str(m.get(k, ""))):
            p.append(f"manifest: {k} not ISO-8601 with offset")
    for r in bundle["ledger"]:
        rid = r.get("row_id")
        if not TS_RE.match(str(r.get("timestamp"))):
            p.append(f"ledger {rid}: timestamp without offset")
        if not (_int(r.get("debit_minor")) and _int(r.get("credit_minor"))):
            p.append(f"ledger {rid}: amounts not integer minor units")
        if r.get("currency") not in exps:
            p.append(f"ledger {rid}: unknown currency exponent")
        for k in ("entity_id", "invoice_id", "payment_id"):
            if k not in r or r[k] == "":
                p.append(f"ledger {rid}: {k} not an explicit null/value")
    for d in bundle["invoices"]:
        did = d.get("doc_id")
        for k in ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor", "line_items"):
            if k not in d:
                p.append(f"invoice {did}: missing {k}")
        for k in ("issued_at", "due_at"):
            if not TS_RE.match(str(d.get(k))):
                p.append(f"invoice {did}: {k} without offset")
        if not _int(d.get("total_minor")) or d.get("currency") not in exps:
            p.append(f"invoice {did}: total/currency format")
        for li in d.get("line_items") or []:
            if not all(_int(li.get(k)) for k in ("quantity", "unit_price_minor", "amount_minor")):
                p.append(f"invoice {did}: line item amounts not integer")
    for d in bundle["payments"]:
        did = d.get("doc_id")
        for k in ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"):
            if k not in d:
                p.append(f"payment {did}: missing {k} (null must be explicit)")
        if not TS_RE.match(str(d.get("timestamp"))):
            p.append(f"payment {did}: timestamp without offset")
        if not _int(d.get("amount_minor")) or d.get("currency") not in exps:
            p.append(f"payment {did}: amount/currency format")
        refs = d.get("invoice_refs")
        if refs is not None and not (isinstance(refs, list) and refs and
                                     all(isinstance(x, dict) and _int(x.get("amount_minor")) and x.get("invoice_id")
                                         for x in refs)):
            p.append(f"payment {did}: invoice_refs must be null or a non-empty list of refs")
    for r in bundle["fx"]:
        if not DATE_RE.match(str(r.get("date"))) or not isinstance(r.get("rate"), str) or not RATE_RE.match(r["rate"]):
            p.append(f"fx {r.get('fx_id')}: date/rate format (rate must be a decimal string)")
    return p


# ----------------------------------------------------------------------------- pre-injection state
def rebuild_clean(bundle: dict, provenance: dict) -> dict:
    """Undo every recorded change -> the pre-injection open bundle."""
    docs = {d["doc_id"]: ("invoices", jsonutil.deep_copy(d)) for d in bundle["invoices"]}
    docs.update({d["doc_id"]: ("payments", jsonutil.deep_copy(d)) for d in bundle["payments"]})
    removed: set[str] = set()
    for ch in reversed(provenance.get("changes", [])):
        tgt, act = ch["target_doc_id"], ch["action"]
        if act.startswith("insert_document"):
            removed.add(tgt)
        elif act in ("modify_document", "document_state") and tgt in docs:
            docs[tgt][1][ch["field"]] = jsonutil.deep_copy(ch["before"])
        elif act == "modify_document+override_posting" and isinstance(ch["before"], dict):
            docs[tgt] = (docs[tgt][0], jsonutil.deep_copy(ch["before"]))
    for tgt in removed:
        docs.pop(tgt, None)
    diff = provenance.get("ledger_diff", {"added": [], "removed": [], "changed": []})
    added = {r["row_id"] for r in diff["added"]}
    before = {c["after"]["row_id"]: c["before"] for c in diff["changed"]}
    ledger = [jsonutil.deep_copy(before.get(r["row_id"], r)) for r in bundle["ledger"] if r["row_id"] not in added]
    ledger += [jsonutil.deep_copy(r) for r in diff["removed"]]
    out = {k: bundle[k] for k in ("manifest", "policy", "fx")}
    out["invoices"] = [d for kind, d in docs.values() if kind == "invoices"]
    out["payments"] = [d for kind, d in docs.values() if kind == "payments"]
    out["ledger"] = ledger
    return out


def _reapply(clean: dict, provenance: dict) -> dict[str, dict]:
    docs = {d["doc_id"]: jsonutil.deep_copy(d) for d in clean["invoices"] + clean["payments"]}
    for ch in provenance.get("changes", []):
        tgt, act = ch["target_doc_id"], ch["action"]
        if act.startswith("insert_document"):
            docs[tgt] = jsonutil.deep_copy(ch["after"])
        elif act in ("modify_document", "document_state"):
            docs[tgt][ch["field"]] = jsonutil.deep_copy(ch["after"])
        elif act == "modify_document+override_posting" and isinstance(ch["after"], dict):
            docs[tgt] = jsonutil.deep_copy(ch["after"])
    return docs


def check_pre_injection(bundle: dict, provenance: dict) -> list[str]:
    p: list[str] = []
    clean = rebuild_clean(bundle, provenance)
    for v in invariants.check_clean(clean):
        p.append(f"pre-injection books inconsistent: {v.get('code')} {v.get('ids')}")
    want = provenance.get("generation", {}).get("clean_counts")
    got = {"ledger": len(clean["ledger"]), "invoices": len(clean["invoices"]), "payments": len(clean["payments"])}
    if want is not None and want != got:
        p.append(f"rebuilt pre-injection counts {got} != provenance clean_counts {want}")
    after = _reapply(clean, provenance)
    now = {d["doc_id"]: d for d in bundle["invoices"] + bundle["payments"]}
    if jsonutil.dumps(after) != jsonutil.dumps(now):
        p.append("re-applying provenance changes does not give the open documents")
    return p


# ----------------------------------------------------------------------------- truth vs changes
def check_truth_vs_changes(truth: dict, provenance: dict) -> list[str]:
    p: list[str] = []
    issues = {i["issue_id"]: i for i in truth["issues"]}
    changes = provenance.get("changes", [])
    for iid in sorted(issues):
        mine = [c for c in changes if c.get("issue_id") == iid]
        if not mine:
            p.append(f"{iid}: no recorded change")
        for c in mine:
            if c["type"] != issues[iid]["type"] or sorted(c["affected_ids"]) != sorted(issues[iid]["affected_ids"]):
                p.append(f"{iid}: change type/affected_ids differ from truth")
    for c in changes:
        if c.get("issue_id") not in issues:
            p.append(f"change on {c.get('target_doc_id')} belongs to no truth issue")
    return p


# ----------------------------------------------------------------------------- evidence reachability
def check_reachable(bundle: dict, truth: dict) -> list[str]:
    store = Store(bundle)
    p: list[str] = []
    for i in truth["issues"]:
        for s in i["acceptable_evidence_sets"]:
            for e in s:
                if _reachable(store, e):
                    continue
                p.append(f"{i['issue_id']}: evidence {e} not reachable through the tools")
    return p


def _reachable(store: Store, eid: str) -> bool:
    try:
        store.get_document(eid)
        return True
    except StoreError:
        pass
    for table in ("ledger", "invoices", "payments", "fx"):
        try:
            store.get_record(table, eid)
            return True
        except StoreError:
            pass
    try:
        return bool(store.list_records("ledger", None, 1, {"txn_id": eid})["records"])
    except StoreError:
        return False


# ----------------------------------------------------------------------------- entry points
def verify_bundle(bundle: dict, truth: dict, provenance: dict) -> list[str]:
    problems = check_formats(bundle)
    problems += [f"schema provenance: {e}" for e in schema.validate(provenance, "provenance")]
    problems += [f"schema truth: {e}" for e in schema.validate(truth, "truth")]
    if problems:  # the deeper checks assume well-formed data
        return problems
    stages = (
        ("pre-injection", lambda: check_pre_injection(bundle, provenance)),
        ("declared discrepancies", lambda: invariants.compare_with_truth(bundle, truth)),
        ("truth vs changes", lambda: check_truth_vs_changes(truth, provenance)),
        ("evidence sufficiency", lambda: evidence.check_sufficiency(bundle, truth)),
        ("evidence reachability", lambda: check_reachable(bundle, truth)),
    )
    for name, run in stages:
        try:
            problems += run()
        except (KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
            problems.append(f"{name}: data cannot be checked ({type(e).__name__})")
    return problems


def verify_dir(open_dir: Path, closed_dir: Path | None = None) -> list[str]:
    bundle = read_open(open_dir)
    truth, provenance = read_closed(closed_dir or closed_dir_of(open_dir))
    return verify_bundle(bundle, truth, provenance)


def main_paths(paths: list[Path]) -> int:
    bad = 0
    for d in paths:
        problems = verify_dir(d)
        print(f"{'OK  ' if not problems else 'FAIL'} {d}" + "".join(f"\n     - {x}" for x in problems[:20]))
        bad += bool(problems)
    return bad


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(main_paths([Path(x) for x in sys.argv[1:]]))
