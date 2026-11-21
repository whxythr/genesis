"""Read-only, in-memory access to the OPEN records of one episode.

The store never touches the filesystem: it is built from the open bundle received
from the grader process at reset. Records are returned as deep copies.
Cursors are stable (same episode + query + offset -> same string) and signed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from .. import jsonutil

PAGE_MAX = 50

TABLES = {
    "ledger": {"key": "row_id",
               "filters": ("txn_id", "account", "entity_id", "invoice_id", "payment_id")},
    "invoices": {"key": "invoice_id", "filters": ("entity_id", "currency", "doc_id")},
    "payments": {"key": "payment_id", "filters": ("entity_id", "currency", "kind", "doc_id")},
    "fx": {"key": "fx_id", "filters": ("date", "source_currency")},
}


class StoreError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _invoice_summary(inv: dict) -> dict:
    out = {k: v for k, v in inv.items() if k != "line_items"}
    out["line_item_count"] = len(inv["line_items"])
    return out


class Store:
    def __init__(self, bundle: dict) -> None:
        self.manifest = jsonutil.deep_copy(bundle["manifest"])
        self.policy = jsonutil.deep_copy(bundle["policy"])
        self._rows = {
            "ledger": [jsonutil.deep_copy(r) for r in bundle["ledger"]],
            "invoices": [_invoice_summary(jsonutil.deep_copy(i)) for i in bundle["invoices"]],
            "payments": [jsonutil.deep_copy(p) for p in bundle["payments"]],
            "fx": [jsonutil.deep_copy(r) for r in bundle["fx"]],
        }
        self._index = {t: {r[TABLES[t]["key"]]: r for r in rows} for t, rows in self._rows.items()}
        # documented secondary keys (used only when the primary key does not exist):
        # invoices/payments also by doc_id, fx also by "date:source_currency:target_currency"
        self._alt = {
            "ledger": {},
            "invoices": {r["doc_id"]: r for r in self._rows["invoices"]},
            "payments": {r["doc_id"]: r for r in self._rows["payments"]},
            "fx": {f"{r['date']}:{r['source_currency']}:{r['target_currency']}": r for r in self._rows["fx"]},
        }
        self._docs = {d["doc_id"]: ("invoice", jsonutil.deep_copy(d)) for d in bundle["invoices"]}
        self._docs.update({d["doc_id"]: ("payment", jsonutil.deep_copy(d)) for d in bundle["payments"]})
        self._key = hashlib.sha256(f"cursor|{self.manifest['episode_id']}".encode()).digest()

    # -- cursors ---------------------------------------------------------------
    def _sign(self, body: str) -> str:
        return hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()[:16]

    def _cursor(self, table: str, fdigest: str, offset: int) -> str:
        body = base64.urlsafe_b64encode(jsonutil.dumps({"f": fdigest, "o": offset, "t": table}).encode()).decode().rstrip("=")
        return f"{body}.{self._sign(body)}"

    def _decode(self, cursor: str, table: str, fdigest: str) -> int:
        if not isinstance(cursor, str) or len(cursor) > 200 or cursor.count(".") != 1:
            raise StoreError("BAD_CURSOR")
        body, sig = cursor.split(".")
        if not hmac.compare_digest(sig, self._sign(body)):
            raise StoreError("BAD_CURSOR")
        try:
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        except (ValueError, TypeError):
            raise StoreError("BAD_CURSOR") from None
        if payload.get("t") != table or payload.get("f") != fdigest:
            raise StoreError("BAD_CURSOR")
        offset = payload.get("o")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise StoreError("BAD_CURSOR")
        return offset

    # -- queries ---------------------------------------------------------------
    def list_records(self, table: str, cursor: str | None, limit: int, flt: dict | None) -> dict:
        if table not in TABLES:
            raise StoreError("UNKNOWN_TABLE")
        flt = flt or {}
        for k in sorted(flt):
            v = flt[k]
            if k not in TABLES[table]["filters"] or not (v is None or isinstance(v, str)) or (isinstance(v, str) and len(v) > 64):
                raise StoreError("BAD_ARGS")
        fdigest = jsonutil.digest(flt, 16)
        offset = 0 if cursor is None else self._decode(cursor, table, fdigest)
        rows = self._rows[table]
        if flt:
            rows = [r for r in rows if all(r.get(k) == flt[k] for k in flt)]
        if offset > len(rows):
            raise StoreError("BAD_CURSOR")
        page = rows[offset: offset + limit]
        nxt = offset + len(page)
        return {
            "table": table,
            "records": jsonutil.deep_copy(page),
            "total": len(rows),
            "next_cursor": self._cursor(table, fdigest, nxt) if nxt < len(rows) else None,
        }

    def get_record(self, table: str, rid: str) -> dict:
        if table not in TABLES:
            raise StoreError("UNKNOWN_TABLE")
        rec = self._index[table].get(rid)
        if rec is None:
            rec = self._alt[table].get(rid)
        if rec is None:
            raise StoreError("NOT_FOUND")
        return {"table": table, "record": jsonutil.deep_copy(rec)}

    def get_document(self, doc_id: str) -> dict:
        hit = self._docs.get(doc_id)
        if hit is None:
            raise StoreError("NOT_FOUND")
        return {"doc_type": hit[0], "document": jsonutil.deep_copy(hit[1])}
