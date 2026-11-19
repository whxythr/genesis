"""The five agent tools: argument validation and dispatch onto the read-only store.

Every failure maps to a fixed error code with a fixed short message: user input is
never echoed back, and no paths, stack traces or hints leak into observations.
"""

from __future__ import annotations

from .calculator import CalcError, evaluate
from .store import PAGE_MAX, Store, StoreError

TOOLS = ("list_records", "get_record", "get_document", "calculator", "submit")

ERROR_MESSAGES = {
    "UNKNOWN_ACTION": "unknown tool",
    "BAD_ARGS": "invalid arguments",
    "UNKNOWN_TABLE": "unknown table",
    "NOT_FOUND": "record not found",
    "BAD_CURSOR": "invalid cursor",
    "CALC_ERROR": "expression rejected",
    "EPISODE_DONE": "episode is over; call reset",
    "TIMEOUT": "episode time limit reached",
}

ARG_SPEC = {
    "list_records": ({"table"}, {"table", "cursor", "limit", "filter"}),
    "get_record": ({"table", "id"}, {"table", "id"}),
    "get_document": ({"doc_id"}, {"doc_id"}),
    "calculator": ({"expression"}, {"expression"}),
    "submit": (set(), {"findings"}),
}

TOOL_HELP = {
    "list_records": {"args": {"table": "ledger | invoices | payments | fx",
                              "cursor": "next_cursor from a previous page, or null",
                              "limit": f"1..{PAGE_MAX} (default {PAGE_MAX})",
                              "filter": "optional {field: value} exact match; fields: "
                                        "ledger(txn_id, account, entity_id, invoice_id, payment_id), "
                                        "invoices(entity_id, currency, doc_id), "
                                        "payments(entity_id, currency, kind, doc_id), fx(date, source_currency)"},
                     "returns": "records, total, next_cursor; invoices are summaries without line_items"},
    "get_record": {"args": {"table": "ledger | invoices | payments | fx",
                            "id": "row_id | invoice_id | payment_id | fx_id"}},
    "get_document": {"args": {"doc_id": "doc_id of an invoice or payment"},
                     "returns": "full document (invoices include line_items)"},
    "calculator": {"args": {"expression": "decimal arithmetic with + - * / ( ), <= 200 chars"},
                   "returns": "value as decimal string, rounded half-up to 12 places"},
    "submit": {"args": {"findings": "object matching findings.schema.json"},
               "returns": "ends the episode"},
}


class ToolError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _str(v, max_len: int = 200) -> str:
    if not isinstance(v, str) or not v or len(v) > max_len:
        raise ToolError("BAD_ARGS")
    return v


def check_args(tool: str, args) -> dict:
    if tool not in ARG_SPEC:
        raise ToolError("UNKNOWN_ACTION")
    if not isinstance(args, dict):
        raise ToolError("BAD_ARGS")
    required, allowed = ARG_SPEC[tool]
    keys = set(args)
    if not required <= keys or not keys <= allowed:
        raise ToolError("BAD_ARGS")
    return args


def run(store: Store, tool: str, args: dict) -> dict:
    """Execute a non-submit tool. Raises ToolError with a fixed code."""
    args = check_args(tool, args)
    try:
        if tool == "list_records":
            table = _str(args["table"], 32)
            cursor = args.get("cursor")
            if cursor is not None:
                cursor = _str(cursor)
            limit = args.get("limit", PAGE_MAX)
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= PAGE_MAX:
                raise ToolError("BAD_ARGS")
            flt = args.get("filter")
            if flt is not None and (not isinstance(flt, dict) or len(flt) > 3):
                raise ToolError("BAD_ARGS")
            return store.list_records(table, cursor, limit, flt)
        if tool == "get_record":
            return store.get_record(_str(args["table"], 32), _str(args["id"], 64))
        if tool == "get_document":
            return store.get_document(_str(args["doc_id"], 64))
        if tool == "calculator":
            expr = args["expression"]
            if not isinstance(expr, str):
                raise ToolError("BAD_ARGS")
            try:
                return {"value": evaluate(expr)}
            except CalcError:
                raise ToolError("CALC_ERROR") from None
    except StoreError as e:
        raise ToolError(e.code) from None
    raise ToolError("UNKNOWN_ACTION")
