"""Canonical JSON helpers (sorted keys, no whitespace variance) for byte-stable output."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pretty(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def digest(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()[:n]


def deep_copy(obj: Any) -> Any:
    return json.loads(dumps(obj))
