"""Opaque, collision-checked identifiers.

IDs carry only a table prefix (``INV-``, ``PAY-``...) and random characters from an
alphabet without 0/1/I/O. They never encode order, error status or meaning.
"""

from __future__ import annotations

import hashlib
import random

ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
PREFIXES = {
    "entity": "E-",
    "invoice": "INV-",
    "payment": "PAY-",
    "doc": "D-",
    "txn": "T-",
    "row": "R-",
    "fx": "FX-",
    "episode": "EP-",
}


class IdFactory:
    """Generates unique opaque IDs from an explicit RNG (and a salt for derived IDs)."""

    def __init__(self, rng: random.Random, length: int = 8) -> None:
        self._rng = rng
        self._length = length
        self._used: set[str] = set()
        self._derived: dict[str, str] = {}
        self.salt = "".join(rng.choice(ALPHABET) for _ in range(16))

    def new(self, kind: str) -> str:
        prefix = PREFIXES[kind]
        while True:
            candidate = prefix + "".join(self._rng.choice(ALPHABET) for _ in range(self._length))
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate

    def derived(self, kind: str, material: str) -> str:
        """Stable ID derived from salt+material: the same material always maps to the
        same ID (ledger rows keep their IDs when a transaction is re-posted)."""
        key = f"{kind}|{material}"
        if key in self._derived:
            return self._derived[key]
        prefix = PREFIXES[kind]
        counter = 0
        while True:
            h = hashlib.sha256(f"{self.salt}|{key}|{counter}".encode()).digest()
            candidate = prefix + "".join(ALPHABET[b % len(ALPHABET)] for b in h[: self._length])
            if candidate not in self._used:
                self._used.add(candidate)
                self._derived[key] = candidate
                return candidate
            counter += 1
