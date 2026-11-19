"""Minimal JSON Schema subset validator (stdlib only).

Supported keywords: type, properties, required, additionalProperties (bool or schema),
items, minItems, maxItems, minLength, maxLength, enum, const, minimum, maximum,
pattern, anyOf. That is all our published schemas in ``schemas/`` use.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


class SchemaError(ValueError):
    pass


def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    raise SchemaError(f"unsupported type {t}")


def errors(value: Any, schema: dict, path: str = "$") -> list[str]:
    out: list[str] = []
    if "anyOf" in schema:
        if not any(not errors(value, s, path) for s in schema["anyOf"]):
            out.append(f"{path}: matches no alternative")
        return out
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            return [f"{path}: expected {'/'.join(types)}"]
    if "const" in schema and value != schema["const"]:
        out.append(f"{path}: const mismatch")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: not in enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path}: too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(f"{path}: pattern mismatch")
    if _type_ok(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path}: above maximum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            out.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{path}: too many items")
        if "items" in schema:
            for i, item in enumerate(value):
                out.extend(errors(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{path}: missing {key}")
        addl = schema.get("additionalProperties", True)
        for key in sorted(value):
            if key in props:
                out.extend(errors(value[key], props[key], f"{path}.{key}"))
            elif addl is False:
                out.append(f"{path}: unexpected key")
            elif isinstance(addl, dict):
                out.extend(errors(value[key], addl, f"{path}.{key}"))
    return out


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as fh:
        return json.load(fh)


def validate(value: Any, name: str) -> list[str]:
    return errors(value, load(name))
