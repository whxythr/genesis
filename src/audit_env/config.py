"""Generator configuration: defaults, strict deep-merge, stable config_id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import jsonutil

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.json"
ERROR_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_error", "period_cutoff")
SCENARIOS = ("partial_fx_payment_at_cutoff", "partially_paid_duplicate")


class ConfigError(ValueError):
    pass


def load_default() -> dict:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _merge(base: Any, override: Any, path: str) -> Any:
    if isinstance(base, dict):
        if not isinstance(override, dict):
            raise ConfigError(f"{path}: expected object")
        out = dict(base)
        for key in sorted(override):
            if key not in base:
                raise ConfigError(f"{path}.{key}: unknown config key")
            out[key] = _merge(base[key], override[key], f"{path}.{key}")
        return out
    return override


def _range(cfg: dict, key: str, lo: int = 0) -> None:
    v = cfg[key]
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v) and lo <= v[0] <= v[1]):
        raise ConfigError(f"{key}: expected [min, max]")


def resolve(override: dict | None = None) -> dict:
    """Default config deep-merged with ``override``; validated."""
    cfg = load_default()
    if override:
        override = {k: v for k, v in override.items() if k not in ("seeds", "split")}
        cfg = _merge(cfg, override, "$")
    for key in ("n_entities", "n_invoices", "n_background_txns"):
        _range(cfg, key, 1 if key != "n_background_txns" else 0)
    for key in sorted(cfg["distractors"]):
        _range(cfg["distractors"], key)
    for et in ERROR_TYPES:
        _range(cfg["errors"][et], "count")
    if not 0 <= cfg["max_errors"] <= 12:
        raise ConfigError("max_errors must be in 0..12")
    if not 0.0 <= cfg["clean_probability"] <= 1.0:
        raise ConfigError("clean_probability must be in [0, 1]")
    if not 20 <= cfg["max_documents"] <= 200:
        raise ConfigError("max_documents must be in 20..200")
    if not 200 <= cfg["max_ledger_rows"] <= 2000:
        raise ConfigError("max_ledger_rows must be in 200..2000")
    return cfg


def config_id(cfg: dict) -> str:
    return f"{cfg['config_name']}-{jsonutil.digest(cfg, 10)}"
