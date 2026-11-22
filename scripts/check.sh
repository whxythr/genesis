#!/usr/bin/env sh
# One-command check on Linux/macOS without Docker:
#
#   sh scripts/check.sh
#
# On a fresh clone it creates .venv with Python >= 3.11 and installs the pinned
# dependencies from requirements.lock (needs internet once); afterwards it runs offline.
set -eu
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
    base=""
    for c in python3.11 python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
            base="$c"; break
        fi
    done
    if [ -z "$base" ]; then
        echo "Python 3.11+ not found. Install it and run this script again." >&2
        exit 1
    fi
    echo "==> creating .venv with: $base"
    "$base" -m venv .venv
fi

if ! .venv/bin/python -c 'import pytest' 2>/dev/null; then
    echo "==> installing pinned dependencies from requirements.lock"
    .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.lock
fi

PYTHONPATH=src exec .venv/bin/python -m audit_env check
