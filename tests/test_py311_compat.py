import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_py311  # noqa: E402


def test_no_python312_only_syntax():
    files = sorted(p for d in ("src", "tests", "scripts") for p in (check_py311.ROOT / d).rglob("*.py"))
    assert [p for f in files for p in check_py311.check_file(f)] == []
