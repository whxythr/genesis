"""Static check for syntax/stdlib that would break on Python 3.11 (when only 3.12+ is installed).

Flags: f-strings reusing their own quote or containing a backslash inside a replacement
field (PEP 701, 3.12+), PEP 695 generics/`type` statements, and a few 3.12+ stdlib APIs.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STDLIB_312 = re.compile(r"\bitertools\.batched\b|\bbatched\(|\bPath\.walk\b|\.walk\(\)|datetime\.UTC\b")


def check_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    problems = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if type(node).__name__ in ("TypeAlias", "TypeVar", "ParamSpec", "TypeVarTuple"):
            problems.append(f"{path}:{node.lineno}: PEP 695 syntax")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and getattr(node, "type_params", None):
            problems.append(f"{path}:{node.lineno}: PEP 695 type params")
    if hasattr(tokenize, "FSTRING_START"):
        stack = []  # (quote, depth_of_braces)
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.FSTRING_START:
                quote = tok.string.lstrip("rRfFbB")
                if stack and stack[-1].startswith(quote[0]) and len(stack[-1]) == len(quote):
                    problems.append(f"{path}:{tok.start[0]}: nested f-string reuses quote {quote}")
                stack.append(quote)
            elif tok.type == tokenize.FSTRING_END:
                stack.pop()
            elif stack and tok.type == tokenize.STRING:
                inner = tok.string.lstrip("rRbBuU")
                if inner.startswith(stack[-1][0]):
                    problems.append(f"{path}:{tok.start[0]}: string inside f-string reuses quote")
                if "\\" in tok.string:
                    problems.append(f"{path}:{tok.start[0]}: backslash inside f-string expression")
    for n, line in enumerate(src.splitlines(), 1):
        if STDLIB_312.search(line):
            problems.append(f"{path}:{n}: 3.12+ stdlib API")
    return problems


def main() -> int:
    files = sorted(p for d in ("src", "tests", "scripts") for p in (ROOT / d).rglob("*.py"))
    problems = [p for f in files for p in check_file(f)]
    for p in problems:
        print(p)
    print(f"checked {len(files)} files: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
