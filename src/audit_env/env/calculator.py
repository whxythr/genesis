"""Safe decimal calculator: own tokenizer + recursive-descent parser. No eval/exec.

Grammar:
    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := ('+' | '-') factor | '(' expr ')' | NUMBER
    NUMBER := digits ['.' digits] | '.' digits
Limits: expression length <= 200, nesting depth <= 20, |value| <= 1e18,
result rounded half-up to 12 decimal places.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, localcontext

MAX_LEN = 200
MAX_DEPTH = 20
MAX_ABS = Decimal(10) ** 18
_QUANT = Decimal(1).scaleb(-12)


class CalcError(ValueError):
    pass


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
        elif ch in "+-*/()":
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            while j < n and text[j].isdigit():
                j += 1
            if j < n and text[j] == ".":
                j += 1
                k = j
                while j < n and text[j].isdigit():
                    j += 1
                if j == k:
                    raise CalcError("bad number")
            if j == i or text[i:j] == ".":
                raise CalcError("bad number")
            if len(text[i:j]) > 40:
                raise CalcError("number too long")
            tokens.append(text[i:j])
            i = j
        else:
            raise CalcError("bad character")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.t = tokens
        self.i = 0
        self.depth = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise CalcError("unexpected end")
        self.i += 1
        return tok

    def _check(self, v: Decimal) -> Decimal:
        if not v.is_finite() or abs(v) > MAX_ABS:
            raise CalcError("value out of range")
        return v

    def expr(self) -> Decimal:
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            r = self.term()
            v = self._check(v + r if op == "+" else v - r)
        return v

    def term(self) -> Decimal:
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            r = self.factor()
            if op == "/":
                if r == 0:
                    raise CalcError("division by zero")
                v = self._check(v / r)
            else:
                v = self._check(v * r)
        return v

    def factor(self) -> Decimal:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError("too deep")
        try:
            tok = self.take()
            if tok in ("+", "-"):
                v = self.factor()
                return -v if tok == "-" else v
            if tok == "(":
                v = self.expr()
                if self.take() != ")":
                    raise CalcError("expected )")
                return v
            if tok in ("*", "/", ")"):
                raise CalcError("unexpected operator")
            return self._check(Decimal(tok))
        finally:
            self.depth -= 1


def evaluate(expression: str) -> str:
    """Evaluate and return a plain decimal string (no exponent)."""
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    if not expression.strip() or len(expression) > MAX_LEN:
        raise CalcError("bad length")
    tokens = _tokenize(expression)
    with localcontext() as ctx:
        ctx.prec = 40
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            p = _Parser(tokens)
            value = p.expr()
            if p.peek() is not None:
                raise CalcError("trailing input")
            value = value.quantize(_QUANT, rounding=ROUND_HALF_UP)
        except (InvalidOperation, DivisionByZero, OverflowError):
            raise CalcError("arithmetic error") from None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text
