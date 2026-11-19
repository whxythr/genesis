"""Money arithmetic: integer minor units, currency exponents, Decimal FX conversion.

Conventions (published, see docs/DESIGN.md):
- Amounts are ``int`` minor units of their currency (never floats).
- An FX rate is a decimal string: units of *target* currency per one unit of
  *source* currency (major units).
- Conversion result is rounded to the target currency's minor unit with
  ROUND_HALF_UP (half away from zero).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, localcontext

EXPONENTS: dict[str, int] = {
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "KWD": 3,
    "KZT": 2,
    "USD": 2,
}
ROUNDING = ROUND_HALF_UP
ROUNDING_NAME = "ROUND_HALF_UP"
_PRECISION = 50
_RATE_RE = re.compile(r"^[0-9]{1,12}(\.[0-9]{1,12})?$")


def exponent(currency: str) -> int:
    try:
        return EXPONENTS[currency]
    except KeyError:
        raise ValueError("unknown currency") from None


def parse_rate(text: str) -> Decimal:
    """Parse a strictly formatted positive decimal rate string."""
    if not isinstance(text, str) or not _RATE_RE.match(text):
        raise ValueError("bad rate format")
    value = Decimal(text)
    if value <= 0:
        raise ValueError("rate must be positive")
    return value


def format_rate(value: Decimal, significant: int = 6) -> str:
    """Round to ``significant`` significant digits and render without exponent."""
    if value <= 0:
        raise ValueError("rate must be positive")
    with localcontext() as ctx:
        ctx.prec = significant
        ctx.rounding = ROUND_HALF_UP
        rounded = +value
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _check_int(amount_minor: int) -> None:
    if not isinstance(amount_minor, int) or isinstance(amount_minor, bool):
        raise TypeError("amount must be int minor units")


def convert(amount_minor: int, source: str, target: str, rate: Decimal | str) -> int:
    """Convert ``amount_minor`` of ``source`` into minor units of ``target``."""
    _check_int(amount_minor)
    if source == target:
        exponent(source)
        return amount_minor
    r = rate if isinstance(rate, Decimal) else parse_rate(rate)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        scaled = Decimal(amount_minor).scaleb(exponent(target) - exponent(source)) * r
        return int(scaled.quantize(Decimal(1), rounding=ROUNDING))


def to_major(amount_minor: int, currency: str) -> Decimal:
    _check_int(amount_minor)
    return Decimal(amount_minor).scaleb(-exponent(currency))


def from_major(value: Decimal | str, currency: str) -> int:
    """Major units (decimal) -> minor units, rounded half-up."""
    d = Decimal(value) if isinstance(value, str) else value
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return int(d.scaleb(exponent(currency)).quantize(Decimal(1), rounding=ROUNDING))
