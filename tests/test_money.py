from decimal import Decimal

import pytest

from audit_env import money
from audit_env.timeutil import local_date, parse_ts


def test_exponents():
    assert money.exponent("JPY") == 0
    assert money.exponent("KZT") == 2
    assert money.exponent("KWD") == 3
    with pytest.raises(ValueError):
        money.exponent("XXX")


def test_convert_scales_between_exponents():
    # 1000 JPY at 3.2 KZT/JPY = 3200.00 KZT
    assert money.convert(1000, "JPY", "KZT", "3.2") == 320000
    # 1.000 KWD (1000 minor) at 3.25 USD/KWD = 3.25 USD
    assert money.convert(1000, "KWD", "USD", "3.25") == 325
    # 1.00 USD -> JPY at 149.5 = 149.5 -> half up 150
    assert money.convert(100, "USD", "JPY", "149.5") == 150


def test_convert_half_up_and_identity():
    assert money.convert(1, "EUR", "USD", "1.5") == 2
    assert money.convert(1, "EUR", "USD", "1.49") == 1
    assert money.convert(12345, "USD", "USD", "1") == 12345
    assert money.convert(12345, "USD", "USD", "999") == 12345  # same currency: no conversion


def test_convert_rejects_floats_and_bad_rates():
    with pytest.raises(TypeError):
        money.convert(1.5, "USD", "EUR", "1.1")
    with pytest.raises(TypeError):
        money.convert(True, "USD", "EUR", "1.1")
    for bad in ["1e3", "-1", "abc", "", "0", "1.2.3", " 1.1"]:
        with pytest.raises(ValueError):
            money.convert(100, "USD", "EUR", bad)


def test_format_rate_is_plain_decimal():
    assert money.format_rate(Decimal("0.0021000449")) == "0.00210004"
    assert money.format_rate(Decimal("0.00210000")) == "0.0021"
    assert money.format_rate(Decimal("514.28571")) == "514.286"
    assert "E" not in money.format_rate(Decimal("0.0000012345678"))


def test_local_date_uses_offset():
    ts = parse_ts("2026-03-31T20:30:00+00:00")
    assert local_date(ts, "+05:00").isoformat() == "2026-04-01"
    assert local_date(ts, "-03:00").isoformat() == "2026-03-31"
    with pytest.raises(ValueError):
        parse_ts("2026-03-31T20:30:00")
