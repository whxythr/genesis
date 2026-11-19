"""ISO-8601 timestamps with explicit offsets. Fixed offsets only (no tz database)."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_OFFSET_RE = re.compile(r"^([+-])([0-9]{2}):([0-9]{2})$")
_TS_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}$")


def tz_from_offset(offset: str) -> timezone:
    m = _OFFSET_RE.match(offset or "")
    if not m:
        raise ValueError("bad utc offset")
    sign = 1 if m.group(1) == "+" else -1
    delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    if delta > timedelta(hours=14):
        raise ValueError("bad utc offset")
    return timezone(sign * delta)


def offset_minutes(offset: str) -> int:
    return int(tz_from_offset(offset).utcoffset(None).total_seconds() // 60)


def parse_ts(text: str) -> datetime:
    """Parse ``YYYY-MM-DDTHH:MM:SS+HH:MM``; offset is mandatory."""
    if not isinstance(text, str) or not _TS_RE.match(text):
        raise ValueError("bad timestamp")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("timestamp without offset")
    return dt


def fmt_ts(dt: datetime, offset: str) -> str:
    return dt.astimezone(tz_from_offset(offset)).isoformat(timespec="seconds")


def local_date(dt: datetime, offset: str) -> date:
    return dt.astimezone(tz_from_offset(offset)).date()


def in_period(dt: datetime, start: datetime, end: datetime) -> bool:
    """Period is [start, end): start inclusive, end exclusive."""
    return start <= dt < end
