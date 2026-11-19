"""Consistent hidden history of a fictional company for one reporting period.

Everything here is document-driven: invoices, payments (receipts and refunds),
background cash transactions and FX rates. The ledger is *derived* from these by
``posting.post_ledger`` so that the clean history balances by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

from .. import money
from ..ids import IdFactory
from ..timeutil import fmt_ts, local_date, offset_minutes, tz_from_offset

USD_VALUE = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.0067"),
    "KWD": Decimal("3.25"),
    "KZT": Decimal("0.0021"),
}
FX_PADDING_DAYS = 7

_SYLLABLES = ["nor", "vex", "tal", "mir", "zen", "kor", "lum", "sar", "bel", "dra", "qui", "tor",
              "ves", "ari", "pol", "nex", "ost", "ran", "fel", "gor", "hal", "ion", "jas", "kel"]
_SUFFIXES = ["Trading LLP", "Systems Ltd", "Logistics LLC", "Holdings", "Partners", "Industries",
             "Solutions", "Group", "Supply Co", "Labs"]
_CATALOG = ["Consulting hours", "Software license", "Hardware maintenance", "Logistics service",
            "Training session", "Cloud hosting", "Spare parts kit", "Audit support",
            "Data processing", "Equipment rental", "Translation service", "Field engineering"]
BACKGROUND_KINDS = [("OPEX", "CASH"), ("PAYROLL", "CASH"), ("BANK_FEES", "CASH"), ("CASH", "INTEREST_INCOME")]


@dataclass
class Company:
    episode_id: str
    base_currency: str
    offset: str
    period_start: datetime
    period_end: datetime
    entities: list[dict]
    invoices: list[dict]
    payments: list[dict]
    background: list[dict]
    rates: dict[tuple[str, str], str]
    fx_rows: list[dict]
    txn_of: dict[str, str]          # doc_id -> txn_id
    tags: dict[str, list[str]] = field(default_factory=dict)  # doc_id -> distractor tags (hidden)

    def invoice_by_id(self) -> dict[str, dict]:
        return {inv["invoice_id"]: inv for inv in self.invoices}

    def rate(self, currency: str, day: date) -> Decimal:
        if currency == self.base_currency:
            return Decimal(1)
        return money.parse_rate(self.rates[(currency, day.isoformat())])

    def policy_rate(self, currency: str, ts: datetime) -> Decimal:
        return self.rate(currency, local_date(ts, self.offset))

    def last_local_day(self) -> date:
        return (self.period_end - timedelta(seconds=1)).astimezone(tz_from_offset(self.offset)).date()


def _name(rng: random.Random) -> str:
    stem = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()
    return f"{stem} {rng.choice(_SUFFIXES)}"


def _price_minor(rng: random.Random, currency: str, lo_exp: float = 3.0, hi_exp: float = 5.4) -> int:
    usd_cents = int(10 ** rng.uniform(lo_exp, hi_exp))
    major = (Decimal(usd_cents) / 100) / USD_VALUE[currency]
    return max(1, money.from_major(major, currency))


def _rand_ts(rng: random.Random, lo: datetime, hi: datetime) -> datetime:
    span = int((hi - lo).total_seconds())
    if span <= 0:
        return lo
    return lo + timedelta(seconds=rng.randrange(span))


class _Builder:
    def __init__(self, rng: random.Random, cfg: dict, ids: IdFactory, episode_id: str) -> None:
        self.rng = rng
        self.cfg = cfg
        self.ids = ids
        self.episode_id = episode_id
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.txn_of: dict[str, str] = {}
        self.tags: dict[str, list[str]] = {}
        self.used_payment_seconds: set[int] = set()
        self.order_refs: set[int] = set()
        self.doc_budget = cfg["max_documents"] - cfg["max_errors"]

    # ---- setup -------------------------------------------------------------
    def setup(self) -> None:
        rng, cfg = self.rng, self.cfg
        self.base = rng.choice(cfg["base_currencies"])
        pool = [c for c in cfg["foreign_currencies"] if c != self.base]
        self.foreign = sorted(rng.sample(pool, k=min(len(pool), rng.randint(1, 2)))) if cfg["foreign_share"] > 0 else []
        self.offset = rng.choice(cfg["timezones"])
        tz = tz_from_offset(self.offset)
        month = rng.randint(1, 12)
        year = cfg["year"]
        self.start = datetime(year, month, 1, tzinfo=tz)
        self.end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=tz)
        self._make_rates()
        self._make_entities()

    def _make_rates(self) -> None:
        self.rates: dict[tuple[str, str], str] = {}
        self.fx_rows: list[dict] = []
        first = self.start.date() - timedelta(days=FX_PADDING_DAYS)
        last = (self.end - timedelta(seconds=1)).astimezone(tz_from_offset(self.offset)).date() + timedelta(days=FX_PADDING_DAYS)
        for cur in self.foreign:
            level = (USD_VALUE[cur] / USD_VALUE[self.base]) * Decimal(str(round(self.rng.uniform(0.97, 1.03), 6)))
            prev = None
            day = first
            while day <= last:
                level = level * Decimal(str(round(1 + self.rng.uniform(-0.006, 0.006), 6)))
                text = money.format_rate(level)
                if text == prev:
                    level = level * Decimal("1.001")
                    text = money.format_rate(level)
                prev = text
                self.rates[(cur, day.isoformat())] = text
                self.fx_rows.append({"fx_id": self.ids.new("fx"), "date": day.isoformat(),
                                     "source_currency": cur, "target_currency": self.base, "rate": text})
                day += timedelta(days=1)

    def _make_entities(self) -> None:
        rng, cfg = self.rng, self.cfg
        n = rng.randint(*cfg["n_entities"])
        names: set[str] = set()
        self.entities = []
        for _ in range(n):
            name = _name(rng)
            while name in names:
                name = _name(rng)
            names.add(name)
            cur = self.base
            if self.foreign and rng.random() < cfg["foreign_share"]:
                cur = rng.choice(self.foreign)
            bank_offset = "+00:00" if rng.random() < 0.5 else rng.choice(cfg["timezones"])
            self.entities.append({"entity_id": self.ids.new("entity"), "name": name,
                                  "currency": cur, "bank_offset": bank_offset})

    # ---- documents ---------------------------------------------------------
    def _docs(self) -> int:
        return len(self.invoices) + len(self.payments)

    def _order_ref(self) -> int:
        ref = self.rng.randint(10000, 99999)
        while ref in self.order_refs:
            ref = self.rng.randint(10000, 99999)
        self.order_refs.add(ref)
        return ref

    def _invoice(self, entity: dict, issued: datetime, lines: list[dict], tags: list[str]) -> dict:
        for i, line in enumerate(lines, start=1):
            line["line_no"] = i
            line["amount_minor"] = line["quantity"] * line["unit_price_minor"]
        due = issued + timedelta(days=self.rng.choice([14, 30, 45]))
        inv = {
            "doc_id": self.ids.new("doc"),
            "invoice_id": self.ids.new("invoice"),
            "entity_id": entity["entity_id"],
            "entity_name": entity["name"],
            "issued_at": fmt_ts(issued, self.offset),
            "due_at": fmt_ts(due, self.offset),
            "currency": entity["currency"],
            "total_minor": sum(line["amount_minor"] for line in lines),
            "line_items": lines,
        }
        self.invoices.append(inv)
        self.txn_of[inv["doc_id"]] = self.ids.new("txn")
        self.tags[inv["doc_id"]] = sorted(tags)
        return inv

    def _regular_lines(self, currency: str) -> list[dict]:
        ref = self._order_ref()
        lines = []
        for item in self.rng.sample(_CATALOG, k=self.rng.randint(1, 3)):
            lines.append({"description": f"{item}, order {ref}",
                          "quantity": self.rng.randint(1, 12),
                          "unit_price_minor": _price_minor(self.rng, currency, 2.5, 4.6)})
        return lines

    def _payment_ts(self, earliest: datetime, latest: datetime) -> datetime | None:
        if earliest >= latest:
            return None
        ts = _rand_ts(self.rng, earliest, latest)
        sec = int(ts.timestamp())
        while sec in self.used_payment_seconds:
            sec += 1
        if sec >= int(latest.timestamp()):
            return None
        self.used_payment_seconds.add(sec)
        return datetime.fromtimestamp(sec, tz=ts.tzinfo)

    def _payment(self, entity: dict, ts: datetime, amount: int, refs: list[dict] | None,
                 kind: str, tags: list[str], display_offset: str | None = None) -> dict:
        pay = {
            "doc_id": self.ids.new("doc"),
            "payment_id": self.ids.new("payment"),
            "kind": kind,
            "entity_id": entity["entity_id"],
            "timestamp": fmt_ts(ts, display_offset or entity["bank_offset"]),
            "currency": entity["currency"],
            "amount_minor": amount,
            "invoice_refs": refs,
        }
        self.payments.append(pay)
        self.txn_of[pay["doc_id"]] = self.ids.new("txn")
        self.tags[pay["doc_id"]] = sorted(tags)
        return pay

    def build_documents(self) -> None:
        rng, cfg = self.rng, self.cfg
        n_target = rng.randint(*cfg["n_invoices"])
        latest_issue = self.end - timedelta(days=2)
        pay_deadline = self.end - timedelta(minutes=5)
        ent_by_id = {e["entity_id"]: e for e in self.entities}

        # subscriptions: same entity, same amount, weekly, different service-period text
        sub_entities = rng.sample(self.entities, k=min(len(self.entities), rng.randint(*cfg["distractors"]["subscriptions"])))
        plans: list[tuple[dict, str]] = []  # (invoice, pattern)
        for ent in sub_entities:
            price = _price_minor(rng, ent["currency"], 3.0, 4.5)
            first = self.start + timedelta(days=rng.randint(0, 3), hours=rng.randint(8, 18), minutes=rng.randint(0, 59))
            for k in range(rng.randint(2, 4)):
                issued = first + timedelta(days=7 * k)
                if issued >= latest_issue:
                    break
                d0 = local_date(issued, self.offset)
                lines = [{"description": f"Platform subscription, service week {d0.isoformat()}/{(d0 + timedelta(days=6)).isoformat()}",
                          "quantity": 1, "unit_price_minor": price}]
                inv = self._invoice(ent, issued, lines, ["subscription"])
                plans.append((inv, rng.choice(["full", "full", "unpaid"])))

        mix = cfg["payment_mix"]
        patterns = sorted(mix)
        weights = [mix[p] for p in patterns]
        while len(self.invoices) < n_target:
            ent = rng.choice(self.entities)
            issued = _rand_ts(rng, self.start, latest_issue)
            inv = self._invoice(ent, issued, self._regular_lines(ent["currency"]), [])
            plans.append((inv, rng.choices(patterns, weights=weights)[0]))

        # combined payments need >=2 invoices of the same entity
        combined: dict[str, list[dict]] = {}
        for inv, pattern in plans:
            if pattern == "combined":
                combined.setdefault(inv["entity_id"], []).append(inv)
        final_plans = []
        for inv, pattern in plans:
            if pattern == "combined" and len(combined[inv["entity_id"]]) < 2:
                pattern = "full"
            final_plans.append((inv, pattern))

        issued_dt = {inv["invoice_id"]: datetime.fromisoformat(inv["issued_at"]) for inv in self.invoices}
        self.paid_in_full: list[dict] = []
        self.unpaid: list[dict] = []
        for inv, pattern in final_plans:
            ent = ent_by_id[inv["entity_id"]]
            t0 = issued_dt[inv["invoice_id"]]
            if pattern == "combined":
                continue
            if pattern == "unpaid" or self._docs() >= self.doc_budget:
                self.unpaid.append(inv)
            elif pattern == "full":
                ts = self._payment_ts(t0 + timedelta(hours=2), min(t0 + timedelta(days=20), pay_deadline))
                if ts is None:
                    self.unpaid.append(inv)
                    continue
                self._payment(ent, ts, inv["total_minor"], [{"invoice_id": inv["invoice_id"], "amount_minor": inv["total_minor"]}], "receipt", [])
                self.paid_in_full.append(inv)
            elif pattern == "split":
                total = inv["total_minor"]
                n_parts = rng.randint(2, 3) if total >= 3 else 1
                cuts = sorted(rng.sample(range(1, total), k=n_parts - 1)) if n_parts > 1 else []
                parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
                t = t0
                paid = 0
                for part in parts:
                    if self._docs() >= self.doc_budget:
                        break
                    ts = self._payment_ts(t + timedelta(hours=2), min(t + timedelta(days=9), pay_deadline))
                    if ts is None:
                        break
                    self._payment(ent, ts, part, [{"invoice_id": inv["invoice_id"], "amount_minor": part}], "receipt", ["split"])
                    paid += part
                    t = ts
                (self.paid_in_full if paid == total else self.unpaid).append(inv)
        # combined: one payment covering several invoices of one entity
        for eid in sorted(combined):
            group = [inv for inv, p in final_plans if p == "combined" and inv["entity_id"] == eid]
            if len(group) < 2:
                continue
            while group:
                chunk, group = group[:3], group[3:]
                if len(chunk) < 2 or self._docs() >= self.doc_budget:
                    self.unpaid.extend(chunk)
                    continue
                t0 = max(issued_dt[i["invoice_id"]] for i in chunk)
                ts = self._payment_ts(t0 + timedelta(hours=2), min(t0 + timedelta(days=15), pay_deadline))
                if ts is None:
                    self.unpaid.extend(chunk)
                    continue
                refs = [{"invoice_id": i["invoice_id"], "amount_minor": i["total_minor"]} for i in chunk]
                self._payment(ent_by_id[eid], ts, sum(r["amount_minor"] for r in refs), refs, "receipt", ["combined"])
                self.paid_in_full.extend(chunk)

        self._boundary_payments(ent_by_id)
        self._refunds(ent_by_id)

    def _boundary_payments(self, ent_by_id: dict) -> None:
        """Receipts in the last hours of the period, displayed in an offset where the
        calendar date already reads as the next month (still inside the period)."""
        rng = self.rng
        k = rng.randint(*self.cfg["distractors"]["boundary_timezone"])
        company_min = offset_minutes(self.offset)
        touched = {r["invoice_id"] for p in self.payments for r in (p["invoice_refs"] or [])}
        candidates = sorted((i for i in self.unpaid if i["invoice_id"] not in touched), key=lambda i: i["invoice_id"])
        rng.shuffle(candidates)
        for inv in candidates:
            if k <= 0 or self._docs() >= self.doc_budget:
                break
            issued = datetime.fromisoformat(inv["issued_at"])
            ts = self._payment_ts(max(issued + timedelta(hours=2), self.end - timedelta(hours=3)), self.end - timedelta(seconds=30))
            if ts is None:
                continue
            gap = int((self.end - ts).total_seconds() // 60)
            offsets = [o for o in self.cfg["timezones"] if offset_minutes(o) - company_min > gap]
            if not offsets:
                self.used_payment_seconds.discard(int(ts.timestamp()))
                continue
            ent = ent_by_id[inv["entity_id"]]
            self._payment(ent, ts, inv["total_minor"], [{"invoice_id": inv["invoice_id"], "amount_minor": inv["total_minor"]}],
                          "receipt", ["boundary_timezone"], display_offset=rng.choice(offsets))
            self.unpaid.remove(inv)
            self.paid_in_full.append(inv)
            k -= 1

    def _refunds(self, ent_by_id: dict) -> None:
        rng = self.rng
        k = rng.randint(*self.cfg["distractors"]["refunds"])
        last_pay: dict[str, datetime] = {}
        for p in self.payments:
            for ref in p["invoice_refs"] or []:
                ts = datetime.fromisoformat(p["timestamp"])
                if ref["invoice_id"] not in last_pay or ts > last_pay[ref["invoice_id"]]:
                    last_pay[ref["invoice_id"]] = ts
        candidates = sorted((i for i in self.paid_in_full if i["total_minor"] >= 20), key=lambda i: i["invoice_id"])
        rng.shuffle(candidates)
        for inv in candidates[:k]:
            if self._docs() >= self.doc_budget:
                break
            t0 = last_pay[inv["invoice_id"]]
            ts = self._payment_ts(t0 + timedelta(hours=3), min(t0 + timedelta(days=10), self.end - timedelta(hours=1)))
            if ts is None:
                continue
            amount = max(1, inv["total_minor"] * rng.randint(5, 50) // 100)
            ent = ent_by_id[inv["entity_id"]]
            self._payment(ent, ts, amount, [{"invoice_id": inv["invoice_id"], "amount_minor": amount}], "refund", ["refund"])

    def build_background(self, max_rows: int) -> list[dict]:
        rng, cfg = self.rng, self.cfg
        doc_rows = 2 * len(self.invoices) + sum(3 + len(p["invoice_refs"] or []) for p in self.payments)
        headroom = max_rows - doc_rows - 6 * cfg["max_errors"] - 20
        # keep the episode within the documented 200..2000 ledger-row range
        lo = max(0, -(-(200 - doc_rows) // 2))
        hi = max(lo, headroom // 2)
        n = min(max(rng.randint(*cfg["n_background_txns"]), lo), hi)
        out = []
        for _ in range(n):
            debit, credit = rng.choice(BACKGROUND_KINDS)
            out.append({
                "txn_id": self.ids.new("txn"),
                "timestamp": fmt_ts(_rand_ts(rng, self.start, self.end - timedelta(seconds=1)), self.offset),
                "debit_account": debit,
                "credit_account": credit,
                "amount_minor": _price_minor(rng, self.base, 2.0, 4.8),
            })
        return out


def build_company(rng: random.Random, cfg: dict, ids: IdFactory, episode_id: str) -> Company:
    b = _Builder(rng, cfg, ids, episode_id)
    b.setup()
    b.build_documents()
    background = b.build_background(cfg["max_ledger_rows"])
    return Company(
        episode_id=episode_id,
        base_currency=b.base,
        offset=b.offset,
        period_start=b.start,
        period_end=b.end,
        entities=b.entities,
        invoices=b.invoices,
        payments=b.payments,
        background=background,
        rates=b.rates,
        fx_rows=b.fx_rows,
        txn_of=b.txn_of,
        tags=b.tags,
    )
