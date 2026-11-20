"""
Synthetic data generator for the October 2026 month-end reconciliation.

Outputs (written to --out-dir, default: current directory):
  platform_transactions.csv  every transaction the platform processed in Oct 2026
  bank_settlements.csv       what the bank actually settled, extract runs to 2026-11-05
  ground_truth.json          exactly which discrepancies were planted, for testing

Conventions
-----------
* All timestamps are UTC. created_at is ISO-8601 with a trailing "Z";
  settled_on is a plain date (YYYY-MM-DD).
* Amounts are signed: payments are positive, refunds are negative (money
  going back out). Platform amounts carry 3 decimals (a % discount is applied);
  the bank truncates toward zero to 2 decimals.
* Money math is done with decimal.Decimal, never floats, and amounts are
  written to CSV as fixed-width strings so nothing gets re-rounded on the way out.

Reproducibility
---------------
The base dataset is drawn from one RNG stream; each plant draws from its own
independent stream. Switching a plant off therefore never changes the base
rows or any of the other plants.

Usage
-----
  python generator.py                      # all plants on
  python generator.py --disable duplicate  # turn one (or more) off
  python generator.py --disable all        # clean data, books balance exactly
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 20261031

MONTH_START = datetime(2026, 10, 1, tzinfo=timezone.utc)
MONTH_END = datetime(2026, 11, 1, tzinfo=timezone.utc)  # exclusive
# Base volume is created Oct 1-29 so that, with a 1-2 day lag, it always
# settles inside October. The only cross-month rows are the planted ones.
BASE_CREATED_END = datetime(2026, 10, 30, tzinfo=timezone.utc)  # exclusive
BANK_EXTRACT_END = date(2026, 11, 5)  # inclusive

N_BASE = 1490            # base rows (payments + refunds); plants add a handful
REFUND_SHARE = 0.05      # ~5% of base rows are refunds
N_CUSTOMERS = 400
DISCOUNT_PCTS = ["0", "2.5", "5", "7.5", "12.5", "15"]
DISCOUNT_WEIGHTS = [0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
N_LATE = 6               # Oct 30-31 txns that settle in November

PLANT_NAMES = [
    "late_settlement",    # a few Oct 30-31 txns that settle in November
    "truncation_drift",   # bank truncates to 2dp; drift accumulates
    "duplicate",          # one platform trn recorded twice
    "orphan_refund",      # one refund whose original_trn_id doesn't exist
    "unsettled_payment",  # one payment that never reaches the bank
]
# Default switches. Edit here or use --disable on the command line.
PLANTS = {name: True for name in PLANT_NAMES}

Q3 = Decimal("0.001")
Q2 = Decimal("0.01")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@dataclass
class Txn:
    trn_id: str
    created_at: datetime
    customer_id: str
    type: str                     # "payment" | "refund"
    amount: Decimal               # signed, 3dp
    original_trn_id: str | None = None
    settle_lag_days: int = 1
    tags: list[str] = field(default_factory=list)  # internal only, not written


def plant_rng(seed: int, name: str) -> np.random.Generator:
    """Independent stream per plant so toggles don't disturb each other."""
    return np.random.default_rng([seed, PLANT_NAMES.index(name) + 1])


def new_trn_id(rng: np.random.Generator, taken: set[str]) -> str:
    while True:
        tid = "TRN-" + "".join(rng.choice(list("0123456789ABCDEF"), size=8))
        if tid not in taken:
            taken.add(tid)
            return tid


def random_ts(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    span = int((end - start).total_seconds())
    return start + timedelta(seconds=int(rng.integers(0, span)))


def discounted_amount(rng: np.random.Generator) -> Decimal:
    """List price in whole cents, minus a % discount -> 3 decimal places."""
    cents = int(rng.integers(500, 50_000))  # $5.00 .. $499.99
    pct = Decimal(str(rng.choice(DISCOUNT_PCTS, p=DISCOUNT_WEIGHTS)))
    price = Decimal(cents) / 100
    return (price * (1 - pct / 100)).quantize(Q3, rounding=ROUND_HALF_UP)


def truncate_2dp(x: Decimal) -> Decimal:
    """Bank behaviour: drop the 3rd decimal, toward zero (so -1.239 -> -1.23)."""
    return x.quantize(Q2, rounding=ROUND_DOWN)


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Base data
# --------------------------------------------------------------------------
def build_base(seed: int, taken: set[str]) -> list[Txn]:
    rng = np.random.default_rng(seed)
    customers = [f"CUS-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    n_refunds = round(N_BASE * REFUND_SHARE)
    n_payments = N_BASE - n_refunds

    payments: list[Txn] = []
    for _ in range(n_payments):
        payments.append(Txn(
            trn_id=new_trn_id(rng, taken),
            created_at=random_ts(rng, MONTH_START, BASE_CREATED_END),
            customer_id=str(rng.choice(customers)),
            type="payment",
            amount=discounted_amount(rng),
            settle_lag_days=int(rng.integers(1, 3)),
        ))

    # Refunds: each refunds a distinct earlier payment, same customer, created
    # after the payment and still within the base window. Full or partial.
    latest_ok = BASE_CREATED_END - timedelta(hours=2)
    eligible = [p for p in payments if p.created_at < latest_ok]
    picks = rng.choice(len(eligible), size=n_refunds, replace=False)
    refunds: list[Txn] = []
    for idx in picks:
        orig = eligible[int(idx)]
        delay = timedelta(minutes=int(rng.integers(30, 5 * 24 * 60)))
        created = min(orig.created_at + delay, latest_ok)
        if rng.random() < 0.7:
            amt = orig.amount
        else:
            frac = Decimal(str(round(float(rng.uniform(0.1, 0.9)), 2)))
            amt = (orig.amount * frac).quantize(Q3, rounding=ROUND_HALF_UP)
        refunds.append(Txn(
            trn_id=new_trn_id(rng, taken),
            created_at=created,
            customer_id=orig.customer_id,
            type="refund",
            amount=-amt,
            original_trn_id=orig.trn_id,
            settle_lag_days=int(rng.integers(1, 3)),
        ))
        orig.tags.append("refunded")

    return payments + refunds


# --------------------------------------------------------------------------
# Plants
# --------------------------------------------------------------------------
def plant_late_settlement(seed, txns, taken, customers):
    rng = plant_rng(seed, "late_settlement")
    start = datetime(2026, 10, 30, tzinfo=timezone.utc)
    added = []
    for _ in range(N_LATE):
        created = random_ts(rng, start, MONTH_END)
        # 1-2 day lag, but never landing on Oct 31: an Oct 30 txn needs 2 days.
        min_lag = (MONTH_END.date() - created.date()).days
        t = Txn(
            trn_id=new_trn_id(rng, taken),
            created_at=created,
            customer_id=str(rng.choice(customers)),
            type="payment",
            amount=discounted_amount(rng),
            settle_lag_days=max(int(rng.integers(1, 3)), min_lag),
            tags=["late_settlement"],
        )
        txns.append(t)
        added.append(t)
    return added


def _pick_clean_payment(rng, txns) -> Txn:
    """A base payment that hasn't been refunded or used by another plant."""
    # Draw a permutation over a pool that doesn't depend on other plants, then
    # skip anything already tagged. Keeps each plant's pick stable across toggles.
    pool = sorted((t for t in txns if t.type == "payment"
                   and not {"late_settlement", "duplicate_copy"} & set(t.tags)),
                  key=lambda t: t.trn_id)
    for idx in rng.permutation(len(pool)):
        if not pool[int(idx)].tags:
            return pool[int(idx)]
    raise RuntimeError("no clean payment left to plant on")


def plant_duplicate(seed, txns):
    rng = plant_rng(seed, "duplicate")
    victim = _pick_clean_payment(rng, txns)
    victim.tags.append("duplicate")
    dup = Txn(**{**victim.__dict__, "tags": ["duplicate_copy"]})
    txns.append(dup)
    return victim


def plant_orphan_refund(seed, txns, taken, customers):
    rng = plant_rng(seed, "orphan_refund")
    missing_orig = new_trn_id(rng, taken)  # reserved id, never emitted
    t = Txn(
        trn_id=new_trn_id(rng, taken),
        created_at=random_ts(rng, MONTH_START + timedelta(days=3), BASE_CREATED_END),
        customer_id=str(rng.choice(customers)),
        type="refund",
        amount=-discounted_amount(rng),
        original_trn_id=missing_orig,
        settle_lag_days=int(rng.integers(1, 3)),
        tags=["orphan_refund"],
    )
    txns.append(t)
    return t


def plant_unsettled_payment(seed, txns):
    rng = plant_rng(seed, "unsettled_payment")
    victim = _pick_clean_payment(rng, txns)
    victim.tags.append("unsettled_payment")
    return victim


# --------------------------------------------------------------------------
# Bank side
# --------------------------------------------------------------------------
def build_bank(txns: list[Txn], truncate: bool) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for t in txns:
        if "unsettled_payment" in t.tags:
            continue
        if t.trn_id in seen:  # duplicate platform record: bank only moved money once
            continue
        seen.add(t.trn_id)
        settled = (t.created_at + timedelta(days=t.settle_lag_days)).date()
        if settled > BANK_EXTRACT_END:
            continue
        amt = truncate_2dp(t.amount) if truncate else t.amount
        rows.append({"trn_ref": t.trn_id, "settled_on": settled, "amount": amt})

    rows.sort(key=lambda r: (r["settled_on"], r["trn_ref"]))
    out = []
    for i, r in enumerate(rows, start=1):
        out.append({
            "settlement_id": f"STL-{i:06d}",
            "batch_id": "BATCH-" + r["settled_on"].strftime("%Y%m%d"),
            "trn_ref": r["trn_ref"],
            "settled_on": r["settled_on"].isoformat(),
            # 2dp when truncating; keep full 3dp otherwise so the clean run matches exactly
            "amount": f"{r['amount']:.2f}" if truncate else f"{r['amount']:.3f}",
            "_amount": r["amount"],
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_dataset(seed: int, plants: dict[str, bool]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build everything in memory: (platform_df, bank_df, ground_truth dict).

    Columns and string formatting are exactly what generate() writes to CSV.
    """
    taken: set[str] = set()
    customers = [f"CUS-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    txns = build_base(seed, taken)

    # Each plant has its own RNG stream, so its ids/picks are the same whether or
    # not the other plants are switched on.
    late = plant_late_settlement(seed, txns, taken, customers) if plants["late_settlement"] else []
    dup = plant_duplicate(seed, txns) if plants["duplicate"] else None
    orphan = plant_orphan_refund(seed, txns, taken, customers) if plants["orphan_refund"] else None
    unsettled = plant_unsettled_payment(seed, txns) if plants["unsettled_payment"] else None

    txns.sort(key=lambda t: (t.created_at, t.trn_id))
    platform = pd.DataFrame([{
        "trn_id": t.trn_id,
        "created_at": fmt_ts(t.created_at),
        "customer_id": t.customer_id,
        "type": t.type,
        "amount": f"{t.amount:.3f}",
        "original_trn_id": t.original_trn_id or "",
    } for t in txns])

    bank = build_bank(txns, truncate=plants["truncation_drift"])

    # ---- ground truth, computed from the data we just built ------------------
    by_id = {t.trn_id: t for t in txns}
    platform_total = sum((t.amount for t in txns), Decimal(0))
    bank_total = sum(bank["_amount"], Decimal(0))
    oct_mask = bank["settled_on"] < "2026-11-01"
    bank_total_oct = sum(bank.loc[oct_mask, "_amount"], Decimal(0))

    drift_rows = [(ref, by_id[ref].amount - amt) for ref, amt in zip(bank["trn_ref"], bank["_amount"])]
    drift_total = sum((d for _, d in drift_rows), Decimal(0))
    drift_nonzero = [d for _, d in drift_rows if d != 0]

    late_info = [{
        "trn_id": t.trn_id, "created_at": fmt_ts(t.created_at), "amount": f"{t.amount:.3f}",
        "settled_on": (t.created_at + timedelta(days=t.settle_lag_days)).date().isoformat(),
    } for t in late]
    late_total = sum((t.amount for t in late), Decimal(0))
    late_bank_total = sum(bank.loc[bank["trn_ref"].isin([t.trn_id for t in late]), "_amount"], Decimal(0))

    # Self-check: every cent of the gap is explained by a plant.
    explained = drift_total + (dup.amount if dup else 0) + (unsettled.amount if unsettled else 0)
    assert platform_total - bank_total == explained, "gap not fully explained by plants"
    assert platform_total - bank_total_oct == explained + late_bank_total, "Oct cut-off gap mismatch"

    d3 = lambda x: f"{x:.3f}"  # noqa: E731
    truth = {
        "seed": seed,
        "month": "2026-10",
        "timezone": "UTC",
        "bank_extract_through": BANK_EXTRACT_END.isoformat(),
        "plants_enabled": plants,
        "row_counts": {"platform": len(platform), "bank": len(bank)},
        "amount_sign_convention": "payments positive, refunds negative (both files)",
        "plants": {
            "late_settlement": {
                "enabled": plants["late_settlement"],
                "description": "Created Oct 30-31, settled by the bank in November.",
                "count": len(late),
                "platform_amount_total": d3(late_total),
                "bank_amount_total": d3(late_bank_total),
                "transactions": late_info,
            },
            "truncation_drift": {
                "enabled": plants["truncation_drift"],
                "description": "Bank truncates each amount toward zero to 2dp. "
                               "Drift = platform_amount - bank_amount, summed over settled rows.",
                "rows_affected": len(drift_nonzero),
                "total_drift": d3(drift_total),
                "max_row_drift": d3(max((abs(d) for d in drift_nonzero), default=Decimal(0))),
            },
            "duplicate": {
                "enabled": plants["duplicate"],
                "description": "Same trn_id appears twice in platform; bank settled it once.",
                "trn_id": dup.trn_id if dup else None,
                "amount": d3(dup.amount) if dup else None,
            },
            "orphan_refund": {
                "enabled": plants["orphan_refund"],
                "description": "Refund whose original_trn_id does not exist in platform data. "
                               "The bank did settle it.",
                "trn_id": orphan.trn_id if orphan else None,
                "original_trn_id": orphan.original_trn_id if orphan else None,
                "amount": d3(orphan.amount) if orphan else None,
            },
            "unsettled_payment": {
                "enabled": plants["unsettled_payment"],
                "description": "Payment present in platform with no bank settlement at all.",
                "trn_id": unsettled.trn_id if unsettled else None,
                "amount": d3(unsettled.amount) if unsettled else None,
            },
        },
        "totals": {
            "platform_total": d3(platform_total),
            "bank_total_full_extract": d3(bank_total),
            "bank_total_settled_in_october": d3(bank_total_oct),
            "gap_vs_full_extract": d3(platform_total - bank_total),
            "gap_vs_october_settlements": d3(platform_total - bank_total_oct),
            "gap_breakdown_full_extract": {
                "truncation_drift": d3(drift_total),
                "duplicate": d3(dup.amount if dup else Decimal(0)),
                "unsettled_payment": d3(unsettled.amount if unsettled else Decimal(0)),
            },
            "note": "October-cut-off gap = full-extract gap + bank amount of late settlements. "
                    "The orphan refund settles normally, so it is a data-integrity issue, not a money gap.",
        },
    }

    return platform, bank.drop(columns="_amount"), truth


def generate(seed: int, plants: dict[str, bool], out_dir: Path) -> dict:
    """Build the dataset and write the three files to out_dir."""
    platform, bank, truth = build_dataset(seed, plants)
    out_dir.mkdir(parents=True, exist_ok=True)
    platform.to_csv(out_dir / "platform_transactions.csv", index=False)
    bank.to_csv(out_dir / "bank_settlements.csv", index=False)
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--disable", nargs="*", default=[], choices=PLANT_NAMES + ["all"],
                    help="plants to switch off")
    args = ap.parse_args()

    plants = dict(PLANTS)
    for name in args.disable:
        for p in (PLANT_NAMES if name == "all" else [name]):
            plants[p] = False

    truth = generate(args.seed, plants, args.out_dir)
    t = truth["totals"]
    print(f"platform rows: {truth['row_counts']['platform']}, bank rows: {truth['row_counts']['bank']}")
    print(f"plants on: {[k for k, v in plants.items() if v]}")
    print(f"platform total {t['platform_total']} | bank (full) {t['bank_total_full_extract']} "
          f"| gap {t['gap_vs_full_extract']} | gap vs Oct-only {t['gap_vs_october_settlements']}")


if __name__ == "__main__":
    main()
