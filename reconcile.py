"""
Month-end reconciliation: platform transactions vs bank settlements.

Pure pandas. Takes the two CSVs, finds every gap on its own (no ground truth),
and returns:

  flags       one row per issue, each with a plain-English reason and its
              effect on the October bridge (bridge_impact)
  exceptions  data-integrity issues that don't move the totals (orphan refunds)
  bridge      platform October total -> bank October total, ending in an
              "Unexplained" line that should be 0.000

Money is handled as integer thousandths (1.234 -> 1234) internally so sums
are exact; amounts in the outputs are decimal.Decimal.

Library use:
    from reconcile import load_platform, load_bank, reconcile
    result = reconcile(load_platform("platform_transactions.csv"),
                       load_bank("bank_settlements.csv"), month="2026-10")
    result.bridge, result.flags, result.exceptions, result.unexplained

CLI:
    python reconcile.py [--platform P.csv] [--bank B.csv] [--month 2026-10] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

PLATFORM_COLS = ["trn_id", "created_at", "customer_id", "type", "amount", "original_trn_id"]
BANK_COLS = ["settlement_id", "batch_id", "trn_ref", "settled_on", "amount"]

# Categories that move money (appear in the bridge) vs. integrity-only ones.
MONEY_CATEGORIES = [
    "duplicate",
    "never_settled",
    "settled_after_month_end",
    "truncation_drift",
    "amount_mismatch",
    "bank_only",
]
EXCEPTION_CATEGORIES = ["orphan_refund"]

FLAG_COLUMNS = ["category", "trn_id", "settlement_id", "created_at", "settled_on",
                "platform_amount", "bank_amount", "bridge_impact", "reason"]


# --------------------------------------------------------------------------
# Loading / money helpers
# --------------------------------------------------------------------------
def _to_milli(value: object, where: str) -> int:
    """'12.345' -> 12345. Exact; rejects blanks and >3 decimals."""
    try:
        d = Decimal(str(value).strip())
    except InvalidOperation:
        raise ValueError(f"{where}: amount {value!r} is not a number") from None
    m = d * 1000
    if m != m.to_integral_value():
        raise ValueError(f"{where}: amount {value!r} has more than 3 decimals")
    return int(m)


def _money(milli: int) -> Decimal:
    return Decimal(int(milli)).scaleb(-3)


def _fmt(milli: int) -> str:
    return f"{_money(milli):.3f}"


def _fmt_bank(milli: int) -> str:
    return f"{_money(milli):.2f}" if milli % 10 == 0 else f"{_money(milli):.3f}"


def _day(d: date) -> str:
    return f"{d:%b} {d.day}"  # "Nov 2"


def _trunc_2dp(milli: int) -> int:
    """Truncate toward zero to 2 decimals: 1239 -> 1230, -1239 -> -1230."""
    return (abs(milli) // 10 * 10) * (1 if milli >= 0 else -1)


def _check_cols(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _load_csv(path: str | Path, name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{name} file not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{name} file is empty (no header row): {path}") from None
    except pd.errors.ParserError as e:
        raise ValueError(f"{name} file is not valid CSV: {path} ({e})") from None


def load_platform(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, "platform")


def load_bank(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, "bank")


def _prep_platform(df: pd.DataFrame) -> pd.DataFrame:
    _check_cols(df, PLATFORM_COLS, "platform data")
    p = df[PLATFORM_COLS].copy().reset_index(drop=True)
    p["original_trn_id"] = p["original_trn_id"].fillna("").astype(str).str.strip()
    p["type"] = p["type"].str.strip().str.lower()
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    p["amt"] = [_to_milli(v, f"platform row {i + 2} ({t})")
                for i, (v, t) in enumerate(zip(p["amount"], p["trn_id"]))]
    return p


def _prep_bank(df: pd.DataFrame) -> pd.DataFrame:
    _check_cols(df, BANK_COLS, "bank data")
    b = df[BANK_COLS].copy().reset_index(drop=True)
    b["settled_on"] = pd.to_datetime(b["settled_on"]).dt.date
    b["amt"] = [_to_milli(v, f"bank row {i + 2} ({s})")
                for i, (v, s) in enumerate(zip(b["amount"], b["settlement_id"]))]
    return b


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class ReconResult:
    month: str
    flags: pd.DataFrame
    exceptions: pd.DataFrame
    bridge: pd.DataFrame
    summary: dict

    @property
    def unexplained(self) -> Decimal:
        return self.bridge.loc[self.bridge["line"] == "unexplained", "amount"].iloc[0]


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def reconcile(platform: pd.DataFrame, bank: pd.DataFrame, month: str = "2026-10") -> ReconResult:
    p_all = _prep_platform(platform)
    b = _prep_bank(bank)

    period = pd.Period(month, freq="M")
    m_start, m_end = period.start_time.date(), period.end_time.date()
    bank_through = max(b["settled_on"]) if len(b) else None

    in_month = (p_all["created_at"].dt.date >= m_start) & (p_all["created_at"].dt.date <= m_end)
    p = p_all[in_month].copy()
    b_month = b[(b["settled_on"] >= m_start) & (b["settled_on"] <= m_end)]

    flags: list[dict] = []

    def flag(category, reason, impact, trn_id="", settlement_id="", created_at=None,
             settled_on=None, platform_amt=None, bank_amt=None):
        flags.append({
            "category": category,
            "trn_id": trn_id,
            "settlement_id": settlement_id,
            "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ") if created_at is not None else "",
            "settled_on": settled_on.isoformat() if settled_on is not None else "",
            "platform_amount": _money(platform_amt) if platform_amt is not None else None,
            "bank_amount": _money(bank_amt) if bank_amt is not None else None,
            "bridge_impact": _money(impact),
            "reason": reason,
        })

    # ---- 1. duplicates in platform -----------------------------------------
    p["copy_no"] = p.groupby("trn_id").cumcount()
    bank_count = b.groupby("trn_ref").size()
    for trn_id, grp in p[p["trn_id"].duplicated(keep=False)].groupby("trn_id", sort=False):
        n = len(grp)
        identical = len(grp.drop(columns="copy_no").astype(str).drop_duplicates()) == 1
        settled_n = int(bank_count.get(trn_id, 0))
        for _, row in grp[grp["copy_no"] > 0].iterrows():
            flag("duplicate",
                 f"trn_id recorded {n} times in platform "
                 f"({'identical copies' if identical else 'copies differ'}); "
                 f"bank settled it {settled_n} time{'s' if settled_n != 1 else ''}. "
                 f"This extra copy overstates platform by {_fmt(row.amt)}",
                 -row.amt, trn_id=trn_id, created_at=row.created_at, platform_amt=row.amt)

    uniq = p[p["copy_no"] == 0]

    # ---- 2. match each unique platform trn to the bank ---------------------
    bank_by_ref = {ref: g for ref, g in b.groupby("trn_ref", sort=False)}
    for row in uniq.itertuples(index=False):
        g = bank_by_ref.get(row.trn_id)
        created = row.created_at.date()

        if g is None:
            through = f" (bank data runs through {_day(bank_through)})" if bank_through else ""
            flag("never_settled",
                 f"created {_day(created)}, no matching settlement anywhere in the bank extract{through}",
                 -row.amt, trn_id=row.trn_id, created_at=row.created_at, platform_amt=row.amt)
            continue

        inside = g[(g["settled_on"] >= m_start) & (g["settled_on"] <= m_end)]
        after = g[g["settled_on"] > m_end]

        if inside.empty and not after.empty:
            s = after.sort_values("settled_on").iloc[0]
            flag("settled_after_month_end",
                 f"created {_day(created)}, settled {_day(s.settled_on)} "
                 f"(batch {s.batch_id}), after month end {_day(m_end)}",
                 -row.amt, trn_id=row.trn_id, settlement_id=s.settlement_id,
                 created_at=row.created_at, settled_on=s.settled_on,
                 platform_amt=row.amt, bank_amt=int(after["amt"].sum()))
            continue

        if inside.empty:  # only settled before the month started: can't be right
            s = g.iloc[0]
            flag("amount_mismatch",
                 f"created {_day(created)} but settled {_day(s.settled_on)}, before the month started",
                 -row.amt, trn_id=row.trn_id, settlement_id=s.settlement_id,
                 created_at=row.created_at, settled_on=s.settled_on,
                 platform_amt=row.amt, bank_amt=int(g["amt"].sum()))
            continue

        bank_amt = int(inside["amt"].sum())
        diff = bank_amt - row.amt
        s = inside.iloc[0]
        common = dict(trn_id=row.trn_id, settlement_id=";".join(inside["settlement_id"]),
                      created_at=row.created_at, settled_on=s.settled_on,
                      platform_amt=row.amt, bank_amt=bank_amt)

        if len(inside) > 1 or not after.empty:
            flag("amount_mismatch",
                 f"settled {len(g)} times in the bank "
                 f"({', '.join(_day(d) for d in sorted(g['settled_on']))}); "
                 f"bank total {_fmt(int(g['amt'].sum()))} vs platform {_fmt(row.amt)}",
                 diff, **common)
        elif diff == 0:
            continue
        elif bank_amt == _trunc_2dp(row.amt):
            flag("truncation_drift",
                 f"bank truncated {_fmt(row.amt)} to {_fmt_bank(bank_amt)} "
                 f"(drift {_fmt(-diff)})",
                 diff, **common)
        else:
            flag("amount_mismatch",
                 f"bank settled {_fmt_bank(bank_amt)} but platform says {_fmt(row.amt)} "
                 f"(difference {_fmt(diff)}, more than truncation explains)",
                 diff, **common)

    # ---- 3. bank settlements in the month with no platform record ----------
    known = set(uniq["trn_id"])
    for s in b_month[~b_month["trn_ref"].isin(known)].itertuples(index=False):
        exists_elsewhere = s.trn_ref in set(p_all["trn_id"])
        why = ("trn exists in platform but was created outside "
               f"{period.strftime('%B %Y')}" if exists_elsewhere
               else "no platform transaction with this trn_ref")
        flag("bank_only", f"settled {_day(s.settled_on)} for {_fmt_bank(s.amt)}; {why}",
             s.amt, trn_id=s.trn_ref, settlement_id=s.settlement_id,
             settled_on=s.settled_on, bank_amt=s.amt)

    # ---- 4. orphan refunds (integrity only, no money effect) ---------------
    all_ids = set(p_all["trn_id"])
    for row in uniq[uniq["type"] == "refund"].itertuples(index=False):
        if not row.original_trn_id:
            why = "refund has no original_trn_id"
        elif row.original_trn_id not in all_ids:
            why = f"refund points to original {row.original_trn_id}, which doesn't exist in platform data"
        else:
            continue
        g = bank_by_ref.get(row.trn_id)
        settled = (f"; bank still settled it ({_fmt_bank(int(g['amt'].sum()))} "
                   f"on {_day(g['settled_on'].min())})" if g is not None else "")
        flag("orphan_refund", why + settled, 0, trn_id=row.trn_id,
             created_at=row.created_at, platform_amt=row.amt,
             settled_on=g["settled_on"].min() if g is not None else None,
             bank_amt=int(g["amt"].sum()) if g is not None else None)

    flags_df = pd.DataFrame(flags, columns=FLAG_COLUMNS)
    order = {c: i for i, c in enumerate(MONEY_CATEGORIES + EXCEPTION_CATEGORIES)}
    flags_df = (flags_df.assign(_o=flags_df["category"].map(order))
                .sort_values(["_o", "settled_on", "trn_id"], kind="stable")
                .drop(columns="_o").reset_index(drop=True))
    exceptions_df = flags_df[flags_df["category"].isin(EXCEPTION_CATEGORIES)].reset_index(drop=True)

    # ---- 5. bridge ----------------------------------------------------------
    platform_total = int(p["amt"].sum())
    bank_total = int(b_month["amt"].sum())

    def cat(c):
        sel = flags_df[flags_df["category"] == c]
        return len(sel), sum(sel["bridge_impact"], Decimal("0.000"))

    mon = period.strftime("%B")
    lines = [("platform_total", f"Platform {mon} total (as recorded)", len(p), _money(platform_total))]
    labels = {
        "duplicate": "Less: duplicate platform records",
        "never_settled": "Less: never settled by the bank",
        "settled_after_month_end": f"Less: settled after {_day(m_end)} (timing)",
        "truncation_drift": "Less: bank truncation to 2 decimals",
        "amount_mismatch": "Adjust: other amount mismatches",
        "bank_only": "Add: bank settlements with no platform record",
    }
    expected = _money(platform_total)
    for c in MONEY_CATEGORIES:
        n, amt = cat(c)
        lines.append((c, labels[c], n, amt))
        expected += amt
    lines.append(("expected_bank_total", f"Expected bank {mon} total", None, expected))
    lines.append(("bank_total", f"Actual bank {mon} total (settled {_day(m_start)}-{_day(m_end)})",
                  len(b_month), _money(bank_total)))
    lines.append(("unexplained", "Unexplained", None, _money(bank_total) - expected))
    bridge_df = pd.DataFrame(lines, columns=["line", "description", "count", "amount"])
    bridge_df["count"] = bridge_df["count"].astype("Int64")

    summary = {
        "month": month,
        "platform_rows_in_month": len(p),
        "platform_rows_outside_month": int((~in_month).sum()),
        "bank_rows_in_month": len(b_month),
        "bank_rows_total": len(b),
        "bank_data_through": bank_through.isoformat() if bank_through else None,
        "platform_total": _money(platform_total),
        "bank_total": _money(bank_total),
        "gap": _money(platform_total - bank_total),
        "flag_counts": flags_df["category"].value_counts().reindex(
            MONEY_CATEGORIES + EXCEPTION_CATEGORIES, fill_value=0).to_dict(),
    }
    return ReconResult(month, flags_df, exceptions_df, bridge_df, summary)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile platform transactions against bank settlements.")
    ap.add_argument("--platform", type=Path, default=Path("platform_transactions.csv"))
    ap.add_argument("--bank", type=Path, default=Path("bank_settlements.csv"))
    ap.add_argument("--month", default="2026-10", help="YYYY-MM")
    ap.add_argument("--out-dir", type=Path, help="write flags.csv, exceptions.csv, bridge.csv here")
    args = ap.parse_args()

    try:
        r = reconcile(load_platform(args.platform), load_bank(args.bank), args.month)
    except ValueError as e:
        hint = "  (no data yet? run: python generator.py)" if "not found" in str(e) else ""
        raise SystemExit(f"error: {e}{hint}") from None

    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(r.bridge[["description", "count", "amount"]].to_string(index=False))
        print("\nFlag counts:", r.summary["flag_counts"])
        for c in MONEY_CATEGORIES + EXCEPTION_CATEGORIES:
            sel = r.flags[r.flags["category"] == c]
            if len(sel):
                print(f"\n[{c}] {len(sel)} row(s), e.g.:")
                print(sel[["trn_id", "bridge_impact", "reason"]].head(3).to_string(index=False))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        r.flags.to_csv(args.out_dir / "flags.csv", index=False)
        r.exceptions.to_csv(args.out_dir / "exceptions.csv", index=False)
        r.bridge.to_csv(args.out_dir / "bridge.csv", index=False)
        print(f"\nwrote flags.csv, exceptions.csv, bridge.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
