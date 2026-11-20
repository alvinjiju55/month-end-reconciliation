"""
Streamlit front end for the month-end reconciliation.

All logic lives in reconcile.py (matching, flags, bridge) and generator.py
(sample data). This file only collects inputs and draws results.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from generator import PLANT_NAMES, SEED, build_dataset
from reconcile import EXCEPTION_CATEGORIES, MONEY_CATEGORIES, reconcile

st.set_page_config(page_title="Month-end reconciliation", page_icon="🧾", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  .muted {color: #6b7280; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

PLANT_LABELS = {
    "late_settlement": ("Late settlements", "A few Oct 30-31 transactions that the bank settles in November."),
    "truncation_drift": ("Truncation drift", "Bank truncates every amount to 2 decimals; tiny per row, adds up."),
    "duplicate": ("Duplicate record", "One platform transaction recorded twice."),
    "orphan_refund": ("Orphan refund", "One refund whose original transaction doesn't exist."),
    "unsettled_payment": ("Never-settled payment", "One payment that never reaches the bank."),
}

# category -> (tab label, one-line explanation)
CATEGORY_INFO = {
    "duplicate": ("Duplicates", "Same trn_id recorded more than once in the platform. The bank only moved the money once, "
                                "so each extra copy overstates the platform total."),
    "never_settled": ("Never settled", "Platform transactions with no matching settlement anywhere in the bank extract."),
    "settled_after_month_end": ("Settled after month end", "October transactions the bank settled in November. "
                                "A timing difference: the money arrived, just not in October."),
    "truncation_drift": ("Truncation", "The bank keeps 2 decimals and drops the third (toward zero). "
                         "Fractions of a cent per row that add up over the month."),
    "amount_mismatch": ("Other mismatches", "Bank amount differs by more than truncation explains, "
                        "or the bank settled the same transaction more than once."),
    "bank_only": ("Bank only", "Money the bank settled in the month with no matching platform transaction."),
    "orphan_refund": ("Exceptions", "Data-integrity issues that don't move the totals. Refunds whose original "
                      "transaction can't be found: the money still left, but nothing explains why."),
}

WATERFALL_LABELS = {
    "duplicate": "Duplicates",
    "never_settled": "Never settled",
    "settled_after_month_end": "Settled after month end",
    "truncation_drift": "Truncation",
    "amount_mismatch": "Other mismatches",
    "bank_only": "Bank only",
}

DEFAULT_MONTH = "2026-10"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating sample data...")
def generated_data(seed: int, plants: tuple[tuple[str, bool], ...]):
    return build_dataset(seed, dict(plants))


@st.cache_data(show_spinner="Reconciling...")
def run_reconcile(platform: pd.DataFrame, bank: pd.DataFrame, month: str):
    return reconcile(platform, bank, month)


def read_upload(file, name: str) -> pd.DataFrame:
    raw = file.getvalue()
    if not raw.strip():
        raise ValueError(f"{name} file '{file.name}' is empty")
    try:
        return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{name} file '{file.name}' has no header row") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ValueError(f"{name} file '{file.name}' is not a readable CSV ({e})") from None


def money(x: Decimal | float) -> str:
    return f"{Decimal(x):,.3f}"


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Decimals -> floats for display only (all math already happened in Decimal)."""
    out = df.copy()
    for c in ["platform_amount", "bank_amount", "bridge_impact"]:
        if c in out:
            out[c] = out[c].map(lambda v: None if v is None or pd.isna(v) else float(v))
    return out


AMOUNT_COLS = {
    "platform_amount": st.column_config.NumberColumn("Platform amount", format="%.3f"),
    "bank_amount": st.column_config.NumberColumn("Bank amount", format="%.3f"),
    "bridge_impact": st.column_config.NumberColumn("Bridge impact", format="%.3f"),
    "reason": st.column_config.TextColumn("Reason", width="large"),
    "trn_id": "Transaction",
    "settlement_id": "Settlement",
    "created_at": "Created (UTC)",
    "settled_on": "Settled on",
}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Generate sample data", "Upload my own CSVs"],
                      label_visibility="collapsed")

    truth = None
    if source == "Generate sample data":
        with st.form("gen"):
            seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=SEED, step=1,
                                   help="Same seed + same toggles = identical data.")
            st.markdown("**Planted problems**")
            plants = {p: st.toggle(PLANT_LABELS[p][0], value=True, help=PLANT_LABELS[p][1])
                      for p in PLANT_NAMES}
            st.form_submit_button("Regenerate", type="primary", width="stretch")
        month = DEFAULT_MONTH
        platform_df, bank_df, truth = generated_data(int(seed), tuple(plants.items()))
        st.caption(f"{len(platform_df):,} platform rows, {len(bank_df):,} bank rows. "
                   "Built in memory, nothing read from disk.")
        c1, c2 = st.columns(2)
        c1.download_button("Platform CSV", platform_df.to_csv(index=False),
                           "platform_transactions.csv", "text/csv", width="stretch")
        c2.download_button("Bank CSV", bank_df.to_csv(index=False),
                           "bank_settlements.csv", "text/csv", width="stretch")
    else:
        st.caption("Platform columns: trn_id, created_at, customer_id, type, amount, original_trn_id.  \n"
                   "Bank columns: settlement_id, batch_id, trn_ref, settled_on, amount.")
        up_platform = st.file_uploader("Platform transactions", type="csv")
        up_bank = st.file_uploader("Bank settlements", type="csv")
        month = st.text_input("Month to reconcile (YYYY-MM)", DEFAULT_MONTH)
        if not (up_platform and up_bank):
            st.info("Upload both files to reconcile.")
            st.title("Month-end reconciliation")
            st.write("Upload a platform file and a bank file in the sidebar, "
                     "or switch back to generated sample data.")
            st.stop()
        try:
            platform_df = read_upload(up_platform, "Platform")
            bank_df = read_upload(up_bank, "Bank")
            pd.Period(month, freq="M")
        except ValueError as e:
            st.error(str(e))
            st.stop()

try:
    r = run_reconcile(platform_df, bank_df, month)
except ValueError as e:
    st.error(f"Couldn't reconcile: {e}")
    st.stop()

period = pd.Period(month, freq="M")
s = r.summary
bridge = r.bridge.set_index("line")

# --------------------------------------------------------------------------
# Header + headline numbers
# --------------------------------------------------------------------------
st.title("Month-end reconciliation")
st.markdown(f"<span class='muted'>{period.strftime('%B %Y')} · platform transactions created in the month "
            f"vs bank settlements dated in the month · bank data through {s['bank_data_through'] or 'n/a'}</span>",
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1, st.container(border=True):
    st.metric("Platform total", money(s["platform_total"]), help=f"{s['platform_rows_in_month']:,} rows")
with m2, st.container(border=True):
    st.metric("Bank total", money(s["bank_total"]), help=f"{s['bank_rows_in_month']:,} settlements in the month")
with m3, st.container(border=True):
    st.metric("Difference", money(s["gap"]), help="Platform total minus bank total")
with m4, st.container(border=True):
    unexplained = r.unexplained
    st.metric("Unexplained", money(unexplained),
              delta="fully explained" if unexplained == 0 else "needs investigation",
              delta_color="normal" if unexplained == 0 else "inverse")

# --------------------------------------------------------------------------
# Waterfall
# --------------------------------------------------------------------------
st.subheader("Where the money goes")
steps = [(WATERFALL_LABELS[c], bridge.loc[c, "amount"]) for c in MONEY_CATEGORIES if bridge.loc[c, "amount"] != 0]
x = ["Platform total"] + [n for n, _ in steps] + ["Unexplained", "Bank total"]
y = [float(s["platform_total"])] + [float(a) for _, a in steps] + [float(unexplained), 0.0]
measure = ["absolute"] + ["relative"] * len(steps) + ["relative", "total"]
text = [money(s["platform_total"])] + [f"{a:+,.3f}" for _, a in steps] + [f"{unexplained:+,.3f}", money(s["bank_total"])]

running, levels = 0.0, []
for v, m in zip(y, measure):
    running = v if m == "absolute" else (running if m == "total" else running + v)
    levels.append(running)
lo, hi = min(levels), max(levels)
pad = max((hi - lo) * 0.25, abs(hi) * 0.0005, 1.0)

fig = go.Figure(go.Waterfall(
    x=x, y=y, measure=measure, text=text, textposition="outside",
    connector={"line": {"color": "rgba(120,120,120,0.4)", "dash": "dot"}},
    decreasing={"marker": {"color": "#e4572e"}},
    increasing={"marker": {"color": "#29a36a"}},
    totals={"marker": {"color": "#3b6fb6"}},
    hovertemplate="%{x}<br>%{text}<extra></extra>",
))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                  yaxis=dict(range=[lo - pad, hi + pad], tickformat=",.0f", title=None))
st.plotly_chart(fig, width="stretch")
st.caption("The y-axis is zoomed in on the gap, so bars are not drawn from zero. "
           "Lines with no effect are left out of the chart; the full bridge is below.")

with st.expander("Bridge table", expanded=False):
    bt = r.bridge.copy()
    bt["amount"] = bt["amount"].map(lambda a: f"{a:,.3f}")
    st.dataframe(bt[["description", "count", "amount"]], hide_index=True, width="stretch",
                 column_config={"description": "Line", "count": "Rows", "amount": "Amount"})
    st.download_button("Download bridge CSV", r.bridge.to_csv(index=False),
                       f"bridge_{month}.csv", "text/csv")

# --------------------------------------------------------------------------
# Tabs per gap type
# --------------------------------------------------------------------------
st.subheader("Flagged rows")
counts = s["flag_counts"]
tab_cats = ["duplicate", "never_settled", "settled_after_month_end", "truncation_drift", "orphan_refund"]
# Only show the generic categories if something landed in them.
tab_cats[4:4] = [c for c in ("amount_mismatch", "bank_only") if counts.get(c)]
tabs = st.tabs([f"{CATEGORY_INFO[c][0]} ({counts.get(c, 0):,})" for c in tab_cats])

for tab, cat in zip(tabs, tab_cats):
    with tab:
        st.markdown(f"<span class='muted'>{CATEGORY_INFO[cat][1]}</span>", unsafe_allow_html=True)
        rows = r.flags[r.flags["category"] == cat]
        if rows.empty:
            st.success("Nothing flagged.")
            continue
        if cat != "orphan_refund":
            total = sum(rows["bridge_impact"], Decimal("0.000"))
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{len(rows):,}")
            c2.metric("Effect on bridge", f"{total:+,.3f}")
        cols = [c for c in ["trn_id", "settlement_id", "created_at", "settled_on", "platform_amount",
                            "bank_amount", "bridge_impact", "reason"] if rows[c].astype(str).str.len().gt(0).any()
                and not rows[c].isna().all()]
        st.dataframe(to_display(rows[cols]), hide_index=True, width="stretch",
                     column_config=AMOUNT_COLS, height=min(38 + 35 * len(rows), 420))

st.download_button(
    f"Download all {len(r.flags):,} flagged rows (CSV)",
    r.flags.to_csv(index=False), f"flagged_rows_{month}.csv", "text/csv", type="primary",
)

# --------------------------------------------------------------------------
# Answer-key check (generated data only)
# --------------------------------------------------------------------------
if truth is not None:
    with st.expander("Check against the planted answer key"):
        st.caption("The reconciler never sees this. It's here so you can confirm it found what was planted.")
        P = truth["plants"]
        found = lambda c: sorted(r.flags.loc[r.flags["category"] == c, "trn_id"])  # noqa: E731
        checks = [
            ("Late settlements", P["late_settlement"]["enabled"],
             sorted(t["trn_id"] for t in P["late_settlement"]["transactions"]), found("settled_after_month_end")),
            ("Duplicate", P["duplicate"]["enabled"],
             [P["duplicate"]["trn_id"]] if P["duplicate"]["enabled"] else [], found("duplicate")),
            ("Orphan refund", P["orphan_refund"]["enabled"],
             [P["orphan_refund"]["trn_id"]] if P["orphan_refund"]["enabled"] else [], found("orphan_refund")),
            ("Never settled", P["unsettled_payment"]["enabled"],
             [P["unsettled_payment"]["trn_id"]] if P["unsettled_payment"]["enabled"] else [], found("never_settled")),
        ]
        rows = [{"Plant": n, "Planted": on, "Expected": ", ".join(e) or "-", "Found": ", ".join(f) or "-",
                 "Match": "✅" if e == f else "❌"} for n, on, e, f in checks]
        drift_on = P["truncation_drift"]["enabled"]
        rows.append({"Plant": "Truncation drift", "Planted": drift_on,
                     "Expected": f"{P['truncation_drift']['rows_affected']} rows over full extract",
                     "Found": f"{counts['truncation_drift']} rows settled in October",
                     "Match": "✅" if (counts["truncation_drift"] > 0) == drift_on else "❌"})
        extra = counts["amount_mismatch"] + counts["bank_only"]
        rows.append({"Plant": "Nothing unplanted", "Planted": True, "Expected": "0", "Found": str(extra),
                     "Match": "✅" if extra == 0 else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------
st.subheader("Assumptions")
st.markdown(f"""
- **Time zone:** all timestamps are UTC. Month boundaries are midnight UTC.
- **Period:** {period.strftime('%B %Y')}. Platform side is transactions *created* in the month;
  bank side is settlements *dated* in the month.
- **Bank rounding:** the bank truncates amounts to 2 decimals (toward zero); the platform keeps 3.
- **Settlement timing:** the bank settles 1-2 days after the transaction, in daily batches.
- **Matching:** platform `trn_id` = bank `trn_ref`, one to one. No fuzzy matching on amount or date.
- **Signs:** payments are positive, refunds are negative, in both files.
- **Out of scope:** no fees, chargebacks, FX, or reversals; the bank pays the gross amount.
""")
