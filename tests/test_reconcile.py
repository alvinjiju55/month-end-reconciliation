"""
Tests for reconcile.py.

ground_truth.json is the answer key. The reconciler never reads it; only these
tests do. Every dataset is generated fresh into a pytest temp dir, so the
CSVs in the project folder are never touched.
"""

from __future__ import annotations

import filecmp
import json
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from generator import PLANT_NAMES, SEED, generate
from reconcile import load_bank, load_platform, reconcile

FILES = ["platform_transactions.csv", "bank_settlements.csv", "ground_truth.json"]
ALL_ON = {p: True for p in PLANT_NAMES}
ALL_OFF = {p: False for p in PLANT_NAMES}
D0 = Decimal("0.000")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_dataset(out_dir: Path, seed: int = SEED, plants: dict | None = None) -> Path:
    generate(seed, dict(plants or ALL_ON), out_dir)
    return out_dir


def run(data_dir: Path):
    truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    result = reconcile(load_platform(data_dir / "platform_transactions.csv"),
                       load_bank(data_dir / "bank_settlements.csv"))
    return truth, result


def flagged_ids(result, category: str) -> list[str]:
    return sorted(result.flags.loc[result.flags["category"] == category, "trn_id"])


def impact(result, category: str) -> Decimal:
    return sum(result.flags.loc[result.flags["category"] == category, "bridge_impact"], D0)


def reason(result, category: str, trn_id: str) -> str:
    """One trn can carry several flags (e.g. duplicate + truncation), so key on both."""
    f = result.flags
    sel = f[(f["category"] == category) & (f["trn_id"] == trn_id)]
    assert len(sel) == 1, f"expected one {category} flag for {trn_id}, got {len(sel)}"
    return sel["reason"].iloc[0]


def bridge_amount(result, line: str) -> Decimal:
    return result.bridge.loc[result.bridge["line"] == line, "amount"].iloc[0]


def expected_from_truth(truth: dict) -> dict:
    """What the reconciler should flag, derived only from ground_truth.json."""
    P = truth["plants"]
    one = lambda name: [P[name]["trn_id"]] if P[name]["enabled"] else []  # noqa: E731
    late = P["late_settlement"]["transactions"]
    late_drift = (Decimal(P["late_settlement"]["platform_amount_total"])
                  - Decimal(P["late_settlement"]["bank_amount_total"]))
    drift_on = P["truncation_drift"]["enabled"]
    # ground truth counts drift over the whole bank extract; the October bridge
    # excludes late rows (they leave the bridge at full platform amount instead).
    late_rows_with_drift = sum(1 for t in late if not t["amount"].endswith("0")) if drift_on else 0
    return {
        "duplicate": one("duplicate"),
        "never_settled": one("unsettled_payment"),
        "orphan_refund": one("orphan_refund"),
        "settled_after_month_end": sorted(t["trn_id"] for t in late),
        "truncation_count": P["truncation_drift"]["rows_affected"] - late_rows_with_drift,
        "truncation_total": Decimal(P["truncation_drift"]["total_drift"]) - late_drift,
    }


def assert_matches_truth(truth: dict, result) -> None:
    exp = expected_from_truth(truth)
    P = truth["plants"]

    # right ids in the right category
    for cat in ["duplicate", "never_settled", "orphan_refund", "settled_after_month_end"]:
        assert flagged_ids(result, cat) == exp[cat], cat

    # nothing extra: categories that were never planted stay empty
    assert flagged_ids(result, "amount_mismatch") == []
    assert flagged_ids(result, "bank_only") == []
    assert (result.flags["category"] == "truncation_drift").sum() == exp["truncation_count"]
    expected_total = (len(exp["duplicate"]) + len(exp["never_settled"]) + len(exp["orphan_refund"])
                      + len(exp["settled_after_month_end"]) + exp["truncation_count"])
    assert len(result.flags) == expected_total

    # amounts
    assert -impact(result, "truncation_drift") == exp["truncation_total"]
    if P["duplicate"]["enabled"]:
        assert -impact(result, "duplicate") == Decimal(P["duplicate"]["amount"])
    if P["unsettled_payment"]["enabled"]:
        assert -impact(result, "never_settled") == Decimal(P["unsettled_payment"]["amount"])
    assert -impact(result, "settled_after_month_end") == Decimal(P["late_settlement"]["platform_amount_total"])

    # truncation flags are each a sub-cent, non-zero difference
    drift = result.flags.loc[result.flags["category"] == "truncation_drift", "bridge_impact"]
    assert all(Decimal("0") < abs(d) < Decimal("0.01") for d in drift)

    # orphan refund lives in exceptions and moves no money
    assert sorted(result.exceptions["trn_id"]) == exp["orphan_refund"]
    assert impact(result, "orphan_refund") == 0

    # totals and the bridge
    assert result.summary["platform_total"] == Decimal(truth["totals"]["platform_total"])
    assert result.summary["bank_total"] == Decimal(truth["totals"]["bank_total_settled_in_october"])
    assert result.summary["gap"] == Decimal(truth["totals"]["gap_vs_october_settlements"])
    assert result.unexplained == Decimal("0.000")
    assert bridge_amount(result, "expected_bank_total") == bridge_amount(result, "bank_total")

    # every flag has a plain reason
    assert result.flags["reason"].str.len().gt(10).all()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def default_dir(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("default"))


@pytest.fixture(scope="session")
def default_run(default_dir):
    return run(default_dir)


# --------------------------------------------------------------------------
# Planted gaps
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [SEED, 7, 12345])
def test_every_planted_gap_found_and_nothing_extra(tmp_path, seed):
    truth, result = run(make_dataset(tmp_path, seed=seed))
    assert_matches_truth(truth, result)


def test_unexplained_is_zero(default_run):
    _, result = default_run
    assert result.unexplained == Decimal("0.000")
    assert str(result.unexplained) == "0.000"


def test_reasons_are_specific(default_run):
    truth, result = default_run
    P = truth["plants"]
    for late in P["late_settlement"]["transactions"]:
        day = int(late["settled_on"][-2:])
        assert re.search(rf"settled Nov {day} .*after month end Oct 31",
                         reason(result, "settled_after_month_end", late["trn_id"]))
    assert "no matching settlement" in reason(result, "never_settled", P["unsettled_payment"]["trn_id"])
    assert "recorded 2 times" in reason(result, "duplicate", P["duplicate"]["trn_id"])
    orphan = P["orphan_refund"]
    assert orphan["original_trn_id"] in reason(result, "orphan_refund", orphan["trn_id"])
    assert result.flags.loc[result.flags["category"] == "truncation_drift", "reason"] \
        .str.match(r"bank truncated -?\d+\.\d{3} to -?\d+\.\d{2} \(drift -?0\.00\d\)").all()


# --------------------------------------------------------------------------
# Clean data
# --------------------------------------------------------------------------
def test_all_plants_off_no_flags_zero_gap(tmp_path):
    truth, result = run(make_dataset(tmp_path, plants=ALL_OFF))
    assert result.flags.empty
    assert result.exceptions.empty
    assert result.summary["gap"] == 0
    assert result.unexplained == 0
    assert result.summary["platform_total"] == Decimal(truth["totals"]["platform_total"])


@pytest.mark.parametrize("plants", [ALL_ON, ALL_OFF], ids=["all_on", "all_off"])
def test_bridge_amounts_always_three_decimals(tmp_path, plants):
    _, result = run(make_dataset(tmp_path, plants=plants))
    # Empty lines included: they must read 0.000, not 0
    for amount in result.bridge["amount"]:
        assert re.fullmatch(r"-?\d+\.\d{3}", str(amount)), amount
    printed = result.bridge.to_string()
    assert re.search(r"\b0\.000\b", printed)
    assert not re.search(r"(?<![\d.])0(?![\d.])\s*$", printed, re.M)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def test_same_seed_identical_data(tmp_path):
    a = make_dataset(tmp_path / "a", seed=SEED)
    b = make_dataset(tmp_path / "b", seed=SEED)
    for name in FILES:
        assert filecmp.cmp(a / name, b / name, shallow=False), name


def test_different_seed_different_data(tmp_path):
    a = make_dataset(tmp_path / "a", seed=SEED)
    b = make_dataset(tmp_path / "b", seed=7)
    assert not filecmp.cmp(a / FILES[0], b / FILES[0], shallow=False)


# --------------------------------------------------------------------------
# One plant off at a time
# --------------------------------------------------------------------------
PLANT_TO_CATEGORY = {
    "late_settlement": "settled_after_month_end",
    "truncation_drift": "truncation_drift",
    "duplicate": "duplicate",
    "orphan_refund": "orphan_refund",
    "unsettled_payment": "never_settled",
}


@pytest.mark.parametrize("plant", PLANT_NAMES)
def test_each_plant_off(tmp_path, plant, default_run):
    truth, result = run(make_dataset(tmp_path, plants={**ALL_ON, plant: False}))
    assert_matches_truth(truth, result)
    assert flagged_ids(result, PLANT_TO_CATEGORY[plant]) == []

    # the other plants are found with exactly the same ids as with everything on
    _, full = default_run
    for other, cat in PLANT_TO_CATEGORY.items():
        if other not in (plant, "truncation_drift"):
            assert flagged_ids(result, cat) == flagged_ids(full, cat), cat


# --------------------------------------------------------------------------
# Unplanted errors made by hand
# --------------------------------------------------------------------------
def test_hand_made_errors_flagged(default_dir):
    truth = json.loads((default_dir / "ground_truth.json").read_text(encoding="utf-8"))
    platform = load_platform(default_dir / "platform_transactions.csv")
    bank = load_bank(default_dir / "bank_settlements.csv")

    planted = {truth["plants"][k].get("trn_id") for k in truth["plants"]}
    clean = bank[(bank["settled_on"] < "2026-11-01") & ~bank["trn_ref"].isin(planted)]
    wrong_ref, twice_ref = clean["trn_ref"].iloc[10], clean["trn_ref"].iloc[20]

    # 1. wrong amount: bank paid 100 more than the platform amount
    i = bank.index[bank["trn_ref"] == wrong_ref][0]
    bank.loc[i, "amount"] = f"{Decimal(bank.loc[i, 'amount']) + 100:.2f}"
    # 2. bank pays one transaction twice
    dup_row = bank[bank["trn_ref"] == twice_ref].iloc[0].to_dict()
    # 3. bank row with no platform record
    stranger = {"settlement_id": "STL-X1", "batch_id": "BATCH-20261015",
                "trn_ref": "TRN-NOTREAL", "settled_on": "2026-10-15", "amount": "50.00"}
    bank = pd.concat([bank, pd.DataFrame([{**dup_row, "settlement_id": "STL-X2"}, stranger])],
                     ignore_index=True)

    result = reconcile(platform, bank)

    assert flagged_ids(result, "amount_mismatch") == sorted([wrong_ref, twice_ref])
    assert "more than truncation explains" in reason(result, "amount_mismatch", wrong_ref)
    assert "settled 2 times" in reason(result, "amount_mismatch", twice_ref)
    assert flagged_ids(result, "bank_only") == ["TRN-NOTREAL"]
    assert "no platform transaction" in reason(result, "bank_only", "TRN-NOTREAL")
    assert bridge_amount(result, "bank_only") == Decimal("50.000")

    # the planted gaps are still found
    assert flagged_ids(result, "duplicate") == [truth["plants"]["duplicate"]["trn_id"]]
    assert flagged_ids(result, "never_settled") == [truth["plants"]["unsettled_payment"]["trn_id"]]
    assert result.unexplained == Decimal("0.000")


# --------------------------------------------------------------------------
# Bad input gives a clear error
# --------------------------------------------------------------------------
@pytest.mark.parametrize("which,column", [("platform", "amount"), ("platform", "trn_id"),
                                          ("bank", "trn_ref"), ("bank", "settled_on")])
def test_missing_column_clear_error(default_dir, which, column):
    platform = load_platform(default_dir / "platform_transactions.csv")
    bank = load_bank(default_dir / "bank_settlements.csv")
    if which == "platform":
        platform = platform.drop(columns=column)
    else:
        bank = bank.drop(columns=column)
    with pytest.raises(ValueError, match=rf"{which} data is missing columns: \['{column}'\]"):
        reconcile(platform, bank)


@pytest.mark.parametrize("loader,name", [(load_platform, "platform"), (load_bank, "bank")])
def test_empty_file_clear_error(tmp_path, loader, name):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match=rf"{name} file is empty"):
        loader(empty)


def test_missing_file_clear_error(tmp_path):
    with pytest.raises(ValueError, match="platform file not found"):
        load_platform(tmp_path / "nope.csv")


def test_non_numeric_amount_clear_error(default_dir):
    platform = load_platform(default_dir / "platform_transactions.csv")
    platform.loc[5, "amount"] = "abc"
    with pytest.raises(ValueError, match=r"platform row 7 .*'abc' is not a number"):
        reconcile(platform, load_bank(default_dir / "bank_settlements.csv"))


def test_header_only_bank_everything_unsettled(default_dir, tmp_path):
    header_only = tmp_path / "bank.csv"
    header_only.write_text((default_dir / "bank_settlements.csv").read_text().splitlines()[0] + "\n")
    platform = load_platform(default_dir / "platform_transactions.csv")
    result = reconcile(platform, load_bank(header_only))
    assert result.summary["flag_counts"]["never_settled"] == platform["trn_id"].nunique()
    assert result.unexplained == 0


def test_header_only_both_no_flags(default_dir, tmp_path):
    for name in ["platform_transactions.csv", "bank_settlements.csv"]:
        (tmp_path / name).write_text((default_dir / name).read_text().splitlines()[0] + "\n")
    result = reconcile(load_platform(tmp_path / FILES[0]), load_bank(tmp_path / FILES[1]))
    assert result.flags.empty
    assert result.unexplained == 0
