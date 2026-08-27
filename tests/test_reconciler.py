"""Tests for the comparison logic.

These run on DataFrames built by hand rather than on the sample files, so a
test says what it is about: one scenario per test, three rows at most.
"""

import pandas as pd
import pytest

from src.core.reconciler import (
    STATUS_DATE_MISMATCH,
    STATUS_MATCH,
    STATUS_MISSING_ACTUAL,
    STATUS_QTY_MISMATCH,
    STATUS_UNPLANNED,
    reconcile,
    summarise,
)


def planned_frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["order_id", "customer", "sku", "planned_qty", "planned_date"]
    )
    frame["planned_qty"] = frame["planned_qty"].astype("Int64")
    frame["planned_date"] = pd.to_datetime(frame["planned_date"])
    return frame


def actual_frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["order_id", "customer", "sku", "actual_qty", "actual_date"]
    )
    frame["actual_qty"] = frame["actual_qty"].astype("Int64")
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    return frame


def line(comparison: pd.DataFrame, order_id: str) -> pd.Series:
    return comparison[comparison["order_id"] == order_id].iloc[0]


def test_missing_actual_is_detected():
    """An order that was planned but never delivered must survive the join."""
    comparison = reconcile(
        planned_frame([("ORD-1", "Velo Parts Ltd", "BRK-1", 100, "2025-07-01")]),
        actual_frame([]),
    )

    record = line(comparison, "ORD-1")
    assert record["status"] == STATUS_MISSING_ACTUAL
    assert pd.isna(record["actual_qty"])
    # The whole planned quantity counts as a shortfall, otherwise the order
    # would never appear in a ranking of the biggest deviations.
    assert record["qty_diff"] == -100
    assert record["qty_diff_pct"] == -100.0


def test_unplanned_is_detected():
    """A delivery nobody planned must survive the join as well."""
    comparison = reconcile(
        planned_frame([]),
        actual_frame([("ORD-9", "Northwind Cycles", "CBL-1", 300, "2025-09-25")]),
    )

    record = line(comparison, "ORD-9")
    assert record["status"] == STATUS_UNPLANNED
    assert pd.isna(record["planned_qty"])
    assert record["qty_diff"] == 300
    # There is no plan to measure against, so the percentage stays undefined
    # instead of being invented.
    assert pd.isna(record["qty_diff_pct"])
    # The customer name is taken from the only side that has it.
    assert record["customer"] == "Northwind Cycles"


def test_identical_lines_match():
    comparison = reconcile(
        planned_frame([("ORD-2", "Alpine Gear Co", "HDL-4", 90, "2025-07-11")]),
        actual_frame([("ORD-2", "Alpine Gear Co", "HDL-4", 90, "2025-07-11")]),
    )

    record = line(comparison, "ORD-2")
    assert record["status"] == STATUS_MATCH
    assert record["qty_diff"] == 0
    assert record["date_diff_days"] == 0


def test_quantity_difference_is_signed_and_relative():
    comparison = reconcile(
        planned_frame(
            [
                ("ORD-3", "Velo Parts Ltd", "CHN-2", 200, "2025-07-05"),
                ("ORD-4", "Velo Parts Ltd", "CRK-3", 50, "2025-07-06"),
            ]
        ),
        actual_frame(
            [
                ("ORD-3", "Velo Parts Ltd", "CHN-2", 185, "2025-07-05"),
                ("ORD-4", "Velo Parts Ltd", "CRK-3", 55, "2025-07-06"),
            ]
        ),
    )

    short = line(comparison, "ORD-3")
    assert short["status"] == STATUS_QTY_MISMATCH
    assert short["qty_diff"] == -15
    assert short["qty_diff_pct"] == -7.5

    surplus = line(comparison, "ORD-4")
    assert surplus["qty_diff"] == 5
    assert surplus["qty_diff_pct"] == 10.0


def test_right_quantity_on_the_wrong_date():
    comparison = reconcile(
        planned_frame([("ORD-5", "Northwind Cycles", "CRK-3", 60, "2025-07-08")]),
        actual_frame([("ORD-5", "Northwind Cycles", "CRK-3", 60, "2025-07-10")]),
    )

    record = line(comparison, "ORD-5")
    assert record["status"] == STATUS_DATE_MISMATCH
    assert record["date_diff_days"] == 2


def test_a_slip_inside_the_tolerance_is_not_a_discrepancy():
    planned = planned_frame(
        [
            ("ORD-A", "Velo Parts Ltd", "BRK-1", 10, "2025-08-01"),
            ("ORD-B", "Velo Parts Ltd", "CHN-2", 10, "2025-08-01"),
        ]
    )
    actual = actual_frame(
        [
            ("ORD-A", "Velo Parts Ltd", "BRK-1", 10, "2025-08-03"),
            ("ORD-B", "Velo Parts Ltd", "CHN-2", 10, "2025-08-06"),
        ]
    )

    strict = reconcile(planned, actual)
    assert line(strict, "ORD-A")["status"] == STATUS_DATE_MISMATCH
    assert line(strict, "ORD-B")["status"] == STATUS_DATE_MISMATCH

    lenient = reconcile(planned, actual, date_tolerance_days=2)
    forgiven = line(lenient, "ORD-A")
    assert forgiven["status"] == STATUS_MATCH
    # Forgiven, not hidden: the slip is still on the line, and so is the fact
    # that a setting is the only reason it does not count.
    assert forgiven["date_diff_days"] == 2
    assert "WITHIN_DATE_TOLERANCE" in forgiven["flags"]
    # Five days is still five days.
    assert line(lenient, "ORD-B")["status"] == STATUS_DATE_MISMATCH


def test_the_tolerance_works_in_both_directions():
    """Two days early is as far from the promise as two days late."""
    comparison = reconcile(
        planned_frame([("ORD-C", "Velo Parts Ltd", "BRK-1", 10, "2025-08-05")]),
        actual_frame([("ORD-C", "Velo Parts Ltd", "BRK-1", 10, "2025-08-03")]),
        date_tolerance_days=2,
    )

    record = line(comparison, "ORD-C")
    assert record["date_diff_days"] == -2
    assert record["status"] == STATUS_MATCH


def test_the_summary_says_which_tolerance_it_used():
    comparison = reconcile(
        planned_frame([("ORD-D", "Velo Parts Ltd", "BRK-1", 10, "2025-08-01")]),
        actual_frame([("ORD-D", "Velo Parts Ltd", "BRK-1", 10, "2025-08-02")]),
        date_tolerance_days=3,
    )

    assert summarise(comparison, 3)["date_tolerance_days"] == 3


def test_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="zero or more"):
        reconcile(planned_frame([]), actual_frame([]), date_tolerance_days=-1)


def test_quantity_outranks_date_when_both_differ():
    """A line that is both short and late is reported as a quantity problem."""
    comparison = reconcile(
        planned_frame([("ORD-6", "Alpine Gear Co", "SDL-5", 130, "2025-08-15")]),
        actual_frame([("ORD-6", "Alpine Gear Co", "SDL-5", 118, "2025-08-19")]),
    )

    record = line(comparison, "ORD-6")
    assert record["status"] == STATUS_QTY_MISMATCH
    # The date slip is not lost, it is just not the headline.
    assert record["date_diff_days"] == 4


def test_a_line_that_is_short_and_late_carries_both_statuses():
    """The headline still says quantity; the line no longer hides the date."""
    comparison = reconcile(
        planned_frame([("ORD-6", "Alpine Gear Co", "SDL-5", 130, "2025-08-15")]),
        actual_frame([("ORD-6", "Alpine Gear Co", "SDL-5", 118, "2025-08-19")]),
    )

    record = line(comparison, "ORD-6")
    assert record["status"] == STATUS_QTY_MISMATCH
    assert set(record["statuses"].split(";")) == {
        STATUS_QTY_MISMATCH,
        STATUS_DATE_MISMATCH,
    }


def test_a_matching_line_carries_only_match():
    comparison = reconcile(
        planned_frame([("ORD-7", "Velo Parts Ltd", "BRK-1", 10, "2025-08-01")]),
        actual_frame([("ORD-7", "Velo Parts Ltd", "BRK-1", 10, "2025-08-01")]),
    )

    assert line(comparison, "ORD-7")["statuses"] == STATUS_MATCH


def test_discrepancy_counts_count_a_line_once_per_problem():
    comparison = reconcile(
        planned_frame(
            [
                ("ORD-8", "Alpine Gear Co", "SDL-5", 130, "2025-08-15"),
                ("ORD-9", "Velo Parts Ltd", "BRK-1", 10, "2025-08-01"),
            ]
        ),
        actual_frame(
            [
                ("ORD-8", "Alpine Gear Co", "SDL-5", 118, "2025-08-19"),
                ("ORD-9", "Velo Parts Ltd", "BRK-1", 10, "2025-08-03"),
            ]
        ),
    )

    summary = summarise(comparison)
    # One line is filed under quantity, the other under date...
    assert summary["status_counts"][STATUS_QTY_MISMATCH] == 1
    assert summary["status_counts"][STATUS_DATE_MISMATCH] == 1
    # ...but two lines arrived on the wrong date, and that is the number
    # somebody asking about late deliveries means.
    assert summary["lines_with_each_discrepancy"][STATUS_DATE_MISMATCH] == 2
    assert summary["lines_with_each_discrepancy"][STATUS_QTY_MISMATCH] == 1


def test_split_delivery_is_summed_and_flagged():
    """The same line delivered twice is one line, with the quantities added."""
    comparison = reconcile(
        planned_frame([("ORD-7", "Velo Parts Ltd", "CHN-2", 180, "2025-08-08")]),
        actual_frame(
            [
                ("ORD-7", "Velo Parts Ltd", "CHN-2", 100, "2025-08-08"),
                ("ORD-7", "Velo Parts Ltd", "CHN-2", 80, "2025-08-11"),
            ]
        ),
    )

    assert len(comparison) == 1
    record = line(comparison, "ORD-7")
    assert record["actual_qty"] == 180
    assert record["qty_diff"] == 0
    # The order was only complete on the later of the two dates.
    assert record["date_diff_days"] == 3
    assert "SPLIT_DELIVERY" in record["flags"]


def test_one_order_with_two_skus_stays_two_lines():
    """The join key is order_id plus SKU, not order_id alone."""
    comparison = reconcile(
        planned_frame(
            [
                ("ORD-8", "Velo Parts Ltd", "SDL-5", 160, "2025-09-12"),
                ("ORD-8", "Velo Parts Ltd", "WHL-6", 45, "2025-09-12"),
            ]
        ),
        actual_frame(
            [
                ("ORD-8", "Velo Parts Ltd", "SDL-5", 160, "2025-09-12"),
                ("ORD-8", "Velo Parts Ltd", "WHL-6", 40, "2025-09-12"),
            ]
        ),
    )

    assert len(comparison) == 2
    assert set(comparison["status"]) == {STATUS_MATCH, STATUS_QTY_MISMATCH}


def test_missing_delivery_date_does_not_become_a_date_mismatch():
    """A blank date is a data-quality flag, not a discrepancy."""
    comparison = reconcile(
        planned_frame([("ORD-10", "Pedalworks Trading", "CBL-1", 450, "2025-08-28")]),
        actual_frame([("ORD-10", "Pedalworks Trading", "CBL-1", 450, None)]),
    )

    record = line(comparison, "ORD-10")
    assert record["status"] == STATUS_MATCH
    assert pd.isna(record["date_diff_days"])
    assert "MISSING_ACTUAL_DATE" in record["flags"]


def test_summarise_totals():
    comparison = reconcile(
        planned_frame(
            [
                ("ORD-11", "Velo Parts Ltd", "BRK-1", 100, "2025-07-01"),
                ("ORD-12", "Alpine Gear Co", "CHN-2", 200, "2025-07-02"),
            ]
        ),
        actual_frame(
            [
                ("ORD-11", "Velo Parts Ltd", "BRK-1", 100, "2025-07-01"),
                ("ORD-12", "Alpine Gear Co", "CHN-2", 150, "2025-07-02"),
                ("ORD-13", "Northwind Cycles", "TYR-8", 30, "2025-07-03"),
            ]
        ),
    )

    summary = summarise(comparison)
    assert summary["total_order_lines"] == 3
    assert summary["matched_lines"] == 1
    assert summary["discrepancy_lines"] == 2
    assert summary["status_counts"]["QTY_MISMATCH"] == 1
    assert summary["status_counts"]["UNPLANNED"] == 1
    assert summary["under_delivered_qty"] == 50
    assert summary["over_delivered_qty"] == 30
    assert summary["net_qty_diff"] == -20
    assert summary["customers"] == [
        "Alpine Gear Co",
        "Northwind Cycles",
        "Velo Parts Ltd",
    ]
    assert summary["date_range"] == {"from": "2025-07-01", "to": "2025-07-03"}
