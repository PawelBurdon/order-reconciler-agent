"""Tests for the tools the model is allowed to call.

Nothing here talks to Gemini. A comparison is built in memory and injected with
set_comparison(), so every test exercises the same code path the agent uses
without spending a token or needing an API key.
"""

import pandas as pd
import pytest

from src.agent import tools
from src.core.reconciler import reconcile


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


@pytest.fixture(autouse=True)
def clear_cache():
    """Never let one test see the comparison another test injected."""
    yield
    tools.set_comparison(None)


@pytest.fixture
def small_dataset():
    """Six lines covering every status, spread over two months."""
    tools.set_comparison(
        reconcile(
            planned_frame(
                [
                    ("ORD-1", "Velo Parts Ltd", "BRK-1", 120, "2025-08-03"),
                    ("ORD-2", "Velo Parts Ltd", "CHN-2", 200, "2025-08-05"),
                    ("ORD-3", "Northwind Cycles", "CRK-3", 60, "2025-08-08"),
                    ("ORD-4", "Alpine Gear Co", "HDL-4", 90, "2025-09-11"),
                    ("ORD-5", "Alpine Gear Co", "TYR-8", 280, "2025-09-19"),
                ]
            ),
            actual_frame(
                [
                    ("ORD-1", "Velo Parts Ltd", "BRK-1", 120, "2025-08-03"),
                    ("ORD-2", "Velo Parts Ltd", "CHN-2", 185, "2025-08-05"),
                    ("ORD-3", "Northwind Cycles", "CRK-3", 60, "2025-08-10"),
                    # Late and short at once, so its status is QTY_MISMATCH
                    # even though it is the latest delivery here - the same
                    # trap the sample data contains.
                    ("ORD-4", "Alpine Gear Co", "HDL-4", 40, "2025-09-16"),
                    ("ORD-6", "Northwind Cycles", "CBL-1", 300, "2025-09-25"),
                ]
            ),
        )
    )


@pytest.fixture
def wide_dataset():
    """More lines than a tool is allowed to return, with distinct shortfalls."""
    planned = [
        (f"ORD-{index:03d}", "Velo Parts Ltd", "BRK-1", 1000, "2025-08-01")
        for index in range(25)
    ]
    actual = [
        (f"ORD-{index:03d}", "Velo Parts Ltd", "BRK-1", 1000 - index, "2025-08-01")
        for index in range(25)
    ]
    tools.set_comparison(reconcile(planned_frame(planned), actual_frame(actual)))


# -- filter_records --------------------------------------------------------


def test_filter_by_customer_is_a_case_insensitive_fragment(small_dataset):
    result = tools.filter_records(customer="velo")

    assert result["total_matching"] == 2
    assert {record["customer"] for record in result["records"]} == {"Velo Parts Ltd"}


def test_filter_by_status(small_dataset):
    result = tools.filter_records(status="UNPLANNED")

    assert result["total_matching"] == 1
    assert result["records"][0]["order_id"] == "ORD-6"


def test_filtering_by_status_finds_lines_filed_under_another_one(small_dataset):
    """ORD-4 is late and short. Its headline says quantity; it is still late."""
    result = tools.filter_records(status="DATE_MISMATCH")

    returned = {record["order_id"] for record in result["records"]}
    assert returned == {"ORD-3", "ORD-4"}

    late_and_short = next(r for r in result["records"] if r["order_id"] == "ORD-4")
    assert late_and_short["status"] == "QTY_MISMATCH"
    assert late_and_short["also"] == "DATE_MISMATCH"


def test_filtering_by_match_does_not_match_qty_mismatch(small_dataset):
    """MATCH is a substring of QTY_MISMATCH, so the boundaries have to hold."""
    result = tools.filter_records(status="MATCH")

    assert {record["order_id"] for record in result["records"]} == {"ORD-1"}


def test_filters_combine_with_and(small_dataset):
    result = tools.filter_records(customer="Alpine", status="QTY_MISMATCH")

    assert result["total_matching"] == 1
    assert result["records"][0]["order_id"] == "ORD-4"


def test_date_filter_uses_the_delivery_date(small_dataset):
    result = tools.filter_records(date_from="2025-09-01", date_to="2025-09-30")

    returned = {record["order_id"] for record in result["records"]}
    # ORD-3 was planned for August and delivered in August; it stays out.
    assert "ORD-3" not in returned
    # ORD-5 was never delivered, so it is matched on its planned date instead
    # of dropping out of the month it was promised for.
    assert "ORD-5" in returned
    assert returned == {"ORD-4", "ORD-5", "ORD-6"}


def test_totals_cover_every_match_not_just_the_returned_rows(wide_dataset):
    result = tools.filter_records(customer="Velo")

    assert result["total_matching"] == 25
    assert result["returned"] == 20
    assert result["truncated"] is True
    # 0 + 1 + ... + 24 units missing across all 25 lines, including the five
    # the model never sees. This is the reason the totals exist.
    assert result["totals_over_all_matches"]["under_delivered_qty"] == 300
    assert result["totals_over_all_matches"]["status_counts"]["QTY_MISMATCH"] == 24


def test_truncation_keeps_the_biggest_deviations(wide_dataset):
    result = tools.filter_records(customer="Velo")

    assert result["records"][0]["qty_diff"] == -24
    # The rows that fall off the end are the small ones, never the large ones.
    smallest_returned = min(abs(record["qty_diff"]) for record in result["records"])
    assert smallest_returned == 5


def test_unknown_customer_returns_an_error_with_the_valid_names(small_dataset):
    result = tools.filter_records(customer="Acme")

    assert "error" in result
    assert "Velo Parts Ltd" in result["known_customers"]


def test_invalid_status_returns_an_error_with_the_valid_values(small_dataset):
    result = tools.filter_records(status="LATE")

    assert "error" in result
    assert "DATE_MISMATCH" in result["valid_statuses"]


def test_invalid_date_returns_an_error(small_dataset):
    result = tools.filter_records(date_from="September")

    assert "error" in result
    assert "YYYY-MM-DD" in result["error"]


def test_reversed_date_range_returns_an_error(small_dataset):
    result = tools.filter_records(date_from="2025-09-30", date_to="2025-09-01")

    assert "error" in result


# -- top_discrepancies -----------------------------------------------------


def test_top_discrepancies_ranks_by_absolute_size(small_dataset):
    result = tools.top_discrepancies(by="qty_diff", limit=3)

    order_ids = [record["order_id"] for record in result["records"]]
    # A surplus of 300 outranks a shortfall of 280: the ranking is by size, and
    # the sign in qty_diff is what tells them apart.
    assert order_ids == ["ORD-6", "ORD-5", "ORD-4"]
    assert result["records"][0]["qty_diff"] == 300
    assert result["records"][1]["qty_diff"] == -280


def test_top_discrepancies_by_percentage_excludes_lines_without_a_plan(small_dataset):
    result = tools.top_discrepancies(by="qty_diff_pct", limit=5)

    order_ids = [record["order_id"] for record in result["records"]]
    assert "ORD-6" not in order_ids
    # The exclusion is reported rather than silently applied.
    assert result["excluded_not_comparable"] == 1
    assert order_ids[0] == "ORD-5"


def test_top_discrepancies_ignores_lines_that_match(small_dataset):
    result = tools.top_discrepancies(by="qty_diff", limit=20)

    assert "ORD-1" not in [record["order_id"] for record in result["records"]]


def test_top_discrepancies_limit_is_capped(wide_dataset):
    result = tools.top_discrepancies(by="qty_diff", limit=500)

    assert result["returned"] == tools.MAX_RECORDS_RETURNED


def test_direction_shortfall_drops_the_surplus(small_dataset):
    """The whole point: ORD-6 is the largest deviation but not a shortfall."""
    result = tools.top_discrepancies(by="qty_diff", direction="shortfall", limit=3)

    order_ids = [record["order_id"] for record in result["records"]]
    assert "ORD-6" not in order_ids
    assert order_ids == ["ORD-5", "ORD-4", "ORD-2"]
    assert all(record["qty_diff"] < 0 for record in result["records"])


def test_direction_surplus_keeps_only_over_deliveries(small_dataset):
    result = tools.top_discrepancies(by="qty_diff", direction="surplus")

    assert [record["order_id"] for record in result["records"]] == ["ORD-6"]
    assert result["records"][0]["qty_diff"] == 300


def test_direction_defaults_to_ranking_both_together(small_dataset):
    """Unchanged behaviour when the question is not about a direction."""
    result = tools.top_discrepancies(by="qty_diff", limit=3)

    assert [record["order_id"] for record in result["records"]] == [
        "ORD-6",
        "ORD-5",
        "ORD-4",
    ]
    assert result["direction"] == "any"


def test_top_discrepancies_honours_a_date_range(small_dataset):
    """Ranking and filtering in one call, so nothing has to be intersected."""
    result = tools.top_discrepancies(
        by="qty_diff", direction="shortfall", date_from="2025-09-01"
    )

    # ORD-2, the August shortfall, is larger than ORD-4 but outside the period.
    assert [record["order_id"] for record in result["records"]] == ["ORD-5", "ORD-4"]
    assert result["filters_applied"] == {"date_from": "2025-09-01"}


def test_ranking_by_lateness_finds_the_line_the_status_hides(small_dataset):
    """ORD-4 is the latest delivery and is not a DATE_MISMATCH."""
    result = tools.top_discrepancies(by="date_diff_days", direction="late", limit=3)

    assert result["records"][0]["order_id"] == "ORD-4"
    assert result["records"][0]["date_diff_days"] == 5
    assert result["records"][0]["status"] == "QTY_MISMATCH"


def test_ranking_by_lateness_ignores_deliveries_that_were_on_time(small_dataset):
    result = tools.top_discrepancies(by="date_diff_days", limit=20)

    assert all(record["date_diff_days"] != 0 for record in result["records"])


def test_a_quantity_direction_is_refused_when_ranking_by_date(small_dataset):
    """A delivery is not a shortfall of days."""
    result = tools.top_discrepancies(by="date_diff_days", direction="shortfall")

    assert "error" in result
    assert result["valid_values"] == ["late", "early", "any"]


def test_a_date_direction_is_refused_when_ranking_by_quantity(small_dataset):
    result = tools.top_discrepancies(by="qty_diff", direction="late")

    assert "error" in result
    assert result["valid_values"] == ["shortfall", "surplus", "any"]


def test_top_discrepancies_rejects_a_bad_date(small_dataset):
    result = tools.top_discrepancies(date_from="last September")

    assert "error" in result
    assert "YYYY-MM-DD" in result["error"]


def test_unknown_direction_returns_an_error(small_dataset):
    result = tools.top_discrepancies(direction="downwards")

    assert "error" in result
    assert result["valid_values"] == ["shortfall", "surplus", "any"]


def test_unknown_ranking_column_returns_an_error(small_dataset):
    result = tools.top_discrepancies(by="delivery_delay")

    assert "error" in result
    assert result["valid_values"] == ["qty_diff", "qty_diff_pct", "date_diff_days"]


# -- group_by --------------------------------------------------------------


def test_group_by_customer_ranks_the_worst_served_first(small_dataset):
    result = tools.group_by(dimension="customer")

    names = [group["group"] for group in result["groups"]]
    # Alpine Gear Co is missing 280 + 50 units, more than anyone else.
    assert names[0] == "Alpine Gear Co"
    assert result["groups"][0]["under_delivered_qty"] == 330
    assert result["groups_total"] == 3


def test_group_by_answers_in_one_call_what_filtering_needs_one_call_each(
    small_dataset,
):
    """Every customer appears, so the model never has to loop over them."""
    result = tools.group_by(dimension="customer")

    assert {group["group"] for group in result["groups"]} == {
        "Alpine Gear Co",
        "Northwind Cycles",
        "Velo Parts Ltd",
    }
    assert sum(group["order_lines"] for group in result["groups"]) == 6


def test_group_by_can_rank_on_a_different_figure(small_dataset):
    result = tools.group_by(dimension="customer", sort_by="over_delivered_qty")

    # Northwind Cycles is the only one with a surplus: the unplanned 300 units.
    assert result["groups"][0]["group"] == "Northwind Cycles"
    assert result["groups"][0]["over_delivered_qty"] == 300


def test_group_by_honours_a_date_range(small_dataset):
    result = tools.group_by(dimension="customer", date_from="2025-09-01")

    names = {group["group"] for group in result["groups"]}
    assert names == {"Alpine Gear Co", "Northwind Cycles"}


def test_group_by_status_counts_the_lines(small_dataset):
    result = tools.group_by(dimension="status", sort_by="discrepancy_lines")

    by_status = {group["group"]: group["order_lines"] for group in result["groups"]}
    assert by_status["MISSING_ACTUAL"] == 1
    assert by_status["UNPLANNED"] == 1


def test_group_by_rejects_an_unknown_dimension(small_dataset):
    result = tools.group_by(dimension="region")

    assert "error" in result
    assert result["valid_dimensions"] == ["customer", "sku", "status"]


def test_group_by_rejects_an_unknown_sort_key(small_dataset):
    result = tools.group_by(sort_by="alphabetical")

    assert "error" in result
    assert "under_delivered_qty" in result["valid_values"]


# -- get_summary and dispatch ---------------------------------------------


def test_get_summary_reports_the_whole_dataset(small_dataset):
    summary = tools.get_summary()

    assert summary["total_order_lines"] == 6
    assert summary["status_counts"]["MISSING_ACTUAL"] == 1
    assert summary["status_counts"]["UNPLANNED"] == 1


def test_execute_tool_rejects_an_invented_tool_name(small_dataset):
    result = tools.execute_tool("drop_database", {})

    assert "error" in result
    assert "filter_records" in result["available_tools"]


def test_a_crashing_tool_returns_an_error_instead_of_raising(small_dataset):
    """The loop must never see an exception, whatever the arguments look like."""
    result = tools.execute_tool("top_discrepancies", {"limit": "not a number"})

    assert "error" in result
    assert "ValueError" in result["error"]
