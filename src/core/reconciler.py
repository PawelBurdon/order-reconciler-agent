"""Comparison of planned against actual orders.

Part of the deterministic core layer: no AI. Every number the agent ever
reports is produced here, by plain pandas, so it is reproducible and testable.
"""

from __future__ import annotations

import pandas as pd

from .loader import KEY_COLUMNS

STATUS_MATCH = "MATCH"
STATUS_QTY_MISMATCH = "QTY_MISMATCH"
STATUS_DATE_MISMATCH = "DATE_MISMATCH"
STATUS_MISSING_ACTUAL = "MISSING_ACTUAL"
STATUS_UNPLANNED = "UNPLANNED"

ALL_STATUSES = [
    STATUS_MATCH,
    STATUS_QTY_MISMATCH,
    STATUS_DATE_MISMATCH,
    STATUS_MISSING_ACTUAL,
    STATUS_UNPLANNED,
]

FLAG_SPLIT_DELIVERY = "SPLIT_DELIVERY"
FLAG_MISSING_PLANNED_DATE = "MISSING_PLANNED_DATE"
FLAG_MISSING_ACTUAL_DATE = "MISSING_ACTUAL_DATE"
FLAG_MISSING_CUSTOMER = "MISSING_CUSTOMER"

COMPARISON_COLUMNS = [
    "order_id",
    "customer",
    "sku",
    "planned_qty",
    "actual_qty",
    "qty_diff",
    "qty_diff_pct",
    "planned_date",
    "actual_date",
    "date_diff_days",
    "status",
    "statuses",
    "flags",
]

# Which single status a line is reported under when only one will fit: the
# most expensive kind of wrong first.
STATUS_PRIORITY = [
    STATUS_UNPLANNED,
    STATUS_MISSING_ACTUAL,
    STATUS_QTY_MISMATCH,
    STATUS_DATE_MISMATCH,
    STATUS_MATCH,
]


def reconcile(planned: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Compare both sides and return one row per (order_id, sku) line.

    The join is an outer join, so a line that exists on only one side survives
    the comparison instead of disappearing - those are exactly the rows a
    reconciliation is looking for.
    """
    planned_lines = _collapse_duplicates(planned, "planned_qty", "planned_date")
    actual_lines = _collapse_duplicates(actual, "actual_qty", "actual_date")

    comparison = planned_lines.merge(
        actual_lines,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_planned", "_actual"),
    )

    # The customer name lives on both sides. The planned side is treated as the
    # reference; the actual side only fills the gap for UNPLANNED lines.
    comparison["customer"] = (
        comparison["customer_planned"].fillna(comparison["customer_actual"]).fillna("")
    )

    comparison["qty_diff"] = _quantity_difference(comparison)
    comparison["qty_diff_pct"] = _quantity_difference_pct(comparison)
    comparison["date_diff_days"] = _date_difference_days(comparison)
    comparison["statuses"] = _collect_statuses(comparison)
    comparison["status"] = _headline_status(comparison["statuses"])
    comparison["flags"] = _collect_flags(comparison)

    comparison = comparison[COMPARISON_COLUMNS]
    return comparison.sort_values(KEY_COLUMNS, ignore_index=True)


def _collapse_duplicates(
    frame: pd.DataFrame, qty_column: str, date_column: str
) -> pd.DataFrame:
    """Make (order_id, sku) unique by summing quantities of repeated lines.

    A repeated key is a split delivery: the same product of the same order
    shipped in two batches. Summing is the honest business answer - 100 + 80
    really is the 180 units that were planned - but the fact that it happened
    must not vanish, so the line is flagged with SPLIT_DELIVERY and keeps the
    latest of the dates (the moment the order was actually complete).
    """
    grouped = frame.groupby(KEY_COLUMNS, as_index=False, dropna=False).agg(
        customer=("customer", "first"),
        **{
            qty_column: (qty_column, "sum"),
            date_column: (date_column, "max"),
        },
        line_count=(qty_column, "size"),
    )
    grouped["is_split"] = grouped["line_count"] > 1
    return grouped.drop(columns=["line_count"])


def _quantity_difference(comparison: pd.DataFrame) -> pd.Series:
    """actual - planned, where a side that does not exist counts as zero.

    Keeping the difference defined for MISSING_ACTUAL and UNPLANNED lines is
    what lets "the biggest shortfalls" include an order that was never
    delivered at all - usually the biggest shortfall there is. The quantity
    columns themselves stay empty for those rows, so "nothing was recorded" is
    still distinguishable from "zero units were recorded".
    """
    planned = comparison["planned_qty"].fillna(0).astype("int64")
    actual = comparison["actual_qty"].fillna(0).astype("int64")
    return (actual - planned).astype("Int64")


def _quantity_difference_pct(comparison: pd.DataFrame) -> pd.Series:
    """The difference relative to the plan, in percent, rounded to 0.1.

    Undefined (NA) when there is no plan to measure against - an UNPLANNED
    line, or the theoretical case of a plan for zero units.
    """
    planned = comparison["planned_qty"].astype("Float64")
    percentage = comparison["qty_diff"].astype("Float64") / planned * 100
    percentage[planned.isna() | (planned == 0)] = pd.NA
    return percentage.round(1)


def _date_difference_days(comparison: pd.DataFrame) -> pd.Series:
    """Delivery date minus planned date, in days; NA if either date is missing."""
    difference = comparison["actual_date"] - comparison["planned_date"]
    return difference.dt.days.astype("Int64")


def _collect_statuses(comparison: pd.DataFrame) -> pd.Series:
    """Every way a line differs from the plan, not just the worst one.

    A line can be short and late at once, and for a long time this column did
    not exist: only the headline did, quantity beat date, and so the latest
    delivery in the sample data was recorded as QTY_MISMATCH. Anything looking
    for late deliveries by asking for DATE_MISMATCH missed it - which is
    exactly what the agent did, fluently and repeatedly, until this was
    measured. The headline is still useful for a report where one word has to
    do; it is a summary, and summaries are the wrong thing to filter on.

    Built with masks rather than a loop over the rows. The first version of
    this walked the frame one row at a time and was, on a 200,000 line extract,
    ninety-five percent of the entire comparison - three seconds where the
    outer join underneath it took a tenth of one. The roadmap had "cache the
    comparison" on it for that reason; profiling said the cost was here, and a
    cache would have hidden it behind a staleness check and a new dependency.
    """
    unplanned = comparison["planned_qty"].isna()
    missing = comparison["actual_qty"].isna() & ~unplanned
    ordinary = ~unplanned & ~missing

    quantity_differs = (comparison["qty_diff"] != 0) & ordinary
    date_differs = (
        comparison["date_diff_days"].notna()
        & (comparison["date_diff_days"] != 0)
        & ordinary
    )

    quantity = pd.Series("", index=comparison.index, dtype="object")
    quantity[quantity_differs] = STATUS_QTY_MISMATCH
    date = pd.Series("", index=comparison.index, dtype="object")
    date[date_differs] = STATUS_DATE_MISMATCH

    # Joining with a separator and then stripping it handles all four
    # combinations without a branch: neither, either, or both.
    statuses = quantity.str.cat(date, sep=";").str.strip(";")
    statuses[statuses == ""] = STATUS_MATCH
    statuses[missing] = STATUS_MISSING_ACTUAL
    statuses[unplanned] = STATUS_UNPLANNED
    return statuses


def _headline_status(statuses: pd.Series) -> pd.Series:
    """Reduce the list to the single status a one-word column can show.

    There are only a handful of distinct combinations however many lines there
    are, so the choice is made once per combination and then looked up.
    """

    def pick(joined: str) -> str:
        entries = joined.split(";")
        for candidate in STATUS_PRIORITY:
            if candidate in entries:
                return candidate
        return STATUS_MATCH

    return statuses.map({value: pick(value) for value in statuses.unique()})


def _collect_flags(comparison: pd.DataFrame) -> pd.Series:
    """Data-quality notes that are not a status on their own.

    Kept separate from `status` on purpose: status answers "how does this line
    differ from the plan", flags answer "how much can you trust this line".
    """
    conditions = {
        FLAG_SPLIT_DELIVERY: (
            comparison["is_split_planned"].fillna(False).astype(bool)
            | comparison["is_split_actual"].fillna(False).astype(bool)
        ),
        FLAG_MISSING_PLANNED_DATE: (
            comparison["planned_qty"].notna() & comparison["planned_date"].isna()
        ),
        FLAG_MISSING_ACTUAL_DATE: (
            comparison["actual_qty"].notna() & comparison["actual_date"].isna()
        ),
        FLAG_MISSING_CUSTOMER: comparison["customer"].astype(str).str.strip() == "",
    }

    collected = pd.Series(
        [[] for _ in range(len(comparison))], index=comparison.index, dtype="object"
    )
    for flag, mask in conditions.items():
        for position in comparison.index[mask]:
            collected[position].append(flag)

    return collected.str.join(";")


def _count_each_status(comparison: pd.DataFrame) -> dict:
    """How many lines carry each status, counting a line once per status."""
    entries = comparison["statuses"].str.split(";").explode()
    counted = entries[entries != ""].value_counts().to_dict()
    return {
        status: int(counted.get(status, 0))
        for status in ALL_STATUSES
        if counted.get(status, 0)
    }


def summarise(comparison: pd.DataFrame) -> dict:
    """Aggregate the comparison into a handful of headline numbers.

    The return value is a small JSON-serialisable dict, because it is fed
    straight to the model as a tool result. It also lists the customer names
    and the date range, which lets the model build a valid filter (it cannot
    guess how a customer is spelled in the data).
    """
    status_counts = {
        status: int((comparison["status"] == status).sum()) for status in ALL_STATUSES
    }
    matched = status_counts[STATUS_MATCH]
    total = len(comparison)
    discrepancies = total - matched

    qty_diff = comparison["qty_diff"].astype("Float64")
    shortfall = qty_diff[qty_diff < 0].sum()
    surplus = qty_diff[qty_diff > 0].sum()

    dates = pd.concat([comparison["planned_date"], comparison["actual_date"]]).dropna()

    flags = comparison["flags"].str.split(";").explode()
    flag_counts = flags[flags != ""].value_counts().to_dict()

    return {
        "total_order_lines": total,
        "matched_lines": matched,
        "discrepancy_lines": discrepancies,
        "discrepancy_rate_pct": round(discrepancies / total * 100, 1) if total else 0.0,
        "status_counts": status_counts,
        # Counted from every status a line carries rather than its headline, so
        # a line that is short and late is counted in both. These add up to
        # more than the number of lines, on purpose: status_counts answers
        # "how is each line filed", this answers "how many lines have this
        # problem at all", and the second is the one people mean.
        "lines_with_each_discrepancy": _count_each_status(comparison),
        "total_planned_qty": int(comparison["planned_qty"].fillna(0).sum()),
        "total_actual_qty": int(comparison["actual_qty"].fillna(0).sum()),
        "net_qty_diff": int(qty_diff.fillna(0).sum()),
        "under_delivered_qty": int(abs(shortfall)) if pd.notna(shortfall) else 0,
        "over_delivered_qty": int(surplus) if pd.notna(surplus) else 0,
        "customers": sorted(c for c in comparison["customer"].unique() if c),
        "customers_with_discrepancies": sorted(
            c
            for c in comparison.loc[
                comparison["status"] != STATUS_MATCH, "customer"
            ].unique()
            if c
        ),
        "date_range": {
            "from": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "to": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
        },
        "data_quality_flags": {key: int(value) for key, value in flag_counts.items()},
    }
