"""The tools the model is allowed to call, plus their schemas.

This module is the whole contract between the AI layer and the deterministic
layer. Two rules run through all of it:

1. A tool never returns the dataset. It returns numbers, aggregates and at most
   MAX_RECORDS_RETURNED rows. Dumping 31 rows here would still "work", but the
   habit does not survive a real dataset: 50k rows is both an enormous token
   bill and, worse, a context the model reasons over badly - it starts adding
   up columns by hand and quietly gets it wrong. Aggregating in pandas and
   handing over the result keeps the arithmetic exact and the context small.

2. A tool never raises. Every failure comes back as {"error": "..."} inside the
   function response, because an exception ends the conversation while an error
   message lets the model correct itself and call the tool again - for instance
   after misspelling a customer name.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from google.genai import types

from ..core.loader import DataValidationError, load_actual_orders, load_planned_orders
from ..core.reconciler import ALL_STATUSES, STATUS_MATCH, reconcile, summarise
from ..core.report import write_report

DEFAULT_PLANNED_PATH = Path("sample_data/planned_orders.csv")
DEFAULT_ACTUAL_PATH = Path("sample_data/actual_orders.csv")
DEFAULT_REPORT_PATH = Path("reconciliation_report.xlsx")

# The hard ceiling on rows handed to the model in a single tool result.
MAX_RECORDS_RETURNED = 20

_state: dict[str, Any] = {
    "planned_path": DEFAULT_PLANNED_PATH,
    "actual_path": DEFAULT_ACTUAL_PATH,
    "comparison": None,
}


def configure_data_sources(planned_path: str | Path, actual_path: str | Path) -> None:
    """Point the tools at different CSV files (used by the CLI)."""
    _state["planned_path"] = Path(planned_path)
    _state["actual_path"] = Path(actual_path)
    _state["comparison"] = None


def set_comparison(comparison: pd.DataFrame | None) -> None:
    """Inject a ready-made comparison, bypassing the files (used by tests)."""
    _state["comparison"] = comparison


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def _tool(function: Callable[..., dict]) -> Callable[..., dict]:
    """Turn any exception raised inside a tool into an error payload."""

    @functools.wraps(function)
    def wrapper(**kwargs: Any) -> dict:
        try:
            return function(**kwargs)
        except DataValidationError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - the agent must never crash here
            return {"error": f"{type(exc).__name__}: {exc}"}

    return wrapper


@_tool
def load_and_compare() -> dict:
    """Read both CSV files, run the comparison and cache it."""
    comparison = _load_comparison()
    summary = summarise(comparison)

    # Deliberately not the data - just enough vocabulary for the model to build
    # a valid follow-up call: the exact customer spellings, the legal status
    # values and the period the files actually cover.
    return {
        "loaded": True,
        "planned_file": str(_state["planned_path"]),
        "actual_file": str(_state["actual_path"]),
        "order_lines_compared": summary["total_order_lines"],
        "customers": summary["customers"],
        "date_range": summary["date_range"],
        "available_statuses": ALL_STATUSES,
    }


@_tool
def get_summary() -> dict:
    """Return the aggregate figures for the whole comparison."""
    return summarise(_require_comparison())


@_tool
def filter_records(
    customer: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Return the order lines matching every filter that was supplied."""
    comparison = _require_comparison()
    filtered = comparison
    applied: dict[str, str] = {}

    if customer:
        # Substring, case-insensitive: the model should not have to reproduce
        # "Crankset Supply Inc" character for character to ask about Crankset.
        matches = filtered["customer"].str.contains(customer, case=False, na=False)
        if not matches.any():
            return {
                "error": f"No customer matches '{customer}'.",
                "known_customers": sorted(c for c in comparison["customer"].unique() if c),
            }
        filtered = filtered[matches]
        applied["customer"] = customer

    if status:
        normalised = status.strip().upper()
        if normalised not in ALL_STATUSES:
            return {
                "error": f"Unknown status '{status}'.",
                "valid_statuses": ALL_STATUSES,
            }
        filtered = filtered[filtered["status"] == normalised]
        applied["status"] = normalised

    if date_from or date_to:
        try:
            boundaries = _parse_date_range(date_from, date_to)
        except ValueError as exc:
            return {"error": str(exc)}

        dates = _effective_dates(filtered)
        if boundaries[0] is not None:
            filtered = filtered[dates >= boundaries[0]]
            applied["date_from"] = date_from
            dates = _effective_dates(filtered)
        if boundaries[1] is not None:
            filtered = filtered[dates <= boundaries[1]]
            applied["date_to"] = date_to

    page = _rank_by_impact(filtered).head(MAX_RECORDS_RETURNED)

    return {
        "filters_applied": applied,
        "date_field_used": "actual_date, falling back to planned_date when nothing was delivered",
        "total_matching": len(filtered),
        "returned": len(page),
        "truncated": len(filtered) > len(page),
        # Totals are computed over every match, not just the rows shown, so the
        # model never has to add up a truncated list.
        "totals_over_all_matches": _totals(filtered),
        "records_sorted_by": "absolute qty_diff, descending",
        "records": [_to_record(row) for _, row in page.iterrows()],
    }


# Ranking by size alone puts a missing delivery and an unrequested one in the
# same list. Both are large deviations; only one of them is a shortfall.
TOP_DIRECTIONS = ["shortfall", "surplus", "any"]


@_tool
def top_discrepancies(
    by: str = "qty_diff", limit: int = 5, direction: str = "any"
) -> dict:
    """Rank the biggest deviations from the plan, optionally in one direction.

    The direction argument exists because of a measurement. Without it, asking
    for the biggest shortfalls returned a ranking by absolute size, whose top
    entry was a 300-unit surplus. The model noticed and rebuilt the ranking out
    of other tools - correctly, but in five to seven calls, by a different
    route on each run. Improvisation is what a model does when the schema
    cannot express the question.
    """
    if by not in {"qty_diff", "qty_diff_pct"}:
        return {
            "error": f"Unknown ranking column '{by}'.",
            "valid_values": ["qty_diff", "qty_diff_pct"],
        }
    if direction not in TOP_DIRECTIONS:
        return {
            "error": f"Unknown direction '{direction}'.",
            "valid_values": TOP_DIRECTIONS,
        }

    comparison = _require_comparison()
    limit = max(1, min(int(limit), MAX_RECORDS_RETURNED))

    ranked = comparison[comparison["qty_diff"].fillna(0) != 0]
    if direction == "shortfall":
        ranked = ranked[ranked["qty_diff"] < 0]
    elif direction == "surplus":
        ranked = ranked[ranked["qty_diff"] > 0]

    excluded_without_baseline = 0
    if by == "qty_diff_pct":
        # A line that was never planned has no baseline, so a percentage of the
        # plan is undefined for it. Saying so beats silently dropping it.
        excluded_without_baseline = int(ranked["qty_diff_pct"].isna().sum())
        ranked = ranked[ranked["qty_diff_pct"].notna()]

    ranked = ranked.reindex(
        ranked[by].abs().sort_values(ascending=False, kind="stable").index
    ).head(limit)

    return {
        "ranked_by": f"absolute {by}, descending",
        "direction": direction,
        "limit": limit,
        "returned": len(ranked),
        "excluded_without_baseline": excluded_without_baseline,
        # Signed values, so the model can tell a shortfall from a surplus.
        "records": [_to_record(row) for _, row in ranked.iterrows()],
    }


# The dimensions worth grouping on, and the columns worth ranking the groups by.
GROUP_DIMENSIONS = ["customer", "sku", "status"]
GROUP_SORT_KEYS = [
    "under_delivered_qty",
    "over_delivered_qty",
    "net_qty_diff",
    "discrepancy_lines",
]


@_tool
def group_by(
    dimension: str = "customer",
    sort_by: str = "under_delivered_qty",
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> dict:
    """Aggregate the comparison per customer, SKU or status, worst group first.

    This tool exists because of something visible in a --verbose trace: without
    it, "which customer is worst" was answered by calling filter_records once
    per customer, which cost one round trip per customer and ran into the
    iteration limit. The model was compensating for a hole in the tool API. The
    fix belongs here rather than in the prompt.
    """
    if dimension not in GROUP_DIMENSIONS:
        return {
            "error": f"Cannot group by '{dimension}'.",
            "valid_dimensions": GROUP_DIMENSIONS,
        }
    if sort_by not in GROUP_SORT_KEYS:
        return {"error": f"Cannot sort by '{sort_by}'.", "valid_values": GROUP_SORT_KEYS}

    frame = _require_comparison()
    applied: dict[str, str] = {}

    if date_from or date_to:
        try:
            boundaries = _parse_date_range(date_from, date_to)
        except ValueError as exc:
            return {"error": str(exc)}

        dates = _effective_dates(frame)
        if boundaries[0] is not None:
            frame = frame[dates >= boundaries[0]]
            applied["date_from"] = date_from
            dates = _effective_dates(frame)
        if boundaries[1] is not None:
            frame = frame[dates <= boundaries[1]]
            applied["date_to"] = date_to

    groups = []
    for key, part in frame.groupby(dimension, dropna=False):
        groups.append(
            {
                "group": str(key) if key else "(unknown)",
                "order_lines": len(part),
                "discrepancy_lines": int((part["status"] != STATUS_MATCH).sum()),
                **_totals(part),
            }
        )

    groups.sort(key=lambda group: abs(group[sort_by]), reverse=True)
    limit = max(1, min(int(limit), MAX_RECORDS_RETURNED))

    return {
        "dimension": dimension,
        "sorted_by": f"{sort_by}, largest first",
        "filters_applied": applied,
        "groups_total": len(groups),
        "returned": min(len(groups), limit),
        "groups": groups[:limit],
    }


@_tool
def generate_report(output_path: str | None = None) -> dict:
    """Write the Excel report to disk and report where it landed."""
    comparison = _require_comparison()
    path = write_report(
        comparison, summarise(comparison), output_path or DEFAULT_REPORT_PATH
    )
    discrepancies = int((comparison["status"] != STATUS_MATCH).sum())

    return {
        "path": str(path.resolve()),
        "sheets": ["Summary", "Discrepancies", "Full comparison"],
        "order_lines": len(comparison),
        "discrepancy_lines": discrepancies,
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _load_comparison() -> pd.DataFrame:
    """Run the core layer and cache the result for the rest of the session."""
    planned = load_planned_orders(_state["planned_path"])
    actual = load_actual_orders(_state["actual_path"])
    comparison = reconcile(planned, actual)
    _state["comparison"] = comparison
    return comparison


def _require_comparison() -> pd.DataFrame:
    """Return the cached comparison, loading it first if the model skipped that.

    The system prompt tells the model to call load_and_compare() first, but a
    tool that only works in the right order is a tool that will eventually fail
    in front of a user.
    """
    if _state["comparison"] is None:
        return _load_comparison()
    return _state["comparison"]


def _effective_dates(frame: pd.DataFrame) -> pd.Series:
    """The date a line is filtered on: when it happened, else when it was due.

    Filtering is on the actual delivery date. Lines that were never delivered
    have none, and they are exactly the lines a question about a month cares
    about most, so they fall back to the planned date instead of dropping out.
    """
    return frame["actual_date"].fillna(frame["planned_date"])


def _parse_date_range(
    date_from: str | None, date_to: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Parse the two boundaries, refusing anything that is not YYYY-MM-DD."""
    boundaries = []
    for label, value in (("date_from", date_from), ("date_to", date_to)):
        if not value:
            boundaries.append(None)
            continue
        parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
        if pd.isna(parsed):
            raise ValueError(
                f"{label}='{value}' is not a valid date. Use the YYYY-MM-DD format."
            )
        boundaries.append(parsed)

    if boundaries[0] is not None and boundaries[1] is not None:
        if boundaries[0] > boundaries[1]:
            raise ValueError(
                f"date_from='{date_from}' is later than date_to='{date_to}'."
            )
    return boundaries[0], boundaries[1]


def _rank_by_impact(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort so that the rows lost to truncation are the least interesting ones."""
    order = frame["qty_diff"].abs().sort_values(ascending=False, kind="stable").index
    return frame.reindex(order)


def _totals(frame: pd.DataFrame) -> dict:
    """Aggregate a filtered set - computed in pandas, never left to the model."""
    qty_diff = frame["qty_diff"].astype("Float64")
    shortfall = qty_diff[qty_diff < 0].sum()
    surplus = qty_diff[qty_diff > 0].sum()
    counts = frame["status"].value_counts().to_dict()

    return {
        "planned_qty": int(frame["planned_qty"].fillna(0).sum()),
        "actual_qty": int(frame["actual_qty"].fillna(0).sum()),
        "net_qty_diff": int(qty_diff.fillna(0).sum()),
        "under_delivered_qty": int(abs(shortfall)) if pd.notna(shortfall) else 0,
        "over_delivered_qty": int(surplus) if pd.notna(surplus) else 0,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
    }


def _to_record(row: pd.Series) -> dict:
    """Serialise one order line, dropping empty fields.

    An absent key reads as "there is no such value"; a null invites the model to
    treat it as zero. It also keeps the payload small.
    """
    record = {
        "order_id": row["order_id"],
        "customer": row["customer"],
        "sku": row["sku"],
        "planned_qty": _number(row["planned_qty"]),
        "actual_qty": _number(row["actual_qty"]),
        "qty_diff": _number(row["qty_diff"]),
        "qty_diff_pct": _number(row["qty_diff_pct"]),
        "planned_date": _date(row["planned_date"]),
        "actual_date": _date(row["actual_date"]),
        "date_diff_days": _number(row["date_diff_days"]),
        "status": row["status"],
        "flags": row["flags"] or None,
    }
    return {key: value for key, value in record.items() if value is not None}


def _number(value: Any) -> int | float | None:
    """Turn a pandas scalar into a plain JSON number."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _date(value: Any) -> str | None:
    """Turn a pandas timestamp into a plain YYYY-MM-DD string."""
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Schemas handed to the model
# --------------------------------------------------------------------------
#
# The descriptions below are the only thing the model sees when it decides what
# to call, so each one says when to use the tool, not merely what it does. The
# two that are easiest to confuse are filter_records ("give me the lines that
# satisfy X") and top_discrepancies ("give me the worst N"); their descriptions
# name each other so the boundary is explicit.

TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [
    types.FunctionDeclaration(
        name="load_and_compare",
        description=(
            "Load the planned and actual order files and reconcile them. Call this "
            "first, before any other tool. Returns the number of order lines, the "
            "exact customer names, the date range covered and the list of valid "
            "status values - use those spellings in later calls."
        ),
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="get_summary",
        description=(
            "Aggregate figures for the entire comparison: how many order lines "
            "match the plan, how many differ and how they break down by status, "
            "total planned versus actual units, and the total under- and "
            "over-delivered quantity. Use this for questions about the overall "
            "picture. It covers all data and takes no filters, so for a single "
            "customer, status or period use filter_records instead."
        ),
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="filter_records",
        description=(
            "Find the order lines matching a customer, a status and/or a date "
            "range. Every argument is optional and they combine with AND. Returns "
            "totals computed over every matching line, plus at most 20 example "
            "records ordered by the size of the quantity difference - so use the "
            "totals for counting and summing, and the records only as examples. "
            "Use this when the question narrows the data; use top_discrepancies "
            "when the question asks for the largest deviations."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "customer": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Customer name or a fragment of it, case-insensitive, "
                        "e.g. 'Velo' matches 'Velo Parts Ltd'."
                    ),
                ),
                "status": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Keep only lines with this status. MATCH: plan and "
                        "delivery agree. QTY_MISMATCH: a different quantity was "
                        "delivered. DATE_MISMATCH: the right quantity on the "
                        "wrong date. MISSING_ACTUAL: planned but never delivered. "
                        "UNPLANNED: delivered although never planned."
                    ),
                    enum=list(ALL_STATUSES),
                ),
                "date_from": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Earliest date to include, YYYY-MM-DD. Filters on the "
                        "actual delivery date, falling back to the planned date "
                        "for lines that were never delivered."
                    ),
                ),
                "date_to": types.Schema(
                    type=types.Type.STRING,
                    description="Latest date to include, YYYY-MM-DD, inclusive.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="top_discrepancies",
        description=(
            "Rank the order lines with the largest deviation from the plan and "
            "return the worst ones. Use this for 'the biggest', 'the worst' or "
            "'top N' questions, and set direction to match what was asked: a "
            "question about shortfalls, missing units or under-delivery wants "
            "direction='shortfall'. This matters - with the default the two "
            "kinds compete on size alone, so a large over-delivery can outrank "
            "every shortfall in the list. Cannot be filtered by customer or "
            "period - for that use filter_records."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "by": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "'qty_diff' ranks by the number of units missing or "
                        "surplus - use it for volume. 'qty_diff_pct' ranks by the "
                        "deviation relative to the plan - use it to find the "
                        "worst-served orders regardless of their size."
                    ),
                    enum=["qty_diff", "qty_diff_pct"],
                ),
                "direction": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "'shortfall' keeps only lines where fewer units arrived "
                        "than were planned - use it for shortfalls, missing "
                        "units, under-delivery, or 'who let us down'. 'surplus' "
                        "keeps only over-deliveries. 'any' ranks both together "
                        "by size and is the default; use it only when the "
                        "question really is about deviation in either "
                        "direction."
                    ),
                    enum=list(TOP_DIRECTIONS),
                ),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description="How many lines to return, 1 to 20. Defaults to 5.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="group_by",
        description=(
            "Aggregate every order line into groups - per customer, per SKU or "
            "per status - and return the groups ranked worst first, each with "
            "its own totals. Use this for any question about which customer, "
            "product or category is the worst, the best, or how they compare: "
            "one call covers all of them, so never call filter_records once per "
            "customer to work this out. Optionally restricted to a date range."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "dimension": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "What to group by. 'customer' for who is worst served, "
                        "'sku' for which product goes wrong, 'status' for how "
                        "the discrepancies break down. Defaults to 'customer'."
                    ),
                    enum=list(GROUP_DIMENSIONS),
                ),
                "sort_by": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Which figure decides the ranking. "
                        "'under_delivered_qty' for missing units (the usual "
                        "meaning of worst), 'over_delivered_qty' for surplus, "
                        "'net_qty_diff' for the balance of the two, "
                        "'discrepancy_lines' for how many lines went wrong "
                        "regardless of size. Defaults to under_delivered_qty."
                    ),
                    enum=list(GROUP_SORT_KEYS),
                ),
                "date_from": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Earliest date to include, YYYY-MM-DD. Same date rule as "
                        "filter_records: the actual delivery date, falling back "
                        "to the planned date for lines never delivered."
                    ),
                ),
                "date_to": types.Schema(
                    type=types.Type.STRING,
                    description="Latest date to include, YYYY-MM-DD, inclusive.",
                ),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description="How many groups to return, 1 to 20. Defaults to 10.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="generate_report",
        description=(
            "Write the full comparison to an Excel file with three sheets: "
            "Summary, Discrepancies and Full comparison. Use this only when the "
            "user asks for a report, a file or an export - it does not answer "
            "questions about the data. Returns the path the file was saved to."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "output_path": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Destination path ending in .xlsx. Defaults to "
                        "reconciliation_report.xlsx in the current directory."
                    ),
                ),
            },
        ),
    ),
]

TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict]] = {
    "load_and_compare": load_and_compare,
    "get_summary": get_summary,
    "filter_records": filter_records,
    "top_discrepancies": top_discrepancies,
    "group_by": group_by,
    "generate_report": generate_report,
}


def execute_tool(name: str, arguments: dict | None) -> dict:
    """Dispatch a function call from the model to its implementation.

    A hallucinated tool name is answered, not raised: the model gets the list of
    tools it actually has and can pick again.
    """
    implementation = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        return {
            "error": f"There is no tool called '{name}'.",
            "available_tools": sorted(TOOL_IMPLEMENTATIONS),
        }
    return implementation(**(arguments or {}))
