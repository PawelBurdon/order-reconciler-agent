"""Loading and validation of the two input CSV files.

Part of the deterministic core layer: no AI, no network calls.

Everything is read as text first and converted explicitly afterwards. That is
slightly more code than letting pandas guess the dtypes, but it buys two
things: a quantity that is silently parsed as text can never reach the
comparison, and every rejected value can be reported with its real CSV line
number instead of a stack trace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PLANNED_COLUMNS = ["order_id", "customer", "sku", "planned_qty", "planned_date"]
ACTUAL_COLUMNS = ["order_id", "customer", "sku", "actual_qty", "actual_date"]

# An order line is identified by the order and the product, not by the order
# alone: a single order_id may legitimately contain several SKUs.
KEY_COLUMNS = ["order_id", "sku"]

DATE_FORMAT = "%Y-%m-%d"


class DataValidationError(Exception):
    """Raised when an input file is missing, malformed or unusable.

    Carries a message that is meant to be shown to a human (or handed to the
    agent as a tool error), so it always says which file and which line failed.
    """


def load_planned_orders(path: str | Path) -> pd.DataFrame:
    """Load planned_orders.csv into a validated DataFrame."""
    return _load_orders(path, PLANNED_COLUMNS, "planned_qty", "planned_date")


def load_actual_orders(path: str | Path) -> pd.DataFrame:
    """Load actual_orders.csv into a validated DataFrame."""
    return _load_orders(path, ACTUAL_COLUMNS, "actual_qty", "actual_date")


def _load_orders(
    path: str | Path,
    expected_columns: list[str],
    qty_column: str,
    date_column: str,
) -> pd.DataFrame:
    """Read one order file and return it with correct dtypes.

    Returned dtypes: string columns as ``str``, the quantity as nullable
    ``Int64`` and the date as ``datetime64[ns]`` (``NaT`` when the cell was
    empty - an empty date is tolerated, an unparseable one is not).
    """
    path = Path(path)

    if not path.exists():
        raise DataValidationError(
            f"Input file not found: {path}. "
            f"Expected a CSV with columns: {', '.join(expected_columns)}."
        )

    try:
        # dtype=str + keep_default_na=False means every cell arrives as a
        # string and an empty cell arrives as "", never as a float NaN. All
        # type conversion below is therefore explicit and checkable.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Input file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse {path} as CSV: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(frame.columns)}."
        )

    frame = frame[expected_columns].copy()

    for column in ["order_id", "customer", "sku"]:
        frame[column] = frame[column].str.strip()

    _reject_empty_values(frame, path, ["order_id", "sku", qty_column])
    frame[qty_column] = _to_integer(frame[qty_column], path, qty_column)
    frame[date_column] = _to_date(frame[date_column], path, date_column)

    if frame.empty:
        raise DataValidationError(f"{path} contains a header but no data rows.")

    return frame


def _reject_empty_values(
    frame: pd.DataFrame, path: Path, required_columns: list[str]
) -> None:
    """Fail if a column that the comparison depends on has blank cells.

    ``customer`` and the date columns are deliberately not required: a missing
    customer name or a missing delivery date is a real-world data-quality
    problem that the reconciler should surface, not a reason to refuse the file.
    """
    for column in required_columns:
        blank_rows = frame.index[frame[column].astype(str).str.strip() == ""]
        if len(blank_rows) > 0:
            raise DataValidationError(
                f"{path}: column '{column}' is empty in line(s) "
                f"{_as_csv_lines(blank_rows)}. This column is required."
            )


def _to_integer(values: pd.Series, path: Path, column: str) -> pd.Series:
    """Convert a text column to nullable integers, refusing junk values."""
    numbers = pd.to_numeric(values, errors="coerce")

    broken_rows = values.index[numbers.isna() & (values.str.strip() != "")]
    if len(broken_rows) > 0:
        examples = ", ".join(repr(v) for v in values[broken_rows][:3])
        raise DataValidationError(
            f"{path}: column '{column}' contains non-numeric value(s) in line(s) "
            f"{_as_csv_lines(broken_rows)}: {examples}."
        )

    non_integer_rows = numbers.index[numbers.notna() & (numbers % 1 != 0)]
    if len(non_integer_rows) > 0:
        raise DataValidationError(
            f"{path}: column '{column}' must contain whole units, but line(s) "
            f"{_as_csv_lines(non_integer_rows)} contain fractional values."
        )

    return numbers.astype("Int64")


def _to_date(values: pd.Series, path: Path, column: str) -> pd.Series:
    """Convert a text column to dates; empty stays empty, junk is rejected."""
    dates = pd.to_datetime(values, format=DATE_FORMAT, errors="coerce")

    broken_rows = values.index[dates.isna() & (values.str.strip() != "")]
    if len(broken_rows) > 0:
        examples = ", ".join(repr(v) for v in values[broken_rows][:3])
        raise DataValidationError(
            f"{path}: column '{column}' must use the {DATE_FORMAT} format, but "
            f"line(s) {_as_csv_lines(broken_rows)} contain: {examples}."
        )

    return dates


def _as_csv_lines(index: pd.Index) -> str:
    """Translate DataFrame positions into CSV line numbers a human can find.

    Row 0 of the DataFrame is line 2 of the file, because line 1 is the header.
    """
    lines = [str(position + 2) for position in index[:5]]
    if len(index) > 5:
        lines.append(f"... and {len(index) - 5} more")
    return ", ".join(lines)
