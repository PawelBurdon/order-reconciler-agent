"""Checks on the evaluation set itself. No model, no key, no network.

An eval set is a fixture of hand-written expectations, and hand-written
expectations rot. A number that was true of the sample data in August is not
automatically true in September, and a case that expects a tool renamed last
week fails for a reason that has nothing to do with the model.

So the fixture is checked twice over: that it is structurally valid against the
real schemas, and that every figure it expects to see in an answer is a figure
the deterministic layer actually produces. If someone edits the sample data,
these fail here rather than looking like the model got worse.
"""

import re

import pytest

from evals.cases import CASES, EvalCase
from src.agent import tools
from src.agent.tools import TOOL_IMPLEMENTATIONS
from src.core.loader import load_actual_orders, load_planned_orders
from src.core.reconciler import STATUS_UNPLANNED, reconcile, summarise

PLANNED_PATH = "sample_data/planned_orders.csv"
ACTUAL_PATH = "sample_data/actual_orders.csv"


def case(case_id: str) -> EvalCase:
    for entry in CASES:
        if entry.id == case_id:
            return entry
    raise AssertionError(f"No eval case with id {case_id!r}")


@pytest.fixture(scope="module")
def comparison():
    return reconcile(load_planned_orders(PLANNED_PATH), load_actual_orders(ACTUAL_PATH))


@pytest.fixture
def loaded_tools(comparison):
    tools.set_comparison(comparison)
    yield
    tools.set_comparison(None)


# -- the fixture is structurally sound -------------------------------------


def test_case_ids_are_unique():
    identifiers = [entry.id for entry in CASES]
    assert len(identifiers) == len(set(identifiers))


@pytest.mark.parametrize("entry", CASES, ids=lambda entry: entry.id)
def test_every_named_tool_exists(entry: EvalCase):
    """A renamed tool must break the eval here, not look like a bad answer."""
    named = entry.expect_tools + entry.expect_any_tools + entry.forbid_tools
    unknown = [name for name in named if name not in TOOL_IMPLEMENTATIONS]
    assert not unknown, f"{entry.id} names tools that do not exist: {unknown}"


@pytest.mark.parametrize("entry", CASES, ids=lambda entry: entry.id)
def test_every_case_asserts_something(entry: EvalCase):
    assert entry.question.strip()
    assert entry.tests.strip()
    assert entry.max_calls >= 1
    assert (
        entry.expect_tools
        or entry.expect_any_tools
        or entry.forbid_tools
        or entry.expect_in_answer
        or entry.expect_any_in_answer
        or entry.expect_not_in_answer
        or entry.expect_answer_matches
    ), f"{entry.id} would pass no matter what the model did"


@pytest.mark.parametrize("entry", CASES, ids=lambda entry: entry.id)
def test_every_pattern_compiles(entry: EvalCase):
    """A broken regex must fail here, not halfway through a billed eval run."""
    for pattern in entry.expect_answer_matches:
        re.compile(pattern)


def test_the_unknown_customer_pattern_reads_denials_and_rejects_inventions():
    """The regex is the assertion, so it gets its own test.

    Three of these are real answers the model has given; the last is the
    fabrication the case exists to catch.
    """
    pattern = case("unknown_customer").expect_answer_matches[0]

    accepted = [
        'I can help you analyze the data, but "Acme Corp" is not in the dataset.',
        "We do not have any orders or records for Acme Corp in the system.",
        "The data does not contain any orders from Acme Corp.",
    ]
    rejected = [
        "Acme Corp under-delivered by 40 units across two order lines.",
        # A denial about something else must not satisfy a denial about Acme.
        "Acme Corp ordered 120 units. The data does not include pricing.",
    ]

    for answer in accepted:
        assert re.search(pattern, answer, re.IGNORECASE | re.DOTALL), answer
    for answer in rejected:
        assert not re.search(pattern, answer, re.IGNORECASE | re.DOTALL), answer


@pytest.mark.parametrize("entry", CASES, ids=lambda entry: entry.id)
def test_a_known_gap_says_what_the_gap_is(entry: EvalCase):
    """An excused failure without a reason is just a disabled test."""
    if entry.known_gap is not None:
        assert entry.known_gap.strip()


# -- the figures the fixture expects are figures the data produces ---------


def test_overview_numbers_are_real(comparison):
    summary = summarise(comparison)
    expected = case("overview").expect_in_answer

    assert str(summary["total_order_lines"]) in expected
    assert str(summary["discrepancy_lines"]) in expected


def test_the_worst_customer_really_is_the_worst(loaded_tools):
    result = tools.group_by(dimension="customer", sort_by="under_delivered_qty")
    worst = result["groups"][0]
    expected = case("worst_customer").expect_in_answer

    assert worst["group"] in expected
    assert str(worst["under_delivered_qty"]) in expected


def test_the_september_customers_are_the_ones_that_fell_short(loaded_tools):
    result = tools.group_by(
        dimension="customer", date_from="2025-09-01", date_to="2025-09-30"
    )
    short = {
        group["group"]
        for group in result["groups"]
        if group["under_delivered_qty"] > 0
    }

    assert short == set(case("customers_in_month").expect_in_answer)


def test_the_five_biggest_shortfalls_are_the_expected_five(loaded_tools):
    result = tools.top_discrepancies(by="qty_diff", limit=20)
    shortfalls = [
        abs(record["qty_diff"])
        for record in result["records"]
        if record["qty_diff"] < 0
    ]

    assert [str(value) for value in shortfalls[:5]] == case(
        "biggest_shortfalls"
    ).expect_in_answer


def test_the_unplanned_delivery_is_the_one_named(comparison):
    unplanned = comparison[comparison["status"] == STATUS_UNPLANNED]
    expected = case("unplanned").expect_in_answer

    assert len(unplanned) == 1
    assert unplanned.iloc[0]["order_id"] in expected
    assert str(unplanned.iloc[0]["actual_qty"]) in expected


def test_the_customer_named_in_the_self_description_case_exists(comparison):
    customers = set(comparison["customer"])

    for expected in case("self_description").expect_in_answer:
        assert expected in customers


def test_the_unknown_customer_really_is_unknown(comparison):
    customers = " ".join(comparison["customer"]).lower()

    for expected in case("unknown_customer").expect_in_answer:
        assert expected.lower() not in customers
