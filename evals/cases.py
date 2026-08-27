"""The evaluation set: questions paired with what a good answer looks like.

The test suite checks what the tools do. This checks something the test suite
cannot: whether the model picks the right one. Those are different failures. A
tool can be perfectly correct and still never be chosen, and a description edit
that reads better can quietly make the model choose worse - with nothing going
red anywhere.

Three things are scored, because "did it answer correctly" hides all three:

  selection  - was the right tool reached for at all
  efficiency - how many calls it took to get there
  grounding  - did the real figures end up in the answer

A case is deliberately not asserted as an exact sequence of calls. The model is
allowed to think differently on different days, and an eval that fails when it
does is an eval nobody keeps. What is asserted is the part that would be wrong
rather than merely different.

Every expected figure below is checked against the sample data by
tests/test_evals.py, so a number here cannot quietly stop being true.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    """One question and the conditions a good answer to it has to meet."""

    id: str
    question: str
    # What this case is really testing, printed in the scorecard.
    tests: str

    # Selection.
    expect_tools: list[str] = field(default_factory=list)
    expect_any_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)

    # Efficiency. Counted across the whole answer, load_and_compare included.
    max_calls: int = 8

    # Grounding. Matched case-insensitively against the answer text.
    expect_in_answer: list[str] = field(default_factory=list)
    expect_any_in_answer: list[str] = field(default_factory=list)
    expect_not_in_answer: list[str] = field(default_factory=list)
    # For conditions a list of phrases cannot express. There are only so many
    # ways to name a figure, so substrings work for those; there are endless
    # ways to say "no", and a case that keeps tripping over which one the model
    # picked is testing vocabulary rather than behaviour.
    expect_answer_matches: list[str] = field(default_factory=list)

    # Set when the case documents a gap that is known and not yet fixed. It is
    # still run and still reported - it just does not fail the suite, because
    # a red build for something already written down in the roadmap teaches
    # nobody anything. The point is to have the measurement ready for the day
    # the gap is closed.
    known_gap: str | None = None


CASES: list[EvalCase] = [
    EvalCase(
        id="overview",
        question="how many order lines have a discrepancy?",
        tests="A whole-dataset question should reach for the aggregate tool "
        "rather than pulling rows and counting them.",
        expect_tools=["get_summary"],
        forbid_tools=["filter_records"],
        max_calls=3,
        expect_in_answer=["31", "17"],
    ),
    EvalCase(
        id="worst_customer",
        question="which customer has the worst discrepancies?",
        tests="The case that produced group_by. Before it existed this took "
        "seven calls, one per customer, and still ran out of iterations.",
        expect_tools=["group_by"],
        max_calls=3,
        expect_in_answer=["Alpine Gear Co", "332"],
    ),
    EvalCase(
        id="customers_in_month",
        question="which customers under-delivered in September?",
        tests="Grouping and a date range in one question. The never-delivered "
        "order has no delivery date and must still be counted in September.",
        expect_any_tools=["group_by", "filter_records"],
        max_calls=4,
        expect_in_answer=["Alpine Gear Co", "Crankset Supply Inc", "Summit Bike Supply"],
    ),
    EvalCase(
        id="biggest_shortfalls",
        question="what are the 5 biggest shortfalls?",
        tests="Ranking by shortfall specifically. This case spent five to "
        "seven calls, varying per run, until top_discrepancies gained a "
        "direction argument; the budget of three is what it should now cost.",
        expect_any_tools=["top_discrepancies"],
        max_calls=3,
        expect_in_answer=["280", "75", "60", "45", "40"],
    ),
    EvalCase(
        id="shortfalls_in_month",
        question="what were the biggest shortfalls in September?",
        tests="A ranking and a date range in one question. Measured before "
        "top_discrepancies took a date range, this cost three calls: one for "
        "the month, one for the ranking, and an intersection performed in the "
        "model's head. It got that right every run, which was the argument for "
        "the fix rather than against it - correctness should not depend on the "
        "model being careful. The budget of two is the version that cannot be "
        "got wrong.",
        expect_tools=["top_discrepancies"],
        max_calls=2,
        expect_in_answer=["280", "60", "45"],
        # July shortfalls. Their presence would mean the period was ignored,
        # which is the failure that matters here - the figures would all be
        # real, just answering a different question than the one asked.
        expect_not_in_answer=["ORD-1007", "ORD-1010"],
    ),
    EvalCase(
        id="unplanned",
        question="did anything arrive that nobody ordered?",
        tests="A status the user names in their own words rather than in the "
        "vocabulary of the data. UNPLANNED never appears in the question.",
        expect_any_tools=["filter_records", "group_by"],
        max_calls=4,
        expect_in_answer=["ORD-1090", "300"],
    ),
    EvalCase(
        id="report",
        question="save the report to eval-report.xlsx",
        tests="An instruction rather than a question. Writing the file is the "
        "answer; describing the data instead would be wrong.",
        expect_tools=["generate_report"],
        max_calls=3,
        expect_in_answer=["eval-report.xlsx"],
    ),
    EvalCase(
        id="unknown_customer",
        question="what happened with orders from Acme Corp?",
        tests="A customer that is not in the data. The failure to catch here "
        "is a confident answer about a company that does not exist. This case "
        "was first written as a list of phrases meaning 'no', and it was wrong "
        "three times: it missed correct answers worded differently, and once "
        "passed on a sentence about pricing that happened to contain one. The "
        "regex asks the question properly - is there a negation in the same "
        "sentence as the company - and forbidding an order id catches the "
        "invented specifics, since the only way one appears here is invention.",
        max_calls=4,
        expect_in_answer=["Acme"],
        # A denial either side of the name: "Acme Corp is not in the dataset",
        # "we have no orders for Acme Corp". Sentence-bounded, so a negation
        # about something else further down the answer cannot satisfy it.
        expect_answer_matches=[
            r"acme[^.!?]*\b(?:not|no|never|nothing)\b"
            r"|\b(?:not|no|never|nothing)\b[^.!?]*acme"
        ],
        expect_not_in_answer=["ORD-"],
    ),
    EvalCase(
        id="out_of_scope",
        question="how much did these orders cost in total?",
        tests="A field that does not exist. Saying so is the correct answer; "
        "a plausible total is the expensive failure.",
        max_calls=4,
        expect_any_in_answer=[
            "no price",
            "no pricing",
            "not include",
            "does not contain",
            "no cost",
            "not available",
            "cannot",
        ],
        expect_not_in_answer=["$", "USD", "EUR"],
    ),
    EvalCase(
        id="self_description",
        question="what is this and what can you tell me?",
        tests="A first-time user with no idea what they are looking at. The "
        "answer should name the real data, not describe a hypothetical one.",
        max_calls=3,
        expect_any_in_answer=["planned", "plan"],
        expect_in_answer=["Velo Parts Ltd"],
    ),
]
