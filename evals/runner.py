"""Runs the evaluation set against the real model and prints a scorecard.

    python -m evals.runner
    python -m evals.runner --case worst_customer --verbose

This is the only part of the project that needs an API key to prove anything,
because the thing under test is the model's judgement. It is therefore not a
pytest test: it costs money, it is not deterministic, and a suite that
sometimes fails because a rate limit was hit is a suite people learn to ignore.
The checks that can be made without the model live in tests/test_evals.py and
run on every push.

Exit code 1 means a case failed. Cases marked with a known gap are reported but
do not fail the run - see evals/cases.py for why.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import errors

from src.agent.agent import DEFAULT_MODEL, MissingApiKeyError, ReconciliationAgent
from src.agent.tools import configure_data_sources

from .cases import CASES, EvalCase

PLANNED_PATH = Path("sample_data/planned_orders.csv")
ACTUAL_PATH = Path("sample_data/actual_orders.csv")

# The free tier allows a handful of requests per minute and one case can spend
# several. Waiting between cases costs a couple of minutes; not waiting costs
# the whole run.
SECONDS_BETWEEN_CASES = 4.0
SECONDS_AFTER_RATE_LIMIT = 65.0
RATE_LIMIT_RETRIES = 3

EXIT_FAILED = 1
EXIT_CONFIG_ERROR = 2

# Which dimension each kind of check belongs to, so the scorecard can say
# whether the model chose badly or answered badly - they need different fixes.
SELECTION = "selection"
EFFICIENCY = "efficiency"
GROUNDING = "grounding"


class Check:
    """One assertion about one answer."""

    def __init__(self, dimension: str, passed: bool, detail: str):
        self.dimension = dimension
        self.passed = passed
        self.detail = detail


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    arguments = _build_parser().parse_args(argv)

    cases = CASES
    if arguments.case:
        cases = [entry for entry in CASES if entry.id in arguments.case]
        if not cases:
            print(f"No case matches {arguments.case}.", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    configure_data_sources(PLANNED_PATH, ACTUAL_PATH)

    try:
        agent = ReconciliationAgent(
            model=arguments.model or DEFAULT_MODEL, verbose=arguments.verbose
        )
    except MissingApiKeyError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(f"Evaluating {len(cases)} cases against {agent.model}\n")

    results = []
    for position, entry in enumerate(cases):
        if position:
            time.sleep(SECONDS_BETWEEN_CASES)
        results.append(_run_case(agent, entry, arguments.verbose))

    _clean_up_generated_files()
    return _report(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals.runner",
        description="Score the model's tool selection against the evaluation set.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this case id. Repeatable.",
    )
    parser.add_argument(
        "--model", default=None, help="Model to evaluate. Defaults to the agent's."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every tool call and the full answer for each case.",
    )
    return parser


def _run_case(agent: ReconciliationAgent, entry: EvalCase, verbose: bool) -> dict:
    """Ask one question and score the answer, retrying through rate limits."""
    agent.reset()

    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            answer = agent.ask(entry.question)
            break
        except errors.APIError as error:
            if getattr(error, "code", None) != 429 or attempt == RATE_LIMIT_RETRIES:
                # Anything other than a rate limit is a real problem, and so is
                # a rate limit that will not clear. Either way the case has no
                # result, which is reported rather than counted as a failure of
                # the model.
                return {
                    "case": entry,
                    "error": f"{getattr(error, 'code', '?')}: "
                    f"{getattr(error, 'message', error)}",
                    "checks": [],
                    "calls": [],
                }
            print(f"  rate limited, waiting {SECONDS_AFTER_RATE_LIMIT:.0f}s")
            time.sleep(SECONDS_AFTER_RATE_LIMIT)

    calls = list(agent.calls)
    checks = _score(entry, calls, answer)

    if verbose:
        print(f"  answer: {answer}\n")

    return {"case": entry, "error": None, "checks": checks, "calls": calls, "answer": answer}


def _score(entry: EvalCase, calls: list[dict], answer: str) -> list[Check]:
    """Turn one answer into a list of pass/fail checks."""
    called = [call["name"] for call in calls]
    haystack = answer.lower()
    checks: list[Check] = []

    for name in entry.expect_tools:
        checks.append(
            Check(SELECTION, name in called, f"expected a call to {name}")
        )

    if entry.expect_any_tools:
        hit = [name for name in entry.expect_any_tools if name in called]
        checks.append(
            Check(
                SELECTION,
                bool(hit),
                f"expected one of {', '.join(entry.expect_any_tools)}",
            )
        )

    for name in entry.forbid_tools:
        checks.append(
            Check(SELECTION, name not in called, f"expected no call to {name}")
        )

    checks.append(
        Check(
            EFFICIENCY,
            len(calls) <= entry.max_calls,
            f"{len(calls)} calls, at most {entry.max_calls} allowed",
        )
    )

    for expected in entry.expect_in_answer:
        checks.append(
            Check(GROUNDING, expected.lower() in haystack, f"answer contains {expected!r}")
        )

    if entry.expect_any_in_answer:
        hit = [
            expected
            for expected in entry.expect_any_in_answer
            if expected.lower() in haystack
        ]
        checks.append(
            Check(GROUNDING, bool(hit), "answer says the data cannot answer this")
        )

    for pattern in entry.expect_answer_matches:
        checks.append(
            Check(
                GROUNDING,
                re.search(pattern, answer, re.IGNORECASE | re.DOTALL) is not None,
                f"answer matches /{pattern}/",
            )
        )

    for forbidden in entry.expect_not_in_answer:
        checks.append(
            Check(
                GROUNDING,
                forbidden.lower() not in haystack,
                f"answer does not contain {forbidden!r}",
            )
        )

    return checks


def _report(results: list[dict]) -> int:
    """Print the scorecard and decide the exit code."""
    failed = 0
    gaps = 0
    errored = 0
    by_dimension: dict[str, list[int]] = {
        SELECTION: [0, 0],
        EFFICIENCY: [0, 0],
        GROUNDING: [0, 0],
    }

    for result in results:
        entry: EvalCase = result["case"]
        checks: list[Check] = result["checks"]

        for check in checks:
            by_dimension[check.dimension][1] += 1
            by_dimension[check.dimension][0] += int(check.passed)

        if result["error"]:
            errored += 1
            print(f"{entry.id:<22} ERROR  {result['error']}")
            continue

        broken = [check for check in checks if not check.passed]
        passed = len(checks) - len(broken)

        if not broken:
            status = "pass"
        elif entry.known_gap:
            status = "GAP"
            gaps += 1
        else:
            status = "FAIL"
            failed += 1

        print(
            f"{entry.id:<22} {status:<6} {passed}/{len(checks)} checks   "
            f"{len(result['calls'])} calls"
        )
        for check in broken:
            print(f"    {check.dimension}: {check.detail}")
        if broken and entry.known_gap:
            print(f"    known gap: {entry.known_gap}")

        # A failure nobody can diagnose is a failure that gets ignored. In CI
        # there is no way to re-run this by hand with --verbose, so the answer
        # and the calls that produced it are printed with the failure. The
        # first question about a red eval is whether the model was wrong or
        # the assertion was, and that cannot be answered without them.
        if broken:
            for call in result["calls"]:
                arguments = ", ".join(
                    f"{key}={value!r}" for key, value in call["arguments"].items()
                )
                print(f"      -> {call['name']}({arguments})")
            print(f'      answer: {result["answer"]}')

    total = len(results)
    print(
        f"\n{total - failed - errored}/{total} cases passed"
        f"{f', {gaps} known gap(s)' if gaps else ''}"
        f"{f', {errored} could not run' if errored else ''}."
    )
    for dimension, (good, seen) in by_dimension.items():
        if seen:
            print(f"  {dimension:<11} {good}/{seen}")

    return EXIT_FAILED if failed or errored else 0


def _clean_up_generated_files() -> None:
    """The report case really does write a file. Do not leave it lying around."""
    generated = Path("eval-report.xlsx")
    if generated.exists():
        generated.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
