"""Command line entry point.

Three commands, and the split between the last one and the rest is the point of
the project:

    python -m src.main ask "which customers under-delivered in September?"
    python -m src.main chat
    python -m src.main report --output report.xlsx

`report` never imports the agent package. It runs on the core layer alone, so
it works with no API key, no network and no model - the numbers do not depend
on the AI. `ask` puts the same core layer behind a model that decides which
part of it to use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core.loader import DataValidationError, load_actual_orders, load_planned_orders
from .core.reconciler import STATUS_MATCH, reconcile, summarise
from .core.report import write_report

DEFAULT_PLANNED_PATH = Path("sample_data/planned_orders.csv")
DEFAULT_ACTUAL_PATH = Path("sample_data/actual_orders.csv")
DEFAULT_REPORT_PATH = Path("reconciliation_report.xlsx")

EXIT_DATA_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_API_ERROR = 3


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "ask":
            return _run_ask(arguments)
        if arguments.command == "chat":
            return _run_chat(arguments)
        return _run_report(arguments)
    except DataValidationError as error:
        # A broken input file is the user's problem to fix, so it gets a plain
        # message rather than a traceback.
        print(f"Data error: {error}", file=sys.stderr)
        return EXIT_DATA_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="order-reconciler-agent",
        description=(
            "Compare planned orders against actual orders, either by asking a "
            "question in plain English or by exporting an Excel report."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser(
        "ask", help="Ask a natural-language question about the discrepancies."
    )
    ask.add_argument("question", help="The question, in quotes.")
    _add_agent_arguments(ask)
    _add_data_source_arguments(ask)

    chat = subparsers.add_parser(
        "chat", help="Ask questions one after another, with follow-ups."
    )
    _add_agent_arguments(chat)
    _add_data_source_arguments(chat)

    report = subparsers.add_parser(
        "report", help="Write the Excel report. Works without an API key."
    )
    report.add_argument(
        "--output",
        default=DEFAULT_REPORT_PATH,
        type=Path,
        help=f"Where to write the .xlsx file (default: {DEFAULT_REPORT_PATH}).",
    )
    _add_data_source_arguments(report)

    return parser


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every tool the agent calls, with its arguments and result.",
    )
    # No default is spelled out here: agent.py owns the model name, and a copy
    # of it in a help string is a copy that goes stale.
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model to use. Defaults to the model configured in the agent.",
    )


def _days(value: str) -> int:
    """A tolerance argparse can reject itself, with argparse's own message.

    Catching this later would mean a traceback or a hand-rolled error for
    something the command line parser already knows how to complain about.
    """
    days = int(value)
    if days < 0:
        raise argparse.ArgumentTypeError(
            f"a tolerance is a number of days, so it cannot be {days}."
        )
    return days


def _add_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--date-tolerance",
        type=_days,
        default=0,
        metavar="DAYS",
        help=(
            "Days a delivery may move without counting as a date discrepancy "
            "(default: 0, every date must match). In agent mode this is only "
            "the starting value - a question can ask for a different one."
        ),
    )
    parser.add_argument(
        "--planned",
        default=DEFAULT_PLANNED_PATH,
        type=Path,
        help=f"Path to the planned orders CSV (default: {DEFAULT_PLANNED_PATH}).",
    )
    parser.add_argument(
        "--actual",
        default=DEFAULT_ACTUAL_PATH,
        type=Path,
        help=f"Path to the actual orders CSV (default: {DEFAULT_ACTUAL_PATH}).",
    )


def _run_ask(arguments: argparse.Namespace) -> int:
    # Imported here, not at the top of the file, so that `report` never loads
    # the agent package at all.
    from google.genai import errors

    from .agent.agent import DEFAULT_MODEL, MissingApiKeyError, ReconciliationAgent
    from .agent.tools import configure_data_sources, configure_date_tolerance

    configure_data_sources(arguments.planned, arguments.actual)
    configure_date_tolerance(arguments.date_tolerance)

    try:
        agent = ReconciliationAgent(
            model=arguments.model or DEFAULT_MODEL, verbose=arguments.verbose
        )
    except MissingApiKeyError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        answer = agent.ask(arguments.question)
    except errors.APIError as error:
        # A rejected request is not a bug in this program, so it gets the same
        # one-line treatment as a broken input file. The free tier rate limit
        # is the one users actually hit, so it is named explicitly.
        print(f"Gemini API error: {_describe_api_error(error)}", file=sys.stderr)
        return EXIT_API_ERROR
    if arguments.verbose:
        print("\n[answer]")
    print(answer)
    return 0


CHAT_BANNER = """\
Ask about the difference between the orders that were planned and the orders
that actually arrived. Each question can build on the previous one, so "and
what went wrong for them?" works.

Not sure what to ask? Ask the agent - "what is this and what can you tell me?"
is a perfectly good first question, and it will answer with the customers and
the period that are actually in your files.

  /reset   forget the conversation so far
  /exit    quit (Ctrl+C works too)
"""


def _run_chat(arguments: argparse.Namespace) -> int:
    from google.genai import errors

    from .agent.agent import DEFAULT_MODEL, MissingApiKeyError, ReconciliationAgent
    from .agent.tools import configure_data_sources, configure_date_tolerance

    configure_data_sources(arguments.planned, arguments.actual)
    configure_date_tolerance(arguments.date_tolerance)

    try:
        agent = ReconciliationAgent(
            model=arguments.model or DEFAULT_MODEL,
            verbose=arguments.verbose,
            remember=True,
        )
    except MissingApiKeyError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(CHAT_BANNER)

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question in {"/exit", "/quit"}:
            return 0
        if question == "/reset":
            agent.reset()
            print("Context cleared.\n")
            continue

        try:
            print(f"\n{agent.ask(question)}\n")
        except errors.APIError as error:
            # A rejected request ends the question, not the session. Hitting a
            # rate limit three questions in should not throw away the
            # conversation that got you there.
            print(f"Gemini API error: {_describe_api_error(error)}\n", file=sys.stderr)


def _describe_api_error(error: Exception) -> str:
    """Reduce an SDK error object to the sentence a user can act on."""
    code = getattr(error, "code", None)
    message = getattr(error, "message", None) or str(error)

    if code == 429:
        return (
            f"{message.strip()} "
            "The free tier allows a handful of requests per minute; wait and "
            "try again, or use a key with a paid quota."
        )
    if code in (401, 403):
        return f"{message.strip()} Check that GEMINI_API_KEY is valid and enabled."
    return message.strip()


def _run_report(arguments: argparse.Namespace) -> int:
    planned = load_planned_orders(arguments.planned)
    actual = load_actual_orders(arguments.actual)
    comparison = reconcile(planned, actual, arguments.date_tolerance)
    summary = summarise(comparison, arguments.date_tolerance)

    path = write_report(comparison, summary, arguments.output)

    discrepancies = int((comparison["status"] != STATUS_MATCH).sum())
    print(f"Report written to {path.resolve()}")
    print(
        f"{len(comparison)} order lines compared, "
        f"{discrepancies} with a discrepancy "
        f"({summary['discrepancy_rate_pct']}%)."
    )
    print(f"Net quantity difference: {summary['net_qty_diff']} units.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
