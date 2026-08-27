"""Tests for the function calling loop.

The Gemini client is replaced with a stub that returns scripted responses, so
these tests exercise the real loop - history order, parallel calls, the
iteration limit, what is kept between questions - without a key, a network or a
token. What the model would decide is not under test; what this code does with
the decision is.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai import types

from src.agent import tools
from src.agent.agent import ReconciliationAgent
from src.core.reconciler import reconcile


class StubResponse:
    """The parts of a GenerateContentResponse that the loop actually reads."""

    def __init__(self, text: str | None = None, calls: tuple = ()):
        self.text = text
        self.function_calls = list(calls)

        parts = [types.Part(text=text)] if text else []
        parts.extend(types.Part(function_call=call) for call in calls)
        self.candidates = [
            SimpleNamespace(content=types.Content(role="model", parts=parts))
        ]


class StubModels:
    """Returns the scripted responses in order and records what it was sent."""

    def __init__(self, responses: list[StubResponse]):
        self.responses = list(responses)
        self.requests: list[list[types.Content]] = []

    def generate_content(self, *, model, contents, config):
        self.requests.append(list(contents))
        # The last response repeats, which is what the iteration-limit test
        # needs: a model that never stops asking for tools.
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def call(name: str, **arguments) -> types.FunctionCall:
    return types.FunctionCall(name=name, args=arguments)


@pytest.fixture(autouse=True)
def comparison():
    """Give the tools something to work on, and take it away afterwards."""
    planned = pd.DataFrame(
        [("ORD-1", "Velo Parts Ltd", "BRK-1", 100, "2025-08-01")],
        columns=["order_id", "customer", "sku", "planned_qty", "planned_date"],
    )
    planned["planned_qty"] = planned["planned_qty"].astype("Int64")
    planned["planned_date"] = pd.to_datetime(planned["planned_date"])

    actual = pd.DataFrame(
        [("ORD-1", "Velo Parts Ltd", "BRK-1", 80, "2025-08-01")],
        columns=["order_id", "customer", "sku", "actual_qty", "actual_date"],
    )
    actual["actual_qty"] = actual["actual_qty"].astype("Int64")
    actual["actual_date"] = pd.to_datetime(actual["actual_date"])

    tools.set_comparison(reconcile(planned, actual))
    yield
    tools.set_comparison(None)


@pytest.fixture
def build_agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-used")

    def build(responses: list[StubResponse], **kwargs):
        agent = ReconciliationAgent(**kwargs)
        stub = StubModels(responses)
        agent.client = SimpleNamespace(models=stub)
        return agent, stub

    return build


def roles(contents: list[types.Content]) -> list[str]:
    return [content.role for content in contents]


def test_a_tool_result_goes_back_before_the_next_request(build_agent):
    agent, stub = build_agent(
        [
            StubResponse(calls=(call("get_summary"),)),
            StubResponse(text="One line is short by 20 units."),
        ]
    )

    assert agent.ask("how bad is it?") == "One line is short by 20 units."
    assert len(stub.requests) == 2

    # The model's own turn must sit between the question and the result,
    # otherwise the second request answers something that was never asked.
    assert roles(stub.requests[1]) == ["user", "model", "user"]

    response_part = stub.requests[1][2].parts[0]
    assert response_part.function_response.name == "get_summary"
    assert response_part.function_response.response["total_order_lines"] == 1


def test_several_calls_in_one_turn_all_come_back_together(build_agent):
    agent, stub = build_agent(
        [
            StubResponse(
                calls=(
                    call("filter_records", customer="Velo"),
                    call("top_discrepancies", by="qty_diff", limit=1),
                )
            ),
            StubResponse(text="Done."),
        ]
    )

    agent.ask("compare two things")

    results_turn = stub.requests[1][-1]
    assert len(results_turn.parts) == 2
    assert [part.function_response.name for part in results_turn.parts] == [
        "filter_records",
        "top_discrepancies",
    ]


def test_a_failing_tool_is_reported_to_the_model_not_raised(build_agent):
    agent, stub = build_agent(
        [
            StubResponse(calls=(call("filter_records", customer="Acme"),)),
            StubResponse(text="There is no such customer."),
        ]
    )

    assert agent.ask("what about Acme?") == "There is no such customer."

    payload = stub.requests[1][-1].parts[0].function_response.response
    assert "error" in payload


def test_the_iteration_limit_stops_a_model_that_never_answers(build_agent):
    agent, stub = build_agent(
        [StubResponse(calls=(call("get_summary"),))], max_iterations=3
    )

    answer = agent.ask("loop forever please")

    assert len(stub.requests) == 3
    assert "Stopped after 3 tool calls" in answer


def test_nothing_is_remembered_by_default(build_agent):
    agent, _ = build_agent([StubResponse(text="An answer.")])

    agent.ask("a question")

    assert agent.history == []


def test_a_follow_up_sees_the_previous_exchange(build_agent):
    agent, stub = build_agent(
        [StubResponse(text="First answer."), StubResponse(text="Second answer.")],
        remember=True,
    )

    agent.ask("first question")
    agent.ask("second question")

    assert roles(stub.requests[1]) == ["user", "model", "user"]
    assert stub.requests[1][0].parts[0].text == "first question"
    assert stub.requests[1][1].parts[0].text == "First answer."


def test_the_history_keeps_the_talking_and_drops_the_tool_traffic(build_agent):
    """The point of the compaction: questions and answers survive, calls do not."""
    agent, _ = build_agent(
        [
            StubResponse(calls=(call("get_summary"),)),
            StubResponse(text="One line is short."),
        ],
        remember=True,
    )

    agent.ask("how bad is it?")

    assert roles(agent.history) == ["user", "model"]
    for content in agent.history:
        for part in content.parts:
            assert part.text is not None
            assert part.function_call is None
            assert part.function_response is None


def test_an_invented_identifier_is_sent_back_to_be_corrected(build_agent):
    """The failure this exists for: ORD-1090 answered as ORD-1000."""
    agent, stub = build_agent(
        [
            StubResponse(calls=(call("filter_records", status="UNPLANNED"),)),
            StubResponse(text="300 units arrived under order ORD-1000."),
            StubResponse(text="300 units arrived under order ORD-1."),
        ]
    )

    answer = agent.ask("did anything arrive unplanned?")

    assert answer == "300 units arrived under order ORD-1."
    correction = stub.requests[-1][-1].parts[0].text
    assert "ORD-1000" in correction
    assert "invented an identifier" in correction


def test_an_identifier_that_appears_in_a_tool_result_is_left_alone(build_agent):
    agent, stub = build_agent(
        [
            StubResponse(calls=(call("filter_records", customer="Velo"),)),
            StubResponse(text="ORD-1 for Velo Parts Ltd is short by 20 units."),
        ]
    )

    assert agent.ask("what is short?") == "ORD-1 for Velo Parts Ltd is short by 20 units."
    # Two requests, not three: nothing needed correcting.
    assert len(stub.requests) == 2


def test_an_identifier_from_the_question_is_not_treated_as_invented(build_agent):
    """The user is allowed to name an order the tools never returned."""
    agent, _ = build_agent([StubResponse(text="ORD-9999 is not in the data.")])

    assert agent.ask("what happened to ORD-9999?") == "ORD-9999 is not in the data."


def test_an_identifier_that_survives_the_correction_is_flagged(build_agent):
    """Told once and still wrong: warn rather than answer cleanly."""
    agent, _ = build_agent(
        [
            StubResponse(text="Order ORD-4242 was never delivered."),
            StubResponse(text="Order ORD-4242 was never delivered."),
        ]
    )

    answer = agent.ask("what is missing?")

    assert "Unverified: ORD-4242" in answer
    assert "may not exist" in answer


def test_reset_clears_the_conversation(build_agent):
    agent, _ = build_agent([StubResponse(text="An answer.")], remember=True)

    agent.ask("a question")
    assert len(agent.history) == 2

    agent.reset()
    assert agent.history == []
