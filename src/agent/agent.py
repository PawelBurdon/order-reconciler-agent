"""The function calling loop.

This is the whole AI layer: send the question together with the tool schemas,
and for as long as the model answers with a function call, run it and send the
result back. The model decides what to call; this module only carries messages
and keeps the conversation from running forever.

The google-genai SDK can do all of this by itself - hand it Python callables
instead of declarations and it will call them for you. That is turned off here
on purpose: the loop is the part of the project worth showing.
"""

from __future__ import annotations

import json
import os
import re

from google import genai
from google.genai import types

from .prompts import SYSTEM_PROMPT
from .tools import TOOL_DECLARATIONS, execute_tool

# Flash-Lite is the default because its free-tier request limit is three times
# higher, which is what a project someone clones and tries out actually needs.
# Override it with --model.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# How many times the model may call tools before answering. Eight is far more
# than any question here needs (two or three is typical); the limit exists so a
# model that keeps re-calling the same tool stops burning tokens instead of
# looping until someone notices.
MAX_ITERATIONS = 8

# Order ids and SKU codes: two or more capitals, a hyphen, digits. Anything
# shaped like this in an answer is a thing the model is naming, and a name it
# did not read somewhere is a name it made up.
IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{2,8}\b")

# One correction, not a loop. If the model cannot fix it when told exactly
# what is wrong, asking again is unlikely to help and the answer should carry
# the warning instead of hiding it.
MAX_CORRECTIONS = 1

CORRECTION_TEMPLATE = (
    "Stop. Your answer refers to {identifiers}, which appear nowhere in the "
    "tool results you were given. You have invented an identifier, most likely "
    "by tidying an unusual one into a more familiar shape. Read the tool "
    "results again and repeat the answer using the identifiers exactly as they "
    "appear there. If you cannot find the one you meant, say so instead of "
    "naming it."
)


class MissingApiKeyError(RuntimeError):
    """Raised when the agent is used without GEMINI_API_KEY being set."""


class ReconciliationAgent:
    """Answers a natural-language question by letting the model use the tools."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        max_iterations: int = MAX_ITERATIONS,
        remember: bool = False,
    ) -> None:
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.remember = remember
        self.history: list[types.Content] = []
        # The calls made while answering the most recent question. --verbose
        # prints them for a human; this records them for a program, which is
        # what the eval harness scores.
        self.calls: list[dict] = []
        self.client = genai.Client(api_key=_read_api_key())

        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            # The tools are the only source of facts, so there is nothing to be
            # gained from a creative sampling temperature.
            temperature=0.0,
            # Without this the SDK would execute the tools itself and hide the
            # loop below.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def ask(self, question: str) -> str:
        """Run the conversation until the model answers with text."""
        contents: list[types.Content] = list(self.history)
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        self.calls = []

        # Everything the model is entitled to quote an identifier from: the
        # question itself, anything it already said in this conversation, and
        # every tool result it has seen while answering this one.
        evidence: list[str] = [question]
        evidence.extend(
            part.text or "" for content in self.history for part in content.parts or []
        )
        corrections = 0

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=self.config
            )

            calls = response.function_calls or []
            if not calls:
                answer = (response.text or "").strip() or (
                    "The model returned an empty answer."
                )

                invented = _invented_identifiers(answer, evidence)
                if invented and corrections < MAX_CORRECTIONS:
                    corrections += 1
                    self._trace_correction(invented)
                    contents.append(response.candidates[0].content)
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text=CORRECTION_TEMPLATE.format(
                                        identifiers=", ".join(invented)
                                    )
                                )
                            ],
                        )
                    )
                    continue

                if invented:
                    # Told once and still wrong. Better a visible warning than
                    # a clean sentence pointing at an order nobody can find.
                    answer += (
                        f"\n\n[Unverified: {', '.join(invented)} does not appear "
                        f"in any tool result and may not exist.]"
                    )

                self._remember_exchange(question, answer)
                return answer

            # The model's own turn has to go into the history before the
            # results, otherwise the next request has answers to questions that
            # were never asked.
            contents.append(response.candidates[0].content)

            self._trace_thinking(iteration, response)

            # A single turn may contain several calls. All of them are executed
            # and all of the results travel back in one turn - responding to
            # only the first would leave the model waiting for an answer that
            # never comes.
            result_parts = []
            for call in calls:
                arguments = dict(call.args or {})
                self.calls.append({"name": call.name, "arguments": arguments})
                self._trace_call(call.name, arguments)
                result = execute_tool(call.name, arguments)
                evidence.append(json.dumps(result, default=str))
                self._trace_result(result)
                result_parts.append(
                    types.Part.from_function_response(
                        name=call.name, response=result
                    )
                )

            contents.append(types.Content(role="user", parts=result_parts))

        return (
            f"Stopped after {self.max_iterations} tool calls without reaching an "
            f"answer. The question may be too broad, or a tool may be returning "
            f"something the model cannot use - rerun with --verbose to see the "
            f"calls it made."
        )

    def reset(self) -> None:
        """Forget the conversation so far, keeping the agent usable."""
        self.history.clear()

    def _remember_exchange(self, question: str, answer: str) -> None:
        """Keep the question and the answer; throw the tool traffic away.

        This is the whole of the follow-up support, and the discarding is the
        interesting half. A conversation carries two very different kinds of
        weight: the questions and answers, which are short and are what a
        follow-up refers back to, and the tool results, which are by far the
        bulkiest thing in the history and are already stale by the next
        question. Keeping everything would make each turn more expensive than
        the last for no benefit; if the model needs a number again it can call
        the tool again, which is cheap and cannot go out of date.

        It also keeps the history structurally simple: text in, text out, so
        there is no way to end up with a function call whose response was
        pruned - which the API rejects.
        """
        if not self.remember:
            return
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=question)])
        )
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )

    # -- verbose tracing ---------------------------------------------------
    #
    # The point of --verbose is to make the agent's reasoning inspectable: which
    # tool it picked, with which arguments, and what came back. Without it an
    # agent is a box that either works or does not.

    def _trace_thinking(self, iteration: int, response: types.GenerateContentResponse) -> None:
        if not self.verbose:
            return
        print(f"\n[iteration {iteration}]")
        for part in response.candidates[0].content.parts or []:
            if part.text:
                print(f"  model: {part.text.strip()}")

    def _trace_call(self, name: str, arguments: dict) -> None:
        if not self.verbose:
            return
        rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        print(f"  -> {name}({rendered})")

    def _trace_correction(self, invented: list[str]) -> None:
        if not self.verbose:
            return
        print(f"  !! answer named {', '.join(invented)} - asking again")

    def _trace_result(self, result: dict) -> None:
        if not self.verbose:
            return
        payload = json.dumps(result, default=str)
        if len(payload) > 500:
            payload = payload[:500] + f"... [{len(payload)} chars total]"
        print(f"  <- {payload}")


def _invented_identifiers(answer: str, evidence: list[str]) -> list[str]:
    """Identifiers the answer names that nothing the model read contains.

    The prompt already asks the model to copy identifiers exactly. It mostly
    does, and then roughly once in six answers it does not: ORD-1090 came back
    as ORD-1000 - every other figure in the sentence correct, the odd id tidied
    into the shape of the twenty-nine others in the data. That answer survives
    a human review and sends somebody looking for an order that does not exist.

    A prompt is a request. This is the check, and it is the same principle the
    rest of the project runs on: where Python can verify the model, it should.
    """
    seen = "\n".join(evidence)
    named = dict.fromkeys(IDENTIFIER_PATTERN.findall(answer))
    return [identifier for identifier in named if identifier not in seen]


def _read_api_key() -> str:
    """Read the key from the environment, or explain how to provide one."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and put your "
            "key in it, or export the variable in your shell. The report command "
            "works without a key."
        )
    return api_key
