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


class MissingApiKeyError(RuntimeError):
    """Raised when the agent is used without GEMINI_API_KEY being set."""


class ReconciliationAgent:
    """Answers a natural-language question by letting the model use the tools."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations
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
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=question)])
        ]

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=self.config
            )

            calls = response.function_calls or []
            if not calls:
                return (response.text or "").strip() or (
                    "The model returned an empty answer."
                )

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
                self._trace_call(call.name, arguments)
                result = execute_tool(call.name, arguments)
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

    def _trace_result(self, result: dict) -> None:
        if not self.verbose:
            return
        payload = json.dumps(result, default=str)
        if len(payload) > 500:
            payload = payload[:500] + f"... [{len(payload)} chars total]"
        print(f"  <- {payload}")


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
