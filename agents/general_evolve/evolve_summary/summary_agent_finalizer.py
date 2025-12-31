#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides SummaryAgentFinalizer, an enhanced version of DefaultFinalizer
that supports structured response generation using `response_format`.
"""

from agentsdk.logger import print_message
from agentsdk.message import Message, Role
from agentsdk.models import CompletionRequest
from evolux.react import AgentContext

from evolux.react.components.default_finalizer import DefaultFinalizer


class SummaryAgentFinalizer(DefaultFinalizer):
    """
    SummaryAgentFinalizer extends DefaultFinalizer to enable structured
    JSON responses aligned with the specified `output_schema`.

    When the ReAct loop exceeds its max steps, this finalizer will:
      1. Summarize the overall reasoning and progress.
      2. Request the model to return a structured response matching
         the `output_schema` (via `response_format`).
    """

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """
        Summarizes the task when ReAct exceeds the maximum number of steps.

        This implementation explicitly enforces structured output by passing
        `response_format=self._output_schema` to the model generation call.

        Args:
            context (AgentContext): The agent's execution context.

        Returns:
            Message | None: A Message object containing the structured JSON result.
        """
        system_message = Message.from_text(
            sender="summary_finalizer",
            role=Role.SYSTEM,
            data=self.summarize_prompt,
        )
        history = await context.memory.get_memory()

        messages = [system_message] + history + [self._hint_message]

        response_format = None
        if self._output_schema:
            schema_dict = self._output_schema.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": self._output_schema.__name__,
                    "schema": schema_dict,
                },
            }

        resp_generator = self.model.generate(
            CompletionRequest(
                messages=messages,
                tools=context.toolkit.get_declarations(),
                response_format=response_format,
            )
        )

        resp = await anext(resp_generator)
        if resp.error_code:
            raise Exception(
                f"Error code: {resp.error_code}, error: {resp.error_message}"
            )
        resp_msg = Message.from_elements(
            sender="summary_finalizer",
            role=Role.ASSISTANT,
            elements=list(resp.content),
            metadata={"usage": resp.usage},
        )
        print_message(resp_msg)
        return resp_msg
