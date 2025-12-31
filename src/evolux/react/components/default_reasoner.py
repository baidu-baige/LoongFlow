#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a default reasoner.
"""

from agentsdk.message import Message, Role
from agentsdk.models import BaseLLMModel, CompletionRequest
from evolux.react import AgentContext
from evolux.react.components import Reasoner


class DefaultReasoner(Reasoner):
    """
    A Reasoner that uses a Language Model to generate thoughts and tool calls.
    """

    def __init__(
        self,
        model: BaseLLMModel,
        system_prompt: str,
        name: str = "reasoner",
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt

    async def reason(self, context: AgentContext) -> Message:
        """
        Invokes the LLM with the current memory and a system prompt to generate the next step.
        """
        system_message = Message.from_text(
            sender=self.name, data=self.system_prompt, role=Role.SYSTEM
        )
        history = await context.get_memory()

        resp_generator = self.model.generate(
            CompletionRequest(
                messages=[system_message] + history,
                tools=context.toolkit.get_declarations(),
            )
        )

        try:
            resp = await anext(resp_generator)
            if resp.error_code:
                raise Exception(
                    f"Error code: {resp.error_code}, error: {resp.error_message}"
                )
        finally:
            async for _ in resp_generator:
                pass

        if resp.error_code:
            raise Exception(
                f"Error code: {resp.error_code}, error: {resp.error_message}"
            )
        return Message.from_elements(
            sender=self.name,
            role=Role.ASSISTANT,
            elements=list(resp.content),
            metadata={"usage": resp.usage},
        )
