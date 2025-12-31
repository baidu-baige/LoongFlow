#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides default finalizer.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel, Field, ValidationError

from agentsdk.message import (
    ContentElement,
    Message,
    MimeType,
    Role,
    ToolCallElement,
    ToolOutputElement,
)
from agentsdk.models import BaseLLMModel, CompletionRequest
from agentsdk.tools import FunctionTool
from evolux.react import AgentContext
from evolux.react.components import Finalizer


class DefaultOutputSchema(BaseModel):
    """
    A default output schema only contains response field
    """

    response: str = Field(
        ...,
        description="The final, complete textual response for the user. "
        "Note: This is the only field that will be passed on as the final answer."
        "Ensure it includes all necessary information.",
    )


class DefaultFinalizer(Finalizer):
    """
    A default Finalizer that provides a tool for delivering the final answer.
    """

    def __init__(
        self,
        model: BaseLLMModel,
        summarize_prompt: str,
        output_schema: Type[BaseModel] | None = None,
        hint_message: Message = None,
        tool_name: str = "generate_final_answer",
        tool_description: str = "Final step: Call this to summarize and structure the concluding answer. "
        "This is mandatory for task completion.",
    ):
        self.model = model
        self.summarize_prompt = summarize_prompt
        if output_schema is None or not output_schema.model_fields:
            self._output_schema = DefaultOutputSchema
        else:
            self._output_schema = output_schema
        self._tool_name = tool_name
        self._tool_description = tool_description

        if hint_message is not None:
            self._hint_message = hint_message
        else:
            self._hint_message = Message.from_text(
                sender="finalizer",
                role=Role.USER,
                data="""The task has been terminated due to reaching the maximum step limit. 
                Please review the entire process and summarize for the user:
                1. What were the main steps attempted.
                2. What progress was made or which obstacles were encountered.
                3. Why a final answer could not be reached.
                Finally, provide a helpful concluding response.
                """,
            )

    @property
    def answer_schema(self) -> FunctionTool:
        """Dynamically creates and returns the 'final answer' tool schema."""

        def final_answer_implementation(**kwargs: Any) -> Dict[str, Any]:
            try:
                validated_model = self._output_schema.model_validate(kwargs)
                return {
                    "output_schema": validated_model.model_dump(),
                }
            except ValidationError as e:
                raise ValueError(f"Response validation failed, error: {e}")

        return FunctionTool(
            name=self._tool_name,
            func=final_answer_implementation,
            args_schema=self._output_schema,
            description=self._tool_description,
        )

    async def resolve_answer(
        self,
        tool_call: ToolCallElement,
        tool_output: ToolOutputElement,
    ) -> Message | None:
        """
        Checks if the executed tool was the final answer tool and if it was successful.
        """
        if tool_call.target != self._tool_name:
            return None
        if tool_output.status != "success":
            return None
        if len(tool_output.result) != 1:
            return None
        if not isinstance(tool_output.result[0], ContentElement):
            return None
        if not isinstance(tool_output.result[0].data, dict):
            return None

        output_data = tool_output.result[0].data.get("output_schema")
        if self._output_schema == DefaultOutputSchema:
            return Message.from_text(
                sender="finalizer",
                data=output_data.get("response", ""),
                role=Role.TOOL,
                mime_type=MimeType.TEXT_PLAIN,
            )
        return Message.from_text(
            sender="finalizer",
            data=output_data,
            role=Role.TOOL,
            mime_type=MimeType.APPLICATION_JSON,
        )

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """
        Summarizes the final answer if react loop exceeds max_steps.
        """
        system_message = Message.from_text(
            sender="finalizer",
            role=Role.SYSTEM,
            data=self.summarize_prompt,
        )
        history = await context.memory.get_memory()

        messages = [system_message] + history + [self._hint_message]

        resp_generator = self.model.generate(
            CompletionRequest(
                messages=messages, tools=context.toolkit.get_declarations()
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

        return Message.from_elements(
            sender="finalizer",
            role=Role.ASSISTANT,
            elements=list(resp.content),
            metadata={"usage": resp.usage},
        )
