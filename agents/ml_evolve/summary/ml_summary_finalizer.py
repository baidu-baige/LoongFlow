# -*- coding: utf-8 -*-

"""
This file provides plan agent finalizer.
"""
import json

from pydantic import BaseModel, Field

from agentsdk.message import (
    Message,
    Role,
)
from agentsdk.models import CompletionRequest
from evolux.react import AgentContext
from evolux.react.components import DefaultFinalizer


class Reflection(BaseModel):
    """
    Represents the final, structured output of the ML Summary Agent's analysis.
    """

    code_technical_summary: str = Field(
        default="",
        description=(
            """
            Compressed technical fingerprint of each pipeline stage.
            Used by Planner to compare solutions WITHOUT reading full code.
            """
        ),
    )

    root_cause_analysis: str = Field(
        default="",
        description=(
            """
            Analysis connecting stage changes to performance outcomes.
            """
        ),
    )

    key_learnings: str = Field(
        default="",
        description=(
            """
            Reusable ML insights distilled from this experiment.
            """
        ),
    )

    actionable_guidance: str = Field(
        default="",
        description=(
            """
            Stage-specific recommendations for the next Planner iteration。
            """
        ),
    )
    fusion_profile: str = Field(
        default="",
        description=(
            """
            Model characteristics for ensemble diversity analysis.
            """
        ),
    )


class MLSummaryFinalizer(DefaultFinalizer):
    """
    MLSummaryFinalizer extends DefaultFinalizer to enable structured
    JSON responses aligned with the specified `output_schema`.

    When the ReAct loop exceeds its max steps, this finalizer will:
      1. Summarize the overall reasoning and progress.
      2. Request the model to return a structured response matching
         the `output_schema` (via `response_format`).
    """

    async def summarize_on_exceed(self, context: AgentContext, **kwargs) -> Message | None:
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
            sender="MLSummary_finalizer",
            role=Role.SYSTEM,
            data=self.summarize_prompt,
        )
        history = await context.memory.get_memory()

        hint_msg = Message.from_text(
            sender="finalizer",
            role=Role.USER,
            data=f"""
            Summarize strictly in JSON following this schema:
            {json.dumps(Reflection.model_json_schema(), indent=2, ensure_ascii=False)}
            """,
        )

        messages = [system_message] + history + [hint_msg]

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
                response_format=response_format
            )
        )
        try:
            resp = await anext(resp_generator)
            if resp.error_code:
                raise Exception(f"Error code: {resp.error_code}, error: {resp.error_message}")
        finally:
            async for _ in resp_generator:
                pass

        resp_msg = Message.from_elements(
            sender="MLSummary_finalizer",
            role=Role.ASSISTANT,
            elements=list(resp.content),
            metadata={"usage": resp.usage},
        )
        return resp_msg
