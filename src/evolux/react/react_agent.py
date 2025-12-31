#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides ReAct Agent
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Type

from pydantic import BaseModel

from agentsdk.logger import get_logger
from agentsdk.memory.grade import GradeMemory
from agentsdk.message import Message, Role, ToolCallElement, ToolOutputElement
from agentsdk.models import BaseLLMModel, CompletionUsage
from agentsdk.tools import Toolkit
from evolux.react.components import Actor, Finalizer, Observer, Reasoner
from evolux.react.context import AgentContext
from evolux.react.react_agent_base import ReactAgentBase

logger = get_logger(__name__)


class ReActAgent(ReactAgentBase):
    """
    Implements the ReAct (Reason, Act) agent architecture.

    This agent orchestrates the interaction between its components to execute
    a cyclical reasoning and action process until a final answer is produced.
    """

    def __init__(
        self,
        context: AgentContext,
        reasoner: Reasoner,
        actor: Actor,
        observer: Observer,
        finalizer: Finalizer,
        name: str = "ReAct",
    ):
        super().__init__()
        self.context = context
        self.reasoner = reasoner
        self.actor = actor
        self.observer = observer
        self.finalizer = finalizer
        self.context.toolkit.register_tool(self.finalizer.answer_schema)
        # we allow user to rewrite the interrupt method
        self._interrupt_handler: Callable[[AgentContext], Awaitable[None]] | None = (
            ReActAgent.default_interrupt_handler
        )
        self.name = name

    @classmethod
    def create_default(
        cls,
        model: BaseLLMModel,
        sys_prompt: str,
        output_format: Type[BaseModel] | None = None,
        toolkit: Toolkit | None = None,
        parallel_tool_run: bool = False,
        max_steps: int = 10,
        hint_message: Message = None,
    ) -> ReActAgent:
        """
        Creates a ReActAgent with a standard set of default components
        """

        from agentsdk.tools import Toolkit
        from evolux.react.components import (
            DefaultFinalizer,
            DefaultObserver,
            DefaultReasoner,
            ParallelActor,
            SequenceActor,
        )

        toolkit = toolkit or Toolkit()

        memory = GradeMemory.create_default(model)
        context = AgentContext(memory, toolkit, max_steps)
        reasoner = DefaultReasoner(model, sys_prompt)
        actor = ParallelActor() if parallel_tool_run else SequenceActor()
        observer = DefaultObserver()
        finalizer = DefaultFinalizer(
            model,
            summarize_prompt=sys_prompt,
            output_schema=output_format,
            hint_message=hint_message,
        )
        return cls(context, reasoner, actor, observer, finalizer, "Default")

    async def run(self, initial_messages: Message | List[Message], **kwargs) -> Message:
        """
        Starts the agent's execution loop with an initial set of messages.
        """
        trace_id = kwargs.get("trace_id")
        await self.context.add(initial_messages)
        total_completion_tokens = 0
        total_prompt_tokens = 0

        default_completion_usage = CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        )

        while self.context.current_step < self.context.max_steps:
            self.context.current_step += 1
            # 1. Reason
            thoughts = await self._reason()

            total_completion_tokens += thoughts.metadata.get(
                "usage", default_completion_usage
            ).completion_tokens
            total_prompt_tokens += thoughts.metadata.get(
                "usage", default_completion_usage
            ).prompt_tokens

            logger.info(
                f"Trace ID: {trace_id}: Agent: {self.name} Reason output: {thoughts}"
            )
            await self.context.add(thoughts)

            # 2. Act
            calls = thoughts.get_elements(ToolCallElement)
            outputs = await self._act(calls)
            for output in outputs:
                total_completion_tokens += output.metadata.get(
                    "usage", default_completion_usage
                ).completion_tokens
                total_prompt_tokens += output.metadata.get(
                    "usage", default_completion_usage
                ).prompt_tokens
                logger.info(
                    f"Trace ID: {trace_id}: Agent: {self.name} Act output: {output}"
                )
            await self.context.add(outputs)

            # finalize check finish
            if final_resp := await self._finalize(calls, outputs):
                total_completion_tokens += final_resp.metadata.get(
                    "usage", default_completion_usage
                ).completion_tokens
                total_prompt_tokens += final_resp.metadata.get(
                    "usage", default_completion_usage
                ).prompt_tokens
                logger.info(
                    f"Trace ID: {trace_id}: Agent: {self.name} Finalizer output: {final_resp}"
                )
                await self.context.add(final_resp)

                final_resp.metadata["total_completion_tokens"] = total_completion_tokens
                final_resp.metadata["total_prompt_tokens"] = total_prompt_tokens
                return final_resp

            # 3. Observe
            observations = await self._observe(outputs)
            if observations:
                total_completion_tokens += observations.metadata.get(
                    "usage", default_completion_usage
                ).completion_tokens
                total_prompt_tokens += observations.metadata.get(
                    "usage", default_completion_usage
                ).prompt_tokens
                logger.info(
                    f"Trace ID: {trace_id}: Agent: {self.name} Observation output: {observations}"
                )
                await self.context.add(observations)

        # if loop exit, no finalizer tool called, we need to summarize the failure task
        message = await self._summarize(**kwargs)
        total_completion_tokens += message.metadata.get("completion_tokens", 0)
        total_prompt_tokens += message.metadata.get("prompt_tokens", 0)
        logger.info(
            f"Trace ID: {trace_id}: Agent: {self.name} Summarize output: {message}"
        )
        await self.context.add(message)
        message.metadata["total_completion_tokens"] = total_completion_tokens
        message.metadata["total_prompt_tokens"] = total_prompt_tokens
        return message

    def register_interrupt(self, handler: Callable[[AgentContext], Awaitable[None]]):
        """
        Registers or overrides the asynchronous handler for interruptions.
        A default handler is pre-registered upon initialization.
        """
        self._interrupt_handler = handler

    @staticmethod
    async def default_interrupt_handler(context: AgentContext):
        """
        The default interrupt handler. Adds a message to the context indicating the interruption occurred.
        """
        interrupt_message = Message.from_text(
            sender="agent",
            role=Role.ASSISTANT,
            data="Agent execution was interrupted by user",
        )
        await context.add(interrupt_message)

    async def interrupt_impl(self):
        """
        handle user interrupt
        """
        await self._interrupt_handler(self.context)

    async def _reason(self) -> Message:
        return await self.reasoner.reason(self.context)

    async def _act(self, tool_calls: List[ToolCallElement]) -> List[Message]:
        return await self.actor.act(self.context, tool_calls)

    async def _observe(self, tool_outputs: List[Message]) -> Message | None:
        return await self.observer.observe(self.context, tool_outputs)

    async def _finalize(
        self, calls: List[ToolCallElement], outputs: List[Message]
    ) -> Message | None:
        output_map = {
            output.call_id: output
            for msg in outputs
            for output in msg.get_elements(ToolOutputElement)
        }

        for call in calls:
            if output := output_map.get(call.call_id):
                if result := await self.finalizer.resolve_answer(call, output):
                    return result
        return None

    async def _summarize(self, **kwargs) -> Message | None:
        return await self.finalizer.summarize_on_exceed(self.context, **kwargs)
