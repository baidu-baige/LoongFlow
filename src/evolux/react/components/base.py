#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is abstraction for components in ReAct Agent
"""

from typing import List, Protocol

from agentsdk.message import Message, ToolCallElement, ToolOutputElement
from agentsdk.tools import FunctionTool
from evolux.react import AgentContext


class Reasoner(Protocol):
    """
    The "Reason" component of the ReAct loop.
    Its role is to analyze the current context and decide on the next action.
    """

    async def reason(self, context: AgentContext) -> Message:
        """
        reason the current context and decide on the next action.
        """
        ...


class Actor(Protocol):
    """
    The "Action" component of the ReAct loop.
    Its role is to execute the actions decided by the Reasoner (e.g., tool calls).
    """

    async def act(
        self, context: AgentContext, tool_calls: List[ToolCallElement]
    ) -> List[Message]:
        """
        act executes the actions decided by the Reasoner (e.g., tool calls).
        """
        ...


class Observer(Protocol):
    """
    The "Observe" component of the ReAct loop.
    Its role is to process the results of actions and prepare them for the next reasoning step.
    """

    async def observe(
        self, context: AgentContext, tool_outputs: List[Message]
    ) -> Message | None:
        """
        observe act results and prepare them for the next reasoning step.
        """
        ...


class Finalizer(Protocol):
    """
    The "Finalize" component of the ReAct loop.
    Its role is to determine if the agent's task is complete and to construct the final response.
    """

    @property
    def answer_schema(self) -> FunctionTool:
        """The schema or definition of the special 'final answer' tool."""
        ...

    async def resolve_answer(
        self,
        tool_call: ToolCallElement,
        tool_output: ToolOutputElement,
    ) -> Message | None:
        """
        Attempts to resolve a tool interaction into the final answer.

        Returns:
            The final message if successful, otherwise None.
        """
        ...

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """
        Summarizes the final answer if react loop exceeds max_steps.
        Returns:
            The summarized message
        """
        ...
