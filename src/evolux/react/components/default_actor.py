#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file provides a sequence and parallel actor
"""

import asyncio
from typing import List

from agentsdk.message import ContentElement, Message, Role, ToolCallElement, ToolStatus
from evolux.react import AgentContext
from evolux.react.components import Actor


class SequenceActor(Actor):
    """
    An Actor that executes tool calls sequentially using the provided toolkit.
    """

    async def act(
            self,
            context: AgentContext,
            tool_calls: List[ToolCallElement],
    ) -> List[Message]:
        """
        Iterates through tool calls and executes them one by one.
        """
        outputs: List[Message] = []
        for call in tool_calls:
            if arguments_err := call.metadata.get("tool_arguments_err"):
                result = [ContentElement(data=arguments_err)]
                status = ToolStatus.ERROR
            else:
                response_message = await context.toolkit.arun(
                        name=call.target,
                        args=call.arguments,
                )
                status = ToolStatus.ERROR if response_message.err_msg else ToolStatus.SUCCESS
                result = [ContentElement(data=response_message.err_msg)] \
                    if response_message.err_msg else response_message.content

            outputs.append(Message.from_tool_output(
                    sender="actor",
                    call_id=call.call_id,
                    tool_name=call.target,
                    status=status,
                    result=result,
                    role=Role.TOOL,
            ))
        return outputs


class ParallelActor(Actor):
    """
    An Actor that executes tool calls in parallel using the provided toolkit.
    """

    async def _execute(
            self, context: AgentContext, call: ToolCallElement
    ) -> Message:
        """
        Executes a single tool call and formats the output into a Message.
        """
        if arguments_err := call.metadata.get("tool_arguments_err"):
            result = [ContentElement(data=arguments_err)]
            status = ToolStatus.ERROR
        else:
            response_message = await context.toolkit.arun(
                    name=call.target,
                    args=call.arguments,
            )
            status = ToolStatus.ERROR if response_message.err_msg else ToolStatus.SUCCESS
            result = [ContentElement(data=response_message.err_msg)] \
                if response_message.err_msg else response_message.content

        return Message.from_tool_output(
                sender="actor",
                call_id=call.call_id,
                tool_name=call.target,
                status=status,
                result=result,
                role=Role.TOOL,
        )

    async def act(
            self,
            context: AgentContext,
            tool_calls: List[ToolCallElement],
    ) -> List[Message]:
        """
        Creates and runs tool call tasks in parallel.
        """
        if not tool_calls:
            return []

        tasks = [
            self._execute(context, call) for call in tool_calls
        ]
        outputs = await asyncio.gather(*tasks)
        return list(outputs)
