#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a default observer, which does nothing
"""

from typing import List

from agentsdk.message import Message, Role, MimeType
from evolux.react import AgentContext
from evolux.react.components import Observer


class DefaultObserver(Observer):
    """
    A dummy Observer that does nothing by default.
    The observation step is optional and can be used for more complex agents
    that need to reflect on or process tool outputs before the next reasoning step.
    """

    async def observe(
            self,
            context: AgentContext,
            tool_outputs: List[Message],
    ) -> Message | None:
        """Returns None, indicating no new observations are added to memory."""
        if not tool_outputs:
            return Message.from_content(
            sender="observer",
            role=Role.USER,
            data="No tool outputs founded, Your response MUST call a tool",
            mime_type=MimeType.TEXT_PLAIN
        )
        return None