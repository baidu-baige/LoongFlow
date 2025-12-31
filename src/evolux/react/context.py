#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides context tests
"""
import uuid
from typing import List, Protocol

from agentsdk.message import Message
from agentsdk.tools import Toolkit


class Memory(Protocol):
    """
    A protocol defining the interface for the agent's memory system.
    """

    async def add(self, messages: Message | List[Message] | None) -> None:
        """
        add a message or message list to the memory system.
        """
        ...

    async def remove(self, message_id: uuid.UUID) -> bool:
        """
        remove a message from the memory system.
        """
        ...

    async def get_memory(self) -> List[Message]:
        """
        get history from the memory system.
        """
        ...


class AgentContext:
    """
    Manages the runtime state and resources for an agent's execution.

    This class holds references to the agent's memory and toolkit
    """

    def __init__(
            self,
            memory: Memory,
            toolkit: Toolkit,
            max_steps: int = 10,
    ):
        self.memory: Memory = memory
        self.toolkit: Toolkit = toolkit
        self.max_steps: int = max_steps
        self.current_step = 0

    async def add(self, messages: Message | List[Message] | None) -> None:
        """Adds one or more messages to the agent's memory."""
        await self.memory.add(messages)

    async def remove(self, message_id: uuid.UUID) -> bool:
        """remove message from the agent's memory."""
        return await self.memory.remove(message_id)

    async def get_memory(self) -> List[Message]:
        """Retrieves the current conversational context from memory."""
        return await self.memory.get_memory()
