#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory component, including stm, mtm, ltm
"""
import uuid
from typing import Any, List, Optional

from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.storage import Storage
from agentsdk.message import Message


class ShortTermMemory:
    """
    Short-Term Memory (STM): Stores the most recent conversation history.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    async def add(self, messages: Message | List[Message]) -> None:
        """Adds messages to the STM."""
        await self.storage.add(messages)

    async def get(self, message_id: uuid.UUID) -> Optional[Message]:
        """Get message from the STM."""
        return await self.storage.get(message_id)

    async def search(self, *args: Any, **kwargs: Any) -> List[Message]:
        """Search STM."""
        raise await self.storage.search(args, kwargs)

    async def remove(self, message_id: uuid.UUID) -> bool:
        """Remove message from the STM."""
        return await self.storage.remove(message_id)

    async def get_memory(self) -> List[Message]:
        """Retrieves all messages from the STM."""
        return await self.storage.get_all()

    async def get_size(self) -> int:
        """Get the size of the STM."""
        return await self.storage.get_size()

    async def clear(self) -> None:
        """Clears all messages from the STM."""
        await self.storage.clear()


class MediumTermMemory:
    """
    Medium-Term Memory (MTM): Stores compressed summaries of conversation history.
    """

    def __init__(self, storage: Storage, compressor: Compressor):
        self.storage = storage
        self.compressor = compressor

    async def add(self, messages: Message | List[Message]) -> None:
        """Adds messages to the MTM."""
        await self.storage.add(messages)

    async def get(self, message_id: uuid.UUID) -> Optional[Message]:
        """Get message from the STM."""
        return await self.storage.get(message_id)

    async def remove(self, message_id: uuid.UUID) -> bool:
        """Remove message from the MTM."""
        return await self.storage.remove(message_id)

    async def compress(self, messages: List[Message]) -> List[Message]:
        """Compresses messages"""
        return await self.compressor.compress(messages)

    async def get_memory(self) -> List[Message]:
        """Retrieves all summarized messages from the MTM."""
        return await self.storage.get_all()

    async def get_size(self) -> int:
        """Get the size of the MTM."""
        return await self.storage.get_size()

    async def clear(self) -> None:
        """Clears all summarized messages from the MTM."""
        await self.storage.clear()


class LongTermMemory:
    """
    Long-Term Memory (LTM): Stores facts, knowledge, and key information.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    async def add(self, messages: Message | List[Message]) -> None:
        """Adds facts or knowledge to the LTM."""
        await self.storage.add(messages)

    async def get(self, message_id: uuid.UUID) -> Optional[Message]:
        """Get message from the STM."""
        return await self.storage.get(message_id)

    async def remove(self, message_id: uuid.UUID) -> bool:
        """Remove facts from the LTM."""
        return await self.storage.remove(message_id)

    async def search(self, *args: Any, **kwargs: Any) -> List[Message]:
        """Searches for relevant information within the LTM."""
        return await self.storage.search(*args, **kwargs)

    async def get_memory(self) -> List[Message]:
        """Retrieves all facts from the LTM."""
        return await self.storage.get_all()

    async def get_size(self) -> int:
        """Get the size of the LTM."""
        return await self.storage.get_size()

    async def clear(self) -> None:
        """Clears all facts from the LTM."""
        await self.storage.clear()
