#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides GradeMemory, which supports stm, mtm, ltm and auto compress.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Type

from agentsdk.logger import get_logger
from agentsdk.memory.grade import LongTermMemory, MediumTermMemory, ShortTermMemory
from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.storage import Storage
from agentsdk.message import Message
from agentsdk.models import BaseLLMModel
from agentsdk.token.base import TokenCounter

logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration settings for the Graded Memory system."""

    token_threshold: int = 65536
    auto_compress: bool = True


class GradeMemory:
    """Grade Memory"""

    def __init__(
        self,
        stm: ShortTermMemory,
        mtm: MediumTermMemory,
        ltm: LongTermMemory,
        token_counter: TokenCounter,
        config: MemoryConfig,
    ):
        self.stm = stm
        self.mtm = mtm
        self.ltm = ltm
        self.token_counter = token_counter
        self.config = config
        self._current_tokens: int = 0

    @classmethod
    def create_default(
        cls,
        model: BaseLLMModel,
        stm_storage: Storage | None = None,
        mtm_storage: Storage | None = None,
        ltm_storage: Type[Storage] | None = None,
        token_counter: TokenCounter | None = None,
        config: MemoryConfig | None = None,
        compressor: Compressor | None = None,
    ) -> GradeMemory:
        """
        Creates a GradedMemory instance with a default, in-memory setup.
        """
        from agentsdk.memory.grade import (
            LongTermMemory,
            MediumTermMemory,
            ShortTermMemory,
        )
        from agentsdk.memory.grade.compressor import LLMCompressor
        from agentsdk.memory.grade.storage import InMemoryStorage, FileStorage
        from agentsdk.token.simple import SimpleTokenCounter

        token_counter = token_counter or SimpleTokenCounter()

        stm_storage = stm_storage or InMemoryStorage()
        stm = ShortTermMemory(storage=stm_storage)

        mtm_storage = mtm_storage or InMemoryStorage()
        compressor = compressor or LLMCompressor(model)
        mtm = MediumTermMemory(storage=mtm_storage, compressor=compressor)

        ltm_storage = ltm_storage or FileStorage(f"./{uuid.uuid4()}.json")
        ltm = LongTermMemory(storage=ltm_storage)

        config = config or MemoryConfig()

        return cls(
            stm=stm, mtm=mtm, ltm=ltm, token_counter=token_counter, config=config
        )

    async def add(self, messages: Message | List[Message] | None) -> None:
        """
        Adds new messages to the memory system and triggers compression if the threshold is exceeded.
        """
        if messages is None:
            return

        await self.stm.add(messages)

        if not self.config.auto_compress:
            return

        if isinstance(messages, Message):
            messages = [messages]

        await self._update_token_count(messages)

        if self._current_tokens > self.config.token_threshold:
            logger.info("Compressing memory due to token threshold exceeded.")
            messages_to_compress = (
                await self.mtm.get_memory() + await self.stm.get_memory()
            )
            messages = await self.mtm.compress(messages_to_compress)
            await self.clear()
            await self.mtm.add(messages)
            await self._update_token_count(await self.get_memory())

    async def remove(self, message_id: uuid.UUID) -> bool:
        """Remove message from GradeMemory."""
        if not self.config.auto_compress:
            return (
                await self.stm.remove(message_id)
                or await self.mtm.remove(message_id)
                or await self.ltm.remove(message_id)
            )

        if msg := await self.stm.get(message_id):
            self._current_tokens -= await self.token_counter.count([msg])
            await self.stm.remove(message_id)
            return True

        if msg := await self.mtm.get(message_id):
            self._current_tokens -= self.token_counter.count([msg])
            await self.mtm.remove(message_id)
            return True

        if msg := await self.ltm.get(message_id):
            self._current_tokens -= self.token_counter.count([msg])
            await self.ltm.remove(message_id)
            return True

        return False

    async def get_memory(self) -> List[Message]:
        """Constructs the final context for the LLM by combining memories from LTM, MTM, and STM."""
        relevant_ltm = await self.ltm.get_memory()
        relevant_mtm = await self.mtm.get_memory()
        current_stm = await self.stm.get_memory()

        final_context: List[Message] = []
        seen_ids = set()
        for msg in relevant_ltm + relevant_mtm + current_stm:
            if msg.id not in seen_ids:
                final_context.append(msg)
                seen_ids.add(msg.id)
        return final_context

    async def get_size(self) -> int:
        """Get size of GradeMemory."""
        return (
            await self.stm.get_size()
            + await self.mtm.get_size()
            + await self.ltm.get_size()
        )

    async def commit_to_ltm(self, messages: Message | List[Message] | None) -> None:
        """Explicitly commits important information to Long-Term Memory."""
        if messages is None:
            return

        await self.ltm.add(messages)

        if not self.config.auto_compress:
            return

        if isinstance(messages, Message):
            messages = [messages]

        await self._update_token_count(messages)

    async def clear(self) -> None:
        """Clears session-specific memory (STM and MTM) while preserving Long-Term Memory."""
        await self.stm.clear()
        await self.mtm.clear()
        self._current_tokens = 0

    async def _update_token_count(self, new_messages: List[Message]) -> None:
        """
        Updates the STM token count based on a batch of new messages.
        """
        count = await self.token_counter.count(new_messages)
        self._current_tokens += count
