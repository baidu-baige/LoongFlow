#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for GradeMemory class
"""
import uuid
from typing import List

import pytest

from agentsdk.memory.grade import LongTermMemory, MediumTermMemory, ShortTermMemory
from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.memory import GradeMemory, MemoryConfig
from agentsdk.memory.grade.storage import InMemoryStorage
from agentsdk.message import ContentElement, Message, Role
from agentsdk.token import SimpleTokenCounter


class MockCompressor(Compressor):
    """Mock compressor for testing MTM"""

    def __init__(self):
        pass

    async def compress(self, messages: List[Message]) -> List[Message]:
        if len(messages) == 0:
            return []
        return [messages[-1]]


def create_test_message(msg_id=None, content="Hello, world!", role=Role.USER):
    """Helper function to create a test message"""
    if msg_id is None:
        msg_id = uuid.uuid4()
    return Message(
            id=msg_id,
            role=role,
            content=[ContentElement(
                    mime_type="text/plain",
                    data=content
            )]
    )


class TestGradeMemory:
    """Test cases for GradeMemory class"""

    @pytest.fixture
    def stm(self):
        return ShortTermMemory(InMemoryStorage())

    @pytest.fixture
    def mtm(self):
        return MediumTermMemory(InMemoryStorage(), MockCompressor())

    @pytest.fixture
    def ltm(self):
        return LongTermMemory(InMemoryStorage())

    @pytest.fixture
    def token_counter(self):
        return SimpleTokenCounter()

    @pytest.fixture
    def default_config(self):
        return MemoryConfig(
                token_threshold=128
        )

    @pytest.fixture
    def grade_memory(self, stm, mtm, ltm, token_counter, default_config):
        return GradeMemory(stm, mtm, ltm, token_counter, default_config)

    @pytest.mark.asyncio
    async def test_add_single_message(self, grade_memory):
        """Test adding a single message to GradeMemory"""
        msg = create_test_message(content="First")
        await grade_memory.add(msg)
        memory = await grade_memory.get_memory()
        assert len(memory) == 1
        assert memory[0].get_elements(ContentElement)[0].data == "First"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, grade_memory):
        """Test adding multiple messages to GradeMemory"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await grade_memory.add(messages)

        msgs = await grade_memory.get_memory()

        assert len(msgs) == 2

        assert msgs[0].get_elements(ContentElement)[0].data == "First"
        assert msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_add_with_auto_compress_disabled(self, stm, mtm, ltm, token_counter):
        """Test adding messages when auto compression is disabled"""
        config = MemoryConfig(
                token_threshold=8,
                auto_compress=False
        )
        grade_memory = GradeMemory(stm, mtm, ltm, token_counter, config)

        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await grade_memory.add(messages)

        msgs = await grade_memory.get_memory()

        assert len(msgs) == 2

        assert msgs[0].get_elements(ContentElement)[0].data == "First"
        assert msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_add_with_auto_compress(self, stm, mtm, ltm, token_counter):
        """Test adding messages when auto compression is disabled"""
        config = MemoryConfig(
                token_threshold=8,
                auto_compress=True
        )
        grade_memory = GradeMemory(stm, mtm, ltm, token_counter, config)

        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await grade_memory.add(messages)

        msgs = await grade_memory.get_memory()

        assert len(msgs) == 1

        assert msgs[0].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_remove_message(self, grade_memory, token_counter):
        """Test removing a message from STM"""
        msg1 = create_test_message(msg_id=uuid.uuid4(), content="First")
        msg2 = create_test_message(msg_id=uuid.uuid4(), content="First")

        await grade_memory.add([msg1, msg2])

        token_count = await token_counter.count([msg1, msg2])
        assert grade_memory._current_tokens == token_count

        await grade_memory.remove(msg1.id)

        assert grade_memory._current_tokens == token_count - await token_counter.count([msg1])

    @pytest.mark.asyncio
    async def test_get_memory(self, grade_memory):
        """Test retrieving combined memory from all levels"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await grade_memory.add(messages)

        result = await grade_memory.get_memory()

        assert len(result) == 2
        assert result[0].get_elements(ContentElement)[0].data == "First"
        assert result[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_commit_to_ltm(self, grade_memory, token_counter):
        """Test committing messages to long-term memory"""
        msg1 = create_test_message(msg_id=uuid.uuid4(), content="First")
        msg2 = create_test_message(msg_id=uuid.uuid4(), content="First")

        await grade_memory.add([msg1, msg2])

        token_count = await token_counter.count([msg1, msg2])
        assert grade_memory._current_tokens == token_count

        msg = create_test_message(content="Important fact")
        await grade_memory.commit_to_ltm(msg)

        assert grade_memory._current_tokens == token_count + await token_counter.count([msg])

    @pytest.mark.asyncio
    async def test_clear_memory(self, grade_memory):
        """Test clearing session memory (STM and MTM)"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await grade_memory.add(messages)

        assert await grade_memory.get_size() == 2
        await grade_memory.clear()
        assert await grade_memory.get_size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
