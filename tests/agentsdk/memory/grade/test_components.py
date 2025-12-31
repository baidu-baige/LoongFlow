#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for memory components (STM, MTM, LTM)
"""
import uuid
from typing import List

import pytest

from agentsdk.memory.grade.components import (
    LongTermMemory,
    MediumTermMemory,
    ShortTermMemory,
)
from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.storage import InMemoryStorage
from agentsdk.message import ContentElement, Message, Role


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
        content=[ContentElement(mime_type="text/plain", data=content)],
    )


class TestShortTermMemory:
    """Test cases for ShortTermMemory class"""

    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def stm(self, storage):
        return ShortTermMemory(storage)

    @pytest.mark.asyncio
    async def test_add_single_message(self, stm, storage):
        """Test adding a single message to STM"""
        msg = create_test_message()

        await stm.add(msg)

        assert await stm.get_size() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, stm, storage):
        """Test adding multiple messages to STM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]

        await stm.add(messages)

        assert await stm.get_size() == 2
        msgs = await stm.get_memory()
        assert msgs[0].get_elements(ContentElement)[0].data == "First"
        assert msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_get_message(self, stm, storage):
        """Test retrieving a message from STM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await stm.add(expected_msg)
        result = await stm.get(msg_id)
        assert result == expected_msg

    @pytest.mark.asyncio
    async def test_get_message_not_found(self, stm, storage):
        """Test retrieving a non-existent message from STM"""
        msg_id = uuid.uuid4()
        result = await stm.get(msg_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_message(self, stm, storage):
        """Test removing a message from STM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await stm.add(expected_msg)

        result = await stm.remove(msg_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_remove_message_not_found(self, stm, storage):
        """Test removing a non-existent message from STM"""
        msg_id = uuid.uuid4()

        result = await stm.remove(msg_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_search_messages(self, stm, storage):
        """Test searching messages in STM"""
        with pytest.raises(NotImplementedError):
            await stm.search("query", limit=10)

    @pytest.mark.asyncio
    async def test_get_memory(self, stm, storage):
        """Test retrieving all messages from STM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await stm.add(messages)

        assert await stm.get_size() == 2
        result = await stm.get_memory()

        assert result[0].get_elements(ContentElement)[0].data == "First"
        assert result[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_get_size(self, stm, storage):
        """Test getting the size of STM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await stm.add(messages)

        assert await stm.get_size() == 2

    @pytest.mark.asyncio
    async def test_clear_memory(self, stm, storage):
        """Test clearing all messages from STM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await stm.add(messages)

        assert await stm.get_size() == 2
        await stm.clear()
        assert await stm.get_size() == 0


class TestMediumTermMemory:
    """Test cases for MediumTermMemory class"""

    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def compressor(self):
        return MockCompressor()

    @pytest.fixture
    def mtm(self, storage, compressor):
        return MediumTermMemory(storage, compressor)

    @pytest.mark.asyncio
    async def test_add_single_message(self, mtm, storage):
        """Test adding a single message to MTM"""
        msg = create_test_message()

        await mtm.add(msg)

        assert await mtm.get_size() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, mtm, storage):
        """Test adding multiple messages to MTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]

        await mtm.add(messages)

        assert await mtm.get_size() == 2
        msgs = await mtm.get_memory()
        assert msgs[0].get_elements(ContentElement)[0].data == "First"
        assert msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_get_message(self, mtm, storage):
        """Test retrieving a message from MTM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await mtm.add(expected_msg)
        result = await mtm.get(msg_id)
        assert result == expected_msg

    @pytest.mark.asyncio
    async def test_remove_message(self, mtm, storage):
        """Test removing a message from MTM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await mtm.add(expected_msg)

        result = await mtm.remove(msg_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_compress_messages(self, mtm, compressor):
        """Test compressing messages in MTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        compressed = await mtm.compress(messages)

        assert len(compressed) == 1

    @pytest.mark.asyncio
    async def test_get_memory(self, mtm, storage):
        """Test retrieving all messages from MTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await mtm.add(messages)

        assert await mtm.get_size() == 2
        result = await mtm.get_memory()

        assert result[0].get_elements(ContentElement)[0].data == "First"
        assert result[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_clear_memory(self, mtm, storage):
        """Test clearing all messages from MTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await mtm.add(messages)

        assert await mtm.get_size() == 2
        await mtm.clear()
        assert await mtm.get_size() == 0


class TestLongTermMemory:
    """Test cases for LongTermMemory class"""

    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def ltm(self, storage):
        return LongTermMemory(storage)

    @pytest.mark.asyncio
    async def test_add_single_message(self, ltm, storage):
        """Test adding a single message to LTM"""
        msg = create_test_message()

        await ltm.add(msg)

        assert await ltm.get_size() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, ltm, storage):
        """Test adding multiple messages to LTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]

        await ltm.add(messages)

        assert await ltm.get_size() == 2
        msgs = await ltm.get_memory()
        assert msgs[0].get_elements(ContentElement)[0].data == "First"
        assert msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_get_message(self, ltm, storage):
        """Test retrieving a message from LTM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await ltm.add(expected_msg)
        result = await ltm.get(msg_id)
        assert result == expected_msg

    @pytest.mark.asyncio
    async def test_remove_message(self, ltm, storage):
        """Test removing a message from LTM"""
        msg_id = uuid.uuid4()
        expected_msg = create_test_message(msg_id=msg_id)

        await ltm.add(expected_msg)

        result = await ltm.remove(msg_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_search_messages(self, ltm, storage):
        """Test searching messages in LTM"""
        with pytest.raises(NotImplementedError):
            await ltm.search("query", limit=10)

    @pytest.mark.asyncio
    async def test_get_memory(self, ltm, storage):
        """Test retrieving all messages from LTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await ltm.add(messages)

        assert await ltm.get_size() == 2
        result = await ltm.get_memory()

        assert result[0].get_elements(ContentElement)[0].data == "First"
        assert result[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_clear_memory(self, ltm, storage):
        """Test clearing all messages from LTM"""
        msg1 = create_test_message(content="First")
        msg2 = create_test_message(content="Second")
        messages = [msg1, msg2]
        await ltm.add(messages)

        assert await ltm.get_size() == 2
        await ltm.clear()
        assert await ltm.get_size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
