# -*- coding: utf-8 -*-
"""
Test cases for AgentContext and Memory protocol
"""
import uuid

import pytest

from agentsdk.memory.grade import GradeMemory, MemoryConfig
from agentsdk.message import ContentElement, Message
from agentsdk.tools import Toolkit
from evolux.react import AgentContext


class TestMemoryProtocol:
    """Test cases for Memory protocol"""

    @pytest.mark.asyncio
    async def test_memory_protocol_implementation(self):
        """Test that Memory protocol can be implemented correctly"""

        memory = GradeMemory.create_default(None, config=MemoryConfig(
                auto_compress=False,
        ))

        # Test adding single message
        message1 = Message.from_text(sender="user", data="test1")
        await memory.add(message1)
        assert len(await memory.get_memory()) == 1

        # Test adding list of messages
        message2 = Message.from_text(sender="assistant", data="test2")
        message3 = Message.from_text(sender="user", data="test3")
        await memory.add([message2, message3])
        memory_list = await memory.get_memory()
        assert len(memory_list) == 3

        # Test get_memory
        assert memory_list[0].get_elements(ContentElement)[0].data == "test1"
        assert memory_list[1].get_elements(ContentElement)[0].data == "test2"
        assert memory_list[2].get_elements(ContentElement)[0].data == "test3"

        # Test remove existing message
        result = await memory.remove(message2.id)
        assert result is True
        assert len(await memory.get_memory()) == 2

        # Test remove non-existing message
        result = await memory.remove(uuid.uuid4())
        assert result is False
        assert len(await memory.get_memory()) == 2

        # Test add None
        await memory.add(None)
        assert len(await memory.get_memory()) == 2


class TestAgentContext:
    """Test cases for AgentContext"""

    @pytest.fixture
    def mock_memory(self):
        return GradeMemory.create_default(None)

    @pytest.fixture
    def mock_toolkit(self):
        return Toolkit()

    @pytest.fixture
    def agent_context(self, mock_memory, mock_toolkit):
        return AgentContext(memory=mock_memory, toolkit=mock_toolkit, max_steps=10)

    @pytest.mark.asyncio
    async def test_add_single_message(self, agent_context, mock_memory):
        """Test adding single message to context"""
        message = Message.from_text(sender="user", data="test message")

        await agent_context.add(message)

        assert len(await agent_context.get_memory()) == 1

    @pytest.mark.asyncio
    async def test_add_list_of_messages(self, agent_context, mock_memory):
        """Test adding list of messages to context"""
        messages = [
            Message.from_text(sender="user", data="message1"),
            Message.from_text(sender="assistant", data="message2")
        ]

        await agent_context.add(messages)

        assert len(await agent_context.get_memory()) == 2

    @pytest.mark.asyncio
    async def test_add_none(self, agent_context, mock_memory):
        """Test adding None to context"""
        await agent_context.add(None)

        assert len(await agent_context.get_memory()) == 0

    @pytest.mark.asyncio
    async def test_remove_message(self, agent_context, mock_memory):
        """Test removing message from context"""
        message = Message.from_text(sender="user", data="test message")

        await agent_context.add(message)

        assert len(await agent_context.get_memory()) == 1

        result = await agent_context.remove(message.id)
        assert result is True

        assert len(await agent_context.get_memory()) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_message(self, agent_context, mock_memory):
        """Test removing non-existent message from context"""
        message = Message.from_text(sender="user", data="test message")

        await agent_context.add(message)

        assert len(await agent_context.get_memory()) == 1

        result = await agent_context.remove(uuid.uuid4())
        assert result is False

        assert len(await agent_context.get_memory()) == 1

    @pytest.mark.asyncio
    async def test_get_memory(self, agent_context, mock_memory):
        """Test getting memory from context"""
        messages = [
            Message.from_text(sender="user", data="test1"),
            Message.from_text(sender="assistant", data="test2")
        ]
        await agent_context.add(messages)

        memory_list = await agent_context.get_memory()
        assert len(memory_list) == 2

        assert memory_list[0].get_elements(ContentElement)[0].data == "test1"
        assert memory_list[1].get_elements(ContentElement)[0].data == "test2"

    @pytest.mark.asyncio
    async def test_get_memory_empty(self, agent_context, mock_memory):
        """Test getting empty memory from context"""
        memory_list = await agent_context.get_memory()
        assert len(memory_list) == 0
