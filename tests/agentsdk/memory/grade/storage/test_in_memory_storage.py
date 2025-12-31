#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test file for in memory storage
"""
import uuid

import pytest

from agentsdk.memory.grade.storage import InMemoryStorage
from agentsdk.message import ContentElement, Message


@pytest.fixture
def storage() -> InMemoryStorage:
    """Provides a clean InMemoryStorage instance for each test."""
    return InMemoryStorage()


class TestInMemoryStorage:
    @pytest.mark.asyncio
    async def test_initial_state(self, storage: InMemoryStorage):
        """Tests that the storage is empty upon creation."""
        assert await storage.get_size() == 0
        assert await storage.get_all() == []

    @pytest.mark.asyncio
    async def test_add_and_get_single_message(self, storage: InMemoryStorage):
        """Tests adding and retrieving a single message."""
        msg_id = uuid.uuid4()
        msg = Message(id=msg_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Hello, world!"
        )])

        await storage.add(msg)

        assert await storage.get_size() == 1
        retrieved = await storage.get(msg_id)
        assert retrieved is not None
        assert retrieved.id == msg_id
        assert len(retrieved.get_elements(ContentElement)) == 1
        assert retrieved.get_elements(ContentElement)[0].data == "Hello, world!"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, storage: InMemoryStorage):
        """Tests adding a list of messages and verifies order."""
        msg1_id, msg2_id = uuid.uuid4(), uuid.uuid4()
        msg1 = Message(id=msg1_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="First"
        )])

        msg2 = Message(id=msg2_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Second"
        )])

        await storage.add([msg1, msg2])

        assert await storage.get_size() == 2
        all_msgs = await storage.get_all()
        assert len(all_msgs) == 2
        assert len(all_msgs[0].get_elements(ContentElement)) == 1
        assert all_msgs[0].get_elements(ContentElement)[0].data == "First"
        assert len(all_msgs[1].get_elements(ContentElement)) == 1
        assert all_msgs[1].get_elements(ContentElement)[0].data == "Second"

    @pytest.mark.asyncio
    async def test_remove_message(self, storage: InMemoryStorage):
        """Tests removing a message."""
        msg1_id, msg2_id = uuid.uuid4(), uuid.uuid4()
        msg1 = Message(id=msg1_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="First"
        )])

        msg2 = Message(id=msg2_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Second"
        )])

        await storage.add([msg1, msg2])

        assert await storage.get_size() == 2

        # Remove the first message
        was_removed = await storage.remove(msg1_id)
        assert was_removed is True
        assert await storage.get_size() == 1
        assert await storage.get(msg1_id) is None
        assert await storage.get(msg2_id) is not None

        # Try to remove a non-existent message
        was_removed_again = await storage.remove(uuid.uuid4())
        assert was_removed_again is False

    @pytest.mark.asyncio
    async def test_clear_storage(self, storage: InMemoryStorage):
        """Tests clearing the storage."""
        msg_id = uuid.uuid4()
        msg = Message(id=msg_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Hello, world!"
        )])
        await storage.add(msg)
        assert await storage.get_size() == 1

        await storage.clear()
        assert await storage.get_size() == 0
        assert await storage.get_all() == []


if __name__ == "__main__":
    pytest.main([__file__])
