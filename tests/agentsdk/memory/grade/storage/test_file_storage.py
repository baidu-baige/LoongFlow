#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file storage test
"""

import json
import uuid

import pytest

from agentsdk.memory.grade.storage import FileStorage
from agentsdk.message import ContentElement, Message

TEST_FILE_PATH = "./test_storage.json"


class TestFileStorage:
    @pytest.mark.asyncio
    async def test_add_and_get_message(self, fs):
        """
        Tests basic add and get functionality.
        """
        # 1. Arrange: Create a storage instance
        storage = FileStorage(TEST_FILE_PATH)

        # Create a sample message
        msg_id = uuid.uuid4()
        msg = Message(id=msg_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Hello, world!"
        )])

        # 2. Act: Add the message
        await storage.add(msg)

        # 3. Assert: Retrieve and verify the message
        retrieved_msg = await storage.get(msg_id)
        assert retrieved_msg is not None
        assert retrieved_msg.id == msg_id
        assert len(retrieved_msg.get_elements(ContentElement)) == 1
        assert retrieved_msg.get_elements(ContentElement)[0].data == "Hello, world!"

        # Verify size and get_all
        all_msgs = await storage.get_all()
        assert len(all_msgs) == 1
        assert await storage.get_size() == 1
        assert all_msgs[0].id == msg_id

    @pytest.mark.asyncio
    async def test_lazy_loading(self, fs):
        """
        Tests that the data file is loaded lazily on the first operation.
        """
        # 1. Arrange: Pre-populate a storage file in the fake filesystem
        msg_id = uuid.UUID("b88ff338-4dc5-4e89-9e5f-c4c06164e2ff")
        msg_data = [
            {
                'id': 'b88ff338-4dc5-4e89-9e5f-c4c06164e2ff',
                'timestamp': '2025-10-27T06:52:21.708311',
                'role': 'user',
                'content': [
                    {'type': 'content', 'mime_type': 'text/plain', 'data': 'Hello, world!'}
                ]
            }

        ]
        # pyfakefs allows creating a file and its parent directories
        fs.create_file(TEST_FILE_PATH, contents=json.dumps(msg_data))

        # 2. Act: Create the storage instance
        storage = FileStorage(TEST_FILE_PATH)

        # 3. Assert: Check that the cache is not yet loaded
        assert storage._cache is None

        # 4. Act: Perform an action that triggers a load
        size = await storage.get_size()

        # 5. Assert: Check that the cache is now loaded and correct
        assert storage._cache is not None
        assert size == 1

        retrieved_msg = await storage.get(msg_id)
        assert retrieved_msg.id == msg_id
        assert len(retrieved_msg.get_elements(ContentElement)) == 1
        assert retrieved_msg.get_elements(ContentElement)[0].data == "Hello, world!"

    @pytest.mark.asyncio
    async def test_remove_message(self, fs):
        """Tests that a message can be successfully removed."""
        storage = FileStorage(TEST_FILE_PATH)
        msg1_id, msg2_id = uuid.uuid4(), uuid.uuid4()
        msg1 = Message(id=msg1_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Message 1"
        )])

        msg2 = Message(id=msg2_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Message 2"
        )])

        await storage.add([msg1, msg2])
        assert await storage.get_size() == 2

        # Remove one message
        was_removed = await storage.remove(msg1_id)
        assert was_removed is True
        assert await storage.get_size() == 1

        # Verify the correct message was removed
        assert await storage.get(msg1_id) is None
        assert (await storage.get(msg2_id)).id == msg2_id

        # Test removing a non-existent message
        was_removed_again = await storage.remove(uuid.uuid4())
        assert was_removed_again is False

    @pytest.mark.asyncio
    async def test_clear_storage(self, fs):
        """Tests that clearing the storage removes all data and the file."""
        storage = FileStorage(TEST_FILE_PATH)
        await storage.add(Message(role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Message 2"
        )]))

        # Verify file exists after adding
        assert fs.exists(TEST_FILE_PATH)

        await storage.clear()

        assert await storage.get_size() == 0
        assert not fs.exists(TEST_FILE_PATH)

    @pytest.mark.asyncio
    async def test_storage_with_empty_file(self, fs):
        """Tests that the storage works correctly with a non-existent file."""
        storage = FileStorage(TEST_FILE_PATH)

        # Should not raise any error
        assert await storage.get_size() == 0
        all_msgs = await storage.get_all()
        assert all_msgs == []

    @pytest.mark.asyncio
    async def test_persistence(self, fs):
        """Tests that data persists between different storage instances."""
        msg_id = uuid.uuid4()
        msg = Message(id=msg_id, role="user", content=[ContentElement(
                mime_type="text/plain",
                data="Hello, world!"
        )])

        # Instance 1: Add a message and it gets saved to the fake file
        storage1 = FileStorage(TEST_FILE_PATH)
        await storage1.add(msg)

        # Instance 2: Should load the data from the file created by instance 1
        storage2 = FileStorage(TEST_FILE_PATH)
        retrieved_msg = await storage2.get(msg_id)

        assert retrieved_msg is not None
        assert retrieved_msg.id == msg_id
        assert len(retrieved_msg.get_elements(ContentElement)) == 1
        assert retrieved_msg.get_elements(ContentElement)[0].data == "Hello, world!"


if __name__ == "__main__":
    pytest.main([__file__])
