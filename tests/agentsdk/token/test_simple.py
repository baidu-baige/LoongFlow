#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for SimpleTokenCounter class
"""
import json
import uuid

import pytest

from agentsdk.message import ContentElement, Message, MimeType, Role
from agentsdk.token.simple import SimpleTokenCounter


def create_test_message(msg_id=None, content="Hello, world!", role=Role.USER):
    """Helper function to create a test message"""
    if msg_id is None:
        msg_id = uuid.uuid4()
    return Message(
        id=msg_id,
        role=role,
        content=[ContentElement(mime_type="text/plain", data=content)],
    )


class TestSimpleTokenCounter:
    """Test cases for SimpleTokenCounter class"""

    @pytest.fixture
    def token_counter(self):
        return SimpleTokenCounter()

    @pytest.mark.asyncio
    async def test_count_single_message(self, token_counter):
        """Test counting tokens for a single message"""
        message = create_test_message(content="Hello, world!")

        result = await token_counter.count([message])

        # Calculate expected token count
        expected_json = json.dumps(
            {
                "role": message.role,
                "content": [elem.get_content() for elem in message.content],
            },
            ensure_ascii=False,
        )
        expected_tokens = len(expected_json) // 4

        assert result == expected_tokens
        assert result > 0  # Should have some tokens

    @pytest.mark.asyncio
    async def test_count_multiple_messages(self, token_counter):
        """Test counting tokens for multiple messages"""
        message1 = create_test_message(content="First message")
        message2 = create_test_message(content="Second message")
        message3 = create_test_message(content="Third message")

        result = await token_counter.count([message1, message2, message3])

        # Calculate expected token count for each message
        total_tokens = 0
        for msg in [message1, message2, message3]:
            expected_json = json.dumps(
                {
                    "role": msg.role,
                    "content": [elem.get_content() for elem in msg.content],
                },
                ensure_ascii=False,
            )
            total_tokens += len(expected_json) // 4

        assert result == total_tokens
        assert result > 0

    @pytest.mark.asyncio
    async def test_count_empty_message_list(self, token_counter):
        """Test counting tokens for an empty message list"""
        result = await token_counter.count([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_message_with_different_roles(self, token_counter):
        """Test counting tokens for messages with different roles"""
        user_message = create_test_message(content="User message", role=Role.USER)
        assistant_message = create_test_message(
            content="Assistant response", role=Role.ASSISTANT
        )
        system_message = create_test_message(content="System prompt", role=Role.SYSTEM)

        result = await token_counter.count(
            [user_message, assistant_message, system_message]
        )

        # All should have different token counts due to different roles
        assert result > 0

    @pytest.mark.asyncio
    async def test_count_message_with_multiple_content_elements(self, token_counter):
        """Test counting tokens for a message with multiple content elements"""
        message = Message(
            id=uuid.uuid4(),
            role=Role.USER,
            content=[
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="First part"),
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Second part"),
                ContentElement(
                    mime_type=MimeType.APPLICATION_JSON, data={"key": "value"}
                ),
            ],
        )

        result = await token_counter.count([message])

        # Should handle multiple content elements correctly
        assert result > 0

    @pytest.mark.asyncio
    async def test_count_message_with_complex_content(self, token_counter):
        """Test counting tokens for a message with complex content"""
        complex_data = {
            "text": "Hello, world!",
            "metadata": {"timestamp": "2024-01-01", "source": "test"},
            "nested": {"level1": {"level2": "deep"}},
        }

        message = Message(
            id=uuid.uuid4(),
            role="system",
            content=[
                ContentElement(mime_type=MimeType.APPLICATION_JSON, data=complex_data)
            ],
        )

        result = await token_counter.count([message])

        # Complex JSON should result in more tokens
        assert result > 10  # Should have reasonable number of tokens

    @pytest.mark.asyncio
    async def test_count_message_with_unicode_content(self, token_counter):
        """Test counting tokens for a message with unicode content"""
        message = create_test_message(content="Hello World！🌍")

        result = await token_counter.count([message])

        # Unicode content should be handled correctly
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__])
