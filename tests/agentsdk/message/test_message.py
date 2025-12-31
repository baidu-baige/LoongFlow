#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for agentsdk.message module
"""

import uuid
from datetime import datetime

import pytest

from agentsdk.message import (
    ContentElement,
    Message,
    MimeType,
    Role,
    ThinkElement,
    ToolCallElement,
    ToolOutputElement,
)


class TestRole:
    """Test cases for Role enumeration"""

    def test_role_values(self):
        """Test that Role enum has correct values"""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"


class TestMimeType:
    """Test cases for MimeType enumeration"""

    def test_mime_type_values(self):
        """Test that MimeType enum has correct values"""
        assert MimeType.TEXT_PLAIN == "text/plain"
        assert MimeType.IMAGE_JPEG == "image/jpeg"
        assert MimeType.IMAGE_PNG == "image/png"
        assert MimeType.AUDIO_MPEG == "audio/mpeg"
        assert MimeType.VIDEO_MP4 == "video/mp4"


class TestContentElement:
    """Test cases for ContentElement"""

    def test_content_element_creation(self):
        """Test basic ContentElement creation"""
        element = ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Hello, World!")

        assert element.type == "content"
        assert element.mime_type == MimeType.TEXT_PLAIN
        assert element.data == "Hello, World!"
        assert element.metadata == {}

    def test_content_element_with_metadata(self):
        """Test ContentElement with metadata"""
        metadata = {"language": "en", "encoding": "utf-8"}
        element = ContentElement(
            mime_type="text/plain", data="Test data", metadata=metadata
        )

        assert element.metadata == metadata

    def test_content_element_get_content(self):
        """Test get_content method of ContentElement"""
        element = ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Test content")

        content = element.get_content()
        expected = {
            "type": "content",
            "content": {"mime_type": MimeType.TEXT_PLAIN, "data": "Test content"},
        }
        assert content == expected


class TestToolCallElement:
    """Test cases for ToolCallElement"""

    def test_tool_call_element_creation(self):
        """Test basic ToolCallElement creation"""
        call_id = uuid.uuid4()
        arguments = {"city": "Beijing", "unit": "celsius"}
        element = ToolCallElement(
            call_id=call_id, target="get_weather", arguments=arguments
        )

        assert element.type == "tool_call"
        assert element.call_id == call_id
        assert element.target == "get_weather"
        assert element.arguments == arguments
        assert element.metadata == {}

    def test_tool_call_element_get_content(self):
        """Test get_content method of ToolCallElement"""
        call_id = uuid.uuid4()
        arguments = {"param": "value"}
        element = ToolCallElement(
            call_id=call_id, target="test_tool", arguments=arguments
        )

        content = element.get_content()
        expected = {
            "type": "tool_call",
            "content": {"target": "test_tool", "arguments": arguments},
        }
        assert content == expected


class TestToolOutputElement:
    """Test cases for ToolOutputElement"""

    def test_tool_output_element_creation(self):
        """Test basic ToolOutputElement creation"""
        call_id = uuid.uuid4()
        result_elements = [
            ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Result 1"),
            ContentElement(mime_type=MimeType.APPLICATION_JSON, data={"key": "value"}),
        ]
        element = ToolOutputElement(
            call_id=call_id, status="success", result=result_elements
        )

        assert element.type == "tool_output"
        assert element.call_id == call_id
        assert element.status == "success"
        assert element.result == result_elements
        assert element.metadata == {}

    def test_tool_output_element_get_content(self):
        """Test get_content method of ToolOutputElement"""
        call_id = uuid.uuid4()
        result_elements = [
            ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Test result")
        ]
        element = ToolOutputElement(
            call_id=call_id, status="success", result=result_elements
        )

        content = element.get_content()
        expected = {
            "type": "tool_output",
            "content": {
                "status": "success",
                "result": [
                    {
                        "type": "content",
                        "content": {"mime_type": "text/plain", "data": "Test result"},
                    }
                ],
            },
        }
        assert content == expected


class TestThinkElement:
    """Test cases for ThinkElement"""

    def test_think_element_creation(self):
        """Test basic ThinkElement creation"""
        element = ThinkElement(content="I need to think about this carefully")

        assert element.type == "think"
        assert element.content == "I need to think about this carefully"
        assert element.metadata == {}

    def test_think_element_get_content(self):
        """Test get_content method of ThinkElement"""
        element = ThinkElement(
            content={"reasoning": "step by step", "conclusion": "final answer"}
        )

        content = element.get_content()
        expected = {
            "type": "think",
            "content": {
                "content": {"reasoning": "step by step", "conclusion": "final answer"}
            },
        }
        assert content == expected


class TestMessage:
    """Test cases for Message class"""

    def test_message_creation_with_minimal_fields(self):
        """Test Message creation with minimal required fields"""
        message = Message(
            role=Role.USER,
            content=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Hello")],
        )

        assert isinstance(message.id, uuid.UUID)
        assert isinstance(message.timestamp, datetime)
        assert message.role == Role.USER
        assert message.sender == ""
        assert message.trace_id == ""
        assert message.conversation_id == ""
        assert message.metadata == {}
        assert len(message.content) == 1
        assert message.content[0].data == "Hello"

    def test_message_creation_with_all_fields(self):
        """Test Message creation with all fields"""
        message_id = uuid.uuid4()
        timestamp = datetime.utcnow()
        content_elements = [
            ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Message text"),
            ThinkElement(content="Internal thought"),
        ]

        message = Message(
            id=message_id,
            timestamp=timestamp,
            trace_id="trace-123",
            conversation_id="conv-456",
            role=Role.ASSISTANT,
            sender="WeatherAgent",
            metadata={"tokens": 150, "model": "gpt-4"},
            content=content_elements,
        )

        assert message.id == message_id
        assert message.timestamp == timestamp
        assert message.trace_id == "trace-123"
        assert message.conversation_id == "conv-456"
        assert message.role == Role.ASSISTANT
        assert message.sender == "WeatherAgent"
        assert message.metadata == {"tokens": 150, "model": "gpt-4"}
        assert message.content == content_elements

    def test_message_creation_with_string_role(self):
        """Test Message creation with string role instead of enum"""
        message = Message(
            role="custom_role",
            content=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Test")],
        )

        assert message.role == "custom_role"

    def test_message_to_dict(self):
        """Test serialization of Message to dictionary"""
        message = Message(
            role=Role.USER,
            content=[
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Test message")
            ],
        )

        message_dict = message.to_dict()

        assert "id" in message_dict
        assert "timestamp" in message_dict
        assert message_dict["role"] == "user"
        assert message_dict["content"][0]["type"] == "content"
        assert message_dict["content"][0]["mime_type"] == "text/plain"
        assert message_dict["content"][0]["data"] == "Test message"

    def test_message_from_dict(self):
        """Test deserialization of Message from dictionary"""
        message_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "role": "user",
            "content": [
                {
                    "type": "content",
                    "mime_type": "text/plain",
                    "data": "Hello from dict",
                }
            ],
        }

        message = Message.from_dict(message_data)

        assert message.role == Role.USER
        assert len(message.content) == 1
        assert message.content[0].type == "content"
        assert message.content[0].data == "Hello from dict"

    def test_message_get_elements(self):
        """Test filtering elements by type"""
        content_element = ContentElement(
            mime_type=MimeType.TEXT_PLAIN, data="Text content"
        )
        think_element = ThinkElement(content="Internal thought")
        tool_call_element = ToolCallElement(target="test_tool", arguments={})

        message = Message(
            role=Role.ASSISTANT,
            content=[content_element, think_element, tool_call_element],
        )

        # Test getting ContentElement
        content_elements = message.get_elements(ContentElement)
        assert len(content_elements) == 1
        assert content_elements[0] == content_element

        # Test getting ThinkElement
        think_elements = message.get_elements(ThinkElement)
        assert len(think_elements) == 1
        assert think_elements[0] == think_element

        # Test getting ToolCallElement
        tool_call_elements = message.get_elements(ToolCallElement)
        assert len(tool_call_elements) == 1
        assert tool_call_elements[0] == tool_call_element

        # Test getting non-existent element type
        tool_output_elements = message.get_elements(ToolOutputElement)
        assert len(tool_output_elements) == 0

    def test_message_inequality(self):
        """Test message inequality with different IDs"""
        message1 = Message(
            role=Role.USER,
            content=[
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Same content")
            ],
        )
        message2 = Message(
            role=Role.USER,
            content=[
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Same content")
            ],
        )

        # Messages with different IDs should not be equal
        assert message1 != message2


class TestElementDiscrimination:
    """Test cases for element type discrimination"""

    def test_element_discrimination_in_message(self):
        """Test that elements are properly discriminated in Message content"""
        elements = [
            {"type": "content", "mime_type": "text/plain", "data": "Text"},
            {"type": "think", "content": "Thought"},
            {"type": "tool_call", "target": "tool", "arguments": {}},
        ]

        message_data = {"role": "user", "content": elements}

        message = Message.from_dict(message_data)

        assert len(message.content) == 3
        assert isinstance(message.content[0], ContentElement)
        assert isinstance(message.content[1], ThinkElement)
        assert isinstance(message.content[2], ToolCallElement)


if __name__ == "__main__":
    pytest.main([__file__])
