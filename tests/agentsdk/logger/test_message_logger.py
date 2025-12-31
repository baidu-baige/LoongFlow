# -*- coding: utf-8 -*-
""""
Unit tests for message logger.
"""
import uuid

from agentsdk.logger import message_logger as ml
from agentsdk.message import (
    Message,
    ContentElement,
    ThinkElement,
    ToolCallElement,
    ToolOutputElement,
    ToolStatus,
    MimeType,
)


def create_test_messages(num: int = 3):
    """Create a list of rich test messages with multiple element types and mime types."""
    messages = []
    for i in range(num):
        elements = [
            ContentElement(mime_type=MimeType.TEXT_PLAIN, data=f"Hello world text {i}"),
            ContentElement(mime_type=MimeType.IMAGE_JPEG, data=b"\xff\xd8\xff"),
            ContentElement(mime_type=MimeType.IMAGE_PNG, data=b"\x89PNG\r\n\x1a\n"),
            ContentElement(mime_type=MimeType.AUDIO_MPEG, data=b"FAKEAUDIO"),
            ContentElement(mime_type=MimeType.VIDEO_MP4, data=b"FAKEVIDEO"),
            ThinkElement(content=f"Thinking about something {i}..."),
            ToolCallElement(target=f"TestToolCall_{i}", arguments={"x": i, "y": i * 2}),
            ToolOutputElement(
                call_id=uuid.uuid4(),
                tool_name=f"TestToolOutput_{i}",
                status=ToolStatus.SUCCESS,
                result=[
                    ContentElement(mime_type=MimeType.TEXT_PLAIN, data=f"Result text {i}"),
                    ContentElement(mime_type=MimeType.IMAGE_PNG, data=b"\x89PNG result"),
                ],
            ),
        ]
        msg = Message(
            role="user",
            sender=f"tester_{i}",
            content=elements,
            metadata={"foo": "bar", "timestamp": str(uuid.uuid4())},
        )
        messages.append(msg)
    return messages


def test_print_multiple_messages():
    """Test printing multiple messages with print mode."""
    msgs = create_test_messages()
    ml.print_message(msgs, show_metadata=True, use_logger=False)


def test_log_multiple_messages():
    """Test logging multiple messages with logger mode."""
    msgs = create_test_messages()
    ml.print_message(msgs, show_metadata=True, use_logger=True)
