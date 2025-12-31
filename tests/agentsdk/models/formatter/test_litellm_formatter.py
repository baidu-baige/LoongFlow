# -*- coding: utf-8 -*-
"""
Unit tests for LiteLLMFormatter.

Covers:
- format_request(): text / image / audio / mixed messages
- parse_response(): dict (stream chunk) and ModelResponse (full)
- internal helpers (_convert_messages, _extract_elements_from_choices)
"""

from types import SimpleNamespace

import pytest

from agentsdk.message import Message
from agentsdk.message.elements import (
    ContentElement,
    ToolCallElement,
    ThinkElement,
    MimeType,
)
from agentsdk.models.formatter.litellm_formatter import LiteLLMFormatter
from agentsdk.models.llm_request import CompletionRequest
from agentsdk.models.llm_response import CompletionResponse


@pytest.fixture
def formatter():
    return LiteLLMFormatter()


@pytest.fixture
def sample_messages():
    return [
        Message(
            role="user",
            content=[
                ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Hello"),
                ThinkElement(content="thinking..."),
                ToolCallElement(target="get_weather", arguments={"city": "Tokyo"}),
            ],
        )
    ]


# ---------------------------------------------------------------------
# Basic request formatting tests
# ---------------------------------------------------------------------
def test_format_request_basic(formatter, sample_messages):
    """Test that format_request builds correct LiteLLM kwargs"""
    req = CompletionRequest(messages=sample_messages, temperature=0.7, top_p=0.9)
    result = formatter.format_request(
        request=req, model_name="gpt-4o-mini", stream=False
    )

    assert result["model"] == "gpt-4o-mini"
    assert result["messages"][0]["role"] == "user"
    assert any(c["type"] == "text" for c in result["messages"][0]["content"])
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9


# ---------------------------------------------------------------------
# New extended format_request tests (image, audio, text)
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_format_request_text_only(formatter):
    """Verify text-only message conversion produces correct LiteLLM payload."""
    request = CompletionRequest(
        messages=[
            Message(
                role="user",
                content=[
                    ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Hello, world!")
                ]
            )
        ],
        temperature=0.7,
    )
    kwargs = formatter.format_request(
        request=request,
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="fake-key",
        stream=False,
    )

    assert kwargs["model"] == "gpt-4o"
    assert kwargs["stream"] is False
    assert isinstance(kwargs["messages"], list)
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"][0]["text"] == "Hello, world!"


@pytest.mark.asyncio
async def test_format_request_with_image(formatter):
    request = CompletionRequest(
        messages=[
            Message(
                role="user",
                content=[
                    ContentElement(mime_type=MimeType.IMAGE_JPEG, data="https://example.com/cat.jpg")
                ]
            )
        ]
    )

    kwargs = formatter.format_request(
        request=request,
        model_name="gpt-4o-mini",
        base_url="https://api.fakeai.com/v1",
        api_key="secret",
        stream=False,
    )
    print(kwargs["messages"][0]["content"])

@pytest.mark.asyncio
async def test_format_request_with_audio(formatter):
    fake_audio_data = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10..."
    request = CompletionRequest(
        messages=[
            Message(
                role="user",
                content=[
                    ContentElement(mime_type=MimeType.AUDIO_MPEG, data=fake_audio_data)
                ]
            )
        ]
    )

    kwargs = formatter.format_request(
        request=request,
        model_name="gpt-4o-audio",
        base_url="https://api.fakeai.com/v1",
        api_key="secret",
        stream=True,
    )

    print(kwargs["messages"][0]["content"])

# ---------------------------------------------------------------------
# parse_response tests
# ---------------------------------------------------------------------
def test_parse_response_dict_chunk_content(formatter):
    """Simulate a streaming delta chunk with text content"""
    chunk = {
        "id": "stream-123",
        "choices": [{"delta": {"content": "partial text"}, "finish_reason": None}],
    }
    response = formatter.parse_response(chunk)

    assert isinstance(response, CompletionResponse)
    assert response.id == "stream-123"
    assert response.content[0].data == "partial text"
    assert response.content[0].mime_type == MimeType.TEXT_PLAIN


def test_parse_response_dict_chunk_tool_call(formatter):
    """Simulate a streaming delta chunk with a tool call"""
    chunk = {
        "id": "stream-456",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    response = formatter.parse_response(chunk)
    assert isinstance(response.content[0], ToolCallElement)
    assert response.content[0].target == "get_weather"
    assert response.content[0].arguments == {"city": "Tokyo"}


def test_parse_response_full_model_response(formatter):
    """Simulate parsing a complete ModelResponse object"""
    mock_response = SimpleNamespace(
        id="resp-1",
        choices=[
            SimpleNamespace(
                message={
                    "content": "This is a test response.",
                    "tool_calls": [
                        {"function": {"name": "mock_tool", "arguments": {"x": 1}}}
                    ],
                },
                finish_reason="stop",
            )
        ],
        usage={"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
    )

    result = formatter._parse_full_response(mock_response)
    assert result.id == "resp-1"
    assert isinstance(result.content[0], ContentElement)
    assert isinstance(result.content[1], ToolCallElement)
    assert result.usage.total_tokens == 15
    assert result.finish_reason == "stop"


def test_parse_response_unknown_type(formatter):
    """Passing an unsupported type should yield an error response"""
    class Dummy: ...
    dummy = Dummy()
    response = formatter.parse_response(dummy)
    assert response.error_code == "unknown_response_type"
    assert "Unsupported" in response.error_message


# ---------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------
def test_extract_elements_from_choices_text_and_tool(formatter):
    """Directly test _extract_elements_from_choices helper"""
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message={
                    "content": "Hi there!",
                    "tool_calls": [
                        {"function": {"name": "do_stuff", "arguments": {"arg1": 1}}}
                    ],
                }
            )
        ]
    )
    elements = formatter._extract_elements_from_choices(raw)
    assert isinstance(elements[0], ContentElement)
    assert isinstance(elements[1], ToolCallElement)
    assert elements[1].target == "do_stuff"


def test_convert_messages_with_mixed_elements(formatter, sample_messages):
    """Ensure mixed content types are correctly converted"""
    converted = formatter._convert_messages(sample_messages)
    msg = converted[0]
    assert msg["role"] == "user"
    types = [c["type"] for c in msg["content"]]
    assert {"text", "tool_call"} & set(types)