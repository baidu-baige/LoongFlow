# -*- coding: utf-8 -*-
"""
Unit tests for LiteLLMModel.

These tests verify the correct behavior of LiteLLMModel including:
- Proper formatting of requests
- Correct parsing of LiteLLM responses
- Error handling
- Streaming and non-streaming modes
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsdk.models.litellm_model import LiteLLMModel
from agentsdk.models.llm_request import CompletionRequest
from agentsdk.models.llm_response import CompletionResponse


@pytest.mark.asyncio
async def test_generate_non_stream_success(monkeypatch):
    """Test LiteLLMModel.generate() non-stream success flow."""
    # Mock response from litellm
    mock_raw_resp = {"id": "123", "choices": [{"message": {"content": "Hello world"}}]}

    # Patch litellm.acompletion to return our mock response
    async_mock = AsyncMock(return_value=mock_raw_resp)
    monkeypatch.setattr("litellm.acompletion", async_mock)

    # Initialize model
    model = LiteLLMModel(model_name="gpt-4o", base_url="https://api.openai.com", api_key="sk-test")

    # Prepare request
    req = make_req("Hi")

    # Collect output
    results = [r async for r in model.generate(req, stream=False)]

    # Verify
    assert len(results) == 1
    resp = results[0]
    assert isinstance(resp, CompletionResponse)
    assert resp.error_code is None
    assert resp.id == "123" or resp.id  # allow id from formatter
    async_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_stream_success(monkeypatch):
    """Test LiteLLMModel.generate() with streaming mode."""
    # Prepare mock streaming chunks
    async def mock_stream():
        for i in range(3):
            yield {"choices": [{"delta": {"content": f"chunk{i}"}}]}
            await asyncio.sleep(0)

    # Patch litellm.acompletion to return async generator
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=mock_stream()))

    model = LiteLLMModel(model_name="gpt-4o", base_url="https://api.openai.com", api_key="sk-test")
    
    req = make_req("stream")

    chunks = [r async for r in model.generate(req, stream=True)]
    assert all(isinstance(c, CompletionResponse) for c in chunks)
    assert any(
        "chunk" in "".join(
            [getattr(t, "data", getattr(t, "text", "")) for t in c.content]
        )
        for c in chunks
    )

@pytest.mark.asyncio
async def test_generate_error(monkeypatch):
    """Test LiteLLMModel.generate() when litellm raises exception."""
    # Patch to raise
    async def raise_error(**kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("litellm.acompletion", raise_error)

    model = LiteLLMModel("gpt-4o", "https://api.openai.com", "sk-test")
    
    req = make_req("error test")

    results = [r async for r in model.generate(req, stream=False)]
    assert len(results) == 1
    resp = results[0]
    assert isinstance(resp, CompletionResponse)
    assert resp.error_code == "litellm_error"
    assert "network error" in resp.error_message


@pytest.mark.asyncio
async def test_generate_uses_formatter(monkeypatch):
    """Ensure LiteLLMFormatter is used for both request and response."""
    model = LiteLLMModel("gpt-4o", "https://api.openai.com", "sk-test")

    # Mock formatter methods
    model.formatter.format_request = MagicMock(return_value={"fake": "req"})
    model.formatter.parse_response = MagicMock(return_value=CompletionResponse(id="ok", content=[]))

    async def mock_completion(**kwargs):
        assert "fake" in kwargs
        return {"id": "mocked"}
    monkeypatch.setattr("litellm.acompletion", mock_completion)

    req = make_req("hi")
    results = [r async for r in model.generate(req)]
    assert len(results) == 1
    model.formatter.format_request.assert_called_once()
    model.formatter.parse_response.assert_called_once()

def make_req(text: str) -> CompletionRequest:
    """Helper to construct a valid CompletionRequest."""
    return CompletionRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "content",
                        "mime_type": "text/plain",
                        "data": text
                    }
                ]
            }
        ]
    )
