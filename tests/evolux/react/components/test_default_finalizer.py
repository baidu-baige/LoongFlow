# -*- coding: utf-8 -*-
"""
Test cases for DefaultFinalizer
"""
import uuid
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from agentsdk.message import (
    ContentElement,
    MimeType,
    ToolCallElement,
    ToolOutputElement,
    ToolStatus,
)
from agentsdk.tools import FunctionTool
from evolux.react.components import DefaultFinalizer


class TestResponse(BaseModel):
    """Test response model"""

    answer: str = Field(description="The answer")


class TestDefaultFinalizer:
    """Test cases for DefaultFinalizer"""

    @pytest.fixture
    def mock_model(self):
        model = AsyncMock()
        model.generate = AsyncMock()
        return model

    @pytest.fixture
    def default_finalizer(self, mock_model):
        return DefaultFinalizer(
            model=mock_model,
            summarize_prompt="Test summarize prompt",
            output_schema=TestResponse,
            tool_name="test_final_answer",
            tool_description="Test final answer tool",
        )

    @pytest.fixture
    def default_finalizer_no_schema(self, mock_model):
        return DefaultFinalizer(
            model=mock_model, summarize_prompt="Test summarize prompt"
        )

    def test_answer_schema_property(self, default_finalizer):
        """Test answer_schema property"""
        schema = default_finalizer.answer_schema

        assert isinstance(schema, FunctionTool)
        assert schema.name == "test_final_answer"
        assert schema.description == "Test final answer tool"

        # Test that the function can be called
        result = schema.func(response="test response", answer="test answer")
        assert result["response"] == "test response"
        assert "answer" in result["output_schema"]

    def test_answer_schema_no_response_field(self, default_finalizer_no_schema):
        """Test answer_schema when base schema doesn't have response field"""
        schema = default_finalizer_no_schema.answer_schema

        assert isinstance(schema, FunctionTool)

        # Test that the function adds response field
        result = schema.func(response="test response")
        assert result["response"] == "test response"

    def test_answer_schema_validation_error(self, default_finalizer):
        """Test answer_schema with validation error"""
        schema = default_finalizer.answer_schema

        # Test with invalid arguments that should cause validation error
        with pytest.raises(Exception) as exc_info:
            schema.func(invalid_arg="value")
        assert "response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_answer_success(self, default_finalizer):
        """Test resolve_answer with successful final answer"""
        call_id = uuid.uuid4()
        tool_call = ToolCallElement(
            call_id=call_id, target="test_final_answer", arguments={}
        )
        content_element = ContentElement(
            data={
                "response": "final answer",
                "output_schema": {"answer": "test answer"},
            },
            mime_type=MimeType.APPLICATION_JSON,
        )
        tool_output = ToolOutputElement(
            call_id=call_id, status=ToolStatus.SUCCESS, result=[content_element]
        )

        result = await default_finalizer.resolve_answer(tool_call, tool_output)

        assert result is not None
        assert result.sender == "finalizer"
        assert result.get_elements(ContentElement)[0].data == {"answer": "test answer"}

    @pytest.mark.asyncio
    async def test_resolve_answer_wrong_tool(self, default_finalizer):
        """Test resolve_answer with non-final-answer tool"""
        call_id = uuid.uuid4()
        tool_call = ToolCallElement(call_id=call_id, target="other_tool", arguments={})
        tool_output = ToolOutputElement(
            call_id=call_id, status=ToolStatus.SUCCESS, result=[]
        )

        result = await default_finalizer.resolve_answer(tool_call, tool_output)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_answer_failed_status(self, default_finalizer):
        """Test resolve_answer with failed tool status"""
        call_id = uuid.uuid4()
        tool_call = ToolCallElement(
            call_id=call_id, target="test_final_answer", arguments={}
        )
        tool_output = ToolOutputElement(
            call_id=call_id, status=ToolStatus.ERROR, result=[]
        )

        result = await default_finalizer.resolve_answer(tool_call, tool_output)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_answer_empty_result(self, default_finalizer):
        """Test resolve_answer with empty result"""
        call_id = uuid.uuid4()
        tool_call = ToolCallElement(
            call_id=call_id, target="test_final_answer", arguments={}
        )
        tool_output = ToolOutputElement(
            call_id=call_id, status=ToolStatus.SUCCESS, result=[]
        )

        result = await default_finalizer.resolve_answer(tool_call, tool_output)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_answer_invalid_result_type(self, default_finalizer):
        """Test resolve_answer with invalid result type"""
        call_id = uuid.uuid4()
        tool_call = ToolCallElement(
            call_id=call_id, target="test_final_answer", arguments={}
        )
        # Create a result that's not a ContentElement
        tool_output = ToolOutputElement(
            call_id=call_id,
            status=ToolStatus.SUCCESS,
            result=[ContentElement(mime_type=MimeType.APPLICATION_JSON, data="test")],
        )

        result = await default_finalizer.resolve_answer(tool_call, tool_output)

        assert result is None

    @pytest.mark.asyncio
    async def test_summarize_on_exceed_success(self, default_finalizer, mock_model):
        """Test summarize_on_exceed method"""
        # todo we need to use real model to finish tests
