# -*- coding: utf-8 -*-
"""
Test cases for default actor implementations
"""

import pytest
from pydantic import BaseModel, Field

from agentsdk.memory.grade import GradeMemory
from agentsdk.message import ContentElement, ToolCallElement, ToolOutputElement
from agentsdk.tools import FunctionTool, Toolkit
from evolux.react import AgentContext
from evolux.react.components import ParallelActor, SequenceActor


class SimpleRunArgs(BaseModel):
    name: str = Field(..., description="Name of actor")


def simple_run(name: str):
    if name == "error":
        raise ValueError(f'{name} exception')
    return name


@pytest.fixture
def mock_context():
    toolkit = Toolkit()
    toolkit.register_tool(FunctionTool(
            func=simple_run,
            name="simple_run1",
            args_schema=SimpleRunArgs,
            description="Simple run",
    ))
    toolkit.register_tool(FunctionTool(
            func=simple_run,
            name="simple_run2",
            args_schema=SimpleRunArgs,
            description="Simple run",
    ))
    return AgentContext(
            memory=GradeMemory.create_default(None),
            toolkit=toolkit,
    )


@pytest.fixture
def mock_tool_calls():
    return [
        ToolCallElement(target="simple_run1", arguments={"name": "name1"}),
        ToolCallElement(target="simple_run2", arguments={"name": "name2"})
    ]


class TestSequenceActor:
    """Test cases for SequenceActor"""

    @pytest.fixture
    def sequence_actor(self):
        return SequenceActor()

    @pytest.mark.asyncio
    async def test_act_empty_tool_calls(self, sequence_actor, mock_context):
        """Test act with empty tool calls list"""
        result = await sequence_actor.act(mock_context, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_act_single_tool_call_success(self, sequence_actor, mock_context, mock_tool_calls):
        """Test act with single successful tool call"""
        result = await sequence_actor.act(mock_context, mock_tool_calls[:1])

        assert len(result) == 1
        assert len(result[0].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[0].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[0].get_elements(ToolOutputElement)[0].result[0].data == "name1"

    @pytest.mark.asyncio
    async def test_act_multiple_tool_calls_sequential(self, sequence_actor, mock_context, mock_tool_calls):
        """Test act with multiple tool calls executed sequentially"""
        result = await sequence_actor.act(mock_context, mock_tool_calls)

        assert len(result) == 2
        assert len(result[0].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[0].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[0].get_elements(ToolOutputElement)[0].result[0].data == "name1"

        assert len(result[1].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[1].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[1].get_elements(ToolOutputElement)[0].result[0].data == "name2"

    @pytest.mark.asyncio
    async def test_act_tool_call_error(self, sequence_actor, mock_context):
        """Test act with tool call that returns error"""
        mock_tool_calls = [
            ToolCallElement(target="simple_run1", arguments={"name": "error"}),
        ]
        result = await sequence_actor.act(mock_context, mock_tool_calls)

        assert len(result) == 1
        assert len(result[0].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[0].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[0].get_elements(ToolOutputElement)[0].result[0].data


class TestParallelActor:
    """Test cases for ParallelActor"""

    @pytest.fixture
    def parallel_actor(self):
        return ParallelActor()

    @pytest.mark.asyncio
    async def test_act_empty_tool_calls(self, parallel_actor, mock_context):
        """Test act with empty tool calls list"""
        result = await parallel_actor.act(mock_context, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_act_single_tool_call_success(self, parallel_actor, mock_context, mock_tool_calls):
        """Test act with single successful tool call"""
        result = await parallel_actor.act(mock_context, mock_tool_calls[:1])

        assert len(result) == 1
        assert len(result[0].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[0].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[0].get_elements(ToolOutputElement)[0].result[0].data == "name1"

    @pytest.mark.asyncio
    async def test_act_multiple_tool_calls_parallel(self, parallel_actor, mock_context, mock_tool_calls):
        """Test act with multiple tool calls executed in parallel"""
        result = await parallel_actor.act(mock_context, mock_tool_calls)

        assert len(result) == 2

        output = set()
        for msg in result:
            assert len(msg.get_elements(ToolOutputElement)) == 1
            assert isinstance(msg.get_elements(ToolOutputElement)[0].result[0], ContentElement)
            output.add(msg.get_elements(ToolOutputElement)[0].result[0].data)

        assert output == {"name1", "name2"}

    @pytest.mark.asyncio
    async def test_act_tool_call_error(self, parallel_actor, mock_context):
        """Test act with tool call that returns error"""
        mock_tool_calls = [
            ToolCallElement(target="simple_run1", arguments={"name": "error"}),
        ]
        result = await parallel_actor.act(mock_context, mock_tool_calls)

        assert len(result) == 1
        assert len(result[0].get_elements(ToolOutputElement)) == 1
        assert isinstance(result[0].get_elements(ToolOutputElement)[0].result[0], ContentElement)
        assert result[0].get_elements(ToolOutputElement)[0].result[0].data


if __name__ == "__main__":
    pytest.main([__file__])
