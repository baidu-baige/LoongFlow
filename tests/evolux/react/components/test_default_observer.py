# -*- coding: utf-8 -*-
"""
Test cases for DefaultObserver
"""

import pytest

from agentsdk.memory.grade import GradeMemory
from agentsdk.message import Message, Role
from agentsdk.tools import Toolkit
from evolux.react import AgentContext
from evolux.react.components import DefaultObserver


@pytest.fixture
def mock_context():
    return AgentContext(
            memory=GradeMemory.create_default(None),
            toolkit=Toolkit()

    )


class TestDefaultObserver:
    """Test cases for DefaultObserver"""

    @pytest.fixture
    def default_observer(self):
        return DefaultObserver()

    @pytest.fixture
    def mock_tool_outputs(self):
        return [
            Message.from_text(sender="tool1", data="output1", role=Role.TOOL),
            Message.from_text(sender="tool2", data="output2", role=Role.TOOL)
        ]

    @pytest.mark.asyncio
    async def test_observe_returns_none(self, default_observer, mock_context, mock_tool_outputs):
        """Test that observe method always returns None"""
        result = await default_observer.observe(mock_context, mock_tool_outputs)

        assert result is None

    @pytest.mark.asyncio
    async def test_observe_empty_tool_outputs(self, default_observer, mock_context):
        """Test observe with empty tool outputs list"""
        result = await default_observer.observe(mock_context, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_observe_single_tool_output(self, default_observer, mock_context):
        """Test observe with single tool output"""
        tool_outputs = [Message.from_text(sender="tool", data="single output", role=Role.TOOL)]

        result = await default_observer.observe(mock_context, tool_outputs)

        assert result is None

    @pytest.mark.asyncio
    async def test_observe_multiple_tool_outputs(self, default_observer, mock_context, mock_tool_outputs):
        """Test observe with multiple tool outputs"""
        result = await default_observer.observe(mock_context, mock_tool_outputs)

        assert result is None
