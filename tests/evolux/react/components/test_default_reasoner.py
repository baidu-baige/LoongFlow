# -*- coding: utf-8 -*-
"""
Test cases for DefaultReasoner
"""
from unittest.mock import AsyncMock

import pytest

from agentsdk.memory.grade import GradeMemory
from agentsdk.message import Message
from agentsdk.tools import Toolkit
from evolux.react import AgentContext
from evolux.react.components import DefaultReasoner


@pytest.fixture
def mock_context():
    memory = GradeMemory.create_default(None)
    memory.add(Message.from_text("reasoner", "message1"))
    memory.add(Message.from_text("reasoner", "message2"))
    return AgentContext(
            memory=GradeMemory.create_default(None),
            toolkit=Toolkit()
    )


class TestDefaultReasoner:
    """Test cases for DefaultReasoner"""

    @pytest.fixture
    def mock_model(self):
        model = AsyncMock()
        model.arun = AsyncMock()
        return model

    @pytest.fixture
    def default_reasoner(self, mock_model):
        return DefaultReasoner(
                model=mock_model,
                system_prompt="Test system prompt",
                name="test_reasoner"
        )

    @pytest.mark.asyncio
    async def test_reason_success(self, default_reasoner, mock_context, mock_model):
        """Test reason method with successful model response"""
