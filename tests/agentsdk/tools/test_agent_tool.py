# -*- coding: utf-8 -*-
"""
Unit tests for AgentTool class.

This test file demonstrates both synchronous and asynchronous execution paths
by wrapping a simple EchoAgent inside an AgentTool.
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from agentsdk.message import ContentElement, MimeType, Message
from agentsdk.tools import AgentTool
from evolux.base.agent_base import AgentBase


# ----------------------------
# 1. Define a simple sub-agent
# ----------------------------
class EchoInput(BaseModel):
    """Input schema for EchoAgent."""
    request: str = Field(..., description="User input text to be echoed.")


class EchoAgent(AgentBase):
    """A simple agent that echoes user input."""

    name = "echo_agent"
    description = "Echoes the user input back as text."
    input_schema = EchoInput

    async def run(self, request: str) -> Message:
        """Main logic of the EchoAgent."""
        response_text = f"EchoAgent received: {request}"
        # Simulate async work
        await asyncio.sleep(0.01)

        return Message.from_text(
            data=response_text,
            sender=self.name,
            role="assistant",
            mime_type=MimeType.TEXT_PLAIN,
        )

    async def interrupt_impl(self):
        """No special interruption logic."""
        return Message.from_text("EchoAgent interrupted.")


# ----------------------------
# 2. Sync test
# ----------------------------
def test_agent_tool_sync():
    """Test AgentTool synchronous run."""
    agent = EchoAgent()
    tool = AgentTool(agent)

    args = {"request": "Hello from sync test"}
    response = tool.run(args=args, tool_context=None)

    # Validate response content
    assert response is not None
    assert len(response.content) > 0

    content = response.content[0]
    assert isinstance(content, ContentElement)
    assert content.mime_type == MimeType.TEXT_PLAIN
    assert "EchoAgent received" in content.data


# ----------------------------
# 3. Async test
# ----------------------------
@pytest.mark.asyncio
async def test_agent_tool_async():
    """Test AgentTool asynchronous run."""
    agent = EchoAgent()
    tool = AgentTool(agent)

    args = {"request": "Hello from async test"}
    response = await tool.arun(args=args, tool_context=None)

    # Validate response content
    assert response is not None
    assert len(response.content) > 0

    content = response.content[0]
    assert isinstance(content, ContentElement)
    assert content.mime_type == MimeType.TEXT_PLAIN
    assert "EchoAgent received" in content.data


# ----------------------------
# 4. Declaration test
# ----------------------------
def test_agent_tool_declaration():
    """Test get_declaration() returns valid schema."""
    agent = EchoAgent()
    tool = AgentTool(agent)
    decl = tool.get_declaration()

    assert decl["name"] == "echo_agent"
    assert "description" in decl
    assert "parameters" in decl
    assert "properties" in decl["parameters"]
    assert "request" in decl["parameters"]["properties"]
