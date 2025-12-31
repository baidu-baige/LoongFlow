# -*- coding: utf-8 -*-
"""
This file provides the test cases for toolkit.
"""
import pytest

from agentsdk.message.elements import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool
from agentsdk.tools.tool_context import (
    ToolContext,
    AuthConfig,
    AuthCredential,
    AuthType,
)
from agentsdk.tools.tool_response import ToolResponse
from agentsdk.tools.toolkit import Toolkit


class DummyTool(FunctionTool):
    """A dummy tool for testing."""

    response_class = ToolResponse

    def run(self, args, tool_context: ToolContext):
        session_id = getattr(tool_context, "function_call_id", "")
        return self._build_response(f"{args.get('a', '')}-{session_id}")

    async def arun(self, args, tool_context: ToolContext):
        session_id = getattr(tool_context, "function_call_id", "")
        return self._build_response(f"{args.get('a', '')}-{session_id}")

    def _build_response(self, data):
        return self.response_class(
            content=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data=data)],
            err_msg="",
        )


@pytest.fixture
def toolkit():
    return Toolkit()


@pytest.fixture
def dummy_tool():
    return DummyTool(name="dummy")


@pytest.mark.asyncio
async def test_toolkit_register_run(toolkit, dummy_tool):
    # Register the tool
    toolkit.register_tool(dummy_tool)

    # Sync call
    resp = toolkit.run("dummy", args={"a": 42})
    assert resp.err_msg == ""
    assert resp.content[0].data == "42-dummy"

    # Async call
    resp = await toolkit.arun("dummy", args={"a": 99})
    assert resp.err_msg == ""
    assert resp.content[0].data == "99-dummy"


@pytest.mark.asyncio
async def test_toolkit_context(toolkit, dummy_tool):
    toolkit.register_tool(dummy_tool)

    # Self-defined ToolContext
    ctx = ToolContext(function_call_id="abc123", state={})
    resp = await toolkit.arun("dummy", args={"a": 1}, tool_context=ctx)
    assert resp.content[0].data == "1-abc123"

    stored_ctx = toolkit.get_context("dummy")
    assert stored_ctx is ctx


def test_toolkit_auth(toolkit, dummy_tool):
    toolkit.register_tool(dummy_tool)

    auth_cfg = AuthConfig(scheme=AuthType.API_KEY, key="key1")
    cred = AuthCredential(auth_type=AuthType.API_KEY, api_key="secret")

    toolkit.set_auth("dummy", auth_cfg, cred)
    stored_cred = toolkit.get_auth("dummy", auth_cfg)
    assert stored_cred.api_key == "secret"


def test_toolkit_unregister(toolkit, dummy_tool):
    toolkit.register_tool(dummy_tool)
    assert "dummy" in toolkit.list_tools()

    toolkit.unregister_tool("dummy")
    assert "dummy" not in toolkit.list_tools()
