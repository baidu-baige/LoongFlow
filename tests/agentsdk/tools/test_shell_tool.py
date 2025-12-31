# -*- coding: utf-8 -*-
"""
Unit tests for ShellTool.
"""

import pytest

from agentsdk.message import MimeType
from agentsdk.tools.shell_tool import ShellTool, _run_command, _run_command_async


@pytest.mark.asyncio
class TestShellTool:
    """Unit tests for ShellTool."""

    def setup_method(self):
        """Create a ShellTool instance before each test."""
        self.tool = ShellTool()

    async def test_arun_single_command(self):
        """Test async run with a single valid command."""
        args = {"commands": [{"command": "echo hello"}]}
        resp = await self.tool.arun(args=args)
        result = resp.content[0].data["results"][0]
        assert result["stdout"] == "hello"
        assert result["returncode"] == 0
        assert resp.content[0].mime_type == MimeType.APPLICATION_JSON

    async def test_arun_multiple_commands(self):
        """Test async run with multiple commands executed sequentially."""
        args = {"commands": [{"command": "echo first"}, {"command": "echo second"}]}
        resp = await self.tool.arun(args=args)
        results = resp.content[0].data["results"]
        assert [r["stdout"] for r in results] == ["first", "second"]

    async def test_arun_missing_command_field(self):
        """Test async run where one item misses the command field."""
        args = {"commands": [{"command": "echo ok"}, {"dir": "/tmp"}]}
        resp = await self.tool.arun(args=args)
        # If validation failed early, error string is returned
        if resp.err_msg:
            assert "Invalid args" in resp.err_msg
        else:
            results = resp.content[0].data["results"]
            assert results[1]["error"] == "Missing `command` field."

    async def test_arun_invalid_commands_type(self):
        """Test async run with invalid commands type."""
        args = {"commands": "not_a_list"}
        resp = await self.tool.arun(args=args)
        assert "Invalid args for ShellTool" in resp.err_msg
        assert resp.content[0].metadata["error"] is True

    async def test_run_sync_success(self):
        """Test synchronous run with valid command."""
        args = {"commands": [{"command": "echo sync"}]}
        resp = self.tool.run(args=args)
        results = resp.content[0].data["results"]
        assert results[0]["stdout"] == "sync"
        assert results[0]["returncode"] == 0

    async def test_run_sync_missing_command(self):
        """Test synchronous run missing command field."""
        args = {"commands": [{"dir": "/tmp"}]}
        resp = self.tool.run(args=args)
        if resp.err_msg:
            assert "Invalid args" in resp.err_msg
        else:
            results = resp.content[0].data["results"]
            assert "error" in results[0]
            assert "Missing `command`" in results[0]["error"]

    async def test_run_sync_invalid_type(self):
        """Test synchronous run with invalid commands type."""
        args = {"commands": "invalid"}
        resp = self.tool.run(args=args)
        assert "Invalid args for ShellTool" in resp.err_msg
        assert resp.content[0].metadata["error"] is True

    async def test_run_command_function(self):
        """Test _run_command helper with real command."""
        result = _run_command("echo test_run")
        assert result["stdout"] == "test_run"
        assert result["returncode"] == 0

    async def test_run_command_async_function(self):
        """Test _run_command_async helper with real command."""
        result = await _run_command_async("echo test_async")
        assert result["stdout"] == "test_async"
        assert result["returncode"] == 0

    async def test_run_command_error(self):
        """Test _run_command handles invalid command gracefully."""
        result = _run_command("nonexistent_command_12345")
        assert result["returncode"] != 0
        assert result["stderr"] != ""

    async def test_run_command_async_error(self):
        """Test _run_command_async handles invalid command gracefully."""
        result = await _run_command_async("nonexistent_command_12345")
        assert result["returncode"] != 0
        assert result["stderr"] != ""
