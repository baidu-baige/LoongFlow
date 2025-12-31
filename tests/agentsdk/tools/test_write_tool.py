# -*- coding: utf-8 -*-
"""
Unit tests for WriteTool.
"""

import os
import tempfile

import pytest

from agentsdk.tools.write_tool import WriteTool


@pytest.fixture
def tool():
    """Fixture providing an instance of WriteTool."""
    return WriteTool()


def test_write_success(tool):
    """Test successful file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        args = {"file_path": file_path, "content": "Hello, LoongFlow!"}

        resp = tool.run(args=args)
        assert resp.err_msg is None
        assert len(resp.content) == 1
        assert "Write successful" in resp.content[0].data

        # Verify content written to file
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Hello, LoongFlow!"


def test_non_absolute_path_error(tool):
    """Test error when using non-absolute path."""
    args = {"file_path": "relative.txt", "content": "data"}
    resp = tool.run(args=args)

    assert resp.err_msg == "File path must be absolute"
    assert resp.content[0].metadata["error"] is True


def test_missing_arguments(tool):
    """Test error when missing required arguments."""
    args = {"file_path": "/tmp/test.txt"}  # missing 'content'
    resp = tool.run(args=args)

    assert resp.err_msg is not None
    assert "content" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_write_exception(monkeypatch, tool):
    """Test handling when file write fails (e.g., permission error)."""

    def mock_open(*args, **kwargs):
        raise PermissionError("No permission")

    monkeypatch.setattr("builtins.open", mock_open)

    args = {"file_path": os.path.abspath("/tmp/test.txt"), "content": "x"}
    resp = tool.run(args=args)

    assert "Error writing file" in resp.err_msg
    assert "PermissionError" not in resp.err_msg  # should not expose traceback
    assert resp.content[0].metadata["error"] is True


@pytest.mark.asyncio
async def test_arun_delegates_to_run(tool):
    """Test that arun calls run internally."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "async.txt")
        args = {"file_path": file_path, "content": "Async content"}

        resp = await tool.arun(args=args)
        assert "Write successful" in resp.content[0].data

        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Async content"
