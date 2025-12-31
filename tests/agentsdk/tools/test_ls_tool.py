# -*- coding: utf-8 -*-
"""
Unit tests for LsTool.
"""

import os
import tempfile

import pytest

from agentsdk.tools.ls_tool import LsTool


@pytest.fixture
def tool():
    """Fixture providing an instance of LsTool."""
    return LsTool()


def test_ls_success(tool):
    """Test listing files and directories successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files and subdirectory
        file_path = os.path.join(tmpdir, "file1.txt")
        subdir_path = os.path.join(tmpdir, "subdir")
        os.mkdir(subdir_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("hello")

        args = {"path": tmpdir}
        resp = tool.run(args=args)

        assert not resp.err_msg
        assert len(resp.content) == 1
        data = resp.content[0].data
        assert "files" in data
        names = [f["name"] for f in data["files"]]
        assert "file1.txt" in names
        assert "subdir" in names

        file_info = next(f for f in data["files"] if f["name"] == "file1.txt")
        assert file_info["is_dir"] is False
        assert file_info["size"] == 5  # content length


def test_ls_ignore_pattern(tool):
    """Test ignore pattern filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "keep.txt"), "w").close()
        open(os.path.join(tmpdir, "skip.log"), "w").close()

        args = {"path": tmpdir, "ignore": ["*.log"]}
        resp = tool.run(args=args)

        data = resp.content[0].data
        names = [f["name"] for f in data["files"]]
        assert "keep.txt" in names
        assert "skip.log" not in names


def test_non_absolute_path_error(tool):
    """Test error when using non-absolute path."""
    args = {"path": "relative/path"}
    resp = tool.run(args=args)

    assert resp.err_msg == "Path must be absolute"
    assert resp.content[0].metadata["error"] is True


def test_path_not_exist(tool):
    """Test error when directory does not exist."""
    tmp_path = os.path.abspath("/tmp/does_not_exist_12345")
    args = {"path": tmp_path}
    resp = tool.run(args=args)

    assert "does not exist" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_path_is_file(tool):
    """Test error when path is a file instead of directory."""
    with tempfile.NamedTemporaryFile() as tmpfile:
        args = {"path": tmpfile.name}
        resp = tool.run(args=args)
        assert "not a directory" in resp.err_msg
        assert resp.content[0].metadata["error"] is True


def test_run_exception(monkeypatch, tool):
    """Test error handling when os.listdir raises exception."""
    def mock_listdir(path):
        raise OSError("listdir failed")

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(os, "listdir", mock_listdir)
        args = {"path": tmpdir}
        resp = tool.run(args=args)

        assert "listdir failed" in resp.err_msg
        assert resp.content[0].metadata["error"] is True


@pytest.mark.asyncio
async def test_arun_delegates_to_run(tool):
    """Test that arun delegates to run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.txt"), "w").close()

        args = {"path": tmpdir}
        resp = await tool.arun(args=args)

        assert not resp.err_msg
        data = resp.content[0].data
        assert "a.txt" in [f["name"] for f in data["files"]]
