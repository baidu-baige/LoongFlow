import os
import tempfile

import pytest

from agentsdk.tools.read_tool import ReadTool


@pytest.fixture
def tool():
    return ReadTool()


def test_read_text_file_success(tool):
    """Test reading a normal text file successfully."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("line1\nline2\nline3\nline4\n")
        tmp_path = tmp.name

    args = {"file_path": tmp_path}
    resp = tool.run(args=args)

    assert not resp.err_msg
    assert resp.content[0].mime_type.value == "application/json"
    data = resp.content[0].data
    assert data["type"] == "text"
    assert "line1" in data["content"]

    os.remove(tmp_path)


def test_read_text_with_offset_and_limit(tool):
    """Test reading text file with offset and limit."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("a\nb\nc\nd\n")
        tmp_path = tmp.name

    args = {"file_path": tmp_path, "offset": 2, "limit": 2}
    resp = tool.run(args=args)
    data = resp.content[0].data

    assert data["total_lines"] == 2
    assert "b" in data["content"]
    assert "c" in data["content"]
    os.remove(tmp_path)


def test_read_empty_file(tool):
    """Test reading an empty file."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    resp = tool.run(args={"file_path": tmp_path})
    assert "File is empty" in resp.content[0].data
    os.remove(tmp_path)


def test_file_not_found(tool):
    """Test file not found error."""
    resp = tool.run(args={"file_path": "/tmp/not_exist_123.txt"})
    assert "File not found" in resp.err_msg


def test_missing_file_path(tool):
    """Test missing file_path field."""
    resp = tool.run(args={})
    assert "validation error" in resp.err_msg
    assert "file_path" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_read_image_file(tool):
    """Test reading an image file (should not actually open)."""
    with tempfile.NamedTemporaryFile("w+", suffix=".png", delete=False) as tmp:
        tmp.write("fakeimage")
        tmp_path = tmp.name

    resp = tool.run(args={"file_path": tmp_path})
    data = resp.content[0].data
    assert data["type"] == "image"
    assert data["path"] == tmp_path
    os.remove(tmp_path)


def test_read_pdf_file(tool):
    """Test reading a PDF file."""
    with tempfile.NamedTemporaryFile("w+", suffix=".pdf", delete=False) as tmp:
        tmp.write("fakepdf")
        tmp_path = tmp.name

    resp = tool.run(args={"file_path": tmp_path})
    assert resp.content[0].data["type"] == "pdf"
    os.remove(tmp_path)


def test_read_ipynb_file(tool):
    """Test reading a Jupyter notebook file."""
    with tempfile.NamedTemporaryFile("w+", suffix=".ipynb", delete=False) as tmp:
        tmp.write("{}")
        tmp_path = tmp.name

    resp = tool.run(args={"file_path": tmp_path})
    assert resp.content[0].data["type"] == "notebook"
    os.remove(tmp_path)


def test_read_binary_file(tool, monkeypatch):
    """Test reading a binary file triggers UnicodeDecodeError."""

    def mock_open(*_, **__):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")

    monkeypatch.setattr("builtins.open", mock_open)
    resp = tool._read_text_file("/tmp/fake.bin", None, None)
    data = resp.content[0].data
    assert data["type"] == "binary"
    assert "warning" in data


@pytest.mark.asyncio
async def test_arun_delegates_to_run(tool):
    """Test that arun delegates to run."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("abc")
        tmp_path = tmp.name

    resp = await tool.arun(args={"file_path": tmp_path})
    assert not resp.err_msg
    assert "abc" in resp.content[0].data["content"]
    os.remove(tmp_path)


def test_exception_handling(tool, monkeypatch):
    """Test generic exception handling path."""

    def mock_exists(_):
        raise RuntimeError("mock failure")

    monkeypatch.setattr("os.path.exists", mock_exists)
    resp = tool.run(args={"file_path": "/tmp/any"})
    assert "mock failure" in resp.err_msg
