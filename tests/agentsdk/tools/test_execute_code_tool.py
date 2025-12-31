import os
import tempfile

import pytest

from agentsdk.tools.execute_code_tool import ExecuteCodeTool


@pytest.fixture
def tool():
    """Fixture for ExecuteCodeTool instance."""
    return ExecuteCodeTool()


def test_get_declaration(tool):
    """Test that get_declaration returns valid metadata."""
    decl = tool.get_declaration()
    assert decl["name"] == "ExecuteCode"
    assert "parameters" in decl
    assert "language" in decl["parameters"]["properties"]


def test_run_inline_code_success(tool):
    """Test executing simple inline Python code."""
    args = {"mode": "code", "code": "print('Hello GPT-5')"}
    resp = tool.run(args=args)
    data = resp.content[0].data
    assert "Hello GPT-5" in data["stdout"]
    assert data["returncode"] == 0
    assert resp.err_msg == ""


def test_run_inline_code_with_error(tool):
    """Test executing Python code that raises an exception."""
    args = {"mode": "code", "code": "raise ValueError('oops')"}
    resp = tool.run(args=args)
    data = resp.content[0].data
    assert "oops" in data["stderr"] or "ValueError" in data["stderr"]
    assert data["returncode"] != 0


def test_run_file_mode_success(tool):
    """Test executing a valid Python file."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("print('File executed')\n")
        file_path = f.name

    args = {"mode": "file", "file_path": file_path}
    resp = tool.run(args=args)
    os.remove(file_path)

    data = resp.content[0].data
    assert "File executed" in data["stdout"]
    assert data["returncode"] == 0


def test_missing_code_for_mode_code(tool):
    """Test missing code in mode='code'."""
    args = {"mode": "code"}
    resp = tool.run(args=args)
    assert "Missing `code`" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_missing_file_path_for_mode_file(tool):
    """Test missing file_path in mode='file'."""
    args = {"mode": "file"}
    resp = tool.run(args=args)
    assert "Missing `file_path`" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_invalid_mode_validation_error(tool):
    """Test that invalid mode raises validation error."""
    args = {"mode": "invalid"}
    resp = tool.run(args=args)
    assert "validation error" in resp.err_msg or "mode must be" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_unsupported_language(tool):
    """Test that unsupported language is rejected."""
    args = {"language": "cpp", "mode": "code", "code": "print(1)"}
    resp = tool.run(args=args)
    assert "Unsupported language" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_timeout_handling(tool):
    """Test that code exceeding timeout returns error."""
    args = {"mode": "code", "code": "import time; time.sleep(2)", "timeout": 1}
    resp = tool.run(args=args)
    data = resp.content[0].data
    # Depending on platform, the error could mention "timed out"
    assert "timed out" in data["error"] or data["returncode"] == -1


def test_arun_delegates_to_run(tool):
    """Test async arun simply delegates to run."""
    args = {"mode": "code", "code": "print('Async OK')"}
    import asyncio
    result = asyncio.run(tool.arun(args=args))
    data = result.content[0].data
    assert "Async OK" in data["stdout"]
    assert result.err_msg == ""


def test_run_file_with_runtime_error(tool):
    """Test executing a Python file that throws an exception."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("raise RuntimeError('boom')\n")
        file_path = f.name

    args = {"mode": "file", "file_path": file_path}
    resp = tool.run(args=args)
    os.remove(file_path)

    data = resp.content[0].data
    assert "boom" in data["stderr"]
    assert data["returncode"] != 0


def test_exception_handling(tool, monkeypatch):
    """Force _run_python_code to raise and ensure graceful fallback."""
    def bad_func(*a, **kw):
        raise RuntimeError("fail-fast")

    monkeypatch.setattr(tool, "_run_python_code", bad_func)
    resp = tool.run(args={"mode": "code", "code": "print(1)"})
    assert "Unexpected error" in resp.err_msg
    assert resp.content[0].metadata["error"] is True
