# -*- coding: utf-8 -*-
"""
Unit tests for TodoReadTool.
"""
import json
import os
import tempfile

import pytest

from agentsdk.tools.todo_read_tool import TodoReadTool
from agentsdk.tools.tool_context import ToolContext


@pytest.fixture
def tool():
    return TodoReadTool()

def test_get_declaration(tool):
    decl = tool.get_declaration()
    assert decl["name"] == "TodoRead"
    assert "parameters" in decl
    assert "TodoReadToolArgs" in str(decl["parameters"])

def test_no_todo_file_returns_message(tool):
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = os.path.join(tmpdir, "nonexistent.json")
        ctx = ToolContext(function_call_id="todo_read", state={"todo_file_path": fake_path})
        resp = tool.run(args={}, tool_context=ctx)
        content = resp.content[0].data
        assert "No todo list found" in content
        assert resp.err_msg == ""

def test_read_existing_todo_file(tool):
    todos = [
        {"task": "Write unit tests", "done": False},
        {"task": "Implement feature X", "done": True}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "todo.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(todos, f)
        ctx = ToolContext(function_call_id="todo_read", state={"todo_file_path": file_path})
        resp = tool.run(args={}, tool_context=ctx)
        data = resp.content[0].data
        assert data["file_path"] == file_path
        assert data["todos"] == todos
        assert "Todo list retrieved successfully" in data["message"]
        assert resp.err_msg == ""

def test_arun_delegates_to_run(tool):
    todos = [{"task": "Async test", "done": False}]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "todo.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(todos, f)
        ctx = ToolContext(function_call_id="todo_read", state={"todo_file_path": file_path})
        import asyncio
        resp = asyncio.run(tool.arun(args={}, tool_context=ctx))
        data = resp.content[0].data
        assert data["todos"] == todos
        assert resp.err_msg == ""
