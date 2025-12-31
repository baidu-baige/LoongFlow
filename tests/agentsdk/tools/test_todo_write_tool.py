# -*- coding: utf-8 -*-
"""
Unit tests for TodoWriteTool.
"""
import json
import os
import tempfile

import pytest

from agentsdk.tools.todo_write_tool import TodoWriteTool, TodoItem
from agentsdk.tools.tool_context import ToolContext


@pytest.fixture
def tool():
    return TodoWriteTool()

def test_get_declaration(tool):
    decl = tool.get_declaration()
    assert decl["name"] == "TodoWrite"
    assert "parameters" in decl
    assert "TodoWriteToolArgs" in str(decl["parameters"])

def test_write_todos_success(tool):
    todos = [
        {"content": "Write tests", "status": "pending"},
        {"content": "Implement feature X", "status": "in_progress"}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "todos.json")
        ctx = ToolContext(function_call_id="todo_write", state={"todo_file_path": file_path})
        resp = tool.run(args={"todos": todos}, tool_context=ctx)
        data = resp.content[0].data
        assert "Todo list updated successfully" in data["message"]
        assert data["file_path"] == file_path
        # Verify file content
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        assert len(file_data) == len(todos)
        assert resp.err_msg == ""

def test_write_todos_with_todoitem_objects(tool):
    items = [TodoItem(content="Task1", status="pending"), TodoItem(content="Task2", status="completed")]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "todos.json")
        ctx = ToolContext(function_call_id="todo_write", state={"todo_file_path": file_path})
        resp = tool.run(args={"todos": items}, tool_context=ctx)
        data = resp.content[0].data
        assert len(data["todos"]) == 2
        assert resp.err_msg == ""

def test_missing_args_returns_error(tool):
    resp = tool.run(args={})
    assert "Field required" in resp.err_msg

def test_arun_delegates_to_run(tool):
    items = [TodoItem(content="Async task", status="pending")]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "todos.json")
        ctx = ToolContext(function_call_id="todo_write", state={"todo_file_path": file_path})
        import asyncio
        resp = asyncio.run(tool.arun(args={"todos": items}, tool_context=ctx))
        data = resp.content[0].data
        assert len(data["todos"]) == 1
        assert resp.err_msg == ""
