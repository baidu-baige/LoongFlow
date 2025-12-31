# -*- coding: utf-8 -*-
"""
This file provides the test cases for function_tool.
"""
import asyncio
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from agentsdk.message import MimeType
from agentsdk.tools.function_tool import FunctionTool
from agentsdk.tools.tool_context import ToolContext
from agentsdk.tools.tool_response import ToolResponse


class Person(BaseModel):
    name: str = Field(..., description="name of user")
    age: int = Field(..., description="age of user")
    achievements: List[str] = Field(..., description="list of achievements")
    hobbies: List[str] = Field(..., description="list of hobbies")


class Output(BaseModel):
    persons: dict[str, Person] = Field(default_factory=dict, description="person info")


def get_person_info(person: Person) -> str:
    return f"{person.name}, {person.age}, {', '.join(person.achievements)}, {', '.join(person.hobbies)}"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def async_add(a: int, b: int) -> int:
    await asyncio.sleep(0.01)
    return a + b


def use_context(a: int, tool_context: Optional[ToolContext] = None):
    if not tool_context:
        raise ValueError("no context")
    return f"{a}-{tool_context.function_call_id}"


def raise_error(a: int):
    raise ValueError("invalid input")


class AddArgs(BaseModel):
    a: int
    b: int


@pytest.fixture
def inline_schema_tool():
    return FunctionTool(
        func=get_person_info,
        args_schema=Person,
        name="person_info_tool",
        description="get person info",
    )


@pytest.fixture
def simple_tool():
    return FunctionTool(func=add, name="add_tool", description="sum two ints")


@pytest.fixture
def async_tool():
    return FunctionTool(
        func=async_add, name="async_add_tool", description="async add two ints"
    )


@pytest.fixture
def schema_tool():
    return FunctionTool(
        func=add, args_schema=AddArgs, name="schema_tool", description="sum two ints"
    )


@pytest.fixture
def context_tool():
    return FunctionTool(
        func=use_context, name="context_tool", description="use context tool"
    )


@pytest.fixture
def error_tool():
    return FunctionTool(
        func=raise_error, name="error_tool", description="raise error tool"
    )


def test_run_success(simple_tool):
    resp: ToolResponse = simple_tool.run(args={"a": 2, "b": 3})
    assert isinstance(resp, ToolResponse)
    assert resp.err_msg == ""
    assert len(resp.content) == 1
    element = resp.content[0]
    assert element.mime_type == MimeType.APPLICATION_JSON
    assert element.data == 5
    assert element.metadata["tool"] == "add_tool"


def test_run_missing_arg(simple_tool):
    resp: ToolResponse = simple_tool.run(args={"a": 2})
    assert resp.err_msg.startswith("Missing mandatory parameters")
    assert resp.content[0].metadata["error"] is True


def test_run_raise_error(error_tool):
    resp: ToolResponse = error_tool.run(args={"a": 10})
    assert "invalid input" in resp.err_msg
    assert resp.content[0].metadata["error"] is True


def test_run_with_schema(schema_tool):
    resp = schema_tool.run(args={"a": 1, "b": 2})
    assert resp.content[0].data == 3


@pytest.mark.asyncio
async def test_arun_success(async_tool):
    resp = await async_tool.arun(args={"a": 5, "b": 7})
    assert isinstance(resp, ToolResponse)
    assert resp.err_msg == ""
    element = resp.content[0]
    assert element.mime_type == MimeType.APPLICATION_JSON
    assert element.data == 12


@pytest.mark.asyncio
async def test_arun_error(async_tool):
    resp = await async_tool.arun(args={"a": 1})
    assert resp.err_msg.startswith("Missing mandatory parameters")
    assert resp.content[0].metadata["error"]


@pytest.mark.asyncio
async def test_arun_with_context(context_tool):
    ctx = ToolContext(function_call_id="abc123", state={})
    resp = await context_tool.arun(args={"a": 42}, tool_context=ctx)
    assert resp.err_msg == ""
    result = resp.content[0].data
    assert result == "42-abc123"


@pytest.mark.asyncio
async def test_arun_raise_error(error_tool):
    resp = await error_tool.arun(args={"a": 100})
    assert "invalid input" in resp.err_msg
    assert resp.content[0].metadata["error"]


def test_get_declaration_with_schema(schema_tool):
    decl = schema_tool.get_declaration()
    assert decl["name"] == "schema_tool"
    assert "parameters" in decl
    props = decl["parameters"]["properties"]
    assert "a" in props and props["a"]["type"] == "integer"


def test_get_declaration_with_input_schema(inline_schema_tool):
    decl = inline_schema_tool.get_declaration()
    assert decl["name"] == "person_info_tool"
    assert "parameters" in decl
    props = decl["parameters"]["properties"]
    assert "name" in props and props["name"]["type"] == "string"


def test_get_declaration_with_func(simple_tool):
    decl = simple_tool.get_declaration()
    assert decl["name"] == "add_tool"
    assert "parameters" in decl
    assert decl["parameters"]["properties"]["a"]["type"] == "integer"
