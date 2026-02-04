# -*- coding: utf-8 -*-
"""
This file provides the TodoWriteTool implementation.
"""

import json
import os
import uuid
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationError
from typing_extensions import override

from loongflow.agentsdk.message import ContentElement, MimeType
from loongflow.agentsdk.tools.function_tool import FunctionTool, ToolResponse
from loongflow.agentsdk.tools.tool_context import ToolContext


class TodoItem(BaseModel):
    """Single todo item model."""

    content: str = Field(..., description="The task description")
    status: str = Field(
        ..., description="Task status: pending | in_progress | completed"
    )
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique task ID"
    )


class TodoWriteToolArgs(BaseModel):
    """
    Arguments for TodoWriteTool.
    - todos: list of todo items to manage
    """

    todos: List[TodoItem] = Field(..., description="The updated todo list")


class TodoWriteTool(FunctionTool):
    """
    TodoWriteTool: manages a structured todo list for coding sessions.

    Features:
    - Tracks progress of complex tasks
    - Organizes multi-step operations
    - Demonstrates progress clearly to the user
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=TodoWriteToolArgs,
            name="TodoWrite",
            description=(
                "Manages a structured task list for coding sessions. "
                "Helps track progress and organize complex tasks."
            ),
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": TodoWriteToolArgs.model_json_schema(),
        }

    def _get_todo_file_path(self, tool_context: Optional[ToolContext]) -> str:
        """Get the path to store the todo list based on ToolContext."""
        if tool_context and tool_context.state.get("todo_file_path"):
            return tool_context.state["todo_file_path"]
        return "./todo_list.json"

    @override
    async def arun(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Asynchronous execution, returns ToolResponse."""
        return self.run(args=args, tool_context=tool_context)

    @override
    def run(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Synchronous execution, returns ToolResponse."""
        validated_args, error = self._prepare_call_args(args, tool_context)
        if error:
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=error,
                        metadata={"error": True},
                    )
                ],
                err_msg=error,
            )

        try:
            validated = TodoWriteToolArgs.model_validate(validated_args)
        except ValidationError as e:
            err = f"Invalid arguments: {str(e)}"
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=err,
                        metadata={"error": True},
                    )
                ],
                err_msg=err,
            )

        file_path = self._get_todo_file_path(tool_context)

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            todos_data = [item.model_dump() for item in validated.todos]

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(todos_data, f, ensure_ascii=False, indent=2)

            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data={
                            "message": "Todo list updated successfully.",
                            "file_path": file_path,
                            "todos": todos_data,
                        },
                        metadata={"tool": self.name},
                    )
                ]
            )

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            err = f"Error writing todo list: {str(e)}"
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=err,
                        metadata={"error": True},
                    )
                ],
                err_msg=err,
            )
