# -*- coding: utf-8 -*-
"""
This file provides the TodoReadTool implementation.
"""

import json
import os
from typing import Any, Optional

from pydantic import BaseModel
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class TodoReadToolArgs(BaseModel):
    """
    Arguments for TodoReadTool.
    - No arguments needed since the file path is obtained from ToolContext.
    """

    pass


class TodoReadTool(FunctionTool):
    """
    TodoReadTool: reads a structured todo list for coding sessions.

    Features:
    - Retrieves the current todo list
    - Displays task progress
    - Uses the same file path as TodoWriteTool
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=TodoReadToolArgs,
            name="TodoRead",
            description=(
                "Reads a structured task list for coding sessions. "
                "Displays current tasks and their status."
            ),
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": TodoReadToolArgs.model_json_schema(),
        }

    def _get_todo_file_path(self, tool_context: Optional[ToolContext]) -> str:
        """Get the path to the todo file based on ToolContext or use default."""
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

        file_path = self._get_todo_file_path(tool_context)

        try:
            if not os.path.exists(file_path):
                msg = f"No todo list found at: {file_path}"
                return ToolResponse(
                    content=[
                        ContentElement(
                            mime_type=MimeType.TEXT_PLAIN,
                            data=msg,
                            metadata={"tool": self.name},
                        )
                    ]
                )

            with open(file_path, "r", encoding="utf-8") as f:
                todos_data = json.load(f)

            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data={
                            "message": "Todo list retrieved successfully.",
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
            err = f"Error reading todo list: {str(e)}"
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
