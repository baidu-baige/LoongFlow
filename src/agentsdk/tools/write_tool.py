# -*- coding: utf-8 -*-
"""
This file provides the WriteTool implementation.
"""

import os
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class WriteToolArgs(BaseModel):
    """
    Arguments for WriteTool.
    - file_path: absolute or relative path to the file to write (relative paths are resolved relative to project root)
    - content: content to write to the file
    """

    file_path: str = Field(
        ..., description="The absolute or relative path to the file to write"
    )
    content: str = Field(..., description="The content to write to the file")


class WriteTool(FunctionTool):
    """
    Write tool: writes files to local filesystem.
    - Will overwrite existing files
    - Avoid creating new files unless explicitly required
    - Supports both absolute and relative paths (relative to project root)
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=WriteToolArgs,
            name="Write",
            description="Writes a file to the local filesystem. Will overwrite existing files. Supports both absolute and relative paths.",
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": WriteToolArgs.model_json_schema(),
        }

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

        file_path = validated_args.get("file_path")
        content = validated_args.get("content")

        try:
            # If path is relative, convert to absolute path relative to project root
            if not os.path.isabs(file_path):
                # Get the project root directory (two levels up from current file)
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                )
                file_path = os.path.join(project_root, file_path)

            # Make sure the target directory exists
            dir_path = os.path.dirname(file_path)
            if (
                dir_path
            ):  # Avoid non-existent directory error for current directory files
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=f"Write successful: {file_path}",
                        metadata={"tool": self.name},
                    )
                ]
            )

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            err = f"Error writing file: {str(e)}"
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

    @override
    async def arun(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Async execution, delegates to sync run."""
        return self.run(args=args, tool_context=tool_context)
