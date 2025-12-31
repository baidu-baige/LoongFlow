# -*- coding: utf-8 -*-
"""
This file provides the LsTool implementation.
"""

import glob
import os
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class LsToolArgs(BaseModel):
    """
    Arguments for LsTool.
    - path: absolute path to directory to list.
    - ignore: optional list of glob patterns to ignore.
    """

    path: str = Field(..., description="The absolute path to the directory to list")
    ignore: Optional[List[str]] = Field(
        None, description="List of glob patterns to ignore"
    )


class LsTool(FunctionTool):
    """
    LS tool: lists files and directories in a given path.
    The path must be absolute. Optionally, ignore files matching glob patterns.
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=LsToolArgs,
            name="LS",
            description=(
                "Lists files and directories in a given path. "
                "Path must be absolute. Optionally provide an array of glob patterns to ignore."
            ),
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": LsToolArgs.model_json_schema(),
        }

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

        path = validated_args.get("path")
        ignore_patterns = validated_args.get("ignore") or []

        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        try:
            if not os.path.exists(path):
                err = f"Path does not exist: {path}"
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

            if not os.path.isdir(path):
                err = f"Path is not a directory: {path}"
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

            all_files = []
            # Only list one layer
            for name in os.listdir(path):
                # Apply ignore patterns
                if any(
                    glob.fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns
                ):
                    continue
                full_path = os.path.join(path, name)
                all_files.append(
                    {
                        "name": name,
                        "path": full_path,
                        "relative_path": name,
                        "is_dir": os.path.isdir(full_path),
                        "size": (
                            os.path.getsize(full_path)
                            if not os.path.isdir(full_path)
                            else 0
                        ),
                    }
                )

            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data={"files": all_files},
                        metadata={"tool": self.name},
                    )
                ]
            )

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            err = str(e)
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
