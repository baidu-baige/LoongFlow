# -*- coding: utf-8 -*-
"""
This file provides the ReadTool implementation.
"""

import os
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class ReadToolArgs(BaseModel):
    """
    Arguments for ReadTool.
    - file_path: absolute path to the file to read.
    - offset: optional line number to start reading from.
    - limit: optional number of lines to read.
    """

    file_path: str = Field(..., description="The absolute path to the file to read.")
    offset: Optional[int] = Field(
        None, description="Line number to start reading from."
    )
    limit: Optional[int] = Field(None, description="Number of lines to read.")


class ReadTool(FunctionTool):
    """
    ReadTool: reads files from local filesystem with optional line range.
    Supports text files, images, PDFs, and Jupyter notebooks.
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=ReadToolArgs,
            name="Read",
            description="Reads a file from the local filesystem. Supports text, images, PDFs, and Jupyter notebooks.",
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": ReadToolArgs.model_json_schema(),
        }

    @override
    async def arun(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool asynchronously."""
        return self.run(args=args, tool_context=tool_context)

    @override
    def run(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool synchronously."""
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
        offset = validated_args.get("offset")
        limit = validated_args.get("limit")

        if not file_path:
            err = "Missing `file_path` field."
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

        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        try:
            if not os.path.exists(file_path):
                err = f"File not found: {file_path}"
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

            if os.path.getsize(file_path) == 0:
                msg = "File is empty"
                return ToolResponse(
                    content=[
                        ContentElement(
                            mime_type=MimeType.TEXT_PLAIN,
                            data=msg,
                            metadata={"warning": True},
                        )
                    ]
                )

            # Dispatch based on file type
            if file_path.endswith(
                (".txt", ".py", ".json", ".csv", ".md", ".html", ".xml")
            ):
                return self._read_text_file(file_path, offset, limit)

            elif file_path.endswith((".png", ".jpg", ".jpeg", ".gif")):
                return ToolResponse(
                    content=[
                        ContentElement(
                            mime_type=MimeType.APPLICATION_JSON,
                            data={"type": "image", "path": file_path},
                            metadata={"tool": self.name},
                        )
                    ]
                )

            elif file_path.endswith(".pdf"):
                return ToolResponse(
                    content=[
                        ContentElement(
                            mime_type=MimeType.APPLICATION_JSON,
                            data={"type": "pdf", "path": file_path},
                            metadata={"tool": self.name},
                        )
                    ]
                )

            elif file_path.endswith(".ipynb"):
                return ToolResponse(
                    content=[
                        ContentElement(
                            mime_type=MimeType.APPLICATION_JSON,
                            data={"type": "notebook", "path": file_path},
                            metadata={"tool": self.name},
                        )
                    ]
                )

            else:
                # Default to text read
                return self._read_text_file(file_path, offset, limit)

        except Exception as e:
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

    def _read_text_file(
        self, file_path: str, offset: Optional[int], limit: Optional[int]
    ) -> ToolResponse:
        """Read text file with optional line range and return ToolResponse."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Apply offset and limit
            start = offset - 1 if offset else 0
            end = start + limit if limit else len(lines)
            lines = lines[start:end]

            # Format with line numbers
            formatted_lines = []
            for i, line in enumerate(lines, start=start + 1):
                line = line[:2000] + "..." if len(line) > 2000 else line
                formatted_lines.append(f"{i:6d}  {line}")

            result = {
                "type": "text",
                "path": file_path,
                "content": "".join(formatted_lines),
                "total_lines": len(lines),
            }

            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data=result,
                        metadata={"tool": self.name},
                    )
                ]
            )

        except UnicodeDecodeError:
            result = {
                "type": "binary",
                "path": file_path,
                "warning": "File appears to be binary data",
            }
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data=result,
                        metadata={"tool": self.name},
                    )
                ]
            )
