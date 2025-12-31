# -*- coding: utf-8 -*-
"""
This file provides the ExecuteCodeTool implementation.
"""

import subprocess
import sys
import time
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class ExecuteCodeToolArgs(BaseModel):
    """
    Arguments for ExecuteCodeTool.
    - language: Programming language to execute (default: python)
    - mode: Execution mode: 'code' for inline code, 'file' for script file
    - code: Python code to execute when mode='code'
    - file_path: Python file path to execute when mode='file'
    - timeout: Execution timeout in seconds
    """

    language: str | None = Field(
        "python", description="Programming language to execute (default python)"
    )
    mode: str = Field(
        "code",
        description="Execution mode: 'code' for inline code, 'file' for script file",
    )
    code: Optional[str] = Field(
        None, description="Python code to execute when mode='code'"
    )
    file_path: Optional[str] = Field(
        None, description="Python file path to execute when mode='file'"
    )
    timeout: int = Field(5, description="Execution timeout in seconds")

    @field_validator("mode")
    def validate_mode(cls, v):
        if v not in ("code", "file"):
            raise ValueError("mode must be 'code' or 'file'")
        return v

    @field_validator("language")
    def validate_language(cls, v: str) -> str:
        supported_languages = ["python"]
        if v not in supported_languages:
            raise ValueError(f"Unsupported language: {v}")
        return v


class ExecuteCodeTool(FunctionTool):
    """Tool to execute Python code or Python scripts."""

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=ExecuteCodeToolArgs,
            name="ExecuteCode",
            description="Executes Python code or Python files with timeout support.",
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": ExecuteCodeToolArgs.model_json_schema(),
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

        mode = validated_args.get("mode")
        timeout = validated_args.get("timeout")

        try:
            if mode == "code":
                code = validated_args.get("code")
                if not code:
                    err = "Missing `code` for mode='code'"
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
                result = self._run_python_code(code, timeout)
            else:  # mode == 'file'
                file_path = validated_args.get("file_path")
                if not file_path:
                    err = "Missing `file_path` for mode='file'"
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
                result = self._run_python_file(file_path, timeout)

            # Return normal result
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data=result,
                        metadata={"tool": self.name},
                    )
                ]
            )

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            err = f"Unexpected error: {str(e)}"
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

    def _run_python_code(self, code: str, timeout: int) -> dict[str, Any]:
        """Run inline Python code with timeout."""
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "error": proc.stderr.strip(),
                "execution_time": time.time() - start,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
                "execution_time": time.time() - start,
            }

    def _run_python_file(self, file_path: str, timeout: int) -> dict[str, Any]:
        """Run Python file with timeout."""
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "error": proc.stderr.strip(),
                "execution_time": time.time() - start,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
                "execution_time": time.time() - start,
            }
