# -*- coding: utf-8 -*-
"""
This file provides the ShellTool implementation.
"""

import asyncio
import subprocess
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.function_tool import FunctionTool, ToolResponse
from agentsdk.tools.tool_context import ToolContext


class CommandItem(BaseModel):
    command: str = Field(..., description="The shell command to execute")
    dir: Optional[str] = Field(
        None, description="Optional working directory for the command"
    )


class ShellToolArgs(BaseModel):
    """
    Arguments for ShellTool.
    - commands: list of CommandItem
    """

    commands: List[CommandItem] = Field(
        ..., description="List of shell commands to execute"
    )


class ShellTool(FunctionTool):
    """
    Shell tool: executes one or more shell commands.
    """

    def __init__(self):
        super().__init__(
            func=None,
            args_schema=ShellToolArgs,
            name="ShellTool",
            description="Execute one or more shell commands.",
        )

    @override
    def get_declaration(self) -> dict[str, Any]:
        """Generate tool declaration from Pydantic model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": ShellToolArgs.model_json_schema(),
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

        commands = validated_args.get("commands", [])
        if not isinstance(commands, list):
            err = "`commands` must be a list."
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

        results = []
        for item in commands:
            cmd = (
                item.get("command")
                if isinstance(item, dict)
                else getattr(item, "command", None)
            )
            cwd = (
                item.get("dir")
                if isinstance(item, dict)
                else getattr(item, "dir", None)
            )
            if not cmd:
                results.append({"error": "Missing `command` field."})
                continue
            results.append(_run_command(cmd, cwd))

        return ToolResponse(
            content=[
                ContentElement(
                    mime_type=MimeType.APPLICATION_JSON,
                    data={"results": results},
                    metadata={"tool": self.name},
                )
            ]
        )

    @override
    async def arun(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Asynchronous execution: sequentially runs shell commands."""
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

        commands = validated_args.get("commands", [])
        if not isinstance(commands, list):
            err = "`commands` must be a list."
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

        # Execute each command sequentially
        results = []
        for item in commands:
            cmd = (
                item.get("command")
                if isinstance(item, dict)
                else getattr(item, "command", None)
            )
            cwd = (
                item.get("dir")
                if isinstance(item, dict)
                else getattr(item, "dir", None)
            )

            if not cmd:
                results.append({"error": "Missing `command` field."})
                continue

            # Run command asynchronously (but sequentially)
            result = await _run_command_async(cmd, cwd)
            results.append(result)

        return ToolResponse(
            content=[
                ContentElement(
                    mime_type=MimeType.APPLICATION_JSON,
                    data={"results": results},
                    metadata={"tool": self.name},
                )
            ]
        )


async def _run_command_async(command: str, dir: Optional[str] = None) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=dir or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return {
            "command": command,
            "dir": dir,
            "returncode": process.returncode,
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip(),
        }
    except Exception as e:
        return {"command": command, "dir": dir, "error": str(e)}


def _run_command(command: str, dir: Optional[str] = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=dir or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {
            "command": command,
            "dir": dir,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as e:
        return {"command": command, "dir": dir, "error": str(e)}
