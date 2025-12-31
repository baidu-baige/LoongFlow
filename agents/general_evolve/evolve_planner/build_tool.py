#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for building tools for the executor.
"""

from pydantic import BaseModel, Field

from agentsdk.tools import FunctionTool
from evolux.evolve.context import Context, Workspace


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


def build_planner_write_tool(context: Context) -> FunctionTool:
    """Build a FunctionTool for running the write function."""

    async def write_func(file_path: str, content: str):
        planner_base = Workspace.get_planner_path(context)

        try:
            if not file_path.startswith(str(planner_base)):
                if "plan1.txt" in file_path:
                    plan1_path = planner_base / "plan1.txt"
                    with open(plan1_path, "w") as f:
                        f.write(content)
                if "plan2.txt" in file_path:
                    plan2_path = planner_base / "plan2.txt"
                    with open(plan2_path, "w") as f:
                        f.write(content)
                if "plan3.txt" in file_path:
                    plan3_path = planner_base / "plan3.txt"
                    with open(plan3_path, "w") as f:
                        f.write(content)
            else:
                with open(file_path, "w") as f:
                    f.write(content)

            return "File written successfully"
        except Exception as e:
            raise ValueError(f"Failed to write file: {e}")

    return FunctionTool(
        func=write_func,
        args_schema=WriteToolArgs,
        name="Write",
        description="Writes a file to the local filesystem. "
        + "Will overwrite existing files. Supports both absolute and relative paths.",
    )
