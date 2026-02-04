#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for building tools for the executor.
"""

import json
import os
import subprocess
import time
import uuid
from typing import Type, Optional

from pydantic import BaseModel, Field

from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import Message
from loongflow.agentsdk.models.base_llm_model import BaseLLMModel
from loongflow.agentsdk.tools import AgentTool, Toolkit, FunctionTool
from loongflow.framework.pes.context import Context, Workspace
from loongflow.framework.pes.evaluator.evaluator import LoongFlowEvaluator
from loongflow.framework.react import ReActAgent

logger = get_logger(__name__)


class EvaluateCodeArgs(BaseModel):
    """Arguments for running LoongFlowEvaluator as a FunctionTool."""

    code_file_path: str = Field(
        description="Python code file path (solution) to be evaluated by LoongFlowEvaluator."
    )


class WriteToolArgs(BaseModel):
    """
    Arguments for WriteTool.
    """

    file_path: str = Field(
        ..., description="The absolute or relative path to the file to write"
    )
    content: str = Field(..., description="The content to write to the file")


class LsToolArgs(BaseModel):
    """
    Arguments for LsTool.
    - path: absolute path to directory to list.
    - ignore: optional list of glob patterns to ignore.
    """

    path: str = Field(
        ..., description="The absolute or relative path to the workspace to list"
    )
    ignore: Optional[list[str]] = Field(
        None, description="List of glob patterns to ignore"
    )


class ReadToolArgs(BaseModel):
    """
    Arguments for ReadTool.
    - file_path: absolute or relative path to the file to read.
    """

    file_path: str = Field(
        ..., description="The absolute or relative path to the file to read"
    )


class ReActAgentInputSchema(BaseModel):
    """Input schema for ReActAgent when wrapped as a tool."""

    request: str = Field(description="User query or task instruction text.")


def build_evaluator_solution_tool(
    evaluator: LoongFlowEvaluator, context: Context, candidate_path: str
) -> FunctionTool:
    """Build a FunctionTool for running LoongFlowEvaluator."""

    async def evaluate_solution_func(code_file_path: str):
        """
        Evaluate the given Python solution and automatically write results

        Args:
            code_file_path (str): Path returned by LLM, may be relative or absolute.

        Returns:
            dict: {
                "evaluation_file_path": <path>,
                "solution_file_path": <path>,
                ...<evaluation JSON fields>...
            }
        """
        # Normalize paths for comparison
        candidate_path_abs = os.path.abspath(candidate_path)

        if os.path.isabs(code_file_path):
            requested_path_abs = os.path.abspath(code_file_path)
        else:
            # Try interpreting path relative to candidate_path
            path_rel_candidate = os.path.abspath(
                os.path.join(candidate_path, code_file_path)
            )
            # Try interpreting path relative to CWD
            path_rel_cwd = os.path.abspath(code_file_path)

            # Heuristic: if path relative to candidate_path does not exist,
            # but path relative to CWD exists and is within candidate_path, use it.
            if (
                not os.path.exists(path_rel_candidate)
                and os.path.exists(path_rel_cwd)
                and path_rel_cwd.startswith(candidate_path_abs)
            ):
                requested_path_abs = path_rel_cwd
            else:
                requested_path_abs = path_rel_candidate

        # Strict path validation: ensure requested path is within candidate_path
        if not requested_path_abs.startswith(candidate_path_abs):
            raise ValueError(
                f"Path '{code_file_path}' is outside permitted workspace '{candidate_path}'"
            )

        # Use code absolute path
        code_file_path = requested_path_abs

        # Verify path exists and is a file
        if not os.path.exists(code_file_path):
            raise ValueError(f"File '{code_file_path}' does not exist")
        if not os.path.isfile(code_file_path):
            raise ValueError(f"Path '{code_file_path}' is not a file")

        logger.info(
            f"Trace ID: {context.trace_id}: Executor: Successfully generated solution in React Mode, "
            + f", solution_file_path={code_file_path}"
        )
        # Read the solution code from file
        try:
            with open(code_file_path, "r") as f:
                code = f.read()
        except Exception as e:
            logger.error(
                "Trace ID: %s: Executor: Failed to read solution file '%s': %s",
                context.trace_id,
                code_file_path,
                str(e),
            )
            return {
                "error": f"Failed to read solution file '{code_file_path}': {str(e)}"
            }

        message = Message.from_text(data=code, role="user")
        try:
            result = await evaluator.evaluate(message)
            if result is None:
                data = {"error": "Evaluation returned None"}
            else:
                data = json.loads(result.to_json())

            random_str = uuid.uuid4().hex[:3]
            Workspace.write_executor_file(
                context,
                f"{candidate_path}/evaluation_{random_str}.json",
                json.dumps(data, ensure_ascii=False, indent=2),
            )

            Workspace.write_executor_file(
                context, f"{candidate_path}/solution_{random_str}.py", code
            )
            logger.info(
                "Trace ID: %s: Executor: React Mode Get Evaluation Result: %s, solution_file_path: %s",
                context.trace_id,
                json.dumps(data, ensure_ascii=False),
                code_file_path,
            )

            history_log_file_path = (
                f"{Workspace.get_executor_path(context)}/history.log"
            )
            with open(history_log_file_path, "r") as f:
                history_log = f.readlines()

            logger.debug(
                f"Trace ID: {context.trace_id}: Executor: Update History log "
                + f"with solution_file_path={code_file_path}"
            )
            history = {
                "solution_file_path": code_file_path,
                "evaluation_result": json.dumps(data, ensure_ascii=False, indent=2),
                "evaluate_timestamp": time.time(),
            }

            history_log.append(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
            with open(history_log_file_path, "w") as f:
                f.writelines(history_log)

            return {**data}

        except Exception as e:
            return {"error": f"Exception during evaluation: {str(e)}"}

    return FunctionTool(
        func=evaluate_solution_func,
        args_schema=EvaluateCodeArgs,
        name="evaluate_solution",
        description="Run LoongFlowEvaluator on provided Python solution code.",
    )


def build_executor_write_tool(context: Context, candidate_path: str) -> FunctionTool:
    """Build a FunctionTool for writing executor files in LoongFlow framework.

    Ensures files are written under candidate_path. Automatically fixes
    paths that are not under candidate_path.
    """

    async def write_func(file_path: str, content: str) -> dict:
        """Write content to the file system with path validation.

        Args:
            file_path (str): Path returned by LLM, may be relative or absolute.
            content (str): File content to write.

        Returns:
            dict: { "path": final_path, "message": info_message }
        """
        try:
            # Determine final path
            if file_path.startswith(candidate_path):
                resolved_path = file_path
            else:
                filename = os.path.basename(file_path)
                resolved_path = os.path.join(candidate_path, filename)
                logger.warning(
                    "Trace ID: %s, Executor, Write Tool: file_path '%s' corrected to '%s'",
                    context.trace_id,
                    file_path,
                    resolved_path,
                )

            # Write the file
            written_path = Workspace.write_executor_file(
                context, resolved_path, content
            )
            logger.debug(
                "Trace ID: %s, Executor, Write file: %s", context.trace_id, written_path
            )

            return {
                "path": written_path,
                "message": f"File written successfully to {written_path}",
            }

        except Exception as e:
            logger.error(
                "Trace ID: Executor: Write Tool Failed to write file '%s' (resolved: '%s'): %s",
                context.trace_id,
                file_path,
                resolved_path,
                e,
            )
            raise ValueError(f"Failed to write file '{file_path}'") from e

    return FunctionTool(
        func=write_func,
        args_schema=WriteToolArgs,
        name="Write",
        description=(
            "Writes a file to the local filesystem. "
            "If path is not under candidate_path, filename is appended automatically. "
            "Overwrites existing files."
        ),
    )


def build_executor_ls_tool(context: Context, candidate_path: str) -> FunctionTool:
    """Build a FunctionTool for listing files in LoongFlow framework.

    Ensures only paths within candidate_path are accessible.
    Rejects paths outside candidate_path with absolute restriction.
    """

    async def ls_func(path: str, ignore: Optional[list[str]] = None) -> dict:
        """List files with strict path validation.

        Args:
            path (str): Path within candidate_path (relative or absolute).
            ignore (list[str]): Optional list of glob patterns to ignore.

        Returns:
            dict: {"files": []}

        Raises:
            ValueError: If path is outside candidate_path
        """
        try:
            # Normalize paths for comparison
            candidate_path_abs = os.path.abspath(candidate_path)

            if os.path.isabs(path):
                requested_path_abs = os.path.abspath(path)
            else:
                # Try interpreting path relative to candidate_path
                path_rel_candidate = os.path.abspath(os.path.join(candidate_path, path))
                # Try interpreting path relative to CWD
                path_rel_cwd = os.path.abspath(path)

                # Heuristic: if path relative to candidate_path does not exist,
                # but path relative to CWD exists and is within candidate_path, use it.
                if (
                    not os.path.exists(path_rel_candidate)
                    and os.path.exists(path_rel_cwd)
                    and path_rel_cwd.startswith(candidate_path_abs)
                ):
                    requested_path_abs = path_rel_cwd
                else:
                    requested_path_abs = path_rel_candidate

            # Strict path validation: ensure requested path is within candidate_path
            if not requested_path_abs.startswith(candidate_path_abs):
                raise ValueError(
                    f"Path '{path}' is outside permitted workspace '{candidate_path}'"
                )

            # Use resolved absolute path
            resolved_path = requested_path_abs

            # Verify path exists and is a directory
            if not os.path.exists(resolved_path):
                raise ValueError(f"Path '{resolved_path}' does not exist")
            if not os.path.isdir(resolved_path):
                raise ValueError(f"Path '{resolved_path}' is not a directory")

            # List files
            all_files = []
            ignore_patterns = ignore or []

            for name in os.listdir(resolved_path):
                # Skip hidden files and patterns
                if name.startswith("."):
                    continue

                # Apply ignore patterns
                import fnmatch

                if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns):
                    continue

                full_path = os.path.join(resolved_path, name)
                all_files.append(
                    {
                        "name": name,
                        "path": full_path,
                        "relative_path": (
                            os.path.relpath(full_path, candidate_path)
                            if full_path.startswith(candidate_path)
                            else name
                        ),
                        "is_dir": os.path.isdir(full_path),
                        "size": (
                            os.path.getsize(full_path)
                            if not os.path.isdir(full_path)
                            else 0
                        ),
                    }
                )

            logger.debug(
                "Trace ID: %s, Executor, Ls files: %s", context.trace_id, resolved_path
            )

            return {"files": all_files}

        except ValueError as e:
            logger.error(
                "Trace ID: %s, Executor: Ls Tool Path Validation Failed for '%s': %s",
                context.trace_id,
                path,
                str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "Trace ID: %s, Executor: Ls Tool Failed to list directory '%s': %s",
                context.trace_id,
                path,
                str(e),
            )
            raise ValueError(f"Failed to list directory '{path}'") from e

    return FunctionTool(
        func=ls_func,
        args_schema=LsToolArgs,
        name="LS",
        description=(
            "Lists files and directories within the permitted workspace. "
            "Rejects paths outside the designated candidate_path."
        ),
    )


def build_executor_read_tool(context: Context, candidate_path: str) -> FunctionTool:
    """Build a FunctionTool for reading files in LoongFlow framework.

    Ensures only files within candidate_path are accessible.
    Rejects paths outside candidate_path with absolute restriction.
    """

    async def read_func(file_path: str) -> dict:
        """Read file content with strict path validation.

        Args:
            file_path (str): Path within candidate_path (relative or absolute).

        Returns:
            dict: {
                "file_path": str,
                "content": str,
                "size": int,
                "message": str
            }

        Raises:
            ValueError: If path is outside candidate_path or file not found
        """
        try:
            # Normalize paths for comparison
            candidate_path_abs = os.path.abspath(candidate_path)

            if os.path.isabs(file_path):
                requested_path_abs = os.path.abspath(file_path)
            else:
                # Try interpreting path relative to candidate_path
                path_rel_candidate = os.path.abspath(
                    os.path.join(candidate_path, file_path)
                )
                # Try interpreting path relative to CWD
                path_rel_cwd = os.path.abspath(file_path)

                # Heuristic: if path relative to candidate_path does not exist,
                # but path relative to CWD exists and is within candidate_path, use it.
                if (
                    not os.path.exists(path_rel_candidate)
                    and os.path.exists(path_rel_cwd)
                    and path_rel_cwd.startswith(candidate_path_abs)
                ):
                    requested_path_abs = path_rel_cwd
                else:
                    requested_path_abs = path_rel_candidate

            # Strict path validation: ensure requested path is within candidate_path
            if not requested_path_abs.startswith(candidate_path_abs):
                raise ValueError(
                    f"Path '{file_path}' is outside permitted workspace '{candidate_path}'"
                )

            # Use resolved absolute path
            resolved_path = requested_path_abs

            # Verify path exists and is a file
            if not os.path.exists(resolved_path):
                raise ValueError(f"File '{resolved_path}' does not exist")
            if not os.path.isfile(resolved_path):
                raise ValueError(f"Path '{resolved_path}' is not a file")

            # Read file content
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "file_path": resolved_path,
                "content": content,
                "message": f"File read successfully from {resolved_path}",
            }

        except ValueError as e:
            logger.error(
                "Trace ID: %s, Executor: Read Tool Path Validation Failed for '%s': %s",
                context.trace_id,
                file_path,
                str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "Trace ID: %s, Executor: Read Tool Failed to read file '%s': %s",
                context.trace_id,
                file_path,
                str(e),
            )
            raise ValueError(f"Failed to read file '{file_path}'") from e

    return FunctionTool(
        func=read_func,
        args_schema=ReadToolArgs,
        name="Read",
        description=(
            "Reads the content of a file within the permitted workspace. "
            "Rejects paths outside the designated candidate_path. "
            "Supports text files with UTF-8 encoding."
        ),
    )


class InstallPackageArgs(BaseModel):
    """Arguments for installing Python packages."""

    package_name: str = Field(..., description="Name of the Python package to install")
    version: Optional[str] = Field(
        None, description="Specific version of the package to install"
    )
    upgrade: bool = Field(
        False, description="Whether to upgrade the package if already installed"
    )


def build_install_package_tool() -> FunctionTool:
    """Build a FunctionTool for installing Python packages."""

    async def install_package_func(
        package_name: str, version: Optional[str] = None, upgrade: bool = False
    ) -> dict:
        """
        Install a Python package using pip.

        Args:
            package_name: Name of the package to install
            version: Specific version to install (optional)
            upgrade: Whether to upgrade if already installed

        Returns:
            dict: {
                "success": bool,
                "message": str,
                "output": str
            }
        """
        package_spec = package_name
        if version:
            package_spec = f"{package_name}=={version}"

        command = ["pip", "install"]
        if upgrade:
            command.append("--upgrade")
        command.append(package_spec)

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            return {
                "success": True,
                "message": f"Successfully installed {package_spec}",
                "output": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            logger.error("Failed to install package %s: %s", package_spec, e.stderr)
            return {
                "success": False,
                "message": f"Failed to install {package_spec}",
                "output": e.stderr,
            }

    return FunctionTool(
        func=install_package_func,
        args_schema=InstallPackageArgs,
        name="install_package",
        description="Install a Python package using pip. Can specify version and upgrade options.",
    )


def build_agent_tool(
    sys_prompt: str,
    model: BaseLLMModel,
    toolkit: Toolkit,
    output_format: Type[BaseModel],
) -> AgentTool:
    """Factory for building an AgentTool from ReActAgent."""

    agent = ReActAgent.create_default(
        model=model,
        sys_prompt=sys_prompt,
        output_format=output_format,
        toolkit=toolkit,
    )

    agent.input_schema = ReActAgentInputSchema
    agent.name = "execute_agent"
    agent.description = "Execute reasoning and tool-using ReActAgent."

    async def wrapped_agent_call(request: str) -> Message:
        message = Message.from_text(request, role="user")
        return await agent.run(initial_messages=message)

    agent.__call__ = wrapped_agent_call  # type: ignore

    return AgentTool(agent)
