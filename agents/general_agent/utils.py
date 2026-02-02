#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for General Agent components (Planner, Executor, Summary).

This module provides shared utilities for converting FunctionTool to ClaudeCodeAgent
custom_tools format, building database tools, and managing skills.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from loongflow.agentsdk.tools import FunctionTool
from loongflow.agentsdk.logger import get_logger

logger = get_logger(__name__)


def load_skills(skill_names: List[str], work_dir: str) -> None:
    """
    Load skills from project root .claude/skills directory to working directory.
    Copies ALL files and subdirectories from the skill directory recursively.

    Args:
        skill_names: List of skill directory names to load
        work_dir: Target working directory to copy skills to

    Raises:
        FileNotFoundError: If skill directory not found
        ValueError: If skill_names is empty or invalid

    Example:
        Given skill directory structure:
        LoongFlow/.claude/skills/skill-creator/
        ├── SKILL.md

        Will be copied to:
        {work_dir}/.claude/skills/skill-creator/ (with all contents)
    """
    if not skill_names or not isinstance(skill_names, list):
        raise ValueError("skill_names must be a non-empty list")

    # Use skills from project root .claude/skills directory
    # Get the root directory of the LoongFlow project
    # utils.py is at agents/general_agent/utils.py, so we need to go up 3 levels
    project_root = Path(__file__).parent.parent.parent
    base_skills_dir = project_root / ".claude" / "skills"

    if not base_skills_dir.exists():
        raise FileNotFoundError(
            f"Global skills directory not found: {base_skills_dir}. "
            f"Please create skills in the project root: {project_root}/.claude/skills/"
        )

    target_skills_dir = Path(work_dir) / ".claude" / "skills"
    target_skills_dir.mkdir(parents=True, exist_ok=True)

    for skill_name in skill_names:
        skill_src = base_skills_dir / skill_name
        if not skill_src.exists():
            raise FileNotFoundError(
                f"Skill '{skill_name}' not found in global skills directory: {base_skills_dir}. "
                f"Available skills: {[d.name for d in base_skills_dir.iterdir() if d.is_dir()]}"
            )

        skill_dst = target_skills_dir / skill_name

        # Remove existing skill directory if exists
        if skill_dst.exists():
            shutil.rmtree(skill_dst)

        # Copy all contents recursively with full metadata
        shutil.copytree(
            src=skill_src,
            dst=skill_dst,
            symlinks=False,  # Copy actual files instead of symlinks
            ignore=None,  # Copy everything
            copy_function=shutil.copy2,  # Preserve file metadata
            ignore_dangling_symlinks=True,
        )

        # Log copied files
        copied_files = [
            str(p.relative_to(skill_src)) for p in skill_src.glob("**/*") if p.is_file()
        ]
        logger.info(
            f"Loaded skill '{skill_name}' with {len(copied_files)} files: "
            f"{', '.join(copied_files[:3])}" + ("..." if len(copied_files) > 3 else "")
        )


def convert_function_tool_to_custom_tool(tool: FunctionTool) -> Dict[str, Any]:
    """
    Convert a FunctionTool to ClaudeCodeAgent custom_tools format.

    This adapter bridges the FunctionTool interface (with Pydantic args_schema)
    to the ClaudeCodeAgent custom_tools format (with async function and parameters dict).

    Args:
        tool: FunctionTool instance with func, args_schema, name, description

    Returns:
        Dict in format: {
            "function": async_func,
            "description": str,
            "parameters": dict
        }
    """
    # Get declaration from FunctionTool (contains name, description, parameters)
    declaration = tool.get_declaration()

    # Extract parameters schema from declaration
    params_schema = declaration.get("parameters", {})
    properties = params_schema.get("properties", {})

    # Convert JSON schema types to Python types for ClaudeCodeAgent
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    parameters = {}
    for param_name, param_info in properties.items():
        json_type = param_info.get("type", "string")
        parameters[param_name] = type_mapping.get(json_type, str)

    # Create async wrapper that calls FunctionTool.arun and converts response
    async def wrapper(args: Dict[str, Any]) -> Dict[str, Any]:
        # Note: ClaudeCodeAgent passes tool_context separately, but FunctionTool.arun
        # requires it as a parameter. Since we don't have access to tool_context here,
        # we call arun without it and handle potential errors.
        try:
            response = await tool.arun(args=args)
            # Convert ToolResponse to ClaudeCodeAgent expected format
            if response.content:
                result_data = response.content[0].data
                if isinstance(result_data, str):
                    text = result_data
                else:
                    text = json.dumps(result_data, ensure_ascii=False, indent=2)
            else:
                text = "No result"
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            # If arun fails (e.g., due to missing tool_context), fallback to calling the function directly
            logger.warning(
                f"Tool arun failed for {tool.name}, falling back to direct call: {e}"
            )
            try:
                # Call the underlying function directly
                result = await tool.func(**args)
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as fallback_error:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool execution failed: {fallback_error}",
                        }
                    ]
                }

    return {
        "function": wrapper,
        "description": declaration.get("description", ""),
        "parameters": parameters,
    }


def build_custom_tools_from_function_tools(
    function_tools: list[FunctionTool],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of FunctionTool instances to ClaudeCodeAgent custom_tools format.

    This is a generic utility for converting any FunctionTool to custom tools,
    not just database tools.

    Args:
        function_tools: List of FunctionTool instances

    Returns:
        Dict of custom tools in ClaudeCodeAgent format
    """
    custom_tools = {}
    for tool in function_tools:
        custom_tools[tool.name] = convert_function_tool_to_custom_tool(tool)
    return custom_tools


def format_loaded_skills(skill_names: Optional[List[str]], work_dir: str) -> str:
    """
    Format loaded skills information for prompt injection.

    This function reads the SKILL.md files from the workspace and extracts
    the name and description from the YAML frontmatter to provide context
    to the agent about available skills.

    Args:
        skill_names: List of skill names that were loaded, or None if no skills
        work_dir: The workspace directory where skills were loaded

    Returns:
        A formatted string describing available skills for prompt injection.
        Returns default message if no skills are loaded.
    """
    from loongflow.framework.claude_code import DEFAULT_LOADED_SKILLS

    if not skill_names:
        return DEFAULT_LOADED_SKILLS

    skills_info = []
    skills_dir = Path(work_dir) / ".claude" / "skills"

    for skill_name in skill_names:
        skill_md_path = skills_dir / skill_name / "SKILL.md"

        if skill_md_path.exists():
            try:
                content = skill_md_path.read_text(encoding="utf-8")
                # Extract YAML frontmatter (between --- markers)
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1].strip()
                        # Parse simple YAML-like frontmatter
                        name = skill_name
                        description = ""
                        for line in frontmatter.split("\n"):
                            if line.startswith("name:"):
                                name = line.split(":", 1)[1].strip().strip('"\'')
                            elif line.startswith("description:"):
                                description = line.split(":", 1)[1].strip().strip('"\'')

                        skills_info.append(f"- **{name}**: {description}")
                        continue
            except Exception as e:
                logger.warning(f"Failed to read SKILL.md for '{skill_name}': {e}")

            # Fallback if parsing fails
            skills_info.append(f"- **{skill_name}**: Check `.claude/skills/{skill_name}/SKILL.md` for details")
        else:
            skills_info.append(f"- **{skill_name}**: Skill loaded but SKILL.md not found")

    if not skills_info:
        return DEFAULT_LOADED_SKILLS

    result = "The following skills are available in this workspace:\n"
    result += "\n".join(skills_info)
    result += "\n\nTo use a skill, read its SKILL.md file for detailed instructions."
    return result
