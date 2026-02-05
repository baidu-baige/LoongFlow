#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides general planner implementation based on Claude Code Agent
"""

import copy
import json
import os
from typing import Any

from agents.general_agent.common import ClaudeAgentConfig
from agents.general_agent.utils import (
    build_custom_tools_from_function_tools,
    format_loaded_skills,
    _prepare_solution_pack_context,
)
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import Message, MimeType, ContentElement
from loongflow.framework.claude_code import GENERAL_PLANNER_USER, GENERAL_PLANNER_SYSTEM
from loongflow.framework.pes.context import Context, Workspace
from loongflow.framework.pes.database import EvolveDatabase
from loongflow.framework.pes.register import Worker
from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent
from loongflow.framework.pes.database.database_tool import (
    GetBestSolutionsTool,
    GetChildsByParentTool,
    GetMemoryStatusTool,
    GetParentsByChildIdTool,
    GetSolutionsTool,
)

logger = get_logger(__name__)

BEST_PLAN_FILE = "best_plan.md"


class GeneralPlanAgent(Worker):
    """Plan Agent Class"""

    def __init__(self, config: Any, db: EvolveDatabase):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ClaudeAgentConfig)
            else ClaudeAgentConfig(**config)
        )

        if self.config.llm_config is None:
            raise ValueError(
                "Planner: No LLMConfig found in config, please check your config."
            )

        llm_config = copy.deepcopy(self.config.llm_config)
        if not llm_config.model.startswith("anthropic/"):
            raise ValueError(
                "Planner: Only support Anthropic model, please use model name like anthropic/xxx."
            )
        llm_config.model = llm_config.model.split("/")[-1]
        self.config.llm_config = llm_config

        self.database = db

        self.custom_tools = [
            GetMemoryStatusTool(self.database.memory_status),
            GetSolutionsTool(self.database.get_solutions),
            GetBestSolutionsTool(self.database.get_best_solutions),
            GetParentsByChildIdTool(self.database.get_parents_by_child_id),
            GetChildsByParentTool(self.database.get_childs_by_parent_id),
        ]

        logger.debug("Planner: Core tools registered successfully")

    def _process_parent_solution(self, parent_dict: dict, context: Context) -> dict:
        """
        Process parent solution to support multi-file solution packs.

        All solutions MUST be directories (solution packs). This method loads the
        solution context (tree + manifest) and formats it for the agent.

        Args:
            parent_dict: Parent solution dictionary from database
            context: Current execution context

        Returns:
            Modified parent_dict with formatted solution content
        """
        solution_path = parent_dict.get("solution", "")

        if "metadata" not in parent_dict:
            parent_dict["metadata"] = {}

        # Ensure absolute path
        if solution_path and not os.path.isabs(solution_path):
            solution_path = os.path.abspath(solution_path)
            parent_dict["solution"] = solution_path
            logger.debug(
                f"[{context.trace_id}] Planner: Converted solution path to absolute: {solution_path}"
            )

        # All solutions must be directories (solution packs)
        if not solution_path or not os.path.exists(solution_path):
            logger.warning(
                f"[{context.trace_id}] Planner: No valid solution path found or path does not exist: {solution_path}"
            )
            parent_dict["solution"] = "No prior solution available."
            parent_dict["metadata"]["solution_pack"] = ""
            return parent_dict

        if not os.path.isdir(solution_path):
            raise ValueError(
                f"Solution path must be a directory (solution pack), got file: {solution_path}. "
                "All solutions in general_agent must follow the Solution Pack protocol."
            )

        try:
            logger.debug(
                f"[{context.trace_id}] Planner: Loading solution pack from {solution_path}"
            )

            # Load solution context (lazy loading: only tree + manifest)
            formatted_solution = _prepare_solution_pack_context(solution_path)

            # Replace solution content with formatted view
            parent_dict["solution"] = solution_path
            parent_dict["metadata"]["solution_pack"] = formatted_solution

            logger.info(f"[{context.trace_id}] Planner: Loaded solution pack ")

        except Exception as e:
            logger.error(
                f"[{context.trace_id}] Planner: Failed to load solution pack: {e}"
            )
            raise ValueError(
                f"Failed to load solution pack from {solution_path}: {e}. "
                "Ensure the directory contains a valid index.json manifest."
            )

        return parent_dict

    async def run(self, context: Context, message: Message) -> Message:
        """Execute planning phase."""
        memory_status = self.database.memory_status()
        logger.info(
            f"[{context.trace_id}] Planner: 📝 Starting iteration {context.current_iteration}/{context.total_iterations} (memory: {memory_status})"
        )

        # Create agent with context-specific work_dir
        work_dir = str(Workspace.get_planner_path(context, True))
        # Ensure work_dir is absolute path
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        logger.debug(
            f"[{context.trace_id}] Planner: Workspace configured at {work_dir}"
        )

        # Load skills if specified
        if self.config.skills:
            from .utils import load_skills

            try:
                load_skills(
                    skill_names=self.config.skills,
                    work_dir=work_dir,
                )
                logger.debug(
                    f"[{context.trace_id}] Planner: Successfully loaded skills: {self.config.skills}"
                )
            except Exception as e:
                logger.error(
                    f"[{context.trace_id}] Planner: Failed to load skills - {str(e)}"
                )
                raise

        # Build database tools for the agent
        database_tools = build_custom_tools_from_function_tools(self.custom_tools)

        agent = ClaudeCodeAgent(
            model=self.config.llm_config.model,
            api_key=self.config.llm_config.api_key,
            url=self.config.llm_config.url,
            work_dir=work_dir,
            tool_list=self.config.build_in_tools,
            disallowed_tools=self.config.disallowed_tools,
            custom_tools=database_tools,
            system_prompt=self.config.system_prompt or GENERAL_PLANNER_SYSTEM,
            permission_mode=self.config.permission_mode or "acceptEdits",
            setting_sources=["project"],
            max_turns=self.config.max_turns,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )

        # Prepare initial parent info
        init_parent = {
            "solution": context.init_solution,
            "score": context.init_score or 0.0,
            "evaluation": context.init_evaluation,
            "summary": "This is the initial solution, it has no parents. Start evolution from here.",
        }

        # Sample parent from database
        parent = self.database.sample_solution(context.island_id)
        parent_dict = parent if parent else init_parent

        # Process parent solution for multi-file support
        parent_dict = self._process_parent_solution(parent_dict, context)

        # Save parent info using Workspace
        parent_json = json.dumps(parent_dict, ensure_ascii=False, indent=2)
        Workspace.write_planner_parent_info(context, parent_json)
        parent_info_path = Workspace.get_planner_parent_info_path(context)
        logger.debug(
            f"[{context.trace_id}] Planner: Parent info saved to {parent_info_path}"
        )

        # Get the expected plan path
        # Use absolute path to avoid any relative path confusion
        best_plan_full_path = str(
            Workspace.get_planner_best_plan_path(context, BEST_PLAN_FILE)
        )
        # Pass absolute path to Claude - Claude agent will handle it correctly regardless of current working directory
        best_plan_path_for_claude = best_plan_full_path  # Already the correct absolute path since Workspace returns absolute paths

        # Format loaded skills information for prompt
        loaded_skills_info = format_loaded_skills(self.config.skills, work_dir)

        user_prompt = GENERAL_PLANNER_USER.format(
            task_info=context.task,
            parent_solution=parent_json,
            workspace=f"{work_dir} (absolute path)",
            island_num=self.database.config.num_islands,
            parent_island=parent.get("island_id") if parent else 0,
            best_plan_path=f"{best_plan_path_for_claude} (absolute path)",
            loaded_skills=loaded_skills_info,
        )

        # Execute planning - Claude should use Write tool to save plan to best_plan_path
        result = await agent.run(user_prompt)

        # Check if Claude wrote the plan file (primary path)
        if os.path.exists(best_plan_full_path):
            logger.info(
                f"[{context.trace_id}] Planner: ✅ Plan generated successfully at {best_plan_full_path}"
            )
        else:
            # Fallback: extract plan from Claude's response and save it manually
            logger.warning(
                f"[{context.trace_id}] Planner: ⚠️ Plan file not found, extracting from response"
            )
            # Extract the plan content from Claude's response
            if (
                result.content
                and len(result.content) > 0
                and isinstance(result.content[0], ContentElement)
            ):
                plan_content = result.content[0].data
            else:
                plan_content = str(result.metadata.get("response", "No plan generated"))

            # Save the extracted plan
            Workspace.write_planner_best_plan(context, plan_content, BEST_PLAN_FILE)
            logger.info(
                f"[{context.trace_id}] Planner: ✅ Plan extracted and saved to {best_plan_full_path}"
            )

        # Save metadata to JSON file
        meta_content = {
            "trace_id": context.trace_id,
            "status": result.metadata.get("status"),
            "tools_used": result.metadata.get("tools_used", []),
            "input_tokens": result.metadata.get("input_tokens"),
            "output_tokens": result.metadata.get("output_tokens"),
            "duration_ms": result.metadata.get("duration_ms"),
        }

        # Save metadata to meta.json in planner directory
        meta_file_path = os.path.join(work_dir, "meta.json")
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        logger.debug(
            "Planner: metadata saved",
            extra={
                "trace_id": context.trace_id,
                "input_tokens": result.metadata.get("input_tokens"),
                "output_tokens": result.metadata.get("output_tokens"),
            },
        )

        return Message.from_text(
            data={
                "parent_info_file_path": parent_info_path,
                "best_plan_file_path": best_plan_full_path,
                "total_prompt_tokens": result.metadata.get("input_tokens"),
                "total_completion_tokens": result.metadata.get("output_tokens"),
            },
            mime_type=MimeType.APPLICATION_JSON,
        )
