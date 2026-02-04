#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides general planner implementation based on Claude Code Agent
"""
import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agents.general_agent.common import ClaudeAgentConfig
from agents.general_agent.utils import (
    build_custom_tools_from_function_tools,
    format_loaded_skills,
    _prepare_solution_pack_context,
)
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.memory.evolution import Solution
from loongflow.agentsdk.message import Message, MimeType, ContentElement
from loongflow.framework.claude_code import GENERAL_SUMMARY_SYSTEM, GENERAL_SUMMARY_USER
from loongflow.framework.pes.context import Context, Workspace
from loongflow.framework.pes.database import EvolveDatabase
from loongflow.framework.pes.register import Worker
from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent
from loongflow.framework.pes.database.database_tool import (
    GetChildsByParentTool,
    GetParentsByChildIdTool,
    GetSolutionsTool,
)

logger = get_logger(__name__)

BEST_SUMMARY_FILE = "best_summary.md"


@dataclass
class Evidence:
    """SummaryAgent evidence"""

    best_plan: str = None
    best_solution: str = None
    best_evaluation: dict[str, Any] = None
    parent_info: Solution = None
    current_solution: Solution = None


class Assessment(Enum):
    """SummaryAgent assessment"""

    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    STALE = "stale"


class GeneralSummaryAgent(Worker):
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
                "Summary: Only support Anthropic model, please use model name like anthropic/xxx."
            )
        llm_config.model = llm_config.model.split("/")[-1]
        self.config.llm_config = llm_config

        self.db = db

        self.custom_tools = [
            GetSolutionsTool(self.db.get_solutions),
            GetParentsByChildIdTool(self.db.get_parents_by_child_id),
            GetChildsByParentTool(self.db.get_childs_by_parent_id),
        ]

        logger.debug("Summary: Core tools registered successfully")

    async def run(self, context: Context, message: Message) -> Message:
        """Execute summary phase."""
        logger.info(
            f"[{context.trace_id}] Summary: Starting iteration {context.current_iteration}/{context.total_iterations}"
        )

        evidence = await self._gather(context, message)
        assessment = await self._assess(context, evidence)
        analysis_str = await self._reflect(context, evidence, assessment)
        analysis = json.loads(analysis_str)
        await self._record(context, evidence, analysis.get("reflection", ""))

        score = evidence.current_solution.score if evidence else 0.0
        total_tokens = analysis.get("total_prompt_tokens", 0) + analysis.get(
            "total_completion_tokens", 0
        )
        logger.info(
            f"[{context.trace_id}] Summary: ✅ Iteration completed - score={score:.4f}, tokens={total_tokens}"
        )

        return Message.from_elements(
            [
                ContentElement(
                    mime_type=MimeType.APPLICATION_JSON,
                    data={
                        "best_summary_file_path": Workspace.get_summarizer_best_summary_path(
                            context
                        ),
                        "total_prompt_tokens": analysis.get("total_prompt_tokens", 0),
                        "total_completion_tokens": analysis.get(
                            "total_completion_tokens", 0
                        ),
                    },
                )
            ]
        )

    async def _gather(self, context: Context, message: Message) -> Evidence:
        content = message.get_elements(ContentElement)
        if not content:
            raise ValueError("Message missing ContentElement data.")
        data = content[0].data

        # sometimes we do not have best_plan, then we could ignore it
        try:
            with open(data.get("best_plan_file_path"), "r", encoding="utf-8") as f:
                plan_content = f.read()
        except Exception:
            plan_content = ""

        # Extract paths from Executor's message
        # Note: Executor now returns best_solution_path (directory), not best_solution_file_path
        best_solution_path = data.get("best_solution_path")
        best_evaluation_path = data.get("best_evaluation_path", {})
        parent_info_path = data.get("parent_info_file_path")

        # Validate paths
        if not best_solution_path or not os.path.exists(best_solution_path):
            raise FileNotFoundError(f"Solution path not found: {best_solution_path}")
        if not best_evaluation_path or not os.path.exists(best_evaluation_path):
            raise FileNotFoundError(
                f"Evaluation path not found: {best_evaluation_path}"
            )
        if not os.path.isdir(best_solution_path):
            raise ValueError(f"Solution path must be a directory: {best_solution_path}")
        if not parent_info_path or not os.path.exists(parent_info_path):
            raise FileNotFoundError(f"Parent info path not found: {parent_info_path}")

        # Ensure absolute path
        if not os.path.isabs(best_solution_path):
            best_solution_path = os.path.abspath(best_solution_path)

        logger.debug(
            f"[{context.trace_id}] Summary: Evidence gathered - solution_pack={best_solution_path}"
        )

        # Load parent info
        try:
            with open(parent_info_path, "r", encoding="utf-8") as f:
                parent_info_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read parent info: {e}")

        # Load evaluation info
        try:
            with open(best_evaluation_path, "r", encoding="utf-8") as f:
                best_evaluation_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read evaluation info: {e}")

        # Filter out fields that are not part of Solution class
        # (e.g., solution_path which is added by Planner for Executor's use)
        from dataclasses import fields

        valid_fields = {f.name for f in fields(Solution)}
        filtered_data = {k: v for k, v in parent_info_data.items() if k in valid_fields}

        # parent_info is not enough in file, we need to read it from db
        parent_info = Solution(**filtered_data)

        if solution_id := parent_info_data.get("solution_id", None):
            solutions = self.db.get_solutions([solution_id])
            if solutions and len(solutions) == 1:
                parent_info = Solution.from_dict(solutions[0])

        # Create new solution with directory path (not file content)
        solution_id = uuid.uuid4().hex[:8]
        trace_list = parent_info.metadata.get("trace", []).copy()
        trace_list.append(solution_id)
        metadata = {
            "trace": trace_list,
            "solution_pack": _prepare_solution_pack_context(best_solution_path),
        }

        return Evidence(
            best_evaluation=best_evaluation_data,
            parent_info=parent_info,
            current_solution=Solution(
                solution=best_solution_path,  # Store directory path, not content
                solution_id=solution_id,
                generate_plan=plan_content,
                parent_id=parent_info.solution_id,
                island_id=context.island_id,
                iteration=context.current_iteration,
                timestamp=time.time(),
                generation=len(trace_list),
                score=best_evaluation_data.get("score", 0),
                evaluation=json.dumps(best_evaluation_data, ensure_ascii=False, indent=2),
                metadata=metadata,
            ),
        )

    async def _assess(self, context: Context, evidence: Evidence) -> Assessment:
        evaluation_score = evidence.current_solution.score
        parent_score = evidence.parent_info.score

        logger.debug(
            f"[{context.trace_id}] Summary: Score assessment - child={evaluation_score:.4f}, parent={parent_score:.4f}"
        )

        if evaluation_score > parent_score:
            assessment_result = Assessment.IMPROVEMENT
        elif evaluation_score < parent_score:
            assessment_result = Assessment.REGRESSION
        else:
            assessment_result = Assessment.STALE

        logger.debug(
            f"[{context.trace_id}] Summary: Assessment result - {assessment_result.value}"
        )

        return assessment_result

    async def _reflect(
        self,
        context: Context,
        evidence: Evidence,
        assessment: Assessment,
    ) -> str:

        # Create agent with context-specific work_dir
        work_dir = str(Workspace.get_summarizer_path(context, True))
        # Ensure work_dir is absolute path
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        logger.debug(f"[{context.trace_id}] Summary: Workspace: {work_dir}")

        # Load skills if specified
        if self.config.skills:
            from .utils import load_skills

            try:
                load_skills(
                    skill_names=self.config.skills,
                    work_dir=work_dir,
                )
                logger.debug(
                    f"[{context.trace_id}] Summary: Loaded skills: {self.config.skills}"
                )
            except Exception as e:
                logger.error(
                    f"[{context.trace_id}] Summary: Failed to load skills: {str(e)}"
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
            custom_tools=database_tools,
            system_prompt=self.config.system_prompt or GENERAL_SUMMARY_SYSTEM,
            permission_mode=self.config.permission_mode or "acceptEdits",
            setting_sources=["project"],
            max_turns=self.config.max_turns,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )

        # Get the expected summary path
        # Use absolute path to avoid any relative path confusion
        best_summary_full_path = str(
            Workspace.get_summarizer_best_summary_path(context, BEST_SUMMARY_FILE)
        )
        # Pass absolute path to Claude - Claude agent will handle it correctly regardless of current working directory
        best_summary_path_for_claude = best_summary_full_path  # Already the correct absolute path since Workspace returns absolute paths

        # Format loaded skills information for prompt
        loaded_skills_info = format_loaded_skills(self.config.skills, work_dir)

        user_prompt = GENERAL_SUMMARY_USER.format(
            task_info=context.task,
            parent_solution=json.dumps(
                evidence.parent_info.to_dict(), indent=2, ensure_ascii=False
            ),
            child_solution=json.dumps(
                evidence.current_solution.to_dict(), indent=2, ensure_ascii=False
            ),
            assessment_result=assessment.value,
            workspace=f"{work_dir} (absolute path)",
            summary_path=f"{best_summary_path_for_claude} (absolute path)",
            loaded_skills=loaded_skills_info,
        )

        result = await agent.run(user_prompt)

        # Check if Claude wrote the plan file (primary path)
        if os.path.exists(best_summary_full_path):
            logger.info(
                f"[{context.trace_id}] Summary: ✅ Summary generated at {best_summary_full_path}"
            )
        else:
            # Fallback: extract plan from Claude's response and save it manually
            logger.warning(
                f"[{context.trace_id}] Summary: ⚠️ Summary file not found, extracting from response"
            )
            # Extract the plan content from Claude's response
            if (
                result.content
                and len(result.content) > 0
                and isinstance(result.content[0], ContentElement)
            ):
                plan_content = result.content[0].data
            else:
                plan_content = str(
                    result.metadata.get("response", "No summary generated")
                )

            # Save the extracted summary
            Workspace.write_summarizer_best_summary(
                context, plan_content, BEST_SUMMARY_FILE
            )
            logger.info(f"[{context.trace_id}] Summary: ✅ Summary extracted and saved")

        reflection = ""
        with open(best_summary_path_for_claude, "r") as f:
            reflection = f.read()

        final_result = {
            "reflection": reflection,
            "total_prompt_tokens": result.metadata.get("input_tokens", 0),
            "total_completion_tokens": result.metadata.get("output_tokens", 0),
        }
        return json.dumps(final_result, ensure_ascii=False, indent=2)

    async def _record(
        self, context: Context, evidence: Evidence, reflection: str
    ) -> None:
        child_weight = 0.05

        # If Parent exists in the database, set child's weight based on parent's weight
        if evidence.parent_info.solution_id:
            parent_solution = evidence.parent_info
            child_solution = evidence.current_solution
            parent_weight = parent_solution.sample_weight
            # Calculate the score difference; the larger the difference, the greater the weight adjustment
            score_diff = child_solution.score - parent_solution.score
            # The amplitude of weight iteration is determined by the iteration progress;
            # larger in early iterations, smaller later on
            step_size = 1 - (context.current_iteration / context.total_iterations)
            # This formula automatically ensures that if score_diff is negative,
            # child_weight is lower than parent_weight, otherwise higher. ALPHA = 2 is the amplification factor
            child_weight = (
                parent_weight + (3 * score_diff * step_size) + 3 * child_solution.score
            )
            if child_weight < 0:
                # Prevent weight from being too small; minimum value is 0.05
                child_weight = 0.05

            logger.debug(
                f"[{context.trace_id}] Summary: Weight adjustment - parent={parent_weight:.4f}, diff={score_diff:.4f}, child={child_weight:.4f}"
            )

            await self.db.update_solution(
                evidence.parent_info.solution_id,
                sample_cnt=evidence.parent_info.sample_cnt + 1,
            )
            logger.debug(f"[{context.trace_id}] Summary: Parent solution updated")

        evidence.current_solution.summary = reflection
        evidence.current_solution.sample_weight = child_weight

        await self.db.add_solution(evidence.current_solution)

        logger.info(
            f"[{context.trace_id}] Summary: ✅ New solution added - id={evidence.current_solution.solution_id}, score={evidence.current_solution.score:.4f}"
        )

        return
