#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides general planner implementation based on Claude Code Agent
"""
import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from agents.general_agent.common import ClaudeAgentConfig
from agents.general_agent.evaluator import GeneralEvaluator
from agents.general_agent.utils import format_loaded_skills
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import Message, ContentElement, Role, MimeType
from loongflow.framework.claude_code import (
    GENERAL_EXECUTOR_SYSTEM,
    GENERAL_EXECUTOR_USER,
)
from loongflow.framework.pes.context import Context, Workspace
from loongflow.framework.pes.register import Worker
from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """Holds previous candidate and stage1 plan information."""

    parent_info_file_path: str
    parent_core: float
    parent_solution: str
    parent_solution_path: str  # Absolute path to parent solution directory
    stage1_plan: str
    stage1_plan_file_path: str


class GeneralExecuteAgent(Worker):
    """Execute Agent - Linear execution with Solution Pack generation"""

    def __init__(self, config: Any, evaluator: GeneralEvaluator):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ClaudeAgentConfig)
            else ClaudeAgentConfig(**config)
        )

        if self.config.llm_config is None:
            raise ValueError(
                "Executor: No LLMConfig found in config, please check your config."
            )

        llm_config = copy.deepcopy(self.config.llm_config)
        if not llm_config.model.startswith("anthropic/"):
            raise ValueError(
                "Executor: Only support Anthropic model, please use model name like anthropic/xxx."
            )
        llm_config.model = llm_config.model.split("/")[-1]
        self.config.llm_config = llm_config

        self.evaluator = evaluator

        logger.debug("Executor: Core configuration loaded successfully")

    async def run(self, context: Context, message: Message) -> Message:
        """
        Execute execution phase with linear workflow.

        Workflow:
        1. Create candidate directory (Solution Pack)
        2. Clone parent solution if exists
        3. Execute agent once
        4. Ensure manifest generation
        5. Return Solution Pack directory path
        """
        logger.info(
            f"[{context.trace_id}] Executor: 🚀 Starting iteration {context.current_iteration}/{context.total_iterations}"
        )

        parent_ctx = self._parse_message_inputs(message)

        # Create agent with context-specific work_dir
        executor_dir = Workspace.get_executor_path(context, True)
        # Create sub-directory work_dir
        work_dir = executor_dir / "work_dir"
        work_dir.mkdir(exist_ok=True)
        work_dir = str(work_dir)

        # Create sub-directory evaluator_dir
        evaluator_dir = executor_dir / "evaluator_dir"
        evaluator_dir.mkdir(exist_ok=True)
        evaluator_dir = str(evaluator_dir)

        # Ensure work_dir is absolute path
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        logger.debug(
            f"[{context.trace_id}] Executor: Workspace configured at {work_dir}"
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
                    f"[{context.trace_id}] Executor: Successfully loaded skills: {self.config.skills}"
                )
            except Exception as e:
                logger.error(
                    f"[{context.trace_id}] Executor: Failed to load skills - {str(e)}"
                )
                raise

        # Step 1: Use work_dir as candidate directory (Solution Pack)
        candidate_dir = work_dir
        logger.debug(
            f"[{context.trace_id}] Executor: Using work_dir as candidate directory: {candidate_dir}"
        )

        # Step 2: Clone parent solution if exists
        parent_solution_path = parent_ctx.parent_solution_path
        if parent_solution_path and os.path.isdir(parent_solution_path):
            try:
                from .utils import copy_solution_files

                copy_solution_files(parent_solution_path, candidate_dir)
                logger.info(
                    f"[{context.trace_id}] Executor: Cloned parent solution from {parent_solution_path}"
                )
            except Exception as e:
                logger.warning(
                    f"[{context.trace_id}] Executor: Failed to clone parent solution: {e}. Starting fresh."
                )

        # Step 3: Execute agent once (linear execution)
        result_data = await self._execute_once(context, parent_ctx, candidate_dir)

        # Step 4: Ensure manifest generation
        try:
            from .utils import ensure_manifest

            await ensure_manifest(candidate_dir, self.config.llm_config.model_dump())
            logger.info(
                f"[{context.trace_id}] Executor: Manifest ensured for {candidate_dir}"
            )
        except Exception as e:
            logger.error(
                f"[{context.trace_id}] Executor: Failed to ensure manifest: {e}"
            )

        # Step 5: Evaluate the solution pack
        logger.info(
            f"[{context.trace_id}] Executor: Starting evaluation of solution pack"
        )
        try:
            # Create a message with the solution directory path
            solution_message = Message.from_text(data=candidate_dir, role=Role.USER)
            evaluation_result = await self.evaluator.evaluate(solution_message, context)

            if evaluation_result is None:
                logger.warning(
                    f"[{context.trace_id}] Executor: Evaluation returned None"
                )
                evaluation_data = {
                    "score": 0.0,
                    "summary": "Evaluation failed or returned no data",
                    "status": "no_evaluation",
                    "metrics": {},
                }
            else:
                evaluation_data = {
                    "score": evaluation_result.score,
                    "summary": evaluation_result.summary,
                    "status": evaluation_result.status.value,
                    "metrics": evaluation_result.metrics,
                }
                logger.info(
                    f"[{context.trace_id}] Executor: Evaluation completed - score={evaluation_result.score:.4f}"
                )
        except Exception as e:
            logger.error(f"[{context.trace_id}] Executor: Evaluation failed: {e}")
            evaluation_data = {
                "score": 0.0,
                "summary": f"Evaluation error: {str(e)}",
                "status": "framework_error",
                "metrics": {"error": str(e)},
            }

        # Step 6: Return Solution Pack directory path with evaluation
        logger.info(
            f"[{context.trace_id}] Executor: ✅ Execution completed - "
            f"solution_pack={candidate_dir}, score={evaluation_data['score']:.4f}"
        )

        evaluation_file = os.path.join(evaluator_dir, "best_evaluation.json")
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)

        return Message.from_text(
            data={
                "best_plan_file_path": parent_ctx.stage1_plan_file_path,
                "best_solution_path": work_dir,  # Directory path (absolute)
                "best_evaluation_path": evaluation_file,
                "parent_info_file_path": parent_ctx.parent_info_file_path,
                "total_prompt_tokens": result_data.get("input_tokens", 0),
                "total_completion_tokens": result_data.get("output_tokens", 0),
            },
            sender="executor",
            role=Role.USER,
            mime_type=MimeType.APPLICATION_JSON,
        )

    def _parse_message_inputs(self, message: Message) -> ExecutionContext:
        """Extract and validate input file paths from the message."""
        elems = message.get_elements(ContentElement)
        if not elems:
            raise ValueError("Message missing ContentElement data.")

        data = elems[0].data
        plan_path = data.get("best_plan_file_path")
        parent_info_path = data.get("parent_info_file_path")

        if not parent_info_path or not os.path.exists(parent_info_path):
            raise FileNotFoundError(f"Missing parent_info.json: {parent_info_path}")

        stage1_plan = ""
        if plan_path:
            with open(plan_path, "r", encoding="utf-8") as f:
                stage1_plan = f.read()
        with open(parent_info_path, "r", encoding="utf-8") as f:
            parent_data = json.load(f)

        # Extract parent solution path (absolute path to solution directory)
        parent_solution_path = parent_data.get("solution", "")
        if parent_solution_path and not os.path.isabs(parent_solution_path):
            parent_solution_path = os.path.abspath(parent_solution_path)

        return ExecutionContext(
            parent_info_file_path=parent_info_path,
            parent_core=float(parent_data.get("score", 0.0)),
            parent_solution=json.dumps(parent_data, ensure_ascii=False, indent=2),
            parent_solution_path=parent_solution_path,
            stage1_plan=stage1_plan,
            stage1_plan_file_path=plan_path,
        )

    async def _execute_once(
        self, context: Context, parent_ctx: ExecutionContext, candidate_dir: str
    ) -> Dict[str, Any]:
        """
        Execute Claude Agent once in the candidate directory.

        Args:
            context: Current execution context
            parent_ctx: Parent execution context
            candidate_dir: Absolute path to candidate solution directory (also work_dir)

        Returns:
            Dict containing execution results (evaluation, tokens, etc.)
        """
        # Format loaded skills information
        loaded_skills_info = format_loaded_skills(self.config.skills, candidate_dir)

        # Build prompt
        user_prompt = GENERAL_EXECUTOR_USER.format(
            task_info=context.task,
            improvement_plan=parent_ctx.stage1_plan,
            parent_solution=parent_ctx.parent_solution,
            solution_path=candidate_dir,  # Pass directory path
            loaded_skills=loaded_skills_info,
        )

        # Create agent
        agent = ClaudeCodeAgent(
            model=self.config.llm_config.model,
            api_key=self.config.llm_config.api_key,
            url=self.config.llm_config.url,
            work_dir=candidate_dir,  # Agent works inside candidate directory
            tool_list=self.config.build_in_tools,
            custom_tools={},  # No custom tools for now
            system_prompt=self.config.system_prompt or GENERAL_EXECUTOR_SYSTEM,
            permission_mode=self.config.permission_mode or "acceptEdits",
            setting_sources=["project"],
            max_turns=self.config.max_turns,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )

        logger.info(
            f"[{context.trace_id}] Executor: Executing agent in {candidate_dir}"
        )

        # Execute agent once
        result = await agent.run(user_prompt)

        # Extract metadata
        result_data = {
            "content": (
                result.content[0].data
                if (
                    result.content
                    and len(result.content) > 0
                    and isinstance(result.content[0], ContentElement)
                )
                else ""
            ),
            "input_tokens": result.metadata.get("input_tokens", 0),
            "output_tokens": result.metadata.get("output_tokens", 0),
            "evaluation": {},  # Placeholder for evaluation results
        }

        logger.info(
            f"[{context.trace_id}] Executor: Agent execution completed, "
            f"tokens={result_data['input_tokens'] + result_data['output_tokens']}"
        )

        return result_data
