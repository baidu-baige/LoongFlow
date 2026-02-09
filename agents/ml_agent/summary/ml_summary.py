# -*- coding: utf-8 -*-
"""
This file provides summary_agent implementation
"""

import json
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from jinja2 import Environment

from agents.ml_agent.prompt.ml_evolve import (
    ML_SUMMARY_SYSTEM_PROMPT,
    ML_SUMMARY_USER_PROMPT,
)
from agents.ml_agent.summary.analysis_tool import build_summary_analysis_tool
from agents.ml_agent.summary.ml_summary_finalizer import MLSummaryFinalizer, Reflection
from agents.ml_agent.utils import solutions
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.memory.evolution import Solution
from loongflow.agentsdk.memory.grade import GradeMemory, MemoryConfig
from loongflow.agentsdk.message import ContentElement, Message, MimeType, Role
from loongflow.agentsdk.models import LiteLLMModel
from loongflow.agentsdk.token import SimpleTokenCounter
from loongflow.agentsdk.tools import Toolkit
from loongflow.framework.pes import Worker
from loongflow.framework.pes.compressor import EvolveCompressor
from loongflow.framework.pes.context import Context, LLMConfig, Workspace
from loongflow.framework.pes.database import (
    EvolveDatabase,
    GetBestSolutionsTool,
    GetChildsByParentTool,
    GetParentsByChildIdTool,
    GetSolutionsTool,
)
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.framework.react.components import (
    DefaultObserver,
    DefaultReasoner,
    SequenceActor,
)

logger = get_logger(__name__)


@dataclass
class MLSummaryAgentConfig:
    """
    Configuration for the ML Summary Agent
    """

    llm_config: LLMConfig = None
    react_max_steps: int = 40
    system_prompt: str = ML_SUMMARY_SYSTEM_PROMPT
    user_prompt: str = ML_SUMMARY_USER_PROMPT


@dataclass
class Evidence:
    """MLSummaryAgent evidence"""

    eda_analysis: str = None
    best_plan: str = None
    best_evaluation: dict[str, Any] = None
    parent_info: Solution = None
    current_solution: Solution = None


class MLSummaryAgent(Worker):
    """Summary Agent class"""

    def __init__(self, config: Any, db: EvolveDatabase):
        """Initializes the Summary Agent."""
        super().__init__()
        self.config = (
            config
            if isinstance(config, MLSummaryAgentConfig)
            else MLSummaryAgentConfig(**config)
        )
        self.db = db

        self.model = self._init_model()

    def _init_model(self) -> LiteLLMModel:
        """Initialize or reuse the LLM model."""
        llm = self.config.llm_config
        if not llm or not all([llm.model, llm.url, llm.api_key]):
            raise ValueError("model_name, url, and api_key are required in llm_config.")

        return LiteLLMModel.from_config(llm.model_dump())

    async def _create_agent(self, model: LiteLLMModel) -> ReActAgent:
        function_tool_list = [
            GetSolutionsTool(solutions.simplify_solution(self.db.get_solutions)),
            GetParentsByChildIdTool(solutions.simplify_solution(self.db.get_parents_by_child_id)),
            GetChildsByParentTool(solutions.simplify_solution(self.db.get_childs_by_parent_id)),
        ]
        toolkit = Toolkit()
        for tool in function_tool_list:
            toolkit.register_tool(tool)

        system_prompt = self.config.system_prompt or ML_SUMMARY_SYSTEM_PROMPT
        system_message = [Message.from_text(system_prompt, role=Role.SYSTEM)]
        token_counter = SimpleTokenCounter()
        system_token_count = await token_counter.count(system_message)
        token_threshold = self.config.llm_config.context_length - system_token_count

        memory_config = MemoryConfig(
            auto_compress=True,
            token_threshold=token_threshold,
        )
        agent_memory = GradeMemory.create_default(
            model=self.model,
            compressor=EvolveCompressor(
                model=self.model,
                token_counter=token_counter,
                token_threshold=token_threshold,
            ),
            config=memory_config,
        )

        agent_context = AgentContext(
            agent_memory, toolkit=toolkit, max_steps=self.config.react_max_steps
        )

        reasoner = DefaultReasoner(model, system_prompt)
        actor = SequenceActor()
        observer = DefaultObserver()

        finalizer = MLSummaryFinalizer(
            model=model,
            summarize_prompt=self.config.system_prompt or ML_SUMMARY_SYSTEM_PROMPT,
            output_schema=Reflection,
        )

        return ReActAgent(
            agent_context,
            reasoner,
            actor,
            observer,
            finalizer,
            name="MLSummary",
        )

    async def run(self, context: Context, message: Message) -> Message:
        """
        execute task
        """
        logger.info(
            f"Trace ID: {context.trace_id}: MLSummaryAgent started for iteration {context.current_iteration}."
        )

        evidence = await self._gather(context, message)
        analysis = await self._reflect(context, evidence)
        await self._record(context, evidence, analysis)
        return Message.from_media(
            sender="MLSummaryAgent",
            role=Role.USER,
            mime_type=MimeType.APPLICATION_JSON,
            data={
                "best_summary_file_path": Workspace.get_summarizer_best_summary_path(
                    context
                ),
            },
        )

    async def _gather(self, context: Context, message: Message) -> Evidence:
        content = message.get_elements(ContentElement)
        if (
                not content
                or len(content) == 0
                or not content[0].data
                or not isinstance(content[0].data, dict)
        ):
            raise ValueError(f"Missing content element data.")
        data = content[0].data

        # sometimes we do not have best_plan, then we could ignore it
        try:
            with open(data.get("best_plan_file_path"), "r", encoding="utf-8") as f:
                plan_content = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read best plan file, error: {e}")
            plan_content = {}

        try:
            with open(data.get("eda_info_file_path"), "r", encoding="utf-8") as f:
                eda_analysis = f.read()
        except Exception as e:
            logger.error(f"Failed to read EDA info file, error: {e}")
            eda_analysis = ""

        with open(data.get("best_evaluation_file_path"), "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)

        with open(data.get("parent_info_file_path"), "r", encoding="utf-8") as f:
            parent_info_data = json.load(f)

        solution_content = self._construct_solutions(
            data.get("best_solution_file_path")
        )

        # parent_info is not enough in file, we need to read it from db
        parent_info = Solution.from_dict(parent_info_data)
        if solution_id := parent_info_data.get("solution_id", None):
            solutions = self.db.get_solutions([solution_id])
            if solutions and len(solutions) == 1:
                parent_info = Solution.from_dict(solutions[0])

        solution_id = uuid.uuid4().hex[:8]
        trace_list = parent_info.metadata.get("trace", []).copy()
        trace_list.append(solution_id)
        metadata = {"trace": trace_list}

        return Evidence(
            eda_analysis=eda_analysis,
            best_evaluation=evaluation_data,
            parent_info=parent_info,
            current_solution=Solution(
                solution=json.dumps(solution_content, ensure_ascii=False, indent=2),
                solution_id=solution_id,
                generate_plan=plan_content,
                parent_id=parent_info.solution_id,
                island_id=context.island_id,
                iteration=context.current_iteration,
                timestamp=time.time(),
                generation=len(trace_list),
                score=evaluation_data.get("score", 0),
                evaluation=json.dumps(evaluation_data, ensure_ascii=False, indent=2),
                metadata=metadata,
            ),
        )

    async def _reflect(self, context: Context, evidence: Evidence) -> Reflection:

        parent_solution = ""
        if evidence.parent_info.solution_id:
            parent_solution = json.dumps(
                evidence.parent_info.to_dict(), indent=2, ensure_ascii=False
            )

        user_prompt = self.config.user_prompt or ML_SUMMARY_USER_PROMPT
        prompt = (
            Environment()
            .from_string(user_prompt)
            .render(
                {
                    "task_info": context.task,
                    "eda_analysis": evidence.eda_analysis,
                    "parent_solution": parent_solution,
                    "current_solution": json.dumps(
                        evidence.current_solution.to_dict(),
                        indent=2,
                        ensure_ascii=False,
                    ),
                }
            )
        )
        agent = await self._create_agent(self.model)
        try:
            agent.context.toolkit.register_tool(
                GetBestSolutionsTool(solutions.simplify_solution(
                    partial(self.db.get_best_solutions, island_id=context.island_id))
                )
            )
            agent.context.toolkit.register_tool(build_summary_analysis_tool(context))
            result = await agent.run(
                Message.from_text(prompt), trace_id=context.trace_id
            )
            content = result.get_elements(ContentElement)
            if not content or len(content) == 0 or not content[0].data:
                raise ValueError("No summary result generated")
            reflection = Reflection.model_validate(content[0].data)
        except Exception as e:
            logger.error(
                f"Trace ID: {context.trace_id}: MLSummary: reflect result validate error: {e}"
            )
            reflection = Reflection()
        logger.info(
            f"Trace ID: {context.trace_id}: Summary: reflection result: {reflection.model_dump()}"
        )
        return reflection

    async def _record(
        self, context: Context, evidence: Evidence, reflection: Reflection
    ) -> None:
        # set initial weight
        child_weight = 0.05

        # If Parent exists in the database, set child's weight based on parent's weight
        if evidence.parent_info.solution_id:
            parent_solution = evidence.parent_info
            child_solution = evidence.current_solution
            parent_weight = parent_solution.sample_weight
            # Calculate the score difference; the larger the difference, the greater the weight adjustment
            score_diff = child_solution.score - parent_solution.score
            # The amplitude of weight iteration is determined by the iteration progress; larger in early iterations,
            # smaller later on
            step_size = 1 - (context.current_iteration / context.total_iterations)
            # This formula automatically ensures that if score_diff is negative,
            # child_weight is lower than parent_weight, otherwise higher. ALPHA = 2 is the amplification factor
            child_weight = (
                    parent_weight + (3 * score_diff * step_size) + 3 * child_solution.score
            )
            if child_weight < 0:
                # Prevent weight from being too small; minimum value is 0.05
                child_weight = 0.05
            logger.info(
                f"Trace ID: {context.trace_id}: Summary: adjust child weight by parent weight, "
                + f"parent_weight: {parent_weight}, score_diff: {score_diff}, step_size: {step_size}, "
                + f"child_weight: {child_weight}"
            )

        if evidence.parent_info.solution_id:
            await self.db.update_solution(
                evidence.parent_info.solution_id,
                sample_cnt=evidence.parent_info.sample_cnt + 1,
            )
            logger.info(
                f"Trace ID: {context.trace_id}: Summary: update parent solution successfully."
            )

        evidence.current_solution.summary = reflection.model_dump_json()
        evidence.current_solution.sample_weight = child_weight

        await self.db.add_solution(evidence.current_solution)

        logger.info(
            f"Trace ID: {context.trace_id}: Summary: Successfully add new solution into database. "
            + f"Solution: {evidence.current_solution.solution_id}"
        )

        Workspace.write_summarizer_best_summary(
            context, reflection.model_dump_json(indent=2, ensure_ascii=False)
        )
        return

    def _construct_solutions(self, solution_dir_path: str) -> dict[str, str]:
        """
        Reads all .py files from a specified directory and returns their content.
        Args:
            solution_dir_path (str): The path to the directory to scan.
        Returns:
            dict: A dictionary where keys are filenames (without the .py extension)
                  and values are the contents of the files. Returns an empty
                  dictionary if the path is invalid or not a directory.
        """
        solution_path = Path(solution_dir_path)
        if not solution_path.is_dir():
            raise FileNotFoundError(f"Solution directory not found: {solution_path}")

        solution_dict = {}
        for py_file in solution_path.glob("*.py"):
            stage_name = py_file.stem  # e.g., 'load_data' from 'load_data.py'
            solution_dict[stage_name] = py_file.read_text(encoding="utf-8")

        if not solution_dict:
            raise FileNotFoundError(
                f"Solution directory not found: {solution_dir_path}"
            )

        return solution_dict
