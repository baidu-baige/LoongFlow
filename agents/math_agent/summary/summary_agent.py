#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides summary_agent implementation
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import ValidationError

from agents.math_agent.prompt.evolve_summary_prompt import (
    EVOLVE_SUMMARY_SYSTEM_PROMPT,
    EVOLVE_SUMMARY_USER_PROMPT,
)
from agents.math_agent.summary.summary_agent_finalizer import (
    SummaryAgentFinalizer,
)
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.memory.evolution import Solution
from loongflow.agentsdk.memory.grade import GradeMemory, MemoryConfig
from loongflow.agentsdk.message import ContentElement, Message, MimeType, Role
from loongflow.agentsdk.models import LiteLLMModel
from loongflow.agentsdk.token import SimpleTokenCounter
from loongflow.agentsdk.tools import (
    Toolkit,
)
from loongflow.framework.pes.compressor import EvolveCompressor
from loongflow.framework.pes.context import Context, LLMConfig, Workspace
from loongflow.framework.pes.database import EvolveDatabase
from loongflow.framework.pes.database.database_tool import (
    GetChildsByParentTool,
    GetParentsByChildIdTool,
    GetSolutionsTool,
)
from loongflow.framework.pes.register import Worker
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.framework.react.components import (
    DefaultObserver,
    DefaultReasoner,
    SequenceActor,
)

logger = get_logger(__name__)


@dataclass
class SummaryAgentConfig:
    """SummaryAgent configuration"""

    llm_config: LLMConfig = None

    system_prompt: str = None
    react_max_steps: int = 40


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


def _init_model(model_config: LLMConfig) -> LiteLLMModel:
    try:
        LLMConfig.model_validate(model_config)
    except ValidationError as e:
        raise ValueError(f"Response validation failed, error: {e}")

    return LiteLLMModel.from_config(model_config.model_dump())


class EvolveSummaryAgent(Worker):
    """Summary Agent class"""

    def __init__(self, config: Any, db: EvolveDatabase):
        super().__init__()
        self.config = SummaryAgentConfig(**config)
        self.db = db
        self.tool_kit = self._build_tool_kit()
        self.model = _init_model(self.config.llm_config)
        self.agent = None
        logger.info(f"Summary: Agent successfully initialized")

    async def run(self, context: Context, message: Message) -> Message:
        """Main method"""
        self.agent, rest_token = await self._create_agent()
        evidence = await self._gather(context, message)
        assessment = await self._assess(context, evidence)
        analysis_str = await self._reflect(context, evidence, assessment, rest_token)
        analysis = json.loads(analysis_str)
        await self._record(context, evidence, analysis.get("reflection", ""))
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

        paths = {
            "best_solution": data.get("best_solution_file_path"),
            "best_evaluation": data.get("best_evaluation_file_path"),
            "parent_info": data.get("parent_info_file_path"),
        }

        for name, path in paths.items():
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"file {name} path {path} not found")

        logger.debug(
            f"Trace ID: {context.trace_id}: Summary: Successfully gathering data from paths"
        )

        try:
            with open(paths["best_solution"], "r", encoding="utf-8") as f:
                solution_content = f.read()
            with open(paths["best_evaluation"], "r", encoding="utf-8") as f:
                evaluation_data = json.load(f)
            with open(paths["parent_info"], "r", encoding="utf-8") as f:
                parent_info_data = json.load(f)
        except FileNotFoundError as e:
            raise e
        except json.JSONDecodeError as e:
            raise ValueError(f"parse json error: {e.msg}")
        except IOError as e:
            raise IOError(f"read file error: {e.filename} - {e.strerror}")
        except Exception as e:
            raise RuntimeError(f"file error: {e}")

        # parent_info is not enough in file, we need to read it from db
        parent_info = Solution(
            **parent_info_data,
        )
        if solution_id := parent_info_data.get("solution_id", None):
            solutions = self.db.get_solutions([solution_id])
            if solutions and len(solutions) == 1:
                parent_info = Solution.from_dict(solutions[0])

        solution_id = uuid.uuid4().hex[:8]
        trace_list = parent_info.metadata.get("trace", []).copy()
        trace_list.append(solution_id)
        metadata = {"trace": trace_list}

        return Evidence(
            best_evaluation=evaluation_data,
            parent_info=parent_info,
            current_solution=Solution(
                solution=solution_content,
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

    async def _assess(self, context: Context, evidence: Evidence) -> Assessment:
        evaluation_score = evidence.current_solution.score
        parent_score = evidence.parent_info.score

        logger.info(
            f"Trace ID: {context.trace_id}: Summary: Get child_score: {evaluation_score}, "
            + f"parent_score: {parent_score}"
        )

        if evaluation_score > parent_score:
            return Assessment.IMPROVEMENT
        elif evaluation_score < parent_score:
            return Assessment.REGRESSION
        else:
            return Assessment.STALE

    async def _reflect(
        self,
        context: Context,
        evidence: Evidence,
        assessment: Assessment,
        rest_token: int,
    ) -> str:
        user_prompt = EVOLVE_SUMMARY_USER_PROMPT.format(
            task_info=context.task,
            parent_solution=json.dumps(
                evidence.parent_info.to_dict(), indent=2, ensure_ascii=False
            ),
            current_solution=json.dumps(
                evidence.current_solution.to_dict(), indent=2, ensure_ascii=False
            ),
            assessment_result=assessment.value,
        )
        logger.info(
            f"Trace ID: {context.trace_id}: Summary: Start generating reflection"
        )

        initial_message = Message.from_text(user_prompt, role=Role.USER)
        token_counter = SimpleTokenCounter()
        user_token_count = await token_counter.count([initial_message])
        if user_token_count > rest_token:
            raise RuntimeError(
                f"Trace ID: {context.trace_id}: Summary: Not enough tokens to complete this request."
                + f"Please check your user prompt tokens and current token usage: {rest_token}/{user_token_count}"
            )

        result = await self.agent.run(initial_message, trace_id=context.trace_id)
        content = result.get_elements(ContentElement)

        reflection = content[0].data if len(content) > 0 else ""

        final_result = {
            "reflection": reflection,
            "total_prompt_tokens": result.metadata.get("total_prompt_tokens", 0),
            "total_completion_tokens": result.metadata.get(
                "total_completion_tokens", 0
            ),
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

            logger.info(
                f"Trace ID: {context.trace_id}: Summary: adjust child weight by parent weight, "
                + f"parent_weight: {parent_weight}, score_diff: {score_diff}, step_size: {step_size}, "
                + f"child_weight: {child_weight}"
            )

            await self.db.update_solution(
                evidence.parent_info.solution_id,
                sample_cnt=evidence.parent_info.sample_cnt + 1,
            )
            logger.info(
                f"Trace ID: {context.trace_id}: Summary: update parent solution successfully."
            )

        evidence.current_solution.summary = reflection
        evidence.current_solution.sample_weight = child_weight

        await self.db.add_solution(evidence.current_solution)

        logger.info(
            f"Trace ID: {context.trace_id}: Summary: Successfully add new solution into database. "
            + f"Solution: {evidence.current_solution.to_dict()}"
        )

        Workspace.write_summarizer_best_summary(context, reflection)
        return

    def _build_tool_kit(self) -> Toolkit:
        function_tool_list = [
            GetSolutionsTool(self.db.get_solutions),
            GetParentsByChildIdTool(self.db.get_parents_by_child_id),
            GetChildsByParentTool(self.db.get_childs_by_parent_id),
        ]

        toolkit = Toolkit()
        for tool in function_tool_list:
            toolkit.register_tool(tool)
        return toolkit

    async def _create_agent(self) -> tuple[ReActAgent, int]:
        system_prompt = self.config.system_prompt or EVOLVE_SUMMARY_SYSTEM_PROMPT
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
            agent_memory, toolkit=self.tool_kit, max_steps=self.config.react_max_steps
        )

        hint_msg = Message.from_text(
            sender="finalizer",
            role=Role.USER,
            data=f"""Summarize and return the final comparative_analysis""",
        )
        finalizer = SummaryAgentFinalizer(
            model=self.model,
            summarize_prompt=self.config.system_prompt or EVOLVE_SUMMARY_SYSTEM_PROMPT,
            hint_message=hint_msg,
        )

        return (
            ReActAgent(
                agent_context,
                DefaultReasoner(self.model, system_prompt),
                SequenceActor(),
                DefaultObserver(),
                finalizer,
                name="Summary",
            ),
            token_threshold,
        )
