#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides plan_agent implementation
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from agents.math_agent.planner.build_tool import build_planner_write_tool
from agents.math_agent.planner.plan_agent_finalizer import PlanAgentFinalizer
from agents.math_agent.prompt.evolve_plan_prompt import (
    EVOLVE_PLANNER_SYSTEM_PROMPT,
    EVOLVE_PLANNER_USER_PROMPT,
)
from loongflow.agentsdk.logger import get_logger
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
    GetBestSolutionsTool,
    GetChildsByParentTool,
    GetMemoryStatusTool,
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
class PlannerAgentConfig:
    """Planner configuration"""

    system_prompt: Optional[str] = None
    llm_config: LLMConfig = None
    react_max_steps: int = 10


def _init_model(model_config: LLMConfig) -> LiteLLMModel:
    try:
        LLMConfig.model_validate(model_config)
    except ValidationError as e:
        raise ValueError(f"Response validation failed, error: {e}")

    return LiteLLMModel.from_config(model_config.model_dump())


class EvolvePlanAgent(Worker):
    """Plan Agent Class"""

    def __init__(self, config: Any, db: EvolveDatabase):
        super().__init__()
        self.config = (
            config
            if isinstance(config, PlannerAgentConfig)
            else PlannerAgentConfig(**config)
        )
        self.database = db
        self.tool_kit = self._build_tool_kit()
        self.model = _init_model(self.config.llm_config)

        logger.info(f"Planner: Agent successfully initialized")

    async def run(self, context: Context, message: Message) -> Message:
        agent, rest_token = await self._create_agent()
        agent.context.toolkit.register_tool(build_planner_write_tool(context))

        """Main method"""
        task = context.task
        island_id = context.island_id
        init_solution = context.init_solution
        init_evaluation = context.init_evaluation
        init_score = 0.0
        if context.init_score is not None:
            init_score = context.init_score
        init_parent = {
            "solution": init_solution,
            "score": init_score,
            "evaluation": init_evaluation,
            "summary": "This is the initial solution, it has no parents, you should start evolution from here",
        }

        workspace = Workspace.get_planner_path(context, True)
        logger.debug(
            f"Trace ID: {context.trace_id}: Planner: Workspace is : {workspace}"
        )

        memory_status = self.database.memory_status()
        logger.info(
            f"Trace ID: {context.trace_id}: Planner: Current Iteration {context.current_iteration} "
            + f"Memory status: {memory_status}"
        )

        parent = self.database.sample_solution(island_id)
        logger.info(
            f"Trace ID: {context.trace_id}: Planner: Get sample parent solution: {parent}"
        )

        # Save parent into parent_info.json
        parent_dict = parent if parent else init_parent
        parent_json = json.dumps(parent_dict, ensure_ascii=False, indent=4)
        Workspace.write_planner_parent_info(context, parent_json)
        parent_info_file_path = Workspace.get_planner_parent_info_path(context)
        logger.debug(
            f"Trace ID: {context.trace_id}: Planner: Write planner parent info to {parent_info_file_path}"
        )

        user_prompt = EVOLVE_PLANNER_USER_PROMPT.format(
            task_info=task,
            parent_solution=parent_json,
            workspace=workspace,
            island_num=self.database.config.num_islands,
            parent_island=parent.get("island_id") if parent else 0,
        )

        initial_message = Message.from_text(user_prompt, role=Role.USER)
        logger.info(
            f"Trace ID: {context.trace_id}: Planner: "
            + f"Start create best plan in iteration {context.current_iteration}"
        )

        token_counter = SimpleTokenCounter()
        user_token_count = await token_counter.count([initial_message])
        if user_token_count > rest_token:
            raise RuntimeError(
                f"Trace ID: {context.trace_id}: Planner: Not enough tokens to complete this request."
                + f"Please check your user prompt tokens and current token usage: {rest_token}/{user_token_count}"
            )

        resp = await agent.run(
            initial_message,
            task=task,
            parent_solution=parent_json,
            workspace=workspace,
            trace_id=context.trace_id,
        )

        best_plan = resp.get_elements(ContentElement)
        generate_plan = best_plan[0].data if len(best_plan) > 0 else ""

        Workspace.write_planner_best_plan(context, generate_plan)
        best_plan_file_path = Workspace.get_planner_best_plan_path(context)
        logger.debug(
            f"Trace ID: {context.trace_id}: Planner: Write best plan to {best_plan_file_path}"
        )

        results = {
            "parent_info_file_path": parent_info_file_path,
            "best_plan_file_path": best_plan_file_path,
            "total_prompt_tokens": resp.metadata.get("total_prompt_tokens", 0),
            "total_completion_tokens": resp.metadata.get("total_completion_tokens", 0),
        }

        logger.info(
            f"Trace ID: {context.trace_id}: Planner: Successfully create best_plan in "
            + f"iteration {context.current_iteration}"
        )

        return Message.from_text(data=results, mime_type=MimeType.APPLICATION_JSON)

    def _build_tool_kit(self) -> Toolkit:
        function_tool_list = [
            GetMemoryStatusTool(self.database.memory_status),
            GetSolutionsTool(self.database.get_solutions),
            GetBestSolutionsTool(self.database.get_best_solutions),
            GetParentsByChildIdTool(self.database.get_parents_by_child_id),
            GetChildsByParentTool(self.database.get_childs_by_parent_id),
        ]

        tool_kit = Toolkit()
        for tool in function_tool_list:
            tool_kit.register_tool(tool)

        return tool_kit

    async def _create_agent(self) -> tuple[ReActAgent, int]:
        system_prompt = self.config.system_prompt or EVOLVE_PLANNER_SYSTEM_PROMPT
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
            agent_memory, self.tool_kit, self.config.react_max_steps
        )
        finalizer = PlanAgentFinalizer(self.model, "")

        return (
            ReActAgent(
                agent_context,
                DefaultReasoner(self.model, system_prompt),
                SequenceActor(),
                DefaultObserver(),
                finalizer,
                name="Planner",
            ),
            token_threshold,
        )
