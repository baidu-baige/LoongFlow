# -*- coding: utf-8 -*-
"""
This file provides implementation of the MLPlannerAgent.
"""

import json
from dataclasses import dataclass
from functools import partial
from typing import Any

from jinja2 import Environment

from agents.ml_agent.evocoder import (
    EDAContextProvider,
    EDAEvaluator,
    EvoCoderConfig,
    EvoCoderEvaluatorConfig,
)
from agents.ml_agent.evocoder.stage_context_provider import Stage
from agents.ml_agent.planner.analysis_tool import build_strategic_analysis_tool
from agents.ml_agent.planner.eda_tool import build_eda_tool
from agents.ml_agent.planner.ml_planner_finalizer import MLPlan, MLPlannerFinalizer
from agents.ml_agent.planner.solution_tool import build_ensemble_tool
from agents.ml_agent.prompt.ml_evolve import (
    ML_PLANNER_SYSTEM_PROMPT,
    ML_PLANNER_USER_PROMPT,
)
from agents.ml_agent.utils import solutions, utils
from loongflow.agentsdk.logger import get_logger
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
    GetMemoryStatusTool,
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
class MLPlannerAgentConfig:
    """Configuration for the ML Planner Agent, using the finalized prompts."""

    llm_config: LLMConfig
    react_max_steps: int = 10
    system_prompt: str = ML_PLANNER_SYSTEM_PROMPT
    user_prompt: str = ML_PLANNER_USER_PROMPT
    evo_coder_timeout: int = 1800


# --- Main Agent Class ---
class MLPlannerAgent(Worker):
    """
    The MLPlannerAgent is responsible for creating strategic plans for ML tasks.
    It operates within an evolutionary framework by:
    1. Sampling a parent solution from a database.
    2. Using a ReAct model to reason based on the parent, feedback, and EDA reports.
    3. Autonomously deciding whether to re-run EDA.
    4. Generating an improved child solution as a structured plan.
    """

    def __init__(self, config: Any, db: EvolveDatabase):
        """Initializes the Planner Agent."""
        super().__init__()
        self.config = (
            config
            if isinstance(config, MLPlannerAgentConfig)
            else MLPlannerAgentConfig(**config)
        )
        self.database = db

        self.model = self._init_model()

    async def run(self, context: Context, message: Message) -> Message:
        """
        The main entry point for the agent, orchestrating the planning process.
        """

        agent = await self._create_agent(self.model)
        # register eda_tool
        agent.context.toolkit.register_tool(
            build_eda_tool(
                context,
                lambda: EvoCoderConfig(
                    llm_config=self.config.llm_config,
                    context_provider=EDAContextProvider(),
                    evaluator=EDAEvaluator(
                        EvoCoderEvaluatorConfig(
                            workspace_path=str(
                                utils.get_evocoder_evaluate_path(
                                    context, Stage.EDA.value
                                )
                            ),
                            timeout=self.config.evo_coder_timeout,
                        )
                    ),
                ),
            )
        )
        # register select_solution_for_ensemble tool
        agent.context.toolkit.register_tool(build_ensemble_tool(context))
        agent.context.toolkit.register_tool(build_strategic_analysis_tool(context))
        # register get_best_solution with specific tool, to avoid search from other island
        agent.context.toolkit.register_tool(
            GetBestSolutionsTool(
                solutions.simplify_solution(
                    partial(self.database.get_best_solutions, island_id=context.island_id))
            )
        )

        workspace = Workspace.get_planner_path(context)
        logger.debug(
            f"Trace ID: {context.trace_id}: MLPlanner: Workspace is : {workspace}"
        )

        memory_status = self.database.memory_status()
        logger.info(
            f"Trace ID: {context.trace_id}: MLPlanner: Current Iteration {context.current_iteration} "
            + f"Memory status: {memory_status}"
        )

        parent = self.database.sample_solution(context.island_id)
        logger.info(
            f"Trace ID: {context.trace_id}: MLPlanner: Get sample parent solution: {parent}"
        )

        parent_dict = parent if parent else {}
        parent_json = json.dumps(parent_dict, ensure_ascii=False, indent=2)
        Workspace.write_planner_parent_info(context, parent_json)
        parent_info_file_path = Workspace.get_planner_parent_info_path(context)
        logger.debug(
            f"Trace ID: {context.trace_id}: MLPlanner: Write planner parent info to {parent_info_file_path}"
        )

        # we need to find eda, but it should not change frequently
        eda_info = utils.get_latest_eda_info(context)
        render_context = {
            "task_description": context.task,
            "parent_solution": parent_json if parent_dict else "",
            "previous_eda_report": eda_info,
            "task_data_path": context.metadata.get("task_data_path"),
            "hardware_info": context.metadata.get("hardware_info"),
            "task_dir_structure": context.metadata.get("task_dir_structure"),
        }

        template = Environment().from_string(self.config.user_prompt)

        user_prompt = template.render(render_context)

        initial_message = Message.from_text(user_prompt, role=Role.USER)

        # actually, resp is message, we could construct it to a more proper way, such as data location
        result = await agent.run(
            initial_message,
        )

        content = result.get_elements(ContentElement)
        if not content or len(content) == 0 or not content[0].data:
            raise ValueError("No plan generated")

        try:
            ml_plan = MLPlan.model_validate(content[0].data)
        except Exception as e:
            logger.error(
                f"Trace ID: {context.trace_id}: MLPlanner: ml plan result validate error: {e}, "
                f"raw response: {content[0].data}"
            )
            ml_plan = MLPlan()
        logger.info(
            f"Trace ID: {context.trace_id}: MLPlanner: ml plan result: {ml_plan.model_dump()}"
        )

        # save plan to data
        Workspace.write_planner_best_plan(context, ml_plan.model_dump_json(indent=2))
        best_plan_file_path = Workspace.get_planner_best_plan_path(context)
        logger.info(
            f"Trace ID: {context.trace_id}: Planner: Write best plan to {best_plan_file_path}"
        )

        # actually we also need to save eda info,simply copy latest is enough
        utils.write_current_eda_info(context, utils.get_latest_eda_info(context))
        utils.write_current_eda_code(context, utils.get_latest_eda_code(context))

        results = {
            "parent_info_file_path": parent_info_file_path,
            "best_plan_file_path": best_plan_file_path,
            "eda_info_file_path": str(utils.get_current_eda_info_path(context)),
            "eda_code_file_path": str(utils.get_current_eda_code_path(context)),
            "model_assemble_file_path": str(utils.get_assemble_model_path(context)),
        }

        logger.info(
            f"Trace ID: {context.trace_id}: MLPlanner: Successfully create best_plan in "
            f"iteration {context.current_iteration}"
        )

        return Message.from_media(
            sender="MLPlanner",
            role=Role.USER,
            data=results,
            mime_type=MimeType.APPLICATION_JSON,
        )

    def _init_model(self) -> LiteLLMModel:
        """Initialize or reuse the LLM model."""
        llm = self.config.llm_config
        if not llm or not all([llm.model, llm.url, llm.api_key]):
            raise ValueError("model_name, url, and api_key are required in llm_config.")

        return LiteLLMModel.from_config(llm.model_dump())

    async def _create_agent(self, model: LiteLLMModel) -> ReActAgent:
        function_tool_list = [
            GetMemoryStatusTool(solutions.simplify_solution(self.database.memory_status)),
            GetSolutionsTool(solutions.simplify_solution(self.database.get_solutions)),
            GetParentsByChildIdTool(solutions.simplify_solution(self.database.get_parents_by_child_id)),
            GetChildsByParentTool(solutions.simplify_solution(self.database.get_childs_by_parent_id)),
        ]
        toolkit = Toolkit()
        for tool in function_tool_list:
            toolkit.register_tool(tool)

        system_prompt = self.config.system_prompt or ML_PLANNER_SYSTEM_PROMPT

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

        finalizer = MLPlannerFinalizer(
            model=model,
            summarize_prompt=self.config.system_prompt or ML_PLANNER_SYSTEM_PROMPT,
            output_schema=MLPlan,
        )

        return ReActAgent(
            agent_context,
            reasoner,
            actor,
            observer,
            finalizer,
            name="MLPlanner",
        )
