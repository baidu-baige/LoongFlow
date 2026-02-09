# -*- coding: utf-8 -*-
"""
This file define eda tool
"""

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from agents.ml_agent.evocoder import EvoCoder, EvoCoderConfig, TaskConfig
from agents.ml_agent.utils import utils
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import ContentElement, Message, MimeType, Role
from loongflow.agentsdk.tools import FunctionTool
from loongflow.framework.pes.context import Context

logger = get_logger(__name__)


class EDAToolArgs(BaseModel):
    """
    Defines the arguments accepted by the EDATool.

    Attributes:
        instruction: A specific directive for the EDA process,
                     allowing the agent to guide the analysis.
    """

    instruction: Optional[str] = Field(
        None,
        description=(
            "Optional instruction for additional analysis beyond the standard report. "
            "The tool outputs quantitative fields (Files, Target, Columns, Missing, etc.). "
        ),
    )


def build_eda_tool(
    context: Context, evocoder_config_factory: Callable[[], EvoCoderConfig]
) -> FunctionTool:
    """
    Factory function to build a context-aware EDATool.

    Args:
        context: The runtime context of the LoongFlow agent, providing task-specific details.
        evocoder_config_factory: The configuration factory for the underlying EvoCoder instance.

    Returns:
        A fully configured FunctionTool ready to be used by a ReAct agent.
    """

    async def eda_run(instruction: str) -> dict[str, Any]:
        """
        The core execution logic for the EDA tool.

        This asynchronous function encapsulates the entire process of:
        1. Configuring a task for the EvoCoder based on the runtime context.
        2. Invoking the EvoCoder to generate, execute, and validate EDA code.
        3. Returning the final, structured output from the EvoCoder.

        Args:
          instruction: An optional, specific directive from the agent to guide
                       the EDA generation (e.g., focus on certain data aspects).

        Returns:
          A dictionary containing the results from the EvoCoder execution. This
          payload is expected to contain the EDA report.
          Raise error if the process fails or yields no content.
        """

        logger.info(f"Generating EDA report, instruction is: {instruction}")
        evocoder_config = evocoder_config_factory()

        evocoder = EvoCoder(config=evocoder_config)

        task_config = TaskConfig(
            task_description=context.task,
            task_data_path=context.metadata.get("task_data_path"),
            plan=instruction,
            gpu_available=context.metadata.get("gpu_available"),
            gpu_count=context.metadata.get("gpu_count"),
            hardware_info=context.metadata.get("hardware_info"),
            task_dir_structure=context.metadata.get("task_dir_structure"),
            code_deps={"eda": utils.get_latest_eda_code(context)},
        )
        message = Message.from_media(
            sender="EDATool",
            mime_type=MimeType.APPLICATION_JSON,
            role=Role.USER,
            data=task_config.to_dict(),
        )
        try:
            result_message = await evocoder(message=message)
        finally:
            evocoder_config.evaluator.interrupt()

        content = result_message.get_elements(ContentElement)
        if not content or len(content) == 0:
            raise RuntimeError("no eda code and report generated")
        if not isinstance(content[0].data, dict):
            raise RuntimeError(f"eda info generated failed {content[0].data}")

        # eda will only output data, we need to save it
        code = content[0].data.get("best_code")
        eda_info = content[0].metadata.get("artifacts", {}).get("eda_info")

        if not code or not eda_info:
            raise RuntimeError("eda info generated failed, no valid info generated")

        # warning: concurrent is unsafe
        utils.write_latest_eda_code(context, code)
        utils.write_latest_eda_info(context, eda_info)
        return {
            "eda_info": eda_info,
        }

    return FunctionTool(
        func=eda_run,
        args_schema=EDAToolArgs,
        name="eda_tool",
        description=(
            "Execute exploratory data analysis (EDA) and return a quantitative report. "
            "Output contains numerical facts: file stats, missing rates, correlations, etc. "
            "The tool generates and runs analysis code, returning a structured report. "
            "Use this when you need data insights to inform your planning decisions."
        ),
    )
