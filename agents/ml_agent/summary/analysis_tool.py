# -*- coding: utf-8 -*-
"""
This file define
"""

from typing import Any

from pydantic import BaseModel, Field

from agents.ml_agent.utils import utils
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.tools import FunctionTool
from loongflow.framework.pes.context import Context

logger = get_logger(__name__)


class WriteSummaryAnalysisArgs(BaseModel):
    """
    Arguments for writing summary analysis before generating final reflection output.
    """
    analysis_content: str = Field(
        ...,
        description=(
            "Complete summary analysis in markdown format. Must include:\n"
            "1. Technical Implementation Review - extract technical fingerprint for each stage\n"
            "2. Performance Attribution - connect score changes to specific stage modifications\n"
            "3. Evidence Collection - gather what worked/failed with supporting evidence\n"
            "4. Strategic Assessment - prioritize stages and recommend next direction"
        )
    )


def build_summary_analysis_tool(context: Context) -> FunctionTool:
    """
    Build a tool for Summary agent to output structured analysis before final reflection.

    Args:
        context: The runtime context providing workspace paths.

    Returns:
        A configured FunctionTool for the ReAct agent.
    """

    async def write_summary_analysis(analysis_content: str) -> dict[str, Any]:
        """
        Write summary analysis to file before generating final reflection output.

        This analysis serves as the intermediate thinking step between
        information gathering and structured output generation.

        Args:
            analysis_content: Complete summary analysis in markdown format.

        Returns:
            Confirmation of successful write.
        """
        logger.info("Executing write_summary_analysis")

        if not analysis_content or not analysis_content.strip():
            raise ValueError("Analysis content cannot be empty.")

        try:
            utils.write_summary_analysis_info(context, analysis_content)
        except Exception as e:
            logger.error(f"Failed to save summary analysis: {e}")
            raise IOError(f"Failed to save summary analysis file: {e}")

        return {
            "status": "success",
            "message": "Summary analysis saved."
        }

    return FunctionTool(
        func=write_summary_analysis,
        args_schema=WriteSummaryAnalysisArgs,
        name="write_summary_analysis",
        description=(
            "Write structured summary analysis before generating final reflection. "
        )
    )
