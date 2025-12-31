# -*- coding: utf-8 -*-
"""
This file define
"""
from typing import Any

from pydantic import BaseModel, Field

from agents.ml_evolve.utils import utils
from agentsdk.logger import get_logger
from agentsdk.tools import FunctionTool
from evolux.evolve.context import Context

logger = get_logger(__name__)


class WriteStrategicAnalysisArgs(BaseModel):
    """
    Arguments for writing strategic analysis before making evolution decisions.
    """
    analysis_content: str = Field(
        ...,
        description=(
            "Complete strategic analysis in markdown format. Must include:\n"
            "1. Plan Effectiveness Analysis - which plans worked/failed and why\n"
            "2. Implementation Gap Analysis - differences vs high-score solutions\n"
            "3. Bottleneck Diagnosis - current weak points and root causes\n"
            "4. Evolution Space - untried directions and recommendations\n"
            "5. Decision - primary evolution direction and fusion decision"
        )
    )


def build_strategic_analysis_tool(context: Context) -> FunctionTool:
    """
    Build a tool for Planner to output structured strategic analysis.

    Args:
        context: The runtime context providing workspace paths.

    Returns:
        A configured FunctionTool for the ReAct agent.
    """

    async def write_strategic_analysis(analysis_content: str) -> dict[str, Any]:
        """
        Write strategic analysis to file before making evolution decisions.

        Args:
            analysis_content: Complete strategic analysis in markdown format.

        Returns:
            Confirmation of successful write.
        """
        logger.info("Executing write_strategic_analysis")

        if not analysis_content or not analysis_content.strip():
            raise ValueError("Analysis content cannot be empty.")

        try:
            utils.write_strategic_analysis_info(context, analysis_content)
        except Exception as e:
            logger.error(f"Failed to save strategic analysis: {e}")
            raise IOError(f"Failed to save strategic analysis file: {e}")

        return {
            "status": "success",
            "message": "Strategic analysis saved."
        }

    return FunctionTool(
        func=write_strategic_analysis,
        args_schema=WriteStrategicAnalysisArgs,
        name="write_strategic_analysis",
        description=(
            "Write structured strategic analysis before making evolution decisions. "
        )
    )
