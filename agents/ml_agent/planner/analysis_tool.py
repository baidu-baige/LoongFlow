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


class WriteStrategicAnalysisArgs(BaseModel):
    """
    Arguments for writing strategic analysis before making evolution decisions.
    """

    analysis_content: str = Field(
        ...,
        description=(
            "Complete strategic analysis in markdown format."
        ),
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

        return {"status": "success", "message": "Strategic analysis saved."}

    return FunctionTool(
        func=write_strategic_analysis,
        args_schema=WriteStrategicAnalysisArgs,
        name="write_strategic_analysis",
        description=(
            "Write structured strategic analysis before making evolution decisions. "
        ),
    )
