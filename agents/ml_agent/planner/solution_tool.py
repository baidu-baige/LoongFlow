# -*- coding: utf-8 -*-
"""
This file defines the tool for selecting solutions for an ensemble plan.
"""

import json
from typing import Any, List

from pydantic import BaseModel, Field

from agents.ml_agent.utils import utils
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.tools import FunctionTool
from loongflow.framework.pes.context import Context

logger = get_logger(__name__)


class SelectSolutionForFusionArgs(BaseModel):
    """
    Arguments for selecting external solutions to fuse into the current solution.
    """
    solution_ids: List[str] = Field(
        ...,
        description="List of solution IDs from historical best solutions to be fused into current solution. "
                    "These should be high-performing solutions with DIFFERENT model architectures than the current plan."
    )
    instruction: str = Field(
        ...,
        description="Fusion strategy describing how to integrate the selected external solutions. "
                    "Example: 'Fuse LightGBM solution (id: xxx) with current XGBoost plan using weighted average.'"
    )


def build_ensemble_tool(context: Context) -> FunctionTool:
    """
    Build a context-aware select_solution_for_fusion tool.
    This tool enables cross-solution fusion by selecting high-performing
    historical solutions to be injected into the current solution's
    train_and_predict stage.
    Args:
        context: The runtime context providing workspace paths.
    Returns:
        A configured FunctionTool for the ReAct agent.
    """

    async def select_solution_for_fusion(solution_ids: List[str], instruction: str) -> dict[str, Any]:
        """
        Select external solutions for fusion with current solution.
        Args:
            solution_ids: IDs of historical solutions to fuse.
            instruction: Strategy for how to fuse these solutions.
        Returns:
            Confirmation of successful selection.
        """
        logger.info(f"Executing select_solution_for_fusion with IDs: {solution_ids}")

        if not isinstance(solution_ids, list):
            raise TypeError(f"Input must be a list of solution IDs, but got {type(solution_ids)}.")

        if not solution_ids:
            return {
                "status": "success",
                "message": f"Successfully selected {len(solution_ids)} solutions for the fusion plan. "
                           f"The selection has been recorded."
            }

        assemble_info = {
            "solution_ids": solution_ids,
            "assemble_plan": instruction,
        }

        try:
            utils.write_assemble_model_info(context, json.dumps(assemble_info, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Failed to save fusion selection: {e}")
            raise IOError(f"Failed to save fusion selection file: {e}")

        return {
            "status": "success",
            "message": f"Successfully selected {len(solution_ids)} external solution(s) for fusion. "
                       f"These will be injected into train_and_predict stage."
        }

    return FunctionTool(
        func=select_solution_for_fusion,
        args_schema=SelectSolutionForFusionArgs,
        name="select_solution_for_fusion",
        description=(
            "Select high-performing historical solutions to FUSE into the current solution. "
            "Use this when Get_Best_Solutions reveals diverse, high-scoring solutions with "
            "different model architectures. Selected solutions will be automatically injected "
            "into the train_and_predict stage for cross-solution fusion."
        )
    )
