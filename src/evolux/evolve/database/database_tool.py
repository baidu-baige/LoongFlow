#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Function tool for LoongFlow.
"""
from typing import Optional, Callable, Any

from pydantic import BaseModel, Field

from agentsdk.tools import FunctionTool


class GetSolutionsArgs(BaseModel):
    """Arguments for getting solutions."""

    solution_ids: list[str] = Field(..., description="List of solution ids.")


class GetSolutionsTool(FunctionTool):
    """Tool for getting solutions by their ids."""

    def __init__(self, func: Optional[Callable[..., Any]] = None):
        super().__init__(
            func=func,
            args_schema=GetSolutionsArgs,
            name="Get_Solutions",
            description="Get solutions by their ids. "
            + "This tool accepts a single argument 'solution_ids' which should contain a list of solution ids."
            + "Returns a list of solutions with the corresponding information. "
            + "The solution id should be nonempty string",
        )


class GetBestSolutionsArgs(BaseModel):
    """Arguments for getting the best solutions."""

    island_id: Optional[int] = Field(None, description="The island id to filter by.")
    top_k: Optional[int] = Field(
        None, description="The count of the top k solutions to retrieve."
    )


class GetBestSolutionsTool(FunctionTool):
    """Tool for getting the best solutions."""

    def __init__(self, func: Optional[Callable[..., Any]] = None):
        super().__init__(
            func=func,
            args_schema=GetBestSolutionsArgs,
            name="Get_Best_Solutions",
            description="Get the best solutions. "
            + "This tool is used when you need to find the best solutions across specific islands or globally."
            + "This tool accepts two arguments 'island_id' and 'top_k'. "
            + "If both are provided, the top k solutions for the specified island will be returned. "
            + "If only island_id is provided, only the top k solutions for that island will be returned. "
            + "If only top_k is provided, the global top k solutions across all islands will be returned."
            + "If neither are provided, the global best solution across all islands will be returned."
            + "Returns a list of solutions with the corresponding information.",
        )


class GetParentsByChildIdArgs(BaseModel):
    """Arguments for getting parents by child id."""

    child_id: str = Field(..., description="The solution_id of the child solution.")
    parent_cnt: int = Field(
        ..., description="The count of the parent solution to retrieve."
    )


class GetParentsByChildIdTool(FunctionTool):
    """Tool for getting parents by child id."""

    def __init__(self, func: Optional[Callable[..., Any]] = None):
        super().__init__(
            func=func,
            args_schema=GetParentsByChildIdArgs,
            name="Get_Parents_By_Child",
            description="Get parents by child id. "
            + "This tool is used when you need to find the parents of a specific solution based on its solution_id."
            + "This tool accepts two arguments 'child_id' and 'parent_cnt'. "
            + "The child_id is string, which is the solution_id of the child solution, "
            + "whose parents you want to fetch. "
            + "The parent_cnt is int, which is the count of the parent solution to retrieve. "
            + "Returns a list of solutions with the corresponding information.",
        )


class GetChildsByParentIdArgs(BaseModel):
    """Arguments for getting childs by parent id."""

    parent_id: str = Field(..., description="The solution_id of the parent solution.")
    child_cnt: int = Field(
        ..., description="The count of the child solution to retrieve."
    )


class GetChildsByParentTool(FunctionTool):
    """Tool for getting childs by parent id."""

    def __init__(self, func: Optional[Callable[..., Any]] = None):
        super().__init__(
            func=func,
            args_schema=GetChildsByParentIdArgs,
            name="Get_Childs_By_Parent",
            description="Get childs by parent id. "
            + "This tool is used when you need to find the childs of a specific solution based on its solution_id."
            + "This tool accepts two arguments 'parent_id' and 'child_cnt'. "
            + "The parent_id is string, which is the solution_id of the parent solution, "
            + "whose childs you want to fetch. "
            + "The child_cnt is int, which is the count of the child solution to retrieve. "
            + "Returns a list of solutions with the corresponding information.",
        )


class GetMemoryStatusArgs(BaseModel):
    """Arguments for getting memory status."""

    island_id: Optional[int] = Field(
        None, description="Island ID, if specified the island status will be returned."
    )


class GetMemoryStatusTool(FunctionTool):
    """Tool for getting memory status."""

    def __init__(self, func: Optional[Callable[..., Any]] = None):
        super().__init__(
            func=func,
            args_schema=GetMemoryStatusArgs,
            name="Get_Memory_Status",
            description="Get memory status. "
            + "This tool is used when you need to know the current memory status of the evolution system."
            + "This tool accepts one available argument 'island_id', if specified the island status will be returned."
            + "Returns a dictionary containing the total memory status and the specified island status."
            + "The island_id should be integer",
        )
