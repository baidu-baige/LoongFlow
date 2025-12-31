#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolux.evolve.executor
"""
import asyncio
import os
import unittest
import uuid
from pathlib import Path

from agents.math_evolve.evolve_executor.execute_react.execute_agent_react import (
    ExecuteAgentReactConfig,
    EvolveExecuteAgentReact,
)
from agentsdk.logger.message_logger import print_message
from agentsdk.message import Message
from agentsdk.message.elements import MimeType
from evolux.evolve.context import Context, EvaluatorConfig, LLMConfig
from evolux.evolve.evaluator.evaluator import LoongFlowEvaluator
from evolux.evolve.executor import Executor
from evolux.evolve.register import register_worker


def get_project_root():
    if root := os.getenv("PROJECT_ROOT"):
        return Path(root)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found.")


# OUTPUT_DIR = Path(get_project_root()) / "output"
OUTPUT_DIR = "./output"
RESOURCE_DIR = Path(get_project_root()) / "tests/evolux/evolve/executor/resource"


def _create_evaluator(timeout: float = 5.0) -> LoongFlowEvaluator:
    """create evaluator"""
    config = EvaluatorConfig(
        workspace_path=str(OUTPUT_DIR),
        evaluate_code=_get_evaluator_code(),
        llm_config=_create_llm_config(),
    )
    return LoongFlowEvaluator(config)


def _get_evaluator_code() -> str:
    """get evaluator code"""
    file = f"{RESOURCE_DIR}/evaluate_code.py"
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def _create_context() -> Context:
    """create context"""
    return Context(
        task_id=uuid.uuid4().hex[:4],
        trace_id=uuid.uuid4().hex[:4],
        island_id=1,
        current_iteration=1,
        base_path=str(OUTPUT_DIR),
        task=""""Task Name: Second autocorrelation inequality Problem
Task Description: Let  𝐶2  be the smallest constant for which one has ‖𝑓∗𝑓‖22≤𝐶2‖𝑓∗𝑓‖1‖𝑓∗𝑓‖∞ for all non-negative  𝑓:ℝ→ℝ . 
It is known that 0.88922 ≤ 𝐶2 ≤ 1, with the lower bound coming from a step function construction by Matolcsi and Vinuesa (2010).
                
Task Goal: Find a step function with 50 equally-spaced intervals on  [−1/4,1/4]  that gives a slightly better lower bound 0.8962 ≤ 𝐶2.
                
Task Requirements: 
1. Use Python to solve this problem. 
2. Fully implement the optimize_lower_bound function. The return result of optimize_lower_bound needs to be able to pass the verification of the verify_heights_sequence function. Ensure that the input and output parameters of optimize_lower_bound remain unchanged.
3. Do not rewrite the cal_lower_bound and verify_heights_sequence functions. Keep them unchanged.
4. Optimize the algorithm to the extreme, striving to raise c_lower_bound to beyond 0.8962.""",
    )


def _create_llm_config() -> LLMConfig:
    """create llm config"""
    return LLMConfig(
        url=os.getenv("LLM_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model="deepseek-r1-0528",
    )


def _create_executor_config() -> ExecuteAgentReactConfig:
    return ExecuteAgentReactConfig(
        llm_config=_create_llm_config(),
        max_rounds=1,
        parallel_candidates=1,
        react_max_steps=5,
    )


class TestEvolveExecutor(unittest.TestCase):
    def test_executor_create(self):
        register_worker("executor", "executor", EvolveExecuteAgentReact)
        executor = Executor("executor", _create_executor_config(), _create_evaluator())
        self.assertIsNotNone(executor)
        self.assertIsNotNone(executor.executor)

    def test_executor_run(self):
        asyncio.run(self._test_executor_run())

    async def _test_executor_run(self):
        register_worker("executor", "executor", EvolveExecuteAgentReact)
        executor = Executor("executor", _create_executor_config(), _create_evaluator())

        result = await executor.run(
            _create_context(),
            Message.from_text(
                data={
                    "best_plan_file_path": f"{RESOURCE_DIR}/best_plan.txt",
                    "parent_info_file_path": f"{RESOURCE_DIR}/parent_info.json",
                },
                mime_type=MimeType.APPLICATION_JSON,
            ),
        )
        print("result=======", result)
        print_message(result)
