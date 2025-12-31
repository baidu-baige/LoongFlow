# -*- coding: utf-8 -*-
"""
Unit tests for evolux.evolve.summary
"""
import time

import pytest

from agents import EvolveSummaryAgent
from agents.math_evolve.evolve_prompt.evolve_summary_prompt import (
    EVOLVE_SUMMARY_SYSTEM_PROMPT,
    EVOLVE_SUMMARY_USER_PROMPT,
)
from agentsdk.memory.evolution import Solution
from agentsdk.message import ContentElement, Message
from evolux.evolve.context import Context, LLMConfig
from evolux.evolve.context.config import DatabaseConfig
from evolux.evolve.database import EvolveDatabase
from evolux.evolve.register import register_worker
from evolux.evolve.summary import Summary


class TestEvolveSummarizer:
    def test_summary_create(self):
        full_config = {
            "summary_config": {
                "system_prompt": EVOLVE_SUMMARY_SYSTEM_PROMPT,
                "user_prompt": EVOLVE_SUMMARY_USER_PROMPT,
                "max_steps": 64,
                "llm_config": LLMConfig(
                    url="xxx",
                    api_key="xxx",
                    model="deepseek-v3",
                ),
            }
        }
        register_worker("summary", "summary", EvolveSummaryAgent)
        summary = Summary(
            "summary", full_config, EvolveDatabase.create_database(DatabaseConfig())
        )
        assert summary is not None
        assert summary.summary is not None

    @pytest.mark.asyncio
    async def test_run(self):
        full_config = {
            "system_prompt": EVOLVE_SUMMARY_SYSTEM_PROMPT,
            "llm_config": LLMConfig(
                url="xxx",
                api_key="xxx",
                model="deepseek-v3",
            ),
        }
        register_worker("summary", "summary", EvolveSummaryAgent)

        db = EvolveDatabase.create_database(DatabaseConfig())
        await db.add_solution(
            Solution(
                solution="xx",
                solution_id="1",
                generate_plan="xx",
                parent_id="",
                island_id=1,
                iteration=1,
                timestamp=time.time(),
                generation=0,
                sample_cnt=1,
                sample_weight=5,
                score=0.2,
                evaluation="xx",
                summary="xx",
                metadata={},
            )
        )

        summary = Summary("summary", full_config, db)
        message = await summary.run(
            Context(
                base_path="./tests/evolux/evolve/summary/resource",
                task="""Task Name: Second autocorrelation inequality Problem
        Task Description: Let  𝐶2  be the smallest constant for which one has ‖𝑓∗𝑓‖22≤𝐶2‖𝑓∗𝑓‖1‖𝑓∗𝑓‖∞ for all non-negative  𝑓:ℝ→ℝ . 
        It is known that 0.88922 ≤ 𝐶2 ≤ 1, with the lower bound coming from a step function construction by Matolcsi and Vinuesa (2010).

        Task Goal: Find a step function with 50 equally-spaced intervals on  [−1/4,1/4]  that gives a slightly better lower bound 0.8962 ≤ 𝐶2.

        Task Requirements: 
        1. Use Python to solve this problem. 
        2. Fully implement the optimize_lower_bound function. The return result of optimize_lower_bound needs to be able to pass the verification of the verify_heights_sequence function. Ensure that the input and output parameters of optimize_lower_bound remain unchanged.
        3. Do not rewrite the cal_lower_bound and verify_heights_sequence functions. Keep them unchanged.
        4. Optimize the algorithm to the extreme, striving to raise c_lower_bound to beyond 0.8962.
        """,
            ),
            Message.from_elements(
                [
                    ContentElement(
                        data={
                            "best_plan_file_path": "./tests/evolux/evolve/summary/resource/best_plan.txt",
                            "best_solution_file_path": "./tests/evolux/evolve/summary/resource/best_solution.txt",
                            "best_evaluation_file_path": "./tests/evolux/evolve/summary/resource/best_evaluator.json",
                            "parent_info_file_path": "./tests/evolux/evolve/summary/resource/parent_info.json",
                        }
                    )
                ]
            ),
        )
        print(message)
