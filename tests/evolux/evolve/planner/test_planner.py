#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolux.evolve.planner
"""
import asyncio
import unittest

from agents import (
    EVOLVE_PLANNER_SYSTEM_PROMPT,
)
from agents.math_evolve.evolve_planner.plan_agent import EvolvePlanAgent
from evolux.evolve.context import LLMConfig, Context
from evolux.evolve.context.config import DatabaseConfig
from evolux.evolve.database import EvolveDatabase
from evolux.evolve.planner import Planner
from evolux.evolve.register import register_worker


class TestEvolvePlanner(unittest.TestCase):
    def test_planner_create(self):
        full_config = {
            "planner_config": {
                "system_prompt": EVOLVE_PLANNER_SYSTEM_PROMPT,
                "llm_config": LLMConfig(
                    url="xxx",
                    api_key="xxx",
                    model="deepseek-v3",
                ),
            }
        }
        register_worker("planner", "planner", EvolvePlanAgent)
        planner = Planner(
            "planner", full_config, EvolveDatabase.create_database(DatabaseConfig())
        )
        self.assertIsNotNone(planner)
        self.assertIsNotNone(planner.planner)

    def test_planner_run(self):
        asyncio.run(self._test_planner_create_planner())

    async def _test_planner_create_planner(self):
        full_config = {
            "planner_config": {
                "system_prompt": EVOLVE_PLANNER_SYSTEM_PROMPT,
                "llm_config": LLMConfig(
                    url="xxx",
                    api_key="xxx",
                    model="deepseek-v3",
                ),
            }
        }
        register_worker("planner", "planner", EvolvePlanAgent)
        planner = Planner(
            "planner", full_config, EvolveDatabase.create_database(DatabaseConfig())
        )
        result = await planner.run(
            Context(
                init_solution="""# EVOLVE-BLOCK-START

import numpy as np

def optimize_lower_bound():
    # This is a sample return. It is necessary to provide a complete implementation
    heights_sequence_2 = np.zeros(50)
    heights_sequence_2.fill(1)

    # This part should remain unchanged
    convolution_2 = np.convolve(heights_sequence_2, heights_sequence_2)
    c_lower_bound = cal_lower_bound(convolution_2)
    return heights_sequence_2, c_lower_bound

# EVOLVE-BLOCK-END

def cal_lower_bound(convolution_2: list[float]):
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    c_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {c_lower_bound}")
    return c_lower_bound

def verify_heights_sequence(heights_sequence_2: np.ndarray, c_lower_bound: float):
    if len(heights_sequence_2) != 50:
        return False, f"len(heights_sequence_2) not 50"

    for i in range(len(heights_sequence_2)):
        if heights_sequence_2[i] < 0:
            return False, f"heights_sequence_2 all elements must be non-negative"

    convolution_2 = np.convolve(heights_sequence_2, heights_sequence_2)
    c_c_lower_bound = cal_lower_bound(convolution_2)
    if c_lower_bound != c_c_lower_bound:
        return False, f"c_lower_bound: {c_lower_bound} miscalculation, the correct value is {c_c_lower_bound}"

    return True, ""

if __name__ == "__main__":
    heights_sequence_2, c_lower_bound = optimize_lower_bound()
    print(f"Step function values: {heights_sequence_2}, C2 >= {c_lower_bound}")

    valid, err = verify_heights_sequence(heights_sequence_2, c_lower_bound)
    print(f"valid = {valid} err = {err}")
""",
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
            None,
        )
        print(result)
