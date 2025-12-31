#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolux.evolve.database.database module.
"""
import asyncio
import logging
import unittest

from agentsdk.memory.evolution import Solution
from evolux.evolve.context.config import DatabaseConfig
from evolux.evolve.database.database import EvolveDatabase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestEvolveDatabase(unittest.TestCase):
    def test_add_solution(self):
        asyncio.run(self._test_add_solution())

    async def _test_add_solution(self):
        database = EvolveDatabase(DatabaseConfig(storage_type="in_memory"))
        solution = Solution(
            solution="test solution",
            evaluation={"test_metric": 1},
            parent_id="parent_id",
            score=0.9,
            island_id=0,
            generate_plan="test generate plan",
            summary="test summary",
            metadata={},
        )

        solution_id = await database.aadd_solution(solution)
        self.assertEqual(len(database._evolution_memory._memory.solutions), 1)
        self.assertEqual(len(database._evolution_memory._memory.populations), 1)
        self.assertEqual(
            database._evolution_memory._memory.solutions[solution_id].solution,
            "test solution",
        )
        self.assertEqual(database._evolution_memory._memory.islands[0], {solution_id})
        self.assertEqual(
            database._evolution_memory._memory.best_solution_id,
            solution_id,
        )


if __name__ == "__main__":
    unittest.main()
