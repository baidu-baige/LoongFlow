#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolution memory_factory
"""

import logging
import time
import unittest

from agentsdk.memory.evolution.base_memory import Solution
from agentsdk.memory.evolution.memory_factory import MemoryFactory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestMemoryFactory(unittest.TestCase):
    def test_add_solution(self):
        self.factory = MemoryFactory(
            storage_type="in_memory",
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
        )

        # Add solution1 without specifying island_id, system will use island 0 as default
        solution1 = Solution(
            generation=0,
            solution="this is a solution A",
            score=0.5,
            timestamp=time.time(),
        )
        solution_id1 = self.factory.add_solution(solution1)
        self.assertIn(solution_id1, self.factory._memory.populations)
        self.assertEqual(self.factory._memory.best_solution_id, solution1.solution_id)
        self.assertEqual(solution1.island_id, 0)
        self.assertEqual(
            self.factory._memory.island_best_solution[0], solution1.solution_id
        )


if __name__ == "__main__":
    unittest.main()
