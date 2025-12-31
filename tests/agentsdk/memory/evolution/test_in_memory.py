#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolution in_memory version
"""

import asyncio
import os
import time
import unittest

from agentsdk.memory.evolution.base_memory import Solution
from agentsdk.memory.evolution.in_memory import InMemory


class TestEvolutionMemory(unittest.TestCase):
    def test_in_memory_initialization(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=3,
        )
        self.assertIsNotNone(memory)
        self.assertEqual(len(memory.islands), 3)

    def test_in_memory_add_solution(self):
        asyncio.run(self._test_in_memory_add_solution())

    async def _test_in_memory_add_solution(self):
        memory = InMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=3,
        )

        # Initial state, all islands are empty, system will use island 0 as default
        solution1 = Solution(
            generation=0,
            solution="this is a solution A",
            score=0.5,
            timestamp=time.time(),
        )
        solution_id1 = await memory.add_solution(solution1)
        self.assertIn(solution_id1, memory.populations)
        self.assertEqual(memory.best_solution_id, solution_id1)
        self.assertEqual(solution1.island_id, 0)
        self.assertEqual(memory.island_best_solution[0], solution_id1)

        # Island 0 now has 1 solution, add solution2, it will be added to island 1
        solution2 = Solution(
            generation=0,
            solution="this is a solution B",
            score=1.0,
            timestamp=time.time(),
        )
        solution_id2 = await memory.add_solution(solution2)
        self.assertIn(solution_id2, memory.populations)
        self.assertEqual(memory.best_solution_id, solution_id2)
        self.assertEqual(solution2.island_id, 1)
        self.assertEqual(memory.island_best_solution[1], solution_id2)
        self.assertEqual(len(memory.islands[0]), 1)
        self.assertEqual(len(memory.islands[1]), 1)

        # Since island 2 is empty, so solution3 will be added to island 2
        solution3 = Solution(
            parent_id=solution_id1,
            solution="this is a solution C",
            score=2.0,
            timestamp=time.time(),
        )
        solution_id3 = await memory.add_solution(solution3)
        self.assertIn(solution_id3, memory.solutions)
        self.assertEqual(memory.best_solution_id, solution3.solution_id)
        self.assertEqual(solution3.island_id, 2)
        self.assertEqual(memory.island_best_solution[2], solution3.solution_id)
        self.assertEqual(len(memory.islands[0]), 1)
        self.assertEqual(len(memory.islands[1]), 1)
        self.assertEqual(len(memory.islands[2]), 1)

        # Add solution4 with parent solution 1, so solution4 will be added to island 0.
        solution4 = Solution(
            parent_id=solution_id1,
            solution="this is a solution D",
            score=1.0,
            timestamp=time.time(),
        )
        solution_id4 = await memory.add_solution(solution4)
        self.assertEqual(memory.best_solution_id, solution3.solution_id)
        self.assertIn(solution_id4, memory.populations)
        self.assertEqual(solution4.island_id, 0)
        self.assertEqual(memory.island_best_solution[0], solution4.solution_id)
        self.assertEqual(len(memory.islands[0]), 2)

        # Add solution5 with parent solution 4, so solution5 will be added to island 0.
        # after adding solution5, migration will be triggered since migration_interval is 3, island0's capacity will be 4
        solution5 = Solution(
            parent_id=solution_id4,
            solution="this is a solution E",
            score=1.5,
            timestamp=time.time(),
        )
        solution_id5 = await memory.add_solution(solution5)
        self.assertEqual(memory.best_solution_id, solution3.solution_id)
        self.assertIn(solution_id5, memory.populations)
        self.assertEqual(solution5.island_id, 0)
        self.assertEqual(memory.island_capacity[0], 4)
        self.assertEqual(memory.island_best_solution[0], f"{solution_id3}_migrated_0")
        self.assertEqual(memory.island_best_solution[1], f"{solution_id3}_migrated_1")
        self.assertEqual(len(memory.islands[1]), 3)
        self.assertEqual(memory.island_best_solution[2], solution_id3)

        # Add solution6 with parent solution 5, so solution6 will be added to island 0.
        solution6 = Solution(
            parent_id=solution_id5,
            solution="this is a solution F",
            score=8,
            timestamp=time.time(),
        )
        solution_id6 = await memory.add_solution(solution6)
        self.assertEqual(memory.best_solution_id, solution6.solution_id)
        self.assertIn(solution_id6, memory.populations)
        self.assertEqual(solution6.island_id, 0)
        self.assertEqual(memory.island_best_solution[0], solution6.solution_id)
        self.assertEqual(memory.island_capacity[0], 5)
        self.assertEqual(memory.solutions[solution_id6].generation, 3)

        metrics = memory.memory_status()
        print(metrics)

        parents_of_solution5 = memory.get_parents_by_child_id(
            child_id=solution_id5, parent_cnt=1
        )
        self.assertEqual(len(parents_of_solution5), 1)
        self.assertEqual(parents_of_solution5[0].solution_id, solution_id4)
        self.assertEqual(parents_of_solution5[0].parent_id, solution_id1)

        childs_of_solution1 = memory.get_childs_by_parent_id(
            parent_id=solution_id1, child_cnt=3
        )
        self.assertEqual(len(childs_of_solution1), 2)
        self.assertIn(solution4, childs_of_solution1)

        update_solution_id = await memory.update_solution(
            solution_id=solution_id6, score=9
        )
        self.assertEqual(update_solution_id, solution_id6)
        self.assertEqual(memory.solutions[solution_id6].score, 9)

    def test_in_memory_get_solutions(self):
        asyncio.run(self._test_in_memory_get_solutions())

    async def _test_in_memory_get_solutions(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
        )

        solutions = [
            Solution(
                solution_id="solution1",
                solution="solution1",
                score=1.0,
                timestamp=time.time(),
            ),
            Solution(
                solution_id="solution2",
                solution="solution2",
                score=2.0,
                timestamp=time.time(),
            ),
            Solution(
                solution_id="solution3",
                solution="solution3",
                score=3.0,
                timestamp=time.time(),
            ),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        self.assertEqual(len(memory.solutions), 3)
        solution1 = memory.get_solutions(solution_ids=["solution1"])
        self.assertEqual(solution1[0].solution_id, "solution1")
        solution2 = memory.get_solutions(solution_ids=["solution1", "solution2"])
        self.assertEqual(len(solution2), 2)

    def test_in_memory_get_best_solutions(self):
        asyncio.run(self._test_in_memory_get_best_solutions())

    async def _test_in_memory_get_best_solutions(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
        )

        solutions = [
            Solution(
                island_id=1,
                solution="solution1",
                score=1.0,
                timestamp=time.time(),
            ),
            Solution(
                island_id=1,
                solution="solution2",
                score=2.0,
                timestamp=time.time(),
            ),
            Solution(
                island_id=2,
                solution="solution3",
                score=3.0,
                timestamp=time.time(),
            ),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        best_solution = memory.get_best_solutions()
        self.assertEqual(best_solution[0].score, 3.0)

        best_solutions = memory.get_best_solutions(island_id=1, top_k=2)
        self.assertEqual(len(best_solutions), 2)
        self.assertEqual(best_solutions[0].score, 2.0)
        self.assertEqual(best_solutions[1].score, 1.0)

        best_solutions = memory.get_best_solutions(island_id=2)
        self.assertEqual(len(best_solutions), 1)
        self.assertEqual(best_solutions[0].score, 3.0)

    def test_in_memory_list_solutions(self):
        asyncio.run(self._test_in_memory_list_solutions())

    async def _test_in_memory_list_solutions(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
        )

        solutions = [
            Solution(solution="solution1", score=1.0, timestamp=time.time()),
            Solution(solution="solution2", score=2.0, timestamp=time.time() + 1),
            Solution(solution="solution3", score=3.0, timestamp=time.time() + 2),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        ascending_solutions = memory.list_solutions("asc", 5)
        self.assertEqual(ascending_solutions[0].score, 1.0)
        self.assertEqual(ascending_solutions[-1].score, 3.0)

        descending_solutions = memory.list_solutions("desc")
        self.assertEqual(descending_solutions[0].score, 3.0)
        self.assertEqual(descending_solutions[-1].score, 1.0)

    def test_in_memory_sample_random(self):
        asyncio.run(self._test_in_memory_sample_random())

    async def _test_in_memory_sample_random(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
            use_sampling_weight=True,
            sampling_weight_power=5,
        )

        solutions = [
            Solution(solution="solution1", score=0.95, sample_weight=1.0),
            Solution(solution="solution2", score=0.8, sample_weight=1.0),
            Solution(solution="solution3", score=0.75, sample_weight=3.0),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        result = {"solution1": 0, "solution2": 0, "solution3": 0}
        for i in range(100):
            sampled_solution = memory.sample()
            result[sampled_solution.solution] += 1
        print(result)

    def test_in_memory_checkpoint(self):
        asyncio.run(self._test_in_memory_checkpoint())

    async def _test_in_memory_checkpoint(self):
        memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
        )

        solutions = [
            Solution(
                solution_id="solution1",
                solution="solution1",
                score=1.0,
                timestamp=time.time(),
            ),
            Solution(
                solution_id="solution2",
                solution="solution2",
                score=2.0,
                timestamp=time.time(),
            ),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        checkpoint_path = "test_checkpoint"
        await memory.save_checkpoint(checkpoint_path, "test")

        # Verify checkpoint files exist
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    checkpoint_path,
                    "checkpoints",
                    "checkpoint-test",
                    "solutions",
                    "solution1.json",
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    checkpoint_path,
                    "checkpoints",
                    "checkpoint-test",
                    "solutions",
                    "solution2.json",
                )
            )
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    checkpoint_path,
                    "checkpoints",
                    "checkpoint-test",
                    "best_solution.json",
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    checkpoint_path, "checkpoints", "checkpoint-test", "metadata.json"
                )
            )
        )

        loaded_memory = InMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
        )
        loaded_memory.load_checkpoint(f"{checkpoint_path}/checkpoints/checkpoint-test")
        self.assertEqual(len(loaded_memory.solutions), 2)
        self.assertIn("solution1", loaded_memory.solutions)
        self.assertIn("solution2", loaded_memory.solutions)
        self.assertEqual(loaded_memory.best_solution_id, "solution2")
        self.assertEqual(loaded_memory.solutions["solution1"], solutions[0])
        self.assertEqual(loaded_memory.populations["solution2"], solutions[1])
        self.assertEqual(len(loaded_memory.islands[0]), 1)
        self.assertEqual(len(loaded_memory.islands[1]), 1)
        self.assertEqual(loaded_memory.islands[0].pop(), "solution1")
        self.assertEqual(loaded_memory.islands[1].pop(), "solution2")


if __name__ == "__main__":
    unittest.main()
