#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for evolution redis_memory version
"""

import asyncio
import json
import os
import time
import unittest

from agentsdk.memory.evolution.base_memory import Solution
from agentsdk.memory.evolution.redis_memory import RedisMemory


class TestEvolutionMemory(unittest.TestCase):
    def test_redis_memory_initialization(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=10,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
        )
        self.assertIsNotNone(memory)
        self.assertEqual(memory.num_islands, 3)

    def test_redis_memory_add_solution(self):
        asyncio.run(self._test_redis_memory_add_solution())

    async def _test_redis_memory_add_solution(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=3,
            redis_url="redis://localhost:6379",
        )

        solution1 = Solution(
            generation=0,
            solution="this is a solution A",
            score=0.5,
            timestamp=time.time(),
        )
        solution_id1 = await memory.add_solution(solution1)
        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }

        self.assertIn(solution_id1, populations_result)

        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution_id1)
        self.assertEqual(solution1.island_id, 0)

        best_key = f"{memory.islands_key}:{0}:best"
        best_island_sid = memory.redis.hget(best_key, "best_solution_id")
        self.assertEqual(best_island_sid.decode("utf-8"), solution_id1)
        solution2 = Solution(
            generation=0,
            solution="this is a solution B",
            score=1.0,
            timestamp=time.time(),
        )
        solution_id2 = await memory.add_solution(solution2)
        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(solution_id2, populations_result)
        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution_id2)
        self.assertEqual(solution2.island_id, 1)
        best_key = f"{memory.islands_key}:{1}:best"
        best_island_sid = memory.redis.hget(best_key, "best_solution_id")

        self.assertEqual(best_island_sid.decode("utf-8"), solution_id2)
        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{0}"), 1)
        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{1}"), 1)

        solution3 = Solution(
            parent_id=solution_id1,
            solution="this is a solution C",
            score=2.0,
            timestamp=time.time(),
        )
        solution_id3 = await memory.add_solution(solution3)
        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(solution_id3, populations_result)

        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution_id3)
        self.assertEqual(solution3.island_id, 2)

        best_key = f"{memory.islands_key}:{2}:best"
        best_island_sid = memory.redis.hget(best_key, "best_solution_id")
        self.assertEqual(best_island_sid.decode("utf-8"), solution_id3)

        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{0}"), 1)
        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{1}"), 1)
        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{2}"), 1)

        solution4 = Solution(
            parent_id=solution_id1,
            solution="this is a solution D",
            score=1.0,
            timestamp=time.time(),
        )
        solution_id4 = await memory.add_solution(solution4)
        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution_id3)

        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(solution_id4, populations_result)

        self.assertEqual(solution4.island_id, 0)
        best_key = f"{memory.islands_key}:{0}:best"
        best_island_sid = memory.redis.hget(best_key, "best_solution_id")
        self.assertEqual(best_island_sid.decode("utf-8"), solution4.solution_id)

        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{0}"), 2)

        solution5 = Solution(
            parent_id=solution_id4,
            solution="this is a solution E",
            score=1.5,
            timestamp=time.time(),
        )
        solution_id5 = await memory.add_solution(solution5)
        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution3.solution_id)
        self.assertEqual(solution5.island_id, 0)

        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(solution_id5, populations_result)

        island_0_solutions = memory.redis.smembers(f"{memory.islands_key}:{0}")
        self.assertEqual(solution4.island_id, 0)
        best_0_key = f"{memory.islands_key}:{0}:best"
        best_island0_sid = memory.redis.hget(best_0_key, "best_solution_id")
        self.assertEqual(best_island0_sid.decode("utf-8"), f"{solution_id3}_migrated_0")

        best_1_key = f"{memory.islands_key}:{1}:best"
        best_island1_sid = memory.redis.hget(best_1_key, "best_solution_id")
        self.assertEqual(best_island1_sid.decode("utf-8"), f"{solution_id3}_migrated_1")

        best_2_key = f"{memory.islands_key}:{2}:best"
        best_island2_sid = memory.redis.hget(best_2_key, "best_solution_id")
        self.assertEqual(best_island2_sid.decode("utf-8"), solution_id3)

        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{0}"), 4)

        solution6 = Solution(
            parent_id=solution_id5,
            solution="this is a solution F",
            score=8,
            timestamp=time.time(),
        )
        solution_id6 = await memory.add_solution(solution6)
        best_sid = memory.redis.hget(memory.metadata_key, "best_solution_id")
        self.assertEqual(best_sid.decode("utf-8"), solution6.solution_id)
        self.assertEqual(solution6.island_id, 0)

        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(solution_id6, populations_result)

        best_0_key = f"{memory.islands_key}:{0}:best"
        best_island0_sid = memory.redis.hget(best_0_key, "best_solution_id")
        self.assertEqual(best_island0_sid.decode("utf-8"), solution6.solution_id)

        self.assertEqual(memory.redis.scard(f"{memory.islands_key}:{0}"), 5)
        solution_id6_obj = memory.redis.hget(memory.solutions_key, solution_id6)
        solution = Solution.from_dict(json.loads(solution_id6_obj))
        self.assertEqual(solution.generation, 3)

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
        solution_id6_obj = memory.redis.hget(memory.solutions_key, solution_id6)
        solution = Solution.from_dict(json.loads(solution_id6_obj))
        self.assertEqual(solution.score, 9)

    def test_redis_memory_get_solutions(self):
        asyncio.run(self._test_redis_memory_get_solutions())

    async def _test_redis_memory_get_solutions(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
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

        self.assertEqual(memory.redis.hlen(memory.solutions_key), 3)
        solution1 = memory.get_solutions(solution_ids=["solution1"])
        self.assertEqual(solution1[0].solution_id, "solution1")
        solution2 = memory.get_solutions(solution_ids=["solution1", "solution2"])
        self.assertEqual(len(solution2), 2)

    def test_redis_memory_get_best_solutions(self):
        asyncio.run(self._test_redis_memory_get_best_solutions())

    async def _test_redis_memory_get_best_solutions(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
        )

        solutions = [
            Solution(
                island_id=1,
                generation=0,
                solution="solution1",
                score=1.0,
                timestamp=time.time(),
            ),
            Solution(
                island_id=1,
                generation=0,
                solution="solution2",
                score=2.0,
                timestamp=time.time(),
            ),
            Solution(
                island_id=2,
                generation=0,
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

    def test_redis_memory_list_solutions(self):
        asyncio.run(self._test_redis_memory_list_solutions())

    async def _test_redis_memory_list_solutions(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
        )

        solutions = [
            Solution(
                generation=0, solution="solution1", score=1.0, timestamp=time.time()
            ),
            Solution(
                generation=0, solution="solution2", score=2.0, timestamp=time.time() + 1
            ),
            Solution(
                generation=0, solution="solution3", score=3.0, timestamp=time.time() + 2
            ),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        ascending_solutions = memory.list_solutions("asc", 5)
        self.assertEqual(ascending_solutions[0].score, 1.0)
        self.assertEqual(ascending_solutions[-1].score, 3.0)

        descending_solutions = memory.list_solutions("desc")
        self.assertEqual(descending_solutions[0].score, 3.0)
        self.assertEqual(descending_solutions[-1].score, 1.0)

    def test_redis_memory_sample_random(self):
        asyncio.run(self._test_redis_memory_sample_random())

    async def _test_redis_memory_sample_random(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            use_sampling_weight=True,
            sampling_weight_power=5,
            redis_url="redis://localhost:6379",
        )

        solutions = [
            Solution(solution="solution1", score=0.95, sample_weight=1.0),
            Solution(solution="solution2", score=0.8, sample_weight=1.0),
            Solution(solution="solution3", score=0.75, sample_weight=3.0),
        ]

        for sol in solutions:
            await memory.add_solution(sol)

        sampled_solution = memory.sample()
        populations = memory.redis.hgetall(memory.populations_key)
        populations_result = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in populations.items()
        }
        self.assertIn(sampled_solution.solution_id, populations_result)

    def test_redis_memory_checkpoint(self):
        asyncio.run(self._test_redis_memory_checkpoint())

    async def _test_redis_memory_checkpoint(self):
        memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
        )

        solutions = [
            Solution(
                solution_id="solution1",
                generation=0,
                solution="solution1",
                score=1.0,
                timestamp=time.time(),
            ),
            Solution(
                solution_id="solution2",
                generation=0,
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

        loaded_memory = RedisMemory(
            num_islands=3,
            population_size=20,
            elite_archive_size=5,
            migration_interval=5,
            redis_url="redis://localhost:6379",
        )
        loaded_memory.load_checkpoint(f"{checkpoint_path}/checkpoints/checkpoint-test")
        self.assertEqual(
            loaded_memory.redis.hlen(loaded_memory.solutions_key),
            2,
        )
        loaded_memory_solutions_raw = loaded_memory.redis.hgetall(
            loaded_memory.solutions_key
        )
        loaded_memory_solutions = {}
        for key, values in loaded_memory_solutions_raw.items():
            key_str = key.decode("utf-8")
            value_str = values.decode("utf-8")
            loaded_solution = json.loads(value_str)
            loaded_memory_solutions[key_str] = loaded_solution

        loaded_memory_populations_raw = loaded_memory.redis.hgetall(
            loaded_memory.populations_key
        )
        loaded_memory_populations = {}
        for key, values in loaded_memory_populations_raw.items():
            key_str = key.decode("utf-8")
            value_str = values.decode("utf-8")
            loaded_population = json.loads(value_str)
            loaded_memory_populations[key_str] = loaded_population

        self.assertIn("solution1", loaded_memory_solutions)
        self.assertIn("solution2", loaded_memory_populations)

        best_solution_id = loaded_memory.redis.hget(
            loaded_memory.metadata_key, "best_solution_id"
        )
        self.assertEqual(best_solution_id.decode("utf-8"), "solution2")
        self.assertEqual(
            loaded_memory_solutions["solution1"],
            solutions[0].to_dict(),
        )
        self.assertEqual(
            loaded_memory_populations["solution2"],
            solutions[1].to_dict(),
        )

        island_0_solutions = loaded_memory.redis.smembers(
            f"{loaded_memory.islands_key}:0"
        )
        island_1_solutions = loaded_memory.redis.smembers(
            f"{loaded_memory.islands_key}:1"
        )
        self.assertEqual(len(island_0_solutions), 1)
        self.assertEqual(len(island_1_solutions), 1)
        self.assertEqual(b"solution1", island_0_solutions.pop())
        self.assertEqual(b"solution2", island_1_solutions.pop())


if __name__ == "__main__":
    unittest.main()
