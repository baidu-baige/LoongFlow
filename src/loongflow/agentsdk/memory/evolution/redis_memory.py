#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provide in-memory implementation of evolution memory.
"""

import heapq
import json
import logging
import os
import threading
import time
import uuid
from operator import attrgetter
from typing import Dict, Optional

from redis import Redis, ConnectionPool

from .base_memory import EvolveMemory, Solution
from .boltzmann import select_parents_with_dynamic_temperature

logger = logging.getLogger(__name__)

update_list_lua_script = """
redis.call('DEL', KEYS[1])
for _, value in ipairs(ARGV) do
    redis.call('RPUSH', KEYS[1], value)
end
"""

reset_island_lua_script = """
redis.call('SREM', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[1], ARGV[2])
"""


class RedisMemory(EvolveMemory):
    """Redis-based implementation of Evolution Memory storage."""

    def __init__(
        self,
        num_islands: int = 3,
        population_size: int = 100,
        elite_archive_size: int = 50,
        migration_interval: int = 10,
        migration_rate: float = 0.2,
        boltzmann_temperature: float = 1.0,
        feature_bins: int = None,
        feature_dimensions=None,
        feature_scaling_method: str = "minmax",
        use_sampling_weight: bool = True,
        sampling_weight_power: float = 1.0,
        output_path: str = "output",
        redis_url: str = "redis://localhost:6379/0",
    ):
        """Initialize Redis connection and data structures"""
        super().__init__()
        if feature_dimensions is None:
            feature_dimensions = ["complexity", "diversity", "score"]
        self.num_islands: int = num_islands
        self.population_size: int = population_size
        self.elite_archive_size: int = elite_archive_size
        self.boltzmann_temperature: float = boltzmann_temperature
        self.migration_interval: int = migration_interval
        self.migration_rate: float = migration_rate
        self.feature_dimensions: list[str] = feature_dimensions
        self.use_sampling_weight: bool = use_sampling_weight
        self.sampling_weight_power: float = sampling_weight_power
        self.output_path: str = output_path

        # Calculate feature_bins if not provided
        if feature_bins is None:
            self.feature_bins: int = int(
                pow(self.elite_archive_size, 1 / len(self.feature_dimensions)) + 0.99
            )
        else:
            self.feature_bins: int = feature_bins

        self.feature_bins_per_dim: Dict[str, int] = {
            dim: self.feature_bins for dim in feature_dimensions
        }
        self.feature_scaling_method: str = feature_scaling_method

        # Redis connection with optimized pool settings
        self.redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=64,  # Increased from default 10
            socket_keepalive=True,
            socket_timeout=30,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self.redis = Redis(
            connection_pool=self.redis_pool,
            decode_responses=True,  # Keep bytes for performance
        )

        # Initialize Redis keys
        self.islands_key = f"evolution:islands:{self.memory_id}"
        self.solutions_key = f"evolution:solutions:{self.memory_id}"
        self.populations_key = f"evolution:populations:{self.memory_id}"
        self.elites_key = f"evolution:elites:{self.memory_id}"
        self.island_feature_maps_key = f"evolution:island_feature_maps:{self.memory_id}"
        self.feature_stats_key = f"evolution:feature_stats:{self.memory_id}"
        self.diversity_cache_key = f"evolution:diversity_cache:{self.memory_id}"
        self.diversity_reference_set_key = (
            f"evolution:diversity_reference_set:{self.memory_id}"
        )
        self.metadata_key = f"evolution:metadata:{self.memory_id}"

        # Thread safety with optimized locking
        self._lock = threading.RLock()
        self._island_locks: Dict[int, threading.Lock] = {  # Changed to simpler Lock
            i: threading.Lock() for i in range(num_islands)  # Lighter than RLock
        }

        # Initialize metadata
        self._init_metadata()

        logger.info(f"Initialized RedisMemory with ID {self.memory_id}")

    def _init_metadata(self):
        """Initialize metadata in Redis"""
        with self._lock:
            if not self.redis.exists(self.metadata_key):
                metadata = {
                    "current_island": 0,
                    "last_iteration": 0,
                    "last_migration_generation": 0,
                    "current_island_counter": 0,
                    "solutions_per_island": max(
                        1, self.population_size // self.num_islands
                    ),
                }
                # Use pipeline for atomic metadata initialization
                with self.redis.pipeline() as pipe:
                    pipe.hset(self.metadata_key, mapping=metadata)
                    pipe.execute()
                logger.debug("Initialized Redis metadata")

    async def add_solution(self, solution: Solution) -> str:
        """
        Add solution to Redis memory

        Args:
            solution: The solution to add to the memory

        Returns:
            str: Solution ID assigned to the solution
        """
        if not isinstance(solution, Solution):
            raise ValueError("solution must be an instance of Solution")
        with self._lock:
            self._prepare_solution(solution)
            solution_dict = solution.to_dict()
            solution_json = json.dumps(solution_dict, ensure_ascii=False)

            # Store in both solutions and populations
            self.redis.hset(self.solutions_key, solution.solution_id, solution_json)

            if not solution.score:
                logger.warning(
                    f"WARNING: No score found for solution {solution.solution_id}. Skipping."
                )
                return solution.solution_id

            # These operations can be done outside the pipeline as they have their own locking
            map_elites_feature = self._calculate_MAP_Elites(solution)
            solution.metadata["MAP_Elite_feature"] = map_elites_feature
            solution_dict = solution.to_dict()
            solution_json = json.dumps(solution_dict, ensure_ascii=False)
            self.redis.hset(self.populations_key, solution.solution_id, solution_json)
            self.redis.hset(self.solutions_key, solution.solution_id, solution_json)

            self._update_island(solution)
            self._update_elites(solution)

            # Enforce population limits
            self._enforce_population_limit(
                exclude_solution_id=solution.solution_id,
            )

            # Update tracking with another pipeline
            self._update_best_solution(solution)
            self._update_island_best_solution(solution, solution.island_id)
            await self._check_migration()

            logger.debug(f"Added solution {solution.solution_id} to memory")
            return solution.solution_id

    async def update_solution(self, solution_id: str, **kwargs) -> str:
        """
        Update solution in memory with optimized workflow.

        Args:
            solution_id: Solution ID to update
            **kwargs: Additional arguments to update solution with

        Returns:
            Updated solution ID
        """
        # Get all solutions from Redis
        if self.redis.hget(self.solutions_key, solution_id) is None:
            raise ValueError("solution_id does not exist in memory")

        for k, v in kwargs.items():
            if k == "island_id" or k == "parent_id":
                raise ValueError("Cannot update island_id or parent_id directly")

        solution_ori = self.redis.hget(self.solutions_key, solution_id)
        solution = Solution.from_dict(json.loads(solution_ori.decode("utf-8")))
        updated_solution = solution.copy()
        updated_solution.update(**kwargs)

        with self._lock:
            solution_dict = updated_solution.to_dict()
            solution_json = json.dumps(solution_dict, ensure_ascii=False)

            # Store in both solutions and populations
            self.redis.hset(self.solutions_key, solution.solution_id, solution_json)
            self.redis.hset(self.populations_key, solution.solution_id, solution_json)

        return solution_id

    def get_solutions(self, solution_ids: Optional[list[str]] = None) -> list[Solution]:
        """
        Get solutions by their IDs.

        Args:
            solution_ids: list of solution IDs to retrieve. If None, returns error.

        Returns:
            list of Solution objects.
        """
        if not solution_ids:
            raise ValueError("No solution IDs provided")

        try:
            with self._lock:
                results = self.redis.hmget(self.solutions_key, solution_ids)
                solutions = []
                for s in results:
                    if s is not None:
                        try:
                            solutions.append(
                                Solution.from_dict(json.loads(s.decode("utf-8")))
                            )
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(
                                f"Failed to parse solution from Redis: {str(e)}"
                            )
                            continue
                return solutions
        except Exception as e:
            logger.error(f"Error retrieving solutions from Redis: {str(e)}")
            raise

    def list_solutions(
        self, filter_type: str = "asc", limit: Optional[int] = None
    ) -> list[Solution]:
        """
        List solutions with optimized sorting and memory usage.

        Args:
            filter_type: "asc" for ascending or "desc" for descending order by timestamp
            limit: Maximum number of solutions to return

        Returns:
            list of Solutions sorted by timestamp
        """
        if filter_type not in ("asc", "desc"):
            raise ValueError("filter_type must be 'asc' or 'desc'")

        with self._lock:
            # Get all solutions from Redis
            solutions = [
                Solution.from_dict(json.loads(s.decode("utf-8")))
                for s in self.redis.hvals(self.solutions_key)
            ]

            # Optimized sorting with key function caching
            key_func = attrgetter("timestamp")
            solutions_len = self.redis.hlen(self.solutions_key)
            if limit is None or limit >= solutions_len:
                return sorted(solutions, key=key_func, reverse=(filter_type == "desc"))

            # Use nsmallest/nlargest with generator for better memory efficiency
            return (
                heapq.nsmallest(limit, solutions, key=key_func)
                if filter_type == "asc"
                else heapq.nlargest(limit, solutions, key=key_func)
            )

    def get_best_solutions(
        self, island_id: Optional[int] = None, top_k: Optional[int] = None
    ) -> list[Solution]:
        """
        Get the best solutions with improved performance using generators.

        Args:
            island_id: Optional island ID to filter by
            top_k: Number of top solutions to return (default: 1)

        Returns:
            list of top Solutions sorted by score
        """
        top_k = 1 if top_k is None else top_k

        with self._lock:
            if island_id is not None:
                # Get solutions from specific island
                island_key = f"{self.islands_key}:{island_id}"
                solution_ids = self.redis.smembers(island_key)
                solutions = [
                    Solution.from_dict(json.loads(s.decode("utf-8")))
                    for s in self.redis.hmget(self.populations_key, list(solution_ids))
                    if s
                ]
            else:
                # Get all solutions
                solutions = [
                    Solution.from_dict(json.loads(s.decode("utf-8")))
                    for s in self.redis.hvals(self.populations_key)
                ]

            return heapq.nlargest(top_k, solutions, key=attrgetter("score"))

    def sample(
        self, island_id: Optional[int] = None, exploration_rate: float = 0.2
    ) -> Optional[Solution]:
        """
        Sample a solution from the memory based on configured selection ratios.

        Selection process:
        1. With probability island_select_ratio: samples from current island
        2. With probability elite_select_ratio: samples from elite archive
        3. Otherwise: samples from all solutions

        Uses Boltzmann selection with dynamic temperature for sampling.

        Returns:
            Optional[Solution]: The sampled solution, or None if no solutions available.
        """
        with self._lock:
            # Sample from all solutions
            solutions = [
                Solution.from_dict(json.loads(s.decode("utf-8")))
                for s in self.redis.hvals(self.populations_key)
            ]

            if island_id is not None:
                solutions = [
                    Solution.from_dict(
                        json.loads(
                            self.redis.hget(self.populations_key, v).decode("utf-8")
                        )
                    )
                    for v in self.redis.smembers(f"{self.islands_key}:{island_id}")
                ]

            if not solutions:
                return None

            elites = []
            elites_ids = self.redis.smembers(self.elites_key)
            for elite_id in elites_ids:
                elite = Solution.from_dict(
                    json.loads(
                        self.redis.hget(self.populations_key, elite_id).decode("utf-8")
                    )
                )
                elites.append(elite)

            return select_parents_with_dynamic_temperature(
                solutions=solutions,
                elites=elites,
                initial_temp=self.boltzmann_temperature,
                use_sampling_weight=self.use_sampling_weight,
                sampling_weight_power=self.sampling_weight_power,
                exploration_rate=exploration_rate,
            )

    async def save_checkpoint(
        self, path: Optional[str] = None, tag: Optional[str] = None
    ) -> None:
        """
        Save the complete memory state to disk from Redis.

        Args:
            path: Optional directory path to save the checkpoint.
                 If None, uses the output_path specified in constructor.
                 Must be a writable directory path.
            tag: Optional identifier for the checkpoint.
                If None, uses current timestamp as identifier.
                Must be a valid filename string without special characters.

        Returns:
            None: This method does not return anything but saves checkpoint files to disk.
        """
        save_path = path or self.output_path
        if not save_path:
            raise ValueError("Path cannot be empty.")

        os.makedirs(save_path, exist_ok=True)

        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        tag = tag if tag else time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{tag}")
        os.makedirs(checkpoint_path, exist_ok=True)

        logger.info(f"Checkpointing Redis memory to {checkpoint_path}")

        with self._lock:
            # Save solutions
            solutions_path = os.path.join(checkpoint_path, "solutions")
            os.makedirs(solutions_path, exist_ok=True)

            # Get all solutions from Redis
            solutions = [
                Solution.from_dict(json.loads(s.decode("utf-8")))
                for s in self.redis.hvals(self.solutions_key)
            ]

            # Save each solution
            for solution in solutions:
                solution_path = os.path.join(
                    solutions_path, f"{solution.solution_id}.json"
                )
                with open(solution_path, "w") as f:
                    json.dump(solution.to_dict(), f, ensure_ascii=False, indent=2)

            feature_stats_raw = self.redis.hgetall(self.feature_stats_key)
            # Convert bytes to string keys and values
            feature_stats = {}
            for key, value in feature_stats_raw.items():
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                if value:
                    try:
                        # Try to parse JSON first, then fallback to string
                        value_str = (
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
                        feature_stats[key_str] = json.loads(value_str)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If JSON parsing fails, use string value
                        feature_stats[key_str] = value_str
                else:
                    feature_stats[key_str] = value

            island_best_solution = []
            for i in range(self.num_islands):
                island_best_solution_key = f"{self.islands_key}:{i}:best"
                best_solution_id = self.redis.hget(
                    island_best_solution_key, "best_solution_id"
                )
                # Convert bytes to string if needed
                if isinstance(best_solution_id, bytes):
                    island_best_solution.append(best_solution_id.decode("utf-8"))
                else:
                    island_best_solution.append(best_solution_id)

            # Save metadata
            metadata = {
                "total_generated_solutions": len(solutions),
                "total_valid_solutions": self.redis.hlen(self.solutions_key),
                "islands": [
                    [
                        member.decode("utf-8") if isinstance(member, bytes) else member
                        for member in self.redis.smembers(f"{self.islands_key}:{i}")
                    ]
                    for i in range(self.num_islands)
                ],
                "island_feature_map": {
                    i: {
                        key.decode("utf-8") if isinstance(key, bytes) else key: (
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
                        for key, value in self.redis.hgetall(
                            f"{self.island_feature_maps_key}:{i}"
                        ).items()
                    }
                    for i in range(self.num_islands)
                },
                "elites": [
                    member.decode("utf-8") if isinstance(member, bytes) else member
                    for member in self.redis.smembers(self.elites_key)
                ],
                "best_solution_id": (
                    self.redis.hget(self.metadata_key, "best_solution_id").decode(
                        "utf-8"
                    )
                ),
                "island_best_solution": island_best_solution,
                "last_iteration": int(
                    self.redis.hget(self.metadata_key, "last_iteration").decode("utf-8")
                    or 0
                ),
                "current_island": int(
                    self.redis.hget(self.metadata_key, "current_island").decode("utf-8")
                    or 0
                ),
                "last_migration_generation": int(
                    self.redis.hget(
                        self.metadata_key, "last_migration_generation"
                    ).decode("utf-8")
                    or 0
                ),
                "island_capacity": [
                    self.redis.scard(f"{self.islands_key}:{i}")
                    for i in range(self.num_islands)
                ],
                "feature_stats": self._serialize_feature_stats(feature_stats),
            }

            def bytes_encoder(obj):
                if isinstance(obj, bytes):
                    return obj.decode("utf-8", errors="replace")
                raise TypeError(
                    f"Object of type {obj.__class__.__name__} is not JSON serializable"
                )

            with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=bytes_encoder)

            logger.info(
                f"Saved checkpoint with {len(solutions)} programs to {checkpoint_path}"
            )

            # Save best solution found so far
            best_solution_id = self.redis.hget(
                self.metadata_key, "best_solution_id"
            ).decode("utf-8")
            if best_solution_id:
                best_solution = Solution.from_dict(
                    json.loads(
                        self.redis.hget(self.populations_key, best_solution_id).decode(
                            "utf-8"
                        )
                    )
                )
            else:
                best_solutions = self.get_best_solutions()
                best_solution = best_solutions[0] if len(best_solutions) > 0 else None

            if best_solution:
                best_solution_path = os.path.join(checkpoint_path, "best_solution.json")
                with open(best_solution_path, "w") as f:
                    json.dump(best_solution.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"Saved checkpoint with tag {tag} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the memory state from disk.

        Args:
            checkpoint_path: Directory path where the checkpoint is stored.
        Returns:
            None: This method does not return anything but loads checkpoint data into memory.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        with self._lock:
            with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

            island_feature_map = metadata.get(
                "island_feature_map", [{} for _ in range(self.num_islands)]
            )
            for i in range(self.num_islands):
                island_feature_map_key = f"{self.island_feature_maps_key}:{i}"
                for k, v in island_feature_map.get(i, {}).items():
                    self.redis.hset(island_feature_map_key, k, v)

            saved_islands = metadata.get("islands", [])

            elites = metadata.get("elites", [])
            if elites:
                self.redis.sadd(self.elites_key, *elites)
            self.redis.hset(
                self.metadata_key,
                mapping={
                    "best_solution_id": metadata.get("best_solution_id", ""),
                    "last_iteration": metadata.get("last_iteration", 0),
                    "current_island": metadata.get("current_island", 0),
                    "last_migration_generation": metadata.get(
                        "last_migration_generation", 0
                    ),
                },
            )
            feature_stats = self._deserialize_feature_stats(
                metadata.get("feature_stats", {})
            )
            for k, v in feature_stats.items():
                if isinstance(v, (dict, list)):
                    self.redis.hset(self.feature_stats_key, k, json.dumps(v, ensure_ascii=False))
                else:
                    self.redis.hset(self.feature_stats_key, k, str(v))

            island_best_solution = metadata.get(
                "island_best_solution", [None] * len(saved_islands)
            )
            for i, best_solution_id in enumerate(island_best_solution):
                if best_solution_id:
                    island_best_solution_key = f"{self.islands_key}:{i}:best"
                    self.redis.hset(
                        island_best_solution_key,
                        "best_solution_id",
                        best_solution_id,
                    )

            # Load solutions
            solutions_path = os.path.join(checkpoint_path, "solutions")
            for filename in os.listdir(solutions_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(solutions_path, filename)
                    try:
                        with open(file_path, "r") as f:
                            solution_dict = json.load(f)
                            solution = Solution.from_dict(solution_dict)
                            solution_json = json.dumps(solution_dict, ensure_ascii=False)
                            self.redis.hset(
                                self.solutions_key, solution.solution_id, solution_json
                            )
                            self.redis.hset(
                                self.populations_key,
                                solution.solution_id,
                                solution_json,
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to load solution from {file_path}: {str(e)}"
                        )
                        raise e

            self._reconstruct_islands(saved_islands)

    def memory_status(self, island_id: int = None) -> dict:
        """Return the status of the memory"""
        top_3_solutions = self.get_best_solutions(top_k=3)
        top_3_iterations = [
            top_3_solutions[i].iteration
            for i in range(min(3, len(top_3_solutions)))
            if top_3_solutions[i]
        ]

        best_solution = top_3_solutions[0] if len(top_3_solutions) > 0 else None
        populations_raw = self.redis.hgetall(self.populations_key)
        population_scores = []
        for key, values in populations_raw.items():
            value_str = values.decode("utf-8")
            population = Solution.from_dict(json.loads(value_str))
            population_scores.append(population.score)

        avg_score = (
            sum(population_scores) / len(population_scores)
            if len(population_scores) > 0
            else 0
        )
        better_ratio = (
            len([s for s in population_scores if s > avg_score])
            / len(population_scores)
            if len(population_scores) > 0
            else 0
        )

        result = {
            "global_status": {
                "current_iteration": int(
                    self.redis.hget(self.metadata_key, "last_iteration").decode("utf-8")
                ),
                "is_full": len(population_scores) == self.population_size,
                "top_3_iterations": top_3_iterations,
                "best_score": best_solution.score if best_solution else 0,
                "best_iteration": (best_solution.iteration if best_solution else 0),
                "avg_score": round(avg_score, 6),
                "better_ratio": round(better_ratio, 2),
            },
        }

        if island_id:
            island_solution_ids = self.redis.smembers(f"{self.islands_key}:{island_id}")
            if len(island_solution_ids) < 1:
                return result

            island_metrics = {"island_id": island_id}
            island_top_3_solutions = self.get_best_solutions(
                island_id=island_id, top_k=3
            )
            island_top_3_iterations = [
                island_top_3_solutions[j].iteration
                for j in range(min(3, len(island_top_3_solutions)))
            ]
            island_metrics["top_3_iterations"] = island_top_3_iterations

            island_best_solution = island_top_3_solutions[0]
            island_metrics["best_score"] = (
                island_best_solution.score if island_best_solution else 0
            )
            island_metrics["best_iteration"] = (
                island_best_solution.iteration if island_best_solution else 0
            )

            island_solutions = [
                Solution.from_dict(
                    json.loads(self.redis.hget(self.populations_key, v).decode("utf-8"))
                )
                for v in island_solution_ids
            ]
            island_scores = [solution.score for solution in island_solutions]
            island_metrics["avg_score"] = round(
                (
                    round(sum(island_scores) / len(island_scores), 6)
                    if len(island_scores) > 0
                    else 0
                ),
                6,
            )
            island_metrics["better_ratio"] = round(
                (
                    len([s for s in island_scores if s > island_metrics["avg_score"]])
                    / len(island_scores)
                    if len(island_scores) > 0
                    else 0
                ),
                2,
            )
            island_feature_maps_key = f"{self.island_feature_maps_key}:{island_id}"
            total_possible_cells = self.feature_bins ** len(self.feature_dimensions)
            feature_map_len = self.redis.hlen(island_feature_maps_key)
            coverage = (feature_map_len + 1) / total_possible_cells
            island_metrics["map_elites_feature_ratio"] = coverage
            result["island_status"] = island_metrics

        return result

    def get_parents_by_child_id(self, child_id: str, parent_cnt: int) -> list[Solution]:
        """
        Get parents by child id

        Args:
            child_id (str): Child solution id
            parent_cnt (int): Number of parents to retrieve

        Returns:
            list[Solution]: Parent solutions
        """
        with self._lock:
            if self.redis.hget(self.solutions_key, child_id) is None:
                raise ValueError(f"Child solution with id '{child_id}' not found.")

            parents = []
            while len(parents) < parent_cnt:
                child_ori = self.redis.hget(self.solutions_key, child_id)
                if child_ori is None:
                    break

                child = Solution.from_dict(json.loads(child_ori.decode("utf-8")))
                parent_id = child.parent_id
                parent_ori = self.redis.hget(self.solutions_key, parent_id)
                if parent_ori is None:
                    break

                parent = Solution.from_dict(json.loads(parent_ori.decode("utf-8")))
                parents.append(parent)

            return parents

    def get_childs_by_parent_id(self, parent_id: str, child_cnt: int) -> list[Solution]:
        """
        Get childs by parent id

        Args:
            parent_id (str): Parent solution id
            child_cnt (int): Number of children to retrieve

        Returns:
            list[Solution]: Child solutions
        """
        with self._lock:
            if self.redis.hget(self.solutions_key, parent_id) is None:
                raise ValueError(f"Parent solution with id '{parent_id}' not found.")

            childs = []
            parent_ori = self.redis.hget(self.solutions_key, parent_id)
            if parent_ori is None:
                return childs

            parent = Solution.from_dict(json.loads(parent_ori.decode("utf-8")))
            island_solutions = [
                Solution.from_dict(
                    json.loads(self.redis.hget(self.populations_key, v).decode("utf-8"))
                )
                for v in self.redis.smembers(f"{self.islands_key}:{parent.island_id}")
            ]
            for child_solution in island_solutions:
                if child_solution.parent_id == parent_id:
                    childs.append(child_solution)

            return childs

    def _reconstruct_islands(self, saved_islands: list[list[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island solution ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.num_islands)
        islands = [set() for _ in range(num_islands)]
        missing_solutions = []

        populations_raw = self.redis.hgetall(self.populations_key)
        populations = {}
        for key, values in populations_raw.items():
            key_str = key.decode("utf-8")
            value_str = values.decode("utf-8")
            population = Solution.from_dict(json.loads(value_str))
            populations[key_str] = population

        # Restore island assignments
        for island_idx, solution_ids in enumerate(saved_islands):
            if island_idx >= len(islands):
                continue

            for solution_id in solution_ids:
                if solution_id in populations:
                    # Solution exists, add to island
                    island_key = f"{self.islands_key}:{island_idx}"
                    self.redis.sadd(island_key, solution_id)
                    islands[island_idx].add(solution_id)
                else:
                    # Solution missing, track it
                    missing_solutions.append((island_idx, solution_id))

        # Clean up archive - remove missing solutions
        elites = self.redis.smembers(self.elites_key)
        for sid in elites:
            if sid not in populations:
                self.redis.srem(self.elites_key, sid)

        # Clean up island_feature_maps - remove missing programs
        feature_keys_to_remove = []
        for i in range(num_islands):
            island_feature_map_key = f"{self.island_feature_maps_key}:{i}"
            feature_map = self.redis.hgetall(island_feature_map_key)
            for feature_key_raw, solution_id_raw in feature_map.items():
                feature_key = (
                    feature_key_raw.decode("utf-8")
                    if isinstance(feature_key_raw, bytes)
                    else feature_key_raw
                )
                solution_id = (
                    solution_id_raw.decode("utf-8")
                    if isinstance(solution_id_raw, bytes)
                    else solution_id_raw
                )
                if solution_id not in populations:
                    feature_keys_to_remove.append((island_feature_map_key, feature_key))
                    self.redis.hdel(island_feature_map_key, feature_key)

        # Clean up island the best solutions - remove stale references
        self._cleanup_stale_island_bests()

        # Check best solution
        best_solution_id = self.redis.hget(self.metadata_key, "best_solution_id")
        if best_solution_id:
            best_solution_id = best_solution_id.decode("utf-8")
        if best_solution_id and best_solution_id not in populations:
            logger.warning(
                f"Best solution {best_solution_id} not found, will recalculate"
            )
            self.redis.hset(self.metadata_key, "best_solution_id", "")

        # If we have solutions but no island assignments, distribute them
        if populations and sum(len(island) for island in islands) == 0:
            logger.info(
                "No island assignments found, distributing programs across islands"
            )
            solution_ids = list(populations.keys())
            for i, solution_id in enumerate(solution_ids):
                island_idx = i % len(islands)
                island_key = f"{self.islands_key}:{island_idx}"
                self.redis.sadd(island_key, solution_id)

    def _get_current_island(self) -> int:
        """Get current island ID from Redis"""
        return int(
            self.redis.hget(self.metadata_key, "current_island").decode("utf-8") or 0
        )

    def _prepare_solution(self, solution: Solution) -> None:
        """Prepare solution for addition"""
        with self._lock:
            last_iteration = int(
                self.redis.hget(self.metadata_key, "last_iteration").decode("utf-8"), 0
            )
            if not solution.iteration:
                last_iteration = self.redis.hincrby(
                    self.metadata_key, "last_iteration", 1
                )
                solution.iteration = last_iteration

            last_iteration = max(last_iteration, solution.iteration)
            self.redis.hset(self.metadata_key, "last_iteration", last_iteration)

            if not solution.solution_id:
                solution.solution_id = uuid.uuid4().hex[:8]

            # if island_id already set, respect it
            if solution.island_id:
                return

            # Assign to island with no programs first
            island_without_program = []
            for i in range(self.num_islands):
                island_key = f"{self.islands_key}:{i}"
                if self.redis.scard(island_key) == 0:
                    island_without_program.append(i)

            if island_without_program:
                solution.island_id = min(island_without_program)
                return

            # If parent solution exists, inherit island and increment generation
            parent_solution_obj = self.redis.hget(
                self.solutions_key, solution.parent_id
            )
            if parent_solution_obj:
                parent_solution = Solution.from_dict(json.loads(parent_solution_obj))
                solution.generation = parent_solution.generation + 1
                solution.island_id = parent_solution.island_id
                return

            # Finally, Round-robin assignment
            current_island = self._get_current_island()
            solution.island_id = current_island
            current_island_counter = int(
                self.redis.hget(self.metadata_key, "current_island_counter").decode(
                    "utf-8"
                )
                or 0
            )
            solutions_per_island = int(
                self.redis.hget(self.metadata_key, "solutions_per_island").decode(
                    "utf-8"
                )
                or 1
            )
            if current_island_counter >= solutions_per_island:
                # Move to next island
                new_island = (current_island + 1) % self.num_islands
                self.redis.hset(self.metadata_key, "current_island", new_island)
                self.redis.hset(self.metadata_key, "current_island_counter", 0)

    def _calculate_MAP_Elites(self, solution: Solution) -> str:
        """
        Adapted from algorithmicsuperintelligence/openevolve (Apache-2.0 License)
        Original source: https://github.com/algorithmicsuperintelligence/openevolve/blob/a7428efeb5a30b7968975f182d5fb7060b36e978/openevolve/database.py#L221

        Update MAP-Elites feature map with the new solution.

        Args:
            solution: The solution to add to the MAP-Elites grid
        """
        with self.redis.pipeline() as pipe:
            # 1. Batch read existing data
            pipe.hgetall(self.populations_key)
            pipe.hgetall(self.feature_stats_key)
            pipe.hgetall(self.diversity_cache_key)
            pipe.lrange(self.diversity_reference_set_key, 0, -1)
            populations_raw, feature_stats, diversity_cache, diversity_reference_set = (
                pipe.execute()
            )
            populations = {
                key.decode("utf-8"): Solution.from_dict(
                    json.loads(value.decode("utf-8"))
                )
                for key, value in populations_raw.items()
            }

            # Repair the diversity_reference_set data type conversion issue
            if diversity_reference_set:
                diversity_reference_set = [
                    ref.decode("utf-8") if isinstance(ref, bytes) else str(ref)
                    for ref in diversity_reference_set
                ]
            else:
                diversity_reference_set = []

            # 2. Calculate new feature coords and diversity reference set
            feature_coords, diversity_reference_set = self._calculate_feature_coords(
                solution,
                populations,
                feature_stats,
                self.feature_bins_per_dim,
                self.feature_bins,
                self.feature_dimensions,
                diversity_cache,
                diversity_reference_set,
            )

            # 3. Batch write updated data
            if feature_stats:
                for k, v in feature_stats.items():
                    if isinstance(v, (dict, list)):
                        pipe.hset(self.feature_stats_key, k, json.dumps(v, ensure_ascii=False))
                    else:
                        pipe.hset(self.feature_stats_key, k, str(v))

            if diversity_cache:
                for k, v in diversity_cache.items():
                    if isinstance(v, (dict, list)):
                        pipe.hset(self.diversity_cache_key, k, json.dumps(v, ensure_ascii=False))
                    else:
                        pipe.hset(self.diversity_cache_key, k, str(v))
            if diversity_reference_set:  # Update diversity reference set
                pipe.eval(
                    update_list_lua_script,
                    1,
                    self.diversity_reference_set_key,
                    *diversity_reference_set,
                )

            pipe.execute()

        logger.debug(
            "Calculated feature coords for %s: %s",
            solution.solution_id[:6],
            feature_coords,
        )

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        island_feature_maps_key = f"{self.island_feature_maps_key}:{solution.island_id}"
        existing_solution_id = self.redis.hget(island_feature_maps_key, feature_key)
        should_replace = existing_solution_id is None

        logger.debug(
            "Feature key %s for %s, replace: %s",
            feature_key[:10],
            solution.solution_id[:6],
            should_replace,
        )

        if not should_replace:
            # Check if the existing program still exists before comparing
            if existing_solution_id not in populations:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale solution reference {existing_solution_id} in feature map"
                )
            else:
                # Solution exists, compare fitness
                should_replace = self._is_better(
                    solution, populations[existing_solution_id]
                )

        if should_replace:
            if feature_key not in island_feature_maps_key:
                # New cell occupation
                logger.info("New MAP-Elites cell occupied: %s", feature_coords)
                # Check coverage milestone
                total_possible_cells = self.feature_bins ** len(self.feature_dimensions)
                feature_map_len = self.redis.hlen(island_feature_maps_key)
                coverage = (feature_map_len + 1) / total_possible_cells
                if coverage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    logger.info(
                        "MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                        coverage * 100,
                        self.redis.hlen(island_feature_maps_key) + 1,
                        total_possible_cells,
                    )
            else:
                # Cell replacement - existing program being replaced
                if existing_solution_id in populations:
                    existing_solution = populations[existing_solution_id]
                    new_fitness = solution.score
                    existing_fitness = existing_solution.score
                    logger.info(
                        "MAP-Elites cell improved: %s (fitness: %.3f -> %.3f)",
                        feature_coords,
                        existing_fitness,
                        new_fitness,
                    )

                    # use MAP-Elites to manage archive
                    if existing_solution_id in self.redis.smembers(self.elites_key):
                        self.redis.srem(self.elites_key, existing_solution_id)
                        self.redis.sadd(self.elites_key, solution.solution_id)

            self.redis.hset(island_feature_maps_key, feature_key, solution.solution_id)
        return json.dumps(feature_coords, ensure_ascii=False)

    def _update_island(self, solution: Solution) -> None:
        """
        Update the island of the given solution based on its island_id.

        Args:
            solution: Solution to update island for.
        """
        island_id = solution.island_id

        with self._island_locks[island_id]:
            island_key = f"{self.islands_key}:{island_id}"
            self.redis.sadd(island_key, solution.solution_id)

        logger.debug(
            f"Solution {solution.solution_id} assigned to island {solution.island_id}"
        )

    def _update_elites(self, solution: Solution) -> None:
        """
        Update the elite archive with the new solution. Only better programs are added.

        Args:
            solution: Solution to consider for elite archive.
        """
        # If elites not full, add program
        if self.redis.scard(self.elites_key) < self.elite_archive_size:
            self.redis.sadd(self.elites_key, solution.solution_id)
            return

        # Clean up stale references and get valid archive programs
        valid_elites_solutions = []
        stale_ids = []

        populations = self.redis.hgetall(self.populations_key)
        for pid in self.redis.smembers(self.elites_key):
            if pid in populations:
                valid_solution = Solution.from_dict(
                    json.loads(populations[pid].decode("utf-8"))
                )
                valid_elites_solutions.append(valid_solution)
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.redis.srem(self.elites_key, stale_id)
            logger.debug(f"Removing stale solution {stale_id} from elites")

        # If archive is now not full after cleanup, just add the new program
        if self.redis.scard(self.elites_key) < self.elite_archive_size:
            self.redis.sadd(self.elites_key, solution.solution_id)
            return

        # Find worst program among valid programs
        if valid_elites_solutions:
            worst_solution = min(valid_elites_solutions, key=lambda s: s.score)

            # Replace if new program is better
            if self._is_better(solution, worst_solution):
                self.redis.srem(self.elites_key, worst_solution.solution_id)
                self.redis.sadd(self.elites_key, solution.solution_id)
        else:
            # No valid programs in archive, just add the new one
            self.redis.sadd(self.elites_key, solution.solution_id)

    def _enforce_population_limit(self, exclude_solution_id: str = None) -> None:
        """
        Enforce population size limit by removing the worst solutions using heap.

        Args:
            exclude_solution_id: Solution ID to protect from removal
        """
        population_len = self.redis.hlen(self.populations_key)
        if population_len <= self.population_size:
            return

        num_to_remove = population_len - self.population_size
        logger.debug(f"Removing {num_to_remove} solutions to enforce population limit")

        protected_ids = {
            self.redis.hget(self.metadata_key, "best_solution_id"),
            exclude_solution_id,
        } - {None}

        # Sort solutions by score (ascending) to remove the worst ones first
        populations = self.redis.hgetall(self.populations_key)
        solutions = [
            Solution.from_dict(json.loads(s.decode("utf-8")))
            for s in populations.values()
            if Solution.from_dict(json.loads(s.decode("utf-8"))).solution_id
            not in protected_ids
        ]
        solutions_sorted = sorted(solutions, key=lambda s: s.score)
        solution_ids_to_remove = {
            s.solution_id for s in solutions_sorted[:num_to_remove]
        }

        with self._lock:
            for sid in solution_ids_to_remove:
                self.redis.hdel(self.populations_key, sid)

            # Remove from feature map
            for island_idx in range(self.num_islands):
                island_map_key = f"{self.island_feature_maps_key}:{island_idx}"
                island_map = self.redis.hgetall(island_map_key)
                keys_to_remove = [
                    key
                    for key, sid in island_map.items()
                    if sid in solution_ids_to_remove
                ]
                for key in keys_to_remove:
                    self.redis.hdel(island_map_key, key)

            # Remove from islands and elites using set difference
            for sid in solution_ids_to_remove:
                for i in range(self.num_islands):
                    island_key = f"{self.islands_key}:{i}"
                    self.redis.srem(island_key, sid)

                self.redis.srem(self.elites_key, sid)

        logger.debug(f"Removed solutions: {sorted(solution_ids_to_remove)[:5]}...")

        # Clean up stale references
        self._cleanup_stale_island_bests()

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best solution references

        Cleans up references to solutions that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        island_best_solution: list[str] = [
            self.redis.hget(f"{self.islands_key}:{i}:best", "best_solution_id")
            for i in range(self.num_islands)
        ]

        for i, best_id in enumerate(island_best_solution):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.redis.hgetall(self.populations_key):
                    logger.debug(
                        f"Clearing stale island {i} best solution {best_id} (solution deleted)"
                    )
                    should_clear = True
                # Check if program still exists in island
                elif best_id not in self.redis.smembers(f"{self.islands_key}:{i}"):
                    logger.debug(
                        f"Clearing stale island {i} best solution {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.redis.hset(
                        f"{self.islands_key}:{i}:best",
                        "best_solution_id",
                        "",
                    )
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale island best solution references"
            )

            # Recalculate best programs for islands that were cleared
            island_best_solution = [
                self.redis.hget(f"{self.islands_key}:{i}:best", "best_solution_id")
                for i in range(self.num_islands)
            ]

            for i, best_id in enumerate(island_best_solution):
                if best_id is None and self.redis.scard(f"{self.islands_key}:{i}") > 0:
                    # Find new best program for this island
                    island_solutions = [
                        Solution.from_dict(
                            json.loads(
                                self.redis.hget(self.populations_key, v).decode("utf-8")
                            )
                        )
                        for v in self.redis.smembers(f"{self.islands_key}:{i}")
                    ]
                    if island_solutions:
                        # Sort by fitness and update
                        best_solution = max(
                            island_solutions,
                            key=lambda s: s.score,
                        )
                        self.redis.hset(
                            f"{self.islands_key}:{i}:best",
                            "best_solution_id",
                            best_solution.solution_id,
                        )
                        logger.debug(
                            f"Recalculated island {i} best solution: {best_solution.solution_id}"
                        )

    def _update_best_solution(self, solution: Solution) -> None:
        """
        Update the best solution tracking

        Args:
            solution: The solution to consider as best
        """
        best_sid = self.redis.hget(self.metadata_key, "best_solution_id")
        if best_sid is None:
            self.redis.hset(self.metadata_key, "best_solution_id", solution.solution_id)
            logger.debug(f"Set initial best solution to {solution.solution_id}")
            return

        # Check if previous best exists
        populations = self.redis.hgetall(self.populations_key)
        if best_sid not in populations:
            logger.debug(f"Previous best solution {best_sid} no longer exists")
            self.redis.hset(self.metadata_key, "best_solution_id", solution.solution_id)
            return

        current_best = Solution.from_dict(
            json.loads(populations[best_sid].decode("utf-8"))
        )
        if self._is_better(solution, current_best):
            old_id = best_sid.decode("utf-8")
            self.redis.hset(self.metadata_key, "best_solution_id", solution.solution_id)

            if solution.score is not None and current_best.score is not None:
                logger.info(  # Changed from debug to info for important events
                    f"New best solution {solution.solution_id} replaces {old_id} "
                    f"(score: {current_best.score:.4f} → {solution.score:.4f})"
                )

    def _update_island_best_solution(self, solution: Solution, island_id: int) -> None:
        """
        Update island's best solution tracking

        Args:
            solution: The solution to consider as best
            island_id: Island ID to update
        """
        if island_id >= self.num_islands:
            logger.warning(f"Invalid island_id {island_id}")
            return

        best_key = f"{self.islands_key}:{island_id}:best"
        best_sid = self.redis.hget(best_key, "best_solution_id")

        if best_sid is None:
            # If no best solution exists, set this one as best
            logger.info(  # Changed from debug to info for important events
                f"Set initial island {island_id} best solution to {solution.solution_id}"
            )
            self.redis.hset(best_key, "best_solution_id", solution.solution_id)
            return

        # Check if previous best exists
        populations = self.redis.hgetall(self.populations_key)
        if best_sid not in populations:
            logger.info(  # Changed from debug to info for important events
                f"Previous island {island_id} best solution {best_sid} no longer exists"
            )
            self.redis.hset(best_key, "best_solution_id", solution.solution_id)
            return

        current_best = Solution.from_dict(
            json.loads(populations[best_sid].decode("utf-8"))
        )
        if self._is_better(solution, current_best):
            old_id = best_sid.decode("utf-8")
            self.redis.hset(best_key, "best_solution_id", solution.solution_id)
            if solution.score is not None and current_best.score is not None:
                logger.info(  # Changed from debug to info for important events
                    f"New island {island_id} best solution {solution.solution_id} replaces {old_id} "
                    f"(score: {current_best.score:.4f} → {solution.score:.4f})"
                )

    async def _check_migration(self) -> None:
        """
        Adapted from algorithmicsuperintelligence/openevolve (Apache-2.0 License)
        Original source: https://github.com/algorithmicsuperintelligence/openevolve/blob/a7428efeb5a30b7968975f182d5fb7060b36e978/openevolve/database.py#L1755

        Enhanced migration with adaptive triggering and targeted transfer.
        """  # Adaptive migration trigger based on island diversity
        should_migrate = False

        # Get current generation from Redis
        max_island_capacity = max(
            [
                self.redis.scard(f"{self.islands_key}:{i}")
                for i in range(self.num_islands)
            ]
        )
        last_migration = int(
            self.redis.hget(self.metadata_key, "last_migration_generation").decode(
                "utf-8"
            )
            or 0
        )

        # Check if should migrate based on interval or high variance
        if max_island_capacity - last_migration >= self.migration_interval:
            should_migrate = True

        if not should_migrate or self.num_islands < 2:
            return

        logger.info("Performing adaptive migration between islands in Redis")

        for src_island in range(self.num_islands):
            island_key = f"{self.islands_key}:{src_island}"
            solution_ids = list(self.redis.smembers(island_key))
            if not solution_ids:
                continue

            island_solutions = [
                Solution.from_dict(json.loads(s.decode("utf-8")))
                for s in self.redis.hmget(self.populations_key, solution_ids)
                if s
            ]

            if not island_solutions:
                continue

            island_solutions.sort(key=lambda s: s.score, reverse=True)

            num_to_migrate = max(1, int(len(island_solutions) * self.migration_rate))
            migrants = island_solutions[:num_to_migrate]

            taget_islands = [
                (src_island + 1) % self.num_islands,
                (src_island - 1) % self.num_islands,
            ]

            for migrant in migrants:
                if migrant.metadata.get("migrated", False):
                    continue
                for target_island in taget_islands:
                    target_island_key = f"{self.islands_key}:{target_island}"
                    target_island_solutions_ids = list(
                        self.redis.smembers(target_island_key)
                    )
                    target_island_solutions = [
                        Solution.from_dict(json.loads(s.decode("utf-8")))
                        for s in self.redis.hmget(
                            self.populations_key, target_island_solutions_ids
                        )
                    ]
                    has_duplicate_code = any(
                        s.solution == migrant.solution for s in target_island_solutions
                    )
                    if has_duplicate_code:
                        logger.debug(
                            f"Skipping migration of {migrant.solution_id} to \
                                island {target_island} due to duplicate code"
                        )
                        continue

                    migrant_copy = Solution(
                        score=migrant.score,
                        solution=migrant.solution,
                        parent_id=migrant.parent_id,
                        generation=migrant.generation,
                        solution_id=f"{migrant.solution_id}_migrated_{target_island}",
                        generate_plan=migrant.generate_plan,
                        island_id=target_island,
                        iteration=migrant.iteration,
                        sample_weight=migrant.sample_weight,
                        evaluation=migrant.evaluation,
                        metadata={**migrant.metadata, "migrated": True},
                    )
                    migrant_copy_json = json.dumps(migrant_copy.to_dict(), ensure_ascii=False)
                    self.redis.hset(
                        self.populations_key,
                        migrant_copy.solution_id,
                        migrant_copy_json,
                    )
                    self.redis.hset(
                        self.solutions_key, migrant_copy.solution_id, migrant_copy_json
                    )
                    self.redis.sadd(target_island_key, migrant_copy.solution_id)
                    self._update_island_best_solution(migrant_copy, target_island)

        # Update last migration generation
        self.redis.hset(
            self.metadata_key, "last_migration_generation", max_island_capacity
        )

        logger.info(f"Migration completed at generation {max_island_capacity}")
