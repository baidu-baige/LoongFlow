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
from typing import Dict, Optional, Set

from .base_memory import EvolveMemory, Solution
from .boltzmann import select_parents_with_dynamic_temperature

logger = logging.getLogger(__name__)


class InMemory(EvolveMemory):
    """In-memory implementation of Evolution Memory storage.

    Features:
    - Thread-safe operations
    - Memory-efficient data structures
    - Optimized for high-frequency access
    """

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
    ):
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
        self.best_solution_id: str = ""
        self.last_iteration: int = 0
        self.current_island: int = 0
        self.current_island_counter: int = 0
        self.solutions_per_island: int = max(1, population_size // num_islands)
        self.last_migration_generation: int = 0

        # Calculate feature_bins if not provided
        if feature_bins is None:
            self.feature_bins: int = int(
                pow(self.elite_archive_size, 1 / len(self.feature_dimensions)) + 0.99
            )
        else:
            self.feature_bins: int = feature_bins

        # Thread-safe data structures
        # Optimized data structures
        self.islands: list[Set[str]] = [set() for _ in range(num_islands)]
        self.island_best_solution: list[Optional[str]] = [None] * num_islands
        self.island_capacity: list[int] = [0] * self.num_islands
        self.island_feature_maps: list[Dict[str, str]] = [
            {} for _ in range(num_islands)
        ]
        self.elites: Set[str] = set()
        self.feature_stats: Dict[str, Dict[str, float | list[float]]] = {}
        self.feature_bins_per_dim: Dict[str, int] = {
            dim: self.feature_bins for dim in self.feature_dimensions
        }
        self.feature_scaling_method: str = feature_scaling_method

        self.diversity_cache: Dict[str, Dict[str, float]] = {}
        self.diversity_reference_set: list[str] = []
        self.solutions: Dict[str, Solution] = {}
        self.populations: Dict[str, Solution] = {}

        self.last_migration_generation: int = 0  # Initialize missing attribute

        # Optimized locking with reentrant locks and context managers
        self._lock = threading.RLock()
        self._island_locks: Dict[int, threading.RLock] = {
            i: threading.RLock() for i in range(num_islands)
        }

    async def add_solution(self, solution: Solution) -> str:
        """
        Add solution to memory with optimized workflow.

        Args:
            solution: Solution to add

        Returns:
            Solution ID assigned to the solution
        """
        if not isinstance(solution, Solution):
            raise ValueError("solution must be an instance of Solution")
        with self._lock:
            self._prepare_solution(solution)
            self.solutions[solution.solution_id] = solution

            if not solution.score:
                logger.warning(
                    f"WARNING: No score found for solution {solution.solution_id}. Skipping."
                )
                return solution.solution_id

            # Process solution through memory components
            map_elites_feature = self._calculate_MAP_Elites(solution)
            solution.metadata["MAP_Elite_feature"] = map_elites_feature
            self.populations[solution.solution_id] = solution
            self.solutions[solution.solution_id] = solution

            self._update_island(solution)
            self._update_elites(solution)

            # Enforce limits before final updates
            self._enforce_population_limit(exclude_solution_id=solution.solution_id)

            # Update tracking
            self._update_best_solution(solution)
            self._update_island_best_solution(solution, solution.island_id)

            # Check Migration
            await self._check_migration()

            # Final update of island capacity after migration
            self.island_capacity[solution.island_id] = len(
                self.islands[solution.island_id]
            )

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
        if solution_id not in self.solutions:
            raise ValueError("solution_id does not exist in memory")
        for k, v in kwargs.items():
            if k == "island_id" or k == "parent_id":
                raise ValueError("Cannot update island_id or parent_id directly")

        solution = self.solutions[solution_id]
        updated_solution = solution.copy()
        updated_solution.update(**kwargs)

        with self._lock:
            self.solutions[solution_id] = updated_solution
            self.populations[solution_id] = updated_solution

        return solution_id

    def get_solutions(self, solution_ids: list[str] = None) -> list[Solution]:
        """
        Get solutions by their IDs.

        Args:
            solution_ids: list of solution IDs to retrieve. If None, returns error.

        Returns:
            list of Solution objects.
        """
        with self._lock:
            if solution_ids is None:
                raise ValueError("No solution IDs provided")

            return [
                self.solutions[sid] for sid in solution_ids if sid in self.solutions
            ]

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
            # Use generator to avoid full list copy
            solutions = (s for s in self.solutions.values())

            # Optimized sorting with key function caching
            key_func = attrgetter("timestamp")
            if limit is None or limit >= len(self.solutions):
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
            # Use generator expression for memory efficiency
            solutions = (
                (
                    self.populations[sid]
                    for sid in self.islands[island_id]
                    if sid in self.populations
                )
                if island_id is not None
                else self.populations.values()
            )

            return heapq.nlargest(top_k, solutions, key=attrgetter("score"))

    def sample(
        self, island_id: Optional[int] = None, exploration_rate: float = 0.2
    ) -> Solution | None:
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
        solutions = list(self.populations.values())

        if island_id is not None:
            island_sids = list(self.islands[island_id])
            solutions = [
                self.populations[sid] for sid in island_sids if sid in self.populations
            ]

        elites = []
        for elite_id in self.elites:
            elite = self.populations[elite_id]
            elites.append(elite)

        if not solutions:
            return None

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

        logger.info(f"Checkpointing memory to {checkpoint_path}")

        # Save solutions
        for solution in self.solutions.values():
            solutions_path = os.path.join(checkpoint_path, "solutions")
            os.makedirs(solutions_path, exist_ok=True)

            # Save solution
            solution_dict = solution.to_dict()
            solution_path = os.path.join(solutions_path, f"{solution.solution_id}.json")
            with open(solution_path, "w") as f:
                json.dump(solution_dict, f, indent=4)

        # Save metadata
        metadata = {
            "total_generated_solutions": len(self.solutions),
            "total_valid_solutions": len(self.populations),
            "island_feature_map": self.island_feature_maps,
            "islands": [list(island) for island in self.islands],
            "elites": list(self.elites),
            "best_solution_id": self.best_solution_id,
            "island_best_solution": [sid for sid in self.island_best_solution],
            "last_iteration": self.last_iteration,
            "current_island": self.current_island,
            "island_capacity": [length for length in self.island_capacity],
            "last_migration_generation": self.last_migration_generation,
            "feature_stats": self._serialize_feature_stats(self.feature_stats),
        }

        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(
            f"Saved checkpoint with {len(self.populations)} programs to {checkpoint_path}"
        )

        # Save best solution found so far
        if self.best_solution_id:
            best_solution = self.populations.get(self.best_solution_id)
        else:
            best_solutions = self.get_best_solutions()
            best_solution = best_solutions[0] if len(best_solutions) > 0 else None

        if best_solution:
            best_solution_path = os.path.join(checkpoint_path, "best_solution.json")
            with open(best_solution_path, "w") as f:
                json.dump(best_solution.to_dict(), f, indent=4)

        logger.info(f"Saved checkpoint with tag {tag} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the memory state from disk.

        Args:
            checkpoint_path: Directory path where the checkpoint is stored.
        Returns:
            None: This method does not return anything but loads checkpoint data into memory.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path {checkpoint_path} does not exist."
            )

        with self._lock:
            logger.info(f"Loading checkpoint from {checkpoint_path}")

            with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

            self.island_feature_maps = metadata.get(
                "island_feature_map", [{} for _ in range(self.num_islands)]
            )
            saved_islands = metadata.get("islands", [])
            self.elites = set(metadata.get("elites", []))
            self.best_solution_id = metadata.get("best_solution_id")
            self.island_best_solution = metadata.get(
                "island_best_solution", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_capacity = metadata.get(
                "island_capacity", [0] * len(saved_islands)
            )
            self.last_migration_generation = metadata.get("last_migration_generation")
            self.feature_stats = self._deserialize_feature_stats(
                metadata.get("feature_stats", {})
            )

            # Load solutions
            solutions_path = os.path.join(checkpoint_path, "solutions")
            for file_name in os.listdir(solutions_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(solutions_path, file_name)
                    try:
                        with open(file_path, "r") as f:
                            solution_dict = json.load(f)
                            solution = Solution.from_dict(solution_dict)
                            self.populations[solution.solution_id] = solution
                            self.solutions[solution.solution_id] = solution
                    except Exception as e:
                        logger.error(
                            f"Failed to load solution from {file_path}: {str(e)}"
                        )
                        raise e

            self._reconstruct_islands(saved_islands)

            if len(self.island_capacity) != len(self.islands):
                self.island_capacity = [len(island) for island in self.islands]

            if len(self.island_best_solution) != len(self.islands):
                self.island_best_solution = [None] * len(self.islands)

    def memory_status(self, island_id: int = None) -> dict:
        """Return the status of the memory"""
        top_3_solutions = self.get_best_solutions(top_k=3)
        top_3_iterations = [
            top_3_solutions[i].iteration
            for i in range(min(3, len(top_3_solutions)))
            if top_3_solutions[i]
        ]

        best_solution = top_3_solutions[0] if len(top_3_solutions) > 0 else None
        population_scores = [
            solution.score for solution in self.populations.values() if solution
        ]
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
                "current_iteration": self.last_iteration,
                "is_full": len(population_scores) == self.population_size,
                "top_3_iterations": top_3_iterations,
                "best_score": best_solution.score if best_solution else 0,
                "best_iteration": best_solution.iteration if best_solution else 0,
                "avg_score": round(avg_score, 6),
                "better_ratio": round(better_ratio, 2),
            },
        }

        if island_id and len(self.islands[island_id]) > 0:
            island_metrics = {"island_id": island_id}
            island_top_3_solutions = self.get_best_solutions(
                island_id=island_id, top_k=3
            )

            if len(island_top_3_solutions) < 1:
                return result

            island_top_3_iterations = [
                island_top_3_solutions[i].iteration
                for i in range(min(3, len(island_top_3_solutions)))
                if island_top_3_solutions[i]
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
                self.populations.get(solution_id)
                for solution_id in self.islands[island_id]
            ]
            island_scores = [
                solution.score for solution in island_solutions if solution
            ]
            island_metrics["avg_score"] = round(
                (
                    sum(island_scores) / len(island_scores)
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
            island_feature_map = self.island_feature_maps[island_id]
            total_possible_cells = self.feature_bins ** len(self.feature_dimensions)
            coverage = (len(island_feature_map) + 1) / total_possible_cells
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
            if child_id not in self.solutions:
                raise ValueError(f"Child solution with id '{child_id}' not found.")

            parents = []
            while len(parents) < parent_cnt:
                child = self.solutions.get(child_id, None)
                if child is None:
                    break

                parent_id = child.parent_id
                parent = self.solutions.get(parent_id, None)
                if parent is None:
                    break

                parents.append(parent)
                child_id = parent_id

            return parents

    def get_childs_by_parent_id(self, parent_id: str, child_cnt: int) -> list[Solution]:
        """
        Get childs by parent id

        Args:
            parent_id (int): Parent solution id
            child_cnt (int): Number of children to retrieve

        Returns:
            list[Solution]: Child solutions
        """
        with self._lock:
            if parent_id not in self.solutions:
                raise ValueError(f"Parent solution with id '{parent_id}' not found.")

            childs = []
            parent = self.solutions.get(parent_id, None)
            if parent is None:
                return childs

            island_solutions = [
                self.solutions[sid] for sid in self.islands[parent.island_id]
            ]
            for child_solution in island_solutions:
                if child_solution.parent_id == parent.solution_id:
                    childs.append(child_solution)
                    if len(childs) == child_cnt:
                        break
            return childs

    def _reconstruct_islands(self, saved_islands: list[list[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island solution ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_solutions = []

        # Restore island assignments
        for island_idx, solution_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for solution_id in solution_ids:
                if solution_id in self.populations:
                    # Solution exists, add to island
                    self.islands[island_idx].add(solution_id)
                else:
                    # Solution missing, track it
                    missing_solutions.append((island_idx, solution_id))

        # Clean up archive - remove missing solutions
        self.elites = {sid for sid in self.elites if sid in self.populations}

        # Clean up island_feature_maps - remove missing programs
        feature_keys_to_remove = []
        for island_idx, island_map in enumerate(self.island_feature_maps):
            island_keys_to_remove = []
            for key, solution_id in island_map.items():
                if solution_id not in self.populations:
                    island_keys_to_remove.append(key)
                    feature_keys_to_remove.append((island_idx, key))
            for key in island_keys_to_remove:
                del island_map[key]

        # Clean up island best solutions - remove stale references
        self._cleanup_stale_island_bests()

        # Check best solution
        if self.best_solution_id and self.best_solution_id not in self.populations:
            logger.warning(
                f"Best solution {self.best_solution_id} not found, will recalculate"
            )
            self.best_solution_id = None

        # If we have solutions but no island assignments, distribute them
        if self.populations and sum(len(island) for island in self.islands) == 0:
            logger.info(
                "No island assignments found, distributing programs across islands"
            )
            solution_ids = list(self.populations.keys())
            for i, solution_id in enumerate(solution_ids):
                island_idx = i % len(self.islands)
                self.islands[island_idx].add(solution_id)

    def _prepare_solution(self, solution: Solution) -> None:
        """Prepare solution for addition by setting IDs and iteration."""
        if not solution.iteration:
            self.last_iteration += 1
            solution.iteration = self.last_iteration

        self.last_iteration = max(self.last_iteration, solution.iteration)

        if not solution.solution_id:
            solution.solution_id = uuid.uuid4().hex[:8]

        # if island_id already set, respect it
        if solution.island_id:
            return

        # Assign to island with no programs first
        island_without_program = []
        for i in range(self.num_islands):
            if len(self.islands[i]) == 0:
                island_without_program.append(i)

        if island_without_program:
            solution.island_id = min(island_without_program)
            return

        # Inherit island from parent if available
        if solution.parent_id:
            parent_solution = self.solutions.get(solution.parent_id)
            if parent_solution:
                solution.generation = parent_solution.generation + 1
                solution.island_id = parent_solution.island_id
                return

        # Finally, Round-robin assignment
        solution.island_id = self.current_island
        self.current_island_counter += 1
        if self.current_island_counter >= self.solutions_per_island:
            self.current_island = (self.current_island + 1) % self.num_islands
            self.current_island_counter = 0

    def _calculate_MAP_Elites(self, solution: Solution) -> str:
        """
        Adapted from algorithmicsuperintelligence/openevolve (Apache-2.0 License)
        Original source: https://github.com/algorithmicsuperintelligence/openevolve/blob/a7428efeb5a30b7968975f182d5fb7060b36e978/openevolve/database.py#L221

        Update MAP-Elites feature map with the new solution.

        Args:
            solution: The solution to add to the MAP-Elites grid
        """
        feature_coords, self.diversity_reference_set = self._calculate_feature_coords(
            solution,
            self.populations,
            self.feature_stats,
            self.feature_bins_per_dim,
            self.feature_bins,
            self.feature_dimensions,
            self.diversity_cache,
            self.diversity_reference_set,
        )

        logger.debug(
            "Calculated feature coords for %s: %s",
            solution.solution_id[:6],
            feature_coords,
        )

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        island_feature_map = self.island_feature_maps[solution.island_id]
        should_replace = feature_key not in island_feature_map

        logger.debug(
            "Feature key %s for %s, replace: %s",
            feature_key[:10],
            solution.solution_id[:6],
            should_replace,
        )

        if not should_replace:
            # Check if the existing program still exists before comparing
            existing_solution_id = island_feature_map[feature_key]
            if existing_solution_id not in self.populations:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale solution reference {existing_solution_id} in feature map"
                )
            else:
                # Solutions exists, compare fitness
                should_replace = self._is_better(
                    solution, self.populations[existing_solution_id]
                )

        if should_replace:
            if feature_key not in island_feature_map:
                # New cell occupation
                logger.info("New MAP-Elites cell occupied: %s", feature_coords)
                # Check coverage milestone
                total_possible_cells = self.feature_bins ** len(self.feature_dimensions)
                coverage = (len(island_feature_map) + 1) / total_possible_cells
                if coverage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    logger.info(
                        "MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                        coverage * 100,
                        len(island_feature_map) + 1,
                        total_possible_cells,
                    )
            else:
                # Cell replacement - existing program being replaced
                existing_solution_id = island_feature_map[feature_key]
                if existing_solution_id in self.populations:
                    existing_solution = self.populations[existing_solution_id]
                    new_fitness = solution.score
                    existing_fitness = existing_solution.score
                    logger.info(
                        "MAP-Elites cell improved: %s (fitness: %.3f -> %.3f)",
                        feature_coords,
                        existing_fitness,
                        new_fitness,
                    )

                    # use MAP-Elites to manage archive
                    if existing_solution_id in self.elites:
                        self.elites.discard(existing_solution_id)
                        self.elites.add(solution.solution_id)

            island_feature_map[feature_key] = solution.solution_id
        return json.dumps(feature_coords, ensure_ascii=False)

    def _update_island(self, solution: Solution) -> None:
        """
        Update the island of the given solution based on its island_id.

        Args:
            solution: Solution to update island for.
        """
        island_id = solution.island_id

        with self._island_locks[island_id]:
            # Update storage
            self.populations[solution.solution_id] = solution
            self.islands[island_id].add(solution.solution_id)
            self.island_capacity[island_id] += 1

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
        if len(self.elites) < self.elite_archive_size:
            self.elites.add(solution.solution_id)
            return

        # Clean up stale references and get valid archive programs
        valid_elites_solutions = []
        stale_ids = []

        for pid in self.elites:
            if pid in self.populations:
                valid_elites_solutions.append(self.populations[pid])
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.elites.discard(stale_id)
            logger.debug(f"Removing stale solution {stale_id} from elites")

        # If archive is now not full after cleanup, just add the new program
        if len(self.elites) < self.elite_archive_size:
            self.elites.add(solution.solution_id)
            return

        # Find worst program among valid programs
        if valid_elites_solutions:
            worst_solution = min(valid_elites_solutions, key=lambda s: s.score)

            # Replace if new program is better
            if self._is_better(solution, worst_solution):
                self.elites.remove(worst_solution.solution_id)
                self.elites.add(solution.solution_id)
        else:
            # No valid programs in archive, just add the new one
            self.elites.add(solution.solution_id)

    def _enforce_population_limit(self, exclude_solution_id: str = None) -> None:
        """
        Enforce population size limit by removing the worst solutions using heap.

        Args:
            exclude_solution_id: Solution ID to protect from removal
        """
        if len(self.populations) <= self.population_size:
            return

        num_to_remove = len(self.populations) - self.population_size
        logger.debug(f"Removing {num_to_remove} solutions to enforce population limit")

        # Get protected IDs
        protected_ids = {
            self.best_solution_id,
            exclude_solution_id,
        } - {None}

        # Sort solutions by score (ascending) to remove the worst ones first
        solutions_sorted = sorted(
            [
                s
                for s in self.populations.values()
                if s.solution_id not in protected_ids
            ],
            key=lambda s: s.score,
        )
        solution_ids_to_remove = {
            s.solution_id for s in solutions_sorted[:num_to_remove]
        }

        # Bulk removal operations
        with self._lock:
            # Remove from populations
            self.populations = {
                k: v
                for k, v in self.populations.items()
                if k not in solution_ids_to_remove
            }

            # Remove from island feature map
            for island_idx, island_map in enumerate(self.island_feature_maps):
                keys_to_remove = [
                    key
                    for key, sid in island_map.items()
                    if sid in solution_ids_to_remove
                ]
                for key in keys_to_remove:
                    del island_map[key]

            # Remove from islands and elites using set difference
            for island in self.islands:
                island.difference_update(solution_ids_to_remove)

            self.elites.difference_update(solution_ids_to_remove)

        logger.debug(f"Removed solutions: {sorted(solution_ids_to_remove)[:5]}...")
        logger.info(f"Population after cleanup: {len(self.populations)}")

        # Clean up stale references
        self._cleanup_stale_island_bests()

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best solution references

        Cleans up references to solutions that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_solution):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.populations:
                    logger.debug(
                        f"Clearing stale island {i} best solution {best_id} (solution deleted)"
                    )
                    should_clear = True
                # Check if program still exists in island
                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best solution {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_solution[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale island best solution references"
            )

            # Recalculate best programs for islands that were cleared
            for i, best_id in enumerate(self.island_best_solution):
                if best_id is None and len(self.islands[i]) > 0:
                    # Find new best program for this island
                    island_solutions = [
                        self.populations[sid]
                        for sid in self.islands[i]
                        if sid in self.populations
                    ]
                    if island_solutions:
                        # Sort by fitness and update
                        best_solution = max(
                            island_solutions,
                            key=lambda s: s.score,
                        )
                        self.island_best_solution[i] = best_solution.solution_id
                        logger.debug(
                            f"Recalculated island {i} best solution: {best_solution.solution_id}"
                        )

    def _update_solution_ranking(
        self,
        solution: Solution,
        target_list: list[str],
        target_idx: int,
        solution_type: str,
    ) -> None:
        """
        Unified method to update solution rankings (best or island best)

        Args:
            solution: Solution to consider
            target_list: List of solution IDs to update
            target_idx: Index in target list to update
            solution_type: Type description for logging
        """
        current_id = target_list[target_idx]

        if current_id is None:
            target_list[target_idx] = solution.solution_id
            logger.debug(
                f"Set initial {solution_type} solution to {solution.solution_id}"
            )
            return

        if current_id not in self.populations:
            logger.debug(
                f"Previous {solution_type} solution {current_id} no longer exists"
            )
            target_list[target_idx] = solution.solution_id
            return

        current_best = self.populations[current_id]

        if self._is_better(solution, current_best):
            old_id = current_id
            target_list[target_idx] = solution.solution_id

            if solution.score is not None and current_best.score is not None:
                logger.debug(
                    f"New {solution_type} solution {solution.solution_id} replaces {old_id} "
                    f"(score: {current_best.score:.4f} → {solution.score:.4f})"
                )

    def _update_best_solution(self, solution: Solution) -> None:
        """
        Update the best solution tracking

        Args:
            solution: The solution to consider as best
        """
        if self.best_solution_id is None:
            self.best_solution_id = solution.solution_id
            logger.debug(f"Set initial best solution to {solution.solution_id}")
            return

        if self.best_solution_id not in self.populations:
            logger.debug(
                f"Previous best solution {self.best_solution_id} no longer exists"
            )
            self.best_solution_id = solution.solution_id
            return

        current_best = self.populations[self.best_solution_id]

        if self._is_better(solution, current_best):
            old_id = self.best_solution_id
            self.best_solution_id = solution.solution_id

            if solution.score is not None and current_best.score is not None:
                logger.debug(
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
            logger.warning(f"Invalid island id {island_id}")
            return

        self._update_solution_ranking(
            solution, self.island_best_solution, island_id, f"island {island_id} best"
        )

    async def _check_migration(self) -> None:
        """
        Adapted from algorithmicsuperintelligence/openevolve (Apache-2.0 License)
        Original source: https://github.com/algorithmicsuperintelligence/openevolve/blob/a7428efeb5a30b7968975f182d5fb7060b36e978/openevolve/database.py#L1755

        Enhanced migration with adaptive triggering and targeted transfer.
        """
        # Adaptive migration trigger based on island diversity
        should_migrate = False
        if (
            max(self.island_capacity) - self.last_migration_generation
            >= self.migration_interval
        ):
            should_migrate = True

        if not should_migrate or len(self.islands) < 2:
            return

        logger.info("Performing adaptive migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) <= 1:
                continue

            island_solutions = [
                self.populations[sid] for sid in island if sid in self.populations
            ]
            if not island_solutions:
                continue

            island_solutions.sort(key=lambda s: s.score, reverse=True)

            num_to_migrate = max(1, int(len(island_solutions) * self.migration_rate))
            migrants = island_solutions[:num_to_migrate]
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            for migrant in migrants:
                if migrant.metadata.get("migrated", False):
                    continue
                for target_island in target_islands:
                    target_island_programs = [
                        self.populations[sid]
                        for sid in self.islands[target_island]
                        if sid in self.populations
                    ]
                    has_duplicate_code = any(
                        s.solution == migrant.solution for s in target_island_programs
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
                        summary=migrant.summary,
                        metadata={**migrant.metadata, "migrated": True},
                    )
                    self.populations[migrant_copy.solution_id] = migrant_copy
                    self.solutions[migrant_copy.solution_id] = migrant_copy
                    self.islands[target_island].add(migrant_copy.solution_id)
                    self.island_capacity[target_island] += 1
                    self._update_island_best_solution(migrant_copy, target_island)

        self.last_migration_generation = max(self.island_capacity)
        logger.info(
            f"Migration completed at generation {self.last_migration_generation}"
        )
