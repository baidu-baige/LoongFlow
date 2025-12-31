#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database for LoongFlow evolve paradigm.
"""
from typing import Optional

from agentsdk.memory.evolution.base_memory import Solution
from agentsdk.memory.evolution.memory_factory import MemoryFactory
from evolux.evolve.context.config import DatabaseConfig


class EvolveDatabase:
    """
    Database for saving and sampling solutions during evolution

    The database is build on agentsdk evolution_memory, which implements
    a combination of MAP-Elites algorithm、boltzmann sampling and island-based
    population model to maintain solution diversity and lead the evolution process
    change better.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database with given configuration.

        Args:
            config: EvolveDatabaseConfig object containing database parameters.
        """
        self.config = config
        self._evolution_memory = MemoryFactory(config.to_dict())

    @classmethod
    def create_database(cls, config: DatabaseConfig) -> "EvolveDatabase":
        """
        Create an instance of the EvolveDatabase class using the provided configuration.

        Args:
            config: An EvolveDatabaseConfig object containing the configuration options for the database.

        Returns:
            An instance of the EvolveDatabase class configured according to the provided configuration.
        """
        return cls(config)

    def sample_solution(self, island_id: Optional[int] = None) -> dict:
        """
        Sample a solution from the database based on the given message.

        Returns:
            The sampled solution dict.
        """
        exploration_rate = self.config.exploration_rate
        # Check the last 5 iteration solutions, if there are no obviously diff, it means we stuck in local optimum
        # If in local optimum, we should increase the exploration rate to select a random solution
        previous_solutions = self._evolution_memory.list_solutions(
            filter_type="desc", limit=5
        )
        # calculate the delta of the last 5 iterations
        deltas = [
            abs(previous_solutions[i].score - previous_solutions[i + 1].score)
            for i in range(len(previous_solutions) - 1)
        ]
        # check if all the deltas are less than 0.01, medium local optimum
        if all(delta < 0.01 for delta in deltas):
            exploration_rate = exploration_rate * 2
        # check if all the deltas are less than 0.001, hard local optimum
        elif all(delta < 0.001 for delta in deltas):
            exploration_rate = exploration_rate * 4

        if exploration_rate >= 1:
            exploration_rate = 0.9

        solution = self._evolution_memory.sample(island_id, exploration_rate)
        return solution.to_dict() if solution is not None else {}

    async def add_solution(self, solution: Solution) -> str:
        """
        Add a new solution to the database.

        Args:
            solution: The new solution.

        Returns:
            The ID of the added solution.
        """
        if not isinstance(solution, Solution):
            raise ValueError("Solution is invalid")
        if solution.solution is None:
            raise ValueError("Solution is empty")
        if solution.evaluation is None:
            raise ValueError("Evaluation is empty")
        if solution.island_id is None:
            raise ValueError("Island id is empty")
        if solution.generate_plan is None:
            raise ValueError("Generate plan is empty")
        if solution.summary is None:
            raise ValueError("Summary is empty")

        return await self._evolution_memory.add_solution(solution)

    async def update_solution(self, solution_id: str, **kwargs) -> str:
        """
        Update a solution in the database.

        Args:
            solution_id: The ID of the solution to update.
            **kwargs: Keyword arguments representing the fields to update.

        Returns:
            The ID of the updated solution.

        Raises:
            ValueError: If the solution_id is not specified.
        """
        if solution_id is None:
            raise ValueError("Solution id is required.")

        return await self._evolution_memory.update_solution(
            solution_id=solution_id, **kwargs
        )

    def memory_status(self, island_id: Optional[int] = None) -> dict:
        """Get current status of the memory."""
        return self._evolution_memory.memory_status(island_id)

    async def save_checkpoint(self, checkpoint_path: str, tag: str):
        """Save the current state of the database to a file at the specified path."""
        await self._evolution_memory.save_checkpoint(checkpoint_path, tag)

    def load_checkpoint(self, checkpoint_path: str):
        """Load the saved state of the database from a file at the specified path."""
        self._evolution_memory.load_checkpoint(checkpoint_path)

    def get_parents_by_child_id(self, child_id: str, parent_cnt: int) -> list[dict]:
        """
        Get parents by child id.

        Args:
            child_id (str): Child id.
            parent_cnt (int): Parent count.

        Returns:
            List[dict]: List of parent solution dict.
        """
        solutions = self._evolution_memory.get_parents_by_child_id(child_id, parent_cnt)
        return [solution.to_dict() for solution in solutions]

    def get_childs_by_parent_id(self, parent_id: str, child_cnt: int) -> list[dict]:
        """
        Get childs by parent id.

        Args:
            parent_id (str): Parent id.
            child_cnt (int): Child count.

        Returns:
            List[dict]: List of child solution dict.
        """
        solutions = self._evolution_memory.get_childs_by_parent_id(parent_id, child_cnt)
        return [solution.to_dict() for solution in solutions]

    def get_solutions(self, solution_ids: list[str]) -> list[dict]:
        """
        Get solutions by ids.

        Args:
            solution_ids (List[str]): List of solution ids.

        Returns:
            List[dict]: List of solution dict.
        """
        solutions = self._evolution_memory.get_solutions(solution_ids)
        return [solution.to_dict() for solution in solutions]

    def get_best_solutions(
        self, island_id: Optional[int] = None, top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Get the best solutions.

        Args:
            island_id (int): Island id.
            top_k (int): Top k.

        Returns:
            List[dict]: List of solution dict.
        """
        solutions = self._evolution_memory.get_best_solutions(island_id, top_k)
        return [solution.to_dict() for solution in solutions]
