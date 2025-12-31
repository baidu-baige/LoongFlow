#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provide factory pattern for evolution memory.
"""

import logging
from typing import Optional

from agentsdk.memory.evolution.base_memory import Solution

logger = logging.getLogger(__name__)

from .redis_memory import RedisMemory
from .in_memory import InMemory


class MemoryFactory:
    """Factory class that provides unified interface for different memory implementations."""

    def __init__(
        self,
        config=None,
        storage_type: str = "in_memory",
        redis_url: str = "redis://localhost:6379/0",
        **kwargs,
    ):
        """Initialize memory factory.

        Args:
            config: Configuration dict containing all parameters
            storage_type: 'in_memory' or 'redis' (overrides config if provided)
            redis_url: Redis connection URL (overrides config if provided)
            **kwargs: Additional arguments passed to underlying memory implementation
        """  # If config is provided, use it as base parameters
        if config is None:
            config = {}
        if config is not None and isinstance(config, dict):
            self._storage_type = config.get("storage_type", storage_type)
            self._redis_url = config.get("redis_url", redis_url)
            if "storage_type" in config:
                del config["storage_type"]
            if "redis_url" in config:
                del config["redis_url"]
            self._kwargs = {**config, **kwargs}  # Merge config and kwargs
        else:
            self._storage_type = storage_type
            self._redis_url = redis_url
            self._kwargs = kwargs

        self._memory: Optional[InMemory | RedisMemory] = None

        # Initialize with specified storage type
        self._init_memory()

    def _init_memory(self):
        """Initialize the underlying memory implementation"""
        logger.debug(f"Initializing memory with type: {self._storage_type}")

        if self._storage_type == "redis":
            self._memory = RedisMemory(redis_url=self._redis_url, **self._kwargs)
        else:
            self._memory = InMemory(**self._kwargs)

    async def add_solution(self, solution: Solution) -> str:
        """
        Add a new solution to memory.
        Args:
            solution: Solution object to add
        Returns:
            solution_id: ID of the added solution
        """
        return await self._memory.add_solution(solution)

    def get_solutions(self, solution_ids: list[str]):
        """
        Retrieve solutions by their IDs.
        Args:
            solution_ids: List of solution IDs to retrieve
        Returns:
            List of solution objects
        """
        return self._memory.get_solutions(solution_ids)

    def list_solutions(self, filter_type: str = "asc", limit: int = None):
        """
        List solutions with optional filtering and limit.
        Args:
            filter_type: 'asc' or 'desc' for sorting by timestamp
            limit: Maximum number of solutions to return
        Returns:
            List of solution objects
        """
        return self._memory.list_solutions(filter_type, limit)

    def get_best_solutions(self, island_id: int = None, top_k: int = None):
        """
        Get the best solutions globally or per island.
        Args:
            island_id: Specific island ID to filter by
            top_k: Number of top solutions to return
        Returns:
            List of the best solution objects
        """
        return self._memory.get_best_solutions(island_id, top_k)

    def sample(self, island_id: Optional[int] = None, exploration_rate: float = 0.2) -> Solution:
        """
        Sample a solution from memory.
        Returns:
            A randomly sampled solution object
        """
        return self._memory.sample(island_id, exploration_rate)

    async def save_checkpoint(self, path=None, tag=None):
        """
        Create a checkpoint of the current memory state.
        Args:
            path: Path to save the checkpoint
            tag: Tag to append to the filename
        """
        return await self._memory.save_checkpoint(path, tag)

    def load_checkpoint(self, path: str):
        """
        Load memory state from a checkpoint.
        Args:
            path: Path to load the checkpoint from
        """
        return self._memory.load_checkpoint(path)

    def memory_status(self, island_id: Optional[int] = None) -> dict:
        """
        Return the status of the memory
        """
        return self._memory.memory_status(island_id)

    async def update_solution(self, solution_id: str, **kwargs) -> str:
        """
        Update an existing solution in memory.
        Args:
            solution_id: Solution ID
            **kwargs: Keyword arguments for updating the solution

        Returns:
            solution_id: ID of the updated solution
        """
        return await self._memory.update_solution(solution_id, **kwargs)

    def get_parents_by_child_id(self, child_id: str, parent_cnt: int):
        """
        Get parents of a given solution based on its ID.
        Args:
            child_id: ID of the solution whose parents are needed
            parent_cnt: Number of parents to fetch
        Returns:
            List of parent solution objects
        """
        return self._memory.get_parents_by_child_id(child_id, parent_cnt)

    def get_childs_by_parent_id(self, parent_id: str, child_cnt: int):
        """
        Get children of a given solution based on its ID.
        Args:
            parent_id: ID of the solution whose children are needed
            child_cnt: Number of children to fetch
        Returns:
            List of child solution objects
        """
        return self._memory.get_childs_by_parent_id(parent_id, child_cnt)

    @property
    def storage_type(self):
        """Get current storage type"""
        return self._storage_type

    def __getattr__(self, name):
        """Delegate any other attributes to underlying memory"""
        return getattr(self._memory, name)
