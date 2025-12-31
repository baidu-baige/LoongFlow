#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides register implementation for planner、executor、summary
"""

import abc
import inspect
from typing import Any

from agentsdk.message import Message

PLANNER = "planner"
EXECUTOR = "executor"
SUMMARY = "summary"

# Planner registry to hold implementations
_planner_registry = {}

# Executor registry to hold implementations
_executor_registry = {}

# Summary registry to hold implementations
_summary_registry = {}


class Worker(abc.ABC):
    """Agent interface"""

    @abc.abstractmethod
    async def run(self, context: Any, message: Message | None) -> Message:
        """
        Args:
            context(Any): Context of evolve information
            message (Message): Message object. Planner - Executor - Summary inner transfer

        Returns:
            Message.
        """
        pass


def register_worker(name: str, phase: str, worker_class: type):
    """Register an agent implementation

    Args:
        name (str): The name to identify the worker.
        phase (str): The phase to identify the worker.
        worker_class (type): The class of the worker that extends the Woker interface.
    """
    if not issubclass(worker_class, Worker):
        raise ValueError(f"{worker_class.__name__} must be a subclass of Worker.")

    phase = phase.lower()
    if phase == PLANNER:
        _planner_registry[name] = worker_class
    elif phase == EXECUTOR:
        _executor_registry[name] = worker_class
    elif phase == SUMMARY:
        _summary_registry[name] = worker_class
    else:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of [{PLANNER}, {EXECUTOR}, {SUMMARY}]."
        )


def get_worker(name: str, phase: str, **kwargs) -> Worker:
    """
    Retrieve a registered worker (Planner, Executor, or Summary) by name.

    This function looks up the worker in the corresponding phase registry
    and instantiates it using the provided keyword arguments. The
    keyword arguments should match the constructor signature of the
    registered worker class.

    Args:
        name (str): The name of the worker to retrieve.
        phase (str): The phase of the worker. Must be one of:
            - "planner"
            - "executor"
            - "summary"
        **kwargs: Arbitrary keyword arguments that will be passed to the
            worker's constructor. Common examples include:
            - config: configuration object for the worker
            - db: database or storage object (planner/summary)
            - evaluator: evaluator instance (executor)

    Returns:
        Worker: The worker implementation.

    Raises:
        KeyError: If the agent is not registered.
    """
    _worker_class = None
    phase = phase.lower()
    if phase == PLANNER:
        _worker_class = _planner_registry[name]
    elif phase == EXECUTOR:
        _worker_class = _executor_registry[name]
    elif phase == SUMMARY:
        _worker_class = _summary_registry[name]

    if _worker_class is None:
        raise KeyError(f"Worker '{name}' not found in Phase '{phase}'.")

    sig = inspect.signature(_worker_class.__init__)

    class_params = set(sig.parameters.keys())
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in class_params
    }

    return _worker_class(**filtered_kwargs)
