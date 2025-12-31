#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides Executor implementation
"""

from typing import Any

from agentsdk.message import Message
from evolux.evolve.register import get_worker


class Executor:
    """Executor class"""

    def __init__(self, executor_name: str, config: Any, evaluator: Any, db: Any):
        self.executor = get_worker(executor_name, "executor", config=config, evaluator=evaluator, db=db)

    async def run(self, context: Any, message: Message | None) -> Message:
        """
        Run the executor

        Args:
            context (Any): Context
            message (Message | None): Message

        Returns:
            Message: Message
        """
        if self.executor is None:
            raise ValueError("Executor has not been registered")
        if not hasattr(self.executor, "run"):
            raise ValueError("Executor does not have 'run' method")
        return await self.executor.run(context, message)
