#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides Planner implementation
"""

from typing import Any

from agentsdk.message import Message
from evolux.evolve.register import get_worker


class Planner:
    """Planner Class"""

    def __init__(self, planner_name: str, config: Any, db: Any):
        self.planner = get_worker(
            name=planner_name, phase="planner", config=config, db=db
        )

    async def run(self, context: Any, message: Message | None) -> Message:
        """
        Run the planner

        Args:
            context (Any): Context
            message (Message | None): Message

        Returns:
            Message: Message
        """
        if self.planner is None:
            raise ValueError("Planner has not been registered")
        if not hasattr(self.planner, "run"):
            raise ValueError("Planner does not have 'run' method")
        return await self.planner.run(context, message)
