#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides Summary Implementation
"""

from typing import Any

from agentsdk.message import Message
from evolux.evolve.register import get_worker


class Summary:
    """Summary Class"""

    def __init__(self, summary_name: str, config: Any, db: Any):
        self.summary = get_worker(name=summary_name, phase="summary", config=config, db=db)

    async def run(self, context: Any, message: Message | None) -> Message:
        """
        Run Summary

        Args:
            context (Any): Context
            message (Message | None): Message

        Returns:
                Message: Message
        """
        if self.summary is None:
            raise ValueError("Summary has not been registered")
        if not hasattr(self.summary, "run"):
            raise ValueError("Summary does not have 'run' method")
        return await self.summary.run(context, message)
