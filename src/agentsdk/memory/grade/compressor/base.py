#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compressor interface
"""
from abc import ABC, abstractmethod
from typing import List

from agentsdk.message import Message


class Compressor(ABC):
    """
    An abstract base class for message history compression strategies.
    Defines the interface for summarizing or compressing a list of messages.
    """

    @abstractmethod
    async def compress(self, messages: List[Message]) -> List[Message]:
        """
        Takes a list of messages and returns a compressed or summarized version.

        Args:
            messages: A list of Message objects to be compressed.

        Returns:
            A list of compressed Message objects. This could be a single summary
            message or a filtered list of key messages.
        """
        pass
