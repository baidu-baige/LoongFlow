#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token abstraction
"""

from abc import ABC, abstractmethod
from typing import List

from agentsdk.message import Message


class TokenCounter(ABC):
    """
    An abstract base class for token counting strategies.
    Defines the interface for calculating the token count of messages.
    """

    @abstractmethod
    async def count(self, messages: List[Message], **kwargs) -> int:
        """
        Calculates or retrieves the token count for given messages.

        Args:
            messages: The Message objects to be evaluated.

        Returns:
            The number of tokens as an integer.
        """
        pass
