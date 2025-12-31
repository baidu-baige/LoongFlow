#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storage abstract interface
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, List

from agentsdk.message import Message


class Storage(ABC):
    """
    An abstract base class for data storage operations.
    Defines the contract for storing, retrieving, and managing messages.
    """

    @abstractmethod
    async def add(self, messages: Message | List[Message]) -> None:
        """Adds one or more messages to the storage."""
        pass

    @abstractmethod
    async def get(self, message_id: uuid.UUID) -> Message | None:
        """Retrieves a single message by its unique ID."""
        pass

    @abstractmethod
    async def remove(self, message_id: uuid.UUID) -> bool:
        """
        Removes a single message by its ID.
        Returns True on success, False otherwise.
        """
        pass

    @abstractmethod
    async def search(self, *args: Any, **kwargs: Any) -> List[Message]:
        """Searches for messages based on query criteria."""
        pass

    @abstractmethod
    async def get_all(self) -> List[Message]:
        """Retrieves all messages from the storage."""
        pass

    @abstractmethod
    async def get_size(self) -> int:
        """Returns the total number of messages in the storage."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Removes all messages from the storage."""
        pass
