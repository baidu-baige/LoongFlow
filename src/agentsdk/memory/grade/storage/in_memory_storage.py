#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory implements Storage
"""

import collections
import uuid
from typing import Any, List, Union

from agentsdk.memory.grade.storage import Storage
from agentsdk.message import Message


class InMemoryStorage(Storage):
    """
    An in-memory implementation of the Storage interface.
    Uses an OrderedDict to maintain message insertion order. Ideal for testing and simple use cases.
    """

    def __init__(self):
        self._data: collections.OrderedDict[uuid.UUID, Message] = collections.OrderedDict()

    async def add(self, messages: Union[Message, List[Message]]) -> None:
        """Adds one or more messages."""
        if isinstance(messages, Message):
            messages = [messages]
        for msg in messages:
            self._data[msg.id] = msg

    async def get(self, message_id: uuid.UUID) -> Message | None:
        """Retrieves a message by its ID."""
        return self._data.get(message_id)

    async def remove(self, message_id: uuid.UUID) -> bool:
        """Removes a message by its ID."""
        return self._data.pop(message_id, None) is not None

    async def search(self, *args: Any, **kwargs: Any) -> List[Message]:
        """Complex search operations are not supported by InMemoryStorage."""
        raise NotImplementedError("InMemoryStorage does not support search operations.")

    async def get_all(self) -> List[Message]:
        """Retrieves all messages."""
        return list(self._data.values())

    async def get_size(self) -> int:
        """Returns the total number of messages."""
        return len(self._data)

    async def clear(self) -> None:
        """Removes all messages."""
        self._data.clear()
