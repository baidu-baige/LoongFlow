#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file storage implements Storage
"""

import collections
import json
import os
import uuid
from functools import wraps
from typing import Any, List, Union

from agentsdk.memory.grade.storage import Storage
from agentsdk.message import Message


def ensure_loaded(func):
    """Decorator to ensure the cache is loaded before calling the method."""

    @wraps(func)
    async def wrapper(self: "FileStorage", *args, **kwargs):
        if self._cache is None:
            await self._load()
        return await func(self, *args, **kwargs)

    return wrapper


class FileStorage(Storage):
    """
    A file-based implementation of the Storage interface using JSON.
    Provides persistence by saving messages to a local file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._cache: collections.OrderedDict[uuid.UUID, Message] | None = None

    async def _load(self) -> None:
        """Loads messages from the JSON file into the in-memory cache."""
        if not os.path.exists(self.file_path):
            self._cache = collections.OrderedDict()
            return
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                message_list = json.load(f)
                self._cache = collections.OrderedDict(
                        (uuid.UUID(msg_data['id']), Message.from_dict(msg_data))
                        for msg_data in message_list
                )
        except (json.JSONDecodeError, FileNotFoundError, TypeError):
            self._cache = collections.OrderedDict()

    async def _save(self) -> None:
        """Saves all messages from the in-memory cache to the JSON file."""
        if self._cache is None:
            return
        message_list = [msg.to_dict() for msg in self._cache.values()]
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(message_list, f, indent=2, ensure_ascii=False)

    @ensure_loaded
    async def add(self, messages: Union[Message, List[Message]]) -> None:
        """Adds one or more messages and persists the changes immediately."""
        if isinstance(messages, Message):
            messages = [messages]
        for msg in messages:
            self._cache[msg.id] = msg
        await self._save()

    @ensure_loaded
    async def get(self, message_id: uuid.UUID) -> Message | None:
        """Retrieve a message from the cache by its ID."""
        return self._cache.get(message_id)

    @ensure_loaded
    async def remove(self, message_id: uuid.UUID) -> bool:
        """Removes a message and persists the changes immediately."""
        if self._cache.pop(message_id, None) is not None:
            await self._save()
            return True
        return False

    async def search(self, *args: Any, **kwargs: Any) -> List[Message]:
        """Provides a basic text search functionality over message content."""
        raise NotImplementedError("FileStorage does not support search operations.")

    @ensure_loaded
    async def get_all(self) -> List[Message]:
        """Retrieves all messages."""
        return list(self._cache.values())

    @ensure_loaded
    async def get_size(self) -> int:
        """Returns the total number of messages."""
        return len(self._cache)

    @ensure_loaded
    async def clear(self) -> None:
        """Clears the cache and deletes the storage file."""
        self._cache.clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
