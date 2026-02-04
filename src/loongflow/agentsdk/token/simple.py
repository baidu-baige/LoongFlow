#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple token counter
"""
import json
from typing import List

from loongflow.agentsdk.message import Message
from loongflow.agentsdk.token import TokenCounter


class SimpleTokenCounter(TokenCounter):
    """
    A simple token counter that estimates token counts.
    """

    async def count(self, messages: List[Message], **kwargs) -> int:
        """
        Get token count of provided messages
        :param messages:
        :param kwargs:
        :return: token count of all messages
        """
        return sum(self._count_message(msg) for msg in messages)

    def _count_message(self, message: Message) -> int:
        """
        Estimates the token count for a given message.
        """
        return len(json.dumps({
            "role": message.role,
            "content": [elem.get_content() for elem in message.content],
        }, ensure_ascii=False, indent=2)) // 4
