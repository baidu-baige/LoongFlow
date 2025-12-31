#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
default compressor using llm to compress
"""

from typing import List, Tuple

from agentsdk.logger import get_logger
from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.compressor.prompt import DEFAULT_COMPRESS_PROMPT, DEFAULT_USER_HINT_MESSAGE
from agentsdk.message import Message, Role, ToolCallElement
from agentsdk.models import BaseLLMModel, CompletionRequest

logger = get_logger(__name__)


class LLMCompressor(Compressor):
    """
    A compressor implementation that uses LLM to summarize message history.
    """

    def __init__(self, model: BaseLLMModel, prompt: str = DEFAULT_COMPRESS_PROMPT):
        """
        Initializes the LLMCompressor.

        Args:
            model: LLM instance
            prompt: The instruction text to prepend to the message list to guide the model's summarization.
        """
        self.model = model
        self.prompt = prompt

    async def compress(self, messages: List[Message]) -> List[Message]:
        """
        Compresses a list of messages using the configured language model.
        """
        if not messages:
            return []

        system_message = Message.from_text(
            sender="compressor",
            role=Role.SYSTEM,
            data=self.prompt,
        )

        # we need to cut message till last user or assistant message
        compressed, kept = self.split_message(messages)
        if len(compressed) == 0:
            return messages

        user_hint_message = Message.from_text(
            sender="compressor",
            role=Role.USER,
            data=DEFAULT_USER_HINT_MESSAGE,
        )

        history = [system_message, *compressed, user_hint_message]

        try:
            resp_generator = self.model.generate(CompletionRequest(
                messages=history,
                tool_choice="none",
            ))
            resp = await anext(resp_generator)
            if resp.error_code:
                raise Exception(f"Error code: {resp.error_code}, error: {resp.error_message}")

            final_message = Message.from_elements(
                sender="compressor",
                role=Role.ASSISTANT,
                elements=list(resp.content),
                metadata={"usage": resp.usage},
            )
        except Exception as e:
            # if exception found, we will use former messages as fallback
            logger.error(f"Error while compressing message: {e}")
            return messages
        return [final_message, *kept]

    def split_message(self, messages: List[Message]) -> Tuple[List[Message], List[Message]]:
        """
        Traverses a list of Pydantic Message objects in reverse order to find a split point based on a set of rules.
        Args:
            messages: A list of validated Pydantic Message objects.
        Returns:
            A tuple containing two lists: (part1, part2). If no split point is
            found, part1 is empty and part2 contains all messages.
        """
        # Iterate through indices in reverse order.
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            # Rule A: The last message with role 'user'.
            if message.role == 'user':
                part1 = messages[:i + 1]
                part2 = messages[i + 1:]
                return part1, part2
            # Rule B: A message with role 'assistant' and no tool elements.
            if message.role == 'assistant':
                # Check if the elements list is empty.
                if not message.get_elements(ToolCallElement) or len(message.get_elements(ToolCallElement)) == 0:
                    part1 = messages[:i + 1]
                    part2 = messages[i + 1:]
                    return part1, part2
            # Rule C: A message with role 'tool'.
            if message.role == 'tool':
                part1 = messages[:i + 1]
                part2 = messages[i + 1:]
                return part1, part2
        # If the loop completes without finding a split point.
        return [], messages
