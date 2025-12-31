#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
default compressor using llm to compress
"""

from typing import List, Tuple

from agentsdk.logger import get_logger
from agentsdk.memory.grade.compressor import Compressor
from agentsdk.memory.grade.compressor.prompt import (
    DEFAULT_COMPRESS_PROMPT,
    DEFAULT_USER_HINT_MESSAGE,
)
from agentsdk.message import Message, Role
from agentsdk.models import BaseLLMModel, CompletionRequest
from agentsdk.token import SimpleTokenCounter

logger = get_logger(__name__)


class EvolveCompressor(Compressor):
    """
    A compressor implementation that uses LLM to summarize message history.
    """

    def __init__(
        self,
        model: BaseLLMModel,
        token_counter: SimpleTokenCounter,
        token_threshold: int,
        prompt: str = DEFAULT_COMPRESS_PROMPT,
    ):
        """
        Initializes the LLMCompressor.

        Args:
            model: LLM instance
            token_counter: SimpleTokenCounter
            token_threshold: The maximum number of tokens allowed in the output.
            prompt: The instruction text to prepend to the message list to guide the model's summarization.
        """
        self.model = model
        self.token_counter = token_counter
        self.token_threshold = token_threshold
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

        first_user_idx = 0
        for i in range(len(messages)):
            message = messages[i]
            # Find the first user message, keep the first user message unchanged, compress the rest
            if message.role == "user":
                first_user_idx = i

        first_user_message = messages[first_user_idx]
        first_user_token_count = await self.token_counter.count([first_user_message])
        rest_messages = messages[first_user_idx + 1:]
        rest_token_count = self.token_threshold - first_user_token_count

        # First user message over threshold, we need to cutoff message
        if rest_token_count < 0:
            raise ValueError(
                f"The total token count of the first user message {first_user_token_count} "
                + f"exceeds the limit of {self.token_threshold}. "
                + f"Please reduce the length of your prompt or increase the model's capacity."
            )

        # we need to cut message till last user or assistant message
        compressed, kept = await self.split_message(rest_messages, rest_token_count)
        if len(compressed) == 0:
            return messages

        user_hint_message = Message.from_text(
            sender="compressor",
            role=Role.USER,
            data=DEFAULT_USER_HINT_MESSAGE,
        )

        history = [system_message, *compressed, user_hint_message]

        try:
            resp_generator = self.model.generate(
                CompletionRequest(
                    messages=history,
                    tool_choice="none",
                )
            )
            try:
                resp = await anext(resp_generator)
                if resp.error_code:
                    raise Exception(
                        f"Error code: {resp.error_code}, error: {resp.error_message}"
                    )
            finally:
                async for _ in resp_generator:
                    pass

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
        return [first_user_message, final_message, *kept]

    async def split_message(
        self, messages: List[Message], rest_token_count: int
    ) -> Tuple[List[Message], List[Message]]:
        """
        Finds the most recent two AI messages from the end and keeps all messages after them.
        Args:
            messages: A list of validated Pydantic Message objects.
            rest_token_count: The maximum number of tokens allowed in the output.
        Returns:
            A tuple containing two lists: (compressed_part, kept_part).
            compressed_part contains messages to be compressed (before the split point)
            kept_part contains messages to be kept (after including the AI message from the end)
        """
        if not messages:
            return [], []

        # Find the positions of AI messages from the end, max keep 2 AI messages
        ai_indices = []
        for i in range(len(messages) - 1, -1, -1):
            # Find the last AI message, check its token count
            if messages[i].role == "assistant":
                # Calculate the token count from the last AI message to the end.
                ai_token_count = await self.token_counter.count(messages[i:])
                # If the message is too long, we should compress all messages
                if ai_token_count >= rest_token_count:
                    break
                ai_indices.append(i)
                if len(ai_indices) >= 2:
                    break

        # Determine the split point
        if ai_indices:
            # Found at least N AI messages, split before the oldest one
            split_point = ai_indices[-1]  # The second AI message from the end
            compressed = messages[:split_point]
            kept = messages[split_point:]
        else:
            # No AI messages found, compress all messages
            compressed = messages
            kept = []

        return compressed, kept
