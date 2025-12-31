# -*- coding: utf-8 -*-
"""
This file provides base model class.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from agentsdk.models.formatter.base_formatter import BaseFormatter
from agentsdk.models.llm_request import CompletionRequest
from agentsdk.models.llm_response import CompletionResponse


class BaseLLMModel(ABC):
    """
    Abstract base class for all LoongFlow LLM model wrappers.

    Each model instance stores static configuration (model_name, base_url, api_key),
    while each `generate` call handles dynamic per-request parameters.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
    ):
        """
        Initialize the base LLM model.

        Args:
            model_name: Model name or deployment ID (e.g. "gpt-4o").
            base_url: Base URL of the model provider.
            api_key: Authentication key for the backend.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.formatter: BaseFormatter = None

    @abstractmethod
    async def generate(
        self,
        request: CompletionRequest,
        stream: bool = False,
    ) -> AsyncGenerator[CompletionResponse, None]:
        """
        Abstract method to generate model completions asynchronously.

        Args:
            request: CompletionRequest containing input messages, tools, etc.
            stream: Whether to stream responses.

        Yields:
            CompletionResponse objects parsed from the backend output.
        """
        raise NotImplementedError
