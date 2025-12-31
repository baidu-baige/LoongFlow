# -*- coding: utf-8 -*-
"""
Base formatter abstraction for LoongFlow model backends.
Defines the common request/response conversion interface
that all formatters (e.g. LiteLLM, Ollama, vLLM) should follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from agentsdk.models.llm_request import CompletionRequest
from agentsdk.models.llm_response import CompletionResponse


class BaseFormatter(ABC):
    """
    Abstract base class for model formatters in LoongFlow.

    Formatters are responsible for:
    1. Converting a generic `CompletionRequest` into backend-specific kwargs
       (for example, `litellm.acompletion` parameters).
    2. Parsing backend-native responses or streamed chunks into
       LoongFlow-standard `CompletionResponse` objects.
    """

    @abstractmethod
    def format_request(
        self,
        request: CompletionRequest,
        model_name: str,
        base_url: str,
        api_key: str,
        model_provider: str,
        timeout: int = 600,
        stream: bool = False,
        **params,
    ) -> Dict[str, Any]:
        """
        Convert LoongFlow request → backend request parameters.

        Args:
            request: High-level LoongFlow CompletionRequest.
            model_name: Model name (e.g. "gpt-4o").
            base_url: Model endpoint base URL.
            api_key: Auth key for backend.
            timeout: Timeout in seconds.
            stream: Whether to stream results.
            model_provider: Name of custom model provider.

        Returns:
            Dict[str, Any]: Backend-specific request kwargs.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self,
        raw_resp: Any,
    ) -> CompletionResponse:
        """
        Parse backend response → LoongFlow CompletionResponse.

        Args:
            raw_resp: Backend-specific raw response or stream chunk.

        Returns:
            Parsed `CompletionResponse`.
        """
        raise NotImplementedError
