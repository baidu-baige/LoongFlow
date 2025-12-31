# -*- coding: utf-8 -*-
"""
This file provides litellm model wrapper
"""
import logging
from typing import AsyncGenerator, Optional

import litellm

from agentsdk.logger import get_logger
from agentsdk.models.base_llm_model import BaseLLMModel
from agentsdk.models.formatter.litellm_formatter import LiteLLMFormatter
from agentsdk.models.llm_request import CompletionRequest
from agentsdk.models.llm_response import CompletionResponse

logger = get_logger(__name__)


class LiteLLMModel(BaseLLMModel):
    """
    LoongFlow model backend implementation based on LiteLLM.

    Each model instance holds static configuration (model name, base_url, api_key),
    while each `generate` call handles dynamic per-request parameters
    (messages, tools, temperature, etc.).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout: int = 600,
        model_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the LiteLLM-based model wrapper.

        Args:
            model_name: Model name or deployment ID (e.g. "gpt-4o").
            base_url: Base URL of the model provider (e.g. OpenAI, Azure, Baidu).
            api_key: API key for authentication.
        """
        # Disable litellm internal debug logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.formatter = LiteLLMFormatter()
        self.model_provider = model_provider
        self.generation_params = kwargs

    @classmethod
    def from_config(cls, config: dict) -> "LiteLLMModel":
        """
        Create a model instance from configuration dictionary.

        Args:
            config: Configuration dictionary containing model settings.
                    Must include 'model', 'url', and 'api_key'.

        Returns:
            LiteLLMModel: Initialized model instance.

        Raises:
            KeyError: If required fields are missing from config.
        """
        # Validate required fields
        required = ["model", "url", "api_key"]
        if missing := [f for f in required if f not in config]:
            raise KeyError(f"Config missing required fields: {missing}")

        # Separate known fields from generation parameters
        known = {"model", "url", "api_key", "model_provider", "timeout"}
        gen_params = {k: v for k, v in config.items() if k not in known}

        return cls(
            model_name=config["model"],
            base_url=config["url"],
            api_key=config["api_key"],
            model_provider=config.get("model_provider"),
            timeout=config.get("timeout", 600),
            **gen_params,
        )

    async def generate(
        self,
        request: CompletionRequest,
        stream: bool = False,
    ) -> AsyncGenerator[CompletionResponse, None]:
        """
        Generate a model completion asynchronously using LiteLLM.

        Args:
            request: CompletionRequest containing input messages, tools, etc.
            stream: Whether to stream responses from the LLM.

        Yields:
            CompletionResponse objects parsed from LiteLLM output.
        """
        # 1. Format the request for LiteLLM
        llm_kwargs = self.formatter.format_request(
            request=request,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            stream=stream,
            timeout=self.timeout,
            model_provider=self.model_provider,
            **self.generation_params,
        )

        # 2. Call LiteLLM asynchronously
        try:
            logger.debug("Start calling LiteLLM...")
            raw_resp = await litellm.acompletion(**llm_kwargs)
        except Exception as e:
            # On error, yield a single CompletionResponse with error info
            yield CompletionResponse(
                id="error",
                content=[],
                error_code="litellm_error",
                error_message=str(e),
            )
            return

        # 3. Handle streaming response
        if stream and hasattr(raw_resp, "__aiter__"):
            async for chunk in raw_resp:
                parsed = self.formatter.parse_response(chunk)
                yield parsed
            return

        # 4. Non-stream response (single ModelResponse)
        parsed = self.formatter.parse_response(raw_resp)
        yield parsed
