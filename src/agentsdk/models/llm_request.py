# -*- coding: utf-8 -*-
"""
This file provides model request parameters.
"""

from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel

from agentsdk.message.message import Message


class CompletionRequest(BaseModel):
    """
    Represents a single model inference request in the LoongFlow framework.
    This class defines only the dynamic, per-request data (not static model config).
    """

    # Core prompt messages (user, assistant, system, tool, etc.)
    messages: List[Message]

    # Optional generation controls
    temperature: Optional[float] = None
    """Sampling temperature (0.0 → deterministic)."""

    top_p: Optional[float] = None
    """Nucleus sampling probability threshold."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    stop: Optional[str | List[str]] = None
    """Stop sequences that terminate generation."""

    timeout: float | int = 600.0
    """Timeout in seconds for the model to generate."""

    # Output structure / function calling
    response_format: Optional[dict | type] = None
    """Controls the output format (e.g., JSON schema or pydantic model)."""

    tools: Optional[List[dict]] = None
    """List of callable tools (OpenAI-style tool specification)."""

    tool_choice: Optional[Literal["auto", "none", "any", "required"] | str] = None
    """Specifies whether/how the model should choose tools."""

    # Optional metadata and custom extension fields
    extra_headers: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
