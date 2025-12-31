# -*- coding: utf-8 -*-
"""
This file provides model response class.
"""

from typing import Literal, Optional, Sequence

from pydantic import BaseModel

from agentsdk.message.elements import ContentElement, ToolCallElement, ThinkElement


class CompletionUsage(BaseModel):
    """Token usage statistics."""
    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt_tokens + completion_tokens)."""

class CompletionResponse(BaseModel):
    """LLM completion response."""
    id: str

    usage: Optional[CompletionUsage] = None
    """Token usage statistics."""

    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    
    content: Sequence[ContentElement | ToolCallElement | ThinkElement]
    """The generated content."""
    
    error_code: Optional[str] = None
    """Error code if the response is an error. Code varies by model."""

    error_message: Optional[str] = None
    """Error message if the response is an error."""

