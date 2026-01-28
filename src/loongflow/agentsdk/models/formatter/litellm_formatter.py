# -*- coding: utf-8 -*-
"""
This file implements LiteLLMFormatter
The adapter between LoongFlow message abstractions and the LiteLLM client interface
"""

import ast
import json
from typing import Any, Dict, List, Optional

from litellm import ModelResponse, ModelResponseStream

from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import (
    ContentElement,
    Message,
    MimeType,
    ThinkElement,
    ToolCallElement,
    ToolOutputElement,
)
from loongflow.agentsdk.models.formatter.base_formatter import BaseFormatter
from loongflow.agentsdk.models.llm_request import CompletionRequest
from loongflow.agentsdk.models.llm_response import CompletionResponse, CompletionUsage

logger = get_logger(__name__)


class LiteLLMFormatter(BaseFormatter):
    """
    Formatter that bridges LoongFlow message and response schemas
    with LiteLLM's API semantics.

    - Request direction: LoongFlow → LiteLLM
    - Response direction: LiteLLM → LoongFlow
    """

    MODEL_PROVIDER_PREFIX_MAP = {
        "gpt": "openai",
        "azure-gpt": "azure_openai",
        "claude": "anthropic",
        "gemini": "google_ai_studio",
        "mistral": "mistral_ai",
        "huggingface": "huggingface",
        "deepseek": "deepseek",
    }

    def __init__(self):
        super().__init__()
        self._current_model_name = None

    def format_request(
        self,
        request: CompletionRequest,
        model_name: str,
        stream: bool = False,
        timeout: int = 600,
        base_url:Optional[str] = None,
        api_key: Optional[str] = None,
        model_provider: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """
        Convert LoongFlow CompletionRequest into kwargs suitable for `litellm.acompletion`.
        """
        self._current_model_name = model_name

        llm_messages = self._convert_messages(request.messages, model_name)
        logger.debug(
            f"convert message after: {json.dumps(llm_messages, ensure_ascii=False)}"
        )

        provider_name = self.get_provider_for_model(
            model_name=model_name, model_provider=model_provider
        )

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": llm_messages,
            "stream": False,  # TODO: support streaming later
            "cache": {"no-cache": True},
            "timeout": timeout,
            "custom_llm_provider": provider_name,
        }

        for key in ["temperature", "top_p", "max_tokens", "stop"]:
            value = params.get(key)
            if value is not None:
                kwargs[key] = value

        if base_url is not None:
            kwargs["base_url"] = base_url

        if api_key is not None:
            kwargs["api_key"] = api_key

        # Determine provider (required for non-OpenAI models)
        logger.debug(f"custom_llm_provider: {provider_name}")

        # Optional generation settings
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop is not None:
            kwargs["stop"] = request.stop

        # Tool/function calling support
        if request.tools is not None:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        else:
            # Default to auto if tools are specified
            if request.tools:
                kwargs["tool_choice"] = "auto"

        # Response format (e.g. JSON mode)
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        # Extra headers / metadata
        if request.extra_headers is not None:
            kwargs["extra_headers"] = request.extra_headers

        return kwargs

    def parse_response(
        self,
        raw: ModelResponse | Dict[str, Any],
    ) -> CompletionResponse:
        """
        Parse LiteLLM response (single or streamed chunk) into LoongFlow CompletionResponse.

        Args:
            raw: The raw response from LiteLLM — can be a full ModelResponse or stream delta.

        Returns:
            CompletionResponse object ready for LoongFlow consumption.
        """
        # Handle dict-like chunks from streamed responses
        if isinstance(raw, ModelResponseStream):
            return self._parse_stream_response(raw)

        # dict-like chunk
        if isinstance(raw, dict):
            return self._parse_dict_response(raw)

        # Handle full ModelResponse object (non-stream)
        if isinstance(raw, ModelResponse):
            return self._parse_full_response(raw)

        # Unexpected type
        return CompletionResponse(
            id="error",
            content=[],
            error_code="unknown_response_type",
            error_message=f"Unsupported response type: {type(raw)}",
        )

    def get_provider_for_model(
        self, model_name: str, model_provider: Optional[str] = None
    ) -> str:
        """
        Determine the LLM provider for a given model.
        """
        if model_provider:
            return model_provider

        for prefix, provider in self.MODEL_PROVIDER_PREFIX_MAP.items():
            if model_name.startswith(prefix):
                return provider

        # Default to OpenAI provider
        return "openai"

    def _is_deepseek_reasoner(self, model_name: str) -> bool:
        """
        check if the model name is deepseek-reasoner model
        """
        if not model_name:
            return False
        return "deepseek-reasoner" in model_name.lower()

    def _convert_messages(self, messages: List[Message], model_name: str) -> List[Dict[str, Any]]:
        """
        Convert LoongFlow Message objects into OpenAI/LiteLLM-compatible messages.

        Each Message may contain multiple element types:
        - ContentElement: text, image, or audio.
        - ToolCallElement: tool invocation (assistant role with `tool_calls`).
        - ToolOutputElement: tool result (standalone `tool` role).
        - ThinkElement: internal reasoning (prefixed with `[think]`).

        The conversion process:
        1. Convert tool outputs (ToolOutputElement) to separate `tool` messages.
        2. Merge assistant tool calls (ToolCallElement) with content.
        3. Convert remaining standard content messages.
        """
        converted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else msg.role

            content_texts, structured_items, tool_calls, tool_outputs, thinking_content = (
                self._collect_message_elements(msg)
            )

            # 1. Convert tool output results → 'tool' messages
            for toe in tool_outputs:
                converted.append(self._convert_tool_output(toe))

            # 2. Merge tool calls + content into assistant message
            if tool_calls:
                converted.append(
                    self._merge_tool_calls(
                        role, content_texts, structured_items, tool_calls, thinking_content, model_name
                    )
                )
                continue

            # 3. Standard text or structured message
            if content_texts or structured_items:
                converted.append(
                    self._convert_standard_message(
                        role, content_texts, structured_items
                    )
                )

        return converted

    def _collect_message_elements(
        self, msg: Message
    ) -> tuple[
        list[str], list[dict[str, Any]], list[dict[str, Any]], list["ToolOutputElement"], list[str]
    ]:
        """
        Extract different element types from a Message and categorize them.
        Returns (content_texts, structured_items, tool_calls, tool_outputs, thinking_content)
        """
        content_texts, structured_items, tool_calls, tool_outputs, thinking_content = [], [], [], [], []

        for elem in msg.content:
            if isinstance(elem, ContentElement):
                self._append_content_element(elem, content_texts, structured_items)
            elif isinstance(elem, ToolCallElement):
                tool_calls.append(
                    {
                        "id": str(elem.call_id),
                        "type": "function",
                        "function": {
                            "name": elem.target,
                            "arguments": json.dumps(elem.arguments, ensure_ascii=False),
                        },
                    }
                )
            elif isinstance(elem, ToolOutputElement):
                tool_outputs.append(elem)
            elif isinstance(elem, ThinkElement):
                thinking_content.append(elem.content)
            else:
                content_texts.append("[unsupported element]")

        return content_texts, structured_items, tool_calls, tool_outputs, thinking_content

    def _append_content_element(
        self,
        elem: "ContentElement",
        content_texts: list[str],
        structured_items: list[dict[str, Any]],
    ):
        """Handle ContentElement based on mime_type."""
        if elem.mime_type == MimeType.TEXT_PLAIN:
            content_texts.append(elem.data)
        elif elem.mime_type == MimeType.APPLICATION_JSON:
            structured_items.append(
                {"type": "text", "text": json.dumps(elem.data, ensure_ascii=False)}
            )
        elif isinstance(elem.mime_type, str) and elem.mime_type.startswith("image/"):
            structured_items.append(
                {"type": "image_url", "image_url": {"url": elem.data}}
            )
        elif isinstance(elem.mime_type, str) and elem.mime_type.startswith("audio/"):
            structured_items.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": elem.data,
                        "format": elem.mime_type.split("/")[-1],
                    },
                }
            )
        else:
            content_texts.append(f"[unsupported {elem.mime_type}]")

    def _convert_tool_output(self, toe: "ToolOutputElement") -> dict[str, Any]:
        """Convert ToolOutputElement → OpenAI-compatible 'tool' message."""
        result_items = []
        for r in toe.result:
            if isinstance(r, ContentElement):
                result_items.append(
                    r.data if r.mime_type == MimeType.TEXT_PLAIN else r.get_content()
                )
            else:
                try:
                    result_items.append(json.dumps(r, ensure_ascii=False))
                except Exception:
                    result_items.append(str(r))

        result_content = (
            result_items[0]
            if len(result_items) == 1 and isinstance(result_items[0], str)
            else json.dumps(result_items, ensure_ascii=False)
        )

        return {
            "role": "tool",
            "name": toe.tool_name,
            "tool_call_id": str(toe.call_id),
            "content": result_content,
        }

    def _merge_tool_calls(
        self,
        role: str,
        content_texts: list[str],
        structured_items: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        thinking_content: list[str],
        model_name: str,
    ) -> dict[str, Any]:
        """Merge content and tool_calls into a single assistant message."""
        combined_text = "\n".join(content_texts).strip() if content_texts else ""
        if structured_items and not combined_text:
            content_field = structured_items
        else:
            content_field = combined_text

        message = {
            "role": "assistant" if role == "assistant" else role,
            "content": content_field,
            "tool_calls": tool_calls,
        }

        if self._is_deepseek_reasoner(model_name) and role == "assistant":
            if thinking_content:
                reasoning = "\n".join(thinking_content)
                message["reasoning_content"] = reasoning

        return message

    def _convert_standard_message(
        self,
        role: str,
        content_texts: list[str],
        structured_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Convert standard (non-tool) message."""
        if structured_items:
            content_field = structured_items
        else:
            content_field = "\n".join(content_texts).strip()
        return {"role": role, "content": content_field}

    def _parse_full_response(self, raw: ModelResponse) -> CompletionResponse:
        """
        Parse a full (non-streaming) LiteLLM ModelResponse into CompletionResponse.
        """
        # Extract main message content
        contents = self._extract_elements_from_choices(raw)

        # Token usage
        usage = None
        if getattr(raw, "usage", None):
            usage = CompletionUsage(
                completion_tokens=raw.usage.get("completion_tokens", 0),
                prompt_tokens=raw.usage.get("prompt_tokens", 0),
                total_tokens=raw.usage.get("total_tokens", 0),
            )

        return CompletionResponse(
            id=getattr(raw, "id", "unknown"),
            usage=usage,
            finish_reason=(
                getattr(raw.choices[0], "finish_reason", None)
                if getattr(raw, "choices", None)
                else None
            ),
            content=contents,
        )

    def _parse_stream_response(self, raw: "ModelResponseStream") -> CompletionResponse:
        """
        Parse a single streamed delta chunk (ModelResponseStream) into CompletionResponse.
        """
        if not raw.choices:
            return CompletionResponse(id=raw.id, content=[])

        delta = raw.choices[0].delta
        finish_reason = raw.choices[0].finish_reason

        elements: list[ContentElement | ToolCallElement | ThinkElement] = []

        if delta:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                elements.append(ThinkElement(content=reasoning))

            if delta.content:
                elements.append(
                    ContentElement(mime_type=MimeType.TEXT_PLAIN, data=delta.content)
                )
            if delta.tool_calls:
                for call in delta.tool_calls:
                    elements.append(
                        ToolCallElement(
                            target=call.function.name,
                            arguments=call.function.arguments,
                        )
                    )

        return CompletionResponse(
            id=raw.id,
            finish_reason=finish_reason,
            content=elements,
        )

    def _parse_dict_response(self, data: Dict[str, Any]) -> CompletionResponse:
        """
        Parse a single streamed delta chunk (dict) into CompletionResponse.
        """
        choices = data.get("choices", [])
        if not choices:
            return CompletionResponse(id=data.get("id", "stream"), content=[])

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason", None)

        elements: List[ContentElement | ToolCallElement | ThinkElement] = []

        if "reasoning_content" in delta and delta["reasoning_content"]:
            elements.append(ThinkElement(content=delta["reasoning_content"]))

        if "content" in delta and delta["content"]:
            elements.append(
                ContentElement(
                    mime_type=MimeType.TEXT_PLAIN,
                    data=delta["content"],
                )
            )
        elif "tool_calls" in delta:
            for tool_call in delta["tool_calls"]:
                elements.append(
                    ToolCallElement(
                        target=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    )
                )

        return CompletionResponse(
            id=data.get("id", "stream"),
            finish_reason=finish_reason,
            content=elements,
        )

    def _extract_elements_from_choices(
        self, raw: ModelResponse
    ) -> List[ContentElement | ToolCallElement | ThinkElement]:
        """
        Convert the first choice's message content into LoongFlow elements.
        Ensures tool_call arguments are always a dict.
        """
        if not getattr(raw, "choices", None):
            return []

        choice = raw.choices[0]
        message = getattr(choice, "message", {})

        content = getattr(message, "content", None) or message.get("content")
        tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls")

        reasoning_content = getattr(message, "reasoning_content", None) or message.get("reasoning_content")

        elements: List[ContentElement | ToolCallElement | ThinkElement] = []

        if reasoning_content:
            elements.append(ThinkElement(content=reasoning_content))

        if content:
            elements.append(ContentElement(mime_type=MimeType.TEXT_PLAIN, data=content))

        # Handle tool calls (OpenAI style)
        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict):
                    func = call.get("function", {})
                else:
                    func = getattr(call, "function", {})
                    if hasattr(func, "to_dict"):
                        func = func.to_dict()
                    elif not isinstance(func, dict):
                        func = {
                            "name": getattr(func, "name", ""),
                            "arguments": getattr(func, "arguments", "")
                        }

                target_name = func.get("name", "")
                args_raw = func.get("arguments", {})

                tool_err = ""
                final_args = {}

                if isinstance(args_raw, str):
                    final_args, method, err_msg = self._safe_parse_json(args_raw)
                    logger.debug(f"[tool-call] parse method used: {method}")
                    if method == "failed":
                        tool_err = err_msg
                elif isinstance(args_raw, dict):
                    final_args = args_raw

                elements.append(
                    ToolCallElement(
                        metadata={
                            "tool_arguments_err": tool_err,
                        },
                        target=target_name,
                        arguments=final_args,
                    )
                )

        return elements

    def _safe_parse_json(self, arg_str: str):
        """
        Safely parse JSON-like strings with multiple fallback strategies.

        Parsing order:
            1. Standard json.loads
            2. ast.literal_eval as a last-resort fallback

        Args:
            arg_str (str): Input string to parse.

        Returns:
            (parsed, method, err_msg):
                parsed: Parsed dict/list, or {}
                method: "json", "literal_eval", "failed"
                err_msg: Error message when method == "failed", otherwise ""
        """
        raw = arg_str
        err_msg = ""

        # 1. Standard JSON
        try:
            parsed = json.loads(raw)
            return parsed, "json", err_msg
        except json.JSONDecodeError as e1:
            logger.warning(
                "[safe_parse_json] JSON decode failed. "
                f"error={e1}. raw_to_reproduce={raw!r}"
            )

        # 2. literal_eval fallback
        try:
            parsed = ast.literal_eval(raw)
            logger.debug(
                "[safe_parse_json] literal_eval fallback used. "
                f"raw_to_reproduce={raw!r} parsed={parsed}"
            )
            return parsed, "literal_eval", err_msg
        except Exception as e2:
            logger.error(
                "[safe_parse_json] All parsing strategies failed. "
                f"literal_eval_error={e2}. raw_to_reproduce={raw!r}"
            )
            err_msg = f"Failed to parse tool arguments: {e2}"

        # All strategies failed → return failure info
        return {}, "failed", err_msg
