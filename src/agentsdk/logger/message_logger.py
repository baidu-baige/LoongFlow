# -*- coding: utf-8 -*-
"""
This module provides structured printing/logging for `Message`
objects, which represent conversation turns between system, user, assistant,
and tool components.

Output mode can be controlled via:
    - Environment variable MESSAGE_LOG_MODE = "print" (default) or "logger"
    - Or explicit argument `use_logger=True` in function call
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import List, Union

from agentsdk.logger import get_logger
from agentsdk.message import (ContentElement, Message, ThinkElement,
                              ToolCallElement, ToolOutputElement, ToolStatus)

try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel

    _console = Console()
    _USE_RICH = True
except ImportError:
    _console = None
    _USE_RICH = False

_logger = get_logger("message")


def print_message(
    msg_or_msgs: Union[Message, List[Message]],
    *,
    show_metadata: bool = False,
    stream: bool = False,
    use_logger: bool | None = False,
) -> None:
    """
    Pretty print or log a single Message or list[Message].

    Args:
        msg_or_msgs: The Message instance or list of Message instances.
        show_metadata: Whether to include metadata details.
        stream: If True, suppress final newline (useful for streaming).
        use_logger: Force logging mode (True/False). If None, read from env var MESSAGE_LOG_MODE.
    """

    if isinstance(msg_or_msgs, list):
        for msg in msg_or_msgs:
            _handle_single_message(msg, show_metadata, stream, use_logger)
    else:
        _handle_single_message(msg_or_msgs, show_metadata, stream, use_logger)


def _handle_single_message(
    msg: Message,
    show_metadata: bool,
    stream: bool,
    use_logger: bool,
) -> None:
    """Internal dispatcher: choose print or logger."""
    if use_logger:
        log_data = {
            "id": str(msg.id),
            "role": msg.role,
            "sender": getattr(msg, "sender", None),
            "timestamp": getattr(msg, "timestamp", datetime.now()).isoformat(),
            "metadata": msg.metadata if show_metadata else None,
            "content": [
                c.model_dump() if hasattr(c, "model_dump") else str(c)
                for c in msg.content
            ],
        }
        _logger.info(json.dumps(log_data, ensure_ascii=False, default=_json_serializer))
    else:
        _print_single_message(msg, show_metadata=show_metadata, stream=stream)


def _print_single_message(msg: Message, *, show_metadata: bool, stream: bool) -> None:
    """Pretty print a single Message to console."""
    header = f"[{msg.role}] {getattr(msg, 'sender', 'Anonymous')}"
    timestamp = getattr(msg, "timestamp", datetime.now()).strftime("%H:%M:%S")

    if _USE_RICH:
        header_text = Text(f"{header} ", style="bold cyan")
        header_text.append(f"({timestamp})", style="dim")
        _console.print(header_text)
    else:
        print(f"{header} ({timestamp})")

    for elem in msg.content:
        _print_element(elem)

    if show_metadata and msg.metadata:
        _print_metadata(msg.metadata)

    if not stream:
        if _USE_RICH:
            _console.print()
        else:
            print()


def _print_element(elem: object) -> None:
    """Dispatch to specific element printer based on type."""
    if isinstance(elem, ContentElement):
        _print_content(elem)
    elif isinstance(elem, ThinkElement):
        _print_think(elem)
    elif isinstance(elem, ToolCallElement):
        _print_tool_call(elem)
    elif isinstance(elem, ToolOutputElement):
        _print_tool_output(elem)
    else:
        _print_unknown_element(elem)


def _print_content(elem: ContentElement) -> None:
    prefix = "🗨️  "
    if elem.mime_type == "text/plain":
        text = str(elem.data)
    elif isinstance(elem.data, bytes):
        text = f"[{elem.mime_type}] (base64): {base64.b64encode(elem.data).decode('utf-8')}"
    else:
        text = f"[{elem.mime_type}] <{type(elem.data).__name__}>"
    (
        _console.print(Text(f"{prefix}{text}", style="white"))
        if _USE_RICH
        else print(f"{prefix}{text}")
    )


def _print_think(elem: ThinkElement) -> None:
    prefix = "💭 "
    content = str(elem.content)
    (
        _console.print(Text(f"{prefix}{content}", style="italic magenta"))
        if _USE_RICH
        else print(f"{prefix}{content}")
    )


def _print_tool_call(elem: ToolCallElement) -> None:
    title = f"🛠️  Tool Call → {elem.target}"
    args_json = json.dumps(elem.arguments, indent=2, ensure_ascii=False)
    (
        _console.print(Panel(args_json, title=title, border_style="blue"))
        if _USE_RICH
        else print(f"{title}\n{args_json}")
    )


def _print_tool_output(elem: ToolOutputElement) -> None:
    title = f"📦 Tool Output ← {elem.tool_name} ({elem.status})"
    style = {
        ToolStatus.SUCCESS: "green",
        ToolStatus.ERROR: "red",
        ToolStatus.IN_PROGRESS: "yellow",
    }.get(elem.status, "white")

    result_data = [r.data for r in elem.result]
    result_json = json.dumps(
        result_data, indent=2, ensure_ascii=False, default=_json_serializer
    )
    (
        _console.print(Panel(result_json, title=title, border_style=style))
        if _USE_RICH
        else print(f"{title}\n{result_json}")
    )


def _print_unknown_element(elem: object) -> None:
    raw = json.dumps(
        getattr(elem, "model_dump", lambda: str(elem))(), indent=2, ensure_ascii=False
    )
    (
        _console.print(
            Panel(
                raw,
                title=f"Unknown Element ({type(elem).__name__})",
                border_style="dim",
            )
        )
        if _USE_RICH
        else print(f"Unknown Element ({type(elem).__name__}):\n{raw}")
    )


def _print_metadata(metadata: dict) -> None:
    meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
    (
        _console.print(Panel(meta_json, title="Metadata", border_style="dim"))
        if _USE_RICH
        else print(f"Metadata:\n{meta_json}")
    )



def _json_serializer(obj):
    import uuid
    from datetime import datetime

    if isinstance(obj, (uuid.UUID, datetime)):
        return str(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    return str(obj)
