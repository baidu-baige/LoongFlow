# -*- coding: utf-8 -*-
"""
This file implements the AgentTool class, which allows an agent
to be exposed and invoked as a tool by other agents.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, TYPE_CHECKING

from agentsdk.message.elements import ContentElement, MimeType
from agentsdk.message.message import Message
from agentsdk.tools.base_tool import FunctionDeclarationDict
from agentsdk.tools.function_tool import FunctionTool
from agentsdk.tools.tool_context import ToolContext
from agentsdk.tools.tool_response import ToolResponse

if TYPE_CHECKING:
    from evolux.base.agent_base import BaseAgent


class AgentTool(FunctionTool):
    """Adapter that exposes an Agent as a callable Tool."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent

    def get_declaration(self) -> Optional[FunctionDeclarationDict]:
        """Return the JSON schema declaration of this agent as a tool."""
        if getattr(self.agent, "input_schema", None):
            raw_schema = self.agent.input_schema.model_json_schema()
            schema = self.resolve_refs(raw_schema)
        else:
            schema = {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "User request text"}
                },
                "required": ["request"],
            }

        return {
            "name": self.agent.name,
            "description": self.agent.description
            or f"Sub-agent tool: {self.agent.name}",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }

    def run(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Synchronously run the async sub-agent."""
        return asyncio.run(self.arun(args=args, tool_context=tool_context))

    async def arun(
        self, *, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the sub-agent asynchronously."""
        # AgentBase.__call__ is async
        message: Message = await self.agent(**args)
        return self._wrap_message_as_response(message)

    @staticmethod
    def _wrap_message_as_response(message: Message) -> ToolResponse:
        """Convert an Agent Message into a ToolResponse."""
        # Extract only ContentElements; if none found, fallback to serialize the whole message
        content_elements = message.get_elements(ContentElement)
        if not content_elements:
            content_elements = [
                ContentElement(
                    mime_type=MimeType.APPLICATION_JSON,
                    data=message.model_dump(),
                )
            ]

        return ToolResponse(content=content_elements)
