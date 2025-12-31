# -*- coding: utf-8 -*-
"""
This file provides the toolkit implementation.
"""

from typing import Any, Dict, List, Optional

from agentsdk.tools.function_tool import FunctionTool
from agentsdk.tools.tool_context import AuthConfig, AuthCredential, ToolContext
from agentsdk.tools.tool_response import ToolResponse
from ..message import ContentElement, MimeType


class Toolkit:
    """
    Toolkit: manages registration and retrieval of multiple FunctionTool instances.
    
    - Toolkit provides unified access for agents to query and run tools.
    """

    def __init__(self):
        self._tools: Dict[str, FunctionTool] = {}
        self._contexts: Dict[str, ToolContext] = {}

    def get_declarations(self) -> List[dict[str, Any]]:
        """Return declarations for all registered tools """
        wrapped_tools = []
        for tool in self._tools.values():
            decl = tool.get_declaration()
            wrapped = {
                "type": "function",
                "function": decl
            }
            wrapped_tools.append(wrapped)
        return wrapped_tools
    
    def run(
        self,
        name: str,
        *,
        args: dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """
        Run a tool synchronously

        Args:
            name (str): Name of the tool to run.
            args (dict[str, Any]): Arguments to pass to the tool.
            tool_context (Optional[ToolContext]): An optional ToolContext instance to use for running the tool.
        Returns:
            ToolResponse: structured result of the tool execution.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=f"Tool '{name}' not found",
                    )
                ],
                err_msg=f"Tool '{name}' not found",
            )

        ctx = self.ensure_context(name, external_context=tool_context)
        return tool.run(args=args, tool_context=ctx)
    
    async def arun(
        self,
        name: str,
        *,
        args: dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """
        Run a tool asynchronously
        Args:
            name (str): Name of the tool to run.
            args (dict[str, Any]): Arguments to pass to the tool.
            tool_context (Optional[ToolContext]): An optional ToolContext instance to use for running the tool.
        Returns:
            ToolResponse: structured result of the tool execution.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=f"Tool '{name}' not found",
                    )
                ],
                err_msg=f"Tool '{name}' not found",
            )

        ctx = self.ensure_context(name, external_context=tool_context)
        return await tool.arun(args=args, tool_context=ctx)
    
    def register_tool(self, tool: FunctionTool, *, auths: Optional[list[tuple[AuthConfig, AuthCredential]]] = None):
        """
        Register a tool with optional authentication credentials.

        Args:
            tool (FunctionTool): The tool to be registered.
            auths (Optional[list[tuple[AuthConfig, AuthCredential]]]): Authentication configuration and credentials.
        """
        self._tools[tool.name] = tool
        if auths:
            context = ToolContext(function_call_id=tool.name)
            for cfg, cred in auths:
                context.set_auth(cfg, cred)
            self._contexts[tool.name] = context

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[FunctionTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())
    
    def get_context(self, name: str) -> Optional[ToolContext]:
        """Retrieve a ToolContext for a specific tool."""
        return self._contexts.get(name)

    def ensure_context(
        self,
        name: str,
        external_context: Optional[ToolContext] = None
    ) -> ToolContext:
        """
        Ensure a ToolContext exists for a tool.
        If external_context is provided, use it and store internally for future use.
        args:
            name (str): The name of the tool.
            external_context (Optional[ToolContext]): An external context to use.
        """
        if external_context is not None:
            self._contexts[name] = external_context
            return external_context

        existing = self._contexts.get(name)
        if existing:
            return existing
        
        context = ToolContext(function_call_id=name, state={})
        self._contexts[name] = context
        return context

    def set_auth(self, name: str, auth_config: AuthConfig, credential: AuthCredential):
        """Store authentication credential for a tool."""
        ctx = self.ensure_context(name)
        ctx.set_auth(auth_config, credential)
        print(f"[Toolkit] Stored credential for tool '{name}' key={auth_config.key}")

    def get_auth(self, name: str, auth_config: AuthConfig) -> Optional[AuthCredential]:
        """Retrieve authentication credential for a tool."""
        ctx = self.ensure_context(name)
        return ctx.get_auth(auth_config)