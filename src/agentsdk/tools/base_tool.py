# -*- coding: utf-8 -*-
"""
This file provides the base class for all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict

from agentsdk.tools.tool_context import ToolContext
from agentsdk.tools.tool_response import ToolResponse


class FunctionDeclarationDict(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: str
    
class BaseTool(ABC):
    """the base class for all tools."""
   
    name: str
    """The name of the tool."""
    description: str
    """The description of the tool."""

    def __init__(self, *, name, description):
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_declaration(self) -> Optional[FunctionDeclarationDict]:
        """ Gets the function declaration of the tool. """
        raise NotImplementedError

    @abstractmethod
    async def arun(
        self,
        *,
        args: Dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool asynchronously."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        args: Dict[str, Any],
        tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool synchronously."""
        raise NotImplementedError