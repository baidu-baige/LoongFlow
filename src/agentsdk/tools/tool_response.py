# -*- coding: utf-8 -*-
"""
This file provides the tool response.
"""

from dataclasses import dataclass
from typing import List

from agentsdk.message.elements import ContentElement


@dataclass
class ToolResponse:
    """The result chunk of a tool call."""

    content: List[ContentElement]
    """The execution output of the tool function."""
    
    err_msg: str = ""
    
    is_interrupted: bool = False
