#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file defines message, common abstraction for information in the framework
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from agentsdk.message import ContentElement, Element, ElementT, MimeType, ThinkElement, ToolCallElement, \
    ToolOutputElement, ToolStatus


class Role(str, Enum):
    """Enumeration for the roles of message senders."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """
    Single message in a conversation.
    """

    id: uuid.UUID = Field(
            default_factory=uuid.uuid4, description="The unique identifier for the message."
    )
    timestamp: datetime = Field(
            default_factory=datetime.utcnow,
            description="Timestamp when the message was created.",
    )
    trace_id: str = Field(
            default="",
            description="An optional ID for tracing a request across multiple services.",
    )
    conversation_id: str = Field(
            default="",
            description="An optional ID to group messages within a conversation.",
    )
    role: Role | str = Field(
            ...,
            description="The role of the message sender, e.g., 'user', 'assistant', 'system', 'tool'.",
    )
    sender: str = Field(
            default="",
            description="The specific identity of the sender, e.g., 'user', 'WeatherAgent'.",
    )
    metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="A dictionary for ancillary information, such as token usage details.",
    )
    content: List[Element] = Field(
            ..., description="Min body of the message, composed of Element objects."
    )

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Serializes the Message instance to a dictionary."""
        return self.model_dump(mode="json", **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "Message":
        """Creates a Message instance from a dictionary."""
        return cls.model_validate(data, **kwargs)

    @classmethod
    def from_text(cls,
                  data: str,
                  sender: str = "",
                  role: Role | str = Role.USER,
                  mime_type: MimeType | str = MimeType.TEXT_PLAIN,
                  **kwargs) -> "Message":
        """Creates a Message instance from a text string."""
        return cls(role=role, sender=sender, content=[ContentElement(mime_type=mime_type, data=data)], **kwargs)

    @classmethod
    def from_content(cls,
                     data: Any,
                     mime_type: MimeType | str,
                     sender: str = "",
                     role: Role | str = Role.USER,
                     **kwargs) -> "Message":
        """Creates a Message instance from content."""
        return cls(role=role, sender=sender, content=[ContentElement(mime_type=mime_type, data=data)], **kwargs)

    @classmethod
    def from_tool_call(cls,
                       target: str,
                       arguments: Dict[str, Any],
                       sender: str = "",
                       role: Role | str = Role.ASSISTANT,
                       **kwargs) -> "Message":
        """Creates a Message instance from a tool call."""
        return cls(role=role, sender=sender, content=[ToolCallElement(target=target, arguments=arguments)], **kwargs)

    @classmethod
    def from_tool_output(cls,
                         call_id: uuid.UUID,
                         tool_name: str,
                         status: ToolStatus,
                         result: List[ContentElement],
                         sender: str,
                         role: Role | str = Role.TOOL,
                         **kwargs) -> "Message":
        """Creates a Message instance from a tool output."""
        return cls(
                role=role,
                sender=sender,
                content=[
                    ToolOutputElement(
                            call_id=call_id,
                            tool_name=tool_name,
                            status=status,
                            result=result
                    )
                ],
                **kwargs)

    @classmethod
    def from_think(cls,
                   content: Any,
                   sender: str = "",
                   role: Role | str = Role.ASSISTANT,
                   **kwargs) -> "Message":
        """Creates a Message instance from a thought."""
        return cls(role=role, sender=sender, content=[ThinkElement(content=content)], **kwargs)

    @classmethod
    def from_elements(cls,
                      elements: List[Element],
                      sender: str = "",
                      role: Role | str = Role.ASSISTANT,
                      **kwargs) -> "Message":
        """Creates a Message instance from a list of elements."""
        return cls(role=role, sender=sender, content=elements, **kwargs)

    @classmethod
    def from_media(cls,
                   sender: str,
                   mime_type: "MimeType | str",
                   data: Any,
                   role: Role | str,
                   **kwargs) -> "Message":
        """Creates a Message instance from media."""
        return cls(role=role, sender=sender, content=[ContentElement(mime_type=mime_type, data=data)], **kwargs)

    def get_elements(self, element_cls: Type[ElementT]) -> List[ElementT]:
        """
        Filters the message content for a specific element class.
        Args:
            element_cls: The class of the element to retrieve
                         (e.g., ContentElement).
        Returns:
            A list containing only instances of the specified element class.
        """
        return [element for element in self.content if isinstance(element, element_cls)]
