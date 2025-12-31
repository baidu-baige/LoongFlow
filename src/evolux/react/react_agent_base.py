# -*- coding: utf-8 -*-
"""
This file define reAct AgentBase
"""
from abc import abstractmethod
from typing import Any, List

from agentsdk.message import Message
from evolux.base import AgentBase


class ReactAgentBase(AgentBase):
    """
    ReAct Agent base class
    """

    supported_hook_types: List[str] = ["pre_run", "post_run", "pre_reason", "post_reason", "pre_act", "post_act",
                                       "pre_observe", "post_observe"]

    def __init__(self):
        super().__init__()
        self._wrap_react_hooks()

    def _wrap_react_hooks(self):
        """
        Manually wrap methods that don't follow the base class's naming convention.
        """
        react_targets = {
            "reason": "_reason",
            "act": "_act",
            "observe": "_observe"
        }
        for target_name, internal_method_name in react_targets.items():
            func = getattr(self, internal_method_name, None)
            if not func:
                continue
            wrapped_func = self._wrap_with_hooks(func, target_name)

            setattr(self, internal_method_name, wrapped_func)

    async def run(self, *args, **kwargs) -> Message:
        """Main agent logic (must be implemented by subclasses)."""
        raise NotImplementedError()

    async def interrupt_impl(self):
        """Custom interruption logic (optional for subclasses)."""
        pass

    @abstractmethod
    async def _reason(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def _act(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def _observe(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
