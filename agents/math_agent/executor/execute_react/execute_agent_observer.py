#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides ExecuteAgentObserver
"""
import json
from typing import List, Dict

from loongflow.agentsdk.logger.logger import get_logger
from loongflow.agentsdk.message import Message
from loongflow.agentsdk.message.elements import ToolOutputElement
from loongflow.agentsdk.message.message import Role
from loongflow.framework.react import AgentContext
from loongflow.framework.react.components import DefaultObserver

logger = get_logger(__name__)

class ToolOutputException(Exception):
    """Raised when a tool output matches an exception rule."""
    pass

class ExecuteAgentObserver(DefaultObserver):
    """
    An observer that stops the agent if a tool output matches an exception rule.
    """
    def __init__(self, exception_rules: List[Dict[str, str]]):
        super().__init__()
        self.exception_rules = exception_rules

    async def observe(self, context: AgentContext, tool_outputs: List[Message]) -> Message | None:
        """Override observe method to implement custom logic."""
        for msg in tool_outputs:
            if msg.role != Role.TOOL:
                continue

            elems = msg.get_elements(ToolOutputElement)
            if not elems:
                continue

            for te in elems:
                tool_name = te.tool_name
                parsed = te.get_content()

                # score > 0 rule
                if tool_name == "evaluate_solution":
                    result = te.result[0].data if (te.result and te.result[0] and te.result[0].data) else None
                    score = result.get("score") if result else None

                    if isinstance(score, (int, float)) and score > 0:
                        logger.debug(f"[Observer] score > 0, score={score}, stop")
                        raise ToolOutputException(f"score>0 ({score}), stop")

                    logger.debug("[Observer] score<=0 or missing")

                # exception_rules matching
                for rule in self.exception_rules:
                    if rule["tool_name"] != tool_name:
                        continue

                    trigger = rule["exception_out"] if rule["exception_out"] else rule["trigger"]

                    for item in parsed:
                        s = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                        if trigger in s:
                            raise ToolOutputException(f"trigger matched: {trigger}")

                    logger.debug("[Observer] no trigger match")

        return None