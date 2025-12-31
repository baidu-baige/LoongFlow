#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides plan agent finalizer.
"""
import os
from os.path import exists

from agents.general_evolve.evolve_prompt.evolve_plan_prompt import (
    EVOLVE_PLANNER_SYSTEM_PROMPT,
    EVOLVE_PLANNER_SUMMARY_PROMPT,
)
from agentsdk.message import (
    ContentElement,
    Message,
    MimeType,
    Role,
)
from agentsdk.models import CompletionRequest
from evolux.react import AgentContext
from evolux.react.components import DefaultFinalizer


class PlanAgentFinalizer(DefaultFinalizer):
    """
    Plan Agent Finalizer that provides a tool for delivering the final answer.
    """

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """
        Summarizes the final answer if react loop exceeds max_steps.
        """
        system_message = Message.from_text(
            sender="finalizer",
            role=Role.SYSTEM,
            data=EVOLVE_PLANNER_SYSTEM_PROMPT,
        )
        history = await context.memory.get_memory()

        task = kwargs.get("task")
        parent_solution = kwargs.get("parent_solution")
        workspace = kwargs.get("workspace")

        plan_outline1 = ""
        if exists(os.path.join(workspace, "plan_1.txt")):
            with open(
                os.path.join(workspace, "plan_1.txt"), "r", encoding="utf-8"
            ) as f:
                plan_outline1 = f.read()

        plan_outline2 = ""
        if exists(os.path.join(workspace, "plan_2.txt")):
            with open(
                os.path.join(workspace, "plan_2.txt"), "r", encoding="utf-8"
            ) as f:
                plan_outline2 = f.read()

        plan_outline3 = ""
        if exists(os.path.join(workspace, "plan_3.txt")):
            with open(
                os.path.join(workspace, "plan_3.txt"), "r", encoding="utf-8"
            ) as f:
                plan_outline3 = f.read()

        hint_message = Message.from_text(
            sender="finalizer",
            role=Role.USER,
            data=EVOLVE_PLANNER_SUMMARY_PROMPT.format(
                task_info=task,
                parent_solution=parent_solution,
                plan_outline1=plan_outline1,
                plan_outline2=plan_outline2,
                plan_outline3=plan_outline3,
            ),
        )

        messages = [system_message] + history + [hint_message]

        resp_generator = self.model.generate(
            CompletionRequest(
                messages=messages, tools=context.toolkit.get_declarations()
            )
        )

        resp = await anext(resp_generator)
        if resp.error_code:
            raise Exception(
                f"Error code: {resp.error_code}, error: {resp.error_message}"
            )

        generated_plan = ""
        for element in resp.content:
            if isinstance(element, ContentElement):
                generated_plan = element.data

        return Message.from_content(
            sender="finalizer",
            role=Role.ASSISTANT,
            data=generated_plan,
            mime_type=MimeType.TEXT_PLAIN,
        )
