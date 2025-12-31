# -*- coding: utf-8 -*-
"""
This file define
"""
from pathlib import Path

import pytest

from agents.ml_evolve.evocoder.evaluator import EDAEvaluator, EvoCoderEvaluatorConfig
from agents.ml_evolve.evocoder.evocoder import EvoCoder, EvoCoderConfig
from agents.ml_evolve.evocoder.stage_context_provider import EDAContextProvider
from agentsdk.message import Message, MimeType, Role
from evolux.evolve.context import LLMConfig


@pytest.mark.asyncio
async def test_run():
    """Create EvoCoder instance"""

    config = EvoCoderConfig(
        llm_config=LLMConfig(
            url="https://qianfan.baidubce.com/v2",
            api_key="xxx",
            model="deepseek-v3",
        ),
        context_provider=EDAContextProvider(),
        max_rounds=10,
        evaluator=EDAEvaluator(
            config=EvoCoderEvaluatorConfig(
                workspace_path="./output",
            )
        ),
    )

    agent = EvoCoder(config=config)
    message = await agent.run(
        Message.from_media(
            sender="EvoCoder",
            mime_type=MimeType.APPLICATION_JSON,
            role=Role.USER,
            data={
                "task_data_path": "./tests/agents/ml_evolve/resource",
                "task_description": Path(
                    "./tests/agents/ml_evolve/resource/description.md"
                ).read_text(encoding="utf-8"),
            },
        )
    )
    print(message)
