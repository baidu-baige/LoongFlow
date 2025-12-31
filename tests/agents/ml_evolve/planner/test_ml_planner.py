# -*- coding: utf-8 -*-
"""
This file define
"""
from pathlib import Path

import pytest

from agents.ml_evolve.planner.ml_planner import MLPlannerAgent
from evolux.evolve.context import Context, LLMConfig
from evolux.evolve.context.config import DatabaseConfig
from evolux.evolve.database import EvolveDatabase
from evolux.evolve.planner import Planner
from evolux.evolve.register import register_worker


@pytest.mark.asyncio
async def test_run():
    """test ml planner"""
    full_config = {
        "llm_config": LLMConfig(
            url="https://qianfan.baidubce.com/v2",
            api_key="xxx",
            model="deepseek-v3",
        ),
    }
    register_worker("ml_planner", "planner", MLPlannerAgent)

    db = EvolveDatabase.create_database(DatabaseConfig())

    planner = Planner("ml_planner", full_config, db)

    message = await planner.run(
        context=Context(
            base_path="./output",
            task=Path("./tests/agents/ml_evolve/resource/description.md").read_text(
                encoding="utf-8"
            ),
            metadata={
                "task_data_path": "./tests/agents/ml_evolve/resource",
            },
        ),
        message=None,
    )

    print(message)
