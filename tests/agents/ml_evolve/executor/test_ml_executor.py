# -*- coding: utf-8 -*-
"""
This file define
"""
import uuid
from pathlib import Path

import pytest

from agents.ml_evolve.evaluator import MLEBenchEvaluator
from agents.ml_evolve.executor import MLExecutorAgent
from agentsdk.message import Message, MimeType, Role
from evolux.evolve.context import Context, EvaluatorConfig, LLMConfig
from evolux.evolve.context.config import DatabaseConfig
from evolux.evolve.database import EvolveDatabase
from evolux.evolve.executor import Executor
from evolux.evolve.register import register_worker


@pytest.mark.asyncio
async def test_run():
    """test ml executor"""
    full_config = {
        "llm_config": LLMConfig(
            url="https://qianfan.baidubce.com/v2",
            api_key="xxx",
            model="deepseek-v3",
        ),
    }

    mle_bench_evaluator = MLEBenchEvaluator(
        EvaluatorConfig(
            workspace_path="./output",
            evaluate_code="",
        )
    )

    register_worker("ml_executor", "executor", MLExecutorAgent)
    db = EvolveDatabase.create_database(DatabaseConfig())
    executor = Executor("ml_executor", full_config, mle_bench_evaluator, db)

    message = await executor.run(
        context=Context(
            base_path="./output",
            task=Path("./tests/agents/ml_evolve/resource/description.md").read_text(
                encoding="utf-8"
            ),
            metadata={
                "task_data_path": "./tests/agents/ml_evolve/resource",
            },
            task_id=uuid.UUID("3162773d-b2e0-468d-b0f6-aba4dacf7e68"),
            island_id=0,
            current_iteration=0,
        ),
        message=Message.from_media(
            sender="MLExecutorAgent",
            mime_type=MimeType.APPLICATION_JSON,
            role=Role.USER,
            data={
                "parent_info_file_path": "output/3162773d-b2e0-468d-b0f6-aba4dacf7e68/0/planner/parent_info.json",
                "best_plan_file_path": "output/3162773d-b2e0-468d-b0f6-aba4dacf7e68/0/planner/best_plan.txt",
                "eda_info_file_path": "output/3162773d-b2e0-468d-b0f6-aba4dacf7e68/0/planner/eda/eda_info.txt",
            },
        ),
    )

    print(message)
