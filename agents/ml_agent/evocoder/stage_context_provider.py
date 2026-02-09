# -*- coding: utf-8 -*-
"""
This file defines the StageContextProvider, which provides the initial context
(system and user prompts) for EvoCoder for each specific machine learning stage.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List

from jinja2 import Environment

from agents.ml_agent.prompt.evocoder import (
    EDAPrompts,
    EnsemblePrompts,
    GetSplitterPrompts,
    LoadDataPrompts,
    PreprocessPrompts,
    TrainAndPredictPrompts,
    WorkflowPrompts,
)
from loongflow.agentsdk.message import Message, Role


@dataclass
class TaskConfig:
    """
    task config for EvoCoder
    """

    task_description: str = None
    task_data_path: str = None
    eda_analysis: str = ""
    plan: str = ""
    parent_code: str = ""
    code_deps: dict[str, Any] = field(default_factory=dict)
    assemble_plan: str = ""
    assemble_models: dict[str, str] = field(default_factory=dict)
    workspace_path: str = ""
    gpu_available: bool = False
    gpu_count: int = 0
    hardware_info: str = ""
    task_dir_structure: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the dataclass instance to a dictionary.
        """
        return asdict(self)


class Stage(str, Enum):
    """
    machine learning stage
    """

    EDA = "eda"
    LOAD_DATA = "load_data"
    GET_SPLITTER = "get_splitter"
    PREPROCESS = "preprocess"
    TRAIN_AND_PREDICT = "train_and_predict"
    ENSEMBLE = "ensemble"
    WORKFLOW = "workflow"


class StageContextProvider(ABC):
    """
    Interface for providing the initial conversation context for a specific
    stage in the machine learning pipeline.
    """

    @abstractmethod
    def stage(self) -> Stage:
        """
        Returns the current stage of the pipeline.
        """
        pass

    @abstractmethod
    def provide(self, task_config: TaskConfig) -> List[Message]:
        """
        Provides the initial system and user messages based on the task configuration.
        """
        pass


class EDAContextProvider(StageContextProvider):
    """Provides context for the 'eda' stage."""

    def stage(self) -> Stage:
        return Stage.EDA

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(EDAPrompts.SYSTEM)
            .render(
                {
                    "task_data_path": task_config.task_data_path,
                    "gpu_available": task_config.gpu_available,
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(EDAPrompts.USER)
            .render(
                {
                    "task_description": task_config.task_description,
                    "plan": task_config.plan,
                    "task_data_path": task_config.task_data_path,
                    "reference_code": task_config.code_deps.get(
                        "eda", "# Eda code not available"
                    ),
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                }
            )
        )

        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class LoadDataContextProvider(StageContextProvider):
    """Provides context for the 'load_data' stage."""

    def stage(self) -> Stage:
        return Stage.LOAD_DATA

    def provide(self, task_config: TaskConfig) -> List[Message]:
        data_num = 50
        if task_config.gpu_available:
            data_num = 200
        system_prompt = (
            Environment()
            .from_string(LoadDataPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                    "data_num": data_num,
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(LoadDataPrompts.USER)
            .render(
                {
                    "plan": task_config.plan,
                    "data_num": data_num,
                    "parent_code": task_config.parent_code,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                }
            )
        )
        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class GetSplitterContextProvider(StageContextProvider):
    """Provides context for the 'get_splitter' stage."""

    def stage(self) -> Stage:
        return Stage.GET_SPLITTER

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(GetSplitterPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# Data loader code not available"
                    ),
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(GetSplitterPrompts.USER)
            .render(
                {
                    "plan": task_config.plan,
                    "parent_code": task_config.parent_code,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                }
            )
        )

        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class PreprocessContextProvider(StageContextProvider):
    """Provides context for the 'preprocess' stage."""

    def stage(self) -> Stage:
        return Stage.PREPROCESS

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(PreprocessPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# Data loader code not available"
                    ),
                    "get_splitter_code": task_config.code_deps.get(
                        "get_splitter", "# Get Splitter code not available"
                    ),
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(PreprocessPrompts.USER)
            .render(
                {
                    "plan": task_config.plan,
                    "parent_code": task_config.parent_code,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                }
            )
        )

        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class TrainAndPredictContextProvider(StageContextProvider):
    """Provides context for the 'train_and_predict' stage."""

    def stage(self) -> Stage:
        return Stage.TRAIN_AND_PREDICT

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(TrainAndPredictPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# Data loader code not available"
                    ),
                    "get_splitter_code": task_config.code_deps.get(
                        "get_splitter", "# Get Splitter code not available"
                    ),
                    "feature_code": task_config.code_deps.get(
                        "preprocess", "# Feature engineering code not available"
                    ),
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "gpu_count": task_config.gpu_count,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                }
            )
        )

        template = Environment().from_string(TrainAndPredictPrompts.USER)
        template.globals["tojson"] = json.dumps
        user_prompt = template.render(
            {
                "train_plan": task_config.plan,
                "parent_code": task_config.parent_code,
                "assemble_models": task_config.assemble_models,
                "assemble_plan": task_config.assemble_plan,
                "task_data_path": task_config.task_data_path,
                "output_data_path": task_config.workspace_path,
            }
        )

        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class EnsembleContextProvider(StageContextProvider):
    """Provides context for the 'ensemble' stage."""

    def stage(self) -> Stage:
        return Stage.ENSEMBLE

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(EnsemblePrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# load_data code not available"
                    ),
                    "feature_code": task_config.code_deps.get(
                        "preprocess", "# preprocess code not available"
                    ),
                    "get_splitter_code": task_config.code_deps.get(
                        "get_splitter", "# get_splitter code not available"
                    ),
                    "model_code": task_config.code_deps.get(
                        "train_and_predict", "# train_and_predict code not available"
                    ),
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(EnsemblePrompts.USER)
            .render(
                {
                    "plan": task_config.plan,
                    "parent_code": task_config.parent_code,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                }
            )
        )
        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]


class WorkflowContextProvider(StageContextProvider):
    """Provides context for the 'workflow' stage."""

    def stage(self) -> Stage:
        return Stage.WORKFLOW

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(WorkflowPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
                    "hardware_info": task_config.hardware_info,
                    "task_dir_structure": task_config.task_dir_structure,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# load_data code not available"
                    ),
                    "feature_code": task_config.code_deps.get(
                        "preprocess", "# preprocess code not available"
                    ),
                    "get_splitter_code": task_config.code_deps.get(
                        "get_splitter", "# get_splitter code not available"
                    ),
                    "model_code": task_config.code_deps.get(
                        "train_and_predict", "# train_and_predict code not available"
                    ),
                    "ensemble_code": task_config.code_deps.get(
                        "ensemble", "# ensemble code not available"
                    ),
                }
            )
        )

        user_prompt = (
            Environment()
            .from_string(WorkflowPrompts.USER)
            .render(
                {
                    "plan": task_config.plan,
                    "parent_code": task_config.parent_code,
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                }
            )
        )

        return [
            Message.from_text(
                sender="ContextProvider", role=Role.SYSTEM, data=system_prompt
            ),
            Message.from_text(
                sender="ContextProvider", role=Role.USER, data=user_prompt
            ),
        ]
