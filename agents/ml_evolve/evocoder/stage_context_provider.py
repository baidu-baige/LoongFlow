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

from agents.ml_evolve.prompt.evocoder import (
    CreateFeaturesPrompts,
    CrossValidationPrompts,
    EDAPrompts,
    EnsemblePrompts,
    LoadDataPrompts,
    TrainAndPredictPrompts,
    WorkflowPrompts,
)
from agentsdk.message import Message, Role


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
    CROSS_VALIDATION = "cross_validation"
    CREATE_FEATURES = "create_features"
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
                    "eda_code": task_config.code_deps.get(
                        "eda", "# Eda code not available"
                    ),
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


class CrossValidationContextProvider(StageContextProvider):
    """Provides context for the 'cross_validation' stage."""

    def stage(self) -> Stage:
        return Stage.CROSS_VALIDATION

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(CrossValidationPrompts.SYSTEM)
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
            .from_string(CrossValidationPrompts.USER)
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


class CreateFeaturesContextProvider(StageContextProvider):
    """Provides context for the 'create_features' stage."""

    def stage(self) -> Stage:
        return Stage.CREATE_FEATURES

    def provide(self, task_config: TaskConfig) -> List[Message]:
        system_prompt = (
            Environment()
            .from_string(CreateFeaturesPrompts.SYSTEM)
            .render(
                {
                    "task_description": task_config.task_description,
                    "eda_analysis": task_config.eda_analysis,
                    "load_data_code": task_config.code_deps.get(
                        "load_data", "# Data loader code not available"
                    ),
                    "cross_validation_code": task_config.code_deps.get(
                        "cross_validation", "# Cross Validation code not available"
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
            .from_string(CreateFeaturesPrompts.USER)
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
                    "cross_validation_code": task_config.code_deps.get(
                        "cross_validation", "# Cross Validation code not available"
                    ),
                    "feature_code": task_config.code_deps.get(
                        "create_features", "# Feature engineering code not available"
                    ),
                    "task_data_path": task_config.task_data_path,
                    "output_data_path": task_config.workspace_path,
                    "gpu_available": task_config.gpu_available,
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
                        "create_features", "# create_features code not available"
                    ),
                    "cross_validation_code": task_config.code_deps.get(
                        "cross_validation", "# cross_validation code not available"
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
                        "create_features", "# create_features code not available"
                    ),
                    "cross_validation_code": task_config.code_deps.get(
                        "cross_validation", "# cross_validation code not available"
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
