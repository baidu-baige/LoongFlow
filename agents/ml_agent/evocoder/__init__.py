# -*- coding: utf-8 -*-
"""
This file define evo coder
"""

from typing import Type

from agents.ml_agent.evocoder.evaluator import (
    EDAEvaluator,
    EnsembleEvaluator,
    EvoCoderEvaluator,
    EvoCoderEvaluatorConfig,
    GetSplitterEvaluator,
    LoadDataEvaluator,
    PreprocessEvaluator,
    TrainAndPredictEvaluator,
    WorkflowEvaluator,
)
from agents.ml_agent.evocoder.evocoder import EvoCoder, EvoCoderConfig
from agents.ml_agent.evocoder.stage_context_provider import (
    EDAContextProvider,
    EnsembleContextProvider,
    GetSplitterContextProvider,
    LoadDataContextProvider,
    PreprocessContextProvider,
    Stage,
    StageContextProvider,
    TaskConfig,
    TrainAndPredictContextProvider,
    WorkflowContextProvider,
)

__all__ = [
    "Stage",
    "EvoCoder",
    "EvoCoderEvaluator",
    "EvoCoderConfig",
    "StageContextProvider",
    "EDAContextProvider",
    "EnsembleContextProvider",
    "GetSplitterContextProvider",
    "LoadDataContextProvider",
    "PreprocessContextProvider",
    "TrainAndPredictContextProvider",
    "WorkflowContextProvider",
    "TaskConfig",
    "EDAEvaluator",
    "EnsembleEvaluator",
    "GetSplitterEvaluator",
    "LoadDataEvaluator",
    "PreprocessEvaluator",
    "TrainAndPredictEvaluator",
    "WorkflowEvaluator",
    "STAGE_PROVIDERS",
    "STAGE_EVALUATORS",
    "EvoCoderEvaluatorConfig",
]

STAGE_PROVIDERS: dict[Stage, Type[StageContextProvider]] = {
    Stage.EDA: EDAContextProvider,
    Stage.LOAD_DATA: LoadDataContextProvider,
    Stage.GET_SPLITTER: GetSplitterContextProvider,
    Stage.PREPROCESS: PreprocessContextProvider,
    Stage.TRAIN_AND_PREDICT: TrainAndPredictContextProvider,
    Stage.ENSEMBLE: EnsembleContextProvider,
    Stage.WORKFLOW: WorkflowContextProvider,
}

STAGE_EVALUATORS: dict[Stage, Type[EvoCoderEvaluator]] = {
    Stage.EDA: EDAEvaluator,
    Stage.LOAD_DATA: LoadDataEvaluator,
    Stage.GET_SPLITTER: GetSplitterEvaluator,
    Stage.PREPROCESS: PreprocessEvaluator,
    Stage.TRAIN_AND_PREDICT: TrainAndPredictEvaluator,
    Stage.ENSEMBLE: EnsembleEvaluator,
    Stage.WORKFLOW: WorkflowEvaluator,
}
