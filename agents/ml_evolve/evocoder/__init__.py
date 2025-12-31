# -*- coding: utf-8 -*-
"""
This file define evo coder
"""
from typing import Type

from agents.ml_evolve.evocoder.evaluator import (CreateFeaturesEvaluator, CrossValidationEvaluator, EDAEvaluator,
                                                 EnsembleEvaluator, EvoCoderEvaluator, EvoCoderEvaluatorConfig,
                                                 LoadDataEvaluator, TrainAndPredictEvaluator, WorkflowEvaluator)
from agents.ml_evolve.evocoder.evocoder import EvoCoder, EvoCoderConfig
from agents.ml_evolve.evocoder.stage_context_provider import (CreateFeaturesContextProvider,
                                                              CrossValidationContextProvider, EDAContextProvider,
                                                              EnsembleContextProvider, LoadDataContextProvider,
                                                              Stage, StageContextProvider, TaskConfig,
                                                              TrainAndPredictContextProvider, WorkflowContextProvider)

__all__ = [
    "Stage",
    "EvoCoder",
    "EvoCoderEvaluator",
    "EvoCoderConfig",
    "StageContextProvider",
    "CreateFeaturesContextProvider",
    "CrossValidationContextProvider",
    "EDAContextProvider",
    "EnsembleContextProvider",
    "LoadDataContextProvider",
    "TrainAndPredictContextProvider",
    "WorkflowContextProvider",
    "TaskConfig",
    "CreateFeaturesEvaluator",
    "CrossValidationEvaluator",
    "EDAEvaluator",
    "EnsembleEvaluator",
    "LoadDataEvaluator",
    "TrainAndPredictEvaluator",
    "WorkflowEvaluator",
    "STAGE_PROVIDERS",
    "STAGE_EVALUATORS",
    "EvoCoderEvaluatorConfig"
]

STAGE_PROVIDERS: dict[Stage, Type[StageContextProvider]] = {
    Stage.EDA: EDAContextProvider,
    Stage.LOAD_DATA: LoadDataContextProvider,
    Stage.CROSS_VALIDATION: CrossValidationContextProvider,
    Stage.CREATE_FEATURES: CreateFeaturesContextProvider,
    Stage.TRAIN_AND_PREDICT: TrainAndPredictContextProvider,
    Stage.ENSEMBLE: EnsembleContextProvider,
    Stage.WORKFLOW: WorkflowContextProvider,
}

STAGE_EVALUATORS: dict[Stage, Type[EvoCoderEvaluator]] = {
    Stage.EDA: EDAEvaluator,
    Stage.LOAD_DATA: LoadDataEvaluator,
    Stage.CROSS_VALIDATION: CrossValidationEvaluator,
    Stage.CREATE_FEATURES: CreateFeaturesEvaluator,
    Stage.TRAIN_AND_PREDICT: TrainAndPredictEvaluator,
    Stage.ENSEMBLE: EnsembleEvaluator,
    Stage.WORKFLOW: WorkflowEvaluator,
}
