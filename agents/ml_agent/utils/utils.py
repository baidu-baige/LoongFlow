# -*- coding: utf-8 -*-
"""
This file define utils
"""

from pathlib import Path

from loongflow.framework.pes.context import Context, Workspace

EDA_INFO_NAME = "eda_info.txt"
EDA_CODE_NAME = "eda.py"
MODEL_ASSEMBLE_NAME = "model_assemble.json"
STRATEGIC_ANALYSIS_NAME = "strategic_analysis.txt"
SUMMARY_ANALYSIS_NAME = "summary_analysis.txt"


def get_ml_executor_output_path(context: Context, create: bool = True) -> Path:
    """
    get ml executor output path
    """
    path = Path(Workspace.get_executor_path(context) / "output")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_ml_executor_best_code_path(context: Context, create: bool = True) -> Path:
    """
    get ml executor bets code path
    """
    path = Path(Workspace.get_executor_path(context) / "best_code")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_evocoder_evaluate_path(context: Context, stage: str, create: bool = True) -> Path:
    """
    get evocoder evaluate path
    """
    base_path = Path(context.base_path)
    path = (
            base_path
            / str(context.task_id)
            / str(context.current_iteration)
            / "evocoder"
            / stage
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_eda_path(context: Context, create: bool = True) -> Path:
    """
    get latest eda path
    """
    base_path = Path(context.base_path)
    path = (
            base_path
            / str(context.task_id)
            / "eda"
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_eda_info_path(context: Context) -> Path:
    """
    get latest eda info path
    """
    return get_latest_eda_path(context) / EDA_INFO_NAME


def get_latest_eda_info(context: Context) -> str:
    """
    get latest eda info
    """
    try:
        return get_latest_eda_info_path(context).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_latest_eda_info(context: Context, eda_info: str) -> None:
    """
    Write latest eda info.
    :param context: task execution context
    :param eda_info: eda info string
    """
    eda_info_path = get_latest_eda_info_path(context)
    eda_info_path.write_text(eda_info, encoding="utf-8")


def get_latest_eda_code_path(context: Context) -> Path:
    """
    get latest eda code path
    """
    return get_latest_eda_path(context) / EDA_CODE_NAME


def get_latest_eda_code(context: Context) -> str:
    """
    get latest eda code
    """
    try:
        return get_latest_eda_code_path(context).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_latest_eda_code(context: Context, eda_code: str) -> None:
    """
    Write latest eda code.
    :param context: task execution context
    :param eda_code: eda code string
    """
    eda_code_path = get_latest_eda_code_path(context)
    eda_code_path.write_text(eda_code, encoding="utf-8")


def get_current_eda_path(context: Context, create: bool = True) -> Path:
    """
    get current eda path
    """
    base_path = Path(context.base_path)
    path = (
            base_path
            / str(context.task_id)
            / str(context.current_iteration)
            / "planner"
            / "eda"
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_current_eda_info_path(context: Context) -> Path:
    """
    get current eda info path
    """
    return get_current_eda_path(context) / EDA_INFO_NAME


def get_current_eda_code_path(context: Context) -> Path:
    """
    get current eda code path
    """
    return get_current_eda_path(context) / EDA_CODE_NAME


def write_current_eda_info(context: Context, eda_info: str) -> None:
    """
    Write current eda info.
    :param context: task execution context
    :param eda_info: eda info string
    """
    get_current_eda_info_path(context).write_text(eda_info, encoding="utf-8")


def write_current_eda_code(context: Context, eda_code: str) -> None:
    """
    Write current eda info.
    :param context: task execution context
    :param eda_code: eda code string
    """
    get_current_eda_code_path(context).write_text(eda_code, encoding="utf-8")


def get_assemble_model_path(context: Context) -> Path:
    """
    get model assemble path
    """
    return Workspace.get_planner_path(context) / MODEL_ASSEMBLE_NAME


def write_assemble_model_info(context: Context, assemble_model_info: str) -> None:
    """
    Write model ensemble info.
    :param context: task execution context
    :param assemble_model_info: model ensemble info string
    """
    get_assemble_model_path(context).write_text(assemble_model_info, encoding="utf-8")


def get_strategic_analysis_path(context: Context) -> Path:
    """
    get strategic analysis path
    """
    return Workspace.get_planner_path(context) / STRATEGIC_ANALYSIS_NAME


def write_strategic_analysis_info(context: Context, analysis_info: str) -> None:
    """
    Write strategic analysis info.
    :param context: task execution context
    :param analysis_info: analysis info string
    """
    get_strategic_analysis_path(context).write_text(analysis_info, encoding="utf-8")


def get_summary_analysis_path(context: Context) -> Path:
    """
    get summary analysis path
    """
    return Workspace.get_summarizer_path(context) / SUMMARY_ANALYSIS_NAME


def write_summary_analysis_info(context: Context, analysis_info: str) -> None:
    """
    Write summary analysis info.
    :param context: task execution context
    :param analysis_info: analysis info string
    """
    get_summary_analysis_path(context).write_text(analysis_info, encoding="utf-8")
