# -*- coding: utf-8 -*-
"""
This file define
"""
from pathlib import Path

import numpy as np
import pandas as pd
from mlebench.data import get_leaderboard, is_dataset_prepared
from mlebench.registry import registry
from mlebench.utils import (
    load_answers,
    read_csv,
)


def normalize_score(leaderboard: pd.DataFrame, my_score: float, is_lower_better: bool = False) -> float:
    """
    normalize score to [0,1]; score larger, result better
    """
    scores = leaderboard['score'].values.copy()

    if is_lower_better:
        ref_point = scores.max() + (scores.max() - scores.min()) * 0.1
        scores = ref_point - scores
        my_score = ref_point - my_score

    unique_scores = np.unique(scores)[::-1]
    best = unique_scores[0]
    worst = unique_scores[-1]

    def _smooth_transform(pct):
        if pct >= 0.99:
            return 0.95 + 0.05 * (pct - 0.99) / 0.01
        elif pct >= 0.90:
            return 0.80 + 0.15 * (pct - 0.90) / 0.09
        elif pct >= 0.50:
            return 0.35 + 0.45 * (pct - 0.50) / 0.40
        else:
            return 0.10 + 0.25 * (pct / 0.50)

    if best == worst:
        return 0.5

    if my_score >= best:
        return 1.0

    if my_score <= worst:
        last_rank_score = _smooth_transform(0.0)
        if my_score == worst:
            return last_rank_score
        deficit = (worst - my_score) / (best - worst)
        return max(0.0, last_rank_score * (1 - deficit))

    upper_idx = np.searchsorted(-unique_scores, -my_score, side='right') - 1
    upper_idx = max(0, upper_idx)
    lower_idx = upper_idx + 1

    if lower_idx >= len(unique_scores):
        lower_idx = len(unique_scores) - 1

    upper = unique_scores[upper_idx]
    lower = unique_scores[lower_idx]

    if upper == lower:
        interval_ratio = 0.5
    else:
        interval_ratio = (my_score - lower) / (upper - lower)

    n_unique = len(unique_scores)
    upper_rank = 1.0 - upper_idx / (n_unique - 1)
    lower_rank = 1.0 - lower_idx / (n_unique - 1)

    raw_rank = lower_rank + interval_ratio * (upper_rank - lower_rank)

    return min(_smooth_transform(raw_rank), 0.99)


def run_evaluate(submission_path: Path, data_dir: Path, competition_id: str):
    """
    run evaluate for mlebench
    """
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".csv"
    if not submission_exists:
        raise ValueError(
            f"Invalid submission file: {submission_path}. Please check that the file exists and it is a CSV."
        )

    new_registry = registry.set_data_dir(data_dir)
    competition = new_registry.get_competition(competition_id)

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
        )

    submission_df = read_csv(submission_path)
    answers = load_answers(competition.answers)

    score = competition.grader(submission_df, answers)
    if score is None:
        raise ValueError(f"Score for competition `{competition.id}` was not found.")

    competition_leaderboard = get_leaderboard(competition)
    lower_better = competition.grader.is_lower_better(competition_leaderboard)
    rank_info = competition.grader.rank_score(score, competition_leaderboard)

    rank = {
        "gold_medal": bool(rank_info["gold_medal"]),
        "silver_medal": bool(rank_info["silver_medal"]),
        "bronze_medal": bool(rank_info["bronze_medal"]),
        "above_median": bool(rank_info["above_median"]),
        "leaderboard_score": score,
        "leaderboard_gold": rank_info["gold_threshold"],
        "leaderboard_silver": rank_info["silver_threshold"],
        "leaderboard_bronze": rank_info["bronze_threshold"],
        "leaderboard_median": rank_info["median_threshold"]
    }

    if rank["gold_medal"]:
        return 1.0,rank

    return normalize_score(competition_leaderboard, score, lower_better), rank


def evaluate(task_data_path, best_code_path, artifacts):
    """
    common evaluate function
    """
    if not artifacts["submission_file_path"]:
        return {
            "status": "validation_failed",
            "summary": f"No submission_file_path provided",
            "score": 0.0,
            "metrics": {},
            "artifacts": {
                "stderr": f"Evaluation failed completely",
                "workflow_result": artifacts,
            },
        }
    submission_file_path = Path(artifacts["submission_file_path"])
    data_dir = Path(task_data_path).parent.parent.parent
    competition_id = Path(task_data_path).parent.parent.name
    try:
        score, rank_info = run_evaluate(submission_file_path, data_dir, competition_id)
    except Exception as e:
        return {
            "status": "execution_failed",
            "summary": f"Program execution failed: {str(e)}",
            "score": 0.0,
            "metrics": {},
            "artifacts": {
                "stderr": f"Evaluation failed completely: {str(e)}",
                "submission_file_path": str(submission_file_path),
            },
        }
    return {
        "status": "success",
        "summary": f"Evaluation successful",
        "score": score,
        "metrics": {
            "norm_score": score,
        },
        "artifacts": {
            "rank_info": rank_info,
            "submission_file_path": str(submission_file_path),
        },
    }