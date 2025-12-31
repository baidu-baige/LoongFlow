"""
iris evaluation
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


def run_evaluate(submission_file_path, data_dir):
    """
    Core evaluation function

    Args:
        submission_file_path: Path to user's submission file (Path object or string)
        data_dir: Root directory path for data

    Returns:
        score: Accuracy score (0.0 - 1.0, higher is better)
    """
    submission_file_path = Path(submission_file_path)
    data_dir = Path(data_dir)

    # Build answer file path
    answer_path = data_dir / "private" / "answer.csv"

    if not answer_path.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_path}")

    if not submission_file_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_file_path}")

    # Read files
    submission = pd.read_csv(submission_file_path)
    answer = pd.read_csv(answer_path)

    # Validate column names
    required_columns = ['id', 'species']
    if not all(col in submission.columns for col in required_columns):
        raise ValueError(f"Submission file must contain columns: {required_columns}. Found: {list(submission.columns)}")

    # Validate ID matching
    if not submission['id'].equals(answer['id']):
        # Try sorting by ID before comparing
        submission_sorted = submission.sort_values('id').reset_index(drop=True)
        answer_sorted = answer.sort_values('id').reset_index(drop=True)
        if not submission_sorted['id'].equals(answer_sorted['id']):
            raise ValueError("Submission IDs do not match answer IDs")
        submission = submission_sorted
        answer = answer_sorted

    # Validate sample count
    if len(submission) != len(answer):
        raise ValueError(f"Submission has {len(submission)} records, expected {len(answer)}")

    # Validate species names
    valid_species = ['setosa', 'versicolor', 'virginica']
    invalid_species = set(submission['species']) - set(valid_species)
    if invalid_species:
        raise ValueError(f"Invalid species names detected: {invalid_species}. Valid names: {valid_species}")

    # Calculate accuracy
    score = accuracy_score(answer['species'], submission['species'])

    return float(score)


def evaluate(task_data_path, best_code_path, artifacts):
    """
    Standard evaluation interface function

    Args:
        task_data_path: Path to task data file
        best_code_path: Path to best code (optional, not used in this task)
        artifacts: Dictionary containing submission file path and other info

    Returns:
        Evaluation result dictionary containing status, score, metrics, etc.
        Format:
        {
            "status": "success" | "validation_failed" | "execution_failed",
            "summary": str,
            "score": float (0.0-1.0, higher is better),
            "metrics": dict,
            "artifacts": dict
        }
    """
    # Validate submission file path
    if not artifacts.get("submission_file_path"):
        return {
            "status": "validation_failed",
            "summary": "No submission_file_path provided",
            "score": 0.0,
            "metrics": {},
            "artifacts": {
                "stderr": "Evaluation failed: submission_file_path is required in artifacts",
                "workflow_result": artifacts,
            },
        }

    submission_file_path = Path(artifacts["submission_file_path"])

    # Parse path structure: data_dir / public / task_data
    task_data_path = Path(task_data_path)
    data_dir = task_data_path.parent

    try:
        score = run_evaluate(submission_file_path, data_dir)

        return {
            "status": "success",
            "summary": f"Evaluation successful. Accuracy: {score:.4f} ({score * 100:.2f}%)",
            "score": score,
            "metrics": {
                "accuracy": score,
            },
            "artifacts": {
                "submission_file_path": str(submission_file_path),
            },
        }

    except FileNotFoundError as e:
        return {
            "status": "validation_failed",
            "summary": f"File not found: {str(e)}",
            "score": 0.0,
            "metrics": {},
            "artifacts": {
                "stderr": f"Evaluation failed: {str(e)}",
                "submission_file_path": str(submission_file_path),
            },
        }

    except ValueError as e:
        return {
            "status": "validation_failed",
            "summary": f"Validation error: {str(e)}",
            "score": 0.0,
            "metrics": {},
            "artifacts": {
                "stderr": f"Validation failed: {str(e)}",
                "submission_file_path": str(submission_file_path),
            },
        }

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
