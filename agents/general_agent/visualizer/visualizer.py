#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask web server for General Agent task visualizer.

Provides real-time monitoring of General Agent tasks, including:
- Task progress tracking
- Iteration history
- Generated code viewing
- Score evolution charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent.parent.parent.parent


class GeneralAgentService:
    """Service for loading and managing General Agent task data."""

    def __init__(self, workspace_paths: List[str]):
        """
        Initialize service with workspace paths.

        Args:
            workspace_paths: List of output directory paths (e.g., ["./output-todo-list"])
        """
        self.workspaces = []
        for workspace in workspace_paths:
            path = Path(workspace)
            if not path.is_absolute():
                path = BASE_DIR / path
            path = path.resolve()
            if path.exists():
                self.workspaces.append(path)
            else:
                print(f"Warning: Workspace not found: {path}")

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        List all available tasks across all workspaces.

        Returns:
            List of task info dicts with id, name, workspace, status, etc.
        """
        tasks = []
        # Skip these directories (not task directories)
        skip_dirs = {"database", "logs", "checkpoints", ".git"}

        for workspace in self.workspaces:
            for task_dir in workspace.iterdir():
                # Skip non-directories and special directories
                if not task_dir.is_dir() or task_dir.name in skip_dirs:
                    continue

                # Check if this looks like a task directory (has iteration subdirs)
                if not any(task_dir.glob("iteration_*")) and not any(task_dir.glob("[0-9]*")):
                    continue

                task_info = self._get_task_info(workspace, task_dir)
                if task_info:
                    tasks.append(task_info)

        # Sort by timestamp (newest first)
        tasks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return tasks

    def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific task.

        Args:
            task_id: Format "workspace_name/task_timestamp"

        Returns:
            Dict containing task metadata, iterations, scores, etc.
        """
        workspace_name, task_name = self._parse_task_id(task_id)
        task_path = self._find_task_path(workspace_name, task_name)

        if not task_path:
            raise ValueError(f"Task not found: {task_id}")

        iterations = self._list_iterations(task_path)
        score_history = self._build_score_history(task_path, iterations)

        return {
            "task_id": task_id,
            "task_name": task_name,
            "workspace": workspace_name,
            "iterations": iterations,
            "score_history": score_history,
            "current_iteration": len(iterations),
            "best_score": max(score_history, key=lambda x: x["score"])
            if score_history
            else None,
        }

    def get_iteration_details(
        self, task_id: str, iteration: int
    ) -> Dict[str, Any]:
        """
        Get details for a specific iteration.

        Args:
            task_id: Task identifier
            iteration: Iteration number (1-indexed)

        Returns:
            Dict with plan, execution, evaluation, summary data
        """
        workspace_name, task_name = self._parse_task_id(task_id)
        task_path = self._find_task_path(workspace_name, task_name)

        if not task_path:
            raise ValueError(f"Task not found: {task_id}")

        iteration_path = self._find_iteration_path(task_path, iteration)
        if not iteration_path:
            raise ValueError(f"Iteration {iteration} not found")

        return {
            "iteration": iteration,
            "plan": self._load_plan(iteration_path),
            "files": self._list_generated_files(iteration_path),
            "evaluation": self._load_evaluation(iteration_path),
            "summary": self._load_summary(iteration_path),
        }

    def get_file_content(
        self, task_id: str, iteration: int, filepath: str
    ) -> Dict[str, Any]:
        """
        Get content of a generated file.

        Args:
            task_id: Task identifier
            iteration: Iteration number
            filepath: Relative path within work_dir

        Returns:
            Dict with filename, content, language
        """
        workspace_name, task_name = self._parse_task_id(task_id)
        task_path = self._find_task_path(workspace_name, task_name)

        if not task_path:
            raise ValueError(f"Task not found: {task_id}")

        iteration_path = self._find_iteration_path(task_path, iteration)
        if not iteration_path:
            raise ValueError(f"Iteration {iteration} not found")

        file_path = iteration_path / "executor" / "work_dir" / filepath

        if not file_path.exists():
            raise ValueError(f"File not found: {filepath}")

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        # Determine language from extension
        ext = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".txt": "text",
            ".sh": "bash",
        }
        language = language_map.get(ext, "text")

        return {"filename": file_path.name, "content": content, "language": language}

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _parse_task_id(self, task_id: str) -> tuple[str, str]:
        """Parse task_id into (workspace_name, task_name)."""
        parts = task_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid task_id format: {task_id}")
        return parts[0], parts[1]

    def _find_task_path(
        self, workspace_name: str, task_name: str
    ) -> Optional[Path]:
        """Find the full path to a task directory."""
        for workspace in self.workspaces:
            if workspace.name == workspace_name:
                task_path = workspace / task_name
                if task_path.exists():
                    return task_path
        return None

    def _find_iteration_path(
        self, task_path: Path, iteration: int
    ) -> Optional[Path]:
        """
        Find the path to an iteration directory.
        Supports both "iteration_N" and "N" formats.
        """
        # Try "iteration_N" format first
        iter_path = task_path / f"iteration_{iteration}"
        if iter_path.exists():
            return iter_path

        # Try numeric format (General Agent uses this)
        iter_path = task_path / str(iteration)
        if iter_path.exists():
            return iter_path

        return None

    def _get_task_info(self, workspace: Path, task_dir: Path) -> Optional[Dict]:
        """Extract basic info about a task."""
        # Support both "iteration_N" and "N" directory formats
        iterations = list(task_dir.glob("iteration_*"))
        if not iterations:
            # Try numeric directories (General Agent uses this format)
            iterations = [
                d for d in task_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]

        if not iterations:
            return None

        # Get latest iteration score
        def get_iter_num(p: Path) -> int:
            if p.name.isdigit():
                return int(p.name)
            return int(p.name.split("_")[1])

        latest_iteration = max(iterations, key=get_iter_num)
        score = self._read_score(latest_iteration)

        return {
            "task_id": f"{workspace.name}/{task_dir.name}",
            "task_name": task_dir.name,
            "workspace": workspace.name,
            "timestamp": task_dir.name.replace("task_", ""),
            "num_iterations": len(iterations),
            "latest_score": score,
            "status": "completed" if score is not None else "running",
        }

    def _list_iterations(self, task_path: Path) -> List[Dict[str, Any]]:
        """List all iterations for a task."""
        iterations = []

        # Support both "iteration_N" and "N" directory formats
        iter_dirs = list(task_path.glob("iteration_*"))
        if not iter_dirs:
            # Try numeric directories (General Agent uses this format)
            iter_dirs = [
                d for d in task_path.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]

        # Helper to extract iteration number
        def get_iter_num(p: Path) -> int:
            if p.name.isdigit():
                return int(p.name)
            return int(p.name.split("_")[1])

        for iter_dir in sorted(iter_dirs, key=get_iter_num):
            iter_num = get_iter_num(iter_dir)
            score = self._read_score(iter_dir)
            num_files = len(
                list((iter_dir / "executor" / "work_dir").glob("*"))
                if (iter_dir / "executor" / "work_dir").exists()
                else []
            )

            iterations.append(
                {
                    "iteration": iter_num,
                    "score": score,
                    "num_files": num_files,
                    "has_plan": (iter_dir / "planner").exists(),
                    "has_evaluation": (iter_dir / "evaluator").exists(),
                    "has_summary": (iter_dir / "summary").exists() or (iter_dir / "summarizer").exists(),
                }
            )
        return iterations

    def _build_score_history(
        self, task_path: Path, iterations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build score history for charting."""
        return [
            {"iteration": it["iteration"], "score": it["score"] or 0.0}
            for it in iterations
        ]

    def _read_score(self, iteration_path: Path) -> Optional[float]:
        """Read score from various evaluation result files."""
        # Try executor/evaluator_dir/best_evaluation.json (General Agent format)
        eval_json = iteration_path / "executor" / "evaluator_dir" / "best_evaluation.json"
        if eval_json.exists():
            try:
                data = json.loads(eval_json.read_text())
                if "score" in data:
                    return float(data["score"])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Try evaluator/score.txt (Math Agent format)
        score_file = iteration_path / "evaluator" / "score.txt"
        if score_file.exists():
            try:
                return float(score_file.read_text().strip())
            except ValueError:
                pass

        # Try to parse from evaluation.log
        log_file = iteration_path / "evaluator" / "evaluation.log"
        if log_file.exists():
            content = log_file.read_text()
            # Look for score patterns like "Score: 0.75" or "score=0.75"
            import re

            match = re.search(r"[Ss]core[:\s=]+(\d+\.?\d*)", content)
            if match:
                return float(match.group(1))

        return None

    def _load_plan(self, iteration_path: Path) -> Optional[str]:
        """Load plan content from best_plan.md."""
        plan_file = iteration_path / "planner" / "best_plan.md"
        if plan_file.exists():
            return plan_file.read_text(encoding="utf-8")
        return None

    def _load_summary(self, iteration_path: Path) -> Optional[str]:
        """Load summary content from insights.md or best_summary.md."""
        # Try 'summary' directory first (Math Agent format)
        summary_file = iteration_path / "summary" / "insights.md"
        if summary_file.exists():
            return summary_file.read_text(encoding="utf-8")

        # Try 'summarizer' directory with insights.md
        summary_file = iteration_path / "summarizer" / "insights.md"
        if summary_file.exists():
            return summary_file.read_text(encoding="utf-8")

        # Try 'summarizer' directory with best_summary.md (General Agent format)
        summary_file = iteration_path / "summarizer" / "best_summary.md"
        if summary_file.exists():
            return summary_file.read_text(encoding="utf-8")

        # Try any .md file in summarizer directory as fallback
        summarizer_dir = iteration_path / "summarizer"
        if summarizer_dir.exists():
            for md_file in summarizer_dir.glob("*.md"):
                return md_file.read_text(encoding="utf-8")

        return None

    def _load_evaluation(self, iteration_path: Path) -> Dict[str, Any]:
        """Load evaluation results."""
        score = self._read_score(iteration_path)
        result = {"score": score, "logs": None, "summary": None, "status": None}

        # Try to load from executor/evaluator_dir/best_evaluation.json (General Agent)
        eval_json = iteration_path / "executor" / "evaluator_dir" / "best_evaluation.json"
        if eval_json.exists():
            try:
                data = json.loads(eval_json.read_text())
                result["summary"] = data.get("summary")
                result["status"] = data.get("status")
                if "metrics" in data and "raw_response" in data["metrics"]:
                    result["logs"] = data["metrics"]["raw_response"]
            except (json.JSONDecodeError, KeyError):
                pass

        # Try to load evaluation logs from evaluator/ directory (Math Agent)
        eval_dir = iteration_path / "evaluator"
        if not result["logs"]:
            log_file = eval_dir / "evaluation.log"
            if log_file.exists():
                result["logs"] = log_file.read_text(encoding="utf-8")

        return result

    def _list_generated_files(self, iteration_path: Path) -> List[Dict[str, Any]]:
        """List all generated files in work_dir."""
        work_dir = iteration_path / "executor" / "work_dir"
        if not work_dir.exists():
            return []

        files = []
        for file_path in sorted(work_dir.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(work_dir)
                files.append(
                    {
                        "path": str(rel_path),
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix,
                    }
                )
        return files


# ------------------------------------------------------------------
# Flask application
# ------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")
service: Optional[GeneralAgentService] = None


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/tasks")
def api_list_tasks():
    """API: List all available tasks."""
    try:
        tasks = service.list_tasks()
        return jsonify({"success": True, "tasks": tasks})
    except Exception as e:
        # Log the full error internally but return a safe message to users
        app.logger.error(f"Error listing tasks: {str(e)}")
        return jsonify({"success": False, "error": "Failed to list tasks"}), 500


@app.route("/api/tasks/<path:task_id>")
def api_get_task(task_id: str):
    """API: Get task details."""
    try:
        details = service.get_task_details(task_id)
        return jsonify({"success": True, "task": details})
    except ValueError as e:
        # Log the original error but return a generic safe message
        app.logger.warning(f"Task not found: {task_id} - {str(e)}")
        return jsonify({"success": False, "error": "Task not found"}), 404
    except Exception as e:
        # Log the full error internally but return a safe message to users
        app.logger.error(f"Error getting task {task_id}: {str(e)}")
        return jsonify({"success": False, "error": "Failed to get task details"}), 500


@app.route("/api/tasks/<path:task_id>/iterations/<int:iteration>")
def api_get_iteration(task_id: str, iteration: int):
    """API: Get iteration details."""
    try:
        details = service.get_iteration_details(task_id, iteration)
        return jsonify({"success": True, "iteration": details})
    except ValueError as e:
        # Log the original error but return a generic safe message
        app.logger.warning(f"Iteration not found: {task_id}/{iteration} - {str(e)}")
        return jsonify({"success": False, "error": "Iteration not found"}), 404
    except Exception as e:
        # Log the full error internally but return a safe message to users
        app.logger.error(f"Error getting iteration {iteration} for task {task_id}: {str(e)}")
        return jsonify({"success": False, "error": "Failed to get iteration details"}), 500


@app.route("/api/tasks/<path:task_id>/iterations/<int:iteration>/files/<path:filepath>")
def api_get_file(task_id: str, iteration: int, filepath: str):
    """API: Get file content."""
    try:
        file_data = service.get_file_content(task_id, iteration, filepath)
        return jsonify({"success": True, "file": file_data})
    except ValueError as e:
        # Log the original error but return a generic safe message
        app.logger.warning(f"File not found: {task_id}/{iteration}/{filepath} - {str(e)}")
        return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        # Log the full error internally but return a safe message to users
        app.logger.error(f"Error getting file {filepath} from task {task_id} iteration {iteration}: {str(e)}")
        return jsonify({"success": False, "error": "Failed to get file content"}), 500


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="General Agent Task Visualizer - Monitor task progress and results"
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        help="Workspace path (e.g., ./output-todo-list)",
    )
    parser.add_argument(
        "--workspaces",
        type=str,
        help="Comma-separated list of workspace paths",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080, help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    # Parse workspace paths
    workspace_paths = []
    if args.workspace:
        workspace_paths.append(args.workspace)
    if args.workspaces:
        workspace_paths.extend(args.workspaces.split(","))

    if not workspace_paths:
        print("Error: At least one workspace path is required")
        print("Usage: python visualizer.py --workspace ./output-todo-list")
        return 1

    # Initialize service
    global service
    service = GeneralAgentService(workspace_paths)

    print("=" * 60)
    print("General Agent Visualizer")
    print("=" * 60)
    print(f"Monitoring workspaces:")
    for ws in service.workspaces:
        print(f"  - {ws}")
    print()
    print(f"Server starting at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Start Flask server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
