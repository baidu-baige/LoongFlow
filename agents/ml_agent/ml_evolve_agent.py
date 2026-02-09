# -*- coding: utf-8 -*-
"""
ML Evolve Agent Runner - Refactored to use BasePESRunner.
"""

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from agents.ml_agent.evaluator import MLEvaluator
from agents.ml_agent.executor import MLExecutorAgent
from agents.ml_agent.planner import MLPlannerAgent
from agents.ml_agent.summary import MLSummaryAgent
from agents.ml_agent.utils import system
from loongflow.framework.pes import Worker
from loongflow.framework.pes.base_runner import BasePESRunner
from loongflow.framework.pes.context import EvolveChainConfig
from loongflow.framework.pes.evaluator import Evaluator


class MLAgent(BasePESRunner):
    """
    ML Evolve Agent runner for machine learning tasks (Kaggle/MLE-Bench).

    Extends BasePESRunner with:
    - Required task data path (--task-data-path)
    - Automatic hardware info detection
    - Custom ML evaluator
    - Multiprocessing spawn method setup
    """

    def _add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        """Add ML-agent specific CLI arguments."""
        parser.add_argument(
            "--task-data-path",
            type=str,
            default=None,
            help="Path to machine learning task directory to load dataset. "
            "Task description will be read from description.md if it's not provided",
        )

    def _merge_custom_configs(
        self, args: argparse.Namespace, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle ML-agent specific config merging."""
        # Validate required argument
        if not args.task_data_path:
            print("Error: ML Task data path must be provided", file=sys.stderr)
            sys.exit(1)

        # Initialize metadata section
        if "metadata" not in config["evolve"]:
            config["evolve"]["metadata"] = {}

        config["evolve"]["metadata"]["task_data_path"] = args.task_data_path

        # Load task description from description.md if not provided via CLI
        try:
            task_description = Path(args.task_data_path + "/description.md").read_text(
                encoding="utf-8"
            )
            config["evolve"]["task"] = task_description
        except FileNotFoundError:
            print(
                f"Error: Task description file 'description.md' not found "
                f"at '{args.task_data_path}'",
                file=sys.stderr,
            )

        # CLI task overrides description.md
        if not config["evolve"].get("task"):
            print("Error: Task description not provided", file=sys.stderr)

        # Add hardware and environment metadata
        hardware_context = system.get_hardware_context()
        config["evolve"]["metadata"]["hardware_info"] = hardware_context.get(
            "hardware_info", ""
        )
        config["evolve"]["metadata"]["gpu_available"] = hardware_context.get(
            "gpu_available", False
        )
        config["evolve"]["metadata"]["gpu_count"] = hardware_context.get(
            "gpu_count", 0
        )
        config["evolve"]["metadata"]["task_dir_structure"] = (
            system.get_directory_structure(args.task_data_path)
        )

        return config

    def _get_process_name(self) -> str:
        return "ML-Evolve"

    def _get_worker_registrations(
        self,
    ) -> Tuple[
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
        List[Tuple[str, Type[Worker]]],
    ]:
        """Register ML agent workers."""
        planners = [("ml_planner", MLPlannerAgent)]
        executors = [("ml_executor", MLExecutorAgent)]
        summarizers = [("ml_summary", MLSummaryAgent)]
        return planners, executors, summarizers

    def _create_evaluator(self, config: EvolveChainConfig) -> Optional[Evaluator]:
        """Create the ML-specific evaluator."""
        return MLEvaluator(config.evolve.evaluator)

    def _pre_run_setup(self) -> None:
        """Set up multiprocessing for ML tasks."""
        multiprocessing.set_start_method("spawn", force=True)


if __name__ == "__main__":
    runner = MLAgent()
    runner.start()
