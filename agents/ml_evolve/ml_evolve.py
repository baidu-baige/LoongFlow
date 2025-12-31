# -*- coding: utf-8 -*-
"""
This file provides the main entrypoint for running the ML-Evolve agent.
"""
import argparse
import asyncio
import logging
import logging.handlers
import multiprocessing
import signal
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from agents.ml_evolve.evaluator import MLEvaluator
from agents.ml_evolve.executor import MLExecutorAgent
from agents.ml_evolve.planner import MLPlannerAgent
from agents.ml_evolve.summary import MLSummaryAgent
from agents.ml_evolve.utils import system
from agentsdk.logger.logger import TraceIdFilter
from evolux.evolve import EvolveAgent
from evolux.evolve.context import EvolveChainConfig


class MLAgent:
    """
    ML-Evolve Framework command-line runner.

    This class orchestrates the process of launching an EvolveAgent run for ML tasks.
    """

    def __init__(self):
        self.parser = self._setup_arg_parser()

    def _setup_arg_parser(self) -> argparse.ArgumentParser:
        """Sets up the command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="LoongFlow Framework Runner: Start an evolutionary task\
with a configuration file and optional overrides.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Core arguments
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="Path to the required YAML configuration file.",
        )
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help="Path to a checkpoint directory to load and resume the database state.",
        )

        # Task and Evaluator overrides
        parser.add_argument(
            "--task",
            type=str,
            default=None,
            help="Override the task description from the config file.",
        )
        parser.add_argument(
            "--task-file",
            type=str,
            default=None,
            help="Override the task description by reading from a file. Takes precedence over --task.",
        )

        parser.add_argument(
            "--task-data-path",
            type=str,
            default=None,
            help="Path to machine learning task directory to load dataset. "
            "Task description will be read from description.md if it's not provided",
        )
        parser.add_argument(
            "--eval-file",
            type=str,
            default=None,
            help="Override the evaluator's 'evaluate_code' by reading from a file.",
        )
        parser.add_argument(
            "--workspace-path",
            type=str,
            default=None,
            help="Override the evaluator's workspace path.",
        )

        # Evolution process overrides
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=None,
            help="Override the maximum number of evolution iterations.",
        )
        parser.add_argument(
            "--target-score",
            type=float,
            default=None,
            help="Override the target score for the evolution process.",
        )
        parser.add_argument(
            "--planner",
            type=str,
            default=None,
            help="Override the planner to use for this run.",
        )
        parser.add_argument(
            "--executor",
            type=str,
            default=None,
            help="Override the executor to use for this run.",
        )
        parser.add_argument(
            "--summary",
            type=str,
            default=None,
            help="Override the summary to use for this run.",
        )
        parser.add_argument(
            "--log-level",
            type=str.upper,
            default=None,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Override the global logging level.",
        )
        parser.add_argument(
            "--log-path",
            type=str,
            default=None,
            help="Override the directory for log files.",
        )

        return parser

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Loads the base configuration from a YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(
                f"Error: Configuration file not found at '{config_path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file '{config_path}':\n{e}", file=sys.stderr)
            sys.exit(1)

    def _merge_configs(
        self, args: argparse.Namespace, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merges CLI arguments into the base configuration dictionary."""
        merged_config = base_config.copy()

        # Ensure 'evolve' and 'evaluator' sections exist for overrides
        if "evolve" not in merged_config:
            merged_config["evolve"] = {}
        if "evaluator" not in merged_config["evolve"]:
            merged_config["evolve"]["evaluator"] = {}
        if "logger" not in merged_config:
            merged_config["logger"] = {}

        # Apply overrides from CLI arguments
        if not args.task_data_path:
            print(f"Error: ML Task data path must be provided", file=sys.stderr)
            sys.exit(1)

        if "metadata" not in merged_config["evolve"]:
            merged_config["evolve"]["metadata"] = {}
        merged_config["evolve"]["metadata"]["task_data_path"] = args.task_data_path

        try:
            task_description = Path(args.task_data_path + "/description.md").read_text(
                encoding="utf-8"
            )
            merged_config["evolve"]["task"] = task_description
        except FileNotFoundError:
            print(
                f"Error: Task description file 'description.md' not found at '{args.task_data_path}'",
                file=sys.stderr,
            )

        if args.task is not None:
            merged_config["evolve"]["task"] = args.task
        if args.task_file is not None:
            try:
                merged_config["evolve"]["task"] = Path(args.task_file).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                print(
                    f"Error: Task file not found at '{args.task_file}'", file=sys.stderr
                )
                sys.exit(1)

        if not merged_config["evolve"]["task"]:
            print(f"Error: Task description not provided", file=sys.stderr)

        if args.eval_file is not None:
            try:
                merged_config["evolve"]["evaluator"]["evaluate_code"] = Path(
                    args.eval_file
                ).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(
                    f"Error: Evaluation code file not found at '{args.eval_file}'",
                    file=sys.stderr,
                )
                sys.exit(1)

        if args.workspace_path is not None:
            merged_config["evolve"]["evaluator"]["workspace_path"] = args.workspace_path

        if args.max_iterations is not None:
            merged_config["evolve"]["max_iterations"] = args.max_iterations
        if args.target_score is not None:
            merged_config["evolve"]["target_score"] = args.target_score
        if args.planner is not None:
            merged_config["evolve"]["planner_name"] = args.planner
        if args.executor is not None:
            merged_config["evolve"]["executor_name"] = args.executor
        if args.summary is not None:
            merged_config["evolve"]["summary_name"] = args.summary
        if args.log_level is not None:
            merged_config["logger"]["level"] = args.log_level
        if args.log_path is not None:
            merged_config["logger"]["log_path"] = args.log_path

        merged_config["evolve"]["metadata"][
            "hardware_info"
        ] = system.get_hardware_context().get("hardware_info", "")
        merged_config["evolve"]["metadata"][
            "gpu_available"
        ] = system.get_hardware_context().get("gpu_available", False)
        merged_config["evolve"]["metadata"]["task_dir_structure"] = (
            system.get_directory_structure(args.task_data_path)
        )

        return merged_config

    def _setup_logging(self, config: EvolveChainConfig):
        """
        Configures the root logger based on the provided LoggerConfig.
        This function MUST be called before any get_logger() is invoked.
        """
        if not config.logger:
            print(
                "Warning: Logger configuration not found. Using default logging.",
                file=sys.stderr,
            )
            return

        logger_config = config.logger
        root_logger = logging.getLogger()

        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        root_logger.setLevel(logger_config.level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [log_id=%(log_id)s] [%(name)s] %(message)s"
        )

        if logger_config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(TraceIdFilter())
            root_logger.addHandler(console_handler)

        if logger_config.file_logging:
            log_dir = Path(logger_config.log_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / logger_config.filename

            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_file_path,
                when=logger_config.rotation,
                interval=1,
                backupCount=logger_config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(TraceIdFilter())
            root_logger.addHandler(file_handler)

        print(
            f"Logging configured. Level: {logger_config.level}, "
            f"Console: {logger_config.console_logging}, "
            f"File: {logger_config.file_logging} (at {logger_config.log_path})"
        )

    async def run(self):
        """
        The main execution method.
        """
        args = self.parser.parse_args()

        print("1. Loading base configuration from YAML...")
        base_config = self._load_yaml_config(args.config)

        print("2. Merging command-line overrides...")
        final_config_dict = self._merge_configs(args, base_config)

        try:
            print("3. Validating final configuration...")
            config = EvolveChainConfig.model_validate(final_config_dict)
            print("   - Configuration is valid.")
        except ValidationError as e:
            print(f"\n--- Configuration Validation Error ---\n{e}", file=sys.stderr)
            sys.exit(1)

        self._setup_logging(config)
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

        print("4. Initializing EvolveAgent...")
        agent = EvolveAgent(
            config=config,
            evaluator=MLEvaluator(config.evolve.evaluator),
            checkpoint_path=checkpoint_path,
        )

        agent.register_planner_worker("ml_planner", MLPlannerAgent)
        agent.register_executor_worker("ml_executor", MLExecutorAgent)
        agent.register_summary_worker("ml_summary", MLSummaryAgent)

        # --- Signal Handling ---
        loop = asyncio.get_running_loop()

        def signal_handler(sig_name):
            """
            handle shutdown signal
            """
            print(f"\n🛑 Received signal {sig_name}. Initiating graceful shutdown...")
            asyncio.create_task(agent.interrupt())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: signal_handler(sig.name))
            except NotImplementedError:
                pass

        print("\n======================================")
        print("🚀 Starting ML-Evolve Process 🚀")
        print(f"🎯 Task: {(config.evolve.task or '')[:80]}...")
        print(f"📈 Target Score: {config.evolve.target_score}")
        print(f"🔄 Max Iterations: {config.evolve.max_iterations}")
        print(f"- Planner: {config.evolve.planner_name}")
        print(f"- Executor: {config.evolve.executor_name}")
        print(f"- Summarizer: {config.evolve.summary_name}")
        if checkpoint_path:
            print(f"↩️ Resuming from checkpoint: {checkpoint_path}")
        print("======================================\n")

        try:
            final_result = await agent.run()
            if final_result:
                print("\n✅ Evolution process finished successfully.")
                print(final_result.model_dump_json(indent=2))
            else:
                print("\n⚠️ Evolution process finished with no result returned.")
        except KeyboardInterrupt:
            print("\n🛑 Process interrupted by user. Shutting down gracefully...")
        except Exception as e:
            print(
                f"\n❌ An unexpected error occurred during evolution: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    runner = MLAgent()
    asyncio.run(runner.run())
