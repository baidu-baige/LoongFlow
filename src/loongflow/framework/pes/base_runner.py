#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base PES Agent Runner - Abstract base class for PES Agent implementations.

This module provides a reusable foundation for building PES (Plan-Execute-Summary)
agent runners, reducing code duplication across different agent types.
"""

import argparse
import asyncio
import logging
import logging.handlers
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from pydantic import ValidationError

from loongflow.agentsdk.logger.logger import TraceIdFilter
from loongflow.framework.pes import PESAgent, Worker
from loongflow.framework.pes.context import EvolveChainConfig
from loongflow.framework.pes.evaluator import Evaluator


class BasePESRunner(ABC):
    """
    Abstract base class for PES Agent runners.

    This class handles the common workflow:
    1. Parsing command-line arguments (base + custom)
    2. Loading YAML configuration
    3. Merging CLI arguments over configuration
    4. Validating configuration using Pydantic
    5. Setting up logging
    6. Instantiating and running the PESAgent

    Subclasses must implement:
    - _add_custom_args(): Add agent-specific CLI arguments
    - _merge_custom_configs(): Handle agent-specific config merging
    - _get_worker_registrations(): Return worker class mappings
    - _get_process_name(): Return display name for the process

    Subclasses may override:
    - _create_evaluator(): Provide a custom evaluator instance
    - _pre_run_setup(): Execute code before agent starts
    """

    def __init__(self):
        self.parser = self._setup_arg_parser()

    # =========================================================================
    # Argument Parsing
    # =========================================================================

    def _setup_arg_parser(self) -> argparse.ArgumentParser:
        """Sets up the command-line argument parser with common arguments."""
        parser = argparse.ArgumentParser(
            description="LoongFlow Framework Runner: Start an evolutionary task "
            "with a configuration file and optional overrides.",
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

        # Task overrides
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
            help="Override the task description by reading from a file. "
            "Takes precedence over --task.",
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

        # Logging overrides
        parser.add_argument(
            "--log-level",
            type=str,
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

        # Allow subclasses to add custom arguments
        self._add_custom_args(parser)

        return parser

    @abstractmethod
    def _add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        """
        Add agent-specific command-line arguments.

        Override this method to add arguments specific to your agent type.

        Args:
            parser: The ArgumentParser to add arguments to.
        """
        pass

    # =========================================================================
    # Configuration Loading & Merging
    # =========================================================================

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

        # Ensure required sections exist
        if "evolve" not in merged_config:
            merged_config["evolve"] = {}
        if "evaluator" not in merged_config["evolve"]:
            merged_config["evolve"]["evaluator"] = {}
        if "logger" not in merged_config:
            merged_config["logger"] = {}

        # Apply common overrides
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

        # Allow subclasses to add custom config merging
        merged_config = self._merge_custom_configs(args, merged_config)

        return merged_config

    @abstractmethod
    def _merge_custom_configs(
        self, args: argparse.Namespace, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge agent-specific configurations.

        Override this method to handle custom CLI arguments and config merging.

        Args:
            args: Parsed command-line arguments.
            config: The configuration dictionary to modify.

        Returns:
            The modified configuration dictionary.
        """
        pass

    # =========================================================================
    # Logging Setup
    # =========================================================================

    def _setup_logging(self, config: EvolveChainConfig) -> None:
        """Configures the root logger based on the provided LoggerConfig."""
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

    # =========================================================================
    # Agent Setup (Abstract Methods)
    # =========================================================================

    @abstractmethod
    def _get_process_name(self) -> str:
        """
        Return the display name for this process.

        Examples: "Math Evolve", "ML-Evolve"
        """
        pass

    @abstractmethod
    def _get_worker_registrations(
        self,
    ) -> Tuple[
        List[Tuple[str, Type[Worker]]],  # planners
        List[Tuple[str, Type[Worker]]],  # executors
        List[Tuple[str, Type[Worker]]],  # summarizers
    ]:
        """
        Return worker class registrations for the agent.

        Returns:
            A tuple of three lists:
            - planners: List of (name, PlannerClass) tuples
            - executors: List of (name, ExecutorClass) tuples
            - summarizers: List of (name, SummaryClass) tuples
        """
        pass

    def _create_evaluator(self, config: EvolveChainConfig) -> Optional[Evaluator]:
        """
        Create and return a custom evaluator instance.

        Override this method if your agent needs a custom evaluator.
        Default returns None (uses framework default evaluator).

        Args:
            config: The validated configuration.

        Returns:
            An evaluator instance or None.
        """
        return None

    def _pre_run_setup(self) -> None:
        """
        Execute any setup code before the agent starts.

        Override this method for agent-specific initialization
        (e.g., multiprocessing settings, environment setup).
        """
        pass

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def run(self) -> None:
        """
        The main execution method. Parses args, loads and merges configs,
        validates them, and starts the PESAgent.
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
            print("\n--- Configuration Validation Error ---", file=sys.stderr)
            print(
                f"There are issues with your merged configuration "
                f"(from {args.config} and CLI args).",
                file=sys.stderr,
            )
            print(f"Details:\n{e}", file=sys.stderr)
            print("--------------------------------------", file=sys.stderr)
            sys.exit(1)

        self._setup_logging(config)

        # Prepare checkpoint path if provided
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

        # Create evaluator (may be None for default)
        evaluator = self._create_evaluator(config)

        print("4. Initializing PESAgent...")
        agent_kwargs = {
            "config": config,
            "checkpoint_path": checkpoint_path,
        }
        if evaluator is not None:
            agent_kwargs["evaluator"] = evaluator

        agent = PESAgent(**agent_kwargs)

        # Register workers
        planners, executors, summarizers = self._get_worker_registrations()
        for name, worker_cls in planners:
            agent.register_planner_worker(name, worker_cls)
        for name, worker_cls in executors:
            agent.register_executor_worker(name, worker_cls)
        for name, worker_cls in summarizers:
            agent.register_summary_worker(name, worker_cls)

        # Signal handling for graceful shutdown
        loop = asyncio.get_running_loop()
        interrupt_task = None

        def signal_handler(sig_name: str) -> None:
            nonlocal interrupt_task
            print(f"\n🛑 Received signal {sig_name}. Initiating graceful shutdown...")
            # Set the stop event immediately to unblock the main loop
            agent._stop_event.set()
            # Create interrupt task and keep reference
            if interrupt_task is None or interrupt_task.done():
                interrupt_task = asyncio.create_task(agent.interrupt())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s.name))
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        # Print startup banner
        process_name = self._get_process_name()
        print("\n======================================")
        print(f"🚀 Starting {process_name} Process 🚀")
        print(f"🎯 Task: {(config.evolve.task or '')[:80]}...")
        print(f"📈 Target Score: {config.evolve.target_score}")
        print(f"🔄 Max Iterations: {config.evolve.max_iterations}")
        print(f"- Planner: {config.evolve.planner_name}")
        print(f"- Executor: {config.evolve.executor_name}")
        print(f"- Summarizer: {config.evolve.summary_name}")
        if checkpoint_path:
            print(f"↩️ Resuming from checkpoint: {checkpoint_path}")
        print("======================================\n")

        # Run the agent
        try:
            final_result = await agent.run()
            if final_result is not None:
                print("\n✅ Evolution process finished successfully.")
                print(final_result.model_dump_json(indent=2))
            else:
                print(
                    "\n⚠️ Evolution process finished with no result returned. "
                    "Maybe it was interrupted."
                )
        except KeyboardInterrupt:
            print("\n🛑 Process interrupted by user. Shutting down gracefully...")
        except Exception as e:
            print(
                f"\n❌ An unexpected error occurred during evolution: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()

    def start(self) -> None:
        """
        Entry point to start the agent.

        Calls _pre_run_setup() then runs the async run() method.
        """
        self._pre_run_setup()
        asyncio.run(self.run())
