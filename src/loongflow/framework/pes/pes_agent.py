#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the PESAgent, which orchestrates an evolutionary
process for generating solutions. It manages concurrent evolution cycles,
each consisting of planning, execution, and summarization steps, until a
target score is achieved or the process is interrupted.
"""

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Set

from loongflow.agentsdk.message import ContentElement
from loongflow.agentsdk.message.message import Message, Role
from loongflow.framework.base.agent_base import AgentBase
from loongflow.framework.pes.context import Context, EvolveChainConfig
from loongflow.framework.pes.database.database import EvolveDatabase
from loongflow.framework.pes.evaluator import Evaluator, LoongFlowEvaluator
from loongflow.framework.pes.finalizer import LoongFlowFinalizer, Finalizer
from loongflow.framework.pes.register import (
    EXECUTOR,
    PLANNER,
    SUMMARY,
    Worker,
    get_worker,
    register_worker,
)


class PESAgent(AgentBase):
    """
    LoongFlow Evolve Agent that manages an in-process, concurrent evolution process.

    This agent dynamically loads and runs Planner, Executor, and Summary workers
    based on the provided configuration in a concurrent loop.
    """

    def __init__(
        self,
        config: EvolveChainConfig,
        database: Optional[EvolveDatabase] = None,
        evaluator: Optional[Evaluator] = None,
        finalizer: Optional[Finalizer] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initializes the PESAgent.

        Args:
            config: The root configuration object for the entire evolve chain.
            database: The database instance for state and memory management.
            evaluator: The evaluator component for solution evaluation.
            finalizer: Optional finalizer component to generate the final result.
        """
        super().__init__()

        if config is None:
            raise ValueError("EvolveChainConfig must be provided for PESAgent.")

        self.config = config
        self.finalizer = finalizer or LoongFlowFinalizer()
        self.evaluator = evaluator or LoongFlowEvaluator(config.evolve.evaluator)

        # Get critical parameters from config
        self.target_score = self.config.evolve.target_score
        self.max_workers = self.config.evolve.concurrency
        self.max_iterations = self.config.evolve.max_iterations or float("inf")

        # State management for concurrency and lifecycle
        self.task_id = uuid.uuid4()  # Unique ID for this agent's run
        self._stop_event = asyncio.Event()
        self._running_tasks: Set[asyncio.Task] = set()

        # Locks
        self._iteration_lock = asyncio.Lock()  # Lock for starting new tasks
        self._completion_lock = (
            asyncio.Lock()
        )  # Lock for finishing tasks and checkpointing
        self._token_lock = asyncio.Lock()  # Lock for token counting

        # --- Task Completion Counter & Checkpoint Parsing Logic ---
        self._completion_count = 0

        if checkpoint_path is not None:
            self.logger.info(f"Initializing from checkpoint: {checkpoint_path}")
            # Parse the checkpoint directory name to restore completion count
            # Expected format: checkpoint-iter-{iteration_id}-{completion_count}
            path_obj = Path(checkpoint_path)
            dir_name = path_obj.name

            # Regex to match: checkpoint-iter-(\d+)-(\d+)
            match = re.search(r"checkpoint-iter-(\d+)-(\d+)$", dir_name)

            if match:
                # Group 1 is iteration_id (we don't use this to set completion_count)
                # Group 2 is completion_count
                try:
                    self._completion_count = int(match.group(2))
                    self.logger.info(
                        f"Restored completion count to {self._completion_count}"
                    )
                except ValueError:
                    raise ValueError(
                        f"Invalid number format in checkpoint path: {dir_name}"
                    )
            else:
                # If format doesn't match, raise error as per requirement
                raise ValueError(
                    f"Invalid checkpoint directory format: '{dir_name}'. "
                    "Expected format: 'checkpoint-iter-{id}-{count}'"
                )

        # Initialize Database
        self.database = database
        if self.database is None:
            self.logger.info("Creating database from config...")
            self.database = EvolveDatabase.create_database(config.evolve.database)
            self.logger.info(f"Created database: {self.database}")
            if checkpoint_path is not None:
                self.logger.info(
                    f"Loading database from checkpoint at {checkpoint_path}"
                )
                self.database.load_checkpoint(str(checkpoint_path))

        # Initialize Iteration ID (from DB, independent of completion count)
        self._current_iteration = (
            self.database.memory_status()
            .get("global_status", {})
            .get("current_iteration", 0)
        )

        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    # region Worker Registration Methods
    def register_planner_worker(self, name: str, worker_class: type[Worker]):
        """
        Registers a Planner worker implementation.

        Args:
            name (str): The name to identify the worker.
            worker_class (type): The class of the worker that extends the Worker interface.
        """
        self.logger.info(f"Registering Planner worker: '{name}'")
        register_worker(name, PLANNER, worker_class)

    def register_executor_worker(self, name: str, worker_class: type[Worker]):
        """
        Registers an Executor worker implementation.

        Args:
            name (str): The name to identify the worker.
            worker_class (type): The class of the worker that extends the Worker interface.
        """
        self.logger.info(f"Registering Executor worker: '{name}'")
        register_worker(name, EXECUTOR, worker_class)

    def register_summary_worker(self, name: str, worker_class: type[Worker]):
        """
        Registers a Summary worker implementation.

        Args:
            name (str): The name to identify the worker.
            worker_class (type): The class of the worker that extends the Worker interface.
        """
        self.logger.info(f"Registering Summary worker: '{name}'")
        register_worker(name, SUMMARY, worker_class)

    async def _evolution_cycle(self, iteration_id: int) -> None:
        """
        Represents a single, complete evolution cycle for a given iteration ID.
        """
        trace_id = str(uuid.uuid4().hex[:12])
        self.logger.info(
            f"Trace ID: {trace_id}, Starting evolution cycle for iteration {iteration_id}."
        )

        previous_prompt_tokens = self.total_prompt_tokens
        previous_completion_tokens = self.total_completion_tokens

        try:
            # 1. Prepare context and configurations for this cycle
            evolve_conf = self.config.evolve

            planner_name = evolve_conf.planner_name
            executor_name = evolve_conf.executor_name
            summary_name = evolve_conf.summary_name

            planner_config = self.config.planners.get(planner_name, {})
            executor_config = self.config.executors.get(executor_name, {})
            summary_config = self.config.summarizers.get(summary_name, {})

            if not all(
                [
                    planner_name in self.config.planners,
                    executor_name in self.config.executors,
                    summary_name in self.config.summarizers,
                ]
            ):
                raise ValueError(
                    "One or more worker configurations are missing in the root config."
                )

            num_islands = evolve_conf.database.num_islands
            current_island_id = 0
            if num_islands > 0:
                current_island_id = iteration_id % num_islands

            context = Context(
                task=evolve_conf.task,
                base_path=evolve_conf.workspace_path,
                init_solution=evolve_conf.initial_code,
                init_evaluation=evolve_conf.initial_evaluation,
                init_score=evolve_conf.initial_score,
                task_id=self.task_id,
                island_id=current_island_id,
                current_iteration=iteration_id,
                total_iterations=self.max_iterations,
                trace_id=trace_id,
                metadata=evolve_conf.metadata,
            )

            evaluator = self.evaluator

            # 2. --- Planner Step ---
            planner = get_worker(
                planner_name,
                PLANNER,
                config=planner_config,
                db=self.database,
            )
            planner_result = await planner.run(context, None)
            plan_content = planner_result.get_elements(ContentElement)
            if plan_content and len(plan_content) > 0:
                async with self._token_lock:
                    self.total_prompt_tokens += plan_content[0].data.get(
                        "total_prompt_tokens", 0
                    )
                    self.total_completion_tokens += plan_content[0].data.get(
                        "total_completion_tokens", 0
                    )

            if self._stop_event.is_set():
                return

            # 3. --- Executor Step ---
            executor = get_worker(
                executor_name,
                EXECUTOR,
                config=executor_config,
                evaluator=evaluator,
                db=self.database,
            )
            executor_result = await executor.run(context, planner_result)
            executor_content = executor_result.get_elements(ContentElement)
            if executor_content and len(executor_content) > 0:
                async with self._token_lock:
                    self.total_prompt_tokens += executor_content[0].data.get(
                        "total_prompt_tokens", 0
                    )
                    self.total_completion_tokens += executor_content[0].data.get(
                        "total_completion_tokens", 0
                    )

            if self._stop_event.is_set():
                return

            # 4. --- Summary Step ---
            summary = get_worker(
                summary_name,
                SUMMARY,
                config=summary_config,
                db=self.database,
            )
            summary_result = await summary.run(context, executor_result)
            summary_content = summary_result.get_elements(ContentElement)
            if summary_content and len(summary_content) > 0:
                async with self._token_lock:
                    self.total_prompt_tokens += summary_content[0].data.get(
                        "total_prompt_tokens", 0
                    )
                    self.total_completion_tokens += summary_content[0].data.get(
                        "total_completion_tokens", 0
                    )

            if self._stop_event.is_set():
                return

            prompt_tokens = self.total_prompt_tokens - previous_prompt_tokens
            completion_tokens = (
                self.total_completion_tokens - previous_completion_tokens
            )
            prompt_cost = (
                prompt_tokens / 1000
            ) * self.config.llm_config.prompt_token_price
            completion_cost = (
                completion_tokens / 1000
            ) * self.config.llm_config.completion_token_price
            total_tokens = prompt_tokens + completion_tokens
            total_cost = prompt_cost + completion_cost

            self.logger.info(
                f"Trace ID: {trace_id}, Evolution cycle for iteration {iteration_id} completed successfully."
                + f"Used Prompt tokens: {prompt_tokens}, cost: {round(prompt_cost, 6)}."
                + f"Used Completion tokens: {completion_tokens}, cost: {round(completion_cost, 6)}."
                + f"Total Tokens: {total_tokens}, total cost: {round(total_cost, 6)}."
            )

            # --- Checkpoint Logic ---
            # Execute only after successful completion of the cycle
            await self._handle_cycle_completion_and_checkpoint(iteration_id)

        except asyncio.CancelledError:
            self.logger.warning(
                f"Evolution cycle for iteration {iteration_id} was cancelled."
            )
            time.sleep(3)
        except Exception as e:
            self.logger.error(
                f"Evolution cycle for iteration {iteration_id} failed: {e}",
                exc_info=True,
            )
            time.sleep(3)

    async def _handle_cycle_completion_and_checkpoint(self, iteration_id: int):
        """
        Handles the logic for task completion counting and triggering checkpoints.
        """
        should_save = False
        current_count = 0

        async with self._completion_lock:
            self._completion_count += 1
            current_count = self._completion_count

            interval = self.config.evolve.database.checkpoint_interval
            if interval > 0 and current_count % interval == 0:
                should_save = True

        if should_save:
            await self._save_checkpoint(iteration_id, current_count)

    async def _save_checkpoint(self, iteration_id: int, completion_count: int) -> None:
        """
        Saves a checkpoint with the specific naming convention.

        Args:
            iteration_id: The iteration ID of the task that just finished.
            completion_count: The total number of completed tasks so far.
        """
        # Requirement 1: Directory format: checkpoint-iter-{iteration_id}-{completion_count}
        dir_name = f"checkpoint-iter-{iteration_id}-{completion_count}"

        self.logger.info(
            f"Triggering checkpoint save. Count: {completion_count}, Iter: {iteration_id}."
        )
        try:
            # We call database save. Note: Assuming database.save_checkpoint is thread-safe
            # or handles its own locking if needed, otherwise we might need to expand lock scope.
            await self.database.save_checkpoint(
                self.config.evolve.database.output_path, dir_name
            )
            self.logger.info(
                f"Checkpoint saved to {self.config.evolve.database.output_path}/{dir_name}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save checkpoint {dir_name}: {e}",
                exc_info=True,
            )

    async def _try_start_new_cycle(self) -> bool:
        """
        Atomically checks conditions and starts a new evolution cycle if possible.
        """
        async with self._iteration_lock:
            if (
                len(self._running_tasks) < self.max_workers
                and self._current_iteration < self.max_iterations
                and not self._stop_event.is_set()
            ):
                # Increment counter *before* creating the task to reserve the ID
                self._current_iteration += 1

                # Requirement 2: Removed the prec-check checkpoint logic from here.

                new_task = asyncio.create_task(
                    self._evolution_cycle(self._current_iteration)
                )
                self._running_tasks.add(new_task)
                return True
        return False

    async def run(self) -> Message:
        """
        Main asynchronous execution loop for the evolution process.
        """
        start_time = int(time.time())
        total_tokens = 0.0
        total_cost = 0.0

        self._stop_event.clear()
        self._running_tasks.clear()
        was_interrupted_flag = False
        init_solution_message = Message.from_text(
            sender="assistant",
            role=Role.ASSISTANT,
            data=self.config.evolve.initial_code,
        )

        # If initial score is not set and the first iteration, evaluate the initial code and set it.
        if self.config.evolve.initial_code and self.config.evolve.evaluator.evaluate_code and self.config.evolve.initial_score is None and self._current_iteration == 0:
            init_evaluation = await self.evaluator.evaluate(init_solution_message)
            self.config.evolve.initial_score = init_evaluation.score
            self.config.evolve.initial_evaluation = json.dumps(
                init_evaluation.to_dict(), ensure_ascii=False, indent=2
            )

        if self.config.evolve.initial_score and self.config.evolve.initial_score >= self.target_score:
            self.logger.info(f"Initial solution meets target score.")
            final_message = await self.finalizer.finalize(
                self.database,
                start_time=start_time,
                was_interrupted=was_interrupted_flag,
                total_cost=total_cost,
                total_tokens=total_tokens,
            )
            return final_message

        try:
            self.logger.info("Starting initial evolution workers...")
            while await self._try_start_new_cycle():
                pass

            self.logger.info(
                f"Started {len(self._running_tasks)} initial evolution workers."
            )

            # Main loop: wait for any task to complete, then try to start a new one.
            while self._running_tasks and not self._stop_event.is_set():
                # Create a task that waits for the stop event
                stop_waiter = asyncio.create_task(self._stop_event.wait())
                tasks_to_wait = self._running_tasks | {stop_waiter}

                done, pending = await asyncio.wait(
                    tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                )

                # If stop_waiter completed, break immediately
                if stop_waiter in done:
                    stop_waiter.cancel()
                    break

                # Cancel the stop_waiter if it's still pending
                if stop_waiter in pending:
                    stop_waiter.cancel()
                    try:
                        await stop_waiter
                    except asyncio.CancelledError:
                        pass

                for task in done:
                    if task in self._running_tasks:
                        self._running_tasks.remove(task)
                    try:
                        await task  # Check for exceptions in the completed task
                    except Exception as e:
                        self.logger.error(
                            f"An evolution cycle task failed unexpectedly: {e}",
                            exc_info=True,
                        )

                    # Check for stop conditions after each completed cycle
                    global_status = self.database.memory_status()
                    best_score = global_status.get("global_status", {}).get(
                        "best_score", 0.0
                    )

                    if (
                        best_score is not None
                        and isinstance(best_score, float)
                        and best_score >= self.target_score
                    ):

                        completion_tokens = self.total_completion_tokens
                        completion_cost = (
                            completion_tokens / 1000
                        ) * self.config.llm_config.completion_token_price
                        prompt_tokens = self.total_prompt_tokens
                        prompt_cost = (
                            prompt_tokens / 1000
                        ) * self.config.llm_config.prompt_token_price
                        total_tokens = completion_tokens + prompt_tokens
                        total_cost = completion_cost + prompt_cost
                        self.logger.info(
                            f"Target score ({self.target_score}) reached. "
                            + f"Save the last checkpoint, Initiating shutdown. \n"
                            + f"Total Completion tokens used: {completion_tokens}, "
                            + f"cost: {round(completion_cost, 6)}. \n"
                            + f"Total Prompt tokens used: {prompt_tokens}, ,"
                            + f"cost: {round(prompt_cost, 6)}. \n"
                            + f"Total tokens: {total_tokens}, Total Cost: {round(total_cost, 6)}."
                        )
                        await self._save_checkpoint(
                            self._current_iteration, self._completion_count
                        )
                        self._stop_event.set()
                        break

                if self._stop_event.is_set():
                    break

                while await self._try_start_new_cycle():
                    pass

        finally:
            if self.interrupted:
                was_interrupted_flag = True

            if (
                self._current_iteration >= self.max_iterations
                and not self._stop_event.is_set()
            ):
                await self._save_checkpoint(
                    self._current_iteration, self._completion_count
                )
                self.logger.info(
                    f"Max iterations ({self.max_iterations}) reached. Cleaning up..."
                )
            else:
                self.logger.info("Main loop finished. Cleaning up running tasks...")
            await self._cleanup_tasks()

        # Calculate total tokens and cost (ensure this happens regardless of exit path)
        if total_tokens == 0.0 and total_cost == 0.0:
            completion_tokens = self.total_completion_tokens
            prompt_tokens = self.total_prompt_tokens
            total_tokens = completion_tokens + prompt_tokens

            # Calculate cost if llm_config has price information
            if self.config.llm_config:
                completion_price = getattr(self.config.llm_config, 'completion_token_price', 0) or 0
                prompt_price = getattr(self.config.llm_config, 'prompt_token_price', 0) or 0
                completion_cost = (completion_tokens / 1000) * completion_price
                prompt_cost = (prompt_tokens / 1000) * prompt_price
                total_cost = completion_cost + prompt_cost
            else:
                completion_cost = 0.0
                prompt_cost = 0.0
                total_cost = 0.0

            self.logger.info(
                f"Final token usage - "
                f"Completion tokens: {completion_tokens}, cost: {round(completion_cost, 6)}. "
                f"Prompt tokens: {prompt_tokens}, cost: {round(prompt_cost, 6)}. "
                f"Total tokens: {total_tokens}, Total Cost: {round(total_cost, 6)}."
            )

        self.logger.info("Evolution process concluded. Invoking Finalizer.")
        final_message = await self.finalizer.finalize(
            self.database,
            start_time=start_time,
            was_interrupted=was_interrupted_flag,
            total_tokens=total_tokens,
            total_cost=total_cost,
        )
        return final_message

    async def _cleanup_tasks(self):
        """Cancels all running asyncio tasks managed by this agent."""
        if not self._running_tasks:
            return

        self.logger.info(
            f"Cancelling {len(self._running_tasks)} active evolution tasks..."
        )
        for task in self._running_tasks:
            task.cancel()

        await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
        self.logger.info("All tasks have been cleaned up.")

    async def interrupt_impl(self):
        """
        Custom interruption logic. Sets the stop event to gracefully halt the process.
        """
        self.logger.info("Interrupt signal received. Setting stop event...")
        if not self._stop_event.is_set():
            self._stop_event.set()

        self.evaluator.interrupt()
        self.logger.info("PESAgent interruption process complete.")
