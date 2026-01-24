# -*- coding: utf-8 -*-
"""
Finalizer component.
"""


import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from loongflow.agentsdk.message import Message, Role
from loongflow.agentsdk.message.elements import EvolveResultElement
from loongflow.framework.evolve.database.database import EvolveDatabase


class Finalizer(ABC):
    """Interface for the Finalizer component."""

    @abstractmethod
    async def finalize(
        self, database: EvolveDatabase, start_time: int, was_interrupted: bool
    ) -> Any:
        """
        Generates the final result of the evolution process based on the state
        stored in the database.

        Args:
            database: The database containing the full history and state.
            start_time: The timestamp when the evolution process started.
            was_interrupted: A boolean indicating if the process was interrupted.

        Returns:
            The final message summarizing the outcome.
        """
        raise NotImplementedError


class LoongFlowFinalizer(Finalizer):
    """
    Default implementation of the Finalizing component.
    It queries the database for the best result, constructs a structured
    EvolveResultElement, and wraps it in a final message.
    """

    async def finalize(
        self, database: EvolveDatabase, start_time: int, was_interrupted: bool
    ) -> Message:
        """
        Queries the database to find the best solution and its metadata,
        then constructs and returns a Message containing an EvolveResultElement.

        Args:
            database: The evolution database to query for results.
            start_time: The timestamp when the evolution process started.
            was_interrupted: A boolean indicating if the process was interrupted.

        Returns:
            A Message object containing either the EvolveResultElement or a status text.
        """
        print("LoongFlow Finalizer: Generating final report from database.")

        end_time = int(time.time())
        cost_time = end_time - start_time
        status_prefix = "Process concluded."
        if was_interrupted:
            status_prefix = "Process was interrupted."

        try:
            memory_status = database.memory_status()
            global_status: Dict[str, Any] = memory_status.get("global_status", {})

            best_score = global_status.get("best_score")
            best_iteration = global_status.get("best_iteration")
            total_iterations = global_status.get("current_iteration", 0)

            if best_score is None or best_iteration is None:
                summary = f"{status_prefix}\nNo solution was successfully scored."
                print(summary)
                return Message.from_text(
                    summary, role=Role.ASSISTANT, sender="LoongFlowFinalizer"
                )

            best_solution = "N/A"
            best_evaluation = "N/A"
            best_solutions = database.get_best_solutions(top_k=1)
            if best_solutions and len(best_solutions) > 0:
                best_solution = best_solutions[0]["solution"]
                best_evaluation = best_solutions[0]["evaluation"]

            start_time_str = datetime.fromtimestamp(start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_time_str = datetime.fromtimestamp(end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            result_element = EvolveResultElement(
                best_score=best_score,
                best_solution=best_solution,
                evaluation=best_evaluation,
                start_time=start_time_str,
                end_time=end_time_str,
                cost_time=cost_time,
                last_iteration=best_iteration,
                total_iterations=total_iterations,
            )

            print(
                f"{status_prefix}\n"
                f"Best score achieved: {result_element.best_score}\n"
                f"Found in iteration: {result_element.last_iteration}\n"
                f"Total iterations: {result_element.total_iterations}\n"
                f"Total cost time: {result_element.cost_time} seconds."
            )

            return Message.from_elements(
                elements=[result_element],
                role=Role.ASSISTANT,
                sender="LoongFlowFinalizer",
            )

        except Exception as e:
            error_summary = (
                f"{status_prefix}\nAn error occurred during finalization: {e}"
            )
            print(f"[ERROR] {error_summary}")
            return Message.from_text(
                error_summary, role=Role.ASSISTANT, sender="LoongFlowFinalizer"
            )
