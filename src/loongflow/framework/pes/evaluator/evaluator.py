"""Evaluator"""

import asyncio
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loongflow.agentsdk.logger.logger import get_logger
from loongflow.agentsdk.message.elements import ContentElement
from loongflow.agentsdk.message.message import Message
from loongflow.framework.pes.context import EvaluatorConfig, Context


class EvaluationStatus(str, Enum):
    """
    Represents the outcome of an evaluation.
    """

    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"
    FRAMEWORK_ERROR = "framework_error"


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing metrics, artifacts, overall score and LLM summary.
    """

    status: EvaluationStatus = EvaluationStatus.FRAMEWORK_ERROR
    summary: str = "Evaluation did not complete due to a framework error."
    score: float = 0.0
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    @classmethod
    def from_any_dict(cls, data: dict) -> "EvaluationResult":
        """Create an EvaluationResult instance from a generic dictionary returned by user's evaluate function."""
        if not isinstance(data, dict):
            return cls(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                summary=f"User's evaluate function returned a non-dict type: {type(data)}",
            )

        status_str = data.get("status")
        try:
            status = EvaluationStatus(status_str)
        except (ValueError, TypeError):
            return cls(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                summary=(
                    f"User's evaluate function must return a dict with a 'status' field. "
                    f"Got status: '{status_str}'."
                ),
                score=float(data.get("score", 0.0)),
                artifacts={"raw_result": data},
            )

        score = data.get("score", 0.0)
        if not isinstance(score, (int, float)):
            return cls(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                summary=(
                    f"User's evaluate function must return a dict with a 'score' field that is either "
                    f"an integer or a float. Got score: '{score}'."
                ),
                artifacts={"raw_result": data},
            )

        return cls(
            status=status,
            summary=data.get("summary", ""),
            score=float(data.get("score", 0.0)),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResult":
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return cls(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                summary="Invalid JSON format.",
                artifacts={"raw_result": json_str},
            )
        return cls.from_any_dict(data)

    def to_dict(self) -> dict:
        """Convert to plain dict."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(
        self, message: Message, context: Optional[Context] = None
    ) -> "EvaluationResult":
        """Run evaluation"""
        pass

    @abstractmethod
    def interrupt(self):
        """Interrupt evaluation"""
        pass


class LoongFlowEvaluator(Evaluator):
    """
    LoongFlow Evaluator
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._logger = get_logger(self.__class__.__name__)
        self._thread_executor = concurrent.futures.ThreadPoolExecutor()

        self._active_processes: Dict[str, multiprocessing.Process] = {}
        self._processes_lock = threading.Lock()

    @staticmethod
    def _run_evaluate_target(evaluator_file_path: str, llm_file_path: str):
        """
        Run the evaluator code in a separate process and write results to a file.
        """
        base_dir = os.path.dirname(evaluator_file_path)
        output_file_path = os.path.join(base_dir, "evaluation_result.json")
        log_file_path = os.path.join(base_dir, "evaluation_process.log")

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        log_file = open(log_file_path, "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file

        logger = get_logger("LoongFlowEvaluator_Child")
        pid = os.getpid()
        logger.debug(
            f"[Child PID:{pid}] Started. Evaluator path: {evaluator_file_path}"
        )

        result_data = {}

        try:
            # Dynamically load evaluator code module.
            spec = importlib.util.spec_from_file_location(
                "evaluator_code", evaluator_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Module spec creation failed for {evaluator_file_path}"
                )

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if not hasattr(mod, "evaluate"):
                raise AttributeError(
                    "The evaluator module must contain an 'evaluate' function."
                )

            start_time = time.time()
            result = mod.evaluate(llm_file_path)
            duration = time.time() - start_time
            logger.debug(f"[Child PID:{pid}] evaluate() finished in {duration:.4f}s.")

            if not isinstance(result, dict):
                raise TypeError(
                    f"The 'evaluate' function must return a dict, but got {type(result)}"
                )

            result_data = result

        except Exception as e:
            logger.error(f"[Child PID:{pid}] Exception occurred: {e}")
            logger.error(traceback.format_exc())
            result_data = {"error": str(e), "traceback": traceback.format_exc()}

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
        except Exception as write_err:
            logger.error(f"[Child PID:{pid}] Failed to write result file: {write_err}")

        logger.debug(f"[Child PID:{pid}] Process logic finished. Forcing exit.")

        try:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
        except:
            pass

        logger.debug(f"Subprocess exit now. Result score: \
{result_data.get('score', '')}. Result summary: {result_data.get('summary', '')}")
        os._exit(0)

    def _execute_in_process_with_timeout(
        self, eval_id: str, evaluator_file_path: str, llm_file_path: str
    ) -> dict:
        """
        Execute the evaluation in a separate process with a timeout, reading result from file.
        """
        self._logger.debug(
            f"[Parent] Preparing to spawn process for eval_id: {eval_id}"
        )

        # Note: We no longer pass a Queue
        process_args = (evaluator_file_path, llm_file_path)
        process = multiprocessing.Process(
            target=self.__class__._run_evaluate_target, args=process_args
        )

        with self._processes_lock:
            self._active_processes[eval_id] = process

        try:
            process.start()
            process.join(timeout=self.config.timeout)

            if process.is_alive():
                self._logger.debug(
                    f"[Parent] TIMEOUT: Process (pid: {process.pid}) is still alive after {self.config.timeout}s."
                )
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    self._logger.debug(
                        f"[Parent] Process (pid: {process.pid}) did not terminate gracefully, killing."
                    )
                    process.kill()
                raise TimeoutError(
                    f"Evaluation execution timed out (>{self.config.timeout}s)"
                )

            self._logger.debug(
                f"[Parent] Process joined. Exit code: {process.exitcode}"
            )

            if process.exitcode != 0:
                self._logger.debug(
                    f"Process exited with non-zero code: {process.exitcode}."
                )
                # We still try to read the file, as the child might have written an error dict before crashing/exiting

            # Read the result from the file
            base_dir = os.path.dirname(evaluator_file_path)
            output_file_path = os.path.join(base_dir, "evaluation_result.json")
            log_file_path = os.path.join(base_dir, "evaluation_process.log")

            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        result_dict = json.load(f)
                    return result_dict
                except json.JSONDecodeError as e:
                    return {
                        "error": f"Failed to decode result JSON: {e}",
                        "traceback": "",
                    }
                except Exception as e:
                    return {
                        "error": f"Failed to read result file: {e}",
                        "traceback": "",
                    }
            else:
                crash_log = ""
                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path, "r", encoding="utf-8") as f:
                            crash_log = f.read()[-500:]  # last 500
                    except:
                        pass

                if process.exitcode != 0:
                    return {
                        "error": f"Evaluation process exited with non-zero code: {process.exitcode}",
                        "traceback": crash_log,
                    }
                return {
                    "error": "Evaluation process finished successfully (exit 0)",
                    "traceback": "",
                }

        except Exception as e:
            if isinstance(e, TimeoutError):
                raise
            self._logger.error(f"[Parent] Exception during process execution: {e}")
            return {
                "error": f"Exception during process execution: {e}",
                "traceback": traceback.format_exc(),
            }
        finally:
            with self._processes_lock:
                if eval_id in self._active_processes:
                    del self._active_processes[eval_id]

            if process.is_alive():
                self._logger.debug(
                    f"[Parent] Cleaning up lingering process (pid: {process.pid}) in finally block."
                )
                process.terminate()
                process.kill()
            else:
                self._logger.debug(
                    f"[Parent] Process (pid: {process.pid}) cleanup complete."
                )

    async def evaluate(
        self, message: Message, context: Optional[Context] = None
    ) -> EvaluationResult:
        try:
            code_to_evaluate = self._extract_evolution_context(message)
        except ValueError as e:
            self._logger.error(f"Failed to extract solution: {e}", exc_info=True)
            return EvaluationResult(
                score=0.0, metrics={"error": f"Failed to extract solution: {e}"}
            )

        workspace_base = self.config.workspace_path
        eval_id = str(uuid.uuid4().hex)
        temp_dir = os.path.join(workspace_base, f"eval_{eval_id}")
        os.makedirs(temp_dir, exist_ok=True)

        self._logger.info(f"Starting evaluation {eval_id} in {temp_dir}")

        try:
            evaluate_code = self.config.evaluate_code
            llm_filename = f"llm_code_{eval_id}.py"
            llm_file_path = os.path.join(temp_dir, llm_filename)
            with open(llm_file_path, "w", encoding="utf-8") as f:
                f.write(code_to_evaluate)

            evaluator_file_path = os.path.join(temp_dir, "evaluator_code.py")
            with open(evaluator_file_path, "w", encoding="utf-8") as f:
                f.write(evaluate_code)

            loop = asyncio.get_running_loop()

            result_dict = await loop.run_in_executor(
                self._thread_executor,
                self._execute_in_process_with_timeout,
                eval_id,
                evaluator_file_path,
                llm_file_path,
            )

            if isinstance(result_dict, dict) and "error" in result_dict:
                if "interrupted" in str(result_dict["error"]):
                    return EvaluationResult(score=0.0, metrics=result_dict)
                return EvaluationResult(score=0.0, metrics=result_dict)

            result = EvaluationResult.from_any_dict(result_dict)
            self._logger.info(f"Evaluation completed. \
Status: {result.status}, Score: {result.score}, Summary: {result.summary}")
            return result
        except TimeoutError as e:
            self._logger.error(str(e))
            return EvaluationResult(
                score=0.0,
                status=EvaluationStatus.FRAMEWORK_ERROR,
                summary="Evaluation execution timed out.",
                metrics={"error": str(e)},
            )
        except Exception as e:
            self._logger.error(
                f"Unexpected error in evaluation orchestration: {e}", exc_info=True
            )
            if isinstance(e, RuntimeError) and "cannot schedule new futures" in str(e):
                return EvaluationResult(
                    score=0.0,
                    status=EvaluationStatus.FRAMEWORK_ERROR,
                    summary="Evaluation execution was interrupted.",
                    metrics={"error": "Evaluator was interrupted."},
                )
            raise
        finally:
            # shutil.rmtree(temp_dir, ignore_errors=True)
            pass

    def _extract_evolution_context(
        self, message: Message
    ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Extracts current solution, parent solution, and parent evaluation from the message.

        This method encapsulates the logic for navigating the message structure to retrieve
        the necessary context for summarization or other evolutionary operations.

        Args:
            message: The input message containing evolution process data.

        Returns:
            A tuple containing:
            - current_solution (str): The code solution from the current generation.
            - parent_solution (str | None): The code solution from the parent, if available.
            - parent_evaluation (dict | None): The evaluation result from the parent, if available.

        Raises:
            ValueError: If the required elements (ContentElement, data) are not found.
        """
        elements = message.get_elements(ContentElement)
        if not elements:
            raise ValueError("No ContentElement found in the message.")
        element = elements[0]
        return element.data

    def interrupt(self):
        """
        Interrupts all ongoing evaluation processes in parallel and shuts down the thread pool.
        This implementation is optimized for high concurrency scenarios.
        """
        self._logger.info(
            "Interrupt signal received. Terminating all active evaluation processes..."
        )

        processes_to_terminate: List[multiprocessing.Process] = []
        with self._processes_lock:
            processes_to_terminate = list(self._active_processes.values())
            self._active_processes.clear()

        if not processes_to_terminate:
            self._logger.info("No active evaluation processes to terminate.")
        else:
            self._logger.info(
                f"Attempting to interrupt {len(processes_to_terminate)} processes."
            )

            for process in processes_to_terminate:
                try:
                    if process.is_alive():
                        self._logger.warning(
                            f"Sending SIGTERM to process pid: {process.pid}"
                        )
                        process.terminate()
                except Exception as e:
                    self._logger.error(
                        f"Error sending SIGTERM to process {process.pid}: {e}",
                        exc_info=False,
                    )

            grace_period = 1.0
            self._logger.info(
                f"Waiting for {grace_period}s for processes to terminate gracefully..."
            )
            time.sleep(grace_period)

            still_alive_processes = [p for p in processes_to_terminate if p.is_alive()]
            if still_alive_processes:
                self._logger.warning(
                    f"{len(still_alive_processes)} processes did not terminate gracefully. Sending SIGKILL."
                )
                for process in still_alive_processes:
                    try:
                        if process.is_alive():
                            self._logger.error(
                                f"Process {process.pid} did not terminate. Forcing kill (SIGKILL)."
                            )
                            process.kill()
                    except Exception as e:
                        self._logger.error(
                            f"Error sending SIGKILL to process {process.pid}: {e}",
                            exc_info=False,
                        )
            else:
                self._logger.info("All processes terminated gracefully.")

        self._logger.info("Shutting down the thread executor.")
        self._thread_executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        if hasattr(self, "_thread_executor") and not self._thread_executor._shutdown:
            self.interrupt()
            self._thread_executor.shutdown(wait=False, cancel_futures=True)
