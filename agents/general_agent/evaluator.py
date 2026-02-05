# -*- coding: utf-8 -*-
"""
General Evaluator for LoongFlow

Supports two evaluation modes:
1. Agent Self-Evaluation: Agent evaluates and returns a 0-1 score with feedback
2. User Evaluation File: Wraps the user's evaluation file as a custom tool for Agent to call

The evaluator inherits from framework's Evaluator base class and is called by executor.

Uses multiprocessing for isolation and precise timeout control.
"""

import asyncio
import copy
import importlib
import json
import multiprocessing
import os
import sys
import traceback
import uuid
from typing import Any, override, Optional

from agents.general_agent.utils import _prepare_solution_pack_context
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import Message, ContentElement
from loongflow.framework.claude_code import (
    GENERAL_EVALUATOR_SIMPLE_SYSTEM,
    GENERAL_EVALUATOR_SIMPLE_USER,
    GENERAL_EVALUATOR_TOOL_SYSTEM,
    GENERAL_EVALUATOR_TOOL_USER,
    DEFAULT_LOADED_SKILLS,
)
from loongflow.framework.pes.context import EvaluatorConfig, Context, Workspace
from loongflow.framework.pes.evaluator.evaluator import (
    Evaluator,
    EvaluationResult,
    EvaluationStatus,
)

logger = get_logger(__name__)


class GeneralEvaluator(Evaluator):
    """
    General Evaluator with two evaluation modes using multiprocessing for isolation.

    1. Self-Evaluation Mode: Agent evaluates directly via system prompt
    2. Custom Tool Mode: User provides an evaluation file that is run in a subprocess

    The evaluate() method is called by executor to trigger the evaluation Agent.
    """

    def __init__(self, config: Any):
        """
        Initialize the evaluator.

        Args:
            config: Can be a GeneralEvaluatorConfig instance, a dict, or an object with llm_config
        """
        # Convert config to GeneralEvaluatorConfig
        if isinstance(config, EvaluatorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = EvaluatorConfig(**config)
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. Expected EvaluatorConfig, dict, or object with llm_config"
            )

        llm_config = copy.deepcopy(self.config.llm_config)
        if not llm_config.model.startswith("anthropic/"):
            raise ValueError(
                "Evaluator: Only support Anthropic model, please use model name like anthropic/xxx."
            )
        llm_config.model = llm_config.model.split("/")[-1]
        self.config.llm_config = llm_config

        self._is_interrupted: bool = False
        self._current_agent = None
        self.work_dir: str = ""
        self._active_processes: dict[str, multiprocessing.Process] = {}

    @override
    async def evaluate(
        self, message: Message, context: Optional[Context] = None
    ) -> EvaluationResult:
        """
        Run evaluation. This method is called by executor.

        Args:
            message: Message containing the solution to evaluate (in ContentElement.data)
            context: Context

        Returns:
            EvaluationResult with score (0-1), status, summary, and metrics
        """
        try:
            if context is None:
                raise ValueError("Context must be provided")

            # Create agent with context-specific work_dir
            work_dir = str(Workspace.get_evaluator_path(context, True))
            # Ensure work_dir is absolute path
            if not os.path.isabs(work_dir):
                work_dir = os.path.abspath(work_dir)
            logger.debug(
                f"[{context.trace_id}] Evaluator: Workspace configured at {work_dir}"
            )

            self.work_dir = work_dir

            # Extract solution directory from message
            solution_dir, plan, task = self._extract_solution(message)

            evaluation_mode = "Custom Tool" if self.config.evaluate_code else "AI Agent"
            logger.info(
                f"[{context.trace_id}] Evaluator: 🔍 Starting {evaluation_mode} evaluation for solution pack: {solution_dir}"
            )

            if self.config.evaluate_code:
                # Custom Tool Mode: Run user's evaluation file in a subprocess
                logger.debug(
                    f"[{context.trace_id}] Evaluator: 🔧 Using custom evaluation file"
                )
                return await self._evaluate_with_custom_file(solution_dir, plan, task)
            else:
                # Self-Evaluation Mode: Use AI Agent to evaluate
                logger.debug(
                    f"[{context.trace_id}] Evaluator: 🤖 Using AI agent evaluation"
                )
                return await self._evaluate_with_ai_agent(solution_dir, plan, task)

        except Exception as e:
            trace_id = context.trace_id if context else "unknown"
            logger.error(f"[{trace_id}] Evaluator: Evaluation failed - {str(e)}")
            return EvaluationResult(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                score=0.0,
                summary=f"Evaluation failed: {str(e)}",
                metrics={"error": str(e)},
            )

    @override
    def interrupt(self) -> None:
        """Interrupt the current evaluation process."""
        logger.warning("Evaluator: Interrupt requested")
        self._is_interrupted = True

        for eval_id, process in list(self._active_processes.items()):
            try:
                if process.is_alive():
                    logger.warning(f"Evaluator: Terminating process {eval_id}")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                logger.error(f"Evaluator: Error terminating process {eval_id}: {e}")

        self._active_processes.clear()

        if self._current_agent:
            # The ClaudeCodeAgent should handle interruption
            pass

    def _extract_solution(self, message: Message) -> tuple[str, str, str]:
        """
        Extract solution directory path from message.

        All solutions MUST be Solution Pack directories.

        Returns:
            Absolute path to solution directory
        """
        elements = message.get_elements(ContentElement)
        if not elements:
            raise ValueError("No ContentElement found in message.")

        data = elements[0].data

        best_solution_path = data.get("best_solution_path")
        best_plan = data.get("best_plan")
        task = data.get("task")

        if not isinstance(best_solution_path, str):
            raise ValueError(
                f"Solution must be a directory path string, got {type(best_solution_path)}"
            )

        if not isinstance(best_plan, str):
            raise ValueError(f"Plan must be a string, got {type(best_plan)}")

        if not isinstance(task, str):
            raise ValueError(f"Task must be a string, got {type(task)}")

        # Ensure absolute path
        if not os.path.isabs(best_solution_path):
            best_solution_path = os.path.abspath(best_solution_path)

        # Validate it's a directory
        if not os.path.exists(best_solution_path):
            raise FileNotFoundError(
                f"Solution directory not found: {best_solution_path}"
            )
        if not os.path.isdir(best_solution_path):
            raise ValueError(
                f"Solution must be a directory (Solution Pack), got file: {best_solution_path}"
            )

        logger.debug(f"Evaluator: Solution Pack directory: {best_solution_path}")
        return best_solution_path, best_plan, task

    def _load_skills(self, work_dir: str) -> None:
        """Load configured skills."""
        if self.config.agent.get("skills"):
            try:
                from .utils import load_skills

                load_skills(skill_names=self.config.agent["skills"], work_dir=work_dir)
                logger.debug(
                    f"Evaluator: Successfully loaded skills: {self.config.agent['skills']}"
                )
            except Exception as e:
                logger.warning(f"Evaluator: Failed to load skills - {e}")

    def _format_loaded_skills(self, work_dir: str) -> str:
        """Format loaded skills information for prompt injection."""
        skills = self.config.agent.get("skills")
        if not skills:
            return DEFAULT_LOADED_SKILLS

        from .utils import format_loaded_skills

        return format_loaded_skills(skills, work_dir)

    def _parse_evaluation_result(self, result_message: Message) -> EvaluationResult:
        """
        Parse the evaluation result from Agent's response.

        The response should contain:
        - Score: a number between 0 and 1
        - Feedback: evaluation feedback
        """
        try:
            # Extract content from message
            content = ""
            if result_message.content and len(result_message.content) > 0:
                if isinstance(result_message.content[0], ContentElement):
                    content = result_message.content[0].data

            # Parse score and feedback from response
            score, feedback = self._extract_score_and_feedback(content)

            # Validate score is non-negative (can be >= 1.0 for exceeding expectations)
            score = max(0.0, float(score))

            # Determine status based on score
            if score > 0.0:
                status = EvaluationStatus.SUCCESS
            else:
                status = EvaluationStatus.VALIDATION_FAILED

            logger.debug(
                "Evaluator parsed result",
                extra={
                    "score": score,
                    "status": status.value,
                    "feedback_length": len(feedback),
                },
            )

            return EvaluationResult(
                status=status,
                score=score,
                summary=feedback or f"Solution evaluated with score {score}",
                metrics={"raw_response": content},
            )

        except Exception as e:
            logger.error(
                "Evaluator failed to parse result",
                extra={
                    "error": str(e),
                    "result_message_type": type(result_message).__name__,
                },
            )
            return EvaluationResult(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                score=0.0,
                summary=f"Failed to parse evaluation result: {str(e)}",
            )

    def _extract_score_and_feedback(self, content: str) -> tuple[float, str]:
        """
        Extract score and feedback from Agent's response.

        Expected format:
        ```
        Score: 0.85
        Feedback: Your detailed feedback here
        ```
        """
        import re

        score = 0.0  # Default score
        feedback = content  # Default feedback

        # Try to extract score
        score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
        if score_match:
            score_str = score_match.group(1)
            try:
                score = float(score_str)
            except ValueError:
                pass

        # Try to extract feedback
        feedback_match = re.search(
            r"Feedback:\s*(.*?)(?:\n\n|\Z)", content, re.IGNORECASE | re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()

        return score, feedback

    async def _run_evaluation_agent(
        self,
        solution: str,
        plan: str,
        task: str,
        system_prompt: str,
        user_prompt: str,
        custom_tools: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Run evaluation using ClaudeCodeAgent.

        This is the common implementation for both evaluation modes.

        Args:
            solution: The solution to evaluate
            plan: The plan chosen by the planner
            task: The original user task
            system_prompt: System prompt for the agent
            user_prompt: User prompt template (will be formatted with solution)
            custom_tools: Optional custom tools for the agent

        Returns:
            EvaluationResult with score and summary
        """
        work_dir = self.work_dir

        # Load skills if configured
        if self.config.agent.get("skills"):
            self._load_skills(work_dir)

        # Format loaded skills information for prompt
        loaded_skills_info = self._format_loaded_skills(work_dir)

        # Import ClaudeCodeAgent here to avoid circular import
        from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent

        # Create evaluation Agent
        agent = ClaudeCodeAgent(
            model=self.config.llm_config.model,
            api_key=self.config.llm_config.api_key,
            url=self.config.llm_config.url,
            work_dir=work_dir,
            disallowed_tools=self.config.agent.get("disallowed_tools"),
            tool_list=self.config.agent.get("build_in_tools"),
            custom_tools=custom_tools or {},
            system_prompt=self.config.agent.get("system_prompt") or system_prompt,
            permission_mode=self.config.agent.get("permission_mode") or "acceptEdits",
            setting_sources=["project"],
            max_turns=self.config.agent.get("max_turns"),
            max_thinking_tokens=self.config.agent.get("max_thinking_tokens"),
        )
        self._current_agent = agent

        # Build evaluation prompt
        prompt = user_prompt.format(
            solution=solution,
            workspace=work_dir,
            loaded_skills=loaded_skills_info,
            plan=plan,
            task=task,
        )

        # Execute evaluation with timeout
        try:
            result_message = await asyncio.wait_for(
                agent.run(prompt), timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Evaluator: Timeout after {self.config.timeout} seconds")
            return EvaluationResult(
                status=EvaluationStatus.FRAMEWORK_ERROR,
                score=0.0,
                summary=f"Evaluation timed out after {self.config.timeout}s",
                metrics={"timeout": self.config.timeout},
            )

        # Parse evaluation result
        return self._parse_evaluation_result(result_message)

    async def _evaluate_with_custom_file(
        self, solution_dir: str, plan: str, task: str
    ) -> EvaluationResult:
        """
        Evaluate using user's evaluation file wrapped as an Agent tool.

        Args:
            solution_dir: Absolute path to Solution Pack directory
            plan: The plan chosen by the planner
            task: The original user task

        Returns:
            EvaluationResult with score, summary, metrics, and artifacts.
        """
        assert self.config.evaluate_code is not None

        # Prepare Solution Pack context (same as AI mode for consistency)
        solution_context = _prepare_solution_pack_context(solution_dir)

        # Create evaluation tool that will pass solution_dir to user script
        eval_tool = _create_user_evaluation_tool(
            workspace_base=self.work_dir,
            evaluate_code=self.config.evaluate_code,
            solution_dir=solution_dir,  # Pass directory path for tool to use
            timeout=self.config.timeout,
            active_processes=self._active_processes,
        )

        return await self._run_evaluation_agent(
            solution=solution_context,  # Show structure to agent
            plan=plan,
            task=task,
            system_prompt=GENERAL_EVALUATOR_TOOL_SYSTEM,
            user_prompt=GENERAL_EVALUATOR_TOOL_USER,
            custom_tools={"evaluate_solution": eval_tool},
        )

    async def _evaluate_with_ai_agent(
        self, solution_dir: str, plan: str, task: str
    ) -> EvaluationResult:
        """
        Evaluate using AI Agent (self-evaluation mode).

        Args:
            solution_dir: Absolute path to Solution Pack directory
            plan: The plan chosen by the planner
            task: The original user task

        Returns:
            EvaluationResult with score and summary.
        """
        # Prepare Solution Pack context for evaluation
        solution_context = _prepare_solution_pack_context(solution_dir)

        return await self._run_evaluation_agent(
            solution=solution_context,
            plan=plan,
            task=task,
            system_prompt=GENERAL_EVALUATOR_SIMPLE_SYSTEM,
            user_prompt=GENERAL_EVALUATOR_SIMPLE_USER,
        )


def _create_user_evaluation_tool(
    workspace_base: str,
    evaluate_code: str,
    solution_dir: str,
    timeout: int = 300,
    active_processes: dict[str, multiprocessing.Process] | None = None,
) -> dict[str, Any]:
    """
    Create a custom tool that wraps the user's evaluation file.

    The tool runs the evaluation in a subprocess with timeout control,
    then returns the full evaluation result for the Agent to analyze.

    Args:
        workspace_base: The workspace base path
        evaluate_code: Evaluation code to wrap
        solution_dir: Absolute path to solution directory (for validation)
        timeout: Maximum time in seconds for the evaluation
        active_processes: Dictionary to track active processes for interruption

    Returns:
        A dict in ClaudeCodeAgent custom_tools format
    """
    if active_processes is None:
        active_processes = {}

    async def run_user_evaluation(args: dict[str, Any]) -> dict[str, Any]:
        """Run the user's evaluation script on the specified file."""
        # Extract file path from agent's call
        file_path = args.get("file_path", "")

        if not file_path:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "framework_error",
                                "score": 0.0,
                                "summary": "file_path parameter is required",
                                "metrics": {},
                                "artifacts": {},
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ]
            }

        # Build absolute path if relative
        if not os.path.isabs(file_path):
            file_path = os.path.join(solution_dir, file_path)

        # Validate file is within solution directory
        file_path_abs = os.path.abspath(file_path)
        solution_dir_abs = os.path.abspath(solution_dir)

        if not file_path_abs.startswith(solution_dir_abs):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "framework_error",
                                "score": 0.0,
                                "summary": f"File path {file_path} is outside solution directory {solution_dir}",
                                "metrics": {},
                                "artifacts": {},
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ]
            }

        # Validate file exists
        if not os.path.exists(file_path_abs):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "framework_error",
                                "score": 0.0,
                                "summary": f"File not found: {file_path}",
                                "metrics": {},
                                "artifacts": {},
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ]
            }

        # Run evaluation in subprocess with the specified file
        result = await _run_evaluation_in_subprocess(
            workspace_base=workspace_base,
            evaluate_code=evaluate_code,
            solution=file_path_abs,  # Pass file path to user script
            timeout=timeout,
            active_processes=active_processes,
        )

        # Return full result for Agent to analyze
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ]
        }

    return {
        "function": run_user_evaluation,
        "description": (
            "Run the user-provided evaluation script on a specific file from the solution pack. "
            "You should analyze the solution pack structure, decide which file to evaluate "
            "(typically the entrypoint specified in index.json), and pass its path to this tool. "
            "Returns a JSON object containing: score (0.0-1.0), summary, status, metrics, and artifacts."
        ),
        "parameters": {"file_path": str},  # Agent specifies which file to evaluate
    }


async def _run_evaluation_in_subprocess(
    workspace_base: str,
    evaluate_code: str,
    solution: str,
    timeout: int,
    active_processes: dict[str, multiprocessing.Process] | None = None,
) -> dict[str, Any]:
    """
    Run user's evaluation file in a subprocess with timeout control.

    Args:
        workspace_base: The workspace base path
        evaluate_code: Evaluation code to wrap.
        solution: The solution content to evaluate
        timeout: Maximum time in seconds for the evaluation
        active_processes: Dictionary to track active processes for interruption

    Returns:
        A dict containing score, summary, status, metrics, and artifacts
    """
    if active_processes is None:
        active_processes = {}

    work_dir = workspace_base
    eval_id = str(uuid.uuid4().hex)
    temp_dir = os.path.join(work_dir, f"eval_{eval_id}")
    os.makedirs(temp_dir, exist_ok=True)

    result_file_path = os.path.join(temp_dir, "evaluation_result.json")
    logger.debug(f"Evaluator: Starting subprocess evaluation {eval_id[:8]}")

    process = None
    try:
        # Validate solution file path exists
        if not os.path.exists(solution):
            raise FileNotFoundError(f"Solution file not found: {solution}")

        solution_file_path = solution
        logger.debug(
            f"Evaluator: Passing solution file to evaluation script: {solution_file_path}"
        )

        evaluation_file_path = os.path.join(temp_dir, "evaluator_code.py")
        with open(evaluation_file_path, "w", encoding="utf-8") as f:
            f.write(evaluate_code)

        # Prepare process args - pass file path
        process_args = (evaluation_file_path, solution_file_path, result_file_path)

        # Create and start process
        process = multiprocessing.Process(
            target=_run_evaluate_target, args=process_args
        )

        active_processes[eval_id] = process
        process.start()

        # Wait for process with timeout
        process.join(timeout=timeout)

        # Check if process is still alive (timeout)
        if process.is_alive():
            logger.error(f"Evaluator: Subprocess timed out after {timeout}s")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

            return {
                "status": "framework_error",
                "score": 0.0,
                "summary": f"Evaluation timed out after {timeout}s",
                "metrics": {"timeout": timeout},
                "artifacts": {},
            }

        # Read evaluation result from file
        if os.path.exists(result_file_path):
            try:
                with open(result_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Evaluator: Failed to decode result file: {e}")
                return {
                    "status": "framework_error",
                    "score": 0.0,
                    "summary": f"Failed to decode evaluation result: {str(e)}",
                    "metrics": {"error": str(e)},
                    "artifacts": {},
                }
            except Exception as e:
                logger.error(f"Evaluator: Failed to read result file: {e}")
                return {
                    "status": "framework_error",
                    "score": 0.0,
                    "summary": f"Failed to read evaluation result: {str(e)}",
                    "metrics": {"error": str(e)},
                    "artifacts": {},
                }
        else:
            logger.error(f"Evaluator: Result file not found")
            return {
                "status": "framework_error",
                "score": 0.0,
                "summary": "Evaluation process completed but result file was not created",
                "metrics": {"result_file": result_file_path},
                "artifacts": {},
            }

    except Exception as e:
        logger.error(f"Evaluator: Subprocess failed: {e}")
        return {
            "status": "framework_error",
            "score": 0.0,
            "summary": f"Evaluation failed: {str(e)}",
            "metrics": {"error": str(e)},
            "artifacts": {},
        }

    finally:
        # Cleanup
        if eval_id in active_processes:
            del active_processes[eval_id]
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        # Note: We don't clean up temp_dir here to allow debugging


def _run_evaluate_target(
    evaluation_file_path: str, solution_path: str, result_file_path: str
) -> None:
    """
    Run user's evaluation function in a separate process and write result to file.

    Supports two modes:
    1. File mode: solution_path is a file -> pass to evaluate()
    2. Directory mode: solution_path is a directory (Solution Pack) -> pass to evaluate()

    The user's evaluate() function should handle both cases:
    - If directory: read index.json to find entrypoint, or use convention
    - If file: evaluate directly

    Args:
        evaluation_file_path: Path to the user's evaluation file
        solution_path: Path to the solution (file or directory)
        result_file_path: Path to write evaluation result JSON file

    Returns:
        None (results are written to result_file_path for parent process to read)
    """
    # Resolve evaluation module
    eval_dir = os.path.dirname(evaluation_file_path)
    eval_filename = os.path.basename(evaluation_file_path)
    module_name = os.path.splitext(eval_filename)[0]

    # Add eval_dir to sys.path for module import
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    result_data = {
        "status": "execution_failed",
        "score": 0.0,
        "summary": "",
        "metrics": {},
        "artifacts": {},
    }

    try:
        # Import evaluation module
        eval_module = importlib.import_module(module_name)

        if not hasattr(eval_module, "evaluate"):
            raise AttributeError(
                f"The evaluator module must contain an 'evaluate' function."
            )

        eval_func = getattr(eval_module, "evaluate")

        # Call evaluate function (supports both file and directory)
        eval_result = eval_func(solution_path)

        # Parse and normalize the result
        if isinstance(eval_result, dict):
            result_data["status"] = eval_result.get("status", "success")
            score = eval_result.get("score", 0.0)
            if isinstance(score, (int, float)):
                score = max(0.0, float(score))  # Allow score >= 1.0
            else:
                score = 0.0
            result_data["score"] = score
            result_data["summary"] = eval_result.get("summary", "Evaluation completed")
            result_data["metrics"] = eval_result.get("metrics", {})
            result_data["artifacts"] = eval_result.get("artifacts", {})

        elif isinstance(eval_result, (int, float)):
            score = max(0.0, float(eval_result))  # Allow score >= 1.0
            result_data["score"] = score
            result_data["status"] = "success"
            result_data["summary"] = f"Evaluation result: {eval_result}"

        elif isinstance(eval_result, str):
            # Try to parse score from string
            try:
                score = float(eval_result)
                score = max(0.0, score)  # Allow score >= 1.0
                result_data["score"] = score
                result_data["status"] = "success"
                result_data["summary"] = eval_result
            except ValueError:
                result_data["score"] = 0.0
                result_data["summary"] = eval_result
        else:
            raise TypeError(
                f"The 'evaluate' function must return a dict, float, int, or str, but got {type(eval_result)}"
            )

        logger.info(
            f"Evaluation completed: status={result_data['status']}, score={result_data['score']}, summary={result_data['summary']}"
        )

    except Exception as e:
        result_data["status"] = "execution_failed"
        result_data["summary"] = f"Evaluation failed: {str(e)}"
        result_data["artifacts"] = {
            "stderr": f"{traceback.format_exc()}",
            "exception": str(e),
        }

    # Write result to file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write result file: {e}")


# Factory function for easy creation
def create_evaluator(config: EvaluatorConfig) -> GeneralEvaluator:
    """
    Factory function to create a GeneralEvaluator.

    Args:
        config: EvaluatorConfig instance
        **kwargs: Additional configuration options (system_prompt, build_in_tools, skills, timeout, etc.)

    Returns:
        GeneralEvaluator instance

    Example:
        # Self-evaluation mode
        evaluator = create_evaluator(llm_config=my_config)

        # Custom tool mode with evaluation file
        evaluator = create_evaluator(
            llm_config=my_config,
            evaluation_file_path="my_evaluator.md"
        )
    """
    return GeneralEvaluator(config)
