#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides general planner implementation based on Claude Code Agent
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from agents.general_agent.common import ClaudeAgentConfig
from agents.general_agent.evaluator import GeneralEvaluator
from agents.general_agent.utils import convert_function_tool_to_custom_tool, format_loaded_skills
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import Message, ContentElement, Role, MimeType
from loongflow.agentsdk.tools import FunctionTool
from loongflow.framework.claude_code import (
    GENERAL_EXECUTOR_SYSTEM,
    GENERAL_EXECUTOR_USER,
)
from loongflow.framework.pes.context import Context, Workspace
from loongflow.framework.pes.register import Worker
from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent

logger = get_logger(__name__)

BEST_SOLUTION_FILE = "best_solution.md"
BEST_EVALUATION_FILE = "best_evaluation.json"
EPSILON = 1e-9


class EvaluateSolutionArgs(BaseModel):
    """Arguments for running LoongFlowEvaluator as a FunctionTool."""

    solution_file_path: str = Field(
        description="Actual Solution Content file path to be evaluated by GeneralEvaluator."
    )


def _create_evaluation_tool(
    evaluator: GeneralEvaluator, context: Context, round_idx: int, candidate_idx: int
) -> FunctionTool:
    """
    Create a custom tool wrapper for the evaluator.

    The tool will call the evaluator's evaluate method and save the result to disk.
    """
    # Pre-compute the evaluation file path
    candidate_dir = Workspace.get_executor_candidate_path(
        context, f"{round_idx}_{candidate_idx}"
    )

    async def evaluate_candidate(solution_file_path: str):
        """Evaluate a candidate solution.

        Args:
            solution_file_path (str): Path to the solution file.
        """

        # Normalize paths for comparison
        candidate_path_abs = os.path.abspath(candidate_dir)

        if os.path.isabs(solution_file_path):
            requested_path_abs = os.path.abspath(solution_file_path)
        else:
            # Try interpreting path relative to candidate_path
            path_rel_candidate = os.path.abspath(
                os.path.join(candidate_dir, solution_file_path)
            )
            # Try interpreting path relative to CWD
            path_rel_cwd = os.path.abspath(solution_file_path)

            # Heuristic: if path relative to candidate_path does not exist,
            # but path relative to CWD exists and is within candidate_path, use it.
            if (
                not os.path.exists(path_rel_candidate)
                and os.path.exists(path_rel_cwd)
                and path_rel_cwd.startswith(candidate_path_abs)
            ):
                requested_path_abs = path_rel_cwd
            else:
                requested_path_abs = path_rel_candidate

        # Strict path validation: ensure requested path is within candidate_path
        if not requested_path_abs.startswith(candidate_path_abs):
            raise ValueError(
                f"Path '{solution_file_path}' is outside permitted workspace '{candidate_dir}'"
            )

        # Use code absolute path
        solution_file_path = requested_path_abs

        # Verify path exists and is a file
        if not os.path.exists(solution_file_path):
            raise ValueError(f"File '{solution_file_path}' does not exist")
        if not os.path.isfile(solution_file_path):
            raise ValueError(f"Path '{solution_file_path}' is not a file")

        logger.debug(
            f"[{context.trace_id}] Executor: Solution file validated: {solution_file_path}"
        )
        # Read the solution code from file
        try:
            with open(solution_file_path, "r") as f:
                full_solution = f.read()
        except Exception as e:
            logger.error(
                f"[{context.trace_id}] Executor: Failed to read solution file: {str(e)}"
            )
            return {
                "error": f"Failed to read solution file '{solution_file_path}': {str(e)}"
            }

        random_str = uuid.uuid4().hex[:3]
        evaluation_file_path = f"{candidate_dir}/evaluation_{random_str}.json"

        actual_solution = _extract_implementation_section(full_solution)
        actual_solution_file_path = f"{candidate_dir}/actual_solution_{random_str}.md"
        with open(actual_solution_file_path, "w", encoding="utf-8") as f:
            f.write(actual_solution)

        message = Message.from_text(data=actual_solution, role="user")
        try:
            result = await evaluator.evaluate(message, context)
            if result is None:
                eval_data = {
                    "score": 0.0,
                    "summary": "No evaluation data available",
                    "status": "no_evaluation_data",
                    "metrics": {},
                }
            else:
                eval_data = {
                    "score": result.score,
                    "summary": result.summary,
                    "status": result.status.value,
                    "metrics": result.metrics,
                }

            # Save to disk for load_results_for_candidate to read
            with open(evaluation_file_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"[{context.trace_id}] Executor: Evaluation completed, score={result.score}"
            )

            return {**eval_data}
        except Exception as e:
            logger.error(f"[{context.trace_id}] Executor: Evaluation failed: {e}")
            error_data = {
                "score": 0.0,
                "summary": f"Evaluation failed: {str(e)}",
                "status": "framework_error",
                "metrics": {"error": str(e)},
            }
            # Also save error result to disk
            try:
                with open(evaluation_file_path, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
            except:
                pass
            return {**error_data}

    return FunctionTool(
        func=evaluate_candidate,
        args_schema=EvaluateSolutionArgs,
        name="evaluate_candidate",
        description="Evaluate a candidate solution.",
    )


@dataclass
class ExecutionContext:
    """Holds previous candidate and stage1 plan information."""

    parent_info_file_path: str
    parent_core: float
    parent_solution: str
    stage1_plan: str
    stage1_plan_file_path: str


@dataclass
class CandidateResult:
    """Internal representation of a single solution/evaluation pair read from disk.

    source: 'disk' when read from evaluation.json/solution files; 'llm' when no disk
    pairs exist, and we only have LLM status/reason.
    """

    round_idx: int = 0
    candidate_idx: int = 0
    random_idx: Optional[str] = None

    solution_file_path: str = ""
    evaluation_file_path: str = ""

    score: float = 0.0

    reason: str = ""

    source: str = "disk"

    def to_dict(self) -> dict:
        """Convert to plain dict."""
        return asdict(self)


@dataclass
class HistoryRecord:
    """Maintains per-round evolution history."""

    records: list = field(default_factory=list)

    def add_round(self, round_idx: int, candidates: list):
        """Add a round into the history."""
        self.records.append({"round": round_idx, "candidates": candidates})

    def to_list(self) -> list:
        """Convert to plain list."""
        return self.records


def _extract_implementation_section(content: str) -> str:
    """
    Extract the Implementation section from the solution markdown.

    Args:
        content: The full solution markdown content.

    Returns:
        The Implementation section content, or the full content if not found.
    """
    # Try to find ## Implementation section (until next ## or end of content)
    pattern = r"## Implementation\s*\n(.*?)(?=\n## Key Improvements |\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        implementation = match.group(1).strip()
        if implementation:
            logger.debug("Executor: Extracted Deliverables section from response")
            return implementation

    # Fallback: return the whole content if no Implementation section found
    logger.debug("Executor: No Deliverables section found, using full response")
    return content


class GeneralExecuteAgent(Worker):
    """Plan Agent Class"""

    def __init__(self, config: Any, evaluator: GeneralEvaluator):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ClaudeAgentConfig)
            else ClaudeAgentConfig(**config)
        )

        if self.config.llm_config is None:
            raise ValueError(
                "Executor: No LLMConfig found in config, please check your config."
            )

        self.evaluator = evaluator

        logger.debug("Executor: Core configuration loaded successfully")

    async def run(self, context: Context, message: Message) -> Message:
        """Execute execution phase."""
        logger.info(
            f"[{context.trace_id}] Executor: 🚀 Starting iteration {context.current_iteration}/{context.total_iterations}"
        )

        parent_ctx = self._parse_message_inputs(message)

        all_results: List[CandidateResult] = []
        history = HistoryRecord()
        previous_attempts = ""
        total_completion_tokens = 0
        total_prompt_tokens = 0

        # Create agent with context-specific work_dir
        work_dir = str(Workspace.get_executor_path(context, True))

        # Ensure work_dir is absolute path
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        logger.debug(
            f"[{context.trace_id}] Executor: Workspace configured at {work_dir}"
        )

        # Load skills if specified
        if self.config.skills:
            from .utils import load_skills

            try:
                load_skills(
                    skill_names=self.config.skills,
                    work_dir=work_dir,
                )
                logger.debug(
                    f"[{context.trace_id}] Executor: Successfully loaded skills: {self.config.skills}"
                )
            except Exception as e:
                logger.error(
                    f"[{context.trace_id}] Executor: Failed to load skills - {str(e)}"
                )
                raise

        # iterate rounds explicitly
        for round_idx in range(self.config.max_rounds):
            # Optimized logging: only log every 10 rounds or first few
            if round_idx % 10 == 0 or round_idx < 5:
                logger.info(
                    f"[{context.trace_id}] Executor: Round {round_idx} started"
                )
            else:
                logger.debug(
                    f"[{context.trace_id}] Executor: Round {round_idx} processing"
                )

            # call gen_multi_candidate (handles concurrency)
            round_results_dict = await self.gen_multi_candidate(
                context, parent_ctx, round_idx, 1, previous_attempts
            )

            round_results = round_results_dict.get("candidates", [])
            total_completion_tokens += round_results_dict.get(
                "total_completion_tokens", 0
            )
            total_prompt_tokens += round_results_dict.get("total_prompt_tokens", 0)

            # Write round-level history (append all candidate entries found this round)
            history.add_round(round_idx, [r.to_dict() for r in round_results])
            Workspace.write_executor_history(
                context, json.dumps(history.to_list(), ensure_ascii=False, indent=2)
            )

            if not round_results:
                logger.debug(
                    f"[{context.trace_id}] Executor: Round {round_idx} - No candidates generated"
                )
                continue

            all_results.extend(round_results)

            # Consider only disk-sourced results for improvement check
            disk_results = [r for r in round_results if r.source == "disk"]
            better = [
                r for r in disk_results if r.score - parent_ctx.parent_core > EPSILON
            ]
            if better:
                best = max(better, key=lambda r: r.score)
                logger.info(
                    f"[{context.trace_id}] Executor: ?? Round {round_idx} - Improvement detected - score={best.score:.4f} > parent={parent_ctx.parent_core:.4f}"
                )
                self._write_best_results(context, best)
                # Stop early on improvement
                break
            else:
                for result in disk_results:
                    previous_attempts += (
                        f"Round {round_idx}, Candidate {result.candidate_idx}, "
                        + f"Evaluation: {result.reason}\n\n"
                    )
                logger.debug(
                    f"[{context.trace_id}] Executor: Round {round_idx} - No improvement"
                )

        if not all_results:
            logger.warning(
                f"[{context.trace_id}] Executor: No candidates generated in any round (attempted {self.config.max_rounds} rounds)"
            )
            return self._make_result_message(
                parent_ctx, None, total_prompt_tokens, total_completion_tokens
            )

        # Ensure best exists on disk; if not, pick top disk-scoring; fallback to any
        disk_all = [r for r in all_results if r.source == "disk"]
        if disk_all:
            chosen = max(disk_all, key=lambda r: r.score)
        else:
            chosen = max(all_results, key=lambda r: r.score)

        best_solution_path = Workspace.get_executor_best_solution_path(context)
        if not Path(best_solution_path).exists() and chosen:
            logger.debug(
                f"[{context.trace_id}] Executor: Writing best candidate - score={chosen.score:.4f}"
            )
            self._write_best_results(context, chosen)

        logger.info(
            f"[{context.trace_id}] Executor: ✅ Iteration completed - final_score={chosen.score:.4f}, tokens={total_prompt_tokens + total_completion_tokens}, candidates={len(all_results)}"
        )
        return self._make_result_message(
            parent_ctx, chosen, total_prompt_tokens, total_completion_tokens
        )

    async def gen_multi_candidate(
        self,
        context: Context,
        parent_ctx: ExecutionContext,
        round_idx: int,
        parallel_candidates: int,
        previous_attempts: str,
    ) -> Dict[str, Any]:
        """Generate multiple candidate slots concurrently.

        Concurrency strategy:
        - Launch gen_one_candidate for each parallel slot concurrently.
        - Gather LLM outputs, then for each slot call load_results_for_candidate which reads disk files
          and attaches LLM status/reason to each disk entry. If load finds no disk pairs, a single
          LLM-only CandidateResult (source='llm') is returned for that slot.
        - Flatten results and return as list[CandidateResult].
        """
        tasks = [
            self.gen_one_candidate(context, parent_ctx, round_idx, i, previous_attempts)
            for i in range(parallel_candidates)
        ]
        llm_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {
            "candidates": [],
            "total_completion_tokens": 0,
            "total_prompt_tokens": 0,
        }

        for idx, llm_out in enumerate(llm_results):
            llm_str = str(llm_out) if isinstance(llm_out, Exception) else llm_out

            llm_out_dict = {}
            try:
                llm_out_dict = json.loads(llm_str)
            except json.JSONDecodeError:
                logger.error(
                    f"[{context.trace_id}] Executor: Failed to parse LLM output JSON"
                )
                logger.debug(f"[{context.trace_id}] Executor: Raw LLM output: {llm_str[:500]}")
            final_results["idx"] = idx
            final_results["total_completion_tokens"] += llm_out_dict.get(
                "total_completion_tokens", 0
            )
            final_results["total_prompt_tokens"] += llm_out_dict.get(
                "total_prompt_tokens", 0
            )

            try:
                slot_results = self.load_results_for_candidate(
                    context, round_idx, idx, llm_str
                )
                final_results["candidates"].extend(slot_results)
            except Exception as e:
                logger.error(
                    f"[{context.trace_id}] Executor: Error loading results for slot {idx}: {e}"
                )
                # ensure at least lLM-only entry
                fallback = CandidateResult(
                    round_idx=round_idx,
                    candidate_idx=idx,
                    solution_file_path="",
                    evaluation_file_path="",
                    score=0.0,
                    reason=llm_str,
                    source="llm",
                )
                final_results["candidates"].append(fallback)

        return final_results

    async def gen_one_candidate(
        self,
        context: Context,
        parent_ctx: ExecutionContext,
        round_idx: int,
        candidate_idx: int,
        previous_attempts: str,
    ) -> str | None:

        work_dir = str(Workspace.get_executor_path(context, True))
        # Ensure work_dir is absolute path
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)

        # Create evaluation tool for candidate assessment
        evaluation_function_tool = _create_evaluation_tool(
            self.evaluator, context, round_idx, candidate_idx
        )
        evaluation_tool_config = convert_function_tool_to_custom_tool(
            evaluation_function_tool
        )
        custom_tools = {"evaluate_candidate": evaluation_tool_config}

        logger.debug(f"[{context.trace_id}] Executor: Created evaluation tool for candidate {round_idx}_{candidate_idx}")

        agent = ClaudeCodeAgent(
            model=self.config.llm_config.model,
            api_key=self.config.llm_config.api_key,
            url=self.config.llm_config.url,
            work_dir=work_dir,
            tool_list=self.config.build_in_tools,
            custom_tools=custom_tools,
            system_prompt=self.config.system_prompt or GENERAL_EXECUTOR_SYSTEM,
            permission_mode=self.config.permission_mode or "acceptEdits",
            setting_sources=["project"],
            max_turns=self.config.max_turns,
            max_thinking_tokens=self.config.max_thinking_tokens,
        )

        # Get the expected execute path
        # Use absolute path to avoid any relative path confusion
        candidate_solution_full_path = str(
            Workspace.get_executor_candidate_path(
                context, f"{round_idx}_{candidate_idx}"
            )
        )
        # Pass absolute path to Claude - Claude agent will handle it correctly regardless of current working directory
        candidate_solution_path_for_claude = f"{candidate_solution_full_path}/full_solution.md"  # Already the correct absolute path since Workspace returns absolute paths

        # Format loaded skills information for prompt
        loaded_skills_info = format_loaded_skills(self.config.skills, work_dir)

        user_prompt = GENERAL_EXECUTOR_USER.format(
            task_info=context.task,
            improvement_plan=parent_ctx.stage1_plan,
            parent_score=parent_ctx.parent_core,
            parent_solution=parent_ctx.parent_solution,
            previous_attempts=previous_attempts,
            solution_path=candidate_solution_path_for_claude,
            workspace=work_dir,
            loaded_skills=loaded_skills_info,
        )

        if previous_attempts:
            logger.debug(
                f"[{context.trace_id}] Executor: Has {len(previous_attempts.split('Round'))-1} previous attempts"
            )

        # Execute execution - Claude should use Write tool to save plan to best_plan_path
        result = await agent.run(user_prompt)

        # Check if Claude wrote the candidate solution file (primary path)
        if os.path.exists(candidate_solution_path_for_claude):
            logger.info(
                f"[{context.trace_id}] Executor: ✅ Candidate {round_idx}_{candidate_idx} generated"
            )
        else:
            # Fallback: extract candidate from Claude's response and save it manually
            logger.warning(
                f"[{context.trace_id}] Executor: Candidate file not found, extracting from response"
            )

        final = {
            "content": (
                result.content[0].data
                if (
                    result.content
                    and len(result.content) > 0
                    and isinstance(result.content[0], ContentElement)
                )
                else ""
            ),
            "total_prompt_tokens": result.metadata.get("input_tokens"),
            "total_completion_tokens": result.metadata.get("output_tokens"),
        }

        return json.dumps(final, ensure_ascii=False)

    def load_results_for_candidate(
        self,
        context: Context,
        round_idx: int,
        candidate_idx: int,
        llm_out: str,
    ) -> List[CandidateResult]:
        """Read a candidate folder and return all paired solution/evaluation results.

        Pairing logic: find all files matching solution*.md and evaluation*.json and pair
        them by the suffix present in the filename. Example:
            solution1_23.md <-> evaluation1_23.json  (suffix: '1_23' or '23' depending on naming)

        Returns list of CandidateResult with source='disk'. If none found, returns single
        CandidateResult with source='llm' containing the LLM status/reason as record.
        """
        candidate_path = Workspace.get_executor_candidate_path(
            context, f"{round_idx}_{candidate_idx}"
        )
        p = Path(candidate_path)

        if not p.exists() or not p.is_dir():
            # no folder -> return llm-only fallback
            logger.debug(
                f"[{context.trace_id}] Executor: Candidate folder not found: {candidate_path}"
            )
            return [
                CandidateResult(
                    round_idx=round_idx,
                    candidate_idx=candidate_idx,
                    solution_file_path="",
                    evaluation_file_path="",
                    score=0.0,
                    reason=llm_out,
                    source="llm",
                )
            ]

        # gather files
        solution_candidates = list(p.glob("actual_solution*.md"))
        eval_candidates = list(p.glob("evaluation*.json"))

        # Build maps keyed by extracted random_idx token
        def extract_suffix(fp: Path, prefix: str):
            name = fp.name
            if not name.startswith(prefix):
                return None
            core = name[len(prefix) :]
            core = core.rsplit(".", 1)[0]
            return core

        sol_map: Dict[str, Path] = {}
        for s in solution_candidates:
            key = extract_suffix(s, "actual_solution")
            if key:
                sol_map[key] = s

        eval_map: Dict[str, Path] = {}
        for e in eval_candidates:
            key = extract_suffix(e, "evaluation")
            if key:
                eval_map[key] = e

        paired_keys = sorted(set(sol_map.keys()) & set(eval_map.keys()))

        results: List[CandidateResult] = []
        for k in paired_keys:
            s_path = str(sol_map[k])
            e_path = str(eval_map[k])

            try:
                with open(e_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
            except Exception as e:
                logger.warning(
                    f"[{context.trace_id}] Executor: Failed to read evaluation file: {e}"
                )
                continue

            score = float(eval_data.get("score", 0.0))

            cr = CandidateResult(
                round_idx=round_idx,
                candidate_idx=candidate_idx,
                solution_file_path=s_path,
                evaluation_file_path=e_path,
                score=score,
                reason=json.dumps(eval_data, ensure_ascii=False),
                source="disk",
            )
            results.append(cr)

        # If no pairs but LLM said something, return fallback single result with LLM info
        if not results:
            return [
                CandidateResult(
                    round_idx=round_idx,
                    candidate_idx=candidate_idx,
                    solution_file_path="",
                    evaluation_file_path="",
                    score=0.0,
                    reason=llm_out,
                    source="llm",
                )
            ]

        return results

    def _write_best_results(self, context: Context, best: CandidateResult):
        """Persist the best candidate paths to workspace."""
        Workspace.write_executor_best_solution(
            context, best.solution_file_path, BEST_SOLUTION_FILE
        )
        Workspace.write_executor_best_eval(
            context, best.evaluation_file_path, BEST_EVALUATION_FILE
        )

    def _make_result_message(
        self,
        parent_ctx: ExecutionContext,
        best: CandidateResult = None,
        total_prompt_tokens: int = 0,
        total_completion_tokens: int = 0,
    ) -> Message:
        """Assemble final output message with file existence checks."""

        def safe_path(path: Optional[str]) -> str:
            """Return path if file exists, otherwise empty string."""
            return path if path and os.path.exists(path) else ""

        return Message.from_text(
            data={
                "best_plan_file_path": parent_ctx.stage1_plan_file_path,
                "best_solution_file_path": safe_path(
                    best.solution_file_path if best else ""
                ),
                "best_evaluation_file_path": safe_path(
                    best.evaluation_file_path if best else ""
                ),
                "parent_info_file_path": parent_ctx.parent_info_file_path,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
            },
            sender="executor",
            role=Role.USER,
            mime_type=MimeType.APPLICATION_JSON,
        )

    def _parse_message_inputs(self, message: Message) -> ExecutionContext:
        """Extract and validate input file paths from the message."""
        elems = message.get_elements(ContentElement)
        if not elems:
            raise ValueError("Message missing ContentElement data.")

        data = elems[0].data
        plan_path = data.get("best_plan_file_path")
        parent_info_path = data.get("parent_info_file_path")

        if not parent_info_path or not os.path.exists(parent_info_path):
            raise FileNotFoundError(f"Missing parent_info.json: {parent_info_path}")

        stage1_plan = ""
        if plan_path:
            with open(plan_path, "r", encoding="utf-8") as f:
                stage1_plan = f.read()
        with open(parent_info_path, "r", encoding="utf-8") as f:
            parent_data = json.load(f)

        return ExecutionContext(
            parent_info_file_path=parent_info_path,
            parent_core=float(parent_data.get("score", 0.0)),
            parent_solution=json.dumps(parent_data, ensure_ascii=False),
            stage1_plan=stage1_plan,
            stage1_plan_file_path=plan_path,
        )
