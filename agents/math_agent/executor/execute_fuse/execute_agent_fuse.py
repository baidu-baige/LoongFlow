#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExecuteAgentChat orchestrates multi-round candidate generation, evaluation,
and best solution selection using ReActAgents and tool-based reasoning.
"""

import asyncio
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.memory.grade import GradeMemory, MemoryConfig
from loongflow.agentsdk.message import ContentElement, Message
from loongflow.agentsdk.message.elements import MimeType
from loongflow.agentsdk.message.message import Role
from loongflow.agentsdk.models import BaseLLMModel, CompletionRequest, LiteLLMModel
from loongflow.agentsdk.token import SimpleTokenCounter
from loongflow.agentsdk.tools import Toolkit
from agents.math_agent.executor.utils import (
    EPSILON,
    parse_full_rewrite,
    parse_missing_package,
)

from agents.math_agent.prompt.evolve_execute_prompt import (
    EVOLVE_EXECUTOR_REACT_USER_PROMPT,
    EVOLVE_EXECUTOR_CHAT_SYSTEM_PROMPT_WITH_PLAN,
    EVOLVE_EXECUTOR_CHAT_USER_PROMPT_WITH_PLAN,
    EVOLVE_EXECUTOR_CHAT_PACKAGE_INSTALL,
    EVOLVE_EXECUTOR_REACT_SYSTEM_PROMPT,
)

from agents.math_agent.executor.execute_react.execute_agent_observer import (
    ToolOutputException,
)

from agents.math_agent.executor.execute_react.build_tool import (
    build_evaluator_solution_tool,
    build_executor_read_tool,
    build_install_package_tool,
    build_executor_write_tool,
    build_executor_ls_tool,
)
from loongflow.framework.pes.compressor import EvolveCompressor
from loongflow.framework.pes.context import Context, LLMConfig, Workspace
from loongflow.framework.pes.evaluator import EvaluationResult
from loongflow.framework.pes.evaluator.evaluator import LoongFlowEvaluator
from loongflow.framework.pes.register import Worker
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.framework.react.components import (
    DefaultFinalizer,
    DefaultObserver,
    DefaultReasoner,
    SequenceActor,
)

logger = get_logger(__name__)


@dataclass
class ExecuteAgentFuseConfig:
    """Configuration for ExecuteAgentChat."""

    react_system_prompt: Optional[str] = None
    chat_system_prompt: Optional[str] = None
    llm_config: Optional[LLMConfig] = None
    max_rounds: int = 1
    react_max_steps: int = 2
    score_threshold: float = 0.9


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
class ExecutionContext:
    """Holds previous candidate and stage1 plan information."""

    parent_info_file_path: str
    parent_core: float
    parent_solution: str
    stage1_plan: str
    stage1_plan_file_path: str


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


class EvolveExecuteAgentFuse(Worker):
    """Agent for iterative candidate generation and evaluation."""

    def __init__(self, config: Any, evaluator: LoongFlowEvaluator):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ExecuteAgentFuseConfig)
            else ExecuteAgentFuseConfig(**config)
        )
        self.evaluator = evaluator
        self.model = self._init_model()
        logger.info(f"Executor: Agent Fuse successfully initialized")

    async def run(self, context: Context, message: Message) -> Message:
        """
        Perform multi-round candidate generation and evaluation until
         an improved solution is found or max rounds are reached.
        """
        parent_ctx = self._parse_message_inputs(message)
        history = HistoryRecord()

        # create a history.log file
        if parent_ctx.parent_core >= self.config.score_threshold:
            history_log_file_path = (
                f"{Workspace.get_executor_path(context)}/history.log"
            )
            with open(history_log_file_path, "w") as f:
                pass

        all_results: List[CandidateResult] = []
        init_parallel = 0
        previous_attempts = ""
        total_completion_tokens = 0
        total_prompt_tokens = 0

        # iterate rounds explicitly
        for round_idx in range(self.config.max_rounds):
            init_parallel += 1
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: [Round {round_idx}] "
                + f"Parallel Generating {init_parallel} candidate slots..."
            )

            # call gen_multi_candidate (handles concurrency)
            round_results_dict = await self.gen_multi_candidate(
                context, parent_ctx, round_idx, init_parallel, previous_attempts
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
                logger.warning(
                    f"Trace ID: {context.trace_id}: Executor Fuse: "
                    + f"[Round {round_idx}] No candidates found for any slot."
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
                    f"Trace ID: {context.trace_id}: Executor Fuse: ✅ Improved candidate found in "
                    + f"round {round_idx}: child_score = {best.score:.6f} > {parent_ctx.parent_core:.6f} = parent_score"
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
                logger.info(
                    f"Trace ID: {context.trace_id}: Executor Fuse: ❌ [Round {round_idx}] "
                    + f"No improved disk-sourced candidates this round."
                )

        if not all_results:
            logger.warning(
                f"Trace ID: {context.trace_id}: Executor Fuse: ⚠️ No candidates generated in any round."
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
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: Writing chosen best from history "
                + f"candidate (score={chosen.score}) to workspace"
            )
            self._write_best_results(context, chosen)

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
                    f"Trace ID: {context.trace_id}: Executor Fuse: Error parsing LLM output JSON: {llm_str}"
                )
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
                logger.exception(
                    f"Trace ID: {context.trace_id}: Executor: Error loading results for slot {idx}: {e}"
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
    ) -> str:
        """Generate a single candidate using ReActAgent."""
        candidate_path = Workspace.get_executor_candidate_path(
            context, f"{round_idx}_{candidate_idx}"
        )

        # If parent score above threshold, use ReAct mode for thorough generation
        if parent_ctx.parent_core >= self.config.score_threshold:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: ▶️ Generating candidate "
                + f"(round={round_idx}, idx={candidate_idx}) using React Mode"
            )

            history_log_file_path = (
                f"{Workspace.get_executor_path(context)}/history.log"
            )
            react_agent, rest_token = await self._create_react_agent(
                context, candidate_path
            )
            user_prompt = EVOLVE_EXECUTOR_REACT_USER_PROMPT.format(
                task=context.task,
                plan=parent_ctx.stage1_plan,
                parent_solution=parent_ctx.parent_solution,
                workspace=candidate_path,
                history_log_path=history_log_file_path,
            )

            initial_message = Message.from_text(user_prompt, role=Role.USER)
            token_counter = SimpleTokenCounter()
            user_token_count = await token_counter.count([initial_message])
            if user_token_count > rest_token:
                raise RuntimeError(
                    f"Trace ID: {context.trace_id}: Executor Fuse: Not enough tokens to complete this request."
                    + f"Please check your user prompt tokens and current token usage: {rest_token}/{user_token_count}"
                )

            try:
                result_msg = await react_agent.run(
                    initial_message,
                    trace_id=context.trace_id,
                )
                return self._parse_message_to_llm_output(result_msg)
            except ToolOutputException as e:
                # Special handling for tool-triggered exception
                error_msg = f"Tool output violated safety rule: {e}"
                logger.error(
                    f"Trace ID: {context.trace_id}, Executor Fuse: ToolOutputException: {error_msg}"
                )
                result = {"content": error_msg}
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as e:
                error_msg = f"react_agent.run failed: {e}"
                logger.exception(
                    f"Trace ID: {context.trace_id}: Executor Fuse: {error_msg}"
                )
                result = {"content": error_msg}
                return json.dumps(result, ensure_ascii=False, indent=2)

        else:  # Else use Chat Mode for faster generation
            total_completion_tokens = 0
            total_prompt_tokens = 0

            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: ▶️ Generating candidate "
                + f"(round={round_idx}, idx={candidate_idx}) using Chat Mode"
            )
            system_prompt = (
                self.config.chat_system_prompt
                or EVOLVE_EXECUTOR_CHAT_SYSTEM_PROMPT_WITH_PLAN
            )
            user_prompt = EVOLVE_EXECUTOR_CHAT_USER_PROMPT_WITH_PLAN.format(
                task=context.task,
                plan=parent_ctx.stage1_plan,
                parent_solution=parent_ctx.parent_solution,
                previous_attempts=previous_attempts,
            )
            if previous_attempts:
                logger.debug(
                    f"Trace ID: {context.trace_id}: Executor Fuse: ⚠️ Previous attempts: {previous_attempts}"
                )

            system_message = Message.from_text(
                sender="system", role=Role.SYSTEM, data=system_prompt
            )
            user_message = Message.from_text(
                sender="user", role=Role.USER, data=user_prompt
            )

            token_counter = SimpleTokenCounter()
            token_count = await token_counter.count([system_message, user_message])
            if token_count > self.config.llm_config.context_length:
                raise RuntimeError(
                    f"Trace ID: {context.trace_id}: Executor Fuse: Not enough tokens to complete this request."
                    + f"Please check your prompt tokens and current token usage: "
                    + f"{self.config.llm_config.context_length}/{token_count}"
                )

            llm_request = CompletionRequest(messages=[system_message] + [user_message])

            resp_generator = self.model.generate(llm_request)
            try:
                resp = await anext(resp_generator)
                if resp.error_code:
                    logger.exception(
                        f"Trace ID: {context.trace_id}: Executor Fuse: {resp.error_message}"
                    )
                    result = {"content": resp.error_message}
                    return json.dumps(result, ensure_ascii=False, indent=2)
            finally:
                async for _ in resp_generator:
                    pass

            code = ""
            for element in resp.content:
                if isinstance(element, ContentElement):
                    code = parse_full_rewrite(element.data, "python")

            if not code:
                logger.error(
                    f"Trace ID: {context.trace_id}: Executor Fuse: Empty code from LLM: {resp}"
                )
                result = {"content": resp}
                return json.dumps(result, ensure_ascii=False, indent=2)

            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: candidate (round={round_idx}, "
                + f"idx={candidate_idx}), Successfully generated solution in Chat Mode"
            )

            total_completion_tokens += resp.usage.completion_tokens
            total_prompt_tokens += resp.usage.prompt_tokens

            random_str = uuid.uuid4().hex[:3]
            Workspace.write_executor_file(
                context, f"{candidate_path}/solution_{random_str}.py", code
            )

            code_message = Message.from_text(
                sender="assistant", role=Role.ASSISTANT, data=code
            )
            evaluation_result = await self.evaluator.evaluate(code_message)
            if evaluation_result is None:
                logger.error(
                    f"Trace ID: {context.trace_id}: Executor Fuse: Failed to get evaluation result: {evaluation_result}"
                )
                result = {"content": evaluation_result}
                return json.dumps(result, ensure_ascii=False, indent=2)

            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: Candidate (round={round_idx}, idx={candidate_idx}), "
                + f"Chat Mode Get Evaluation Result: "
                + f"{json.dumps(json.loads(evaluation_result.to_json()), ensure_ascii=False)}",
            )
            missing_pacakge = parse_missing_package(evaluation_result.summary)
            if missing_pacakge:
                logger.info(
                    f"Trace ID: {context.trace_id}: Executor Fuse: Candidate (round={round_idx}, "
                    + f"idx={candidate_idx}), Missing package: {missing_pacakge}"
                )
                completion_tokens, prompt_tokens = await self.install_missing_package(
                    context, evaluation_result
                )
                total_completion_tokens += completion_tokens
                total_prompt_tokens += prompt_tokens

            Workspace.write_executor_file(
                context,
                f"{candidate_path}/evaluation_{random_str}.json",
                evaluation_result.to_json(),
            )

            result = {
                "content": json.dumps(evaluation_result.to_dict(), ensure_ascii=False, indent=2),
                "total_completion_tokens": total_completion_tokens,
                "total_prompt_tokens": total_prompt_tokens,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

    def _init_model(self) -> BaseLLMModel:
        """Initialize or reuse the LLM model."""
        llm = self.config.llm_config
        if not llm or not all([llm.model]):
            raise ValueError("model_name, url, and api_key are required in llm_config.")

        return LiteLLMModel.from_config(llm.model_dump())

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
            parent_solution=json.dumps(parent_data, ensure_ascii=False, indent=2),
            stage1_plan=stage1_plan,
            stage1_plan_file_path=plan_path,
        )

    def _parse_message_to_llm_output(self, result_msg: Message) -> str:
        """Extract the model output from Message as a plain string."""
        content = result_msg.get_elements(ContentElement)
        if content is not None and len(content) > 0:
            result = {
                "content": content[0].data,
                "total_prompt_tokens": result_msg.metadata.get(
                    "total_prompt_tokens", 0
                ),
                "total_completion_tokens": result_msg.metadata.get(
                    "total_completion_tokens", 0
                ),
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        result = {
            "content": f"parse llm ContentElement failed, "
            + f"result_msg:{json.dumps(result_msg.to_dict(), ensure_ascii=False, indent=2)}",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def load_results_for_candidate(
        self,
        context: Context,
        round_idx: int,
        candidate_idx: int,
        llm_out: str,
    ) -> List[CandidateResult]:
        """Read a candidate folder and return all paired solution/evaluation results.

        Pairing logic: find all files matching solution*.py and evaluation*.json and pair
        them by the suffix present in the filename. Example:
            solution1_23.py <-> evaluation1_23.json  (suffix: '1_23' or '23' depending on naming)

        Returns list of CandidateResult with source='disk'. If none found, returns single
        CandidateResult with source='llm' containing the LLM status/reason as record.
        """
        candidate_path = Workspace.get_executor_candidate_path(
            context, f"{round_idx}_{candidate_idx}"
        )
        p = Path(candidate_path)

        if not p.exists() or not p.is_dir():
            # no folder -> return llm-only fallback
            logger.warning(
                f"Trace ID: {context.trace_id}: Executor Fuse: Candidate folder not found: {candidate_path}"
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
        solution_candidates = list(p.glob("solution*.py"))
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
            key = extract_suffix(s, "solution")
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
                    f"Trace ID: {context.trace_id}: Executor Fuse: Failed to "
                    + f"read evaluation file {e_path}: {e}"
                )
                continue

            score = float(eval_data.get("score", 0.0))

            cr = CandidateResult(
                round_idx=round_idx,
                candidate_idx=candidate_idx,
                random_idx=str(k),
                solution_file_path=s_path,
                evaluation_file_path=e_path,
                score=score,
                reason=json.dumps(eval_data, ensure_ascii=False, indent=2),
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
        Workspace.write_executor_best_solution(context, best.solution_file_path)
        Workspace.write_executor_best_eval(context, best.evaluation_file_path)

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

    async def install_missing_package(
        self, context: Context, evaluation_result: EvaluationResult
    ):
        """Install missing package."""
        user_message = Message.from_text(
            sender="user",
            role=Role.USER,
            data=EVOLVE_EXECUTOR_CHAT_PACKAGE_INSTALL.format(
                error_msg=evaluation_result.summary, language="python"
            ),
        )

        llm_request = CompletionRequest(messages=[user_message])

        resp_generator = self.model.generate(llm_request)
        try:
            resp = await anext(resp_generator)
            if resp.error_code:
                raise Exception(
                    f"Error code: {resp.error_code}, error: {resp.error_message}"
                )
        finally:
            async for _ in resp_generator:
                pass

        cmd = ""
        for element in resp.content:
            if isinstance(element, ContentElement):
                cmd = parse_full_rewrite(element.data)

        total_completion_tokens = resp.usage.completion_tokens or 0
        total_prompt_tokens = resp.usage.prompt_tokens or 0

        if not cmd:
            logger.warning(
                f"Trace ID: {context.trace_id}: Executor Fuse: No package install command generated"
            )
            return total_completion_tokens, total_prompt_tokens

        try:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: Start Install missing package: {cmd}"
            )
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Fuse: Install package result: {result}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Trace ID: {context.trace_id}: Executor Fuse: Error install package: {e}"
            )

        return total_completion_tokens, total_prompt_tokens

    async def _create_react_agent(
        self, context: Context, candidate_path: str
    ) -> tuple[ReActAgent, int]:
        """Create and configure a ReActAgent for execution."""
        system_prompt = (
            self.config.react_system_prompt or EVOLVE_EXECUTOR_REACT_SYSTEM_PROMPT
        )
        system_message = [Message.from_text(system_prompt, role=Role.SYSTEM)]
        token_counter = SimpleTokenCounter()
        system_token_count = await token_counter.count(system_message)
        token_threshold = self.config.llm_config.context_length - system_token_count

        memory_config = MemoryConfig(
            auto_compress=True,
            token_threshold=token_threshold,
        )
        agent_memory = GradeMemory.create_default(
            model=self.model,
            compressor=EvolveCompressor(
                model=self.model,
                token_counter=token_counter,
                token_threshold=token_threshold,
            ),
            config=memory_config,
        )
        tool_kit = self._build_toolkit(context, candidate_path)
        agent_context = AgentContext(
            agent_memory, toolkit=tool_kit, max_steps=self.config.react_max_steps
        )

        hint_message = Message.from_text(
            sender="finalizer",
            role=Role.USER,
            data=(
                "Maximum execution attempts reached. "
                "Please provide a short textual summary of the current candidate results, "
                "focusing on their status and scores."
            ),
        )
        finalizer = DefaultFinalizer(
            self.model,
            summarize_prompt=system_prompt,
            output_schema=None,
            hint_message=hint_message,
        )

        return (
            ReActAgent(
                agent_context,
                DefaultReasoner(self.model, system_prompt),
                SequenceActor(),
                DefaultObserver(),
                finalizer,
                name="Executor",
            ),
            token_threshold,
        )

    def _build_toolkit(self, context: Context, candidate_path: str) -> Toolkit:
        """Register tools used by the ReActAgent."""
        toolkit = Toolkit()
        toolkit.register_tool(
            build_evaluator_solution_tool(self.evaluator, context, candidate_path)
        )
        toolkit.register_tool(build_executor_read_tool(context, candidate_path))
        toolkit.register_tool(build_install_package_tool())
        toolkit.register_tool(build_executor_write_tool(context, candidate_path))
        toolkit.register_tool(build_executor_ls_tool(context, candidate_path))
        return toolkit
