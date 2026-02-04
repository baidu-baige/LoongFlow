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

from agents.math_agent.executor.utils import (
    EPSILON,
    parse_full_rewrite,
    parse_missing_package,
)
from agents.math_agent.prompt.evolve_execute_prompt import (
    EVOLVE_EXECUTOR_CHAT_PACKAGE_INSTALL,
    EVOLVE_EXECUTOR_CHAT_SYSTEM_PROMPT_WITH_PLAN,
    EVOLVE_EXECUTOR_CHAT_USER_PROMPT_WITH_PLAN,
)
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import ContentElement, Message
from loongflow.agentsdk.message.elements import MimeType
from loongflow.agentsdk.message.message import Role
from loongflow.agentsdk.models import BaseLLMModel, CompletionRequest, LiteLLMModel
from loongflow.framework.pes.context import Context, LLMConfig, Workspace
from loongflow.framework.pes.evaluator import EvaluationResult
from loongflow.framework.pes.evaluator.evaluator import LoongFlowEvaluator
from loongflow.framework.pes.register import Worker

logger = get_logger(__name__)


@dataclass
class ExecuteAgentChatConfig:
    """Configuration for ExecuteAgentChat."""

    system_prompt: Optional[str] = None
    llm_config: Optional[LLMConfig] = None
    max_rounds: int = 1


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


class EvolveExecuteAgentChat(Worker):
    """Agent for iterative candidate generation and evaluation."""

    def __init__(self, config: Any, evaluator: LoongFlowEvaluator):
        super().__init__()
        self.config = (
            config
            if isinstance(config, ExecuteAgentChatConfig)
            else ExecuteAgentChatConfig(**config)
        )
        self.evaluator = evaluator
        self.model = self._init_model()
        logger.info(f"Executor: Agent Chat successfully initialized")

    async def run(self, context: Context, message: Message) -> Message:
        """
        Perform multi-round candidate generation and evaluation until
         an improved solution is found or max rounds are reached.
        """
        parent_ctx = self._parse_message_inputs(message)
        history = HistoryRecord()
        all_results: List[CandidateResult] = []
        init_parallel = 0
        previous_evaluation = ""
        # iterate rounds explicitly
        for round_idx in range(self.config.max_rounds):
            init_parallel += 1
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Chat: [Round {round_idx}] "
                + f"Generating {init_parallel} candidate slots..."
            )

            # call gen_multi_candidate (handles concurrency)
            round_results = await self.gen_multi_candidate(
                context, parent_ctx, round_idx, init_parallel, previous_evaluation
            )

            # Write round-level history (append all candidate entries found this round)
            history.add_round(round_idx, [r.to_dict() for r in round_results])
            Workspace.write_executor_history(
                context, json.dumps(history.to_list(), ensure_ascii=False, indent=2)
            )

            if not round_results:
                logger.warning(
                    f"Trace ID: {context.trace_id}: Executor Chat: "
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
                    f"Trace ID: {context.trace_id}: Executor Chat: ✅ Improved candidate found in "
                    + f"round {round_idx}: {best.score:.6f} > {parent_ctx.parent_core:.6f}"
                )
                self._write_best_results(context, best)
                # Stop early on improvement
                break
            else:
                for result in disk_results:
                    previous_evaluation += (
                        f"Round {round_idx}, Candidate {result.candidate_idx}, "
                        + f"Evaluation Result: {result.reason}\n\n"
                    )
                logger.info(
                    f"Trace ID: {context.trace_id}: Executor Chat: [Round {round_idx}] "
                    + f"No improved disk-sourced candidates this round."
                )

        if not all_results:
            logger.warning(
                f"Trace ID: {context.trace_id}: Executor Chat: ⚠️ No candidates generated in any round."
            )
            return self._make_result_message(parent_ctx, None)

        # Ensure best exists on disk; if not, pick top disk-scoring; fallback to any
        disk_all = [r for r in all_results if r.source == "disk"]
        if disk_all:
            chosen = max(disk_all, key=lambda r: r.score)
        else:
            chosen = max(all_results, key=lambda r: r.score)

        best_solution_path = Workspace.get_executor_best_solution_path(context)
        if not Path(best_solution_path).exists() and chosen:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Chat: Writing chosen best "
                + f"candidate (score={chosen.score}) to workspace"
            )
            self._write_best_results(context, chosen)

        return self._make_result_message(parent_ctx, chosen)

    async def gen_multi_candidate(
        self,
        context: Context,
        parent_ctx: ExecutionContext,
        round_idx: int,
        parallel_candidates: int,
        previous_evaluation: str,
    ) -> List[CandidateResult]:
        """Generate multiple candidate slots concurrently.

        Concurrency strategy:
        - Launch gen_one_candidate for each parallel slot concurrently.
        - Gather LLM outputs, then for each slot call load_results_for_candidate which reads disk files
          and attaches LLM status/reason to each disk entry. If load finds no disk pairs, a single
          LLM-only CandidateResult (source='llm') is returned for that slot.
        - Flatten results and return as list[CandidateResult].
        """
        tasks = [
            self.gen_one_candidate(
                context, parent_ctx, round_idx, i, previous_evaluation
            )
            for i in range(parallel_candidates)
        ]
        llm_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: List[CandidateResult] = []

        for idx, llm_out in enumerate(llm_results):
            llm_str = str(llm_out) if isinstance(llm_out, Exception) else llm_out
            try:
                slot_results = self.load_results_for_candidate(
                    context, round_idx, idx, llm_str
                )
                final_results.extend(slot_results)
            except Exception as e:
                logger.exception(
                    f"Trace ID: {context.trace_id}: Executor Chat: "
                    + f"Error loading results for slot {idx}: {e}"
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
                final_results.append(fallback)

        return final_results

    async def gen_one_candidate(
        self,
        context: Context,
        parent_ctx: ExecutionContext,
        round_idx: int,
        candidate_idx: int,
        previous_evaluation: str,
    ) -> str:
        """Generate a single candidate using ReActAgent."""
        logger.info(
            f"Trace ID: {context.trace_id}: Executor Chat: ▶️ Generating candidate "
            + f"(round={round_idx}, idx={candidate_idx})"
        )
        candidate_path = Workspace.get_executor_candidate_path(
            context, f"{round_idx}_{candidate_idx}"
        )
        system_prompt = (
            self.config.system_prompt or EVOLVE_EXECUTOR_CHAT_SYSTEM_PROMPT_WITH_PLAN
        )
        user_prompt = EVOLVE_EXECUTOR_CHAT_USER_PROMPT_WITH_PLAN.format(
            task=context.task,
            plan=parent_ctx.stage1_plan,
            parent_solution=parent_ctx.parent_solution,
            previous_attempts=previous_evaluation,
        )
        if previous_evaluation:
            logger.debug(
                f"Trace ID: {context.trace_id}: Executor Chat"
                + f": ⚠️ Previous evaluation result: {previous_evaluation}"
            )

        system_message = Message.from_text(
            sender="system", role=Role.SYSTEM, data=system_prompt
        )
        user_message = Message.from_text(
            sender="user", role=Role.USER, data=user_prompt
        )

        llm_request = CompletionRequest(messages=[system_message] + [user_message])

        resp_generator = self.model.generate(llm_request)
        try:
            resp = await anext(resp_generator)
            if resp.error_code:
                raise Exception(
                    f"Error code: {resp.error_code}, error: {resp.error_message}"
                )
        finally:
            # Ensure that the generator is fully consumed to properly close the internal session
            async for _ in resp_generator:
                pass

        code = ""
        for element in resp.content:
            if isinstance(element, ContentElement):
                code = parse_full_rewrite(element.data, "python")

        if not code:
            raise Exception(f"Empty code from LLM: {resp}")

        logger.info(
            f"Trace ID: {context.trace_id}: Executor Chat: "
            f"candidate (round={round_idx}, idx={candidate_idx}), Successfully generated solution"
        )

        random_str = uuid.uuid4().hex[:3]
        Workspace.write_executor_file(
            context, f"{candidate_path}/solution_{random_str}.py", code
        )

        code_message = Message.from_text(
            sender="assistant", role=Role.ASSISTANT, data=code
        )
        evaluation_result = await self.evaluator.evaluate(code_message)
        if evaluation_result is None:
            raise Exception("Evaluation returned None")

        logger.info(
            f"Trace ID: {context.trace_id}: Executor Chat: candidate (round={round_idx}, idx={candidate_idx}), "
            + f"Evaluation result: {json.dumps(json.loads(evaluation_result.to_json()), ensure_ascii=False)}",
        )
        # Check for missing packages; invoke LLM to generate installation comman
        missing_pacakge = parse_missing_package(evaluation_result.summary)
        if missing_pacakge:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Chat: candidate (round={round_idx}, "
                + f"idx={candidate_idx}), Missing package: {missing_pacakge}"
            )
            await self.install_missing_package(context, evaluation_result)

        evaluation_result_json = json.dumps(evaluation_result.to_dict(), ensure_ascii=False, indent=2)
        Workspace.write_executor_file(
            context,
            f"{candidate_path}/evaluation_{random_str}.json",
            evaluation_result.to_json(),
        )
        return evaluation_result_json

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
            parent_solution=parent_data.get("solution", ""),
            stage1_plan=stage1_plan,
            stage1_plan_file_path=plan_path,
        )

    def _parse_message_to_llm_output(self, result_msg: Message) -> str:
        """Extract the model output from Message as a plain string."""
        content = result_msg.get_elements(ContentElement)
        if content is not None and len(content) > 0:
            return content[0].data
        return f"parse llm ContentElement failed, result_msg:{json.dumps(result_msg, ensure_ascii=False)}"

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
                f"Trace ID: {context.trace_id}: Executor Chat: Candidate folder not found: {candidate_path}"
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
                    f"Trace ID: {context.trace_id}: Executor Chat: "
                    + f"Failed to read evaluation file {e_path}: {e}"
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
                reason=llm_out,
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
        """Persist best candidate paths to workspace."""
        Workspace.write_executor_best_solution(context, best.solution_file_path)
        Workspace.write_executor_best_eval(context, best.evaluation_file_path)

    def _make_result_message(
        self,
        parent_ctx: ExecutionContext,
        best: CandidateResult = None,
    ) -> Message:
        """Assemble final output message with file existence checks."""
        logger.info(f"best: {best}")

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
            # Make sure to consume the generator to avoid leaking resources
            async for _ in resp_generator:
                pass

        cmd = ""
        for element in resp.content:
            if isinstance(element, ContentElement):
                cmd = parse_full_rewrite(element.data)

        if not cmd:
            logger.warning(
                f"Trace ID: {context.trace_id}: Executor Chat: No command generated"
            )
            return None

        try:
            logger.info(
                f"Trace ID: {context.trace_id}: Executor Chat: Installing missing package: {cmd}"
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
                f"Trace ID: {context.trace_id}: Executor Chat: Installed result: {result}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Trace ID: {context.trace_id}: Executor Chat: Error installing package: {e}"
            )
