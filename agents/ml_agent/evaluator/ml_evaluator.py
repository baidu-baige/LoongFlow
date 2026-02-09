# -*- coding: utf-8 -*-
"""
This file provides the implementation for the final pipeline evaluator,
which is responsible for scoring the output of the MLExecutorAgent.
"""

import importlib.util
import json
import os
import sys
import time
import traceback

from loongflow.agentsdk.logger.logger import get_logger
from loongflow.framework.pes.context import EvaluatorConfig
from loongflow.framework.pes.evaluator import LoongFlowEvaluator


class MLEvaluator(LoongFlowEvaluator):
    """
    Abstract base class for evaluators that score the final output of a ML pipeline.

    It inherits from LoongFlowEvaluator to reuse its robust, process-isolated
    execution mechanism. It is specialized to handle the specific data bundle
    (task_data_path, best_code_path, artifacts) passed by the MLExecutorAgent.
    """

    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)

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

        logger = get_logger("MLEvaluator_Child")
        pid = os.getpid()
        logger.debug(
            f"[Child PID:{pid}] Started. Evaluator path: {evaluator_file_path}"
        )
        logger.debug(f"[Child PID:{pid}] Target LLM path: {llm_file_path}")

        result_data = {}

        try:
            # 1. Dynamically load the LLM code module to extract ML_AGENT_RESULT
            spec = importlib.util.spec_from_file_location(
                "llm_code_module", llm_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Module spec creation failed for {llm_file_path}")

            llm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_module)

            # Check for ML_AGENT_RESULT variable
            if not hasattr(llm_module, "ML_AGENT_RESULT"):
                raise AttributeError("Invalid llm_code, ML_AGENT_RESULT not found")

            ml_agent_result = llm_module.ML_AGENT_RESULT

            if not isinstance(ml_agent_result, dict):
                raise TypeError("ML_AGENT_RESULT must be a dict")

            logger.debug(f"[Child PID:{pid}] ML_AGENT_RESULT loaded successfully.")

            # 2. Dynamically load evaluator code module
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

            # 3. Call evaluate with the extracted parameters
            start_time = time.time()
            result = mod.evaluate(
                ml_agent_result.get("task_data_path"),
                ml_agent_result.get("best_code_path"),
                ml_agent_result.get("artifacts"),
            )
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

        logger.debug(
            f"Subprocess exit now. Result score: \
    {result_data.get('score', '')}. Result summary: {result_data.get('summary', '')}"
        )
        os._exit(0)
