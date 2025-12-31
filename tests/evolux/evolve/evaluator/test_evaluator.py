# test_evaluator.py

import asyncio
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from agentsdk.message import Message, ContentElement
from evolux.evolve.context import EvaluatorConfig
from evolux.evolve.evaluator import LoongFlowEvaluator, EvaluationResult

CONFIGURABLE_EVALUATOR_CODE = """
import time
import re
import os

def evaluate(llm_file_path: str) -> dict:
    sleep_duration = 0
    try:
        with open(llm_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'# SLEEP: (\\d+\\.?\\d*)', content)
            if match:
                sleep_duration = float(match.group(1))
    except Exception as e:
        return {"error": f"Failed to read or parse llm_file: {e}"}

    print(f"Evaluator subprocess (pid: {os.getpid()}) will sleep for {sleep_duration} seconds.")
    time.sleep(sleep_duration)
    print(f"Evaluator subprocess (pid: {os.getpid()}) finished sleeping.")
    
    return {
        "score": 1.0,
        "metrics": {"status": "success", "execution_time": sleep_duration},
        "artifacts": {"log": "Evaluation completed successfully."}
    }
"""


class TestLoongFlowEvaluator(unittest.IsolatedAsyncioTestCase):
    """
    LoongFlowEvaluator Class Tests。
    """

    def setUp(self):
        """
        Buidling a temporary workspace for each test case.
        """
        self.workspace_path = tempfile.mkdtemp(prefix="evolux_test_")
        self.evaluator = None
        print(f"\n--- Starting test: {self._testMethodName} ---")
        print(f"Created temporary workspace: {self.workspace_path}")

    def tearDown(self):
        """
        Makes sure the evaluator is properly closed and temporary directory is cleaned up after each test.
        """
        if self.evaluator:
            # Calling interrupt explicitly to clean up resources is more reliable than relying on __del__
            self.evaluator.interrupt()
        shutil.rmtree(self.workspace_path)
        print(f"Cleaned up temporary workspace: {self.workspace_path}")
        print(f"--- Finished test: {self._testMethodName} ---")

    def _create_evaluator(self, timeout: float = 5.0) -> LoongFlowEvaluator:
        """Helpful function that creates an instance of LoongFlowEvaluator"""
        config = EvaluatorConfig(
            workspace_path=self.workspace_path,
            evaluate_code=CONFIGURABLE_EVALUATOR_CODE,
            timeout=timeout,
        )
        self.evaluator = LoongFlowEvaluator(config)
        return self.evaluator

    def _create_message(self, llm_code: str) -> Message:
        """Helpful function that creates a Message object containing the given LLM code."""
        element = ContentElement(data=llm_code)
        message = MagicMock(spec=Message)
        message.get_elements.return_value = [element]
        return message

    async def test_evaluate_success_single(self):
        """
        Tests a successful evaluation task that should complete within the timeout.
        """
        evaluator = self._create_evaluator(timeout=5)
        # Child process will sleep for 1 second, well within the 5 second timeout
        llm_code = "# SLEEP: 1\n# My awesome solution"
        message = self._create_message(llm_code)

        result = await evaluator.evaluate(message)
        print("Result:", result)

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics.get("status"), "success")
        self.assertNotIn("error", result.metrics)

    async def test_evaluate_timeout(self):
        """
        Tests a timeout scenario where the evaluation task takes too long to complete.
        """
        # Set a very short timeout: 1.5 seconds
        evaluator = self._create_evaluator(timeout=1.5)
        # Child process will sleep for 3 seconds
        llm_code = "# SLEEP: 3\n# This code will take too long"
        message = self._create_message(llm_code)

        start_time = time.perf_counter()
        result = await evaluator.evaluate(message)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Timeout test duration: {duration:.2f}s")

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metrics)
        self.assertIn("timed out", result.metrics["error"])

        # Verify that the execution time is approximately equal to the timeout, not the sleep time of the child process
        self.assertAlmostEqual(duration, 1.5, delta=0.5)

    async def test_evaluate_interrupt(self):
        """
        Test that calls interrupt() during the evaluation task execution.
        """
        evaluator = self._create_evaluator(timeout=20)
        # Child process will sleep for 10 seconds
        llm_code = "# SLEEP: 10\n# This code will be interrupted"
        message = self._create_message(llm_code)

        # Start the evaluation task in the background
        eval_task = asyncio.create_task(evaluator.evaluate(message))

        # Wait for the child process to start
        await asyncio.sleep(1)

        # Check if the process is in the active processes list
        self.assertEqual(len(evaluator._active_processes), 1)

        print("Calling interrupt()...")
        evaluator.interrupt()
        print("Interrupt() returned.")

        # Wait for the evaluation task to complete
        result = await eval_task
        print("Result:", result)

        # Verify that the evaluation task was interrupted
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metrics)
        error_msg = result.metrics["error"]
        self.assertTrue(
            "Evaluation process exited with non-zero" in error_msg
            or "interrupted" in error_msg
            or "Evaluator was interrupted." in error_msg
        )

        # Verify that the active processes list is empty
        self.assertEqual(len(evaluator._active_processes), 0)

    async def test_evaluate_concurrent(self):
        """
        Tests concurrent evaluation of multiple tasks.
        """
        num_concurrent_tasks = 4
        # Each task sleep 2 seconds
        sleep_per_task = 2
        # Total timeout set to 5 seconds, enough for concurrent tasks but not for serial completion
        timeout = 5

        evaluator = self._create_evaluator(timeout=timeout)
        llm_code = f"# SLEEP: {sleep_per_task}\n# Concurrent task"
        message = self._create_message(llm_code)

        tasks = [evaluator.evaluate(message) for _ in range(num_concurrent_tasks)]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(
            f"Concurrent evaluation of {num_concurrent_tasks} tasks took {duration:.2f}s"
        )

        # Check that all tasks were successful
        self.assertEqual(len(results), num_concurrent_tasks)
        for result in results:
            print(result)
            self.assertIsInstance(result, EvaluationResult)
            self.assertEqual(result.score, 1.0)
            self.assertNotIn("error", result.metrics)

        # Check that the total duration is not the sum of all task durations
        # Expected time should be slightly larger than the sum of all task times, plus some overhead
        total_serial_time = num_concurrent_tasks * sleep_per_task
        self.assertLess(duration, total_serial_time)
        self.assertGreater(duration, sleep_per_task)
        # Set a reasonable upper limit, e.g., single task time + 2 seconds overhead
        self.assertLess(duration, sleep_per_task + 2)


if __name__ == "__main__":
    unittest.main()
