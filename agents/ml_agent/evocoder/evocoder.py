# -*- coding: utf-8 -*-
"""
This file define EvoCoder
"""

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from agents.ml_agent.evocoder.evaluator import EvoCoderEvaluator
from agents.ml_agent.evocoder.stage_context_provider import (
    Stage,
    StageContextProvider,
    TaskConfig,
)
from agents.ml_agent.prompt.evocoder import PackageInstallerPrompts
from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.message import ContentElement, Message, MimeType, Role
from loongflow.agentsdk.models import CompletionRequest, LiteLLMModel
from loongflow.framework.base import AgentBase
from loongflow.framework.pes.context import LLMConfig
from loongflow.framework.pes.evaluator.evaluator import (
    EvaluationResult,
    EvaluationStatus,
)

logger = get_logger(__name__)


@dataclass
class EvoCoderConfig:
    """EvoCoder configuration"""

    llm_config: LLMConfig
    context_provider: StageContextProvider
    evaluator: EvoCoderEvaluator
    max_rounds: int = 10


class EvoCoder(AgentBase):
    """
    EvoCoder
    """

    def __init__(self, config: EvoCoderConfig):
        """
        Initializes the EvoCoder.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.model = LiteLLMModel.from_config(config.llm_config.model_dump())
        self.context_provider = self.config.context_provider
        self.evaluator = self.config.evaluator

    async def run(self, message: Message) -> Message:
        """
        Executes the generate-evaluate-retry loop.

        Args:
            message: The message from the Planner containing generation instructions.

        Returns:
            A message containing the final validated code on success, or an error
            report on failure.
        """
        # Step 1: Parse instructions from the message
        content = message.get_elements(ContentElement)
        if not content or len(content) == 0 or not isinstance(content[0].data, dict):
            raise ValueError(
                "Message missing ContentElement data or invalid content type, required dict."
            )
        data = content[0].data

        # Step 2: Start the retry loop
        # This is a simple loop to generate runnable code
        task_config = TaskConfig(**data)
        messages = self.context_provider.provide(task_config)
        stage = self.context_provider.stage()

        for attempt_num in range(self.config.max_rounds):
            logger.info(
                f"EvoCoder: Starting attempt for {stage} at {attempt_num + 1}/{self.config.max_rounds}."
            )

            # generate code
            code_message = await self._generate_code(stage, messages)
            if not code_message:
                continue
            messages.append(code_message)

            # extract code
            try:
                code = self._extract_code(code_message)
            except Exception as e:
                logger.error(f"Failed to extract code: {e}")
                messages.append(
                    Message.from_text(sender="EvoCoder", role=Role.USER, data=str(e))
                )
                continue

            logger.info(f"EvoCoder: code for {stage}: {code}")

            # 3c: Evaluate the generated code
            all_codes = repr(
                {
                    **task_config.code_deps,
                    self.context_provider.stage().value: code,
                }
            )
            try:
                eval_result = await self.evaluator.evaluate(
                    Message.from_text(data=f"EVOCODER_FILES={all_codes}")
                )
            except Exception as e:
                logger.error(
                    f"EvoCoder: evaluation failed on attempt {attempt_num + 1}: {e}"
                )
                messages.append(
                    Message.from_text(
                        sender="EvoCoder",
                        role=Role.USER,
                        data=f"evaluation failed: {str(e)}",
                    )
                )
                continue

            # 3d: Check the result
            if eval_result.status == EvaluationStatus.SUCCESS:
                logger.info(
                    f"Code validation for {stage} evaluated successfully on attempt {attempt_num + 1}."
                )
                return Message.from_elements(
                    sender="EvoCoder",
                    role=Role.USER,
                    elements=[
                        ContentElement(
                            mime_type=MimeType.APPLICATION_JSON,
                            data={"best_code": code},
                            metadata={"artifacts": eval_result.artifacts},
                        )
                    ],
                )

            logger.warning(
                f"Code validation for {stage} evaluated failed on attempt {attempt_num + 1},"
                f"evaluation result: {eval_result}"
            )
            msg = await self.solve_evaluation_fail(stage, eval_result)
            messages.append(msg)

        # Step 4: If the loop finishes without success, return a failure message
        logger.error(
            f"EvoCoder failed for {stage} after {self.config.max_rounds} attempts."
        )
        return Message.from_media(
            sender="EvoCoder",
            role=Role.USER,
            mime_type=MimeType.APPLICATION_JSON,
            data={},
        )

    async def interrupt_impl(self):
        """
        interrupt to stop evaluator
        """
        logger.info("EvoCoder interrupting... Passing interrupt to evaluator.")
        if self.evaluator:
            self.evaluator.interrupt()

    async def _generate_code(
        self, stage: Stage, messages: list[Message]
    ) -> Message | None:
        """Generate code from LLM and extract it."""
        llm_request = CompletionRequest(messages=messages)
        resp_generator = self.model.generate(llm_request)

        try:
            resp = await anext(resp_generator)
            if resp.error_code:
                logger.warning(
                    f"EvoCoder generate code for {stage} failed, "
                    f"error code: {resp.error_code}, error: {resp.error_message}"
                )
                return None
        finally:
            async for _ in resp_generator:
                pass
        # Append assistant response
        return Message.from_elements(
            sender="EvoCoder",
            role=Role.ASSISTANT,
            elements=list(resp.content),
            metadata={"usage": resp.usage},
        )

    def _extract_code(self, code_message: Message) -> Message | str:
        code_content = code_message.get_elements(ContentElement)
        if (
                not code_content
                or len(code_content) == 0
                or not isinstance(code_content[0].data, str)
        ):
            raise ValueError(
                "Python code not found or incomplete code block detected. Please provide complete code with matching delimiters"
            )

        raw_code = code_content[0].data

        code = parse_full_rewrite(raw_code)

        # if no code found, we need to try to generate python code
        if not code:
            raise ValueError(
                f"Python code not found or incomplete code block detected. Please provide complete code with matching delimiters"
            )

        return code

    async def solve_evaluation_fail(
        self, stage: Stage, eval_result: EvaluationResult
    ) -> Message:
        """
        solve evaluation failed message
        """
        error_details = eval_result.to_dict()

        missing_package = parse_missing_package(eval_result.summary)
        if not missing_package:
            return Message.from_text(
                sender="EvoCoder",
                role=Role.USER,
                data=f"code run failed: {json.dumps(error_details, ensure_ascii=False)}",
            )

        logger.info(
            f"Code evaluate for {stage} detect Missing package: {missing_package}, try to install it"
        )
        result = await self.install_missing_package(eval_result)
        return Message.from_text(
            sender="EvoCoder",
            role=Role.USER,
            data=f"trying to install missing package: {missing_package}, install result: {result}",
        )

    async def install_missing_package(self, evaluation_result: EvaluationResult):
        """Install missing package."""
        user_message = Message.from_text(
            sender="user",
            role=Role.USER,
            data=PackageInstallerPrompts.USER.format(
                error_msg=evaluation_result.summary,
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
                break

        if not cmd:
            logger.warning(f"EvoCoder: No command generated")
            return "No command generated, package not installed"
        try:
            logger.info(f"EvoCoder: Installing missing package: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                text=True,
                timeout=600,  # set a default timeout
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"EvoCoder: Installed result: {result}")
            return result
        except Exception as e:
            logger.error(f"EvoCoder: Error installing package: {e}")
            return str(e)


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    if not llm_response:
        return None

    xml_pattern = r"<(\w+)_code>(.*?)</\1_code>"
    matches = re.findall(xml_pattern, llm_response, re.DOTALL)
    if matches:
        return matches[0][1].strip()

    code_pattern = r"<code>(.*?)</code>"
    matches = re.findall(code_pattern, llm_response, re.DOTALL)
    if matches:
        return matches[0].strip()

    code_block_pattern = r"```" + re.escape(language) + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        content = matches[0].strip()
        if "\n" in content:
            first_line, rest_of_content = content.split("\n", 1)
            if first_line.strip().lower() in [
                "python",
                "javascript",
                "java",
                "c++",
                "go",
                "sql",
                "typescript",
                "html",
                "css",
                "bash",
                "shell",
            ]:
                return rest_of_content.strip()
        return content

    # Fallback to plain text
    return llm_response.strip()


def parse_missing_package(eval_response: str) -> Optional[str]:
    """
    Extract missing package from evaluation response

    Args:
        eval_response: Response from the evaluator tool

    Returns:
        Missing package or None if not found
    """
    if not eval_response:
        return None

    pattern = r"No module named '([^']*)'"

    match = re.search(pattern, eval_response)

    if match:
        extracted_word = match.group(1)
        return extracted_word
    else:
        return None
