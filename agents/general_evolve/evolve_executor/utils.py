#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common parsing utilities for executor modules.
"""

import re
from typing import Optional


EPSILON = 1e-9


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

    code_block_pattern = r"```" + re.escape(language) + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        # Check if the first line is a language identifier
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

    # Match the word after "No module named '"
    pattern = r"No module named '([^']*)'"

    # Use the regular expression to search for the word
    match = re.search(pattern, eval_response)

    # Extract the word and return it
    if match:
        extracted_word = match.group(1)
        return extracted_word
    else:
        return None
