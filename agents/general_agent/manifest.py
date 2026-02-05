#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manifest management for Solution Pack protocol.

This module provides:
1. SolutionManifest: Pydantic model for index.json
2. ManifestGenerator: Lightweight LLM-based manifest generator
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator

from loongflow.agentsdk.logger import get_logger

logger = get_logger(__name__)


class FileEntry(BaseModel):
    """Represents a file in the solution manifest."""

    path: str = Field(description="Relative path from solution root")
    type: Literal["code", "test", "doc", "config", "asset"] = Field(
        description="File type classification"
    )
    description: str = Field(description="Brief description of file purpose")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is relative and normalized."""
        if os.path.isabs(v):
            raise ValueError(f"Path must be relative, got: {v}")
        return os.path.normpath(v)


class ManifestMetadata(BaseModel):
    """Metadata section of the manifest."""

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    generation: int = Field(default=0, description="PES iteration number")
    parent_solution: Optional[str] = Field(
        default=None, description="Absolute path to parent solution"
    )


class SolutionManifest(BaseModel):
    """Pydantic model for index.json manifest."""

    version: str = Field(default="1.0", description="Manifest protocol version")
    entrypoint: Optional[str] = Field(
        default=None, description="Primary execution entry point"
    )
    test_entrypoint: Optional[str] = Field(
        default=None, description="Primary test entry point"
    )
    description: str = Field(
        default="", description="High-level summary of the solution"
    )
    metadata: ManifestMetadata = Field(default_factory=ManifestMetadata)
    files: List[FileEntry] = Field(default_factory=list, description="File inventory")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(ensure_ascii=False, indent=2, exclude_none=True)

    @classmethod
    def from_json(cls, json_str: str) -> "SolutionManifest":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, manifest_path: str) -> "SolutionManifest":
        """Load manifest from file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def save(self, manifest_path: str) -> None:
        """Save manifest to file."""
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


class ManifestGenerator:
    """Lightweight LLM-based manifest generator."""

    SYSTEM_PROMPT = """You are a JSON generator for code manifests.

CRITICAL RULES:
1. Your ENTIRE response must be a single valid JSON object
2. The output must be blocked between ```json and ```
3. NO explanations, NO extra text before or after the JSON
4. The JSON structure MUST exactly match this schema:

```json
{
  "version": "1.0",
  "entrypoint": "path/to/main.py",
  "test_entrypoint": "path/to/test.py",
  "description": "1-2 sentence summary",
  "files": [
    {
      "path": "relative/path/to/file.py",
      "type": "code|test|doc|config|asset",
      "description": "Brief 1-sentence description"
    }
  ]
}
```

Type classification rules:
- code: .py, .js, .ts, .java, .cpp, .go, etc.
- test: files in test/ or tests/ directories, or files with test_ prefix
- doc: .md, .txt, .rst, README files
- config: .json, .yaml, .toml, .ini, .env files
- asset: images, data files, models, etc.

Entrypoint heuristics:
- Look for main.py, __main__.py, run.py, app.py, index.py
- test_entrypoint: look for test_main.py, run_tests.py, or main test file
"""

    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize ManifestGenerator.

        Args:
            llm_config: LLM configuration with keys: model, api_key, url
        """
        self.llm_config = llm_config

    async def generate(self, solution_dir: str) -> str:
        """
        Generate manifest by analyzing directory structure.

        Args:
            solution_dir: Absolute path to solution directory

        Returns:
            SolutionManifest instance
        """
        # Scan directory structure
        file_tree = self._build_file_tree(solution_dir)

        # Build prompt
        user_prompt = f"""Analyze this solution directory structure and generate index.json:

```
{file_tree}
```

The manifest is: """

        # Call LLM
        try:
            raw_output = await self._call_llm(user_prompt)

            # Try to extract JSON from the output
            manifest_json = self._extract_json(raw_output)

            logger.info(f"ManifestGenerator: Generated manifest")
            return manifest_json
        except Exception as e:
            logger.error(f"ManifestGenerator: Failed to parse LLM output - {e}")
            # Log the exact problematic JSON content for debugging
            debug_info = {}
            if "raw_output" in locals():
                debug_info["raw_output_first_500"] = raw_output[:500]
            if "manifest_json" in locals():
                debug_info["json_first_500"] = manifest_json[:500]

            logger.debug(f"ManifestGenerator: Debug info: {debug_info}")
            # Fallback: create minimal manifest
            return self._create_fallback_manifest(solution_dir)

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM output, handling markdown code blocks and other noise.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned JSON string
        """
        import re

        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Remove control characters (ASCII 0-31) except for newlines (\n, \r), tabs (\t)
        # This fixes errors with control characters like \u0000-\u001F
        # Note: We need to handle both actual control chars and their escape sequences
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Also remove JSON-invalid escape sequences that might appear in strings
        # Replace problematic escape sequences with safe alternatives
        text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)  # Remove \xXX sequences
        text = re.sub(r'\\([^"\\/bfnrtu])', r"\1", text)  # Remove invalid escapes

        # Try to find JSON object pattern
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, return original (will fail validation and trigger fallback)
        return text.strip()

    def _build_file_tree(
        self, solution_dir: str, max_depth: int = 5, max_files: int = 100
    ) -> str:
        """Build a tree visualization of the directory."""
        lines = []
        solution_path = Path(solution_dir)

        file_count = 0
        for root, dirs, files in os.walk(solution_dir):
            # Skip hidden directories and common excludes
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["__pycache__", "node_modules", ".git"]
            ]

            level = len(Path(root).relative_to(solution_path).parts)
            if level > max_depth:
                continue

            indent = "  " * level
            rel_root = Path(root).relative_to(solution_path)
            if rel_root != Path("."):
                lines.append(f"{indent}{rel_root}/")

            for file in sorted(files):
                if file_count >= max_files:
                    lines.append(f"{indent}  ... (truncated)")
                    return "\n".join(lines)

                lines.append(f"{indent}  {file}")
                file_count += 1

        return "\n".join(lines) if lines else "(empty directory)"

    async def _call_llm(self, user_prompt: str) -> str:
        """Call LLM to generate manifest."""
        from loongflow.framework.claude_code.claude_code_agent import ClaudeCodeAgent

        # Create a temporary agent for manifest generation
        agent = ClaudeCodeAgent(
            model=self.llm_config.get("model", "gpt-4o-mini"),
            api_key=self.llm_config["api_key"],
            url=self.llm_config.get("url"),
            work_dir="/tmp",  # Dummy work dir
            tool_list=[],  # No tools needed
            disallowed_tools=["Read"],
            custom_tools={},
            system_prompt=self.SYSTEM_PROMPT,
            max_turns=10,  # Single turn
        )

        result = await agent.run(user_prompt)

        # Extract content from message
        if result.content and len(result.content) > 0:
            from loongflow.agentsdk.message import ContentElement

            if isinstance(result.content[0], ContentElement):
                return result.content[0].data

        raise ValueError("ManifestGenerator: LLM returned no content")

    def _create_fallback_manifest(self, solution_dir: str) -> str:
        """Create a minimal fallback manifest when LLM fails."""
        logger.warning("ManifestGenerator: Creating fallback manifest")

        files = []
        entrypoint = None
        test_entrypoint = None

        for root, dirs, filenames in os.walk(solution_dir):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                file_path = Path(root) / filename
                rel_path = str(file_path.relative_to(solution_dir))

                # Infer type from extension and path
                file_type = self._infer_file_type(rel_path, filename)

                files.append(
                    FileEntry(
                        path=rel_path,
                        type=file_type,
                        description=f"{file_type.capitalize()} file",
                    )
                )

                # Detect entrypoints
                if not entrypoint and filename in [
                    "main.py",
                    "__main__.py",
                    "run.py",
                    "app.py",
                ]:
                    entrypoint = rel_path
                if not test_entrypoint and filename in [
                    "test_main.py",
                    "run_tests.py",
                ]:
                    test_entrypoint = rel_path

        manifest = SolutionManifest(
            description="Auto-generated manifest (fallback mode)",
            entrypoint=entrypoint,
            test_entrypoint=test_entrypoint,
            files=files,
        )

        return manifest.to_json()

    def _infer_file_type(self, rel_path: str, filename: str) -> str:
        """Infer file type from path and extension."""
        lower_path = rel_path.lower()
        lower_name = filename.lower()

        # Test files
        if "test" in lower_path or lower_name.startswith("test_"):
            return "test"

        # Documentation
        if any(
            lower_name.endswith(ext)
            for ext in [".md", ".txt", ".rst", ".adoc", "readme"]
        ):
            return "doc"

        # Config files
        if any(
            lower_name.endswith(ext)
            for ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".cfg"]
        ):
            return "config"

        # Code files
        code_extensions = [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".rb",
            ".php",
        ]
        if any(lower_name.endswith(ext) for ext in code_extensions):
            return "code"

        # Default to asset
        return "asset"
