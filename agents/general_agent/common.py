from dataclasses import dataclass

from typing import Optional, List

from loongflow.framework.pes.context import LLMConfig


@dataclass
class ClaudeAgentConfig:
    """Claude Agent configuration"""

    llm_config: LLMConfig
    system_prompt: Optional[str] = None
    build_in_tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    max_turns: Optional[int] = None
    max_thinking_tokens: Optional[int] = None
    permission_mode: Optional[str] = "acceptEdits"
