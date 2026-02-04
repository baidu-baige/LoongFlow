# -*- coding: utf-8 -*-
"""
This file defines the claude code agent
"""

import asyncio
import os
from typing import Optional, List, Dict, Any, Callable

from loongflow.agentsdk.logger import get_logger
from loongflow.framework.base import AgentBase
from loongflow.agentsdk.message import Message, Role
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)

logger = get_logger(__name__)


def apply_llm_config(api_key: Optional[str], url: Optional[str]) -> None:
    """
    Apply LLMConfig to environment variables for claude_agent_sdk.

    Args:
        llm_config: LLMConfig from EvolveChainConfig
    """
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        logger.debug("Set ANTHROPIC_API_KEY from config")

    if url:
        os.environ["ANTHROPIC_BASE_URL"] = url
        logger.debug(f"Set ANTHROPIC_BASE_URL to: {url}")


class ClaudeCodeAgent(AgentBase):
    """
    Wraps the Claude Agent SDK as a generic Agent node in LoongFlow.

    This agent provides Claude with full terminal/file system permissions.
    Claude SDK internally handles Planning -> Execution (Coding/Bash) -> Solving.

    Supports both built-in tools and custom user-defined tools.

    Usage:
        # Basic usage with built-in tools
        agent = ClaudeCodeAgent(
            work_dir="./workspace",
            tool_list=["Read", "Edit", "Glob", "Bash", "Skill"]
        )
        result = await agent.run("Review and fix bugs in src/main.py")

        # With custom system prompt
        agent = ClaudeCodeAgent(
            work_dir="./workspace",
            tool_list=["Read", "Edit"],
            system_prompt="You are a security expert. Focus on finding vulnerabilities."
        )
        result = await agent.run("Audit the code in src/")

        # With custom tools
        async def my_custom_tool(args):
            return {"content": [{"type": "text", "text": f"Result: {args}"}]}

        agent = ClaudeCodeAgent(
            work_dir="./workspace",
            tool_list=["Read", "Edit"],
            custom_tools={
                "my_tool": {
                    "function": my_custom_tool,
                    "description": "My custom tool",
                    "parameters": {"input": str}
                }
            },
            system_prompt="You are a helpful coding assistant."
        )
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        work_dir: Optional[str] = None,
        tool_list: Optional[List[str]] = None,
        custom_tools: Optional[Dict[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        permission_mode: Optional[str] = None,
        setting_sources: Optional[List[str]] = None,
        max_turns: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
    ):
        """
        Initialize ClaudeCodeAgent.

        Args:
            work_dir: Working directory for the agent (default: "./")
            tool_list: List of allowed built-in tools (default: ["Read", "Edit", "Glob", "Bash", "Skill"])
                      Available built-in tools: Read, Edit, Glob, Bash, Skill, etc.
            custom_tools: Dictionary of custom tools to register
                         Format: {
                             "tool_name": {
                                 "function": callable,      # async function(args) -> dict
                                 "description": str,        # Tool description
                                 "parameters": dict         # {param_name: type}
                             }
                         }
                         Example:
                         {
                             "calculator": {
                                 "function": my_calc_func,
                                 "description": "Performs calculations",
                                 "parameters": {"expression": str}
                             }
                         }
            system_prompt: Custom system prompt to guide Claude's behavior (optional)
                          If not provided, Claude will use its default system prompt.
                          Example: "You are a code review expert. Focus on security and performance."
            permission_mode: Permission mode for file operations
                           - "prompt": Ask for confirmation (default)
                           - "acceptEdits": Auto-approve file edits
                           - "acceptAll": Auto-approve all operations
            verbose: Whether to enable verbose logging
        """
        super().__init__()

        # Set default working directory
        self.work_dir = work_dir or "./"

        # Set default tools if not provided
        self.tool_list = tool_list or [
            "Read",
            "Write",
            "Edit",
            "Grep",
            "Glob",
            "Bash",
            "Skill",
            "Task",
        ]

        # Store custom tools configuration
        self.custom_tools = custom_tools or {}

        # Store system prompt
        self.system_prompt = system_prompt

        self.model = model

        # Store permission and setting_sources
        self.permission_mode = permission_mode or "acceptEdits"
        self.setting_sources = setting_sources or ["project"]

        apply_llm_config(api_key, url)

        # Build allowed tools list
        allowed_tools = self.tool_list.copy()

        # Setup MCP servers for custom tools
        mcp_servers = {}
        if self.custom_tools:
            custom_tool_functions = self._register_custom_tools()
            if custom_tool_functions:
                # Create SDK MCP server with custom tools
                custom_server = create_sdk_mcp_server(
                    name="custom_tools", version="1.0.0", tools=custom_tool_functions
                )
                mcp_servers["custom"] = custom_server

                # Add custom tool names to allowed_tools
                for tool_name in self.custom_tools.keys():
                    allowed_tools.append(f"mcp__custom__{tool_name}")

        # Configure Claude Agent options - only include parameters with actual values
        options_kwargs = {
            "model": self.model,
            "allowed_tools": allowed_tools,
            "permission_mode": permission_mode,
        }

        # Only add optional parameters if they have values
        if self.work_dir and self.work_dir != "./":
            options_kwargs["cwd"] = self.work_dir

        if self.system_prompt:
            options_kwargs["system_prompt"] = self.system_prompt

        if setting_sources:
            options_kwargs["setting_sources"] = setting_sources

        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

        if max_turns:
            options_kwargs["max_turns"] = max_turns

        if max_thinking_tokens:
            options_kwargs["max_thinking_tokens"] = max_thinking_tokens

        self.options = ClaudeAgentOptions(**options_kwargs)

    def _validate_tool_name(self, name: str) -> None:
        """
        Validate tool name format.

        Args:
            name: Tool name to validate

        Raises:
            ValueError: If tool name is invalid
        """
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")

        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Tool name '{name}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        if name in self.tool_list:
            raise ValueError(
                f"Tool name '{name}' conflicts with built-in tool. "
                f"Please choose a different name."
            )

    def _validate_tool_function(self, func: Callable, name: str) -> None:
        """
        Validate that the tool function is async and properly formatted.

        Args:
            func: Function to validate
            name: Tool name for error messages

        Raises:
            TypeError: If function is not async or not callable
        """
        if not callable(func):
            raise TypeError(f"Tool '{name}' function must be callable")

        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"Tool '{name}' function must be async. "
                f"Please define it with 'async def'."
            )

    def _register_custom_tools(self) -> List[Any]:
        """
        Register custom tools as SDK MCP tools.

        Returns:
            List of SdkMcpTool instances ready for MCP server.
            Type annotation is List[Any] to avoid runtime import issues.
        """
        tool_functions = []

        for tool_name, tool_config in self.custom_tools.items():
            func = tool_config.get("function")
            description = tool_config.get("description", f"Custom tool: {tool_name}")
            parameters = tool_config.get("parameters", {})

            if not func:
                self.logger.warning(
                    f"Custom tool '{tool_name}' has no function, skipping"
                )
                continue

            try:
                # Validate tool function
                self._validate_tool_function(func, tool_name)

                # Wrap the function with @tool decorator
                decorated_func = tool(tool_name, description, parameters)(func)
                tool_functions.append(decorated_func)

                self.logger.info(f"Registered custom tool: {tool_name}")

            except (TypeError, ValueError) as e:
                self.logger.error(f"Failed to register tool '{tool_name}': {e}")
                # Remove invalid tool from custom_tools
                if tool_name in self.custom_tools:
                    del self.custom_tools[tool_name]

        return tool_functions

    def _update_options(self) -> None:
        """
        Update ClaudeAgentOptions with current custom tools configuration.
        This method consolidates the logic for updating options after tool changes.
        """
        allowed_tools = self.tool_list.copy()
        mcp_servers = None

        if self.custom_tools:
            custom_tool_functions = self._register_custom_tools()
            if custom_tool_functions:
                custom_server = create_sdk_mcp_server(
                    name="custom_tools", version="1.0.0", tools=custom_tool_functions
                )
                mcp_servers = {"custom": custom_server}

                # Add custom tool names to allowed_tools
                for tool_name in self.custom_tools.keys():
                    allowed_tools.append(f"mcp__custom__{tool_name}")

        options_kwargs = {
            "model": self.model,
            "allowed_tools": allowed_tools,
            "permission_mode": self.permission_mode,
        }

        # Only add optional parameters if they have values
        if self.work_dir and self.work_dir != "./":
            options_kwargs["cwd"] = self.work_dir

        if self.system_prompt:
            options_kwargs["system_prompt"] = self.system_prompt

        if self.setting_sources:
            options_kwargs["setting_sources"] = self.setting_sources

        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

        self.options = ClaudeAgentOptions(**options_kwargs)

    async def run(self, input_query: str, **kwargs) -> Message:
        """
        Execute the agent with the given query using ClaudeSDKClient.

        Claude SDK internally performs:
        - Planning: Analyze the task and decide on actions
        - Execution: Use tools (terminal, file system) to perform actions
        - Solving: Return the final result

        Args:
            input_query: The task/query for Claude to execute
            **kwargs: Additional context (currently unused)

        Returns:
            Message: A LoongFlow Message containing the agent's response
                     with token usage in metadata:
                     - input_tokens: Number of input tokens consumed
                     - output_tokens: Number of output tokens generated
                     - total_cost_usd: Total cost in USD (if available)
                     - duration_ms: Duration in milliseconds (if available)
        """
        full_response = []
        tool_calls = []
        final_status = "success"

        # Token usage tracking
        input_tokens = 0
        output_tokens = 0
        duration_ms = None

        # Consolidated initialization log with structured data
        self.logger.info(
            f"[Claude Agent] 🚀 Starting query, model: {self.model}, "
            f"work_dir: {self.work_dir}, "
            f"permission_mode: {self.permission_mode}, "
            f"query_length: {len(input_query)}, "
            f"query_preview: {input_query[:100] + '...' if len(input_query) > 100 else input_query}, "
            f"tool_count: {len(self.tool_list) + len(self.custom_tools)}"
        )

        try:
            # Use ClaudeSDKClient for better connection management
            self.logger.debug("Connecting to Claude SDK...")
            async with ClaudeSDKClient(options=self.options) as client:
                self.logger.debug("Connection established, sending query")
                # Send the query
                await client.query(input_query)
                self.logger.debug("Query sent, awaiting response")

                # Receive and process messages
                message_count = 0
                async for message in client.receive_messages():
                    message_count += 1
                    self.logger.debug(
                        f"[Claude Agent] Received message #{message_count}: {type(message).__name__}"
                    )

                    # Process assistant messages (Claude's reasoning and text output)
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if hasattr(block, "text"):
                                if block.text != "(no content)":
                                    full_response.append(block.text)
                                    self.logger.debug(f"[Claude]: {block.text[:200]}")
                            elif hasattr(block, "name"):
                                tool_info = f"Tool: {block.name}"
                                tool_calls.append(tool_info)
                                self.logger.info(
                                    f"[Claude Tool]: {block.name}, args={block.input}"
                                )

                    # Process result messages (final status and usage)
                    elif isinstance(message, ResultMessage):
                        final_status = message.subtype
                        self.logger.info(f"[Claude Done]: {message.subtype}")

                        # Extract token usage from ResultMessage
                        if hasattr(message, "usage") and message.usage:
                            input_tokens = message.usage.get("input_tokens", 0)
                            output_tokens = message.usage.get("output_tokens", 0)
                            self.logger.info(
                                f"[Claude Usage]: input_tokens={input_tokens}, output_tokens={output_tokens}"
                            )

                        if hasattr(message, "duration_ms"):
                            duration_ms = message.duration_ms

                        # ResultMessage indicates completion
                        break

        except ExceptionGroup as eg:
            # Handle TaskGroup exceptions (Python 3.11+)
            error_messages = []
            for exc in eg.exceptions:
                error_messages.append(f"{type(exc).__name__}: {str(exc)}")
                self.logger.error(
                    f"[Claude SubError]: {type(exc).__name__}: {str(exc)}"
                )
            self.logger.error(
                f"[Claude Error]: TaskGroup failed with {len(eg.exceptions)} sub-exceptions"
            )
            final_status = "error"
            full_response.append(f"Error occurred: {'; '.join(error_messages)}")
        except Exception as e:
            self.logger.error(f"[Claude Error]: {type(e).__name__}: {str(e)}")
            # Log more details for debugging
            import traceback

            self.logger.debug(f"[Claude Traceback]: {traceback.format_exc()}")
            final_status = "error"
            full_response.append(f"Error occurred: {str(e)}")

        # Combine response content
        content_text = "\n".join(full_response) if full_response else "Task completed"

        # Build metadata with token usage
        metadata = {
            "status": final_status,
            "tools_used": tool_calls,
            "work_dir": self.work_dir,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Add optional fields if available
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms

        # Return as LoongFlow Message
        return Message.from_text(
            data=content_text,
            sender=self.name or "ClaudeCodeAgent",
            role=Role.ASSISTANT,
            metadata=metadata,
        )

    async def interrupt_impl(self):
        """Handle agent interruption."""
        self.logger.warning(f"[{self.name}] Claude Code Agent interrupted")
        return Message.from_text(
            data="Agent execution was interrupted",
            sender=self.name or "ClaudeCodeAgent",
            role=Role.ASSISTANT,
            metadata={"status": "interrupted"},
        )

    def add_custom_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict[str, type]] = None,
    ) -> None:
        """
        Add a custom tool dynamically after initialization.

        Args:
            name: Tool name
            function: The tool function to execute (must be async)
            description: Tool description for Claude
            parameters: Parameter schema {param_name: type}

        Raises:
            ValueError: If tool name is invalid or conflicts with built-in tools
            TypeError: If function is not async or not callable

        Example:
            async def my_calculator(args):
                result = eval(args['expression'])
                return {
                    "content": [
                        {"type": "text", "text": f"Result: {result}"}
                    ]
                }

            agent.add_custom_tool(
                "calculator",
                my_calculator,
                "Evaluates mathematical expressions",
                {"expression": str}
            )
        """
        # Validate inputs
        self._validate_tool_name(name)
        self._validate_tool_function(function, name)

        if name in self.custom_tools:
            self.logger.warning(f"Tool '{name}' already exists, will be replaced")

        # Add tool
        self.custom_tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters or {},
        }

        # Update options with new tool configuration
        self._update_options()
        self.logger.info(f"Added custom tool: {name}")

    def remove_custom_tool(self, name: str) -> bool:
        """
        Remove a custom tool by name.

        Args:
            name: The tool name to remove

        Returns:
            bool: True if tool was removed, False if not found
        """
        if name not in self.custom_tools:
            self.logger.warning(f"Custom tool '{name}' not found")
            return False

        del self.custom_tools[name]
        self.logger.info(f"Removed custom tool: {name}")

        # Update options to reflect removal
        self._update_options()
        return True

    def list_custom_tools(self) -> List[str]:
        """
        List all registered custom tools.

        Returns:
            List of custom tool names
        """
        return list(self.custom_tools.keys())

    def get_custom_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific custom tool.

        Args:
            name: Tool name

        Returns:
            Tool configuration dict or None if not found
        """
        return self.custom_tools.get(name)
