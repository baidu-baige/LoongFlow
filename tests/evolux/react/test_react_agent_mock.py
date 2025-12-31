# -*- coding: utf-8 -*-
"""
Test cases for ReActAgent
"""
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from agentsdk.message import ContentElement, Message, Role, ToolCallElement
from agentsdk.models import CompletionResponse
from evolux.react import ReActAgent


class TestResponse(BaseModel):
    """Test response model"""
    answer: str = Field(description="The answer")


class TestReActAgent:
    """Test cases for ReActAgent"""

    @pytest.fixture
    def mock_model(self):
        """Create ReActAgent instance"""
        model = AsyncMock()
        model.generate = AsyncMock()
        return model

    @pytest.fixture
    def react_agent(self, mock_model):
        """Create ReActAgent instance"""
        agent = ReActAgent.create_default(mock_model, "You are a helpful assistant")
        return agent

    @pytest.mark.asyncio
    async def test_run_successful_flow(self, react_agent, mock_model):
        """Test successful ReAct flow with final answer"""
        init_message = Message.from_text(
                sender="agent",
                data="What is the capital of China?",
                role=Role.USER,
        )
        mock_reasoning_response = CompletionResponse(
                id="test",
                content=[
                    ContentElement(data="I've found the answer. It is Beijing."),
                    ToolCallElement(
                            target="generate_final_answer",
                            arguments={"response": "The capital of China is Beijing."}
                    )
                ]
        )

        async def async_generator_mock():
            """
            mock generator
            """
            yield mock_reasoning_response

        mock_model.generate.return_value = async_generator_mock()

        result_message = await react_agent.run(init_message)

        # The agent should have added the initial message, the reasoning, the tool output, and the final response to context
        assert len(await react_agent.context.memory.get_memory()) == 4

        # The final result should be a message containing the ToolOutputElement from the finalizer
        assert result_message.role == "tool"
        final_output_elements = result_message.get_elements(ContentElement)
        assert len(final_output_elements) == 1
        assert final_output_elements[0].data.get("response") == "The capital of China is Beijing."

    @pytest.mark.asyncio
    async def test_run_exceeds_max_steps(self, react_agent, mock_model):
        """Test ReAct flow that exceeds max steps"""

        max_steps = 2
        agent = ReActAgent.create_default(
                model=mock_model,
                sys_prompt="You are a helpful assistant.",
                max_steps=max_steps
        )

        init_message = Message.from_text(
                sender="agent",
                data="Solve this complex problem.",
                role=Role.USER,
        )
        mock_reasoning_response = CompletionResponse(
                id="test",
                content=[
                    ContentElement(data="I've found the answer. It is Beijing."),
                    ToolCallElement(
                            target="generate_final_answer",
                            arguments={"response": "The capital of France is Beijing."}
                    )
                ]
        )
        mock_summarize_response = CompletionResponse(
                id="test",
                content=[
                    ContentElement(data="I could not solve the problem within the step limit."),
                ]
        )

        def create_async_generator(response_to_yield):
            """A factory that creates and returns a new async generator object."""

            async def _generator():
                yield response_to_yield

            return _generator()

        side_effect_generators = [
            create_async_generator(mock_reasoning_response) for _ in range(max_steps)
        ]
        side_effect_generators.append(create_async_generator(mock_summarize_response))
        mock_model.generate.side_effect = side_effect_generators

        result_message = await agent.run(init_message)

        final_text_elements = result_message.get_elements(ContentElement)
        assert len(final_text_elements) == 1
