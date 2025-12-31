# -*- coding: utf-8 -*-
"""
Unit tests for agentsdk.memory.grade.compressor.default_compressor module
"""

from unittest.mock import MagicMock

import pytest

from agentsdk.memory.grade.compressor import LLMCompressor
from agentsdk.models import LiteLLMModel
from tests.agentsdk.memory.grade.compressor.test_message_compression import generate_test_messages


class TestLLMCompressor:
    """Test cases for LLMCompressor class"""

    @pytest.mark.asyncio
    async def test_compress_empty_messages(self):
        """Test compress method with empty message list"""
        mock_model = MagicMock()
        compressor = LLMCompressor(mock_model)

        result = await compressor.compress([])

        # Should return empty list without calling the model
        assert result == []

    @pytest.mark.asyncio
    async def test_compress(self):
        """Test compress method with multiple messages"""
        message = generate_test_messages()
        model = LiteLLMModel(
                model_name="deepseek-v3",
                base_url="https://qianfan.baidubce.com/v2",
                api_key="xxx",
        )
        compressor = LLMCompressor(model)
        messages = await compressor.compress(message)
        print(messages)


if __name__ == "__main__":
    pytest.main([__file__])
