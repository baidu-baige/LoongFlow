# -*- coding: utf-8 -*-
"""
Unit tests for AgentBase
"""
import asyncio
from types import SimpleNamespace

import pytest

from evolux.base.agent_base import AgentBase


class DummyAgent(AgentBase):
    """A simple agent implementation for testing."""

    def __init__(self):
        super().__init__()
        self.run_called = False
        self.interrupt_called = False
        self.hook_order = []

    async def run(self, *args, **kwargs):
        self.run_called = True
        await asyncio.sleep(0.01)
        return {"result": kwargs.get("x", 42)}

    async def interrupt_impl(self):
        self.interrupt_called = True
        await super().interrupt()

@pytest.mark.asyncio
async def test_agent_run_and_interrupt():
    """Agent runs normally and can be interrupted."""
    agent = DummyAgent()

    # run() should execute successfully
    result = await agent(x=10)
    assert result == {"result": 10}
    assert agent.run_called
    assert not agent._interrupted

    # interrupt() should mark agent as interrupted
    await agent.interrupt()
    assert agent._interrupted
    assert agent.interrupt_called


@pytest.mark.asyncio
async def test_safe_run_handles_exception(monkeypatch):
    """_safe_run should capture exceptions and call handle_error."""
    agent = DummyAgent()

    async def faulty_run(*args, **kwargs):
        raise RuntimeError("boom")

    async def fake_handle_error(error):
        return {"error": str(error)}

    monkeypatch.setattr(agent, "run", faulty_run)
    monkeypatch.setattr(agent, "handle_error", fake_handle_error)

    result = await agent._safe_run()
    assert result == {"error": "boom"}


@pytest.mark.asyncio
async def test_hook_execution_order():
    """Verify pre/post hooks execution order."""
    agent = DummyAgent()

    async def pre_hook(agent_instance, *args, **kwargs):
        agent_instance.hook_order.append("pre")

    async def post_hook(agent_instance, *args, result=None):
        agent_instance.hook_order.append("post")

    agent.register_hook("pre_run", pre_hook)
    agent.register_hook("post_run", post_hook)

    result = await agent()
    assert result["result"] == 42
    assert agent.hook_order == ["pre", "post"]


def test_register_and_remove_hooks():
    """Register and remove hooks correctly."""
    agent = DummyAgent()

    async def dummy_hook(agent_instance, *args, **kwargs):
        pass

    agent.register_hook("pre_run", dummy_hook)
    assert dummy_hook in agent._instance_hooks["pre_run"]

    agent.remove_hook("pre_run", dummy_hook)
    assert dummy_hook not in agent._instance_hooks["pre_run"]

    with pytest.raises(ValueError):
        agent.remove_hook("pre_run", dummy_hook)

    with pytest.raises(ValueError):
        agent.register_hook("invalid_hook", dummy_hook)


def test_wrap_supported_hooks_invalid_name(monkeypatch):
    """_wrap_supported_hooks should raise for invalid hook names."""
    class BadAgent(DummyAgent):
        supported_hook_types = ["invalidhook"]

    with pytest.raises(ValueError):
        BadAgent()


@pytest.mark.asyncio
async def test_interrupt_without_task():
    """Interrupt should work even when no task is running."""
    agent = DummyAgent()
    assert not agent._interrupted
    await agent.interrupt()
    assert agent._interrupted


@pytest.mark.asyncio
async def test_is_running_property():
    """Check is_running and interrupted properties."""
    agent = DummyAgent()

    assert not agent.is_running
    task = asyncio.create_task(agent._safe_run())
    agent._task = task
    assert agent.is_running

    await agent.interrupt()
    assert agent.interrupted


@pytest.mark.asyncio
async def test_handle_error_logs(monkeypatch):
    """Verify handle_error returns expected structure."""
    agent = DummyAgent()

    called = SimpleNamespace(msg=None)

    def fake_error(msg):
        called.msg = msg

    monkeypatch.setattr(agent.logger, "error", fake_error)

    result = await agent.handle_error(RuntimeError("oops"))
    assert "oops" in called.msg
    assert result["error"] == "oops"
