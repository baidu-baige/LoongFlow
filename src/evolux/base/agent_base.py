
# -*- coding: utf-8 -*-
"""
This file defines the base class for all agents
"""
import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Type, Callable, Dict, List

from pydantic import BaseModel

from agentsdk.logger.context import get_log_id, new_log_id, set_log_id
from agentsdk.logger.logger import get_logger
from agentsdk.message.message import Message


class AgentBase(ABC):
    """
    Unified asynchronous Agent base class with:
    - Cancellable lifecycle
    - Hook registration system
    - Structured logging
    - Optional input schema definition

    Subclasses should implement:
        - `run`: main execution logic
        - `interrupt_impl`: custom interruption handling (optional)
    """

    name: str = "base_agent"
    description: Optional[str] = None
    input_schema: Optional[Type[BaseModel]] = None
    supported_hook_types: List[str] = ["pre_run", "post_run"]

    def __init__(self):
        # Runtime state
        self._task: Optional[asyncio.Task] = None
        self._interrupted: bool = False
        self._instance_hooks: Dict[str, List[Callable]] = {
            t: [] for t in self.supported_hook_types
        }

        # Initialize trace and logger
        if not get_log_id():
            set_log_id(new_log_id())
        self.logger = get_logger(self.__class__.__name__)

        # Default schema if not provided
        if self.input_schema is None:
            class DefaultInput(BaseModel):
                """Default input schema."""
                request: str
            self.input_schema = DefaultInput

        # Automatically wrap supported hooks
        self._wrap_supported_hooks()

    async def __call__(self, *args, **kwargs) -> Message:
        """
        Agent entrypoint.
        - Creates async task for execution
        - Handles cancellation and cleanup
        """
        self._interrupted = False
        self._task = asyncio.create_task(self._safe_run(*args, **kwargs))

        try:
            return await self._task
        except asyncio.CancelledError:
            # External cancel or Ctrl+C
            return await self.interrupt_impl()
        finally:
            self._task = None

    async def _safe_run(self, *args, **kwargs) -> Message:
        """Run with unified error handling."""
        try:
            return await self.run(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return await self.handle_error(e)

    @abstractmethod
    async def run(self, *args, **kwargs) -> Message:
        """Main agent logic (must be implemented by subclasses)."""
        pass

    async def interrupt(self):
        """Trigger agent interruption."""
        if self._interrupted:
            return

        self._interrupted = True

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self.interrupt_impl()

    @abstractmethod
    async def interrupt_impl(self):
        """Custom interruption logic (optional for subclasses)."""
        pass

    def _wrap_supported_hooks(self):
        """
        Wrap all target functions defined in `supported_hook_types`
        with corresponding pre/post hooks.
        """
        wrapped_targets = set()

        for hook_type in self.supported_hook_types:
            # Validate hook name format
            if not hook_type.startswith(("pre_", "post_")):
                raise ValueError(
                    f"Invalid hook name '{hook_type}', must start with 'pre_' or 'post_'."
                )

            # Extract the target method name
            target_name = hook_type.split("_", 1)[1]
            if target_name in wrapped_targets:
                continue

            # Get the method from the instance
            func = getattr(self, target_name, None)
            if not func:
                continue

            # Wrap the method with pre/post hooks
            setattr(self, target_name, self._wrap_with_hooks(func, target_name))
            wrapped_targets.add(target_name)

    def _wrap_with_hooks(self, func: Callable[..., Any], name: str) -> Callable[..., Any]:
        """
        Wrap a specific method with pre/post hooks.
        Pre-hooks run before the method and can modify input arguments.
        Post-hooks run after the method and can modify the return value.
        """

        async def wrapper(*args, **kwargs):
            instance = self

            # Prepare mutable copies for hooks to modify
            hook_args = list(args)
            hook_kwargs = deepcopy(kwargs)

            # Execute pre-hooks
            for hook in self._instance_hooks.get(f"pre_{name}", []):
                modified = await _execute_async_or_sync_func(hook, instance, *hook_args, **hook_kwargs)
                if modified is not None:
                    assert isinstance(modified, dict), (
                        f"Pre-hook {hook.__name__} must return a dict with 'args' and 'kwargs'."
                    )
                    hook_args = modified.get("args", hook_args)
                    hook_kwargs = modified.get("kwargs", hook_kwargs)

            # Call the original function
            result = await func(*hook_args, **hook_kwargs)

            # Execute post-hooks
            for hook in self._instance_hooks.get(f"post_{name}", []):
                modified_result = await _execute_async_or_sync_func(
                    hook, instance, *hook_args, **hook_kwargs, result=result
                )
                if modified_result is not None:
                    result = modified_result

            return result

        return wrapper

    def register_hook(self, hook_type: str, hook_fn: Callable):
        """Register a hook function to a specific hook type."""
        if hook_type not in self._instance_hooks:
            raise ValueError(f"Unsupported hook type: {hook_type}")
        self._instance_hooks[hook_type].append(hook_fn)

    def remove_hook(self, hook_type: str, hook_fn: Callable):
        """Remove a registered hook function."""
        hooks = self._instance_hooks.get(hook_type)
        if hooks is None:
            raise ValueError(f"Unsupported hook type: {hook_type}")
        try:
            hooks.remove(hook_fn)
        except ValueError:
            raise ValueError(f"Hook not found: {hook_fn}")
        
    async def handle_error(self, error: Exception):
        """Default error handling (can be overridden)."""
        self.logger.error(f"[{self.name}] Error: {error}")
        return {"error": str(error)}

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._task is not None and not self._task.done()

    @property
    def interrupted(self) -> bool:
        """Whether the agent has been interrupted."""
        return self._interrupted

async def _execute_async_or_sync_func(func: Callable, *args, **kwargs):
    """
    Helper to execute a function which can be async or sync.
    Returns the function result.
    """
    result = func(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result
