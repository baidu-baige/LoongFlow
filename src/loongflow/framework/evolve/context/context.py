# -*- coding: utf-8 -*-
"""
This file define context for runtime
"""
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Context:
    """
    Context for all stage component
    """

    task: str
    base_path: str | Path = "./workspace"
    init_solution: str = ""
    init_evaluation: str = ""
    init_score: float = 0.0
    task_id: uuid.UUID = uuid.uuid4()
    island_id: int = 0
    current_iteration: int = 0
    total_iterations: int = 1000
    trace_id: str = str(uuid.uuid4())
    metadata: dict[str, Any] = field(default_factory=dict)

    def increment_iteration(self):
        """
        increment the current iteration
        """
        self.current_iteration += 1
