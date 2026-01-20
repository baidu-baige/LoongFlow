# Memory Components

LoongFlow's memory system provides hierarchical storage management for agents, including state management for evolutionary algorithms and conversation history management.

## Core Architecture

The memory system is divided into two main parts:

### Evolution Memory (`src/loongflow/agentsdk/memory/evolution/`)
- Manages the storage of solutions and population management in evolutionary algorithms
- Supports island-model parallel evolutionary algorithms
- Provides checkpoint functionality to support saving and restoring evolution progress

### Grade Memory (`src/loongflow/agentsdk/memory/grade/`)
- Based on three-level storage: STM (Short-Term Memory), MTM (Medium-Term Memory), and LTM (Long-Term Memory)
- Supports automatic compression of message history
- Manages agent conversation context and history

## Key Features

- **Evolution State Management**: Island allocation, population management, elite archiving
- **History Compression**: LLM-based intelligent message compression
- **Persistent Storage**: Supports in-memory and file storage backends
- **Automatic Management**: Token counting, automatic compression, memory cleanup

## Quick Start

```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory
from loongflow.agentsdk.memory.grade.memory import GradeMemory

# Evolution memory
evolution_memory = InMemory(num_islands=3, population_size=100)

# Grade memory
grade_memory = GradeMemory.create_default(model)
```

Continue reading the following documents for more details:
- [Evolution Memory](evolution.md) - Evolution state management
- [Grade Memory](grade.md) - Conversation history management
- [Configuration Guide](configuration.md) - Configuration options