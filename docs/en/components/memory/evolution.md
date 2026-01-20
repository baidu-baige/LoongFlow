# Evolution Memory

Evolution Memory manages solution states in evolutionary algorithms, supporting parallel evolution, population management, and checkpoint functions.

## Core Functions

### Infrastructure
```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory

memory = InMemory(
    num_islands=3,           # Number of islands
    population_size=100,     # Population size
    elite_archive_size=50,   # Elite archive size
    migration_interval=10    # Migration interval
)
```

### Solution Management
- **Solution Representation**: Uses the `Solution` data class to store code, scores, and parent relationships.
- **Island Allocation**: Solutions are assigned to different islands based on parents or round-robin.
- **Population Control**: Automatically maintains population size and eliminates low-scoring solutions.

### Evolutionary Algorithm Support
- **Elite Selection**: Maintains an elite archive to preserve optimal solutions.
- **Migration Mechanism**: Periodically migrates excellent solutions between islands.
- **Diversity Management**: Feature-based MAP-Elites algorithm.

## Usage Methods

### Add Solution
```python
solution = Solution(
    solution="def solve(): return 42",
    score=0.85,
    parent_id="parent_001"
)
solution_id = await memory.add_solution(solution)
```

### Retrieve Information
```python
# Get best solutions
best_solutions = memory.get_best_solutions(island_id=0, top_k=5)

# Sample solutions for crossover/mutation
parent = memory.sample(island_id=0, exploration_rate=0.1)

# View memory status
status = memory.memory_status()
```

### Checkpoint Management
```python
# Save checkpoint
await memory.save_checkpoint("./checkpoints/", "iteration_50")

# Load checkpoint
memory.load_checkpoint("./checkpoints/iteration_50/")
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_islands` | Number of parallel evolution islands | 3 |
| `population_size` | Maximum number of solutions | 100 |
| `elite_archive_size` | Elite archive size | 50 |
| `migration_interval` | Migration interval (generations) | 10 |

## Best Practices

1. **Island Configuration**: Use 3-5 islands for parallel exploration of complex problems.
2. **Checkpoint Frequency**: Save checkpoints every 10-20 iterations for long-running tasks.
3. **Population Size**: Adjust according to problem complexity to avoid excessive memory usage.

For more configuration options, see the [Configuration Guide](configuration.md).