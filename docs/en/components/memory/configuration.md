# Memory Configuration Guide

A complete configuration reference for the LoongFlow memory system, including Evolution Memory and Grade Memory.

## Evolution Memory Configuration

### Basic Configuration
```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory

memory = InMemory(
    num_islands=3,           # Number of islands
    population_size=100,     # Max solutions per island
    elite_archive_size=50,   # Elite archive size
    migration_interval=10,   # Migration interval
    output_path="./output"   # Checkpoint output path
)
```

### Advanced Parameters
```python
memory = InMemory(
    # Evolutionary algorithm parameters
    boltzmann_temperature=1.0,      # Sampling temperature
    migration_rate=0.2,             # Migration rate
    use_sampling_weight=True,       # Use sampling weight
    sampling_weight_power=1.0,      # Weight power
    
    # Diversity management
    feature_dimensions=["complexity", "diversity", "score"],
    feature_bins=10,                # Number of feature bins
    feature_scaling_method="minmax" # Feature scaling method
)
```

## Grade Memory Configuration

### Basic Configuration
```python
from loongflow.agentsdk.memory.grade.memory import GradeMemory, MemoryConfig

grade_memory = GradeMemory.create_default(
    model=llm_model,
    config=MemoryConfig(
        token_threshold=65536,  # Token threshold for auto-compression
        auto_compress=True      # Enable auto-compression
    )
)
```

### Custom Storage Backend
```python
from loongflow.agentsdk.memory.grade.storage import FileStorage
from loongflow.agentsdk.memory.grade.compressor import LLMCompressor

# Custom storage and compressor
stm_storage = FileStorage("./stm_data/")
mtm_storage = FileStorage("./mtm_data/") 
ltm_storage = FileStorage("./ltm_data/")
compressor = LLMCompressor(model, custom_prompt="Please compress the following conversation history")

grade_memory = GradeMemory(
    stm=ShortTermMemory(stm_storage),
    mtm=MediumTermMemory(mtm_storage, compressor),
    ltm=LongTermMemory(ltm_storage),
    token_counter=token_counter,
    config=MemoryConfig(token_threshold=32768)
)
```

## YAML Configuration Files

### Evolution Memory Configuration
```yaml
evolution:
  num_islands: 3
  population_size: 100
  elite_archive_size: 50
  migration_interval: 10
  migration_rate: 0.2
  output_path: "./evolution_output"
```

### Grade Memory Configuration
```yaml
grade:
  token_threshold: 65536
  auto_compress: true
  storage:
    stm:
      type: "in_memory"
    mtm: 
      type: "file"
      path: "./mtm_data"
    ltm:
      type: "file" 
      path: "./ltm_data"
```

## Environment Variable Configuration

```bash
# Evolution Memory
export EVOLUTION_NUM_ISLANDS=3
export EVOLUTION_POPULATION_SIZE=100
export EVOLUTION_OUTPUT_PATH="./output"

# Grade Memory  
export GRADE_TOKEN_THRESHOLD=65536
export GRADE_AUTO_COMPRESS=true
```

## Performance Tuning

### Memory Optimization
```python
# Reduce population size and number of islands
memory = InMemory(
    num_islands=2,           # Reduce islands
    population_size=50,      # Reduce population
    elite_archive_size=25    # Streamline elite archive
)
```

### Compression Optimization
```python
# Adjust compression threshold
config = MemoryConfig(
    token_threshold=32768,  # Lower threshold, compress more frequently
    auto_compress=True
)
```

## Troubleshooting

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Memory Monitoring
```python
# Check evolution memory status
status = memory.memory_status()
print(f"Current population: {status['global_status']['total_valid_solutions']}")

# Check grade memory size  
size = await grade_memory.get_size()
print(f"Total messages: {size}")
```

## Best Practices

1. **Development Environment**: Use default configuration, focus on feature implementation.
2. **Production Environment**: Adjust the number of islands and population size according to task complexity.
3. **Large-scale Deployment**: Consider distributed storage backends like Redis.

For more implementation details, refer to [Evolution Memory](evolution.md) and [Grade Memory](grade.md).