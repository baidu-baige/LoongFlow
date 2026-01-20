# Grade Memory

Grade Memory provides hierarchical management of agent conversation history, based on a three-layer architecture of STM (Short-Term), MTM (Medium-Term), and LTM (Long-Term), supporting automatic compression and token management.

## Three-Layer Memory Architecture

### Short-Term Memory (STM)
- Stores recent conversation history
- Fast access, non-persistent
- Defaults to in-memory storage

### Medium-Term Memory (MTM)
- Stores compressed conversation summaries
- Supports LLM automatic compression
- Persistable across sessions

### Long-Term Memory (LTM)
- Stores important facts and knowledge
- Permanent storage
- Supports file persistence

## Core Features

### Initialization
```python
from loongflow.agentsdk.memory.grade.memory import GradeMemory

grade_memory = GradeMemory.create_default(
    model=llm_model,
    config=MemoryConfig(
        token_threshold=65536,  # Token threshold
        auto_compress=True      # Auto compression
    )
)
```

### Message Management
```python
# Add message
await grade_memory.add(message)

# Get all memory content
context = await grade_memory.get_memory()

# Manually commit important information to long-term memory
await grade_memory.commit_to_ltm(important_message)
```

### Automatic Compression
When the token count exceeds the threshold, the system automatically triggers compression:
1. Merges messages from STM and MTM
2. Uses LLM to generate a summary
3. Clears session memory, retaining the compression results

## Storage Backends

### Memory Storage
```python
from loongflow.agentsdk.memory.grade.storage import InMemoryStorage
storage = InMemoryStorage()  # Suitable for development and testing
```

### File Storage
```python
from loongflow.agentsdk.memory.grade.storage import FileStorage  
storage = FileStorage("./memory_data/")  # Suitable for production environments
```

## Compressor

### LLM Compressor
Uses a language model to intelligently compress conversation history:
```python
from loongflow.agentsdk.memory.grade.compressor import LLMCompressor
compressor = LLMCompressor(model)
```

## Best Practices

1. **Token Management**: Set a reasonable `token_threshold` based on the model context length.
2. **Compression Strategy**: Manually commit important information to LTM to avoid loss.
3. **Storage Selection**: Use in-memory storage for development environments and file storage for production environments.

For more configuration options, please refer to the [Configuration Guide](configuration.md).