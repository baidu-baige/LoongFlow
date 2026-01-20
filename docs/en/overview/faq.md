# FAQ

## Getting Started

### 🔧 Can I use my own LLM?

**Supports all OpenAI-compatible APIs**:
- Commercial: OpenAI, Google (Gemini series)
- Local Deployment: vLLM, sglang, Ollama

```yaml
llm_config:
  url: "https://your-endpoint.com/v1"
  api_key: "your-api-key"
  model: "your-model-name"
```

### 🐍 Why is Python 3.12+ required?

Python 3.12 provides key features:
- Enhanced type system and error messages
- Improved asynchronous performance
- Better support for Agent frameworks

---

## Framework Concepts

### 🆚 How is it different from OpenEvolve/AlphaEvolve?

The core difference lies in the **thinking paradigm**:

| Aspect | OpenEvolve/AlphaEvolve | LoongFlow |
|------|------------------------|-----------|
| **Core Abstraction** | Mutation-Selection Evolution | **PES Thinking Paradigm** |
| **Learning Method** | Task-Specific Improvement | **Cross-Task Experience Accumulation** |
| **Reasoning Depth** | Limited | **Structured Long-Range Reasoning** |

### 🧠 How is PES different from ReAct?

PES provides a structured improvement loop:
- **Planning**: Deep strategic thinking
- **Executing**: Structured experimental validation
- **Summarizing**: Systematic reflection and experience extraction

### 📚 How does the memory system work?

Integrates diverse memory structures:
- **Evolution Tree**: Tracks solution lineage
- **Multi-Island Map**: Maintains solution diversity
- **Experience Patterns**: Stores successful strategies

---

## Technical Issues

### ⚙️ Why use UV instead of pip?

UV advantages:
- **Faster Installation**: Significantly faster than pip
- **Better Dependency Resolution**: More reliable environment
- **Modern Tools**: Designed for Python 3.12+

### 🚀 Why not distribute on PyPI?

Technical reasons:
- **Complex Dependencies**: Different Agents have specific requirements
- **Flexible Configuration**: Users can customize Agent and LLM settings
- **Transparency**: Full access to source code

### 🔍 How to debug agents?

Debugging steps:
1. **Check Logs**: `tail -f ./agents/.../run.log`
2. **Verify LLM Configuration**: API accessibility, quota limits
3. **Check Generated Code**: View the `./output` directory

---

## Performance and Scalability

### 📊 How about scalability?

**Performance Characteristics**:
- Small problems: Fast convergence (<30 generations)
- Medium complexity: Stable improvement (30-100 generations)
- Large-scale optimization: Progressive refinement (>100 generations)

**Resource Requirements**:
- Memory: ~500MB base + ~100MB per island
- CPU: Single core suffices, multi-core supports parallelism

### 🎯 What problems is it suitable for?

**Ideal Problem Characteristics**:
- Clearly defined goals and evaluation criteria
- Complexity requiring strategic reasoning
- Can benefit from iterative improvement

**Success Cases**:
- ✅ Mathematical optimization and discovery
- ✅ Machine Learning competitions
- ✅ Algorithm design improvement

---

## Community and Support

### 🤝 How to contribute?

All contributions are welcome:
- **Code**: Bug fixes, new Agent implementations
- **Documentation**: Tutorials, usage examples
- **Community**: Answering questions, sharing experiences

### 📞 Where to get help?

Support channels:
- **GitHub Discussions**: Technical questions and community help
- **Discord Community**: Real-time discussion and collaboration
- **Issue Tracker**: Bug reports and feature requests

---

## Advanced Topics

### 🎨 Can I create custom agents?

**Supports Custom Development**:
1. Define PES components (Planner, Executor, Summarizer)
2. Implement necessary interfaces
3. Configure task-specific settings
4. Test and validate

### 🔄 How does transfer learning work?

Implemented through the memory system:
- **Automatic Retrieval**: Identifies relevant past experiences
- **Strategy Adaptation**: Adapts experience to the current context
- **Validation Improvement**: Tests and optimizes adapted strategies

---

Still have questions? Join our [GitHub Discussions](https://github.com/baidu-baige/LoongFlow/discussions) or [Discord Community](https://discord.gg/YSfdrC8HJh) for help!