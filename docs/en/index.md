# LoongFlow: A Thinking and Learning Expert-Level Agent Development Framework

**Set creation free! LoongFlow allows you to effortlessly transform your expert experience into professional AI productivity.**

LoongFlow is an open-source expert-level agent development framework that empowers agents with thinking and continuous learning capabilities through the PES thinking paradigm.

## ✨ Why Choose LoongFlow?

**An expert-level agent development framework that thinks and learns, enabling agents to think like scientists and helping developers quickly transform their professional experience into expert-level agents.**

### Core Advantages

- **Smart Thinking**: Innovative PES (Plan-Execute-Summarize) paradigm, enabling agents with structured thinking capabilities to solve long-horizon complex reasoning problems.
- **Continuous Learning**: Innovative multi-structure fusion memory, achieving lightweight learning and evolution.
- **Stable & Efficient**: Demonstrates exceptional stability and efficiency in complex tasks.

### Proven Achievements

| **Domain** | **Achievement** | **Example** |
|---------------|-------------------|-------------------|
| **Math Challenges** | Surpassed best results on 11 problems, reached SOTA on 7 problems | Packing circles in a unit square |
| **Machine Learning Competitions** | Verified in 40 Kaggle competitions, 22 gold medals | Stanford COVID-19 Vaccine Competition |

## 🚀 Quick Start

### Installation Requirements

LoongFlow requires **Python 3.12** or higher.

```bash
# Use uv (recommended)
cd LoongFlow
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Or use conda
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .
```

### Running Examples

#### Run General Evolutionary Agent

```bash
# Run the first evolution task
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt
./run_task.sh packing_circle_in_unit_square --background

# View task logs
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log
```

#### Run Machine Learning Agent

```bash
./run_ml.sh init
./run_ml.sh run ml_example --background

# View task logs
tail -f ./agents/ml_evolve/examples/ml_example/agent.log
```

## 🔧 Core Features

### PES Thinking Paradigm

The core of LoongFlow is the **PES (Plan-Execute-Summarize) thinking paradigm**, where every Agent iteration follows a clear structure:

- **Plan**: Understand tasks and constraints, design high-quality execution blueprints.
- **Execute**: Conduct structured experiments, verify intermediate results.
- **Summarize**: Deeply reflect on successes and failures, extract reusable insights.

Through the learning and evolutionary memory system, it achieves **leapfrog reasoning**, breaking through the limitations of local search.

## 🌟 Evaluation Results

### Performance in Math Challenges

Achieved breakthrough progress in geometry and algebra challenges, surpassing AlphaEvolve results and reaching the latest SOTA.

### Machine Learning Competitions

Demonstrated strong full-process autonomous construction capabilities in 40 Kaggle competitions within the MLE-bench evaluation set.

## 📖 Documentation Navigation

- [General Evolutionary Agent Documentation](./agents/general_evolve.md)
- [Machine Learning Agent Documentation](./agents/ml_evolve.md)
- [Framework Design Documentation](./overview/design.md)

## 🤝 Contribution

Welcome code contributions! Please read [CONTRIBUTING.md](https://github.com/baidu-baige/LoongFlow/blob/main/CONTRIBUTING.md) to understand the contribution guidelines.

## 📜 License

LoongFlow is licensed under the Apache License 2.0.

---

**Maintained by the LoongFlow Community**

*If LoongFlow is helpful to you, please consider giving this repository a star ⭐*