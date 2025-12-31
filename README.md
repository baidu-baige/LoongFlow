[**中文版**](./README_zh.md)

<div align="center">

**LoongFlow：Evolve Agent Development Framework**

_From atomic components and development frameworks to core scenario Agents, comprehensive evolutionary Agent construction and application support is provided._

<p align="center">
    <a href="https://arxiv.org/abs/xxxx.xxxxxx">
        <img
            src="https://img.shields.io/badge/cs.MA-xxxx.xxxx-B31C1C?logo=arxiv&logoColor=B31C1C"
            alt="arxiv"
        />
    </a>
    <a href="https://pypi.org/project/LoongFlow/">
        <img
            src="https://img.shields.io/badge/python-3.12+-blue?logo=python"
            alt="pypi"
        />
    </a>
    <a href="https://pypi.org/project/evolux/">
        <img
            src="https://img.shields.io/badge/version-v1.0.0-blue"
            alt="pypi"
        />
    </a>
    <a href="./LICENSE">
        <img
            src="https://img.shields.io/badge/license-Apache--2.0-green"
            alt="license"
        />
    </a>       
</p>

[**General-Evolve**](./agents/general_evolve) • [**ML-Evolve**](./agents/ml_evolve) • [**EvolveAgent**](./src/evolux/evolve) • [**ReactAgent**](./src/evolux/react) • [**AgentSDK**](./src/agentsdk)

</div>

<br/>

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>

<td width="30%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🚀 <strong>General-Evolve</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-General_Evolve-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>General Code Evolve Agent </strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Automatically</strong>, <strong>efficiently</strong>, and <strong>stably</strong> perform optimization tasks such as algorithms, mathematical puzzles, and prompts.</p>
</div>
</td>

<td width="30%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🔥 <strong>ML-Evolve</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-ML_Evolve-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Machine Learning Agent</strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Self-evolving</strong> ML Agent that <strong>autonomously</strong> understands data, builds models, and delivers an optimized solution.</p>
</div>

</td>
<td width="30%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">⭐ <strong>LoongFlow</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/FRAMEWORK-EVOLUX-blue" alt="Backend Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Evolve Agent Framework</strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">A modular, <strong>highly extensible Agent framework</strong> for flexible customization and seamless integration.</p>
</div>
</td>

</tr>
</table>

<br/>

**LoongFlow**: Inspired by Wang Yangming's "Enlightenment at Longchang," this concept signifies the deep integration of the model's "knowing" and the tools' "doing" — knowledge propels action, and action yields insight, ushering in the era of Agent cognitive autonomy. It transcends the role of a mere mechanical executor; through iterative refinement in the PES (Plan-Execute-Summarize) cycle, it shatters cognitive boundaries, achieving an evolutionary leap from a "passive tool" to an "autonomous intelligence."

## ✨ Why LoongFlow?

**A high-performance, stable, and scalable framework for evolutionary Agent development, featuring an innovative PES evolutionary paradigm to empower developers in building high-quality evolutionary Agents efficiently.**

<p align="center">
<img src="./assets/images/loongflow_fr_v1.jpg" alt="LoongFlow Framework" width="80%"/>
</p>

- **High Efficiency**: The innovative PES evolutionary paradigm, combined with multi-structural fused evolutionary memory, shifts from "random mutation" to "directed cognitive evolution." It significantly mitigates issues in traditional evolutionary methods, such as low generation quality, excessive ineffective evaluations, repetitive trial-and-error, and high randomness, thereby substantially enhancing evolutionary efficiency and convergence certainty. Compared to conventional methods, overall evolutionary efficiency is improved by approximately 60%.

- **Stability**: Upholding the principle of "engineering certainty," the system systematically encapsulates the inherent uncertainties of models through its architectural design, thereby reducing the burden of model reasoning and establishing a highly stable, reproducible intelligent evolution system. In practical evaluations, LoongFlow has demonstrated significant performance advantages.

- **Ease of Use**: LoongFlow provides comprehensive support, ranging from task-specific evolutionary Agents and a highly scalable evolutionary Agent development framework to modular atomic components. From applications to frameworks, it empowers developers to rapidly deploy evolutionary Agents for solving domain-specific problems, significantly reducing development and fine-tuning costs.

### Experimental Results

#### Math Problem

| Problem                           | Previously best known    |     AlphaEvolve      | LoongFlow Evolve Result |     Details     |
| --------------------------------- | -----------------------  | -------------------- | ----------------------- | --------------- |
| Circle packing in a square        | 2.634 (Higher is Better) |  2.6358627564136983  |  **2.6359829624734026** | [packing_circle_in_unit_square](./agents/general_evolve/examples/packing_circle_in_unit_square)               |
| Circle packing in a rectangle     | 2.364 (Higher is Better) |  2.3658321334167627  |  **2.365832229500823**  | [packing_circle_in_rectangle](./agents/general_evolve/examples/packing_circle_in_rectangle)                   |
| Packing hexagons in hexagons      | 3.943 (Lower is Better)  |  3.930092            |  **3.928906855463712**  | [packing_hexagons_in_hexagons](./agents/general_evolve/examples/packing_hexagons_in_hexagons)                 |
| Max to min ratios                 | 12.89（Lower is Better） |  12.88926611203463   |  **12.889243547212832** | [max_to_min_ratios](./agents/general_evolve/examples/max_to_min_ratios)                                       |
| Minimum Overlap Problem           | 0.380927 (Lower is Better) |  0.380924      | **0.3809137564083654**    | [minimum_overlap_problem](./agents/general_evolve/examples/minimum_overlap_problem)                           |
| An uncertainty inequality         | 0.3523 (Lower is Better)   |  0.35209910442252773  |  **0.352099104421844**   | [uncertainty_inequality](./agents/general_evolve/examples/uncertainty_inequality)                             |
| Second autocorrelation inequality | 0.88922 (Higher is Better) |  0.8962799441554083   | **0.9027021077220739**  | [second_autocorrelation_inequality](./agents/general_evolve/examples/second_autocorrelation_inequality)       |
| First autocorrelation inequality  | 1.5098 (Lower is Better)   |  1.5052939684401607   |  1.509527314861778   | [first_autocorrelation_inequality](./agents/general_evolve/examples/first_autocorrelation_inequality)         |
| Sums differences problems         | 1.059793 (Higher is Better) | 1.1219357374860444   |  1.103534711409646   | [sums_and_differences_problems_1](./agents/general_evolve/examples/sums_and_differences_problems_1)           |
| heilbronn triangles               | 0.036（Higher is Better）|  0.036529889880030156  | 0.0365298898793351    | [heilbronn_problem_for_triangles](./agents/general_evolve/examples/heilbronn_problem_for_triangles)           |
| heilbronn convex regions          | 0.0306（Higher is Better） |  0.030936889034895654  | 0.030900663674639613   | [heilbronn_problem_for_convex_regions](./agents/general_evolve/examples/heilbronn_problem_for_convex_regions) |

Validated on open mathematical problems proposed by Terence Tao and the AlphaEvolve team, the system outperformed all previously known best results on 11 of the problems.

#### ML Task

| Problem                                  | LoongFlow Evolve Result | Details                                          |
| ---------------------------------------- | ----------------------- | ------------------------------------------------ |
| aerial-cactus-identification             | 🥇 Gold                 | [aerial-cactus-identification](./agents/ml_evolve/examples/mlebench/competitions/simple/aerial-cactus-identification) |
| denoising-dirty-documents                | 🥇 Gold                 | [denoising-dirty-documents](./agents/ml_evolve/examples/mlebench/competitions/simple/denoising-dirty-documents) |
| detecting-insults-in-social-commentary   | 🥇 Gold                 | [detecting-insults-in-social-commentary](./agents/ml_evolve/examples/mlebench/competitions/simple/detecting-insults-in-social-commentary) |
| dogs-vs-cats-redux-kernels-edition       | 🥇 Gold                 | [dogs-vs-cats-redux-kernels-edition](./agents/ml_evolve/examples/mlebench/competitions/simple/dogs-vs-cats-redux-kernels-edition) |
| histopathologic-cancer-detection         | 🥇 Gold                 | [histopathologic-cancer-detection](./agents/ml_evolve/examples/mlebench/competitions/simple/histopathologic-cancer-detection) |
| nomad2018-predict-transparent-conductors | 🥇 Gold                 | [nomad2018-predict-transparent-conductors](./agents/ml_evolve/examples/mlebench/competitions/simple/nomad2018-predict-transparent-conductors) |
| plant-pathology-2020-fgvc7               | 🥇 Gold                 | [plant-pathology-2020-fgvc7](./agents/ml_evolve/examples/mlebench/competitions/simple/plant-pathology-2020-fgvc7) |
| tabular-playground-series-dec-2021       | 🥇 Gold                 | [tabular-playground-series-dec-2021](./agents/ml_evolve/examples/mlebench/competitions/simple/tabular-playground-series-dec-2021) |
| the-icml-2013-whale-challenge-right-whale-redux   | 🥇 Gold        | [the-icml-2013-whale-challenge-right-whale-redux](./agents/ml_evolve/examples/mlebench/competitions/simple/the-icml-2013-whale-challenge-right-whale-redux) |
| google-quest-challenge          | 🥇 Gold                 | [google-quest-challenge](./agents/ml_evolve/examples/mlebench/competitions/medium/google-quest-challenge) |
| plant-pathology-2021-fgvc8      | 🥇 Gold                 | [plant-pathology-2021-fgvc8](./agents/ml_evolve/examples/mlebench/competitions/medium/plant-pathology-2021-fgvc8) |
| us-patent-phrase-to-phrase-matching     | 🥇 Gold                 | [us-patent-phrase-to-phrase-matching](./agents/ml_evolve/examples/mlebench/competitions/medium/us-patent-phrase-to-phrase-matching) |
| predict-volcanic-eruptions-ingv-oe      | 🥇 Gold                 | [predict-volcanic-eruptions-ingv-oe](./agents/ml_evolve/examples/mlebench/competitions/hard/predict-volcanic-eruptions-ingv-oe) |
| stanford-covid-vaccine                  | 🥇 Gold                 | [stanford-covid-vaccine](./agents/ml_evolve/examples/mlebench/competitions/hard/stanford-covid-vaccine) |

Validated on 20 Kaggle machine learning competitions from the OpenAI MLE-Bench benchmark, the system achieved gold medals in 14 contests. Complete results will be announced after all competitions are concluded.

#### Others

Additionally, validation was conducted on problems such as [mathematical puzzles](./agents/general_evolve/examples/math_flip) and [MOE load balancing algorithms](./agents/general_evolve/examples/moe_lb_time)，Detailed examples can be found in [Examples](./agents/general_evolve/examples).

## 🚀 Quick Start

### Installation

> LoongFlow requires **Python 3.12** or higher.

```bash
# Install uv/conda and clone repository
uv: https://docs.astral.sh/uv/getting-started/installation/
Miniforge: https://conda-forge.org/download/

# Install with uv
cd LoongFlow
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Install with conda
cd LoongFlow
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .

```

### Run Examples

#### Run General Evolve Agent

```bash
# Config LLM: Edit task_config.yaml, recommend to use gemini-3-pro-preview or deepseek-r1-250528
# Example: ./agents/general_evolve/examples/packing_circle_in_unit_square/task_config.yaml
# The model needs to configure providers as needed, default provider is openai. for example: openai/gemini-3-pro-preview
llm_config:
  url: "https://xxxxxx/v1"
  api_key: "******"
  model: "openai/gemini-3-pro-preview"

# Run your first evolve task, the evolution results are in the ./output directory
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt
./run_task.sh packing_circle_in_unit_square --background

# Check task log
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# Stop task
./run_task.sh stop packing_circle_in_unit_square

```

#### Run ML Evolve Agent

```bash
# Config LLM: Edit task_config.yaml, recommend to use gemini-3-pro-preview or deepseek-r1-250528
# Example: ./agents/ml_evolve/examples/ml_example/task_config.yaml
# The model needs to configure providers as needed, default provider is openai. for example: openai/gemini-3-pro-preview
llm_config:
  url: "https://xxxxxx/v1"
  api_key: "******"
  model: "openai/gemini-3-pro-preview"

# Init ml evolve
./run_ml.sh init

# Run your first evolve task, the evolution results are in the ./output directory
# ./run_ml.sh run <task_name> [--background] [other Python args]
./run_ml.sh run ml_example --background

# Check task log
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# Stop task
./run_ml.sh stop ml_example

```

### LoongFlow Usage

#### EvolveAgent

```python
from evolux.evolve import EvolveAgent

# Config evolve agent
agent = EvolveAgent(
    config=config,
    checkpoint_path=checkpoint_path,
)

# Register worker（Implement the Planner, Executor, and Summary interfaces）
agent.register_planner_worker("planner", PlanAgent)
agent.register_executor_worker("executor", ExecuteAgent)
agent.register_summary_worker("summary", SummaryAgent)

# Run agent
result = await agent()
```

For more details, please refer to [EvolveAgent](./src/evolux/evolve/README.md)

#### ReActAgent

```python
from evolux.react import AgentContext, ReActAgent
from agentsdk.tools import TodoReadTool, TodoWriteTool, Toolkit

# Build agent context
toolkit = Toolkit()
toolkit.register_tool(TodoReadTool())
toolkit.register_tool(TodoWriteTool())

# Build default react agent
agent = ReActAgent.create_default(model=model, sys_prompt=sys_prompt, toolkit=toolkit)

# Run agent
result = await agent(message)
```

For more details, please refer to [ReActAgent](./src/evolux/react/README.md)

## 🤝 Contribution

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## 📜 License

LoongFlow is licensed under the Apache License 2.
