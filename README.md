[**中文版**](./README_zh.md)

<div align="center">

<h2 align="center">LoongFlow: A Thinking & Learning Framework for Expert-Grade AI Agents.</h2>

_Set Creativity Free! LoongFlow turns your expertise into professional AI productivity._

LoongFlow is an open-source **expert-grade Agent development framework**.

Enable Agents to think and learn through the PES paradigm, and accumulate experience through iteration.

<p align="center">
    <a href="https://github.com/baidu-baige/LoongFlow/stargazers"><img src="https://img.shields.io/github/stars/baidu-baige/LoongFlow?style=social" alt="GitHub stars"></a>
    <a href="https://arxiv.org/abs/2512.24077">
        <img
            src="https://img.shields.io/badge/cs.AI-2512.24077-B31C1C?logo=arxiv&logoColor=B31C1C"
            alt="arxiv"
        />
    </a>
    <a href="https://pypi.org/project/LoongFlow/">
        <img
            src="https://img.shields.io/badge/python-3.12+-blue?logo=python"
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

[🚀 **Quick Start**](#quick-start) • [**Examples**](#loongflow-examples) • [**Math-Agent**](./agents/math_agent) • [**ML-Agent**](./agents/ml_agent) • [**Discussions**](https://github.com/baidu-baige/LoongFlow/discussions)

</div>

<br/>

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>

<td width="33%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🚀 <strong>Math-Agent</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-math_agent-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Math Problem Agent </strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Efficient</strong>,<strong>stable</strong> driving of math algorithm design and continuous evolution.</p>
</div>
</td>

<td width="30%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🔥 <strong>ML-Agent</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-ml_agent-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Machine Learning Agent</strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Full-process</strong>,<strong>autonomous</strong> construction and continuous evolutionary breakthrough.</p>
</div>

</td>
<td width="30%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">⭐ <strong>LoongFlow</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/FRAMEWORK-LoongFlow-blue" alt="Backend Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Universal Agent Framework</strong></p>
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">A Universal Agent Framework for <strong>Expert-Grade</strong> AI Productivity.</p>
</div>
</td>

</tr>
</table>

<br/>

**LoongFlow**: Inspired by Wang Yangming's "Enlightenment at Longchang".LoongFlow is dedicated to breaking the barrier between Knowing and Doing. We enable wisdom to awaken through the unity of knowledge and action, ensuring that every drop of professional expertise is transformed into powerful **AI productivity**.

## ✨ Why LoongFlow?

---

**An expert-grade Agent framework that thinks and learns. It empowers Agents to think like scientists, helping developers rapidly transform their professional expertise into expert-level Agents.**

<p align="center">
<img src="./assets/images/loongflow_fr_v1.jpg" alt="LoongFlow Framework" width="80%"/>
</p>

- **Intelligent Thinking**:
  Innovative PES (Planning-Execution-Summary) Paradigm. LoongFlow empowers Agents with structured thinking to tackle long-range complex reasoning challenges. This enables Agents to iterate through high-difficulty tasks with the rigorous mindset of a human scientist.
- **Continuous Learning**:
  Innovative Multi-Structure Fusion Memory. By actively generating model reasoning contexts, LoongFlow allows Agents to continuously synthesize experience during task iterations. This results in a "run-and-improve" mechanism, achieving lightweight learning and evolution without heavy retraining.

We believe that the key to designing an expert-level Agent capable of solving complex problems lies in the **Agent’s thinking paradigm**. The thinking paradigm determines the complexity of problems an Agent can handle and sets the ceiling for its effectiveness. LoongFlow is built specifically for complex tasks requiring long-range reasoning, helping developers rapidly build Agents with domain-expert performance.

### Proven Achievements

<div align="center">

| **Domain**                                             | **Achievement**                                                                                                                  | **Example**                                                                                            |
| ------------------------------------------------------ |----------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------ |
| **Mathematical Challenges (Tao’s & AlphaEvolve sets)** | Outperformed the best human results on 11 problems and surpassed AlphaEvolve’s results on 7 problems, achieving the latest SOTA. | [Circle Packing](./agents/math_agent/examples/packing_circle_in_unit_square)                          |
| **MLE-bench (Kaggle Challenges)**                      | Achieved medals across all 48 Kaggle competitions, including 26 Gold Medals.                                                                | [Stanford-Covid-Vaccine](./agents/ml_agent/examples/mlebench/competitions/hard/stanford-covid-vaccine) |

</div>

### LoongFlow vs Traditional Agent Approaches:

<table> <tr> <th align="left">Aspect</th> <th align="left">Prompt / Tool-Based Agents</th> <th align="left">OpenEvolve-Style Evolution</th> <th align="left">LoongFlow</th> </tr> <tr> <td><strong>Core Loop</strong></td> <td>Generate → Retry</td> <td>Mutate → Select</td> <td>Plan → Execute → Summary</td> </tr> <tr> <td><strong>Reasoning Depth</strong></td> <td>Shallow</td> <td>Limited</td> <td>Long-horizon, structured</td> </tr> <tr> <td><strong>Learning from Failure</strong></td> <td>❌</td> <td>Partial</td> <td>✅ Explicit reflection</td> </tr> <tr> <td><strong>Experience Reuse</strong></td> <td>❌</td> <td>❌</td> <td>✅ Structured memory</td> </tr> <tr> <td><strong>Stability</strong></td> <td>Fragile</td> <td>Often unstable</td> <td>Stable convergence</td> </tr> <tr> <td><strong>Best Use Case</strong></td> <td>Simple automation</td> <td>Search-heavy tasks</td> <td>Expert-level problem solving</td> </tr> </table>

## Quick Start

---

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

#### Run Math Agent

```bash
# Config LLM: Edit task_config.yaml, recommend to use gemini-3-pro-preview or deepseek-r1-250528
# Example: ./agents/math_agent/examples/packing_circle_in_unit_square/task_config.yaml
# The model needs to configure providers as needed, default provider is openai. for example: openai/gemini-3-pro-preview
llm_config:
  url: "https://xxxxxx/v1"
  api_key: "******"
  model: "openai/gemini-3-pro-preview"

# Run your first evolve task, the evolution results are in the ./output directory
uv pip install -r ./agents/math_agent/examples/packing_circle_in_unit_square/requirements.txt
./run_math.sh packing_circle_in_unit_square --background

# Check task log
tail -f ./agents/math_agent/examples/packing_circle_in_unit_square/run.log

# Stop task
./run_math.sh stop packing_circle_in_unit_square

```

#### Run ML Agent

```bash
# Config LLM: Edit task_config.yaml, recommend to use gemini-3-pro-preview or deepseek-r1-250528
# Example: ./agents/ml_agent/examples/ml_example/task_config.yaml
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
tail -f ./agents/ml_agent/examples/ml_example/agent.log

# Stop task
./run_ml.sh stop ml_example

```

---

## How LoongFlow Works

LoongFlow is designed around a simple idea:

> Expert-level performance emerges not from better mutations, but from better thinking, reflection, and accumulated experience.

To achieve this, LoongFlow organizes agent behavior into a thinking–learning–evolving loop.

---

### From Evolutionary Agents to Thinking Agents

From Evolutionary Agents to Thinking Agents

Frameworks such as **OpenEvolve** and **AlphaEvolve** demonstrated that agents can improve through iteration, evaluation, and selection.

This marked a clear step beyond static prompting.

However, in real-world expert tasks, purely evolutionary loops often struggle because:

- Exploration is blind or weakly guided
- Long-horizon reasoning breaks easily
- Experience remains task-specific
- Agents converge prematurely to local optima

The core issue is not evolution itself, but **the lack of a structured thinking process**.

LoongFlow addresses this by shifting the abstraction:

from _evolving outputs_ to **standardizing how agents think, act, and learn**.

---

### PES Thinking Paradigm

At the core of LoongFlow is the **PES (Plan–Execute–Summary) thinking paradigm**, inspired by how human experts conduct research:

Each agent iteration follows the same explicit structure:

<table> <tr> <td width="33%">
Plan

- Understand the task and constraints
- Retrieve relevant past experience
- Design a clear, high-quality execution blueprint

> Planning ensures generation is deliberate rather than blind.

</td> <td width="33%">
Execute

- Perform structured experimentation
- Verify intermediate results
- Avoid low-value or redundant trials

> Execution becomes controlled experimentation, not guesswork.

</td> <td width="33%">
Summary

- Reflect deeply on successes and failures
- Extract reusable insights
- Persist experience into structured memory

> Summary prevents agents from repeating the same mistakes.

</td> </tr> </table>

<p align="center">
<img src="./assets/images/pes-flow.jpg" alt="LoongFlow Framework" width="80%"/>
</p>

PES transforms evolution from a mutation-driven process into a **reasoning-guided improvement loop**.

---

### Learning & Evolutionary Memory

Thinking alone is not enough. To improve over time, agents must **remember, generalize, and escape local optima**.

LoongFlow integrates PES with a hybrid evolutionary memory system:

- Multi-Island + MAP-Elites to preserve diversity
- Adaptive Boltzmann selection to balance exploration and exploitation
- Global evolutionary tree memory for long-range experience retrieval

This allows agents to perform **jump-style reasoning** — leveraging past discoveries to move beyond incremental local search.

## LoongFlow Examples

---

### Mathematical Challenges (Tao’s & AlphaEvolve sets)

| Problem                           | Previously best known       | AlphaEvolve          | LoongFlow Evolve Result | Details                                                                                                    |
| --------------------------------- | --------------------------- | -------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| Circle packing in a square        | 2.634 (Higher is Better)    | 2.6358627564136983   | **2.6359829624734026**  | [packing_circle_in_unit_square](./agents/math_agent/examples/packing_circle_in_unit_square)               |
| Circle packing in a rectangle     | 2.364 (Higher is Better)    | 2.3658321334167627   | **2.365832229500823**   | [packing_circle_in_rectangle](./agents/math_agent/examples/packing_circle_in_rectangle)                   |
| Packing hexagons in hexagons      | 3.943 (Lower is Better)     | 3.930092             | **3.928906855463712**   | [packing_hexagons_in_hexagons](./agents/math_agent/examples/packing_hexagons_in_hexagons)                 |
| Max to min ratios                 | 12.89（Lower is Better）    | 12.88926611203463    | **12.889243547212832**  | [max_to_min_ratios](./agents/math_agent/examples/max_to_min_ratios)                                       |
| Minimum Overlap Problem           | 0.380927 (Lower is Better)  | 0.380924             | **0.3809137564083654**  | [minimum_overlap_problem](./agents/math_agent/examples/minimum_overlap_problem)                           |
| An uncertainty inequality         | 0.3523 (Lower is Better)    | 0.35209910442252773  | **0.352099104421844**   | [uncertainty_inequality](./agents/math_agent/examples/uncertainty_inequality)                             |
| Second autocorrelation inequality | 0.88922 (Higher is Better)  | 0.8962799441554083   | **0.9027021077220739**  | [second_autocorrelation_inequality](./agents/math_agent/examples/second_autocorrelation_inequality)       |
| First autocorrelation inequality  | 1.5098 (Lower is Better)    | 1.5052939684401607   | 1.509527314861778       | [first_autocorrelation_inequality](./agents/math_agent/examples/first_autocorrelation_inequality)         |
| Sums differences problems         | 1.059793 (Higher is Better) | 1.1219357374860444   | 1.103534711409646       | [sums_and_differences_problems_1](./agents/math_agent/examples/sums_and_differences_problems_1)           |
| heilbronn triangles               | 0.036（Higher is Better）   | 0.036529889880030156 | 0.0365298898793351      | [heilbronn_problem_for_triangles](./agents/math_agent/examples/heilbronn_problem_for_triangles)           |
| heilbronn convex regions          | 0.0306（Higher is Better）  | 0.030936889034895654 | 0.030900663674639613    | [heilbronn_problem_for_convex_regions](./agents/math_agent/examples/heilbronn_problem_for_convex_regions) |

Across 11 challenges in geometry and algebra, LoongFlow outperformed all known best results and surpassed AlphaEvolve on 7 specific problems, achieving the latest SOTA.

### MLE-bench (Kaggle Challenges)

| Problem | LoongFlow Evolve Result | Details |
| :--- | :--- | :--- |
| aerial-cactus-identification | 🥇 Gold | [aerial-cactus-identification](./agents/ml_agent/examples/mlebench/competitions/simple/aerial-cactus-identification) |
| aptos2019-blindness-detection | 🥇 Gold | [aptos2019-blindness-detection](./agents/ml_agent/examples/mlebench/competitions/simple/aptos2019-blindness-detection) |
| denoising-dirty-documents | 🥇 Gold | [denoising-dirty-documents](./agents/ml_agent/examples/mlebench/competitions/simple/denoising-dirty-documents) |
| detecting-insults-in-social-commentary | 🥇 Gold | [detecting-insults-in-social-commentary](./agents/ml_agent/examples/mlebench/competitions/simple/detecting-insults-in-social-commentary) |
| dogs-vs-cats-redux-kernels-edition | 🥇 Gold | [dogs-vs-cats-redux-kernels-edition](./agents/ml_agent/examples/mlebench/competitions/simple/dogs-vs-cats-redux-kernels-edition) |
| histopathologic-cancer-detection | 🥇 Gold | [histopathologic-cancer-detection](./agents/ml_agent/examples/mlebench/competitions/simple/histopathologic-cancer-detection) |
| nomad2018-predict-transparent-conductors | 🥇 Gold | [nomad2018-predict-transparent-conductors](./agents/ml_agent/examples/mlebench/competitions/simple/nomad2018-predict-transparent-conductors) |
| plant-pathology-2020-fgvc7 | 🥇 Gold | [plant-pathology-2020-fgvc7](./agents/ml_agent/examples/mlebench/competitions/simple/plant-pathology-2020-fgvc7) |
| tabular-playground-series-dec-2021 | 🥇 Gold | [tabular-playground-series-dec-2021](./agents/ml_agent/examples/mlebench/competitions/simple/tabular-playground-series-dec-2021) |
| the-icml-2013-whale-challenge-right-whale-redux | 🥇 Gold | [the-icml-2013-whale-challenge-right-whale-redux](./agents/ml_agent/examples/mlebench/competitions/simple/the-icml-2013-whale-challenge-right-whale-redux) |
| chaii-hindi-and-tamil-question-answering | 🥇 Gold | [chaii-hindi-and-tamil-question-answering](./agents/ml_agent/examples/mlebench/competitions/medium/chaii-hindi-and-tamil-question-answering) |
| google-quest-challenge | 🥇 Gold | [google-quest-challenge](./agents/ml_agent/examples/mlebench/competitions/medium/google-quest-challenge) |
| hubmap-kidney-segmentation | 🥇 Gold | [hubmap-kidney-segmentation](./agents/ml_agent/examples/mlebench/competitions/medium/hubmap-kidney-segmentation) |
| inaturalist-2019-fgvc6 | 🥇 Gold | [inaturalist-2019-fgvc6](./agents/ml_agent/examples/mlebench/competitions/medium/inaturalist-2019-fgvc6) |
| learning-agency-lab-automated-essay-scoring-2 | 🥇 Gold | [learning-agency-lab-automated-essay-scoring-2](./agents/ml_agent/examples/mlebench/competitions/medium/learning-agency-lab-automated-essay-scoring-2) |
| plant-pathology-2021-fgvc8 | 🥇 Gold | [plant-pathology-2021-fgvc8](./agents/ml_agent/examples/mlebench/competitions/medium/plant-pathology-2021-fgvc8) |
| seti-breakthrough-listen | 🥇 Gold | [seti-breakthrough-listen](./agents/ml_agent/examples/mlebench/competitions/medium/seti-breakthrough-listen) |
| tensorflow-speech-recognition-challenge | 🥇 Gold | [tensorflow-speech-recognition-challenge](./agents/ml_agent/examples/mlebench/competitions/medium/tensorflow-speech-recognition-challenge) |
| us-patent-phrase-to-phrase-matching | 🥇 Gold | [us-patent-phrase-to-phrase-matching](./agents/ml_agent/examples/mlebench/competitions/medium/us-patent-phrase-to-phrase-matching) |
| whale-categorization-playground | 🥇 Gold | [whale-categorization-playground](./agents/ml_agent/examples/mlebench/competitions/medium/whale-categorization-playground) |
| 3d-object-detection-for-autonomous-vehicles | 🥇 Gold | [3d-object-detection-for-autonomous-vehicles](./agents/ml_agent/examples/mlebench/competitions/hard/3d-object-detection-for-autonomous-vehicles) |
| iwildcam-2019-fgvc6 | 🥇 Gold | [iwildcam-2019-fgvc6](./agents/ml_agent/examples/mlebench/competitions/hard/iwildcam-2019-fgvc6) |
| predict-volcanic-eruptions-ingv-oe | 🥇 Gold | [predict-volcanic-eruptions-ingv-oe](./agents/ml_agent/examples/mlebench/competitions/hard/predict-volcanic-eruptions-ingv-oe) |
| rsna-miccai-brain-tumor-radiogenomic-classification | 🥇 Gold | [rsna-miccai-brain-tumor-radiogenomic-classification](./agents/ml_agent/examples/mlebench/competitions/hard/rsna-miccai-brain-tumor-radiogenomic-classification) |
| stanford-covid-vaccine | 🥇 Gold | [stanford-covid-vaccine](./agents/ml_agent/examples/mlebench/competitions/hard/stanford-covid-vaccine) |
| vinbigdata-chest-xray-abnormalities-detection | 🥇 Gold | [vinbigdata-chest-xray-abnormalities-detection](./agents/ml_agent/examples/mlebench/competitions/hard/vinbigdata-chest-xray-abnormalities-detection) |

Achieved medals across all 48 Kaggle competitions within the MLE-bench, including 26 Gold Medals. The full results can be found at [competitions](./agents/ml_agent/examples/mlebench/competitions)

### Others

Additionally, validation was conducted on problems such as [mathematical puzzles](./agents/math_agent/examples/math_flip) and [MOE load balancing algorithms](./agents/math_agent/examples/moe_lb_time)，Detailed examples can be found in [Examples](./agents/math_agent/examples).

## 🧩 Advanced Usage

---

### PESAgent

```python
from loongflow.framework.pes import PESAgent

# Config evolve agent
agent = PESAgent(
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

For more details, please refer to [PESAgent](./src/loongflow/framework/pes/README.md)

### ReActAgent

```python
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.agentsdk.tools import TodoReadTool, TodoWriteTool, Toolkit

# Build agent context
toolkit = Toolkit()
toolkit.register_tool(TodoReadTool())
toolkit.register_tool(TodoWriteTool())

# Build default react agent
agent = ReActAgent.create_default(model=model, sys_prompt=sys_prompt, toolkit=toolkit)

# Run agent
result = await agent(message)
```

For more details, please refer to [ReActAgent](./src/loongflow/framework/react/README.md)

## Visualization

---

**Real-time evolution tracking** with interactive web interface:

```
# Launch visualization server
python agents/math_agent/visualizer/visualizer.py --port 8888 --checkpoint-path output-circle-packing/database/checkpoints
```

**Features:**

- 🌳 Evolution tree with parent-child relationships
- 📈 Performance tracking across generations
- 🔍 Code diff viewer showing mutations
- 📊 Island map for visualizing the distribution of solutions

<figure align="center">
<img src="./assets/images/visualize.png" alt="LoongFlow Framework" width="1000%"/>
</figure>

## FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

Like CirclePacking problem, if use Gemini 3 Pro, the cost is about **$10** in total

</details>

<details>
<summary><b>🆚 How is LoongFlow related to OpenEvolve or AlphaEvolve?</b></summary>

OpenEvolve and AlphaEvolve explore evolutionary improvement through mutation and selection.
LoongFlow builds on these ideas but introduces a higher-level abstraction:

**A structured thinking and learning paradigm inspired by human experts.**

Rather than optimizing mutations, LoongFlow focuses on how agents plan, execute, reflect, and accumulate experience across iterations.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** LoongFlow supports any OpenAI-compatible API:

- **Commercial**: OpenAI, Google
- **Local**: vLLM, sglang

Just set the `llm_config` in your config to point to your endpoint.

</details>

## 🤝 Contribution

We welcome contributions! Here's how to get started:

1. 🍴 Fork the repository
2. 🌿 Create your feature branch: git checkout -b feat-amazing-feature
3. ✨ Add your changes and tests
4. 📝 Commit with a clear message
5. 🚀 Push and create a Pull Request

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## 💬 Contact

Welcome to join our community on

| [Discord](https://discord.gg/YSfdrC8HJh)                                | Wechat                                                                 |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| <img src="./assets/images/discord_invite.png" width="200" height="200"> | <img src="./assets/images/wechat_invite.jpg" width="200" height="200"> |

## 📜 License

LoongFlow is licensed under the Apache License 2.0.

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{LoongFlow2025,
      title={LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm},
      author={Chunhui Wan and Xunan Dai and Zhuo Wang and Minglei Li and Yanpeng Wang and Yinan Mao and Yu Lan and Zhiwen Xiao},
      year={2025},
      eprint={2512.24077},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24077},
}
```

---

<div align="center">

### **🚀 Ready to build your expert agent?**

**Maintained by the LoongFlow community**

_If LoongFlow helps you, please consider starring this repository._

</div>
