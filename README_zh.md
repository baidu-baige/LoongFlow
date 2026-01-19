[**English Version**](./README.md)

<div align="center">

<h2 align="center">LoongFlow：会思考、会学习的专家级Agent开发框架</h2>

_让创造更自由！LoongFlow，让你的经验轻松转化为专业的 AI 生产力。_

通过PES思考范式让Agent会思考、会学习，具备长程复杂推理能力，并且能够跳过局部最优，在迭代中积累经验实现专家级效果突破。

<p align="center">
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
    <a href="https://pypi.org/project/LoongFlow/">
        <img
            src="https://img.shields.io/badge/version-v1.0.0-blue"
            alt="pypi"
        />
    </a>
    <a href="./LICENSE">
        <img
            src="https://img.shields.io/badge/license-Apache--2.0-black"
            alt="license"
        />
    </a>       
</p>

[🚀 **Quick Start**](#快速开始) • [**Examples**](#相关示例) • [**General-Evolve**](./agents/general_evolve) • [**ML-Evolve**](./agents/ml_evolve) • [**Discussions**](https://github.com/baidu-baige/LoongFlow/discussions)

</div>

<br/>

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>

<td width="33%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🚀 <strong>General-Evolve</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-General_Evolve-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>通用进化智能体</strong></p>
</div>
<div style="height: 120px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>高效</strong>、<strong>稳定</strong>驱动通用算法设计和持续进化</p>
</div>
</td>

<td width="33%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🔥 <strong>ML-Evolve</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/AGENT-ML_Evolve-blue" alt="agent Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>机器学习智能体</strong></p>
</div>
<div style="height: 120px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>全流程、全自主</strong>完整构建，持续进化突破</p>
</div>

</td>
<td width="33%" align="center" style="vertical-align: top; padding: 20px;">
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">⭐ <strong>LoongFlow</strong></h3>
</div>
<div align="center" style="margin: 10px 0;">
  <img src="https://img.shields.io/badge/FRAMEWORK-LoongFlow-blue" alt="Backend Badge" />
</div>
<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>通用Agent框架</strong></p>
</div>
<div style="height: 120px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>会思考、会学习</strong>的专家级通用Agent开发框架</p>
</div>
</td>

</tr>
</table>

<br/>

**LoongFlow**：取名源自“龙场悟道”，寓意LoongFlow 致力于打破“知”与“行”的藩篱，让经验在知行合一中觉醒，让每一份专业积淀都能转化为强大的AI生产力。

## ✨ Why LoongFlow?

--- 
**会思考、会学习的专家级Agent开发框架，让Agent像科学家一样思考，助力开发者快速把自己的专业经验转化为专家级Agent。**

<p align="center">
<img src="./assets/images/loongflow_fr_v1.jpg" alt="LoongFlow Framework" width="80%"/>
</p>

- **会思考**：创新PES思考范式，让Agent具备结构化思考能力，解决长程复杂推理难题。让Agent可以像人类科学家一样，迭代解决高难度任务。
- **会学习**：创新多结构融合记忆，通过主动生成模型推理上下文，让Agent在任务迭代中，持续总结经验，越跑越好，实现轻量级学习进化。


我们认为，设计一个能解决复杂问题的专家级Agent，关键就在于Agent的思考模式，思考模式决定了这个Agent能解决问题的复杂度和效果上限。LoongFlow就是为解决需要长程思考的复杂任务而生，帮助开发者快速构建领域专家级效果Agent。

### 已证实的成果

<div align="center">

| **领域**                              | **成果**                                                               | **示例**                                                                                                  |
|-------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **数学挑战 (Tao’s & AlphaEvolve sets)** | 在 11 个问题上超越了人类最佳成绩，在 7 个问题上超越了 AlphaEvolve 的成绩，达到了最新的 SOTA（最先进技术）水平。 | [Circle Packing](./agents/general_evolve/examples/packing_circle_in_unit_square)                        |
| **MLE-bench (Kaggle Challenges)**   | 经40项Kaggle竞赛验证，获得22枚金牌。                                              | [Stanford-Covid-Vaccine](./agents/ml_evolve/examples/mlebench/competitions/hard/stanford-covid-vaccine) |

</div>

### LoongFlow对比传统Agent框架:

<table> <tr> <th align="left">方面</th> <th align="left">提示/工具型Agent</th> <th align="left">OpenEvolve式演化</th> <th align="left">LoongFlow</th> </tr> <tr> <td><strong>核心循环</strong></td> <td>生成 → 重试</td> <td>变异 → 选择</td> <td>计划 → 执行 → 总结</td> </tr> <tr> <td><strong>推理深度</strong></td> <td>浅</td> <td>有限</td> <td>长周期、结构化</td> </tr> <tr> <td><strong>从失败中学习</strong></td> <td>❌</td> <td>部分</td> <td>✅ 显式反思</td> </tr> <tr> <td><strong>经验重用</strong></td> <td>❌</td> <td>❌</td> <td>✅ 结构化记忆</td> </tr> <tr> <td><strong>稳定性</strong></td> <td>脆弱</td> <td>通常不稳定</td> <td>稳定收敛</td> </tr> <tr> <td><strong>最佳用例</strong></td> <td>简单自动化</td> <td>搜索密集型任务</td> <td>专家级问题解决</td> </tr> </table>

## 快速开始

---

### 安装

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

### 运行示例

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

--- 

## LoongFlow 是如何工作的？

LoongFlow 的设计理念很简单：

> 专家级表现并非源于更优的变异，而是源于更优秀的思考、反思和经验积累。

为了实现这一点，LoongFlow将智能体的行为组织成一个思考-学习-演化的循环。

---
### 从进化Agent到思考Agent

诸如 **OpenEvolve** 和 **AlphaEvolve** 之类的框架引入了一个重要的理念：智能体可以通过迭代、评估和选择来改进自身。

这标志着智能体在静态提示的基础上迈出了重要一步。

然而，在现实世界的专家任务中，纯粹的进化循环往往难以奏效，原因如下：

- 探索往往是盲目的或缺乏引导
- 长远推理容易失效
- 经验仍然局限于特定任务
- 智能体经常陷入局部最优解

问题的核心不在于进化本身，而在于**缺乏结构化的思考过程**。

LoongFlow 通过转变抽象概念来解决这个问题：

从 _演化输出_ 转变为**标准化智能体的思考、行动和学习方式**。

---

### PES 思考范式

LoongFlow 的核心是**PES（计划-执行-总结）思考范式**，其灵感来源于人类专家开展研究的方式：

每次智能体迭代都遵循相同的、明确的流程：

<table> <tr> <td width="33%">
计划

- 理解任务和限制条件
- 回顾相关经验
- 设计清晰、高质量的执行方案

> 规划确保方案的生成是经过深思熟虑的，而不是盲目的。


</td> <td width="33%">
执行

- 进行结构化实验
- 验证中间结果
- 避免低价值或重复的试验

> 执行过程应成为受控实验，而非猜测。

</td> <td width="33%">
总结

- 深入反思成功与失败
- 提取可复用的洞见
- 将经验巩固到结构化记忆中

> 总结有助于防止智能体重蹈覆辙。
> 
</td> </tr> </table>

<p align="center">
<img src="./assets/images/pes-flow.jpg" alt="LoongFlow Framework" width="80%"/>
</p>

PES 将进化从突变驱动的过程转变为**推理引导的改进循环**。

---

### 从进化记忆中学习

仅靠思考是不够的。智能体还必须**记住、概括并跳出局部最优解**。

LoongFlow 引入了一种混合进化记忆系统：

- **多岛 + MAP-Elites** 用于保持多样性
- **自适应玻尔兹曼选择** 用于平衡探索与利用
- **全局进化树记忆** 用于长程上下文检索

这使得智能体能够进行**跳跃式推理**，而非增量式局部搜索。

### LoongFlow 对比其他框架

| 维度 | 基于工具的智能体框架 | 进化智能体（例如 OpenEvolve、AlphaEvolve） | 长流 |
|---------|----------------------------|---------------------------------------------------|-----------|
| 核心抽象 | 工具链 | 变异与选择 | PES 思维范式 |
| 长时推理 | ❌ | ⚠️ | ✅ |
| 结构化反思 | ❌ | ❌ | ✅ |
| 从失败中学习 | ❌ | 有限 | ✅ |
| 专家知识重用 | ❌ | ❌ | ✅ |
| 跳出局部最优 | ❌ | 部分 | ✅ |

## 相关示例

---

### 陶哲轩&AlphaEvolve发布数学挑战

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

在11个几何和代数问题挑战中，取得了超过已知最好结果，并在7个问题上超过AlphaEvolve进化结果，取得最新SOTA。

### Kaggle机器学习竞赛

| Problem                                  | LoongFlow Evolve Result | Details                                          | Description                                                  |
| ---------------------------------------- | ----------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| aerial-cactus-identification             | 🥇 Gold                 | [aerial-cactus-identification](./agents/ml_evolve/examples/mlebench/competitions/simple/aerial-cactus-identification) | 用无人机拍的照片，识别图片里有没有仙人掌，目标是训练AI自动找到沙漠里的仙人掌。 |
| denoising-dirty-documents                | 🥇 Gold                 | [denoising-dirty-documents](./agents/ml_evolve/examples/mlebench/competitions/simple/denoising-dirty-documents) | 把发黄、有污渍的老旧文件照片清理干净，目标是让扫描后的文字清晰可读。 |
| detecting-insults-in-social-commentary   | 🥇 Gold                 | [detecting-insults-in-social-commentary](./agents/ml_evolve/examples/mlebench/competitions/simple/detecting-insults-in-social-commentary) | 识别社交媒体评论里有没有骂人的话，目标是自动过滤网络暴力内容。 |
| dogs-vs-cats-redux-kernels-edition       | 🥇 Gold                 | [dogs-vs-cats-redux-kernels-edition](./agents/ml_evolve/examples/mlebench/competitions/simple/dogs-vs-cats-redux-kernels-edition) | 分类任务，把狗猫图片判别出来，目标是区分图片里是狗还是猫。 |
| histopathologic-cancer-detection         | 🥇 Gold                 | [histopathologic-cancer-detection](./agents/ml_evolve/examples/mlebench/competitions/simple/histopathologic-cancer-detection) | 用AI看病理切片，判断有没有癌细胞，目标是帮助医生更快更准地诊断癌症。 |
| nomad2018-predict-transparent-conductors | 🥇 Gold                 | [nomad2018-predict-transparent-conductors](./agents/ml_evolve/examples/mlebench/competitions/simple/nomad2018-predict-transparent-conductors) | 预测新材料能不能当透明导体用，目标是找到能导电又透明的材料，做手机屏幕、太阳能板啥的。 |
| plant-pathology-2020-fgvc7               | 🥇 Gold                 | [plant-pathology-2020-fgvc7](./agents/ml_evolve/examples/mlebench/competitions/simple/plant-pathology-2020-fgvc7) | 看苹果叶子照片，判断是健康还是有病，目标是帮农民及时发现病害，减少损失。  |
| tabular-playground-series-dec-2021       | 🥇 Gold                 | [tabular-playground-series-dec-2021](./agents/ml_evolve/examples/mlebench/competitions/simple/tabular-playground-series-dec-2021) | 给一堆数据，预测结果，这是Kaggle的入门练习赛，目标是练手学数据科学。 |
| the-icml-2013-whale-challenge-right-whale-redux   | 🥇 Gold        | [the-icml-2013-whale-challenge-right-whale-redux](./agents/ml_evolve/examples/mlebench/competitions/simple/the-icml-2013-whale-challenge-right-whale-redux) | 看鲸鱼照片，认出是哪条鲸鱼，目标是保护濒危的露脊鲸。 |
| google-quest-challenge          | 🥇 Gold                 | [google-quest-challenge](./agents/ml_evolve/examples/mlebench/competitions/medium/google-quest-challenge) | 给问答内容打标签，判断问题好坏和答案相关性，目标是提升问答系统的质量。 |
| plant-pathology-2021-fgvc8      | 🥇 Gold                 | [plant-pathology-2021-fgvc8](./agents/ml_evolve/examples/mlebench/competitions/medium/plant-pathology-2021-fgvc8) | 通过苹果叶子照片判断有没有病害，目标是帮助农民及时发现植物疾病。 |
| us-patent-phrase-to-phrase-matching     | 🥇 Gold                 | [us-patent-phrase-to-phrase-matching](./agents/ml_evolve/examples/mlebench/competitions/medium/us-patent-phrase-to-phrase-matching) | 判断两个专利短语的相似程度，目标是帮助专利审查员快速找到相关专利文件。 |
| predict-volcanic-eruptions-ingv-oe      | 🥇 Gold                 | [predict-volcanic-eruptions-ingv-oe](./agents/ml_evolve/examples/mlebench/competitions/hard/predict-volcanic-eruptions-ingv-oe) | 分析火山传感器数据预测火山喷发时间，目标是帮助提前预警减少灾害损失。   |
| stanford-covid-vaccine                  | 🥇 Gold                 | [stanford-covid-vaccine](./agents/ml_evolve/examples/mlebench/competitions/hard/stanford-covid-vaccine) | 预测RNA疫苗的稳定性，目标是设计出更稳定的新冠mRNA疫苗。 |

在MLE-bench评测集中40场kaggle机器学习赛事验证，取得22个金牌，完整结果将在完成全部赛事后公布。

### 其他尝试

另外在[数学谜题](./agents/general_evolve/examples/math_flip)，[MOE负载均衡](./agents/general_evolve/examples/moe_lb)等问题上验证，具体可在[Examples](./agents/general_evolve/examples)查看。

## 🧩 高级使用

---

### EvolveAgent

```python
from loongflow.framework.evolve import EvolveAgent

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

更多细节，可以查看 [EvolveAgent](./src/evolux/evolve)

#### ReActAgent

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

更多细节，可以查看 [ReActAgent](./src/evolux/react)

## 可视化界面

---

通过交互式网页界面进行**实时演化跟踪**：

```
# Launch visualization server
python agents/general_evolve/visualizer/visualizer.py --port 8888 --checkpoint-path output-circle-packing/database/checkpoints
```

**特点:**

- 🌳 具有亲子关系的进化树
- 📈 跨代的表现追踪
- 🔍 代码差异查看器显示每个个体的代码差异
- 📊 用于可视化解决方案分布的岛状图

<figure align="center">
<img src="./assets/images/visualize.png" alt="LoongFlow Framework" width="1000%"/>
</figure>

## FAQ

<details>
<summary><b>💰跑一次要多少钱</b></summary>

与 CirclePacking 问题类似，如果使用 Gemini 3 Pro，总成本约为 **10 美元**。

</details>

<details>
<summary><b>🆚 LoongFlow 与 OpenEvolve 或 AlphaEvolve 有什么关系？</b></summary>

OpenEvolve 和 AlphaEvolve 探索通过变异和选择实现进化改进。

LoongFlow 在这些理念的基础上，引入了更高层次的抽象：

**一种受人类专家启发而构建的结构化思维和学习范式。**

LoongFlow 并非着眼于优化变异，而是关注智能体如何在迭代过程中进行规划、执行、反思和经验积累。
</details>

<details>
<summary><b>🔧 我能用自己部署的LLM么?</b></summary>

**是的！** LoongFlow 支持所有 OpenAI 兼容的 API：

- **商业版**：OpenAI、Google
- **本地版**：vllm、sglang

只需在您的配置中设置 `llm_config` 指向您的端点即可。

</details>

## 🤝 贡献

欢迎贡献！以下是入门指南：

1. 🍴 Fork 此仓库

2. 🌿 创建你的特性分支：git checkout -b feat-amazing-feature

3. ✨ 添加你的更改和测试

4. 📝 提交更改并附上清晰的提交信息

5. 🚀 推送并创建拉取请求

更详细的请阅读 [CONTRIBUTING.md](./CONTRIBUTING.md) 文件，了解行为准则以及提交拉取请求的流程。

## 💬 讨论

欢迎加入我们的社区进行讨论：

| [Discord](https://discord.gg/YSfdrC8HJh)       | Wechat                                                                 |
|------------------------------------------------|------------------------------------------------------------------------|
| <img src="./assets/images/discord_invite.png" width="200" height="200"> | <img src="./assets/images/wechat_invite.jpg" width="200" height="200"> |

## 📜 许可

LoongFlow 采用 Apache License 2.0 许可。


## 📚 引用

如果您觉得我们的工作对您有帮助，请考虑引用我们的论文：

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

### **🚀 准备好构建您的专家智能体了吗？**

**由 LoongFlow 社区维护**

*如果 LoongFlow 对您有所帮助，请考虑为该代码库点赞。*

</div>
