# LoongFlow：会思考、会学习的专家级Agent开发框架

**让创造更自由！LoongFlow，让你的专家经验轻松转化为专业AI生产力。**

LoongFlow 是一个开源专家级Agent开发框架，通过PES思考范式赋能Agent具备思考和持续学习能力。

## ✨ 为什么选择LoongFlow？

**会思考、会学习的专家级Agent开发框架，让Agent像科学家一样思考，助力开发者快速把自己的专业经验转化为专家级Agent。**

### 核心优势

- **智能思考**：创新PES（计划-执行-总结）范式，让Agent具备结构化思考能力，解决长程复杂推理难题
- **持续学习**：创新多结构融合记忆，实现轻量级学习进化
- **稳定高效**：在复杂任务中展现卓越的稳定性和效率

### 已验证的成就

| **领域** | **成就** | **示例** |
|---------------|-------------------|-------------------|
| **数学挑战** | 11个问题上超越最佳结果，7个问题达到SOTA | 圆形装箱问题 |
| **机器学习竞赛** | 40个Kaggle竞赛验证，22枚金牌 | 斯坦福新冠疫苗竞赛 |

## 🚀 快速开始

### 安装要求

LoongFlow 需要 **Python 3.12** 或更高版本

```bash
# 使用uv（推荐）
cd LoongFlow
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# 或使用conda
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .
```

### 运行示例

#### 运行通用进化智能体

```bash
# 运行首个进化任务
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt
./run_task.sh packing_circle_in_unit_square --background

# 查看任务日志
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log
```

#### 运行机器学习智能体

```bash
./run_ml.sh init
./run_ml.sh run ml_example --background

# 查看任务日志
tail -f ./agents/ml_evolve/examples/ml_example/agent.log
```

## 🔧 核心特性

### PES思考范式

LoongFlow的核心是**PES（计划-执行-总结）思考范式**，每个Agent迭代都遵循明确的结构：

- **计划**：理解任务和约束，设计高质量执行蓝图
- **执行**：进行结构化实验，验证中间结果  
- **总结**：深入反思成功与失败，提取可复用洞察

通过学习与进化记忆系统，实现**跳跃式推理**，突破局部搜索局限。

## 🌟 评估效果

### 数学挑战表现

在几何和代数挑战中取得突破性进展，超越AlphaEvolve结果，达到最新SOTA。

### 机器学习竞赛

在MLE-bench评测集的40场Kaggle赛事中，展现强大的全流程自主构建能力。

## 📖 文档导航

- [通用进化智能体文档](./agents/general_evolve.md)
- [机器学习智能体文档](./agents/ml_evolve.md)
- [框架设计文档](./overview/design.md)

## 🤝 贡献

欢迎贡献代码！请阅读[CONTRIBUTING.md](https://github.com/baidu-baige/LoongFlow/blob/main/CONTRIBUTING.md)了解贡献指南。

## 📜 许可证

LoongFlow采用Apache License 2.0许可证。

---

**由LoongFlow社区维护**

*如果LoongFlow对你有帮助，请考虑给这个仓库点个星⭐*