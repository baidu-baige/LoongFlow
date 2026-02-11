# General Agent for LoongFlow

## 🚀 快速开始

General Agent 是基于 LoongFlow Plan-Execute-Summary (PES) 范式的通用智能体，支持技能驱动的任务执行。

### 1. 环境配置
```bash
# 进入项目根目录
cd LoongFlow

# 创建虚拟环境（推荐使用 Python 3.12+）
uv venv .venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 2. 配置 API 密钥 和 URL

当前，General Agent仅支持Anthropic模型。您可以通过设置环境变量来配置API密钥和URL，或在配置文件的`llm_config`部分填写信息。

```bash
# 设置 OpenAI 或 Litellm兼容的 API 密钥 和 URL
export ANTHROPIC_API_KEY="your-api-key-here"
export ANTHROPIC_BASE_URL="https://api.deepseek.com/v1"
```

### 3. 运行示例任务

```bash
# 运行 hello_world 示例
./run_general.sh hello_world

# 后台运行
./run_general.sh hello_world --background

# 带参数运行
./run_general.sh hello_world --log-level DEBUG --max-iterations 100
```

### 4. 监控进度

- **前台运行**: 输出直接显示在终端
- **后台运行**: 查看日志文件 `agents/general_agent/examples/hello_world/run.log`
- **停止后台任务**: `./run_general.sh stop hello_world`

---

## 🏗️ 任务目录结构

**📌 重要说明**: 自定义技能需要放在LoongFlow根目录的 `.claude/skills/` 文件夹下才能被正确加载。

```
task_name/                    # 任务名称
├── task_config.yaml          # 主配置文件（必需）
├── eval_program.py           # 评估脚本（可选）
```

---

## ⚙️ 配置文件详解

### 基础配置示例

```yaml
# workspace_path: 输出目录配置
workspace_path: "./output-task-name"

# llm_config: LLM 配置
llm_config:
  model: "anthropic/deepseek-v3.2"  #  模型协议/模型名称
  url: "https://api.anthropic.com"  # （可选）如果配置会优先使用，否则读取环境变量
  api_key: "xxx"                    # （可选）如果配置会优先使用，否则读取环境变量

# evolve: 进化流程配置
evolve:
  task: |
    你是一名专家软件开发工程师。你的任务是迭代改进现有的代码库。
    具体目标：开发一个高效的数据处理系统。
  max_iterations: 100               # 最大迭代次数
  target_score: 0.9                 # 任务目标分数，如果评估分数达到或者超过该值，任务停止
  concurrency: 5                    # 并行运行数
```

### 智能体组件配置

```yaml
# planners: 规划器配置
planners:
  general_planner:
    skills: ["file_io", "data_processing"]  # 加载的技能
    max_turns: 10                           # 最大对话轮次
    permission_mode: "acceptEdits"          # 权限模式

# executors: 执行器配置
executors:
  general_executor:
    skills: ["code_generation", "testing"]
    max_turns: 20
    permission_mode: "acceptEdits"

# summarizers: 总结器配置
summarizers:
  general_summarizer:
    skills: ["analysis", "reporting"]
    max_turns: 10
    permission_mode: "acceptEdits"
```

---

## 📊 实时可视化仪表板

General Agent 包含了一个强大的 Web 可视化工具，用于实时监控任务进度、探索迭代历史和分析结果。

### 功能特性

- **📈 分数演化追踪**：动态图表展示各迭代间的性能改进
- **🔄 迭代历史**：浏览和对比所有迭代的详细指标
- **📝 Markdown 渲染**：对计划、总结和评估日志进行富文本格式化
- **📁 层级文件树**：可折叠的文件夹结构探索生成的代码
- **💻 代码查看器**：语法高亮的文件内容查看器
- **🔄 自动刷新**：任务执行时实时更新数据

### 快速开始

启动可视化工具来监控你的任务执行：

```bash
# 监控单个工作空间
python agents/general_agent/visualizer/visualizer.py --workspace ./output-todo-list

# 同时监控多个工作空间
python agents/general_agent/visualizer/visualizer.py \
  --workspaces "./output-todo-list,./output-file-processor,./output-bug-hunter"

# 自定义端口和主机
python agents/general_agent/visualizer/visualizer.py \
  --workspace ./output-todo-list \
  --port 8080 \
  --host 0.0.0.0
```

然后在浏览器中打开 `http://127.0.0.1:8080` 查看仪表板。

### 仪表板界面

**任务列表侧边栏**：
- 查看所有被监控工作空间中的任务
- 一目了然地查看迭代次数和最新分数
- 点击任意任务查看详细进度

**任务概览面板**：
- 带动态 Y 轴缩放的分数历史图表
- 显示所有迭代的网格视图，包含分数和文件数量
- 点击任意迭代查看详情

**迭代详情视图**：
- **计划**：Markdown 渲染的规划文档，展示任务分解
- **生成的文件**：层级树状结构
  - 点击文件夹（📁）展开/折叠
  - 点击文件查看语法高亮的内容
- **评估**：详细的评分和反馈日志
- **总结**：从该迭代中获得的洞察和学习

### 使用示例

**监控运行中的任务**：
```bash
# 终端 1：启动任务
./run_general.sh 01_todo_list

# 终端 2：启动可视化工具
python agents/general_agent/visualizer/visualizer.py --workspace ./output-todo-list
```

**分析已完成的任务**：
```bash
# 任务完成后查看结果
python agents/general_agent/visualizer/visualizer.py --workspace ./output-bug-hunter
```

**对比多个任务**：
```bash
# 一起监控所有示例任务
python agents/general_agent/visualizer/visualizer.py \
  --workspaces "./output-todo-list,./output-file-processor,./output-bug-hunter,./output-circle-packing"
```

### 技术细节

- **后端**：Flask REST API 提供任务数据服务
- **前端**：原生 JavaScript 配合 Chart.js 和 Marked.js
- **数据格式**：读取 General Agent 的标准目录结构
  - 任务目录：基于 UUID 的组织方式
  - 迭代：数字子目录（1, 2, 3, ...）
  - 评估结果：`executor/evaluator_dir/best_evaluation.json`
  - 生成的文件：`executor/work_dir/`
- **自动刷新**：查看任务时每 10 秒更新一次详情

### 故障排除

**没有显示任务**：
- 确保工作空间路径指向正确的输出目录
- 检查任务目录是否存在并包含迭代子目录

**分数未显示**：
- 验证迭代文件夹中是否存在 `executor/evaluator_dir/best_evaluation.json`
- 检查 JSON 是否包含有效的 `score` 字段

**文件无法加载**：
- 确认迭代中存在 `executor/work_dir/` 目录
- 检查文件权限

---

## 🔧 Claude 技能系统

### 什么是技能？

技能是扩展智能体能力的模块化包，包含：
- **技能描述** (SKILL.md)：YAML元数据和markdown说明
- **脚本工具** (scripts/)：可执行的Python/Bash代码
- **参考文档** (references/)：领域知识文档
- **资源文件** (assets/)：模板和样例文件

### 使用现有技能

**📌 重要说明**: 当前版本仅支持从LoongFlow项目根目录下的`.claude/skills/`目录加载技能。

```yaml
# 从项目根目录的.claude/skills/加载技能
planners:
  general_planner:
    skills: ["skill-creator", "your-skill-name"]  # 技能名称对应根目录.claude/skills/下的文件夹名
```

**技能目录结构示例**:
```
LoongFlow/
├── .claude/
│   └── skills/                  # 全局技能库
│       ├── skill-creator/       # 技能文件夹（对应技能名"skill-creator"）
│       │   ├── SKILL.md         # 技能描述文件
│       │   └── scripts/         # 相关脚本
│       └── your-skill-name/     # 你的自定义技能
└── agents/general_agent/
    └── examples/
        └── task_name/
            ├── task_config.yaml # 配置中指定：skills: ["skill-creator"]
```

### 创建自定义技能

**📌 重要说明**: 自定义技能需要放在项目根目录的 `.claude/skills/` 文件夹下才能被正确加载。

1. **在项目根目录创建技能**：
```bash
# 在项目根目录下创建技能文件夹
cd LoongFlow
mkdir -p .claude/skills/my_skill/scripts
```

2. **创建 SKILL.md**：
```markdown
---
name: "my_skill"
description: "处理数据文件的技能。用于数据清理、转换和分析。"
---

# My Skill

## 功能特性
- 数据文件读取和解析
- 数据清理和预处理
- 常用的数据转换操作

## 使用方法
使用内置的 file_io 工具读取数据文件，然后进行相应的处理。
```

3. **在配置中引用**：
```yaml
planners:
  general_planner:
    skills: ["my_skill"]  # 技能名称对应 .claude/skills/my_skill 文件夹
```

---

## 📋 创建新任务的完整流程

### 步骤1：创建任务目录
```bash
cd LoongFlow/agents/general_agent/examples
mkdir my_custom_task
```

### 步骤2：创建配置文件
```bash
# 创建 task_config.yaml
cat > my_custom_task/task_config.yaml << 'EOF'
workspace_path: "./output-my-task"
llm_config:
  model: "anthropic/deepseek-v3.2"

planners:
  general_planner:
    skills: ["skill-creator"]
    max_turns: 10

executors:
  general_executor:
    skills: ["skill-creator"]

summarizers:
  general_summarizer:
    max_turns: 10

evolve:
  task: |
    开发一个高效的数据分析系统，能够处理CSV文件并进行基本统计分析。
  max_iterations: 50
  target_score: 0.85
EOF
```

### 步骤3：（可选）添加技能
```bash
# 注意：技能需要放在项目根目录下才能生效
cd LoongFlow
mkdir -p .claude/skills/my_data_skill
# 在 .claude/skills/my_data_skill/ 目录下创建 SKILL.md 和相关脚本
```

### 步骤4：运行任务
```bash
cd LoongFlow
./run_general.sh my_custom_task --log-level INFO
```

---

## 🔧 高级配置选项

### 权限模式
- `"default"`：标准权限模式
- `"acceptEdits"`：自动批准文件编辑（推荐）

### 内置工具
```yaml
build_in_tools: ["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Skill", "Task"]
```

### 性能调优
```yaml
general_planner:
  max_turns: 15                    # 增加轮次提高规划质量
  max_thinking_tokens: 2000        # 控制思考令牌数
```

---

## 🐛 故障排除

### 常见问题

1. **API 连接失败**：
   - 确认是否使用Anthropic协议的模型（如 anthropic/claude-3 */
   - 验证 API 端点 URL 是否正确

2. **技能加载失败**：
   - 确认技能目录结构正确
   - 检查 SKILL.md 的 YAML 格式

3. **权限错误**：
   - 设置 `permission_mode: "acceptEdits"` 避免频繁确认

4. **结果在哪**:
   - 最终结果会保存在 `{workspace_path}/task_id/iteration_id` 子目录下
   - 每个迭代子目录包含 4 个子目录: `planner`, `executor`, `evaluator`, 和 `summary`

5. **结果解释**：
    - 现在 General_agent 会生成多文件结果，在每个轮次中，我们会将所有生成文件放在 `executor/work_dir` 子目录下
    - 评估器会将整个 `executor` 目录作为一个整体评估任务，给出该目录的最终评估结果
    - 每个迭代的 `solution` 字段都会设置为该迭代 `executor/work_dir` 目录的绝对路径，你可以在那个绝对路径下查看生成的文件。

### 日志级别控制
```bash
# 不同详细程度的日志
./run_general.sh task_name --log-level DEBUG    # 最详细
./run_general.sh task_name --log-level INFO     # 一般信息  
./run_general.sh task_name --log-level WARNING  # 仅警告和错误
./run_general.sh task_name --log-level ERROR    # 仅错误信息
```

---

## 🆘 获取帮助

如果遇到问题：
1. 检查日志文件中的详细错误信息
2. 验证配置文件格式是否正确
3. 确保环境变量已正确设置
4. 参考现有的 `hello_world` 示例进行对比

**开始你的第一个任务：**
```bash
cd LoongFlow
./run_general.sh hello_world
```