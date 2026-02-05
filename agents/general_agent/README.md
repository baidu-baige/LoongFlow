# General Agent for LoongFlow

## 🚀 Quick Start

General Agent is a flexible, general-purpose agent built on LoongFlow's Plan-Execute-Summary (PES) paradigm, supporting skill-driven task execution.

### 1. Environment Setup

```bash
# Navigate to project root
cd LoongFlow

# Create virtual environment (Python 3.12+ recommended)
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 2. Configure API Key and URL

Currently, General Agent only supports Anthropic models. You can set environment variables to configure API key and URL or fill in the information in the `llm_config` section of the configuration file.

```bash
# Set OpenAI or Anthropic or Litellm-supported API key and URL
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_BASE_URL="your-model-endpoint"
```

### 3. Run Example Task

```bash
# Run hello_world example
./run_general.sh hello_world

# Run in background
./run_general.sh hello_world --background

# Run with custom options
./run_general.sh hello_world --log-level DEBUG --max-iterations 100
```

### 4. Monitor Progress

- **Foreground**: Output appears in terminal
- **Background**: Check log file at `agents/general_agent/examples/hello_world/run.log`
- **Stop background task**: `./run_general.sh stop hello_world`

---

## 🏗️ Task Directory Structure

**📌 Important**: Self-designed skills need to be placed in the `.claude/skills/` folder of the LoongFlow root directory to be loaded correctly.

```
task_name/                    # Task name
├── task_config.yaml          # Main configuration file (required)
├── eval_program.py           # Optional: Custom evaluation script
```

---

## ⚙️ Configuration File Details

### Basic Configuration Example

```yaml
# workspace_path: Output directory configuration
workspace_path: "./output-task-name"

# llm_config: LLM configuration
llm_config:
  model: "anthropic/model-name"         # model-provider/model-name
  url: "https://api.anthropic.com"      # Optional: If set, it will be used first; otherwise, it will be read from ENV
  api_key: "xxx"                        # Optional: If set, it will be used first; otherwise, it will be read from ENV

# evolve: Evolution process configuration
evolve:
  task: |
    You are an expert software developer. Your task is to iteratively improve existing codebase.
    Specific goal: Develop an efficient data processing system.
  max_iterations: 100               # Max evolution iterations
  target_score: 0.9                 # Target score for evolution to stop
  concurrency: 5                    # Number of parallel evolutions
```

### Agent Component Configuration

```yaml
# planners: Planner configuration
planners:
  general_planner:
    skills: ["file_io", "data_processing"]  # Skills to load
    max_turns: 10                           # Max conversation turns
    permission_mode: "acceptEdits"          # Permission mode

# executors: Executor configuration
executors:
  general_executor:
    skills: ["code_generation", "testing"]
    permission_mode: "acceptEdits"

# summarizers: Summarizer configuration
summarizers:
  general_summarizer:
    skills: ["analysis", "reporting"]
    max_turns: 10
```

---

## 🔧 Claude Skills System

### What are Skills?

Skills are modular, self-contained packages that extend agent capabilities, including:
- **Skill description** (SKILL.md): YAML metadata and markdown instructions
- **Script tools** (scripts/): Executable Python/Bash code
- **Reference docs** (references/): Domain knowledge documentation
- **Resource files** (assets/): Templates and sample files

### Using Existing Skills

**📌 Important Note**: Current version only supports loading skills from the `.claude/skills/` directory under the LoongFlow project root.

```yaml
# Load skills from project root .claude/skills/
planners:
  general_planner:
    skills: ["skill-creator", "your-skill-name"]  # Skill names correspond to folder names under .claude/skills/
```

**Skill Directory Structure Example**:
```
LoongFlow/
├── .claude/
│   └── skills/                  # Global skills library
│       ├── skill-creator/       # Skill folder (corresponds to "skill-creator")
│       │   ├── SKILL.md         # Skill description file
│       │   └── scripts/         # Related scripts
│       └── your-skill-name/     # Your custom skill
└── agents/general_agent/
    └── examples/
        └── task_name/
            ├── task_config.yaml # Configuration specifies: skills: ["skill-creator"]
```

### Creating Custom Skills

**📌 Important Note**: Current version only supports loading skills from the `.claude/skills/` directory under the LoongFlow project root.

1. **Create skill directory structure**:
```bash
cd LoongFlow
mkdir -p .claude/skills/my_skill/scripts
```

2. **Create SKILL.md**:
```markdown
---
name: "my_skill"
description: "Processing data files skill. Used for data cleaning, transformation, and analysis."
---

# My Skill

## Features
- Data file reading and parsing
- Data cleaning and preprocessing
- Common data transformation operations

## Usage
Use built-in file_io tools to read data files, then perform corresponding processing.
```

3. **Reference in configuration**:
```yaml
planners:
  general_planner:
    skills: ["my_skill"]
```

---

## 📋 Complete Workflow for Creating New Tasks

### Step 1: Create Task Directory
```bash
cd LoongFlow/agents/general_agent/examples
mkdir my_custom_task
```

### Step 2: Create Configuration File
```bash
# Create task_config.yaml
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
    Develop an efficient data analysis system that can process CSV files and perform basic statistical analysis.
  max_iterations: 50
  target_score: 0.85
EOF
```

### Step 3: (Optional) Add Skills
```bash
cd LoongFlow
mkdir -p .claude/skills/my_data_skill
# Create SKILL.md and related scripts
```

### Step 4: Run Task
```bash
cd LoongFlow
./run_general.sh my_custom_task --log-level INFO
```

---

## 🔧 Advanced Configuration Options

### Permission Modes
- `"default"`: Standard permission behavior
- `"acceptEdits"`: Auto-approve file edits (recommended)

### Built-in Tools
```yaml
build_in_tools: ["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Skill", "Task"]
```

### Performance Tuning
```yaml
general_planner:
  max_turns: 15                    # Increase turns for better planning quality
  max_thinking_tokens: 2000        # Control thinking tokens
```

---

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Failure**:
   - Confirm whether to use an Anthropic protocol model (e.g., anthropic/claude-3 */)
   - Verify API endpoint URL is correct

2. **Skill Loading Failure**:
   - Confirm skill directory structure is correct
   - Check SKILL.md YAML format

3. **Permission Errors**:
   - Set `permission_mode: "acceptEdits"` to avoid frequent confirmations

4. **Where is the Result**:
   - The result will be saved at `{workspace_path}/task_id/iteration_id` sub-directory
   - Each iteration sub-directory include 4 sub-directories: `planner`, `executor`, `evaluator`, and `summary`

5. **Solution Explanation**:
   - Now General_agent will generate multi-files in each iteration, we put all generation files in `executor/work_dir` sub-directory
   - The Evaluator will evaluate the whole `work_dir` as a single evaluation task, and give a final evaluation result for the whole `work_dir`
   - The solution field of the Solution class is set to the absolute path of `executor/work_dir` in each iteration, and you can check the generated files in that absolute path.

### Log Level Control
```bash
# Different verbosity levels
./run_general.sh task_name --log-level DEBUG    # Most detailed
./run_general.sh task_name --log-level INFO     # General information
./run_general.sh task_name --log-level WARNING  # Warnings and errors only
./run_general.sh task_name --log-level ERROR    # Errors only
```

---

## 🆘 Getting Help

If you encounter issues:
1. Check detailed error information in log files
2. Verify configuration file format is correct
3. Ensure environment variables are properly set
4. Reference existing `hello_world` example for comparison

**Start your first task:**
```bash
cd LoongFlow
./run_general.sh hello_world
```
