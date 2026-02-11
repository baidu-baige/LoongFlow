# 📊 Example 02: File Processor

**Difficulty**: ⭐⭐ Intermediate
**Time**: 10-15 minutes
**Goal**: Build a multi-file CSV/JSON processing system with custom skills

---

## 🎯 What You'll Learn

Building on Example 01, this example introduces:

- ✅ **Custom Skills**: How to create and use domain-specific knowledge packages
- ✅ **Multi-file Projects**: Generate complex applications with 5+ files
- ✅ **Professional Structure**: Learn proper code organization patterns
- ✅ **Data Processing**: CSV/JSON parsing, validation, and analysis

**Prerequisites**: Complete [Example 01: TODO List](../01_todo_list/) first

---

## 🚀 Quick Start

```bash
# 1. Navigate to LoongFlow root
cd /path/to/LoongFlow

# 2. Run the example
./run_general.sh 02_file_processor

# 3. Find your generated code
cd output-file-processor/task_*/iteration_*/executor/work_dir/

# 4. Test with sample data
python main.py ../../../../../../agents/general_agent/examples/02_file_processor/sample_data/sales_data.csv
```

---

## 📦 What This Example Generates

A professional file processing system with:

- ✅ **main.py**: CLI entry point with argument parsing
- ✅ **file_readers.py**: CSV/JSON parsing modules
- ✅ **processors.py**: Data cleaning and transformation
- ✅ **analyzers.py**: Statistical analysis functions
- ✅ **output_formatters.py**: Multiple output formats
- ✅ **config.py**: Configuration management
- ✅ **utils.py**: Helper functions

**Total**: 5-7 Python files, ~400-500 lines of code

---

## 🎨 What Makes This Different from Example 01?

### Example 01 (TODO List):
- Single file: `todo_app.py`
- Simple structure
- No external knowledge

### Example 02 (File Processor):
- **Multi-file project** with modular design
- Uses **file-processing skill** for best practices
- Professional code organization
- Advanced error handling

---

## 🧠 Understanding Skills

### What is a Skill?

A skill is a **knowledge package** that guides the agent. It contains:
- Best practices for the domain
- Code templates and patterns
- Common pitfalls to avoid
- Tool references

### The file-processing Skill

Check it out:
```bash
cat .claude/skills/file-processing/SKILL.md
```

This skill teaches the agent:
- How to structure file processing projects
- CSV/JSON best practices
- Error handling patterns
- Performance optimization tips

### Using Skills in Your Task

In [task_config.yaml](task_config.yaml), notice:

```yaml
planners:
  general_planner:
    skills: ["file-processing"]  # 🎯 Load the skill here

executors:
  general_executor:
    skills: ["file-processing"]  # Agent gets domain knowledge
```

---

## 🧪 Test Your Generated App

### Test 1: Basic CSV Analysis

```bash
cd output-file-processor/task_*/iteration_*/executor/work_dir/

# Analyze sales data
python main.py ../../../../../../agents/general_agent/examples/02_file_processor/sample_data/sales_data.csv

# Should output:
# - Total sales sum
# - Average transaction
# - Top products
# - Monthly trends
```

### Test 2: Different Output Formats

```bash
# Text output (default)
python main.py sample.csv --output report.txt

# JSON output
python main.py sample.csv --output stats.json --format json

# CSV summary
python main.py sample.csv --output summary.csv --format csv
```

### Test 3: Error Handling

```bash
# Try with invalid file
python main.py nonexistent.csv
# Should show clear error message, not crash

# Try with corrupted data
echo "bad,data,format" > bad.csv
python main.py bad.csv
# Should handle gracefully with error reporting
```

---

## 📁 Sample Data Provided

### sales_data.csv

Located at [sample_data/sales_data.csv](sample_data/sales_data.csv):

```csv
date,product,quantity,price,customer_id
2024-01-01,Laptop,2,999.99,C001
2024-01-02,Mouse,5,29.99,C002
...
```

Contains 50+ rows of realistic sales transactions.

### students.csv

Located at [sample_data/students.csv](sample_data/students.csv):

```csv
name,age,grade,gpa,major
Alice,20,Junior,3.8,CS
Bob,19,Sophomore,3.2,Math
...
```

Contains 30+ student records for testing.

---

## 📊 Understanding the Output Structure

```
output-file-processor/
└── task_<timestamp>/
    └── iteration_1/
        ├── planner/
        │   └── best_plan.md       # Multi-file architecture plan
        ├── executor/
        │   └── work_dir/          # 🎯 ALL FILES HERE
        │       ├── main.py
        │       ├── file_readers.py
        │       ├── processors.py
        │       ├── analyzers.py
        │       ├── output_formatters.py
        │       ├── config.py
        │       └── utils.py
        ├── evaluator/
        │   └── score.txt          # Score for all files
        └── summary/
            └── insights.md        # Lessons learned
```

**Key Difference**: One `work_dir/` contains multiple files, and the evaluator judges the **entire project** as a whole.

---

## ⚙️ Configuration Highlights

Key differences from Example 01:

```yaml
planners:
  general_planner:
    skills: ["file-processing"]    # 🎯 Load skill
    max_turns: 30                  # More turns for complex tasks

evolve:
  task: |
    Create a professional CSV/JSON file processing system...

    The system should:
    - Support multiple input formats (CSV, JSON)
    - Provide data validation and cleaning
    - Calculate statistics (mean, median, counts, etc.)
    - Support multiple output formats
    - Handle errors gracefully
    - Be well-structured with separate modules

  max_iterations: 30               # More iterations for complexity
  target_score: 0.85
```

---

## 🎯 Expected Results

### Iteration 1 (Score: 0.4-0.5)
- Basic file reading works
- Single-file implementation
- Limited error handling

### Iteration 2 (Score: 0.6-0.7)
- Multi-file structure emerging
- Better error handling
- Some statistics implemented

### Iteration 3 (Score: 0.8-0.9) ✅
- Professional multi-file structure
- Comprehensive statistics
- Multiple output formats
- Robust error handling

---

## 🛠️ Common Issues & Solutions

### Issue: "Skill not found: file-processing"

```bash
# Check if skill exists
ls -la .claude/skills/file-processing/SKILL.md

# If missing, create it (see tutorial)
mkdir -p .claude/skills/file-processing
```

### Issue: "Generated only one file, not multi-file"

- Increase `max_turns: 50` to give agent more chances
- Check if skill loaded properly in logs
- Task description might need to emphasize multi-file structure

### Issue: "Can't process sample data"

```bash
# Check paths - use absolute paths or correct relative paths
pwd  # Make sure you're in work_dir/
ls ../../../../../../agents/general_agent/examples/02_file_processor/sample_data/

# Or copy sample data to work_dir:
cp ../../../../../../agents/general_agent/examples/02_file_processor/sample_data/*.csv .
python main.py sales_data.csv
```

---

## 🎓 Creating Your Own Skills

Want to create a custom skill? Here's how:

### Step 1: Create skill directory

```bash
cd LoongFlow
mkdir -p .claude/skills/my-data-skill
```

### Step 2: Write SKILL.md

```bash
cat > .claude/skills/my-data-skill/SKILL.md << 'EOF'
---
name: "my-data-skill"
description: "Custom data processing patterns for my domain"
---

# My Data Skill

## Purpose
This skill provides best practices for processing domain-specific data formats.

## Best Practices
1. Always validate input data schema
2. Use pandas for large datasets
3. Implement streaming for files > 100MB

## Code Patterns
```python
import pandas as pd

def process_data(filepath):
    """Standard data processing template"""
    df = pd.read_csv(filepath)
    # validation, cleaning, analysis
    return results
```
EOF
```

### Step 3: Use in your task

```yaml
planners:
  general_planner:
    skills: ["file-processing", "my-data-skill"]  # Multiple skills!
```

---

## 🎉 Next Steps

Great job learning about skills and multi-file projects! Now try:

1. **Modify the task**: Add new requirements
   ```yaml
   task: |
     Create a file processor that also:
     - Supports Excel files (.xlsx)
     - Generates plots and charts
     - Has a web dashboard
   ```

2. **Create your own skill**: Follow the steps above to build domain expertise

3. **Try Example 03**: [03_bug_hunter](../03_bug_hunter/) - Learn about production-ready tools and security analysis

---

## 📚 Key Takeaways

- ✅ **Skills** provide domain knowledge to guide the agent
- ✅ **Multi-file projects** are organized in `work_dir/`
- ✅ **Skills are reusable** across different tasks
- ✅ **One evaluation** judges the entire project
- ✅ **Professional structure** emerges from skill guidance

**Ready for advanced features?** Head to [Example 03: Bug Hunter](../03_bug_hunter/) to build production-ready analysis agents! 🚀
