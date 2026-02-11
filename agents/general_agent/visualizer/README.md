# General Agent Visualizer - Quick Start Guide

## 🎯 Overview

The General Agent Visualizer is a web-based dashboard for monitoring your General Agent tasks in real-time. It provides an intuitive interface to track progress, view generated code, and analyze results.

## 🚀 Quick Start

### 1. Run a Task

First, run a General Agent task:

```bash
cd LoongFlow
./run_general.sh 01_todo_list --background
```

### 2. Start the Visualizer

Launch the visualizer pointing to your output directory:

```bash
python agents/general_agent/visualizer/visualizer.py \
    --port 8080 \
    --workspace ./output-todo-list
```

### 3. Open in Browser

Visit [http://localhost:8080](http://localhost:8080) in your web browser.

## 📖 Usage Examples

### Monitor a Single Task

```bash
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-todo-list \
    --port 8080
```

### Monitor Multiple Tasks

```bash
python agents/general_agent/visualizer/visualizer.py \
    --workspaces "output-todo-list,output-file-processor,output-bug-hunter" \
    --port 8080
```

### Use a Different Host/Port

```bash
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-todo-list \
    --host 0.0.0.0 \
    --port 3000
```

## 🎨 Dashboard Features

### 📋 Task List (Left Sidebar)

- View all available tasks
- See iteration count and latest score
- Click to select and view details

### 📈 Score History Chart

- Visual graph of score evolution
- Track quality improvements over iterations
- Identify best-performing iterations

### 🔄 Iteration Browser

- Grid view of all iterations
- Click any iteration to view details:
  - **Plan**: What the agent planned to do
  - **Generated Files**: All created files
  - **Evaluation**: Test results and score
  - **Summary**: Insights and lessons learned

### 💻 Code Viewer

- Click any file to view its contents
- Syntax highlighting for Python, JSON, YAML, Markdown
- Full-screen modal for comfortable reading

### 🔄 Auto-refresh

- Dashboard automatically refreshes every 10 seconds
- Stay up-to-date with running tasks
- Manual refresh button available

## 🔧 Command-Line Options

```bash
python agents/general_agent/visualizer/visualizer.py --help
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--workspace` | Single workspace path | Required (if --workspaces not used) |
| `--workspaces` | Comma-separated list of workspaces | - |
| `--port` | Server port | 8080 |
| `--host` | Server host | 127.0.0.1 |

## 📁 Expected Directory Structure

The visualizer expects tasks to follow this structure:

```
output-<task>/
└── task_<timestamp>/
    ├── iteration_1/
    │   ├── planner/
    │   │   └── best_plan.md
    │   ├── executor/
    │   │   └── work_dir/
    │   │       ├── todo_app.py
    │   │       └── todos.json
    │   ├── evaluator/
    │   │   ├── score.txt          # Score: 0.75
    │   │   └── evaluation.log
    │   └── summary/
    │       └── insights.md
    ├── iteration_2/
    │   └── ...
    └── iteration_3/
        └── ...
```

## 🐛 Troubleshooting

### Issue: "Workspace not found"

**Solution**: Ensure the workspace path exists and contains task directories:

```bash
# Check if output directory exists
ls -la output-todo-list/

# Should show task_* directories
```

### Issue: "Port already in use"

**Solution**: Use a different port:

```bash
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-todo-list \
    --port 8081  # Try a different port
```

### Issue: "Cannot connect to http://localhost:8080"

**Checklist:**
1. Verify the server started successfully
2. Check for firewall blocking
3. Try accessing from `http://127.0.0.1:8080` instead

### Issue: "No tasks shown"

**Solution**: Ensure tasks have been run and have at least one iteration:

```bash
# Run an example task first
./run_general.sh 01_todo_list

# Then start visualizer
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-todo-list
```

## 🎯 Best Practices

### 1. Run Visualizer in Background

```bash
# Terminal 1: Run task
./run_general.sh 02_file_processor --background

# Terminal 2: Monitor with visualizer
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-file-processor
```

### 2. Monitor Multiple Tasks

Compare different approaches side-by-side:

```bash
./run_general.sh 01_todo_list --background
./run_general.sh 02_file_processor --background

python agents/general_agent/visualizer/visualizer.py \
    --workspaces "output-todo-list,output-file-processor"
```

### 3. Share with Team

Make accessible to your team:

```bash
python agents/general_agent/visualizer/visualizer.py \
    --workspace ./output-bug-hunter \
    --host 0.0.0.0 \
    --port 8080

# Team members can access at:
# http://your-machine-ip:8080
```

## 🔒 Security Notes

- The visualizer is intended for **local development use**
- Default binding to `127.0.0.1` (localhost only)
- Use `--host 0.0.0.0` only on trusted networks
- No authentication is provided in current version

## 💡 Tips & Tricks

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Esc` | Close code viewer |
| `F5` | Refresh page |

### Browser Compatibility

Tested and supported on:
- Chrome 100+
- Firefox 100+
- Safari 15+
- Edge 100+

### Performance

- Handles up to 100+ iterations per task
- Supports multiple concurrent tasks
- Chart rendering optimized for 50+ data points

## 🆕 Coming Soon

Future enhancements planned:

- [ ] Real-time WebSocket updates
- [ ] Dark mode toggle
- [ ] Export results to PDF/JSON
- [ ] Diff view between iterations
- [ ] Search and filter functionality
- [ ] Custom dashboard layouts
- [ ] Integration with CI/CD pipelines

## 📞 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify your workspace structure matches expected format
3. Check server logs for error messages
4. Report issues on GitHub with:
   - Command used to start visualizer
   - Browser and OS information
   - Error messages from console

---

**Ready to visualize?** Start your first task and launch the visualizer! 🚀

```bash
./run_general.sh 01_todo_list --background
python agents/general_agent/visualizer/visualizer.py --workspace ./output-todo-list
open http://localhost:8080
```
