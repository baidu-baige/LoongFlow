#!/bin/bash

set -e
set -o pipefail

# =============================================================================
# Machine Learning Task Management Script
# =============================================================================
# Functions: Initialize, run and stop machine learning tasks
#
# Usage:
#   ./run_ml.sh init
#   ./run_ml.sh run <task_name> [--background] [other Python args]
#   ./run_ml.sh stop <task_name>
# =============================================================================

# --- GLOBAL PATHS ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_TASK_BASE="$SCRIPT_DIR/agents/ml_agent/examples"
ENV_NAME="loongflow_ml"

# --- Helper Functions ---
info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

# Initialize conda/mamba for shell usage
init_conda() {
    # Find conda/mamba installation
    local conda_base=""

    if [ -n "${CONDA_EXE:-}" ]; then
        conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [ -n "${MAMBA_EXE:-}" ]; then
        conda_base="$(dirname "$(dirname "$MAMBA_EXE")")"
    elif command -v mamba >/dev/null 2>&1; then
        conda_base="$(mamba info --base 2>/dev/null || conda info --base 2>/dev/null)"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null)"
    else
        error "Neither conda nor mamba found. Please install mambaforge or miniconda first."
    fi

    # Source conda.sh to enable activate command
    if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
        source "$conda_base/etc/profile.d/conda.sh"
    else
        error "Cannot find conda.sh at $conda_base/etc/profile.d/conda.sh"
    fi

    # Also source mamba.sh if available
    if [ -f "$conda_base/etc/profile.d/mamba.sh" ]; then
        source "$conda_base/etc/profile.d/mamba.sh"
    fi

    eval "$(conda shell.posix hook)"
}

# Activate the target environment and set up LD_LIBRARY_PATH
activate_env() {
    init_conda

    info "Activating environment: $ENV_NAME"
    mamba activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
        error "Failed to activate environment '$ENV_NAME'"
    fi

    # Set LD_LIBRARY_PATH to include conda environment's lib directory
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        info "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
    else
        warning "CONDA_PREFIX not set, LD_LIBRARY_PATH may not be configured correctly"
    fi
}

# Recursively get all child processes
get_descendants() {
    local parent="$1"
    local children

    if command -v pgrep >/dev/null 2>&1; then
        children=$(pgrep -P "$parent" 2>/dev/null || true)
    else
        children=$(ps -o pid= --ppid "$parent" 2>/dev/null | awk '{print $1}' || true)
    fi

    for c in $children; do
        echo "$c"
        get_descendants "$c"
    done
}

# Check if environment exists
check_env_exists() {
    if ! mamba env list | grep -q "${ENV_NAME} "; then
        error "Environment '$ENV_NAME' not found. Please run '$0 init' first."
    fi
}

# --- Command Implementations ---

do_init() {
    info "Initializing ML environment..."

    # --- Detect GPU availability ---
    local env_file
    local pip_file
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        info "Detected NVIDIA GPU, using GPU environment"
        env_file="$SCRIPT_DIR/agents/ml_agent/examples/environment_gpu.yaml"
        pip_file="$SCRIPT_DIR/agents/ml_agent/examples/requirements_gpu.txt"
    else
        info "No GPU detected, using CPU environment"
        env_file="$SCRIPT_DIR/agents/ml_agent/examples/environment_cpu.yaml"
        pip_file="$SCRIPT_DIR/agents/ml_agent/examples/requirements_cpu.txt"
    fi

    # --- Verify environment file exists ---
    if [ ! -f "$env_file" ]; then
        error "Environment file not found: $env_file"
    fi

    # --- Check if mamba is available ---
    if ! command -v mamba >/dev/null 2>&1; then
        error "mamba is not installed. Please install mambaforge or mamba first."
    fi

    info "Target environment name: $ENV_NAME"

    # --- Create/update conda environment ---
    if mamba env list | grep -q "${ENV_NAME} "; then
        warning "Environment '$ENV_NAME' already exists. skip"
    else
        info "Creating new environment from $env_file..."
        mamba env create -n "$ENV_NAME" -f "$env_file"
    fi

    success "Conda environment '$ENV_NAME' is ready."

    # Activate environment and install packages
    activate_env

    info "Installing common requirements in environment '$ENV_NAME'..."
    python -u -m pip install --no-build-isolation -r "$pip_file"

    cd "$SCRIPT_DIR"

    success "Initialization complete!"
    echo ""
    echo "=================================================================="
    info "To activate the environment, run:"
    echo "    mamba activate $ENV_NAME"
    echo "=================================================================="
}

do_run() {
    # --- Parse arguments ---
    local run_in_background=false
    local task_name=""
    local python_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --background)
                run_in_background=true
                shift
                ;;
            *)
                if [ -z "$task_name" ]; then
                    task_name="$1"
                else
                    python_args+=("$1")
                fi
                shift
                ;;
        esac
    done

    # --- Check if task_name is provided ---
    if [ -z "$task_name" ]; then
        error "task_name is required. Usage: $0 run <task_name> [--background] [other args]"
    fi

    # --- Determine task directory ---
    local task_dir="$DEFAULT_TASK_BASE/$task_name"

    # Verify task directory exists
    if [ ! -d "$task_dir" ]; then
        error "Task directory not found: $task_dir"
    fi

    # Check if environment exists
    check_env_exists

    # --- Path definitions ---
    local task_config="$task_dir/task_config.yaml"
    local description_file="$task_dir/public/description.md"
    local eval_program="$task_dir/eval_program.py"
    local task_data_path="$task_dir/public"
    local pid_file="$task_dir/.agent.pid"
    local log_file="$task_dir/agent.log"
    local evolve_script="$SCRIPT_DIR/agents/ml_agent/ml_evolve_agent.py"

    # --- Verify required files exist ---
    if [ ! -f "$task_config" ]; then
        error "Task config not found: $task_config"
    fi
    if [ ! -f "$description_file" ]; then
        error "Description file not found: $description_file"
    fi
    if [ ! -f "$eval_program" ]; then
        error "Eval program not found: $eval_program"
    fi
    if [ ! -f "$evolve_script" ]; then
        error "Evolve script not found: $evolve_script"
    fi

    # Check if already running
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            error "Agent is already running with PID $old_pid. Use 'stop' first."
        else
            warning "Found stale PID file. Removing it."
            rm "$pid_file"
        fi
    fi

    # --- Activate environment (this also sets LD_LIBRARY_PATH) ---
    activate_env

    # --- Set PYTHONPATH ---
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR:$SCRIPT_DIR/src"

    # --- Build command array ---
    local command_array=(
        "python3" "-u" "$evolve_script"
        "--config" "$task_config"
        "--task-data-path" "$task_data_path"
        "--task-file" "$description_file"
        "--eval-file" "$eval_program"
    )

    if [ ${#python_args[@]} -gt 0 ]; then
        command_array+=("${python_args[@]}")
    fi

    # --- Print execution info ---
    echo "=================================================================="
    info "Starting ML-Evolve Agent"
    echo "📋 Task Name: $task_name"
    echo "🔧 Environment: $ENV_NAME (activated)"
    echo "📁 Task Directory: $task_dir"
    echo "📁 Task Data: $task_data_path"
    echo "📝 Config: $task_config"
    echo "📄 Description: $description_file"
    echo "🔧 Evaluator: $eval_program"
    echo "🚀 Command: ${command_array[*]}"
    echo "=================================================================="

    # --- Execute ---
    if [ "$run_in_background" = true ]; then
        info "Starting agent in background mode..."
        nohup "${command_array[@]}" > "$log_file" 2>&1 &
        local pid=$!
        echo "$pid" > "$pid_file"

        success "Agent started in background with PID: $pid"
        info "Log file: $log_file"
        info "Monitor logs: tail -f $log_file"
        info "Stop agent: $0 stop $task_name"
    else
        info "Starting agent in foreground mode..."
        "${command_array[@]}"
        success "Agent execution completed."
    fi
}

do_stop() {
    # --- Parse arguments ---
    local task_name="$1"

    if [ -z "$task_name" ]; then
        error "task_name is required. Usage: $0 stop <task_name>"
    fi

    # --- Determine task directory ---
    local task_dir="$DEFAULT_TASK_BASE/$task_name"
    local pid_file="$task_dir/.agent.pid"

    if [ ! -f "$pid_file" ]; then
        warning "PID file not found: $pid_file"
        warning "Agent may not be running or was already stopped."
        # Perform global cleanup
        do_global_cleanup
        exit 0
    fi

    local pid=$(cat "$pid_file")
    info "Stopping agent with PID: $pid..."

    if ! ps -p "$pid" > /dev/null 2>&1; then
        warning "Process $pid is not running."
        rm -f "$pid_file"
        do_global_cleanup
        exit 0
    fi

    # Get all child processes
    local descendants=$(get_descendants "$pid")
    local desc_array=()
    if [ -n "$descendants" ]; then
        read -r -a desc_array <<< "$descendants"
    fi

    # Gracefully terminate child processes
    if [ ${#desc_array[@]} -gt 0 ]; then
        info "Terminating child processes (SIGTERM): ${desc_array[*]}"
        kill "${desc_array[@]}" 2>/dev/null || true
    fi

    # Terminate parent process
    info "Terminating parent process (SIGTERM): $pid"
    kill "$pid" 2>/dev/null || true

    # Wait for processes to exit
    sleep 10

    # Check if force kill is needed
    local to_force=()
    if ps -p "$pid" > /dev/null 2>&1; then
        to_force+=("$pid")
    fi
    for p in "${desc_array[@]}"; do
        if [ -n "$p" ] && ps -p "$p" > /dev/null 2>&1; then
            to_force+=("$p")
        fi
    done

    if [ ${#to_force[@]} -gt 0 ]; then
        warning "Force killing remaining processes (SIGKILL): ${to_force[*]}"
        kill -9 "${to_force[@]}" 2>/dev/null || true
    fi

    success "Agent and all child processes stopped."
    rm -f "$pid_file"

    # Perform global cleanup
    do_global_cleanup
}

do_global_cleanup() {
    info "Performing global cleanup..."

    # Clean up potentially remaining ml_evolve_agent.py processes
    pkill -f "agents/ml_agent/ml_evolve_agent.py" 2>/dev/null || true

    success "Cleanup complete."
}

# --- Main Script Logic ---

if [ $# -lt 1 ]; then
    echo "Usage: $0 {init|run|stop} [task_name] [options]"
    echo ""
    echo "Commands:"
    echo "  init                               Initialize ML environment"
    echo "  run <task_name> [opts]             Run agent for specified task"
    echo "  stop <task_name>                   Stop running agent"
    echo ""
    echo "Run Options:"
    echo "  --background                       Run in background mode"
    echo "  [other args]                       Pass through to Python script"
    echo ""
    echo "Task Directory:"
    echo "  Tasks are located in: $DEFAULT_TASK_BASE/<task_name>"
    echo ""
    echo "Environment: $ENV_NAME"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 run ml_example --background"
    echo "  $0 run ml_example --some-python-arg value"
    echo "  $0 stop ml_example"
    exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
    init)
        do_init
        ;;
    run)
        do_run "$@"
        ;;
    stop)
        do_stop "$@"
        ;;
    *)
        error "Unknown command: '$COMMAND'. Use {init|run|stop}"
        ;;
esac
