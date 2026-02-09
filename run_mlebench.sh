#!/bin/bash

set -e
set -o pipefail

# =============================================================================
# MLE-Bench task management script
# =============================================================================
# Function: Initialize, prepare, run, and stop MLE-Bench competition tasks
#
# Usage:
#   ./run_mlebench.sh init
#   ./run_mlebench.sh prepare <competition_id>
#   ./run_mlebench.sh run <competition_id> [--background] [other Python parameters]
#   ./run_mlebench.sh stop <competition_id>
# =============================================================================

# --- GLOBAL PATHS ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR="$SCRIPT_DIR/output"
MLEBENCH_DIR="$OUTPUT_DIR/mlebench"
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

# Recursively get all child processes (compatible with Linux and macOS)
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

# Check if the environment exists
check_env_exists() {
    if ! mamba env list | grep -q "${ENV_NAME}"; then
        error "Environment '$ENV_NAME' not found. Please run '$0 init' first."
    fi
}

# --- Command Implementations ---

do_init() {
    info "Initializing MLE-Bench environment..."

    # --- Check if GPU is available ---
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

    # --- Create or update the conda environment ---
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

    # --- Install mlebench ---
    info "Installing mlebench library..."
    local mlebench_repo_path="$SCRIPT_DIR/mle-bench"

    # Clone the repository (if it does not exist)
    if [ ! -d "$mlebench_repo_path" ]; then
        info "Cloning mle-bench repository..."
        git clone --depth 1 --single-branch --branch main https://github.com/openai/mle-bench "$mlebench_repo_path"
    else
        info "Repository already exists at '$mlebench_repo_path'"
    fi

    # Pull LFS files
    info "Fetching Git LFS files..."
    cd "$mlebench_repo_path"
    git lfs fetch --all
    git lfs pull

    local pyproject_file="$mlebench_repo_path/pyproject.toml"
    if [ -f "$pyproject_file" ]; then
        info "Modifying TensorFlow version requirement in pyproject.toml..."
        # macOS use sed -i.bak，Linux use sed -i
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i.bak 's/tensorflow>=2\.16/tensorflow>=2.15/' "$pyproject_file"
        else
            sed -i 's/tensorflow>=2\.16/tensorflow>=2.15/' "$pyproject_file"
        fi
        success "TensorFlow version updated to >=2.15"
    else
        warning "pyproject.toml not found, skipping version modification"
    fi

    info "Installing mlebench in environment '$ENV_NAME'..."
    python -u -m pip install --no-build-isolation -e "$mlebench_repo_path"

    cd "$SCRIPT_DIR"

    success "Initialization complete!"
    echo ""
    echo "=================================================================="
    info "To activate the environment, run:"
    echo "    mamba activate $ENV_NAME"
    echo "=================================================================="
}

do_prepare() {
    local competition_id=$1
    if [ -z "$competition_id" ]; then
        error "Competition ID is required for the 'prepare' command."
    fi

    # Check if the environment exists
    check_env_exists

    local competition_dir="$MLEBENCH_DIR/$competition_id"
    info "Preparing competition: $competition_id"
    info "Data will be downloaded to: $competition_dir"

    # Activate environment and run mlebench prepare
    activate_env

    info "Running mlebench prepare in environment '$ENV_NAME'..."
    mlebench prepare \
        --competition-id "$competition_id" \
        --data-dir "$MLEBENCH_DIR"

    success "Preparation complete for '$competition_id'."
}

do_run() {
    local competition_id=$1
    if [ -z "$competition_id" ]; then
        error "Competition ID is required for the 'run' command."
    fi

    # Check if the environment exists
    check_env_exists

    # --- Path define ---
    local competition_dir="$MLEBENCH_DIR/$competition_id"
    local task_data_path="$competition_dir/prepared/public"
    local pid_file="$competition_dir/.agent.pid"
    local log_file="$competition_dir/agent.log"
    local task_file="$task_data_path/description.md"

    local mlebench_examples="$SCRIPT_DIR/agents/ml_agent/examples/mlebench"
    local config_template="$mlebench_examples/task_config.yaml"
    local eval_program="$mlebench_examples/eval_program.py"
    local evolve_script="$SCRIPT_DIR/agents/ml_agent/ml_evolve_agent.py"
    local agent_config_path="$competition_dir/task_config.yaml"

    # --- Path vertification ---
    [ -d "$task_data_path" ] || error "Task data path '$task_data_path' not found. Run 'prepare' first."
    [ -f "$task_file" ] || error "Task file '$task_file' not found. Run 'prepare' first."
    [ -f "$config_template" ] || error "Config template '$config_template' not found."
    [ -f "$eval_program" ] || error "Eval program '$eval_program' not found."
    [ -f "$evolve_script" ] || error "Evolve script '$evolve_script' not found."

    # Check if it is already running
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            error "Agent is already running with PID $old_pid. Use 'stop' first."
        else
            warning "Found stale PID file. Removing it."
            rm "$pid_file"
        fi
    fi

    # --- Prepare the configuration file  ---
    if [ -f "$agent_config_path" ]; then
        info "Using existing configuration: $agent_config_path"
    else
        info "Copying configuration template to '$agent_config_path'..."
        cp "$config_template" "$agent_config_path"
    fi

    # --- Parameter parsing ---
    local run_in_background=false
    local python_args=()

    for arg in "${@:2}"; do
        if [ "$arg" == "--background" ]; then
            run_in_background=true
        else
            python_args+=("$arg")
        fi
    done

    # --- Activate environment (this also sets LD_LIBRARY_PATH) ---
    activate_env

    # --- Set PYTHONPATH ---
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR:$SCRIPT_DIR/src"

    # --- Construct the command array ---
    local command_array=(
        "python3" "-u" "$evolve_script"
        "--config" "$agent_config_path"
        "--task-data-path" "$task_data_path"
        "--task-file" "$task_file"
        "--eval-file" "$eval_program"
    )

    if [ ${#python_args[@]} -gt 0 ]; then
        command_array+=("${python_args[@]}")
    fi

    # --- Print execution information---
    echo "=================================================================="
    info "Starting ML-Evolve Agent for: $competition_id"
    echo "🔧 Environment: $ENV_NAME (activated)"
    echo "📁 Task Data: $task_data_path"
    echo "📝 Config: $agent_config_path"
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
        info "Stop agent: $0 stop $competition_id"
    else
        info "Starting agent in foreground mode..."
        "${command_array[@]}"
        success "Agent execution completed."
    fi
}

do_stop() {
    local competition_id=$1
    if [ -z "$competition_id" ]; then
        error "Competition ID is required for the 'stop' command."
    fi

    local competition_dir="$MLEBENCH_DIR/$competition_id"
    local pid_file="$competition_dir/.agent.pid"

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

    # Terminate child processes gracefully
    if [ ${#desc_array[@]} -gt 0 ]; then
        info "Terminating child processes (SIGTERM): ${desc_array[*]}"
        kill "${desc_array[@]}" 2>/dev/null || true
    fi

    # Terminate the parent process
    info "Terminating parent process (SIGTERM): $pid"
    kill "$pid" 2>/dev/null || true

    # Wait for processes to exit
    sleep 10

    # Check if forced termination is required
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

    # Clean up any residual ml_evolve_agent.py processes
    pkill -f "agents/ml_agent/ml_evolve_agent.py" 2>/dev/null || true

    success "Cleanup complete."
}

# --- Main Script Logic ---

if [ $# -lt 1 ]; then
    echo "Usage: $0 {init|prepare|run|stop} [competition_id] [options]"
    echo ""
    echo "Commands:"
    echo "  init                        Initialize MLE-Bench environment"
    echo "  prepare <competition_id>    Download and prepare competition data"
    echo "  run <competition_id> [opts] Run agent (use --background for background mode)"
    echo "  stop <competition_id>       Stop running agent"
    echo ""
    echo "Environment: $ENV_NAME"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 prepare spooky-author-identification"
    echo "  $0 run spooky-author-identification --background"
    echo "  $0 stop spooky-author-identification"
    exit 1
fi

COMMAND=$1
COMPETITION_ID=${2:-}

# Create output directory
mkdir -p "$MLEBENCH_DIR"

case "$COMMAND" in
    init)
        do_init
        ;;
    prepare)
        do_prepare "$COMPETITION_ID"
        ;;
    run)
        do_run "$COMPETITION_ID" "${@:3}"
        ;;
    stop)
        do_stop "$COMPETITION_ID"
        ;;
    *)
        error "Unknown command: '$COMMAND'. Use {init|prepare|run|stop}"
        ;;
esac
