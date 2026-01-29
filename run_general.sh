#!/bin/bash

# Function: Automatically execute and manage general_agent agents based on the task directory name.
#
# Usage:
# 1. Run task in foreground (blocks terminal):
#    ./run_general.sh <task_directory_name> [python_script_options]
#    Example: ./run_general.sh hello_world --log-level DEBUG --max-iterations 100
#
# 2. Run task in background (does not block terminal):
#    Example: ./run_general.sh hello_world --background --log-level DEBUG --max-iterations 100
#
# 3. Stop a background task:
#    ./run_general.sh stop <task_directory_name>
#    Example: ./run_general.sh stop hello_world

# --- 1. Core Functions ---

# Set task-related variables
setup_task_vars() {
  TASK_NAME="$1"

  BASE_PATH="./agents/general_agent/examples"
  MAIN_SCRIPT="agents/general_agent/general_evolve_agent.py"

  TASK_DIR="${BASE_PATH}/${TASK_NAME}"
  EVAL_FILE="${TASK_DIR}/eval_program.py"
  CONFIG_FILE="${TASK_DIR}/task_config.yaml"

  # Hidden PID file to store background process ID
  PID_FILE="${TASK_DIR}/.run.pid"
}

# Get all descendant PIDs recursively
get_descendants() {
  local parent="$1"
  local children

  # Prefer pgrep -P; fallback to ps --ppid
  if command -v pgrep >/dev/null 2>&1; then
    children=$(pgrep -P "$parent")
  else
    children=$(ps -o pid= --ppid "$parent" 2>/dev/null | awk '{print $1}')
  fi

  for c in $children; do
    echo "$c"
    get_descendants "$c"
  done
}

# --- 2. Main Logic: Stop or Start ---

if [ -z "$1" ]; then
  echo "❌ Error: No arguments provided."
  echo "   Usage: $0 <task_directory_name> [options] | stop <task_directory_name>"
  exit 1
fi

# Stop Logic
if [ "$1" == "stop" ]; then
  if [ -z "$2" ]; then
    echo "❌ Error: Task directory name required for stop command."
    echo "   Usage: $0 stop <task_directory_name>"
    exit 1
  fi

  setup_task_vars "$2"

  if [ ! -f "$PID_FILE" ]; then
    echo "🟡 Warning: PID file not found: ${PID_FILE}"
    echo "   Task '${TASK_NAME}' might not be running or was stopped manually."
    exit 0
  fi

  PID=$(cat "$PID_FILE")

  if ps -p "$PID" > /dev/null; then
    echo "⏹️ Stopping task '${TASK_NAME}' (PID: ${PID})..."

    # Get all descendant processes
    DESC_LIST_STR="$(get_descendants "$PID")"
    read -r -a DESC_LIST <<< "$DESC_LIST_STR"

    # Terminate children first
    if [ "${#DESC_LIST[@]}" -gt 0 ]; then
      echo "   Terminating children (SIGTERM): ${DESC_LIST[*]}"
      kill "${DESC_LIST[@]}" 2>/dev/null
    fi

    # Terminate parent
    kill "$PID" 2>/dev/null

    sleep 10

    # Force kill remaining processes if any
    TO_FORCE=()
    if ps -p "$PID" > /dev/null; then
      TO_FORCE+=("$PID")
    fi
    for p in "${DESC_LIST[@]}"; do
      if [ -n "$p" ] && ps -p "$p" > /dev/null; then
        TO_FORCE+=("$p")
      fi
    done

    if [ "${#TO_FORCE[@]}" -gt 0 ]; then
      echo "   Force killing remaining processes (SIGKILL): ${TO_FORCE[*]}"
      kill -9 "${TO_FORCE[@]}" 2>/dev/null
    fi

    echo "✅ Task and its descendants have stopped."
  else
    echo "🟡 Warning: Process (PID: ${PID}) in PID file no longer exists."
  fi

  echo "🧹 Performing global cleanup..."

  # Safety net: cleanup all general_evolve_agent.py processes
  echo "   -> pkill -f \"agents/general_agent/general_evolve_agent.py\""
  pkill -f "agents/general_agent/general_evolve_agent.py" 2>/dev/null

  rm -f "$PID_FILE"
  exit 0
fi

# --- 3. Start Logic ---

setup_task_vars "$1"

# Check paths
if [ ! -d "$TASK_DIR" ]; then
  echo "❌ Error: Task directory not found: ${TASK_DIR}"
  exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Error: Required file not found: ${CONFIG_FILE}"
  exit 1
fi

# --- 4. Build and Execute Command ---

PYTHON_ARGS=()
RUN_IN_BACKGROUND=false

# Parse arguments starting from the second one
for arg in "${@:2}"; do
  if [ "$arg" == "--background" ]; then
    RUN_IN_BACKGROUND=true
  else
    PYTHON_ARGS+=("$arg")
  fi
done

# Construct the command array (safer than eval)
COMMAND_ARRAY=(
  "python" "${MAIN_SCRIPT}"
  "--config" "${CONFIG_FILE}"
  "${PYTHON_ARGS[@]}"
)

echo "✅ Checks passed. Preparing to run task: ${TASK_NAME}"
echo "------------------------------------------------------------------"
echo "🚀 Command to execute:"
echo "PYTHONPATH=$PYTHONPATH:./src ${COMMAND_ARRAY[@]}"
echo "------------------------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:./src

if [ "$RUN_IN_BACKGROUND" = true ]; then
  echo "🏃 Starting task in background..."
  # Use nohup to keep running after terminal closes
  nohup "${COMMAND_ARRAY[@]}" > "${TASK_DIR}/run.log" 2>&1 &

  PID=$!
  echo "$PID" > "$PID_FILE"

  echo "✅ Task started in background. PID: ${PID}"
  echo "   Log file: ${TASK_DIR}/run.log"
  echo "   To stop, run: $0 stop ${TASK_NAME}"
else
  # Foreground execution
  "${COMMAND_ARRAY[@]}"
  echo "✅ Task finished."
fi