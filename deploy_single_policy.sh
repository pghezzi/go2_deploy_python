#!/usr/bin/env bash
# Launch fixed-policy debugging in MuJoCo or on the robot.
# Usage: ./deploy_single_policy.sh <interface> [config_name] [--policy-index N] [--model PATH]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <interface> [config_name] [--policy-index N] [--model PATH]" >&2
    echo "Example: $0 enp3s0 single_policy.yaml --policy-index 2" >&2
    exit 2
fi

interface="$1"
shift
config_name="single_policy.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    config_name="$1"
    shift
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

# Apply before either Python process imports NumPy or PyTorch. The robot's
# OpenBLAS pools retain four workers even after torch.set_num_threads(1).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [[ ! -d "/sys/class/net/$interface" ]]; then
    echo "Network interface '$interface' does not exist." >&2
    exit 2
fi

if [[ ! -f "$script_dir/configs/$config_name" ]]; then
    echo "Missing configuration: $script_dir/configs/$config_name" >&2
    exit 2
fi

depth_pid=""
controller_pid=""
depth_log="/tmp/single_policy_depth_publisher.log"

cleanup() {
    trap - EXIT INT TERM
    [[ -n "$controller_pid" ]] && kill -TERM "$controller_pid" 2>/dev/null || true
    [[ -n "$depth_pid" ]] && kill -TERM "$depth_pid" 2>/dev/null || true
    [[ -n "$controller_pid" ]] && wait "$controller_pid" 2>/dev/null || true
    [[ -n "$depth_pid" ]] && wait "$depth_pid" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$script_dir"

if [[ "$interface" != "lo" ]]; then
    echo "Starting RealSense depth publisher (logging to $depth_log)..."
    "$python_bin" -u rough_depth_image.py --interface "$interface" --config "configs/$config_name" >"$depth_log" 2>&1 &
    depth_pid=$!
else
    echo "Using depth published by MuJoCo; start the simulator separately."
fi

echo "Starting single-policy controller on $interface with $config_name..."
"$python_bin" -u deploy.py --interface "$interface" --config "$config_name" --type single_policy "$@" &
controller_pid=$!

# Stop the companion process if either the controller or camera exits.
if [[ -n "$depth_pid" ]]; then
    exit_status=0
    wait -n "$controller_pid" "$depth_pid" || exit_status=$?
    if ! kill -0 "$depth_pid" 2>/dev/null; then
        echo "Depth publisher exited; stopping the controller. Camera log: $depth_log" >&2
        tail -n 80 "$depth_log" >&2 || true
    else
        echo "Controller exited; stopping the depth publisher." >&2
    fi
    echo "Child process exit status: $exit_status" >&2
    exit "$exit_status"
else
    wait "$controller_pid"
fi
