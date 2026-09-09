#!/usr/bin/env bash
# Launch the real-camera publisher and DreamWaQ low-level controller together.
# Usage: ./deploy_depthwaq.sh <robot_network_interface> [config_name]

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <robot_network_interface> [config_name]" >&2
    echo "Example: $0 enp3s0 depthwaq.yaml" >&2
    exit 2
fi

interface="$1"
config_name="${2:-depthwaq.yaml}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

# Apply before either Python process imports NumPy or PyTorch. The robot's
# OpenBLAS pools retain four workers even after torch.set_num_threads(1).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [[ "$interface" == "lo" ]]; then
    echo "Refusing to deploy to loopback. Provide the robot Ethernet interface." >&2
    exit 2
fi

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
depth_log="/tmp/depthwaq_depth_publisher.log"

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

echo "Starting RealSense depth publisher (logging to $depth_log)..."
"$python_bin" -u rough_depth_image.py --interface "$interface" --config "configs/$config_name" >"$depth_log" 2>&1 &
depth_pid=$!

echo "Starting DepthWaQ controller on $interface with $config_name..."
"$python_bin" deploy.py --interface "$interface" --config "$config_name" --type depthwaq &
controller_pid=$!

wait "$controller_pid"
