#!/bin/bash
# Launch mini-SWE-agent training on this machine (8x H100, Ubuntu 22.04).
# See LAUNCH.md for full details.
set -e

cd "$(dirname "$0")/../../.."  # ~/SkyRL

echo "[launch] Sourcing environment..."
source setup_env.sh

echo "[launch] Checking Ray..."
if ! ray status &>/dev/null; then
    echo "[launch] Starting Ray head node..."
    ray start --head --disable-usage-stats
else
    echo "[launch] Ray already running."
fi

echo "[launch] Checking Docker Hub login..."
if ! podman login --get-login docker.io &>/dev/null; then
    echo "[launch] WARNING: Not logged in to Docker Hub. You may hit rate limits."
    echo "         Run: podman login docker.io -u <user> -p <password>"
fi

echo "[launch] Verifying cached dataset files..."
python3 examples/train/mini_swe_agent/rebuild_cached_datasets.py --verify-only

echo "[launch] Starting training (logging to ~/training.log)..."
bash examples/train/mini_swe_agent/run_mini_swe_8B.sh 2>&1 | tee ~/training.log
