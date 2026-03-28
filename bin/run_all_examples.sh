#!/usr/bin/env bash
set -euo pipefail

# Run a lightweight smoke check for representative MuJoCo examples.
# Each script is executed with offscreen rendering to work in CI.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export MUJOCO_GL=${MUJOCO_GL:-osmesa}

examples=(
	"examples/run_drawer_open_demo.py"
	"examples/run_button_push_demo.py"
	"examples/run_sliding_door_open.py"
	"examples/run_assembly_ring_demo.py"
	"examples/run_hinged_box_open_demo.py"
	"examples/run_lever_pull_demo.py"
	"examples/run_lidded_box_reach_demo.py"
	"examples/run_soccer_demo.py"
)

for example in "${examples[@]}"; do
	echo "[CI] Running: ${example}"
	timeout 10s python "$example" --render_mode rgb_array
done

echo "[CI] All example smoke tests completed."
