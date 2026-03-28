#!/usr/bin/env bash
set -euo pipefail

# Run a lightweight smoke check for representative MuJoCo tasks.
# Instead of running full scripted demos, build each env and execute a few random steps.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export MUJOCO_GL=${MUJOCO_GL:-osmesa}

python - <<'PY'
import pathlib
import tempfile

TASKS = [
	(
		"Drawer",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.drawer",
		"DrawerRobotEnvBuilder",
		"DrawerRobotEnvBuilderConfig",
	),
	(
		"Button",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.button",
		"ButtonRobotEnvBuilder",
		"ButtonRobotEnvBuilderConfig",
	),
	(
		"SlidingDoor",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.door",
		"SlidingDoorRobotEnvBuilder",
		"SlidingDoorRobotEnvBuilderConfig",
	),
	(
		"AssemblyRing",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.assembly_ring",
		"AssemblyRingRobotEnvBuilder",
		"AssemblyRingRobotEnvBuilderConfig",
	),
	(
		"HingedBox",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.hinged_box",
		"HingedBoxRobotEnvBuilder",
		"HingedBoxRobotEnvBuilderConfig",
	),
	(
		"Lever",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.lever",
		"LeverRobotEnvBuilder",
		"LeverRobotEnvBuilderConfig",
	),
	(
		"LiddedBox",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.lidded_box",
		"LiddedBoxRobotEnvBuilder",
		"LiddedBoxRobotEnvBuilderConfig",
	),
	(
		"Soccer",
		"mujaco_gym_lite.environment_tools.mujoco_env.builders.soccer",
		"SoccerRobotEnvBuilder",
		"SoccerRobotEnvBuilderConfig",
	),
]

with tempfile.TemporaryDirectory(prefix="mgl_ci_smoke_") as tmp:
	base_dir = pathlib.Path(tmp)

	for task_name, module_name, builder_name, config_name in TASKS:
		print(f"[CI] Smoke: {task_name}")
		module = __import__(module_name, fromlist=[builder_name, config_name])
		builder_cls = getattr(module, builder_name)
		config_cls = getattr(module, config_name)

		config = config_cls(render_mode="rgb_array")
		builder = builder_cls(config)
		output_dir = base_dir / task_name.lower()
		output_dir.mkdir(parents=True, exist_ok=True)
		env = builder.build_env(output_dir_path=output_dir)

		try:
			env.reset()
			for _ in range(3):
				action = env.action_space.sample()
				_, _, terminated, truncated, _ = env.step(action)
				if terminated or truncated:
					env.reset()
		finally:
			env.close()

print("[CI] All task smoke tests completed.")
PY
