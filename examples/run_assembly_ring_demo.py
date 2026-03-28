import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Union

import numpy as np
import numpy.typing as npt
from gymnasium.wrappers.rendering import RecordVideo

from mujaco_gym_lite.environment_tools.mujoco_env.builders.assembly_ring import (
    AssemblyRingRobotEnvBuilder,
    AssemblyRingRobotEnvBuilderConfig,
)
from mujaco_gym_lite.logger import logger, setup_logger
from mujaco_gym_lite.utils.files import write_json
from mujaco_gym_lite.utils.transforms import matrix_to_quat, slerp_quaternion
from mujaco_gym_lite.utils.views import view_to_rotation


def _get_num_points(num_points: Union[int, list[int]], index: int) -> int:
    return num_points[index] if isinstance(num_points, list) else num_points


def _plan_linear_path(
    way_points: list[npt.NDArray], num_points_between_way_points: Union[int, list[int]]
) -> npt.NDArray:
    segments = []
    n_segments = len(way_points) - 1
    for i, (start, end) in enumerate(zip(way_points[:-1], way_points[1:])):
        # Each linear path point should have 3 elements.
        assert len(start) == 3, f"Expected 3 elements, got {len(start)}"
        assert len(end) == 3, f"Expected 3 elements, got {len(end)}"

        n_points = _get_num_points(num_points_between_way_points, i)
        # For the last segment, include the endpoint to ensure continuity.
        include_endpoint = i == n_segments - 1
        segment = np.linspace(start, end, n_points, endpoint=include_endpoint)
        segments.append(segment)
    return np.concatenate(segments)


def _plan_rotation_path(
    way_points: list[npt.NDArray], num_points_between_way_points: Union[int, list[int]]
) -> npt.NDArray:
    segments = []
    n_segments = len(way_points) - 1
    for i, (start, end) in enumerate(zip(way_points[:-1], way_points[1:])):
        # Each quaternion is expected to have 4 elements.
        assert len(start) == 4, f"Expected 4 elements, got {len(start)}"
        assert len(end) == 4, f"Expected 4 elements, got {len(end)}"

        n_points = _get_num_points(num_points_between_way_points, i)
        # For the last segment, use spherical linear interpolation (slerp).
        if i == n_segments - 1:
            segment = slerp_quaternion(start, end, n_points)
        else:
            segment = np.linspace(start, end, n_points)  # type: ignore
        segments.append(segment)
    return np.concatenate(segments)


def generate_path() -> tuple[npt.NDArray, npt.NDArray]:
    # Define constants.
    ring_handle_x = 0.275
    ring_handle_y = -0.1
    peg_rod_x = 0.125
    peg_rod_y = 0.0

    # Initial configuration.
    init_position = np.array([ring_handle_x, ring_handle_y, 0.08], dtype=np.float32)
    init_rotation = np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float32)

    middle3_position = np.array([ring_handle_x, ring_handle_y, 0.3], dtype=np.float32)
    middle3_rotation = init_rotation

    stay3_position = np.array([peg_rod_x, peg_rod_y, 0.3], dtype=np.float32)
    stay3_rotation = init_rotation

    middle4_position = np.array([peg_rod_x, peg_rod_y, 0.075], dtype=np.float32)
    middle4_rotation = init_rotation

    stay4_position = np.array([peg_rod_x, peg_rod_y, 0.075], dtype=np.float32)
    stay4_rotation = init_rotation

    # Group configurations into lists.
    positions = [
        init_position,
        middle3_position,
        stay3_position,
        middle4_position,
        stay4_position,
    ]
    rotations = [
        init_rotation,
        middle3_rotation,
        stay3_rotation,
        middle4_rotation,
        stay4_rotation,
    ]
    # Define the number of points for each segment.
    segment_points = [80, 80, 80, 80]

    pos_path = _plan_linear_path(positions, segment_points)
    rot_path = _plan_rotation_path(rotations, segment_points)

    return pos_path, rot_path


def run_env(output_dir_path: pathlib.Path, render_mode: str, record: bool):
    output_dir_path.mkdir(parents=True, exist_ok=True)
    config = AssemblyRingRobotEnvBuilderConfig(
        home_arm_qpos=np.array([-1.09862707, 4.04584791, 1.52457248, -0.18611493, 0.72032405, -1.28474199]),
        render_mode=render_mode,
        window_render_type_and_name=("mocap", "mocap_camera"),
        add_end_effector_marker=True,
        end_effector_marker_name="endeffector_marker",
        add_mocap_coordinate_marker=False,
        mocap_coordinate_name="mocap_coordinate_marker",
        enable_all_finger_control=True,
        obstacle_positions=None,
        add_task_area_marker=False,
        initial_lift=False,
    )
    builder = AssemblyRingRobotEnvBuilder(config)
    env = builder.build_env(output_dir_path=output_dir_path)

    if record:
        assert render_mode == "rgb_array"
        env = RecordVideo(
            env, str(output_dir_path / "video"), episode_trigger=lambda x: True, video_length=500, fps=100
        )

    initial_condition = {}
    look_at_point = np.array([0.0, 0.0, 0.2])
    initial_camera_position = np.array([look_at_point[0] + 0.5, look_at_point[1] - 0.5, look_at_point[2] + 0.3])
    initial_camera_rotation = matrix_to_quat(view_to_rotation(look_at_point - initial_camera_position))
    initial_condition[f"camera/{config.mocap_camera_names[0]}/position"] = initial_camera_position
    initial_condition[f"camera/{config.mocap_camera_names[0]}/rotation"] = initial_camera_rotation

    env.reset(options=initial_condition)
    robot_position_path, robot_rotation_path = generate_path()
    assert len(robot_position_path) == len(robot_rotation_path)

    t = 0

    target_state = defaultdict(list)
    while True:
        action = {}
        t = min(t, len(robot_position_path) - 1)
        action["robot/end_effector/position"] = robot_position_path[t]
        action["robot/end_effector/rotation"] = robot_rotation_path[t]
        action["task/end_episode"] = np.zeros(1, dtype=np.float32)
        action["robot/home"] = np.zeros(1, dtype=np.float32)
        t += 1
        for mocap_camera_name in config.mocap_camera_names:
            # Set camera actions
            action[f"camera/{mocap_camera_name}/position"] = initial_camera_position
            action[f"camera/{mocap_camera_name}/rotation"] = initial_camera_rotation

        next_state, reward, termination, truncated, info = env.step(action)
        target_state["robot/end_effector/position"].append(next_state["robot/end_effector/position"].tolist())
        target_state["assembly_ring/ring/position"].append(next_state["assembly_ring/ring/position"].tolist())
        target_state["assembly_ring/handle/position"].append(next_state["assembly_ring/handle/position"].tolist())
        logger.info(f"Reward info: Reward {reward}")
        env.render()

        if info["task/success"]:
            logger.info("Task Success!")

        if t >= len(robot_position_path) - 1:
            break

        if termination or truncated:
            break

    write_json(file_path=output_dir_path / "demo_trajectory.json", data=target_state)
    env.close()


def run(args):
    setup_logger([sys.stderr], ["DEBUG"])
    run_env(pathlib.Path(args.output_dir_path), render_mode=args.render_mode, record=args.record)


def main():
    parser = argparse.ArgumentParser()
    result_dir = pathlib.Path(__file__).parent / "demo_result" / "assembly_ring"
    parser.add_argument("--output_dir_path", type=str, default=str(result_dir))
    parser.add_argument("--render_mode", default="human", type=str)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
