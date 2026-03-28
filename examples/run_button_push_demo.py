import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Union

import numpy as np
import numpy.typing as npt
from gymnasium.wrappers.rendering import RecordVideo

from mujaco_gym_lite.environment_tools.mujoco_env.builders.button import (
    ButtonRobotEnvBuilder,
    ButtonRobotEnvBuilderConfig,
)
from mujaco_gym_lite.logger import logger, setup_logger
from mujaco_gym_lite.utils.files import write_json
from mujaco_gym_lite.utils.transforms import combine_quat, euler_to_quat, matrix_to_quat, slerp_quaternion
from mujaco_gym_lite.utils.views import view_to_rotation


def _get_num_points(num_points: Union[int, list[int]], index: int) -> int:
    return num_points[index] if isinstance(num_points, list) else num_points


def _plan_linear_path(
    way_points: list[npt.NDArray], num_points_between_way_points: Union[int, list[int]]
) -> npt.NDArray:
    path_segments = []
    n_segments = len(way_points) - 1

    for i, (start, end) in enumerate(zip(way_points[:-1], way_points[1:])):
        # Ensure that each point has 3 elements
        assert len(start) == 3, f"Expected 3 elements, got {len(start)}"
        assert len(end) == 3, f"Expected 3 elements, got {len(end)}"

        n_points = _get_num_points(num_points_between_way_points, i)
        # For the last segment, include the endpoint; otherwise, exclude it.
        endpoint = i == n_segments - 1
        segment = np.linspace(start, end, n_points, endpoint=endpoint)
        path_segments.append(segment)

    return np.concatenate(path_segments)


def _plan_rotation_path(
    way_points: list[npt.NDArray], num_points_between_way_points: Union[int, list[int]]
) -> npt.NDArray:
    path_segments = []
    n_segments = len(way_points) - 1

    for i, (start, end) in enumerate(zip(way_points[:-1], way_points[1:])):
        # Ensure that each quaternion has 4 elements
        assert len(start) == 4, f"Expected 4 elements, got {len(start)}"
        assert len(end) == 4, f"Expected 4 elements, got {len(end)}"

        n_points = _get_num_points(num_points_between_way_points, i)
        # For the last segment, use spherical linear interpolation (slerp)
        if i == n_segments - 1:
            segment = slerp_quaternion(start, end, n_points)
        else:
            segment = np.linspace(start, end, n_points)  # type: ignore
        path_segments.append(segment)

    return np.concatenate(path_segments)


def generate_path() -> tuple[npt.NDArray, npt.NDArray]:
    button_x = 0.1

    init_position = np.array([0.3, 0.0, 0.45], dtype=np.float32)
    init_rotation = combine_quat(
        np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float32),
        euler_to_quat([np.pi * 0.5, np.pi * 0.5], order="xz"),
    )

    middle_position = np.array([button_x, 0.15, 0.25], dtype=np.float32)
    middle_rotation = init_rotation

    stay_position = np.array([button_x, 0.3, 0.25], dtype=np.float32)
    stay_rotation = init_rotation

    end_position = np.array([button_x, 0.3, 0.25], dtype=np.float32)
    end_rotation = init_rotation

    position = _plan_linear_path(
        [init_position, middle_position, stay_position, end_position], num_points_between_way_points=[60, 60, 60]
    )
    rotation = _plan_rotation_path(
        [init_rotation, middle_rotation, stay_rotation, end_rotation], num_points_between_way_points=[60, 60, 60]
    )
    return position, rotation


def run_env(output_dir_path: pathlib.Path, render_mode: str, record: bool):
    output_dir_path.mkdir(parents=True, exist_ok=True)
    config = ButtonRobotEnvBuilderConfig(
        home_arm_qpos=np.array([-0.64840576, 3.61841829, 1.55457843, -2.23992233, 1.18533837, -1.30244344]),
        render_mode=render_mode,
        add_mocap_coordinate_marker=False,
        add_end_effector_marker=True,
        end_effector_marker_name="endeffector_marker",
        mocap_coordinate_name="mocap_coordiante",
        window_render_type_and_name=("mocap", "mocap_camera"),
        add_task_area_marker=False,
        initial_position_noise=True,
        obstacle_position=None,
    )
    builder = ButtonRobotEnvBuilder(config)
    env = builder.build_env(output_dir_path=output_dir_path)

    if record:
        assert render_mode == "rgb_array"
        env = RecordVideo(
            env, str(output_dir_path / "video"), episode_trigger=lambda x: True, video_length=500, fps=100
        )

    initial_condition = {}
    look_at_point = np.array([0.0, 0.0, 0.3])
    initial_camera_position = np.array([look_at_point[0] + 0.75, look_at_point[1] - 0.25, look_at_point[2] + 0.3])
    initial_camera_rotation = matrix_to_quat(view_to_rotation(look_at_point - initial_camera_position))
    initial_condition[f"camera/{config.mocap_camera_names[0]}/position"] = initial_camera_position
    initial_condition[f"camera/{config.mocap_camera_names[0]}/rotation"] = initial_camera_rotation

    state, _ = env.reset(options=initial_condition)
    robot_position_path, robot_rotation_path = generate_path()
    assert len(robot_position_path) == len(robot_rotation_path)
    t = 0
    total_return = 0.0
    target_state = defaultdict(list)
    while True:
        action = {}
        t = min(t, len(robot_position_path) - 1)
        action["robot/end_effector/position"] = robot_position_path[t]
        action["robot/end_effector/rotation"] = robot_rotation_path[t]
        action["task/end_episode"] = np.zeros(1, dtype=np.float32)
        t += 1
        action["robot/home"] = np.zeros(1, dtype=np.float32)

        for mocap_camera_name in config.mocap_camera_names:
            # Set camera actions
            action[f"camera/{mocap_camera_name}/position"] = initial_camera_position
            action[f"camera/{mocap_camera_name}/rotation"] = initial_camera_rotation

        next_state, reward, terminal, truncated, info = env.step(action)
        logger.info(f"Reward info: Reward {reward} ")

        target_state["robot/end_effector/position"].append(next_state["robot/end_effector/position"].tolist())
        target_state["button/surface_center_position"].append(next_state["button/surface_center_position"].tolist())

        if bool(next_state["robot/touch_button"]):
            logger.info(f"{next_state['robot/touch_button']}")
        if info["task/success"]:
            logger.info("Task Success!")
            logger.info(f"Drawer handle position: {next_state['button/surface_center_position']}")
            # NOTE: Follow metaworld settings, we dont done even when a robot achieves the task

        total_return += reward
        env.render()

        if t >= len(robot_position_path) - 1:
            break

        if terminal or truncated:
            break

    write_json(file_path=output_dir_path / "demo_trajectory.json", data=target_state)
    logger.info(f"Total Return] {total_return}")
    env.close()


def run(args):
    setup_logger([sys.stderr], ["DEBUG"])
    run_env(pathlib.Path(args.output_dir_path), render_mode=args.render_mode, record=args.record)


def main():
    parser = argparse.ArgumentParser()
    result_dir = pathlib.Path(__file__).parent / "demo_result" / "button"
    parser.add_argument("--output_dir_path", type=str, default=str(result_dir))
    parser.add_argument("--render_mode", default="human", type=str)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
