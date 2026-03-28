import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Union

import numpy as np
import numpy.typing as npt
from gymnasium.wrappers.rendering import RecordVideo

from mujaco_gym_lite.environment_tools.mujoco_env.builders.soccer import (
    SoccerRobotEnvBuilder,
    SoccerRobotEnvBuilderConfig,
)
from mujaco_gym_lite.logger import logger, setup_logger
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


def generate_path(robot_position: npt.NDArray, ball_position: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    ball_x = ball_position[0]
    ball_y = ball_position[1]
    ball_z = 0.05

    init_position = robot_position
    init_rotation = combine_quat(
        np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float32), euler_to_quat(np.pi, order="z")
    )

    middle_position = np.array([ball_x, ball_y - 0.125, ball_z], dtype=np.float32)
    middle_rotation = combine_quat(init_rotation, euler_to_quat(np.pi * 0.0, order="z"))

    stay1_position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
    stay1_rotation = combine_quat(init_rotation, euler_to_quat(np.pi * 0.0, order="z"))

    stay2_position = np.array([ball_x, ball_y + 0.1, ball_z], dtype=np.float32)
    stay2_rotation = combine_quat(init_rotation, euler_to_quat(np.pi * 0.0, order="z"))

    end_position = np.array([ball_x, ball_y + 0.25, ball_z], dtype=np.float32)
    end_rotation = combine_quat(init_rotation, euler_to_quat(np.pi * 0.0, order="z"))

    position = _plan_linear_path(
        [init_position, middle_position, stay1_position, stay2_position, end_position],
        num_points_between_way_points=[50, 50, 50, 50],
    )
    rotation = _plan_rotation_path(
        [init_rotation, middle_rotation, stay1_rotation, stay2_rotation, end_rotation],
        num_points_between_way_points=[50, 50, 50, 50],
    )
    return position, rotation


def run_env(output_dir_path: pathlib.Path, render_mode: str, record: bool):
    output_dir_path.mkdir(parents=True, exist_ok=True)
    config = SoccerRobotEnvBuilderConfig(
        home_arm_qpos=np.array([-2.47129327, 3.43295249, 0.76351186, -0.13933894, 0.54693593, 0.53096044]),
        render_mode=render_mode,
        window_render_type_and_name=("mocap", "mocap_camera"),
        add_end_effector_marker=True,
        end_effector_marker_name="endeffector_marker",
        add_mocap_coordinate_marker=False,
        mocap_coordinate_name="mocap_coordinate_marker",
        add_task_area_marker=False,
    )
    builder = SoccerRobotEnvBuilder(config)
    env = builder.build_env(output_dir_path=output_dir_path)

    if record:
        assert render_mode == "rgb_array"
        env = RecordVideo(
            env, str(output_dir_path / "video"), episode_trigger=lambda x: True, video_length=500, fps=100
        )

    initial_condition = {}
    look_at_point = np.array([0.0, 0.0, 0.3])
    initial_camera_position = np.array([look_at_point[0] + 0.5, look_at_point[1] - 0.25, look_at_point[2]])
    initial_camera_rotation = matrix_to_quat(view_to_rotation(look_at_point - initial_camera_position))
    initial_condition[f"camera/{config.mocap_camera_names[0]}/position"] = initial_camera_position
    initial_condition[f"camera/{config.mocap_camera_names[0]}/rotation"] = initial_camera_rotation

    state, _ = env.reset(options=initial_condition)
    robot_position_path, robot_rotation_path = generate_path(
        state["robot/end_effector/position"], state["soccer_ball/position"]
    )
    assert len(robot_position_path) == len(robot_rotation_path)
    t = 0
    total_return = 0.0
    prev_reward = 0.0
    target_state = defaultdict(list)

    state_history = []
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
        print(reward, state["soccer_ball/position"], next_state["soccer_ball/position"])
        if prev_reward > reward:
            logger.info("####################################")
            logger.info(f"prev {prev_reward} / current {reward}")

        prev_reward = reward

        target_state["robot/end_effector/position"].append(next_state["robot/end_effector/position"].tolist())
        state_history.append(next_state["robot/end_effector/position"].tolist())

        if info["task/success"]:
            logger.info("Task Success!")
            # NOTE: Follow metaworld settings, we dont done even when a robot achieves the task

        total_return += reward
        env.render()

        if t >= len(robot_position_path) - 1:
            break

        if terminal or truncated:
            break

    logger.info(f"Total Return] {total_return}")
    logger.info(f"Max robot position: {np.max(np.array(state_history), axis=0)}")
    logger.info(f"Min robot position: {np.min(np.array(state_history), axis=0)}")
    env.close()


def run(args):
    setup_logger([sys.stderr], ["DEBUG"])
    run_env(pathlib.Path(args.output_dir_path), render_mode=args.render_mode, record=args.record)


def main():
    parser = argparse.ArgumentParser()
    result_dir = pathlib.Path(__file__).parent / "demo_result" / "soccer"
    parser.add_argument("--output_dir_path", type=str, default=str(result_dir))
    parser.add_argument("--render_mode", default="human", type=str)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
