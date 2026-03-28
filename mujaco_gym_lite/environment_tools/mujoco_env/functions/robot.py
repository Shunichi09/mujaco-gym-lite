from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robot import Robot
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robots.jaco2 import J2n6s300
from mujaco_gym_lite.logger import logger
from mujaco_gym_lite.utils.transforms import create_transformation_matrix, extract_position, extract_rotation


def robot_observation(
    robot: Robot,
    target_initial_end_effector_rotation: npt.NDArray,
    end_effector_position_min: Optional[npt.NDArray],
    end_effector_position_max: Optional[npt.NDArray],
) -> tuple[dict[str, npt.NDArray], dict[str, Any]]:
    endeffector_pose = robot.end_effector_pose()
    joint_positions = np.array(robot.arm_joint_qpos()).flatten()
    gripper_joint_positions = np.array(robot.gripper_joint_qpos()).flatten()

    contact_positions, contact_names = (
        robot.contact_gripper_positions()
    )  # TODO: Do we have to remove table? exclude_abstract_geom_names=["table"]
    # change format for later use
    contact = np.array([float(len(c)) for c in contact_positions.values()], dtype=np.float32)
    contact_positions = {f"robot/contact_positions/{key}": value for key, value in contact_positions.items()}
    contact_names = {f"robot/contact_names/{key}": value for key, value in contact_names.items()}

    if isinstance(robot, J2n6s300):
        assert len(contact) == 3
    else:
        raise ValueError

    robot_observation = {
        "robot/base/position": extract_position(robot.base_body_pose()).astype(np.float32),
        "robot/base/rotation": extract_rotation(robot.base_body_pose()).astype(np.float32),
        "robot/end_effector/position": extract_position(endeffector_pose).astype(np.float32),
        "robot/end_effector/rotation": extract_rotation(endeffector_pose).astype(np.float32),
        "robot/target_initial_end_effector_rotation": target_initial_end_effector_rotation.astype(np.float32),
        "robot/contact": contact,  # Which finger is the contact on
        "robot/joint_angles": joint_positions.astype(np.float32),
        "robot/gripper_joint_angles": gripper_joint_positions.astype(np.float32),
    }

    if end_effector_position_max is not None and end_effector_position_min is not None:
        pos = robot_observation["robot/end_effector/position"]
        orig_pos = pos.copy()

        clipped_pos = np.clip(
            orig_pos, a_min=end_effector_position_min, a_max=end_effector_position_max, dtype=np.float32
        )
        if not np.array_equal(orig_pos, clipped_pos):
            logger.debug(f"End-effector position clipped:\n  before={orig_pos}\n  after ={clipped_pos}")

        robot_observation["robot/end_effector/position"] = clipped_pos

    robot_observation_info = {}
    robot_observation_info.update(contact_positions)
    robot_observation_info.update(contact_names)  # type: ignore

    return robot_observation, robot_observation_info


def robot_action(
    robot: Robot,
    ee_pose_action: Optional[npt.NDArray],
    home: bool,
    open: bool = False,
    close: bool = False,
    gripper_command: Optional[npt.NDArray] = None,
    joint_angle: Optional[npt.NDArray] = None,
    mj_ctrl_start_index: int = 0,
):
    if joint_angle is None:
        assert ee_pose_action is not None and len(ee_pose_action) == 7
        target_matrix = create_transformation_matrix(ee_pose_action[:3], ee_pose_action[3:], rotation_type="quaternion")
        target_joint_angle = robot.solve_ik(target_matrix)
    else:
        assert ee_pose_action is None
        target_joint_angle = joint_angle

    if target_joint_angle is not None:
        robot.apply_arm_control(target_joint_angle, mj_ctrl_start_index=mj_ctrl_start_index)

    if close:
        assert not open
        assert gripper_command is None
        robot.gripper_close()

    if open:
        assert not close
        assert gripper_command is None
        robot.gripper_open()

    if gripper_command is not None:
        assert (not open) and (not close)
        robot.apply_gripper_control(gripper_command)

    if home:
        robot.appply_home_arm_qpos()
        robot.appply_home_gripper_qpos()
