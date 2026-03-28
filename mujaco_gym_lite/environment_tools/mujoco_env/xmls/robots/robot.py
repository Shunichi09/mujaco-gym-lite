import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.camera import add_camera
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.robots.kinova_jaco2.arm import add_j2n6s300
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.robots.kinova_jaco2.gripper import add_jaco2_hand_3finger
from mujaco_gym_lite.utils.transforms import euler_to_quat


def add_robot_xml(
    generator: MJCFGenerator,
    robot_name: str,
    asset_dir_path: pathlib.Path,
    robot_position: npt.NDArray,
    robot_rotation: npt.NDArray,
    with_ego_camera: bool = False,
    with_ego_camera_mesh: bool = False,
    ego_camera_name: Optional[str] = None,
    add_end_effector_marker: bool = False,
    end_effector_marker_name: Optional[str] = None,
):
    if "j2n6s300" in robot_name:
        actor_arm_end_body = add_j2n6s300(
            generator,
            robot_name,
            robot_asset_dir=asset_dir_path / "robots" / "kinova_jaco2" / "assets",
            robot_position=robot_position,
            robot_rotation=robot_rotation,  # np.array([0.0, 0.0, 0.0, 1.0]),
        )
        gripper_end_effector_body, _ = add_jaco2_hand_3finger(
            generator,
            robot_name,
            robot_asset_dir=asset_dir_path / "robots" / "kinova_jaco2" / "assets",
            attach_body=actor_arm_end_body,
            add_end_effector_marker=add_end_effector_marker,
            end_effector_marker_name=end_effector_marker_name,
        )
        if with_ego_camera:
            assert ego_camera_name is not None
            add_camera(
                generator,
                camera_name=f"{ego_camera_name}",
                camera_position=np.array([0.0, 0.1, -0.175]),
                camera_rotation=euler_to_quat([float(np.pi), float(np.pi) * 1.1], "zx"),
                attach_body=gripper_end_effector_body,
                mesh_file_path=(
                    asset_dir_path / "objects" / "realsense" / "d435" / "model.obj" if with_ego_camera_mesh else None
                ),
                texture_file_path=(
                    asset_dir_path / "textures" / "poligon" / "ConcretePoured001_COL_2K_METALNESS.png"
                    if with_ego_camera_mesh
                    else None
                ),
            )
    else:
        raise ValueError(f"Invalid robot_name: {robot_name}")
