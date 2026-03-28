import pathlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robots.jaco2 import J2n6s300
from mujaco_gym_lite.environment_tools.mujoco_env.functions.camera import opencv_camera_to_mujoco_camera
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.camera import add_camera, add_mocap_camera
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.robots.robot import add_robot_xml
from mujaco_gym_lite.environments.mujoco_env.mujoco_env import MujocoEnv


def _default_asset_dir_path() -> pathlib.Path:
    # Allow explicit override when assets are installed outside the source tree.
    env_asset_dir = os.environ.get("MUJACO_GYM_LITE_ASSETS_DIR")
    if env_asset_dir:
        return pathlib.Path(env_asset_dir).expanduser()

    candidate_paths = (
        # Source tree layout: <repo_root>/assets
        pathlib.Path(__file__).resolve().parents[4] / "assets",
        # Installed package layout (future-proof): <site-packages>/mujaco_gym_lite/assets
        pathlib.Path(__file__).resolve().parents[3] / "assets",
        # setuptools data-files fallback: <sys.prefix>/mujaco_gym_lite/assets
        pathlib.Path(sys.prefix) / "mujaco_gym_lite" / "assets",
        pathlib.Path.cwd() / "assets",
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    # Keep previous behavior of failing fast in __post_init__ when missing.
    return candidate_paths[0]


@dataclass
class EnvBuilderConfig:
    asset_dir_path: pathlib.Path = field(default_factory=_default_asset_dir_path)
    crop_scene_min_pos: npt.NDArray = np.array([-0.5, -0.5, 0.01])
    crop_scene_max_pos: npt.NDArray = np.array([0.5, 0.5, 0.5])
    mocap_camera_names: tuple[str, ...] = ("mocap_camera",)
    fixed_camera_names: tuple[str, ...] = ()
    ego_camera_names: tuple[str, ...] = ()
    fixed_camera_poses: tuple[tuple[npt.NDArray, npt.NDArray], ...] = ()
    add_camera_mesh: bool = False
    camera_fovy: int = 69

    def __post_init__(self):
        assert pathlib.Path(self.asset_dir_path).exists(), (
            "Asset directory not found. Set MUJACO_GYM_LITE_ASSETS_DIR or pass asset_dir_path explicitly. "
            f"Current value: {self.asset_dir_path}"
        )
        assert np.all(self.crop_scene_max_pos >= self.crop_scene_min_pos)
        assert len(self.fixed_camera_names) == len(self.fixed_camera_poses)
        assert self.camera_fovy > 0.0


class EnvBuilder:
    def __init__(self, config: EnvBuilderConfig):
        self._config = config

    def build_env(self, output_dir_path: pathlib.Path) -> MujocoEnv:
        raise NotImplementedError

    def build_xml(self, xml_file_path: pathlib.Path):
        raise NotImplementedError

    def build_env_model(
        self,
    ) -> dict[str, Union[EnvModel, dict[str, Union[dict[str, Camera], dict[str, MocapCamera]]]]]:
        raise NotImplementedError

    def build_env_from_xml(
        self, xml_file_path: pathlib.Path, output_dir_path: pathlib.Path, copy_xml_file: bool
    ) -> MujocoEnv:
        raise NotImplementedError

    def _build_camera_model(self) -> tuple[dict[str, MocapCamera], dict[str, Camera], dict[str, Camera]]:
        mocap_cameras: dict[str, MocapCamera] = {}
        for i, name in enumerate(self._config.mocap_camera_names):
            mocap_cameras[name] = MocapCamera(f"{name}_body", [], [], name, name, i)

        fixed_cameras: dict[str, Camera] = {}
        for i, name in enumerate(self._config.fixed_camera_names):
            fixed_cameras[name] = Camera(f"{name}_body", [], [], name, name)

        ego_cameras: dict[str, Camera] = {}
        for i, name in enumerate(self._config.ego_camera_names):
            ego_cameras[name] = Camera(f"{name}_body", [], [], name, name)

        return mocap_cameras, fixed_cameras, ego_cameras

    def _build_camera_xml(self, generator: MJCFGenerator):
        # TODO: Support other type camer mesh
        camera_mesh_file_path = (
            self._config.asset_dir_path / "objects" / "realsense" / "d435" / "model.obj"
            if self._config.add_camera_mesh
            else None
        )
        camera_texture_file_path = (
            self._config.asset_dir_path / "textures" / "poligon" / "ConcretePoured001_COL_2K_METALNESS.png"
            if self._config.add_camera_mesh
            else None
        )

        # mocap camera
        for i, name in enumerate(self._config.mocap_camera_names):
            add_mocap_camera(
                generator,
                name,
                # NOTE: +i only debugging reason.
                camera_position=np.array([0.0, -0.2 + i * 0.1, 2.0]),
                camera_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                mesh_file_path=camera_mesh_file_path,
                texture_file_path=camera_texture_file_path,
                fovy=self._config.camera_fovy,
            )

        # fixed camera
        for (pos, rot), name in zip(self._config.fixed_camera_poses, self._config.fixed_camera_names):
            add_camera(
                generator,
                name,
                camera_position=pos,
                camera_rotation=opencv_camera_to_mujoco_camera(rot),
                mesh_file_path=camera_mesh_file_path,
                texture_file_path=camera_texture_file_path,
                fovy=self._config.camera_fovy,
            )


@dataclass
class RobotEnvBuilderConfig(EnvBuilderConfig):
    # robot settings
    robot_name: str = "j2n6s300"
    robot_position: npt.NDArray = np.array([0.0, -0.5, 0.0])
    robot_rotation: npt.NDArray = np.array([0.0, 0.0, 0.0, 1.0])
    target_initial_end_effector_position: npt.NDArray = np.array([0.0, -0.3, 0.4])
    target_initial_end_effector_rotation: npt.NDArray = np.array([0.0, 0.7071067811865475, 0.7071067811865475, 0.0])
    home_arm_qpos: npt.NDArray = np.array([-1.52721949, 3.58571133, 1.96776086, -0.60883485, 1.85972993, -0.55478329])
    home_gripper_qpos: npt.NDArray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add_end_effector_marker: bool = False
    end_effector_marker_name: Optional[str] = "j2n6s300_marker"
    add_mocap_coordinate_marker: bool = False
    mocap_coordinate_name: Optional[str] = None
    enable_all_finger_control: bool = False


class RobotEnvBuilder(EnvBuilder):
    _config: RobotEnvBuilderConfig

    def __init__(self, config: RobotEnvBuilderConfig):
        super().__init__(config)

    def _build_robot_xml(self, generator: MJCFGenerator):
        add_robot_xml(
            generator=generator,
            asset_dir_path=self._config.asset_dir_path,
            robot_name=self._config.robot_name,
            robot_position=self._config.robot_position,
            robot_rotation=self._config.robot_rotation,
            with_ego_camera=False,
            add_end_effector_marker=self._config.add_end_effector_marker,
            end_effector_marker_name=self._config.end_effector_marker_name,
        )

    def _build_robot_model(self):
        if "j2n6s300" in self._config.robot_name:
            base_body_name = f"{self._config.robot_name}_link0_body"
            arm_joint_names = [
                f"{self._config.robot_name}_link1_joint",
                f"{self._config.robot_name}_link2_joint",
                f"{self._config.robot_name}_link3_joint",
                f"{self._config.robot_name}_link4_joint",
                f"{self._config.robot_name}_link5_joint",
                f"{self._config.robot_name}_link6_joint",
            ]
            arm_body_names = [
                f"{self._config.robot_name}_link0_body",
                f"{self._config.robot_name}_link1_body",
                f"{self._config.robot_name}_link2_body",
                f"{self._config.robot_name}_link3_body",
                f"{self._config.robot_name}_link4_body",
                f"{self._config.robot_name}_link5_body",
                f"{self._config.robot_name}_link6_body",
            ]
            geom_root_name = f"{self._config.robot_name}"
            endeffector_body_name = f"{self._config.robot_name}_end_effector_body"

            robot = J2n6s300(
                base_body_name=base_body_name,
                arm_joint_names=arm_joint_names,
                arm_body_names=arm_body_names,
                gripper_joint_names=[
                    f"{self._config.robot_name}_finger1_proximal_link_joint",
                    f"{self._config.robot_name}_finger1_distal_link_joint",
                    f"{self._config.robot_name}_finger2_proximal_link_joint",
                    f"{self._config.robot_name}_finger2_distal_link_joint",
                    f"{self._config.robot_name}_finger3_proximal_link_joint",
                    f"{self._config.robot_name}_finger3_distal_link_joint",
                ],
                gripper_body_names=[
                    f"{self._config.robot_name}_finger1_proximal_link_body",
                    f"{self._config.robot_name}_finger1_distal_link_body",
                    f"{self._config.robot_name}_finger2_proximal_link_body",
                    f"{self._config.robot_name}_finger2_distal_link_body",
                    f"{self._config.robot_name}_finger3_proximal_link_body",
                    f"{self._config.robot_name}_finger3_distal_link_body",
                ],
                geom_root_name=geom_root_name,
                gripper_finger_tip_geom_names={
                    "finger1": [
                        f"{self._config.robot_name}_finger1_proximal_link_collision_geom_0",
                        f"{self._config.robot_name}_finger1_proximal_link_collision_geom_1",
                        f"{self._config.robot_name}_finger1_proximal_link_collision_geom_2",
                        f"{self._config.robot_name}_finger1_distal_link_collision_geom_0",
                    ],
                    "finger2": [
                        f"{self._config.robot_name}_finger2_proximal_link_collision_geom_0",
                        f"{self._config.robot_name}_finger2_proximal_link_collision_geom_1",
                        f"{self._config.robot_name}_finger2_proximal_link_collision_geom_2",
                        f"{self._config.robot_name}_finger2_distal_link_collision_geom_0",
                    ],
                    "finger3": [
                        f"{self._config.robot_name}_finger3_proximal_link_collision_geom_0",
                        f"{self._config.robot_name}_finger3_proximal_link_collision_geom_1",
                        f"{self._config.robot_name}_finger3_proximal_link_collision_geom_2",
                        f"{self._config.robot_name}_finger3_distal_link_collision_geom_0",
                    ],
                },
                end_effector_body_name=endeffector_body_name,
                home_gripper_qpos=self._config.home_gripper_qpos,
                home_joint_qpos=self._config.home_arm_qpos,
                apply_distal_finger_control=self._config.enable_all_finger_control,
            )
        else:
            raise NotImplementedError
        return robot
