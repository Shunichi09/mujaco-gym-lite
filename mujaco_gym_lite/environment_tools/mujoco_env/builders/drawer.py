import pathlib
import shutil
from dataclasses import asdict, dataclass
from typing import Optional, Union, cast

import gymnasium
import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.builders.builder import RobotEnvBuilder, RobotEnvBuilderConfig
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.drawer import Drawer
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mocap_object import MocapObject
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.metaworld_objects.drawer import add_drawer
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.asset import add_asset_model
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.geom import add_mujoco_principal_geom
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.marker import (
    add_mocap_coordinate_marker,
    add_workspace_marker,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.scene.floor_scene import add_basic_scene_setting
from mujaco_gym_lite.environments.mujoco_env.mujoco_env import MujocoEnv
from mujaco_gym_lite.utils.files import write_json, write_txt
from mujaco_gym_lite.utils.transforms import euler_to_quat

import mujaco_gym_lite.environments  # noqa


@dataclass
class DrawerRobotEnvBuilderConfig(RobotEnvBuilderConfig):
    # model settings
    drawer_name: str = "drawer"
    drawer_joint_name: str = "drawer_joint"
    drawer_body_name: str = "drawer_body"
    drawer_joint_range: tuple[float, float] = (0.0, 0.155)
    drawer_position: npt.NDArray = np.array([0.1, 0.125, 0.05])
    drawer_rotation: npt.NDArray = euler_to_quat(0.0, order="z")
    drawer_handle_name: str = "drawer_handle"
    drawer_handle_site_name: str = "drawer_handle_site"
    drawer_core_name: str = "drawer_core"
    drawer_case_name: str = "drawer_case"
    obstacle_position: Optional[tuple[float, float, float]] = (-0.2, 0.125, 0.3)
    drawer_scale: npt.NDArray = np.array([1.0, 1.0, 1.0])

    # robot
    home_arm_qpos: npt.NDArray = np.array([-1.61779153, 3.81199834, 2.64718577, -1.06878415, 2.60586587, -2.68657568])
    home_gripper_qpos: npt.NDArray = np.array([0.75, 0.0, 0.75, 0.0, 0.75, 0.0])
    robot_hand_geom_name: str = "j2n6s300_hand"  # TODO: add robot name
    robot_finger_geom_name: str = "j2n6s300_finger"  # TODO: add robot name
    end_effector_min_position: npt.NDArray = np.array([-0.35, -0.35, 0.05])
    end_effector_max_position: npt.NDArray = np.array([0.35, 0.1, 0.45])

    # task area marker
    add_task_area_marker: bool = False

    # episode settings
    max_episode_steps: Optional[int] = None

    # render settings
    # NOTE: None means using mujoco free camera.
    # example (mocap, mocap_camera)
    window_render_type_and_name: Optional[tuple[str, str]] = None
    render_segmentation: bool = False
    render_depth: bool = False
    render_color: bool = False
    frame_skip: int = 10
    render_mode: str = "human"
    width: int = 1280
    height: int = 720
    default_camera_config: Optional[dict] = None
    use_fixed_camera_in_human_mode: bool = False

    initial_position_noise: bool = False


class DrawerRobotEnvBuilder(RobotEnvBuilder):
    _config: DrawerRobotEnvBuilderConfig

    def __init__(self, config: DrawerRobotEnvBuilderConfig):
        super().__init__(config)

    def build_env(self, output_dir_path: pathlib.Path) -> MujocoEnv:
        config_file_path = output_dir_path / "env.json"
        write_json(config_file_path, asdict(self._config))

        xml_file_path = output_dir_path / "env.xml"
        self.build_xml(xml_file_path)

        env_models = self.build_env_model()
        camera = cast(dict[str, dict[str, EnvModel]], env_models["camera"])

        return self._build_env(xml_file_path, camera, env_models)  # type: ignore

    def build_env_from_xml(
        self, xml_file_path: pathlib.Path, output_dir_path: pathlib.Path, copy_xml_file: bool
    ) -> MujocoEnv:
        config_file_path = output_dir_path / "env.json"
        write_json(config_file_path, asdict(self._config))

        if copy_xml_file:
            shutil.copy(xml_file_path, output_dir_path / "env.xml")

        env_models = self.build_env_model()
        camera = cast(dict[str, dict[str, EnvModel]], env_models["camera"])

        return self._build_env(xml_file_path, camera, env_models)  # type: ignore

    def build_xml(self, xml_file_path: pathlib.Path):
        generator = MJCFGenerator()

        # camera
        self._build_camera_xml(generator)
        # robot
        self._build_robot_xml(generator)
        # objects
        self._build_objects_xml(generator)
        # basics
        self._build_scene_xml(generator)

        xml_str = generator.generate()
        write_txt(xml_file_path, xml_str)

    def build_env_model(
        self,
    ) -> dict[str, Union[EnvModel, dict[str, Union[dict[str, Camera], dict[str, MocapCamera]]]]]:
        # drawer
        drawer = Drawer(
            self._config.drawer_body_name,
            [self._config.drawer_joint_name],
            [self._config.drawer_body_name],
            self._config.drawer_name,
            drawer_joint_range=self._config.drawer_joint_range,
            drawer_handle_geom_name=self._config.drawer_handle_name,
            drawer_handle_site_name=self._config.drawer_handle_site_name,
            open_and_close_threshold=0.02,
        )

        # camera
        mocap_cameras, fixed_cameras, ego_cameras = self._build_camera_model()

        # robot
        robot = self._build_robot_model()

        # coordinate marker
        if self._config.add_mocap_coordinate_marker:
            coordinate_marker = MocapObject(
                base_body_name=self._config.mocap_coordinate_name + "_body",
                joint_names=[],
                body_names=[],
                geom_root_name="",
                mocap_idx=len(mocap_cameras),
            )
        else:
            coordinate_marker = None

        return {
            "drawer": drawer,
            "robot": robot,
            "camera": {"mocap": mocap_cameras, "fixed": fixed_cameras, "ego": ego_cameras},
            "coordinate_marker": coordinate_marker,
        }

    def _build_scene_xml(self, generator: MJCFGenerator):
        # light and skyboxes
        add_basic_scene_setting(generator, self._config.asset_dir_path)

        # table
        add_asset_model(
            generator,
            self._config.asset_dir_path / "objects" / "basemesh_edited" / "coffee_table_a",
            "table",
            texture_file_path=self._config.asset_dir_path
            / "textures"
            / "poligon"
            / "VeneerWhiteOakRandomMatched001_COL_2K_METALNESS.png",
            model_position=np.array([0.0, 0.0, -1.0]),
            model_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            model_scale=np.array([3.0, 2.0, 1.0]),
        )

    def _build_objects_xml(self, generator: MJCFGenerator):
        add_drawer(
            generator=generator,
            drawer_name=self._config.drawer_name,
            asset_dir_path=self._config.asset_dir_path,
            drawer_position=self._config.drawer_position,
            drawer_rotation=self._config.drawer_rotation,
            drawer_scale=self._config.drawer_scale,
            drawer_handle_site_name=self._config.drawer_handle_site_name,
            drawer_handle_name=self._config.drawer_handle_name,
            drawer_joint_name=self._config.drawer_joint_name,
            drawer_body_name=self._config.drawer_body_name,
            drawer_case_name=self._config.drawer_case_name,
        )

        # obstacle
        if self._config.obstacle_position is not None:
            add_mujoco_principal_geom(
                generator=generator,
                model_name="obstacle",
                model_position=np.array(self._config.obstacle_position),
                model_rotation=np.array([0.70710678, 0.0, -0.70710678, 0.0]),
                model_size=np.array([0.25, 0.2, 0.02]),
                model_type_name="box",
                with_free_joint=False,
                texture_file_path=self._config.asset_dir_path / "textures" / "metaworld" / "navy_blue.png",
            )

        # task area marker
        if self._config.add_task_area_marker:
            add_workspace_marker(
                generator,
                "task_area_marker_1",
                marker_positions=[np.array([0.27, 0.21, 0.225])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )
            add_workspace_marker(
                generator,
                "task_area_marker_2",
                marker_positions=[np.array([-0.075, 0.21, 0.225])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )
            add_workspace_marker(
                generator,
                "task_area_marker_3",
                marker_positions=[np.array([-0.075, 0.04, 0.225])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )
            add_workspace_marker(
                generator,
                "task_area_marker_4",
                marker_positions=[np.array([0.27, 0.04, 0.225])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )
            add_workspace_marker(
                generator,
                "task_area_marker_5",
                marker_positions=[np.array([0.2, -0.21, 0.15])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )
            add_workspace_marker(
                generator,
                "task_area_marker_6",
                marker_positions=[np.array([0.0, -0.21, 0.15])],
                rgbas=[np.array([0.0, 0.5, 0.0, 1.0]) for _ in range(1)],
                size=np.array([0.015, 0.015, 0.015]),
                attach_body="worldbody",
            )

        if self._config.add_mocap_coordinate_marker:
            assert self._config.mocap_coordinate_name is not None
            add_mocap_coordinate_marker(
                generator,
                coordinate_name=self._config.mocap_coordinate_name,
                coordinate_position=np.zeros(3),
                coordinate_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                attach_body="worldbody",
            )

    def _build_env(
        self, xml_file_path: pathlib.Path, camera: dict[str, dict[str, EnvModel]], env_models: dict[str, EnvModel]
    ) -> MujocoEnv:
        # TODO: Support ego camera and window render camera
        env: MujocoEnv = gymnasium.make(  # type: ignore
            id="DrawerRobot-v0",
            max_episode_steps=self._config.max_episode_steps,
            model_path=xml_file_path,
            mocap_cameras=tuple(camera["mocap"].values()),
            fixed_cameras=tuple(camera["fixed"].values()),
            robot=env_models["robot"],
            drawer=env_models["drawer"],
            # render setting
            # NOTE: None means use mujoco free camera.
            window_render_camera=(
                None
                if self._config.window_render_type_and_name is None
                else camera[self._config.window_render_type_and_name[0]][self._config.window_render_type_and_name[1]]
            ),
            render_segmentation=self._config.render_segmentation,
            segmentation_object_names=(
                self._config.robot_finger_geom_name,
                self._config.robot_hand_geom_name,
                self._config.drawer_handle_name,
                "task_area_marker_1",
                "task_area_marker_2",
                "task_area_marker_3",
                "task_area_marker_4",
                "task_area_marker_5",
                "task_area_marker_6",
                self._config.end_effector_marker_name,
            ),
            render_color=self._config.render_color,
            render_depth=self._config.render_depth,
            frame_skip=self._config.frame_skip,
            render_mode=self._config.render_mode,
            width=self._config.width,
            height=self._config.height,
            default_camera_config=self._config.default_camera_config,
            use_fixed_camera_in_human_mode=self._config.use_fixed_camera_in_human_mode,
            target_initial_end_effector_position=self._config.target_initial_end_effector_position,
            target_initial_end_effector_rotation=self._config.target_initial_end_effector_rotation,
            end_effector_min_position=self._config.end_effector_min_position,
            end_effector_max_position=self._config.end_effector_max_position,
            mocap_coordinate_marker=env_models["coordinate_marker"],
            initial_position_noise=self._config.initial_position_noise,
        )
        return env
