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
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.lidded_box import Lid, OpenTopBox
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mocap_object import MocapObject
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.metaworld_objects.lidded_box import add_lidded_box
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.asset import add_asset_model
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.marker import add_mocap_coordinate_marker
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.scene.floor_scene import add_basic_scene_setting
from mujaco_gym_lite.environments.mujoco_env.mujoco_env import MujocoEnv
from mujaco_gym_lite.utils.files import write_json, write_txt
from mujaco_gym_lite.utils.transforms import euler_to_quat

import mujaco_gym_lite.environments  # noqa


@dataclass
class LiddedBoxRobotEnvBuilderConfig(RobotEnvBuilderConfig):
    # model settings
    box_name: str = "box"
    box_joint_name: str = "box_joint"
    box_body_name: str = "box_body"
    box_position: npt.NDArray = np.array([0.0, 0.0, 0.0])
    box_rotation: npt.NDArray = np.array([1.0, 0.0, 0.0, 0.0])
    box_geom_name: str = "box"
    box_collision_name: str = "box_collision"
    box_center_top_site_name: str = "box_center_top_site"

    lid_name: str = "lid"
    lid_joint_name: str = "lid_joint"
    lid_body_name: str = "lid_body"
    lid_collision_name: str = "lid_collision"
    lid_handle_mesh_name: str = "lid_handle_mesh"
    lid_handle_site_name: str = "lid_handle_site"
    lid_handle_low_site_name: str = "lid_handle_low_site"
    lid_position: npt.NDArray = np.array([0.3, -0.1, 0.1])
    lid_rotation: npt.NDArray = euler_to_quat(np.array([0.0, 0.0, np.pi * 0.5]))
    lid_min_position: npt.NDArray = np.array([-0.5, -0.4, 0.0])
    lid_max_position: npt.NDArray = np.array([0.5, 0.4, 0.5])

    # robot
    home_arm_qpos: npt.NDArray = np.array([-1.04906454, 3.8513547, 1.68162966, -0.30987864, 1.13917531, -4.50053584])
    home_gripper_qpos: npt.NDArray = np.array([0.2, 0.0, 0.2, 0.0, 0.2, 0.0])
    robot_hand_geom_name: str = "j2n6s300_hand"  # TODO: support different robot name
    robot_finger_geom_name: str = "j2n6s300_finger"  # TODO: support different robot name
    end_effector_min_position: npt.NDArray = np.array([-0.2, -0.15, 0.075])
    end_effector_max_position: npt.NDArray = np.array([0.4, 0.45, 0.4])

    initial_position_noise: bool = True
    initial_lift: bool = True

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


class LiddedBoxRobotEnvBuilder(RobotEnvBuilder):
    _config: LiddedBoxRobotEnvBuilderConfig

    def __init__(self, config: LiddedBoxRobotEnvBuilderConfig):
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
        lid = Lid(
            base_body_name=self._config.lid_body_name,
            lid_joint_names=[self._config.lid_joint_name],
            body_names=[self._config.lid_body_name],
            geom_root_name=self._config.lid_collision_name,
            lid_handle_geom_name=self._config.lid_handle_mesh_name,
            lid_handle_site_name=self._config.lid_handle_site_name,
            lid_handle_low_site_name=self._config.lid_handle_low_site_name,
            lid_max_position=self._config.lid_max_position,
            lid_min_position=self._config.lid_min_position,
        )
        box = OpenTopBox(
            base_body_name=self._config.box_body_name,
            body_names=[self._config.box_body_name],
            geom_root_name=self._config.box_geom_name,
            open_top_box_center_top_site_name=self._config.box_center_top_site_name,
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
            "lid": lid,
            "box": box,
            "robot": robot,
            "coordinate_marker": coordinate_marker,
            "camera": {"mocap": mocap_cameras, "fixed": fixed_cameras, "ego": ego_cameras},
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
        add_lidded_box(
            generator=generator,
            asset_dir_path=self._config.asset_dir_path,
            box_name=self._config.box_name,
            box_position=self._config.box_position,
            box_rotation=self._config.box_rotation,
            box_body_name=self._config.box_body_name,
            box_collision_name=self._config.box_collision_name,
            box_center_top_site_name=self._config.box_center_top_site_name,
            lid_body_name=self._config.lid_body_name,
            lid_joint_name=self._config.lid_joint_name,
            lid_position=self._config.lid_position,
            lid_rotation=self._config.lid_rotation,
            lid_collision_name=self._config.lid_collision_name,
            lid_handle_mesh_name=self._config.lid_handle_mesh_name,
            lid_handle_site_name=self._config.lid_handle_site_name,
            lid_handle_low_site_name=self._config.lid_handle_low_site_name,
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
            id="LiddedBoxRobot-v0",
            max_episode_steps=self._config.max_episode_steps,
            model_path=xml_file_path,
            mocap_cameras=tuple(camera["mocap"].values()),
            fixed_cameras=tuple(camera["fixed"].values()),
            robot=env_models["robot"],
            lid=env_models["lid"],
            open_top_box=env_models["box"],
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
                self._config.box_geom_name,
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
            initial_lift=self._config.initial_lift,
        )
        return env
