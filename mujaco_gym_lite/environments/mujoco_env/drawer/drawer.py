import pathlib
from typing import Any, Optional, Union

import gymnasium
import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.drawer import Drawer
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mocap_object import MocapObject
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robot import Robot
from mujaco_gym_lite.environment_tools.mujoco_env.functions.camera import camera_observation, move_mocap_cameras
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mocap import reset_mocap
from mujaco_gym_lite.environment_tools.mujoco_env.functions.reward import tolerance
from mujaco_gym_lite.environment_tools.mujoco_env.functions.robot import robot_action, robot_observation
from mujaco_gym_lite.environments.mujoco_env.mujoco_env import MujocoEnv
from mujaco_gym_lite.logger import logger
from mujaco_gym_lite.utils.randoms import rand_min_max
from mujaco_gym_lite.utils.transforms import (
    create_transformation_matrix,
    extract_position,
    extract_rotation,
    matrix_to_quat,
)
from mujaco_gym_lite.utils.views import view_to_rotation


def _object_obervation(drawer: Drawer, qpos_bounds) -> tuple[dict[str, npt.NDArray], dict[str, list[npt.NDArray]]]:
    drawer_qpos = np.array(drawer.joint_qpos(), dtype=np.float32).reshape((1,))
    orig_qpos = drawer_qpos.copy()
    clipped_qpos = np.clip(orig_qpos, a_min=qpos_bounds[0], a_max=qpos_bounds[1], dtype=np.float32)
    if not np.array_equal(orig_qpos, clipped_qpos):
        logger.debug(f"Drawer position clipped:\n  before={orig_qpos}\n  after ={clipped_qpos}")

    object_observation = {
        "drawer/qpos": clipped_qpos,
        "drawer/handle_position": np.array(drawer.handle_center_position(), dtype=np.float32),
    }
    return object_observation, {}


def _env_observation(
    end_effector_position_min: Optional[npt.NDArray],
    end_effector_position_max: Optional[npt.NDArray],
    drawer_qpos_min: npt.NDArray,
    drawer_qpos_max: npt.NDArray,
    # object obs settings
    drawer: Drawer,
    robot: Robot,
    fixed_cameras: tuple[Camera, ...],
    mocap_cameras: tuple[MocapCamera, ...],
    # camera obs settings
    render_color: bool,
    render_depth: bool,
    render_segmentation: bool,
    segmentation_object_names: tuple[str, ...],
    target_initial_end_effector_rotation: npt.NDArray,
    # env obs
    mj_data: "mujoco.MjData",
) -> tuple[dict[str, npt.NDArray], dict[str, Union[list[npt.NDArray], dict[str, int], dict[int, str]]]]:
    observation: dict[str, npt.NDArray] = {}
    obs_info: dict[str, Union[list[npt.NDArray], dict[str, int], dict[int, str]]] = {}

    object_obs, object_info = _object_obervation(drawer=drawer, qpos_bounds=[drawer_qpos_min, drawer_qpos_max])

    camera_obs, camera_info = camera_observation(
        fixed_cameras=fixed_cameras,
        mocap_cameras=mocap_cameras,
        render_color=render_color,
        render_segmentation=render_segmentation,
        render_depth=render_depth,
        segmentation_object_names=segmentation_object_names,
    )
    robot_obs, robot_observation_info = robot_observation(
        robot=robot,
        target_initial_end_effector_rotation=target_initial_end_effector_rotation,
        end_effector_position_min=end_effector_position_min,
        end_effector_position_max=end_effector_position_max,
    )
    env_obs = {
        "qpos": np.array(mj_data.qpos, dtype=np.float32),
        "qvel": np.array(mj_data.qvel.copy(), dtype=np.float32),
    }
    observation.update(object_obs)
    observation.update(camera_obs)
    observation.update(robot_obs)
    observation.update(env_obs)

    # add special robot observation
    touch_handle, touch_handle_info = drawer.has_handle_touch(robot.geom_root_name())
    robot_special_obs = {
        "robot/distance_to_handle_center": np.array(
            [drawer.distance_to_handle_center(extract_position(robot.end_effector_pose()))], dtype=np.float32
        ),
        "robot/touch_handle": np.array([touch_handle], dtype=np.float32),
    }
    observation.update(robot_special_obs)

    obs_info.update(object_info)
    obs_info.update(camera_info)
    obs_info.update(robot_observation_info)
    obs_info.update(touch_handle_info)  # type: ignore

    return observation, obs_info


def _compute_reward(
    end_effector_position: npt.NDArray,
    handle_position: npt.NDArray,
    target_handle_position: npt.NDArray,
    initial_handle_position: npt.NDArray,
    initial_end_effector_position: npt.NDArray,
    drawer_open_threshold: float,
    drawer_max_joint_qpos: float,
):
    """compute reward of drawer opening task"""
    handle_error = float(np.linalg.norm(handle_position - target_handle_position))
    reward_for_opening = tolerance(
        handle_error,
        bounds=(0, max(0.0, drawer_open_threshold - 0.01)),
        margin=np.abs(drawer_max_joint_qpos),
        sigmoid="long_tail",
    )

    gripper_error = (handle_position - end_effector_position) * np.array([1.0, 1.0, 3.0])
    gripper_error_init = initial_handle_position - initial_end_effector_position * np.array([1.0, 1.0, 3.0])

    reward_for_caging = tolerance(
        float(np.linalg.norm(gripper_error)),
        bounds=(0.0, 0.025),  # target range
        margin=np.linalg.norm(gripper_error_init),  # margin for decays
        sigmoid="long_tail",  # decay function shape
    )
    reward = 3.0 * reward_for_caging + 6.0 * reward_for_opening
    return float(reward), {
        "reward/handle_error": handle_error,
        "reward/reward_for_caging": reward_for_caging,
        "reward/reward_for_opening": reward_for_opening,
    }


def _step_to_target(robot, model, data, frame_skip, target_pos, target_quat, max_trans_step=0.05, render_fn=None):
    for _ in range(100):
        ee_pose = robot.end_effector_pose()
        curr_pos = extract_position(ee_pose)

        delta_pos = target_pos - curr_pos
        dist = np.linalg.norm(delta_pos)
        if dist > max_trans_step:
            delta_pos = delta_pos / dist * max_trans_step

        next_pos = curr_pos + delta_pos
        next_quat = target_quat

        robot.apply_gripper_control(np.array([135.0, 135.0]))
        robot_action(
            robot,
            np.hstack([next_pos, next_quat]),
            close=False,
            open=False,
            home=False,
            gripper_command=np.array([135.0, 135.0]),  # always close
        )

        mujoco.mj_step(model, data, nstep=frame_skip)
        if render_fn is not None:
            render_fn()


def _compute_distance_reward(
    robot_position,
    handle_position,
    target_robot_position,
    target_handle_position,
):
    robot_error = np.exp(-5.0 * np.sum((target_robot_position - robot_position) ** 2))
    handle_error = np.exp(-5.0 * np.sum((target_handle_position - handle_position) ** 2))
    return 5.0 * (0.1 * robot_error + handle_error), {}


class DrawerRobotEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "segmentation_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        model_path: pathlib.Path,
        mocap_cameras: tuple[MocapCamera, ...],
        fixed_cameras: tuple[Camera, ...],
        drawer: Drawer,
        robot: Robot,
        target_initial_end_effector_position: npt.NDArray,
        target_initial_end_effector_rotation: npt.NDArray,
        end_effector_min_position: npt.NDArray,
        end_effector_max_position: npt.NDArray,
        # render setting
        window_render_camera: Optional[Camera] = None,  # None means use mujoco free camera.
        render_color: bool = True,
        render_depth: bool = False,
        render_segmentation: bool = False,
        segmentation_object_names: tuple[str, ...] = (
            "j2n6s300",
            "drawer_handle",
        ),
        frame_skip: int = 10,
        render_mode: str = "human",
        width: int = 1280,
        height: int = 720,
        default_camera_config: Optional[dict] = None,
        use_fixed_camera_in_human_mode: bool = False,
        mocap_coordinate_marker: Optional[MocapObject] = None,
        initial_position_noise: bool = False,
    ):
        self._window_render_camera = window_render_camera
        self._mocap_cameras = mocap_cameras
        self._fixed_cameras = fixed_cameras

        self._render_color = render_color
        self._render_depth = render_depth
        self._render_segmentation = render_segmentation
        self._segmentation_object_names = segmentation_object_names

        self._robot = robot
        self._drawer = drawer

        self._end_effector_min_position = end_effector_min_position
        self._end_effector_max_position = end_effector_max_position
        assert np.all(self._end_effector_max_position >= self._end_effector_min_position)

        # NOTE: Work around for getting mj_model before calling super class constructor
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self._initial_position_noise = initial_position_noise

        observation_space = self._build_observation_space(
            fixed_cameras=fixed_cameras,
            mocap_cameras=mocap_cameras,
            height=height,
            width=width,
            render_color=render_color,
            render_depth=render_depth,
            render_segmentation=render_segmentation,
        )
        action_space = self._build_action_space(mocap_cameras=mocap_cameras)

        super().__init__(
            model_path=str(model_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            action_space=action_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=(None if window_render_camera is None else window_render_camera.camera_name()),
            default_camera_config=default_camera_config,
            use_fixed_camera_in_human_mode=use_fixed_camera_in_human_mode,
        )
        self._target_initial_end_effector_position = target_initial_end_effector_position
        self._target_initial_end_effector_rotation = target_initial_end_effector_rotation

        reset_mocap(self.model, self.data)

        self._drawer.load_mj_model_and_data(self.model, self.data)
        self._robot.load_mj_model_and_data(self.model, self.data)

        for mocap_camera in self._mocap_cameras:
            mocap_camera.load_mj_model_and_data(self.model, self.data)

        for fixed_camera in self._fixed_cameras:
            fixed_camera.load_mj_model_and_data(self.model, self.data)

        for mocap_camera in self._mocap_cameras:
            mocap_camera.load_renderer(self.mujoco_renderer)

        for fixed_camera in self._fixed_cameras:
            fixed_camera.load_renderer(self.mujoco_renderer)

        self._mocap_coordinate_marker = None
        if mocap_coordinate_marker is not None:
            self._mocap_coordinate_marker = mocap_coordinate_marker
            self._mocap_coordinate_marker.load_mj_model_and_data(self.model, self.data)

    @staticmethod
    def workspace() -> tuple[npt.NDArray, npt.NDArray]:
        return np.array([-0.3, -0.3, 0.05]), np.array([0.3, 0.1, 0.45])

    def step(self, action: Any):
        self._update_timesteps()
        # action serialization
        robot_pos = action["robot/end_effector/position"]

        # apply clip
        clipped_robot_pos = np.clip(
            robot_pos, a_min=self._end_effector_min_position, a_max=self._end_effector_max_position
        )

        robot_quat = action["robot/end_effector/rotation"]
        assert len(robot_quat) == 4
        robot_home = action["robot/home"]

        camera_actions = []
        for mocap_camera in self._mocap_cameras:
            camera_actions.append(
                [
                    *action[f"camera/{mocap_camera.camera_name()}/position"],
                    *action[f"camera/{mocap_camera.camera_name()}/rotation"],
                ]
            )

        robot_action(
            self._robot,
            np.array([*clipped_robot_pos, *robot_quat]),
            gripper_command=np.array([135.0, 135.0]),
            open=False,
            close=False,
            home=bool(robot_home),
        )
        move_mocap_cameras(self._mocap_cameras, np.array(camera_actions))

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        ob, ob_info = _env_observation(
            end_effector_position_min=self._end_effector_min_position,
            end_effector_position_max=self._end_effector_max_position,
            drawer_qpos_min=self.observation_space["drawer/qpos"].low,
            drawer_qpos_max=self.observation_space["drawer/qpos"].high,
            drawer=self._drawer,
            robot=self._robot,
            fixed_cameras=self._fixed_cameras,
            mocap_cameras=self._mocap_cameras,
            render_color=self._render_color,
            render_depth=self._render_depth,
            render_segmentation=self._render_segmentation,
            segmentation_object_names=self._segmentation_object_names,
            target_initial_end_effector_rotation=self._target_initial_end_effector_rotation,
            mj_data=self.data,
        )

        # TODO: Support randomization and set parameter
        bias = np.array([0.0, -0.0275, -0.02])
        reward, reward_info = _compute_reward(
            handle_position=ob["drawer/handle_position"] + bias,
            end_effector_position=ob["robot/end_effector/position"],
            target_handle_position=self._initial_handle_position
            + np.array([0.0, -np.abs(self._drawer._drawer_joint_range[1]), 0.0])
            + bias,  # NOTE: We should add bias to target handle position as well as the current handle position
            initial_end_effector_position=self._initial_end_effector_position,
            initial_handle_position=self._initial_handle_position + bias,
            drawer_open_threshold=self._drawer._open_and_close_thredshold,
            drawer_max_joint_qpos=self._drawer._drawer_joint_range[1],
        )
        ob_info.update(reward_info)
        ob_info["task/success"] = self._drawer.is_open()
        if ob_info["task/success"]:
            reward += 3.0

        if self._mocap_coordinate_marker is not None:
            self._mocap_coordinate_marker.move(
                create_transformation_matrix(translation=ob["drawer/handle_position"] + bias)
            )

        return ob, reward, False, False, ob_info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._reset_timesteps()
        self._reset_simulation()

        self._robot.appply_home_arm_qpos()
        home_qpos = self._robot.arm_joint_qpos()
        self._robot.apply_arm_control(np.array(home_qpos))
        self._robot.appply_home_gripper_qpos()

        if options is None:
            # logger.debug("Option is None! Use default mocap camera settings")
            camera_options = {}
            for mocap_camera in self._mocap_cameras:
                look_at_point = np.array([0.0, 0.0, 0.3])
                initial_camera_position = np.array(
                    [look_at_point[0] + 0.75, look_at_point[1] - 0.25, look_at_point[2] + 0.3]
                )
                initial_camera_rotation = matrix_to_quat(view_to_rotation(look_at_point - initial_camera_position))
                camera_options[f"camera/{mocap_camera.camera_name()}/position"] = initial_camera_position
                camera_options[f"camera/{mocap_camera.camera_name()}/rotation"] = initial_camera_rotation

            options = camera_options

        camera_actions = []
        for mocap_camera in self._mocap_cameras:
            camera_actions.append(
                [
                    *options[f"camera/{mocap_camera.camera_name()}/position"],
                    *options[f"camera/{mocap_camera.camera_name()}/rotation"],
                ]
            )
        move_mocap_cameras(self._mocap_cameras, np.array(camera_actions))

        mujoco.mj_forward(self.model, self.data)

        warm_up_step = 1
        for _ in range(warm_up_step):
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        if self._initial_position_noise:
            # initial robot position: np.array([0.0, 0.025, 0.45], dtype=np.float32)
            # drawer: np.array([0.1, -0.08, 0.125], dtype=np.float32)
            ee_pose = self._robot.end_effector_pose()
            drawer_handle_pos = self._drawer.handle_center_position()
            initial_noise = rand_min_max(min_val=np.array([-0.15, -0.1, 0.1]), max_val=np.array([0.15, 0.1, 0.25]))
            pos = drawer_handle_pos + initial_noise
            quat = extract_rotation(ee_pose, rotation_type="quaternion")
            _step_to_target(
                self._robot,
                self.model,
                self.data,
                self.frame_skip,
                pos,
                quat,
                max_trans_step=0.05,
            )

        ob, ob_info = _env_observation(
            end_effector_position_min=self._end_effector_min_position,
            end_effector_position_max=self._end_effector_max_position,
            drawer_qpos_min=self.observation_space["drawer/qpos"].low,
            drawer_qpos_max=self.observation_space["drawer/qpos"].high,
            drawer=self._drawer,
            robot=self._robot,
            fixed_cameras=self._fixed_cameras,
            mocap_cameras=self._mocap_cameras,
            render_color=self._render_color,
            render_depth=self._render_depth,
            render_segmentation=self._render_segmentation,
            segmentation_object_names=self._segmentation_object_names,
            target_initial_end_effector_rotation=self._target_initial_end_effector_rotation,
            mj_data=self.data,
        )

        self._initial_handle_position = ob["drawer/handle_position"]
        self._initial_end_effector_position = ob["robot/end_effector/position"]

        if self._mocap_coordinate_marker is not None:
            self._mocap_coordinate_marker.move(
                create_transformation_matrix(translation=self._initial_handle_position + np.array([0.0, -0.0275, 0.0]))
            )
        return ob, ob_info

    def observation_info_keys(self) -> tuple[str, ...]:
        keys = []
        for i in range(self._robot.num_fingers):
            keys.append(f"robot/contact_positions/finger{i}")
            keys.append(f"robot/contact_names/finger{i}")

        keys.append("drawer/contact_positions/handle")
        keys.append("drawer/contact_names/handle")

        for mocap_camera in self._mocap_cameras:
            keys.append(f"camera/{mocap_camera.camera_name()}/segmentation/id_to_name")
            keys.append(f"camera/{mocap_camera.camera_name()}/segmentation/name_to_id")

        for fixed_camera in self._fixed_cameras:
            keys.append(f"camera/{fixed_camera.camera_name()}/segmentation/id_to_name")
            keys.append(f"camera/{fixed_camera.camera_name()}/segmentation/name_to_id")

        return tuple(keys)

    def _build_observation_space(
        self,
        mocap_cameras: tuple[MocapCamera, ...],
        fixed_cameras: tuple[Camera, ...],
        height: int,
        width: int,
        render_color: bool,
        render_depth: bool,
        render_segmentation: bool,
    ):
        assert self._drawer._drawer_joint_range[0] < self._drawer._drawer_joint_range[1]
        object_observation_space = {
            "drawer/qpos": gymnasium.spaces.Box(
                low=self._drawer._drawer_joint_range[0],
                high=self._drawer._drawer_joint_range[1],
                shape=(1,),
                dtype=np.float32,
            ),
            "drawer/handle_position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }

        observation_space = {}
        observation_space.update(object_observation_space)
        observation_space.update(
            self._build_camera_observation_space(
                fixed_cameras=fixed_cameras,
                mocap_cameras=mocap_cameras,
                height=height,
                width=width,
                color=render_color,
                depth=render_depth,
                segmentation=render_segmentation,
            )
        )
        observation_space.update(
            self._build_robot_observation_space(
                robot=self._robot,
                end_effector_min_position=self._end_effector_min_position,
                end_effector_max_position=self._end_effector_max_position,
            )
        )
        observation_space.update(self._build_env_observation_space())

        # add special
        observation_space.update(
            {
                "robot/distance_to_handle_center": gymnasium.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "robot/touch_handle": gymnasium.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        return gymnasium.spaces.Dict(observation_space)  # type: ignore

    def _build_action_space(self, mocap_cameras: tuple[MocapCamera, ...]):
        robot_action_space = {
            "robot/end_effector/position": gymnasium.spaces.Box(
                self._end_effector_min_position, self._end_effector_max_position, shape=(3,), dtype=np.float32
            ),
            "robot/end_effector/rotation": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "robot/home": gymnasium.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        camera_action_space = {}
        for mocap_camera in mocap_cameras:
            camera_action_space[f"camera/{mocap_camera.camera_name()}/position"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
            camera_action_space[f"camera/{mocap_camera.camera_name()}/rotation"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            )
        task_action_space = {"task/end_episode": gymnasium.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)}

        action_space = {}
        action_space.update(robot_action_space)
        action_space.update(camera_action_space)
        action_space.update(task_action_space)

        return gymnasium.spaces.Dict(action_space)  # type: ignore
