import os
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

import gymnasium
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Space

import mujoco

if TYPE_CHECKING:
    from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera
    from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robot import Robot

from mujaco_gym_lite.environment_tools.mujoco_env.renderer import MujocoRenderer


class MujocoEnv(gymnasium.Env, metaclass=ABCMeta):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space,
        action_space: Space,
        render_mode: str,
        width: int = 1280,
        height: int = 720,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[dict] = None,
        max_geom: int = 1000,
        use_fixed_camera_in_human_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise OSError(f"File {self.model_path} does not exist")

        self._initialize_simulation()  # may use width and height

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        self.observation_space = observation_space
        self.action_space = action_space

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config=default_camera_config,
            width=width,
            height=height,
            max_geom=max_geom,
            use_fixed_camera_in_human_mode=use_fixed_camera_in_human_mode,
        )

        self._timesteps = 0

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
            "segmentation_array",
        ], self.metadata["render_modes"]

    def _update_timesteps(self):
        self._timesteps += 1

    def _reset_timesteps(self):
        self._timesteps = 0

    def _current_timesteps(self):
        return self._timesteps

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def change_render_camera_name(self, camera_name: str):
        self.camera_name = camera_name

    def render(self):
        return self.mujoco_renderer.render(self.render_mode, self.camera_id, self.camera_name)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def observation_info_keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    @staticmethod
    def workspace() -> tuple[npt.NDArray, npt.NDArray]:
        raise NotImplementedError

    def _build_camera_observation_space(
        self,
        fixed_cameras: tuple["Camera", ...],
        mocap_cameras: tuple["MocapCamera", ...],
        height: int,
        width: int,
        color: bool,
        depth: bool,
        segmentation: bool,
    ) -> dict[str, gymnasium.spaces.Box]:
        camera_observation_space = {}
        for mocap_camera in mocap_cameras:
            if depth:
                camera_observation_space[f"camera/{mocap_camera.camera_name()}/depth"] = gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(height, width), dtype=np.float32
                )
            if color:
                camera_observation_space[f"camera/{mocap_camera.camera_name()}/color"] = gymnasium.spaces.Box(
                    0, 255, shape=(height, width, 3), dtype=np.uint8
                )
            if segmentation:
                camera_observation_space[f"camera/{mocap_camera.camera_name()}/segmentation"] = gymnasium.spaces.Box(
                    0, 2**16 - 1, shape=(height, width), dtype=np.uint16
                )

            camera_observation_space[f"camera/{mocap_camera.camera_name()}/camera_extrinsic"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(4, 4), dtype=np.float32
            )
            camera_observation_space[f"camera/{mocap_camera.camera_name()}/camera_intrinsic"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(3, 3), dtype=np.float32
            )

        for fixed_camera in fixed_cameras:
            if depth:
                camera_observation_space[f"camera/{fixed_camera.camera_name()}/depth"] = gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(height, width), dtype=np.float32
                )
            if color:
                camera_observation_space[f"camera/{fixed_camera.camera_name()}/color"] = gymnasium.spaces.Box(
                    0, 255, shape=(height, width, 3), dtype=np.uint8
                )
            if segmentation:
                camera_observation_space[f"camera/{fixed_camera.camera_name()}/segmentation"] = gymnasium.spaces.Box(
                    0, 2**16, shape=(height, width), dtype=np.uint16
                )

            camera_observation_space[f"camera/{fixed_camera.camera_name()}/camera_extrinsic"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(4, 4), dtype=np.float32
            )
            camera_observation_space[f"camera/{fixed_camera.camera_name()}/camera_intrinsic"] = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(3, 3), dtype=np.float32
            )

        return camera_observation_space

    def _build_robot_observation_space(
        self, robot: "Robot", end_effector_min_position: npt.NDArray, end_effector_max_position: npt.NDArray
    ):
        robot_observation_space = {
            "robot/base/position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "robot/base/rotation": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "robot/end_effector/position": gymnasium.spaces.Box(
                end_effector_min_position, end_effector_max_position, shape=(3,), dtype=np.float32
            ),
            "robot/end_effector/rotation": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "robot/target_initial_end_effector_rotation": gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32
            ),
            "robot/contact": gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(robot.num_fingers,), dtype=np.float32
            ),  # bool flag for each finger, supports
            "robot/joint_angles": gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(robot.num_arm_joints,), dtype=np.float32
            ),
            "robot/gripper_joint_angles": gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(robot.num_gripper_joints,), dtype=np.float32
            ),
        }
        return robot_observation_space

    def _build_env_observation_space(self):
        env_space = {
            "qpos": gymnasium.spaces.Box(0.0, 1.0, shape=(self.model.nq,), dtype=np.float32),
            "qvel": gymnasium.spaces.Box(0.0, 1.0, shape=(self.model.nv,), dtype=np.float32),
        }
        return env_space
