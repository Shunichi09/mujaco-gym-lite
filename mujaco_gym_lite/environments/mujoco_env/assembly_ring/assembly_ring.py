import pathlib
from typing import Any, Optional, Union

import gymnasium
import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.assembly_ring import AssemblyRing
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.assembly_rod import AssemblyRod
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mocap_object import MocapObject
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robot import Robot
from mujaco_gym_lite.environment_tools.mujoco_env.functions.camera import camera_observation, move_mocap_cameras
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mocap import reset_mocap
from mujaco_gym_lite.environment_tools.mujoco_env.functions.reward import tolerance
from mujaco_gym_lite.environment_tools.mujoco_env.functions.robot import robot_action, robot_observation
from mujaco_gym_lite.environments.mujoco_env.mujoco_env import MujocoEnv
from mujaco_gym_lite.logger import logger
from mujaco_gym_lite.utils.transforms import (
    create_transformation_matrix,
    extract_position,
    extract_rotation,
    matrix_to_quat,
)
from mujaco_gym_lite.utils.views import view_to_rotation


def _object_obervation(
    robot: Robot,
    assembly_ring: AssemblyRing,
    assembly_rod: AssemblyRod,
    ring_position_min: Optional[npt.NDArray],
    ring_position_max: Optional[npt.NDArray],
) -> tuple[dict[str, npt.NDArray], dict[str, list[npt.NDArray]]]:
    ring_pos = np.array(assembly_ring.ring_position(), dtype=np.float32).reshape((3,))
    orig_qpos = ring_pos.copy()
    clipped_qpos = np.clip(orig_qpos, a_min=ring_position_min, a_max=ring_position_max, dtype=np.float32)
    if not np.array_equal(orig_qpos, clipped_qpos):
        logger.debug(f"Ring position clipped:\n  before={orig_qpos}\n  after ={clipped_qpos}")

    object_observation = {
        "assembly_ring/handle/position": np.array(assembly_ring.ring_handle_position(), dtype=np.float32),
        "assembly_ring/ring/position": np.array(assembly_ring.ring_position(), dtype=np.float32),
        "assembly_ring/ring/qpos": clipped_qpos,
        "assembly_rod/bottom/position": np.array(assembly_rod.rod_bottom_position(), dtype=np.float32),
        "assembly_rod/top/position": np.array(assembly_rod.rod_top_position(), dtype=np.float32),
    }
    gripper_hold_handle, _ = assembly_ring.has_handle_touch(robot.geom_root_name())
    return object_observation, {"assembly_ring/holded_handle_by_gripper": gripper_hold_handle}


def _env_observation(
    end_effector_position_min: Optional[npt.NDArray],
    end_effector_position_max: Optional[npt.NDArray],
    ring_position_min: Optional[npt.NDArray],
    ring_position_max: Optional[npt.NDArray],
    # object obs settings
    assembly_ring: AssemblyRing,
    assembly_rod: AssemblyRod,
    robot: Robot,
    fixed_cameras: tuple[Camera, ...],
    mocap_cameras: tuple[MocapCamera, ...],
    # camera obs settings
    render_color: bool,
    render_depth: bool,
    render_segmentation: bool,
    segmentation_object_names: tuple[str, ...],
    target_initial_end_effector_rotation: npt.NDArray,
) -> tuple[dict[str, npt.NDArray], dict[str, Union[list[npt.NDArray], dict[str, int], dict[int, str]]]]:
    observation: dict[str, npt.NDArray] = {}
    obs_info: dict[str, Union[list[npt.NDArray], dict[str, int], dict[int, str]]] = {}

    object_obs, object_info = _object_obervation(
        assembly_ring=assembly_ring,
        assembly_rod=assembly_rod,
        ring_position_min=ring_position_min,
        ring_position_max=ring_position_max,
        robot=robot,
    )

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

    observation.update(object_obs)
    observation.update(camera_obs)
    observation.update(robot_obs)

    # add special robot observation

    obs_info.update(object_info)
    obs_info.update(camera_info)
    obs_info.update(robot_observation_info)

    return observation, obs_info


def _compute_reward_ring(
    ring_center: npt.NDArray[Any],
    init_ring_center: npt.NDArray[Any],
    target_rod_top_position: npt.NDArray[Any],
) -> tuple[float, dict]:

    pos_error = target_rod_top_position - ring_center
    init_pos_error = target_rod_top_position - init_ring_center
    radius = np.linalg.norm(pos_error[:2])
    height = abs(pos_error[2] + 0.05)
    init_radius = np.linalg.norm(init_pos_error[:2])

    reward_for_xy_alignment = tolerance(
        radius,
        bounds=(0, 0.025),
        margin=np.abs(init_radius),
        sigmoid="long_tail",
    )
    reward_for_z_alignment = tolerance(
        height,
        bounds=(0, 0.01),
        margin=0.3,
        sigmoid="long_tail",
    )
    if radius < 0.0525:
        reward_for_z_alignment = 1.0
        reward_for_height = tolerance(
            abs(max(float(ring_center[2] - 0.015), 0.0)),
            bounds=(0, 0.01),
            margin=0.4,
            sigmoid="long_tail",
        )

    else:
        reward_for_height = 0.0

    reward = 3.0 * reward_for_height + 1.0 * reward_for_xy_alignment + 1.0 * reward_for_z_alignment
    return reward, {"reward/xy_error": radius}


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

        robot.apply_gripper_control(np.array([200.0, 200.0], dtype=np.float32))
        robot_action(
            robot,
            np.hstack([next_pos, next_quat]),
            close=False,
            open=False,
            home=False,
            gripper_command=np.array([200.0, 200.0], dtype=np.float32),
        )

        mujoco.mj_step(model, data, nstep=frame_skip)
        if render_fn is not None:
            render_fn()


def _sample_donut(center_xy=(0.0, 0.0), r_inner=0.05, r_outer=0.20, z_range=(0.275, 0.325)):
    cx, cy = center_xy[:2]
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.sqrt(np.random.uniform(r_inner**2, r_outer**2))
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.random.uniform(z_range[0], z_range[1])
    return np.array([x, y, z]).flatten()


class AssemblyRingRobotEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "segmentation_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        model_path: pathlib.Path,
        mocap_cameras: tuple[MocapCamera, ...],
        fixed_cameras: tuple[Camera, ...],
        assembly_ring: AssemblyRing,
        assembly_rod: AssemblyRod,
        robot: Robot,
        target_initial_end_effector_position: npt.NDArray,
        target_initial_end_effector_rotation: npt.NDArray,
        end_effector_min_position: npt.NDArray,
        end_effector_max_position: npt.NDArray,
        initial_lift: bool,
        # render setting
        window_render_camera: Optional[Camera] = None,  # None means use mujoco free camera.
        render_color: bool = True,
        render_depth: bool = False,
        render_segmentation: bool = False,
        segmentation_object_names: tuple[str, ...] = ("ball",),
        frame_skip: int = 10,
        render_mode: str = "human",
        width: int = 1280,
        height: int = 720,
        default_camera_config: Optional[dict] = None,
        use_fixed_camera_in_human_mode: bool = False,
        mocap_coordinate_marker: Optional[MocapObject] = None,
    ):
        self._window_render_camera = window_render_camera
        self._mocap_cameras = mocap_cameras
        self._fixed_cameras = fixed_cameras

        self._render_color = render_color
        self._render_depth = render_depth
        self._render_segmentation = render_segmentation
        self._segmentation_object_names = segmentation_object_names

        self._robot = robot
        self._assembly_ring = assembly_ring
        self._assembly_rod = assembly_rod
        self._initial_lift = initial_lift
        self._end_effector_min_position = end_effector_min_position
        self._end_effector_max_position = end_effector_max_position
        assert np.all(self._end_effector_max_position >= self._end_effector_min_position)

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

        self._assembly_ring.load_mj_model_and_data(self.model, self.data)
        self._assembly_rod.load_mj_model_and_data(self.model, self.data)
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
        return np.array([-0.45, -0.45, -0.1]), np.array([0.45, 0.45, 0.44])

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
            close=False,
            open=False,
            home=bool(robot_home),
            gripper_command=np.array([200.0, 200.0], dtype=np.float32),  # always close
        )
        move_mocap_cameras(self._mocap_cameras, np.array(camera_actions))

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        ob, ob_info = _env_observation(
            end_effector_position_min=self._end_effector_min_position,
            end_effector_position_max=self._end_effector_max_position,
            ring_position_min=self.unwrapped.observation_space["assembly_ring/ring/qpos"].low,
            ring_position_max=self.unwrapped.observation_space["assembly_ring/ring/qpos"].high,
            assembly_ring=self._assembly_ring,
            assembly_rod=self._assembly_rod,
            robot=self._robot,
            fixed_cameras=self._fixed_cameras,
            mocap_cameras=self._mocap_cameras,
            render_color=self._render_color,
            render_depth=self._render_depth,
            render_segmentation=self._render_segmentation,
            segmentation_object_names=self._segmentation_object_names,
            target_initial_end_effector_rotation=self._target_initial_end_effector_rotation,
        )

        if self._mocap_coordinate_marker is not None:
            self._mocap_coordinate_marker.move(
                create_transformation_matrix(translation=ob["assembly_rod/top/position"] + np.array([0.0, 0.0, 0.075]))
            )

        reward, reward_info = _compute_reward_ring(
            ring_center=ob["assembly_ring/ring/position"],
            target_rod_top_position=ob["assembly_rod/top/position"],
            init_ring_center=self._initial_ring_center_position,
        )
        is_ring_inserting, info = self._assembly_rod.is_ring_inserting(ring=self._assembly_ring)
        ob_info["task/success"] = is_ring_inserting
        if is_ring_inserting:
            reward += 5.0  # additional bonus if success

        if not ob_info["assembly_ring/holded_handle_by_gripper"]:
            reward = 0.0  # set reward to zero, if not grasping
            ob_info["task/success"] = False

        ob_info.update(reward_info)
        return ob, reward, False, False, ob_info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_timesteps()
        self._reset_simulation()

        self._robot.appply_home_arm_qpos()
        home_qpos = self._robot.arm_joint_qpos()
        self._robot.apply_arm_control(np.array(home_qpos))
        # gripper
        self._robot.appply_home_gripper_qpos()

        if options is None:
            # logger.debug("Option is None! Use default mocap camera settings")
            camera_options = {}
            for mocap_camera in self._mocap_cameras:
                initial_camera_position = np.array([-0.1, 0.7, 0.5])
                initial_camera_rotation = matrix_to_quat(
                    view_to_rotation(np.array([-0.1, 0.1, 0.0]) - np.array([-0.1, 0.7, 0.5]))
                )
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

        # apply close gripper
        for _ in range(100):
            self._robot.apply_gripper_control(np.array([200.0, 200.0], dtype=np.float32))
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        if self._initial_lift:
            ee_pose = self._robot.end_effector_pose()
            # initial: np.array([ 0.275, -0.1, 0.08], dtype=np.float32)
            # peg: np.array([0.175, 0.1, 0.3], dtype=np.float32)
            rod_top_position = np.array(self._assembly_rod.rod_top_position(), dtype=np.float32)
            rod_ring_position = np.array(self._assembly_ring.ring_position(), dtype=np.float32)
            curr_pos = extract_position(ee_pose)
            difference_position = rod_ring_position - curr_pos

            target_pos = _sample_donut(center_xy=tuple(rod_top_position.tolist()), r_inner=0.1, r_outer=0.15)
            target_pos[:2] -= difference_position[:2]

            curr_quat = extract_rotation(ee_pose, rotation_type="quaternion")
            target_quat = curr_quat

            z_target_pos = np.array([curr_pos[0], curr_pos[1], target_pos[2]])
            _step_to_target(self._robot, self.model, self.data, self.frame_skip, z_target_pos, target_quat)
            xy_target_pos = np.array([target_pos[0], target_pos[1], z_target_pos[2]])
            _step_to_target(self._robot, self.model, self.data, self.frame_skip, xy_target_pos, target_quat)

        ob, ob_info = _env_observation(
            end_effector_position_min=self._end_effector_min_position,
            end_effector_position_max=self._end_effector_max_position,
            ring_position_min=self.unwrapped.observation_space["assembly_ring/ring/qpos"].low,
            ring_position_max=self.unwrapped.observation_space["assembly_ring/ring/qpos"].high,
            assembly_ring=self._assembly_ring,
            assembly_rod=self._assembly_rod,
            robot=self._robot,
            fixed_cameras=self._fixed_cameras,
            mocap_cameras=self._mocap_cameras,
            render_color=self._render_color,
            render_depth=self._render_depth,
            render_segmentation=self._render_segmentation,
            segmentation_object_names=self._segmentation_object_names,
            target_initial_end_effector_rotation=self._target_initial_end_effector_rotation,
        )

        if self._mocap_coordinate_marker is not None:
            self._mocap_coordinate_marker.move(
                create_transformation_matrix(translation=ob["assembly_rod/top/position"] + np.array([0.0, 0.0, 0.075]))
            )

        self._initial_ring_center_position = ob["assembly_ring/ring/position"]
        return ob, {}

    def observation_info_keys(self) -> tuple[str, ...]:
        keys = []
        for i in range(self._robot.num_fingers):
            keys.append(f"robot/contact_positions/finger{i}")
            keys.append(f"robot/contact_names/finger{i}")

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
        # HACK: Do not consider rotation.
        ring_position_from_robot_ee = np.array([-0.27494479, -0.00098435, -0.05267889])
        object_observation_space = {
            "assembly_ring/handle/position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "assembly_ring/ring/position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "assembly_ring/ring/qpos": gymnasium.spaces.Box(
                low=self._end_effector_min_position + ring_position_from_robot_ee - 0.025,
                high=self._end_effector_max_position + ring_position_from_robot_ee + 0.025,
                shape=(3,),
                dtype=np.float32,
            ),
            "assembly_rod/bottom/position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "assembly_rod/top/position": gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
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

        # add special
        observation_space.update({})

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    center_xy = np.array([-0.15, 0.0, 0.15])
    arm_xy = np.array([0.27392003, -0.09938303, 0.08472654])
    ring_pos = np.array([-0.00102476, -0.10036738, 0.03204765])
    difference = ring_pos - arm_xy
    pts = []
    for _ in range(1000):
        pt = _sample_donut(center_xy=tuple(center_xy.tolist()), r_inner=0.05, r_outer=0.15)
        pts.append(pt)

    pts = np.array(pts)
    arm_pts = pts - difference

    # 2D可視化 (x-y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=15)
    ax.scatter(arm_pts[:, 0], arm_pts[:, 1], s=15)

    xmin, xmax = -0.2, 0.0
    ymin, ymax = -0.05, 0.15
    r_forbidden = 0.05

    # ボックス枠
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, lw=1.5)
    ax.add_patch(rect)

    # 禁止円
    circ = Circle(center_xy, r_forbidden, fill=False, color="r", lw=1.5)
    ax.add_patch(circ)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Samples in box (circle excluded)")
    plt.show()

    # z の分布も確認したければヒストグラム
    plt.figure()
    plt.hist(pts[:, 2], bins=10)
    plt.xlabel("z")
    plt.ylabel("count")
    plt.title("z distribution")
    plt.show()
