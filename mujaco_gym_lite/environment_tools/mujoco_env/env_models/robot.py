import itertools
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d

import mujoco
from mujaco_gym_lite.environment_tools.ik_solvers.ik_sovler import InverseKinematicsSolver
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import (
    apply_joint_qpos_and_qvel,
    joint_qpos,
    joint_qvel,
)
from mujaco_gym_lite.utils.transforms import create_transformation_matrix, extract_position, extract_rotation


class Robot(EnvModel, metaclass=ABCMeta):
    _mj_model: Optional["mujoco.MjModel"]
    _mj_data: Optional["mujoco.MjData"]

    def __init__(
        self,
        base_body_name: str,
        arm_joint_names: list[str],
        arm_body_names: list[str],
        gripper_joint_names: list[str],
        gripper_body_names: list[str],
        geom_root_name: str,
        gripper_finger_tip_geom_names: dict[str, list[str]],  # key means a group
        end_effector_body_name: str,
        ik_solver: InverseKinematicsSolver,
        joint_meshes: list[o3d.geometry.TriangleMesh],
        disable_collision_mesh_pairs: list[tuple[int, int]],
        home_joint_qpos: list[float],
        home_gripper_qpos: list[float],
        num_actuators: int,
        num_arm_joints: int = 6,
        num_gripper_joints: int = 6,
        num_fingers: int = 3,
    ):
        super().__init__(
            base_body_name,
            arm_joint_names + gripper_joint_names,
            arm_body_names + gripper_body_names,
            geom_root_name,
        )
        self._arm_body_names = arm_body_names
        self._gripper_body_names = gripper_body_names
        self._arm_joint_names = arm_joint_names
        self._gripper_joint_names = gripper_joint_names
        self._ik_solver = ik_solver
        self._joint_meshes = joint_meshes
        self._end_effector_body_name = end_effector_body_name
        self._gripper_finger_tip_geom_names = gripper_finger_tip_geom_names
        self._disable_collision_mesh_pairs = disable_collision_mesh_pairs
        self._home_arm_qpos = home_joint_qpos
        self._home_gripper_qpos = home_gripper_qpos
        self.num_actuators = num_actuators
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.num_fingers = num_fingers

    def end_effector_pose(self) -> npt.NDArray:
        assert self._mj_data is not None
        return create_transformation_matrix(
            self._mj_data.body(self._end_effector_body_name).xpos,
            self._mj_data.body(self._end_effector_body_name).xquat,
            rotation_type="quaternion",
        )

    def solve_ik(
        self,
        target_pose: npt.NDArray,
        robot_base_pose: Optional[npt.NDArray] = None,
        joint_qpos: Optional[npt.NDArray] = None,
    ) -> Optional[npt.NDArray]:
        if robot_base_pose is None:
            world_T_robot_base = self.base_body_pose()
        else:
            world_T_robot_base = robot_base_pose

        robot_base_T_target_pose = np.matmul(np.linalg.inv(world_T_robot_base), target_pose)
        solutions = self._ik_solver.solve_ik(
            target_position=extract_position(robot_base_T_target_pose),
            target_rotation=extract_rotation(robot_base_T_target_pose, rotation_type="matrix"),
            current_joint_angles=np.array(self.arm_joint_qpos() if joint_qpos is None else joint_qpos),
            free_joint_values=self._free_joint_value_for_ik(),
        )
        if solutions is None:
            return None

        for solusion in solutions:
            if not self._has_self_collision(solusion):
                return np.array(solusion)
        return None

    def apply_arm_control(self, actuator_vals: npt.NDArray, mj_ctrl_start_index: int = 0):
        assert self._mj_data is not None
        # Start index is for dual robot env
        # TODO: how to check if actuator_vals has the correct length.
        for i, val in enumerate(actuator_vals):
            self._mj_data.ctrl[mj_ctrl_start_index + i] = val if val.ndim == 0 else val[0]

    def _has_self_collision(self, joint_angles: npt.NDArray):
        transformed_joint_mesh, _ = self._transform_joint_mesh(self._joint_meshes, joint_angles)
        for pair in itertools.combinations(range(len(transformed_joint_mesh)), 2):
            if pair in self._disable_collision_mesh_pairs:
                continue
            if bool(transformed_joint_mesh[pair[0]].is_intersecting(transformed_joint_mesh[pair[1]])):
                return True
        return False

    def apply_gripper_control(self, actuator_vals: npt.NDArray):
        raise NotImplementedError

    def arm_joint_qpos(self) -> list[npt.NDArray]:
        return joint_qpos(self._mj_data, self._arm_joint_names)

    def gripper_joint_qpos(self) -> list[npt.NDArray]:
        return joint_qpos(self._mj_data, self._gripper_joint_names)

    def arm_joint_qvel(self) -> list[npt.NDArray]:
        return joint_qvel(self._mj_data, self._arm_joint_names)

    def gripper_joint_qvel(self) -> list[npt.NDArray]:
        return joint_qvel(self._mj_data, self._gripper_joint_names)

    def apply_arm_joint_qpos_and_qvel(
        self,
        arm_joint_qpos: npt.NDArray,
        arm_joint_velocities: Optional[list[npt.NDArray]] = None,
    ) -> None:
        apply_joint_qpos_and_qvel(self._mj_data, self._arm_joint_names, arm_joint_qpos, arm_joint_velocities)

    def apply_gripper_joint_qpos_and_qvel(
        self,
        gripper_joint_qpos: npt.NDArray,
        gripper_joint_velocities: Optional[list[npt.NDArray]] = None,
    ) -> None:
        apply_joint_qpos_and_qvel(
            self._mj_data,
            self._gripper_joint_names,
            gripper_joint_qpos,
            gripper_joint_velocities,
        )

    def appply_home_arm_qpos(self) -> None:
        apply_joint_qpos_and_qvel(self._mj_data, self._arm_joint_names, self._home_arm_qpos)

    def appply_home_gripper_qpos(self) -> None:
        apply_joint_qpos_and_qvel(self._mj_data, self._gripper_joint_names, self._home_gripper_qpos)

    def contact_gripper_positions(
        self, exclude_abstract_geom_names: list[str] = []
    ) -> tuple[dict[str, list[npt.NDArray]], dict[str, list[tuple[str, str]]]]:
        raise NotImplementedError

    @abstractmethod
    def _free_joint_value_for_ik(self) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def _transform_joint_mesh(
        self,
        joint_meshes: list[o3d.geometry.TriangleMesh],
        joint_angles: Union[list[float], npt.NDArray],
    ) -> tuple[o3d.geometry.TriangleMesh, npt.NDArray]:
        raise NotImplementedError

    @staticmethod
    def generate_mesh(
        arm_joint_qpos: npt.NDArray,
        gripper_joint_qpos: Optional[npt.NDArray] = None,
        robot_base_pose: npt.NDArray = np.eye(4),
    ):
        raise NotImplementedError

    @abstractmethod
    def gripper_open(self):
        raise NotImplementedError

    @abstractmethod
    def gripper_close(self):
        raise NotImplementedError
