from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d

from mujaco_gym_lite.environment_tools.ik_solvers.ik_sovler import InverseKinematicsSolver, J2n6s300IKSolver
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.assets.robots.jaco2 import (
    load_j2n6s300_mesh,
    transform_j2n6s300_mesh,
)
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.robot import Robot
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_of_geom_names
from mujaco_gym_lite.utils.solids import fusion_mesh


class J2n6s300(Robot):
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
        apply_distal_finger_control: bool = False,
        joint_meshes: list[o3d.geometry.TriangleMesh] = load_j2n6s300_mesh(),
        ik_solver: InverseKinematicsSolver = J2n6s300IKSolver(),
        # https://github.com/Kinovarobotics/kinova-ros/blob/924781d3dfe241b2b94b3c72a804b80d3658cf02/kinova_moveit/robot_configs/j2n6s300_moveit_config/config/j2n6s300.srdf#L26
        disable_collision_mesh_pairs: list[tuple[int, int]] = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ],
        home_joint_qpos: list[float] = [
            -1.52721949,
            3.58571133,
            1.96776086,
            -0.60883485,
            1.85972993,
            -0.55478329,
        ],
        home_gripper_qpos: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ):
        super().__init__(
            base_body_name,
            arm_joint_names,
            arm_body_names,
            gripper_joint_names,
            gripper_body_names,
            geom_root_name,
            gripper_finger_tip_geom_names,
            end_effector_body_name,
            ik_solver,
            joint_meshes,
            disable_collision_mesh_pairs,
            home_joint_qpos,
            home_gripper_qpos,
            num_actuators=8,
        )
        self._apply_distal_finger_control = apply_distal_finger_control

    def gripper_open(self, rate: float = 25.0):
        self._mj_data.ctrl[-2] = rate
        self._mj_data.ctrl[-1] = rate

    def gripper_close(self, rate: float = 255.0):
        # get contact info and if it has a contact then apply second actuator
        has_contact_info = self.has_finger_contact()
        has_contact = (has_contact_info["finger1"] and has_contact_info["finger2"]) or (
            has_contact_info["finger1"] and has_contact_info["finger3"]
        )
        if has_contact and self._apply_distal_finger_control:
            self._mj_data.ctrl[-1] = 255.0
        else:
            self._mj_data.ctrl[-1] = 0.0

        self._mj_data.ctrl[-2] = 255.0

    def has_finger_contact(self) -> dict[str, bool]:
        has_contact = {}
        for key, geom_names in self._gripper_finger_tip_geom_names.items():
            num_contact, contact_geoms, _ = get_contact_info_of_geom_names(
                self._mj_model,
                self._mj_data,
                geom_names,
                exclude_geom_names=self._get_exclude_names(key),
            )

            only_finger_contact = False
            for geoms in contact_geoms:
                assert len(geoms) == 2, f"{geoms}! length of geoms is {len(geoms)}"
                finger1 = any(["finger1" in g for g in geoms])
                finger2 = any(["finger2" in g for g in geoms])
                finger3 = any(["finger3" in g for g in geoms])
                only_finger_contact |= bool(np.sum([finger1, finger2, finger3]) >= 2)

            if num_contact > 0 and (not only_finger_contact):
                has_contact[key] = True
            else:
                has_contact[key] = False
        return has_contact

    def contact_gripper_positions(
        self, exclude_abstract_geom_names: list[str] = []
    ) -> tuple[dict[str, list[npt.NDArray]], dict[str, list[tuple[str, str]]]]:
        contact_positions = {}
        contact_names = {}
        for key, geom_names in self._gripper_finger_tip_geom_names.items():
            num_contact, contact_name, position = get_contact_info_of_geom_names(
                self._mj_model,
                self._mj_data,
                geom_names,
                exclude_geom_names=self._get_exclude_names(key),
                exclude_abstract_geom_names=exclude_abstract_geom_names,
            )
            if num_contact > 0:
                contact_positions[key] = position
                contact_names[key] = contact_name
            else:
                contact_positions[key] = []
                contact_names[key] = []

        return contact_positions, contact_names

    def _get_exclude_names(self, curr_key: str) -> list[str]:
        exclude_names = []
        for key, geom_names in self._gripper_finger_tip_geom_names.items():
            if curr_key != key:
                exclude_names.extend(geom_names)
        return exclude_names

    def apply_gripper_control(self, actuator_vals: npt.NDArray):
        # TODO: Support each finger can have different values
        assert len(actuator_vals) == 2
        self._mj_data.ctrl[-2] = actuator_vals[0]

        has_contact_info = self.has_finger_contact()
        has_contact = (has_contact_info["finger1"] and has_contact_info["finger2"]) or (
            has_contact_info["finger1"] and has_contact_info["finger3"]
        )
        if has_contact and self._apply_distal_finger_control:
            self._mj_data.ctrl[-1] = actuator_vals[-1]
        else:
            self._mj_data.ctrl[-1] = 0.0

    def appply_home_gripper_qpos(self) -> None:
        super().appply_home_gripper_qpos()

        # NOTE: position control so apply ctrl value
        max_gripper_joint = np.array([1.51, 1.0])
        self.apply_gripper_control(self._home_gripper_qpos[:2] / max_gripper_joint * 255.0)

    def _free_joint_value_for_ik(self):
        return []

    def _transform_joint_mesh(
        self,
        joint_meshes: list[o3d.geometry.TriangleMesh],
        joint_angles: Union[list[float], npt.NDArray],
    ) -> tuple[o3d.geometry.TriangleMesh, npt.NDArray]:
        return transform_j2n6s300_mesh(joint_meshes, joint_angles, visualize=False)

    @staticmethod
    def generate_mesh(
        arm_joint_qpos: npt.NDArray,
        gripper_joint_qpos: Optional[npt.NDArray] = None,
        robot_base_pose: npt.NDArray = np.eye(4),
    ):
        if gripper_joint_qpos is None:
            mesh, _ = transform_j2n6s300_mesh(load_j2n6s300_mesh("collision"), arm_joint_qpos, visualize=False)
        else:
            mesh, _ = transform_j2n6s300_mesh(
                load_j2n6s300_mesh("collision", load_finger=True),
                np.concatenate([arm_joint_qpos, gripper_joint_qpos]),
                with_finger=True,
                visualize=False,
            )
        return fusion_mesh(mesh).transform(robot_base_pose)
