from typing import Optional, Union

import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import (
    get_contact_info_between_abstract_geom_names,
    get_contact_info_of_model,
)
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import (
    apply_joint_qpos_and_qvel,
    body_pose,
    joint_qpos,
    joint_qvel,
)
from mujaco_gym_lite.utils.transforms import create_transformation_matrix


class EnvModel:
    _mj_model: Optional["mujoco.MjModel"]
    _mj_data: Optional["mujoco.MjData"]

    def __init__(
        self,
        base_body_name: str,
        joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
    ) -> None:
        self._base_body_name = base_body_name
        self._joint_names = joint_names
        self._body_names = body_names
        self._geom_root_name = geom_root_name
        self._mj_model = None
        self._mj_data = None

    def geom_root_name(self) -> str:
        return self._geom_root_name

    def load_mj_model_and_data(self, mj_model: "mujoco.mjModel", mj_data: "mujoco.mjData"):
        self._mj_model = mj_model
        self._mj_data = mj_data

    def has_contact(self, exclude_geom_names: list[str]) -> bool:
        num_contact, _, _ = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._geom_root_name,
            geom2_abstract_name="",
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0)

    def contact_positions(self, exclude_geom_names: list[str]) -> list[npt.NDArray]:
        _, _, contact_positions = get_contact_info_of_model(
            self._mj_model,
            self._mj_data,
            self._geom_root_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return contact_positions

    def joint_qpos(self) -> list[npt.NDArray]:
        return joint_qpos(self._mj_data, self._joint_names)

    def joint_qvel(self) -> list[npt.NDArray]:
        return joint_qvel(self._mj_data, self._joint_names)

    def base_body_pose(self) -> npt.NDArray:
        assert self._mj_data is not None
        return create_transformation_matrix(
            self._mj_data.body(self._base_body_name).xpos,
            self._mj_data.body(self._base_body_name).xquat,
            rotation_type="quaternion",
        )

    def body_pose(self) -> list[npt.NDArray]:
        return body_pose(self._mj_data, self._body_names)

    def apply_joint_qpos_and_qvel(
        self,
        joint_qpos: Union[list[npt.NDArray], npt.NDArray, list[float]],
        joint_velocities: Optional[list[npt.NDArray]] = None,
    ):
        apply_joint_qpos_and_qvel(self._mj_data, self._joint_names, joint_qpos, joint_velocities)
