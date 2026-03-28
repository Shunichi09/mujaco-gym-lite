from typing import Union

import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position


class AssemblyRing(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        ring_joint_names: list[str],
        ring_body_names: list[str],
        geom_root_name: str,
        ring_site_name: str,
        ring_handle_site_name: str,
    ):
        super().__init__(base_body_name, ring_joint_names, ring_body_names, geom_root_name)
        self._ring_site_name = ring_site_name
        self._ring_handle_site_name = ring_handle_site_name

    def ring_position(self) -> npt.NDArray:
        ring_pose = site_pose(self._mj_data, [self._ring_site_name])[0]
        return extract_position(ring_pose)

    def ring_handle_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._ring_handle_site_name])[0]
        return extract_position(handle_pose)

    def has_handle_touch(
        self, abstract_geom_name: str, exclude_geom_names: list[str] = []
    ) -> tuple[bool, dict[str, Union[list[npt.NDArray], list[tuple[str, str]]]]]:
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._geom_root_name,
            geom2_abstract_name=abstract_geom_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0), {
            "assembly_ring/contact_positions": contact_points,
            "assembly_ring/contact_names": contact_names,
        }
