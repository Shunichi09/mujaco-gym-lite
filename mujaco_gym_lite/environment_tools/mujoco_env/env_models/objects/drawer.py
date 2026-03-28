from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class Drawer(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        drawer_joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        drawer_joint_range: tuple[float, float],
        drawer_handle_geom_name: str,
        drawer_handle_site_name: str,
        open_and_close_threshold: float = 0.015,
    ):
        super().__init__(base_body_name, drawer_joint_names, body_names, geom_root_name)
        self._drawer_joint_range = drawer_joint_range
        self._drawer_handle_geom_name = drawer_handle_geom_name
        self._drawer_handle_site_name = drawer_handle_site_name
        self._open_and_close_thredshold = open_and_close_threshold

    def is_close(self):
        drawer_joint_pos = self.joint_qpos()[0]
        return np.abs(drawer_joint_pos - self._drawer_joint_range[0]) < self._open_and_close_thredshold

    def is_open(self):
        drawer_joint_pos = self.joint_qpos()[0]
        return np.abs(drawer_joint_pos - self._drawer_joint_range[1]) < self._open_and_close_thredshold

    def open(self):
        self.apply_joint_qpos_and_qvel([self._drawer_joint_range[1]])

    def close(self):
        self.apply_joint_qpos_and_qvel([self._drawer_joint_range[0]])

    def has_handle_touch(
        self, abstract_geom_name: str, exclude_geom_names: list[str] = []
    ) -> tuple[bool, dict[str, Union[list[npt.NDArray], list[tuple[str, str]]]]]:
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._drawer_handle_geom_name,
            geom2_abstract_name=abstract_geom_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0), {
            "drawer/contact_positions/handle": contact_points,
            "drawer/contact_names/handle": contact_names,
        }

    def distance_to_handle_center(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.handle_center_position()))

    def handle_center_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._drawer_handle_site_name])[0]
        return extract_position(handle_pose)

    def handle_center_rotation(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._drawer_handle_site_name])[0]
        return extract_rotation(handle_pose)
