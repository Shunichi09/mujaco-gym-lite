from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class Lever(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        lever_joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        lever_joint_range: tuple[float, float],
        lever_handle_geom_name: str,
        lever_handle_site_name: str,
        up_threshold: float = 0.015,
    ):
        super().__init__(base_body_name, lever_joint_names, body_names, geom_root_name)
        self._lever_joint_range = lever_joint_range
        self._lever_handle_geom_name = lever_handle_geom_name
        self._lever_handle_site_name = lever_handle_site_name
        self._up_threshold = up_threshold

    def is_down(self, target_angle: float) -> bool:
        assert target_angle > 0.0
        assert len(self.joint_qpos()[0]) == 1
        lever_joint_pos = self.joint_qpos()[0]
        return lever_joint_pos > target_angle

    def is_up(self, target_angle: float) -> bool:
        assert target_angle > 0.0
        assert len(self.joint_qpos()[0]) == 1
        lever_joint_pos = self.joint_qpos()[0]
        return np.abs(lever_joint_pos - target_angle) < self._up_threshold

    def up(self):
        self.apply_joint_qpos_and_qvel([self._lever_joint_range[1]])

    def down(self):
        self.apply_joint_qpos_and_qvel([self._lever_joint_range[0]])

    def has_handle_touch(
        self, abstract_geom_name: str, exclude_geom_names: list[str] = []
    ) -> tuple[bool, dict[str, Union[list[npt.NDArray], list[tuple[str, str]]]]]:
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._lever_handle_geom_name,
            geom2_abstract_name=abstract_geom_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0), {
            "lever/contact_positions/handle": contact_points,
            "lever/contact_names/handle": contact_names,
        }

    def distance_to_handle_center(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.handle_center_position()))

    def handle_center_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._lever_handle_site_name])[0]
        return extract_position(handle_pose)

    def handle_center_rotation(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._lever_handle_site_name])[0]
        return extract_rotation(handle_pose)
