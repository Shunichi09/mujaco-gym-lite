from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class Button(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        button_joint_name: str,
        button_body_names: list[str],
        geom_root_name: str,
        max_button_pos: float,
        button_geom_name: str,
        button_site_name: str,
        push_thredshold: float = 0.0075,
    ):
        super().__init__(base_body_name, [button_joint_name], button_body_names, geom_root_name)
        self._max_button_pos = max_button_pos
        self._button_geom_name = button_geom_name
        self._button_site_name = button_site_name
        self._push_threshold = push_thredshold

    def is_pushed(self):
        button_joint_pos = self.joint_qpos()[0]
        return bool(button_joint_pos <= -1 * self._max_button_pos + self._push_threshold)

    def reset_button(self):
        # TODO: Always resetting 0.0 it OK??
        self.apply_joint_qpos_and_qvel([0.0])

    def has_button_touch(
        self, abstract_geom_name: str, exclude_geom_names: list[str] = []
    ) -> tuple[bool, dict[str, Union[list[npt.NDArray], list[tuple[str, str]]]]]:
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._button_geom_name,
            geom2_abstract_name=abstract_geom_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0), {
            "button/contact_positions": contact_points,
            "button/contact_names": contact_names,
        }

    def distance_to_button_surface_center(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.button_surface_center_position()))

    def button_surface_center_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._button_site_name])[0]
        return extract_position(handle_pose)

    def button_surface_center_rotation(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._button_site_name])[0]
        return extract_rotation(handle_pose)
