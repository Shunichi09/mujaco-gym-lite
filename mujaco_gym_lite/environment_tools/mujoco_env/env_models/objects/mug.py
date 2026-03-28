from typing import Union

import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.utils.transforms import extract_position


class Mug(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        mug_joint_names: list[str],
        mug_body_names: list[str],
        geom_root_name: str,
        mug_height: float,  # NOTE: Assume mug tree define in under the
    ):
        super().__init__(base_body_name, mug_joint_names, mug_body_names, geom_root_name)
        self._mug_height = mug_height

    def center_position(self) -> npt.NDArray:
        mug_pose = self.body_pose()[0]
        mug_pose[2] += self._mug_height
        return extract_position(mug_pose)

    def has_touch(
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
            "mug/contact_positions": contact_points,
            "mug/contact_names": contact_names,
        }
