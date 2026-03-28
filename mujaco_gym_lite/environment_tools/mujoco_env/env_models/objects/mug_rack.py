import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mug import Mug
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position


class MugRack(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        mug_rack_joint_names: list[str],
        mug_rack_body_names: list[str],
        geom_root_name: str,
        target_rack_site_name: str,
    ):
        super().__init__(base_body_name, mug_rack_joint_names, mug_rack_body_names, geom_root_name)
        self._target_rack_site_name = target_rack_site_name

    def mug_is_hanging(self, mug: Mug) -> tuple[bool, dict[str, list[npt.NDArray]]]:
        mug_rack_position = extract_position(self.body_pose()[0])
        mug_position = extract_position(mug.body_pose()[0])

        # NOTE: assuming z-axis vertical directions
        has_height = mug_position[2] > 0.1
        within_range = (
            np.abs(mug_position[0] - mug_rack_position[0]) < 0.01
            or np.abs(mug_position[1] - mug_rack_position[1]) < 0.01
        )
        num_contact, _, contact_positions = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._geom_root_name,
            geom2_abstract_name=mug.geom_root_name(),
            exclude_abstract_names=[],
        )
        return has_height and within_range and num_contact > 0, {"contact_positions": contact_positions}

    def distance_to_target_rack(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.target_rack_position()))

    def target_rack_position(self) -> npt.NDArray:
        target_rack_pose = site_pose(self._mj_data, [self._target_rack_site_name])[0]
        return extract_position(target_rack_pose)
