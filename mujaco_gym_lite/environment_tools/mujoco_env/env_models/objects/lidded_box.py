import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class Lid(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        lid_joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        lid_handle_geom_name: str,
        lid_handle_site_name: str,
        lid_handle_low_site_name: str,
        *,
        lid_max_position: npt.NDArray,
        lid_min_position: npt.NDArray,
    ):
        super().__init__(base_body_name, lid_joint_names, body_names, geom_root_name)
        self._lid_handle_geom_name = lid_handle_geom_name
        self._lid_handle_site_name = lid_handle_site_name
        self._lid_handle_low_site_name = lid_handle_low_site_name
        self._lid_max_position = lid_max_position
        self._lid_min_position = lid_min_position

    def handle_and_lid_distance(self) -> float:
        lid_pose = site_pose(self._mj_data, [self._lid_handle_site_name])[0]
        lid_position = extract_position(lid_pose)
        base_body_position = extract_position(self.body_pose()[0])
        return float(np.linalg.norm(lid_position - base_body_position))

    def distance_to_handle_center(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.handle_center_position()))

    def handle_center_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._lid_handle_site_name])[0]
        return extract_position(handle_pose)

    def handle_low_center_position(self) -> npt.NDArray:
        handle_low_pose = site_pose(self._mj_data, [self._lid_handle_low_site_name])[0]
        return extract_position(handle_low_pose)

    def handle_center_rotation(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._lid_handle_site_name])[0]
        return extract_rotation(handle_pose)


class OpenTopBox(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        body_names: list[str],
        geom_root_name: str,
        open_top_box_center_top_site_name: str,
    ):
        super().__init__(base_body_name, [], body_names, geom_root_name)
        self._open_top_box_center_top_site_name = open_top_box_center_top_site_name

    def open_top_center_position(self) -> npt.NDArray:
        site_pose_array = site_pose(self._mj_data, [self._open_top_box_center_top_site_name])
        return extract_position(site_pose_array[0])

    def is_covered(self, lid: Lid) -> bool:
        open_top_box_center_top_position = extract_position(
            site_pose(self._mj_data, [self._open_top_box_center_top_site_name])[0]
        )
        handle_position = lid.handle_center_position()

        # x,y distance between box top center and lid handle
        distance_xy = np.linalg.norm(open_top_box_center_top_position[:2] - handle_position[:2])
        xy_is_covered = distance_xy < 0.035

        if not xy_is_covered:
            return False

        z_is_covered = (handle_position[2] - lid.handle_and_lid_distance()) > open_top_box_center_top_position[2]

        if not z_is_covered:
            return False

        # check contact between box and lid
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._geom_root_name,
            geom2_abstract_name=lid._geom_root_name,
            exclude_abstract_names=[],
        )
        has_contact = bool(num_contact > 0)

        if not has_contact:
            return False

        return True
