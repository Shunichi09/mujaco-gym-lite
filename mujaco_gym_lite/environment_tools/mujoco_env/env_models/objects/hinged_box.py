import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class HingedBox(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        hinged_box_joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        hinged_box_joint_range: tuple[float, float],
        hinged_box_handle_geom_name: str,
        hinged_box_handle_site_name: str,
        close_threshold: float = 0.01,
    ):
        super().__init__(base_body_name, hinged_box_joint_names, body_names, geom_root_name)
        self._hinged_box_joint_range = hinged_box_joint_range
        self._hinged_box_handle_geom_name = hinged_box_handle_geom_name
        self._hinged_box_handle_site_name = hinged_box_handle_site_name
        self._close_threshold = close_threshold

    def is_close(self):
        hinged_box_joint_pos = self.joint_qpos()[0]
        return np.abs(hinged_box_joint_pos - self._hinged_box_joint_range[1]) < self._close_threshold

    def is_open(self, target_angle=None):
        assert len(self.joint_qpos()) == 1
        hinged_box_joint_pos = self.joint_qpos()[0]
        if target_angle is None:
            target_angle = self._hinged_box_joint_range[0]
        else:
            assert target_angle < 0.0
        return hinged_box_joint_pos < target_angle

    def open(self):
        self.apply_joint_qpos_and_qvel([self._hinged_box_joint_range[1]])

    def close(self):
        self.apply_joint_qpos_and_qvel([self._hinged_box_joint_range[0]])

    def distance_to_handle_center(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.handle_center_position()))

    def handle_center_position(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._hinged_box_handle_site_name])[0]
        return extract_position(handle_pose)

    def handle_center_rotation(self) -> npt.NDArray:
        handle_pose = site_pose(self._mj_data, [self._hinged_box_handle_site_name])[0]
        return extract_rotation(handle_pose)
