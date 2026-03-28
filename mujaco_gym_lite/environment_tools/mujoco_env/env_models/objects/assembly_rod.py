import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.assembly_ring import AssemblyRing
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import extract_position


class AssemblyRod(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        assembly_rod_joint_names: list[str],
        assembly_rod_body_names: list[str],
        geom_root_name: str,
        rod_top_site_name: str,
        rod_bottom_site_name: str,
        insertion_height_threshold: float,
        insertion_radius_threshold: float,
    ):
        super().__init__(base_body_name, assembly_rod_joint_names, assembly_rod_body_names, geom_root_name)
        self._rod_bottom_site_name = rod_bottom_site_name
        self._rod_top_site_name = rod_top_site_name
        self._insertion_height_threshold = insertion_height_threshold
        self._insertion_radius_threshold = insertion_radius_threshold

    def is_ring_inserting(self, ring: AssemblyRing) -> tuple[bool, dict[str, npt.NDArray | bool]]:
        target_rod_position = self.rod_bottom_position()
        ring_position = ring.ring_position()
        # TODO: Support not xy plane settings
        is_inserted_height = (ring_position[2] - target_rod_position[2]) < self._insertion_height_threshold
        is_inserted_radius = bool(
            np.linalg.norm(target_rod_position[:2] - ring_position[:2]) < self._insertion_radius_threshold
        )
        return is_inserted_height and is_inserted_radius, {
            "ring_position": ring_position,
            "target_rod_position": target_rod_position,
            "is_inserted_height": is_inserted_height,
            "is_inserted_radius": is_inserted_radius,
        }

    def distance_to_target_rack(self, position: npt.NDArray) -> float:
        return float(np.linalg.norm(position - self.rod_bottom_position()))

    def rod_bottom_position(self) -> npt.NDArray:
        rod_bottom_pose = site_pose(self._mj_data, [self._rod_bottom_site_name])[0]
        return extract_position(rod_bottom_pose)

    def rod_top_position(self) -> npt.NDArray:
        rod_top_position = site_pose(self._mj_data, [self._rod_top_site_name])[0]
        return extract_position(rod_top_position)
