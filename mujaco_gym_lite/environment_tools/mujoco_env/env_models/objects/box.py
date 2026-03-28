import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.utils.solids import are_points_inside_bbox


class TargetEmptyBox(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        bboxes: npt.NDArray,
    ):
        super().__init__(base_body_name, joint_names, body_names, geom_root_name)
        self._bboxes = np.array(bboxes)
        assert len(self._bboxes) == 8
        self._center_point = np.mean(bboxes, axis=0)

    def is_inside(self, points: npt.NDArray) -> bool:
        assert len(points.shape) == 2
        points_are_inside = are_points_inside_bbox(points, self._bboxes)
        return bool(np.all(points_are_inside))

    def distance_to_center(self, point: npt.NDArray) -> float:
        assert point.shape == (3,)
        return float(np.linalg.norm(point - self._center_point))
