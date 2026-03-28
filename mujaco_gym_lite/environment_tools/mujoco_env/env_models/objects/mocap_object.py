from typing import Optional

import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class MocapObject(EnvModel):
    _mj_model: Optional[mujoco.MjModel]
    _mj_data: Optional[mujoco.MjData]

    def __init__(
        self,
        base_body_name: str,
        joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        mocap_idx: int,
    ):
        super().__init__(base_body_name, joint_names, body_names, geom_root_name)
        self._mocap_idx = mocap_idx

    def move(self, target_pose: npt.NDArray):
        assert target_pose.shape == (4, 4)
        assert self._mj_data is not None
        target_position = extract_position(target_pose)
        target_opencv_quaternion = extract_rotation(target_pose)
        self._mj_data.mocap_pos[self._mocap_idx, :] = target_position.copy()
        self._mj_data.mocap_quat[self._mocap_idx, :] = target_opencv_quaternion.copy()


"""
    def approximate_bbox(self) -> npt.NDArray:
        body_pose = self.body_pose()
        assert len(body_pose) == 1
        position = extract_position(body_pose[0])
        bbox_3d = np.array(
            [
                [
                    position[0] - self._width / 2,
                    position[1] + self._height / 2,
                    position[2] - self._depth / 2,
                ],  # Front up left
                [
                    position[0] + self._width / 2,
                    position[1] + self._height / 2,
                    position[2] - self._depth / 2,
                ],  # Front up right
                [
                    position[0] - self._width / 2,
                    position[1] - self._height / 2,
                    position[2] - self._depth / 2,
                ],  # Front down left
                [
                    position[0] + self._width / 2,
                    position[1] - self._height / 2,
                    position[2] - self._depth / 2,
                ],  # Front down right
                [
                    position[0] - self._width / 2,
                    position[1] + self._height / 2,
                    position[2] + self._depth / 2,
                ],  # Back up left
                [
                    position[0] + self._width / 2,
                    position[1] + self._height / 2,
                    position[2] + self._depth / 2,
                ],  # Back up right
                [
                    position[0] - self._width / 2,
                    position[1] - self._height / 2,
                    position[2] + self._depth / 2,
                ],  # Back down left
                [
                    position[0] + self._width / 2,
                    position[1] - self._height / 2,
                    position[2] + self._depth / 2,
                ],  # Back down right
            ]
        )
        return bbox_3d
"""
