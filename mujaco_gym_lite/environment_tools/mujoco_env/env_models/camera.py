from typing import Optional

import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.camera import (
    camera_extrinsic,
    camera_intrinsic,
    opencv_camera_to_mujoco_camera,
)
from mujaco_gym_lite.environment_tools.mujoco_env.functions.depth import zbuffer_to_depth
from mujaco_gym_lite.environment_tools.mujoco_env.functions.segmentation import (
    object_segmentation_from_segmentation_buffer,
)
from mujaco_gym_lite.environment_tools.mujoco_env.renderer import MujocoRenderer
from mujaco_gym_lite.utils.images import remove_small_regions
from mujaco_gym_lite.utils.transforms import extract_position, extract_rotation


class Camera(EnvModel):
    _renderer: Optional[MujocoRenderer]

    def __init__(
        self,
        base_body_name: str,
        joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        camera_name: str,
    ):
        super().__init__(base_body_name, joint_names, body_names, geom_root_name)
        self._renderer = None
        self._camera_name = camera_name

    def load_renderer(self, renderer: MujocoRenderer):
        self._renderer = renderer

    def camera_name(self) -> str:
        return self._camera_name

    def camera_extrinsic(self) -> npt.NDArray:
        return camera_extrinsic(self._mj_data, self._camera_name)

    def camera_intrinsic(self) -> npt.NDArray:
        return camera_intrinsic(self._mj_model, self._camera_name)

    def color(self) -> npt.NDArray:
        assert self._renderer is not None
        color = self._renderer.render("rgb_array", None, self._camera_name)
        assert color is not None
        return color

    def segmentation(
        self, segmentation_object_names: tuple[str, ...], min_segmentation_area_size: int = 5
    ) -> tuple[npt.NDArray, dict[int, str], dict[str, int]]:
        assert self._renderer is not None
        # NOTE:
        # mujoco_type_image = segmentation_array[:, :, 0]
        # mujoco_id_image = segmentation_array[:, :, 1]
        segmentation_array = self._renderer.render("segmentation_array", None, self._camera_name)
        assert segmentation_array is not None
        segmentation, segmentation_id_to_name, name_to_segmentation_id = object_segmentation_from_segmentation_buffer(
            self._mj_model, segmentation_array, segmentation_object_names
        )
        remove_segmentation = remove_small_regions(segmentation, min_size=min_segmentation_area_size)
        return remove_segmentation, segmentation_id_to_name, name_to_segmentation_id

    def depth(self):
        assert self._renderer is not None
        depth = self._renderer.render("depth_array", None, self._camera_name)
        assert depth is not None
        return zbuffer_to_depth(self._mj_model, depth_buffer=depth)


class MocapCamera(Camera):
    _mj_model: Optional["mujoco.MjModel"]
    _mj_data: Optional["mujoco.MjData"]

    def __init__(
        self,
        base_body_name: str,
        joint_names: list[str],
        body_names: list[str],
        geom_root_name: str,
        camera_name: str,
        mocap_idx: int,
    ):
        super().__init__(base_body_name, joint_names, body_names, geom_root_name, camera_name)
        self._mocap_idx = mocap_idx

    def move(self, target_pose: npt.NDArray):
        assert self._mj_data is not None
        target_position = extract_position(target_pose)
        target_opencv_quaternion = extract_rotation(target_pose)
        self._mj_data.mocap_pos[self._mocap_idx, :] = target_position.copy()
        self._mj_data.mocap_quat[self._mocap_idx, :] = opencv_camera_to_mujoco_camera(target_opencv_quaternion).copy()
