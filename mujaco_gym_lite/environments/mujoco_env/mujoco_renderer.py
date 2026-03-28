import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.renderer import MujocoRenderer
from mujaco_gym_lite.utils.images import object_segmentation_from_segmentation_buffer, remove_small_regions
from mujaco_gym_lite.utils.transforms import combine_quat


def opencv_camera_to_mujoco_camera(opencv_quat: npt.NDArray) -> npt.NDArray:
    # NOTE:
    # Mujoco's camera pose is not correct, we should add base transformation to fit the normal camera definition
    # x_180_quat
    base_quat = np.array([np.cos(np.pi * 0.5), np.sin(np.pi * 0.5), 0.0, 0.0])
    mujoco_quat = combine_quat(opencv_quat, base_quat)
    return mujoco_quat


class MujocoQposQvelMocapCameraRender:
    def __init__(
        self,
        xml_file_path: pathlib.Path,
        mocap_camera_index: int = 0,
        mocap_camera_name: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        max_geom: int = 1000,
        default_camera_config: Optional[dict] = None,
    ):
        self._xml_file_path = xml_file_path
        self._width = width
        self._height = height
        self._initialize_simulation()
        self._mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config=default_camera_config,
            width=width,
            height=height,
            max_geom=max_geom,
        )
        self._mocap_camera_name = mocap_camera_name
        self._mocap_camera_index = mocap_camera_index

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(str(self._xml_file_path))
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self._width
        self.model.vis.global_.offheight = self._height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(
        self,
        qpos: npt.NDArray,
        qvel: npt.NDArray,
        camera_position: npt.NDArray,
        camera_rotation: npt.NDArray,
    ):
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        self.data.mocap_pos[self._mocap_camera_index, :] = camera_position.copy()
        self.data.mocap_quat[self._mocap_camera_index, :] = opencv_camera_to_mujoco_camera(camera_rotation).copy()
        mujoco.mj_forward(self.model, self.data)

    def render_color(
        self, qpos: npt.NDArray, qvel: npt.NDArray, camera_pose: tuple[npt.NDArray, npt.NDArray]
    ) -> npt.NDArray:
        self.set_state(qpos, qvel, camera_position=camera_pose[0], camera_rotation=camera_pose[1])
        return self._mujoco_renderer.render("rgb_array", None, self._mocap_camera_name)  # type: ignore

    def render_color_all(
        self,
        qpos: list[npt.NDArray],
        qvel: list[npt.NDArray],
        camera_pose: tuple[npt.NDArray, npt.NDArray],
    ) -> list[npt.NDArray]:
        assert len(qpos) == len(qvel)

        if qpos[0].ndim < 2 and qvel[0].ndim < 2:
            qpos = [qpos]
            qvel = [qvel]

        frames = []
        for qp, qv in zip(qpos, qvel):
            assert len(qp) == len(qv)
            for p, v in zip(qp, qv):
                frames.append(self.render_color(p, v, camera_pose))
        return frames

    def render_segmentation(
        self,
        qpos: npt.NDArray,
        qvel: npt.NDArray,
        camera_pose: tuple[npt.NDArray, npt.NDArray],
        segmentation_object_names: list[str],
        min_segmentation_area_size: int = 10,
    ) -> tuple[npt.NDArray, dict[int, str], dict[str, int]]:
        self.set_state(qpos, qvel, camera_position=camera_pose[0], camera_rotation=camera_pose[1])
        segmentation_array = self._mujoco_renderer.render("segmentation_array", None, self._mocap_camera_name)
        segmentation, segmentation_id_to_name, name_to_segmentation_id = object_segmentation_from_segmentation_buffer(
            self.model, segmentation_array, segmentation_object_names
        )
        remove_segmentation = remove_small_regions(segmentation, min_size=min_segmentation_area_size)
        return remove_segmentation, segmentation_id_to_name, name_to_segmentation_id

    def render_segmentation_all(
        self,
        qpos: list[npt.NDArray],
        qvel: list[npt.NDArray],
        camera_pose: tuple[npt.NDArray, npt.NDArray],
        segmentation_object_names: list[str],
        min_segmentation_area_size: int = 10,
    ) -> list[tuple[npt.NDArray, dict[int, str], dict[str, int]]]:
        assert len(qpos) == len(qvel)

        if qpos[0].ndim < 2 and qvel[0].ndim < 2:
            qpos = [qpos]
            qvel = [qvel]

        frames = []
        for qp, qv in zip(qpos, qvel):
            assert len(qp) == len(qv)
            for p, v in zip(qp, qv):
                frames.append(
                    self.render_segmentation(p, v, camera_pose, segmentation_object_names, min_segmentation_area_size)
                )
        return frames

    def reset(self):
        self._reset_simulation()

    def close(self):
        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
