from typing import TYPE_CHECKING, Union, cast

import numpy as np
import numpy.typing as npt

import mujoco

if TYPE_CHECKING:
    from mujaco_gym_lite.environment_tools.mujoco_env.env_models.camera import Camera, MocapCamera

from mujaco_gym_lite.utils.transforms import combine_quat, create_transformation_matrix, matrix_to_quat


def render_camera(
    camera: "Camera",
    color: bool,
    depth: bool,
    segmentation: bool,
    segmentation_object_names: tuple[str, ...],
) -> dict[str, Union[npt.NDArray, tuple[npt.NDArray, dict[int, str], dict[str, int]]]]:
    camera_info: dict[str, Union[npt.NDArray, tuple[npt.NDArray, dict[int, str], dict[str, int]]]] = {}

    camera_info["camera_intrinsic"] = camera.camera_intrinsic()
    camera_info["camera_extrinsic"] = camera.camera_extrinsic()

    if color:
        camera_info["color"] = camera.color()

    if segmentation:
        camera_info["segmentation"] = camera.segmentation(segmentation_object_names)

    if depth:
        camera_info["depth"] = camera.depth()

    return camera_info


def camera_observation(
    fixed_cameras: tuple["Camera", ...],
    mocap_cameras: tuple["MocapCamera", ...],
    render_color: bool,
    render_depth: bool,
    render_segmentation: bool,
    segmentation_object_names: tuple[str, ...],
) -> tuple[dict[str, npt.NDArray], dict[str, Union[dict[str, int], dict[int, str]]]]:
    camera_observation = {}
    camera_info: dict[str, Union[dict[str, int], dict[int, str]]] = {}
    for mocap_camera in mocap_cameras:
        render_info = render_camera(
            mocap_camera,
            color=render_color,
            depth=render_depth,
            segmentation=render_segmentation,
            segmentation_object_names=segmentation_object_names,
        )
        if render_depth:
            depth = cast(np.ndarray, render_info["depth"])
            camera_observation[f"camera/{mocap_camera.camera_name()}/depth"] = depth.astype(np.float32)

        if render_color:
            color = cast(np.ndarray, render_info["color"])
            camera_observation[f"camera/{mocap_camera.camera_name()}/color"] = color.astype(np.uint8)

        if render_segmentation:
            segmentations = cast(tuple[npt.NDArray, dict[int, str], dict[str, int]], render_info["segmentation"])
            segmentation_array, segmentation_id_to_name, name_to_segmentation_id = segmentations
            camera_observation[f"camera/{mocap_camera.camera_name()}/segmentation"] = segmentation_array
            camera_info[f"camera/{mocap_camera.camera_name()}/segmentation/id_to_name"] = segmentation_id_to_name
            camera_info[f"camera/{mocap_camera.camera_name()}/segmentation/name_to_id"] = name_to_segmentation_id

        camera_extrinsic = cast(np.ndarray, render_info["camera_extrinsic"])
        camera_observation[f"camera/{mocap_camera.camera_name()}/camera_extrinsic"] = camera_extrinsic.astype(
            np.float32
        )
        camera_intrinsic = cast(np.ndarray, render_info["camera_intrinsic"])
        camera_observation[f"camera/{mocap_camera.camera_name()}/camera_intrinsic"] = camera_intrinsic.astype(
            np.float32
        )

    for fixed_camera in fixed_cameras:
        render_info = render_camera(
            fixed_camera,
            color=render_color,
            depth=render_depth,
            segmentation=render_segmentation,
            segmentation_object_names=segmentation_object_names,
        )

        if render_depth:
            depth = cast(np.ndarray, render_info["depth"])
            camera_observation[f"camera/{fixed_camera.camera_name()}/depth"] = depth.astype(np.float32)

        if render_color:
            color = cast(np.ndarray, render_info["color"])
            camera_observation[f"camera/{fixed_camera.camera_name()}/color"] = color.astype(np.uint8)

        if render_segmentation:
            segmentations = cast(tuple[npt.NDArray, dict[int, str], dict[str, int]], render_info["segmentation"])
            segmentation_array, segmentation_id_to_name, name_to_segmentation_id = segmentations
            camera_observation[f"camera/{fixed_camera.camera_name()}/segmentation"] = segmentation_array
            camera_info[f"camera/{fixed_camera.camera_name()}/segmentation/id_to_name"] = segmentation_id_to_name
            camera_info[f"camera/{fixed_camera.camera_name()}/segmentation/name_to_id"] = name_to_segmentation_id

        camera_extrinsic = cast(np.ndarray, render_info["camera_extrinsic"])
        camera_observation[f"camera/{fixed_camera.camera_name()}/camera_extrinsic"] = camera_extrinsic.astype(
            np.float32
        )
        camera_intrinsic = cast(np.ndarray, render_info["camera_intrinsic"])
        camera_observation[f"camera/{fixed_camera.camera_name()}/camera_intrinsic"] = camera_intrinsic.astype(
            np.float32
        )

    return camera_observation, camera_info


def move_mocap_cameras(mocap_cameras: Union[list["MocapCamera"], tuple["MocapCamera", ...]], actions: npt.NDArray):
    assert len(actions) == len(mocap_cameras)
    for action, camera in zip(actions, mocap_cameras):
        assert len(action) == 7
        camera.move(
            create_transformation_matrix(
                action[:3],
                action[3:],
                rotation_type="quaternion",
            )
        )


def opencv_camera_to_mujoco_camera(opencv_quat: npt.NDArray) -> npt.NDArray:
    # NOTE:
    # Mujoco's camera pose is not correct, we should add base transformation to fit the normal camera definition
    # x_180_quat
    base_quat = np.array([np.cos(np.pi * 0.5), np.sin(np.pi * 0.5), 0.0, 0.0])
    mujoco_quat = combine_quat(opencv_quat, base_quat)
    return mujoco_quat


def camera_extrinsic(mj_data: "mujoco.mjData", camera_name: str) -> npt.NDArray:
    camera_position = mj_data.cam(camera_name).xpos.ravel().copy()
    camera_rotation = matrix_to_quat(mj_data.cam(camera_name).xmat.ravel().copy().reshape(3, 3))
    # NOTE:
    # Mujoco's camera pose is not correct, we should add base transformation to fit the normal camera definition
    # x_-180_quat
    base_quat = np.array([np.cos(-np.pi * 0.5), np.sin(-np.pi * 0.5), 0.0, 0.0])
    camera_rotation = combine_quat(camera_rotation, base_quat)
    return np.linalg.inv(create_transformation_matrix(camera_position, camera_rotation, rotation_type="quaternion"))


def camera_intrinsic(mj_model: "mujoco.mjModel", camera_name: str) -> npt.NDArray:
    """get camera intrinsic
    Returns:
        K (npt.NDArray): camera_intrinsic matrix, K = [[fx  0 cx], [ 0 fy cy], [ 0  0  1]]
    """
    width = mj_model.vis.global_.offwidth
    height = mj_model.vis.global_.offheight

    camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = mj_model.cam_fovy[camera_id]
    focal_scaling = (1.0 / np.tan(np.deg2rad(fovy) / 2)) * height / 2.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[focal_scaling, 0.0, cx], [0.0, focal_scaling, cy], [0.0, 0.0, 1.0]])
    return K
