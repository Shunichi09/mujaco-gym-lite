import pathlib
from typing import Union

import cv2
import numpy as np
import numpy.typing as npt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.ndimage import label as scipy_label

import mujoco
from mujaco_gym_lite.logger import logger
from mujaco_gym_lite.random import np_drng
from mujaco_gym_lite.utils.transforms import transform_position


def segmentation_map_to_segmentation_image(segmentation_map: npt.NDArray, skip_ids: list[int] = [0]) -> npt.NDArray:
    assert len(segmentation_map.shape) == 2
    segmentation_image = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for i in np.unique(segmentation_map):
        if i in skip_ids:
            continue
        segmentation_image[segmentation_map == i] = np.array(np_drng.integers(255, size=3), dtype=np.uint8)
    return segmentation_image


def point_to_camera_coordinate(point: npt.NDArray, camera_intrinsic: npt.NDArray, camera_extrinsic: npt.NDArray):
    point_at_camera_3d = transform_position(point, np.linalg.inv(camera_extrinsic))
    point_at_camera_2d = np.matmul(camera_intrinsic, point_at_camera_3d[:, np.newaxis])

    if point_at_camera_3d[2] <= 0.0001:
        logger.info("Point is behind the camera or too close to the projection plane.")
        return None, None

    width = point_at_camera_2d[0] / point_at_camera_2d[2]
    height = point_at_camera_2d[1] / point_at_camera_2d[2]
    return int(width), int(height)


def images_to_mp4(frames: Union[list[npt.NDArray], npt.NDArray], video_file_path: pathlib.Path, fps: int):
    clip = ImageSequenceClip(frames, fps=fps)
    moviepy_logger = "bar"
    clip.write_videofile(str(video_file_path), logger=moviepy_logger)


def add_border(
    image: npt.NDArray, border_thickness: int = 5, color: tuple[int, int, int] = (255, 255, 255)
) -> npt.NDArray:
    bordered_image = cv2.copyMakeBorder(
        image,
        top=border_thickness,
        bottom=border_thickness,
        left=border_thickness,
        right=border_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=color,  # White color
    )
    return bordered_image


def draw_transparent_text(image, text, alpha=0.5, font_scale=1.5, font_thickness=2, color=(255, 255, 255)):
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    position = (width - text_width - 20, height - text_height - 1)

    overlay = image.copy()
    cv2.putText(overlay, text, position, font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def sample_and_concatenate_image_pair_horizontal(
    traj1_images: npt.NDArray,
    traj2_images: npt.NDArray,
    num_parts: int,
    resize_ratio: float,
    left_frame_color=(255, 0, 0),
    right_frame_color=(0, 0, 255),
) -> npt.NDArray:
    def _divided_indices(n: int, num_parts: int) -> npt.NDArray:
        step = n / num_parts
        return np.array([min(n - 1, int(round(i * step))) for i in range(num_parts + 1)])

    def resize_image(image: npt.NDArray, ratio: float) -> npt.NDArray:
        new_shape = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))  # OpenCV expects (width, height)
        return cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

    traj1_indices = _divided_indices(traj1_images.shape[0], num_parts=num_parts)
    traj2_indices = _divided_indices(traj2_images.shape[0], num_parts=num_parts)

    traj1_concated = np.hstack(
        [
            add_border(
                draw_transparent_text(resize_image(traj1_images[j], resize_ratio), f"t = {i}"), color=left_frame_color
            )
            for i, j in enumerate(traj1_indices)
        ]
    )
    traj2_concated = np.hstack(
        [
            add_border(
                draw_transparent_text(resize_image(traj2_images[j], resize_ratio), f"t = {i}"), color=right_frame_color
            )
            for i, j in enumerate(traj2_indices)
        ]
    )

    black_line = np.zeros((15, traj1_concated.shape[1], traj1_concated.shape[2]), dtype=traj1_concated.dtype)
    concated_image = np.vstack([traj1_concated, black_line, traj2_concated])

    return concated_image


def sample_and_concatenate_image_pair_vertical(
    traj1_images: npt.NDArray,
    traj2_images: npt.NDArray,
    num_parts: int,
    resize_ratio: float,
    left_frame_color=(255, 0, 0),
    right_frame_color=(0, 0, 255),
) -> npt.NDArray:
    def _divided_indices(n: int, num_parts: int) -> npt.NDArray:
        step = n / num_parts
        return np.array([min(n - 1, int(round(i * step))) for i in range(num_parts + 1)])

    def resize_image(image: npt.NDArray, ratio: float) -> npt.NDArray:
        new_shape = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))  # OpenCV expects (width, height)
        return cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

    traj1_indices = _divided_indices(traj1_images.shape[0], num_parts=num_parts)
    traj2_indices = _divided_indices(traj2_images.shape[0], num_parts=num_parts)

    traj1_concated = np.vstack(
        [
            add_border(
                draw_transparent_text(resize_image(traj1_images[j], resize_ratio), f"t = {i}"), color=left_frame_color
            )
            for i, j in enumerate(traj1_indices)
        ]
    )
    traj2_concated = np.vstack(
        [
            add_border(
                draw_transparent_text(resize_image(traj2_images[j], resize_ratio), f"t = {i}"), color=right_frame_color
            )
            for i, j in enumerate(traj2_indices)
        ]
    )
    black_line = np.zeros((traj1_concated.shape[0], 15, traj1_concated.shape[2]), dtype=traj1_concated.dtype)
    concated_image = np.hstack([traj1_concated, black_line, traj2_concated])

    return concated_image


def remove_small_regions(segmentation: npt.NDArray, min_size: int = 3) -> np.ndarray:
    """
    Remove small regions (connected components) in a segmentation map.

    Parameters:
        segmentation (np.ndarray): Input segmentation map (e.g., np.uint16).
        min_size (int): Minimum size of regions to retain. Regions smaller than this are removed.

    Returns:
        np.ndarray: Segmentation map with small regions removed.
    """
    # Ensure the output is a copy of the input
    output_segmentation = np.copy(segmentation)

    # Get unique labels in the segmentation map
    unique_labels = np.unique(segmentation)

    for label_value in unique_labels:
        if label_value == 0:
            continue  # Skip background

        # Isolate the current label
        binary_label = (segmentation == label_value).astype(np.int32)

        # Label connected components within the current label
        labeled_array, num_features = scipy_label(binary_label)

        # Get the size of each connected component
        unique, counts = np.unique(labeled_array, return_counts=True)
        label_sizes = dict(zip(unique, counts))

        # Remove small regions
        for sub_label, size in label_sizes.items():
            if sub_label == 0:
                continue  # Skip background
            if size < min_size:
                output_segmentation[labeled_array == sub_label] = 0

    return output_segmentation


def remove_non_target_regions(segmentation: npt.NDArray, id_to_object_name: dict[int, str]) -> np.ndarray:
    # Ensure the output is a copy of the input
    output_segmentation = np.copy(segmentation)

    # Get unique labels in the segmentation map
    unique_labels = np.unique(segmentation)

    for label_value in unique_labels:
        if label_value == 0:
            continue  # Skip background

        if label_value in id_to_object_name.keys():
            continue

        output_segmentation[output_segmentation == label_value] = 0.0

    return output_segmentation


def draw_key_points(
    image: npt.NDArray,
    points: npt.NDArray,
    file_path: pathlib.Path,
    normalized: bool = True,
    color: tuple = (0, 0, 255),
    radius: int = 3,
    thickness: int = -1,
):
    """
    Draw key point coordinates on an image using OpenCV (cv2).

    Args:
        image (np.ndarray):      The image (H, W, C). It can be in BGR or RGB format,
                                so be cautious during visualization.
        points (np.ndarray):     Key point coordinates (N, 2). Each row is in the format [x, y].
        normalized (bool):       If True, coordinates in the range [-1, 1] will be converted to pixel values.
        color (tuple):           The color for the drawn points (B, G, R) or (R, G, B).
        radius (int):            The radius of the point (for cv2.circle).
        thickness (int):         The thickness of the point; if -1, the circle will be filled.
        window_name (str):       (For cv2.imshow) The name of the display window.
        wait_key (int):          (For cv2.waitKey) The wait time in milliseconds; if 0, waits indefinitely.

    Note:
        - OpenCV typically uses BGR channel order; however, images converted directly
          from PyTorch are often in RGB, so take care when visualizing.
        - If you want to close the display window immediately after showing the image,
          call `cv2.destroyAllWindows()` after `cv2.waitKey(wait_key)`.
    """
    h, w = image.shape[:2]

    if normalized:
        # Convert x: [-1,1] -> [0, w-1]
        # Convert y: [-1,1] -> [0, h-1]
        points_px = points.copy()
        points_px[:, 0] = (points[:, 0] + 1) / 2.0 * (w - 1)
        points_px[:, 1] = (points[:, 1] + 1) / 2.0 * (h - 1)
    else:
        points_px = points

    # Convert coordinates to integers for drawing (cv2.circle requires integer coordinates)
    points_px = np.round(points_px).astype(np.int32)

    # Draw each point
    for x, y in points_px:
        cv2.circle(image, (x, y), radius, color, thickness)

    cv2.imwrite(filename=str(file_path), img=image)


def object_segmentation_from_segmentation_buffer(
    mj_model: "mujoco.MjModel", segmentation_buffer: npt.NDArray, object_names: tuple[str, ...]
) -> tuple[npt.NDArray, dict[int, str], dict[str, int]]:
    segmentation_id_to_name: dict[int, str] = {}
    name_to_segmentation_id: dict[str, int] = {}
    for i, object_name in enumerate(object_names):
        # NOTE: 0 means back ground
        segmentation_id_to_name[i + 1] = object_name
        name_to_segmentation_id[object_name] = i + 1

    mujoco_type_image = segmentation_buffer[:, :, 0]
    mujoco_id_image = segmentation_buffer[:, :, 1]

    geoms = mujoco_type_image == mujoco.mjtObj.mjOBJ_GEOM
    geom_ids = np.unique(mujoco_id_image[geoms])

    object_segmentation = np.zeros((segmentation_buffer.shape[:2]), np.uint16)
    for geom_id in geom_ids:
        model_name = str(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        for object_name in object_names:
            if object_name in model_name:
                object_segmentation[mujoco_id_image == geom_id] = name_to_segmentation_id[object_name]

    return object_segmentation, segmentation_id_to_name, name_to_segmentation_id
