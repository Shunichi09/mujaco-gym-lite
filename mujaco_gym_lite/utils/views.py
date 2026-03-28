from typing import cast

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.logger import logger
from mujaco_gym_lite.random import np_drng
from mujaco_gym_lite.utils.math import fit_angle_in_range
from mujaco_gym_lite.utils.randoms import rand_min_max
from mujaco_gym_lite.utils.solids import aligned_sphere_points, generate_sphere_points
from mujaco_gym_lite.utils.transforms import (
    create_transformation_matrix,
    euler_to_matrix,
    extract_position,
    extract_rotation,
    matrix_to_quat,
    rotvec_from_two_vectors,
    rotvec_to_matrix,
)


def _compute_align_xy_plane(x: float, y: float) -> npt.NDArray:
    xy_plane_angle = np.arctan2(y, x)
    xy_plane_angle = fit_angle_in_range(xy_plane_angle, min_angle=0.0, max_angle=2.0 * np.pi)
    return fit_angle_in_range(xy_plane_angle + np.pi * 0.5, min_angle=-np.pi, max_angle=np.pi)


def view_to_rotation(direction: npt.NDArray) -> npt.NDArray:
    assert len(direction) == 3

    if np.allclose(np.array([0.0, 0.0, -1.0]), direction / np.linalg.norm(direction), atol=1e-5):
        raise ValueError

    # TODO: Support x, y axis
    direction_rotation_vector = rotvec_from_two_vectors(np.array([0.0, 0.0, 1.0]), direction)
    direction_rotation_matrix = rotvec_to_matrix(direction_rotation_vector)

    # TODO: Support x, y axis
    angle_to_align_plane = _compute_align_xy_plane(-direction[0], -direction[1])
    align_rotation_matrix = euler_to_matrix(angle_to_align_plane, order="z")
    rotation = np.matmul(direction_rotation_matrix, align_rotation_matrix)
    return cast(npt.NDArray, rotation)  # in shape (3, 3)


def sample_sphere_view(
    num_samples: int,
    min_radius: float,
    max_radius: float,
    num_sphere_points: int,
    min_height: float,  # in z
    base_transformation_matrixes: list[npt.NDArray],
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    assert max_radius >= min_radius
    assert len(base_transformation_matrixes) == num_samples
    sampled_transformations: list[npt.NDArray] = []

    while True:
        # sample sphere points
        sampled_radius = rand_min_max(max_val=max_radius, min_val=min_radius)
        sphere_points = generate_sphere_points(float(sampled_radius), num_sphere_points, method="uniform")
        sampled_sphere_point = sphere_points[np_drng.integers(len(sphere_points))]
        approach_vector = np.zeros(3) - sampled_sphere_point
        approach_vector /= np.linalg.norm(approach_vector)
        sampled_rotation = view_to_rotation(approach_vector)

        # apply base transformation
        current_index = len(sampled_transformations)
        base_matrix = base_transformation_matrixes[current_index]
        sampled_transformation = np.matmul(
            base_matrix, create_transformation_matrix(sampled_sphere_point, sampled_rotation)
        )
        sampled_position = extract_position(sampled_transformation)

        if sampled_position[2] < min_height:
            logger.debug("Not meet the height requirement.")
        else:
            sampled_transformations.append(sampled_transformation)

        if len(sampled_transformations) >= num_samples:
            break

    sampled_pos_rot_pair = [(extract_position(t), extract_rotation(t)) for t in sampled_transformations]
    return sampled_pos_rot_pair


def aligned_sphere_view(
    num_max_samples_per_radius: int,
    radius: list[float],
    min_height: float,  # in z
    base_transformation_matrix: npt.NDArray,
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    views = []
    for r in radius:
        points = aligned_sphere_points(num_points=num_max_samples_per_radius, radius=r)
        for p in points:
            approach_vector = np.zeros(3) - p
            approach_vector /= np.linalg.norm(approach_vector)
            p_rotation = view_to_rotation(approach_vector)

            p_transformation = np.matmul(base_transformation_matrix, create_transformation_matrix(p, p_rotation))
            sampled_position = extract_position(p_transformation)

            if sampled_position[2] < min_height:
                logger.debug("Not meet the height requirement.")
            else:
                views.append((extract_position(p_transformation), extract_rotation(p_transformation)))
    return views


def surrounding_view(
    point: npt.NDArray, radius: float
) -> tuple[list[tuple[npt.NDArray, npt.NDArray]], dict[str, tuple[npt.NDArray, npt.NDArray]]]:
    views = []
    views.append(_view_from_top(point, radius))
    views.append(_view_from_left(point, radius))
    views.append(_view_from_right(point, radius))
    views.append(_view_from_left_diagonal(point, radius))
    views.append(_view_from_right_diagonal(point, radius))
    views.append(_view_from_back(point, radius))
    views.append(_view_from_front(point, radius))
    views.append(_view_from_front_left_diagonal(point, radius))
    views.append(_view_from_front_right_diagonal(point, radius))

    return views, {
        "top": views[0],
        "left_side": views[1],
        "right_side": views[2],
        "left_diagonal": views[3],
        "right_diagonal": views[4],
        "back": views[5],
        "front": views[6],
        "left_front_diagonal": views[7],
        "right_front_diagonal": views[8],
    }


def _view_from_top(points: npt.NDArray, radius: float):
    assert len(points) == 3
    camera_pos = points + np.array([0.0, 0.0, radius])
    camera_rot = view_to_rotation(
        direction=points - camera_pos + [0.0, -0.0001, 0.0]
    )  # NOTE: avoid computational error
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_left(points: npt.NDArray, radius: float):
    assert len(points) == 3
    camera_pos = points + np.array([-radius, 0.0, 0.0])
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_right(points: npt.NDArray, radius: float):
    assert len(points) == 3
    camera_pos = points + np.array([radius, 0.0, 0.0])
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_left_diagonal(
    points: npt.NDArray[np.float64], radius: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dir = np.array([-0.5, -0.5, np.sqrt(2) / 2], dtype=points.dtype)
    camera_pos = points + dir * radius
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_right_diagonal(points: npt.NDArray[np.float64], radius: float):
    dir = np.array([0.5, -0.5, np.sqrt(2) / 2], dtype=points.dtype)
    camera_pos = points + dir * radius
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_front_left_diagonal(
    points: npt.NDArray[np.float64], radius: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    offset = np.array([-1.0, 1.0, 1.0], dtype=points.dtype)
    unit = offset / np.linalg.norm(offset)
    camera_pos = points + unit * radius
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_front_right_diagonal(
    points: npt.NDArray[np.float64], radius: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    offset = np.array([1.0, 1.0, 1.0], dtype=points.dtype)
    unit = offset / np.linalg.norm(offset)
    camera_pos = points + unit * radius
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_back(points: npt.NDArray, radius: float):
    assert len(points) == 3
    camera_pos = points + np.array([0.0, -np.sin(np.pi / 4.0) * radius, np.sin(np.pi / 4.0) * radius])
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)


def _view_from_front(points: npt.NDArray, radius: float):
    assert len(points) == 3
    camera_pos = points + np.array([0.0, np.sin(np.pi / 4.0) * radius, np.sin(np.pi / 4.0) * radius])
    camera_rot = view_to_rotation(direction=points - camera_pos)
    return camera_pos, matrix_to_quat(camera_rot)
