from typing import List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp


def create_transformation_matrix(
    translation: Union[List[float], npt.NDArray] = np.zeros(3),
    rotation: Union[List[float], npt.NDArray] = np.eye(3),
    rotation_type: str = "matrix",
) -> npt.NDArray:
    translation = np.array(translation)
    rotation = np.array(rotation)

    transformation_mat = np.eye(4)
    if rotation_type == "matrix":
        transformation_mat[:3, :3] = rotation.copy()
    elif rotation_type == "quaternion":
        transformation_mat[:3, :3] = quat_to_matrix(rotation)
    elif rotation_type == "rotvec":
        transformation_mat[:3, :3] = rotvec_to_matrix(rotation)
    else:
        raise ValueError

    transformation_mat[:-1, -1] = translation.copy()
    return transformation_mat


def extract_position(transformation_matrix: npt.NDArray) -> npt.NDArray:
    if len(transformation_matrix.shape) == 3:
        return transformation_matrix[:, :3, -1]
    elif len(transformation_matrix.shape) == 2:
        return transformation_matrix[:3, -1]
    else:
        raise ValueError


def extract_rotation(transformation_matrix: npt.NDArray, rotation_type: str = "quaternion") -> npt.NDArray:
    if len(transformation_matrix.shape) == 3:
        rotation_matrix = transformation_matrix[:, :3, :3]
    elif len(transformation_matrix.shape) == 2:
        rotation_matrix = transformation_matrix[:3, :3]
    else:
        raise ValueError

    if rotation_type == "quaternion":
        return matrix_to_quat(rotation_matrix)
    elif rotation_type == "matrix":
        return rotation_matrix.copy()
    else:
        raise ValueError


def flatten_transformation_matrix(matrix: npt.NDArray):
    position = extract_position(matrix)
    rotation = extract_rotation(matrix)
    # TODO: support batch
    return np.concatenate([position, rotation])


def _mujoco_quat_to_scipy_quat(mujoco_quat: npt.NDArray) -> npt.NDArray:
    """(w, x, y, z) to (x, y, z, w)"""
    if len(mujoco_quat.shape) == 2:
        scipy_quat = mujoco_quat[:, np.array([1, 2, 3, 0])]
        assert scipy_quat.shape[1] == 4
    elif len(mujoco_quat.shape) == 1:
        scipy_quat = mujoco_quat[np.array([1, 2, 3, 0])]
        assert scipy_quat.shape[0] == 4
    else:
        raise ValueError

    return scipy_quat


def _scipy_quat_to_mujoco_quat(scipy_quat: npt.NDArray) -> npt.NDArray:
    """(x, y, z, w) to (w, x, y, z)"""
    if len(scipy_quat.shape) == 2:
        mujoco_quat = scipy_quat[:, np.array([3, 0, 1, 2])]
        assert mujoco_quat.shape[1] == 4
    elif len(scipy_quat.shape) == 1:
        mujoco_quat = scipy_quat[np.array([3, 0, 1, 2])]
        assert mujoco_quat.shape[0] == 4
    else:
        raise ValueError

    return mujoco_quat


def quat_to_matrix(quaternion: npt.NDArray) -> npt.NDArray:
    # (w x y z) to (x y z w)
    scipy_quaternion = _mujoco_quat_to_scipy_quat(quaternion)
    scipy_rotation = Rotation.from_quat(scipy_quaternion)
    return np.array(scipy_rotation.as_matrix())


def quat_to_rotvec(quaternion: npt.NDArray) -> npt.NDArray:
    # (w x y z) to (x y z w)
    scipy_quaternion = _mujoco_quat_to_scipy_quat(quaternion)
    scipy_rotation = Rotation.from_quat(scipy_quaternion)
    return np.array(scipy_rotation.as_rotvec())


def rotvec_to_quat(rotvec: npt.NDArray) -> npt.NDArray:
    scipy_rotation = Rotation.from_rotvec(rotvec)
    scipy_quaternion = scipy_rotation.as_quat()
    mujoco_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(mujoco_quaternion)


def rotvec_to_matrix(rotvec: npt.NDArray) -> npt.NDArray:
    scipy_rotation = Rotation.from_rotvec(rotvec)
    return np.array(scipy_rotation.as_matrix())


def matrix_to_quat(matrix: npt.NDArray) -> npt.NDArray:
    # (w x y z) to (x y z w)
    scipy_rotation = Rotation.from_matrix(matrix)
    scipy_quaternion = scipy_rotation.as_quat()
    mujoco_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(mujoco_quaternion)


def euler_to_quat(
    angles: Union[npt.NDArray, float, list[float], Tuple[float, ...], Tuple[Tuple[float, ...], ...], npt.NDArray],
    order: str = "xyz",
) -> npt.NDArray:
    scipy_rotation = Rotation.from_euler(order, angles, degrees=False)
    scipy_quaternion = scipy_rotation.as_quat()
    mujoco_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(mujoco_quaternion)


def euler_to_matrix(
    angles: Union[npt.NDArray, float, list[float], Tuple[float, ...], Tuple[Tuple[float, ...], ...]],
    order: str = "xyz",
) -> npt.NDArray:
    scipy_rotation = Rotation.from_euler(order, angles, degrees=False)
    return np.array(scipy_rotation.as_matrix())


def matrix_to_euler(matrix: npt.NDArray, order="xyz") -> npt.NDArray:
    scipy_rotation = Rotation.from_matrix(matrix)
    return cast(npt.NDArray, scipy_rotation.as_euler(order))


def quat_to_euler(quaternion: npt.NDArray, order="xyz") -> npt.NDArray:
    # (w x y z) to (x y z w)
    scipy_quaternion = _mujoco_quat_to_scipy_quat(quaternion)
    scipy_rotation = Rotation.from_quat(scipy_quaternion)
    return np.array(scipy_rotation.as_euler(order))


def matrix_to_rotvec(matrix: npt.NDArray) -> npt.NDArray:
    scipy_rotation = Rotation.from_matrix(matrix)
    return cast(npt.NDArray, scipy_rotation.as_rotvec())


def compute_approach_vector(quaternion: npt.NDArray, axis="z") -> npt.NDArray:
    if axis == "x":
        return quat_to_matrix(quaternion)[:, 0]
    elif axis == "y":
        return quat_to_matrix(quaternion)[:, 1]
    elif axis == "z":
        return quat_to_matrix(quaternion)[:, 2]
    else:
        raise ValueError


def combine_quat(quat1: npt.NDArray, quat2: npt.NDArray, normalize=True) -> npt.NDArray:
    quat1_mat = np.array(
        [
            [quat1[0], -quat1[1], -quat1[2], -quat1[3]],
            [quat1[1], quat1[0], -quat1[3], quat1[2]],
            [quat1[2], quat1[3], quat1[0], -quat1[1]],
            [quat1[3], -quat1[2], quat1[1], quat1[0]],
        ]
    )
    combined_quat = np.dot(quat1_mat, quat2[:, np.newaxis]).flatten()

    if normalize:
        return cast(npt.NDArray, combined_quat / np.linalg.norm(combined_quat))

    return cast(npt.NDArray, combined_quat)


def inverse_quat(quaternion: npt.NDArray):
    return np.array([quaternion[0], -1 * quaternion[1], -1 * quaternion[2], -1 * quaternion[3]])


def rotvec_from_two_vectors(vector1: npt.NDArray, vector2: npt.NDArray) -> npt.NDArray:
    normal_vector = np.cross(vector1, vector2)
    angle = angle_from_two_vectors(vector1, vector2)
    norm = np.linalg.norm(normal_vector)
    denom = norm if norm != 0 else 1.0
    return cast(npt.NDArray, normal_vector / denom * angle)


def angle_from_two_vectors(vector1: npt.NDArray, vector2: npt.NDArray) -> float:
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_theta)
    return float(angle)


def transform_position(vector: npt.NDArray, transformation_matrix: npt.NDArray) -> npt.NDArray:
    assert len(vector) == 3
    tmp_vector = np.ones((4, 1))
    tmp_vector[:3, 0] = vector
    transformed_vector = np.matmul(transformation_matrix, tmp_vector)
    return cast(npt.NDArray, transformed_vector.flatten()[:3])  # FIXME: Is this a correct way to cast ??


def slerp_quaternion(quat1: npt.NDArray, quat2: npt.NDArray, num_points: int) -> list[npt.NDArray]:
    scipy_quat1 = _mujoco_quat_to_scipy_quat(quat1)
    scipy_quat2 = _mujoco_quat_to_scipy_quat(quat2)

    rot1 = Rotation.from_quat(scipy_quat1)
    rot2 = Rotation.from_quat(scipy_quat2)
    rot_c = Rotation.concatenate([rot1, rot2])

    t_values = np.linspace(0, 1, num=num_points)

    slerp = Slerp([0, 1], rot_c)
    rotations = slerp(t_values)

    return [_scipy_quat_to_mujoco_quat(np.array(r.as_quat())) for r in rotations]
