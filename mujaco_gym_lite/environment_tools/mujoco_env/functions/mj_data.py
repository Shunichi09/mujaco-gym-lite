from typing import Optional, Union

import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.utils.transforms import create_transformation_matrix


def joint_qpos(mj_data: "mujoco.mjData", names: list[str]) -> list[npt.NDArray]:
    return [np.array(mj_data.joint(name).qpos) for name in names]


def joint_qvel(mj_data: "mujoco.mjData", names: list[str]) -> list[npt.NDArray]:
    return [np.array(mj_data.joint(name).qvel) for name in names]


def body_pose(mj_data: "mujoco.mjData", names: list[str]) -> list[npt.NDArray]:
    return [
        create_transformation_matrix(
            mj_data.body(name).xpos,
            mj_data.body(name).xquat,
            rotation_type="quaternion",
        )
        for name in names
    ]


def site_pose(mj_data: "mujoco.mjData", names: list[str]) -> list[npt.NDArray]:
    return [
        create_transformation_matrix(
            mj_data.site(name).xpos,
            np.array(mj_data.site(name).xmat).reshape(3, 3),
            rotation_type="matrix",
        )
        for name in names
    ]


def apply_joint_qpos_and_qvel(
    mj_data: "mujoco.mjData",
    names: list[str],
    joint_qpos: Union[list[npt.NDArray], npt.NDArray, list[float]],
    joint_velocities: Optional[list[npt.NDArray]] = None,
):
    assert len(joint_qpos) == len(names), "Invalid joint_angles"
    joint_qpos_ndarray = np.array(joint_qpos)
    if joint_velocities is not None:
        assert len(joint_velocities) == len(names), "Invalid joint_velocities"

    for i, name in enumerate(names):
        mj_data.joint(name).qpos[:] = joint_qpos_ndarray[i].copy()
        if joint_velocities is None:
            mj_data.joint(name).qvel[:] = np.zeros_like(mj_data.joint(name).qvel[:])
        else:
            mj_data.joint(name).qvel[:] = joint_velocities[i].copy()
