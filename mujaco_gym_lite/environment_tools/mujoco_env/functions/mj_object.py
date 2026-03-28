import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.env_models.objects.mocap_object import MocapObject
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import is_contact_between_models
from mujaco_gym_lite.utils.randoms import rand_min_max
from mujaco_gym_lite.utils.transforms import (
    create_transformation_matrix,
    euler_to_quat,
    extract_position,
    extract_rotation,
)


def get_model_name_state(mj_data: "mujoco.mjData", model_name: str, suffix_str: str = "_body") -> npt.NDArray:
    return create_transformation_matrix(
        mj_data.body(model_name + suffix_str).xpos,
        mj_data.body(model_name + suffix_str).xquat,
        rotation_type="quaternion",
    )


def get_model_name_states(
    mj_data: "mujoco.mjData", model_names: list[str], suffix_str: str = "_body"
) -> dict[str, npt.NDArray]:
    name_to_state = {}
    for model_name in model_names:
        name_to_state[model_name] = get_model_name_state(mj_data, model_name, suffix_str)
    return name_to_state


def move_model(
    mj_data: "mujoco.mjData", model_name: str, transformation_matrix: npt.NDArray, suffix_str: str = "_joint"
):
    target_qpos = np.concatenate(
        [
            extract_position(transformation_matrix),
            extract_rotation(transformation_matrix),
        ]
    )
    mj_data.joint(model_name + suffix_str).qpos[:] = target_qpos.copy()
    mj_data.joint(model_name + suffix_str).qvel[:] = np.zeros(6)  # NOTE: always 0 to stabilize a simulation


def move_models(
    mj_data: "mujoco.mjData",
    model_names: list[str],
    transformation_matrixes: list[npt.NDArray],
    suffix_str: str = "_joint",
):
    """move given model name via set mj_data.joint.qpos value and qvel value
    NOTE: qvel value is always 0
    """
    for model_name, transformation_matrix in zip(model_names, transformation_matrixes):
        move_model(mj_data, model_name, transformation_matrix, suffix_str)


def sample_object_model_states(
    mj_model: "mujoco.mjModel",
    mj_data: "mujoco.mjData",
    object_model_names: list[str],
    max_trials_per_each_object_model: int,
    min_workspace: npt.NDArray,
    max_workspace: npt.NDArray,
) -> dict[str, npt.NDArray]:
    """
    This function will call mj_forward.
    DO NOT call this function in environment's step function,
    because your simulation will not run correctly.
    We also recommend to call mj_reset after calling this function.
    """
    object_model_states = {}
    for _, object_model_name in enumerate(object_model_names):
        default_object_model_state = get_model_name_state(mj_data, object_model_name).copy()
        for _ in range(max_trials_per_each_object_model):
            position = np.array(rand_min_max(min_val=min_workspace, max_val=max_workspace))
            z_angle = rand_min_max(min_val=0.0, max_val=2 * np.pi)
            rotation = euler_to_quat(z_angle, order="z")

            object_model_state = create_transformation_matrix(position, rotation, "quaternion")
            move_model(
                mj_data,
                object_model_name,
                object_model_state,
            )

            mujoco.mj_forward(mj_model, mj_data)

            if is_contact_between_models(mj_model, mj_data, object_model_names):
                # reset state
                move_model(
                    mj_data,
                    object_model_name,
                    default_object_model_state,
                )
                mujoco.mj_forward(mj_model, mj_data)
                continue
            else:
                object_model_states[object_model_name] = object_model_state
                break

    return object_model_states


def mocap_object_action(mocap_object: MocapObject, object_action: npt.NDArray, force: bool):
    """
    NOTE: If force True, we set the qpos and qvel first, and then apply mocap as well
    """
    assert len(object_action) == 7
    if force:
        # NOTE: If force True, we set the qpos and qvel first, and then apply mocap as well
        mocap_object.apply_joint_qpos_and_qvel(joint_qpos=[object_action])

    mocap_object.move(
        create_transformation_matrix(
            object_action[:3],
            object_action[3:],
            rotation_type="quaternion",
        )
    )
