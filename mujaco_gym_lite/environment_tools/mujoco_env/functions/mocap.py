import numpy as np

import mujoco


def reset_mocap(mj_model: "mujoco.MjModel", mj_data: "mujoco.MjData"):
    """
    This function will call mj_forward.
    DO NOT call this function in environment's step function,
    because your simulation will not run correctly.
    """
    for i in range(mj_model.eq_data.shape[0]):
        # See: https://mujoco.readthedocs.io/en/stable/python.html#enums-and-constants
        if mj_model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            mj_model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    mujoco.mj_forward(mj_model, mj_data)
