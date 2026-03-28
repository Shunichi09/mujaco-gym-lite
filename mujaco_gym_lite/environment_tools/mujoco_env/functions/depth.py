import numpy.typing as npt

import mujoco


def zbuffer_to_depth(mj_model: "mujoco.mjModel", depth_buffer: npt.NDArray) -> npt.NDArray:
    """TODO: Do we have to use this version?
    zfar = np.float32(far)
    znear = np.float32(near)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

    # In reverse Z mode the perspective matrix is transformed by the following
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef

    # We need 64 bits to convert Z from ndc to metric depth without noticeable
    # losses in precision
    out_64 = np.array(depth_buffer, dtype=np.float64)

    # Undo OpenGL projection
    # Note: We do not need to take action to convert from window coordinates
    # to normalized device coordinates because in reversed Z mode the mapping
    # is identity
    depth = d_coef / (out_64 + c_coef)

    # Cast result back to float32 for backwards compatibility
    # This has a small accuracy cost
    return np.array(depth, np.float32)
    """
    # https://github.com/google-deepmind/mujoco/blob/8f1ea05bef131e71bb374678ca0a777322dea2ae/python/mujoco/renderer.py#L179C1-L179C27
    # depth_buffer
    extent = float(mj_model.stat.extent)
    near = float(mj_model.vis.map.znear * extent)
    far = float(mj_model.vis.map.zfar * extent)

    # Convert from [0 1] to depth in meters, see links below:
    # http://stackoverflow.com/a/6657284/1461210
    # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
    depth = near / (1 - depth_buffer * (1 - near / far))
    return depth
