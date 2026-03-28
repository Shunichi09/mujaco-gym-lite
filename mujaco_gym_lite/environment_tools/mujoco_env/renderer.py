from typing import Optional, Union, cast

import numpy as np
import numpy.typing as npt

import mujoco
from mujaco_gym_lite.environment_tools.mujoco_env.externals.renderer import OffScreenViewer, WindowViewer


class MujocoRenderer:
    """This is the MuJoCo renderer manager class for every MuJoCo environment.
    Almost same as the original code, but support segmentaiton.
    """

    _viewers: dict[str, Union[WindowViewer, OffScreenViewer]]
    viewer: Optional[Union[WindowViewer, OffScreenViewer]]

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        width: int,
        height: int,
        default_cam_config: Optional[dict] = None,
        max_geom: int = 1000,
        use_fixed_camera_in_human_mode: bool = False,
    ):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            model: MjModel data structure of the MuJoCo simulation
            data: MjData data structure of the MuJoCo simulation
            default_cam_config: dictionary with attribute values of the viewer's default camera,
            https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global
            width: width of the OpenGL rendering context
            height: height of the OpenGL rendering context
            max_geom: maximum number of geometries to render
        """
        self.model = model
        self.data = data
        self._viewers = {}
        self.viewer = None
        self.default_cam_config = default_cam_config
        self.width = width
        self.height = height
        self.max_geom = max_geom
        self._use_fixed_camera_in_human_mode = use_fixed_camera_in_human_mode

    def render(
        self,
        render_mode: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ) -> Optional[npt.NDArray]:
        """Renders a frame of the simulation in a specific format and camera view.

        Args:
            render_mode: The format to render the frame, it can be: "human", "rgb_array", or "depth_array"
            camera_id: The integer camera id from which to render the frame in the MuJoCo simulation
            camera_name: The string name of the camera from which to render the frame in the MuJoCo simulation.
            This argument should not be passed if using cameara_id instead and vice versa

        Returns:
            If render_mode is "rgb_array" or "depth_arra" it returns a numpy array in the specified format.
            "human" render mode does not return anything.
        """

        viewer = self._get_viewer(render_mode=render_mode)
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be" " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = "track"

        if camera_id is None:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

        if render_mode in {"rgb_array", "depth_array", "segmentation_array"}:
            assert isinstance(viewer, OffScreenViewer)
            if render_mode == "segmentation_array":
                segmentation = True
                # render mode to rbg array
                render_mode = "rgb_array"
            else:
                segmentation = False

            img = viewer.render(render_mode=render_mode, camera_id=camera_id, segmentation=segmentation)
            return cast(np.ndarray, img)

        elif render_mode == "human":
            assert isinstance(viewer, WindowViewer)
            if self._use_fixed_camera_in_human_mode:
                return cast(np.ndarray, viewer.render(camera_id=camera_id))
            else:
                return cast(np.ndarray, viewer.render())

        else:
            raise ValueError("Not supported render mode")

    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = WindowViewer(self.model, self.data, self.width, self.height, self.max_geom)

            elif render_mode in {"rgb_array", "depth_array", "segmentation_array"}:
                self.viewer = OffScreenViewer(self.model, self.data, self.width, self.height, self.max_geom)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, depth_array, segmentation_array"
                )
            # Add default camera parameters
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer

    def _set_cam_config(self):
        """Set the default camera parameters"""
        assert self.viewer is not None
        if self.default_cam_config is not None:
            for key, value in self.default_cam_config.items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = value
                else:
                    setattr(self.viewer.cam, key, value)

    def close(self):
        """Close the OpenGL rendering contexts of all viewer modes"""
        for _, viewer in self._viewers.items():
            viewer.close()


def qpos_qvel_to_rgb_frames(env, qpos: np.ndarray, qvel: np.ndarray) -> list[np.ndarray]:
    env.reset()
    frames = []
    assert len(qpos) == len(qvel)
    for p, v in zip(qpos, qvel):
        env.set_state(p, v)
        frame = env.render()
        frames.append(frame)
    return frames
