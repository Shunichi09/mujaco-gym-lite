import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    CameraConfig,
    GeomConfig,
    MaterialConfig,
    MeshConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def _add_camera_mesh(
    generator: MJCFGenerator,
    camera_name: str,
    attach_body: str,
    mesh_file_path: pathlib.Path,
    texture_file_path: pathlib.Path,
):
    visual_mesh_name = camera_name + "_visual_mesh"
    generator.add_mesh(MeshConfig(name=visual_mesh_name, file=mesh_file_path))

    texture_name = camera_name + "_texture"
    generator.add_texture(TextureConfig(name=texture_name, type="2d", file=texture_file_path))
    material_name = camera_name + "_material"
    generator.add_material(MaterialConfig(name=material_name, texture=texture_name))

    visual_geom_name = camera_name + "_visual_geom"
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=visual_geom_name,
            type="mesh",
            mesh=visual_mesh_name,
            material=material_name,
            group=2,
            contype=0,
            conaffinity=0,
            quat=np.array([0.0, 0.0, 1.0, 0.0]),  # NOTE: For being consistent with mujoco camera definition
        ),
        parent_name=attach_body,
    )


def add_camera(
    generator: MJCFGenerator,
    camera_name: str,
    camera_position: npt.NDArray,
    camera_rotation: npt.NDArray,
    fovy: int = 69,
    attach_body: str = "worldbody",
    mesh_file_path: Optional[pathlib.Path] = None,
    texture_file_path: Optional[pathlib.Path] = None,
):
    # NOTE: for a clarification, first add body for it
    camera_body_name = camera_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(
            name=camera_body_name,
            pos=camera_position,
            quat=camera_rotation,
        ),
        parent_name=attach_body,
    )
    generator.add_camera(
        mjcf_config=CameraConfig(
            name=camera_name,
            mode="fixed",
            fovy=fovy,
        ),
        parent_name=camera_body_name,
    )

    if mesh_file_path is not None and texture_file_path is not None:
        _add_camera_mesh(generator, camera_name, camera_body_name, mesh_file_path, texture_file_path)

    return camera_body_name


def add_mocap_camera(
    generator: MJCFGenerator,
    camera_name: str,
    camera_position: npt.NDArray,
    camera_rotation: npt.NDArray,
    fovy: int = 69,
    attach_body: str = "worldbody",
    mesh_file_path: Optional[pathlib.Path] = None,
    texture_file_path: Optional[pathlib.Path] = None,
):
    # NOTE: to use mocap, first add body for it
    camera_body_name = camera_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(
            name=camera_body_name,
            pos=camera_position,
            quat=camera_rotation,
            mocap="true",
        ),
        parent_name=attach_body,
    )
    generator.add_camera(
        mjcf_config=CameraConfig(
            name=camera_name,
            mode="fixed",
            fovy=fovy,
        ),
        parent_name=camera_body_name,
    )

    if mesh_file_path is not None and texture_file_path is not None:
        _add_camera_mesh(generator, camera_name, camera_body_name, mesh_file_path, texture_file_path)
