import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    MaterialConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_floor(
    generator: MJCFGenerator,
    texture_file_path: Optional[pathlib.Path],
    floor_name: str,
    floor_position: npt.NDArray,
    floor_rotation: npt.NDArray,
    floor_size: npt.NDArray = np.array([5.0, 5.0, 0.01]),
    floor_color: Optional[npt.NDArray] = None,
):
    if texture_file_path is None and floor_color is None:
        raise ValueError("Invalid texture_file_path and floor_color")

    if texture_file_path is not None and floor_color is not None:
        raise ValueError("Invalid texture_file_path and floor_color")

    material_name = None
    if texture_file_path is not None:
        texture_name = floor_name + "_texture"
        generator.add_texture(mjcf_config=TextureConfig(name=texture_name, type="2d", file=texture_file_path))
        material_name = floor_name + "_material"
        generator.add_material(mjcf_config=MaterialConfig(name=material_name, texture=texture_name))

    # For body
    # floor body
    body_name = floor_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(name=body_name, pos=floor_position, quat=floor_rotation),
        parent_name="worldbody",
    )
    # floor geom
    geom_name = floor_name + "_geom"

    if texture_file_path is not None:
        assert material_name is not None
        generator.add_geom(
            mjcf_config=GeomConfig(name=geom_name, type="plane", material=material_name, size=floor_size),
            parent_name=body_name,
        )
    else:
        generator.add_geom(
            mjcf_config=GeomConfig(name=geom_name, type="plane", rgba=floor_color, size=floor_size),
            parent_name=body_name,
        )
