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


def add_empty_box(
    generator: MJCFGenerator,
    box_name: str,
    box_position: npt.NDArray,
    texture_file_path: Optional[pathlib.Path] = None,
    material_name: Optional[str] = None,
    rgba: Optional[npt.NDArray] = None,
    box_rotation: npt.NDArray = np.array([1.0, 0.0, 0.0, 0.0]),
    box_size: npt.NDArray = np.array([0.5, 0.5, 0.1]),
    box_thickness: float = 0.01,
    attach_body: str = "worldbody",
    box_density: int = 1000,
    solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2]),
    solref: npt.NDArray = np.array([0.02, 0.1]),
    box_texture_shininess: float = 0.25,
    box_texture_reflectance: float = 0.0,
    box_texture_emission: float = 0.0,
    box_texture_specular: float = 0.15,
    box_texture_roughness: float = 0.9,
):
    """
    Note:
        This function is for adding an empty box with basic color.
        If you want to add an empty box with texture, please use add_asset_model.
    """
    texture_name = box_name + "_texture"
    if (texture_file_path is None) and (rgba is None) and (material_name is None):
        raise ValueError

    if (texture_file_path is not None) and (rgba is not None) and (material_name is not None):
        raise ValueError

    if texture_file_path is not None:
        generator.add_texture(TextureConfig(name=texture_name, type="2d", file=texture_file_path))
        material_name = box_name + "_material"
        generator.add_material(
            MaterialConfig(
                name=material_name,
                texture=texture_name,
                shininess=box_texture_shininess,
                reflectance=box_texture_reflectance,
                emission=box_texture_emission,
                specular=box_texture_specular,
                roughness=box_texture_roughness,
            )
        )

    box_body_name = box_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(name=box_body_name, pos=box_position, quat=box_rotation),
        parent_name=attach_body,
    )

    eps = 0.001  # NOTE: This is for avoiding transparent
    geom_1_2_size = np.array([box_thickness, box_size[1] / 2.0, box_size[2] / 2.0 + eps])
    geom_1_pos = np.array([box_size[0] / 2.0, 0.0, box_size[2] / 2.0 + box_thickness * 2.0])
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=f"{box_name}_side_geom_1",
            type="box",
            pos=geom_1_pos,
            size=geom_1_2_size,
            density=box_density,
            material=None if material_name is None else material_name,
            rgba=None if rgba is None else rgba,
            condim=3,
            solimp=solimp,
            solref=solref,
        ),
        parent_name=box_body_name,
    )

    geom_2_pos = np.array([-box_size[0] / 2.0, 0.0, box_size[2] / 2.0 + box_thickness * 2.0])
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=f"{box_name}_side_geom_2",
            type="box",
            pos=geom_2_pos,
            size=geom_1_2_size,
            density=box_density,
            material=None if material_name is None else material_name,
            rgba=None if rgba is None else rgba,
            condim=3,
            solimp=solimp,
            solref=solref,
        ),
        parent_name=box_body_name,
    )

    geom_3_4_size = np.array([box_size[0] / 2.0 - box_thickness, box_thickness, box_size[2] / 2.0 + eps])
    geom_3_pos = np.array(
        [
            0.0,
            box_size[1] / 2.0 - box_thickness,
            box_size[2] / 2.0 + box_thickness * 2.0,
        ]
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=f"{box_name}_side_geom_3",
            type="box",
            pos=geom_3_pos,
            size=geom_3_4_size,
            density=box_density,
            material=None if material_name is None else material_name,
            rgba=None if rgba is None else rgba,
            condim=3,
            solimp=solimp,
            solref=solref,
        ),
        parent_name=box_body_name,
    )

    geom_4_pos = np.array(
        [
            0.0,
            -box_size[1] / 2.0 + box_thickness,
            box_size[2] / 2.0 + box_thickness * 2.0,
        ]
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=f"{box_name}_side_geom_4",
            type="box",
            pos=geom_4_pos,
            size=geom_3_4_size,
            density=box_density,
            material=None if material_name is None else material_name,
            rgba=None if rgba is None else rgba,
            condim=3,
            solimp=solimp,
            solref=solref,
        ),
        parent_name=box_body_name,
    )

    geom_5_size = np.array([box_size[0] / 2.0 + box_thickness, box_size[1] / 2.0, box_thickness])
    geom_5_pos = np.array([0.0, 0.0, box_thickness])
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=f"{box_name}_bottom_geom_5",
            type="box",
            pos=geom_5_pos,
            size=geom_5_size,
            density=box_density,
            material=None if material_name is None else material_name,
            rgba=None if rgba is None else rgba,
            condim=3,
            solimp=solimp,
            solref=solref,
        ),
        parent_name=box_body_name,
    )
