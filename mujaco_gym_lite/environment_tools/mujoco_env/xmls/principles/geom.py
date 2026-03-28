import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    JointConfig,
    MaterialConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_mujoco_principal_geom(
    generator: MJCFGenerator,
    model_name: str,
    model_position: npt.NDArray,
    model_rotation: npt.NDArray,
    model_type_name: str,
    model_color: Optional[npt.NDArray] = None,
    model_size: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    model_density: float = 500,
    model_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    model_solref: npt.NDArray = np.array([0.01, 1]),
    model_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    model_group: Optional[int] = None,
    condim: int = 4,
    contype: Optional[int] = None,
    conaffinity: Optional[int] = None,
    with_free_joint: bool = True,
    parent_name: str = "worldbody",
    mocap: Optional[str] = None,
    model_joint_name: Optional[str] = None,
    texture_file_path: Optional[pathlib.Path] = None,
) -> dict[str, Optional[str]]:
    texture_name = model_name + "_texture"
    if (texture_file_path is None) and (model_color is None):
        raise ValueError

    if (texture_file_path is not None) and (model_color is not None):
        raise ValueError

    material_name = None
    if texture_file_path is not None:
        generator.add_texture(TextureConfig(name=texture_name, type="2d", file=texture_file_path))
        material_name = model_name + "_material"
        generator.add_material(MaterialConfig(name=material_name, texture=texture_name))

    # body
    body_name = model_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(name=body_name, pos=model_position, quat=model_rotation, mocap=mocap),
        parent_name=parent_name,
    )
    # joint
    if with_free_joint:
        assert model_joint_name is not None
        generator.add_joint(mjcf_config=JointConfig(name=model_joint_name, type="free"), parent_name=body_name)

        end_effector_name = model_name + "_end_effector_body"
        generator.add_body(
            BodyConfig(name=end_effector_name),
            parent_name=body_name,
        )
    else:
        end_effector_name = None

    generator.add_geom(
        mjcf_config=GeomConfig(
            name=model_name + "_collision_geom",
            type=model_type_name,
            friction=model_friction,
            density=model_density,
            solimp=model_solimp,
            solref=model_solref,
            condim=condim,
            size=model_size,
            contype=contype,
            conaffinity=conaffinity,
            material=None if material_name is None else material_name,
            rgba=None if model_color is None else model_color,
            group=model_group,
        ),
        parent_name=body_name,
    )
    return {"body_name": body_name, "end_effector_name": end_effector_name}
