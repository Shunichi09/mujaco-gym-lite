import pathlib
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    JointConfig,
    MaterialConfig,
    MeshConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.utils.files import get_files


def add_asset_model(
    generator: MJCFGenerator,
    model_dir_path: pathlib.Path,
    model_name: str,
    model_position: npt.NDArray,
    model_rotation: npt.NDArray,
    texture_file_path: Optional[pathlib.Path] = None,
    material_name: Optional[str] = None,
    model_scale: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    model_density: float = 500,
    model_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    model_solref: npt.NDArray = np.array([0.01, 1]),
    model_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    parent_name: str = "worldbody",
    add_free_joint: bool = True,
) -> dict[str, str]:
    assert not (texture_file_path is not None and material_name is not None)
    assert texture_file_path is not None or material_name is not None

    if texture_file_path is not None:
        texture_name = model_name + "_texture"
        generator.add_texture(TextureConfig(name=texture_name, type="2d", file=texture_file_path))
        material_name = model_name + "_material"
        generator.add_material(MaterialConfig(name=material_name, texture=texture_name))

    # visual mesh
    mesh_file_path = model_dir_path / "model.obj"
    visual_mesh_name = model_name + "_visual_mesh"
    generator.add_mesh(MeshConfig(name=visual_mesh_name, file=mesh_file_path, scale=model_scale))

    # collision mesh
    collision_mesh_files = get_files(model_dir_path, file_format="model_collision*.obj")
    collision_mesh_names = []
    for i, collision_mesh_file in enumerate(collision_mesh_files):
        collision_mesh_name = model_name + f"_collision_mesh_{i}"
        generator.add_mesh(MeshConfig(name=collision_mesh_name, file=collision_mesh_file, scale=model_scale))
        collision_mesh_names.append(collision_mesh_name)

    # for body tag
    # body
    body_name = model_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(name=body_name, pos=model_position, quat=model_rotation),
        parent_name=parent_name,
    )
    # joint
    if add_free_joint:
        joint_name = model_name + "_joint"
        generator.add_joint(mjcf_config=JointConfig(name=joint_name, type="free"), parent_name=body_name)

    # visual geom
    visual_geom_name = model_name + "_visual_geom"
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=visual_geom_name,
            type="mesh",
            mesh=visual_mesh_name,
            material=material_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=body_name,
    )
    # collision geom
    for i, collision_mesh_name in enumerate(collision_mesh_names):
        generator.add_geom(
            mjcf_config=GeomConfig(
                name=model_name + f"_collision_geom_{i}",
                type="mesh",
                mesh=collision_mesh_name,
                group=3,
                friction=model_friction,
                density=model_density,
                solimp=model_solimp,
                solref=model_solref,
                condim=4,
            ),
            parent_name=body_name,
        )

    return {"body_name": body_name}
