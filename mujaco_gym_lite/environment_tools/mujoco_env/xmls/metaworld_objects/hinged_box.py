import pathlib

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    JointConfig,
    MaterialConfig,
    MeshConfig,
    SiteConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_hinged_box(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    box_name: str,
    box_position: npt.NDArray,
    box_rotation: npt.NDArray,
    box_body_name: str,
    box_joint_name: str,
    box_handle_site_name: str,
    box_handle_mesh_name: str,
    box_density: float = 500,
    box_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    box_solref: npt.NDArray = np.array([0.01, 1]),
):
    metal1_texture_file_path = asset_dir_path / "textures" / "metaworld" / "metal1.png"
    metal1_texture_name = box_name + "_metal1_texture"
    generator.add_texture(TextureConfig(name=metal1_texture_name, type="2d", file=metal1_texture_file_path))
    metal1_material_name = box_name + "_metal1_material"
    generator.add_material(
        MaterialConfig(
            name=metal1_material_name,
            texture=metal1_texture_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0.75, 0.75, 0.75, 1.0]),
        )
    )

    metal2_texture_file_path = asset_dir_path / "textures" / "metaworld" / "metal2.png"
    metal2_texture_name = box_name + "_metal2_texture"
    generator.add_texture(TextureConfig(name=metal2_texture_name, type="2d", file=metal2_texture_file_path))
    metal2_material_name = box_name + "_metal2_material"
    generator.add_material(
        MaterialConfig(
            name=metal2_material_name,
            texture=metal2_texture_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0.3, 0.32, 0.35, 1.0]),
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    # add assets
    safe_mesh_name = box_name + "_safe_mesh"
    safe_mesh_file_path = metaworld_asset_dir_path / "hinged_box" / "safe.stl"
    generator.add_mesh(
        MeshConfig(
            name=safe_mesh_name,
            file=safe_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    door_mesh_name = box_name + "_door_mesh"
    door_mesh_file_path = metaworld_asset_dir_path / "hinged_box" / "door.stl"
    generator.add_mesh(
        MeshConfig(
            name=door_mesh_name,
            file=door_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    door_handle_mesh_name = box_name + "_door_handle_mesh"
    door_handle_mesh_file_path = metaworld_asset_dir_path / "hinged_box" / "door_handle.stl"
    generator.add_mesh(
        MeshConfig(
            name=door_handle_mesh_name,
            file=door_handle_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    handle_base_mesh_name = box_name + "_handle_base_mesh"
    handle_base_mesh_file_path = metaworld_asset_dir_path / "hinged_box" / "handle_base.stl"
    generator.add_mesh(
        MeshConfig(
            name=handle_base_mesh_name,
            file=handle_base_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )

    # box body
    generator.add_body(
        mjcf_config=BodyConfig(name=box_body_name, pos=box_position, quat=box_rotation),
        parent_name="worldbody",
    )

    # safe geom (visual)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_geom",
            mesh=safe_mesh_name,
            type="mesh",
            material=metal2_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=box_body_name,
    )
    # safe geom (collision)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_collision_geom_1",
            pos=np.array([-0.204, 0.0, 0.0]),
            size=np.array([0.016, 0.106, 0.15]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_collision_geom_2",
            pos=np.array([0.204, 0.0, 0.0]),
            size=np.array([0.016, 0.106, 0.15]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_collision_geom_3",
            pos=np.array([0.0, 0.0, 0.138]),
            size=np.array([0.188, 0.106, 0.012]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_collision_geom_4",
            pos=np.array([0.0, 0.0, -0.138]),
            size=np.array([0.189, 0.106, 0.012]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_safe_collision_geom_5",
            pos=np.array([0.0, 0.094, 0.0]),
            size=np.array([0.188, 0.012, 0.126]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
        ),
        parent_name=box_body_name,
    )

    # door body
    door_body_name = box_name + "_door_body"
    generator.add_body(
        mjcf_config=BodyConfig(name=door_body_name, pos=np.array([-0.185, -0.1, 0.0])),
        parent_name=box_body_name,
    )
    # door joint
    generator.add_joint(
        mjcf_config=JointConfig(
            name=box_joint_name,
            type="hinge",
            axis=np.array([0.0, 0.0, 1.0]),
            range=np.array([-np.deg2rad(60.0), 0.0]),
            armature=0.01,
            damping=1.0,
        ),
        parent_name=door_body_name,
    )
    # door geom (visual)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_geom",
            mesh=door_mesh_name,
            type="mesh",
            material=metal2_material_name,
            pos=np.array([0.185, 0.0, 0.0]),
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=door_body_name,
    )
    # door handle geom (visual)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_handle_geom",
            mesh=door_handle_mesh_name,
            type="mesh",
            material=metal1_material_name,
            euler=np.array([1.57, 0.0, 0.0]),
            pos=np.array([0.325, -0.062, 0.0]),
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=door_body_name,
    )
    # handle base geom (visual)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_handle_base_geom",
            mesh=handle_base_mesh_name,
            type="mesh",
            material=metal1_material_name,
            pos=np.array([0.325, -0.006, 0.0]),
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=door_body_name,
    )
    # door geom (cylinder visual)
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_cylinder_geom_1",
            pos=np.array([0.0, 0.0, 0.07]),
            size=np.array([0.013, 0.045]),
            type="cylinder",
            material=metal1_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=door_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_cylinder_geom_2",
            pos=np.array([0.0, 0.0, -0.07]),
            size=np.array([0.013, 0.045]),
            type="cylinder",
            material=metal1_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=door_body_name,
    )

    # collision geoms for door
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_handle_collision_geom_1",
            euler=np.array([1.57, 0.0, 0.0]),
            pos=np.array([0.325, -0.006, 0.0]),
            size=np.array([0.028, 0.012]),
            type="cylinder",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=door_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_handle_mesh_name + "_collision_geom_2",
            euler=np.array([1.57, 0.0, 0.0]),
            pos=np.array([0.325, -0.065, 0.0]),
            size=np.array([0.013, 0.047]),
            type="cylinder",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=door_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_handle_mesh_name + "_collision_geom_3",
            euler=np.array([0.0, 1.57, 0.0]),
            pos=np.array([0.381, -0.12, 0.0]),
            size=np.array([0.019, 0.075]),
            type="cylinder",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=door_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_handle_mesh_name + "_collision_geom_4",
            euler=np.array([0.0, 1.57, 0.0]),
            pos=np.array([0.395, -0.12, 0.0]),
            size=np.array([0.023, 0.054]),
            type="cylinder",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=door_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_name + "_door_collision_geom_5",
            pos=np.array([0.185, 0.0, 0.0]),
            size=np.array([0.18, 0.01, 0.123]),
            type="box",
            material=None,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=door_body_name,
    )

    generator.add_site(
        mjcf_config=SiteConfig(
            name=box_handle_site_name,
            pos=np.array([0.39, -0.11, 0.0]),
            size=np.array([0.025]),
            rgba=np.array([1.0, 0.0, 0.0, 0.0]),
            type="sphere",
        ),
        parent_name=door_body_name,
    )

    return box_body_name
