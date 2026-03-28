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


def add_lidded_box(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    box_name: str,
    box_position: npt.NDArray,
    box_rotation: npt.NDArray,
    box_body_name: str,
    box_collision_name: str,
    box_center_top_site_name: str,
    lid_body_name: str,
    lid_position: npt.NDArray,
    lid_rotation: npt.NDArray,
    lid_joint_name: str,
    lid_handle_site_name: str,
    lid_handle_low_site_name: str,
    lid_handle_mesh_name: str,
    lid_collision_name: str,
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

    blue_material_name = box_name + "_blue_material"
    generator.add_material(
        MaterialConfig(
            name=blue_material_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0, 0, 0.8, 1.0]),
        )
    )

    red_material_name = box_name + "_red_material"
    generator.add_material(
        MaterialConfig(
            name=red_material_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0.8, 0, 0, 1.0]),
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    # add assets
    bin_mesh_name = box_name + "_bin_mesh"
    generator.add_mesh(
        MeshConfig(
            name=bin_mesh_name,
            file=metaworld_asset_dir_path / "lidded_box" / "bin.stl",
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    box_lid_mesh_name = box_name + "_box_lid_mesh"
    generator.add_mesh(
        MeshConfig(
            name=box_lid_mesh_name,
            file=metaworld_asset_dir_path / "lidded_box" / "boxtop.stl",
            scale=np.array([1.0, 1.0, 1.15]),
        )
    )
    box_handle_mesh_name = box_name + "_box_handle_mesh"
    generator.add_mesh(
        MeshConfig(
            name=box_handle_mesh_name,
            file=metaworld_asset_dir_path / "lidded_box" / "boxhandle.stl",
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )

    # box body
    generator.add_body(
        mjcf_config=BodyConfig(name=box_body_name, pos=box_position, quat=box_rotation),
        parent_name="worldbody",
    )
    # lid body
    generator.add_body(
        mjcf_config=BodyConfig(name=lid_body_name, pos=lid_position, quat=lid_rotation),
        parent_name="worldbody",
    )

    # visual mesh
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=bin_mesh_name + "_geom",
            type="mesh",
            mesh=bin_mesh_name,
            material=blue_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.0, 0.03]),
            conaffinity=0,
            contype=0,
            group=2,
        ),
        parent_name=box_body_name,
    )
    # collision geoms
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_collision_name + "_front_geom",
            type="box",
            size=np.array([0.1, 0.005, 0.03]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, -0.095, 0.03]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_collision_name + "_back_geom",
            type="box",
            size=np.array([0.1, 0.005, 0.03]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.095, 0.03]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_collision_name + "_right_geom",
            type="box",
            size=np.array([0.005, 0.09, 0.03]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.095, 0.0, 0.03]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_collision_name + "_left_geom",
            type="box",
            size=np.array([0.005, 0.09, 0.03]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([-0.095, 0.0, 0.03]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=box_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=box_collision_name + "_bottom_geom",
            type="box",
            size=np.array([0.1, 0.1, 0.005]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.0, 0.005]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=box_body_name,
    )
    generator.add_site(
        mjcf_config=SiteConfig(
            name=box_center_top_site_name,
            pos=np.array([0.0, 0.0, 0.06]),
            size=np.array([0.01]),
            type="sphere",
            rgba=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        parent_name=box_body_name,
    )

    yellow_rgba = np.array([0.8, 0.8, 0.0, 1.0])
    site_r = 0.01
    top_z = 0.03 + 0.03
    eps_z = -0.0055
    corner_xy = 0.095
    corner_positions = [
        (+corner_xy, +corner_xy, top_z + eps_z),
        (+corner_xy, -corner_xy, top_z + eps_z),
        (-corner_xy, +corner_xy, top_z + eps_z),
        (-corner_xy, -corner_xy, top_z + eps_z),
    ]
    for i, (x, y, z) in enumerate(corner_positions, start=1):
        generator.add_site(
            mjcf_config=SiteConfig(
                name=f"{box_name}_corner{i}_site",
                pos=np.array([x, y, z]),
                size=np.array([site_r]),
                type="sphere",
                rgba=yellow_rgba,
            ),
            parent_name=box_body_name,
        )

    # lid joint
    generator.add_joint(
        JointConfig(type="slide", axis=np.array([1.0, 0.0, 0.0]), name=lid_joint_name + "_x"), parent_name=lid_body_name
    )
    generator.add_joint(
        JointConfig(type="slide", axis=np.array([0.0, 1.0, 0.0]), name=lid_joint_name + "_y"), parent_name=lid_body_name
    )
    generator.add_joint(
        JointConfig(type="slide", axis=np.array([0.0, 0.0, 1.0]), name=lid_joint_name + "_z"), parent_name=lid_body_name
    )

    # lid visual mesh
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_body_name + "_geom",
            type="mesh",
            mesh=box_lid_mesh_name,
            material=red_material_name,
            pos=np.array([0.0, 0.0, -0.005]),
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            conaffinity=0,
            contype=0,
            group=2,
        ),
        parent_name=lid_body_name,
    )
    # handle visual mesh
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_handle_mesh_name + "_geom",
            type="mesh",
            mesh=box_handle_mesh_name,
            material=metal1_material_name,
            pos=np.array([0.0, 0.0, 0.082]),
            euler=np.array([1.57, 0.0, 0.0]),
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            conaffinity=0,
            contype=0,
            group=2,
        ),
        parent_name=lid_body_name,
    )
    # add collision geoms
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_center_collision_geom",
            type="box",
            size=np.array([0.115, 0.115, 0.003]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.0, 0.003]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_right_collision_geom",
            type="box",
            size=np.array([0.005, 0.115, 0.008]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.11, 0.0, -0.008]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_left_collision_geom",
            type="box",
            size=np.array([0.005, 0.115, 0.008]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([-0.11, 0.0, -0.008]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_front_collision_geom",
            type="box",
            size=np.array([0.115, 0.005, 0.008]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, -0.11, -0.008]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_back_collision_geom",
            type="box",
            size=np.array([0.115, 0.005, 0.008]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.11, -0.008]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_top_collision_geom",
            type="capsule",
            size=np.array([0.008, 0.05]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.0, 0.082]),
            euler=np.array([1.57, 0.0, 0.0]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_left_side_collision_geom",
            type="capsule",
            size=np.array([0.008, 0.035]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, -0.05, 0.043]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    generator.add_geom(
        mjcf_config=GeomConfig(
            name=lid_collision_name + "_right_side_collision_geom",
            type="capsule",
            size=np.array([0.008, 0.035]),
            material=red_material_name,
            density=box_density,
            solimp=box_solimp,
            solref=box_solref,
            pos=np.array([0.0, 0.05, 0.043]),
            conaffinity=1,
            contype=1,
            condim=4,
            group=3,
        ),
        parent_name=lid_body_name,
    )
    # handle site
    generator.add_site(
        mjcf_config=SiteConfig(
            name=lid_handle_site_name,
            pos=np.array([0.0, 0.0, 0.082]),
            size=np.array([0.01]),
            type="sphere",
            rgba=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        parent_name=lid_body_name,
    )
    generator.add_site(
        mjcf_config=SiteConfig(
            name=lid_handle_low_site_name,
            pos=np.array([0.0, 0.0, -0.01]),
            size=np.array([0.01]),
            type="sphere",
            rgba=np.array([0.0, 1.0, 0.0, 0.0]),
        ),
        parent_name=lid_body_name,
    )

    lid_cian_rgba = np.array([0.0, 1.0, 1.0, 1.0])
    lid_site_r = 0.01

    lid_top_z = 0.003 + 0.003
    lid_eps_z = -0.025
    lid_corner_xy = 0.11

    lid_corner_positions = [
        (+lid_corner_xy, +lid_corner_xy, lid_top_z + lid_eps_z),
        (+lid_corner_xy, -lid_corner_xy, lid_top_z + lid_eps_z),
        (-lid_corner_xy, +lid_corner_xy, lid_top_z + lid_eps_z),
        (-lid_corner_xy, -lid_corner_xy, lid_top_z + lid_eps_z),
    ]

    for i, (x, y, z) in enumerate(lid_corner_positions, start=1):
        generator.add_site(
            mjcf_config=SiteConfig(
                name=f"{box_name}_lid_corner{i}_site",
                pos=np.array([x, y, z]),
                size=np.array([lid_site_r]),
                type="sphere",
                rgba=lid_cian_rgba,
            ),
            parent_name=lid_body_name,
        )

    return {}
