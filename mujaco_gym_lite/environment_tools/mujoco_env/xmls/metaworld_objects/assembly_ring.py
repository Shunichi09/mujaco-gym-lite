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


def add_assembly_ring(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    assembly_ring_name: str,
    assembly_ring_position: npt.NDArray,
    assembly_ring_rotation: npt.NDArray,
    assembly_ring_body_name: str,
    assembly_ring_handle_site_name: str,
    assembly_ring_site_name: str,
    assembly_ring_joint_name: str,
    assembly_ring_density: float = 500,
    assembly_ring_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    assembly_ring_solref: npt.NDArray = np.array([0.01, 1]),
    assembly_ring_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    material_col_name = assembly_ring_name + "_col_material"
    generator.add_material(
        MaterialConfig(
            name=material_col_name,
            rgba=np.array([0.3, 0.3, 1.0, 0.5]),
            shininess=0.0,
            specular=0.0,
        )
    )
    material_green_name = assembly_ring_name + "_green_material"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    texture_name = assembly_ring_name + "_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = assembly_ring_name + "_metal_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_metal_name,
            rgba=np.array([0.65, 0.65, 0.65, 1]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    assembly_ring_handle_mesh_name = assembly_ring_name + "_ring_handle"
    assembly_ring_handle_mesh_file_path = metaworld_asset_dir_path / "assembly_ring" / "assembly_ring_handle.stl"
    generator.add_mesh(
        MeshConfig(
            name=assembly_ring_handle_mesh_name,
            file=assembly_ring_handle_mesh_file_path,
            scale=np.array([2.0, 3.0, 3.5]),
        )
    )
    assembly_ring_mesh_name = assembly_ring_name + "_ring"
    assembly_ring_mesh_file_path = metaworld_asset_dir_path / "assembly_ring" / "assembly_ring.stl"
    generator.add_mesh(
        MeshConfig(name=assembly_ring_mesh_name, file=assembly_ring_mesh_file_path, scale=np.array([2.0, 2.0, 2.0]))
    )
    assembly_rod_mesh_name = assembly_ring_name + "_ring_rod"
    assembly_rod_mesh_file_path = metaworld_asset_dir_path / "assembly_ring" / "assembly_rod.stl"
    generator.add_mesh(
        MeshConfig(name=assembly_rod_mesh_name, file=assembly_rod_mesh_file_path, scale=np.array([2.0, 2.0, 2.0]))
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=assembly_ring_body_name, pos=assembly_ring_position, quat=assembly_ring_rotation),
        parent_name="worldbody",
    )
    generator.add_joint(JointConfig(type="free", name=assembly_ring_joint_name), parent_name=assembly_ring_body_name)

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=assembly_ring_name + "_main_visual_geom",
            mesh=assembly_ring_mesh_name,
            type="mesh",
            material=material_metal_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=assembly_ring_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=assembly_ring_name + "_handle_visual_geom",
            mesh=assembly_ring_handle_mesh_name,
            type="mesh",
            material=material_green_name,
            pos=np.array([0.0, -0.275, 0.025]),
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=assembly_ring_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=assembly_ring_name + "_cylinder_visual_geom",
            type="cylinder",
            material=material_metal_name,
            pos=np.array([0.0, -0.15, 0.0]),
            size=np.array([0.012, 0.05]),
            euler=np.array([1.57, 0.0, 0.0]),
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=assembly_ring_body_name,
    )

    # collision mesh
    ring_configs = [
        {
            "suffix": "pos_x",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([0.048 * 2.0, 0.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "pos_y",
            "euler": np.array([0.0, 1.57, 0.0]),
            "pos": np.array([0.0, 0.048 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "diag_ne",
            "euler": np.array([1.57, 0.785, 0.785]),
            "pos": np.array([0.036 * 2.0, 0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "neg_x",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([-0.048 * 2.0, 0.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "neg_y",
            "euler": np.array([0.0, 1.57, 0.0]),
            "pos": np.array([0.0, -0.048 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "diag_nw",
            "euler": np.array([1.57, -0.785, -0.785]),
            "pos": np.array([-0.036 * 2.0, 0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "diag_sw",
            "euler": np.array([1.57, 0.785, 0.785]),
            "pos": np.array([-0.036 * 2.0, -0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "diag_se",
            "euler": np.array([1.57, -0.785, -0.785]),
            "pos": np.array([0.036 * 2.0, -0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.02,
        },
        {
            "suffix": "box_low",
            "euler": np.array([0.0, 0.0, 0.0]),
            "pos": np.array([0.0, -0.275, 0.025]),
            "size": np.array([0.019 * 2.0, 0.038 * 3.0, 0.016 * 3.5]),
            "type": "box",
            "mass": 0.1,
        },
        {
            "suffix": "cyl_mid",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([0.0, -0.15, 0.0]),
            "size": np.array([0.012, 0.05]),
            "type": "cylinder",
            "mass": 0.02,
        },
    ]

    for cfg in ring_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{assembly_ring_name}_main_{cfg['suffix']}_geom",
                type=cfg["type"],
                pos=cfg["pos"],
                euler=cfg["euler"],
                size=cfg["size"],
                mass=cfg["mass"],
                density=assembly_ring_density,
                friction=assembly_ring_friction,
                solimp=assembly_ring_solimp,
                solref=assembly_ring_solref,
                group=3,
                condim=4,
            ),
            parent_name=assembly_ring_body_name,
        )

    generator.add_site(
        SiteConfig(
            name=assembly_ring_handle_site_name,
            pos=np.array([-0.0, -0.275, 0.0275]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.05]),
        ),
        parent_name=assembly_ring_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=assembly_ring_site_name,
            pos=np.array([-0.0, 0.0, 0.0]),
            rgba=np.array([0.1, 0.8, 0.1, 0.5]),
            size=np.array([0.015, 0.015, 0.015]),
            type="sphere",
        ),
        parent_name=assembly_ring_body_name,
    )
    return {"ring_body": assembly_ring_body_name}


def add_assembly_rod(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    assembly_rod_name: str,
    assembly_rod_position: npt.NDArray,
    assembly_rod_rotation: npt.NDArray,
    assembly_rod_body_name: str,
    assembly_rod_target_top_site_name: str,
    assembly_rod_target_bottom_site_name: str,
    assembly_rod_density: float = 500,
    assembly_rod_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    assembly_rod_solref: npt.NDArray = np.array([0.01, 1]),
    assembly_rod_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    texture_name = assembly_rod_name + "_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = assembly_rod_name + "_metal_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_metal_name,
            rgba=np.array([0.65, 0.65, 0.65, 1]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    material_light_pink_name = assembly_rod_name + "_light_pink_material"
    generator.add_material(
        MaterialConfig(
            name=material_light_pink_name,
            rgba=np.array([0.85, 0.45, 0.55, 1.0]),
            shininess=0.15,
            reflectance=0.05,
            specular=0.08,
        )
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=assembly_rod_body_name, pos=assembly_rod_position, quat=assembly_rod_rotation),
        parent_name="worldbody",
    )

    generator.add_geom(
        GeomConfig(
            name=assembly_rod_name + "_main",
            group=1,
            condim=4,
            material=material_light_pink_name,
            type="cylinder",
            size=np.array([0.03, 0.075]),
            pos=np.array([0.0, 0.0, 0.075]),
            density=assembly_rod_density,
            friction=assembly_rod_friction,
            solimp=assembly_rod_solimp,
            solref=assembly_rod_solref,
        ),
        parent_name=assembly_rod_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=assembly_rod_target_bottom_site_name,
            pos=np.array([0.0, 0.0, 0.0]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.05]),
        ),
        parent_name=assembly_rod_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=assembly_rod_target_top_site_name,
            pos=np.array([0.0, 0.0, 0.15]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.05]),
        ),
        parent_name=assembly_rod_body_name,
    )
    return {}
