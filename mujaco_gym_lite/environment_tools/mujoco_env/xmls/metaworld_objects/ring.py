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
from mujaco_gym_lite.utils.transforms import euler_to_quat


def add_ring(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    ring_name: str,
    ring_position: npt.NDArray,
    ring_rotation: npt.NDArray,
    ring_body_name: str,
    ring_handle_site_name: str,
    ring_site_name: str,
    ring_joint_name: str,
    ring_density: float = 500,
    ring_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    ring_solref: npt.NDArray = np.array([0.01, 1]),
    ring_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    material_col_name = ring_name + "_col_material"
    generator.add_material(
        MaterialConfig(
            name=material_col_name,
            rgba=np.array([0.3, 0.3, 1.0, 0.5]),
            shininess=0.0,
            specular=0.0,
        )
    )
    material_green_name = ring_name + "_green_material"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    texture_name = ring_name + "_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = ring_name + "_metal_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_metal_name,
            rgba=np.array([0.65, 0.65, 0.65, 1]),
            shininess=1.0,
            reflectance=0.2,
            specular=0.5,
        )
    )

    texture_name = ring_name + "_wood_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "wood2.png")
    )
    material_wood_name = ring_name + "_wood_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_wood_name,
            shininess=1.0,
            reflectance=0.0,
            specular=0.5,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    ring_handle_mesh_name = ring_name + "_ring_handle"
    ring_handle_mesh_file_path = metaworld_asset_dir_path / "ring" / "ring_handle.stl"
    generator.add_mesh(
        MeshConfig(
            name=ring_handle_mesh_name,
            file=ring_handle_mesh_file_path,
            scale=np.array([2.0, 3.0, 3.5]),
        )
    )
    ring_mesh_name = ring_name + "_ring"
    ring_mesh_file_path = metaworld_asset_dir_path / "ring" / "ring.stl"
    generator.add_mesh(MeshConfig(name=ring_mesh_name, file=ring_mesh_file_path, scale=np.array([2.0, 2.0, 2.0])))
    rod_mesh_name = ring_name + "_rod"
    rod_mesh_file_path = metaworld_asset_dir_path / "ring" / "rod.stl"
    generator.add_mesh(MeshConfig(name=rod_mesh_name, file=rod_mesh_file_path, scale=np.array([2.0, 2.0, 2.0])))

    generator.add_body(
        mjcf_config=BodyConfig(name=ring_body_name, pos=ring_position, quat=ring_rotation),
        parent_name="worldbody",
    )
    generator.add_joint(JointConfig(type="free", name=ring_joint_name), parent_name=ring_body_name)

    ring_box_body_name = ring_name + "_box_body"
    ring_box_joint_name = ring_name + "_box_joint"
    generator.add_body(
        mjcf_config=BodyConfig(name=ring_box_body_name, pos=ring_position, quat=ring_rotation),
        parent_name="worldbody",
    )
    generator.add_joint(
        JointConfig(type="slide", axis=np.array([1, 0, 0]), name=ring_box_joint_name), parent_name=ring_box_body_name
    )
    generator.add_geom(
        GeomConfig(
            name=ring_name + "_box",
            group=1,
            condim=4,
            material=material_wood_name,
            type="box",
            size=np.array([0.05, 0.1, 0.1]),
            pos=np.array([-0.09, -0.275, 0.025]),
            quat=euler_to_quat([0.0, 0.0, 0.0], order="zxy"),
            density=ring_density,
            friction=ring_friction,
            solimp=ring_solimp,
            solref=ring_solref,
        ),
        parent_name=ring_box_body_name,
    )

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=ring_name + "_main_visual_geom",
            mesh=ring_mesh_name,
            type="mesh",
            material=material_metal_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=ring_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=ring_name + "_handle_visual_geom",
            mesh=ring_handle_mesh_name,
            type="mesh",
            material=material_green_name,
            pos=np.array([0.0, -0.275, 0.025]),
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=ring_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=ring_name + "_cylinder_visual_geom",
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
        parent_name=ring_body_name,
    )

    # collision mesh
    ring_configs = [
        {
            "suffix": "pos_x",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([0.048 * 2.0, 0.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "pos_y",
            "euler": np.array([0.0, 1.57, 0.0]),
            "pos": np.array([0.0, 0.048 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "diag_ne",
            "euler": np.array([1.57, 0.785, 0.785]),
            "pos": np.array([0.036 * 2.0, 0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "neg_x",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([-0.048 * 2.0, 0.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "neg_y",
            "euler": np.array([0.0, 1.57, 0.0]),
            "pos": np.array([0.0, -0.048 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "diag_nw",
            "euler": np.array([1.57, -0.785, -0.785]),
            "pos": np.array([-0.036 * 2.0, 0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "diag_sw",
            "euler": np.array([1.57, 0.785, 0.785]),
            "pos": np.array([-0.036 * 2.0, -0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "diag_se",
            "euler": np.array([1.57, -0.785, -0.785]),
            "pos": np.array([0.036 * 2.0, -0.036 * 2.0, 0.0]),
            "size": np.array([0.019 * 1.7, 0.015 * 2.0]),
            "type": "capsule",
            "mass": 0.005,
        },
        {
            "suffix": "box_low",
            "euler": np.array([0.0, 0.0, 0.0]),
            "pos": np.array([0.0, -0.275, 0.025]),
            "size": np.array([0.019 * 2.0, 0.038 * 3.0, 0.016 * 3.5]),
            "type": "box",
            "mass": 0.25,
        },
        {
            "suffix": "cyl_mid",
            "euler": np.array([1.57, 0.0, 0.0]),
            "pos": np.array([0.0, -0.15, 0.0]),
            "size": np.array([0.012, 0.05]),
            "type": "cylinder",
            "mass": 0.005,
        },
    ]

    for cfg in ring_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{ring_name}_main_{cfg['suffix']}_geom",
                type=cfg["type"],
                pos=cfg["pos"],
                euler=cfg["euler"],
                size=cfg["size"],
                mass=cfg["mass"],
                density=ring_density,
                friction=ring_friction,
                solimp=ring_solimp,
                solref=ring_solref,
                group=3,
                condim=4,
            ),
            parent_name=ring_body_name,
        )

    generator.add_site(
        SiteConfig(
            name=ring_handle_site_name,
            pos=np.array([-0.0, -0.275, 0.0275]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.05]),
        ),
        parent_name=ring_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=ring_site_name,
            pos=np.array([-0.0, 0.0, 0.0]),
            rgba=np.array([0.1, 0.8, 0.1, 0.5]),
            size=np.array([0.015, 0.015, 0.015]),
            type="sphere",
        ),
        parent_name=ring_body_name,
    )
    return {"ring_body": ring_body_name}


def add_rod(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    rod_name: str,
    rod_position: npt.NDArray,
    rod_rotation: npt.NDArray,
    rod_body_name: str,
    rod_target_top_site_name: str,
    rod_target_bottom_site_name: str,
    rod_density: float = 500,
    rod_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    rod_solref: npt.NDArray = np.array([0.01, 1]),
    rod_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    texture_name = rod_name + "_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "darkwood.png")
    )
    material_darkwood_name = rod_name + "_darkwood_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_darkwood_name,
            shininess=1.0,
            reflectance=0.0,
            specular=0.5,
        )
    )

    texture_name = rod_name + "_navy_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "navy_blue.png")
    )
    material_navy_name = rod_name + "_navy_material"
    generator.add_material(
        MaterialConfig(
            texture=texture_name,
            name=material_navy_name,
            shininess=1.0,
            reflectance=0.0,
            specular=0.5,
        )
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=rod_body_name, pos=rod_position, quat=rod_rotation),
        parent_name="worldbody",
    )

    generator.add_geom(
        GeomConfig(
            name=rod_name + "_main",
            group=1,
            condim=4,
            material=material_navy_name,
            type="cylinder",
            size=np.array([0.03, 0.075]),
            pos=np.array([0.0, 0.0, 0.075]),
            density=rod_density,
            friction=rod_friction,
            solimp=rod_solimp,
            solref=rod_solref,
        ),
        parent_name=rod_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=rod_name + "_wall",
            group=1,
            condim=4,
            material=material_darkwood_name,
            type="box",
            size=np.array([0.01, 0.3, 0.125]),
            pos=np.array([0.0, -0.075, 0.0]),
            quat=euler_to_quat([np.pi * 0.5, 0.0, 0.0], order="yxz"),
            density=rod_density,
            friction=rod_friction,
            solimp=rod_solimp,
            solref=rod_solref,
        ),
        parent_name=rod_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=rod_target_bottom_site_name,
            pos=np.array([0.0, 0.0, 0.0]),
            rgba=np.array([0.2, 0.5, 0.0, 1.0]),
            size=np.array([0.01, 0.01, 0.01]),
        ),
        parent_name=rod_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=rod_target_top_site_name,
            pos=np.array([0.0, 0.0, 0.15]),
            rgba=np.array([0.2, 0.5, 0.0, 1.0]),
            size=np.array([0.01, 0.01, 0.01]),
        ),
        parent_name=rod_body_name,
    )
    return {}
