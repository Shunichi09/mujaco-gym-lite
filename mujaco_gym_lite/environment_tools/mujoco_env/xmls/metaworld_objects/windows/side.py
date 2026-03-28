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


def add_side_window(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    window_name: str,
    window_position: npt.NDArray,
    window_rotation: npt.NDArray,
    window_body_name: str,
    window_handle_site_name: str,
    window_handle_name: str,
    window_joint_name: str,
    window_density: float = 500,
    window_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    window_solref: npt.NDArray = np.array([0.01, 1]),
    window_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    texture_name = window_name + "_metal_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = window_name + "_metal_material"
    generator.add_material(
        MaterialConfig(
            name=material_metal_name,
            texture=texture_name,
            rgba=np.array([0.65, 0.65, 0.65, 1]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    texture_name = window_name + "_navy_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="2d", file=asset_dir_path / "textures" / "metaworld" / "navy_blue.png")
    )
    material_navy_name = window_name + "_navy_material"
    generator.add_material(MaterialConfig(name=material_navy_name, texture=texture_name))

    material_green_name = window_name + "_green_texture"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.51, 0.58, 0.55, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    texture_name = window_name + "_wood_texture"
    generator.add_texture(
        TextureConfig(
            name=texture_name,
            type="2d",
            file=asset_dir_path / "textures" / "metaworld" / "wood1.png",
        )
    )
    material_wood_name = window_name + "_wood_material"
    generator.add_material(MaterialConfig(name=material_wood_name, texture=texture_name))

    material_glass_name = window_name + "_glass_material"
    generator.add_material(
        MaterialConfig(
            name=material_glass_name,
            rgba=np.array([0.0, 0.3, 0.4, 0.8]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"

    window_h_base_mesh_name = window_name + "_window_h_base_mesh"
    window_h_base_mesh_file_path = metaworld_asset_dir_path / "window" / "window_h_base.stl"
    generator.add_mesh(
        MeshConfig(name=window_h_base_mesh_name, file=window_h_base_mesh_file_path, scale=np.array([1.3, 1.0, 1.0]))
    )

    window_h_frame_mesh_name = window_name + "_window_h_frame_mesh"
    window_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "window_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=window_h_frame_mesh_name, file=window_h_frame_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    windowa_h_frame_mesh_name = window_name + "_windowa_h_frame_mesh"
    windowa_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=windowa_h_frame_mesh_name, file=windowa_h_frame_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    windowa_h_glass_mesh_name = window_name + "_windowa_h_glass_mesh"
    windowa_h_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_h_glass.stl"
    generator.add_mesh(
        MeshConfig(name=windowa_h_glass_mesh_name, file=windowa_h_glass_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    windowb_h_frame_mesh_name = window_name + "_windowb_h_frame_mesh"
    windowb_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=windowb_h_frame_mesh_name, file=windowb_h_frame_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    windowb_h_glass_mesh_name = window_name + "_windowb_h_glass_mesh"
    windowb_h_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_h_glass.stl"
    generator.add_mesh(
        MeshConfig(name=windowb_h_glass_mesh_name, file=windowb_h_glass_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=window_body_name, pos=window_position, quat=window_rotation),
        parent_name="worldbody",
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=window_name + "_joint_attached_body", pos=np.array([0.0, 0.0, 0.0])),
        parent_name=window_body_name,
    )
    generator.add_joint(
        JointConfig(
            name=window_joint_name,
            pos=np.array([0.0, 0.0, 0.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            type="slide",
            limited="true",
            armature=0.001,
            range=np.array([0, 0.25]),
            damping=1.5,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=window_name + "_window_base_visual_geom",
            mesh=window_h_base_mesh_name,
            pos=np.array([0.0, 0.0, -0.192 - 0.05]),
            type="mesh",
            material=material_navy_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_frame_visual_geom",
            mesh=window_h_frame_mesh_name,
            type="mesh",
            material=material_navy_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_body_name,
    )

    # collision mesh
    column_configs = [
        {
            "suffix": "col_left",
            "pos": np.array([-0.216 - 0.06, 0.0, 0.0]),
            "size": np.array([0.015 * 1.3, 0.03, 0.181 * 1.3]),
        },
        {
            "suffix": "horz_bottom",
            "pos": np.array([0.0, 0.0, -0.192 - 0.05]),
            "size": np.array([0.25 * 1.3, 0.06, 0.01 * 1.3]),
        },
        {
            "suffix": "col_right",
            "pos": np.array([0.216 + 0.06, 0.0, 0.0]),
            "size": np.array([0.015 * 1.3, 0.03, 0.181 * 1.3]),
        },
        {
            "suffix": "mid_lower",
            "pos": np.array([0.0, 0.0, -0.166 - 0.05]),
            "size": np.array([0.201 * 1.3, 0.03, 0.015 * 1.3]),
        },
        {
            "suffix": "mid_upper",
            "pos": np.array([0.0, 0.0, 0.166 + 0.05]),
            "size": np.array([0.201 * 1.3, 0.03, 0.015 * 1.3]),
        },
    ]
    for cfg in column_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{window_name}_window_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                group=3,
                condim=4,
                density=window_density,
                friction=window_friction,
                solimp=window_solimp,
                solref=window_solref,
            ),
            parent_name=window_body_name,
        )
    #################################################################
    # visual mesh ###################################################
    #################################################################
    attach_body = window_name + "_joint_attached_body"
    # handle visual mesh
    configs = [
        {
            "suffix": "cylinder_1",
            "type": "cylinder",
            "pos": np.array([-0.014, -0.028, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": "cylinder_2",
            "type": "cylinder",
            "pos": np.array([-0.014, -0.028, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": "capsule_1",
            "type": "capsule",
            "pos": np.array([-0.014, -0.06 - 0.005, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": "capsule_2",
            "type": "capsule",
            "pos": np.array([-0.014, -0.06 - 0.005, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": "capsule_3",
            "type": "capsule",
            "pos": np.array([-0.014, -0.11, 0.0]),
            "euler": None,
            "size": np.array([0.008 + 0.005, 0.045 + 0.075]),
        },
    ]
    for cfg in configs:
        params = dict(
            name=f"{window_handle_name}_{cfg['suffix']}_geom",
            pos=cfg["pos"],
            size=cfg["size"],
            type=cfg["type"],
            material=material_metal_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        )
        if cfg["euler"] is not None:
            params["euler"] = cfg["euler"]
        generator.add_geom(
            GeomConfig(**params),
            parent_name=attach_body,
        )

    # Slide body visual mesh geoms
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowb_h_frame_visual_geom",
            mesh=windowb_h_frame_mesh_name,
            pos=np.array([-0.104 - 0.028, -0.022, 0.0]),
            type="mesh",
            material=material_wood_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowb_h_glass_visual_geom",
            mesh=windowb_h_glass_mesh_name,
            pos=np.array([-0.104 - 0.028, -0.018, 0.0]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    # capsule collisions
    capsule_cylinder_configs = [
        {
            "suffix": 1,
            "type": "cylinder",
            "pos": np.array([-0.014, -0.028, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": 2,
            "type": "cylinder",
            "pos": np.array([-0.014, -0.028, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": 3,
            "type": "capsule",
            "pos": np.array([-0.014, -0.06 - 0.005, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": 4,
            "type": "capsule",
            "pos": np.array([-0.014, -0.06 - 0.005, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": 5,
            "type": "capsule",
            "pos": np.array([-0.014, -0.11, 0.0]),
            "euler": None,
            "size": np.array([0.008 + 0.005, 0.045 + 0.075]),
        },
    ]
    for cfg in capsule_cylinder_configs:
        params = dict(
            name=f"{window_handle_name}_window_col_{cfg['suffix']}_geom",
            pos=cfg["pos"],
            size=cfg["size"],
            type=cfg["type"],
            material=material_green_name,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        )
        if cfg["euler"] is not None:
            params["euler"] = cfg["euler"]

        generator.add_geom(
            GeomConfig(**params),
            parent_name=attach_body,
        )

    box_configs = [
        {"suffix": 12, "pos": np.array([-0.014, -0.013, 0.0]), "size": np.array([0.014 * 1.5, 0.012, 0.15 * 1.3])},
        {
            "suffix": 13,
            "pos": np.array([-0.19 - 0.045, -0.013, 0.0]),
            "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3]),
        },
        {
            "suffix": 14,
            "pos": np.array([-0.104 - 0.03, -0.013, 0.14 + 0.04]),
            "size": np.array([0.076 * 1.5, 0.012, 0.01 * 1.3]),
        },
        {
            "suffix": 15,
            "pos": np.array([-0.104 - 0.03, -0.013, -0.14 - 0.04]),
            "size": np.array([0.076 * 1.5, 0.012, 0.01 * 1.3]),
        },
        {
            "suffix": 16,
            "pos": np.array([-0.104 - 0.03, -0.018, 0.0]),
            "size": np.array([0.076 * 1.5, 0.001, 0.13 * 1.3]),
        },
        {
            "suffix": 17,
            "pos": np.array([-0.104 - 0.03, -0.022, 0.0]),
            "size": np.array([0.076 * 1.5, 0.003, 0.005 * 1.3]),
        },
    ]

    for cfg in box_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{window_name}_window_col_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                material=material_green_name,
                group=3,
                condim=4,
                density=window_density,
                friction=window_friction,
                solimp=window_solimp,
                solref=window_solref,
            ),
            parent_name=attach_body,
        )

    #################################################################
    # visual mesh ################################################
    #################################################################
    generator.add_body(
        mjcf_config=BodyConfig(name=window_name + "_b_body", pos=np.array([0.1, 0.013, 0.0])),
        parent_name=window_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowa_h_frame_visual_geom",
            mesh=windowa_h_frame_mesh_name,
            pos=np.array([0.0 + 0.028, 0.0, 0.0]),
            type="mesh",
            material=material_wood_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_b_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowa_h_glass_visual_geom",
            mesh=windowa_h_glass_mesh_name,
            pos=np.array([0.0 + 0.028, -0.005, 0.0]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_b_body",
    )

    additional_box_configs = [
        # <geom pos="0 0 -0.14" size="0.08 0.012 0.01" type="box" />
        {
            "suffix": 18,
            "pos": np.array([0.0 + 0.03, 0.0, -0.14 - 0.04]),
            "size": np.array([0.08 * 1.5, 0.012, 0.01 * 1.3]),
        },
        # <geom pos="0 0 0.14"  size="0.08 0.012 0.01" type="box" />
        {
            "suffix": 19,
            "pos": np.array([0.0 + 0.03, 0.0, 0.14 + 0.04]),
            "size": np.array([0.08 * 1.5, 0.012, 0.01 * 1.3]),
        },
        # <geom pos="0 -0.005 0" size="0.08 0.001 0.13" type="box" />
        {"suffix": 20, "pos": np.array([0.0 + 0.03, -0.005, 0.0]), "size": np.array([0.08 * 1.5, 0.001, 0.13 * 1.3])},
        # <geom pos="-0.0 -0.009 0" size="0.08 0.003 0.005" type="box" />
        {"suffix": 21, "pos": np.array([0.0 + 0.03, -0.009, 0.0]), "size": np.array([0.08 * 1.5, 0.003, 0.005 * 1.3])},
        # <geom pos="-0.09 0 0"   size="0.01 0.012 0.15" type="box" />
        {"suffix": 22, "pos": np.array([-0.09 - 0.0, 0.0, 0.0]), "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3])},
        # <geom pos="0.09 0 0"    size="0.01 0.012 0.15" type="box" />
        {"suffix": 23, "pos": np.array([0.09 + 0.045, 0.0, 0.0]), "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3])},
    ]
    for cfg in additional_box_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{window_name}_window_col_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                material=material_green_name,
                group=3,
                condim=4,
                density=window_density,
                friction=window_friction,
                solimp=window_solimp,
                solref=window_solref,
            ),
            parent_name=window_name + "_b_body",
        )

    generator.add_site(
        SiteConfig(
            name=window_handle_site_name,
            pos=np.array([-0.014, -0.06 - 0.05, 0.0]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.025, 0.025, 0.025]),
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    return {"window_handle": {"body_name": window_name + "_joint_attached_body"}}
