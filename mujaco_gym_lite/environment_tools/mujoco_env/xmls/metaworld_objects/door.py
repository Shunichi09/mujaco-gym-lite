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


def add_sliding_door(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    door_name: str,
    door_position: npt.NDArray,
    door_rotation: npt.NDArray,
    door_body_name: str,
    door_handle_site_name: str,
    door_handle_name: str,
    door_joint_name: str,
    door_density: float = 500,
    door_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    door_solref: npt.NDArray = np.array([0.01, 1]),
    door_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    texture_name = door_name + "_metal_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = door_name + "_metal_material"
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

    texture_name = door_name + "_navy_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="2d", file=asset_dir_path / "textures" / "metaworld" / "navy_blue.png")
    )
    material_navy_name = door_name + "_navy_material"
    generator.add_material(MaterialConfig(name=material_navy_name, texture=texture_name))

    texture_name = door_name + "_dark_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="2d", file=asset_dir_path / "textures" / "metaworld" / "navy_blue.png")
    )
    material_dark_navy_name = door_name + "_dark_navy_material"
    generator.add_material(
        MaterialConfig(
            name=material_dark_navy_name,
            texture=texture_name,
            specular=0.0,
            shininess=0.0,
            reflectance=0.1,
            emission=0.0,
        )
    )

    material_green_name = door_name + "_green_texture"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.4, 0.65, 0.4, 1.0]),
            specular=0.3,
            shininess=0.3,
        )
    )

    material_light_blue_name = door_name + "_light_blue_material"
    generator.add_material(
        MaterialConfig(
            name=material_light_blue_name,
            rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    material_light_pink_name = door_name + "_light_pink_material"
    generator.add_material(
        MaterialConfig(
            name=material_light_pink_name,
            rgba=np.array([0.85, 0.45, 0.55, 1.0]),
            shininess=0.15,
            reflectance=0.05,
            specular=0.08,
        )
    )

    texture_name = door_name + "_wood_texture"
    generator.add_texture(
        TextureConfig(
            name=texture_name,
            type="2d",
            file=asset_dir_path / "textures" / "metaworld" / "wood3.png",
        )
    )
    material_wood_name = door_name + "_wood_material"
    generator.add_material(MaterialConfig(name=material_wood_name, texture=texture_name))

    texture_name = door_name + "_wood2_texture"
    generator.add_texture(
        TextureConfig(
            name=texture_name,
            type="2d",
            file=asset_dir_path / "textures" / "metaworld" / "wood1.png",
        )
    )
    material_glass_name = door_name + "_glass_material"
    generator.add_material(MaterialConfig(name=material_glass_name, texture=texture_name))

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"

    door_h_base_mesh_name = door_name + "_door_h_base_mesh"
    door_h_base_mesh_file_path = metaworld_asset_dir_path / "door" / "door_h_base.stl"
    generator.add_mesh(
        MeshConfig(name=door_h_base_mesh_name, file=door_h_base_mesh_file_path, scale=np.array([1.3, 2.5, 1.0]))
    )

    door_h_frame_mesh_name = door_name + "_door_h_frame_mesh"
    door_h_frame_mesh_file_path = metaworld_asset_dir_path / "door" / "door_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=door_h_frame_mesh_name, file=door_h_frame_mesh_file_path, scale=np.array([1.3, 5.0, 1.3]))
    )

    doora_h_frame_mesh_name = door_name + "_doora_h_frame_mesh"
    doora_h_frame_mesh_file_path = metaworld_asset_dir_path / "door" / "doora_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=doora_h_frame_mesh_name, file=doora_h_frame_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    doora_h_glass_mesh_name = door_name + "_doora_h_glass_mesh"
    doora_h_glass_mesh_file_path = metaworld_asset_dir_path / "door" / "doora_h_glass.stl"
    generator.add_mesh(
        MeshConfig(name=doora_h_glass_mesh_name, file=doora_h_glass_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    doorb_h_frame_mesh_name = door_name + "_doorb_h_frame_mesh"
    doorb_h_frame_mesh_file_path = metaworld_asset_dir_path / "door" / "doorb_h_frame.stl"
    generator.add_mesh(
        MeshConfig(name=doorb_h_frame_mesh_name, file=doorb_h_frame_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    doorb_h_glass_mesh_name = door_name + "_doorb_h_glass_mesh"
    doorb_h_glass_mesh_file_path = metaworld_asset_dir_path / "door" / "doorb_h_glass.stl"
    generator.add_mesh(
        MeshConfig(name=doorb_h_glass_mesh_name, file=doorb_h_glass_mesh_file_path, scale=np.array([1.3, 1.0, 1.3]))
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=door_body_name, pos=door_position, quat=door_rotation),
        parent_name="worldbody",
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=door_name + "_joint_attached_body", pos=np.array([0.0, 0.0, 0.0])),
        parent_name=door_body_name,
    )
    generator.add_joint(
        JointConfig(
            name=door_joint_name,
            pos=np.array([0.0, 0.0, 0.0]),
            axis=np.array([-1.0, 0.0, 0.0]),
            type="slide",
            limited="true",
            armature=0.001,
            range=np.array([0, 0.25]),
            damping=1.5,
        ),
        parent_name=door_name + "_joint_attached_body",
    )

    # visual mesh
    # DoorFrame ########################################
    generator.add_geom(
        GeomConfig(
            name=door_name + "_door_base_visual_geom",
            mesh=door_h_base_mesh_name,
            pos=np.array([0.0, 0.0 + 0.125, -0.192 - 0.05]),
            type="mesh",
            material=material_navy_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_frame_visual_geom",
            mesh=door_h_frame_mesh_name,
            pos=np.array([0.0, 0.0 + 0.125, 0.0]),
            type="mesh",
            material=material_navy_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_frame_back_visual_geom",
            pos=np.array([0.0, 0.125 + 0.125, 0.0]),
            type="box",
            material=material_dark_navy_name,
            size=np.array([0.265, 0.02, 0.195]),
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_frame_back_visual_geom_2",
            pos=np.array([0.0, 0.125 + 0.12, 0.0]),
            type="box",
            material=material_light_blue_name,
            size=np.array([0.265, 0.02, 0.195]),
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_body_name,
    )

    # collision mesh
    column_configs = [
        {
            "suffix": "col_left",
            "pos": np.array([-0.216 - 0.06, 0.0 + 0.125, 0.0]),
            "size": np.array([0.015 * 1.3, 0.03 * 5.0, 0.181 * 1.3]),
        },
        {
            "suffix": "horz_bottom",
            "pos": np.array([0.0, 0.0 + 0.125, -0.192 - 0.05]),
            "size": np.array([0.25 * 1.3, 0.06 * 2.5, 0.01 * 1.3]),
        },
        {
            "suffix": "col_right",
            "pos": np.array([0.216 + 0.06, 0.0 + 0.125, 0.0]),
            "size": np.array([0.015 * 1.3, 0.03 * 5.0, 0.181 * 1.3]),
        },
        {
            "suffix": "mid_lower",
            "pos": np.array([0.0, 0.0 + 0.125, -0.166 - 0.05]),
            "size": np.array([0.201 * 1.3, 0.03 * 5.0, 0.015 * 1.3]),
        },
        {
            "suffix": "mid_upper",
            "pos": np.array([0.0, 0.0 + 0.125, 0.166 + 0.05]),
            "size": np.array([0.201 * 1.3, 0.03 * 5.0, 0.015 * 1.3]),
        },
        {
            "suffix": "back",
            "pos": np.array([0.0, 0.125 + 0.125, 0.0]),
            "size": np.array([0.255, 0.03, 0.21]),
        },
    ]
    for cfg in column_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{door_name}_door_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                group=3,
                condim=4,
                density=door_density,
                friction=door_friction,
                solimp=door_solimp,
                solref=door_solref,
            ),
            parent_name=door_body_name,
        )

    # DoorFrame ########################################

    #################################################################
    # visual mesh ###################################################
    #################################################################
    attach_body = door_name + "_joint_attached_body"
    # handle visual mesh
    configs = [
        {
            "suffix": "cylinder_1",
            "type": "cylinder",
            "pos": np.array([-0.014 + 0.255, -0.028, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
            "material": material_metal_name,
        },
        {
            "suffix": "cylinder_2",
            "type": "cylinder",
            "pos": np.array([-0.014 + 0.255, -0.028, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
            "material": material_metal_name,
        },
        {
            "suffix": "capsule_1",
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.06 - 0.005, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
            "material": material_metal_name,
        },
        {
            "suffix": "capsule_2",
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.06 - 0.005, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
            "material": material_metal_name,
        },
        {
            "suffix": "capsule_3",
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.11, 0.0]),
            "euler": None,
            "size": np.array([0.008 + 0.005, 0.045 + 0.075]),
            "material": material_light_pink_name,
        },
    ]
    for cfg in configs:
        params = dict(
            name=f"{door_handle_name}_{cfg['suffix']}_geom",
            pos=cfg["pos"],
            size=cfg["size"],
            type=cfg["type"],
            material=cfg["material"],
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
            name=f"{door_name}_right_doorb_h_frame_visual_geom",
            mesh=doorb_h_frame_mesh_name,
            pos=np.array([-0.104 - 0.028 + 0.255, -0.022, 0.0]),
            type="mesh",
            material=material_wood_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_right_doorb_h_glass_visual_geom",
            mesh=doorb_h_glass_mesh_name,
            pos=np.array([-0.104 - 0.028 + 0.255, -0.018, 0.0]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_name + "_joint_attached_body",
    )

    # capsule collisions
    capsule_cylinder_configs = [
        {
            "suffix": 1,
            "type": "cylinder",
            "pos": np.array([-0.014 + 0.255, -0.028, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": 2,
            "type": "cylinder",
            "pos": np.array([-0.014 + 0.255, -0.028, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.012, 0.003]),
        },
        {
            "suffix": 3,
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.06 - 0.005, 0.045 + 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": 4,
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.06 - 0.005, -0.045 - 0.05]),
            "euler": np.array([1.57, 0.0, 0.0]),
            "size": np.array([0.008, 0.045]),
        },
        {
            "suffix": 5,
            "type": "capsule",
            "pos": np.array([-0.014 + 0.255, -0.11, 0.0]),
            "euler": None,
            "size": np.array([0.008 + 0.005, 0.045 + 0.075]),
        },
    ]
    for cfg in capsule_cylinder_configs:
        params = dict(
            name=f"{door_handle_name}_door_col_{cfg['suffix']}_geom",
            pos=cfg["pos"],
            size=cfg["size"],
            type=cfg["type"],
            material=material_green_name,
            group=3,
            condim=4,
            density=door_density,
            friction=door_friction,
            solimp=door_solimp,
            solref=door_solref,
        )
        if cfg["euler"] is not None:
            params["euler"] = cfg["euler"]

        generator.add_geom(
            GeomConfig(**params),
            parent_name=attach_body,
        )

    # Right door
    box_configs = [
        {
            "suffix": 12,
            "pos": np.array([-0.014 + 0.245, -0.013, 0.0]),
            "size": np.array([0.014 * 1.5, 0.012, 0.15 * 1.3]),
        },
        {
            "suffix": 13,
            "pos": np.array([-0.19 - 0.045 + 0.245, -0.013, 0.0]),
            "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3]),
        },
        {
            "suffix": 14,
            "pos": np.array([-0.104 - 0.03 + 0.245, -0.013, 0.14 + 0.04]),
            "size": np.array([0.076 * 1.5, 0.012, 0.01 * 1.3]),
        },
        {
            "suffix": 15,
            "pos": np.array([-0.104 - 0.03 + 0.245, -0.013, -0.14 - 0.04]),
            "size": np.array([0.076 * 1.5, 0.012, 0.01 * 1.3]),
        },
        {
            "suffix": 16,
            "pos": np.array([-0.104 - 0.03 + 0.245, -0.018, 0.0]),
            "size": np.array([0.076 * 1.5, 0.001, 0.13 * 1.3]),
        },
        {
            "suffix": 17,
            "pos": np.array([-0.104 - 0.03 + 0.245, -0.022, 0.0]),
            "size": np.array([0.076 * 1.5, 0.003, 0.005 * 1.3]),
        },
    ]

    for cfg in box_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{door_name}_door_col_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                material=material_green_name,
                group=3,
                condim=4,
                density=door_density,
                friction=door_friction,
                solimp=door_solimp,
                solref=door_solref,
            ),
            parent_name=attach_body,
        )

    #################################################################
    # visual mesh ################################################
    # left door
    #################################################################
    generator.add_body(
        mjcf_config=BodyConfig(name=door_name + "_b_body", pos=np.array([0.1, 0.013, 0.0])),
        parent_name=door_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_left_doora_h_frame_visual_geom",
            mesh=doora_h_frame_mesh_name,
            pos=np.array([0.0 + 0.028 - 0.255, 0.0, 0.0]),
            type="mesh",
            material=material_wood_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_name + "_b_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{door_name}_left_doora_h_glass_visual_geom",
            mesh=doora_h_glass_mesh_name,
            pos=np.array([0.0 + 0.028 - 0.255, -0.005, 0.0]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=door_name + "_b_body",
    )

    additional_box_configs = [
        # <geom pos="0 0 -0.14" size="0.08 0.012 0.01" type="box" />
        {
            "suffix": 18,
            "pos": np.array([0.0 + 0.03 - 0.255, 0.0, -0.14 - 0.04]),
            "size": np.array([0.08 * 1.5, 0.012, 0.01 * 1.3]),
        },
        # <geom pos="0 0 0.14"  size="0.08 0.012 0.01" type="box" />
        {
            "suffix": 19,
            "pos": np.array([0.0 + 0.03 - 0.255, 0.0, 0.14 + 0.04]),
            "size": np.array([0.08 * 1.5, 0.012, 0.01 * 1.3]),
        },
        # <geom pos="0 -0.005 0" size="0.08 0.001 0.13" type="box" />
        {
            "suffix": 20,
            "pos": np.array([0.0 + 0.03 - 0.255, -0.005, 0.0]),
            "size": np.array([0.08 * 1.5, 0.001, 0.13 * 1.3]),
        },
        # <geom pos="-0.0 -0.009 0" size="0.08 0.003 0.005" type="box" />
        {
            "suffix": 21,
            "pos": np.array([0.0 + 0.03 - 0.255, -0.009, 0.0]),
            "size": np.array([0.08 * 1.5, 0.003, 0.005 * 1.3]),
        },
        # <geom pos="-0.09 0 0"   size="0.01 0.012 0.15" type="box" />
        {
            "suffix": 22,
            "pos": np.array([-0.09 - 0.0 - 0.255, 0.0, 0.0]),
            "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3]),
        },
        # <geom pos="0.09 0 0"    size="0.01 0.012 0.15" type="box" />
        {
            "suffix": 23,
            "pos": np.array([0.09 + 0.045 - 0.255, 0.0, 0.0]),
            "size": np.array([0.01 * 1.5, 0.012, 0.15 * 1.3]),
        },
    ]
    for cfg in additional_box_configs:
        generator.add_geom(
            GeomConfig(
                name=f"{door_name}_door_col_{cfg['suffix']}_geom",
                pos=cfg["pos"],
                size=cfg["size"],
                type="box",
                material=material_green_name,
                group=3,
                condim=4,
                density=door_density,
                friction=door_friction,
                solimp=door_solimp,
                solref=door_solref,
            ),
            parent_name=door_name + "_b_body",
        )

    generator.add_site(
        SiteConfig(
            name=door_handle_site_name,
            pos=np.array([-0.014, -0.06 - 0.05, 0.0]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.05]),
        ),
        parent_name=door_name + "_joint_attached_body",
    )

    return {"door_handle": {"body_name": door_name + "_joint_attached_body"}}
