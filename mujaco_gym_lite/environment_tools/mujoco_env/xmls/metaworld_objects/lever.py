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


def add_lever(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    lever_name: str,
    lever_position: npt.NDArray,
    lever_rotation: npt.NDArray,
    lever_body_name: str,
    lever_joint_name: str,
    lever_handle_site_name: str,
    lever_handle_geom_name: str,
    lever_density: float = 500,
    lever_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    lever_solref: npt.NDArray = np.array([0.01, 1]),
):
    metal1_texture_file_path = asset_dir_path / "textures" / "metaworld" / "metal1.png"
    metal1_texture_name = lever_name + "_metal1_texture"
    generator.add_texture(TextureConfig(name=metal1_texture_name, type="2d", file=metal1_texture_file_path))
    metal1_material_name = lever_name + "_metal1_material"
    generator.add_material(
        MaterialConfig(
            name=metal1_material_name,
            texture=metal1_texture_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0.9, 0.9, 0.9, 1.0]),
        )
    )
    metal2_material_name = lever_name + "_metal2_material"
    generator.add_material(
        MaterialConfig(
            name=metal2_material_name,
            texture=metal1_texture_name,
            shininess=0.5,
            specular=0.3,
            reflectance=0.2,
            rgba=np.array([0.35, 0.35, 0.35, 1.0]),
        )
    )

    material_blue_name = lever_name + "_black_material"
    generator.add_material(
        MaterialConfig(
            name=material_blue_name,
            rgba=np.array([0.0, 0.0, 0.5, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )
    material_yellow_name = lever_name + "_yellow_material"
    generator.add_material(
        MaterialConfig(
            name=material_yellow_name,
            rgba=np.array([0.7, 0.5, 0.0, 1.0]),
            specular=0.5,
            reflectance=0.7,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    lever_axis_mesh_name = lever_name + "_axis_mesh"
    lever_axis_mesh_file_path = metaworld_asset_dir_path / "lever" / "lever_axis.stl"
    generator.add_mesh(
        MeshConfig(
            name=lever_axis_mesh_name,
            file=lever_axis_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    lever_base_mesh_name = lever_name + "_base_mesh"
    lever_base_mesh_file_path = metaworld_asset_dir_path / "lever" / "lever_base.stl"
    generator.add_mesh(
        MeshConfig(
            name=lever_base_mesh_name,
            file=lever_base_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    lever_handle_mesh_name = lever_name + "_handle_mesh"
    lever_handle_mesh_file_path = metaworld_asset_dir_path / "lever" / "lever_handle.stl"
    generator.add_mesh(
        MeshConfig(
            name=lever_handle_mesh_name,
            file=lever_handle_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    lever_rod_mesh_name = lever_name + "_rod_mesh"
    lever_rod_mesh_file_path = metaworld_asset_dir_path / "lever" / "lever_rod.stl"
    generator.add_mesh(
        MeshConfig(
            name=lever_rod_mesh_name,
            file=lever_rod_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )
    lever_rodbase_mesh_name = lever_name + "_rodbase_mesh"
    lever_rodbase_mesh_file_path = metaworld_asset_dir_path / "lever" / "lever_rodbase.stl"
    generator.add_mesh(
        MeshConfig(
            name=lever_rodbase_mesh_name,
            file=lever_rodbase_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    )

    # body
    generator.add_body(
        BodyConfig(name=lever_body_name, pos=lever_position, quat=lever_rotation),
        parent_name="worldbody",
    )

    # visual mesh geoms
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_base_geom",
            mesh=lever_base_mesh_name,
            euler=np.array([0.0, 1.57, 0.0]),
            type="mesh",
            material=metal2_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=lever_body_name,
    )

    # collision geoms
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_base_collision_geom1",
            pos=np.array([0.0, 0.0, 0.125]),
            size=np.array([0.041, 0.083, 0.125]),
            type="box",
            material=metal2_material_name,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_base_collision_geom2",
            euler=np.array([0.0, 1.57, 0.0]),
            pos=np.array([0.0, 0.0, 0.25]),
            size=np.array([0.083, 0.041]),
            type="cylinder",
            material=metal2_material_name,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_base_collision_geom3",
            pos=np.array([0.0, 0.0, 0.013]),
            size=np.array([0.05, 0.092, 0.013]),
            type="box",
            material=metal2_material_name,
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_body_name,
    )

    # lever link body
    lever_link_body_name = lever_body_name + "_link1"
    generator.add_body(
        BodyConfig(
            name=lever_link_body_name,
            pos=np.array([0.12, 0.0, 0.25]),
            quat=euler_to_quat(np.array([-np.pi * 0.5, 0.0, 0.0])),
        ),
        parent_name=lever_body_name,
    )

    # joint
    generator.add_joint(
        JointConfig(
            name=lever_joint_name,
            type="hinge",
            axis=np.array([1.0, 0.0, 0.0]),
            range=np.array([0.0, np.deg2rad(75.0)]),
            armature=0.01,
            damping=1.0,
            stiffness=1.0,
        ),
        parent_name=lever_link_body_name,
    )

    # visual mesh geoms
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_axis_geom",
            mesh=lever_axis_mesh_name,
            type="mesh",
            material=metal1_material_name,
            rgba=np.array([0.05, 0.25, 1.0, 1.0]),
            euler=np.array([0.0, 1.57, 0.0]),
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=lever_link_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_rodbase_geom",
            mesh=lever_rodbase_mesh_name,
            euler=np.array([0.0, 1.57, 0.0]),
            type="mesh",
            material=metal1_material_name,
            rgba=np.array([0.05, 0.25, 1.0, 1.0]),
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=lever_link_body_name,
    )
    """
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_handle_geom",
            mesh=lever_handle_mesh_name,
            pos=np.array([0.0, -0.2, 0.0]),
            type="mesh",
            material=material_yellow_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=lever_link_body_name,
    )
    """
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_rod_geom",
            mesh=lever_rod_mesh_name,
            euler=np.array([1.57, 0.0, 0.0]),
            pos=np.array([0.0, -0.1, 0.0]),
            type="mesh",
            material=metal1_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=lever_link_body_name,
    )

    # collision geoms
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_link1_collision_geom1",
            euler=np.array([0.0, 1.57, 0.0]),
            size=np.array([0.038, 0.016]),
            type="cylinder",
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_link_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_link1_collision_geom2",
            euler=np.array([1.57, 0.0, 0.0]),
            pos=np.array([0.0, -0.091, 0.0]),
            size=np.array([0.012, 0.1]),
            type="cylinder",
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_link_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_handle_geom_name,
            pos=np.array([0.0, -0.2, 0.0]),
            euler=np.array([0.0, 1.57, 0.0]),
            size=np.array([0.0225, 0.05]),
            material=material_yellow_name,
            type="cylinder",
            contype=1,
            conaffinity=1,
        ),
        parent_name=lever_link_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=lever_name + "_link1_collision_geom4",
            euler=np.array([0.0, 1.57, 0.0]),
            pos=np.array([-0.025, 0.0, 0.0]),
            size=np.array([0.016, 0.046]),
            type="cylinder",
            contype=1,
            conaffinity=1,
            group=3,
        ),
        parent_name=lever_link_body_name,
    )

    # site
    generator.add_site(
        SiteConfig(
            name=lever_handle_site_name,
            pos=np.array([0.0, -0.2, 0.0]),
            size=np.array([0.025]),
            rgba=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        parent_name=lever_link_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=lever_name + "_guide_line_site",
            pos=np.array([0.1, -0.25, 0.0]),
            size=np.array([0.2, 0.02, 0.005]),
            rgba=np.array([0.8, 0.0, 0.0, 1.0]),
            type="box",
        ),
        parent_name=lever_body_name,
    )

    return {}
