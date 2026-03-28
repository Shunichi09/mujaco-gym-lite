import pathlib

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    InertialConfig,
    JointConfig,
    MaterialConfig,
    MeshConfig,
    SiteConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_button(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    button_name: str,
    button_surface_name: str,
    button_rod_name: str,
    button_position: npt.NDArray,
    button_rotation: npt.NDArray,
    button_site_name: str,
    button_scale: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    button_density: float = 500,
    button_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    button_solref: npt.NDArray = np.array([0.01, 1]),
    button_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    # NOTE: Only metaworld assets can be used.
    # add materials
    texture_file_path = asset_dir_path / "textures" / "metaworld" / "metal1.png"
    texture_name = button_name + "_texture"
    generator.add_texture(TextureConfig(name=texture_name, type="cube", file=texture_file_path))
    material_metal_name = button_name + "_metal_material"
    generator.add_material(
        MaterialConfig(
            name=material_metal_name,
            texture=texture_name,
            reflectance=1,
            shininess=1,
            specular=1,
        )
    )
    material_col_name = button_name + "_col_material"
    generator.add_material(
        MaterialConfig(
            name=material_col_name,
            rgba=np.array([0.3, 0.3, 1.0, 0.5]),
            shininess=0.0,
            specular=0.0,
        )
    )
    material_red_name = button_name + "_red_material"
    generator.add_material(
        MaterialConfig(
            name=material_red_name,
            rgba=np.array([0.6, 0.0, 0.0, 1.0]),
            specular=0.5,
            reflectance=0.7,
        )
    )
    material_yellow_name = button_name + "_yellow_material"
    generator.add_material(
        MaterialConfig(
            name=material_yellow_name,
            rgba=np.array([0.7, 0.5, 0.0, 1.0]),
            specular=0.5,
            reflectance=0.7,
        )
    )
    material_black_name = button_name + "_black_material"
    generator.add_material(
        MaterialConfig(
            name=material_black_name,
            rgba=np.array([0.15, 0.15, 0.15, 1.0]),
            specular=0.5,
            reflectance=0.7,
        )
    )
    material_blue_name = button_name + "_blue_material"
    generator.add_material(
        MaterialConfig(
            name=material_blue_name,
            rgba=np.array([0.0, 0.0, 0.5, 1.0]),
            specular=0.5,
            reflectance=0.7,
        )
    )
    # add mesh
    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"
    stop_bot_mesh_name = button_name + "_stopbot_mesh"
    stop_bot_mesh_file_path = metaworld_asset_dir_path / "buttonbox" / "stopbot.stl"
    generator.add_mesh(MeshConfig(name=stop_bot_mesh_name, file=stop_bot_mesh_file_path))

    stop_button_mesh_name = button_name + "_stop_button_mesh"
    stop_button_mesh_file_path = metaworld_asset_dir_path / "buttonbox" / "stopbutton.stl"
    generator.add_mesh(
        MeshConfig(
            name=stop_button_mesh_name,
            file=stop_button_mesh_file_path,
            scale=button_scale,
        )
    )

    stop_button_rim_mesh_name = button_name + "_stop_button_rim_mesh"
    stop_button_rim_mesh_file_path = metaworld_asset_dir_path / "buttonbox" / "stopbuttonrim.stl"
    generator.add_mesh(
        MeshConfig(
            name=stop_button_rim_mesh_name,
            file=stop_button_rim_mesh_file_path,
        )
    )

    stop_button_rod_mesh_name = button_name + "_stop_button_rod_mesh"
    stop_button_rod_file_path = metaworld_asset_dir_path / "buttonbox" / "stopbuttonrod.stl"
    generator.add_mesh(
        MeshConfig(name=stop_button_rod_mesh_name, file=stop_button_rod_file_path, scale=np.array([1.0, 1.0, 1.5]))
    )

    stop_top_mesh_name = button_name + "_stop_top_mesh"
    stop_top_mesh_file_path = metaworld_asset_dir_path / "buttonbox" / "stoptop.stl"
    generator.add_mesh(
        MeshConfig(
            name=stop_top_mesh_name,
            file=stop_top_mesh_file_path,
        )
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=button_name + "_box_body", pos=button_position, quat=button_rotation),
        parent_name="worldbody",
    )

    generator.add_geom(
        GeomConfig(
            name=button_name + "_stopbot_visual_geom",
            mesh=stop_bot_mesh_name,
            type="mesh",
            pos=np.array([0.0, -0.06 + 0.03, 0.0]),
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=button_name + "_box_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_name + "_stop_button_rim_visual_geom",
            mesh=stop_button_rim_mesh_name,
            pos=np.array([0.0, -0.089 + 0.03, 0.0]),
            euler=np.array([-1.57, 0.0, 0.0]),
            type="mesh",
            material=material_metal_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=button_name + "_box_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_name + "_stop_top_visual_geom",
            type="mesh",
            mesh=stop_top_mesh_name,
            pos=np.array([0.0, -0.06 + 0.03, 0.0]),
            material=material_yellow_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=button_name + "_box_body",
    )

    # add collisions
    generator.add_geom(
        GeomConfig(
            name=button_name + "_box_collision_0",
            type="box",
            pos=np.array([0.0, 0.012 + 0.03, 0.072]),
            size=np.array([0.12, 0.102, 0.048]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        parent_name=button_name + "_box_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_name + "_box_collision_1",
            type="box",
            pos=np.array([0.0, 0.012 + 0.03, -0.072]),
            size=np.array([0.12, 0.102, 0.048]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        parent_name=button_name + "_box_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_name + "_box_collision_2",
            type="box",
            pos=np.array([-0.073, 0.012 + 0.03, 0.0]),
            size=np.array([0.047, 0.102, 0.024]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        parent_name=button_name + "_box_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_name + "_box_collision_3",
            type="box",
            pos=np.array([0.073, 0.012 + 0.03, 0.0]),
            size=np.array([0.047, 0.102, 0.024]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        parent_name=button_name + "_box_body",
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=button_name + "_body"),
        parent_name=button_name + "_box_body",
    )

    generator.add_inertial(
        InertialConfig(
            pos=np.array([0, -0.1935, 0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            mass=0.01,
            diaginertia=np.array([0.001, 0.001, 0.001]),
        ),
        parent_name=button_name + "_body",
    )
    generator.add_joint(
        JointConfig(
            name=button_name + "_joint",
            pos=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, -1.0, 0.0]),
            type="slide",
            springref=0.5,
            limited="true",
            armature=0.001,
            stiffness=5.0,
            range=np.array([-0.09, 0]),
            damping=2.5,
        ),
        parent_name=button_name + "_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_surface_name + "_stop_button_visual_geom",
            mesh=stop_button_mesh_name,
            type="mesh",
            pos=np.array([0.0, -0.158, 0.0]),
            euler=np.array([1.57, 0.0, 0.0]),
            material=material_red_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        button_name + "_body",
    )

    # boxの棒の部分
    generator.add_geom(
        GeomConfig(
            name=button_rod_name + "_stop_button_rod_visual_geom",
            mesh=stop_button_rod_mesh_name,
            type="mesh",
            pos=np.array([0.0, -0.126 + 0.025, 0.0]),
            euler=np.array([1.57, 0.0, 0.0]),
            material=material_blue_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        button_name + "_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_surface_name + "_collision_0",
            type="cylinder",
            pos=np.array([0.0, -0.128, 0.0]),
            size=np.array([0.021, 0.039 + 0.025]),
            euler=np.array([1.57, 0.0, 0.0]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        button_name + "_body",
    )
    generator.add_geom(
        GeomConfig(
            name=button_surface_name + "_collision_1",
            type="cylinder",
            pos=np.array([0.0, -0.166, 0.0]),
            size=np.array([0.026, 0.008]),
            euler=np.array([1.57, 0.0, 0.0]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        button_name + "_body",
    )
    # button surface size
    generator.add_geom(
        GeomConfig(
            name=button_surface_name + "_collision_2",
            type="cylinder",
            pos=np.array([0.0, -0.1832, 0.0]),
            size=np.array([0.08 * (1.15 / 2.0), 0.0105]),
            euler=np.array([1.57, 0.0, 0.0]),
            group=3,
            condim=4,
            density=button_density,
            friction=button_friction,
            solimp=button_solimp,
            solref=button_solref,
        ),
        button_name + "_body",
    )

    generator.add_site(
        mjcf_config=SiteConfig(
            name=button_site_name,
            pos=np.array([0.0, -0.18, 0.0]),
            rgba=np.array([0.2, 0.5, 0.0, 0.0]),
            size=np.array([0.025, 0.025, 0.025]),
        ),
        parent_name=button_name + "_body",
    )
    return {}
