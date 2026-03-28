import pathlib

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    JointConfig,
    MaterialConfig,
    MeshConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_updown_window(
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
    material_col_name = window_name + "_col_material"
    generator.add_material(
        MaterialConfig(
            name=material_col_name,
            rgba=np.array([0.3, 0.3, 1.0, 0.5]),
            shininess=0.0,
            specular=0.0,
        )
    )
    material_white_name = window_name + "_white_material"
    generator.add_material(
        MaterialConfig(
            name=material_white_name,
            rgba=np.array([0.65, 0.65, 0.65, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )
    material_red_name = window_name + "_red_material"
    generator.add_material(
        MaterialConfig(
            name=material_red_name,
            rgba=np.array([0.36, 0.26, 0.27, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )
    material_green_name = window_name + "_green_material"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.51, 0.58, 0.55, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )
    material_black_name = window_name + "_black_material"
    generator.add_material(
        MaterialConfig(
            name=material_black_name,
            rgba=np.array([0.3, 0.3, 0.3, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )
    material_glass_name = window_name + "_glass_material"
    generator.add_material(
        MaterialConfig(
            name=material_glass_name,
            rgba=np.array([0.0, 0.3, 0.4, 0.1]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"

    window_h_base_mesh_name = window_name + "_window_h_base_mesh"
    window_h_base_mesh_file_path = metaworld_asset_dir_path / "window" / "window_h_base.stl"
    generator.add_mesh(MeshConfig(name=window_h_base_mesh_name, file=window_h_base_mesh_file_path))

    window_h_frame_mesh_name = window_name + "_window_h_frame_mesh"
    window_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "window_h_frame.stl"
    generator.add_mesh(MeshConfig(name=window_h_frame_mesh_name, file=window_h_frame_mesh_file_path))

    windowa_h_frame_mesh_name = window_name + "_windowa_h_frame_mesh"
    windowa_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_h_frame.stl"
    generator.add_mesh(MeshConfig(name=windowa_h_frame_mesh_name, file=windowa_h_frame_mesh_file_path))

    windowa_h_glass_mesh_name = window_name + "_windowa_h_glass_mesh"
    windowa_h_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_h_glass.stl"
    generator.add_mesh(MeshConfig(name=windowa_h_glass_mesh_name, file=windowa_h_glass_mesh_file_path))

    windowb_h_frame_mesh_name = window_name + "_windowb_h_frame_mesh"
    windowb_h_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_h_frame.stl"
    generator.add_mesh(MeshConfig(name=windowb_h_frame_mesh_name, file=windowb_h_frame_mesh_file_path))

    windowb_h_glass_mesh_name = window_name + "_windowb_h_glass_mesh"
    windowb_h_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_h_glass.stl"
    generator.add_mesh(MeshConfig(name=windowb_h_glass_mesh_name, file=windowb_h_glass_mesh_file_path))

    window_base_mesh_name = window_name + "_window_base_mesh"
    window_base_mesh_file_path = metaworld_asset_dir_path / "window" / "window_base.stl"
    generator.add_mesh(MeshConfig(name=window_base_mesh_name, file=window_base_mesh_file_path))

    window_frame_mesh_name = window_name + "_window_frame_mesh"
    window_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "window_frame.stl"
    generator.add_mesh(MeshConfig(name=window_frame_mesh_name, file=window_frame_mesh_file_path))

    windowa_frame_mesh_name = window_name + "_windowa_frame_mesh"
    windowa_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_frame.stl"
    generator.add_mesh(MeshConfig(name=windowa_frame_mesh_name, file=windowa_frame_mesh_file_path))

    windowa_glass_mesh_name = window_name + "_windowa_glass_mesh"
    windowa_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowa_glass.stl"
    generator.add_mesh(MeshConfig(name=windowa_glass_mesh_name, file=windowa_glass_mesh_file_path))

    windowb_frame_mesh_name = window_name + "_windowb_frame_mesh"
    windowb_frame_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_frame.stl"
    generator.add_mesh(MeshConfig(name=windowb_frame_mesh_name, file=windowb_frame_mesh_file_path))

    windowb_glass_mesh_name = window_name + "_windowb_glass_mesh"
    windowb_glass_mesh_file_path = metaworld_asset_dir_path / "window" / "windowb_glass.stl"
    generator.add_mesh(MeshConfig(name=windowb_glass_mesh_name, file=windowb_glass_mesh_file_path))

    generator.add_body(
        mjcf_config=BodyConfig(name=window_body_name, pos=window_position, quat=window_rotation),
        parent_name="worldbody",
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=window_name + "_joint_attached_body", pos=np.array([0.0, 0.0, 0.051])),
        parent_name=window_body_name,
    )
    generator.add_joint(
        JointConfig(
            name=window_joint_name,
            pos=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            type="slide",
            limited="true",
            armature=0.001,
            range=np.array([0, 0.2]),
            damping=10.0,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=window_name + "_window_base_visual_geom",
            mesh=window_base_mesh_name,
            pos=np.array([0.0, 0.0, 0.01]),
            type="mesh",
            material=material_green_name,
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
            mesh=window_frame_mesh_name,
            pos=np.array([0.0, 0.0, 0.467]),
            type="mesh",
            material=material_green_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_a_frame_visual_geom",
            mesh=windowa_frame_mesh_name,
            pos=np.array([0.0, 0.012, 0.261]),
            type="mesh",
            material=material_green_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_a_glass_visual_geom",
            mesh=windowa_glass_mesh_name,
            pos=np.array([0.0, 0.007, 0.351]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_body_name,
    )

    # collision mesh
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_1_geom",
            pos=np.array([0.0, 0.0, 0.035]),
            size=np.array([0.181, 0.03, 0.015]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_2_geom",
            pos=np.array([0.0, 0.0, 0.01]),
            size=np.array([0.195, 0.06, 0.01]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_3_geom",
            pos=np.array([0.0, 0.0, 0.467]),
            size=np.array([0.181, 0.03, 0.015]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_4_geom",
            pos=np.array([0.166, 0.0, 0.251]),
            size=np.array([0.015, 0.03, 0.201]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_5_geom",
            pos=np.array([-0.166, 0.0, 0.251]),
            size=np.array([0.015, 0.03, 0.201]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_6_geom",
            pos=np.array([0.14, 0.012, 0.351]),
            size=np.array([0.01, 0.012, 0.08]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_7_geom",
            pos=np.array([-0.14, 0.012, 0.351]),
            size=np.array([0.01, 0.012, 0.08]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_8_geom",
            pos=np.array([0.0, 0.007, 0.351]),
            size=np.array([0.13, 0.001, 0.08]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_9_geom",
            pos=np.array([0.0, 0.003, 0.351]),
            size=np.array([0.005, 0.003, 0.08]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_10_geom",
            pos=np.array([0.0, 0.012, 0.261]),
            size=np.array([0.15, 0.012, 0.01]),
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

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_11_geom",
            pos=np.array([0.0, 0.012, 0.441]),
            size=np.array([0.15, 0.012, 0.01]),
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
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_black_cylinder_1_geom",
            pos=np.array([0.045, -0.028, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.012, 0.003]),
            type="cylinder",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_black_capsule_1_geom",
            pos=np.array([0.045, -0.06, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.008, 0.035]),
            type="capsule",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_black_capsule_2_geom",
            pos=np.array([-0.045, -0.06, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.008, 0.035]),
            type="capsule",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_black_cylinder_2_geom",
            pos=np.array([-0.045, -0.028, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.012, 0.003]),
            type="cylinder",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_black_capsule_3_geom",
            pos=np.array([0.0, -0.095, 0.014]),
            euler=np.array([0.0, 1.571, 0.0]),
            size=np.array([0.008, 0.045]),
            type="capsule",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )
    # Slide body visual mesh geoms
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowb_frame_visual_geom",
            mesh=windowb_frame_mesh_name,
            pos=np.array([0.0, -0.022, 0.104]),
            type="mesh",
            material=material_green_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_windowb_glass_visual_geom",
            mesh=windowb_glass_mesh_name,
            pos=np.array([0.0, -0.018, 0.104]),
            type="mesh",
            material=material_glass_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    #################################################################
    # collision mesh ################################################
    #################################################################
    # Slide body box geoms (with mass)
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_12_geom",
            pos=np.array([0.0, -0.013, 0.014]),
            size=np.array([0.15, 0.012, 0.014]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_13_geom",
            pos=np.array([0.0, -0.013, 0.19]),
            size=np.array([0.15, 0.012, 0.01]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_14_geom",
            pos=np.array([0.14, -0.013, 0.104]),
            size=np.array([0.01, 0.012, 0.076]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_15_geom",
            pos=np.array([-0.14, -0.013, 0.104]),
            size=np.array([0.01, 0.012, 0.076]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_16_geom",
            pos=np.array([0.0, -0.018, 0.104]),
            size=np.array([0.13, 0.001, 0.076]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_17_geom",
            pos=np.array([0.0, -0.022, 0.104]),
            size=np.array([0.005, 0.003, 0.076]),
            type="box",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    # Slide body cylinder/capsule geoms (with mass)
    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_cyl_mass_1_geom",
            pos=np.array([0.045, -0.028, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.012, 0.003]),
            type="cylinder",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_caps_mass_1_geom",
            pos=np.array([0.045, -0.06, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.008, 0.035]),
            type="capsule",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_caps_mass_2_geom",
            pos=np.array([-0.045, -0.06, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.008, 0.035]),
            type="capsule",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_cyl_mass_2_geom",
            pos=np.array([-0.045, -0.028, 0.014]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.012, 0.003]),
            type="cylinder",
            material=material_green_name,
            mass=0.001,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    generator.add_geom(
        GeomConfig(
            name=f"{window_name}_window_col_caps_mass_3_geom",
            pos=np.array([0.0, -0.095, 0.014]),
            euler=np.array([0.0, 1.57, 0.0]),
            size=np.array([0.008, 0.045]),
            type="capsule",
            material=material_green_name,
            group=3,
            condim=4,
            density=window_density,
            friction=window_friction,
            solimp=window_solimp,
            solref=window_solref,
        ),
        parent_name=window_name + "_joint_attached_body",
    )

    return {"case": {"body_name": window_name + "_joint_attached_body"}}
