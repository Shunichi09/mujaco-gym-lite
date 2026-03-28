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
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_soccer_ball(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    ball_name: str,
    ball_position: npt.NDArray,
    ball_body_name: str,
    ball_joint_name: str,
    ball_density: float = 500,
    ball_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    ball_solref: npt.NDArray = np.array([0.01, 1]),
):
    material_black_name = ball_name + "_black_material"
    generator.add_material(
        MaterialConfig(
            name=material_black_name,
            rgba=np.array([0.15, 0.15, 0.15, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )
    material_white_name = ball_name + "_white_material"
    generator.add_material(
        MaterialConfig(
            name=material_white_name,
            rgba=np.array([0.85, 0.85, 0.85, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )

    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld"

    # visual mesh
    black_ball_mesh_name = ball_name + "_ball_black"
    black_ball_mesh_file_path = metaworld_asset_dir_path / "soccer" / "soccer_black.stl"
    generator.add_mesh(
        MeshConfig(
            name=black_ball_mesh_name,
            file=black_ball_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]) * 2,
        )
    )

    white_ball_mesh_name = ball_name + "_ball_white"
    white_ball_mesh_file_path = metaworld_asset_dir_path / "soccer" / "soccer_white.stl"
    generator.add_mesh(
        MeshConfig(
            name=white_ball_mesh_name,
            file=white_ball_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]) * 2,
        )
    )

    generator.add_body(
        mjcf_config=BodyConfig(name=ball_body_name, pos=ball_position),
        parent_name="worldbody",
    )
    generator.add_joint(
        JointConfig(type="free", name=ball_joint_name, armature=0.0001, damping=0.0), parent_name=ball_body_name
    )

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=ball_name + "_ball_black_geom",
            mesh=black_ball_mesh_name,
            type="mesh",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=ball_body_name,
    )

    generator.add_geom(
        GeomConfig(
            name=ball_name + "_ball_white_geom",
            mesh=white_ball_mesh_name,
            type="mesh",
            material=material_white_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=ball_body_name,
    )

    # collsions
    generator.add_geom(
        GeomConfig(
            name=ball_name + "_collision_geom",
            type="sphere",
            size=np.array([0.026 * 2, 0.0, 0.0]),
            group=3,
            condim=4,
            density=ball_density,
            solimp=ball_solimp,
            solref=ball_solref,
            friction=np.array([0.1, 0.005, 0.005]),
        ),
        parent_name=ball_body_name,
    )
    return {"ball_body": ball_body_name}


def add_soccer_goal(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    goal_name: str,
    goal_position: npt.NDArray,
    goal_rotation: npt.NDArray,
    goal_body_name: str,
    goal_left_front_post_site_name: str,
    goal_right_front_post_site_name: str,
    goal_left_back_post_site_name: str,
    goal_right_back_post_site_name: str,
    goal_center_top_site_name: str,
    goal_center_bottom_back_site_name: str,
    goal_density: float = 500,
    goal_solimp: npt.NDArray = np.array([0.99, 0.99, 0.001, 0.5, 2.0]),
    goal_solref: npt.NDArray = np.array([0.01, 1]),
    goal_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    material_white_name = goal_name + "_white_material"
    generator.add_material(
        MaterialConfig(
            name=material_white_name,
            rgba=np.array([0.75, 0.75, 0.75, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )

    material_black_name = goal_name + "_black_material"
    generator.add_material(
        MaterialConfig(
            name=material_black_name,
            rgba=np.array([0.35, 0.35, 0.35, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )

    material_blue_name = goal_name + "_blue_material"
    generator.add_material(
        MaterialConfig(
            name=material_blue_name,
            rgba=np.array([0.0, 0.4, 0.6, 1.0]),
            shininess=1.0,
            specular=0.5,
        )
    )

    # visual mesh
    metaworld_asset_dir_path = asset_dir_path / "objects" / "metaworld" / "soccer"
    goal_net_mesh_name = goal_name + "_net_mesh"
    goal_net_mesh_file_path = metaworld_asset_dir_path / "soccer_net.stl"
    generator.add_mesh(
        MeshConfig(
            name=goal_net_mesh_name,
            file=goal_net_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]) * 1.5,
        )
    )

    goal_frame_mesh_name = goal_name + "_frame_mesh"
    goal_frame_mesh_file_path = metaworld_asset_dir_path / "soccer_frame.stl"
    generator.add_mesh(
        MeshConfig(
            name=goal_frame_mesh_name,
            file=goal_frame_mesh_file_path,
            scale=np.array([1.0, 1.0, 1.0]) * 1.5,
        )
    )

    # collision mesh
    collision_meshes = []
    for i in range(1, 5):
        collision_mesh_name = goal_name + f"_collision_mesh_{i}"
        collision_mesh_file_path = metaworld_asset_dir_path / f"goal_col{i}.stl"
        generator.add_mesh(
            MeshConfig(
                name=collision_mesh_name,
                file=collision_mesh_file_path,
                scale=np.array([1.0, 1.0, 1.0]) * 1.5,
            )
        )
        collision_meshes.append(collision_mesh_name)

    # body
    generator.add_body(
        mjcf_config=BodyConfig(
            name=goal_body_name,
            pos=goal_position,
            quat=goal_rotation,
        ),
        parent_name="worldbody",
    )

    # visual mesh
    generator.add_geom(
        GeomConfig(
            name=goal_name + "_net_geom",
            mesh=goal_net_mesh_name,
            type="mesh",
            material=material_white_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=goal_body_name,
    )
    generator.add_geom(
        GeomConfig(
            name=goal_name + "_frame_geom",
            mesh=goal_frame_mesh_name,
            type="mesh",
            material=material_black_name,
            contype=0,
            conaffinity=0,
            density=0,
            group=2,
        ),
        parent_name=goal_body_name,
    )

    for collision_mesh_name in collision_meshes:
        generator.add_geom(
            GeomConfig(
                name=collision_mesh_name + "_geom",
                mesh=collision_mesh_name,
                type="mesh",
                contype=1,
                conaffinity=1,
                density=goal_density,
                friction=goal_friction,
                solimp=goal_solimp,
                solref=goal_solref,
                pos=np.array((0.0, 0.0, 0.08)) * 1.5,
                group=3,
            ),
            parent_name=goal_body_name,
        )

    # add sites
    generator.add_site(
        SiteConfig(
            name=goal_left_front_post_site_name,
            pos=np.array([-0.1 * 1.5, 0.05 * 1.5, 0.0]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        parent_name=goal_body_name,
    )
    generator.add_site(
        SiteConfig(
            name=goal_right_front_post_site_name,
            pos=np.array([-0.1 * 1.5, -0.05 * 1.5, 0.0]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([0.0, 1.0, 0.0, 0.0]),
        ),
        parent_name=goal_body_name,
    )
    generator.add_site(
        SiteConfig(
            name=goal_left_back_post_site_name,
            pos=np.array([0.1 * 1.5, 0.05 * 1.5, 0.0]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([0.0, 0.0, 1.0, 0.0]),
        ),
        parent_name=goal_body_name,
    )
    generator.add_site(
        SiteConfig(
            name=goal_right_back_post_site_name,
            pos=np.array([0.1 * 1.5, -0.05 * 1.5, 0.0]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([0.0, 0.5, 0.5, 0.0]),
        ),
        parent_name=goal_body_name,
    )
    generator.add_site(
        SiteConfig(
            name=goal_center_top_site_name,
            pos=np.array([0.0, -0.05 * 1.5, 0.16 * 1.5]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([0.5, 0.0, 0.5, 0.0]),
        ),
        parent_name=goal_body_name,
    )
    generator.add_site(
        SiteConfig(
            name=goal_center_bottom_back_site_name,
            pos=np.array([0.0, 0.05 * 1.5, 0.0]),
            size=np.array([0.01, 0.01, 0.01]),
            rgba=np.array([0.5, 0.8, 0.5, 0.0]),
        ),
        parent_name=goal_body_name,
    )

    generator.add_site(
        SiteConfig(
            name=goal_name + "_guide_line_site",
            pos=np.array([0.0, -0.09, 0.0]),
            size=np.array([0.135, 0.0075, 0.001]),
            rgba=np.array([0.75, 0.0, 0.0, 1.0]),
            type="box",
        ),
        parent_name=goal_body_name,
    )

    return {}
