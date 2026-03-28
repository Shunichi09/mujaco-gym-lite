import pathlib

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeomConfig,
    JointConfig,
    MaterialConfig,
    SiteConfig,
    TextureConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_drawer(
    generator: MJCFGenerator,
    asset_dir_path: pathlib.Path,
    drawer_name,
    drawer_position: npt.NDArray,
    drawer_rotation: npt.NDArray,
    drawer_body_name: str,
    drawer_case_name: str,
    drawer_handle_site_name: str,
    drawer_handle_name: str,
    drawer_joint_name: str,
    drawer_scale: npt.NDArray = np.array([1.0, 1.0, 1.0]),
    drawer_density: float = 500,
    drawer_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    drawer_solref: npt.NDArray = np.array([0.01, 1]),
    drawer_friction: npt.NDArray = np.array([1.0, 1.0, 1.0]),
):
    # NOTE: Only metaworld assets can be used.
    # add materials
    texture_name = drawer_name + "_metal_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="cube", file=asset_dir_path / "textures" / "metaworld" / "metal1.png")
    )
    material_metal_name = drawer_name + "_metal_material"
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

    material_light_pink_name = drawer_name + "_light_pink_material"
    generator.add_material(
        MaterialConfig(
            name=material_light_pink_name,
            rgba=np.array([0.85, 0.45, 0.55, 1.0]),
            shininess=0.15,
            reflectance=0.05,
            specular=0.08,
        )
    )

    texture_name = drawer_name + "_navy_texture"
    generator.add_texture(
        TextureConfig(name=texture_name, type="2d", file=asset_dir_path / "textures" / "metaworld" / "navy_blue.png")
    )
    material_navy_name = drawer_name + "_navy_material"
    generator.add_material(MaterialConfig(name=material_navy_name, texture=texture_name))

    texture_name = drawer_name + "_wood_texture"
    generator.add_texture(
        TextureConfig(
            name=texture_name,
            type="2d",
            file=asset_dir_path / "textures" / "metaworld" / "wood3.png",
        )
    )
    material_wood_name = drawer_name + "_wood_material"
    generator.add_material(MaterialConfig(name=material_wood_name, texture=texture_name))

    #  texture_file_path=self._config.asset_dir_path / "textures" / "metaworld" / "navy_blue.png",
    material_green_name = drawer_name + "_green_material"
    generator.add_material(
        MaterialConfig(
            name=material_green_name,
            rgba=np.array([0.4, 0.65, 0.4, 1.0]),
            specular=0.3,
            shininess=0.3,
        )
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=drawer_body_name, pos=drawer_position, quat=drawer_rotation),
        parent_name="worldbody",
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=drawer_case_name + "_body", pos=np.array([0.0, 0.0, 0.084])),
        parent_name=drawer_body_name,
    )
    generator.add_body(
        mjcf_config=BodyConfig(name=drawer_name + "_joint_attached_body", pos=np.array([0.0, -0.01, 0.006])),
        parent_name=drawer_case_name + "_body",
    )

    material_light_blue_name = drawer_name + "_light_blue_material"
    generator.add_material(
        MaterialConfig(
            name=material_light_blue_name,
            rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            shininess=1.0,
            reflectance=0.7,
            specular=0.5,
        )
    )

    # add collision
    # 取っ手を正面に見て、左側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_case_collision_0",
            type="box",
            pos=np.array([-0.175, 0.0, 0.0]),
            size=np.array([0.008, 0.1, 0.084]),
            group=2,
            condim=4,
            mass=0.05,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_navy_name,
        ),
        parent_name=drawer_case_name + "_body",
    )
    # 取っ手を正面に見て、右側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_case_collision_1",
            type="box",
            pos=np.array([0.175, 0.0, 0.0]),
            size=np.array([0.008, 0.1, 0.084]),
            group=2,
            condim=4,
            mass=0.05,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_navy_name,
        ),
        parent_name=drawer_case_name + "_body",
    )
    # 取っ手を正面に見て、奥側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_case_collision_2",
            type="box",
            pos=np.array([0.0, 0.092, -0.008]),
            size=np.array([0.1675, 0.008, 0.076]),
            group=2,
            condim=4,
            mass=0.05,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_navy_name,
        ),
        parent_name=drawer_case_name + "_body",
    )
    # 取っ手を正面に見て、下側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_case_collision_3",
            type="box",
            pos=np.array([0.0, -0.008, -0.07]),
            size=np.array([0.1675, 0.092, 0.014]),
            group=2,
            condim=4,
            mass=0.05,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_navy_name,
        ),
        parent_name=drawer_case_name + "_body",
    )
    # 取っ手を正面に見て、上側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_case_collision_4",
            type="box",
            pos=np.array([0.0, 0.0, 0.076]),
            size=np.array([0.1675, 0.1, 0.008]),
            group=2,
            condim=4,
            mass=0.05,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_navy_name,
        ),
        parent_name=drawer_case_name + "_body",
    )

    # add joints
    generator.add_joint(
        JointConfig(
            name=drawer_joint_name,
            pos=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, -1.0, 0.0]),
            type="slide",
            limited="true",
            armature=0.001,
            range=np.array([0, 0.16]),
            damping=5.0,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )

    # add case
    # ケースの表側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_front_collision_0",
            type="box",
            pos=np.array([0.0, -0.082, 0.008]),
            size=np.array([0.165, 0.008, 0.052]),
            group=2,
            condim=4,
            mass=0.04,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_wood_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    # ケースの奥側
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_collision_1",
            type="box",
            pos=np.array([0.0, 0.082, 0.008]),
            size=np.array([0.165, 0.008, 0.052]),
            group=2,
            condim=4,
            mass=0.04,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_wood_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    # ケースの側面（左側）
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_collision_2",
            type="box",
            pos=np.array([-0.092 - 0.065, 0, 0.008]),
            size=np.array([0.008, 0.074, 0.052]),
            group=2,
            condim=4,
            mass=0.04,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_wood_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    # ケースの側面（右側）
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_collision_3",
            type="box",
            pos=np.array([0.092 + 0.065, 0, 0.008]),
            size=np.array([0.008, 0.074, 0.052]),
            group=2,
            condim=4,
            mass=0.04,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_wood_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=drawer_name + "_collision_4",
            type="box",
            pos=np.array([0.0, 0.0, -0.052]),
            size=np.array([0.165, 0.09, 0.008]),
            group=2,
            condim=4,
            mass=0.04,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_wood_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )

    generator.add_site(
        SiteConfig(
            name=drawer_name + "_site_4",
            type="box",
            pos=np.array([0.0, 0.01, -0.05]),
            size=np.array([0.1625, 0.095, 0.008]),
            rgba=np.array([0.0, 0.5, 1.0, 1.0]),
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )

    # 以下が、handle
    generator.add_geom(
        GeomConfig(
            name=drawer_handle_name + "_collision_1",
            type="capsule",
            pos=np.array([-0.05 - 0.07, -0.135, 0.0]),
            euler=np.array([1.571, 0.0, 0.0]),
            size=np.array([0.009, 0.05]),
            group=2,
            condim=1,
            friction=np.array([0.0, 0.0, 0.0]),
            mass=0.06,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_metal_name,
            priority=1,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=drawer_handle_name + "_collision_2",
            type="capsule",
            pos=np.array([0.0, -0.15 - 0.035, 0.0]),
            euler=np.array([0.0, 1.57, 0.0]),
            size=np.array([0.009, 0.115]),
            group=2,
            condim=4,
            mass=0.06,
            friction=drawer_friction,
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_light_pink_name,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )
    generator.add_geom(
        GeomConfig(
            name=drawer_handle_name + "_collision_3",
            type="capsule",
            pos=np.array([0.05 + 0.07, -0.135, 0.0]),
            euler=np.array([1.57, 0.0, 0.0]),
            size=np.array([0.009, 0.05]),
            group=2,
            condim=1,
            mass=0.06,
            friction=np.array([0.0, 0.0, 0.0]),
            solimp=drawer_solimp,
            solref=drawer_solref,
            material=material_metal_name,
            priority=1,
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )

    generator.add_site(
        mjcf_config=SiteConfig(
            name=drawer_handle_site_name,
            pos=np.array([0.0, -0.185, 0.0]),
            rgba=np.array([0.0, 0.0, 0.0, 0.0]),
        ),
        parent_name=drawer_name + "_joint_attached_body",
    )

    return {"case": {"body_name": drawer_name + "_joint_attached_body"}}
