import pathlib

import numpy as np

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import LightConfig, TextureConfig
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.floor import add_floor
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.geom import add_mujoco_principal_geom
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.principles.settings import (
    setup_option_size_compiler,
    setup_visual,
)


def add_basic_scene_setting(
    generator: MJCFGenerator, asset_dir_path: pathlib.Path, add_field_box: bool = False
) -> None:
    add_floor(
        generator,
        asset_dir_path / "textures" / "poligon" / "ConcretePoured001_COL_2K_METALNESS.png",
        floor_name="floor_base",
        floor_position=np.array([0.0, 0.0, -1.0]),
        floor_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        floor_size=np.array([10.0, 10.0, 0.01]),
    )

    if add_field_box:
        add_mujoco_principal_geom(
            generator=generator,
            model_name="field_box_right",
            model_position=np.array([5.0, 0.0, 4.0]),
            model_rotation=np.array([0.70710678, 0.0, -0.70710678, 0.0]),
            model_size=np.array([5.0, 5.0, 0.01]),
            model_type_name="box",
            model_density=0,
            with_free_joint=False,
            contype=0,
            conaffinity=0,
            model_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

        add_mujoco_principal_geom(
            generator=generator,
            model_name="field_box_left",
            model_position=np.array([-5.0, 0.0, 4.0]),
            model_rotation=np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            model_size=np.array([5.0, 5.0, 0.01]),
            model_type_name="box",
            model_density=0,
            with_free_joint=False,
            contype=0,
            conaffinity=0,
            model_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

        add_mujoco_principal_geom(
            generator=generator,
            model_name="field_box_up",
            model_position=np.array([0.0, 5.0, 4.0]),
            model_rotation=np.array([0.70710678, -0.70710678, 0.0, 0.0]),
            model_size=np.array([5.0, 5.0, 0.01]),
            model_type_name="box",
            model_density=0,
            with_free_joint=False,
            contype=0,
            conaffinity=0,
            model_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

        add_mujoco_principal_geom(
            generator=generator,
            model_name="field_box_down",
            model_position=np.array([0.0, -5.0, 4.0]),
            model_rotation=np.array([0.70710678, 0.70710678, 0.0, 0.0]),
            model_size=np.array([5.0, 5.0, 0.01]),
            model_type_name="box",
            model_density=0,
            with_free_joint=False,
            contype=0,
            conaffinity=0,
            model_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

        add_mujoco_principal_geom(
            generator=generator,
            model_name="field_box_top",
            model_position=np.array([0.0, 0.0, 9.0]),
            model_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            model_size=np.array([5.0, 5.0, 0.01]),
            model_type_name="box",
            model_density=0,
            with_free_joint=False,
            contype=0,
            conaffinity=0,
            model_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

    setup_visual(generator)
    generator.add_texture(
        TextureConfig(
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=".50 .495 .48",
            rgb2=".50 .495 .48",
            width="32",
            height="32",
        )
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([-1.0, 1.0, -1.0]),
            dir=np.array([1.0, 1.0, -1.0]),
        ),
        "worldbody",
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([1.0, -1.0, -1.0]),
            dir=np.array([-1.0, 1.0, -1.0]),
        ),
        "worldbody",
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([0.0, 1.0, 1.0]),
            dir=np.array([0.0, -1.0, -1.0]),
        ),
        "worldbody",
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([-1.0, 0.0, -1.0]),
            dir=np.array([1.0, 0.0, 1.0]),
        ),
        "worldbody",
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([0.0, 0.0, -1.0]),
            dir=np.array([0.0, 0.0, 1.0]),
        ),
        "worldbody",
    )
    generator.add_light(
        LightConfig(
            castshadow="false",
            directional="true",
            diffuse=np.array([0.3, 0.3, 0.3]),
            specular=np.array([0.3, 0.3, 0.3]),
            pos=np.array([0.0, 0.0, 1.0]),
            dir=np.array([0.0, 0.0, -1.0]),
        ),
        "worldbody",
    )
    setup_option_size_compiler(generator, time_step=0.001, noslip_iterations=0, impratio=100)
