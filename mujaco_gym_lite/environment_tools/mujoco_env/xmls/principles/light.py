from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import LightConfig
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_light(
    generator: MJCFGenerator,
    light_name: Optional[str] = None,
    directional: bool = True,
    cast_shadow: bool = False,
    direction: npt.NDArray = np.array([0.0, 0.0, -1.0]),
    light_position: npt.NDArray = np.array([0.0, 0.0, 0.0]),
    ambient_color: npt.NDArray = np.array([0.0, 0.0, 0.0]),
    diffuse_color: npt.NDArray = np.array([0.7, 0.7, 0.7]),
    specular_color: npt.NDArray = np.array([0.3, 0.3, 0.3]),
    attach_body: str = "worldbody",
):
    mjcf_config = LightConfig(
        name=light_name,
        directional="true" if directional else "false",
        dir=direction,
        pos=light_position,
        ambient=ambient_color,
        diffuse=diffuse_color,
        specular=specular_color,
        castshadow="true" if cast_shadow else "false",
    )
    generator.add_light(mjcf_config, parent_name=attach_body)
