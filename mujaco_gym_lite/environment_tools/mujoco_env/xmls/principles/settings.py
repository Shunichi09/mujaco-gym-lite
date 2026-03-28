import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    CompilerConfig,
    HeadlightConfig,
    MapConfig,
    OptionConfig,
    OptionFlagConfig,
    QualityConfig,
    SizeConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def setup_visual(
    generator: MJCFGenerator,
    headlight_ambient_color: npt.NDArray = np.array([0.1, 0.1, 0.1]),
    headlight_diffuse_color: npt.NDArray = np.array([0.4, 0.4, 0.4]),
    headlight_specular_color: npt.NDArray = np.array([0.5, 0.5, 0.5]),
    headlight_active: bool = True,
    shadow_size: int = 8192,
    num_slices: int = 64,
    z_near: float = 0.0025,
    z_far: int = 15,
    offsamples: int = 4,
):
    headlight_mjcf_config = HeadlightConfig(
        ambient=headlight_ambient_color,
        diffuse=headlight_diffuse_color,
        specular=headlight_specular_color,
        active=1 if headlight_active else 0,
    )
    generator.add_headlight(headlight_mjcf_config)
    map_mjcf_config = MapConfig(znear=z_near, zfar=z_far)
    generator.add_map(map_mjcf_config)
    quality_mjcf_config = QualityConfig(shadowsize=shadow_size, numslices=num_slices, offsamples=offsamples)
    generator.add_quality(quality_mjcf_config)


def setup_option_size_compiler(
    generator: MJCFGenerator,
    time_step: float = 0.001,
    noslip_iterations: int = 0,
    jacobian: str = "dense",
    cone: str = "elliptic",
    impratio: int = 1,
    gravity: npt.NDArray = np.array([0, 0, -9.81]),
    compiler_angle: str = "radian",
    memory_size: str = "-1",
    multiccd: bool = True,
    nativeccd: bool = True,
    autolimits: bool = True,
):
    option_mjcf_config = OptionConfig(
        timestep=time_step,
        noslip_iterations=noslip_iterations,
        jacobian=jacobian,
        cone=cone,
        impratio=impratio,
        gravity=gravity,
    )
    generator.add_option(option_mjcf_config)
    option_flag_mjcf_config = OptionFlagConfig(
        multiccd="enable" if multiccd else "disable", nativeccd="enable" if nativeccd else "disable"
    )
    generator.add_option_flag(option_flag_mjcf_config)
    compiler_mjcf_config = CompilerConfig(angle=compiler_angle, autolimits="true" if autolimits else "false")
    generator.add_compiler(compiler_mjcf_config)
    size_mjcf_config = SizeConfig(memory=memory_size)
    generator.add_size(size_mjcf_config)
