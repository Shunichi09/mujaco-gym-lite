import pathlib

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import TextureConfig
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_texture_skybox(
    generator: MJCFGenerator,
    texture_file_path: pathlib.Path,
    skybox_name: str,
):
    generator.add_texture(TextureConfig(name=skybox_name, file=texture_file_path, type="skybox"))
