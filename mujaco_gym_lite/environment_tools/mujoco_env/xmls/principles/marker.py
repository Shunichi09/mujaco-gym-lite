from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import BodyConfig, GeomConfig, SiteConfig
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.utils.transforms import quat_to_matrix


def add_workspace_marker(
    generator: MJCFGenerator,
    workspace_name: str,
    marker_positions: list[npt.NDArray],
    rgbas: list[npt.NDArray],
    shape_type: str = "box",
    size: npt.NDArray = np.array([0.005, 0.005, 0.005]),
    attach_body: str = "worldbody",
    mujoco_type: str = "geom",
):
    for i, (marker_position, rgba) in enumerate(zip(marker_positions, rgbas)):
        mjcf_config: Union[GeomConfig, SiteConfig]
        if mujoco_type == "geom":
            mjcf_config = GeomConfig(
                name=f"{workspace_name}_{i}",
                pos=marker_position,
                rgba=rgba,
                type=shape_type,
                size=size,
                contype=0,
                conaffinity=0,
            )
            generator.add_geom(mjcf_config, parent_name=attach_body)
        elif mujoco_type == "site":
            mjcf_config = SiteConfig(
                name=f"{workspace_name}_{i}",
                pos=marker_position,
                rgba=rgba,
                type=shape_type,
                size=size,
            )
            generator.add_site(mjcf_config, parent_name=attach_body)
        else:
            raise ValueError


def add_mocap_coordinate_marker(
    generator: MJCFGenerator,
    coordinate_name: str,
    coordinate_position: npt.NDArray,
    coordinate_rotation: npt.NDArray,
    rgba: npt.NDArray = np.array([0.2, 0.2, 0.2, 0.5]),
    marker_size: npt.NDArray = np.array([0.02, 0.02, 0.02]),
    coordinate_size: npt.NDArray = np.array([0.1, 0.1, 0.1]),
    attach_body: str = "worldbody",
    mujoco_type: str = "site",
):
    coordinate_body_name = coordinate_name + "_body"
    generator.add_body(
        mjcf_config=BodyConfig(
            name=coordinate_body_name,
            pos=coordinate_position,
            quat=coordinate_rotation,
            mocap="true",
        ),
        parent_name=attach_body,
    )
    mjcf_config: Union[GeomConfig, SiteConfig]
    assert len(coordinate_rotation) == 4, "Only quaternion supported"
    marker_rotation_matrix = quat_to_matrix(coordinate_rotation)
    # add center marker
    x_marker_position = marker_rotation_matrix[:, 0] * coordinate_size[0]
    y_marker_position = marker_rotation_matrix[:, 1] * coordinate_size[1]
    z_marker_position = marker_rotation_matrix[:, 2] * coordinate_size[2]

    if mujoco_type == "geom":
        # center
        mjcf_config = GeomConfig(
            name=f"{coordinate_name}_center",
            rgba=rgba,
            type="sphere",
            size=marker_size,
            contype=0,
            conaffinity=0,
        )
        generator.add_geom(mjcf_config, parent_name=coordinate_body_name)
        # x axis
        mjcf_config = GeomConfig(
            name=f"{coordinate_name}_x_axis",
            pos=x_marker_position,
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
            type="box",
            size=marker_size,
            contype=0,
            conaffinity=0,
        )
        generator.add_geom(mjcf_config, parent_name=coordinate_body_name)
        # y axis
        mjcf_config = GeomConfig(
            name=f"{coordinate_name}_y_axis",
            pos=y_marker_position,
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
            type="box",
            size=marker_size,
            contype=0,
            conaffinity=0,
        )
        generator.add_geom(mjcf_config, parent_name=coordinate_body_name)
        # z axis
        mjcf_config = GeomConfig(
            name=f"{coordinate_name}_z_axis",
            pos=z_marker_position,
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
            type="box",
            size=marker_size,
            contype=0,
            conaffinity=0,
        )
        generator.add_geom(mjcf_config, parent_name=coordinate_body_name)
    elif mujoco_type == "site":
        # center
        mjcf_config = SiteConfig(
            name=f"{coordinate_name}_center",
            rgba=rgba,
            type="sphere",
            size=marker_size,
        )
        generator.add_site(mjcf_config, parent_name=coordinate_body_name)
        # x axis
        mjcf_config = SiteConfig(
            name=f"{coordinate_name}_x_axis",
            pos=x_marker_position,
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
            type="box",
            size=marker_size,
        )
        generator.add_site(mjcf_config, parent_name=coordinate_body_name)
        # y axis
        mjcf_config = SiteConfig(
            name=f"{coordinate_name}_y_axis",
            pos=y_marker_position,
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
            type="box",
            size=marker_size,
        )
        generator.add_site(mjcf_config, parent_name=coordinate_body_name)
        # z axis
        mjcf_config = SiteConfig(
            name=f"{coordinate_name}_z_axis",
            pos=z_marker_position,
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
            type="box",
            size=marker_size,
        )
        generator.add_site(mjcf_config, parent_name=coordinate_body_name)
    else:
        raise ValueError
