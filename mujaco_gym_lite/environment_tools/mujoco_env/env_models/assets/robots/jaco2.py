import os
import pathlib
from typing import Union

import numpy as np
import numpy.typing as npt
import open3d as o3d

from mujaco_gym_lite.utils.solids import create_coordinate_mesh, load_mesh
from mujaco_gym_lite.utils.transforms import create_transformation_matrix, euler_to_quat


def load_j2n6s300_mesh(mesh_type: str = "box", load_finger: bool = False) -> list[o3d.geometry.TriangleMesh]:
    if mesh_type == "box":
        assert not load_finger, "You don't have to load finger mesh when mesh type is box."

    mesh_file_path_list = [
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "base.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "shoulder.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "arm.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "forearm.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "wrist.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "wrist.stl",
        pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "hand_3finger.stl",
    ]
    if load_finger:
        # Add finger meshes.
        mesh_file_path_list.extend(
            [
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_proximal.stl",
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_distal.stl",
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_proximal.stl",
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_distal.stl",
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_proximal.stl",
                pathlib.Path(os.path.dirname(__file__)) / "jaco2_mesh" / mesh_type / "finger_distal.stl",
            ]
        )
    j2n6s300_meshes = []
    for mesh_file in mesh_file_path_list:
        j2n6s300_meshes.append(load_mesh(mesh_file))
    return j2n6s300_meshes


def _compute_link_matrix(pos, rot, joint_angle):
    base_T_child = create_transformation_matrix(pos, rot, rotation_type="quaternion")
    child_T_additional_angle = create_transformation_matrix(
        rotation=euler_to_quat((0.0, 0.0, joint_angle), "xyz"),
        rotation_type="quaternion",
    )
    return np.matmul(base_T_child, child_T_additional_angle)


def transform_j2n6s300_mesh(
    joint_mesh: list[o3d.geometry.TriangleMesh],
    joint_angle: Union[list[float], npt.NDArray],
    with_finger: bool = False,
    visualize: bool = True,
) -> tuple[list[o3d.geometry.TriangleMesh], npt.NDArray]:
    if with_finger:
        assert len(joint_mesh) == 13
        assert len(joint_angle) == 12
    else:
        assert len(joint_mesh) == 7
        assert len(joint_angle) == 6

    # copy all
    copied_joint_mesh = []
    for mesh in joint_mesh:
        copied_joint_mesh.append(o3d.geometry.TriangleMesh(mesh))

    # colors
    carbon = np.array([0.15, 0.15, 0.15])
    grey_plastic = np.array([0.5, 0.5, 0.5])

    # base
    pos = np.array([0.0, 0.0, 0.0])
    rot = np.array([1.0, 0.0, 0.0, 0.0])
    world_T_base = _compute_link_matrix(pos, rot, 0.0)
    copied_joint_mesh[0].transform(world_T_base)
    copied_joint_mesh[0].paint_uniform_color(carbon)
    # link1 (base.stl)
    pos = np.array([0.0, 0.0, 0.15675])
    rot = np.array([0.0, 0.0, 1.0, 0.0])
    base_T_link1 = _compute_link_matrix(pos, rot, joint_angle[0])
    copied_joint_mesh[1].transform(base_T_link1)
    copied_joint_mesh[1].paint_uniform_color(carbon)

    # link2 (arm.stl)
    pos = np.array([0.0, 0.0016, -0.11875])
    rot = np.array([0.0, 0.0, -0.707107, 0.707107])
    link1_T_link2 = _compute_link_matrix(pos, rot, joint_angle[1])
    world_T_link2 = np.matmul(base_T_link1, link1_T_link2)
    copied_joint_mesh[2].transform(world_T_link2)
    copied_joint_mesh[2].paint_uniform_color(carbon)

    # link3 (forearm.stl)
    pos = np.array([0.0, -0.410, 0.0])
    rot = np.array([0.0, 0.0, 1.0, 0.0])
    link2_T_link3 = _compute_link_matrix(pos, rot, joint_angle[2])
    world_T_link3 = np.matmul(world_T_link2, link2_T_link3)
    copied_joint_mesh[3].transform(world_T_link3)
    copied_joint_mesh[3].paint_uniform_color(carbon)

    # link4 (wrist.stl)
    pos = np.array([0.0, 0.2073, -0.0114])
    rot = np.array([0.0, 0.0, -0.707107, 0.707107])
    link3_T_link4 = _compute_link_matrix(pos, rot, joint_angle[3])
    world_T_link4 = np.matmul(world_T_link3, link3_T_link4)
    copied_joint_mesh[4].transform(world_T_link4)
    copied_joint_mesh[4].paint_uniform_color(carbon)

    # link5
    pos = np.array([0.0, -0.03703, -0.06414])
    rot = np.array([0.0, 0.0, 0.5, 0.8660254])
    link4_T_link5 = _compute_link_matrix(pos, rot, joint_angle[4])
    world_T_link5 = np.matmul(world_T_link4, link4_T_link5)
    copied_joint_mesh[5].transform(world_T_link5)
    copied_joint_mesh[5].paint_uniform_color(carbon)

    # link6
    pos = np.array([0.0, -0.03703, -0.06414])
    rot = np.array([0.0, 0.0, 0.5, 0.8660254])
    link5_T_link6 = _compute_link_matrix(pos, rot, joint_angle[5])
    world_T_link6 = np.matmul(world_T_link5, link5_T_link6)
    copied_joint_mesh[6].transform(world_T_link6)
    copied_joint_mesh[6].paint_uniform_color(carbon)

    # fingers
    if with_finger:

        def transform_finger(
            proximal_mesh,
            distal_mesh,
            proximal_joint_angle,
            distal_joint_angle,
            proximal_position,
            proximal_rotation,
        ):
            link6_T_proximal = _compute_link_matrix(proximal_position, proximal_rotation, proximal_joint_angle)
            world_T_proximal = np.matmul(world_T_link6, link6_T_proximal)
            proximal_mesh.transform(world_T_proximal)
            proximal_mesh.paint_uniform_color(grey_plastic)

            proximal_T_distal = _compute_link_matrix(
                np.array([0.044, -0.003, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                distal_joint_angle,
            )
            world_T_distal = np.matmul(world_T_proximal, proximal_T_distal)
            distal_mesh.transform(world_T_distal)
            distal_mesh.paint_uniform_color(grey_plastic)

        transform_finger(
            proximal_mesh=copied_joint_mesh[7],
            distal_mesh=copied_joint_mesh[8],
            proximal_joint_angle=joint_angle[6],
            distal_joint_angle=joint_angle[7],
            proximal_position=np.array([0.00279, 0.03126, -0.11467]),
            proximal_rotation=np.array([0.24023797, -0.63366596, -0.38953603, 0.62371055]),
        )
        transform_finger(
            proximal_mesh=copied_joint_mesh[9],
            distal_mesh=copied_joint_mesh[10],
            proximal_joint_angle=joint_angle[8],
            distal_joint_angle=joint_angle[9],
            proximal_position=np.array([0.02226, -0.02707, -0.11482]),
            proximal_rotation=np.array([0.65965333, -0.37145956, 0.60167915, -0.25467132]),
        )
        transform_finger(
            proximal_mesh=copied_joint_mesh[11],
            distal_mesh=copied_joint_mesh[12],
            proximal_joint_angle=joint_angle[10],
            distal_joint_angle=joint_angle[11],
            proximal_position=np.array([-0.02226, -0.02707, -0.11482]),
            proximal_rotation=np.array([0.60167915, -0.25467132, 0.65965333, -0.37145956]),
        )

    # end effector
    pos = np.array([0.0, 0.0, -0.16])
    rot = np.array([0.0, 0.707107, 0.707107, 0.0])
    link6_T_ee = _compute_link_matrix(pos, rot, 0.0)
    world_T_ee = np.matmul(world_T_link6, link6_T_ee)

    if visualize:
        o3d.visualization.draw_geometries(
            [*copied_joint_mesh, create_coordinate_mesh(np.eye(4), size=0.1)],
            mesh_show_wireframe=True,
        )

    return copied_joint_mesh, world_T_ee
