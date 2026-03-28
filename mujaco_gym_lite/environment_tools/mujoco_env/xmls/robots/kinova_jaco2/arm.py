from collections import defaultdict
from pathlib import Path

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
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.robots.kinova_jaco2.utils import (
    add_jaco2_actuator,
    add_jaco2_link,
)
from mujaco_gym_lite.utils.files import get_files


def add_j2n6s300(
    mjcf_generator: MJCFGenerator,
    robot_name: str,
    robot_asset_dir: Path,
    robot_position: npt.NDArray,
    robot_rotation: npt.NDArray,
    attach_body: str = "worldbody",
    with_end_effector: bool = False,
):
    # assets
    # materials
    carbon_fiber_material_name = robot_name + "_carbon_fiber_material"
    if not mjcf_generator.has_material(carbon_fiber_material_name):
        mjcf_generator.add_material(
            MaterialConfig(
                name=carbon_fiber_material_name,
                specular=1.0,  # 1, NOTE: dont know why but set 1 does not work
                shininess=1.0,  # 2, NOTE: dont know why but set 1 does not work
                reflectance=1,
                rgba=np.array([0.05, 0.05, 0.05, 1.0]),
            )
        )

    grey_plastic_material_name = robot_name + "_grey_plastic_material"
    if not mjcf_generator.has_material(grey_plastic_material_name):
        mjcf_generator.add_material(
            MaterialConfig(
                name=grey_plastic_material_name,
                specular=0.5,
                shininess=0,
                reflectance=0.0,
                emission=1.0,  # 1, NOTE: dont know why but set 1 does not work
                rgba=np.array([0.12, 0.14, 0.14, 1.0]),
            )
        )

    # meshes
    jaco2_arm_mesh_names = [
        "base",
        "shoulder",
        "arm",
        "forearm",
        "wrist",
        "ring_big",
        "ring_small",
    ]
    # visual
    visual_mesh_names: dict[str, str] = {}
    for name in jaco2_arm_mesh_names:
        visual_mesh_file = robot_asset_dir / "visual" / str(name + ".STL")
        visual_mesh_name = robot_name + "_" + name + "_visual_mesh"
        mjcf_generator.add_mesh(MeshConfig(name=visual_mesh_name, file=visual_mesh_file))
        visual_mesh_names[name] = visual_mesh_name

    # collision
    collision_mesh_names = defaultdict(list)
    for name in jaco2_arm_mesh_names[:-2]:  # without ring
        collision_mesh_files = get_files(robot_asset_dir / "collision" / name, file_format=["*.obj"])
        for i, collision_mesh_file in enumerate(collision_mesh_files):
            collision_mesh_name = robot_name + "_" + name + f"_collision_mesh_{i}"
            mjcf_generator.add_mesh(MeshConfig(name=collision_mesh_name, file=collision_mesh_file))
            collision_mesh_names[name].append(collision_mesh_name)

    # kinematic tree
    link_base_body_name = robot_name + "_link0_body"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link0",
        body_name=link_base_body_name,
        joint_name=None,
        attach_body_name=attach_body,
        body_position=robot_position,
        body_rotation=robot_rotation,
        joint_range=None,
        joint_damping=None,
        material_names=[carbon_fiber_material_name],
        visual_mesh_names=[visual_mesh_names["base"]],
        collision_mesh_names=collision_mesh_names["base"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.46784,
        inertial_position=np.array([0.0, 0.0, 0.1255]),
        inertial_full=np.array([0.000951270861568, 0.000951270861568, 0.000374272, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )

    link1_body_name = robot_name + "_link1_body"
    link1_joint_name = robot_name + "_link1_joint"
    link1_actuator_name = robot_name + "_link1_actuator"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link1",
        body_name=link1_body_name,
        joint_name=link1_joint_name,
        attach_body_name=link_base_body_name,
        body_position=np.array([0.0, 0.0, 0.15675]),
        body_rotation=np.array([0.0, 0.0, 1.0, 0.0]),
        joint_range=np.array([-6.28319, 6.28319]),
        joint_damping=1,
        material_names=[carbon_fiber_material_name, grey_plastic_material_name],
        visual_mesh_names=[
            visual_mesh_names["shoulder"],
            visual_mesh_names["ring_big"],
        ],
        collision_mesh_names=collision_mesh_names["shoulder"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.7477,
        inertial_position=np.array([0.0, -0.002, -0.0605]),
        inertial_full=np.array([0.00152031725204, 0.00152031725204, 0.00059816, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )
    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link1_actuator_name,
        joint=link1_joint_name,
        gainrpm=4500.0,
        biasprm=np.array([0.0, -4500, -450]),
        ctrlrange=np.array([-6.28319, 6.28319]),
        forcerange=np.array([-100.0, 100.0]),
        # forcerange=np.array([-87.0, 87.0]),
    )

    link2_body_name = robot_name + "_link2_body"
    link2_joint_name = robot_name + "_link2_joint"
    link2_actuator_name = robot_name + "_link2_actuator"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link2",
        body_name=link2_body_name,
        joint_name=link2_joint_name,
        attach_body_name=link1_body_name,
        body_position=np.array([0.0, 0.0016, -0.11875]),
        body_rotation=np.array([0.0, 0.0, -0.707107, 0.707107]),
        joint_range=np.array([0.820305, 5.46288]),
        joint_damping=1,
        material_names=[carbon_fiber_material_name, grey_plastic_material_name],
        visual_mesh_names=[
            visual_mesh_names["arm"],
            visual_mesh_names["ring_big"],
        ],
        collision_mesh_names=collision_mesh_names["arm"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.99,
        inertial_position=np.array([0.0, -0.2065, -0.01]),
        inertial_full=np.array(
            [
                0.010502207990999999,
                0.0007920000000000001,
                0.010502207990999999,
                0.0,
                0.0,
                0.0,
            ]
        ),
        inertial_diag=None,
    )
    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link2_actuator_name,
        joint=link2_joint_name,
        gainrpm=4500.0,
        biasprm=np.array([0.0, -4500, -450]),
        ctrlrange=np.array([0.820305, 5.46288]),
        forcerange=np.array([-500.0, 500.0]),
        # forcerange=np.array([-87.0, 87.0]),
    )

    link3_body_name = robot_name + "_link3_body"
    link3_joint_name = robot_name + "_link3_joint"
    link3_actuator_name = robot_name + "_link3_actuator"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link3",
        body_name=link3_body_name,
        joint_name=link3_joint_name,
        attach_body_name=link2_body_name,
        body_position=np.array([0.0, -0.410, 0.0]),
        body_rotation=np.array([0.0, 0.0, 1.0, 0.0]),
        joint_range=np.array([0.33161255787892263, 5.951572749300664]),
        joint_damping=1,
        material_names=[carbon_fiber_material_name, grey_plastic_material_name],
        visual_mesh_names=[
            visual_mesh_names["forearm"],
            visual_mesh_names["ring_big"],
        ],
        collision_mesh_names=collision_mesh_names["forearm"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.6763,
        inertial_position=np.array([0.0, 0.081, -0.0086]),
        inertial_full=np.array([0.0014202243190800001, 0.000304335, 0.0014202243190800001, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )
    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link3_actuator_name,
        joint=link3_joint_name,
        gainrpm=3500.0,
        biasprm=np.array([0.0, -3500, -350]),
        ctrlrange=np.array([0.33161255787892263, 5.951572749300664]),
        forcerange=np.array([-87.0, 87.0]),
    )

    link4_body_name = robot_name + "_link4_body"
    link4_joint_name = robot_name + "_link4_joint"
    link4_actuator_name = robot_name + "_link4_actuator"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link4",
        body_name=link4_body_name,
        joint_name=link4_joint_name,
        attach_body_name=link3_body_name,
        body_position=np.array([0.0, 0.2073, -0.0114]),
        body_rotation=np.array([0.0, 0.0, -0.707107, 0.707107]),
        joint_range=np.array([-6.28319, 6.28319]),
        joint_damping=1,
        material_names=[carbon_fiber_material_name, grey_plastic_material_name],
        visual_mesh_names=[visual_mesh_names["wrist"], visual_mesh_names["ring_small"]],
        collision_mesh_names=collision_mesh_names["wrist"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.426367,
        inertial_position=np.array([0.0, -0.037, -0.0642]),
        inertial_full=np.array([7.734969059999999e-05, 7.734969059999999e-05, 0.0001428, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )
    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link4_actuator_name,
        joint=link4_joint_name,
        gainrpm=3500.0,
        biasprm=np.array([0.0, -3500, -350]),
        ctrlrange=np.array([-6.28319, 6.28319]),  # TODO: Check limits
        forcerange=np.array([-87.0, 87.0]),
    )

    link5_body_name = robot_name + "_link5_body"
    link5_joint_name = robot_name + "_link5_joint"
    link5_actuator_name = robot_name + "_link5_actuator"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_link5",
        body_name=link5_body_name,
        joint_name=link5_joint_name,
        attach_body_name=link4_body_name,
        body_position=np.array([0.0, -0.03703, -0.06414]),
        body_rotation=np.array([0.0, 0.0, 0.5, 0.8660254]),
        joint_range=np.array([-6.28319, 6.28319]),
        joint_damping=1,
        material_names=[carbon_fiber_material_name, grey_plastic_material_name],
        visual_mesh_names=[
            visual_mesh_names["wrist"],
            visual_mesh_names["ring_small"],
        ],
        collision_mesh_names=collision_mesh_names["wrist"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1]),
        inertia_mass=0.426367,
        inertial_position=np.array([0.0, -0.037, -0.0642]),
        inertial_full=np.array([7.734969059999999e-05, 7.734969059999999e-05, 0.0001428, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )
    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link5_actuator_name,
        joint=link5_joint_name,
        gainrpm=3500.0,
        biasprm=np.array([0.0, -3500, -350]),
        ctrlrange=np.array([-6.28319, 6.28319]),  # TODO: Check limits
        forcerange=np.array([-87.0, 87.0]),
    )

    link6_body_name = robot_name + "_link6_body"
    link6_joint_name = robot_name + "_link6_joint"
    link6_actuator_name = robot_name + "_link6_actuator"
    # NOTE: Add body and joint only
    mjcf_generator.add_body(
        BodyConfig(
            name=link6_body_name,
            pos=np.array([0.0, -0.03703, -0.06414]),
            quat=np.array([0.0, 0.0, 0.5, 0.8660254]),
        ),
        parent_name=link5_body_name,
    )
    mjcf_generator.add_joint(
        JointConfig(
            name=link6_joint_name,
            axis=np.array([0.0, 0.0, 1.0]),
            type="hinge",
            armature=0.1,
            range=np.array([-6.28319, 6.28319]),
            damping=1,
            limited="true",
        ),
        parent_name=link6_body_name,
    )
    mjcf_generator.add_geom(
        GeomConfig(
            name=robot_name + "_link6" + "_visual_mesh_0",
            type="mesh",
            mesh=visual_mesh_names["ring_small"],
            material=grey_plastic_material_name,
            contype=0,
            conaffinity=0,
            group=2,
        ),
        parent_name=link6_body_name,
    )

    add_jaco2_actuator(
        mjcf_generator,
        actuator_name=link6_actuator_name,
        joint=link6_joint_name,
        gainrpm=2000.0,
        biasprm=np.array([0.0, -2000, -200]),
        ctrlrange=np.array([-6.28319, 6.28319]),
        forcerange=np.array([-12.0, 12.0]),
    )

    if with_end_effector:
        raise NotImplementedError

    return link6_body_name
