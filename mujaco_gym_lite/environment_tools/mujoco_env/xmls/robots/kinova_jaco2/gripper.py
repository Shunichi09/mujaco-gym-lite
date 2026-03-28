from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    EqualityJointConfig,
    FixedTendonConfig,
    FixedTendonJointConfig,
    GeneralActuatorConfig,
    GeomConfig,
    MaterialConfig,
    MeshConfig,
    SiteConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.robots.kinova_jaco2.utils import add_jaco2_link
from mujaco_gym_lite.utils.files import get_files


def add_jaco2_hand_3finger(
    mjcf_generator: MJCFGenerator,
    robot_name: str,
    robot_asset_dir: Path,
    attach_body: str,
    gripper_position: npt.NDArray = np.array([0.0, 0.0, 0.0]),
    gripper_rotation: npt.NDArray = np.array([1.0, 0.0, 0.0, 0.0]),
    add_end_effector_marker: bool = False,
    end_effector_marker_name: Optional[str] = None,
    end_effector_marker_type: str = "site",
) -> tuple[str, dict[str, str]]:
    """
    Notes:
        All parameters are following: https://github.com/deepmind/mujoco_menagerie/tree/main/franka_emika_panda
    """
    # assets
    # materials
    carbon_fiber_material_name = robot_name + "_carbon_fiber_material"
    if not mjcf_generator.has_material(carbon_fiber_material_name):
        mjcf_generator.add_material(
            MaterialConfig(
                name=carbon_fiber_material_name,
                specular=1,
                shininess=2,
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
                emission=1,
                rgba=np.array([0.12, 0.14, 0.14, 1.0]),
            )
        )

    jaco2_gripper_mesh_names = ["hand_3finger", "finger_proximal", "finger_distal"]
    # visual
    visual_mesh_names: dict[str, list[str]] = defaultdict(list)
    for name in jaco2_gripper_mesh_names:
        visual_mesh_file = robot_asset_dir / "visual" / str(name + ".STL")
        visual_mesh_name = robot_name + "_" + name + "_visual_mesh"
        mjcf_generator.add_mesh(MeshConfig(name=visual_mesh_name, file=visual_mesh_file))
        visual_mesh_names[name].append(visual_mesh_name)

    # collision
    collision_mesh_names: dict[str, list[str]] = defaultdict(list)
    for name in jaco2_gripper_mesh_names:
        collision_mesh_files = get_files(robot_asset_dir / "collision" / name, file_format=["*.obj"])
        for i, collision_mesh_file in enumerate(collision_mesh_files):
            collision_mesh_name = robot_name + "_" + name + f"_fingertip_collision_mesh_{i}"
            mjcf_generator.add_mesh(MeshConfig(name=collision_mesh_name, file=collision_mesh_file))
            collision_mesh_names[name].append(collision_mesh_name)

    # hand link
    hand_link_body_name = robot_name + "_hand_link_body"
    add_jaco2_link(
        mjcf_generator,
        link_name=robot_name + "_hand_link",
        body_name=hand_link_body_name,
        joint_name=None,
        attach_body_name=attach_body,
        body_position=gripper_position,
        body_rotation=gripper_rotation,
        joint_range=None,
        joint_damping=None,
        material_names=[carbon_fiber_material_name],
        visual_mesh_names=visual_mesh_names["hand_3finger"],
        collision_mesh_names=collision_mesh_names["hand_3finger"],
        collision_conaffinity=1,
        collision_contype=1,
        collision_friction=np.array([1.0, 0.005, 0.0001]),
        collision_solimp=np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
        collision_solref=np.array([0.02, 1.0]),
        inertia_mass=0.99,
        inertial_position=np.array([0.0, 0.0, -0.06]),
        inertial_full=np.array([0.0003453236187, 0.0003453236187, 0.0005816, 0.0, 0.0, 0.0]),
        inertial_diag=None,
    )

    def add_finger(finger_number, position, rotation):
        finger_proximal_link_name = robot_name + f"_finger{finger_number}_proximal_link"
        finger_proximal_link_body_name = robot_name + f"_finger{finger_number}_proximal_link_body"
        finger_proximal_link_joint_name = robot_name + f"_finger{finger_number}_proximal_link_joint"
        # proximal
        add_jaco2_link(
            mjcf_generator,
            link_name=finger_proximal_link_name,
            body_name=finger_proximal_link_body_name,
            joint_name=finger_proximal_link_joint_name,
            attach_body_name=hand_link_body_name,
            body_position=position,
            body_rotation=rotation,
            joint_range=np.array([0.0, 1.51]),  # TODO: check limit
            joint_damping=1,
            joint_type="hinge",
            joint_axis=np.array([0.0, 0.0, 1.0]),
            material_names=[grey_plastic_material_name],
            visual_mesh_names=visual_mesh_names["finger_proximal"],
            collision_mesh_names=collision_mesh_names["finger_proximal"],
            collision_conaffinity=1,
            collision_contype=1,
            collision_friction=np.array([1.0, 1.0, 1.0]),
            collision_solimp=np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
            collision_solref=np.array([0.001, 1.0]),
            inertia_mass=0.01,
            inertial_position=np.array([0.022, 0.0, 0.0]),
            inertial_full=np.array([7.8999684e-07, 7.8999684e-07, 8e-08, 0.0, 0.0, 0.0]),
            inertial_diag=None,
            condim=4,
        )
        finger_distal_link_name = robot_name + f"_finger{finger_number}_distal_link"
        finger_distal_link_body_name = robot_name + f"_finger{finger_number}_distal_link_body"
        finger_distal_link_joint_name = robot_name + f"_finger{finger_number}_distal_link_joint"
        # distal
        add_jaco2_link(
            mjcf_generator,
            link_name=finger_distal_link_name,
            body_name=finger_distal_link_body_name,
            joint_name=finger_distal_link_joint_name,
            attach_body_name=finger_proximal_link_body_name,
            body_position=np.array([0.044, -0.003, 0.0]),
            body_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_range=np.array([0.0, 1.0]),  # TODO: check limit
            joint_damping=1.0,
            joint_type="hinge",
            joint_axis=np.array([0.0, 0.0, 1.0]),
            material_names=[grey_plastic_material_name],
            visual_mesh_names=visual_mesh_names["finger_distal"],
            collision_mesh_names=collision_mesh_names["finger_distal"],
            collision_conaffinity=1,
            collision_contype=1,
            collision_friction=np.array([1.0, 1.0, 1.0]),
            collision_solimp=np.array([0.99, 0.95, 0.001, 0.5, 2.0]),
            collision_solref=np.array([0.001, 1]),
            inertia_mass=0.01,
            inertial_position=np.array([0.022, 0.0, 0.0]),
            inertial_full=np.array([7.8999684e-07, 7.8999684e-07, 8e-08, 0.0, 0.0, 0.0]),
            inertial_diag=None,
            joint_stiffness=0.0,
            condim=4,
        )
        return (
            finger_proximal_link_body_name,
            finger_proximal_link_joint_name,
            finger_distal_link_body_name,
            finger_distal_link_joint_name,
        )

    # thumb
    (
        finger1_proximal_link_body_name,
        finger1_proximal_link_joint_name,
        finger1_distal_link_body_name,
        finger1_distal_link_joint_name,
    ) = add_finger(
        1,
        position=np.array([0.00279, 0.03126, -0.11467]),
        rotation=np.array([0.24023797, -0.63366596, -0.38953603, 0.62371055]),
    )

    (
        finger2_proximal_link_body_name,
        finger2_proximal_link_joint_name,
        finger2_distal_link_body_name,
        finger2_distal_link_joint_name,
    ) = add_finger(
        2,
        position=np.array([0.02226, -0.02707, -0.11482]),
        rotation=np.array([0.65965333, -0.37145956, 0.60167915, -0.25467132]),
    )

    (
        finger3_proximal_link_body_name,
        finger3_proximal_link_joint_name,
        finger3_distal_link_body_name,
        finger3_distal_link_joint_name,
    ) = add_finger(
        3,
        position=np.array([-0.02226, -0.02707, -0.11482]),
        rotation=np.array([0.60167915, -0.25467132, 0.65965333, -0.37145956]),
    )

    # equality
    mjcf_generator.add_equality_joint(
        mjcf_config=EqualityJointConfig(
            name=None,
            joint1=finger1_proximal_link_joint_name,
            joint2=finger2_proximal_link_joint_name,
            solimp=np.array([0.95, 0.99, 0.001]),
            solref=np.array([0.005, 1]),
        )
    )
    mjcf_generator.add_equality_joint(
        mjcf_config=EqualityJointConfig(
            name=None,
            joint1=finger1_proximal_link_joint_name,
            joint2=finger3_proximal_link_joint_name,
            solimp=np.array([0.95, 0.99, 0.001]),
            solref=np.array([0.005, 1]),
        )
    )

    mjcf_generator.add_equality_joint(
        mjcf_config=EqualityJointConfig(
            name=None,
            joint1=finger1_distal_link_joint_name,
            joint2=finger2_distal_link_joint_name,
            solimp=np.array([0.95, 0.99, 0.001]),
            solref=np.array([0.005, 1]),
        )
    )
    mjcf_generator.add_equality_joint(
        mjcf_config=EqualityJointConfig(
            name=None,
            joint1=finger1_distal_link_joint_name,
            joint2=finger3_distal_link_joint_name,
            solimp=np.array([0.95, 0.99, 0.001]),
            solref=np.array([0.005, 1]),
        )
    )

    # tendon
    proximal_tendon_name = robot_name + "_proximal_tendon_split"
    mjcf_generator.add_fixed_tendon(mjcf_config=FixedTendonConfig(proximal_tendon_name))
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger1_proximal_link_joint_name, coef=0.333),
        parent_name=proximal_tendon_name,
    )
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger2_proximal_link_joint_name, coef=0.333),
        parent_name=proximal_tendon_name,
    )
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger3_proximal_link_joint_name, coef=0.333),
        parent_name=proximal_tendon_name,
    )

    distal_tendon_name = robot_name + "_distal_tendon_split"
    mjcf_generator.add_fixed_tendon(mjcf_config=FixedTendonConfig(distal_tendon_name))
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger1_distal_link_joint_name, coef=0.333),
        parent_name=distal_tendon_name,
    )
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger2_distal_link_joint_name, coef=0.333),
        parent_name=distal_tendon_name,
    )
    mjcf_generator.add_fixed_joint_tendon(
        mjcf_config=FixedTendonJointConfig(joint=finger3_distal_link_joint_name, coef=0.333),
        parent_name=distal_tendon_name,
    )

    # actuator
    gains = 0.4
    proximal_actuator_name = robot_name + "_gripper_proximal_actuator"
    mjcf_generator.add_general_actuator(
        mjcf_config=GeneralActuatorConfig(
            name=proximal_actuator_name,
            tendon=proximal_tendon_name,
            forcerange=np.array([-100.0, 100.0]),
            ctrlrange=np.array([0.0, 255.0]),
            gainprm=np.array([(1.51 * 100 / 255) * gains, 0, 0]),
            biasprm=np.array([0.0, -100.0 * gains, -10.0 * gains]),
            biastype="affine",
            dyntype="none",
        )
    )

    gains = 0.2
    distal_actuator_name = robot_name + "_gripper_distal_actuator"
    mjcf_generator.add_general_actuator(
        mjcf_config=GeneralActuatorConfig(
            name=distal_actuator_name,
            tendon=distal_tendon_name,
            forcerange=np.array([-100.0, 100.0]),
            ctrlrange=np.array([0.0, 255.0]),
            gainprm=np.array([(1.0 * 100 / 255) * gains, 0, 0]),
            biasprm=np.array([0.0, -100.0 * gains, -10.0 * gains]),
            biastype="affine",
            dyntype="none",
        )
    )

    end_effector_body_name = robot_name + "_end_effector_body"
    mjcf_generator.add_body(
        BodyConfig(
            end_effector_body_name,
            pos=np.array([0.0, 0.0, -0.16]),
            quat=np.array([0.0, 0.707106781, 0.707106781, 0.0]),
        ),
        parent_name=hand_link_body_name,
    )

    if add_end_effector_marker:
        assert end_effector_marker_name is not None
        if end_effector_marker_type == "site":
            mjcf_generator.add_site(
                SiteConfig(
                    name=end_effector_marker_name,
                    rgba=np.array([0.2, 0.8, 0.2, 0.5]),
                    type="sphere",
                    size=np.array([0.0125, 0.0125, 0.0125]),
                ),
                end_effector_body_name,
            )
        elif end_effector_marker_type == "geom":
            mjcf_generator.add_geom(
                GeomConfig(
                    name=end_effector_marker_name,
                    rgba=np.array([0.2, 0.8, 0.2, 0.5]),
                    type="sphere",
                    size=np.array([0.0125, 0.0125, 0.0125]),
                    contype=0,
                    conaffinity=0,
                ),
                end_effector_body_name,
            )
        else:
            raise ValueError

    info = {}
    return end_effector_body_name, info
