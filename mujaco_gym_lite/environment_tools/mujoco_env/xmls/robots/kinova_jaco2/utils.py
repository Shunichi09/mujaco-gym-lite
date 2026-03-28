from typing import List, Optional

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    BodyConfig,
    GeneralActuatorConfig,
    GeomConfig,
    InertialConfig,
    JointConfig,
)
from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_generator import MJCFGenerator


def add_jaco2_link(
    mjcf_generator: MJCFGenerator,
    link_name: str,
    body_name: str,
    joint_name: Optional[str],
    attach_body_name: str,
    body_position: npt.NDArray,
    body_rotation: npt.NDArray,
    joint_range: Optional[npt.NDArray],
    joint_damping: Optional[float],
    material_names: List[str],
    visual_mesh_names: List[str],
    collision_mesh_names: List[str],
    collision_conaffinity: int,
    collision_contype: int,
    inertia_mass: float,
    inertial_position: npt.NDArray,
    inertial_full: Optional[npt.NDArray],
    inertial_diag: Optional[npt.NDArray],
    joint_type: Optional[str] = "hinge",
    joint_axis: Optional[npt.NDArray] = np.array([0.0, 0.0, 1.0]),
    collision_friction: npt.NDArray = np.array([1.0, 0.005, 0.0001]),
    collision_solimp: npt.NDArray = np.array([0.95, 0.99, 0.001, 0.5, 2.0]),
    collision_solref: npt.NDArray = np.array([0.02, 1]),
    condim: int = 4,
    joint_stiffness: float = 0.0,
):
    mjcf_generator.add_body(
        BodyConfig(name=body_name, pos=body_position, quat=body_rotation),
        parent_name=attach_body_name,
    )

    # inertia
    mjcf_generator.add_inertial(
        InertialConfig(
            mass=inertia_mass,
            pos=inertial_position,
            quat=None,
            fullinertia=inertial_full,
            diaginertia=inertial_diag,
        ),
        parent_name=body_name,
    )

    if joint_name is not None:
        mjcf_generator.add_joint(
            JointConfig(
                name=joint_name,
                axis=joint_axis,
                type=joint_type,
                armature=0.1,
                range=joint_range,
                damping=joint_damping,
                limited="true",
                stiffness=joint_stiffness,
            ),
            parent_name=body_name,
        )
    # visual mesh
    for i, (material_name, visual_mesh_name) in enumerate(zip(material_names, visual_mesh_names)):
        mjcf_generator.add_geom(
            GeomConfig(
                name=link_name + f"_visual_mesh_{i}",
                type="mesh",
                mesh=visual_mesh_name,
                material=material_name,
                contype=0,
                conaffinity=0,
                group=2,
            ),
            parent_name=body_name,
        )
    # collision mesh
    for i, collision_mesh_name in enumerate(collision_mesh_names):
        mjcf_generator.add_geom(
            mjcf_config=GeomConfig(
                name=link_name + f"_collision_geom_{i}",
                type="mesh",
                mesh=collision_mesh_name,
                group=3,
                friction=collision_friction,
                density=1000,
                solimp=collision_solimp,
                solref=collision_solref,
                contype=collision_contype,
                conaffinity=collision_conaffinity,
                condim=condim,
            ),
            parent_name=body_name,
        )


def add_jaco2_actuator(
    mjcf_generator: MJCFGenerator,
    actuator_name: str,
    joint: str,
    gainrpm: float,
    biasprm: npt.NDArray,
    ctrlrange: npt.NDArray,
    forcerange: npt.NDArray,
):
    mjcf_generator.add_general_actuator(
        GeneralActuatorConfig(
            name=actuator_name,
            joint=joint,
            biastype="affine",
            dyntype="none",
            ctrlrange=ctrlrange,
            forcerange=forcerange,
            gainprm=gainrpm,
            biasprm=biasprm,
        )
    )
