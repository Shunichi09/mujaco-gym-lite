import pathlib
from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt


def if_array_to_str(value: Union[npt.NDArray, float, str, int, list]) -> str:
    if isinstance(value, np.ndarray) or isinstance(value, list):
        array_list = [str(val) for val in value]
        return " ".join(array_list)
    else:
        return str(value)


def mjcf_config_to_xml(mjcf_config: "MJCFConfig") -> str:
    tag_dict = asdict(mjcf_config)
    xml_str = []
    for key, value in tag_dict.items():
        key = "class" if key == "cls" else key
        if value is not None:
            string = f'{key}="{if_array_to_str(value)}"'
            xml_str.append(string)
    return " ".join(xml_str)


@dataclass
class MJCFConfig:
    pass


@dataclass
class NameConfig(MJCFConfig):
    name: str


# Asset
@dataclass
class TextureConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
    """

    name: Optional[str] = None
    file: Optional[pathlib.Path] = None
    type: Optional[str] = None
    builtin: Optional[str] = None
    rgb1: Optional[str] = None
    rgb2: Optional[str] = None
    width: Optional[str] = None
    height: Optional[str] = None


@dataclass
class MaterialConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
    """

    name: str
    cls: Optional[str] = None
    texture: Optional[str] = None
    texrepeat: Optional[npt.NDArray] = None
    shininess: Optional[float] = None
    specular: Optional[float] = None
    reflectance: Optional[float] = None
    rgba: Optional[npt.NDArray] = None
    emission: Optional[float] = None
    roughness: Optional[float] = None


@dataclass
class MeshConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    file: Optional[pathlib.Path] = None
    scale: Optional[npt.NDArray] = None


# Body
@dataclass
class GeomConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    type: Optional[str] = None
    contype: Optional[int] = None
    conaffinity: Optional[int] = None
    condim: Optional[int] = None
    group: Optional[int] = None
    size: Optional[npt.NDArray] = None
    material: Optional[str] = None
    rgba: Optional[npt.NDArray] = None
    friction: Optional[npt.NDArray] = None
    mass: Optional[float] = None
    mesh: Optional[str] = None
    density: Optional[float] = None
    solref: Optional[npt.NDArray] = None
    solimp: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None
    quat: Optional[npt.NDArray] = None
    euler: Optional[npt.NDArray] = None
    gap: Optional[int] = None
    priority: Optional[int] = None


@dataclass
class BodyConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#world-body-r
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    pos: Optional[npt.NDArray] = None
    quat: Optional[npt.NDArray] = None
    mocap: Optional[str] = None  # false or true


@dataclass
class JointConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint
    """

    name: Optional[str] = None
    type: Optional[str] = None
    cls: Optional[str] = None
    group: Optional[int] = None
    axis: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None
    damping: Optional[float] = None
    armature: Optional[float] = None
    springref: Optional[float] = None
    ref: Optional[float] = None
    limited: Optional[str] = None
    margin: Optional[float] = None
    range: Optional[npt.NDArray] = None
    stiffness: Optional[float] = None
    solreflimit: Optional[npt.NDArray] = None
    solimplimit: Optional[npt.NDArray] = None


@dataclass
class LightConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-light
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    mode: Optional[str] = None
    target: Optional[str] = None
    directional: Optional[str] = None
    castshadow: Optional[str] = None
    pos: Optional[npt.NDArray] = None
    dir: Optional[npt.NDArray] = None
    cutoff: Optional[float] = None
    exponent: Optional[float] = None
    ambient: Optional[npt.NDArray] = None
    diffuse: Optional[npt.NDArray] = None
    specular: Optional[npt.NDArray] = None


@dataclass
class InertialConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-inertial
    """

    pos: Optional[npt.NDArray]
    quat: Optional[npt.NDArray]
    mass: float
    diaginertia: Optional[npt.NDArray] = None
    fullinertia: Optional[npt.NDArray] = None


@dataclass
class SiteConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-site
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    type: Optional[str] = None
    group: Optional[int] = None
    size: Optional[npt.NDArray] = None
    rgba: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None
    quat: Optional[npt.NDArray] = None
    euler: Optional[npt.NDArray] = None


@dataclass
class CameraConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-camera
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    mode: Optional[str] = None
    target: Optional[str] = None
    fovy: Optional[float] = None
    pos: Optional[npt.NDArray] = None
    quat: Optional[npt.NDArray] = None


# Visual
@dataclass
class QualityConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#visual-quality
    """

    shadowsize: Optional[int] = None
    offsamples: Optional[int] = None
    numslices: Optional[int] = None
    numstacks: Optional[int] = None
    numquads: Optional[int] = None


@dataclass
class HeadlightConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#visual-headlight
    """

    ambient: Optional[npt.NDArray] = None
    diffuse: Optional[npt.NDArray] = None
    specular: Optional[npt.NDArray] = None
    active: Optional[int] = None


@dataclass
class MapConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#visual-map
    """

    znear: Optional[float] = None
    zfar: Optional[float] = None


# Option
@dataclass
class OptionFlagConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-flag
    """

    multiccd: Optional[str] = None
    sensornoise: Optional[str] = None
    nativeccd: Optional[str] = None


@dataclass
class OptionConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#option
    """

    timestep: Optional[float] = None
    noslip_iterations: Optional[int] = None
    jacobian: Optional[str] = None
    cone: Optional[str] = None
    impratio: Optional[int] = None
    gravity: Optional[npt.NDArray] = None
    iterations: Optional[int] = None
    solver: Optional[str] = None
    apirate: Optional[int] = None
    integrator: Optional[str] = None


# Compiler
@dataclass
class CompilerConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#compiler
    """

    angle: Optional[str] = None
    autolimits: Optional[str] = None


@dataclass
class SizeConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#size
    """

    memory: Optional[str] = None


# Equality
@dataclass
class EqualityWeldConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld
    """

    body1: str
    cls: Optional[str] = None
    name: Optional[str] = None
    body2: Optional[str] = None
    relpose: Optional[npt.NDArray] = None
    anchor: Optional[npt.NDArray] = None
    torquescale: Optional[float] = None
    solref: Optional[npt.NDArray] = None
    solimp: Optional[npt.NDArray] = None


@dataclass
class EqualityJointConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality-joint
    """

    joint1: str
    joint2: str
    cls: Optional[str] = None
    name: Optional[str] = None
    solref: Optional[npt.NDArray] = None
    solimp: Optional[npt.NDArray] = None
    polycoef: Optional[npt.NDArray] = None


@dataclass
class EqualityConnectConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality-connect
    """

    anchor: npt.NDArray
    body1: str
    body2: str
    cls: Optional[str] = None
    name: Optional[str] = None
    solref: Optional[npt.NDArray] = None
    solimp: Optional[npt.NDArray] = None


# contact
@dataclass
class ContactExcludeConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#contact-exclude
    """

    body1: str
    body2: str
    name: Optional[str] = None


# actuators
@dataclass
class MotorActuatorConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-motor
    """

    name: Optional[str] = None
    cls: Optional[str] = None
    joint: Optional[str] = None
    jointinparent: Optional[str] = None
    site: Optional[str] = None
    refsite: Optional[str] = None
    tendon: Optional[str] = None
    body: Optional[str] = None
    gear: Optional[npt.NDArray] = None
    ctrllimited: Optional[str] = None
    forcelimited: Optional[str] = None
    ctrlrange: Optional[npt.NDArray] = None
    forcerange: Optional[npt.NDArray] = None


@dataclass
class GeneralActuatorConfig(MotorActuatorConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-general
    """

    dyntype: Optional[str] = None
    biastype: Optional[str] = None
    gainprm: Optional[Union[npt.NDArray, float]] = None
    biasprm: Optional[Union[npt.NDArray, float]] = None


@dataclass
class PositionActuatorConfig(MotorActuatorConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-position
    """

    kp: Optional[float] = None


@dataclass
class VelocityActuatorConfig(MotorActuatorConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-velocity
    """

    kv: Optional[float] = None


@dataclass
class AdhesionActuatorConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-adhesion
    """

    body: str
    cls: Optional[str] = None
    ctrlrange: Optional[npt.NDArray] = None
    name: Optional[str] = None
    gain: Optional[float] = None
    forcelimited: Optional[str] = None
    forcerange: Optional[npt.NDArray] = None


@dataclass
class FixedTendonConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#tendon
    """

    name: Optional[str] = None
    limited: Optional[str] = None
    range: Optional[npt.NDArray] = None
    frictionloss: Optional[float] = None
    damping: Optional[float] = None


@dataclass
class FixedTendonJointConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#fixed-joint
    """

    joint: str
    coef: float


@dataclass
class DefaultConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-r
    """

    cls: str


@dataclass
class DefaultMaterialConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-material-texuniform
    """

    cls: str
    texture: Optional[str] = None
    texrepeat: Optional[npt.NDArray] = None
    shininess: Optional[float] = None
    specular: Optional[float] = None
    reflectance: Optional[float] = None
    rgba: Optional[npt.NDArray] = None
    emission: Optional[int] = None


@dataclass
class DefaultMeshConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-mesh-scale
    """

    cls: str
    file: Optional[pathlib.Path] = None
    scale: Optional[npt.NDArray] = None


@dataclass
class DefaultGeomConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-geom-user
    """

    cls: str
    type: Optional[str] = None
    contype: Optional[int] = None
    conaffinity: Optional[int] = None
    condim: Optional[int] = None
    group: Optional[int] = None
    size: Optional[npt.NDArray] = None
    material: Optional[str] = None
    rgba: Optional[npt.NDArray] = None
    friction: Optional[npt.NDArray] = None
    mass: Optional[float] = None
    mesh: Optional[str] = None
    density: Optional[float] = None
    solref: Optional[npt.NDArray] = None
    solimp: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None
    quat: Optional[npt.NDArray] = None
    euler: Optional[npt.NDArray] = None
    gap: Optional[int] = None
    priority: Optional[int] = None


@dataclass
class DefaultJointConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-joint-user
    """

    type: Optional[str] = None
    cls: Optional[str] = None
    group: Optional[int] = None
    axis: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None
    damping: Optional[float] = None
    armature: Optional[float] = None
    springref: Optional[float] = None
    ref: Optional[float] = None
    limited: Optional[str] = None
    margin: Optional[float] = None
    range: Optional[npt.NDArray] = None
    stiffness: Optional[float] = None
    solreflimit: Optional[npt.NDArray] = None
    solimplimit: Optional[npt.NDArray] = None


@dataclass
class DefaultMotorActuatorConfig(MJCFConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-general-actearly
    """

    cls: str
    joint: Optional[str] = None
    jointinparent: Optional[str] = None
    site: Optional[str] = None
    refsite: Optional[str] = None
    tendon: Optional[str] = None
    body: Optional[str] = None
    gear: Optional[npt.NDArray] = None
    ctrllimited: Optional[str] = None
    forcelimited: Optional[str] = None
    ctrlrange: Optional[npt.NDArray] = None
    forcerange: Optional[npt.NDArray] = None


@dataclass
class DefaultGeneralActuatorConfig(DefaultMotorActuatorConfig):
    """
    Note:
        See: https://mujoco.readthedocs.io/en/latest/XMLreference.html#default-general-actearly
    """

    dyntype: Optional[str] = None
    biastype: Optional[str] = None
    gainprm: Optional[Union[npt.NDArray, float]] = None
    biasprm: Optional[Union[npt.NDArray, float]] = None
