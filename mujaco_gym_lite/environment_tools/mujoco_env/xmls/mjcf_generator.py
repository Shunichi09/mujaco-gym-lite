import pathlib
from typing import Optional, Type

from mujaco_gym_lite.environment_tools.mujoco_env.xmls.mjcf_configs import (
    AdhesionActuatorConfig,
    BodyConfig,
    CameraConfig,
    CompilerConfig,
    ContactExcludeConfig,
    DefaultConfig,
    DefaultGeneralActuatorConfig,
    DefaultGeomConfig,
    DefaultJointConfig,
    DefaultMaterialConfig,
    DefaultMeshConfig,
    EqualityConnectConfig,
    EqualityJointConfig,
    EqualityWeldConfig,
    FixedTendonConfig,
    FixedTendonJointConfig,
    GeneralActuatorConfig,
    GeomConfig,
    HeadlightConfig,
    InertialConfig,
    JointConfig,
    LightConfig,
    MapConfig,
    MaterialConfig,
    MeshConfig,
    MJCFConfig,
    MotorActuatorConfig,
    NameConfig,
    OptionConfig,
    OptionFlagConfig,
    PositionActuatorConfig,
    QualityConfig,
    SiteConfig,
    SizeConfig,
    TextureConfig,
    VelocityActuatorConfig,
    mjcf_config_to_xml,
)


class _MJCFElement:
    _mjcf_config_type: str
    _mjcf_config: MJCFConfig

    def __init__(self, mjcf_config_type: str, mjcf_config: "MJCFConfig") -> None:
        self._mjcf_config_type = mjcf_config_type
        self._mjcf_config = mjcf_config

    def generate(self) -> str:
        return f"<{self._mjcf_config_type} " + mjcf_config_to_xml(self._mjcf_config) + " />\n"


class _MJCFList:
    _elements: list[_MJCFElement]

    def __init__(self) -> None:
        self._elements = []

    def add(self, mjcf_config_type: str, mjcf_config: "MJCFConfig"):
        element = _MJCFElement(mjcf_config_type, mjcf_config)
        self._elements.append(element)

    def generate(self) -> str:
        return "".join([element.generate() for element in self._elements])


class _MJCFNode:
    _mjcf_config_type: str
    _mjcf_config: MJCFConfig
    _parent: Optional["_MJCFNode"]
    children: list["_MJCFNode"]
    elements: _MJCFList

    def __init__(
        self,
        mjcf_config_type: str,
        mjcf_config: MJCFConfig,
        parent: Optional["_MJCFNode"],
    ) -> None:
        self._mjcf_config_type = mjcf_config_type
        self._mjcf_config = mjcf_config
        self._parent = parent
        self.children = []
        self.elements = _MJCFList()

    def generate(self) -> str:
        child_str = ""
        for child in self.children:
            child_str += child.generate()

        element_str = self.elements.generate()

        if isinstance(self._mjcf_config, BodyConfig) or isinstance(self._mjcf_config, FixedTendonConfig):
            node_str = f"<{self._mjcf_config_type} " + mjcf_config_to_xml(self._mjcf_config) + " >\n"
        else:
            node_str = f"<{self._mjcf_config_type}>\n"

        return node_str + element_str + child_str + f"</{self._mjcf_config_type}>\n"

    def has_mjcf_config_in_elements(self, mjcf_config_class: Type["MJCFConfig"]) -> bool:
        return any([isinstance(element._mjcf_config, mjcf_config_class) for element in self.elements._elements])


def _is_default_mjcf_config(mjcf_config):
    return (
        isinstance(mjcf_config, DefaultConfig)
        or isinstance(mjcf_config, DefaultMaterialConfig)
        or isinstance(mjcf_config, DefaultMeshConfig)
        or isinstance(mjcf_config, DefaultJointConfig)
        or isinstance(mjcf_config, DefaultGeomConfig)
        or isinstance(mjcf_config, DefaultGeneralActuatorConfig)
    )


class _MJCFTree:
    _nodes: list[_MJCFNode]

    def __init__(self, root_tag_name: str, root_config: NameConfig) -> None:
        self._nodes = [_MJCFNode(root_tag_name, root_config, None)]

    def add(self, mjcf_config_type: str, mjcf_config: "MJCFConfig", parent_name: str):
        if _is_default_mjcf_config(mjcf_config):
            parent = self.search_with_node_mjcf_config_cls_name(parent_name)
        else:
            parent = self.search_with_node_mjcf_config_name(parent_name)

        if (
            isinstance(mjcf_config, BodyConfig)
            or isinstance(mjcf_config, FixedTendonConfig)
            or _is_default_mjcf_config(mjcf_config)
        ):
            node = _MJCFNode(mjcf_config_type, mjcf_config, parent)
            self._nodes.append(node)
            parent.children.append(node)
        else:
            parent.elements.add(mjcf_config_type, mjcf_config)

    def generate(self) -> str:
        return self._nodes[0].generate()

    def search_with_element_mjcf_config(self, mjcf_config_class: Type[MJCFConfig]) -> list[_MJCFNode]:
        nodes = []
        for node in self._nodes:
            if node.has_mjcf_config_in_elements(mjcf_config_class):
                nodes.append(node)
        return nodes

    def search_with_node_mjcf_config_name(
        self, mjcf_config_name: str
    ) -> _MJCFNode:  # NOTE: each config can't have the same name, we don't have to search all.
        for node in self._nodes:
            if hasattr(node._mjcf_config, "name"):
                if (node._mjcf_config.name is not None) and (node._mjcf_config.name == mjcf_config_name):
                    return node

        raise ValueError(f"{mjcf_config_name} does not exists")

    def search_with_node_mjcf_config_cls_name(
        self, mjcf_config_cls_name: str
    ) -> _MJCFNode:  # NOTE: each config can't have the same name, we don't have to search all.
        for node in self._nodes:
            if hasattr(node._mjcf_config, "cls"):
                if (node._mjcf_config.cls is not None) and (node._mjcf_config.cls == mjcf_config_cls_name):
                    return node

        raise ValueError(f"{mjcf_config_cls_name} does not exists")


class MJCFGenerator:
    _visual: _MJCFTree
    _compiler: _MJCFList
    _option: _MJCFList
    _option_flag: _MJCFTree
    _asset: _MJCFTree
    _world_body: _MJCFTree
    _actuator: _MJCFTree
    _equality: _MJCFTree
    _contact: _MJCFTree
    _tendon: _MJCFTree
    _keyframe: _MJCFTree
    _size: _MJCFList

    def __init__(self) -> None:
        super().__init__()
        self._default = _MJCFTree("default", NameConfig("default"))
        self._visual = _MJCFTree("visual", NameConfig("visual"))
        self._size = _MJCFList()
        self._compiler = _MJCFList()
        self._option = _MJCFList()
        self._option_flag = _MJCFTree("option", NameConfig("option"))
        self._asset = _MJCFTree("asset", NameConfig("asset"))
        self._world_body = _MJCFTree("worldbody", NameConfig("worldbody"))
        self._tendon = _MJCFTree("tendon", NameConfig("tendon"))
        self._keyframe = _MJCFTree("keyframe", NameConfig("keyframe"))
        self._equality = _MJCFTree("equality", NameConfig("equality"))
        self._actuator = _MJCFTree("actuator", NameConfig("actuator"))
        self._contact = _MJCFTree("contact", NameConfig("contact"))

    def add_compiler(self, mjcf_config: CompilerConfig):
        self._compiler.add("compiler", mjcf_config)

    def add_size(self, mjcf_config: SizeConfig):
        self._size.add("size", mjcf_config)

    def add_option(self, mjcf_config: OptionConfig):
        self._option.add("option", mjcf_config)

    def add_option_flag(self, mjcf_config: OptionFlagConfig):
        self._option_flag.add("flag", mjcf_config, "option")

    def add_map(self, mjcf_config: MapConfig):
        self._visual.add("map", mjcf_config, "visual")

    def add_quality(self, mjcf_config: QualityConfig):
        self._visual.add("quality", mjcf_config, "visual")

    def add_headlight(self, mjcf_config: HeadlightConfig):
        self._visual.add("headlight", mjcf_config, "visual")

    def add_material(self, mjcf_config: MaterialConfig):
        self._asset.add("material", mjcf_config, "asset")

    def add_mesh(self, mjcf_config: MeshConfig):
        self._asset.add("mesh", mjcf_config, "asset")

    def add_texture(self, mjcf_config: TextureConfig):
        self._asset.add("texture", mjcf_config, "asset")

    def add_default_parent(self, mjcf_config: DefaultConfig, parent_class_name: str):
        self._default.add("default", mjcf_config, parent_class_name)

    def add_default_material(self, mjcf_config: DefaultMaterialConfig, parent_class_name: str):
        self._default.add("material", mjcf_config, parent_class_name)

    def add_default_mesh(self, mjcf_config: DefaultMeshConfig, parent_class_name: str):
        self._default.add("mesh", mjcf_config, parent_class_name)

    def add_default_geom(self, mjcf_config: DefaultGeomConfig, parent_class_name: str):
        self._default.add("geom", mjcf_config, parent_class_name)

    def add_default_joint(self, mjcf_config: DefaultJointConfig, parent_class_name: str):
        self._default.add("joint", mjcf_config, parent_class_name)

    def add_default_general_actuator(self, mjcf_config: DefaultGeneralActuatorConfig, parent_class_name: str):
        self._actuator.add("general", mjcf_config, parent_class_name)

    def add_body(self, mjcf_config: BodyConfig, parent_name: str):
        self._world_body.add("body", mjcf_config, parent_name)

    def add_geom(self, mjcf_config: GeomConfig, parent_name: str):
        self._world_body.add("geom", mjcf_config, parent_name)

    def add_inertial(self, mjcf_config: InertialConfig, parent_name: str):
        self._world_body.add("inertial", mjcf_config, parent_name)

    def add_joint(self, mjcf_config: JointConfig, parent_name: str):
        self._world_body.add("joint", mjcf_config, parent_name)

    def add_camera(self, mjcf_config: CameraConfig, parent_name: str):
        self._world_body.add("camera", mjcf_config, parent_name)

    def add_site(self, mjcf_config: SiteConfig, parent_name: str):
        self._world_body.add("site", mjcf_config, parent_name)

    def add_light(self, mjcf_config: LightConfig, parent_name: str):
        self._world_body.add("light", mjcf_config, parent_name)

    def add_equality_joint(self, mjcf_config: EqualityJointConfig):
        self._equality.add("joint", mjcf_config, "equality")

    def add_equality_weld(self, mjcf_config: EqualityWeldConfig):
        self._equality.add("weld", mjcf_config, "equality")

    def add_equality_connect(self, mjcf_config: EqualityConnectConfig):
        self._equality.add("connect", mjcf_config, "equality")

    def add_contact_exclude(self, mjcf_config: ContactExcludeConfig):
        self._contact.add("exclude", mjcf_config, "contact")

    def add_general_actuator(self, mjcf_config: GeneralActuatorConfig):
        self._actuator.add("general", mjcf_config, "actuator")

    def add_motor_actuator(self, mjcf_config: MotorActuatorConfig):
        self._actuator.add("motor", mjcf_config, "actuator")

    def add_position_actuator(self, mjcf_config: PositionActuatorConfig):
        self._actuator.add("position", mjcf_config, "actuator")

    def add_velocity_actuator(self, mjcf_config: VelocityActuatorConfig):
        self._actuator.add("velocity", mjcf_config, "actuator")

    def add_adhesion_actuator(self, mjcf_config: AdhesionActuatorConfig):
        self._actuator.add("adhesion", mjcf_config, "actuator")

    def add_fixed_tendon(self, mjcf_config: FixedTendonConfig):
        self._tendon.add("fixed", mjcf_config, "tendon")

    def add_fixed_joint_tendon(self, mjcf_config: FixedTendonJointConfig, parent_name: str):
        self._tendon.add("joint", mjcf_config, parent_name)

    def if_has_texture_file_return_name(self, texture_file_path: pathlib.Path) -> Optional[str]:
        return self._search_with_file_path(TextureConfig, texture_file_path)

    def has_material(self, material_name: str) -> bool:
        return self._search_with_mjcf_config_name(MaterialConfig, material_name) is not None

    def generate(self) -> str:
        lines = "<mujoco>\n"
        for value in self.__dict__.values():
            lines += value.generate()
        lines += "</mujoco>\n"
        return lines

    def _search_with_file_path(self, mjcf_config_class: Type[TextureConfig], file_path: pathlib.Path) -> Optional[str]:
        nodes = self._asset.search_with_element_mjcf_config(mjcf_config_class)
        for node in nodes:
            for element in node.elements._elements:
                if isinstance(element._mjcf_config, mjcf_config_class):
                    if element._mjcf_config.file == file_path:
                        return element._mjcf_config.name
        return None

    def _search_with_mjcf_config_name(self, mjcf_config_class: Type[MaterialConfig], tag_name: str) -> Optional[str]:
        nodes = self._asset.search_with_element_mjcf_config(mjcf_config_class)
        for node in nodes:
            for element in node.elements._elements:
                if isinstance(element._mjcf_config, mjcf_config_class):
                    if element._mjcf_config.name == tag_name:
                        return element._mjcf_config.name
        return None
