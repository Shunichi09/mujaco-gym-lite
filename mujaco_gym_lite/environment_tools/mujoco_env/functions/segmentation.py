import re
from typing import Callable

import numpy as np
import numpy.typing as npt

import mujoco


def object_segmentation_from_segmentation_buffer(
    mj_model: "mujoco.MjModel", segmentation_buffer: npt.NDArray, object_names: tuple[str, ...]
) -> tuple[npt.NDArray, dict[int, str], dict[str, int]]:
    segmentation_id_to_name: dict[int, str] = {}
    name_to_segmentation_id: dict[str, int] = {}
    for i, object_name in enumerate(object_names):
        # NOTE: 0 means back ground
        segmentation_id_to_name[i + 1] = object_name
        name_to_segmentation_id[object_name] = i + 1

    mujoco_type_image = segmentation_buffer[:, :, 0]
    mujoco_id_image = segmentation_buffer[:, :, 1]

    geoms = mujoco_type_image == mujoco.mjtObj.mjOBJ_GEOM
    geom_ids = np.unique(mujoco_id_image[geoms])

    object_segmentation = np.zeros((segmentation_buffer.shape[:2]), np.uint16)
    for geom_id in geom_ids:
        model_name = str(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        for object_name in object_names:
            if object_name in model_name:
                object_segmentation[mujoco_id_image == geom_id] = name_to_segmentation_id[object_name]

    return object_segmentation, segmentation_id_to_name, name_to_segmentation_id


def _remove_suffixes(name: str) -> str:
    name = re.sub(r"_geom|_collision|_visual", "", name)
    name = re.sub(r"_[0-9]+$", "", name)  # remove _XX (X is number)
    return re.sub(r"_$", "", name)  # remove last underbar


def segmentation_object_id_map(
    mj_model: "mujoco.mjModel",
    segmentation_array: npt.NDArray,
    remove_suffixes_func: Callable[[str], str] = _remove_suffixes,
):
    mujoco_type_image = segmentation_array[:, :, 0]
    mujoco_id_image = segmentation_array[:, :, 1]

    geoms = mujoco_type_image == mujoco.mjtObj.mjOBJ_GEOM
    geom_ids = np.unique(mujoco_id_image[geoms])

    unified_id_map: dict[str, int] = {}
    id2object_map: dict[int, str] = {}
    for geom_id in geom_ids:
        object_name = str(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        object_name = remove_suffixes_func(object_name)
        geom_id = int(geom_id)
        if object_name not in unified_id_map.keys():
            # register only once to unify the id
            unified_id_map[object_name] = geom_id
        id2object_map[geom_id] = object_name
    return unified_id_map, id2object_map
