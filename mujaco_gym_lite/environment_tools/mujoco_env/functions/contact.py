import itertools

import numpy as np
import numpy.typing as npt

import mujoco


def is_contact_between_models(
    mj_model: "mujoco.mjModel",
    mj_data: "mujoco.mjData",
    object_model_names: list[str],
    exclude_abstract_names: list[str] = [],
) -> bool:
    num_contacts = []
    for pair in itertools.combinations(object_model_names, 2):
        n_con, geoms, _ = get_contact_info_between_abstract_geom_names(
            mj_model, mj_data, pair[0], pair[1], exclude_abstract_names
        )
        num_contacts.append(n_con)
    return any(np.array(num_contacts) > 0)


def _get_contact(mj_model: "mujoco.mjModel", mj_data: "mujoco.mjData", contact_id: int):
    contact = mj_data.contact[contact_id]
    contact_geom1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    contact_geom2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    return contact, contact_geom1, contact_geom2


def get_contact_info_of_geom_names(
    mj_model: "mujoco.mjModel",
    mj_data: "mujoco.mjData",
    geom_names: list[str],
    exclude_geom_names: list[str] = [],
    exclude_abstract_geom_names: list[str] = [],
) -> tuple[int, list[tuple[str, str]], list[npt.NDArray]]:
    """
    Returns:
        Tuple[int, List[Tuple[str, str]], List[npt.NDArray]]:
            number of contacts, list of the pair of contact geom names, list of contact points
    """
    # you can get contact info between banana_XXXX and apple_YYYY
    num_contacts = 0
    contact_geoms = []
    contact_points = []
    for contact_id in range(mj_data.ncon):
        contact, contact_geom1, contact_geom2 = _get_contact(mj_model, mj_data, contact_id)
        if (contact_geom1 in geom_names) or (contact_geom2 in geom_names):
            if not ((contact_geom1 in exclude_geom_names) or (contact_geom2 in exclude_geom_names)):
                exclude = False
                for exclude_name in exclude_abstract_geom_names:
                    if (exclude_name in contact_geom1) or (exclude_name in contact_geom2):
                        exclude = True
                        break

                if exclude:
                    continue

                num_contacts += 1
                contact_geoms.append((contact_geom1, contact_geom2))
                contact_points.append(contact.pos.copy().tolist())

    return num_contacts, contact_geoms, contact_points


def get_contact_info_between_abstract_geom_names(
    mj_model: "mujoco.mjModel",
    mj_data: "mujoco.mjData",
    geom1_abstract_name: str,
    geom2_abstract_name: str,
    exclude_abstract_names: list[str],
) -> tuple[int, list[tuple[str, str]], list[npt.NDArray]]:
    """get contact info between the abstract geom names
    For example, you can get contact info between banana_XXXX and apple_YYYY
    if you call this function (apple, banana)
    """
    # you can get contact info between banana_XXXX and apple_YYYY
    num_contacts = 0
    contact_geoms = []
    contact_points = []
    for contact_id in range(mj_data.ncon):
        contact, contact_geom1, contact_geom2 = _get_contact(mj_model, mj_data, contact_id)

        if (geom1_abstract_name in contact_geom1 and geom2_abstract_name in contact_geom2) or (
            geom1_abstract_name in contact_geom2 and geom2_abstract_name in contact_geom1
        ):
            exclude = False
            for exclude_name in exclude_abstract_names:
                if (exclude_name in contact_geom1) or (exclude_name in contact_geom2):
                    exclude = True
                    break

            if exclude:
                continue

            num_contacts += 1
            contact_geoms.append((contact_geom1, contact_geom2))
            contact_points.append(contact.pos.copy().tolist())

    return num_contacts, contact_geoms, contact_points


def get_contact_info_of_model(
    mj_model: "mujoco.mjModel",
    mj_data: "mujoco.mjData",
    geom_abstract_name: str,
    exclude_abstract_names: list[str],
) -> tuple[int, list[str], list[npt.NDArray]]:
    # you can get contact info between banana_XXXX and apple_YYYY
    num_contacts = 0
    contact_geoms = []
    contact_points = []
    for contact_id in range(mj_data.ncon):
        contact, contact_geom1, contact_geom2 = _get_contact(mj_model, mj_data, contact_id)
        for exclude_name in exclude_abstract_names:
            if (exclude_name in contact_geom1) or (exclude_name in contact_geom2):
                continue

        if geom_abstract_name in contact_geom1:
            if geom_abstract_name in contact_geom2:  # without self collision
                continue
            num_contacts += 1
            contact_geoms.append(contact_geom2)
            contact_points.append(contact.pos.copy().tolist())
        elif geom_abstract_name in contact_geom2:  # without self collision
            if geom_abstract_name in contact_geom1:
                continue
            num_contacts += 1
            contact_geoms.append(contact_geom1)
            contact_points.append(contact.pos.copy().tolist())

    return num_contacts, contact_geoms, contact_points
