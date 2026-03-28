import datetime
import glob
import json
import pathlib
import pickle
import re
from typing import Any, Union, cast

import cv2
import numpy as np
import numpy.typing as npt


def _natural_sort(letter: list[str]) -> list[str]:
    def convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(letter, key=alphanum_key)


def get_files(dir_path: pathlib.Path, file_format: Union[str, list[str]] = "**/") -> list[pathlib.Path]:
    file_format = [file_format] if isinstance(file_format, str) else file_format
    file_lists = []
    for format in file_format:
        load_file_path = dir_path / format
        format_files = glob.glob(str(load_file_path))
        file_lists.extend(format_files)
    return [pathlib.Path(f) for f in _natural_sort(file_lists)]


def write_txt(file_path: pathlib.Path, string: str):
    with open(file_path, mode="w") as f:
        f.write(string)


def _numpy_to_list(data):
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            data[key] = val.tolist()
        elif isinstance(val, dict):
            data[key] = _numpy_to_list(val)
    return data


def _pathlib_path_to_str(data):
    for key, val in data.items():
        if isinstance(val, pathlib.Path):
            data[key] = str(val)
        elif isinstance(val, dict):
            data[key] = _pathlib_path_to_str(val)
    return data


def _sanitize_data(data):
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if not isinstance(key, str):
                key = str(key)
            sanitized_value = _sanitize_data(value)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        return sanitized
    elif isinstance(data, list):
        sanitized_list = []
        for item in data:
            sanitized_item = _sanitize_data(item)
            if sanitized_item is not None:
                sanitized_list.append(sanitized_item)
        return sanitized_list
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    else:
        return None


def write_json(file_path: pathlib.Path, data: dict, apply_sanitize: bool = False):
    data = _numpy_to_list(data)
    data = _pathlib_path_to_str(data)
    if apply_sanitize:
        data = _sanitize_data(data)
    with open(str(file_path), mode="w") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: pathlib.Path) -> dict:
    with open(str(file_path), mode="r") as f:
        dict_data = json.load(f)
    return cast(dict, dict_data)


def load_txt_as_str_list(file_path: pathlib.Path) -> list[str]:
    with open(str(file_path), mode="r") as f:
        lines = f.read().splitlines()
    return lines


def write_to_pickle(file_path: pathlib.Path, data: Any):
    with open(str(file_path), mode="wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: pathlib.Path):
    with open(str(file_path), mode="rb") as f:
        data = pickle.load(f)
    return data


def write_to_cv_png(file_path: pathlib.Path, image: npt.NDArray, image_type: str = "color"):
    if image_type == "color":
        cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif image_type == "depth":
        raise NotImplementedError
    else:
        raise ValueError


def datetime_as_str(format: str = "%Y%m%d-%H%M%S") -> str:
    now = datetime.datetime.now()
    return now.strftime(format)


def extract_view_number(file_path_name: str) -> int:
    match = re.search(r"view_(\d+)", file_path_name)
    if match:
        view_number = int(match.group(1))
        return view_number
    else:
        raise ValueError
