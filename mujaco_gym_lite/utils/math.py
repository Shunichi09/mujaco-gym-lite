from typing import Union

import numpy as np
import numpy.typing as npt


def generate_2d_grid(min_x: float, max_x: float, num_x: int, min_y: float, max_y: float, num_y: int) -> npt.NDArray:
    x_arange = np.linspace(min_x, max_x, num_x)
    y_arange = np.linspace(min_y, max_y, num_y)
    x_pos, y_pos = np.meshgrid(x_arange, y_arange)
    return np.array([[x, y] for x, y in zip(x_pos.flatten(), y_pos.flatten())])


def fit_angle_in_range(
    angles: Union[float, list[float], tuple[float], npt.NDArray], min_angle: float = -np.pi, max_angle: float = np.pi
) -> npt.NDArray:
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError("difference between max_angle and min_angle must be greater than 2.0 * pi")

    output = np.array(angles)
    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    return output.reshape(output_shape)


def calculate_part_accuracy(data: npt.NDArray, indexes: npt.NDArray) -> npt.NDArray:
    if len(indexes) < 2:
        raise ValueError("Indexes must have at least two values.")

    if any(idx < 0 or idx > len(data) for idx in indexes):
        raise ValueError("Indexes are out of bounds.")

    cumulative_accuracies = []
    for i in range(len(indexes) - 1):
        s, e = indexes[i], indexes[i + 1]
        subset = data[s:e]
        accuracy = np.mean(subset) if len(subset) > 0 else 0
        cumulative_accuracies.append(accuracy)

    return np.array(cumulative_accuracies)


def calculate_cumulative_accuracy(data, indexes) -> npt.NDArray:
    cumulative_accuracies = []
    for index in indexes:
        subset = data[:index]
        accuracy = float(np.mean(subset)) if len(subset) > 0 else 0
        cumulative_accuracies.append(accuracy)
    return np.array(cumulative_accuracies)


def divided_indices(n: int, num_parts: int) -> npt.NDArray:
    step = n / num_parts
    return np.array([int(round(i * step)) for i in range(num_parts + 1)])
