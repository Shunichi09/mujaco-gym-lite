from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.random import np_drng


def rand_min_max(max_val: Union[npt.NDArray, float], min_val: Union[npt.NDArray, float]) -> Union[npt.NDArray, float]:
    if isinstance(max_val, np.ndarray) and isinstance(min_val, np.ndarray):
        assert max_val.shape == min_val.shape
        assert np.all(max_val >= min_val)
        return np.array(np_drng.random(*max_val.shape) * (max_val - min_val) + min_val)
    elif isinstance(max_val, float) and isinstance(min_val, float):
        assert max_val >= min_val
        return float(np_drng.random() * (max_val - min_val) + min_val)
    else:
        raise ValueError
