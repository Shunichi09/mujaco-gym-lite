import random

import numpy as np

import torch

np_drng = np.random.default_rng()


def numpy_seed(seed: int):
    global np_drng
    np_drng = np.random.default_rng(seed)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    numpy_seed(seed)
