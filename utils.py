import os
import torch
from typing import *

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


def get_all_files(path, prefix: Union[str, Tuple] = "", suffix: Union[str, Tuple] = "", contains=""):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and contains in name:
                files.append(os.path.join(pre, name))
    return files


def list_dir(path, prefix: Union[str, Tuple] = "", suffix: Union[str, Tuple] = "", contains=""):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    return [
        f"{path}/{name}"
        for name in os.listdir(path)
        if name.startswith(prefix) and name.endswith(suffix) and contains in name
    ]


get_filename = lambda path: os.path.splitext(os.path.basename(path))[0]


def rand_split(x, r, seed=42):
    return random_split(x, [int(len(x) * r), len(x) - int(len(x) * r)], generator=torch.Generator().manual_seed(seed))


def normal_split(x, r):
    return train_test_split(range(len(x)), train_size=r, shuffle=False)
