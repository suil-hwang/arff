from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


def _load_mat_cells(path: Path) -> list[np.ndarray]:
    names = [name for name, _, _ in whosmat(str(path))]
    data = loadmat(str(path))
    return [np.asarray(data[name], dtype=np.float64) for name in names]


@lru_cache(maxsize=1)
def _load_octa_mats_cached() -> tuple[np.ndarray, ...]:
    path = Path(__file__).with_name("OctaMat.mat")
    return tuple(_load_mat_cells(path))


def load_octa_mats() -> tuple[np.ndarray, ...]:
    return tuple(m.copy() for m in _load_octa_mats_cached())


def LoadOctaMats() -> list[np.ndarray]:
    return [m.copy() for m in _load_octa_mats_cached()]
