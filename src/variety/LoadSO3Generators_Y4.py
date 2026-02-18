from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm


@lru_cache(maxsize=1)
def _load_so3_generators_y4_cached() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s2 = np.sqrt(2.0)
    s72 = np.sqrt(7.0 / 2.0)
    s10 = np.sqrt(10.0)
    s3o2 = 3.0 / np.sqrt(2.0)

    Lx = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -s2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -s72, 0.0, -s2],
            [0.0, 0.0, 0.0, 0.0, 0.0, -s3o2, 0.0, -s72, 0.0],
            [0.0, 0.0, 0.0, 0.0, -s10, 0.0, -s3o2, 0.0, 0.0],
            [0.0, 0.0, 0.0, s10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, s3o2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, s72, 0.0, s3o2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [s2, 0.0, s72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, s2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    Ly = np.array(
        [
            [0.0, s2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-s2, 0.0, s72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -s72, 0.0, s3o2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -s3o2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -s10, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, s10, 0.0, -s3o2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, s3o2, 0.0, -s72, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s72, 0.0, -s2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s2, 0.0],
        ],
        dtype=np.float64,
    )

    Lz = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    YZ = expm(0.5 * np.pi * Lx)
    return Lx, Ly, Lz, YZ


def load_so3_generators_y4() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lx, Ly, Lz, YZ = _load_so3_generators_y4_cached()
    return Lx.copy(), Ly.copy(), Lz.copy(), YZ.copy()


def LoadSO3Generators_Y4() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_so3_generators_y4()
