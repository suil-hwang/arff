from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm


@lru_cache(maxsize=1)
def _load_so3_generators_y2_cached() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s3 = np.sqrt(3.0)

    Lx = np.array(
        [
            [0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -s3, 0.0, -1.0],
            [0.0, s3, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    Ly = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -s3, 0.0],
            [0.0, 0.0, s3, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    Lz = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    YZ = expm(0.5 * np.pi * Lx)
    return Lx, Ly, Lz, YZ


def load_so3_generators_y2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lx, Ly, Lz, YZ = _load_so3_generators_y2_cached()
    return Lx.copy(), Ly.copy(), Lz.copy(), YZ.copy()


def LoadSO3Generators_Y2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_so3_generators_y2()
