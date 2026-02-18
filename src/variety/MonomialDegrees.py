from __future__ import annotations

import numpy as np
from scipy.linalg import hankel


def monomial_degrees(d: int) -> np.ndarray:
    if d < 0:
        raise ValueError("d must be nonnegative.")

    index_mask = hankel(np.arange(d + 1, 0, -1, dtype=np.int64)) != 0
    deg_x = np.tile(np.arange(0, d + 1, dtype=np.int64)[:, None], (1, d + 1))
    deg_y = np.tile(np.arange(0, d + 1, dtype=np.int64)[None, :], (d + 1, 1))
    deg_z = hankel(np.arange(d, -1, -1, dtype=np.int64))

    mask_flat = index_mask.reshape(-1, order="F")
    dx = deg_x.reshape(-1, order="F")[mask_flat]
    dy = deg_y.reshape(-1, order="F")[mask_flat]
    dz = deg_z.reshape(-1, order="F")[mask_flat]
    return np.vstack((dx, dy, dz))


def MonomialDegrees(d: int) -> np.ndarray:
    return monomial_degrees(d)

