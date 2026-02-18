from __future__ import annotations

import numpy as np


def octa_z_aligned(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    n = theta.shape[0]
    out = np.zeros((9, n), dtype=np.float64)
    out[0, :] = -np.sqrt(5.0 / 12.0) * np.sin(4.0 * theta)
    out[4, :] = np.sqrt(7.0 / 12.0)
    out[8, :] = np.sqrt(5.0 / 12.0) * np.cos(4.0 * theta)
    return out


def OctaZAligned(theta: np.ndarray) -> np.ndarray:
    return octa_z_aligned(theta)

