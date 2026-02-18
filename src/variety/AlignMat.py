from __future__ import annotations

import numpy as np

from .ExpSO3 import exp_so3


def align_mat(axes: np.ndarray, YZ: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64)
    YZ = np.asarray(YZ, dtype=np.float64)
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("axes must have shape (n, 3).")
    if YZ.ndim != 2 or YZ.shape[0] != YZ.shape[1]:
        raise ValueError("YZ must be square.")

    n = axes.shape[0]
    d = YZ.shape[0]
    axis_angles = np.repeat(axes, d, axis=0)
    q0 = np.tile(np.eye(d, dtype=np.float64), (1, n))
    D = exp_so3(axis_angles=axis_angles, q=q0, YZ=YZ, rotate_north_only=True)
    return np.reshape(D, (d, d, n), order="F")


def AlignMat(axes: np.ndarray, YZ: np.ndarray) -> np.ndarray:
    return align_mat(axes, YZ)

