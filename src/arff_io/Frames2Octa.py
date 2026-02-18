from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from variety import exp_so3, load_so3_generators_y4, octa_z_aligned

from ._common import as_real64


def frames2_octa(frames: np.ndarray) -> np.ndarray:
    f = as_real64("frames", frames)
    if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
        raise ValueError("frames must have shape (3, 3, n).")
    n = f.shape[2]

    rot_mats = np.transpose(f, (2, 0, 1))
    rotvec = Rotation.from_matrix(rot_mats).as_rotvec()  # (n, 3)

    _, _, _, yz = load_so3_generators_y4()
    q0 = np.repeat(octa_z_aligned(np.array([0.0], dtype=np.float64)), n, axis=1)
    return exp_so3(rotvec, q0, yz)


def Frames2Octa(frames: np.ndarray) -> np.ndarray:
    return frames2_octa(frames)
