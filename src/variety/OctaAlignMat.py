from __future__ import annotations

import numpy as np

from .AlignMat import align_mat
from .LoadSO3Generators_Y4 import load_so3_generators_y4


def octa_align_mat(
    normals: np.ndarray, return_align: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (n, 3).")

    _, _, _, YZ = load_so3_generators_y4()
    rot_mat_stacked = align_mat(normals, YZ)
    if return_align:
        align_mat_stacked = rot_mat_stacked[1:8, :, :]
        return rot_mat_stacked, align_mat_stacked
    return rot_mat_stacked


def OctaAlignMat(
    normals: np.ndarray, return_align: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    return octa_align_mat(normals, return_align=return_align)

