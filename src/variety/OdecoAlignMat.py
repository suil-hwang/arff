from __future__ import annotations

import numpy as np

from .AlignMat import align_mat
from .LoadSO3Generators_Y2 import load_so3_generators_y2
from .LoadSO3Generators_Y4 import load_so3_generators_y4


def odeco_align_mat(
    normals: np.ndarray, return_align: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (n, 3).")

    n = normals.shape[0]
    _, _, _, YZ2 = load_so3_generators_y2()
    _, _, _, YZ4 = load_so3_generators_y4()
    D2 = align_mat(normals, YZ2)
    D4 = align_mat(normals, YZ4)

    rot = np.zeros((15, 15, n), dtype=np.float64)
    rot[0, 0, :] = 1.0
    rot[1:6, 1:6, :] = D2
    rot[6:, 6:, :] = D4

    if not return_align:
        return rot

    align = np.zeros((8, 15, n), dtype=np.float64)
    align[0, 0, :] = np.sqrt(7.0) / 5.0
    align[0, 1:6, :] = np.sqrt(5.0 / 7.0) * D2[2, :, :]
    align[0, 6:, :] = (np.sqrt(7.0) / 35.0) * D4[4, :, :]

    align[1, 0, :] = 1.0 / (5.0 * np.sqrt(2.0))
    align[1, 6:, :] = -7.0 / (5.0 * np.sqrt(2.0)) * D4[4, :, :]

    align[2, 1:6, :] = D2[1, :, :]
    align[3, 1:6, :] = D2[3, :, :]

    align[4, 6:, :] = D4[1, :, :]
    align[5, 6:, :] = D4[3, :, :]
    align[6, 6:, :] = D4[5, :, :]
    align[7, 6:, :] = D4[7, :, :]
    return rot, align


def OdecoAlignMat(
    normals: np.ndarray, return_align: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    return odeco_align_mat(normals, return_align=return_align)

