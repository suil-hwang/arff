from __future__ import annotations

import numpy as np


def tet_volumes(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    verts = np.asarray(verts, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must have shape (n, 3).")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (m, 4).")

    v0 = verts[tets[:, 0]]
    v1 = verts[tets[:, 1]]
    v2 = verts[tets[:, 2]]
    v3 = verts[tets[:, 3]]
    vol6 = np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0))
    return np.abs(vol6) / 6.0


TetVolumes = tet_volumes

