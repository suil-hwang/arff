from __future__ import annotations

from typing import Any

import numpy as np

from arff_io import coeff2_frames

from .VisualizeFrameField import visualize_frame_field


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def visualize_result(mesh_data: Any, q: np.ndarray, *, show: bool = False):
    q2 = np.asarray(q, dtype=np.float64)
    if q2.ndim != 2:
        raise ValueError("q must have shape (d, n).")

    L = _mesh_get(mesh_data, "L")
    tets = _mesh_get(mesh_data, "tets", "tetra")
    verts = _mesh_get(mesh_data, "verts", "points")
    bdry = None
    for name in ("bdry", "bdry_faces", "bdryFaces"):
        try:
            bdry = _mesh_get(mesh_data, name)
            break
        except ValueError:
            continue

    frames = coeff2_frames(q2)
    if q2.shape[0] == 9:
        lq = L @ q2.T
        energy = np.einsum("ij,ij->i", q2.T, lq, optimize=True)
    else:
        qq = q2[6:15, :]
        lq = L @ qq.T
        energy = np.einsum("ij,ij->i", qq.T, lq, optimize=True)

    return visualize_frame_field(
        frames,
        energy,
        (np.asarray(verts, dtype=np.float64), np.asarray(tets, dtype=np.int64)),
        None if bdry is None else np.asarray(bdry, dtype=np.int64),
        show=show,
    )


def VisualizeResult(meshData: Any, q: np.ndarray):
    return visualize_result(meshData, q)

