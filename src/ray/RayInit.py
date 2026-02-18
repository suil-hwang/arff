from __future__ import annotations

from typing import Any

import numpy as np

from arff_io import frames2_octa

from ._core import _as_real64, compute_ff


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def _as_index(name: str, x: np.ndarray, n: int) -> np.ndarray:
    idx = np.asarray(x, dtype=np.int64).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"{name} contains out-of-range indices.")
    return idx


def _prepare_tets(tets: np.ndarray, nv: int) -> np.ndarray:
    tet = np.asarray(tets, dtype=np.int64)
    if tet.ndim != 2 or tet.shape[1] != 4:
        raise ValueError("tets must have shape (nt, 4).")

    if tet.size > 0 and tet.min() == 1 and tet.max() <= nv:
        tet = tet - 1
    if tet.size > 0 and (tet.min() < 0 or tet.max() >= nv):
        raise ValueError("tets contains out-of-range indices.")

    if tet.shape[0] <= 1:
        return tet
    order = np.lexsort((tet[:, 3], tet[:, 2], tet[:, 1], tet[:, 0]))
    return tet[order, :]


def ray_init(mesh_data: Any) -> tuple[np.ndarray, np.ndarray]:
    nv = int(_mesh_get(mesh_data, "nv"))
    if nv <= 0:
        raise ValueError("mesh_data.nv must be positive.")

    bdry_idx = _as_index("bdry_idx", _mesh_get(mesh_data, "bdry_idx", "bdryIdx"), nv)
    bdry_normals = _as_real64("bdry_normals", _mesh_get(mesh_data, "bdry_normals", "bdryNormals"))
    if bdry_normals.shape != (bdry_idx.size, 3):
        raise ValueError("bdry_normals must have shape (len(bdry_idx), 3).")

    tet = _prepare_tets(_mesh_get(mesh_data, "tets", "tetra"), nv)
    normals = np.zeros((nv, 3), dtype=np.float64)
    if bdry_idx.size > 0:
        normals[bdry_idx, :] = bdry_normals

    frames = compute_ff(normals, tet)
    q = frames2_octa(frames)
    return q, frames


def RayInit(meshData: Any) -> tuple[np.ndarray, np.ndarray]:
    return ray_init(meshData)

