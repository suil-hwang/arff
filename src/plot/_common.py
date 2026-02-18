from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def as_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


@dataclass(slots=True)
class TetData:
    points: np.ndarray
    tets: np.ndarray


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def unpack_tet_data(tetra: Any) -> TetData:
    """Extract tetrahedral mesh arrays from common Python representations."""
    if isinstance(tetra, TetData):
        points = tetra.points
        tets = tetra.tets
    elif isinstance(tetra, tuple) and len(tetra) == 2:
        points, tets = tetra
    else:
        points = _mesh_get(tetra, "points", "verts", "Points")
        tets = _mesh_get(tetra, "tets", "tetra", "ConnectivityList")

    points2 = np.asarray(points, dtype=np.float64)
    tets2 = np.asarray(tets, dtype=np.int64)
    if points2.ndim != 2 or points2.shape[1] != 3:
        raise ValueError("tetra points must have shape (n, 3).")
    if tets2.ndim != 2 or tets2.shape[1] != 4:
        raise ValueError("tetra connectivity must have shape (m, 4).")
    return TetData(points=points2, tets=tets2)


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    t = np.asarray(tets, dtype=np.int64)
    faces = np.vstack(
        [
            t[:, [0, 1, 2]],
            t[:, [0, 1, 3]],
            t[:, [0, 2, 3]],
            t[:, [1, 2, 3]],
        ]
    )
    faces_sorted = np.sort(faces, axis=1)
    uniq, first_idx, counts = np.unique(
        faces_sorted, axis=0, return_index=True, return_counts=True
    )
    _ = uniq
    return faces[first_idx[counts == 1]]


def tet_edges(tets: np.ndarray) -> np.ndarray:
    t = np.asarray(tets, dtype=np.int64)
    edges = np.vstack(
        [
            t[:, [0, 1]],
            t[:, [0, 2]],
            t[:, [0, 3]],
            t[:, [1, 2]],
            t[:, [1, 3]],
            t[:, [2, 3]],
        ]
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def build_triangles_and_tet_tri_idx(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tets, dtype=np.int64)
    tri_stack = np.vstack(
        [
            t[:, [0, 1, 2]],
            t[:, [0, 1, 3]],
            t[:, [0, 2, 3]],
            t[:, [1, 2, 3]],
        ]
    )
    triangles, inv = np.unique(np.sort(tri_stack, axis=1), axis=0, return_inverse=True)
    tet_tri_idx = inv.reshape(-1, 4)
    return triangles, tet_tri_idx


def _solve_barycentric(
    points: np.ndarray, tets: np.ndarray, tet_idx: np.ndarray, query: np.ndarray
) -> np.ndarray:
    bary = np.full((query.shape[0], 4), np.nan, dtype=np.float64)
    inside = tet_idx >= 0
    if not np.any(inside):
        return bary
    q = query[inside]
    tri = tets[tet_idx[inside]]
    v0 = points[tri[:, 0]]
    v1 = points[tri[:, 1]]
    v2 = points[tri[:, 2]]
    v3 = points[tri[:, 3]]
    mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=2)  # (k, 3, 3)
    rhs = q - v0
    try:
        l123 = np.linalg.solve(mat, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        l123 = np.zeros((q.shape[0], 3), dtype=np.float64)
        for i in range(q.shape[0]):
            l123[i] = np.linalg.lstsq(mat[i], rhs[i], rcond=None)[0]
    l0 = 1.0 - np.sum(l123, axis=1)
    bary_inside = np.column_stack([l0, l123])
    bary[inside] = bary_inside
    return bary


def point_location(
    points: np.ndarray,
    tets: np.ndarray,
    query: np.ndarray,
    *,
    prefer_pyvista: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("query must have shape (k, 3).")

    tet_idx = np.full((q.shape[0],), -1, dtype=np.int64)
    if q.shape[0] == 0:
        return tet_idx, np.full((0, 4), np.nan, dtype=np.float64)

    if prefer_pyvista:
        try:
            pv = __import__("pyvista")
            cells = np.hstack(
                [np.full((tets.shape[0], 1), 4, dtype=np.int64), tets]
            ).ravel()
            celltypes = np.full(tets.shape[0], pv.CellType.TETRA, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, celltypes, points)
            tet_idx = np.asarray(grid.find_containing_cell(q), dtype=np.int64)
        except Exception:
            tet_idx[:] = -1

    if np.all(tet_idx < 0):
        # Fallback: brute-force barycentric test (slow but dependency-free).
        tol = 1e-10
        v = points[tets]  # (nt, 4, 3)
        v0 = v[:, 0, :]
        mats = np.stack([v[:, 1, :] - v0, v[:, 2, :] - v0, v[:, 3, :] - v0], axis=2)
        inv_mats = np.zeros_like(mats)
        valid = np.ones((tets.shape[0],), dtype=bool)
        for i in range(tets.shape[0]):
            try:
                inv_mats[i] = np.linalg.inv(mats[i])
            except np.linalg.LinAlgError:
                valid[i] = False
        for i in range(q.shape[0]):
            rhs = (q[i][None, :] - v0)  # (nt, 3)
            l123 = np.einsum("nij,nj->ni", inv_mats, rhs, optimize=True)
            l0 = 1.0 - np.sum(l123, axis=1)
            bary = np.column_stack([l0, l123])
            inside = np.all(bary >= -tol, axis=1) & np.all(bary <= 1.0 + tol, axis=1) & valid
            hit = np.where(inside)[0]
            if hit.size > 0:
                tet_idx[i] = int(hit[0])

    bary = _solve_barycentric(points, tets, tet_idx, q)
    return tet_idx, bary


def barycentric_to_cartesian(
    points: np.ndarray, tets: np.ndarray, tet_idx: np.ndarray, bary: np.ndarray
) -> np.ndarray:
    tet_idx2 = np.asarray(tet_idx, dtype=np.int64).reshape(-1)
    bary2 = np.asarray(bary, dtype=np.float64)
    if bary2.ndim != 2 or bary2.shape[1] != 4:
        raise ValueError("bary must have shape (k, 4).")
    out = np.full((tet_idx2.shape[0], 3), np.nan, dtype=np.float64)
    inside = tet_idx2 >= 0
    if not np.any(inside):
        return out
    tri = tets[tet_idx2[inside]]
    verts = points[tri]  # (k, 4, 3)
    out[inside] = np.einsum("ki,kij->kj", bary2[inside], verts, optimize=True)
    return out


def require_pyvista() -> Any:
    try:
        pv = __import__("pyvista")
    except Exception as exc:
        raise ImportError(
            "pyvista is required for this plotting function. Install with: pip install pyvista"
        ) from exc
    return pv
