from __future__ import annotations

import numpy as np

from mesh import tet_volumes

from ._common import as_real64, barycentric_to_cartesian, point_location, tet_edges, unpack_tet_data


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros_like(v, dtype=np.float64)
    nz = nrm[:, 0] > 0.0
    out[nz] = v[nz] / nrm[nz]
    return out


def integral_curves(
    frames: np.ndarray,
    tetra,
    *,
    num_seeds: int = 1000,
    prune: bool = True,
    color_field: np.ndarray | None = None,
    curve_length: float | None = None,
    prefer_pyvista_locator: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    f = as_real64("frames", frames)
    if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
        raise ValueError("frames must have shape (3, 3, n).")
    tet = unpack_tet_data(tetra)
    points = tet.points
    tets = tet.tets
    nv = points.shape[0]
    if f.shape[2] != nv:
        raise ValueError("frames vertex count must match tetra points.")

    if color_field is None:
        interp_color = False
        cf = np.zeros((nv,), dtype=np.float64)
    else:
        cf = as_real64("color_field", color_field).reshape(-1)
        if cf.shape[0] != nv:
            raise ValueError("color_field must have one value per vertex.")
        interp_color = True

    lb = np.min(points, axis=0)
    ub = np.max(points, axis=0)
    default_curve_length = 0.25 * float(np.max(ub - lb))
    cl = default_curve_length if curve_length is None else float(curve_length)

    edges = tet_edges(tets)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    dt = 0.5 * float(np.mean(edge_lengths) - np.std(edge_lengths))
    dt = max(dt, np.finfo(np.float64).eps)
    n_steps = max(1, int(cl / dt))

    n_seeds = int(num_seeds)
    curve_heads = np.full((n_steps, n_seeds, 3), np.nan, dtype=np.float64)
    curve_color = np.full((n_seeds, n_steps), np.nan, dtype=np.float64)

    vol = tet_volumes(points, tets)
    if np.any(vol < 0.0) or float(np.sum(vol)) <= 0.0:
        raise ValueError("Invalid tetrahedral mesh volume.")
    vol = vol / np.sum(vol)
    rand_tets = np.random.choice(tets.shape[0], size=n_seeds, replace=True, p=vol)
    rand_bary = np.random.rand(n_seeds, 4)
    rand_bary = rand_bary / np.sum(rand_bary, axis=1, keepdims=True)
    curve_heads[0] = barycentric_to_cartesian(points, tets, rand_tets, rand_bary)
    if interp_color:
        curve_color[:, 0] = np.sum(rand_bary * cf[tets[rand_tets]], axis=1)

    curve_vel = _normalize_rows(np.random.randn(n_seeds, 3).astype(np.float64))
    axis_dirs = np.transpose(
        np.concatenate([f, -f], axis=1), (1, 0, 2)
    )  # (6, 3, nv), rows are candidate directions

    inside_idx = np.arange(n_seeds, dtype=np.int64)
    for k in range(1, n_steps):
        prev_inside = inside_idx.copy()
        tet_idx, bary = point_location(
            points,
            tets,
            curve_heads[k - 1, inside_idx, :],
            prefer_pyvista=prefer_pyvista_locator,
        )
        in_mask = tet_idx >= 0
        inside_idx = prev_inside[in_mask]
        if inside_idx.size == 0:
            break

        tet_idx = tet_idx[in_mask]
        bary = bary[in_mask]
        neigh_vert_idx = tets[tet_idx]  # (nc, 4)
        if interp_color:
            curve_color[inside_idx, k] = np.sum(bary * cf[neigh_vert_idx], axis=1)

        nc = inside_idx.size
        v_old = curve_vel[inside_idx]
        neigh_vel = np.zeros((nc, 4, 3), dtype=np.float64)
        for j in range(4):
            dirs_j = np.transpose(axis_dirs[:, :, neigh_vert_idx[:, j]], (2, 0, 1))  # (nc, 6, 3)
            score = np.einsum("nij,nj->ni", dirs_j, v_old, optimize=True)
            choice = np.argmax(score, axis=1)
            neigh_vel[:, j, :] = dirs_j[np.arange(nc), choice, :]

        v_new = np.einsum("ni,nij->nj", bary, neigh_vel, optimize=True)
        v_new = _normalize_rows(v_new)
        curve_vel[inside_idx] = v_new
        curve_heads[k, inside_idx, :] = curve_heads[k - 1, inside_idx, :] + dt * v_new

    if prune:
        finite_steps = np.sum(np.all(np.isfinite(curve_heads), axis=2), axis=0)
        prune_threshold = 0.1 * float(n_steps)
        if prune_threshold > 5.0:
            prune_threshold = 5.0
        retain = np.where(finite_steps > prune_threshold)[0]
        curve_heads = curve_heads[:, retain, :]
        if interp_color:
            curve_color = curve_color[retain, :]
        else:
            curve_color = np.full((curve_heads.shape[1], n_steps), np.nan, dtype=np.float64)

    return curve_heads, curve_color


def IntegralCurves(frames: np.ndarray, tetra, **kwargs):
    arg_map = {
        "NumSeeds": "num_seeds",
        "Prune": "prune",
        "ColorField": "color_field",
        "CurveLength": "curve_length",
    }
    py_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
    return integral_curves(frames, tetra, **py_kwargs)
