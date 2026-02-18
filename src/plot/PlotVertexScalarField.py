from __future__ import annotations

import numpy as np

from ._common import point_location, require_pyvista, tet_edges, unpack_tet_data


def _matlab_alphamap() -> np.ndarray:
    """MATLAB-matching piece-wise opacity: zeros(32), linspace(0,1,192), ones(32)."""
    return np.concatenate(
        [
            np.zeros(32, dtype=np.float64),
            np.linspace(0.0, 1.0, 192, dtype=np.float64),
            np.ones(32, dtype=np.float64),
        ]
    )


def plot_vertex_scalar_field(
    tetra,
    values: np.ndarray,
    *,
    plotter=None,
    show: bool = False,
):
    pv = require_pyvista()
    tet = unpack_tet_data(tetra)
    points = tet.points
    tets = tet.tets
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.shape[0] != points.shape[0]:
        raise ValueError("values must have one value per tetra vertex.")

    lb = np.min(points, axis=0)
    ub = np.max(points, axis=0)
    edges = tet_edges(tets)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    voxel_size = 0.5 * float(np.min(edge_lengths))
    voxel_size = max(voxel_size, np.finfo(np.float64).eps)

    xs = np.arange(lb[0], ub[0] + 0.5 * voxel_size, voxel_size)
    ys = np.arange(lb[1], ub[1] + 0.5 * voxel_size, voxel_size)
    zs = np.arange(lb[2], ub[2] + 0.5 * voxel_size, voxel_size)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    samples = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])

    tet_idx, bary = point_location(points, tets, samples)
    grid_vals = np.zeros((samples.shape[0],), dtype=np.float64)
    inside = tet_idx >= 0
    if np.any(inside):
        vert_vals = vals[tets[tet_idx[inside]]]
        grid_vals[inside] = np.einsum("ki,ki->k", bary[inside], vert_vals, optimize=True)

    img = pv.ImageData()
    img.origin = (float(xs[0]), float(ys[0]), float(zs[0]))
    img.spacing = (voxel_size, voxel_size, voxel_size)
    img.dimensions = (xs.size, ys.size, zs.size)
    img.point_data["values"] = grid_vals

    if plotter is None:
        plotter = pv.Plotter()
    actor = plotter.add_volume(
        img,
        scalars="values",
        cmap="inferno",
        opacity=_matlab_alphamap().tolist(),
        blending="maximum",
        shade=False,
    )
    plotter.set_background("white")
    if show:
        plotter.show()
    return plotter, img, actor


def PlotVertexScalarField(tetra, values: np.ndarray):
    return plot_vertex_scalar_field(tetra, values)

