from __future__ import annotations

import numpy as np

from .ExtractSingularities import extract_singularities
from ._common import require_pyvista


def _edges_from_face_rows(face_rows: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for row in face_rows:
        v = row[row >= 0]
        if v.size < 2:
            continue
        if v.size == 2:
            a, b = int(v[0]), int(v[1])
            edge_set.add((min(a, b), max(a, b)))
            continue
        for i in range(v.size - 1):
            a = int(v[i])
            b = int(v[i + 1])
            edge_set.add((min(a, b), max(a, b)))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edge_set), dtype=np.int64)


def plot_singular_graph(
    q: np.ndarray,
    tetra,
    *,
    plotter=None,
    line_width: float = 2.0,
    show: bool = False,
):
    pv = require_pyvista()
    _, _, _, sing_points, sing_edges_rows = extract_singularities(q, tetra, return_graph=True)
    if sing_points is None or sing_edges_rows is None:
        raise RuntimeError("Failed to extract singular graph.")

    edges = _edges_from_face_rows(sing_edges_rows)
    if plotter is None:
        plotter = pv.Plotter()

    if sing_points.shape[0] == 0 or edges.shape[0] == 0:
        poly = pv.PolyData()
        actor = plotter.add_mesh(poly)
        if show:
            plotter.show()
        return plotter, poly, actor

    lines = np.column_stack(
        [np.full((edges.shape[0],), 2, dtype=np.int64), edges[:, 0], edges[:, 1]]
    ).ravel()
    poly = pv.PolyData(sing_points)
    poly.lines = lines
    actor = plotter.add_mesh(poly, color="white", line_width=float(line_width), render_lines_as_tubes=False)
    plotter.set_background("white")
    if show:
        plotter.show()
    return plotter, poly, actor


def PlotSingularGraph(q: np.ndarray, tetra):
    return plot_singular_graph(q, tetra)

