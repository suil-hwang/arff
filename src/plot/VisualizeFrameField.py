from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ExtractSingularities import extract_singularities
from .PlotIntegralCurves import CurveRenderResult, plot_integral_curves
from ._common import boundary_faces_from_tets, require_pyvista, unpack_tet_data


def _boundary_polydata(points: np.ndarray, faces: np.ndarray):
    pv = require_pyvista()
    if faces.size == 0:
        return pv.PolyData()
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Boundary faces must have shape (m, 3).")
    return pv.PolyData(points, faces=np.column_stack([np.full((f.shape[0],), 3), f]).ravel())


@dataclass(slots=True)
class VisualizeFrameFieldResult:
    plotter: object
    singular_curves: CurveRenderResult | None
    all_curves: CurveRenderResult | None
    boundary_actor_left: object | None
    boundary_actor_right: object | None


def visualize_frame_field(
    frames: np.ndarray,
    energy_density: np.ndarray,
    tetra,
    bdry: np.ndarray | None = None,
    *,
    num_seeds: int = 10000,
    show: bool = False,
) -> VisualizeFrameFieldResult:
    pv = require_pyvista()
    f = np.asarray(frames, dtype=np.float64)
    if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
        raise ValueError("frames must have shape (3, 3, n).")

    tet = unpack_tet_data(tetra)
    points = tet.points
    tets = tet.tets
    nv = points.shape[0]
    if f.shape[2] != nv:
        raise ValueError("frames and tetra vertex counts must match.")

    e = np.asarray(energy_density, dtype=np.float64).reshape(-1)
    if e.shape[0] != nv:
        raise ValueError("energy_density must have one value per tetra vertex.")

    fnrm = np.linalg.norm(f, axis=0, keepdims=True)
    f_normed = np.divide(f, fnrm, out=np.zeros_like(f), where=fnrm > 0.0)

    sing_tets, _, _, _, _ = extract_singularities(f_normed, tet, return_graph=False)
    singular_curves = None
    all_curves = None
    boundary_actor_left = None
    boundary_actor_right = None

    if bdry is None:
        bdry_faces = boundary_faces_from_tets(tets)
    else:
        bdry_faces = np.asarray(bdry, dtype=np.int64)

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background("white")

    plotter.subplot(0, 0)
    if sing_tets.size > 0:
        singular_tet_data = (points, tets[sing_tets])
        singular_curves = plot_integral_curves(
            f_normed,
            singular_tet_data,
            color_field=e,
            num_seeds=num_seeds,
            prune=True,
            plotter=plotter,
            show=False,
        )
    bpoly = _boundary_polydata(points, bdry_faces)
    boundary_actor_left = plotter.add_mesh(
        bpoly,
        color="black",
        opacity=0.01,
        show_scalar_bar=False,
    )
    plotter.view_isometric()

    plotter.subplot(0, 1)
    all_curves = plot_integral_curves(
        f_normed,
        tet,
        color_field=None,
        num_seeds=num_seeds,
        prune=True,
        plotter=plotter,
        show=False,
    )
    boundary_actor_right = plotter.add_mesh(
        bpoly,
        color="black",
        opacity=0.01,
        show_scalar_bar=False,
    )
    plotter.view_isometric()
    plotter.link_views()

    if show:
        plotter.show()

    return VisualizeFrameFieldResult(
        plotter=plotter,
        singular_curves=singular_curves,
        all_curves=all_curves,
        boundary_actor_left=boundary_actor_left,
        boundary_actor_right=boundary_actor_right,
    )


def VisualizeFrameField(
    frames: np.ndarray,
    energyDensity: np.ndarray,
    tetra,
    bdry: np.ndarray,
):
    return visualize_frame_field(frames, energyDensity, tetra, bdry)
