from __future__ import annotations

from typing import Any

import numpy as np

from arff_io import coeff2_frames

from .ExtractSingularities import extract_singularities
from .PlotIntegralCurves import plot_integral_curves
from .VisualizeFrameField import _boundary_polydata


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def _solve_get_q(item: Any) -> np.ndarray:
    if isinstance(item, dict) and "q" in item:
        return np.asarray(item["q"], dtype=np.float64)
    if hasattr(item, "q"):
        return np.asarray(getattr(item, "q"), dtype=np.float64)
    raise ValueError("Each solve_info entry must provide 'q'.")


def frame_field_movie(mesh_data: Any, solve_info: list[Any], out_file: str) -> str:
    pv = __import__("pyvista")

    verts = np.asarray(_mesh_get(mesh_data, "verts", "points"), dtype=np.float64)
    tets = np.asarray(_mesh_get(mesh_data, "tets", "tetra"), dtype=np.int64)
    L = _mesh_get(mesh_data, "L")
    try:
        bdry = np.asarray(_mesh_get(mesh_data, "bdry", "bdry_faces"), dtype=np.int64)
    except ValueError:
        from ._common import boundary_faces_from_tets

        bdry = boundary_faces_from_tets(tets)

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.open_movie(out_file)
    bpoly = _boundary_polydata(verts, bdry)

    for i, row in enumerate(solve_info):
        plotter.clear()
        q = _solve_get_q(row)
        if q.shape[1] != verts.shape[0]:
            raise ValueError("q columns must match mesh vertex count.")

        if q.shape[0] == 9:
            energy = np.einsum("ij,ij->i", q.T, L @ q.T, optimize=True)
        else:
            q9 = q[6:15, :]
            energy = np.einsum("ij,ij->i", q9.T, L @ q9.T, optimize=True)
        frames = coeff2_frames(q, normalize=True)

        sing_tets, _, _, _, _ = extract_singularities(frames, (verts, tets), return_graph=False)
        if sing_tets.size > 0:
            plot_integral_curves(
                frames,
                (verts, tets[sing_tets]),
                color_field=energy,
                num_seeds=10000,
                prune=True,
                plotter=plotter,
                show=False,
            )

        plotter.add_mesh(bpoly, color="black", opacity=0.01, show_scalar_bar=False)
        if i == 0:
            plotter.set_background("black")
            plotter.view_isometric()
        plotter.write_frame()

    plotter.close()
    return out_file


def FrameFieldMovie(meshData: Any, solveInfo: list[Any], outFile: str) -> str:
    return frame_field_movie(meshData, solveInfo, outFile)
