from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .IntegralCurves import integral_curves
from ._common import require_pyvista


@dataclass(slots=True)
class CurveRenderResult:
    curve_heads: np.ndarray
    curve_color: np.ndarray
    polydata: object
    actor: object
    plotter: object


def _equalize_color(colors: np.ndarray) -> np.ndarray:
    """Rank-based histogram equalization matching MATLAB's
    ``[~,perm]=sort(c); eq(perm)=1:n; c=eq;``."""
    out = colors.copy()
    finite_mask = np.isfinite(out)
    finite_vals = out[finite_mask]
    if finite_vals.size == 0:
        return out
    perm = np.argsort(finite_vals, kind="stable")
    equalized = np.empty_like(finite_vals)
    equalized[perm] = np.arange(1, finite_vals.size + 1, dtype=np.float64)
    out[finite_mask] = equalized
    return out


def _build_polyline_polydata(
    curve_heads: np.ndarray,
    curve_color: np.ndarray | None,
    *,
    interp_color: bool,
) -> object:
    pv = require_pyvista()

    n_steps, n_curves, _ = curve_heads.shape
    points_list: list[np.ndarray] = []
    cells: list[int] = []
    point_scalar: list[float] = []

    for j in range(n_curves):
        finite = np.where(np.all(np.isfinite(curve_heads[:, j, :]), axis=1))[0]
        if finite.size < 2:
            continue
        pts = curve_heads[finite, j, :]
        start = len(points_list)
        points_list.extend(list(pts))
        cells.extend([pts.shape[0], *range(start, start + pts.shape[0])])
        if interp_color and curve_color is not None:
            vals = curve_color[j, finite]
            point_scalar.extend(list(vals))
        else:
            color_id = float(np.random.randint(1, 21))
            point_scalar.extend([color_id] * pts.shape[0])

    if len(points_list) == 0:
        poly = pv.PolyData()
        poly["curve_scalar"] = np.zeros((0,), dtype=np.float64)
        return poly

    poly = pv.PolyData(np.asarray(points_list, dtype=np.float64))
    poly.lines = np.asarray(cells, dtype=np.int64)
    scalars = np.asarray(point_scalar, dtype=np.float64)
    if interp_color:
        scalars = _equalize_color(scalars)
    poly["curve_scalar"] = scalars
    return poly


def plot_integral_curves(
    frames: np.ndarray,
    tetra,
    *,
    color_field: np.ndarray | None = None,
    plotter=None,
    line_width: float = 1.5,
    show: bool = False,
    **kwargs,
) -> CurveRenderResult:
    pv = require_pyvista()
    interp_color = color_field is not None

    curve_heads, curve_color = integral_curves(
        frames,
        tetra,
        color_field=color_field,
        **kwargs,
    )

    poly = _build_polyline_polydata(
        curve_heads,
        curve_color if interp_color else None,
        interp_color=interp_color,
    )
    if plotter is None:
        plotter = pv.Plotter()

    if interp_color:
        actor = plotter.add_mesh(
            poly,
            scalars="curve_scalar",
            cmap="inferno",
            line_width=float(line_width),
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )
    else:
        actor = plotter.add_mesh(
            poly,
            scalars="curve_scalar",
            cmap="tab20",
            line_width=float(line_width),
            render_lines_as_tubes=False,
            show_scalar_bar=False,
        )

    if show:
        plotter.show()

    return CurveRenderResult(
        curve_heads=curve_heads,
        curve_color=curve_color,
        polydata=poly,
        actor=actor,
        plotter=plotter,
    )


def PlotIntegralCurves(frames: np.ndarray, tetra, **kwargs):
    arg_map = {"ColorField": "color_field"}
    py_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
    return plot_integral_curves(frames, tetra, **py_kwargs)

