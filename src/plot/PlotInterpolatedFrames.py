from __future__ import annotations

from typing import Callable

import numpy as np

from arff_io import coeff2_frames

from .PlotFrames import plot_frames
from ._common import as_real64, point_location, unpack_tet_data


def _project_coeff_samples(q_samples: np.ndarray) -> np.ndarray:
    d = q_samples.shape[0]
    try:
        if d == 9:
            from mbo import octa_mbo

            return octa_mbo().proj(q_samples)
        from mbo import odeco_mbo

        return odeco_mbo().proj(q_samples)
    except Exception:
        # Fallback when SDP dependencies (e.g., MOSEK) are unavailable.
        return q_samples


def plot_interpolated_frames(
    q: np.ndarray,
    tetra,
    samples: np.ndarray,
    *,
    color_field: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
    normal_color_rot: np.ndarray | None = None,
    plotter=None,
    show: bool = False,
):
    q2 = as_real64("q", q)
    s = as_real64("samples", samples)
    if q2.ndim != 2:
        raise ValueError("q must have shape (d, n).")
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("samples must have shape (k, 3).")

    tet = unpack_tet_data(tetra)
    points = tet.points
    tets = tet.tets
    if q2.shape[1] != points.shape[0]:
        raise ValueError("q columns must match tetra vertex count.")

    tet_idx, bary = point_location(points, tets, s)
    inside = np.where(tet_idx >= 0)[0]
    if inside.size == 0:
        return None

    s_in = s[inside]
    t_in = tet_idx[inside]
    b_in = bary[inside]
    d = q2.shape[0]

    q_verts = q2[:, tets[t_in]]  # (d, k, 4)
    q_samples = np.einsum("ki,dki->dk", b_in, q_verts, optimize=True)
    q_samples = _project_coeff_samples(q_samples)
    f_samples = coeff2_frames(q_samples, normalize=False)

    cf_vals = None
    if color_field is not None:
        if callable(color_field):
            cf_vals = np.asarray(color_field(q_samples), dtype=np.float64).reshape(-1)
        else:
            cf_vals = np.asarray(color_field, dtype=np.float64).reshape(-1)
        if cf_vals.shape[0] != s.shape[0]:
            if cf_vals.shape[0] != s_in.shape[0]:
                raise ValueError("color_field output must have one scalar per plotted sample.")
            pass
        else:
            cf_vals = cf_vals[inside]

    return plot_frames(
        f_samples,
        s_in,
        global_scale=1.0,
        plot_cubes=True,
        color_field=cf_vals,
        normal_color_rot=normal_color_rot,
        plotter=plotter,
        show=show,
    )


def PlotInterpolatedFrames(q: np.ndarray, tetra, samples: np.ndarray, **kwargs):
    arg_map = {
        "ColorField": "color_field",
        "NormalColorRot": "normal_color_rot",
    }
    py_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
    return plot_interpolated_frames(q, tetra, samples, **py_kwargs)

