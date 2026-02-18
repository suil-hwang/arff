from __future__ import annotations

from pathlib import Path

import numpy as np

from .IntegralCurves import integral_curves


def export_integral_curves(
    filename: str,
    frames: np.ndarray,
    tetra,
    *,
    clip_plane: np.ndarray | None = None,
    matlab_indexing: bool = False,
    **kwargs,
) -> str:
    curve_heads, _ = integral_curves(frames, tetra, **kwargs)
    curves = curve_heads.copy()

    if clip_plane is not None:
        clip = np.asarray(clip_plane, dtype=np.float64).reshape(1, 1, 3)
        remove = np.any(np.sum(curves * clip, axis=2) > 1.0, axis=0)
        curves[:, remove, :] = np.nan

    n_steps, n_curves, _ = curves.shape
    idx_offset = 1 if matlab_indexing else 0
    points: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    idx = -np.ones((n_steps, n_curves), dtype=np.int64)

    for j in range(n_curves):
        last = -1
        for i in range(n_steps):
            p = curves[i, j]
            if not np.all(np.isfinite(p)):
                continue
            vid = len(points) + idx_offset
            points.append(p)
            idx[i, j] = vid
            if last >= 0:
                edges.append((last, vid))
            last = vid

    out = Path(filename)
    with out.open("w", encoding="utf-8") as f:
        for p in points:
            f.write(f"v {p[0]:.16g} {p[1]:.16g} {p[2]:.16g}\n")
        for e0, e1 in edges:
            f.write(f"f {e0} {e1}\n")
    return str(out)


def ExportIntegralCurves(filename: str, frames: np.ndarray, tetra, **kwargs) -> str:
    arg_map = {"ClipPlane": "clip_plane", "MatlabIndexing": "matlab_indexing"}
    py_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
    return export_integral_curves(filename, frames, tetra, **py_kwargs)
