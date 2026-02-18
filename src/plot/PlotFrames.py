from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import as_real64, require_pyvista


@dataclass(slots=True)
class FrameRenderResult:
    polydata: object
    actor: object
    plotter: object


_CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)
_CUBE_FACES = np.array(
    [
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
        [3, 2, 6, 7],
        [4, 5, 6, 7],
    ],
    dtype=np.int64,
)
_CUBE_NORMALS = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _default_scale(centers: np.ndarray) -> float:
    if centers.shape[0] <= 1:
        return 1.0
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_d = float(np.sqrt(np.min(d2)))
    if not np.isfinite(min_d) or min_d <= 0.0:
        return 1.0
    return min_d / np.sqrt(3.0)


def _build_all_cubes(
    frames: np.ndarray, centers: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized cube geometry: returns (8*n, 3) points and (6*n, 4) face indices."""
    n = frames.shape[2]
    # frames: (3, 3, n) → batch (n, 3, 3)
    f_batch = np.moveaxis(frames, 2, 0)  # (n, 3, 3)
    # Transform cube verts: centers[i] + f[i] @ (CUBE_VERTS.T * scale)
    # scaled_verts: (8, 3) * scale, then f_batch @ scaled_verts.T → (n, 3, 8)
    transformed = np.matmul(f_batch, _CUBE_VERTS.T * scale)  # (n, 3, 8)
    transformed = np.transpose(transformed, (0, 2, 1))  # (n, 8, 3)
    all_points = transformed + centers[:, None, :]  # (n, 8, 3)
    all_points = all_points.reshape(8 * n, 3)

    offsets = (np.arange(n, dtype=np.int64) * 8)[:, None]  # (n, 1)
    face_idx = _CUBE_FACES[None, :, :] + offsets[:, :, None]  # (n, 6, 4)... broadcast
    # Actually: _CUBE_FACES is (6, 4), offsets is (n, 1)
    face_idx = np.tile(_CUBE_FACES, (n, 1, 1)) + offsets[:, None, :]  # (n, 6, 4)
    all_faces = face_idx.reshape(6 * n, 4)
    return all_points, all_faces


def _build_cube_polydata(
    frames: np.ndarray,
    centers: np.ndarray,
    *,
    global_scale: float,
    color_field: np.ndarray | None,
    normal_color_rot: np.ndarray | None,
) -> tuple[object, dict[str, np.ndarray]]:
    pv = require_pyvista()
    n = frames.shape[2]
    scale = _default_scale(centers) * float(global_scale)

    all_points, raw_faces = _build_all_cubes(frames, centers, scale)
    faces = np.column_stack(
        [np.full((raw_faces.shape[0],), 4, dtype=np.int64), raw_faces]
    )

    face_rgb = np.full((6 * n, 3), 0.9, dtype=np.float64)
    face_scalar = np.zeros((6 * n,), dtype=np.float64)

    if color_field is not None:
        face_scalar = np.repeat(color_field, 6)
    elif normal_color_rot is not None:
        f_batch = np.moveaxis(frames, 2, 0)  # (n, 3, 3)
        normals = np.matmul(f_batch, _CUBE_NORMALS.T)  # (n, 3, 6)
        normals = np.transpose(normals, (0, 2, 1))  # (n, 6, 3)
        nrm = np.linalg.norm(normals, axis=2, keepdims=True)
        nz = nrm > 0.0
        normals = np.where(nz, normals / np.where(nz, nrm, 1.0), 0.0)
        face_rgb = np.abs(normals.reshape(6 * n, 3) @ normal_color_rot)

    poly = pv.PolyData(all_points, faces=faces.ravel())
    arrays: dict[str, np.ndarray] = {}
    if color_field is not None:
        arrays["face_scalar"] = face_scalar
    else:
        arrays["face_rgb"] = np.clip(face_rgb, 0.0, 1.0)
    return poly, arrays


def _build_line_polydata(
    frames: np.ndarray,
    centers: np.ndarray,
    *,
    global_scale: float,
    color_field: np.ndarray | None,
) -> tuple[object, dict[str, np.ndarray]]:
    pv = require_pyvista()
    n = frames.shape[2]
    v1 = frames[:, 0, :].T
    v2 = frames[:, 1, :].T
    v3 = frames[:, 2, :].T
    vectors = np.vstack([v1, v2, v3, -v1, -v2, -v3])
    starts = np.tile(centers, (6, 1))
    ends = starts + float(global_scale) * vectors

    m = starts.shape[0]
    points = np.vstack([starts, ends])
    lines = np.column_stack(
        [np.full((m,), 2, dtype=np.int64), np.arange(m, dtype=np.int64), np.arange(m, dtype=np.int64) + m]
    ).ravel()

    poly = pv.PolyData(points)
    poly.lines = lines
    arrays: dict[str, np.ndarray] = {}
    if color_field is not None:
        arrays["line_scalar"] = np.tile(color_field, 6)
    return poly, arrays


def plot_frames(
    frames: np.ndarray,
    centers: np.ndarray,
    *,
    global_scale: float = 1.0,
    scale_field: np.ndarray | None = None,
    color_field: np.ndarray | None = None,
    normal_color_rot: np.ndarray | None = None,
    color_map: str = "hot",
    line_width: float = 1.0,
    plot_cubes: bool = False,
    plotter=None,
    show: bool = False,
) -> FrameRenderResult:
    pv = require_pyvista()

    f = as_real64("frames", frames)
    c = as_real64("centers", centers)
    if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
        raise ValueError("frames must have shape (3, 3, n).")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("centers must have shape (n, 3).")
    n = f.shape[2]
    if c.shape[0] != n:
        raise ValueError("centers rows must equal number of frames.")

    f2 = f.copy()
    if scale_field is not None:
        sf = as_real64("scale_field", scale_field).reshape(-1)
        if sf.shape[0] != n:
            raise ValueError("scale_field must have length n.")
        f2 = f2 * sf.reshape(1, 1, n)

    cf = None
    if color_field is not None:
        cf = as_real64("color_field", color_field).reshape(-1)
        if cf.shape[0] != n:
            raise ValueError("color_field must have length n.")

    rot = None
    if normal_color_rot is not None:
        rot = as_real64("normal_color_rot", normal_color_rot)
        if rot.shape != (3, 3):
            raise ValueError("normal_color_rot must have shape (3, 3).")

    if plotter is None:
        plotter = pv.Plotter()

    if plot_cubes:
        poly, arrays = _build_cube_polydata(
            f2,
            c,
            global_scale=global_scale,
            color_field=cf,
            normal_color_rot=rot,
        )
        for key, value in arrays.items():
            getattr(poly, "cell_data")[key] = value
        if "face_scalar" in arrays:
            actor = plotter.add_mesh(
                poly,
                scalars="face_scalar",
                cmap=color_map,
                show_scalar_bar=False,
                opacity=1.0,
            )
        else:
            actor = plotter.add_mesh(
                poly,
                scalars="face_rgb",
                rgb=True,
                show_scalar_bar=False,
                opacity=1.0,
            )
    else:
        poly, arrays = _build_line_polydata(
            f2,
            c,
            global_scale=global_scale,
            color_field=cf,
        )
        for key, value in arrays.items():
            getattr(poly, "cell_data")[key] = value
        if "line_scalar" in arrays:
            actor = plotter.add_mesh(
                poly,
                scalars="line_scalar",
                cmap=color_map,
                line_width=float(line_width),
                show_scalar_bar=False,
                render_lines_as_tubes=False,
            )
        else:
            actor = plotter.add_mesh(
                poly,
                color="white",
                line_width=float(line_width),
                render_lines_as_tubes=False,
            )

    if show:
        plotter.show()

    return FrameRenderResult(polydata=poly, actor=actor, plotter=plotter)


def PlotFrames(frames: np.ndarray, centers: np.ndarray, **kwargs):
    arg_map = {
        "GlobalScale": "global_scale",
        "ScaleField": "scale_field",
        "ColorField": "color_field",
        "NormalColorRot": "normal_color_rot",
        "ColorMap": "color_map",
        "LineWidth": "line_width",
        "PlotCubes": "plot_cubes",
    }
    py_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
    return plot_frames(frames, centers, **py_kwargs)
