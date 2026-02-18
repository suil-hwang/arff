from __future__ import annotations

from math import pi
import numpy as np

from arff_io import Octa2Frames
from mbo import OctaMBO
from plot import PlotRealY4
from ray import RayProjection


def _plot_frames_with_opposites(q_column: np.ndarray) -> np.ndarray:
    frame = np.asarray(Octa2Frames(q_column[:, None]), dtype=np.float64)
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return 2.0 * np.hstack([frame, -frame])


def projection_comparison(sample_count: int = 100000):
    q0 = np.random.randn(9, int(sample_count))
    q0 = q0 / np.linalg.norm(q0, axis=0, keepdims=True)

    q_ray = RayProjection(q0)
    q_ours = OctaMBO().proj(q0)

    d_ours = np.linalg.norm(q_ours - q0, axis=0)
    d_ray = np.linalg.norm(q_ray - q0, axis=0)
    idx = np.argsort(d_ray - d_ours)[::-1]

    try:
        import matplotlib.pyplot as plt

        max_show = min(5, idx.size)
        for k in range(max_show):
            j = int(idx[k])
            f_ours = _plot_frames_with_opposites(q_ours[:, j])
            f_ray = _plot_frames_with_opposites(q_ray[:, j])

            ax, _ = PlotRealY4(
                q0[:, j],
                shift=np.sqrt(189.0 / pi) / 8.0,
                scale=1.0,
                center=(0.0, 0.0, 0.0),
                nTicks=500,
            )

            ax.quiver(
                np.zeros(6),
                np.zeros(6),
                np.zeros(6),
                f_ours[0],
                f_ours[1],
                f_ours[2],
                color="b",
                arrow_length_ratio=0.0,
                linewidth=3,
                length=1.0,
            )
            ax.quiver(
                np.zeros(6),
                np.zeros(6),
                np.zeros(6),
                f_ray[0],
                f_ray[1],
                f_ray[2],
                color="r",
                arrow_length_ratio=0.0,
                linewidth=3,
                length=1.0,
            )
            ax.set_axis_off()
            ax.set_box_aspect((1.0, 1.0, 1.0))

            ax.figure.canvas.draw_idle()
            ax.figure.set_facecolor("white")

        plt.close("all")
    except Exception:
        pass

    return {
        "q0": q0,
        "qRay": q_ray,
        "qOurs": q_ours,
        "idx": idx,
        "dRay": d_ray,
        "dOurs": d_ours,
    }


ProjectionComparison = projection_comparison

__all__ = ["projection_comparison", "ProjectionComparison"]
