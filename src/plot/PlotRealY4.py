from __future__ import annotations

import numpy as np

from .RealY4Basis import real_y4_basis


def plot_real_y4(
    real_coeffs: np.ndarray,
    shift: float = 1.0,
    scale: float = 1.0,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_ticks: int = 100,
    *,
    ax=None,
    cmap: str = "viridis",
):
    import matplotlib.pyplot as plt

    c = np.asarray(center, dtype=np.float64).reshape(3)
    u, v = np.meshgrid(
        np.linspace(0.0, np.pi, int(n_ticks)),
        np.linspace(0.0, 2.0 * np.pi, int(n_ticks)),
        indexing="ij",
    )
    r_data = real_y4_basis(real_coeffs, u, v)
    r_scaled = scale * r_data + shift

    x = c[0] + r_scaled * (np.sin(u) * np.cos(v))
    y = c[1] + r_scaled * (np.sin(u) * np.sin(v))
    z = c[2] + r_scaled * np.cos(u)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    surf = getattr(ax, "plot_surface")(
        x,
        y,
        z,
        facecolors=plt.get_cmap(cmap)(
            (r_data - np.nanmin(r_data))
            / max(np.nanmax(r_data) - np.nanmin(r_data), np.finfo(np.float64).eps)
        ),
        linewidth=0.0,
        antialiased=True,
    )
    getattr(ax, "set_box_aspect")((1.0, 1.0, 1.0))
    ax.set_axis_off()
    return ax, surf


def PlotRealY4(
    realCoeffs: np.ndarray,
    shift: float = 1.0,
    scale: float = 1.0,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    nTicks: int = 100,
):
    return plot_real_y4(
        real_coeffs=realCoeffs,
        shift=shift,
        scale=scale,
        center=center,
        n_ticks=nTicks,
    )
