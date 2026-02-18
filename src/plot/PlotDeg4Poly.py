from __future__ import annotations

from math import factorial

import numpy as np

from variety import monomial_degrees

from ._common import as_real64


def plot_deg4_poly(
    coeffs: np.ndarray,
    shift: float = 1.0,
    scale: float = 1.0,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_ticks: int = 100,
    *,
    ax=None,
    cmap: str = "viridis",
):
    import matplotlib.pyplot as plt

    c = as_real64("coeffs", coeffs).reshape(-1)
    degree = monomial_degrees(4)  # (3, 15)
    if c.shape[0] != degree.shape[1]:
        raise ValueError("coeffs must have length 15 for degree-4 homogeneous polynomial.")

    denom = np.array(
        [factorial(int(degree[0, i])) * factorial(int(degree[1, i])) * factorial(int(degree[2, i])) for i in range(degree.shape[1])],
        dtype=np.float64,
    )
    scaled_coeffs = (1.0 / denom) * c
    ctr = np.asarray(center, dtype=np.float64).reshape(3)

    u, v = np.meshgrid(
        np.linspace(0.0, np.pi, int(n_ticks)),
        np.linspace(0.0, 2.0 * np.pi, int(n_ticks)),
        indexing="ij",
    )
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    shape = x.shape

    mon = (
        (x.reshape(-1, 1) ** degree[0][None, :])
        * (y.reshape(-1, 1) ** degree[1][None, :])
        * (z.reshape(-1, 1) ** degree[2][None, :])
    )
    r = (mon @ scaled_coeffs).reshape(shape)

    rs = shift + scale * r
    x2 = ctr[0] + x * rs
    y2 = ctr[1] + y * rs
    z2 = ctr[2] + z * rs

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    surf = getattr(ax, "plot_surface")(
        x2,
        y2,
        z2,
        facecolors=plt.get_cmap(cmap)(
            (r - np.nanmin(r))
            / max(np.nanmax(r) - np.nanmin(r), np.finfo(np.float64).eps)
        ),
        linewidth=0.0,
        antialiased=True,
    )
    getattr(ax, "set_box_aspect")((1.0, 1.0, 1.0))
    ax.set_axis_off()
    return ax, surf


def PlotDeg4Poly(
    coeffs: np.ndarray,
    shift: float = 1.0,
    scale: float = 1.0,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    nTicks: int = 100,
):
    return plot_deg4_poly(
        coeffs=coeffs,
        shift=shift,
        scale=scale,
        center=center,
        n_ticks=nTicks,
    )
