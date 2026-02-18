from __future__ import annotations

import numpy as np

from .OctaAlignMat import octa_align_mat
from .OctaZAligned import octa_z_aligned


def rand_octahedral_field(
    n: int,
    align_idx: np.ndarray | list[int] | tuple[int, ...] | None,
    align_axes: np.ndarray | list[list[float]] | None,
    gpuflag: bool = False,
) -> np.ndarray:
    del gpuflag
    q = octa_z_aligned(2.0 * np.pi * np.random.rand(n))

    axes = np.random.randn(n, 3)
    axes_norm = np.linalg.norm(axes, axis=1)
    axes = axes / np.where(axes_norm > 0.0, axes_norm, 1.0)[:, None]

    if align_idx is not None and align_axes is not None:
        idx = np.asarray(align_idx, dtype=np.int64).reshape(-1)
        if idx.size > 0:
            axes[idx, :] = np.asarray(align_axes, dtype=np.float64)

    D = octa_align_mat(axes)
    return np.einsum("ban,bn->an", D, q)


def RandOctahedralField(
    n: int,
    alignIdx: np.ndarray | list[int] | tuple[int, ...] | None,
    alignAxes: np.ndarray | list[list[float]] | None,
    gpuflag: bool = False,
) -> np.ndarray:
    return rand_octahedral_field(n, alignIdx, alignAxes, gpuflag=gpuflag)

