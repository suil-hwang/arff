from __future__ import annotations

import numpy as np

from ._core import project_sph_field


def ray_projection(
    q0: np.ndarray,
    grad_threshold: float = 1e-8,
    dot_threshold: float = 1e-8,
) -> np.ndarray:
    return project_sph_field(q0, grad_threshold=grad_threshold, dot_threshold=dot_threshold)


def RayProjection(
    q0: np.ndarray,
    grad_threshold: float = 1e-8,
    dot_threshold: float = 1e-8,
) -> np.ndarray:
    return ray_projection(q0, grad_threshold=grad_threshold, dot_threshold=dot_threshold)

