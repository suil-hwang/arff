from __future__ import annotations

import numpy as np

from .OctaTensorGradient import octa_tensor_gradient
from ._common import as_real64, frames_from_row_vectors


def octa2_frames(
    q: np.ndarray,
    *,
    v1_init: np.ndarray | None = None,
    v2_init: np.ndarray | None = None,
    tol: float | None = None,
    max_iters: int = 10000,
) -> np.ndarray:
    q2 = as_real64("q", q)
    if q2.ndim != 2 or q2.shape[0] != 9:
        raise ValueError("q must have shape (9, n).")
    n = q2.shape[1]

    nrm = np.linalg.norm(q2, axis=0, keepdims=True)
    qn = np.zeros_like(q2)
    nz = nrm > 0.0
    qn[:, nz[0]] = q2[:, nz[0]] / nrm[:, nz[0]]
    qa = np.vstack([((np.sqrt(189.0) / 4.0) * np.ones((1, n), dtype=np.float64)), qn])

    if v1_init is None:
        v1 = np.random.randn(n, 3).astype(np.float64)
    else:
        v1 = as_real64("v1_init", v1_init)
        if v1.shape != (n, 3):
            raise ValueError("v1_init must have shape (n, 3).")

    if v2_init is None:
        v2 = np.random.randn(n, 3).astype(np.float64)
    else:
        v2 = as_real64("v2_init", v2_init)
        if v2.shape != (n, 3):
            raise ValueError("v2_init must have shape (n, 3).")

    threshold = float((np.finfo(np.float64).eps ** 0.9) * np.sqrt(n)) if tol is None else float(tol)
    delta = np.inf
    it = 0
    while delta > threshold and it < int(max_iters):
        it += 1
        x = v1[:, 0].copy()
        y = v1[:, 1].copy()
        z = v1[:, 2].copy()

        w1 = octa_tensor_gradient(qa, x, y, z)
        w2 = octa_tensor_gradient(qa, v2[:, 0], v2[:, 1], v2[:, 2])

        v1 = w1 / np.linalg.norm(w1, axis=1, keepdims=True)
        w2_orth = w2 - np.sum(w2 * v1, axis=1, keepdims=True) * v1
        v2 = w2_orth / np.linalg.norm(w2_orth, axis=1, keepdims=True)
        delta = float(np.linalg.norm(v1 - np.column_stack([x, y, z]), ord="fro"))

    v3 = np.cross(v1, v2, axis=1)
    return frames_from_row_vectors(v1, v2, v3)


def Octa2Frames(q: np.ndarray) -> np.ndarray:
    return octa2_frames(q)
