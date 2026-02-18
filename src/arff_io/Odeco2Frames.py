from __future__ import annotations

import numpy as np

from .OdecoTensorGradient import odeco_tensor_gradient
from ._common import as_real64, frames_from_row_vectors


def odeco2_frames(
    odeco_coeffs: np.ndarray,
    *,
    v1_init: np.ndarray | None = None,
    v2_init: np.ndarray | None = None,
    tol: float | None = None,
    max_iters: int = 10000,
) -> np.ndarray:
    coeffs = as_real64("odeco_coeffs", odeco_coeffs)
    if coeffs.ndim != 2 or coeffs.shape[0] != 15:
        raise ValueError("odeco_coeffs must have shape (15, n).")
    n = coeffs.shape[1]

    if v1_init is None:
        v1 = np.random.randn(n, 3).astype(np.float64)
    else:
        v1_in = as_real64("v1_init", v1_init)
        if v1_in.shape != (n, 3):
            raise ValueError("v1_init must have shape (n, 3).")
        v1 = v1_in.copy()

    if v2_init is None:
        v2 = np.random.randn(n, 3).astype(np.float64)
    else:
        v2_in = as_real64("v2_init", v2_init)
        if v2_in.shape != (n, 3):
            raise ValueError("v2_init must have shape (n, 3).")
        v2 = v2_in.copy()

    threshold = float((np.finfo(np.float64).eps ** 0.9) * np.sqrt(n)) if tol is None else float(tol)
    delta = np.inf
    it = 0

    v1_old = np.empty((n, 3), dtype=np.float64)
    w1 = np.zeros((n, 3), dtype=np.float64)
    w2 = np.zeros((n, 3), dtype=np.float64)
    w3 = np.empty((n, 3), dtype=np.float64)
    work = np.empty((n, 3), dtype=np.float64)
    row_scalars = np.empty((n, 1), dtype=np.float64)
    while delta > threshold and it < int(max_iters):
        it += 1
        v1_old[:] = v1

        odeco_tensor_gradient(coeffs, v1, out=w1)
        odeco_tensor_gradient(coeffs, v2, out=w2)

        np.multiply(v1, w1, out=work)
        np.sum(work, axis=1, keepdims=True, out=row_scalars)
        np.multiply(row_scalars, w1, out=v1)
        np.multiply(v1, v1, out=work)
        np.sum(work, axis=1, keepdims=True, out=row_scalars)
        np.sqrt(row_scalars, out=row_scalars)
        np.divide(v1, row_scalars, out=v1)

        np.multiply(w2, v1, out=work)
        np.sum(work, axis=1, keepdims=True, out=row_scalars)
        np.multiply(row_scalars, v1, out=work)
        np.subtract(w2, work, out=w2)
        np.multiply(v2, w2, out=work)
        np.sum(work, axis=1, keepdims=True, out=row_scalars)
        np.multiply(row_scalars, w2, out=v2)
        np.multiply(v2, v2, out=work)
        np.sum(work, axis=1, keepdims=True, out=row_scalars)
        np.sqrt(row_scalars, out=row_scalars)
        np.divide(v2, row_scalars, out=v2)

        np.subtract(v1, v1_old, out=work)
        np.multiply(work, work, out=work)
        delta = float(np.sqrt(np.sum(work)))

    work[:, 0] = (v1[:, 1] * v2[:, 2]) - (v1[:, 2] * v2[:, 1])
    work[:, 1] = (v1[:, 2] * v2[:, 0]) - (v1[:, 0] * v2[:, 2])
    work[:, 2] = (v1[:, 0] * v2[:, 1]) - (v1[:, 1] * v2[:, 0])
    odeco_tensor_gradient(coeffs, work, out=w3)
    return 0.25 * frames_from_row_vectors(w1, w2, w3)


def Odeco2Frames(odecoCoeffs: np.ndarray) -> np.ndarray:
    return odeco2_frames(odecoCoeffs)
