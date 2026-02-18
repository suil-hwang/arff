from __future__ import annotations

import numpy as np

from ._common import as_real64


def odeco_tensor_gradient(
    odeco_coeffs: np.ndarray,
    v: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    coeffs = as_real64("odeco_coeffs", odeco_coeffs)
    v2 = as_real64("v", v)
    if coeffs.ndim != 2 or coeffs.shape[0] != 15:
        raise ValueError("odeco_coeffs must have shape (15, n).")
    if v2.ndim != 2 or v2.shape[1] != 3:
        raise ValueError("v must have shape (n, 3).")
    n = v2.shape[0]
    if coeffs.shape[1] != n:
        raise ValueError("odeco_coeffs and v have inconsistent sample counts.")
    if out is None:
        grad = np.empty((n, 3), dtype=np.float64)
    else:
        grad = as_real64("out", out)
        if grad.shape != (n, 3):
            raise ValueError("out must have shape (n, 3).")

    x = v2[:, 0]
    y = v2[:, 1]
    z = v2[:, 2]

    t0 = (1.0 / 6.0) * z**3
    t1 = 0.5 * x * z**2
    t2 = 0.5 * x**2 * z
    t3 = (1.0 / 6.0) * x**3
    t4 = 0.5 * y * z**2
    t5 = x * y * z
    t6 = 0.5 * x**2 * y
    t7 = 0.5 * y**2 * z
    t8 = 0.5 * x * y**2
    t9 = (1.0 / 6.0) * y**3

    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = coeffs

    grad[:, 0] = (
        c1 * t0
        + c2 * t1
        + c3 * t2
        + c4 * t3
        + c6 * t4
        + c7 * t5
        + c8 * t6
        + c10 * t7
        + c11 * t8
        + c13 * t9
    )
    grad[:, 1] = (
        c5 * t0
        + c6 * t1
        + c7 * t2
        + c8 * t3
        + c9 * t4
        + c10 * t5
        + c11 * t6
        + c12 * t7
        + c13 * t8
        + c14 * t9
    )
    grad[:, 2] = (
        c0 * t0
        + c1 * t1
        + c2 * t2
        + c3 * t3
        + c5 * t4
        + c6 * t5
        + c7 * t6
        + c9 * t7
        + c10 * t8
        + c12 * t9
    )
    return grad


def OdecoTensorGradient(odecoCoeffs: np.ndarray, v: np.ndarray) -> np.ndarray:
    return odeco_tensor_gradient(odecoCoeffs, v)
