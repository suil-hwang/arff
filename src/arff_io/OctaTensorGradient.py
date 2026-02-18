from __future__ import annotations

import numpy as np

from ._common import as_real64


def octa_tensor_gradient(q: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    q2 = as_real64("q", q)
    x2 = as_real64("x", x).reshape(-1)
    y2 = as_real64("y", y).reshape(-1)
    z2 = as_real64("z", z).reshape(-1)

    n = x2.size
    if y2.size != n or z2.size != n:
        raise ValueError("x, y, z must have the same length.")
    if q2.ndim != 2 or q2.shape != (10, n):
        raise ValueError("q must have shape (10, n).")

    qt = q2.T
    p = np.pi

    dx = np.column_stack(
        [
            2.0 * p ** (-0.5) * x2 * (x2**2 + y2**2 + z2**2),
            (-3.0 / 4.0) * np.sqrt(35.0 / p) * y2 * ((-3.0) * x2**2 + y2**2),
            (9.0 / 2.0) * np.sqrt((35.0 / 2.0) / p) * x2 * y2 * z2,
            (-3.0 / 4.0) * np.sqrt(5.0 / p) * y2 * (3.0 * x2**2 + y2**2 + (-6.0) * z2**2),
            (-9.0 / 2.0) * np.sqrt((5.0 / 2.0) / p) * x2 * y2 * z2,
            (9.0 / 4.0) * p ** (-0.5) * x2 * (x2**2 + y2**2 + (-4.0) * z2**2),
            (3.0 / 4.0) * np.sqrt((5.0 / 2.0) / p) * z2 * ((-9.0) * x2**2 + (-3.0) * y2**2 + 4.0 * z2**2),
            (-3.0 / 2.0) * np.sqrt(5.0 / p) * x2 * (x2**2 + (-3.0) * z2**2),
            (9.0 / 4.0) * np.sqrt((35.0 / 2.0) / p) * (x2**2 + (-1.0) * y2**2) * z2,
            (3.0 / 4.0) * np.sqrt(35.0 / p) * (x2**3 + (-3.0) * x2 * y2**2),
        ]
    )
    dy = np.column_stack(
        [
            2.0 * p ** (-0.5) * y2 * (x2**2 + y2**2 + z2**2),
            (3.0 / 4.0) * np.sqrt(35.0 / p) * x2 * (x2**2 + (-3.0) * y2**2),
            (9.0 / 4.0) * np.sqrt((35.0 / 2.0) / p) * (x2**2 + (-1.0) * y2**2) * z2,
            (-3.0 / 4.0) * np.sqrt(5.0 / p) * x2 * (x2**2 + 3.0 * y2**2 + (-6.0) * z2**2),
            (3.0 / 4.0) * np.sqrt((5.0 / 2.0) / p) * z2 * ((-3.0) * x2**2 + (-9.0) * y2**2 + 4.0 * z2**2),
            (9.0 / 4.0) * p ** (-0.5) * y2 * (x2**2 + y2**2 + (-4.0) * z2**2),
            (-9.0 / 2.0) * np.sqrt((5.0 / 2.0) / p) * x2 * y2 * z2,
            (3.0 / 2.0) * np.sqrt(5.0 / p) * y2 * (y2**2 + (-3.0) * z2**2),
            (-9.0 / 2.0) * np.sqrt((35.0 / 2.0) / p) * x2 * y2 * z2,
            (-3.0 / 4.0) * np.sqrt(35.0 / p) * (3.0 * x2**2 * y2 + (-1.0) * y2**3),
        ]
    )
    dz = np.column_stack(
        [
            2.0 * p ** (-0.5) * z2 * (x2**2 + y2**2 + z2**2),
            np.zeros((n,), dtype=np.float64),
            (-3.0 / 4.0) * np.sqrt((35.0 / 2.0) / p) * y2 * ((-3.0) * x2**2 + y2**2),
            9.0 * np.sqrt(5.0 / p) * x2 * y2 * z2,
            (-9.0 / 4.0) * np.sqrt((5.0 / 2.0) / p) * y2 * (x2**2 + y2**2 + (-4.0) * z2**2),
            p ** (-0.5) * ((-9.0) * x2**2 * z2 + (-9.0) * y2**2 * z2 + 6.0 * z2**3),
            (-9.0 / 4.0) * np.sqrt((5.0 / 2.0) / p) * x2 * (x2**2 + y2**2 + (-4.0) * z2**2),
            (9.0 / 2.0) * np.sqrt(5.0 / p) * (x2**2 + (-1.0) * y2**2) * z2,
            (3.0 / 4.0) * np.sqrt((35.0 / 2.0) / p) * x2 * (x2**2 + (-3.0) * y2**2),
            np.zeros((n,), dtype=np.float64),
        ]
    )

    grad_x = np.sum(qt * dx, axis=1)
    grad_y = np.sum(qt * dy, axis=1)
    grad_z = np.sum(qt * dz, axis=1)
    return np.column_stack([grad_x, grad_y, grad_z])


def OctaTensorGradient(q: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return octa_tensor_gradient(q, x, y, z)
