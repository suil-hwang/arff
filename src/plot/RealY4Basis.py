from __future__ import annotations

import numpy as np

from ._common import as_real64


def real_y4_basis(real_coeffs: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    coeffs = as_real64("real_coeffs", real_coeffs).reshape(-1)
    if coeffs.shape[0] != 9:
        raise ValueError("real_coeffs must have length 9.")

    theta2 = np.asarray(theta, dtype=np.float64)
    phi2 = np.asarray(phi, dtype=np.float64)
    if theta2.shape != phi2.shape:
        raise ValueError("theta and phi must have the same shape.")

    dims = theta2.shape
    theta1 = theta2.reshape(-1)
    phi1 = phi2.reshape(-1)

    basis_scale = np.diag(
        np.array(
            [
                (3.0 / 16.0) * np.sqrt(35.0 / np.pi),
                (3.0 / 4.0) * np.sqrt(35.0 / (2.0 * np.pi)),
                (3.0 / 16.0) * np.sqrt(5.0 / np.pi),
                (3.0 / 8.0) * np.sqrt(5.0 / (2.0 * np.pi)),
                3.0 / (128.0 * np.sqrt(np.pi)),
                (3.0 / 8.0) * np.sqrt(5.0 / (2.0 * np.pi)),
                (3.0 / 16.0) * np.sqrt(5.0 / np.pi),
                (3.0 / 4.0) * np.sqrt(35.0 / (2.0 * np.pi)),
                (3.0 / 16.0) * np.sqrt(35.0 / np.pi),
            ],
            dtype=np.float64,
        )
    )

    phi_part = np.column_stack(
        [
            np.sin(4.0 * phi1),
            np.sin(3.0 * phi1),
            np.sin(2.0 * phi1),
            np.sin(phi1),
            np.ones_like(phi1),
            np.cos(phi1),
            np.cos(2.0 * phi1),
            np.cos(3.0 * phi1),
            np.cos(4.0 * phi1),
        ]
    )

    s = np.sin(theta1)
    c = np.cos(theta1)
    c2 = np.cos(2.0 * theta1)
    c4 = np.cos(4.0 * theta1)
    theta_part = np.column_stack(
        [
            s**4,
            (s**3) * c,
            (s**2) * (5.0 + 7.0 * c2),
            s * ((1.0 + 7.0 * c2) * c),
            9.0 + 20.0 * c2 + 35.0 * c4,
            s * ((1.0 + 7.0 * c2) * c),
            (s**2) * (5.0 + 7.0 * c2),
            (s**3) * c,
            s**4,
        ]
    )

    y = (phi_part * theta_part) @ (basis_scale @ coeffs)
    return y.reshape(dims)


def RealY4Basis(realCoeffs: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return real_y4_basis(realCoeffs, theta, phi)

