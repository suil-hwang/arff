from __future__ import annotations

import numpy as np


def transpose_pages(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("Expected a 3D page-stacked array.")
    return np.swapaxes(arr, 0, 1)


def apply_basis(basis: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply per-page basis matrices.

    basis: (D, d, n)
    coeffs: (d, n)
    returns: (D, n)
    """
    B = np.asarray(basis, dtype=np.float64)
    X = np.asarray(coeffs, dtype=np.float64)
    if B.ndim != 3 or X.ndim != 2:
        raise ValueError("Invalid basis application dimensions.")
    D, d, n = B.shape
    if X.shape != (d, n):
        raise ValueError("Invalid basis coefficient dimensions.")
    if n == 0:
        return np.zeros((D, 0), dtype=np.float64)
    return np.einsum("adn,dn->an", B, X, optimize=True)


def apply_basis_t(basis: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply per-page transposed basis matrices.

    basis: (D, d, n)
    vectors: (D, n)
    returns: (d, n)
    """
    B = np.asarray(basis, dtype=np.float64)
    V = np.asarray(vectors, dtype=np.float64)
    if B.ndim != 3 or V.ndim != 2:
        raise ValueError("Invalid transposed basis application dimensions.")
    D, d, n = B.shape
    if V.shape != (D, n):
        raise ValueError("Invalid transposed basis vector dimensions.")
    if n == 0:
        return np.zeros((d, 0), dtype=np.float64)
    return np.einsum("adn,an->dn", B, V, optimize=True)

