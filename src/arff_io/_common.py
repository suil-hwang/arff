from __future__ import annotations

import numpy as np


def as_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


def normalize_rows(v: np.ndarray) -> np.ndarray:
    v2 = np.asarray(v, dtype=np.float64)
    if v2.ndim != 2:
        raise ValueError("Expected a 2D array.")
    nrm = np.linalg.norm(v2, axis=1, keepdims=True)
    out = np.zeros_like(v2)
    nz = nrm[:, 0] > 0.0
    out[nz, :] = v2[nz, :] / nrm[nz, :]
    return out


def frames_from_row_vectors(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """Build 3x3xn frames from n x 3 row-vector triplets."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    c = np.asarray(v3, dtype=np.float64)
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("v1, v2, v3 must all have shape (n, 3).")
    return np.stack([a.T, b.T, c.T], axis=1)
