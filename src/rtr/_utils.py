from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def as_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


def as_index(name: str, x: np.ndarray, n: int) -> np.ndarray:
    idx = np.asarray(x, dtype=np.int64).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"{name} contains out-of-range indices.")
    return idx


def mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def matvec_rows(A: sp.spmatrix | np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute (A * x^T)^T for row-stacked vectors."""
    return (A @ x.T).T


def matrix_diag_inverse(A: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(A):
        diag = np.asarray(sp.csr_matrix(A).diagonal(), dtype=np.float64).reshape(-1)
    else:
        diag = np.asarray(np.diag(np.asarray(A, dtype=np.float64)), dtype=np.float64).reshape(-1)
    if np.any(diag == 0.0):
        raise ValueError("Matrix diagonal must be nonzero for preconditioning.")
    return 1.0 / diag


def octa_to_odeco(q_octa: np.ndarray) -> np.ndarray:
    q = as_real64("q_octa", q_octa)
    if q.ndim != 2 or q.shape[0] != 9:
        raise ValueError("q_octa must have shape (9, n).")
    n = q.shape[1]
    top = ((6.0 / 5.0) * np.sqrt(np.pi)) * np.ones((1, n), dtype=np.float64)
    middle = np.zeros((5, n), dtype=np.float64)
    bottom = ((8.0 / 5.0) * np.sqrt(np.pi / 21.0)) * q
    return np.vstack([top, middle, bottom])
