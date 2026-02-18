from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._common import as_real64


@lru_cache(maxsize=1)
def _sph024_to_monomial_matrix() -> np.ndarray:
    p = np.pi
    a = np.zeros((15, 15), dtype=np.float64)

    s1 = np.sqrt(1.0 / p)
    s5 = np.sqrt(5.0 / p)
    s10 = np.sqrt(10.0 / p)
    s15 = np.sqrt(15.0 / p)
    s5o2 = np.sqrt((5.0 / 2.0) / p)
    s35 = np.sqrt(35.0 / p)
    s35o2 = np.sqrt((35.0 / 2.0) / p)

    a[0, :] = np.array(
        [12 * s1, 0, 4 * s1, 0, 12 * s1, 0, 0, 0, 0, 4 * s1, 0, 4 * s1, 0, 0, 12 * s1],
        dtype=np.float64,
    )
    a[1, :] = np.array([0, 0, 0, 0, 0, 0, s15, 0, 3 * s15, 0, 0, 0, 0, 3 * s15, 0], dtype=np.float64)
    a[2, :] = np.array([0, 0, 0, 0, 0, 3 * s15, 0, s15, 0, 0, 0, 0, 3 * s15, 0, 0], dtype=np.float64)
    a[3, :] = np.array(
        [12 * s5, 0, s5, 0, -6 * s5, 0, 0, 0, 0, s5, 0, -2 * s5, 0, 0, -6 * s5],
        dtype=np.float64,
    )
    a[4, :] = np.array([0, 3 * s15, 0, 3 * s15, 0, 0, 0, 0, 0, 0, s15, 0, 0, 0, 0], dtype=np.float64)
    a[5, :] = np.array([0, 0, s15, 0, 6 * s15, 0, 0, 0, 0, -s15, 0, 0, 0, 0, -6 * s15], dtype=np.float64)
    a[6, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, (9.0 / 2.0) * s35, 0, 0, 0, 0, (-9.0 / 2.0) * s35, 0], dtype=np.float64)
    a[7, :] = np.array([0, 0, 0, 0, 0, 0, 0, (9.0 / 2.0) * s35o2, 0, 0, 0, 0, (-9.0 / 2.0) * s35o2, 0, 0], dtype=np.float64)
    a[8, :] = np.array([0, 0, 0, 0, 0, 0, 9 * s5, 0, (-9.0 / 2.0) * s5, 0, 0, 0, 0, (-9.0 / 2.0) * s5, 0], dtype=np.float64)
    a[9, :] = np.array([0, 0, 0, 0, 0, 9 * s10, 0, (-9.0 / 2.0) * s5o2, 0, 0, 0, 0, (-27.0 / 2.0) * s5o2, 0, 0], dtype=np.float64)
    a[10, :] = np.array([36 * s1, 0, -18 * s1, 0, (27.0 / 2.0) * s1, 0, 0, 0, 0, -18 * s1, 0, (9.0 / 2.0) * s1, 0, 0, (27.0 / 2.0) * s1], dtype=np.float64)
    a[11, :] = np.array([0, 9 * s10, 0, (-27.0 / 2.0) * s5o2, 0, 0, 0, 0, 0, 0, (-9.0 / 2.0) * s5o2, 0, 0, 0, 0], dtype=np.float64)
    a[12, :] = np.array([0, 0, 9 * s5, 0, -9 * s5, 0, 0, 0, 0, -9 * s5, 0, 0, 0, 0, 9 * s5], dtype=np.float64)
    a[13, :] = np.array([0, 0, 0, (9.0 / 2.0) * s35o2, 0, 0, 0, 0, 0, 0, (-9.0 / 2.0) * s35o2, 0, 0, 0, 0], dtype=np.float64)
    a[14, :] = np.array([0, 0, 0, 0, (9.0 / 2.0) * s35, 0, 0, 0, 0, 0, 0, (-9.0 / 2.0) * s35, 0, 0, (9.0 / 2.0) * s35], dtype=np.float64)

    return a.T


def sph024_to_monomial(q: np.ndarray) -> np.ndarray:
    q2 = as_real64("q", q)
    if q2.ndim != 2 or q2.shape[0] != 15:
        raise ValueError("q must have shape (15, n).")
    m = _sph024_to_monomial_matrix()
    return m @ q2


def Sph024ToMonomial(q: np.ndarray) -> np.ndarray:
    return sph024_to_monomial(q)
