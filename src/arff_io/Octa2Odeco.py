from __future__ import annotations

import numpy as np

from ._common import as_real64


def octa2_odeco(q: np.ndarray) -> np.ndarray:
    q2 = as_real64("q", q)
    if q2.ndim != 2 or q2.shape[0] != 9:
        raise ValueError("q must have shape (9, n).")
    n = q2.shape[1]
    out = np.zeros((15, n), dtype=np.float64)
    out[0, :] = ((6.0 / 5.0) * np.sqrt(np.pi))
    out[6:, :] = ((8.0 / 5.0) * np.sqrt(np.pi / 21.0)) * q2
    return out


def Octa2Odeco(q: np.ndarray) -> np.ndarray:
    return octa2_odeco(q)
