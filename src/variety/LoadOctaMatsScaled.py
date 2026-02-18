from __future__ import annotations

import numpy as np

from .LoadOctaMats import load_octa_mats


def load_octa_mats_scaled() -> list[np.ndarray]:
    S = np.diag([np.sqrt(189.0) / 4.0] + [1.0] * 9)
    out = []
    for M in load_octa_mats():
        out.append(S @ M @ S)
    return out


def LoadOctaMatsScaled() -> list[np.ndarray]:
    return load_octa_mats_scaled()

