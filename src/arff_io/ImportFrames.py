from __future__ import annotations

from pathlib import Path

import numpy as np


def import_frames(filename: str | Path) -> np.ndarray:
    data = np.loadtxt(str(filename), dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError("Frame file must contain n rows and 9 columns.")
    return np.reshape(data.T, (3, 3, -1), order="F")


def ImportFrames(filename: str | Path) -> np.ndarray:
    return import_frames(filename)
