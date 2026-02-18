from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import as_real64


def export_frames(filename: str | Path, frames: np.ndarray) -> None:
    f = as_real64("frames", frames)
    if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
        raise ValueError("frames must have shape (3, 3, n).")
    rows = np.reshape(f, (9, -1), order="F").T
    np.savetxt(str(filename), rows, fmt="%.18e", delimiter=" ")


def ExportFrames(filename: str | Path, frames: np.ndarray) -> None:
    export_frames(filename, frames)
