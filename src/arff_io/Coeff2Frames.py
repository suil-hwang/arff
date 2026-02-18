from __future__ import annotations

import numpy as np

from .Odeco2Frames import odeco2_frames
from .Octa2Frames import octa2_frames
from .Sph024ToMonomial import sph024_to_monomial
from ._common import as_real64


def coeff2_frames(q: np.ndarray, normalize: bool = False) -> np.ndarray:
    q2 = as_real64("q", q)
    if q2.ndim != 2:
        raise ValueError("q must be a 2D array with shape (d, n).")

    if q2.shape[0] == 9:
        return octa2_frames(q2)

    # Odeco branch 
    frames = odeco2_frames(sph024_to_monomial(q2))
    if normalize:
        norms = np.linalg.norm(frames, axis=0, keepdims=True)
        return np.divide(
            frames,
            norms,
            out=np.zeros_like(frames),
            where=norms > 0.0,
        )
    return frames


def Coeff2Frames(q: np.ndarray, normalize: bool = False) -> np.ndarray:
    return coeff2_frames(q, normalize=normalize)
