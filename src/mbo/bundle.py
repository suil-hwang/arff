from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class MBOFiber:
    dim: int
    proj: Callable[[np.ndarray], np.ndarray]
    proj_aligned: Callable[[np.ndarray], np.ndarray]
    bdry_basis: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    rand: Callable[[int, np.ndarray, np.ndarray], np.ndarray]

    # MATLAB-style aliases (field names used in .m code)
    @property
    def projAligned(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.proj_aligned

    @property
    def bdryBasis(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        return self.bdry_basis

