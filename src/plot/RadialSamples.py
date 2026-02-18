from __future__ import annotations

import numpy as np


def radial_samples(base: int, R: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    if base <= 0:
        raise ValueError("base must be positive.")
    if R <= 0:
        raise ValueError("R must be positive.")

    r = np.repeat(np.arange(1, R + 1, dtype=np.float64), base * np.arange(1, R + 1))
    step = np.exp((2j * np.pi) / (base * r))
    z = np.exp(1j * phase) * r * np.cumprod(step)
    points = np.column_stack([np.real(z), np.imag(z), np.zeros_like(r)])
    angles = np.angle(z)
    return points.astype(np.float64), angles.astype(np.float64)


def RadialSamples(base: int, R: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    return radial_samples(base, R, phase)

