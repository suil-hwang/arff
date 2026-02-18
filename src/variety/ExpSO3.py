from __future__ import annotations

import numpy as np


def _cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.arctan2(y, x)
    hxy = np.hypot(x, y)
    el = np.arctan2(z, hxy)
    r = np.sqrt(x * x + y * y + z * z)
    return az, el, r


def exp_so3(
    axis_angles: np.ndarray,
    q: np.ndarray,
    YZ: np.ndarray,
    rotate_north_only: bool = False,
) -> np.ndarray:
    axis_angles = np.asarray(axis_angles, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    YZ = np.asarray(YZ, dtype=np.float64)

    if axis_angles.ndim != 2 or axis_angles.shape[1] != 3:
        raise ValueError("axis_angles must have shape (n, 3).")
    if q.ndim != 2:
        raise ValueError("q must have shape (d, n).")
    if YZ.ndim != 2 or YZ.shape[0] != YZ.shape[1]:
        raise ValueError("YZ must be square.")
    if q.shape[0] != YZ.shape[0]:
        raise ValueError("q and YZ dimension mismatch.")
    if q.shape[1] != axis_angles.shape[0]:
        raise ValueError("axis_angles rows must equal q columns.")

    band_idx = YZ.shape[0] // 2
    bands = np.arange(-band_idx, band_idx + 1, dtype=np.float64)[:, None]

    az, el, rot = _cart2sph(axis_angles[:, 0], axis_angles[:, 1], axis_angles[:, 2])
    preserved_idx = rot == 0.0
    q_preserved = q[:, preserved_idx].copy()
    el = np.pi / 2.0 - el

    az_angs = bands * az[None, :]
    el_angs = bands * el[None, :]
    rot_angs = bands * rot[None, :]
    cos_az = np.cos(az_angs)
    sin_az = np.sin(az_angs)
    cos_el = np.cos(el_angs)
    sin_el = np.sin(el_angs)
    cos_rot = np.cos(rot_angs)
    sin_rot = np.sin(rot_angs)

    q_flip = np.flipud(q)
    q = cos_az * q + sin_az * q_flip
    q = YZ @ q

    q_flip = np.flipud(q)
    q = cos_el * q + sin_el * q_flip
    q = YZ.T @ q

    if not rotate_north_only:
        q_flip = np.flipud(q)
        q = cos_rot * q - sin_rot * q_flip

        q = YZ @ q
        q_flip = np.flipud(q)
        q = cos_el * q - sin_el * q_flip
        q = YZ.T @ q

        q_flip = np.flipud(q)
        q = cos_az * q - sin_az * q_flip
        q[:, preserved_idx] = q_preserved

    return q


def ExpSO3(
    axisAngles: np.ndarray,
    q: np.ndarray,
    YZ: np.ndarray,
    rotateNorthOnly: bool = False,
) -> np.ndarray:
    return exp_so3(
        axis_angles=axisAngles,
        q=q,
        YZ=YZ,
        rotate_north_only=rotateNorthOnly,
    )

