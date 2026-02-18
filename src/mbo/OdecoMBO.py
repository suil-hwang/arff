from __future__ import annotations

import numpy as np

from sdp import MultiSdpSolver
from variety import load_odeco_mats_sph, odeco_align_mat, rand_octahedral_field

from ._basis import transpose_pages
from .bundle import MBOFiber


def _build_odeco_sdp_data() -> tuple[np.ndarray, np.ndarray]:
    odeco = np.stack(load_odeco_mats_sph(), axis=2)  # (15, 15, 27)

    first = np.zeros((16, 16), dtype=np.float64)
    first[0, 0] = 1.0

    lifted = np.zeros((16, 16, 27), dtype=np.float64)
    lifted[1:, 1:, :] = odeco

    sdp_pages = np.concatenate([first[:, :, None], lifted], axis=2)  # (16, 16, 28)
    sdp_a = np.reshape(sdp_pages, (16 * 16, 28), order="F").T
    sdp_b = np.concatenate([np.array([1.0], dtype=np.float64), np.zeros(27, dtype=np.float64)])[:, None]
    return sdp_a, sdp_b


def _build_odeco_aligned_sdp_data() -> tuple[np.ndarray, np.ndarray]:
    m1 = np.diag([1.0, 0.0, 0.0, -18.0, -18.0])
    m2 = np.diag([np.sqrt(2.0), -6.0, -6.0], 2)
    m3_left = np.diag(np.array([np.sqrt(2.0), 0.0, -6.0, 0.0], dtype=np.float64), 1)
    m3_right = np.diag(np.array([0.0, 6.0], dtype=np.float64), 3)
    m3 = m3_left + m3_right
    small = np.stack([m1, m2, m3], axis=2)  # (5, 5, 3)

    first = np.zeros((6, 6), dtype=np.float64)
    first[0, 0] = 1.0

    second = np.zeros((6, 6), dtype=np.float64)
    second[1:, 1:] = np.eye(5, dtype=np.float64)

    lifted = np.zeros((6, 6, 3), dtype=np.float64)
    lifted[1:, 1:, :] = small

    pages = np.concatenate([first[:, :, None], second[:, :, None], lifted], axis=2)  # (6, 6, 5)
    sdp_a = np.reshape(pages, (6 * 6, 5), order="F").T
    sdp_b = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)[:, None]
    return sdp_a, sdp_b


def _odeco_z_aligned_basis() -> tuple[np.ndarray, np.ndarray]:
    z_basis = np.array(
        [
            [(4 / 15) * np.sqrt(2 * np.pi), 0, 0, 0, 0],
            [0, (-4 / 7) * np.sqrt((3 / 5) * np.pi), 0, 0, 0],
            [0, 0, 0, 0, 0],
            [(-8 / 21) * np.sqrt((2 / 5) * np.pi), 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, (4 / 7) * np.sqrt((3 / 5) * np.pi), 0, 0],
            [0, 0, 0, (-8 / 3) * np.sqrt((1 / 35) * np.pi), 0],
            [0, 0, 0, 0, 0],
            [0, (4 / 21) * np.sqrt((1 / 5) * np.pi), 0, 0, 0],
            [0, 0, 0, 0, 0],
            [(4 / 105) * np.sqrt(2 * np.pi), 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, (-4 / 21) * np.sqrt((1 / 5) * np.pi), 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, (8 / 3) * np.sqrt((1 / 35) * np.pi)],
        ],
        dtype=np.float64,
    )
    z_fixed = np.array(
        [
            (2 / 5) * np.sqrt(np.pi),
            0,
            0,
            (8 / 7) * np.sqrt((1 / 5) * np.pi),
            0,
            0,
            0,
            0,
            0,
            0,
            (16 / 105) * np.sqrt(np.pi),
            0,
            0,
            0,
            0,
        ],
        dtype=np.float64,
    )
    return z_basis, z_fixed


def _octa_to_odeco(q_octa: np.ndarray) -> np.ndarray:
    q = np.asarray(q_octa, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != 9:
        raise ValueError("Octahedral field must have shape (9, n).")
    n = q.shape[1]
    top = ((6.0 / 5.0) * np.sqrt(np.pi)) * np.ones((1, n), dtype=np.float64)
    middle = np.zeros((5, n), dtype=np.float64)
    bottom = ((8.0 / 5.0) * np.sqrt(np.pi / 21.0)) * q
    return np.vstack([top, middle, bottom])


def odeco_mbo() -> MBOFiber:
    sdp_a, sdp_b = _build_odeco_sdp_data()
    aligned_a, aligned_b = _build_odeco_aligned_sdp_data()
    z_basis, z_fixed = _odeco_z_aligned_basis()

    proj_solver: MultiSdpSolver | None = None
    aligned_solver: MultiSdpSolver | None = None

    def _proj(q0: np.ndarray) -> np.ndarray:
        nonlocal proj_solver
        if proj_solver is None:
            proj_solver = MultiSdpSolver(sdp_a, sdp_b)
        out = proj_solver.project(np.asarray(q0, dtype=np.float64), return_Q=False)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _proj_aligned(q0: np.ndarray) -> np.ndarray:
        nonlocal aligned_solver
        q0m = np.asarray(q0, dtype=np.float64)
        if q0m.ndim != 2 or q0m.shape[0] != 5:
            raise ValueError("Odeco aligned projection expects shape (5, n).")
        if aligned_solver is None:
            aligned_solver = MultiSdpSolver(aligned_a, aligned_b)
        out = aligned_solver.project(q0m, return_Q=False)
        q = out[0] if isinstance(out, tuple) else out
        scale = (315.0 / (64.0 * np.pi)) * np.sum(q0m * q, axis=0, keepdims=True)
        return scale * q

    def _bdry_basis(bdry_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot_obj = odeco_align_mat(np.asarray(bdry_normals, dtype=np.float64), return_align=False)
        rot = rot_obj[0] if isinstance(rot_obj, tuple) else rot_obj
        rot_t = transpose_pages(rot)
        bdry_basis = np.einsum("abn,bc->acn", rot_t, z_basis, optimize=True)
        bdry_fixed = np.einsum("abn,b->an", rot_t, z_fixed, optimize=True)
        return bdry_fixed, bdry_basis

    def _rand(nv: int, bdry_idx: np.ndarray, bdry_normals: np.ndarray) -> np.ndarray:
        return _octa_to_odeco(rand_octahedral_field(nv, bdry_idx, bdry_normals))

    return MBOFiber(
        dim=15,
        proj=_proj,
        proj_aligned=_proj_aligned,
        bdry_basis=_bdry_basis,
        rand=_rand,
    )


def OdecoMBO() -> MBOFiber:
    return odeco_mbo()
