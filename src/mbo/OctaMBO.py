from __future__ import annotations

import numpy as np

from sdp import MultiSdpSolver
from variety import load_octa_mats_scaled, octa_align_mat, rand_octahedral_field

from ._basis import transpose_pages
from .bundle import MBOFiber


def _build_octa_sdp_data() -> tuple[np.ndarray, np.ndarray]:
    octa_mats = np.stack(load_octa_mats_scaled(), axis=2)  # (10, 10, 15)
    first = np.zeros((10, 10), dtype=np.float64)
    first[0, 0] = 1.0
    sdp_pages = np.concatenate([first[:, :, None], octa_mats], axis=2)  # (10, 10, 16)
    sdp_a = np.reshape(sdp_pages, (10 * 10, 16), order="F").T
    sdp_b = np.concatenate([np.array([1.0], dtype=np.float64), np.zeros(15, dtype=np.float64)])[:, None]
    return sdp_a, sdp_b


def octa_mbo() -> MBOFiber:
    sdp_a, sdp_b = _build_octa_sdp_data()
    solver: MultiSdpSolver | None = None

    def _proj(q: np.ndarray) -> np.ndarray:
        nonlocal solver
        if solver is None:
            solver = MultiSdpSolver(sdp_a, sdp_b)
        out = solver.project(np.asarray(q, dtype=np.float64), return_Q=False)
        if isinstance(out, tuple):
            return out[0]
        return out

    scale = np.sqrt(5.0 / 12.0)

    def _proj_aligned(q: np.ndarray) -> np.ndarray:
        q2 = np.asarray(q, dtype=np.float64)
        if q2.ndim != 2 or q2.shape[0] != 2:
            raise ValueError("Octa aligned projection expects shape (2, n).")
        nrm = np.linalg.norm(q2, axis=0, keepdims=True)
        out = np.zeros_like(q2)
        nz = nrm > 0.0
        out[:, nz[0]] = q2[:, nz[0]] / nrm[:, nz[0]]
        return scale * out

    basis_const = np.zeros((9, 2), dtype=np.float64)
    basis_const[0, 0] = 1.0
    basis_const[8, 1] = 1.0
    fixed_const = np.zeros((9,), dtype=np.float64)
    fixed_const[4] = np.sqrt(7.0 / 12.0)

    def _bdry_basis(bdry_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot_obj = octa_align_mat(np.asarray(bdry_normals, dtype=np.float64), return_align=False)
        rot = rot_obj[0] if isinstance(rot_obj, tuple) else rot_obj
        rot_t = transpose_pages(rot)
        bdry_basis = np.einsum("abn,bc->acn", rot_t, basis_const, optimize=True)
        bdry_fixed = np.einsum("abn,b->an", rot_t, fixed_const, optimize=True)
        return bdry_fixed, bdry_basis

    def _rand(nv: int, bdry_idx: np.ndarray, bdry_normals: np.ndarray) -> np.ndarray:
        return rand_octahedral_field(nv, bdry_idx, bdry_normals)

    return MBOFiber(
        dim=9,
        proj=_proj,
        proj_aligned=_proj_aligned,
        bdry_basis=_bdry_basis,
        rand=_rand,
    )


def OctaMBO() -> MBOFiber:
    return octa_mbo()
