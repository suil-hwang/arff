from __future__ import annotations

import numpy as np

from mbo.bundle import MBOFiber
from variety import octa_align_mat, rand_octahedral_field

from .RayProjection import ray_projection


def ray_mbo() -> MBOFiber:
    scale = np.sqrt(5.0 / 12.0)
    basis_const = np.zeros((9, 2), dtype=np.float64)
    basis_const[0, 0] = 1.0
    basis_const[8, 1] = 1.0
    fixed_const = np.zeros((9,), dtype=np.float64)
    fixed_const[4] = np.sqrt(7.0 / 12.0)

    def _proj(q: np.ndarray) -> np.ndarray:
        q2 = np.asarray(q, dtype=np.float64)
        if q2.ndim != 2 or q2.shape[0] != 9:
            raise ValueError("q must have shape (9, n).")
        return ray_projection(q2)

    def _proj_aligned(q: np.ndarray) -> np.ndarray:
        q2 = np.asarray(q, dtype=np.float64)
        if q2.ndim != 2 or q2.shape[0] != 2:
            raise ValueError("Ray aligned projection expects shape (2, n).")
        out = np.zeros_like(q2)
        norms = np.linalg.norm(q2, axis=0, keepdims=True)
        nz = norms > 0.0
        out[:, nz[0]] = q2[:, nz[0]] / norms[:, nz[0]]
        return scale * out

    def _bdry_basis(bdry_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot = octa_align_mat(np.asarray(bdry_normals, dtype=np.float64), return_align=False)
        rot_t = np.swapaxes(rot, 0, 1)
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


def RayMBO() -> MBOFiber:
    return ray_mbo()

