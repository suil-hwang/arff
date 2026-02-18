from __future__ import annotations

from typing import Any

import numpy as np
from pymanopt.manifolds.manifold import Manifold
from scipy.linalg import block_diag

from batchop import batchop
from variety import (
    exp_so3,
    load_odeco_mats_sph,
    load_so3_generators_y2,
    load_so3_generators_y4,
    odeco_align_mat,
    rand_octahedral_field,
)

from ._utils import as_index, as_real64, octa_to_odeco


class OdecoBundleManifold(Manifold):
    """OdecoBundleFactory"""

    def __init__(self, n: int, bdry_idx: np.ndarray, bdry_normals: np.ndarray):
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("n must be positive.")

        self.bdry_idx = as_index("bdry_idx", bdry_idx, self.n)
        self.int_idx = np.setdiff1d(
            np.arange(self.n, dtype=np.int64),
            self.bdry_idx,
            assume_unique=False,
        )
        self.nb = self.bdry_idx.size
        self.ni = self.int_idx.size

        normals = as_real64("bdry_normals", bdry_normals)
        if normals.shape != (self.nb, 3):
            raise ValueError("bdry_normals must have shape (len(bdry_idx), 3).")
        self.bdry_normals = normals

        self._odeco_mats = np.stack(load_odeco_mats_sph(), axis=2)  # (15, 15, 27)
        lx4, ly4, lz4, yz4 = load_so3_generators_y4()
        lx2, ly2, lz2, yz2 = load_so3_generators_y2()
        lx = block_diag(np.zeros((1, 1), dtype=np.float64), lx2, lx4)
        ly = block_diag(np.zeros((1, 1), dtype=np.float64), ly2, ly4)
        lz = block_diag(np.zeros((1, 1), dtype=np.float64), lz2, lz4)

        self._lxyz = np.stack([lx, ly, lz], axis=2)  # (15, 15, 3)
        self._yz2 = yz2
        self._yz4 = yz4
        self._id15 = np.eye(15, dtype=np.float64)

        if self.nb > 0:
            _, bdry_align = odeco_align_mat(self.bdry_normals, return_align=True)
            self._bdry_proj = self._id15[:, :, None] - batchop(
                "mult",
                bdry_align,
                bdry_align,
                "T",
                "N",
            )
        else:
            self._bdry_proj = np.zeros((15, 15, 0), dtype=np.float64)

        # Pre-allocate reusable buffer for pinv (27, 15, n) to avoid
        # repeated allocation in _build_projection_cache.
        self._pinv_buf = np.empty((27, 15, self.n), dtype=np.float64)

        super().__init__(
            name=f"OdecoBundle({self.n})",
            dimension=6 * self.ni + 3 * self.nb,
            point_layout=1,
        )

    def _check_point(self, name: str, q: np.ndarray) -> np.ndarray:
        arr = as_real64(name, q)
        if arr.ndim != 2 or arr.shape != (15, self.n):
            raise ValueError(f"{name} must have shape (15, {self.n}).")
        return arr

    def _mul_lxyz(self, v: np.ndarray) -> np.ndarray:
        v2 = self._check_point("v", v)
        return np.einsum("abk,bn->ank", self._lxyz, v2, optimize=True)

    def mul_o(self, q: np.ndarray) -> np.ndarray:
        q2 = self._check_point("q", q)
        return np.einsum("abk,bn->akn", self._odeco_mats, q2, optimize=True)

    def mulO(self, q: np.ndarray) -> np.ndarray:
        return self.mul_o(q)

    def _build_projection_cache(self, q: np.ndarray) -> dict[str, np.ndarray]:
        oq = self.mul_o(q)  # (15, 27, n)

        # MATLAB uses copy-on-write: NqM.basis = NqM.Oq only copies when
        # boundary pages are modified.  Mirror that by skipping the copy
        # when there are no boundary vertices.
        if self.nb > 0:
            basis = np.array(oq, dtype=np.float64, copy=True)
            basis[:, :, self.bdry_idx] = batchop(
                "mult",
                self._bdry_proj,
                oq[:, :, self.bdry_idx],
            )
        else:
            basis = oq

        # Reuse pre-allocated pinv buffer to avoid (27, 15, n) allocation
        # every call.  The buffer is created once in __init__.
        pinv = self._pinv_buf
        if self.nb > 0:
            pinv[:, :, self.bdry_idx] = batchop("pinv", basis[:, :, self.bdry_idx], 4)
        if self.ni > 0:
            pinv[:, :, self.int_idx] = batchop("pinv", basis[:, :, self.int_idx], 9)

        # Compute proj_tang = Id - Oq * pinv, avoiding an intermediate
        # (15, 15, n) broadcast allocation for Id[:, :, None].
        proj_tang = -batchop("mult", oq, pinv)
        for i in range(15):
            proj_tang[i, i, :] += 1.0
        if self.nb > 0:
            proj_tang[:, :, self.bdry_idx] = batchop(
                "mult",
                self._bdry_proj,
                proj_tang[:, :, self.bdry_idx],
            )

        return {
            "Oq": oq,
            "basis": basis,
            "pinv": pinv,
            "proj_tang": proj_tang,
        }

    def project_with_cache(
        self,
        q: np.ndarray,
        v0: np.ndarray,
        cache: dict[str, np.ndarray] | None = None,
        *,
        return_y_o: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
        q2 = self._check_point("q", q)
        v2 = self._check_point("v0", v0)

        local_cache = self._build_projection_cache(q2) if cache is None else cache

        v3 = batchop("mult", local_cache["proj_tang"], v2[:, None, :])
        v = np.asarray(v3[:, 0, :], dtype=np.float64)

        if not return_y_o:
            return v, local_cache, None

        dual_y3 = batchop("mult", local_cache["pinv"], v2[:, None, :])  # (27, 1, n)
        dual_y = np.asarray(dual_y3[:, 0, :], dtype=np.float64)  # (27, n)
        y_o = np.einsum("abk,kn->abn", self._odeco_mats, dual_y, optimize=True)
        return v, local_cache, y_o

    def projection(self, point, vector):
        projected, _, _ = self.project_with_cache(point, vector, return_y_o=False)
        return projected

    def tangent(self, point: np.ndarray, vector: np.ndarray) -> np.ndarray:
        del point
        return self._check_point("vector", vector)

    def egrad2rgrad(
        self,
        q: np.ndarray,
        egrad: np.ndarray,
        cache: dict[str, np.ndarray] | None = None,
        *,
        return_y_o: bool = False,
    ) -> Any:
        projected, local_cache, y_o = self.project_with_cache(
            q,
            egrad,
            cache,
            return_y_o=return_y_o,
        )
        if return_y_o:
            return projected, local_cache, y_o
        return projected

    def ehess2rhess(
        self,
        q: np.ndarray,
        egrad: np.ndarray,
        ehess: np.ndarray,
        tangent_vector: np.ndarray,
        cache: dict[str, np.ndarray],
        grad_y_o: np.ndarray,
    ) -> np.ndarray:
        del egrad
        q2 = self._check_point("q", q)
        eh = self._check_point("ehess", ehess)
        tv = self._check_point("tangent_vector", tangent_vector)
        yo = as_real64("grad_y_o", grad_y_o)
        if yo.shape != (15, 15, self.n):
            raise ValueError(f"grad_y_o must have shape (15, 15, {self.n}).")

        correction = batchop("mult", yo, tv[:, None, :])[:, 0, :]
        rhess = eh - correction
        projected, _, _ = self.project_with_cache(q2, rhess, cache, return_y_o=False)
        return projected

    def inner_product(
        self,
        point: np.ndarray,
        tangent_vector_a: np.ndarray,
        tangent_vector_b: np.ndarray,
    ) -> float:
        del point
        a = self._check_point("tangent_vector_a", tangent_vector_a)
        b = self._check_point("tangent_vector_b", tangent_vector_b)
        return float(np.sum(a * b))

    def norm(self, point, tangent_vector):
        del point
        tv = self._check_point("tangent_vector", tangent_vector)
        return float(np.linalg.norm(tv))

    def random_point(self):
        q_octa = rand_octahedral_field(self.n, self.bdry_idx, self.bdry_normals)
        return octa_to_odeco(q_octa)

    def random_tangent_vector(self, point):
        q = self._check_point("point", point)
        rnd = np.random.randn(15, self.n).astype(np.float64)
        s = self.projection(q, rnd)
        norm_s = np.linalg.norm(s)
        if norm_s == 0.0:
            return s
        return s / norm_s

    def zero_vector(self, point):
        del point
        return np.zeros((15, self.n), dtype=np.float64)

    def _retract(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        q0 = self._check_point("point", point)
        v = self._check_point("tangent_vector", tangent_vector)
        tv = v if t is None else float(t) * v

        rot_basis = np.transpose(self._mul_lxyz(q0), (0, 2, 1))  # (15, 3, n)
        rot_part3 = batchop("leastsq", rot_basis, tv[:, None, :])  # (3, 1, n)
        linear_part = tv - batchop("mult", rot_basis, rot_part3)[:, 0, :]
        rot_part = rot_part3[:, 0, :]  # (3, n)

        q = q0 + linear_part
        axis_angles = rot_part.T  # (n, 3)
        q = np.array(q, dtype=np.float64, copy=True)
        q[1:6, :] = exp_so3(axis_angles, q[1:6, :], self._yz2)
        q[6:, :] = exp_so3(axis_angles, q[6:, :], self._yz4)
        return q

    def retraction(self, point, tangent_vector):
        return self._retract(point, tangent_vector)

    def to_tangent_space(self, point, vector):
        del point
        return self._check_point("vector", vector)

    # MATLAB-style aliases
    def retr(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        return self._retract(point, tangent_vector, t=t)

    def rand(self) -> np.ndarray:
        return self.random_point()

    def randvec(self, point: np.ndarray) -> np.ndarray:
        return self.random_tangent_vector(point)

    def zerovec(self, point: np.ndarray) -> np.ndarray:
        return self.zero_vector(point)


def odeco_bundle_factory(
    n: int,
    bdry_idx: np.ndarray,
    bdry_normals: np.ndarray,
    gpuflag: bool = False,
) -> OdecoBundleManifold:
    del gpuflag
    return OdecoBundleManifold(n, bdry_idx, bdry_normals)


def OdecoBundleFactory(
    n: int,
    bdryIdx: np.ndarray,
    bdryNormals: np.ndarray,
    gpuflag: bool = False,
) -> OdecoBundleManifold:
    return odeco_bundle_factory(n, bdryIdx, bdryNormals, gpuflag=gpuflag)
