from __future__ import annotations

import numpy as np
from pymanopt.manifolds.manifold import Manifold

from variety import exp_so3, load_so3_generators_y4, rand_octahedral_field

from ._utils import as_index, as_real64


class OctahedralBundleManifold(Manifold):
    """OctahedralBundleFactory"""

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

        normals = as_real64("bdry_normals", bdry_normals)
        if normals.shape != (self.bdry_idx.size, 3):
            raise ValueError("bdry_normals must have shape (len(bdry_idx), 3).")
        self.bdry_normals = normals

        lx, ly, lz, yz = load_so3_generators_y4()
        self._lxyz = np.stack([lx, ly, lz], axis=2)  # (9, 9, 3)
        self._lxyz_t = -self._lxyz
        self._yz = yz

        super().__init__(
            name=f"OctahedralBundle({self.n})",
            dimension=3 * self.n,
            point_layout=1,
        )

    @property
    def typical_dist(self) -> float:
        return float((np.pi / 2.0) * np.sqrt(3.0 * self.n))

    def _check_point(self, name: str, q: np.ndarray) -> np.ndarray:
        arr = as_real64(name, q)
        if arr.ndim != 2 or arr.shape != (9, self.n):
            raise ValueError(f"{name} must have shape (9, {self.n}).")
        return arr

    def _check_tangent(self, name: str, s: np.ndarray) -> np.ndarray:
        arr = as_real64(name, s)
        if arr.ndim != 2 or arr.shape != (self.n, 3):
            raise ValueError(f"{name} must have shape ({self.n}, 3).")
        return arr

    def _apply_boundary_tangent_constraint(self, s: np.ndarray) -> np.ndarray:
        # MATLAB modifies s in-place; Python must copy to avoid aliasing.
        # Skip the copy entirely when there are no boundary vertices.
        if self.bdry_idx.size == 0:
            return s
        out = np.array(s, dtype=np.float64, copy=True)
        bdry = out[self.bdry_idx, :]
        scales = np.sum(bdry * self.bdry_normals, axis=1, keepdims=True)
        out[self.bdry_idx, :] = scales * self.bdry_normals
        return out

    def mul_lxyz(self, v: np.ndarray) -> np.ndarray:
        v2 = self._check_point("v", v)
        return np.einsum("abk,bn->ank", self._lxyz, v2, optimize=True)

    def mul_lxyz_t(self, v: np.ndarray) -> np.ndarray:
        v2 = self._check_point("v", v)
        return np.einsum("abk,bn->ank", self._lxyz_t, v2, optimize=True)

    # MATLAB aliases
    def mulLxyz(self, v: np.ndarray) -> np.ndarray:
        return self.mul_lxyz(v)

    def mulLxyzT(self, v: np.ndarray) -> np.ndarray:
        return self.mul_lxyz_t(v)

    def tangentbasis(self, q: np.ndarray) -> np.ndarray:
        return self.mul_lxyz(q)

    def projection(
        self,
        point,
        vector,
    ):
        return self.proj(point, vector)

    def proj(
        self,
        point: np.ndarray,
        vector: np.ndarray,
        lxyzq: np.ndarray | None = None,
    ) -> np.ndarray:
        q2 = self._check_point("point", point)
        v = self._check_point("vector", vector)
        basis = self.tangentbasis(q2) if lxyzq is None else as_real64("lxyzq", lxyzq)
        if basis.shape != (9, self.n, 3):
            raise ValueError(f"lxyzq must have shape (9, {self.n}, 3).")
        s = np.sum(v[:, :, None] * basis, axis=0)
        return self._apply_boundary_tangent_constraint(s)

    def tangentialize(self, point: np.ndarray, vector: np.ndarray) -> np.ndarray:
        del point
        s = self._check_tangent("vector", vector)
        return self._apply_boundary_tangent_constraint(s)

    def tangent2ambient(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray,
        lxyzq: np.ndarray | None = None,
    ) -> np.ndarray:
        q2 = self._check_point("point", point)
        s = self._check_tangent("tangent_vector", tangent_vector)
        basis = (
            self.tangentbasis(q2)
            if lxyzq is None
            else as_real64("lxyzq", lxyzq)
        )
        if basis.shape != (9, self.n, 3):
            raise ValueError(f"lxyzq must have shape (9, {self.n}, 3).")
        return np.einsum("ank,nk->an", basis, s, optimize=True)

    def egrad2rgrad(
        self,
        q: np.ndarray,
        egrad: np.ndarray,
        lxyzq: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.proj(q, egrad, lxyzq)

    def ehess2rhess(
        self,
        q: np.ndarray,
        egrad: np.ndarray,
        ehess: np.ndarray,
        tangent_vector: np.ndarray,
        s_ambient: np.ndarray | None = None,
        rgrad: np.ndarray | None = None,
        lxyz_t_egrad: np.ndarray | None = None,
        lxyzq: np.ndarray | None = None,
    ) -> np.ndarray:
        q2 = self._check_point("q", q)
        egrad2 = self._check_point("egrad", egrad)
        ehess2 = self._check_point("ehess", ehess)
        s = self._check_tangent("tangent_vector", tangent_vector)

        basis = self.tangentbasis(q2) if lxyzq is None else as_real64("lxyzq", lxyzq)
        if basis.shape != (9, self.n, 3):
            raise ValueError(f"lxyzq must have shape (9, {self.n}, 3).")

        s_amb = (
            self.tangent2ambient(q2, s, basis)
            if s_ambient is None
            else self._check_point("s_ambient", s_ambient)
        )
        rg = (
            self.egrad2rgrad(q2, egrad2, basis)
            if rgrad is None
            else self._check_tangent("rgrad", rgrad)
        )
        lte = (
            self.mul_lxyz_t(egrad2)
            if lxyz_t_egrad is None
            else as_real64("lxyz_t_egrad", lxyz_t_egrad)
        )
        if lte.shape != (9, self.n, 3):
            raise ValueError(f"lxyz_t_egrad must have shape (9, {self.n}, 3).")

        rhess = self.proj(q2, ehess2, basis)

        interior_term = 0.5 * np.cross(rg, s) + np.sum(s_amb[:, :, None] * lte, axis=0)
        if self.int_idx.size > 0:
            rhess[self.int_idx, :] += interior_term[self.int_idx, :]

        if self.bdry_idx.size > 0:
            bdry_lte = lte[:, self.bdry_idx, :]
            bdry_basis = np.einsum("nk,ank->an", self.bdry_normals, bdry_lte, optimize=True)
            scalars = np.sum(s_amb[:, self.bdry_idx] * bdry_basis, axis=0, keepdims=True).T
            rhess[self.bdry_idx, :] += scalars * self.bdry_normals

        return rhess

    def inner_product(
        self,
        point: np.ndarray,
        tangent_vector_a: np.ndarray,
        tangent_vector_b: np.ndarray,
    ) -> float:
        del point
        a = self._check_tangent("tangent_vector_a", tangent_vector_a)
        b = self._check_tangent("tangent_vector_b", tangent_vector_b)
        return float(np.sum(a * b))

    def norm(self, point, tangent_vector):
        del point
        s = self._check_tangent("tangent_vector", tangent_vector)
        return float(np.linalg.norm(s))

    def random_point(self):
        return np.sqrt(3.0 / 20.0) * rand_octahedral_field(
            self.n,
            self.bdry_idx,
            self.bdry_normals,
        )

    def random_tangent_vector(self, point):
        del point
        s = np.random.randn(self.n, 3).astype(np.float64)
        if self.bdry_idx.size > 0:
            coeff = np.random.randn(self.bdry_idx.size, 1).astype(np.float64)
            s[self.bdry_idx, :] = coeff * self.bdry_normals
        norm_s = np.linalg.norm(s)
        if norm_s == 0.0:
            return s
        return s / norm_s

    def zero_vector(self, point):
        del point
        return np.zeros((self.n, 3), dtype=np.float64)

    def _retract(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        q0 = self._check_point("point", point)
        s = self._check_tangent("tangent_vector", tangent_vector)
        ts = s if t is None else float(t) * s
        return exp_so3(ts, q0, self._yz)

    def retraction(self, point, tangent_vector):
        return self._retract(point, tangent_vector)

    def exp(self, point, tangent_vector):
        return self._retract(point, tangent_vector)

    def to_tangent_space(self, point, vector):
        return self.tangentialize(point, vector)

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


def octahedral_bundle_factory(
    n: int,
    bdry_idx: np.ndarray,
    bdry_normals: np.ndarray,
    gpuflag: bool = False,
) -> OctahedralBundleManifold:
    del gpuflag
    return OctahedralBundleManifold(n, bdry_idx, bdry_normals)


def OctahedralBundleFactory(
    n: int,
    bdryIdx: np.ndarray,
    bdryNormals: np.ndarray,
    gpuflag: bool = False,
) -> OctahedralBundleManifold:
    return octahedral_bundle_factory(n, bdryIdx, bdryNormals, gpuflag=gpuflag)
