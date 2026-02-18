from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_EDGE_LOCAL = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ],
    dtype=np.int64,
)

_RX90 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(14.0) / 4.0, 0.0, -np.sqrt(2.0) / 4.0, 0.0],
        [0.0, -3.0 / 4.0, 0.0, np.sqrt(7.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(2.0) / 4.0, 0.0, np.sqrt(14.0) / 4.0, 0.0],
        [0.0, np.sqrt(7.0) / 4.0, 0.0, 3.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, np.sqrt(5.0) / 4.0, 0.0, np.sqrt(35.0) / 8.0],
        [-np.sqrt(14.0) / 4.0, 0.0, -np.sqrt(2.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.sqrt(5.0) / 4.0, 0.0, 1.0 / 2.0, 0.0, -np.sqrt(7.0) / 4.0],
        [np.sqrt(2.0) / 4.0, 0.0, -np.sqrt(14.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.sqrt(35.0) / 8.0, 0.0, -np.sqrt(7.0) / 4.0, 0.0, 1.0 / 8.0],
    ],
    dtype=np.float64,
)

_RX_MINUS90 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, -np.sqrt(14.0) / 4.0, 0.0, np.sqrt(2.0) / 4.0, 0.0],
        [0.0, -3.0 / 4.0, 0.0, np.sqrt(7.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -np.sqrt(2.0) / 4.0, 0.0, -np.sqrt(14.0) / 4.0, 0.0],
        [0.0, np.sqrt(7.0) / 4.0, 0.0, 3.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, np.sqrt(5.0) / 4.0, 0.0, np.sqrt(35.0) / 8.0],
        [np.sqrt(14.0) / 4.0, 0.0, np.sqrt(2.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.sqrt(5.0) / 4.0, 0.0, 1.0 / 2.0, 0.0, -np.sqrt(7.0) / 4.0],
        [-np.sqrt(2.0) / 4.0, 0.0, np.sqrt(14.0) / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.sqrt(35.0) / 8.0, 0.0, -np.sqrt(7.0) / 4.0, 0.0, 1.0 / 8.0],
    ],
    dtype=np.float64,
)

_QUERY_GRAD_SCALE = 1.0 / 8.0
_MAX_PROJ_ITERS = 10000


def _as_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


def _as_int64_tetra(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (nt, 4).")
    return arr


@dataclass(slots=True)
class SphericalHarmonicL4:
    coeff: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.coeff, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 9:
            raise ValueError("SphericalHarmonicL4 expects 9 coefficients.")
        self.coeff = arr.copy()

    def copy(self) -> "SphericalHarmonicL4":
        return SphericalHarmonicL4(self.coeff.copy())

    def mult9(self, M: np.ndarray) -> None:
        self.coeff = np.asarray(M, dtype=np.float64) @ self.coeff

    def rz(self, a: float) -> None:
        c1 = np.cos(a)
        c2 = np.cos(2.0 * a)
        c3 = np.cos(3.0 * a)
        c4 = np.cos(4.0 * a)
        s1 = np.sin(a)
        s2 = np.sin(2.0 * a)
        s3 = np.sin(3.0 * a)
        s4 = np.sin(4.0 * a)
        M = np.array(
            [
                [c4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s4],
                [0.0, c3, 0.0, 0.0, 0.0, 0.0, 0.0, s3, 0.0],
                [0.0, 0.0, c2, 0.0, 0.0, 0.0, s2, 0.0, 0.0],
                [0.0, 0.0, 0.0, c1, 0.0, s1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -s1, 0.0, c1, 0.0, 0.0, 0.0],
                [0.0, 0.0, -s2, 0.0, 0.0, 0.0, c2, 0.0, 0.0],
                [0.0, -s3, 0.0, 0.0, 0.0, 0.0, 0.0, c3, 0.0],
                [-s4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c4],
            ],
            dtype=np.float64,
        )
        self.mult9(M)

    def rx90(self) -> None:
        self.mult9(_RX90)

    def rx_minus90(self) -> None:
        self.mult9(_RX_MINUS90)

    def ry(self, alpha: float) -> None:
        self.rx90()
        self.rz(alpha)
        self.rx_minus90()

    def rx(self, alpha: float) -> None:
        self.ry(-0.5 * np.pi)
        self.rz(alpha)
        self.ry(0.5 * np.pi)

    def rot(self, r: Rotation) -> None:
        """Apply R^{-1} to the SH coefficients (matches C++ Rot convention)."""
        # C++ uses Quaternion::to_Euler followed by Rx->Ry->Rz updates.
        # The equivalent SciPy decomposition is intrinsic xyz on R^{-1}.
        alpha, beta, gamma = r.inv().as_euler("xyz")
        self.rx(float(alpha))
        self.ry(float(beta))
        self.rz(float(gamma))

    def ex(self) -> "SphericalHarmonicL4":
        c = self.coeff
        return SphericalHarmonicL4(
            np.array(
                [
                    -np.sqrt(2.0) * c[7],
                    -np.sqrt(2.0) * c[8] - np.sqrt(3.5) * c[6],
                    -np.sqrt(3.5) * c[7] - np.sqrt(4.5) * c[5],
                    -np.sqrt(4.5) * c[6] - np.sqrt(10.0) * c[4],
                    np.sqrt(10.0) * c[3],
                    np.sqrt(4.5) * c[2],
                    np.sqrt(3.5) * c[1] + np.sqrt(4.5) * c[3],
                    np.sqrt(2.0) * c[0] + np.sqrt(3.5) * c[2],
                    np.sqrt(2.0) * c[1],
                ],
                dtype=np.float64,
            )
        )

    def ey(self) -> "SphericalHarmonicL4":
        c = self.coeff
        return SphericalHarmonicL4(
            np.array(
                [
                    np.sqrt(2.0) * c[1],
                    -np.sqrt(2.0) * c[0] + np.sqrt(3.5) * c[2],
                    -np.sqrt(3.5) * c[1] + np.sqrt(4.5) * c[3],
                    -np.sqrt(4.5) * c[2],
                    -np.sqrt(10.0) * c[5],
                    -np.sqrt(4.5) * c[6] + np.sqrt(10.0) * c[4],
                    -np.sqrt(3.5) * c[7] + np.sqrt(4.5) * c[5],
                    -np.sqrt(2.0) * c[8] + np.sqrt(3.5) * c[6],
                    np.sqrt(2.0) * c[7],
                ],
                dtype=np.float64,
            )
        )

    def ez(self) -> "SphericalHarmonicL4":
        c = self.coeff
        return SphericalHarmonicL4(
            np.array(
                [4.0 * c[8], 3.0 * c[7], 2.0 * c[6], c[5], 0.0, -c[3], -2.0 * c[2], -3.0 * c[1], -4.0 * c[0]],
                dtype=np.float64,
            )
        )

    def dot(self, other: "SphericalHarmonicL4") -> float:
        return float(np.dot(self.coeff, other.coeff))

    def norm(self) -> float:
        return float(np.sqrt(self.dot(self)))

    def normalize(self) -> None:
        n = self.norm()
        if n > 0.0:
            self.coeff /= n

    @staticmethod
    def project_helper(
        query: "SphericalHarmonicL4", grad_threshold: float, dot_threshold: float
    ) -> tuple[Rotation, "SphericalHarmonicL4"]:
        query_n = query.copy()
        query_n.normalize()

        best_dot = -1.0
        W = _INIT_ROT[0]
        v = _INIT_HARMONICS[0].copy()
        for i in range(len(_INIT_HARMONICS)):
            tdot = _INIT_HARMONICS[i].dot(query_n)
            if tdot > best_dot:
                best_dot = tdot
                W = _INIT_ROT[i]
                v = _INIT_HARMONICS[i].copy()

        olddot = best_dot
        for _ in range(_MAX_PROJ_ITERS):
            grad = np.array(
                [query_n.dot(v.ex()), query_n.dot(v.ey()), query_n.dot(v.ez())],
                dtype=np.float64,
            )
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < grad_threshold:
                break

            grad *= _QUERY_GRAD_SCALE
            v.rx(float(grad[0]))
            v.ry(float(grad[1]))
            v.rz(float(grad[2]))

            W = Rotation.from_rotvec([float(grad[0]), 0.0, 0.0]) * W
            W = Rotation.from_rotvec([0.0, float(grad[1]), 0.0]) * W
            W = Rotation.from_rotvec([0.0, 0.0, float(grad[2])]) * W

            dot = v.dot(query_n)
            if dot - olddot < dot_threshold:
                break
            olddot = dot
        else:
            logger.warning("Spherical-harmonic projection reached maximum iterations.")

        return W, v

    @staticmethod
    def project(
        query: "SphericalHarmonicL4", grad_threshold: float, dot_threshold: float
    ) -> Rotation:
        return SphericalHarmonicL4.project_helper(query, grad_threshold, dot_threshold)[0]

    @staticmethod
    def project_sph(
        query: "SphericalHarmonicL4", grad_threshold: float, dot_threshold: float
    ) -> "SphericalHarmonicL4":
        return SphericalHarmonicL4.project_helper(query, grad_threshold, dot_threshold)[1]


_INIT_HARMONICS = (
    SphericalHarmonicL4(
        np.array([0.0, 0.0, 0.0, 0.0, np.sqrt(7.0 / 12.0), 0.0, 0.0, 0.0, np.sqrt(5.0 / 12.0)], dtype=np.float64)
    ),
    SphericalHarmonicL4(np.array([0.0, 0.0, 0.0, 0.0, -0.190941, 0.0, -0.853913, 0.0, 0.484123], dtype=np.float64)),
    SphericalHarmonicL4(np.array([0.0, 0.0, 0.0, 0.0, -0.190941, 0.0, 0.853913, 0.0, 0.484123], dtype=np.float64)),
    SphericalHarmonicL4(np.array([0.0, 0.0, 0.0, 0.0, 0.763763, 0.0, 0.0, 0.0, -0.645497], dtype=np.float64)),
    SphericalHarmonicL4(np.array([0.0, 0.0, -0.853913, 0.0, -0.190941, 0.0, 0.0, 0.0, -0.484123], dtype=np.float64)),
)

_INIT_ROT = (
    Rotation.identity(),
    Rotation.from_rotvec([0.78539816339, 0.0, 0.0]),
    Rotation.from_rotvec([0.0, 0.78539816339, 0.0]),
    Rotation.from_rotvec([0.0, 0.0, 0.78539816339]),
    Rotation.from_rotvec(0.78539816339 * np.array([0.70710678118, 0.70710678118, 0.0])),
)


def project_sph_field(q0: np.ndarray, grad_threshold: float, dot_threshold: float) -> np.ndarray:
    q0m = _as_real64("q0", q0)
    if q0m.ndim != 2 or q0m.shape[0] != 9:
        raise ValueError("q0 must have shape (9, n).")

    out = np.empty_like(q0m)
    for i in range(q0m.shape[1]):
        qi = SphericalHarmonicL4(q0m[:, i])
        proj = SphericalHarmonicL4.project_sph(qi, grad_threshold, dot_threshold)
        out[:, i] = proj.coeff
    return out


def _normal_to_alignment_rotation(normal: np.ndarray) -> Rotation:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm == 0.0:
        return Rotation.identity()

    n_unit = n / nrm
    if abs(float(n_unit[2])) < 0.99:
        axis = np.cross(n_unit, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis_nrm = float(np.linalg.norm(axis))
        if abs(axis_nrm) > 1.0:
            axis = 0.99 * axis / axis_nrm
            axis_nrm = float(np.linalg.norm(axis))
        if axis_nrm > 0.0:
            angle = float(np.arctan2(axis_nrm, n_unit[2]))
            return Rotation.from_rotvec(angle * axis / axis_nrm)
    return Rotation.identity()


def compute_ff(normals: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    normals2 = _as_real64("normals", normals)
    if normals2.ndim != 2 or normals2.shape[1] != 3:
        raise ValueError("normals must have shape (nv, 3).")

    tet = _as_int64_tetra("tetra", tetra)
    nv = normals2.shape[0]
    nt = tet.shape[0]
    if tet.size > 0 and (tet.min() < 0 or tet.max() >= nv):
        raise ValueError("tetra contains out-of-range vertex indices.")

    n_vars = 11 * nv
    row_offset = 0

    # Smoothing term: one equation per (tet edge, coefficient).
    if nt > 0:
        tet_edges = tet[:, _EDGE_LOCAL]
        v0 = tet_edges[:, :, 0].reshape(-1)
        v1 = tet_edges[:, :, 1].reshape(-1)
        n_edges = v0.size
        n_rows_smooth = n_edges * 9

        edge_idx = np.repeat(np.arange(n_edges, dtype=np.int64), 9)
        coeff_idx = np.tile(np.arange(9, dtype=np.int64), n_edges)
        row_smooth = np.arange(n_rows_smooth, dtype=np.int64)

        left_cols = 11 * v0[edge_idx] + coeff_idx
        right_cols = 11 * v1[edge_idx] + coeff_idx

        rows = np.concatenate([row_smooth, row_smooth])
        cols = np.concatenate([left_cols, right_cols])
        data = np.concatenate([np.ones(n_rows_smooth, dtype=np.float64), -np.ones(n_rows_smooth, dtype=np.float64)])
        rhs = np.zeros(n_rows_smooth, dtype=np.float64)
        row_offset = n_rows_smooth
    else:
        rows = np.empty((0,), dtype=np.int64)
        cols = np.empty((0,), dtype=np.int64)
        data = np.empty((0,), dtype=np.float64)
        rhs = np.empty((0,), dtype=np.float64)

    # Boundary penalty term.
    nonzero = np.where(np.linalg.norm(normals2, axis=1) > 0.0)[0]
    if nonzero.size > 0:
        sh0_stack = np.empty((nonzero.size, 9), dtype=np.float64)
        sh4_stack = np.empty((nonzero.size, 9), dtype=np.float64)
        sh8_stack = np.empty((nonzero.size, 9), dtype=np.float64)

        for i, v in enumerate(nonzero):
            rot = _normal_to_alignment_rotation(normals2[v, :])
            sh0 = SphericalHarmonicL4(np.array([np.sqrt(5.0 / 12.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            sh4 = SphericalHarmonicL4(np.array([0.0, 0.0, 0.0, 0.0, np.sqrt(7.0 / 12.0), 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            sh8 = SphericalHarmonicL4(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(5.0 / 12.0)], dtype=np.float64))
            sh4.rot(rot)
            sh0.rot(rot)
            sh8.rot(rot)
            sh0_stack[i, :] = sh0.coeff
            sh4_stack[i, :] = sh4.coeff
            sh8_stack[i, :] = sh8.coeff

        n_rows_bdry = nonzero.size * 9
        row_bdry = row_offset + np.arange(n_rows_bdry, dtype=np.int64)
        v_idx = np.repeat(nonzero, 9)
        b_idx = np.repeat(np.arange(nonzero.size, dtype=np.int64), 9)
        d_idx = np.tile(np.arange(9, dtype=np.int64), nonzero.size)

        sh0_vals = sh0_stack[b_idx, d_idx]
        sh4_vals = sh4_stack[b_idx, d_idx]
        sh8_vals = sh8_stack[b_idx, d_idx]

        rows_b = np.concatenate([row_bdry, row_bdry, row_bdry])
        cols_b = np.concatenate([11 * v_idx + d_idx, 11 * v_idx + 9, 11 * v_idx + 10])
        data_b = 100.0 * np.concatenate([np.ones(n_rows_bdry, dtype=np.float64), sh0_vals, sh8_vals])
        rhs_b = 100.0 * sh4_vals

        rows = np.concatenate([rows, rows_b])
        cols = np.concatenate([cols, cols_b])
        data = np.concatenate([data, data_b])
        rhs = np.concatenate([rhs, rhs_b])
        row_offset += n_rows_bdry

    A = sp.coo_matrix((data, (rows, cols)), shape=(row_offset, n_vars), dtype=np.float64).tocsr()
    x = spla.lsqr(A, rhs, atol=1e-10, btol=1e-10, iter_lim=20000)[0]

    frames = np.empty((3, 3, nv), dtype=np.float64)
    for v in range(nv):
        sh = SphericalHarmonicL4(x[11 * v : 11 * v + 9])
        r = SphericalHarmonicL4.project(sh, 1e-3, 1e-5)
        frames[:, :, v] = r.as_matrix()
    return frames
