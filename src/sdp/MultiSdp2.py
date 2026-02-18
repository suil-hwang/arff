from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from batchop import batchop


def _as_real64_2d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return arr


def _as_real64_3d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array.")
    return arr


def _as_real64_column(name: str, x: np.ndarray, rows: int) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    if arr.ndim == 1:
        if arr.shape[0] != rows:
            raise ValueError("Dimensions do not match.")
        return arr[:, None]
    if arr.ndim == 2:
        if arr.shape != (rows, 1):
            raise ValueError("b must have shape (m,) or (m, 1).")
        return arr
    raise ValueError("b must have shape (m,) or (m, 1).")


def _multitransp(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, 0, 1)


def _multisym(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + _multitransp(x))


def _scale_pages(scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x * scale[None, None, :]


class MultiSdp2Solver:
    """Python port of MATLAB ``MultiSdp2`` (primal-dual predictor-corrector)."""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        *,
        eps_primal: float = 1.0e-7,
        eps_gap: float = 1.0e-7,
    ):
        Aarr = np.asarray(A)
        if Aarr.ndim == 3:
            A3 = _as_real64_3d("A", Aarr)
            d = A3.shape[0]
            if d != A3.shape[1]:
                raise ValueError("A pages must be square.")
            k = A3.shape[2]
        elif Aarr.ndim == 2:
            A2 = _as_real64_2d("A", Aarr)
            k, cols = A2.shape
            d = int(round(np.sqrt(cols)))
            if d * d != cols:
                raise ValueError("A must have d^2 columns when 2D.")
            A3 = np.reshape(A2.T, (d, d, k), order="F")
        else:
            raise ValueError("A must be either 2D or 3D.")

        bcol = _as_real64_column("b", b, k)

        if eps_primal <= 0.0 or eps_gap <= 0.0:
            raise ValueError("eps_primal and eps_gap must be positive.")

        self.d = d
        self.k = k
        self.eps_primal = float(eps_primal)
        self.eps_gap = float(eps_gap)

        self._id = np.eye(self.d, dtype=np.float64)
        self._a_pages, self._b = self._rescale_problem(A3, bcol[:, 0])
        self._at = np.reshape(self._a_pages, (self.d * self.d, self.k), order="F")
        self._aflat = self._at.T
        self._aoi, self._ioa = self._build_kron_operators()

    @staticmethod
    def _rescale_problem(A3: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Per-page Frobenius rescaling, matching MATLAB MultiSdp2.m
        norms = np.sqrt(np.sum(np.sum(A3 * A3, axis=0), axis=0))
        if np.any(norms <= 0.0):
            raise ValueError("Each constraint page in A must have nonzero Frobenius norm.")
        A_scaled = A3 / norms[None, None, :]
        b_scaled = b / norms
        return A_scaled, b_scaled

    def _build_c(self, q0: np.ndarray) -> np.ndarray:
        n = q0.shape[2]
        d_minus_1 = self.d - 1
        if q0.shape[0] != d_minus_1:
            raise ValueError("Dimensions do not match.")

        top_left = -2.0 * np.sum(q0 * q0, axis=0, keepdims=True)  # (1,1,n)
        top_right = _multitransp(q0)  # (1,d-1,n)
        bottom_left = q0  # (d-1,1,n)
        bottom_right = -np.repeat(np.eye(d_minus_1, dtype=np.float64)[:, :, None], n, axis=2)

        top = np.concatenate([top_left, top_right], axis=1)
        bot = np.concatenate([bottom_left, bottom_right], axis=1)
        C = np.concatenate([top, bot], axis=0)

        c_norm = np.sqrt(np.sum(np.sum(C * C, axis=0), axis=0))
        if np.any(c_norm <= 0.0):
            raise RuntimeError("Encountered zero objective page norm in MultiSdp2.")
        C = C / c_norm[None, None, :]
        return C

    def _build_kron_operators(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        # Match MATLAB's sparse kron construction to avoid dense memory blowups.
        id_sparse = sparse.eye(self.d, format="csr", dtype=np.float64)

        aoi_blocks: list[sparse.csr_matrix] = []
        ioa_blocks: list[sparse.csr_matrix] = []
        for j in range(self.k):
            a_page = sparse.csr_matrix(self._a_pages[:, :, j])
            aoi_blocks.append(sparse.kron(a_page.T, id_sparse, format="csr"))
            ioa_blocks.append(
                sparse.kron(id_sparse, a_page, format="csr").transpose().tocsr(),
            )

        aoi = sparse.vstack(aoi_blocks, format="csr")
        ioa = sparse.vstack(ioa_blocks, format="csr")
        return aoi, ioa

    def _form_system_matrix(self, z_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        # MATLAB-equivalent construction:
        # ZinvA = reshape(AoI * reshape(Zinv, d^2, n), d^2, k, n)
        # AX    = reshape(IoA * reshape(X,    d^2, n), d^2, k, n)
        # G     = batchop('mult', AX, ZinvA, 'T', 'N')
        n = x.shape[2]
        d2 = self.d * self.d
        zinv_prod = self._aoi @ np.reshape(z_inv, (d2, n), order="F")
        zinv_a = np.reshape(
            np.asarray(zinv_prod),
            (d2, self.k, n),
            order="F",
        )
        ax_prod = self._ioa @ np.reshape(x, (d2, n), order="F")
        ax = np.reshape(
            np.asarray(ax_prod),
            (d2, self.k, n),
            order="F",
        )
        return batchop("mult", ax, zinv_a, "T", "N")

    def close(self) -> None:
        return None

    def project(
        self,
        q0: np.ndarray,
        maxiter: int,
        pred_corr: int | bool = False,
        *,
        matlab_indexing: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MATLAB-compatible API.

        Returns
        -------
        tuple
            ``(X_final, Z_final, unconvergedIdx, gap, feasPrimal)``.
            ``unconvergedIdx`` is 0-based by default.
            Set ``matlab_indexing=True`` to get 1-based indices.
        """
        if not isinstance(maxiter, (int, np.integer)):
            raise ValueError("maxiter must be an integer.")
        maxiter_i = int(maxiter)
        if maxiter_i <= 0:
            raise ValueError("maxiter must be positive.")

        q0m = _as_real64_2d("q0", q0)
        if q0m.shape[0] != self.d - 1:
            raise ValueError("Dimensions do not match.")

        pred_corr_steps = 1 if not bool(pred_corr) else 2

        n = q0m.shape[1]
        q0p = q0m[:, None, :]
        C = self._build_c(q0p)

        X = np.repeat(self._id[:, :, None], n, axis=2)
        y = np.zeros((self.k, n), dtype=np.float64)
        Aty = np.reshape(self._at @ y, (self.d, self.d, n), order="F")
        Z = Aty - C

        X_final = np.zeros((self.d, self.d, n), dtype=np.float64)
        Z_final = np.zeros((self.d, self.d, n), dtype=np.float64)
        unconverged_idx = np.arange(n, dtype=np.int64)

        gap = np.empty((0, 1), dtype=np.float64)
        feas_primal = np.empty((0, 1), dtype=np.float64)
        G: np.ndarray | None = None

        for _ in range(maxiter_i):
            x_vec = np.reshape(X, (self.d * self.d, n), order="F")
            feas_vec = np.linalg.norm(self._aflat @ x_vec - self._b[:, None], axis=0)
            gap_vec = np.sum(self._b[:, None] * y, axis=0) - np.einsum("ijn,ijn->n", C, X)
            feasible_mask = feas_vec < self.eps_primal
            converged_mask = feasible_mask & (np.abs(gap_vec) < self.eps_gap)

            if np.any(converged_mask):
                target_idx = unconverged_idx[converged_mask]
                X_final[:, :, target_idx] = X[:, :, converged_mask]
                Z_final[:, :, target_idx] = Z[:, :, converged_mask]

                keep_mask = ~converged_mask
                n_unconverged = int(np.sum(keep_mask))
                if n_unconverged == 0:
                    return (
                        X_final,
                        Z_final,
                        np.empty((0,), dtype=np.int64),
                        np.empty((0, 1), dtype=np.float64),
                        np.empty((0, 1), dtype=np.float64),
                    )

                unconverged_idx = unconverged_idx[keep_mask]
                X = X[:, :, keep_mask]
                Z = Z[:, :, keep_mask]
                C = C[:, :, keep_mask]
                if G is not None:
                    G = G[:, :, keep_mask]
                y = y[:, keep_mask]
                feas_vec = feas_vec[keep_mask]
                gap_vec = gap_vec[keep_mask]
                feasible_mask = feasible_mask[keep_mask]
                n = n_unconverged

            Aty = np.reshape(self._at @ y, (self.d, self.d, n), order="F")
            F_dual = Z + C - Aty
            F_dual_X = batchop("mult", F_dual, X)

            ZX = batchop("mult", Z, X)
            ZdotX = np.einsum("ijn,ijn->n", Z, X)
            LX = batchop("chol", X)
            LZ = batchop("chol", Z)
            Zinv = batchop("cholsolve", LZ, np.repeat(self._id[:, :, None], n, axis=2))

            G = self._form_system_matrix(Zinv, X)
            LG = batchop("chol", G)

            sigma = np.full((n,), 0.5, dtype=np.float64)
            sigma[feasible_mask] = 0.0
            Dy = np.zeros((self.k, n), dtype=np.float64)
            DZ = np.zeros((self.d, self.d, n), dtype=np.float64)
            DX = np.zeros((self.d, self.d, n), dtype=np.float64)
            a_primal = np.ones((n,), dtype=np.float64)
            a_dual = np.ones((n,), dtype=np.float64)
            for substep in range(pred_corr_steps):
                mu = sigma * (ZdotX / self.d)

                F_ZX = ZX - _scale_pages(mu, self._id[:, :, None])
                if substep == 1:
                    F_ZX = F_ZX + batchop("mult", DZ, DX)

                rhs = batchop("mult", Zinv, F_dual_X - F_ZX) + X
                rhs = self._aflat @ np.reshape(rhs, (self.d * self.d, n), order="F")
                Dy_rhs = -self._b[:, None] + rhs

                Dy = batchop("cholsolve", LG, Dy_rhs[:, None, :])[:, 0, :]
                AtDy = np.reshape(self._at @ Dy, (self.d, self.d, n), order="F")
                DZ = -F_dual + AtDy
                DX = _multisym(batchop("mult", Zinv, -F_ZX - batchop("mult", DZ, X)))

                LDXL = batchop("cholcong", LX, DX)
                LDZL = batchop("cholcong", LZ, DZ)
                diag_LDXL = self._id[:, :, None] * LDXL
                diag_LDZL = self._id[:, :, None] * LDZL

                ub_DX = np.max(np.sum(-diag_LDXL + np.abs(-LDXL + diag_LDXL), axis=1), axis=0)
                ub_DZ = np.max(np.sum(-diag_LDZL + np.abs(-LDZL + diag_LDZL), axis=1), axis=0)
                ub_DX = np.minimum(ub_DX, np.sqrt(np.sum(np.sum(LDXL * LDXL, axis=0), axis=0)))
                ub_DZ = np.minimum(ub_DZ, np.sqrt(np.sum(np.sum(LDZL * LDZL, axis=0), axis=0)))

                inv_ub_dx = np.divide(
                    0.98,
                    ub_DX,
                    out=np.full_like(ub_DX, np.inf),
                    where=ub_DX > 0.0,
                )
                inv_ub_dz = np.divide(
                    0.98,
                    ub_DZ,
                    out=np.full_like(ub_DZ, np.inf),
                    where=ub_DZ > 0.0,
                )
                a_primal = np.minimum(inv_ub_dx, 1.0)
                a_dual = np.minimum(inv_ub_dz, 1.0)

                if substep == 0:
                    zx_next = np.einsum(
                        "ijn,ijn->n",
                        Z + _scale_pages(a_dual, DZ),
                        X + _scale_pages(a_primal, DX),
                    )
                    sigma = np.power(zx_next / ZdotX, 3)
                    sigma = np.clip(sigma, 0.2, 1.0)

            y = y + a_dual[None, :] * Dy
            Z = Z + _scale_pages(a_dual, DZ)
            X = X + _scale_pages(a_primal, DX)

            gap = gap_vec[:, None]
            feas_primal = feas_vec[:, None]

        unconverged_out = (
            unconverged_idx + 1 if matlab_indexing else unconverged_idx.copy()
        )
        return X_final, Z_final, unconverged_out, gap, feas_primal


def multi_sdp2(
    A: np.ndarray,
    b: np.ndarray,
    *,
    eps_primal: float = 1.0e-7,
    eps_gap: float = 1.0e-7,
) -> MultiSdp2Solver:
    """Factory matching MATLAB ``MultiSdp2(A, b)``."""
    return MultiSdp2Solver(A, b, eps_primal=eps_primal, eps_gap=eps_gap)


MultiSdp2 = multi_sdp2
