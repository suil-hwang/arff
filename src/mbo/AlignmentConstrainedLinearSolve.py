from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._basis import apply_basis, apply_basis_t


def _to_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


def _to_index(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{name} must contain nonnegative indices.")
    return arr


def alignment_constrained_linear_solve(
    A: sp.spmatrix | np.ndarray,
    rhs: np.ndarray,
    int_idx: np.ndarray,
    bdry_idx: np.ndarray,
    bdry_basis: np.ndarray,
    q_fixed: np.ndarray,
    warmstart: np.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Python port of AlignmentConstrainedLinearSolve.m."""
    if not (sp.issparse(A) or isinstance(A, np.ndarray)):
        raise ValueError("A must be a dense ndarray or a scipy sparse matrix.")

    rhs2 = _to_real64("rhs", rhs)
    q_fixed2 = _to_real64("q_fixed", q_fixed)
    basis = _to_real64("bdry_basis", bdry_basis)
    int_idx2 = _to_index("int_idx", int_idx)
    bdry_idx2 = _to_index("bdry_idx", bdry_idx)

    if rhs2.ndim != 2 or q_fixed2.ndim != 2:
        raise ValueError("rhs and q_fixed must be 2D arrays.")
    if basis.ndim != 3:
        raise ValueError("bdry_basis must have shape (D, d, nb).")

    D, nv = rhs2.shape
    if q_fixed2.shape != (D, nv):
        raise ValueError("q_fixed dimensions do not match rhs.")
    if basis.shape[0] != D:
        raise ValueError("bdry_basis leading dimension must match rhs.")
    if basis.shape[2] != bdry_idx2.size:
        raise ValueError("bdry_basis page count must match number of boundary indices.")

    if A.shape != (nv, nv):
        raise ValueError("A dimensions do not match rhs.")
    if int_idx2.size + bdry_idx2.size > nv:
        raise ValueError("Index sets exceed vertex count.")

    d = basis.shape[1]
    nb = bdry_idx2.size
    ni = int_idx2.size

    def _apply_A(q_full: np.ndarray) -> np.ndarray:
        # (A * q').'
        return (A @ q_full.T).T

    rhs_shifted = rhs2 - _apply_A(q_fixed2)
    rhs_bdry = apply_basis_t(basis, rhs_shifted[:, bdry_idx2]) if nb > 0 else np.zeros((d, 0), dtype=np.float64)
    rhs_red = np.concatenate(
        [
            rhs_bdry.reshape(-1, order="F"),
            rhs_shifted[:, int_idx2].reshape(-1, order="F"),
        ]
    )

    n_unknown = d * nb + D * ni
    if rhs_red.shape[0] != n_unknown:
        raise RuntimeError("Internal reduced-system dimension mismatch.")

    # Pre-allocate q_full buffer outside CG loop (matches MATLAB's persistent
    # qFull in nested function scope).  Fortran order makes .T C-contiguous
    # so scipy's sparse multiply avoids an internal ravel copy each call.
    _q_full = np.zeros((D, nv), dtype=np.float64, order="F")

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != n_unknown:
            raise ValueError("Reduced vector has invalid size.")

        _q_full[:] = 0.0
        if nb > 0:
            _q_full[:, bdry_idx2] = apply_basis(basis, x[: d * nb].reshape((d, nb), order="F"))
        if ni > 0:
            _q_full[:, int_idx2] = x[d * nb :].reshape((D, ni), order="F")

        Aq_full = _apply_A(_q_full)
        Aq_bdry = apply_basis_t(basis, Aq_full[:, bdry_idx2]) if nb > 0 else np.zeros((d, 0), dtype=np.float64)
        return np.concatenate(
            [
                Aq_bdry.reshape(-1, order="F"),
                Aq_full[:, int_idx2].reshape(-1, order="F"),
            ]
        )

    class _ReducedOperator(spla.LinearOperator):
        def __init__(self) -> None:
            super().__init__(dtype=np.float64, shape=(n_unknown, n_unknown))

        def _matvec(self, x: np.ndarray) -> np.ndarray:
            return matvec(x)

    linop = _ReducedOperator()

    x0 = None
    if warmstart is not None:
        ws = np.asarray(warmstart, dtype=np.float64).reshape(-1)
        if ws.shape[0] == n_unknown:
            x0 = ws

    it_counter = {"n": 0}

    def _cb(_: np.ndarray) -> None:
        it_counter["n"] += 1

    soln, info = spla.cg(linop, rhs_red, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter, callback=_cb)
    if info < 0:
        raise RuntimeError("CG solver failed with an illegal input or breakdown.")

    q = np.zeros((D, nv), dtype=np.float64)
    if nb > 0:
        q[:, bdry_idx2] = apply_basis(basis, soln[: d * nb].reshape((d, nb), order="F"))
    if ni > 0:
        q[:, int_idx2] = soln[d * nb :].reshape((D, ni), order="F")
    q += q_fixed2

    return q, int(it_counter["n"]), soln


def AlignmentConstrainedLinearSolve(
    A: sp.spmatrix | np.ndarray,
    rhs: np.ndarray,
    intIdx: np.ndarray,
    bdryIdx: np.ndarray,
    BdryBasis: np.ndarray,
    qFixed: np.ndarray,
    warmstart: np.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[np.ndarray, int, np.ndarray]:
    return alignment_constrained_linear_solve(
        A,
        rhs,
        intIdx,
        bdryIdx,
        BdryBasis,
        qFixed,
        warmstart,
        rtol=rtol,
        maxiter=maxiter,
    )
