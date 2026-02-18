from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import linalg as sla
from scipy.linalg import lapack

# SciPy exposes many LAPACK wrappers dynamically at runtime.
# Bind through getattr so static analyzers do not flag missing attributes.
_DPOTRF = getattr(lapack, "dpotrf")
_DPOTRS = getattr(lapack, "dpotrs")
_DGELS = getattr(lapack, "dgels")
_DGESVD = getattr(lapack, "dgesvd")
_DGEQP3 = getattr(lapack, "dgeqp3")
_DORGQR = getattr(lapack, "dorgqr")


def _to_real64_pages(name: str, x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")

    if arr.ndim == 2:
        return arr[:, :, None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError("batchop operates on 2D or 3D arrays only.")


def _validate_binary_dims(A3: np.ndarray, A_is_2d: bool, B3: np.ndarray, B_is_2d: bool) -> None:
    # MATLAB often treats trailing singleton page dimensions as absent.
    # Allow 2D/3D mixing only when both inputs are effectively single-page.
    if A_is_2d != B_is_2d:
        if A3.shape[2] == 1 and B3.shape[2] == 1:
            return
        raise ValueError("Dimensions do not match.")
    if A3.shape[2] != B3.shape[2]:
        raise ValueError("Dimensions do not match.")


def _parse_transpose(flag: str, arg_name: str) -> str:
    if flag not in {"N", "T"}:
        raise ValueError(
            f"Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T']) for {arg_name}."
        )
    return flag


def _parse_uplo(flag: str) -> str:
    if flag not in {"U", "L"}:
        raise ValueError("Usage: batchop('trisolve', T, B, 'U' or 'L').")
    return flag


def _restore_pages(arr: np.ndarray, squeeze: bool) -> np.ndarray:
    if squeeze:
        return arr[:, :, 0]
    return arr


def _restore_vec_pages(arr: np.ndarray, squeeze: bool) -> np.ndarray:
    if squeeze:
        return arr[:, [0]]
    return arr


def _restore_info(info: np.ndarray, squeeze: bool) -> np.ndarray | np.int64:
    out = info.astype(np.int64, copy=False)
    if squeeze:
        return np.int64(out[0])
    return out


def _batch_to_first(A3: np.ndarray) -> np.ndarray:
    return np.moveaxis(A3, 2, 0)


def _first_to_batch(Ab: np.ndarray) -> np.ndarray:
    return np.moveaxis(Ab, 0, 2)


def _finite_page_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)

    mask = np.ones(arrays[0].shape[2], dtype=bool)
    for arr in arrays:
        if arr.ndim != 3:
            raise ValueError("Internal arrays must be 3D.")
        mask &= np.isfinite(arr).all(axis=(0, 1))
    return mask


def _run_batch_fallback(
    indices: np.ndarray,
    run_batch: Callable[[np.ndarray], None],
    run_single: Callable[[int], None],
) -> None:
    """Run a batched operation, recursively falling back to single-page processing.

    Only failing batches are split and retried, so pages with valid data are kept
    on the fast path.
    """
    if indices.size == 0:
        return
    try:
        run_batch(indices)
        return
    except (np.linalg.LinAlgError, ValueError):
        if indices.size == 1:
            run_single(int(indices[0]))
            return
        mid = indices.size // 2
        _run_batch_fallback(indices[:mid], run_batch, run_single)
        _run_batch_fallback(indices[mid:], run_batch, run_single)


def _mult_pages(A3: np.ndarray, B3: np.ndarray, transp_a: str, transp_b: str) -> np.ndarray:
    A_eff = np.swapaxes(A3, 0, 1) if transp_a == "T" else A3
    B_eff = np.swapaxes(B3, 0, 1) if transp_b == "T" else B3

    if A_eff.shape[1] != B_eff.shape[0]:
        raise ValueError("Inner dimensions do not match.")

    A_batch = np.moveaxis(A_eff, 2, 0)
    B_batch = np.moveaxis(B_eff, 2, 0)
    C_batch = A_batch @ B_batch
    return np.moveaxis(C_batch, 0, 2)


def _fcopy(page: np.ndarray) -> np.ndarray:
    """Create a Fortran-contiguous copy of a 2D page (never a view)."""
    return np.array(page, dtype=np.float64, order="F", copy=True)


def _chol_output_with_input_upper(L3: np.ndarray, A3: np.ndarray) -> np.ndarray:
    """Keep Cholesky factor in lower triangle and preserve input upper triangle."""
    upper = np.triu(np.ones(L3.shape[:2], dtype=bool), k=1)[:, :, None]
    return np.where(upper, A3, L3)


def _chol_pages(A3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n, p = A3.shape
    if m != n:
        raise ValueError("Matrices are not square.")

    valid = _finite_page_mask(A3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)
    L = np.empty_like(A3, dtype=np.float64)
    info = np.empty((p,), dtype=np.int64)

    if valid_idx.size > 0:
        def run_batch(idxs: np.ndarray) -> None:
            A_batch = _batch_to_first(A3[:, :, idxs])
            L_batch = sla.cholesky(
                A_batch,
                lower=True,
                check_finite=False,
                overwrite_a=False,
            )
            L[:, :, idxs] = _first_to_batch(L_batch)
            info[idxs] = 0

        def run_single(k: int) -> None:
            try:
                Lk, inf = _DPOTRF(
                    _fcopy(A3[:, :, k]),
                    lower=1,
                    clean=0,
                    overwrite_a=1,
                )
                L[:, :, k] = Lk
                info[k] = inf
            except (np.linalg.LinAlgError, ValueError):
                L[:, :, k] = np.nan
                info[k] = 1

        _run_batch_fallback(valid_idx, run_batch, run_single)

    if invalid_idx.size > 0:
        # Non-finite inputs should be treated as failed factorizations.
        info[invalid_idx] = 1
        for k in invalid_idx:
            L[:, :, k], inf = _DPOTRF(
                _fcopy(A3[:, :, k]),
                lower=1,
                clean=0,
                overwrite_a=1,
            )
            info[k] = inf

    L = _chol_output_with_input_upper(L, A3)
    return L, info


def _cholsolve_pages(L3: np.ndarray, B3: np.ndarray) -> np.ndarray:
    m, n, p = L3.shape
    if m != n or B3.shape[0] != m:
        raise ValueError("Dimensions do not match.")

    try:
        Lb = _batch_to_first(L3)
        Bb = _batch_to_first(B3)
        Xb = sla.solve_triangular(
            Lb,
            Bb,
            lower=True,
            trans="N",
            check_finite=False,
            overwrite_b=False,
        )
        Xb = sla.solve_triangular(
            Lb,
            Xb,
            lower=True,
            trans="T",
            check_finite=False,
            overwrite_b=False,
        )
        return _first_to_batch(Xb)
    except (np.linalg.LinAlgError, ValueError):
        nrhs = B3.shape[1]
        X = np.empty((m, nrhs, p), dtype=np.float64)
        for k in range(p):
            Xk, _ = _DPOTRS(
                _fcopy(L3[:, :, k]),
                _fcopy(B3[:, :, k]),
                lower=1,
                overwrite_b=1,
            )
            X[:, :, k] = Xk
        return X


def _cholcong_pages(L3: np.ndarray, B3: np.ndarray) -> np.ndarray:
    m, n, p = L3.shape
    if m != n or B3.shape[0] != m or B3.shape[1] != m:
        raise ValueError("Dimensions do not match.")

    try:
        Lb = _batch_to_first(L3)
        Bb = _batch_to_first(B3)
        Yb = sla.solve_triangular(
            Lb,
            Bb,
            lower=True,
            trans="N",
            check_finite=False,
            overwrite_b=False,
        )
        Xb = sla.solve_triangular(
            Lb,
            np.swapaxes(Yb, 1, 2),
            lower=True,
            trans="N",
            check_finite=False,
            overwrite_b=False,
        )
        return _first_to_batch(np.swapaxes(Xb, 1, 2))
    except (np.linalg.LinAlgError, ValueError):
        X = np.empty((m, m, p), dtype=np.float64)
        for k in range(p):
            Lk = L3[:, :, k]
            Bk = B3[:, :, k]
            Yk = np.asarray(
                sla.solve_triangular(
                    Lk,
                    Bk,
                    lower=True,
                    trans="N",
                    unit_diagonal=False,
                    check_finite=False,
                    overwrite_b=False,
                ),
                dtype=np.float64,
            )
            Xk_t = np.asarray(
                sla.solve_triangular(
                    Lk,
                    Yk.T,
                    lower=True,
                    trans="N",
                    unit_diagonal=False,
                    check_finite=False,
                    overwrite_b=False,
                ),
                dtype=np.float64,
            )
            Xk = Xk_t.T
            X[:, :, k] = Xk
        return X


def _trisolve_pages(T3: np.ndarray, B3: np.ndarray, uplo: str) -> np.ndarray:
    m, n, p = T3.shape
    if m != n or B3.shape[0] != m:
        raise ValueError("Dimensions do not match.")

    nrhs = B3.shape[1]
    lower = uplo == "L"
    try:
        Tb = _batch_to_first(T3)
        Bb = _batch_to_first(B3)
        Xb = sla.solve_triangular(
            Tb,
            Bb,
            lower=lower,
            trans="N",
            unit_diagonal=False,
            check_finite=False,
            overwrite_b=False,
        )
        return _first_to_batch(Xb)
    except (np.linalg.LinAlgError, ValueError):
        X = np.empty((m, nrhs, p), dtype=np.float64)
        for k in range(p):
            X[:, :, k] = np.asarray(
                sla.solve_triangular(
                    T3[:, :, k],
                    B3[:, :, k],
                    lower=lower,
                    trans="N",
                    unit_diagonal=False,
                    check_finite=False,
                    overwrite_b=False,
                ),
                dtype=np.float64,
            )
        return X


def _leastsq_pages(A3: np.ndarray, B3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, r, p = A3.shape
    if B3.shape[0] != m:
        raise ValueError("Dimensions do not match.")

    n = B3.shape[1]
    valid = _finite_page_mask(A3, B3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)

    X = np.empty((r, n, p), dtype=np.float64)
    info = np.empty((p,), dtype=np.int64)

    if valid_idx.size > 0:
        ldb = max(m, r)

        def run_batch(idxs: np.ndarray) -> None:
            Ab = _batch_to_first(A3[:, :, idxs])
            Bb = _batch_to_first(B3[:, :, idxs])
            X_batch, _, rank_out, _ = sla.lstsq(
                Ab,
                Bb,
                cond=None,
                overwrite_a=False,
                overwrite_b=False,
                check_finite=False,
            )
            X[:, :, idxs] = _first_to_batch(np.asarray(X_batch, dtype=np.float64))
            info[idxs] = 0

        def run_single(k: int) -> None:
            X[:, :, k] = np.nan
            try:
                # LAPACK dgels requires B leading dimension >= max(m, r).
                Bk = np.zeros((ldb, n), dtype=np.float64, order="F")
                Bk[:m, :] = B3[:, :, k]
                _, B_sol, inf = _DGELS(
                    _fcopy(A3[:, :, k]),
                    Bk,
                    trans="N",
                    overwrite_a=1,
                    overwrite_b=1,
                )
                X[:, :, k] = B_sol[:r, :]
                info[k] = inf
            except (np.linalg.LinAlgError, ValueError):
                info[k] = 1

        _run_batch_fallback(valid_idx, run_batch, run_single)

    if invalid_idx.size > 0:
        ldb = max(m, r)
        for k in invalid_idx:
            X[:, :, k] = np.nan
            try:
                Bk = np.zeros((ldb, n), dtype=np.float64, order="F")
                Bk[:m, :] = B3[:, :, k]
                _, B_sol, inf = _DGELS(
                    _fcopy(A3[:, :, k]),
                    Bk,
                    trans="N",
                    overwrite_a=1,
                    overwrite_b=1,
                )
                X[:, :, k] = B_sol[:r, :]
                info[k] = inf
            except (np.linalg.LinAlgError, ValueError):
                info[k] = 1

    return X, info


def _pinv_pages(A3: np.ndarray, rank: int) -> np.ndarray:
    m, n, p = A3.shape
    rank = min(rank, min(m, n))

    valid = _finite_page_mask(A3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)

    B = np.empty((n, m, p), dtype=np.float64)

    if rank == 0:
        B.fill(0.0)
        B[:, :, invalid_idx] = np.nan
        return B

    if valid_idx.size > 0:
        def run_batch(idxs: np.ndarray) -> None:
            A_batch = _batch_to_first(A3[:, :, idxs])
            U_batch, S_batch, Vt_batch = sla.svd(
                A_batch,
                full_matrices=False,
                check_finite=False,
                overwrite_a=False,
            )
            U = _first_to_batch(U_batch)[:, :rank, :]
            S = np.asarray(S_batch, dtype=np.float64).T[:rank, :]
            Vt = _first_to_batch(Vt_batch)[:rank, :, :]
            Vt_div = Vt / S[:, None, :]
            B[:, :, idxs] = _mult_pages(Vt_div, U, "T", "T")

        def run_single(k: int) -> None:
            try:
                Uk, Sk, Vtk, inf = _DGESVD(
                    _fcopy(A3[:, :, k]),
                    compute_uv=1,
                    full_matrices=0,
                    overwrite_a=1,
                )
                if inf != 0:
                    B[:, :, k] = np.nan
                    return
                B[:, :, k] = (Vtk[:rank, :] / Sk[:rank, None]) @ Uk[:, :rank].T
            except (np.linalg.LinAlgError, ValueError):
                B[:, :, k] = np.nan

        _run_batch_fallback(valid_idx, run_batch, run_single)

    if invalid_idx.size > 0:
        B[:, :, invalid_idx] = np.nan

    return B


def _svd_pages(A3: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m, n, p = A3.shape
    r = min(m, n)
    rank = min(rank, r)

    valid = _finite_page_mask(A3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)

    U = np.empty((m, rank, p), dtype=np.float64)
    S = np.empty((rank, p), dtype=np.float64)
    Vt = np.empty((rank, n, p), dtype=np.float64)
    info = np.empty((p,), dtype=np.int64)

    if rank == 0:
        U.fill(0.0)
        Vt.fill(0.0)
        S.fill(0.0)
        info[:] = 0
        return U, S, Vt, info

    if valid_idx.size > 0:
        def run_batch(idxs: np.ndarray) -> None:
            A_batch = _batch_to_first(A3[:, :, idxs])
            U_batch, S_batch, Vt_batch = sla.svd(
                A_batch,
                full_matrices=False,
                check_finite=False,
                overwrite_a=False,
            )
            U[:, :, idxs] = np.moveaxis(U_batch[:, :, :rank], 0, 2)
            S[:, idxs] = S_batch[:, :rank].T
            Vt[:, :, idxs] = np.moveaxis(Vt_batch[:, :rank, :], 0, 2)
            info[idxs] = 0

        def run_single(k: int) -> None:
            try:
                Uk, Sk, Vtk, inf = _DGESVD(
                    _fcopy(A3[:, :, k]),
                    compute_uv=1,
                    full_matrices=0,
                    overwrite_a=1,
                )
                U[:, :, k] = Uk[:, :rank]
                S[:, k] = Sk[:rank]
                Vt[:, :, k] = Vtk[:rank, :]
                info[k] = inf
            except (np.linalg.LinAlgError, ValueError):
                U[:, :, k] = np.nan
                S[:, k] = np.nan
                Vt[:, :, k] = np.nan
                info[k] = 1

        _run_batch_fallback(valid_idx, run_batch, run_single)

    if invalid_idx.size > 0:
        for k in invalid_idx:
            U[:, :, k] = np.nan
            S[:, k] = np.nan
            Vt[:, :, k] = np.nan
            info[k] = 1

    return (
        np.asarray(U, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(Vt, dtype=np.float64),
        info,
    )


def _qr_pages(
    A3: np.ndarray, *, include_info: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m, n, p = A3.shape
    r = min(m, n)

    valid = _finite_page_mask(A3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)

    Q = np.empty((m, r, p), dtype=np.float64)
    R = np.zeros((m, n, p), dtype=np.float64)
    P = np.empty((n, p), dtype=np.int64)
    info = np.empty((p,), dtype=np.int64) if include_info else None

    if valid_idx.size > 0:
        try:
            A_batch = _batch_to_first(A3[:, :, valid_idx])
            Q_batch, R_batch, P_batch = sla.qr(
                A_batch,
                mode="economic",
                pivoting=True,
                check_finite=False,
                overwrite_a=False,
            )
            Q[:, :, valid_idx] = np.asarray(_first_to_batch(Q_batch[:, :, :r]), dtype=np.float64)
            R_part = _first_to_batch(R_batch)
            R[: R_part.shape[0], : R_part.shape[1], valid_idx] = R_part
            P[:, valid_idx] = np.asarray(P_batch.T, dtype=np.int64) + 1
            if include_info:
                info[valid_idx] = 0
        except (np.linalg.LinAlgError, ValueError):
            for k in valid_idx:
                qr, jpvt, tau, _, info_qr = _DGEQP3(
                    _fcopy(A3[:, :, k]),
                    overwrite_a=1,
                )
                # Extract R BEFORE dorgqr overwrites the QR buffer (matches C++ order)
                R[:, :, k] = np.triu(qr)
                P[:, k] = jpvt.astype(np.int64, copy=False)  # 1-based, matching LAPACK/MATLAB
                qk, _, info_q = _DORGQR(
                    _fcopy(qr[:, :r]),
                    tau,
                    overwrite_a=1,
                )
                Q[:, :, k] = qk[:, :r]
                if include_info and info is not None:
                    info[k] = info_qr if info_qr != 0 else info_q

    if invalid_idx.size > 0:
        for k in invalid_idx:
            R[:, :, k] = np.nan
            P[:, k] = -1
            if include_info and info is not None:
                info[k] = 1
            try:
                qr, jpvt, tau, _, info_qr = _DGEQP3(
                    _fcopy(A3[:, :, k]),
                    overwrite_a=1,
                )
                R[:, :, k] = np.triu(qr)
                P[:, k] = jpvt.astype(np.int64, copy=False)  # 1-based, matching LAPACK/MATLAB
                qk, _, info_q = _DORGQR(
                    _fcopy(qr[:, :r]),
                    tau,
                    overwrite_a=1,
                )
                Q[:, :, k] = qk[:, :r]
                if include_info and info is not None:
                    info[k] = info_qr if info_qr != 0 else info_q
            except (np.linalg.LinAlgError, ValueError):
                Q[:, :, k] = np.nan
                if include_info and info is not None:
                    info[k] = 1

    if include_info and info is not None:
        return Q, R, P, info
    return Q, R, P


def _eig_pages(A3: np.ndarray, vectors: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    m, n, p = A3.shape
    if m != n:
        raise ValueError("Matrices are not square.")

    valid = _finite_page_mask(A3)
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)

    E = np.empty((m, p), dtype=np.float64)
    Q = np.empty((m, m, p), dtype=np.float64) if vectors else None
    info = np.empty((p,), dtype=np.int64)

    if valid_idx.size > 0:
        if vectors:
            def run_batch(idxs: np.ndarray) -> None:
                A_batch = _batch_to_first(A3[:, :, idxs])
                evals_batch, evecs_batch = sla.eig(
                    A_batch,
                    left=False,
                    right=True,
                    check_finite=False,
                )
                order = np.argsort(np.real(evals_batch), axis=1)
                evals_batch = np.real_if_close(
                    np.take_along_axis(evals_batch, order, axis=1),
                    tol=100,
                )
                evecs_batch = np.real_if_close(
                    np.take_along_axis(evecs_batch, order[:, None, :], axis=2),
                    tol=100,
                )
                E[:, idxs] = np.moveaxis(evals_batch, 0, 1)
                Q[:, :, idxs] = np.moveaxis(evecs_batch, 0, 2)
                info[idxs] = 0

            def run_single(k: int) -> None:
                try:
                    evals, evecs = np.linalg.eig(A3[:, :, k])
                    order = np.argsort(np.real(evals))
                    E[:, k] = np.real_if_close(evals[order], tol=100)
                    if Q is not None:
                        Q[:, :, k] = np.real_if_close(evecs[:, order], tol=100)
                    info[k] = 0
                except (np.linalg.LinAlgError, ValueError):
                    info[k] = 1
                    E[:, k] = np.nan
                    if Q is not None:
                        Q[:, :, k] = np.nan

            _run_batch_fallback(valid_idx, run_batch, run_single)
        else:
            def run_batch(idxs: np.ndarray) -> None:
                A_batch = _batch_to_first(A3[:, :, idxs])
                evals_batch = np.real_if_close(sla.eigvals(A_batch, check_finite=False), tol=100)
                E[:, idxs] = np.moveaxis(np.sort(np.real(evals_batch), axis=1), 0, 1)
                info[idxs] = 0

            def run_single(k: int) -> None:
                try:
                    E[:, k] = np.sort(np.real(np.linalg.eigvals(A3[:, :, k])))
                    info[k] = 0
                except (np.linalg.LinAlgError, ValueError):
                    info[k] = 1
                    E[:, k] = np.nan

            _run_batch_fallback(valid_idx, run_batch, run_single)

    if invalid_idx.size > 0:
        for k in invalid_idx:
            info[k] = 1
            E[:, k] = np.nan
            if Q is not None:
                Q[:, :, k] = np.nan

    return np.asarray(E, dtype=np.float64), None if Q is None else np.asarray(Q, dtype=np.float64), info


def batchop(
    op: str,
    A: np.ndarray,
    *args: Any,
    return_info: bool = False,
    allow_extended_ops: bool = False,
    matlab_indexing: bool = False,
) -> Any:
    """
    Batched linear algebra operations on page-stacked arrays.

    Inputs follow MATLAB-style page layout `(m, n, k)` and support 2D/3D arrays.
    This is a CPU implementation using NumPy/SciPy.

    `return_info=False` (default) mimics MATLAB calls that do not request status
    outputs via `nargout`:
      - `chol` -> `L`
      - `leastsq` -> `X`
      - `svd` -> `(U, S, Vt)`
      - `qr` -> `(Q, R, P)`  (CPU MEX behavior)
      - `eig` -> `(E, Q)`
      - `eigval` -> `E`

    With `return_info=True`, status outputs are included for operations that
    expose them in this Python API:
      - `chol` -> `(L, info)`
      - `leastsq` -> `(X, info)`
      - `svd` -> `(U, S, Vt, info)`
      - `eig` -> `(E, Q, info)`
      - `eigval` -> `(E, info)`

    `allow_extended_ops=True` enables non-MEX extensions. Currently this only
    affects `qr`: with both `return_info=True` and `allow_extended_ops=True`,
    `qr` returns `(Q, R, P, info)`.

    `matlab_indexing` controls whether the pivot vector `P` from `qr` is
    returned using MATLAB/LAPACK 1-based indices when set to `True`, or
    converted to zero-based Python indices when `False` (default).
    """

    if not isinstance(op, str):
        raise ValueError("Must provide an operation and input matrix.")

    A3, A_is_2d = _to_real64_pages("A", A)

    if op == "mult":
        if not (1 <= len(args) <= 3):
            raise ValueError("Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T']).")
        B3, B_is_2d = _to_real64_pages("B", args[0])
        _validate_binary_dims(A3, A_is_2d, B3, B_is_2d)
        transp_a = _parse_transpose(args[1] if len(args) >= 2 else "N", "transpA")
        transp_b = _parse_transpose(args[2] if len(args) >= 3 else "N", "transpB")
        C3 = _mult_pages(A3, B3, transp_a, transp_b)
        return _restore_pages(C3, A_is_2d)

    if op == "chol":
        if len(args) != 0:
            raise ValueError("Usage: batchop('chol', A).")
        L3, info = _chol_pages(A3)
        L = _restore_pages(L3, A_is_2d)
        if return_info:
            return L, _restore_info(info, A_is_2d)
        return L

    if op == "cholsolve":
        if len(args) != 1:
            raise ValueError("Usage: batchop('cholsolve', L, B).")
        B3, B_is_2d = _to_real64_pages("B", args[0])
        _validate_binary_dims(A3, A_is_2d, B3, B_is_2d)
        X3 = _cholsolve_pages(A3, B3)
        return _restore_pages(X3, A_is_2d)

    if op == "cholcong":
        if len(args) != 1:
            raise ValueError("Usage: batchop('cholcong', L, B).")
        B3, B_is_2d = _to_real64_pages("B", args[0])
        _validate_binary_dims(A3, A_is_2d, B3, B_is_2d)
        X3 = _cholcong_pages(A3, B3)
        return _restore_pages(X3, A_is_2d)

    if op == "trisolve":
        if len(args) != 2:
            raise ValueError("Usage: batchop('trisolve', T, B, 'U' or 'L').")
        B3, B_is_2d = _to_real64_pages("B", args[0])
        _validate_binary_dims(A3, A_is_2d, B3, B_is_2d)
        uplo = _parse_uplo(args[1])
        X3 = _trisolve_pages(A3, B3, uplo)
        return _restore_pages(X3, A_is_2d)

    if op == "leastsq":
        if len(args) != 1:
            raise ValueError("Usage: batchop('leastsq', A, B).")
        B3, B_is_2d = _to_real64_pages("B", args[0])
        _validate_binary_dims(A3, A_is_2d, B3, B_is_2d)
        X3, info = _leastsq_pages(A3, B3)
        X = _restore_pages(X3, A_is_2d)
        if return_info:
            return X, _restore_info(info, A_is_2d)
        return X

    if op == "svd":
        if len(args) != 1:
            raise ValueError("Must provide desired rank.")
        rank = int(args[0])
        if rank < 0:
            raise ValueError("rank must be nonnegative.")

        U3, S2, Vt3, info = _svd_pages(A3, rank)
        U = _restore_pages(U3, A_is_2d)
        S = _restore_vec_pages(S2, A_is_2d)
        Vt = _restore_pages(Vt3, A_is_2d)
        if return_info:
            return U, S, Vt, _restore_info(info, A_is_2d)
        return U, S, Vt

    if op == "pinv":
        if len(args) != 1:
            raise ValueError("Usage: batchop('pinv', A, rank).")
        rank = int(args[0])
        if rank < 0:
            raise ValueError("rank must be nonnegative.")

        B3 = _pinv_pages(A3, rank)
        return _restore_pages(B3, A_is_2d)

    if op == "qr":
        if len(args) != 0:
            raise ValueError("Usage: batchop('qr', A).")
        include_info = return_info and allow_extended_ops
        qr_out = _qr_pages(A3, include_info=include_info)
        if include_info:
            Q3, R3, P2, info = qr_out
        else:
            Q3, R3, P2 = qr_out
        Q = _restore_pages(Q3, A_is_2d)
        R = _restore_pages(R3, A_is_2d)
        P = _restore_vec_pages(P2, A_is_2d)
        if not matlab_indexing:
            P = np.asarray(P, dtype=np.int64) - 1
        if include_info:
            return Q, R, P, _restore_info(info, A_is_2d)
        return Q, R, P

    if op == "eig":
        if len(args) != 0:
            raise ValueError("Usage: batchop('eig', A).")
        E2, Q3, info = _eig_pages(A3, vectors=True)
        E = _restore_vec_pages(E2, A_is_2d)
        assert Q3 is not None
        Q = _restore_pages(Q3, A_is_2d)
        if return_info:
            return E, Q, _restore_info(info, A_is_2d)
        return E, Q

    if op == "eigval":
        if len(args) != 0:
            raise ValueError("Usage: batchop('eigval', A).")
        E2, _, info = _eig_pages(A3, vectors=False)
        E = _restore_vec_pages(E2, A_is_2d)
        if return_info:
            return E, _restore_info(info, A_is_2d)
        return E

    raise ValueError("Unknown operation.")


def Batchop(
    op: str,
    A: np.ndarray,
    *args: Any,
    return_info: bool = False,
    allow_extended_ops: bool = False,
    matlab_indexing: bool = False,
) -> Any:
    return batchop(
        op,
        A,
        *args,
        return_info=return_info,
        allow_extended_ops=allow_extended_ops,
        matlab_indexing=matlab_indexing,
    )
