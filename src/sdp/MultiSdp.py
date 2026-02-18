from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from mosek.fusion import *  # noqa: F403
from mosek.fusion import (
    OptimizeError,
    ProblemStatus,
    SolutionStatus,
)

_OBJ_MINIMIZE = getattr(ObjectiveSense, "Minimize")  # noqa: F405
_SOLN_OPTIMAL = getattr(SolutionStatus, "Optimal")
logger = logging.getLogger(__name__)


def _as_real64_2d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
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


def _set_solver_params(model: Any, solver_params: dict[str, Any] | None) -> None:
    # Match the original C++ MEX defaults when supported by installed MOSEK.
    # Newer MOSEK versions may rename parameters, so we ignore unknown keys.
    params: dict[str, Any] = {
        "intpntMultiThread": "off",
        "intpntCoTolRelGap": "1.0e-12",
    }
    if solver_params is not None:
        params.update(solver_params)

    for name, value in params.items():
        try:
            model.setSolverParam(name, value)
        except Exception:
            continue


@dataclass(slots=True)
class _ProblemData:
    A: np.ndarray
    b: np.ndarray
    d: int
    d1: int
    m: int


def _validate_problem_data(A: np.ndarray, b: np.ndarray) -> _ProblemData:
    A2 = _as_real64_2d("A", A)
    m, cols = A2.shape
    d1 = int(round(np.sqrt(cols)))
    if d1 * d1 != cols:
        raise ValueError("A must have (d + 1)^2 columns.")
    if d1 < 2:
        raise ValueError("Invalid SDP size derived from A.")
    d = d1 - 1
    bcol = _as_real64_column("b", b, m)
    return _ProblemData(A=A2, b=bcol, d=d, d1=d1, m=m)


class MultiSdpSolver:
    """Reusable solver for many small SDPs with fixed linear constraints.

    Solves, for each column q0_j:
        min_Q trace(Q) - 2 * q0_j^T q
        s.t.  A * vec(Q) = b
              Q in S_+^{d+1}
        where q = Q[1:, 0].
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        *,
        solver_params: dict[str, Any] | None = None,
    ):
        self._closed = False  # Set early for __del__ safety

        data = _validate_problem_data(A, b)
        self._A = data.A
        self._b = data.b
        self.d = data.d
        self.d1 = data.d1
        self.m = data.m

        self._model = Model("MultiSdp")
        _set_solver_params(self._model, solver_params)

        self._Q = self._model.variable(  # noqa: F405
            "Q", Domain.inPSDCone(self.d1),
        )
        A_mat = Matrix.dense(self._A)  # noqa: F405
        b_mat = Matrix.dense(self._b)  # noqa: F405
        self._model.constraint(
            Expr.mul(  # noqa: F405
                A_mat,
                self._Q.reshape(self.d1 * self.d1, 1),
            ),
            Domain.equalsTo(b_mat),  # noqa: F405
        )
        self._q_expr = self._Q.slice(
            [1, 0], [self.d1, 1],
        )
        self._trace_expr = Expr.sum(  # noqa: F405
            self._Q.diag(),
        )

    def close(self) -> None:
        if not self._closed:
            self._model.dispose()
            self._closed = True

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        self.close()

    def project(
        self,
        q0: np.ndarray,
        *,
        return_Q: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Project each column of q0 onto the SDP-relaxed variety.

        Parameters
        ----------
        q0 : np.ndarray
            Shape (d, n) matrix whose columns are the
            points to project.
        return_Q : bool
            If True, also return the full PSD matrices.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Projected points q (d, n). If return_Q is True,
            also returns Qflat ((d+1)^2, n).

        Raises
        ------
        RuntimeError
            If the solver is closed or MOSEK fails to find
            an optimal solution.
        """
        if self._closed:
            raise RuntimeError("MultiSdpSolver has been closed.")

        q0m = _as_real64_2d("q0", q0)
        if q0m.shape[0] != self.d:
            raise ValueError("Dimensions do not match.")

        n = q0m.shape[1]
        q = np.empty((self.d, n), dtype=np.float64)
        Qflat = (
            np.empty((self.d1 * self.d1, n), dtype=np.float64)
            if return_Q
            else None
        )

        for j in range(n):
            # Rebuild objective per iteration because
            # MOSEK prohibits M.parameter() in expressions
            # involving PSD cone variables.
            q0_col = Matrix.dense(  # noqa: F405
                q0m[:, [j]],
            )
            self._model.objective(
                _OBJ_MINIMIZE,
                Expr.add(  # noqa: F405
                    self._trace_expr,
                    Expr.mul(  # noqa: F405
                        -2.0,
                        Expr.dot(  # noqa: F405
                            q0_col, self._q_expr,
                        ),
                    ),
                ),
            )
            self._solve_and_check(j)

            q[:, j] = np.asarray(
                self._q_expr.level(), dtype=np.float64,
            ).reshape(self.d)
            if return_Q and Qflat is not None:
                Qflat[:, j] = np.asarray(
                    self._Q.level(), dtype=np.float64,
                ).reshape(self.d1 * self.d1)

        if return_Q and Qflat is not None:
            return q, Qflat
        return q

    def _solve_and_check(self, column_index: int) -> None:
        """Run MOSEK solver and verify optimality.

        Raises
        ------
        RuntimeError
            If the solution is not optimal.
        """
        try:
            self._model.solve()
        except OptimizeError as exc:
            raise RuntimeError(
                f"MOSEK optimization error at column {column_index}: "
                f"{exc}",
            ) from exc

        sol_status = self._model.getPrimalSolutionStatus()
        if sol_status != _SOLN_OPTIMAL:
            prob_status = self._model.getProblemStatus()
            raise RuntimeError(
                f"SDP not optimal at column {column_index}: "
                f"solution_status={sol_status}, "
                f"problem_status={prob_status}",
            )


def multi_sdp(
    q0: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    return_Q: bool = False,
    solver_params: dict[str, Any] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """One-shot SDP projection wrapper matching MATLAB MultiSdp(q0, A, b)."""
    solver = MultiSdpSolver(A, b, solver_params=solver_params)
    try:
        return solver.project(q0, return_Q=return_Q)
    finally:
        solver.close()


MultiSdp = multi_sdp
