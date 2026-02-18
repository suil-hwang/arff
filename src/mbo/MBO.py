from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._basis import apply_basis, apply_basis_t
from .AlignmentConstrainedLinearSolve import alignment_constrained_linear_solve
from .bundle import MBOFiber


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def _fiber_get(fiber: Any, *names: str) -> Any:
    for name in names:
        if isinstance(fiber, dict) and name in fiber:
            return fiber[name]
        if hasattr(fiber, name):
            return getattr(fiber, name)
    raise ValueError(f"Missing fiber field. Tried: {names!r}")


def _as_real64(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real.")
    if arr.dtype != np.float64:
        raise ValueError("Inputs must be of type double.")
    return arr


def _energy_quadratic(q: np.ndarray, L: sp.spmatrix | np.ndarray) -> float:
    q_t = q.T  # (nv, dim)
    Lq = L @ q_t
    return float(0.5 * np.sum(q_t * Lq))


def _mass_norm(q: np.ndarray, M: sp.spmatrix | np.ndarray) -> float:
    q_t = q.T  # (nv, dim)
    Mq = M @ q_t
    val = float(np.sum(q_t * Mq))
    return float(np.sqrt(max(val, 0.0)))


def mbo(
    mesh_data: Any,
    fiber: MBOFiber | dict[str, Any] | Callable[[], MBOFiber | dict[str, Any]],
    q0: np.ndarray | None = None,
    tau_mult: float = 1.0,
    tau_exponent: float = 0.0,
    save_iterates: bool = False,
    *,
    max_iters: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Diffusion-generated algorithm for variety-valued maps (MBO)."""
    if callable(fiber) and not isinstance(fiber, (dict, MBOFiber)):
        fiber = fiber()

    nv = int(_mesh_get(mesh_data, "nv"))
    bdry_idx = np.asarray(_mesh_get(mesh_data, "bdry_idx", "bdryIdx"), dtype=np.int64).reshape(-1)
    int_idx = np.asarray(_mesh_get(mesh_data, "int_idx", "intIdx"), dtype=np.int64).reshape(-1)
    bdry_normals = _as_real64("bdry_normals", _mesh_get(mesh_data, "bdry_normals", "bdryNormals"))
    L = _mesh_get(mesh_data, "L")
    M = _mesh_get(mesh_data, "M")
    lambda1_l = float(_mesh_get(mesh_data, "lambda1L"))

    dim = int(_fiber_get(fiber, "dim"))
    fiber_proj = _fiber_get(fiber, "proj")
    fiber_proj_aligned = _fiber_get(fiber, "proj_aligned", "projAligned")
    fiber_bdry_basis = _fiber_get(fiber, "bdry_basis", "bdryBasis")
    fiber_rand = _fiber_get(fiber, "rand")

    if lambda1_l <= 0.0:
        raise ValueError("mesh_data.lambda1L must be positive.")
    if not (sp.issparse(L) or isinstance(L, np.ndarray)):
        raise ValueError("mesh_data.L must be sparse or dense matrix.")
    if not (sp.issparse(M) or isinstance(M, np.ndarray)):
        raise ValueError("mesh_data.M must be sparse or dense matrix.")

    bdry_fixed, bdry_basis = fiber_bdry_basis(bdry_normals)
    bdry_fixed = _as_real64("bdry_fixed", bdry_fixed)
    bdry_basis = _as_real64("bdry_basis", bdry_basis)

    q_fixed = np.zeros((dim, nv), dtype=np.float64)
    if bdry_idx.size > 0:
        q_fixed[:, bdry_idx] = bdry_fixed

    if q0 is None:
        q0m = fiber_rand(nv, bdry_idx, bdry_normals)
    else:
        q0m = _as_real64("q0", q0)
    if q0m.shape != (dim, nv):
        raise ValueError("q0 has invalid shape.")

    tau0 = float(tau_mult) / lambda1_l

    q = q0m.copy()
    q_proj = np.zeros_like(q)
    warmstart: np.ndarray | None = None

    info: list[dict[str, Any]] = []
    cost0 = _energy_quadratic(q, L)
    info0: dict[str, Any] = {
        "tau": tau0,
        "cost": cost0,
        "costdelta": cost0,
        "gradnorm": float(np.linalg.norm((L @ q.T), ord="fro")),
        "time": 0.0,
    }
    if save_iterates:
        info0["q"] = q.copy()
    info.append(info0)

    t0 = time.perf_counter()
    for k in range(2, max_iters + 1):
        tau_k = tau0 / ((k - 1) ** float(tau_exponent))
        A = M + tau_k * L

        q_diffused, iter_count, warmstart = alignment_constrained_linear_solve(
            A,
            (M @ q.T).T,
            int_idx,
            bdry_idx,
            bdry_basis,
            q_fixed,
            warmstart,
        )

        if bdry_idx.size > 0:
            q_diff_bdry = apply_basis_t(bdry_basis, q_diffused[:, bdry_idx] - bdry_fixed)
            q_proj_bdry = _as_real64("qProjBdry", fiber_proj_aligned(q_diff_bdry))
            q_proj[:, bdry_idx] = apply_basis(bdry_basis, q_proj_bdry) + bdry_fixed
        if int_idx.size > 0:
            q_proj[:, int_idx] = _as_real64("qProjInt", fiber_proj(q_diffused[:, int_idx]))

        dq = q_proj - q
        delta = _mass_norm(dq, M)

        q = q_proj.copy()
        cost = _energy_quadratic(q, L)
        prev_cost = info[-1]["cost"]
        costdelta = prev_cost - cost
        gradnorm = float(np.linalg.norm((L @ q.T), ord="fro"))
        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {
            "tau": tau_k,
            "delta": delta,
            "cost": cost,
            "costdelta": costdelta,
            "gradnorm": gradnorm,
            "time": elapsed,
            "inner_iters": iter_count,
        }
        if save_iterates:
            row["q"] = q.copy()
        info.append(row)

        if verbose:
            print(
                f"t = {elapsed:3.3g}s, cost = {cost:3.6g}, "
                f"delta = {delta:3.3g}, inner iters = {iter_count}"
            )

        cost_rel = np.inf if cost == 0.0 else abs(costdelta / cost)
        q_norm = _mass_norm(q, M)
        delta_rel = np.inf if q_norm == 0.0 else delta / q_norm
        if cost_rel < 1e-5 or delta_rel < 1e-5:
            break

    return q, q0m, info


def MBO(
    meshData: Any,
    fiber: MBOFiber | dict[str, Any] | Callable[[], MBOFiber | dict[str, Any]],
    q0: np.ndarray | None = None,
    tauMult: float = 1.0,
    tauExponent: float = 0.0,
    saveIterates: bool = False,
    *,
    max_iters: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return mbo(
        meshData,
        fiber,
        q0=q0,
        tau_mult=tauMult,
        tau_exponent=tauExponent,
        save_iterates=saveIterates,
        max_iters=max_iters,
        verbose=verbose,
    )
