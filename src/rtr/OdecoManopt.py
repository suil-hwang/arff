from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import scipy.sparse as sp
from pymanopt import Problem
from pymanopt.function import numpy as pnp
from pymanopt.optimizers import TrustRegions

from .OdecoBundleFactory import OdecoBundleManifold, odeco_bundle_factory
from ._utils import as_index, as_real64, matvec_rows, mesh_get


def _as_square_matrix(name: str, A: sp.spmatrix | np.ndarray, n: int) -> sp.spmatrix | np.ndarray:
    if not (sp.issparse(A) or isinstance(A, np.ndarray)):
        raise ValueError(f"{name} must be a dense ndarray or scipy sparse matrix.")
    if A.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}).")
    return A


def odeco_manopt(
    mesh_data: Any,
    q0: np.ndarray | None = None,
    save_iterates: bool = False,
    gpuflag: bool = False,
    *,
    max_iterations: int = 1000,
    min_gradient_norm: float = 1e-6,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    del gpuflag
    nv = int(mesh_get(mesh_data, "nv"))
    if nv <= 0:
        raise ValueError("mesh_data.nv must be positive.")

    bdry_idx = as_index("bdry_idx", mesh_get(mesh_data, "bdry_idx", "bdryIdx"), nv)
    bdry_normals = as_real64("bdry_normals", mesh_get(mesh_data, "bdry_normals", "bdryNormals"))
    if bdry_normals.shape != (bdry_idx.size, 3):
        raise ValueError("bdry_normals must have shape (len(bdry_idx), 3).")

    manifold: OdecoBundleManifold = odeco_bundle_factory(nv, bdry_idx, bdry_normals)
    A = _as_square_matrix("mesh_data.L", mesh_get(mesh_data, "L"), nv)

    # -- Cache (MATLAB store pattern via closure) --
    # Matches MATLAB's isfield(store, ...) pattern: Aq is computed lazily
    # for cost(); rgrad/cache/grad_yo are added only when gradient or
    # hessian need them.  This avoids the expensive egrad2rgrad +
    # _build_projection_cache (~130 MB temporaries) when the TR solver
    # only evaluates cost for a trial point.
    _store: dict[str, Any] = {}
    _iter_log: list[dict[str, Any]] = []
    _t0 = _time.perf_counter()

    def _prepare(q: np.ndarray) -> dict[str, Any]:
        """Ensure Aq is cached (minimum for cost evaluation)."""
        if _store.get("_q_id") != id(q):
            _store.clear()
            _store["_q_id"] = id(q)
        if "aq" not in _store:
            _store["aq"] = matvec_rows(A, q)
        return _store

    def _prepare_grad(q: np.ndarray) -> dict[str, Any]:
        """Ensure Aq, rgrad, cache, grad_yo are cached."""
        s = _prepare(q)
        if "rgrad" not in s:
            rgrad, cache, grad_yo = manifold.egrad2rgrad(
                q, s["aq"], return_y_o=True,
            )
            s["rgrad"] = rgrad
            s["cache"] = cache
            s["grad_yo"] = grad_yo
        return s

    @pnp(manifold)
    def cost(q: np.ndarray) -> float:
        s = _prepare(q)
        return float(0.5 * np.sum(q * s["aq"]))

    @pnp(manifold)
    def gradient(q: np.ndarray) -> np.ndarray:
        s = _prepare_grad(q)
        rgrad = s["rgrad"]
        _iter_log.append({
            "cost": float(0.5 * np.sum(q * s["aq"])),
            "gradnorm": float(np.linalg.norm(rgrad)),
            "time": _time.perf_counter() - _t0,
        })
        return rgrad

    @pnp(manifold)
    def hessian(q: np.ndarray, tangent_vector: np.ndarray) -> np.ndarray:
        s = _prepare_grad(q)
        ehess = matvec_rows(A, tangent_vector)
        return manifold.ehess2rhess(
            q, s["aq"], ehess, tangent_vector, s["cache"], s["grad_yo"],
        )

    problem = Problem(
        manifold=manifold,
        cost=cost,
        riemannian_gradient=gradient,
        riemannian_hessian=hessian,
    )

    if q0 is None:
        initial = manifold.random_point()
        q0_used = initial.copy()
    else:
        q0_used = as_real64("q0", q0)
        if q0_used.shape != (15, nv):
            raise ValueError(f"q0 must have shape (15, {nv}).")
        initial = q0_used

    optimizer = TrustRegions(
        max_iterations=int(max_iterations),
        min_gradient_norm=float(min_gradient_norm),
        verbosity=int(verbosity),
    )
    result = optimizer.run(
        problem,
        initial_point=initial,
        Delta_bar=float(np.sqrt(6.0 * nv)),
    )
    q = np.asarray(result.point, dtype=np.float64)

    # -- Build per-iteration info (matches MATLAB info struct array) --
    info: list[dict[str, Any]] = []
    for i, entry in enumerate(_iter_log):
        row: dict[str, Any] = {
            "iter": i,
            "cost": entry["cost"],
            "gradnorm": entry["gradnorm"],
            "time": entry["time"],
        }
        if save_iterates and i == len(_iter_log) - 1:
            row["q"] = q.copy()
        info.append(row)
    if not info:
        info.append({
            "iter": 0,
            "cost": float(result.cost),
            "gradnorm": float(result.gradient_norm) if result.gradient_norm is not None else np.nan,
            "time": float(result.time),
            "stopping_criterion": str(result.stopping_criterion),
        })
    info[-1]["stopping_criterion"] = str(result.stopping_criterion)

    return q, q0_used, info


def OdecoManopt(
    meshData: Any,
    q0: np.ndarray | None = None,
    saveIterates: bool = False,
    gpuflag: bool = False,
    *,
    max_iterations: int = 1000,
    min_gradient_norm: float = 1e-6,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return odeco_manopt(
        meshData,
        q0=q0,
        save_iterates=saveIterates,
        gpuflag=gpuflag,
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
        verbosity=verbosity,
    )
