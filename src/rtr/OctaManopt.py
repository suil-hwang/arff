from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import scipy.sparse as sp
from pymanopt import Problem
from pymanopt.function import numpy as pnp
from pymanopt.optimizers import TrustRegions

from .OctahedralBundleFactory import OctahedralBundleManifold, octahedral_bundle_factory
from ._utils import as_index, as_real64, matvec_rows, matrix_diag_inverse, mesh_get


def _build_combinatorial_laplacian(tetra: np.ndarray, nv: int) -> sp.csr_matrix:
    tet = np.asarray(tetra, dtype=np.int64)
    if tet.ndim != 2 or tet.shape[1] != 4:
        raise ValueError("mesh_data.tetra must have shape (nt, 4).")
    if tet.size == 0:
        return sp.csr_matrix((nv, nv), dtype=np.float64)

    if tet.min() == 1 and tet.max() <= nv:
        tet = tet - 1
    if tet.min() < 0 or tet.max() >= nv:
        raise ValueError("mesh_data.tetra contains out-of-range indices.")

    edges = np.concatenate(
        [
            tet[:, [0, 1]],
            tet[:, [0, 2]],
            tet[:, [0, 3]],
            tet[:, [1, 2]],
            tet[:, [1, 3]],
            tet[:, [2, 3]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.size, dtype=np.float64)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(nv, nv), dtype=np.float64).tocsr()
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).reshape(-1)
    return sp.diags(degree, offsets=0, shape=(nv, nv), dtype=np.float64) - adjacency


def _as_square_matrix(name: str, A: sp.spmatrix | np.ndarray, n: int) -> sp.spmatrix | np.ndarray:
    if not (sp.issparse(A) or isinstance(A, np.ndarray)):
        raise ValueError(f"{name} must be a dense ndarray or scipy sparse matrix.")
    if A.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}).")
    return A


def octa_manopt(
    mesh_data: Any,
    q0: np.ndarray | None = None,
    save_iterates: bool = False,
    gpuflag: bool = False,
    use_combinatorial_laplacian: bool = False,
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

    manifold: OctahedralBundleManifold = octahedral_bundle_factory(nv, bdry_idx, bdry_normals)
    base_A = _as_square_matrix("mesh_data.L", mesh_get(mesh_data, "L"), nv)
    if use_combinatorial_laplacian:
        tetra = mesh_get(mesh_data, "tetra")
        A = _build_combinatorial_laplacian(tetra, nv)
    else:
        A = base_A
    inv_diag_a = matrix_diag_inverse(A)

    # -- Cache (MATLAB store pattern via closure) --
    # Matches MATLAB's isfield(store, ...) pattern: each field is computed
    # lazily on first access.  cost() only needs Aq; gradient() adds basis
    # and rgrad; hessian() adds lxyz_t.  This avoids recomputing expensive
    # fields when the TR solver evaluates cost for a trial point.
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
        """Ensure Aq, basis, rgrad are cached (for gradient)."""
        s = _prepare(q)
        if "basis" not in s:
            s["basis"] = manifold.tangentbasis(q)
        if "rgrad" not in s:
            s["rgrad"] = manifold.egrad2rgrad(q, s["aq"], s["basis"])
        return s

    def _prepare_hess(q: np.ndarray) -> dict[str, Any]:
        """Ensure all fields are cached (for hessian)."""
        s = _prepare_grad(q)
        if "lxyz_t" not in s:
            s["lxyz_t"] = manifold.mul_lxyz_t(s["aq"])
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
        s = _prepare_hess(q)
        s_ambient = manifold.tangent2ambient(
            q, tangent_vector, s["basis"],
        )
        ehess = matvec_rows(A, s_ambient)
        return manifold.ehess2rhess(
            q,
            s["aq"],
            ehess,
            tangent_vector,
            s_ambient,
            s["rgrad"],
            s["lxyz_t"],
            s["basis"],
        )

    def precon(_: np.ndarray, tangent_vector: np.ndarray) -> np.ndarray:
        tv = np.asarray(tangent_vector, dtype=np.float64)
        return inv_diag_a[:, None] * tv

    problem = Problem(
        manifold=manifold,
        cost=cost,
        riemannian_gradient=gradient,
        riemannian_hessian=hessian,
        preconditioner=precon,
    )

    if q0 is None:
        initial = manifold.random_point()
        q0_used = np.sqrt(20.0 / 3.0) * initial
    else:
        q0_used = as_real64("q0", q0)
        if q0_used.shape != (9, nv):
            raise ValueError(f"q0 must have shape (9, {nv}).")
        initial = np.sqrt(3.0 / 20.0) * q0_used

    optimizer = TrustRegions(
        max_iterations=int(max_iterations),
        min_gradient_norm=float(min_gradient_norm),
        verbosity=int(verbosity),
    )
    result = optimizer.run(problem, initial_point=initial)
    q = np.sqrt(20.0 / 3.0) * np.asarray(result.point, dtype=np.float64)

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


def OctaManopt(
    meshData: Any,
    q0: np.ndarray | None = None,
    saveIterates: bool = False,
    gpuflag: bool = False,
    useCombinatorialLaplacian: bool = False,
    *,
    max_iterations: int = 1000,
    min_gradient_norm: float = 1e-6,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return octa_manopt(
        meshData,
        q0=q0,
        save_iterates=saveIterates,
        gpuflag=gpuflag,
        use_combinatorial_laplacian=useCombinatorialLaplacian,
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
        verbosity=verbosity,
    )
