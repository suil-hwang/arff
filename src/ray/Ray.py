from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from arff_io import coeff2_frames
from mesh import geometric_primal_lm
from variety import load_so3_generators_y4, octa_align_mat, octa_z_aligned

from .RayInit import ray_init
from ._core import _as_real64

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
_BANDS = np.arange(-4.0, 5.0, dtype=np.float64)[:, None]


class _StopByGradnorm(RuntimeError):
    pass


def _mesh_get(mesh_data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(mesh_data, dict) and name in mesh_data:
            return mesh_data[name]
        if hasattr(mesh_data, name):
            return getattr(mesh_data, name)
    raise ValueError(f"Missing mesh field. Tried: {names!r}")


def _as_index(name: str, x: np.ndarray, n: int) -> np.ndarray:
    idx = np.asarray(x, dtype=np.int64).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"{name} contains out-of-range indices.")
    return idx


def _to_zero_based_tets(tets: np.ndarray, nv: int) -> np.ndarray:
    tet = np.asarray(tets, dtype=np.int64)
    if tet.ndim != 2 or tet.shape[1] != 4:
        raise ValueError("tets must have shape (nt, 4).")
    if tet.size > 0 and tet.min() == 1 and tet.max() <= nv:
        tet = tet - 1
    if tet.size > 0 and (tet.min() < 0 or tet.max() >= nv):
        raise ValueError("tets contains out-of-range indices.")
    return tet


def _build_combinatorial_laplacian(tets: np.ndarray, nv: int) -> sp.csr_matrix:
    tet = _to_zero_based_tets(tets, nv)
    if tet.size == 0:
        return sp.csr_matrix((nv, nv), dtype=np.float64)

    edges = tet[:, _EDGE_LOCAL].reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.ones(rows.size, dtype=np.float64)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(nv, nv), dtype=np.float64).tocsr()
    deg = np.asarray(adj.sum(axis=1), dtype=np.float64).reshape(-1)
    return sp.diags(deg, offsets=0, shape=(nv, nv), dtype=np.float64) - adj


def _rotate_z(eul_z: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return q.copy()
    eul = np.asarray(eul_z, dtype=np.float64).reshape(1, -1)
    if eul.shape[1] != q.shape[1]:
        raise ValueError("eul_z must have one entry per q column.")
    phase = _BANDS * eul
    q_flip = np.flipud(q)
    return np.cos(phase) * q - np.sin(phase) * q_flip


def _rotate_y(eul_y: np.ndarray, q: np.ndarray, YZ: np.ndarray) -> np.ndarray:
    return YZ.T @ _rotate_z(eul_y, YZ @ q)


def _rotate_x(eul_x: np.ndarray, q: np.ndarray, XZ: np.ndarray, YZ: np.ndarray) -> np.ndarray:
    del YZ
    return XZ.T @ _rotate_z(eul_x, XZ @ q)


def ray(
    mesh_data: Any,
    q_init: np.ndarray | None = None,
    use_geometric_laplacian: bool = False,
    time_initialization: bool = True,
    *,
    max_iterations: int = 5000,
    max_function_evaluations: int = 5000,
    gradnorm_stop: float = 1e-6,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    del verbosity
    nv = int(_mesh_get(mesh_data, "nv"))
    if nv <= 0:
        raise ValueError("mesh_data.nv must be positive.")

    bdry_idx = _as_index("bdry_idx", _mesh_get(mesh_data, "bdry_idx", "bdryIdx"), nv)
    int_idx = _as_index("int_idx", _mesh_get(mesh_data, "int_idx", "intIdx"), nv)
    bdry_normals = _as_real64("bdry_normals", _mesh_get(mesh_data, "bdry_normals", "bdryNormals"))
    if bdry_normals.shape != (bdry_idx.size, 3):
        raise ValueError("bdry_normals must have shape (len(bdry_idx), 3).")

    tets = _to_zero_based_tets(_mesh_get(mesh_data, "tets", "tetra"), nv)
    if use_geometric_laplacian:
        verts = _as_real64("verts", _mesh_get(mesh_data, "verts"))
        L, _ = geometric_primal_lm(verts, tets)
    else:
        L = _build_combinatorial_laplacian(tets, nv)

    Lx, Ly, Lz, YZ = load_so3_generators_y4()
    XZ = sla.expm(-(np.pi / 2.0) * Ly)
    B = octa_align_mat(bdry_normals)
    q0 = np.repeat(octa_z_aligned(np.array([0.0], dtype=np.float64)), nv, axis=1)

    if time_initialization:
        t0 = _time.perf_counter()

    q_init_used: np.ndarray
    f_init: np.ndarray
    if q_init is not None:
        q_init_arr = _as_real64("qInit", q_init)
        if q_init_arr.size == 0:
            q_init_arr = None
        elif q_init_arr.shape != (9, nv):
            raise ValueError(f"qInit must have shape (9, {nv}).")
        if q_init_arr is not None:
            q_init_used = q_init_arr
            f_init = coeff2_frames(q_init_used)
        else:
            q_init_used, f_init = ray_init(mesh_data)
    else:
        q_init_used, f_init = ray_init(mesh_data)

    if not time_initialization:
        t0 = _time.perf_counter()

    ni = int_idx.size
    nb = bdry_idx.size
    if ni > 0:
        rotm = np.transpose(f_init[:, :, int_idx], (2, 0, 1))
        eul_init_int = np.fliplr(Rotation.from_matrix(rotm).as_euler("ZYX"))
    else:
        eul_init_int = np.zeros((0, 3), dtype=np.float64)

    if nb > 0:
        q_init_bdry = np.einsum("abn,an->bn", B, q_init_used[:, bdry_idx], optimize=True)
        eul_init_bdry = -0.25 * np.arctan2(-q_init_bdry[0, :], q_init_bdry[8, :])
    else:
        eul_init_bdry = np.zeros((0,), dtype=np.float64)

    eul_init = np.concatenate([eul_init_bdry, eul_init_int.reshape(-1, order="F")])

    def split_bdry_int(eul: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eul2 = np.asarray(eul, dtype=np.float64).reshape(-1)
        if eul2.shape[0] != nb + 3 * ni:
            raise ValueError("Euler parameter vector has invalid size.")
        eul_bdry = eul2[:nb]
        eul_int = eul2[nb:].reshape((ni, 3), order="F")
        return eul_bdry, eul_int

    def q0_rotated_by(
        eul_bdry: np.ndarray, eul_int: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if nb > 0:
            Rbq0 = _rotate_z(eul_bdry, q0[:, bdry_idx])
            BRbq0 = np.einsum("abn,an->bn", B, Rbq0, optimize=True)
        else:
            Rbq0 = np.zeros((9, 0), dtype=np.float64)
            BRbq0 = np.zeros((9, 0), dtype=np.float64)

        if ni > 0:
            Rxq0 = _rotate_x(eul_int[:, 0], q0[:, int_idx], XZ, YZ)
            RyRxq0 = _rotate_y(eul_int[:, 1], Rxq0, YZ)
            RzRyRxq0 = _rotate_z(eul_int[:, 2], RyRxq0)
        else:
            Rxq0 = np.zeros((9, 0), dtype=np.float64)
            RyRxq0 = np.zeros((9, 0), dtype=np.float64)
            RzRyRxq0 = np.zeros((9, 0), dtype=np.float64)
        return RzRyRxq0, BRbq0, Rbq0, RyRxq0, Rxq0

    def ray_obj_grad(eul: np.ndarray) -> tuple[float, np.ndarray]:
        eul_bdry, eul_int = split_bdry_int(eul)
        q_int, q_bdry, Rbq0, RyRxq0, Rxq0 = q0_rotated_by(eul_bdry, eul_int)

        qi = np.zeros((9, nv), dtype=np.float64)
        if nb > 0:
            qi[:, bdry_idx] = q_bdry
        if ni > 0:
            qi[:, int_idx] = q_int

        egrad = np.asarray(L @ qi.T, dtype=np.float64)
        obj = float(0.5 * np.sum(qi.T * egrad))

        grad_bdry = np.zeros((nb,), dtype=np.float64)
        if nb > 0:
            Dbq = np.einsum("abn,an->bn", B, Lz @ Rbq0, optimize=True)
            grad_bdry = np.sum(Dbq.T * egrad[bdry_idx, :], axis=1)

        if ni > 0:
            Dzq = Lz @ q_int
            Dyq = _rotate_z(eul_int[:, 2], Ly @ RyRxq0)
            Dxq = _rotate_z(eul_int[:, 2], _rotate_y(eul_int[:, 1], Lx @ Rxq0, YZ))
            egrad_int = egrad[int_idx, :]
            grad_x = np.sum(egrad_int.T * Dxq, axis=0)
            grad_y = np.sum(egrad_int.T * Dyq, axis=0)
            grad_z = np.sum(egrad_int.T * Dzq, axis=0)
            grad_int = np.column_stack([grad_x, grad_y, grad_z])
            grad = np.concatenate([grad_bdry, grad_int.reshape(-1, order="F")])
        else:
            grad = grad_bdry

        return obj, grad

    info: list[dict[str, Any]] = []
    latest_x: np.ndarray | None = None

    def callback(xk: np.ndarray) -> None:
        nonlocal latest_x
        latest_x = np.asarray(xk, dtype=np.float64).copy()
        cost, grad = ray_obj_grad(latest_x)
        gradnorm = float(np.linalg.norm(grad))
        info.append({"cost": cost, "gradnorm": gradnorm, "time": _time.perf_counter() - t0})
        if gradnorm < gradnorm_stop:
            raise _StopByGradnorm

    options = {
        "maxiter": int(max_iterations),
        "maxfun": int(max_function_evaluations),
        "gtol": 0.0,
        "ftol": 0.0,
    }

    try:
        result = minimize(
            fun=lambda x: ray_obj_grad(x)[0],
            x0=eul_init,
            jac=lambda x: ray_obj_grad(x)[1],
            method="L-BFGS-B",
            callback=callback,
            options=options,
        )
        eul = np.asarray(result.x, dtype=np.float64)
    except _StopByGradnorm:
        eul = latest_x.copy() if latest_x is not None else eul_init.copy()

    if not info:
        cost, grad = ray_obj_grad(eul)
        info.append({"cost": cost, "gradnorm": float(np.linalg.norm(grad)), "time": _time.perf_counter() - t0})

    eul_bdry, eul_int = split_bdry_int(eul)
    q_int, q_bdry, _, _, _ = q0_rotated_by(eul_bdry, eul_int)
    q = np.zeros((9, nv), dtype=np.float64)
    if nb > 0:
        q[:, bdry_idx] = q_bdry
    if ni > 0:
        q[:, int_idx] = q_int
    return q, q_init_used, info


def Ray(
    meshData: Any,
    qInit: np.ndarray | None = None,
    useGeometricLaplacian: bool = False,
    timeInitialization: bool = True,
    *,
    max_iterations: int = 5000,
    max_function_evaluations: int = 5000,
    gradnorm_stop: float = 1e-6,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return ray(
        meshData,
        q_init=qInit,
        use_geometric_laplacian=useGeometricLaplacian,
        time_initialization=timeInitialization,
        max_iterations=max_iterations,
        max_function_evaluations=max_function_evaluations,
        gradnorm_stop=gradnorm_stop,
        verbosity=verbosity,
    )
