from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .TetVolumes import tet_volumes

_TET_TO_EDGE = np.array(
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


def geometric_primal_lm(
    verts: np.ndarray, tets: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    verts = np.asarray(verts, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must have shape (n, 3).")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (m, 4).")

    nv = verts.shape[0]
    tets = np.sort(tets, axis=1)
    nt = tets.shape[0]

    tet_edges = tets[:, _TET_TO_EDGE]
    tet_edges_opp = np.flip(tet_edges, axis=1)

    v0 = verts[tet_edges[:, :, 0]]
    v1 = verts[tet_edges[:, :, 1]]
    w0 = verts[tet_edges_opp[:, :, 0]]
    w1 = verts[tet_edges_opp[:, :, 1]]

    opp_edge_vec = w1 - w0
    opp_edge_len = np.linalg.norm(opp_edge_vec, axis=2)
    safe_opp_edge_len = np.where(opp_edge_len > 0.0, opp_edge_len, 1.0)
    opp_edge_unit = opp_edge_vec / safe_opp_edge_len[:, :, None]

    n0 = np.cross(opp_edge_unit, v0 - w1)
    n1 = np.cross(opp_edge_unit, v1 - w1)
    n0_norm = np.linalg.norm(n0, axis=2)
    n1_norm = np.linalg.norm(n1, axis=2)
    n0 = n0 / np.where(n0_norm > 0.0, n0_norm, 1.0)[:, :, None]
    n1 = n1 / np.where(n1_norm > 0.0, n1_norm, 1.0)[:, :, None]
    t1 = np.cross(opp_edge_unit, n1)

    alpha = np.abs(
        np.arctan2(
            np.einsum("ijk,ijk->ij", n0, t1),
            np.einsum("ijk,ijk->ij", n0, n1),
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        opposite_cotans = 1.0 / np.tan(alpha)
    opposite_cotans = np.nan_to_num(opposite_cotans, nan=0.0, posinf=0.0, neginf=0.0)

    Lij = opp_edge_len * opposite_cotans / 6.0
    edge_rows = tet_edges[:, :, 0].reshape(-1)
    edge_cols = tet_edges[:, :, 1].reshape(-1)
    L = sp.coo_matrix((Lij.reshape(-1), (edge_rows, edge_cols)), shape=(nv, nv)).tocsr()
    L = L + L.T
    L = sp.diags(np.asarray(L.sum(axis=1)).reshape(-1)) - L

    Mij = np.abs(
        np.einsum("ijk,ijk->ij", v1 - v0, np.cross(w0 - v0, w1 - v0))
    ) / 120.0
    Mii = np.repeat(tet_volumes(verts, tets) / 10.0, 4)
    diag_idx = tets.reshape(-1)

    M = sp.coo_matrix((Mij.reshape(-1), (edge_rows, edge_cols)), shape=(nv, nv)).tocsr()
    M = (
        M
        + M.T
        + sp.coo_matrix((Mii, (diag_idx, diag_idx)), shape=(nv, nv)).tocsr()
    )
    return L.tocsr(), M.tocsr()


GeometricPrimalLM = geometric_primal_lm

