from __future__ import annotations

from functools import lru_cache

import numpy as np

from arff_io import coeff2_frames

from ._common import build_triangles_and_tet_tri_idx, unpack_tet_data


def _project_coeff_samples(q: np.ndarray) -> np.ndarray:
    d = q.shape[0]
    try:
        if d == 9:
            from mbo import octa_mbo

            return octa_mbo().proj(q)
        from mbo import odeco_mbo

        return odeco_mbo().proj(q)
    except Exception:
        return q


@lru_cache(maxsize=1)
def _octa_group() -> tuple[np.ndarray, np.ndarray, int]:
    rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    gens = [np.eye(3, dtype=np.float64), rx, ry, rz]

    mats: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    head = 0
    while head < len(mats):
        cur = mats[head]
        head += 1
        for g in gens:
            nxt = cur @ g
            if not any(np.allclose(nxt, m, atol=1e-12, rtol=0.0) for m in mats):
                mats.append(nxt)

    octa = np.stack(mats, axis=2)
    octa_flat = octa.reshape(9, octa.shape[2], order="F")
    eye_idx = int(
        np.argmax(
            np.sum(
                octa_flat * np.eye(3, dtype=np.float64).reshape(9, 1, order="F"),
                axis=0,
            )
        )
    )
    return octa, octa_flat, eye_idx


def _compute_holonomy(vert_frames: np.ndarray, octa: np.ndarray, octa_flat: np.ndarray) -> np.ndarray:
    # MATLAB input layout: (3, 3, m, n), m loop points, n triangles/pages.
    # Internal layout: (3, 3, n, m) for per-step page operations.
    vf = np.transpose(vert_frames, (0, 1, 3, 2)).copy()
    m = vf.shape[3]
    oct_idx = np.zeros((vf.shape[2],), dtype=np.int64)

    for i in range(m):
        j = (i + 1) % m
        a = vf[:, :, :, i]
        b = vf[:, :, :, j]
        a_b = np.moveaxis(a, 2, 0)
        b_b = np.moveaxis(b, 2, 0)
        rij = np.matmul(np.transpose(a_b, (0, 2, 1)), b_b)
        rij_flat = np.moveaxis(rij, 0, 2).reshape(9, -1, order="F")
        oct_idx = np.argmax(octa_flat.T @ rij_flat, axis=0)
        if i == m - 1:
            break
        corr = octa[:, :, oct_idx]
        corr_b = np.moveaxis(corr, 2, 0)
        b_upd = np.matmul(b_b, np.transpose(corr_b, (0, 2, 1)))
        vf[:, :, :, j] = np.moveaxis(b_upd, 0, 2)
    return oct_idx


def extract_singularities(
    frames: np.ndarray,
    tetra,
    *,
    return_graph: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    tet = unpack_tet_data(tetra)
    points = tet.points
    tets = tet.tets
    triangles, tet_tri_idx = build_triangles_and_tet_tri_idx(tets)
    n_tri = triangles.shape[0]

    octa, octa_flat, eye_idx = _octa_group()
    q = None
    d = None

    if frames.ndim == 2:
        q = np.asarray(frames, dtype=np.float64)
        if q.shape[1] != points.shape[0]:
            raise ValueError("Coefficient field must have one column per tetra vertex.")
        d = q.shape[0]

        tri_edges = np.vstack(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        )
        tri_edges = np.sort(tri_edges, axis=1)
        edges, tri_edge_inv = np.unique(tri_edges, axis=0, return_inverse=True)
        tri_edge_idx = tri_edge_inv.reshape(-1, 3)
        n_edges = edges.shape[0]

        q_e0 = q[:, edges[:, 0]]
        q_e1 = q[:, edges[:, 1]]
        mid1 = (2.0 / 3.0) * q_e0 + (1.0 / 3.0) * q_e1
        mid2 = (1.0 / 3.0) * q_e0 + (2.0 / 3.0) * q_e1
        q_mid_flat = np.empty((d, 2 * n_edges), dtype=np.float64)
        q_mid_flat[:, 0::2] = mid1
        q_mid_flat[:, 1::2] = mid2
        q_mid = _project_coeff_samples(q_mid_flat).reshape(d, 2, n_edges, order="F")

        q_tri = q[:, triangles.T]  # (d, 3, nTri)
        q_loop = np.concatenate(
            [
                q_tri[:, [0], :],
                q_mid[:, :, tri_edge_idx[:, 0]],
                q_tri[:, [1], :],
                q_mid[:, :, tri_edge_idx[:, 1]],
                q_tri[:, [2], :],
                q_mid[:, :, tri_edge_idx[:, 2]],
            ],
            axis=1,
        )  # (d, 9, nTri)
        f_loop = coeff2_frames(q_loop.reshape(d, -1, order="F"), normalize=True)
        f_loop = f_loop.reshape(3, 3, 9, n_tri, order="F")
        holonomy = _compute_holonomy(f_loop, octa, octa_flat)
    else:
        f = np.asarray(frames, dtype=np.float64)
        if f.ndim != 3 or f.shape[0] != 3 or f.shape[1] != 3:
            raise ValueError("frames must have shape (3, 3, n) or coefficients (d, n).")
        if f.shape[2] != points.shape[0]:
            raise ValueError("frames must have one frame per tetra vertex.")
        f_tri = f[:, :, triangles.T]  # (3, 3, 3, nTri)
        holonomy = _compute_holonomy(f_tri, octa, octa_flat)

    sing_tri_mask = holonomy != eye_idx
    sing_tri_type = holonomy[sing_tri_mask]
    sing_tri = np.where(sing_tri_mask)[0]
    sing_tet_mask = np.any(sing_tri_mask[tet_tri_idx], axis=1)
    sing_tet = np.where(sing_tet_mask)[0]

    if not return_graph:
        return sing_tet, sing_tri, sing_tri_type, None, None

    if q is None or d is None:
        raise ValueError("return_graph=True requires coefficient input with shape (d, n).")

    subtri = np.array(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0]],
            [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
        ],
        dtype=np.float64,
    )  # (4, 3, 3)
    loop = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.75, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
            [0.25, 0.0, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.0, 0.25],
        ],
        dtype=np.float64,
    )  # (12, 3)

    tri_active = sing_tri.copy()
    tri_verts = points[triangles[tri_active]]  # (k, 3, 3)
    tri_q = np.transpose(q[:, triangles[tri_active]], (1, 0, 2))  # (k, d, 3)

    # Precompute subtri-loop composition: subtri_loop[s] = subtri[s].T @ loop.T
    # q_sub @ loop.T = (tri_q @ subtri[s].T) @ loop.T = tri_q @ (subtri[s].T @ loop.T)
    subtri_loop = np.einsum("sji,lj->sil", subtri, loop)  # (4, 3, 12)

    for _ in range(4):
        n_sing = tri_active.shape[0]
        if n_sing == 0:
            tri_verts = np.zeros((0, 3, 3), dtype=np.float64)
            tri_q = np.zeros((0, d, 3), dtype=np.float64)
            break

        # All 4 subtriangles × all n_sing triangles at once
        # q_sub[k, s] = tri_q[k] @ subtri[s].T → (k, 4, d, 3)
        q_sub = np.einsum("kdj,smj->ksdm", tri_q, subtri)

        # v_sub[k, s] = subtri[s] @ tri_verts[k] → (k, 4, 3, 3)
        v_sub = np.einsum("smj,kjx->ksmx", subtri, tri_verts)

        # q_loop = q_sub @ loop.T → (k, 4, d, 12)
        q_loop = np.einsum("ksdj,lj->ksdl", q_sub, loop)

        # Project and compute holonomy for all 4*n_sing subtriangles
        total = n_sing * 4
        q_loop_flat = q_loop.reshape(total, d, 12)
        q_proj_all = _project_coeff_samples(q_loop_flat.reshape(d, total * 12))
        q_proj_all = q_proj_all.reshape(d, total, 12, order="F")
        q_proj_all = np.moveaxis(q_proj_all, 1, 0)  # (total, d, 12)

        f_all = np.empty((total, 3, 3, 12), dtype=np.float64)
        for idx in range(total):
            f_all[idx] = coeff2_frames(q_proj_all[idx], normalize=True).reshape(
                3, 3, 12, order="F"
            )

        hol_all = np.empty((total,), dtype=np.int64)
        for idx in range(total):
            hol_all[idx] = _compute_holonomy(
                f_all[idx][:, :, :, None], octa, octa_flat
            )[0]

        hol_mat = hol_all.reshape(n_sing, 4)
        sub_is_singular = hol_mat != eye_idx
        is_singular = np.any(sub_is_singular, axis=1)

        # Take first singular subtriangle per triangle (MATLAB: max(..., 'linear'))
        subtri_choice = np.argmax(sub_is_singular, axis=1)

        keep = np.where(is_singular)[0]
        if keep.size == 0:
            tri_active = np.zeros((0,), dtype=np.int64)
            tri_verts = np.zeros((0, 3, 3), dtype=np.float64)
            tri_q = np.zeros((0, d, 3), dtype=np.float64)
            break

        chosen_s = subtri_choice[keep]
        tri_active = tri_active[keep]
        tri_q = q_sub[keep, chosen_s]  # (keep_count, d, 3)
        tri_verts = v_sub[keep, chosen_s]  # (keep_count, 3, 3)

    sing_points = np.mean(tri_verts, axis=1) if tri_verts.size > 0 else np.zeros((0, 3), dtype=np.float64)
    sing_tri_refined = tri_active

    sing_tri_order = np.full((n_tri,), -1, dtype=np.int64)
    sing_tri_order[sing_tri_refined] = np.arange(sing_tri_refined.shape[0], dtype=np.int64)
    sing_edges = sing_tri_order[tet_tri_idx[sing_tet]]
    sing_edges = np.sort(sing_edges, axis=1)
    return sing_tet, sing_tri_refined, sing_tri_type, sing_points, sing_edges


def ExtractSingularities(frames: np.ndarray, tetra, returnGraph: bool = False):
    return extract_singularities(frames, tetra, return_graph=returnGraph)
