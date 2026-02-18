from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from sdp import MultiSdp
from variety import LoadOdecoMatsSph


_CLEBSCH_CACHE: np.ndarray | None = None


def _eval_matlab_expr(expr: str) -> float:
    expr2 = expr.strip()
    if not expr2:
        return 0.0
    expr2 = expr2.replace("...", "")
    expr2 = re.sub(r"\s*\n\s*", " ", expr2)
    expr2 = expr2.replace("^", "**")
    expr2 = expr2.replace(".*", "*")
    expr2 = re.sub(r"\bpi\b", "np.pi", expr2)
    expr2 = expr2.replace("sqrt", "np.sqrt")
    return float(eval(expr2, {"np": np}, {}))


def _parse_clebsch_gordan() -> np.ndarray:
    global _CLEBSCH_CACHE
    if _CLEBSCH_CACHE is not None:
        return _CLEBSCH_CACHE

    mfile = Path(__file__).with_name("OdecoExactnessTest.m")
    text = mfile.read_text(encoding="utf-8")
    pattern = re.compile(r"clebschGordan\(:,:,(\d+)\)\s*=\s*\[(.*?)\];", re.S)
    matches = pattern.findall(text)
    if len(matches) < 15:
        raise RuntimeError("Could not parse clebschGordan from OdecoExactnessTest.m")

    matrices: list[np.ndarray] = []
    for _, raw in sorted(matches, key=lambda item: int(item[0])):
        body = raw.replace("...", "")
        rows = []
        for row in body.split(";"):
            row2 = row.strip()
            if not row2:
                continue
            entries = [entry for entry in row2.split(",") if entry.strip()]
            values = [_eval_matlab_expr(entry) for entry in entries]
            rows.append(values)
        mat = np.array(rows, dtype=np.float64)
        if mat.shape != (6, 6):
            raise ValueError(f"Parsed clebsch block with invalid shape {mat.shape}.")
        matrices.append(mat)

    if len(matrices) != 15:
        raise RuntimeError(f"Expected 15 clebsch blocks, found {len(matrices)}.")

    _CLEBSCH_CACHE = np.stack(matrices, axis=0)
    return _CLEBSCH_CACHE


def odeco_exactness_test(n):
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    n_int = int(n)

    clebsch = _parse_clebsch_gordan()  # (15, 6, 6)

    rand_m = np.random.randn(6, 6, n_int)
    rand_psd = np.einsum("jin,jkn->ikn", rand_m, rand_m, optimize=True)
    q0 = np.einsum("ijn,aij->an", rand_psd, clebsch, optimize=True)
    q0 = q0 / np.linalg.norm(q0, axis=0, keepdims=True)

    odeco_mat = np.stack(LoadOdecoMatsSph(), axis=2)
    n_constr = odeco_mat.shape[2]
    odeco_block = np.zeros((16, 16, n_constr), dtype=np.float64)
    odeco_block[1:, 1:, :] = odeco_mat
    head = np.zeros((16, 16), dtype=np.float64)
    head[0, 0] = 1.0
    sdp_a = np.concatenate([odeco_block, head[:, :, None]], axis=2)
    sdp_a = np.reshape(sdp_a, (16 * 16, n_constr + 1), order="F").T
    sdp_b = np.concatenate([np.zeros((n_constr, 1), dtype=np.float64), np.array([[1.0]])], axis=0)

    q, q_flat = MultiSdp(q0, sdp_a, sdp_b, return_Q=True)
    q = np.reshape(q, (15, n_int), order="F")
    Q = np.reshape(q_flat, (16, 16, n_int), order="F")

    first_eig = np.empty(n_int, dtype=np.float64)
    second_eig = np.empty(n_int, dtype=np.float64)
    for idx in range(n_int):
        eigvals = np.linalg.eigvalsh(Q[:, :, idx])
        first_eig[idx] = eigvals[-1]
        second_eig[idx] = eigvals[-2]

    eig_ratio = second_eig / first_eig
    worst_eig2 = float(np.max(second_eig))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(np.log10(eig_ratio), bins=40)
        ax.set_ylabel("count")
        ax.set_xlabel("log10(second/first eigenvalue)")
        fig.tight_layout()
        plt.close(fig)
    except Exception:
        pass

    return worst_eig2, eig_ratio, q0, q, Q


OdecoExactnessTest = odeco_exactness_test

__all__ = ["odeco_exactness_test", "OdecoExactnessTest"]
