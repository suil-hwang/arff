from __future__ import annotations

import numpy as np

from sdp import MultiSdp
from variety import LoadOctaMatsScaled


def octa_exactness_test(n):
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    n_int = int(n)

    q0 = np.random.randn(9, n_int)
    q0 = q0 / np.linalg.norm(q0, axis=0, keepdims=True)

    octa_mat = np.stack(LoadOctaMatsScaled(), axis=2)
    first = np.zeros((10, 10), dtype=np.float64)
    first[0, 0] = 1.0
    sdp_a = np.concatenate([first[:, :, None], octa_mat], axis=2)
    sdp_a = np.reshape(sdp_a, (10 * 10, 16), order="F").T
    sdp_b = np.concatenate([np.array([1.0]), np.zeros(15, dtype=np.float64)])[:, None]

    q, q_flat = MultiSdp(q0, sdp_a, sdp_b, return_Q=True)
    q = np.reshape(q, (9, n_int), order="F")
    Q = np.reshape(q_flat, (10, 10, n_int), order="F")

    first_eig = np.empty(n_int, dtype=np.float64)
    second_eig = np.empty(n_int, dtype=np.float64)
    for idx in range(n_int):
        eigvals = np.linalg.eigvalsh(Q[:, :, idx])
        first_eig[idx] = eigvals[9]
        second_eig[idx] = eigvals[8]

    worst_eig2 = float(np.max(second_eig))
    eig_ratio = second_eig / first_eig

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


OctaExactnessTest = octa_exactness_test

__all__ = ["octa_exactness_test", "OctaExactnessTest"]
