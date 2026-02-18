from __future__ import annotations

from pathlib import Path
from math import pi

import numpy as np
from scipy.spatial.transform import Rotation

from mesh import ImportMesh
from mbo import MBO, OdecoMBO
from plot import PlotInterpolatedFrames, RadialSamples
from rtr import OdecoManopt


def prism_figures():
    mesh_path = Path(__file__).resolve().parent.parent / "meshes" / "prism" / "prism.node"
    prism = ImportMesh(mesh_path, pi / 2 + 1e-2)

    q_prism_odeco_mbo, _, _ = MBO(prism, OdecoMBO, None, 1, 0)
    q_prism_odeco_mbo_rtr, _, _ = OdecoManopt(prism, q_prism_odeco_mbo)

    samples, _ = RadialSamples(6, 10, pi / 2.0)
    samples = samples / 8.0 + np.array([0.0, np.sqrt(3.0) / 3.0, 0.0], dtype=np.float64)
    samples_stacked = np.vstack(
        [
            samples + np.array([0.0, 0.0, 1.0], dtype=np.float64),
            samples,
            samples + np.array([0.0, 0.0, -1.0], dtype=np.float64),
        ],
    )
    samples_line = np.column_stack(
        [np.zeros(14, dtype=np.float64), np.linspace(0.0, np.sqrt(3.0), 14), np.zeros(14, dtype=np.float64)],
    )
    rand_rot = Rotation.from_euler("xyz", 2.0 * np.pi * np.random.rand(3)).as_matrix()

    PlotInterpolatedFrames(
        q_prism_odeco_mbo_rtr,
        prism,
        samples_stacked,
        ColorField=lambda q: np.linalg.norm(q[1:6, :], axis=0),
    )

    PlotInterpolatedFrames(
        q_prism_odeco_mbo_rtr,
        prism,
        samples,
        NormalColorRot=rand_rot,
    )

    PlotInterpolatedFrames(
        q_prism_odeco_mbo_rtr,
        prism,
        samples_line,
        NormalColorRot=rand_rot,
    )


PrismFigures = prism_figures

__all__ = ["prism_figures", "PrismFigures"]
