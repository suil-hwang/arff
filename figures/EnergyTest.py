from __future__ import annotations

from pathlib import Path

import numpy as np

from arff_io import Octa2Odeco
from mesh import ImportMesh, tet_volumes
from rtr import OctaManopt, OdecoManopt


def energy_test(datadir: str | Path | None = None):
    default_dir = Path(__file__).resolve().parent.parent / "meshes" / "spheres"
    data_path = Path(datadir) if datadir is not None else default_dir
    if not data_path.exists():
        raise FileNotFoundError(f"Sphere mesh directory not found: {data_path}")

    meshfiles = sorted(data_path.glob("sphere*"))
    n_samples = 10
    results = []

    for mesh_file in meshfiles:
        mesh_data = ImportMesh(mesh_file)
        vols = tet_volumes(mesh_data.verts, mesh_data.tets)
        item = {
            "volGeomean": float(np.exp(np.mean(np.log(vols)))),
            "volMean": float(np.mean(vols)),
            "volMax": float(np.max(vols)),
            "volMin": float(np.min(vols)),
            "numTets": int(len(mesh_data.tets)),
            "numVerts": int(mesh_data.nv),
            "EOcta": np.empty(n_samples, dtype=np.float64),
            "EOdeco": np.empty(n_samples, dtype=np.float64),
        }

        for sample_idx in range(n_samples):
            q_octa, _, _ = OctaManopt(mesh_data)
            q_odeco = Octa2Odeco(q_octa)
            energy_octa = 0.5 * float(np.sum(q_odeco.T * (mesh_data.L @ q_odeco.T)))
            item["EOcta"][sample_idx] = energy_octa

            q_odeco_only, _, _ = OdecoManopt(mesh_data)
            energy_odeco = 0.5 * float(np.sum(q_odeco_only.T * (mesh_data.L @ q_odeco_only.T)))
            item["EOdeco"][sample_idx] = energy_odeco

        results.append(item)

    if len(results) > 0:
        vols_rep = np.repeat([item["volGeomean"] for item in results], n_samples)
        e_octa = np.concatenate([item["EOcta"] for item in results], axis=0)
        e_odeco = np.concatenate([item["EOdeco"] for item in results], axis=0)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.scatter(vols_rep, e_octa, s=10, marker=".")
            ax.scatter(vols_rep, e_odeco, s=10, marker=".")
            ax.set_xscale("log")
            ax.set_xlabel("volume (mean)")
            ax.invert_xaxis()
            fig.tight_layout()
            plt.close(fig)
        except Exception:
            pass

    return results


EnergyTest = energy_test

__all__ = ["energy_test", "EnergyTest"]
