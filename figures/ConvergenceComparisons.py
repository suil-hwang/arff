from __future__ import annotations

from pathlib import Path

import numpy as np

from mesh import ImportMesh
from rtr import OctaManopt
from ray import Ray, RayInit


def convergence_comparisons(inputdir, outputdir):
    input_path = Path(inputdir)
    output_path = Path(outputdir)
    output_path.mkdir(parents=True, exist_ok=True)

    meshfiles = sorted(input_path.glob("*.mesh"))
    cond = ("raycond", "ourcond")

    for mesh_file in meshfiles:
        mesh_data = ImportMesh(mesh_file)
        print(f"fullname {mesh_file}")

        for idx, cond_name in enumerate(cond):
            if idx == 0:
                q0_init, _ = RayInit(mesh_data)
                _, _, info_rtr = OctaManopt(
                    mesh_data,
                    q0=q0_init,
                    saveIterates=False,
                    gpuflag=False,
                    useCombinatorialLaplacian=True,
                )
                _, _, info_ray = Ray(
                    mesh_data,
                    qInit=None,
                    useGeometricLaplacian=False,
                    timeInitialization=False,
                )
            else:
                _, q0, info_rtr = OctaManopt(
                    mesh_data,
                    q0=None,
                    saveIterates=False,
                    gpuflag=False,
                    useCombinatorialLaplacian=False,
                )
                _, _, info_ray = Ray(
                    mesh_data,
                    qInit=q0,
                    useGeometricLaplacian=True,
                    timeInitialization=False,
                )

            if len(info_rtr) == 0 or len(info_ray) == 0:
                raise RuntimeError(f"Missing optimization trace in {mesh_file} for {cond_name}")

            grad_rtr = (20.0 / 3.0) * np.array([row["gradnorm"] for row in info_rtr], dtype=np.float64)
            time_rtr = np.array([row["time"] for row in info_rtr], dtype=np.float64)
            grad_ray = np.array([row["gradnorm"] for row in info_ray], dtype=np.float64)
            time_ray = np.array([row["time"] for row in info_ray], dtype=np.float64)

            np.savetxt(
                output_path / f"{mesh_file.stem}_{cond_name}_grad_ray.csv",
                np.column_stack([time_ray, grad_ray]),
            )
            np.savetxt(
                output_path / f"{mesh_file.stem}_{cond_name}_grad_rtr.csv",
                np.column_stack([time_rtr, grad_rtr]),
            )

            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.semilogy(time_ray, grad_ray)
                ax.semilogy(time_rtr, grad_rtr)
                fig.tight_layout()
                plt.close(fig)
            except Exception:
                # Plotting is optional for parity with MATLAB script behavior.
                pass


ConvergenceComparisons = convergence_comparisons

__all__ = ["convergence_comparisons", "ConvergenceComparisons"]
