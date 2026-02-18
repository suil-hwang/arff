from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.io import savemat

from arff_io import Coeff2Frames, ExportFrames, Octa2Odeco
from mesh import ImportMesh
from mbo import MBO, OdecoMBO, OctaMBO
from ray import Ray, RayMBO
from rtr import OctaManopt, OdecoManopt
from variety import RandOctahedralField


def _to_struct_array(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.empty((0, 1), dtype=object)
    fields = list(rows[0].keys())
    out = np.empty(len(rows), dtype=[(name, object) for name in fields])
    for i, row in enumerate(rows):
        for name in fields:
            out[name][i] = row[name]
    return out


def _build_config():
    types = [
        {
            "name": "Octa",
            "fiber": OctaMBO,
            "rtr": lambda mesh_data, q0: OctaManopt(mesh_data, q0, False, False),
        },
        {
            "name": "Odeco",
            "fiber": OdecoMBO,
            "rtr": lambda mesh_data, q0: OdecoManopt(mesh_data, q0, False, False),
        },
    ]

    config: list[dict] = []
    config.append(
        {
            "type": 0,
            "method": ["Ray"],
            "run": [lambda mesh_data, q0: Ray(mesh_data)],
        },
    )
    config.append(
        {
            "type": 0,
            "method": ["Ray2"],
            "run": [lambda mesh_data, q0: Ray(mesh_data, qInit=q0, useGeometricLaplacian=True)],
        },
    )
    config.append(
        {
            "type": 0,
            "method": ["RayMBO"],
            "run": [lambda mesh_data, q0, _fiber=RayMBO: MBO(mesh_data, _fiber, q0)],
        },
    )
    config.append(
        {
            "type": 0,
            "method": ["RayMMBO"],
            "run": [lambda mesh_data, q0, _fiber=RayMBO: MBO(mesh_data, _fiber, q0, 50, 3)],
        },
    )

    for type_idx, type_info in enumerate(types):
        config.append(
            {
                "type": type_idx,
                "method": ["RTR"],
                "run": [type_info["rtr"]],
            },
        )
        config.append(
            {
                "type": type_idx,
                "method": ["MBO"],
                "run": [lambda mesh_data, q0, _fiber=type_info["fiber"]: MBO(mesh_data, _fiber, q0)],
            },
        )
        config.append(
            {
                "type": type_idx,
                "method": ["mMBO"],
                "run": [lambda mesh_data, q0, _fiber=type_info["fiber"]: MBO(mesh_data, _fiber, q0, 50, 3)],
            },
        )

    for cfg in config:
        if cfg["method"][0] != "RTR":
            cfg["method"].append(f"{cfg['method'][0]}+RTR")
            cfg["run"].append(types[cfg["type"]]["rtr"])

    return types, config


def generate_comparisons(inputdir, outputdir):
    input_path = Path(inputdir)
    output_path = Path(outputdir)
    output_path.mkdir(parents=True, exist_ok=True)

    types, config = _build_config()

    meshfiles = sorted(input_path.glob("*.mesh"))
    trial = 0
    succeeded: list[bool] = []
    results = []

    for mesh_file in meshfiles:
        mesh_data = ImportMesh(mesh_file)
        q0_octa = RandOctahedralField(
            mesh_data.nv,
            mesh_data.bdry_idx,
            mesh_data.bdry_normals,
        )
        q0_odeco = Octa2Odeco(q0_octa)
        print(f"fullname {mesh_file}")

        for cfg in config:
            total_time = 0.0
            total_iters = 0
            q_prev = None

            for stage_idx, method_name in enumerate(cfg["method"]):
                trial += 1
                trial_name = f"{mesh_file.stem}_{types[cfg['type']]['name']}_{method_name}"

                try:
                    run = cfg["run"][stage_idx]
                    if stage_idx > 0:
                        q, q0, info = run(mesh_data, q_prev)
                    elif cfg["type"] == 0:
                        q, q0, info = run(mesh_data, q0_octa)
                    else:
                        q, q0, info = run(mesh_data, q0_odeco)

                    q_odeco = q if q.shape[0] == 15 else Octa2Odeco(q)
                    energy = 0.5 * float(np.sum(q_odeco.T * (mesh_data.L @ q_odeco.T)))

                    if isinstance(info, list):
                        total_iters += len(info)
                        if info:
                            total_time += float(info[-1].get("time", 0.0))

                    results.append(
                        {
                            "Model": mesh_file.stem,
                            "Vertices": int(mesh_data.nv),
                            "Type": types[cfg["type"]]["name"],
                            "Method": method_name,
                            "Energy": energy,
                            "Time": total_time,
                            "Iterations": total_iters,
                        },
                    )

                    ExportFrames(output_path / f"{trial_name}_frames.txt", Coeff2Frames(q))
                    ExportFrames(output_path / f"{trial_name}_init_frames.txt", Coeff2Frames(q0))
                    savemat(
                        output_path / f"{trial_name}.mat",
                        {
                            "trialName": np.array([trial_name], dtype=object),
                            "q": q,
                            "q0": q0,
                            "info": _to_struct_array(info) if isinstance(info, list) else np.empty((0, 1), dtype=object),
                        },
                    )

                    q_prev = q
                    succeeded.append(True)
                except Exception as exc:
                    warnings.warn(f"{trial_name} GenerateComparisons: {exc}")
                    succeeded.append(False)
                    break

    results_path = output_path / "results.mat"
    savemat(results_path, {"results": _to_struct_array(results)})

    csv_path = output_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Model", "Vertices", "Type", "Method", "Energy", "Time", "Iterations"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return succeeded, results


GenerateComparisons = generate_comparisons

__all__ = ["generate_comparisons", "GenerateComparisons"]
