from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import meshio
import numpy as np

from .ProcessMesh import MeshData


def _extract_verts_tets(mesh_data: MeshData | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh_data, MeshData):
        verts = mesh_data.verts
        tets = mesh_data.tets
    elif isinstance(mesh_data, Mapping):
        if "verts" not in mesh_data or "tets" not in mesh_data:
            raise KeyError("mesh_data mapping must contain 'verts' and 'tets'.")
        verts = mesh_data["verts"]
        tets = mesh_data["tets"]
    else:
        raise TypeError("mesh_data must be MeshData or mapping with verts/tets.")

    verts = np.asarray(verts, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must have shape (n, 3).")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (m, 4).")
    return verts, tets


def export_mesh(filename: str | Path, mesh_data: MeshData | Mapping[str, Any]) -> None:
    """Write a tetrahedral mesh to file via meshio.

    Output format is inferred from the file extension by meshio.
    Accepts MeshData or a mapping with 'verts' and 'tets' keys.
    """
    path = Path(filename)
    verts, tets = _extract_verts_tets(mesh_data)
    m = meshio.Mesh(
        points=verts,
        cells=[("tetra", tets)],
    )
    meshio.write(str(path), m)


ExportMesh = export_mesh
