from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from .ProcessMesh import MeshData, process_mesh


def _read_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and tetrahedra from a mesh file via meshio.

    Supports all formats meshio handles (Medit .mesh, Tetgen .node/.ele, etc.).
    Returns 0-based int64 tetrahedra and float64 vertices.
    """
    m = meshio.read(str(mesh_path))
    verts = np.asarray(m.points, dtype=np.float64)
    if "tetra" not in m.cells_dict:
        raise ValueError(f"No tetrahedra found in {mesh_path}.")
    tets = np.asarray(m.cells_dict["tetra"], dtype=np.int64)
    return verts, tets


def import_mesh(mesh_file: str | Path, bdry_angle_cutoff: float = 0.0) -> MeshData:
    mesh_path = Path(mesh_file)
    verts, tets = _read_mesh(mesh_path)
    mesh = process_mesh(verts, tets, bdry_angle_cutoff=bdry_angle_cutoff)
    mesh.file = str(mesh_path)
    return mesh


ImportMesh = import_mesh
