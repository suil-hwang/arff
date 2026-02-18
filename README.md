# Algebraic Representations for Volumetric Frame Fields

## Introduction

This code includes algorithms for computing volumetric (octahedral and odeco) frame fields, described in detail in the paper:

> Palmer, D., Bommes, D., & Solomon, J. (2020). [Algebraic Representations for Volumetric Frame Fields.](https://dl.acm.org/doi/pdf/10.1145/3366786) ACM Transactions on Graphics (TOG), 39(2), 1-17.

[![View ARFF on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/75297-arff)

[![DOI](https://zenodo.org/badge/208910749.svg)](https://zenodo.org/badge/latestdoi/208910749)

## External Dependencies (Python)

- Python 3.10+
- `numpy`, `scipy` (core math)
- `meshio` (mesh I/O)
- `pymanopt` (RTR solvers)
- `mosek` (required for SDP projection used by MBO projection operators)
- Optional for visualization: `matplotlib`, `pyvista`

## Installation (Python)

Install in editable mode:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Install with common optional dependencies:

```bash
python -m pip install -e ".[mesh,rtr,sdp,plot,test]"
```

If you use MOSEK-based SDP projection, ensure your MOSEK license is available in your environment.

## Usage (Python)

Main entry points are `MBO`, `OctaManopt`, and `OdecoManopt`.

### Loading Models

Some tetrahedral meshes in `Medit` format are included in the `meshes` directory.

```python
from mesh import ImportMesh

# Medit format
mesh = ImportMesh("meshes/rockerarm_91k.mesh")

# TetGen format (.node/.ele pair)
mesh_tetgen = ImportMesh("path/to/file.node")
```

### Computing Frame Fields

Compute octahedral and odeco fields with MBO (random initialization):

```python
from mbo import MBO, OctaMBO, OdecoMBO

q_octa, q0_octa, info_octa = MBO(mesh, OctaMBO, None, 1, 0)
q_odeco, q0_odeco, info_odeco = MBO(mesh, OdecoMBO, None, 1, 0)
```

Modified MBO (diffusion multiplier/exponent from the paper):

```python
q_octa, _, _ = MBO(mesh, OctaMBO, None, 50, 3)
q_odeco, _, _ = MBO(mesh, OdecoMBO, None, 50, 3)
```

Run RTR with specified initial fields (or pass `None` for random initialization):

```python
from rtr import OctaManopt, OdecoManopt

q_octa_rtr, _, info_octa_rtr = OctaManopt(mesh, q_octa)
q_odeco_rtr, _, info_odeco_rtr = OdecoManopt(mesh, q_odeco)
```

### Visualization

Visualize an octahedral or odeco field:

```python
from plot import VisualizeResult

plotter = VisualizeResult(mesh, q_odeco)
```

Plot frame-aligned cubes at sample points:

```python
import numpy as np
from plot import PlotInterpolatedFrames

samples = mesh.verts[:100, :]  # example sample points (k, 3)
PlotInterpolatedFrames(q_octa, mesh, samples)
```

## Notes on MATLAB-Only Components
- `ext/ray` contains the original MATLAB/MEX/C++ Ray implementation.  
- The Python entry point is available in `src/ray` (`from ray import Ray, ray`) and mirrors the MATLAB behavior.
- Scripts in `figures/` (`EnergyTest`, `PrismFigures`, `ConvergenceComparisons`, `GenerateComparisons`, `OctaExactnessTest`, `OdecoExactnessTest`) are MATLAB scripts.
