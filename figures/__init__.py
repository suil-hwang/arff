"""Python ports for MATLAB scripts in :mod:`figures`.

The module mirrors the MATLAB script names to allow the same call patterns for
Python scripts and notebooks:

* ``ConvergenceComparisons``
* ``EnergyTest``
* ``GenerateComparisons``
* ``OctaExactnessTest``
* ``OdecoExactnessTest``
* ``PrismFigures``
* ``ProjectionComparison``
"""

from .ConvergenceComparisons import ConvergenceComparisons, convergence_comparisons
from .EnergyTest import EnergyTest, energy_test
from .GenerateComparisons import GenerateComparisons, generate_comparisons
from .OctaExactnessTest import OctaExactnessTest, octa_exactness_test
from .OdecoExactnessTest import OdecoExactnessTest, odeco_exactness_test
from .PrismFigures import PrismFigures, prism_figures
from .ProjectionComparison import ProjectionComparison, projection_comparison

__all__ = [
    "ConvergenceComparisons",
    "convergence_comparisons",
    "EnergyTest",
    "energy_test",
    "GenerateComparisons",
    "generate_comparisons",
    "OctaExactnessTest",
    "octa_exactness_test",
    "OdecoExactnessTest",
    "odeco_exactness_test",
    "PrismFigures",
    "prism_figures",
    "ProjectionComparison",
    "projection_comparison",
]
