"""Plotting utilities and tools for AthenaK, interfacing plot-tools.

Two subpackages, mirrored by the drivers in ``scripts/``:

- :mod:`kplot.system` — whole-run diagnostics (history, trackers, horizons,
  waveforms, slice series, the full-run ``SystemPlotter``).
- :mod:`kplot.sphere` — analysis of spherical-surface (``sph``) extraction
  output: ejecta, neutrinos, merger time.

The :mod:`kplot.system` names are re-exported here, so ``from kplot import *``
keeps working.
"""

from .system import (
    HorizonFinder,
    BatchMerger,
    Waveform,
    SeriesPlot,
    History,
    Units,
    CGS,
    MKS,
    GEOMETRIC_SOLAR,
    SystemPlotter,
    athenak_units,
)
from . import system
from . import sphere

__all__ = [
    "HorizonFinder",
    "BatchMerger",
    "Waveform",
    "SeriesPlot",
    "History",
    "Units",
    "CGS",
    "MKS",
    "GEOMETRIC_SOLAR",
    "SystemPlotter",
    "athenak_units",
    "system",
    "sphere",
]
