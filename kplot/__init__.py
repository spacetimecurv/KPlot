"""Plotting utilities and tools for AthenaK, interfacing plot-tools.

Three subpackages:

- :mod:`kplot.system` — whole-run diagnostics (history, trackers, horizons,
  waveforms, slice series, the full-run ``SystemPlotter``).
- :mod:`kplot.sphere` — analysis of spherical-surface (``sph``) extraction
  output: ejecta, neutrinos, merger time.
- :mod:`kplot.volume` — analysis of 3D volume-domain output: post-merger disk
  diagnostics etc..

Names live under their subpackage and submodule: ``from kplot.system.plotter
import SystemPlotter``, ``from kplot.sphere import ejecta``, ``from
kplot.volume import disk``.
"""

from . import system
from . import sphere
from . import volume

__all__ = [
    "system",
    "sphere",
    "volume",
]
