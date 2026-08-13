"""Whole-system diagnostics for an AthenaK run.

Everything keyed on the simulation as a whole — history files, compact-object
trackers, apparent horizons, waveforms, 2-D slice series and the full-run
visualizer that combines them across all ``output-XXXX`` restart segments —
as opposed to :mod:`kplot.sphere`, which analyses what crosses a spherical
extraction surface.

Every name here is also re-exported at the top level, so ``from kplot import
SystemPlotter`` and ``from kplot.system import SystemPlotter`` are equivalent.

A ready-to-edit driver lives in ``scripts/system/plot_system.sh``.
"""

from .horizon import HorizonFinder
from .batchmerge import BatchMerger
from .waveforms import Waveform
from .seriesplot import SeriesPlot
from .history import History
from .checkpoints import list_checkpoints, read_checkpoint
from .units import Units, CGS, MKS, GEOMETRIC_SOLAR
from .plotter import SystemPlotter, main, use_latex
from . import athenak_units

__all__ = [
    "HorizonFinder",
    "BatchMerger",
    "Waveform",
    "SeriesPlot",
    "History",
    "list_checkpoints",
    "read_checkpoint",
    "Units",
    "CGS",
    "MKS",
    "GEOMETRIC_SOLAR",
    "SystemPlotter",
    "main",
    "use_latex",
    "athenak_units",
]
