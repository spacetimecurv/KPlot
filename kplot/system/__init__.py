"""Whole-system diagnostics for an AthenaK run.

Names live under their submodule, e.g. ``from kplot.system.plotter import
SystemPlotter``, ``from kplot.system.horizon import HorizonFinder``.

A automated pipeline lives in ``scripts/system/plot_system.sh``.
"""

from . import batchutils
from . import bin_convert
from . import checkpoints
from . import history
from . import horizon
from . import meshblocks
from . import parfile
from . import plotter
from . import seriesplot
from . import tracker
from . import units
from . import waveforms

__all__ = [
  "batchutils",
  "bin_convert",
  "checkpoints",
  "history",
  "horizon",
  "meshblocks",
  "parfile",
  "plotter",
  "seriesplot",
  "tracker",
  "units",
  "waveforms",
]
