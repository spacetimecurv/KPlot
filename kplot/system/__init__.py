"""Whole-system diagnostics for an AthenaK run.

Names live under their submodule, e.g. ``from kplot.system.plotter import
SystemPlotter``, ``from kplot.system.horizon import HorizonFinder``.

A automated pipeline lives in ``scripts/system/plot_system.sh``.
"""

from . import athenak_units
from . import batchmerge
from . import bin_convert
from . import checkpoints
from . import history
from . import horizon
from . import plotter
from . import seriesplot
from . import tracker
from . import units
from . import waveforms

__all__ = [
  "athenak_units",
  "batchmerge",
  "bin_convert",
  "checkpoints",
  "history",
  "horizon",
  "plotter",
  "seriesplot",
  "tracker",
  "units",
  "waveforms",
]
