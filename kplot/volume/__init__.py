"""Analysis of AthenaK 3D volume-domain output.

Usage:
  from kplot.volume import disk, plots
  disk.analyze(args)
  plots.plot_all(outdir, figdir)

An automated script lives in ``scripts/volume/run_analysis.sh``.
"""

import importlib

__all__ = ["disk", "plots"]


def __getattr__(name):
  if name in __all__:
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
  return sorted(__all__)
