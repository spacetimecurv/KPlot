"""Analysis of AthenaK spherical-surface (``file_type = sph``) extraction output.

Usage:
  from kplot.sphere import ejecta
  ejecta.analyze(["/sim/output-0000/sph"], "DD2.h5", "/sim/analysis")

A automated pipeline lives in ``scripts/sphere/run_analysis.sh``.
"""

import importlib

__all__ = ["accretion", "butterfly", "comparison", "ejecta", "mergertime", "neutrinos",
           "plots", "poynting"]


def __getattr__(name):
  if name in __all__:
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
  return sorted(__all__)
