"""Analysis of AthenaK spherical-surface (``file_type = sph``) extraction output.

Measures what crosses the extraction sphere: ejecta mass flux and composition
(:mod:`~kplot.sphere.ejecta`) and neutrino luminosities and mean energies
(:mod:`~kplot.sphere.neutrinos`), plus the GW merger time
(:mod:`~kplot.sphere.mergertime`) the other diagnostics are referenced to.
:mod:`~kplot.sphere.plots` and :mod:`~kplot.sphere.comparison` render the
results, and :mod:`~kplot.sphere.accretion` derives a post-merger accretion
rate from them.

Submodules are imported on first access: several of them select the matplotlib
Agg backend or pull in h5py, which should not happen on ``import kplot``.

    from kplot.sphere import ejecta
    ejecta.analyze(["/sim/output-0000/sph"], "DD2.h5", "/sim/analysis")

A ready-to-edit driver lives in ``scripts/sphere/run_analysis.sh``.
"""

import importlib

__all__ = ["accretion", "comparison", "ejecta", "mergertime", "neutrinos", "plots"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
