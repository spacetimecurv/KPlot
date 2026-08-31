"""Analysis of AthenaK 3D volume-domain output, as opposed to :mod:`kplot.sphere`,
which analyses spherical-surface extraction output.

Submodules are imported on first access: :mod:`~kplot.volume.disk` pulls in
h5py and scipy, and :mod:`~kplot.volume.plots` pulls in matplotlib, neither of
which should happen on ``import kplot``.

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
