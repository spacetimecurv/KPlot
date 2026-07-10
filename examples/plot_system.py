#########################################################################
# File: plot_system.py                                                 #
# Description: Full-run simulation visualization from AthenaK output.  #
#########################################################################

# Import the full-run system plotter class.
from kplot import SystemPlotter

# Point at the parent directory that contains the output-XXXX restart
# segments.  Frames repeated at restart boundaries are deduplicated by
# simulation time (the later segment wins).
plotter = SystemPlotter(
  simpath="/leonardo_scratch/large/userexternal/osteppoh/basem1/temp",
  figpath=None,          # default: <simpath>/Figs
  units="cgs",           # 'code', 'cgs', or 'ngs'
  jobname="bhns",        # AthenaK job name prefixing all input/output files
  plane="xy",            # slice plane: 'xy' (auto xz companion), 'xz', or 'yz'
  time_units="Msun",     # time display in plot titles/axes: 'Msun' or 'ms'
  show_trackers=False,   # overlay compact-object tracker markers on 2-D slices
  show_horizon=False,    # overlay black-hole apparent-horizon circles on 2-D slices
  full_domain=False,     # False -> +/-60 Msun window
  skip_existing=True,    # skip frames whose PNG already exists
)
# Each frame series lands in its own <figpath>/<diagnostic>_<plane>/ subfolder,
# e.g. Figs/temperature_xy/.  scripts/make_movies.sh then builds one movie per
# folder into Figs/all_movies/.

# Run any subset of the available diagnostics.  Pass ["all"] to run everything.
plotter.run([
  "history",       # baryonic mass & Hamiltonian constraint norm vs time
  "rho_max",       # central density & min lapse vs time
  "run_speed",     # AthenaK cycles/sec from log files
  "density",       # density slice frames  [+ trackers/horizon + AMR grid]
  "temperature",   # temperature slice frames
  "s00",           # electron fraction (s_00) frames
  "rad_E",         # radiation energy E:0 frames
  "rad_N",         # radiation number density N:0 frames
  "nu_energy",     # neutrino energy E/N species 0 (nu_e)
])
