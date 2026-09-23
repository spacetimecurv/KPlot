#########################################################################
# File: system.py                                                       #
# Description: Multi-segment full-run visualization driver for AthenaK. #
#########################################################################
#
# Combines all output-XXXX restart segments of an AthenaK simulation and
# produces a full-run set of diagnostics: 1-D history/time-series plots
# and parallel 2-D slice frames (density, temperature, Y_e, radiation-M1
# fields, neutrino energies) with optional NS-tracker, black-hole horizon
# and AMR-grid overlays.
#
# Binary files have to either exist in a batchtools substructure, i.e.
# output-XXXX (in this case pass --batchtools) or in a single folder.

# Each frame series is written to its own figpath/<diagnostic>_<plane>/ subfolder.
#
# Use in Python:
#     from kplot.system.plotter import SystemPlotter
#     SystemPlotter(simpath="/path/to/sim", units="cgs").run(["density", "history"])

# Import necessary standard libraries.
import argparse
import functools
import glob
import multiprocessing
import os
import re
import configparser

# Import necessary third-party libraries.
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import athplot utilities from plot-tools.
from athplot.load_ath_bin import BinaryData
from athplot.group_data import GroupData
from athplot.tracker import Tracker

# Import AthenaK unit conversions / render parameters.
from kplot.system.units import AthenaK_Units
from kplot.system.history import History

# ---------------------------------------------------------------------------
# Global plot style.
# ---------------------------------------------------------------------------
_STYLE = {
  'lines.markersize': 6,
  'font.family': 'serif',
  'font.serif': ['Latin Modern Roman', 'CMU Serif', 'cmr10'],
  'mathtext.fontset': 'cm',
  'font.size': 10.0,
  'text.usetex': False,
  'axes.labelsize': 'large',
  'xtick.major.size': 10,
  'xtick.minor.size': 6,
  'xtick.minor.visible': True,
  'xtick.labelsize': 'large',
  'xtick.direction': 'in',
  'xtick.top': True,
  'ytick.major.size': 10,
  'ytick.minor.size': 6,
  'ytick.minor.visible': True,
  'ytick.labelsize': 'large',
  'ytick.direction': 'in',
  'ytick.right': True,
  'legend.numpoints': 1,
  'legend.fontsize': 'medium',
  'legend.frameon': False,
  'legend.scatterpoints': 1,
  'figure.figsize': (5, 3),
}

FIGSIZE = (10, 6)

# ---------------------------------------------------------------------------
# Units, scales and labels.
# ---------------------------------------------------------------------------
au = AthenaK_Units()

# Scale factors: (var, units) -> multiplicative factor applied to code values
_SCALES = {
  # rest-mass density
  ('dens',      'cgs'): au.CODE2CGS_DENS,
  # radiation energy density
  ('E:0',       'ngs'): au.GK2NGS_ENEDENS,
  ('E:0',       'cgs'): au.CODE2CGS_ENEDENS,
  # radiation number density
  ('N:0',       'ngs'): au.CODE2NGS_NUMDENS,
  ('N:0',       'cgs'): au.CODE2CGS_NUMDENS,
  ('N:2',       'ngs'): au.CODE2NGS_NUMDENS,
  ('N:2',       'cgs'): au.CODE2CGS_NUMDENS,
  # absorption opacities
  ('abs_0:0',   'ngs'): au.CODE2NGS_OPACITY,
  ('abs_0:0',   'cgs'): au.CODE2CGS_OPACITY,
  ('abs_1:0',   'ngs'): au.CODE2NGS_OPACITY,
  ('abs_1:0',   'cgs'): au.CODE2CGS_OPACITY,
  # number emissivity
  ('eta_0:0',   'ngs'): au.UNIT_ND_DOT,
  ('eta_0:0',   'cgs'): au.CODE2CGS_ND_DOT,
  # energy emissivity
  ('eta_1:0',   'ngs'): au.UNIT_ED_DOT,
  ('eta_1:0',   'cgs'): au.CODE2CGS_ED_DOT,
  # average neutrino energy (ngs and cgs both yield MeV)
  ('avg_energy', 'ngs'): au.CODE2MEV_AVGENE,
  ('avg_energy', 'cgs'): au.CODE2MEV_AVGENE,
  ('nu_energy',  'ngs'): au.CODE2MEV_AVGENE,
  ('nu_energy',  'cgs'): au.CODE2MEV_AVGENE,
}

# Axis / colorbar labels: (var, units) -> LaTeX string
_LABELS = {
  # density
  ('dens',      'code'): r"$\rho~[\mathrm{km}^{-2}]$",
  ('dens',      'cgs' ): r"$\rho~[\mathrm{g\,cm^{-3}}]$",
  # radiation energy density
  ('E:0',       'code'): r"$E_\nu^{(0)}~[\mathrm{km}^{-2}]$",
  ('E:0',       'ngs' ): r"$E_\nu^{(0)}~[\mathrm{MeV\,nm^{-3}}]$",
  ('E:0',       'cgs' ): r"$E_\nu^{(0)}~[\mathrm{erg\,cm^{-3}}]$",
  # radiation number density
  ('N:0',       'code'): r"$N_\nu^{(0)}~[\mathrm{fm}^{-3}]$",
  ('N:0',       'ngs' ): r"$N_\nu^{(0)}~[\mathrm{nm^{-3}}]$",
  ('N:0',       'cgs' ): r"$N_\nu^{(0)}~[\mathrm{cm^{-3}}]$",
  ('N:2',       'code'): r"$N_\nu^{(2)}~[\mathrm{fm}^{-3}]$",
  ('N:2',       'ngs' ): r"$N_\nu^{(2)}~[\mathrm{nm^{-3}}]$",
  ('N:2',       'cgs' ): r"$N_\nu^{(2)}~[\mathrm{cm^{-3}}]$",
  # absorption opacities
  ('abs_0:0',   'code'): r"$\kappa_{a,0}^{(0)}~[\mathrm{km}^{-1}]$",
  ('abs_0:0',   'ngs' ): r"$\kappa_{a,0}^{(0)}~[\mathrm{nm}^{-1}]$",
  ('abs_0:0',   'cgs' ): r"$\kappa_{a,0}^{(0)}~[\mathrm{cm}^{-1}]$",
  ('abs_1:0',   'code'): r"$\kappa_{a,1}^{(0)}~[\mathrm{km}^{-1}]$",
  ('abs_1:0',   'ngs' ): r"$\kappa_{a,1}^{(0)}~[\mathrm{nm}^{-1}]$",
  ('abs_1:0',   'cgs' ): r"$\kappa_{a,1}^{(0)}~[\mathrm{cm}^{-1}]$",
  # number emissivity
  ('eta_0:0',   'code'): r"$\eta_0^{(0)}~[\mathrm{code}]$",
  ('eta_0:0',   'ngs' ): r"$\eta_0^{(0)}~[\mathrm{nm^{-3}\,s^{-1}}]$",
  ('eta_0:0',   'cgs' ): r"$\eta_0^{(0)}~[\mathrm{cm^{-3}\,s^{-1}}]$",
  # energy emissivity
  ('eta_1:0',   'code'): r"$\eta_1^{(0)}~[\mathrm{code}]$",
  ('eta_1:0',   'ngs' ): r"$\eta_1^{(0)}~[\mathrm{MeV\,nm^{-3}\,s^{-1}}]$",
  ('eta_1:0',   'cgs' ): r"$\eta_1^{(0)}~[\mathrm{erg\,cm^{-3}\,s^{-1}}]$",
  # average energy (report in MeV for both ngs and cgs)
  ('avg_energy', 'code'): r"$\langle\varepsilon_\nu\rangle~[\mathrm{km^{-2}/fm^{-3}}]$",
  ('avg_energy', 'ngs' ): r"$\langle\varepsilon_\nu\rangle~[\mathrm{MeV}]$",
  ('avg_energy', 'cgs' ): r"$\langle\varepsilon_\nu\rangle~[\mathrm{MeV}]$",
  ('nu_energy',  'code'): r"$\langle\varepsilon_\nu\rangle~[\mathrm{km^{-2}/fm^{-3}}]$",
  ('nu_energy',  'ngs' ): r"$\langle\varepsilon_\nu\rangle~[\mathrm{MeV}]$",
  ('nu_energy',  'cgs' ): r"$\langle\varepsilon_\nu\rangle~[\mathrm{MeV}]$",
  # dimensionless / already physical
  ('temperature', 'code'): r"$T~[\mathrm{MeV}]$",
  ('temperature', 'ngs' ): r"$T~[\mathrm{MeV}]$",
  ('temperature', 'cgs' ): r"$T~[\mathrm{MeV}]$",
  ('s_00',      'code'): r"$Y_e$",
  ('s_00',      'ngs' ): r"$Y_e$",
  ('s_00',      'cgs' ): r"$Y_e$",
  ('con_H',     'code'): r"$|\mathcal{H}|$",
  ('con_H',     'ngs' ): r"$|\mathcal{H}|$",
  ('con_H',     'cgs' ): r"$|\mathcal{H}|$",
  ('alpha',     'code'): r"$\alpha$",
  ('alpha',     'ngs' ): r"$\alpha$",
  ('alpha',     'cgs' ): r"$\alpha$",
}

# Variables that use a linear (not log) colorscale
_LINEAR_VARS = {'temperature', 's_00', 'nu_energy', 'alpha'}

# Convert geometrized to physical units.
MSUN_TO_KM = au.unit_len_cgs / 1.0e5
MSUN_TO_MS = (au.unit_len_cgs / au.cgs.c) * 1.0e3

# Choose spatial coordinates for the plot.
def spatial_units(units):
  """Return (coord_scale, axis_label) for the spatial axes given a unit system."""
  if units == "code":
    return 1.0, r"\mathrm{M}_\odot"
  return MSUN_TO_KM, r"\mathrm{km}"

# Choose time units display for the plot.
TIME_UNITS = ("Msun", "ms")
def time_units_scale(time_units):
  """Return (scale, latex_label, axis_label) for the chosen time-display units."""
  if time_units == "ms":
    return MSUN_TO_MS, r"\mathrm{ms}", "ms"
  return 1.0, r"\mathrm{M}_\odot", "Msun"

# Ranges for the plotting variables in code units (DEFAULTS).
_CODE_VMIN = {
  'dens':        6e8 / au.CODE2CGS_DENS,   # ~ 9.7e-10
  'temperature': 0.0,
  's_00':        0.0,
  'E:0':         1e-11,
  'N:0':         1e-7,
  'N:2':         1e-7,
  'abs_0:0':     1e-18,
  'abs_1:0':     1e-18,
  'eta_0:0':     1e-10,
  'eta_1:0':     1e-10,
  'con_H':       1e-9,
  'alpha':       0.0,
  'avg_energy':  1e-4,
  'nu_energy':   0.0 / au.CODE2MEV_AVGENE,
}
_CODE_VMAX = {
  'dens':        2e15 / au.CODE2CGS_DENS,  # ~ 3.2e-3
  'temperature': 70.0,
  's_00':        0.5,
  'E:0':         1e-8,
  'N:0':         1e-3,
  'N:2':         1e-3,
  'abs_0:0':     1e-9,
  'abs_1:0':     1e-9,
  'eta_0:0':     1e-4,
  'eta_1:0':     1e-4,
  'con_H':       1e-4,
  'alpha':       1.0,
  'avg_energy':  1e-3,
  'nu_energy':   300.0 / au.CODE2MEV_AVGENE,
}

# Load the sections specified by the user.
def load_sections(path):
  """Return the [sections] entries of the config.ini as 'name[=lo,hi,cmap]' tokens."""
  if not path: return []

  # Extract the sections block.
  block, inside = [], False
  with open(path) as f:
    for line in f:
      header = line.strip()
      if header.startswith('[') and header.endswith(']'):
        inside = header[1:-1].strip().lower() == 'sections'
        continue
      if inside:
        block.append(line)
  if not block: return []

  cp = configparser.ConfigParser(delimiters=('=',), allow_no_value=True,
                                 inline_comment_prefixes=("#", ";"))
  cp.optionxform = str
  cp.read_string('[sections]\n' + ''.join(block))
  return [f"{name}={val}" if val else name for name, val in cp.items('sections')]

# Parse a 'name[=lo,hi,cmap]' section token.
def parse_section_spec(token):
  """Return (name, vmin, vmax, cmap); empty fields become None."""
  name, _, rest = token.partition('=')
  name = name.strip()
  if name not in SECTIONS + ["all"]:
    raise argparse.ArgumentTypeError(
      f"invalid section '{name}' (choose from {', '.join(SECTIONS + ['all'])})")
  lo, hi, cmap = ([p.strip() for p in rest.split(',')] + ['', '', ''])[:3]
  if cmap and cmap not in mpl.colormaps:
    raise argparse.ArgumentTypeError(f"unknown colormap '{cmap}' for section '{name}'")
  return (name, float(lo) if lo else None, float(hi) if hi else None, cmap or None)

# ---------------------------------------------------------------------------
# Section-specific variables.
# ---------------------------------------------------------------------------
# Neutrino-energy sections: (section name, species index, label)
_NU_ENERGY_SECTIONS = [
  ("nu_energy",  0, r"$\nu_e$"),
  ("nu_energy1", 1, r"$\bar{\nu}_e$"),
  ("nu_energy2", 2, r"$\nu_x$"),
]

# All valid section names, in execution order.
SECTIONS = [
  "history", "rho_max", "trackers", "run_speed",
  "density", "temperature", "s00",
  "rad_E", "rad_N", "rad_N2", "rad_abs", "rad_abs0",
  "rad_eta0", "rad_eta1", "conH", "alpha",
  "s00_avg", "avg_energy",
  "nu_energy", "nu_energy1", "nu_energy2",
]

# ---------------------------------------------------------------------------
# Configure plotting environment.
# ---------------------------------------------------------------------------
# Configure mpl.
def configure_matplotlib():
  """Select the non-interactive Agg backend and apply the plot style."""
  matplotlib.use("Agg", force=True)
  mpl.rcParams.update(_STYLE)

# Slice planes selectable for 2-D frame rendering, and their axis letters.
PLANES = ("xy", "xz", "yz")
_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}

# Fetch the coordinate labels for the given plane.
def plane_axis_labels(plane):
  """Return (horizontal, vertical) coordinate letters for a slice plane."""
  return _PLANE_AXES[plane]

# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------
# List of binary files.
def glob_bin_files(directory, prefix):
  """Return sorted list of existing binary files matching prefix.NNNNN.bin."""
  # If there is no 'bin' directory, then directly search in the parent folder.
  for sub in ("bin", ""):
    files = glob.glob(os.path.join(directory, sub, f"{prefix}.?????.bin"))
    if files:
      return sorted(files)
  return []

# Print stats from the frame.
def print_stats(data, var_name, time_scale=1.0, time_axis="Msun"):
  """Print the frame time plus min/max of var_name across all blocks."""
  max_val, min_val = -np.inf, np.inf
  for _, arr in data.get_block_data(var_name):
    max_val = max(max_val, arr.max())
    min_val = min(min_val, arr.min())
  print(f"    t={data.time * time_scale:8.3f} {time_axis}  |  "
        f"max={max_val:10.3e}  min={min_val:10.3e}")

# ---------------------------------------------------------------------------
# Horizon/Tracker.
# ---------------------------------------------------------------------------
# CO tracker class holding the necessary data.
class _CombinedTracker:
  """Lightweight tracker holding concatenated time/x/y/z arrays from all segments."""
  def __init__(self, time, x, y, z):
    self.time = time
    self.x = x
    self.y = y
    self.z = z

# Horizon class holding the times/radii.
class _CombinedHorizon:
  """Lightweight horizon holding concatenated time/radius arrays from all segments."""
  def __init__(self, time, radius):
    self.time = time
    self.radius = radius

# Select the tracker position in 2D based on the specified plane.
def _tracker_plane_coords(tracker, plane):
  """Return the (horizontal, vertical) tracker coordinate arrays for a plane."""
  if plane == "xy":
    return tracker.x, tracker.y
  if plane == "xz":
    return tracker.x, tracker.z
  return tracker.y, tracker.z

# Load the CO tracker files into a CombinedTracker object.
def load_trackers_all(output_dirs, jobname="bhns"):
  """Load and concatenate tracker files across all segments."""
  trackers = []
  for k in (0, 1):
    times_list, x_list, y_list, z_list = [], [], [], []
    for d in output_dirs:
      fpath = os.path.join(d, f"{jobname}.co_{k}.txt")
      if os.path.exists(fpath):
        try:
          t = Tracker(fpath)
          times_list.append(t.time)
          x_list.append(t.x)
          y_list.append(t.y)
          z_list.append(t.z)
        except Exception as e:
          print(f"  Warning: could not load {fpath}: {e}")

    if not times_list:
      print(f"  No tracker files found for object {k}")
      trackers.append(None)
      continue

    times, xs, ys, zs = _dedup_by_time(np.concatenate(times_list),
                                       np.concatenate(x_list),
                                       np.concatenate(y_list),
                                       np.concatenate(z_list))
    trackers.append(_CombinedTracker(times, xs, ys, zs))
  return trackers

# Load the apparent horizon times/radii into a CombinedHorizon class.
def load_horizons_all(output_dirs, jobname="bhns"):
  """Load and concatenate apparent-horizon summary files across all segments."""
  horizons = []
  for k in (0, 1):
    times_list, r_list = [], []
    for d in output_dirs:
      fpath = os.path.join(d, f"{jobname}.horizon_summary_{k}.txt")
      if os.path.exists(fpath):
        try:
          arr = np.loadtxt(fpath, comments="#")
          if arr.ndim == 1:
            arr = arr.reshape(1, -1)
          if arr.size > 0:
            times_list.append(arr[:, 1])
            r_list.append(arr[:, -1])
        except Exception as e:
          print(f"  Warning: could not load {fpath}: {e}")

    if not times_list:
      horizons.append(None)
      continue

    times, radii = _dedup_by_time(np.concatenate(times_list),
                                  np.concatenate(r_list))
    horizons.append(_CombinedHorizon(times, radii))
  return horizons

# Overlay the tracker positions.
def draw_trackers(ax, trackers, time, plane="xy", colors=("cyan", "orange"),
                  marker="x", markersize=80, coord_scale=1.0):
  """Overlay compact-object positions on ax at the given simulation time."""
  labels = ["CO 1", "CO 2"]
  for tracker, color, label in zip(trackers, colors, labels):
    if tracker is None:
      continue
    idx = np.argmin(np.abs(tracker.time - time))
    h, v = _tracker_plane_coords(tracker, plane)
    ax.scatter(h[idx] * coord_scale, v[idx] * coord_scale,
               marker=marker, c=color, s=markersize,
               linewidths=1.5, zorder=10, label=label)

# Overlay apparent-horizon position.
def draw_horizons(ax, horizons, trackers, time, plane="xy", coord_scale=1.0,
                  edgecolor="white", facecolor="black"):
  """Overlay filled apparent-horizon circles at the paired tracker positions."""
  if horizons is None or trackers is None:
    return
  for horizon, tracker in zip(horizons, trackers):
    if horizon is None or tracker is None:
      continue
    hi = np.argmin(np.abs(horizon.time - time))
    ti = np.argmin(np.abs(tracker.time - time))
    radius = horizon.radius[hi] * coord_scale
    h, v = _tracker_plane_coords(tracker, plane)
    circle = mpatches.Circle((h[ti] * coord_scale, v[ti] * coord_scale), radius,
                             edgecolor=edgecolor, facecolor=facecolor,
                             fill=True, zorder=9)
    ax.add_patch(circle)

# ---------------------------------------------------------------------------
# Data segment helpers.
# ---------------------------------------------------------------------------
# Read the time from the binary file.
def _read_frame_time(fpath):
  """Read simulation time from a binary data file. Returns (fpath, time_or_None)."""
  try:
    return (fpath, BinaryData(fpath).time)
  except Exception:
    return (fpath, None)

# Find the output directories under the simulation path.
def find_output_dirs(simpath, batchtools=True):
  """Return sorted list of output-XXXX subdirectories under simpath."""
  if not batchtools:
    return [simpath] if os.path.isdir(simpath) else []
  pattern = os.path.join(simpath, "output-[0-9][0-9][0-9][0-9]")
  return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))

# Collec the frames from the segments.
def _collect_frames_by_time(output_dirs, file_prefix):
  """Gather all files for file_prefix across output_dirs."""
  all_files = []
  for d in output_dirs:
    all_files.extend(glob_bin_files(d, file_prefix))

  if not all_files:
    return {}

  n_workers = min(multiprocessing.cpu_count(), len(all_files))
  with multiprocessing.Pool(n_workers) as pool:
    results = pool.map(_read_frame_time, all_files)

  # Iterate in order so later segments overwrite earlier ones
  time_to_fpath = {}
  for fpath, t in results:
    if t is not None:
      time_to_fpath[t] = fpath
  return time_to_fpath

# Collect the frames for a given file prefix of all segments.
def collect_combined_frames(output_dirs, file_prefix, comp_prefix=None, step=1):
  """Collect all frames for file_prefix across all segments."""
  prefixes = [file_prefix] if isinstance(file_prefix, str) else list(file_prefix)
  comp_prefixes = None
  if comp_prefix is not None:
    comp_prefixes = [comp_prefix] if isinstance(comp_prefix, str) else list(comp_prefix)

  time_maps = [_collect_frames_by_time(output_dirs, pre) for pre in prefixes]
  if not all(time_maps):
    return []
  common_times = set(time_maps[0]).intersection(*time_maps[1:])

  # Build set of all existing XZ files for quick lookup
  xz_file_set = set()
  if comp_prefixes is not None:
    for d in output_dirs:
      for pre in comp_prefixes:
        xz_file_set.update(glob_bin_files(d, pre))

  sorted_times = sorted(common_times)[::step]

  result = []
  for global_idx, t in enumerate(sorted_times):
    xy_fpaths = [tm[t] for tm in time_maps]
    xz_fpaths = None
    if comp_prefixes is not None:
      candidates = []
      for xy_fpath, pre in zip(xy_fpaths, comp_prefixes):
        m = re.search(r'\.(\d{5})\.bin$', os.path.basename(xy_fpath))
        if m:
          candidates.append(os.path.join(os.path.dirname(xy_fpath),
                                         f"{pre}.{m.group(1)}.bin"))
      if len(candidates) == len(comp_prefixes) and all(c in xz_file_set for c in candidates):
        xz_fpaths = candidates
    result.append((global_idx, xy_fpaths, xz_fpaths))
  return result

# Get the combined history of a file.
def combine_history(output_dirs, filename):
  parts = []
  for d in output_dirs:
    fpath = os.path.join(d, filename)
    if os.path.exists(fpath):
      try:
        h = History(fpath)
        h.load_history_data()
        if len(h.history_data["time"]) > 0:
          parts.append(h.history_data)
      except Exception as e:
        print(f"  Warning: could not load {fpath}: {e}")

  if not parts:
    return None

  keys = list(parts[0].keys())
  merged = [np.concatenate([p[k] for p in parts]) for k in keys]
  _, *cols = _dedup_by_time(merged[keys.index("time")], *merged)
  return dict(zip(keys, cols))

# Deduplicate frames.
def _dedup_by_time(all_times, *arrays):
  """Sort by time and keep the last occurrence of each time."""
  sort_idx = np.argsort(all_times, kind='stable')
  all_times = all_times[sort_idx]
  arrays = [a[sort_idx] for a in arrays]

  _, first_in_rev = np.unique(all_times[::-1], return_index=True)
  keep = np.sort(len(all_times) - 1 - first_in_rev)
  return (all_times[keep], *(a[keep] for a in arrays))

# ---------------------------------------------------------------------------
# Diagnostic history / time-series plots
# ---------------------------------------------------------------------------
# Plot the baryon mass conservation and hamiltonian constraint.
def _plot_timeseries(time, values, ylabel, figpath, fname, time_axis="Msun", logy=False):
  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  ax.plot(time, values)
  ax.set_xlabel(f"t ({time_axis})")
  ax.set_ylabel(ylabel)
  ax.set_xlim(np.min(time), np.max(time))
  if logy:
    ax.set_yscale("log")
  fig.savefig(os.path.join(figpath, fname), dpi=150)
  plt.close(fig)
  print(f"  Saved {fname}")

# Plot a column of a history file.
def _plot_history_columns(output_dirs, figpath, jobname, suffix, columns,
                          time_scale=1.0, time_axis="Msun"):
  d = combine_history(output_dirs, f"{jobname}.{suffix}")
  if d is None:
    print(f"  Skipping {suffix}: no {jobname}.{suffix} found")
    return
  for col, ylabel, name, logy in columns:
    if col not in d:
      print(f"  Skipping {col}: not in {jobname}.{suffix}")
      continue
    _plot_timeseries(d["time"] * time_scale, d[col], ylabel, figpath,
                     f"{jobname}_{name}.png", time_axis=time_axis, logy=logy)

# Plot the baryon mass conservation and hamiltonian constraint.
def plot_history_all(output_dirs, figpath, jobname="bhns",
                     time_scale=1.0, time_axis="Msun"):
  """Baryonic mass and Hamiltonian constraint norm across all segments."""
  _plot_history_columns(output_dirs, figpath, jobname, "mhd.hst",
                        [("mass", r"$M_b$", "history_mass", False)],
                        time_scale=time_scale, time_axis=time_axis)
  _plot_history_columns(output_dirs, figpath, jobname, "z4c.user.hst",
                        [("H-norm2", r"$||\mathcal{H}||_2$", "history_hnorm2", True)],
                        time_scale=time_scale, time_axis=time_axis)

# Plot the max. density and min. lapse.
def plot_user_history_all(output_dirs, figpath, jobname="bhns",
                          time_scale=1.0, time_axis="Msun"):
  """Max density and min lapse from <jobname>.user.hst across all segments."""
  _plot_history_columns(output_dirs, figpath, jobname, "user.hst",
                        [("rho-max", r"$\rho_\mathrm{max}$", "user_rho_max", True),
                         ("alpha-min", r"$\alpha_\mathrm{min}$", "user_alpha_min", False)],
                        time_scale=time_scale, time_axis=time_axis)

# Plot the CO trajectories.
def plot_tracker_trajectories_all(output_dirs, figpath, jobname="bhns",
                                  coord_scale=1.0, coord_label=r"\mathrm{M}_\odot"):
  """Plot compact-object trajectories (raw + smoothed) from tracker data."""
  trackers = load_trackers_all(output_dirs, jobname)
  if all(t is None for t in trackers):
    print("  No tracker files found, skipping trajectory plot")
    return

  colors  = ["tab:red",  "tab:blue"]
  colors2 = ["salmon",   "skyblue"]
  labels  = ["CO 1", "CO 2"]

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  for tracker, color, color2, label in zip(trackers, colors, colors2, labels):
    if tracker is None:
      continue
    ax.plot(tracker.x * coord_scale, tracker.y * coord_scale,
            ".", color=color2, ms=1, label=f"{label} raw")
    try:
      xs, ys = tracker.smooth_walk()
      ax.plot(xs * coord_scale, ys * coord_scale,
              "-", color=color, lw=1, label=f"{label} smoothed")
    except Exception:
      pass

  ax.set_xlabel(rf"$x~[{coord_label}]$")
  ax.set_ylabel(rf"$y~[{coord_label}]$")
  ax.set_aspect("equal")
  ax.legend()
  fig.savefig(os.path.join(figpath, f"{jobname}_tracker_trajectories.png"), dpi=150)
  plt.close(fig)
  print(f"  Saved {jobname}_tracker_trajectories.png")

# Parse the run speed from the log file.
def _parse_run_speed_log(logfile):
  """Parse one AthenaK log file.

  Returns (sim_time_mid_array, speed_array) or (None, None).  Elapsed time in
  each log resets to 0 at the start of that segment, so each log is processed
  independently.
  """
  if not os.path.exists(logfile):
    return None, None

  pattern = re.compile(
    r"elapsed=([0-9.eE+-]+)\s+cycle=([0-9]+)\s+time=([0-9.eE+-]+)"
  )
  elapsed, cycle, sim_time = [], [], []
  with open(logfile) as f:
    for line in f:
      m = pattern.search(line)
      if m:
        elapsed.append(float(m.group(1)))
        cycle.append(int(m.group(2)))
        sim_time.append(float(m.group(3)))

  if len(elapsed) < 2:
    return None, None

  speed, sim_time_mid = [], []
  for i in range(1, len(elapsed)):
    dt_wall = elapsed[i] - elapsed[i - 1]
    dcycle  = cycle[i] - cycle[i - 1]
    if dt_wall > 0:
      speed.append(dcycle / dt_wall)
      sim_time_mid.append(0.5 * (sim_time[i] + sim_time[i - 1]))

  if not speed:
    return None, None
  return np.array(sim_time_mid), np.array(speed)

# Plot the run speed with the log parser.
def plot_run_speed_all(output_dirs, figpath, jobname="bhns",
                       time_scale=1.0, time_axis="Msun"):
  """Plot cycles/second vs simulation time, stitched across all segments."""
  all_times, all_speeds = [], []
  for d in output_dirs:
    log_files = sorted(glob.glob(os.path.join(d, "*.out")))
    if not log_files:
      print(f"  Warning: no *.out files found in {d}")
      continue
    for log_file in log_files:
      t, s = _parse_run_speed_log(log_file)
      if t is not None:
        all_times.append(t)
        all_speeds.append(s)

  if not all_times:
    print("  No log data found for run speed")
    return

  sim_time = np.concatenate(all_times)
  speed    = np.concatenate(all_speeds)
  sort_idx = np.argsort(sim_time)
  sim_time = sim_time[sort_idx]
  speed    = speed[sort_idx]

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  ax.plot(sim_time * time_scale, speed)
  ax.set_xlabel(f"Simulation time ({time_axis})")
  ax.set_ylabel("Cycles / wall-second")
  fig.savefig(os.path.join(figpath, f"{jobname}_run_speed.png"), dpi=150)
  plt.close(fig)
  print(f"  Saved {jobname}_run_speed.png")

# ---------------------------------------------------------------------------
# Per-frame slice renderers.
# ---------------------------------------------------------------------------
# Main plotting interface function, to plot variable
# in specific unit system.
def plot_params(var, units='code'):
  """Return render parameters for *var* in *units*.

  Parameters:
    var   : str -- variable name (e.g. 'dens', 'abs_1:0')
    units : str -- 'code', 'cgs', or 'ngs'

  Returns:
    scale : float   -- multiply code values by this before plotting
    label : str     -- colorbar / axis label (LaTeX)
    use_log : bool  -- True -> LogNorm, False -> linear Normalize
    vmin  : float   -- suggested colorbar minimum (in target units)
    vmax  : float   -- suggested colorbar maximum (in target units)
  """
  scale   = _SCALES.get((var, units), 1.0)
  label   = _LABELS.get((var, units), _LABELS.get((var, 'code'), var))
  use_log = var not in _LINEAR_VARS

  # Defaults stored in code units and converted to target units.
  vmin_code = _CODE_VMIN.get(var)
  vmax_code = _CODE_VMAX.get(var)
  vmin = vmin_code * scale if vmin_code is not None else None
  vmax = vmax_code * scale if vmax_code is not None else None

  return scale, label, use_log, vmin, vmax

# Rescale derived variables.
def _rescaled(fn, scale, *args):
  return fn(*args) * scale

# Load frame with binary data.
def _load_frame(fpaths):
  return BinaryData(fpaths[0]) if len(fpaths) == 1 else GroupData(fpaths)

# Load frame base.
def _frame_base(data):
  return data.datasets[0] if isinstance(data, GroupData) else data

# Rescale image extent by coordinate scale.
def rescale_image_extents(ax, coord_scale):
  """Multiply the extent of every image already drawn on ax by coord_scale."""
  if coord_scale == 1.0:
    return
  for im in ax.get_images():
    x0, x1, y0, y1 = im.get_extent()
    im.set_extent([x0 * coord_scale, x1 * coord_scale,
                   y0 * coord_scale, y1 * coord_scale])

# Draw AMR grid structure.
def draw_refinement_grid(ax, data, alpha=0.4, linewidth=0.6, coord_scale=1.0):
  """Draw block boundaries colored by AMR refinement level."""
  levels = sorted(set(b.level for b in data.blocks))
  if len(levels) <= 1:
    return

  cmap = mpl.colormaps.get_cmap("cool")
  level_min, level_max = levels[0], levels[-1]

  def level_color(lv):
    if level_max == level_min:
      return cmap(0.5)
    return cmap((lv - level_min) / (level_max - level_min))

  for block in data.blocks:
    ext = block.get_extent()
    if len(ext) < 4:
      continue
    xmin, xmax, ymin, ymax = (ext[0] * coord_scale, ext[1] * coord_scale,
                              ext[2] * coord_scale, ext[3] * coord_scale)
    rect = mpatches.Rectangle(
      (xmin, ymin), xmax - xmin, ymax - ymin,
      linewidth=linewidth,
      edgecolor=level_color(block.level),
      facecolor="none",
      alpha=alpha,
      zorder=5,
    )
    ax.add_patch(rect)

# Each worker renders a frame.
def _render_frame_worker_all(kw):
  """Per-frame plot worker for combined multi-segment data."""
  global_idx      = kw['global_idx']
  xy_fpaths       = kw['xy_fpaths']
  xz_fpaths       = kw['xz_fpaths']   # None if not available
  var_name        = kw['var_name']
  plot_kwargs     = kw['plot_kwargs']
  xlim            = kw['xlim']
  ylim            = kw['ylim']
  zlim            = kw['zlim']
  label           = kw['label']
  out_prefix      = kw['out_prefix']
  figpath         = kw['figpath']
  print_minmax    = kw['print_minmax']
  show_refinement = kw['show_refinement']
  show_trackers   = kw['show_trackers']
  trackers        = kw['trackers']
  show_horizon    = kw.get('show_horizon', False)
  horizons        = kw.get('horizons', None)
  derived         = kw.get('derived', None)
  plane           = kw.get('plane', 'xy')
  companion       = kw.get("companion", "xz")
  coord_scale     = kw.get('coord_scale', 1.0)
  coord_label     = kw.get('coord_label', r"\mathrm{M}_\odot")
  time_scale      = kw.get('time_scale', 1.0)
  time_tex        = kw.get('time_tex', r"\mathrm{M}_\odot")
  time_axis       = kw.get('time_axis', "Msun")
  hlab, vlab      = plane_axis_labels(plane)

  # Convert the (M_sun) plot limits to the displayed spatial units.
  if xlim is not None: xlim = tuple(v * coord_scale for v in xlim)
  if ylim is not None: ylim = tuple(v * coord_scale for v in ylim)
  if zlim is not None: zlim = tuple(v * coord_scale for v in zlim)

  try:
    data_xy = _load_frame(xy_fpaths)
  except FileNotFoundError as err:
    print(f"  Missing file: {err}, skipping frame {global_idx}")
    return

  data_xz = None
  if xz_fpaths is not None:
    try:
      data_xz = _load_frame(xz_fpaths)
    except FileNotFoundError:
      pass

  if derived is not None:
    fn, args = derived
    data_xy.register_derived_variable(var_name, fn, *args)
    if data_xz is not None:
      data_xz.register_derived_variable(var_name, fn, *args)

  base_xy = _frame_base(data_xy)
  if print_minmax and hasattr(data_xy, 'get_block_data'):
    print_stats(data_xy, var_name, time_scale=time_scale, time_axis=time_axis)

  if data_xz is not None:
    fig = plt.figure(figsize=(4.5, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.03, right=0.83)
    ax_xz = fig.add_subplot(gs[0])
    ax_xy = fig.add_subplot(gs[1], sharex=ax_xz)

    pcm = data_xz.plot_slice(var_name, ax=ax_xz, **plot_kwargs)
    data_xy.plot_slice(var_name, ax=ax_xy, **plot_kwargs)
    rescale_image_extents(ax_xz, coord_scale)
    rescale_image_extents(ax_xy, coord_scale)

    if show_refinement:
      draw_refinement_grid(ax_xz, _frame_base(data_xz), coord_scale=coord_scale)
      draw_refinement_grid(ax_xy, base_xy, coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax_xy, trackers, base_xy.time, plane="xy", coord_scale=coord_scale)
      draw_trackers(ax_xz, trackers, base_xy.time, plane=companion, coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax_xy, horizons, trackers, base_xy.time, plane="xy", coord_scale=coord_scale)
      draw_horizons(ax_xz, horizons, trackers, base_xy.time, plane=companion, coord_scale=coord_scale)

    if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
    if zlim is not None: ax_xz.set_ylim(*zlim)
    if ylim is not None: ax_xy.set_ylim(*ylim)
    plt.setp(ax_xz.get_xticklabels(), visible=False)
    ax_xz.set_xlabel(""); ax_xz.set_ylabel(rf"$z~[{coord_label}]$")
    top_h, _ = plane_axis_labels(companion)
    ax_xy.set_xlabel(rf"$x,\,{top_h}~[{coord_label}]$" if top_h != "x"
                    else rf"$x~[{coord_label}]$")
    ax_xy.set_ylabel(rf"$y~[{coord_label}]$")
    ax_xz.set_title(rf"$t = {base_xy.time * time_scale:.2f}~{time_tex}$")
    ax_xy.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='upper'))
    fig.align_ylabels([ax_xz, ax_xy])
    p_top = ax_xz.get_position(); p_bot = ax_xy.get_position()
    cax = fig.add_axes([p_top.x1 + 0.015, p_bot.y0, 0.04, p_top.y1 - p_bot.y0])
    cb = fig.colorbar(pcm, cax=cax)
    cb.ax.yaxis.set_label_position('right')
    cb.set_label(label or var_name)
  else:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pcm = data_xy.plot_slice(var_name, ax=ax, **plot_kwargs)
    rescale_image_extents(ax, coord_scale)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(pcm, cax=cax, label=label or var_name)

    if show_refinement:
      draw_refinement_grid(ax, base_xy, coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax, trackers, base_xy.time, plane=plane, coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax, horizons, trackers, base_xy.time, plane=plane, coord_scale=coord_scale)

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel(rf"${hlab}~[{coord_label}]$")
    ax.set_ylabel(rf"${vlab}~[{coord_label}]$")
    ax.set_title(rf"$t = {base_xy.time * time_scale:.2f}~{time_tex}$")
    fig.tight_layout()

  fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
              bbox_inches='tight')
  plt.close(fig)

# Render all the frames by setting the worker args.
def render_frames_all(output_dirs, figpath, file_prefix, var_name,
                      cmap, norm=None, vmin=None, vmax=None,
                      label=None, xlim=(-60, 60), ylim=(-60, 60),
                      step=1, out_prefix=None,
                      print_minmax=False,
                      show_trackers=True, show_refinement=True,
                      trackers=None,
                      show_horizon=False, horizons=None,
                      scale_factor=1.0,
                      skip_existing=False,
                      derived=None,
                      plane="xy", companion="xz",
                      subdir=None,coord_scale=1.0,
                      coord_label=r"\mathrm{M}_\odot",
                      time_scale=1.0, time_tex=r"\mathrm{M}_\odot",
                      time_axis="Msun"):
  """Render one PNG per frame across all segments, in parallel."""
  prefixes = [file_prefix] if isinstance(file_prefix, str) else list(file_prefix)
  comp_prefix = None
  if companion != "none" and all("_xy" in pre for pre in prefixes):
    comp_prefix = [pre.replace("_xy", f"_{companion}") for pre in prefixes]
  frames = collect_combined_frames(output_dirs, prefixes, comp_prefix, step)

  if not frames:
    print(f"  No files found for prefix '{file_prefix}'")
    return

  if out_prefix is None:
    out_prefix = f"frame_{var_name.replace(':', '_')}"

  outdir = os.path.join(figpath, subdir) if subdir else figpath
  os.makedirs(outdir, exist_ok=True)

  if skip_existing:
    frames = [(gi, xy, xz) for gi, xy, xz in frames
              if not os.path.exists(os.path.join(outdir, f"{out_prefix}_{gi:05d}.png"))]
    if not frames:
      print(f"  All frames already exist, skipping '{var_name}'")
      return
    print(f"  Skipping existing, rendering {len(frames)} new frame(s)")

  zlim = (0, ylim[1]) if ylim is not None else None
  if derived is not None:
    fn, args = derived
    derived = (functools.partial(_rescaled, fn, scale_factor), args)
    scale_factor = 1.0
  plot_kwargs = dict(cmap=cmap, interpolation="nearest", rescale=scale_factor)
  if norm is not None: plot_kwargs["norm"] = norm
  if vmin is not None: plot_kwargs["vmin"] = vmin
  if vmax is not None: plot_kwargs["vmax"] = vmax

  worker_args = [
    dict(global_idx=global_idx, xy_fpaths=xy_fpaths, xz_fpaths=xz_fpaths,
         var_name=var_name, plot_kwargs=plot_kwargs,
         xlim=xlim, ylim=ylim, zlim=zlim,
         label=label, out_prefix=out_prefix, figpath=outdir,
         print_minmax=print_minmax,
         show_refinement=show_refinement,
         show_trackers=show_trackers, trackers=trackers,
         show_horizon=show_horizon, horizons=horizons,
         derived=derived, plane=plane, companion=companion,
         coord_scale=coord_scale, coord_label=coord_label,
         time_scale=time_scale, time_tex=time_tex, time_axis=time_axis)
    for global_idx, xy_fpaths, xz_fpaths in frames
  ]

  n_workers = min(multiprocessing.cpu_count(), len(worker_args))
  with multiprocessing.Pool(n_workers, initializer=configure_matplotlib) as pool:
    pool.map(_render_frame_worker_all, worker_args)

  print(f"  Rendered {len(frames)} frames for '{var_name}'")

# ---------------------------------------------------------------------------
# Derived average neutrino energy (E:s / N:s)
# ---------------------------------------------------------------------------
def _calc_average_energy(rE, rN):
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(rN > 0, rE / rN, 0.0)

# ---------------------------------------------------------------------------
# Area-weighted average Y_e (s_00) time series
# ---------------------------------------------------------------------------
# Compute the average Ye on a given slice per worker.
def _compute_s00_avg_worker_fpath(fpath):
  """Worker: load one frame by path and return (time, area-weighted mean of s_00)."""
  try:
    data = BinaryData(fpath)
  except FileNotFoundError:
    print(f"  Missing {fpath}, skipping")
    return None

  total_sum  = 0.0
  total_area = 0.0
  for block, arr in data.get_block_data("s_00"):
    ext = block.get_extent()
    if len(ext) < 4:
      continue
    xmin, xmax, ymin, ymax = ext[0], ext[1], ext[2], ext[3]
    block_area  = (xmax - xmin) * (ymax - ymin)
    total_sum  += np.mean(arr) * block_area
    total_area += block_area

  if total_area == 0:
    return None
  return (data.time, total_sum / total_area)

# Plot the 1D time series of avg. Ye against time.
def plot_s00_timeseries_all(output_dirs, figpath, jobname="bhns", file_prefix=None,
                            time_scale=1.0, time_axis="Msun"):
  """Compute and plot the area-weighted average Y_e (s_00) vs time across all segments."""
  time_to_fpath = _collect_frames_by_time(output_dirs, file_prefix)
  if not time_to_fpath:
    print(f"  No files found for prefix '{file_prefix}' across segments")
    return

  sorted_fpaths = [time_to_fpath[t] for t in sorted(time_to_fpath.keys())]

  n_workers = min(multiprocessing.cpu_count(), len(sorted_fpaths))
  with multiprocessing.Pool(n_workers) as pool:
    results = pool.map(_compute_s00_avg_worker_fpath, sorted_fpaths)

  results = sorted((r for r in results if r is not None), key=lambda x: x[0])
  if not results:
    print("  No s_00 data found")
    return

  times  = np.array([r[0] for r in results])
  avg_ye = np.array([r[1] for r in results])

  _plot_timeseries(times * time_scale, avg_ye, r"$\langle Y_e \rangle$", figpath,
                   f"{jobname}_avg_s00_timeseries.png", time_axis=time_axis)

# ---------------------------------------------------------------------------
# System plotter class.
# ---------------------------------------------------------------------------
class SystemPlotter:
  """
  Full-run visualization driver for a AthenaK simulation.
  Options:
    - batchtools segments output-XXXX or single folder
    - overlay AMR grid
    - plot CO trackers or BH apparent-horizons
    - choose between Msun/ms time units and code units/km spatial units
  """
  def __init__(self, simpath, figpath=None, jobname="bhns",
               prim_prefix="mhd_w_bcc", rad_prefix="rad_m1", batchtools=True,
               units='cgs', time_units="Msun", plane="xy", companion="xz",
               domain=(-60,60), show_grid=True, show_trackers=True,
               show_horizon=False, skip_existing=False):
    """
    Initialize the SystemPlotter.

    Parameters:
      simpath (str): Parent directory containing the output-XXXX subdirectories.
      figpath (str): Output figure directory (default: <simpath>/Figs).
      jobname (str): AthenaK job name prefixing all input/output files.
      prim_prefix (str): Primitive-variable file prefix (before "_{plane}") for
                         density/temperature/s00, e.g. "mhd_w_bcc" or "prim".
      rad_prefix (str): M1 radiation file prefix (before "_<var>_{plane}") for
                        E/N/abs/eta, e.g. "rad_m1".
      batchtools (bool): Whether to assume a output-XXXX segment structure.
      units (str): Unit system for physical quantities ('code', 'cgs', 'ngs').
      time_units (str): Time display units for titles/axes, 'Msun' or 'ms'.
      plane (str): Slice plane to render, one of 'xy', 'xz', 'yz'.
      companion (str): Top-panel to render, one of 'xz', 'yz'.
      domain (tuple): Size of the plotting domain in code units.
      show_grid (bool): Overlay the AMR grid.
      show_trackers (bool): Overlay compact-object tracker markers on 2-D slices.
      show_horizon (bool): Overlay black-hole apparent-horizon circles on 2-D slices.
      skip_existing (bool): Skip frames whose output PNG already exists.
    """
    # Check if plane and time units fit the supported formats.
    if plane not in PLANES:
      raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
    if time_units not in TIME_UNITS:
      raise ValueError(f"time_units must be one of {TIME_UNITS}, got {time_units!r}")

    # Paths and file names/prefixes.
    self.simpath       = simpath
    self.figpath       = figpath or os.path.join(simpath, "Figs")
    self.jobname       = jobname
    self.prim_prefix   = prim_prefix
    self.rad_prefix    = rad_prefix
    self.batchtools    = batchtools

    # Plotting-specific choices.
    self.units         = units
    self.time_units    = time_units
    self.plane         = plane
    self.companion     = companion
    self.domain        = domain
    self.show_grid     = show_grid
    self.show_trackers = show_trackers
    self.show_horizon  = show_horizon

    # Set the spatial/temporal units/scales
    self.coord_scale, self.coord_label             = spatial_units(units)
    self.time_scale, self.time_tex, self.time_axis = time_units_scale(time_units)

    self.output_dirs   = find_output_dirs(simpath, batchtools)
    self.skip_existing = skip_existing
    self._trackers     = None
    self._horizons     = None
    self.overrides     = {}
    self.section       = None

  # Frame output name.
  def frame_out(self, name):
    """Output PNG basename prefix for a frame series."""
    suffix = f"_{self.plane}"
    return f"{self.jobname}_frame_{name}{suffix}"

  # Secton output directory.
  def frame_dir(self, name):
    """Per-series output subfolder, named '<diagnostic>_<plane>'."""
    return f"{name}_{self.plane}"

  # Shared tracker/horizon data.
  @property
  def trackers(self):
    """Combined compact-object trackers across all segments."""
    if self._trackers is None:
      self._trackers = load_trackers_all(self.output_dirs, self.jobname)
    return self._trackers

  @property
  def horizons(self):
    """Combined apparent-horizon radii across all segments."""
    if self._horizons is None:
      self._horizons = load_horizons_all(self.output_dirs, self.jobname)
    return self._horizons

  def rad_file(self, var):
    """File prefix of an M1 radiation variable, e.g. '<job>.rad_m1_E_xy'."""
    return f"{self.jobname}.{self.rad_prefix}_{var}_{self.plane}"

  # Shared render function.
  def render_field(self, file_prefix, var_name, name, cmap="inferno",
                   var_key=None, print_minmax=True, derived=None):
    """Render a scalar-field frame series."""
    var_key = var_key or var_name
    scale, label, use_log, lo, hi = plot_params(var_key, self.units)
    u_lo, u_hi, u_cmap = self.overrides.get(self.section, (None, None, None))
    cmap = u_cmap or cmap
    if u_lo is not None: lo = u_lo
    if u_hi is not None: hi = u_hi
    norm = LogNorm(vmin=lo, vmax=hi) if use_log else Normalize(vmin=lo, vmax=hi)
    render_frames_all(self.output_dirs, self.figpath,
                      file_prefix=file_prefix, var_name=var_name,
                      cmap=cmap, norm=norm, label=label,
                      out_prefix=self.frame_out(name), subdir=self.frame_dir(name),
                      xlim=self.domain, ylim=self.domain,
                      print_minmax=print_minmax,
                      show_trackers=self.show_trackers, show_refinement=self.show_grid,
                      trackers=self.trackers,
                      show_horizon=self.show_horizon,
                      horizons=self.horizons if self.show_horizon else None,
                      scale_factor=scale,
                      skip_existing=self.skip_existing, derived=derived,
                      plane=self.plane, companion=self.companion,
                      coord_scale=self.coord_scale, coord_label=self.coord_label,
                      time_scale=self.time_scale, time_tex=self.time_tex,
                      time_axis=self.time_axis)

  # Section renderer.
  def history(self):
    print("[history] Plotting mass & H-norm2...")
    plot_history_all(self.output_dirs, self.figpath, self.jobname,
                     time_scale=self.time_scale, time_axis=self.time_axis)

  def rho_max(self):
    print("[rho_max] Plotting rho_max & alpha_min...")
    plot_user_history_all(self.output_dirs, self.figpath, self.jobname,
                          time_scale=self.time_scale, time_axis=self.time_axis)

  def trackers_plot(self):
    print("[trackers] Plotting object trajectories...")
    plot_tracker_trajectories_all(self.output_dirs, self.figpath, self.jobname,
                                  coord_scale=self.coord_scale,
                                  coord_label=self.coord_label)

  def run_speed(self):
    print("[run_speed] Plotting run speed from logs...")
    plot_run_speed_all(self.output_dirs, self.figpath, self.jobname,
                       time_scale=self.time_scale, time_axis=self.time_axis)

  def density(self):
    print("[density] Rendering density frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "dens",
                       "dens")

  def temperature(self):
    print("[temperature] Rendering temperature frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "temperature",
                       "temperature")

  def s00(self):
    print("[s00] Rendering s_00 (Ye proxy) frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "s_00",
                       "s00")

  def rad_E(self):
    print("[rad_E] Rendering radiation energy E:0 frames...")
    self.render_field(self.rad_file("E"), "E:0", "rad_E0")

  def rad_N(self):
    print("[rad_N] Rendering radiation number density N:0 frames...")
    self.render_field(self.rad_file("N"), "N:0", "rad_N0")

  def rad_N2(self):
    print("[rad_N2] Rendering radiation number density N:2 (heavy lepton) frames...")
    self.render_field(self.rad_file("N"), "N:2", "rad_N2")

  def rad_abs(self):
    print("[rad_abs] Rendering absorption opacity abs_1:0 frames...")
    self.render_field(self.rad_file("abs_1"), "abs_1:0", "abs1_0")

  def rad_abs0(self):
    print("[rad_abs0] Rendering absorption opacity abs_0:0 frames...")
    self.render_field(self.rad_file("abs_0"), "abs_0:0", "abs0_0")

  def rad_eta0(self):
    print("[rad_eta0] Rendering emissivity eta_0:0 frames...")
    self.render_field(self.rad_file("eta_0"), "eta_0:0", "eta0_0")

  def rad_eta1(self):
    print("[rad_eta1] Rendering emissivity eta_1:0 frames...")
    self.render_field(self.rad_file("eta_1"), "eta_1:0", "eta1_0")

  def conH(self):
    print("[z4c] Rendering con_H (Hamiltonian constraint) frames...")
    self.render_field(f"{self.jobname}.con_{self.plane}", "derived:abs_con_H",
                       "con_H", var_key="con_H", derived=(np.abs, ["con_H"]))

  def alpha(self):
    print("[z4c] Rendering z4c_alpha (lapse) frames...")
    self.render_field(f"{self.jobname}.z4c_{self.plane}", "z4c_alpha", "alpha",
                      var_key="alpha")

  def s00_avg(self):
    print("[s00_avg] Computing average Y_e (s_00) time series...")
    plot_s00_timeseries_all(self.output_dirs, self.figpath, self.jobname,
                            file_prefix=f"{self.jobname}.{self.prim_prefix}_{self.plane}",
                            time_scale=self.time_scale, time_axis=self.time_axis)

  def avg_energy(self):
    print("[avg_energy] Rendering average neutrino energy frames...")
    self.render_field([self.rad_file("N"), self.rad_file("E")], "derived:avg_energy",
                      "avg_energy", cmap="viridis", var_key="avg_energy",
                      derived=(_calc_average_energy, ["E:0", "N:0"]))

  def _nu_energy(self, section, species, label):
    print(f"[{section}] Rendering neutrino energy E/{label} (species {species}) frames...")
    self.render_field([self.rad_file("N"), self.rad_file("E")], "derived:avg_energy",
                      section, cmap="viridis", var_key="nu_energy",
                      derived=(_calc_average_energy, [f"E:{species}", f"N:{species}"]))

  # Dispatch table mapping the section name to the render function.
  def _dispatch_table(self):
    """Ordered mapping of section name -> zero-argument callable."""
    table = {
      "history":     self.history,
      "rho_max":     self.rho_max,
      "trackers":    self.trackers_plot,
      "run_speed":   self.run_speed,
      "density":     self.density,
      "temperature": self.temperature,
      "s00":         self.s00,
      "rad_E":       self.rad_E,
      "rad_N":       self.rad_N,
      "rad_N2":      self.rad_N2,
      "rad_abs":     self.rad_abs,
      "rad_abs0":    self.rad_abs0,
      "rad_eta0":    self.rad_eta0,
      "rad_eta1":    self.rad_eta1,
      "conH":        self.conH,
      "alpha":       self.alpha,
      "s00_avg":     self.s00_avg,
      "avg_energy":  self.avg_energy,
    }
    for section, species, label in _NU_ENERGY_SECTIONS:
      table[section] = (lambda sec=section, sp=species, lb=label:
                        self._nu_energy(sec, sp, lb))
    return table

  # Main function to run the renderer.
  def run(self, sections=("all",)):
    """Run the requested sections.

    Parameters:
    sections: iterable of section names or 'name=vmin,vmax,cmap' tokens
              (or parsed tuples), or containing 'all' to run everything.
    """
    specs = [s if isinstance(s, tuple) else parse_section_spec(s) for s in sections]
    self.overrides = {name: (lo, hi, cmap) for name, lo, hi, cmap in specs}
    sections = [name for name, *_ in specs]
    if not self.output_dirs:
      if self.batchtools:
        print(f"ERROR: No output-XXXX directories found under {self.simpath}")
      else:
        print(f"ERROR: {self.simpath} is not a directory")
      raise SystemExit(1)

    configure_matplotlib()
    os.makedirs(self.figpath, exist_ok=True)

    sections = set(sections)
    run_all = "all" in sections

    print(f"SIMPATH  : {self.simpath}")
    print(f"FIGPATH  : {self.figpath}")
    print(f"JOBNAME  : {self.jobname}")
    print(f"Plane    : {self.plane}")
    print(f"Top-panel: {self.companion}")
    print(f"Segments : {len(self.output_dirs)}")
    for d in self.output_dirs:
      print(f"  {d}")
    print(f"Sections : {', '.join(sorted(sections))}")
    print(f"Trackers : {'on' if self.show_trackers else 'off'}")
    print(f"Horizon  : {'on' if self.show_horizon else 'off'}")
    print(f"Domain   : {self.domain}")
    print(f"Units    : {self.units}  (spatial axes in "
          f"{'M_sun' if self.coord_scale == 1.0 else 'km'})")
    print(f"Time     : {self.time_axis}\n")

    table = self._dispatch_table()
    for name, func in table.items():
      if run_all or name in sections:
        self.section = name
        func()

    print("\nDone.")

# ---------------------------------------------------------------------------
# CLI options.
# ---------------------------------------------------------------------------
def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    prog="python3 -m kplot.system",
    description="Multi-segment full-run visualization driver for AthenaK.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--simpath", required=True,
                      help="Parent simulation directory containing output-XXXX subdirs")
  parser.add_argument("--figpath", default=None,
                      help="Output figure directory (default: <simpath>/Figs)")
  parser.add_argument("--jobname", default="bhns",
                      help="AthenaK job name prefixing input/output files (default: bhns)")
  parser.add_argument("--prim-prefix", dest="prim_prefix", default="mhd_w_bcc",
                      help="Primitive-variable file prefix (before '_<plane>') for "
                           "density/temperature/s00, e.g. 'mhd_w_bcc' [default] or 'prim'")
  parser.add_argument("--rad-prefix", dest="rad_prefix", default="rad_m1",
                      help="M1 radiation file prefix (before '_<var>_<plane>') for "
                           "E/N/abs/eta, e.g. 'rad_m1' [default]")
  parser.add_argument("--plane", default="xy", choices=list(PLANES),
                      help="Slice plane to render: xy [default], xz, or yz")
  parser.add_argument("--companion", default="xz", choices=["xz", "yz", "none"],
                        help="Companion panel to the main panel: xz [default], yz, or none")
  parser.add_argument("--time-units", dest="time_units", default="Msun",
                      choices=list(TIME_UNITS),
                      help="Time display units for titles/axes: Msun [default] or ms")
  parser.add_argument("--no-trackers", dest="no_trackers", action="store_true",
                      help="Disable tracker overlays on all 2-D slice plots")
  parser.add_argument("--show-grid", dest="show_grid", action="store_true",
                      help="Overlay AMR grid")
  parser.add_argument("--show-horizon", dest="show_horizon", action="store_true",
                      help="Overlay black-hole apparent-horizon circles on 2-D slice plots")
  parser.add_argument("--domain", dest="domain", default=(-60,60),
                      type=lambda s: tuple(float(x) for x in s.strip("()[] ").split(",")),
                      help="Tuple holding the size of the plotting domain in code units.")
  parser.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                      help="Skip frames whose output PNG already exists in figpath")
  parser.add_argument("--batchtools", action=argparse.BooleanOptionalAction, default=True,
                      help="Read data from output-XXXX subdirs of simpath [default]; "
                           "--no-batchtools reads it directly from simpath")
  parser.add_argument("--sections", nargs="+", default=None, type=parse_section_spec,
                      metavar="SECTION[=VMIN,VMAX,CMAP]",
                      help="Which sections to run, with optional colorbar limits and "
                           "colormap, e.g. density=1e6,1e15,inferno s00=,,plasma "
                           "(default: [sections] of --config, else all)")
  parser.add_argument("--units", default="cgs",
                      choices=["code", "cgs", "ngs"],
                      help="Unit system for physical quantities: "
                           "'code' = raw AthenaK GeometricKilometer units, "
                           "'cgs' = Gaussian CGS (g, cm, s, erg) [default], "
                           "'ngs' = bns_nurates units (nm, g, s, MeV)")
  parser.add_argument("--config", default=None, help="config.ini with a [sections] section")
  return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv=None):
  args = parse_args(argv)
  plotter = SystemPlotter(
    simpath=args.simpath,
    figpath=args.figpath,
    jobname=args.jobname,
    prim_prefix=args.prim_prefix,
    rad_prefix=args.rad_prefix,
    batchtools=args.batchtools,
    units=args.units,
    time_units=args.time_units,
    plane=args.plane,
    companion=args.companion,
    domain=args.domain,
    show_grid=args.show_grid,
    show_trackers=not args.no_trackers,
    show_horizon=args.show_horizon,
    skip_existing=args.skip_existing,
  )
  sections = args.sections or load_sections(args.config) or ["all"]
  plotter.run(sections)

if __name__ == "__main__":
  main()
