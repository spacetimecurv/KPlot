"""
Global plotter function of AthenaK .bin slices.

Combines all output-XXXX restart segments of an AthenaK simulation and
produces a full-run set of diagnostics: 1-D history/time-series plots
and parallel 2-D slice frames (density, temperature, Y_e, radiation-M1
fields, neutrino energies) with optional NS-tracker, black-hole horizon
and AMR-grid overlays.

Binary files have to either exist in a batchtools substructure, i.e.
output-XXXX (in this case pass --batchtools) or in a single folder.

Each frame series is written to its own figpath/<diagnostic>_<plane>/ subfolder.

Use in Python:
    from kplot.system.plotter import SystemPlotter
    SystemPlotter(simpath="/path/to/sim", units="cgs").run(["density", "history"])

Import necessary standard libraries.
"""

# Built-in libraries.
import argparse
import functools
import glob
import multiprocessing
import os
import re
import configparser
from collections.abc import Callable

# Import necessary third-party libraries.
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LogNorm, Normalize, to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

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
  'axes.formatter.use_mathtext': True,
  'axes.unicode_minus': False,
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
  ('con_M',     'code'): r"$|\mathcal{M}|$",
  ('con_M',     'ngs' ): r"$|\mathcal{M}|$",
  ('con_M',     'cgs' ): r"$|\mathcal{M}|$",
  ('con_C',     'code'): r"$|\mathcal{C}|$",
  ('con_C',     'ngs' ): r"$|\mathcal{C}|$",
  ('con_C',     'cgs' ): r"$|\mathcal{C}|$",
  ('con_Z',     'code'): r"$|\mathcal{Z}|$",
  ('con_Z',     'ngs' ): r"$|\mathcal{Z}|$",
  ('con_Z',     'cgs' ): r"$|\mathcal{Z}|$",
  ('alpha',     'code'): r"$\alpha$",
  ('alpha',     'ngs' ): r"$\alpha$",
  ('alpha',     'cgs' ): r"$\alpha$",
}

# Variables that use a linear (not log) colorscale
_LINEAR_VARS = {'temperature', 's_00', 'nu_energy', 'alpha'}

# Convert geometrized to physical units.
MSUN_TO_KM = au.unit_len_cgs / 1.0e5
MSUN_TO_MS = (au.unit_len_cgs / au.cgs.c) * 1.0e3

# Default density contours.
_DEFAULT_DENS_CONT = (1e9, 1e10, 1e11, 1e12, 1e13)

# Choose spatial coordinates for the plot.
def spatial_units(units: str) -> tuple[float, str]:
  """
  Choose the scale and label of the spatial axes.

  Parameters:
  units (str): Unit system, 'code', 'cgs' or 'ngs'.

  Returns:
  (coord_scale, axis_label).
  """
  if units == "code":
    return 1.0, r"\mathrm{M}_\odot"
  return MSUN_TO_KM, r"\mathrm{km}"

# Choose time units display for the plot.
TIME_UNITS = ("Msun", "ms")
def time_units_scale(time_units: str) -> tuple[float, str, str]:
  """
  Choose the scale and labels of the displayed time.

  Parameters:
  time_units (str): 'Msun' or 'ms'.

  Returns:
  (scale, latex_label, axis_label).
  """
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
def load_sections(path: str | None) -> list[str]:
  """
  Read the [sections] entries of a config.ini.

  Parameters:
  path (str): Path to the config.ini (None gives no sections).

  Returns:
  List of 'name[=lo,hi,cmap]' tokens.
  """
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
def parse_section_spec(token: str) -> tuple[str, float | None, float | None, str | None]:
  """
  Parse a 'name[=lo,hi,cmap]' section token.

  Parameters:
  token (str): Section token.

  Returns:
  (name, vmin, vmax, cmap); empty fields become None.
  """
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
  "rad_eta0", "rad_eta1", "alpha",
  "s00_avg", "avg_energy",
  "conH", "conM", "conC", "conZ",
  "nu_energy", "nu_energy1", "nu_energy2",
]

# ---------------------------------------------------------------------------
# Configure plotting environment.
# ---------------------------------------------------------------------------
# Configure mpl.
def configure_matplotlib() -> None:
  """
  Select the non-interactive Agg backend and apply the plot style.

  Returns:
  None.
  """
  matplotlib.use("Agg", force=True)
  mpl.rcParams.update(_STYLE)

# Slice planes selectable for 2-D frame rendering, and their axis letters.
PLANES = ("xy", "xz", "yz")
_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}

# Fetch the coordinate labels for the given plane.
def plane_axis_labels(plane: str) -> tuple[str, str]:
  """
  Coordinate letters of a slice plane.

  Parameters:
  plane (str): 'xy', 'xz' or 'yz'.

  Returns:
  (horizontal, vertical) coordinate letters.
  """
  return _PLANE_AXES[plane]

# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------
# List of binary files.
def glob_bin_files(directory: str, prefix: str) -> list[str]:
  """
  Find the binary files prefix.NNNNN.bin in directory (or its bin/ subfolder).

  Parameters:
  directory (str): Segment directory.
  prefix (str): File prefix, e.g. '<job>.mhd_w_bcc_xy'.

  Returns:
  Sorted list of file paths.
  """
  # If there is no 'bin' directory, then directly search in the parent folder.
  for sub in ("bin", ""):
    files = glob.glob(os.path.join(directory, sub, f"{prefix}.?????.bin"))
    if files:
      return sorted(files)
  return []

# Print stats from the frame.
def print_stats(data: BinaryData, var_name: str, time_scale: float = 1.0,
                time_axis: str = "Msun") -> None:
  """
  Print the frame time plus min/max of var_name across all blocks.

  Parameters:
  data (BinaryData): Loaded frame.
  var_name (str): Variable name.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
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
  def __init__(self, time: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Parameters:
    time (np.ndarray): Times.
    x, y, z (np.ndarray): Tracker positions.
    """
    self.time = time
    self.x = x
    self.y = y
    self.z = z

# Horizon class holding the times/radii.
class _CombinedHorizon:
  """Lightweight horizon holding concatenated time/radius arrays from all segments."""
  def __init__(self, time: np.ndarray, radius: np.ndarray) -> None:
    """
    Parameters:
    time (np.ndarray): Times.
    radius (np.ndarray): Mean horizon radii.
    """
    self.time = time
    self.radius = radius

# Select the tracker position in 2D based on the specified plane.
def _tracker_plane_coords(tracker: _CombinedTracker,
                          plane: str) -> tuple[np.ndarray, np.ndarray]:
  """
  Tracker coordinates in a slice plane.

  Parameters:
  tracker (_CombinedTracker): Tracker data.
  plane (str): 'xy', 'xz' or 'yz'.

  Returns:
  (horizontal, vertical) coordinate arrays.
  """
  if plane == "xy":
    return tracker.x, tracker.y
  if plane == "xz":
    return tracker.x, tracker.z
  return tracker.y, tracker.z

# Load the CO tracker files into a CombinedTracker object.
def load_trackers_all(output_dirs: list[str],
                      jobname: str = "bhns") -> list[_CombinedTracker | None]:
  """
  Load and concatenate the tracker files co_0/co_1 across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  jobname (str): AthenaK job name.

  Returns:
  Both trackers, None where no file is found.
  """
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
def load_horizons_all(output_dirs: list[str],
                      jobname: str = "bhns") -> list[_CombinedHorizon | None]:
  """
  Load and concatenate the apparent-horizon summary files across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  jobname (str): AthenaK job name.

  Returns:
  Both horizons, None where no file is found.
  """
  horizons = []
  for k in (0, 1):
    times_list, r_list = [], []
    for d in output_dirs:
      fpath = os.path.join(d, f"{jobname}.horizon_summary_{k}.txt")
      if os.path.exists(fpath):
        try:
          with open(fpath) as f:
            headers = re.findall(r"\d+:(\S+)", f.readline())
          t_col = headers.index("time") if "time" in headers else 1
          r_col = headers.index("meanradius") if "meanradius" in headers else -1
          arr = np.loadtxt(fpath, comments="#")
          if arr.ndim == 1:
            arr = arr.reshape(1, -1)
          if arr.size > 0:
            times_list.append(arr[:, t_col])
            r_list.append(arr[:, r_col])
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
def draw_trackers(ax: plt.Axes, trackers: list, time: float, plane: str = "xy",
                  colors: tuple = ("white", "white"), marker: str = "x",
                  markersize: float = 20, coord_scale: float = 1.0,
                  show_path: bool = False, path_length: float | None = None,
                  path_fade: bool = True, path_lw: float = 2.0,
                  path_alpha: float = 0.9, path_nseg: int = 50) -> None:
  """
  Overlay compact-object positions on ax at the given simulation time.

  Parameters:
  ax (plt.Axes): Axes to draw on.
  trackers (list): Trackers from load_trackers_all.
  time (float): Simulation time.
  plane (str): Slice plane.
  colors (tuple): Marker color per tracker.
  marker (str): Marker style.
  markersize (float): Marker size.
  coord_scale (float): Factor converting code lengths to the axis units.
  show_path (bool): Also draw the trajectory ip to the current time.
  path_length(float): Length of the drawn trail in code time (None = full history).
  path_fade (bool): Fade the trail from transparent (oldest) to opaque (current).
  path_lw (float): Trail line width.
  path_alpha (float): Trail opacity at the current position.
  path_nseg (int): Maximum number of segments of the faded trail.

  Returns:
  None.
  """
  labels = ["CO 1", "CO 2"]
  for tracker, color, label in zip(trackers, colors, labels):
    if tracker is None:
      continue
    idx = np.argmin(np.abs(tracker.time - time))
    h, v = _tracker_plane_coords(tracker, plane)
    if show_path:
      t_start = tracker.time[0] if path_length is None else time - path_length
      lo = 0 if path_length is None else np.searchsorted(tracker.time, time - path_length)
      if idx - lo >= 1:
        pts = np.column_stack([h[lo:idx + 1], v[lo:idx + 1]]) * coord_scale
        if path_fade:
          sel = np.unique(np.linspace(0, idx - lo, min(path_nseg, idx - lo) + 1).round().astype(int))
          pts = pts[sel]
          t_pts = tracker.time[lo:idx + 1][sel]
          segs = np.stack([pts[:-1], pts[1:]], axis=1)
          t_seg = 0.5 * (t_pts[:-1] + t_pts[1:])
          t0, t1 = t_pts[0], t_pts[-1]
          frac = (t_seg - t0) / (t1 - t0) if t1 > t0 else np.ones_like(t_seg)
          rgba = np.tile(to_rgba(color), (len(segs), 1))
          rgba[:, 3] = path_alpha * np.clip(frac, 0.0, 1.0)
          ax.add_collection(LineCollection(segs, colors=rgba, linewidths=path_lw,
                                           capstyle="butt", zorder=9.5))
        else:
          ax.plot(pts[:, 0], pts[:, 1], color=color, lw=path_lw,
                  alpha=path_alpha, zorder=9.5)
          
    ax.scatter(h[idx] * coord_scale, v[idx] * coord_scale,
               marker=marker, c=color, s=markersize,
               linewidths=1.5, zorder=10, label=label)

# Overlay apparent-horizon position.
def draw_horizons(ax: plt.Axes, horizons: list | None, trackers: list | None, time: float,
                  plane: str = "xy", coord_scale: float = 1.0, edgecolor: str = "white",
                  facecolor: str = "black") -> None:
  """
  Overlay filled apparent-horizon circles at the paired tracker positions.

  Parameters:
  ax (plt.Axes): Axes to draw on.
  horizons (list): Horizons from load_horizons_all.
  trackers (list): Trackers from load_trackers_all.
  time (float): Simulation time.
  plane (str): Slice plane.
  coord_scale (float): Factor converting code lengths to the axis units.
  edgecolor (str): Circle edge color.
  facecolor (str): Circle fill color.

  Returns:
  None.
  """
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
# Contour helpers.
# ---------------------------------------------------------------------------
# Resample AMR block data onto a uniform grid.
def _resample_uniform(data: BinaryData, var: str, extent: tuple,
                      n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Resample a 2-D slice onto a uniform n x n grid; finer levels overwrite coarser ones.

  Parameters:
  data (BinaryData): Loaded 2-D slice.
  var (str): Variable to resample.
  extent (tuple): (x0, x1, y0, y1) of the grid in code units.
  n (int): Number of cells per direction.

  Returns:
  (x cell centers, y cell centers, grid[y, x]); NaN where no block covers the grid.
  """
  x0, x1, y0, y1 = extent
  xc = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
  yc = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
  grid = np.full((n, n), np.nan)
  for block, arr in sorted(data.get_block_data(var), key=lambda b: b[0].level):
    bx0, bx1, by0, by1 = block.get_extent()
    ny, nx = arr.shape                
    mi = (xc >= bx0) & (xc < bx1); mj = (yc >= by0) & (yc < by1)
    if not (mi.any() and mj.any()): continue
    fi = np.clip((xc[mi] - bx0) / (bx1 - bx0) * nx - 0.5, 0, nx - 1)
    fj = np.clip((yc[mj] - by0) / (by1 - by0) * ny - 0.5, 0, ny - 1)
    i0 = np.minimum(fi.astype(int), nx - 2); wi = fi - i0
    j0 = np.minimum(fj.astype(int), ny - 2); wj = (fj - j0)[:, None]
    lo = arr[j0][:, i0] * (1 - wi) + arr[j0][:, i0 + 1] * wi
    hi = arr[j0 + 1][:, i0] * (1 - wi) + arr[j0 + 1][:, i0 + 1] * wi
    grid[np.ix_(mj, mi)] = lo * (1 - wj) + hi * wj
  return xc, yc, grid

# Draw density contours.
def draw_density_contours(ax: plt.Axes, fpath: str | None, levels: tuple | None,
                          extent: tuple | None, coord_scale: float = 1.0,
                          color: str = "w", linewidth: float = 0.6) -> None:
  """
  Overlay density contours from a primitive-variable slice.

  Parameters:
  ax (plt.Axes): Axes to draw on.
  fpath (str): Slice file holding 'dens'; nothing is drawn if None.
  levels (tuple): Contour levels in code units; nothing is drawn if None.
  extent (tuple): (x0, x1, y0, y1) of the contoured region in code units
                  (default: the full slice).
  coord_scale (float): Factor converting code lengths to the axis units.
  color (str): Line color.
  linewidth (float): Line width.

  Returns:
  None.
  """
  if fpath is None or not levels:
    return
  try:
    data = BinaryData(fpath)
  except FileNotFoundError:
    return
  if 'dens' not in data.variable_names:
    return
  if extent is None:
    exts = np.array([b.get_extent()[:4] for b in data.blocks])
    extent = (exts[:, 0].min(), exts[:, 1].max(), exts[:, 2].min(), exts[:, 3].max())

  xc, yc, rho = _resample_uniform(data, 'dens', extent)
  with np.errstate(divide='ignore', invalid='ignore'):
    log_rho = np.ma.masked_invalid(np.log10(rho))
  log_levels = np.log10(sorted(levels))
  cs = ax.contour(xc * coord_scale, yc * coord_scale, log_rho, levels=log_levels,
                  colors=color, linewidths=linewidth, linestyles="solid", zorder=4)

  def fmt(v: float) -> str:
    """
    Contour label in g/cm^3.

    Parameters:
    v (float): log10 of the level in code units.

    Returns:
    LaTeX label.
    """
    mant, exp = f"{10**v * au.CODE2CGS_DENS:.1e}".split("e")
    mant, exp = float(mant), int(exp)
    return rf"$10^{{{exp}}}$" if mant == 1.0 else rf"${mant:g}\times10^{{{exp}}}$"

  ax.clabel(cs, fmt=fmt, fontsize=5, inline=True)

# ---------------------------------------------------------------------------
# Data segment helpers.
# ---------------------------------------------------------------------------
# Read the time from the binary file.
def _read_frame_time(fpath: str) -> tuple[str, float | None]:
  """
  Read the simulation time from a binary data file.

  Parameters:
  fpath (str): Path to the .bin file.

  Returns:
  (fpath, time), with time None if the file cannot be read.
  """
  try:
    return (fpath, BinaryData(fpath).time)
  except Exception:
    return (fpath, None)

# Find the output directories under the simulation path.
def find_output_dirs(simpath: str, batchtools: bool = True) -> list[str]:
  """
  Find the output-XXXX segment directories under simpath.

  Parameters:
  simpath (str): Simulation directory.
  batchtools (bool): Whether simpath holds batchtools output-XXXX segments.

  Returns:
  Sorted list of segment directories ([simpath] without batchtools).
  """
  if not batchtools:
    return [simpath] if os.path.isdir(simpath) else []
  pattern = os.path.join(simpath, "output-[0-9][0-9][0-9][0-9]")
  return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))

# Collec the frames from the segments.
def _collect_frames_by_time(output_dirs: list[str], file_prefix: str) -> dict[float, str]:
  """
  Gather all files for file_prefix across output_dirs; later segments win.

  Parameters:
  output_dirs (list[str]): Segment directories.
  file_prefix (str): File prefix.

  Returns:
  {time: path}.
  """
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
def collect_combined_frames(output_dirs: list[str], file_prefix: str | list[str],
                            comp_prefix: str | list[str] | None = None,
                            step: int = 1) -> list[tuple[int, list[str], list[str] | None]]:
  """
  Collect all frames for file_prefix across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  file_prefix (str or list[str]): Prefix(es) that must all exist at a time.
  comp_prefix (str or list[str]): Prefix(es) of the companion-plane files.
  step (int): Use every step-th frame.

  Returns:
  [(frame index, file paths, companion file paths or None)].
  """
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
def combine_history(output_dirs: list[str], filename: str) -> dict | None:
  """
  Load and merge a history file across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  filename (str): History file name, e.g. '<job>.mhd.hst'.

  Returns:
  {column: data} sorted by time, or None if no file is found.
  """
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
def _dedup_by_time(all_times: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
  """
  Sort by time and keep the last occurrence of each time.

  Parameters:
  all_times (np.ndarray): Times.
  *arrays (np.ndarray): Arrays aligned with all_times.

  Returns:
  (times, *arrays) after sorting and deduplication.
  """
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
def _plot_timeseries(time: np.ndarray, values: np.ndarray, ylabel: str, figpath: str,
                     fname: str, time_axis: str = "Msun", logy: bool = False) -> None:
  """
  Plot and save one time series.

  Parameters:
  time (np.ndarray): Times.
  values (np.ndarray): Values.
  ylabel (str): y-axis label.
  figpath (str): Output directory.
  fname (str): Output file name.
  time_axis (str): Displayed time unit.
  logy (bool): Log-scale y-axis.

  Returns:
  None.
  """
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
def _plot_history_columns(output_dirs: list[str], figpath: str, jobname: str, suffix: str,
                          columns: list[tuple], time_scale: float = 1.0,
                          time_axis: str = "Msun") -> None:
  """
  Plot columns of a history file, one figure each.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  suffix (str): History file suffix, e.g. 'mhd.hst'.
  columns (list[tuple]): (column, ylabel, name, logy) per figure.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
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
def plot_history_all(output_dirs: list[str], figpath: str, jobname: str = "bhns",
                     time_scale: float = 1.0, time_axis: str = "Msun") -> None:
  """
  Plot the baryonic mass and Hamiltonian constraint norm across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
  _plot_history_columns(output_dirs, figpath, jobname, "mhd.hst",
                        [("mass", r"$M_b$", "history_mass", False)],
                        time_scale=time_scale, time_axis=time_axis)
  _plot_history_columns(output_dirs, figpath, jobname, "z4c.user.hst",
                        [("H-norm2", r"$||\mathcal{H}||_2$", "history_hnorm2", True)],
                        time_scale=time_scale, time_axis=time_axis)

# Plot the max. density and min. lapse.
def plot_user_history_all(output_dirs: list[str], figpath: str, jobname: str = "bhns",
                          time_scale: float = 1.0, time_axis: str = "Msun") -> None:
  """
  Plot the max density and min lapse from <jobname>.user.hst across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
  _plot_history_columns(output_dirs, figpath, jobname, "user.hst",
                        [("rho-max", r"$\rho_\mathrm{max}$", "user_rho_max", True),
                         ("alpha-min", r"$\alpha_\mathrm{min}$", "user_alpha_min", False)],
                        time_scale=time_scale, time_axis=time_axis)

# Plot the CO trajectories.
def plot_tracker_trajectories_all(output_dirs: list[str], figpath: str, jobname: str = "bhns",
                                  coord_scale: float = 1.0,
                                  coord_label: str = r"\mathrm{M}_\odot") -> None:
  """
  Plot compact-object trajectories (raw + smoothed) from tracker data.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  coord_scale (float): Factor converting code lengths to the axis units.
  coord_label (str): LaTeX label of the axis units.

  Returns:
  None.
  """
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
def _parse_run_speed_log(logfile: str) -> tuple[np.ndarray | None, np.ndarray | None]:
  """
  Parse one AthenaK log file. Elapsed time in each log resets to 0 at the
  start of that segment, so each log is processed independently.

  Parameters:
  logfile (str): Path to the log file.

  Returns:
  (simulation time, cycles per wall-second), or (None, None).
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
def plot_run_speed_all(output_dirs: list[str], figpath: str, jobname: str = "bhns",
                       time_scale: float = 1.0, time_axis: str = "Msun") -> None:
  """
  Plot cycles/second vs simulation time, stitched across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
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
def plot_params(var: str, units: str = 'code') -> tuple[float, str, bool, float | None,
                                                        float | None]:
  """
  Render parameters for var in units.

  Parameters:
  var (str): Variable name (e.g. 'dens', 'abs_1:0').
  units (str): 'code', 'cgs', or 'ngs'.

  Returns:
  (scale, label, use_log, vmin, vmax): scale multiplies the code values, use_log
  selects LogNorm, and vmin/vmax are suggested colorbar limits in target units.
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
def _rescaled(fn: Callable, scale: float, *args) -> np.ndarray:
  """
  Evaluate fn(*args) and multiply by scale.

  Parameters:
  fn (callable): Derived-variable function.
  scale (float): Factor applied to the result.
  *args: Arguments of fn.

  Returns:
  fn(*args) * scale.
  """
  return fn(*args) * scale

# Load frame with binary data.
def _load_frame(fpaths: list[str]) -> BinaryData | GroupData:
  """
  Load one frame from one or several files.

  Parameters:
  fpaths (list[str]): Paths of the frame files.

  Returns:
  BinaryData for one file, GroupData for several.
  """
  return BinaryData(fpaths[0]) if len(fpaths) == 1 else GroupData(fpaths)

# Load frame base.
def _frame_base(data: BinaryData | GroupData) -> BinaryData:
  """
  First dataset of a frame, holding time and blocks.

  Parameters:
  data (BinaryData or GroupData): Loaded frame.

  Returns:
  The (first) BinaryData.
  """
  return data.datasets[0] if isinstance(data, GroupData) else data

# Rescale image extent by coordinate scale.
def rescale_image_extents(ax: plt.Axes, coord_scale: float) -> None:
  """
  Multiply the extent of every image already drawn on ax by coord_scale.

  Parameters:
  ax (plt.Axes): Axes holding the images.
  coord_scale (float): Factor applied to the extents.

  Returns:
  None.
  """
  if coord_scale == 1.0:
    return
  for im in ax.get_images():
    x0, x1, y0, y1 = im.get_extent()
    im.set_extent([x0 * coord_scale, x1 * coord_scale,
                   y0 * coord_scale, y1 * coord_scale])

# Draw AMR grid structure.
def draw_refinement_grid(ax: plt.Axes, data: BinaryData, alpha: float = 0.4,
                         linewidth: float = 0.6, coord_scale: float = 1.0) -> None:
  """
  Draw block boundaries colored by AMR refinement level.

  Parameters:
  ax (plt.Axes): Axes to draw on.
  data (BinaryData): Loaded frame.
  alpha (float): Line transparency.
  linewidth (float): Line width.
  coord_scale (float): Factor converting code lengths to the axis units.

  Returns:
  None.
  """
  levels = sorted(set(b.level for b in data.blocks))
  if len(levels) <= 1:
    return

  cmap = mpl.colormaps.get_cmap("cool")
  level_min, level_max = levels[0], levels[-1]

  def level_color(lv: int) -> tuple:
    """
    Color of a refinement level.

    Parameters:
    lv (int): Refinement level.

    Returns:
    RGBA color.
    """
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
def _render_frame_worker_all(kw: dict) -> None:
  """
  Per-frame plot worker for combined multi-segment data.

  Parameters:
  kw (dict): Frame files and plot settings, as built in render_frames_all.

  Returns:
  None.
  """
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
  show_path       = kw.get('show_path', False)
  path_length     = kw.get('path_length', None)
  path_fade       = kw.get('path_fade', True)
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
  contour_levels  = kw.get('contour_levels', None)
  dens_xy_fpath   = kw.get('dens_xy_fpath', None)
  dens_xz_fpath   = kw.get('dens_xz_fpath', None)
  hlab, vlab      = plane_axis_labels(plane)

  # Contour regions in code units, before the limits are rescaled.
  ext_xy = (*xlim, *ylim) if xlim is not None and ylim is not None else None
  ext_xz = (*xlim, *zlim) if xlim is not None and zlim is not None else None

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
      draw_trackers(ax_xy, trackers, base_xy.time, plane="xy", coord_scale=coord_scale,
                    show_path=show_path, path_length=path_length, path_fade=path_fade)
      draw_trackers(ax_xz, trackers, base_xy.time, plane=companion, coord_scale=coord_scale,
                    show_path=show_path, path_length=path_length, path_fade=path_fade)
    if show_horizon:
      draw_horizons(ax_xy, horizons, trackers, base_xy.time, plane="xy", coord_scale=coord_scale)
      draw_horizons(ax_xz, horizons, trackers, base_xy.time, plane=companion, coord_scale=coord_scale)

    if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
    if zlim is not None: ax_xz.set_ylim(*zlim)
    if ylim is not None: ax_xy.set_ylim(*ylim)
    # After the limits, so the inline labels are placed in the final view.
    if contour_levels is not None:
      draw_density_contours(ax_xy, dens_xy_fpath, contour_levels, ext_xy, coord_scale)
      draw_density_contours(ax_xz, dens_xz_fpath, contour_levels, ext_xz, coord_scale)
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
      draw_trackers(ax, trackers, base_xy.time, plane=plane, coord_scale=coord_scale,
                    show_path=show_path, path_length=path_length, path_fade=path_fade)
    if show_horizon:
      draw_horizons(ax, horizons, trackers, base_xy.time, plane=plane, coord_scale=coord_scale)

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    if contour_levels is not None:
      draw_density_contours(ax, dens_xy_fpath, contour_levels, ext_xy, coord_scale)
    ax.set_xlabel(rf"${hlab}~[{coord_label}]$")
    ax.set_ylabel(rf"${vlab}~[{coord_label}]$")
    ax.set_title(rf"$t = {base_xy.time * time_scale:.2f}~{time_tex}$")
    fig.tight_layout()

  fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
              bbox_inches='tight')
  plt.close(fig)

# Render all the frames by setting the worker args.
def render_frames_all(output_dirs: list[str], figpath: str, file_prefix: str | list[str],
                      var_name: str, cmap: str, norm: Normalize | None = None,
                      vmin: float | None = None, vmax: float | None = None,
                      label: str | None = None, xlim: tuple = (-60, 60),
                      ylim: tuple = (-60, 60), step: int = 1, out_prefix: str | None = None,
                      print_minmax: bool = False, show_trackers: bool = True,
                      show_refinement: bool = True, trackers: list | None = None,
                      show_horizon: bool = False, horizons: list | None = None,
                      scale_factor: float = 1.0, skip_existing: bool = False,
                      derived: tuple | None = None, plane: str = "xy",
                      companion: str = "xz", subdir: str | None = None,
                      coord_scale: float = 1.0, coord_label: str = r"\mathrm{M}_\odot",
                      time_scale: float = 1.0, time_tex: str = r"\mathrm{M}_\odot",
                      time_axis: str = "Msun", contour_levels: tuple | None = None,
                      dens_prefix: str | None = None,
                      show_path: bool = False, path_length: float | None = None,
                      path_fade: bool = True) -> None:
  """
  Render one PNG per frame across all segments, in parallel.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  file_prefix (str or list[str]): Prefix(es) of the slice files.
  var_name (str): Variable to plot, or 'derived:<name>' with derived.
  cmap (str): Colormap.
  norm (Normalize): Colorbar norm.
  vmin (float): Colorbar minimum.
  vmax (float): Colorbar maximum.
  label (str): Colorbar label.
  xlim (tuple): Horizontal plot range.
  ylim (tuple): Vertical plot range.
  step (int): Use every step-th frame.
  out_prefix (str): Output PNG basename prefix.
  print_minmax (bool): Print min/max of each frame.
  show_trackers (bool): Overlay the tracker positions.
  show_refinement (bool): Overlay the AMR grid.
  trackers (list): Trackers from load_trackers_all.
  show_horizon (bool): Overlay the apparent horizons.
  horizons (list): Horizons from load_horizons_all.
  scale_factor (float): Factor applied to the plotted values.
  skip_existing (bool): Skip frames whose PNG already exists.
  derived (tuple): (function, [variable names]) of a derived variable.
  plane (str): Slice plane.
  companion (str): Companion-plane top panel, 'xz', 'yz' or 'none'.
  subdir (str): Subfolder of figpath for the PNGs.
  coord_scale (float): Factor converting code lengths to the axis units.
  coord_label (str): LaTeX label of the axis units.
  time_scale (float): Factor converting the time to the displayed units.
  time_tex (str): LaTeX label of the time unit.
  time_axis (str): Displayed time unit.
  contour_levels (tuple): Density contour levels in g/cm^3 (None: no contours).
  dens_prefix (str): Prefix of the slice files holding 'dens' for the contours
                     (default: file_prefix).
  show_path (bool): Also draw the trajectory ip to the current time.
  path_length(float): Length of the drawn trail in code time (None = full history).
  path_fade (bool): Fade the trail from transparent (oldest) to opaque (current).

  Returns:
  None.
  """
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

  # Match every frame to the density slices at the same time.
  dens_fpaths = {}
  if contour_levels is not None:
    contour_levels = tuple(np.asarray(contour_levels) / au.CODE2CGS_DENS)
    dens_prefix = dens_prefix or prefixes[0]
    frame_time = {p: t for t, p in _collect_frames_by_time(output_dirs, prefixes[0]).items()}
    dens_xy = _collect_frames_by_time(output_dirs, dens_prefix)
    dens_comp = {}
    if comp_prefix is not None and "_xy" in dens_prefix:
      dens_comp = _collect_frames_by_time(output_dirs,
                                          dens_prefix.replace("_xy", f"_{companion}"))
    for gi, xy, _ in frames:
      t = frame_time.get(xy[0])
      dens_fpaths[gi] = (dens_xy.get(t), dens_comp.get(t))

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
         time_scale=time_scale, time_tex=time_tex, time_axis=time_axis,
         contour_levels=contour_levels,
         dens_xy_fpath=dens_fpaths.get(global_idx, (None, None))[0],
         dens_xz_fpath=dens_fpaths.get(global_idx, (None, None))[1],
         show_path=show_path, path_length=path_length, path_fade=path_fade)
    for global_idx, xy_fpaths, xz_fpaths in frames
  ]

  n_workers = min(multiprocessing.cpu_count(), len(worker_args))
  with multiprocessing.Pool(n_workers, initializer=configure_matplotlib) as pool:
    pool.map(_render_frame_worker_all, worker_args)

  print(f"  Rendered {len(frames)} frames for '{var_name}'")

# ---------------------------------------------------------------------------
# Derived average neutrino energy (E:s / N:s)
# ---------------------------------------------------------------------------
def _calc_average_energy(rE: np.ndarray, rN: np.ndarray) -> np.ndarray:
  """
  Average neutrino energy E/N (0 where N = 0).

  Parameters:
  rE (np.ndarray): Radiation energy density.
  rN (np.ndarray): Radiation number density.

  Returns:
  E/N.
  """
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(rN > 0, rE / rN, 0.0)

# ---------------------------------------------------------------------------
# Area-weighted average Y_e (s_00) time series
# ---------------------------------------------------------------------------
# Compute the average Ye on a given slice per worker.
def _compute_s00_avg_worker_fpath(fpath: str) -> tuple[float, float] | None:
  """
  Load one frame and compute the area-weighted mean of s_00.

  Parameters:
  fpath (str): Path to the .bin file.

  Returns:
  (time, mean s_00), or None if the frame cannot be used.
  """
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
def plot_s00_timeseries_all(output_dirs: list[str], figpath: str, jobname: str = "bhns",
                            file_prefix: str | None = None, time_scale: float = 1.0,
                            time_axis: str = "Msun") -> None:
  """
  Compute and plot the area-weighted average Y_e (s_00) vs time across all segments.

  Parameters:
  output_dirs (list[str]): Segment directories.
  figpath (str): Output directory.
  jobname (str): AthenaK job name.
  file_prefix (str): Prefix of the primitive slice files.
  time_scale (float): Factor converting the time to the displayed units.
  time_axis (str): Displayed time unit.

  Returns:
  None.
  """
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
  def __init__(self, simpath: str, figpath: str | None = None, jobname: str = "bhns",
               prim_prefix: str = "mhd_w_bcc", rad_prefix: str = "rad_m1",
               batchtools: bool = True, units: str = 'cgs', time_units: str = "Msun",
               plane: str = "xy", companion: str = "xz", domain: tuple = (-60, 60),
               plot_contours: bool = False, dens_contours: tuple = _DEFAULT_DENS_CONT,
               show_grid: bool = True, show_trackers: bool = True,
               show_horizon: bool = False, skip_existing: bool = False,
               show_path: bool = False, path_length: float | None = None,
               path_fade: bool = True) -> None:
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
    plot_contours (bool): Whether to plot density contours or not.
    dens_contours (tuple): Density contour levels in g/cm^3 (only used with plot_contours).
    show_grid (bool): Overlay the AMR grid.
    show_trackers (bool): Overlay compact-object tracker markers on 2-D slices.
    show_horizon (bool): Overlay black-hole apparent-horizon circles on 2-D slices.
    skip_existing (bool): Skip frames whose output PNG already exists.
    show_path (bool): Also draw the trajectory ip to the current time.
    path_length(float): Length of the drawn trail in code time (None = full history).
    path_fade (bool): Fade the trail from transparent (oldest) to opaque (current).
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
    self.plot_contours = plot_contours
    self.dens_contours = dens_contours

    # Set the spatial/temporal units/scales
    self.coord_scale, self.coord_label             = spatial_units(units)
    self.time_scale, self.time_tex, self.time_axis = time_units_scale(time_units)

    self.output_dirs   = find_output_dirs(simpath, batchtools)
    self.skip_existing = skip_existing
    self._trackers     = None
    self._horizons     = None
    self.overrides     = {}
    self.section       = None
    self.show_path     = show_path
    self.path_length   = path_length
    self.path_fade     = path_fade

  # Frame output name.
  def frame_out(self, name: str) -> str:
    """
    Output PNG basename prefix for a frame series.

    Parameters:
    name (str): Series name.

    Returns:
    '<job>_frame_<name>_<plane>'.
    """
    suffix = f"_{self.plane}"
    return f"{self.jobname}_frame_{name}{suffix}"

  # Secton output directory.
  def frame_dir(self, name: str) -> str:
    """
    Per-series output subfolder.

    Parameters:
    name (str): Series name.

    Returns:
    '<name>_<plane>'.
    """
    return f"{name}_{self.plane}"

  # Shared tracker/horizon data.
  @property
  def trackers(self) -> list[_CombinedTracker | None]:
    """
    Combined compact-object trackers across all segments (loaded once).

    Returns:
    Trackers from load_trackers_all.
    """
    if self._trackers is None:
      self._trackers = load_trackers_all(self.output_dirs, self.jobname)
    return self._trackers

  @property
  def horizons(self) -> list[_CombinedHorizon | None]:
    """
    Combined apparent-horizon radii across all segments (loaded once).

    Returns:
    Horizons from load_horizons_all.
    """
    if self._horizons is None:
      self._horizons = load_horizons_all(self.output_dirs, self.jobname)
    return self._horizons

  def rad_file(self, var: str) -> str:
    """
    File prefix of an M1 radiation variable.

    Parameters:
    var (str): Radiation variable, e.g. 'E'.

    Returns:
    '<job>.<rad_prefix>_<var>_<plane>'.
    """
    return f"{self.jobname}.{self.rad_prefix}_{var}_{self.plane}"

  # Shared render function.
  def render_field(self, file_prefix: str | list[str], var_name: str, name: str,
                   cmap: str = "inferno", var_key: str | None = None,
                   print_minmax: bool = True, derived: tuple | None = None) -> None:
    """
    Render a scalar-field frame series.

    Parameters:
    file_prefix (str or list[str]): Prefix(es) of the slice files.
    var_name (str): Variable to plot, or 'derived:<name>' with derived.
    name (str): Series name for the output files.
    cmap (str): Default colormap.
    var_key (str): Key for the render parameters (default: var_name).
    print_minmax (bool): Print min/max of each frame.
    derived (tuple): (function, [variable names]) of a derived variable.

    Returns:
    None.
    """
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
                      time_axis=self.time_axis,
                      contour_levels=self.dens_contours if self.plot_contours else None,
                      dens_prefix=f"{self.jobname}.{self.prim_prefix}_{self.plane}",
                      show_path=self.show_path, path_length=self.path_length,
                      path_fade=self.path_fade)

  # Section renderer.
  def history(self) -> None:
    """
    Plot the baryonic mass and Hamiltonian constraint history.

    Returns:
    None.
    """
    print("[history] Plotting mass & H-norm2...")
    plot_history_all(self.output_dirs, self.figpath, self.jobname,
                     time_scale=self.time_scale, time_axis=self.time_axis)

  def rho_max(self) -> None:
    """
    Plot the max density and min lapse history.

    Returns:
    None.
    """
    print("[rho_max] Plotting rho_max & alpha_min...")
    plot_user_history_all(self.output_dirs, self.figpath, self.jobname,
                          time_scale=self.time_scale, time_axis=self.time_axis)

  def trackers_plot(self) -> None:
    """
    Plot the compact-object trajectories.

    Returns:
    None.
    """
    print("[trackers] Plotting object trajectories...")
    plot_tracker_trajectories_all(self.output_dirs, self.figpath, self.jobname,
                                  coord_scale=self.coord_scale,
                                  coord_label=self.coord_label)

  def run_speed(self) -> None:
    """
    Plot the run speed from the logs.

    Returns:
    None.
    """
    print("[run_speed] Plotting run speed from logs...")
    plot_run_speed_all(self.output_dirs, self.figpath, self.jobname,
                       time_scale=self.time_scale, time_axis=self.time_axis)

  def density(self) -> None:
    """
    Render the density frames.

    Returns:
    None.
    """
    print("[density] Rendering density frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "dens",
                       "dens")

  def temperature(self) -> None:
    """
    Render the temperature frames.

    Returns:
    None.
    """
    print("[temperature] Rendering temperature frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "temperature",
                       "temperature")

  def s00(self) -> None:
    """
    Render the s_00 (Ye proxy) frames.

    Returns:
    None.
    """
    print("[s00] Rendering s_00 (Ye proxy) frames...")
    self.render_field(f"{self.jobname}.{self.prim_prefix}_{self.plane}", "s_00",
                       "s00")

  def rad_E(self) -> None:
    """
    Render the radiation energy E:0 frames.

    Returns:
    None.
    """
    print("[rad_E] Rendering radiation energy E:0 frames...")
    self.render_field(self.rad_file("E"), "E:0", "rad_E0")

  def rad_N(self) -> None:
    """
    Render the radiation number density N:0 frames.

    Returns:
    None.
    """
    print("[rad_N] Rendering radiation number density N:0 frames...")
    self.render_field(self.rad_file("N"), "N:0", "rad_N0")

  def rad_N2(self) -> None:
    """
    Render the radiation number density N:2 (heavy lepton) frames.

    Returns:
    None.
    """
    print("[rad_N2] Rendering radiation number density N:2 (heavy lepton) frames...")
    self.render_field(self.rad_file("N"), "N:2", "rad_N2")

  def rad_abs(self) -> None:
    """
    Render the absorption opacity abs_1:0 frames.

    Returns:
    None.
    """
    print("[rad_abs] Rendering absorption opacity abs_1:0 frames...")
    self.render_field(self.rad_file("abs_1"), "abs_1:0", "abs1_0")

  def rad_abs0(self) -> None:
    """
    Render the absorption opacity abs_0:0 frames.

    Returns:
    None.
    """
    print("[rad_abs0] Rendering absorption opacity abs_0:0 frames...")
    self.render_field(self.rad_file("abs_0"), "abs_0:0", "abs0_0")

  def rad_eta0(self) -> None:
    """
    Render the emissivity eta_0:0 frames.

    Returns:
    None.
    """
    print("[rad_eta0] Rendering emissivity eta_0:0 frames...")
    self.render_field(self.rad_file("eta_0"), "eta_0:0", "eta0_0")

  def rad_eta1(self) -> None:
    """
    Render the emissivity eta_1:0 frames.

    Returns:
    None.
    """
    print("[rad_eta1] Rendering emissivity eta_1:0 frames...")
    self.render_field(self.rad_file("eta_1"), "eta_1:0", "eta1_0")

  def conH(self) -> None:
    """
    Render the Hamiltonian constraint |con_H| frames.

    Returns:
    None.
    """
    print("[z4c] Rendering con_H (Hamiltonian constraint) frames...")
    self.render_field(f"{self.jobname}.con_{self.plane}", "derived:abs_con_H",
                       "con_H", var_key="con_H", derived=(np.abs, ["con_H"]))

  def conM(self) -> None:
    """
    Render the momentum constraint |con_M| frames.

    Returns:
    None.
    """
    print("[z4c] Rendering con_M (momentum constraint) frames...")
    self.render_field(f"{self.jobname}.con_{self.plane}", "derived:abs_con_M",
                        "con_M", var_key="con_M", derived=(np.abs, ["con_M"]))

  def conC(self) -> None:
    """
    Render the constraint |con_C| frames.

    Returns:
    None.
    """
    print("[z4c] Rendering con_C (constraint) frames...")
    self.render_field(f"{self.jobname}.con_{self.plane}", "derived:abs_con_C",
                        "con_C", var_key="con_C", derived=(np.abs, ["con_C"]))

  def conZ(self) -> None:
    """
    Render the constraint |con_Z| frames.

    Returns:
    None.
    """
    print("[z4c] Rendering con_Z (constraint) frames...")
    self.render_field(f"{self.jobname}.con_{self.plane}", "derived:abs_con_Z",
                        "con_Z", var_key="con_Z", derived=(np.abs, ["con_Z"]))

  def alpha(self) -> None:
    """
    Render the lapse frames.

    Returns:
    None.
    """
    print("[z4c] Rendering z4c_alpha (lapse) frames...")
    self.render_field(f"{self.jobname}.z4c_{self.plane}", "z4c_alpha", "alpha",
                      var_key="alpha")

  def s00_avg(self) -> None:
    """
    Plot the average Y_e (s_00) time series.

    Returns:
    None.
    """
    print("[s00_avg] Computing average Y_e (s_00) time series...")
    plot_s00_timeseries_all(self.output_dirs, self.figpath, self.jobname,
                            file_prefix=f"{self.jobname}.{self.prim_prefix}_{self.plane}",
                            time_scale=self.time_scale, time_axis=self.time_axis)

  def avg_energy(self) -> None:
    """
    Render the average neutrino energy frames.

    Returns:
    None.
    """
    print("[avg_energy] Rendering average neutrino energy frames...")
    self.render_field([self.rad_file("N"), self.rad_file("E")], "derived:avg_energy",
                      "avg_energy", cmap="viridis", var_key="avg_energy",
                      derived=(_calc_average_energy, ["E:0", "N:0"]))

  def _nu_energy(self, section: str, species: int, label: str) -> None:
    """
    Render the average neutrino energy frames of one species.

    Parameters:
    section (str): Section name for the output files.
    species (int): M1 species index.
    label (str): Species label.

    Returns:
    None.
    """
    print(f"[{section}] Rendering neutrino energy E/{label} (species {species}) frames...")
    self.render_field([self.rad_file("N"), self.rad_file("E")], "derived:avg_energy",
                      section, cmap="viridis", var_key="nu_energy",
                      derived=(_calc_average_energy, [f"E:{species}", f"N:{species}"]))

  # Dispatch table mapping the section name to the render function.
  def _dispatch_table(self) -> dict[str, Callable[[], None]]:
    """
    Ordered mapping of section name to its render function.

    Returns:
    {section: zero-argument callable}.
    """
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
      "conM":        self.conM,
      "conC":        self.conC,
      "conZ":        self.conZ,
      "alpha":       self.alpha,
      "s00_avg":     self.s00_avg,
      "avg_energy":  self.avg_energy,
    }
    for section, species, label in _NU_ENERGY_SECTIONS:
      table[section] = (lambda sec=section, sp=species, lb=label:
                        self._nu_energy(sec, sp, lb))
    return table

  # Main function to run the renderer.
  def run(self, sections: tuple | list = ("all",)) -> None:
    """
    Run the requested sections.

    Parameters:
    sections (list): Section names or 'name=vmin,vmax,cmap' tokens (or parsed
                     tuples), or containing 'all' to run everything.

    Returns:
    None.
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
    print(f"Path     : {'off' if not self.show_path else (self.path_length or 'full')}")
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
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """
  Parse the command-line arguments.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  The parsed arguments.
  """
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
  parser.add_argument("--show-path", dest="show_path", action="store_true",
                      help="Draw the compact-object trajectories up to the current time")
  parser.add_argument("--path-length", dest="path_length", type=float, default=None,
                      help="Trail length in code time (default: full history)")
  parser.add_argument("--no-path-fade", dest="no_path_fade", action="store_true",
                      help="Draw the trajectory with constant opacity instead of fading")
  parser.add_argument("--show-grid", dest="show_grid", action="store_true",
                      help="Overlay AMR grid")
  parser.add_argument("--show-horizon", dest="show_horizon", action="store_true",
                      help="Overlay black-hole apparent-horizon circles on 2-D slice plots")
  parser.add_argument("--domain", dest="domain", default=(-60,60),
                      type=lambda s: tuple(float(x) for x in s.strip("()[] ").split(",")),
                      help="Tuple holding the size of the plotting domain in code units.")
  parser.add_argument("--plot-contours", dest="plot_contours", action="store_true",
                      help="Plot contours of either density or fraction of disk mass.")
  parser.add_argument("--dens-contours", dest="dens_contours", nargs="+", type=float, 
                      default=list(_DEFAULT_DENS_CONT), 
                      help="Density contour levels in g/cm^3 (used with --plot-contours).")
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
def main(argv: list[str] | None = None) -> None:
  """
  Command-line entry point of the system plotter.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
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
    plot_contours=args.plot_contours,
    dens_contours=args.dens_contours,
    domain=args.domain,
    show_grid=args.show_grid,
    show_trackers=not args.no_trackers,
    show_horizon=args.show_horizon,
    skip_existing=args.skip_existing,
    show_path=args.show_path,
    path_length=args.path_length,
    path_fade=not args.no_path_fade
  )
  sections = args.sections or load_sections(args.config) or ["all"]
  plotter.run(sections)

if __name__ == "__main__":
  main()
