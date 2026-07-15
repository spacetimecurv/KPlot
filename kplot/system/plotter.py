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
# Although the default diagnostics target binary-neutron-star / neutron-star
# radiation-M1 runs, the driver is system-agnostic: the AthenaK job name is
# configurable (default 'bhns'), and any section whose input files are absent
# is simply skipped, so the same class serves BNS, BHNS, BBH, etc.
#
# output-XXXX subdirectories are discovered automatically under a parent
# simulation directory.  Frames that appear in multiple segments (at restart
# boundaries) are deduplicated by simulation time; the later segment wins.
#
# Command-line entry point (used by the plot_system.sh driver):
#     python3 -m kplot.system --simpath PATH [--figpath PATH] [--jobname NAME]
#                             [--plane {xy,xz,yz}] [--time-units {Msun,ms}]
#                             [--no-trackers] [--show-horizon]
#                             [--full-domain] [--skip-existing]
#                             [--units {code,cgs,ngs}]
#                             [--sections SECTION [SECTION ...]]
#
# Each frame series is written to its own figpath/<diagnostic>_<plane>/ subfolder.
#
# Programmatic use:
#     from kplot import SystemPlotter
#     SystemPlotter(simpath="/path/to/sim", units="cgs").run(["density", "history"])

# Import necessary standard libraries.
import argparse
import glob
import multiprocessing
import os
import re
import sys

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

# Import BNS-specific unit conversions / render parameters.
from . import athenak_units as au

# ---------------------------------------------------------------------------
# Global plot style (applied lazily so importing kplot does not clobber the
# matplotlib state used by other classes such as History/HorizonFinder).
# ---------------------------------------------------------------------------
_STYLE = {
  'lines.markersize': 6,
  'font.family': 'serif',
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

# Spatial coordinate grid is in geometrized M_sun units (G=c=M_sun=1),
# 1 code length = G*M_sun/c^2 = 1.476955 km.  Field values are rescaled by
# the --units choice, but the coordinate axes are only converted to km for
# non-code units.
MSUN_TO_KM = au._unit_len_cgs / 1.0e5

# Time is stored in geometrized M_sun units; 1 code time = G*M_sun/c^3 seconds.
MSUN_TO_MS = (au._unit_len_cgs / au.c) * 1.0e3   # -> milliseconds


def configure_matplotlib():
  """Select the non-interactive Agg backend and apply the plot style.

  Used as the pool initializer for rendering workers and called once by the
  parent process before drawing the 1-D diagnostic plots.
  """
  matplotlib.use("Agg", force=True)
  mpl.rcParams.update(_STYLE)


def spatial_units(units):
  """Return (coord_scale, axis_label) for the spatial axes given a unit system.

  'code' keeps native M_sun coordinates; 'cgs'/'ngs' display km.
  """
  if units == "code":
    return 1.0, r"\mathrm{M}_\odot"
  return MSUN_TO_KM, r"\mathrm{km}"


# Slice planes selectable for 2-D frame rendering, and their axis letters.
PLANES = ("xy", "xz", "yz")
_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}


def plane_axis_labels(plane):
  """Return (horizontal, vertical) coordinate letters for a slice plane."""
  return _PLANE_AXES[plane]


# Time-display units selectable for frame titles and 1-D time-series axes.
TIME_UNITS = ("Msun", "ms")


def time_units_scale(time_units):
  """Return (scale, latex_label, axis_label) for the chosen time-display units.

  scale multiplies the stored (M_sun) simulation time; latex_label is for plot
  titles (math mode) and axis_label is a plain string for axis captions.
  """
  if time_units == "ms":
    return MSUN_TO_MS, r"\mathrm{ms}", "ms"
  return 1.0, r"\mathrm{M}_\odot", "Msun"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def glob_bin_files(directory, prefix):
  """Return sorted list of existing binary files matching prefix.NNNNN.bin."""
  pattern = os.path.join(directory, "bin", f"{prefix}.?????.bin")
  return sorted(glob.glob(pattern))


def _detect_prefix_all(output_dirs, new_prefix, old_prefix):
  """Return new_prefix if files exist in any segment, else old_prefix."""
  for d in output_dirs:
    if glob_bin_files(d, new_prefix):
      return new_prefix
  return old_prefix


def print_stats(data, var_name, time_scale=1.0, time_axis="Msun"):
  """Print the frame time plus min/max of var_name across all blocks."""
  max_val, min_val = -np.inf, np.inf
  for _, arr in data.get_block_data(var_name):
    max_val = max(max_val, arr.max())
    min_val = min(min_val, arr.min())
  print(f"    t={data.time * time_scale:.3f} {time_axis}  |  "
        f"max={max_val:.3e}  min={min_val:.3e}")


def _tracker_plane_coords(tracker, plane):
  """Return the (horizontal, vertical) tracker coordinate arrays for a plane."""
  if plane == "xy":
    return tracker.x, tracker.y
  if plane == "xz":
    return tracker.x, tracker.z
  return tracker.y, tracker.z   # yz


def draw_trackers(ax, trackers, time, plane="xy",
                  colors=("cyan", "orange"), marker="x", markersize=80,
                  coord_scale=1.0):
  """Overlay compact-object positions on ax at the given simulation time.

  The tracker coordinates are projected onto *plane* (xy/xz/yz).  coord_scale
  multiplies the stored (M_sun) coordinates so they match the displayed spatial
  units (e.g. km when coord_scale = MSUN_TO_KM).
  """
  labels = ["obj 1", "obj 2"]
  for tracker, color, label in zip(trackers, colors, labels):
    if tracker is None:
      continue
    idx = np.argmin(np.abs(tracker.time - time))
    h, v = _tracker_plane_coords(tracker, plane)
    ax.scatter(h[idx] * coord_scale, v[idx] * coord_scale,
               marker=marker, c=color, s=markersize,
               linewidths=1.5, zorder=10, label=label)


def draw_horizons(ax, horizons, trackers, time, plane="xy", coord_scale=1.0,
                  edgecolor="cyan", facecolor="black"):
  """Overlay filled apparent-horizon circles at the paired tracker positions.

  For each black hole k, the circle is centered on tracker k's position
  (projected onto *plane*) at the nearest sample to *time* and sized by horizon
  k's radius (last column of the horizon-summary file), mirroring SeriesPlot's
  black-hole overlay.  Requires both a horizon summary (radius) and a tracker
  (center) for that object.
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


def rescale_image_extents(ax, coord_scale):
  """Multiply the extent of every image already drawn on ax by coord_scale.

  Used to convert the native M_sun grid coordinates of plot_slice() output to
  the displayed spatial units (km).  No-op when coord_scale == 1.0.
  """
  if coord_scale == 1.0:
    return
  for im in ax.get_images():
    x0, x1, y0, y1 = im.get_extent()
    im.set_extent([x0 * coord_scale, x1 * coord_scale,
                   y0 * coord_scale, y1 * coord_scale])


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


class _CombinedTracker:
  """Lightweight tracker holding concatenated time/x/y/z arrays from all segments."""

  def __init__(self, time, x, y, z):
    self.time = time
    self.x = x
    self.y = y
    self.z = z


class _CombinedHorizon:
  """Lightweight horizon holding concatenated time/radius arrays from all segments."""

  def __init__(self, time, radius):
    self.time = time
    self.radius = radius


# ---------------------------------------------------------------------------
# Multi-segment data collection helpers
# ---------------------------------------------------------------------------

def _read_frame_time(fpath):
  """Read simulation time from a binary data file. Returns (fpath, time_or_None)."""
  try:
    return (fpath, BinaryData(fpath).time)
  except Exception:
    return (fpath, None)


def find_output_dirs(simpath):
  """Return sorted list of output-XXXX subdirectories under simpath."""
  pattern = os.path.join(simpath, "output-[0-9][0-9][0-9][0-9]")
  return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def _collect_frames_by_time(output_dirs, file_prefix):
  """Gather all files for file_prefix across output_dirs.

  Returns dict: sim_time -> xy_fpath.  Later segments overwrite earlier ones
  for duplicate times.
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


def collect_combined_frames(output_dirs, file_prefix, xz_prefix=None, step=1):
  """Collect all frames for file_prefix across all segments, deduplicated by
  simulation time (later segment wins).

  Returns list of (global_idx, xy_fpath, xz_fpath_or_None) sorted by sim time,
  with step applied after deduplication.
  """
  time_to_xy = _collect_frames_by_time(output_dirs, file_prefix)
  if not time_to_xy:
    return []

  # Build set of all existing XZ files for quick lookup
  xz_file_set = set()
  if xz_prefix is not None:
    for d in output_dirs:
      xz_file_set.update(glob_bin_files(d, xz_prefix))

  sorted_times = sorted(time_to_xy.keys())[::step]

  result = []
  for global_idx, t in enumerate(sorted_times):
    xy_fpath = time_to_xy[t]
    xz_fpath = None
    if xz_prefix is not None:
      m = re.search(r'\.(\d{5})\.bin$', os.path.basename(xy_fpath))
      if m:
        xz_candidate = os.path.join(
          os.path.dirname(xy_fpath), f"{xz_prefix}.{m.group(1)}.bin"
        )
        if xz_candidate in xz_file_set:
          xz_fpath = xz_candidate
    result.append((global_idx, xy_fpath, xz_fpath))
  return result


def combine_hst(output_dirs, filename):
  """Load and concatenate history (.hst) files across all segments.

  Rows are sorted by simulation time (col 0); duplicate times keep the later
  segment's row.  Returns numpy array or None if no files found.
  """
  arrays = []
  for d in output_dirs:
    fpath = os.path.join(d, filename)
    if os.path.exists(fpath):
      try:
        a = np.loadtxt(fpath, skiprows=2)
        if a.ndim == 1:
          a = a.reshape(1, -1)
        if a.size > 0:
          arrays.append(a)
      except Exception as e:
        print(f"  Warning: could not load {fpath}: {e}")

  if not arrays:
    return None

  combined = np.vstack(arrays)
  sort_idx = np.argsort(combined[:, 0], kind='stable')
  combined = combined[sort_idx]

  # Keep last occurrence for each time (later segment wins)
  times = combined[:, 0]
  _, first_in_rev = np.unique(times[::-1], return_index=True)
  last_in_orig = np.sort(len(times) - 1 - first_in_rev)
  return combined[last_in_orig]


def _dedup_by_time(all_times, *arrays):
  """Sort by time and keep the last occurrence of each time (later segment wins).

  Returns (sorted_unique_times, *deduped_arrays) aligned to the kept indices.
  """
  sort_idx = np.argsort(all_times, kind='stable')
  all_times = all_times[sort_idx]
  arrays = [a[sort_idx] for a in arrays]

  _, first_in_rev = np.unique(all_times[::-1], return_index=True)
  keep = np.sort(len(all_times) - 1 - first_in_rev)
  return (all_times[keep], *(a[keep] for a in arrays))


def load_trackers_all(output_dirs, jobname="bhns"):
  """Load and concatenate tracker files across all segments.

  Returns list of two _CombinedTracker objects (entries may be None).
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


def load_horizons_all(output_dirs, jobname="bhns"):
  """Load and concatenate apparent-horizon summary files across all segments.

  Reads <jobname>.horizon_summary_k.txt (time = column 1, radius = last
  column, matching SeriesPlot).  Returns list of two _CombinedHorizon objects
  (entries may be None when no horizon file exists for that object).
  """
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


# ---------------------------------------------------------------------------
# Diagnostic history / time-series plots
# ---------------------------------------------------------------------------

def plot_history_all(output_dirs, figpath, jobname="bhns",
                     time_scale=1.0, time_axis="Msun"):
  """Baryonic mass and Hamiltonian constraint norm across all segments."""
  mhd = combine_hst(output_dirs, f"{jobname}.mhd.hst")
  if mhd is not None:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(mhd[:, 0] * time_scale, mhd[:, 2])
    ax.set_xlabel(f"t ({time_axis})")
    ax.set_ylabel("Baryonic mass")
    fig.savefig(os.path.join(figpath, f"{jobname}_history_mass.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {jobname}_history_mass.png")
  else:
    print(f"  Skipping mass history: no {jobname}.mhd.hst found")

  z4c = combine_hst(output_dirs, f"{jobname}.z4c.user.hst")
  if z4c is not None:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(z4c[:, 0] * time_scale, z4c[:, 3])
    ax.set_xlabel(f"t ({time_axis})")
    ax.set_ylabel("H-norm2")
    ax.set_yscale("log")
    fig.savefig(os.path.join(figpath, f"{jobname}_history_hnorm2.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {jobname}_history_hnorm2.png")
  else:
    print(f"  Skipping H-norm2: no {jobname}.z4c.user.hst found")


def plot_user_history_all(output_dirs, figpath, jobname="bhns",
                          time_scale=1.0, time_axis="Msun"):
  """Max density and min lapse from <jobname>.user.hst across all segments."""
  d = combine_hst(output_dirs, f"{jobname}.user.hst")
  if d is None:
    print(f"  Skipping user history: no {jobname}.user.hst found")
    return

  time_u   = d[:, 0] * time_scale
  rho_max  = d[:, 2]
  alph_min = d[:, 3]

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  ax.plot(time_u, rho_max)
  ax.set_xlabel(f"t ({time_axis})")
  ax.set_ylabel(r"$\rho_\mathrm{max}$")
  ax.set_yscale("log")
  fig.savefig(os.path.join(figpath, f"{jobname}_user_rho_max.png"), dpi=150)
  plt.close(fig)
  print(f"  Saved {jobname}_user_rho_max.png")

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  ax.plot(time_u, alph_min)
  ax.set_xlabel(f"t ({time_axis})")
  ax.set_ylabel(r"$\alpha_\mathrm{min}$")
  fig.savefig(os.path.join(figpath, f"{jobname}_user_alpha_min.png"), dpi=150)
  plt.close(fig)
  print(f"  Saved {jobname}_user_alpha_min.png")


def plot_tracker_trajectories_all(output_dirs, figpath, jobname="bhns",
                                  coord_scale=1.0, coord_label=r"\mathrm{M}_\odot"):
  """Plot compact-object trajectories (raw + smoothed) from tracker data."""
  trackers = load_trackers_all(output_dirs, jobname)
  if all(t is None for t in trackers):
    print("  No tracker files found, skipping trajectory plot")
    return

  colors  = ["tab:red",  "tab:blue"]
  colors2 = ["salmon",   "skyblue"]
  labels  = ["obj 1", "obj 2"]

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  for tracker, color, color2, label in zip(trackers, colors, colors2, labels):
    if tracker is None:
      continue
    ax.plot(tracker.x * coord_scale, tracker.y * coord_scale,
            ".", color=color2, ms=2, label=f"{label} raw")
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
# Per-frame slice renderers (module-level workers for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _render_frame_worker_all(kw):
  """Per-frame plot worker for combined multi-segment data."""
  global_idx      = kw['global_idx']
  xy_fpath        = kw['xy_fpath']
  xz_fpath        = kw['xz_fpath']   # None if not available
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
  use_abs         = kw.get('use_abs', False)
  plane           = kw.get('plane', 'xy')
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
    data_xy = BinaryData(xy_fpath)
  except FileNotFoundError:
    print(f"  Missing {xy_fpath}, skipping")
    return

  data_xz = None
  if xz_fpath is not None:
    try:
      data_xz = BinaryData(xz_fpath)
    except FileNotFoundError:
      pass

  if use_abs:
    abs_var = f"derived:abs_{var_name}"
    data_xy.register_derived_variable(abs_var, np.abs, var_name)
    if data_xz is not None:
      data_xz.register_derived_variable(abs_var, np.abs, var_name)
    var_name = abs_var

  if print_minmax:
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
      draw_refinement_grid(ax_xz, data_xz, coord_scale=coord_scale)
      draw_refinement_grid(ax_xy, data_xy, coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax_xy, trackers, data_xy.time, plane="xy", coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax_xy, horizons, trackers, data_xy.time, plane="xy", coord_scale=coord_scale)

    if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
    ax_xz.set_ylim(*zlim)
    if ylim is not None: ax_xy.set_ylim(*ylim)
    plt.setp(ax_xz.get_xticklabels(), visible=False)
    ax_xz.set_xlabel(""); ax_xz.set_ylabel(rf"$z~[{coord_label}]$")
    ax_xy.set_xlabel(rf"$x~[{coord_label}]$")
    ax_xy.set_ylabel(rf"$y~[{coord_label}]$")
    ax_xz.set_title(rf"$t = {data_xy.time * time_scale:.2f}~{time_tex}$")
    if ylim is not None and coord_scale == 1.0:
      ax_xz.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 20, 40, 60]))
      ax_xy.yaxis.set_major_locator(mpl.ticker.FixedLocator([-60, -40, -20, 0, 20, 40]))
    else:
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
      draw_refinement_grid(ax, data_xy, coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax, trackers, data_xy.time, plane=plane, coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax, horizons, trackers, data_xy.time, plane=plane, coord_scale=coord_scale)

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel(rf"${hlab}~[{coord_label}]$")
    ax.set_ylabel(rf"${vlab}~[{coord_label}]$")
    ax.set_title(rf"$t = {data_xy.time * time_scale:.2f}~{time_tex}$")
    fig.tight_layout()

  fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
              bbox_inches='tight')
  plt.close(fig)


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
                      use_abs=False,
                      plane="xy", subdir=None,
                      coord_scale=1.0,
                      coord_label=r"\mathrm{M}_\odot",
                      time_scale=1.0, time_tex=r"\mathrm{M}_\odot",
                      time_axis="Msun"):
  """Render one PNG per deduplicated frame across all segments, in parallel.

  For the default xy plane an XZ companion panel is auto-detected via the
  '_xy' -> '_xz' substitution in file_prefix; xz/yz planes render a single panel.
  PNGs are written to figpath/subdir (created on demand) when subdir is given.
  """
  xz_prefix = file_prefix.replace("_xy", "_xz") if "_xy" in file_prefix else None
  frames = collect_combined_frames(output_dirs, file_prefix, xz_prefix, step)

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

  zlim = (0, ylim[1]) if ylim is not None else (0, None)
  plot_kwargs = dict(cmap=cmap, interpolation="nearest", rescale=scale_factor)
  if norm is not None: plot_kwargs["norm"] = norm
  if vmin is not None: plot_kwargs["vmin"] = vmin
  if vmax is not None: plot_kwargs["vmax"] = vmax

  worker_args = [
    dict(global_idx=global_idx, xy_fpath=xy_fpath, xz_fpath=xz_fpath,
         var_name=var_name, plot_kwargs=plot_kwargs,
         xlim=xlim, ylim=ylim, zlim=zlim,
         label=label, out_prefix=out_prefix, figpath=outdir,
         print_minmax=print_minmax,
         show_refinement=show_refinement,
         show_trackers=show_trackers, trackers=trackers,
         show_horizon=show_horizon, horizons=horizons,
         use_abs=use_abs, plane=plane,
         coord_scale=coord_scale, coord_label=coord_label,
         time_scale=time_scale, time_tex=time_tex, time_axis=time_axis)
    for global_idx, xy_fpath, xz_fpath in frames
  ]

  n_workers = min(multiprocessing.cpu_count(), len(worker_args))
  with multiprocessing.Pool(n_workers, initializer=configure_matplotlib) as pool:
    pool.map(_render_frame_worker_all, worker_args)

  print(f"  Rendered {len(frames)} frames for '{var_name}'")


# ---------------------------------------------------------------------------
# Derived average neutrino energy (E:s / N:s) renderer
# ---------------------------------------------------------------------------

def _calc_average_energy(rE, rN):
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(rN > 0, rE / rN, 0.0)


def _calc_average_energy_scaled(scale):
  """Return a version of _calc_average_energy that applies a unit scale."""
  def _fn(rE, rN):
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.where(rN > 0, rE / rN * scale, 0.0)
  return _fn


def _render_avg_energy_worker_all(kw):
  """Per-frame avg_energy worker for combined multi-segment data."""
  global_idx      = kw['global_idx']
  xy_n_fpath      = kw['xy_n_fpath']
  xy_e_fpath      = kw['xy_e_fpath']
  xz_n_fpath      = kw['xz_n_fpath']   # None if not available
  xz_e_fpath      = kw['xz_e_fpath']
  xlim            = kw['xlim']
  ylim            = kw['ylim']
  zlim            = kw['zlim']
  figpath         = kw['figpath']
  print_minmax    = kw['print_minmax']
  show_refinement = kw['show_refinement']
  show_trackers   = kw['show_trackers']
  trackers        = kw['trackers']
  show_horizon    = kw.get('show_horizon', False)
  horizons        = kw.get('horizons', None)

  avg_energy_scale = kw.get('avg_energy_scale', 1.0)
  avg_energy_label = kw.get('avg_energy_label', "Average energy (code units)")
  avg_energy_norm  = kw.get('avg_energy_norm', LogNorm(vmin=1e-4, vmax=1e-3))
  species          = kw.get('species', 0)
  out_prefix       = kw.get('out_prefix', 'frame_avg_energy')
  plane            = kw.get('plane', 'xy')
  coord_scale      = kw.get('coord_scale', 1.0)
  coord_label      = kw.get('coord_label', r"\mathrm{M}_\odot")
  time_scale       = kw.get('time_scale', 1.0)
  time_tex         = kw.get('time_tex', r"\mathrm{M}_\odot")
  time_axis        = kw.get('time_axis', "Msun")
  hlab, vlab       = plane_axis_labels(plane)

  if xlim is not None: xlim = tuple(v * coord_scale for v in xlim)
  if ylim is not None: ylim = tuple(v * coord_scale for v in ylim)
  if zlim is not None: zlim = tuple(v * coord_scale for v in zlim)

  try:
    data_xy = GroupData([xy_n_fpath, xy_e_fpath])
  except FileNotFoundError as err:
    print(f"  Missing file: {err}, skipping frame {global_idx}")
    return

  e_var = f"E:{species}"
  n_var = f"N:{species}"
  _avg_fn = _calc_average_energy_scaled(avg_energy_scale)
  data_xy.register_derived_variable("derived:avg_energy", _avg_fn, e_var, n_var)
  if print_minmax and hasattr(data_xy, 'get_block_data'):
    print_stats(data_xy, "derived:avg_energy",
                time_scale=time_scale, time_axis=time_axis)

  plot_kwargs = dict(norm=avg_energy_norm, interpolation="nearest")

  data_xz = None
  if xz_n_fpath is not None and xz_e_fpath is not None:
    try:
      data_xz = GroupData([xz_n_fpath, xz_e_fpath])
      data_xz.register_derived_variable("derived:avg_energy",
                                        _avg_fn, e_var, n_var)
    except FileNotFoundError:
      data_xz = None

  if data_xz is not None:
    fig = plt.figure(figsize=(4.5, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.03, right=0.83)
    ax_xz = fig.add_subplot(gs[0])
    ax_xy = fig.add_subplot(gs[1], sharex=ax_xz)

    pcm = data_xz.plot_slice("derived:avg_energy", ax=ax_xz, **plot_kwargs)
    data_xy.plot_slice("derived:avg_energy", ax=ax_xy, **plot_kwargs)
    rescale_image_extents(ax_xz, coord_scale)
    rescale_image_extents(ax_xy, coord_scale)

    if show_refinement:
      draw_refinement_grid(ax_xz, data_xz.datasets[0], coord_scale=coord_scale)
      draw_refinement_grid(ax_xy, data_xy.datasets[0], coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax_xy, trackers, data_xy.datasets[0].time, plane="xy", coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax_xy, horizons, trackers, data_xy.datasets[0].time, plane="xy", coord_scale=coord_scale)

    if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
    ax_xz.set_ylim(*zlim)
    if ylim is not None: ax_xy.set_ylim(*ylim)
    plt.setp(ax_xz.get_xticklabels(), visible=False)
    ax_xz.set_xlabel(""); ax_xz.set_ylabel(rf"$z~[{coord_label}]$")
    ax_xy.set_xlabel(rf"$x~[{coord_label}]$")
    ax_xy.set_ylabel(rf"$y~[{coord_label}]$")
    ax_xz.set_title(rf"$t = {data_xy.datasets[0].time * time_scale:.2f}~{time_tex}$")
    if ylim is not None and coord_scale == 1.0:
      ax_xz.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 20, 40, 60]))
      ax_xy.yaxis.set_major_locator(mpl.ticker.FixedLocator([-60, -40, -20, 0, 20, 40]))
    else:
      ax_xy.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='upper'))
    fig.align_ylabels([ax_xz, ax_xy])
    p_top = ax_xz.get_position(); p_bot = ax_xy.get_position()
    cax = fig.add_axes([p_top.x1 + 0.015, p_bot.y0, 0.04, p_top.y1 - p_bot.y0])
    cb = fig.colorbar(pcm, cax=cax)
    cb.ax.yaxis.set_label_position('right')
    cb.set_label(avg_energy_label)
  else:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pcm = data_xy.plot_slice("derived:avg_energy", ax=ax, **plot_kwargs)
    rescale_image_extents(ax, coord_scale)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(pcm, cax=cax, label=avg_energy_label)

    if show_refinement:
      draw_refinement_grid(ax, data_xy.datasets[0], coord_scale=coord_scale)
    if show_trackers and trackers is not None:
      draw_trackers(ax, trackers, data_xy.datasets[0].time, plane=plane, coord_scale=coord_scale)
    if show_horizon:
      draw_horizons(ax, horizons, trackers, data_xy.datasets[0].time, plane=plane, coord_scale=coord_scale)

    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_xlabel(rf"${hlab}~[{coord_label}]$")
    ax.set_ylabel(rf"${vlab}~[{coord_label}]$")
    ax.set_title(rf"$t = {data_xy.datasets[0].time * time_scale:.2f}~{time_tex}$")
    fig.tight_layout()

  fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
              bbox_inches='tight')
  plt.close(fig)


def render_avg_energy_all(output_dirs, figpath, print_minmax=False,
                          show_trackers=True, show_refinement=True,
                          trackers=None, show_horizon=False, horizons=None,
                          xlim=(-60, 60), ylim=(-60, 60),
                          units='code', species=0, jobname="bhns",
                          out_prefix='frame_avg_energy',
                          var_key='avg_energy',
                          plane="xy", subdir=None,
                          coord_scale=1.0,
                          coord_label=r"\mathrm{M}_\odot",
                          time_scale=1.0, time_tex=r"\mathrm{M}_\odot",
                          time_axis="Msun"):
  """Derived average neutrino energy = E:s / N:s, rendered in parallel across
  all segments.  Frames are deduplicated by simulation time.
  PNGs are written to figpath/subdir (created on demand) when subdir is given.
  """
  _sc, _lb, _log, _lo, _hi = au.plot_params(var_key, units)
  _norm = LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi)
  n_prefix = _detect_prefix_all(output_dirs, f"{jobname}.rad_m1_N_{plane}", f"{jobname}.rad_m1_N")
  e_prefix = _detect_prefix_all(output_dirs, f"{jobname}.rad_m1_E_{plane}", f"{jobname}.rad_m1_E")

  t_to_n = _collect_frames_by_time(output_dirs, n_prefix)
  t_to_e = _collect_frames_by_time(output_dirs, e_prefix)

  if not t_to_n or not t_to_e:
    print("  No N/E files found for avg_energy")
    return

  common_times = sorted(set(t_to_n.keys()) & set(t_to_e.keys()))
  if not common_times:
    print("  No matching N/E frame pairs found for avg_energy")
    return

  outdir = os.path.join(figpath, subdir) if subdir else figpath
  os.makedirs(outdir, exist_ok=True)

  zlim = (0, ylim[1]) if ylim is not None else (0, None)
  worker_args = [
    dict(global_idx=idx,
         xy_n_fpath=t_to_n[t], xy_e_fpath=t_to_e[t],
         xz_n_fpath=None, xz_e_fpath=None,
         xlim=xlim, ylim=ylim, zlim=zlim, figpath=outdir,
         print_minmax=print_minmax,
         show_refinement=show_refinement,
         show_trackers=show_trackers, trackers=trackers,
         show_horizon=show_horizon, horizons=horizons,
         avg_energy_scale=_sc,
         avg_energy_label=_lb,
         avg_energy_norm=_norm,
         species=species,
         out_prefix=out_prefix,
         plane=plane,
         coord_scale=coord_scale, coord_label=coord_label,
         time_scale=time_scale, time_tex=time_tex, time_axis=time_axis)
    for idx, t in enumerate(common_times)
  ]

  n_workers = min(multiprocessing.cpu_count(), len(worker_args))
  with multiprocessing.Pool(n_workers, initializer=configure_matplotlib) as pool:
    pool.map(_render_avg_energy_worker_all, worker_args)

  print(f"  Rendered {len(common_times)} {out_prefix} frames")


# ---------------------------------------------------------------------------
# Area-weighted average Y_e (s_00) time series
# ---------------------------------------------------------------------------

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


def plot_s00_timeseries_all(output_dirs, figpath, jobname="bhns", file_prefix=None,
                            time_scale=1.0, time_axis="Msun"):
  """Compute and plot the area-weighted average Y_e (s_00) vs time across all segments.

  Collects all <jobname>.prim_xy frames across output-XXXX subdirs (deduplicated
  by simulation time), computes the 2-D slice area-weighted mean of s_00 for each
  frame in parallel, then saves a 1-D time-series plot.
  """
  if file_prefix is None:
    file_prefix = f"{jobname}.prim_xy"
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

  fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
  ax.plot(times * time_scale, avg_ye)
  ax.set_xlabel(f"t ({time_axis})")
  ax.set_ylabel(r"$\langle Y_e \rangle$")
  fig.savefig(os.path.join(figpath, f"{jobname}_avg_s00_timeseries.png"), dpi=150)
  plt.close(fig)
  print(f"  Saved {jobname}_avg_s00_timeseries.png ({len(times)} frames)")


# ---------------------------------------------------------------------------
# Orchestrator
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
  "rad_eta0", "rad_eta1", "z4c",
  "s00_avg", "avg_energy",
  "nu_energy", "nu_energy1", "nu_energy2",
]


class SystemPlotter:
  """Full-run visualization driver for a multi-segment AthenaK simulation.

  Discovers the output-XXXX restart segments under *simpath* and renders any
  requested subset of diagnostics into *figpath*.  Input and output file names
  are keyed on the AthenaK job name (default 'bhns'), so the same driver serves
  different physical systems (BNS, BHNS, BBH, ...); sections whose input files
  are absent are simply skipped.

  Field values are converted to the chosen unit system; spatial axes are shown
  in M_sun for 'code' units and km otherwise.  2-D slices can be overlaid with
  compact-object trackers, black-hole apparent-horizon circles, and the AMR grid.
  """

  def __init__(self, simpath, figpath=None, units='cgs', jobname="bhns",
               plane="xy", time_units="Msun",
               show_trackers=True, show_horizon=False,
               full_domain=False, skip_existing=False):
    """Initialize the SystemPlotter.

    Parameters:
    simpath (str): Parent directory containing the output-XXXX subdirectories.
    figpath (str): Output figure directory (default: <simpath>/Figs).
    units (str): Unit system for physical quantities ('code', 'cgs', 'ngs').
    jobname (str): AthenaK job name prefixing all input/output files.
    plane (str): Slice plane to render, one of 'xy', 'xz', 'yz'.  'xy' also
                 auto-detects an 'xz' companion panel when those files exist.
    time_units (str): Time display units for titles/axes, 'Msun' or 'ms'.
    show_trackers (bool): Overlay compact-object tracker markers on 2-D slices.
    show_horizon (bool): Overlay black-hole apparent-horizon circles on 2-D slices.
    full_domain (bool): Show the full domain instead of the +/-60 Msun window.
    skip_existing (bool): Skip frames whose output PNG already exists.
    """
    if plane not in PLANES:
      raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
    if time_units not in TIME_UNITS:
      raise ValueError(f"time_units must be one of {TIME_UNITS}, got {time_units!r}")
    self.simpath = simpath
    self.figpath = figpath or os.path.join(simpath, "Figs")
    self.units = units
    self.jobname = jobname
    self.plane = plane
    self.time_units = time_units
    self.show_trackers = show_trackers
    self.show_horizon = show_horizon
    self.full_domain = full_domain
    self.skip_existing = skip_existing
    self.coord_scale, self.coord_label = spatial_units(units)
    self.time_scale, self.time_tex, self.time_axis = time_units_scale(time_units)
    self.xy_lim = None if full_domain else (-60, 60)

    self.output_dirs = find_output_dirs(simpath)
    self._trackers = None
    self._horizons = None
    self._rad_prefixes = None

  def _frame_out(self, name):
    """Output PNG basename prefix for a frame series (plane-suffixed except xy)."""
    suffix = "" if self.plane == "xy" else f"_{self.plane}"
    return f"{self.jobname}_frame_{name}{suffix}"

  def _frame_dir(self, name):
    """Per-series output subfolder, named '<diagnostic>_<plane>'."""
    return f"{name}_{self.plane}"

  # --- lazily-loaded shared data -----------------------------------------

  @property
  def trackers(self):
    """Combined compact-object trackers across all segments (loaded once)."""
    if self._trackers is None:
      self._trackers = load_trackers_all(self.output_dirs, self.jobname)
    return self._trackers

  @property
  def horizons(self):
    """Combined apparent-horizon radii across all segments (loaded once)."""
    if self._horizons is None:
      self._horizons = load_horizons_all(self.output_dirs, self.jobname)
    return self._horizons

  def _detect_rad_prefixes(self):
    """Auto-detect the radiation file prefix scheme (new '_xy' vs old)."""
    if self._rad_prefixes is None:
      d, j, p = self.output_dirs, self.jobname, self.plane
      self._rad_prefixes = {
        'E':     _detect_prefix_all(d, f"{j}.rad_m1_E_{p}",     f"{j}.rad_m1_E"),
        'N':     _detect_prefix_all(d, f"{j}.rad_m1_N_{p}",     f"{j}.rad_m1_N"),
        'abs_0': _detect_prefix_all(d, f"{j}.rad_m1_abs_0_{p}", f"{j}.rad_m1_abs_0"),
        'abs_1': _detect_prefix_all(d, f"{j}.rad_m1_abs_1_{p}", f"{j}.rad_m1_abs_1"),
        'eta_0': _detect_prefix_all(d, f"{j}.rad_m1_eta_0_{p}", f"{j}.rad_m1_eta_0"),
        'eta_1': _detect_prefix_all(d, f"{j}.rad_m1_eta_1_{p}", f"{j}.rad_m1_eta_1"),
      }
    return self._rad_prefixes

  # --- shared render helper ----------------------------------------------

  def _render_field(self, file_prefix, var_name, name, cmap="inferno",
                    var_key=None, norm_from_params=True,
                    vmin=None, vmax=None, print_minmax=True, use_abs=False):
    """Render a scalar-field frame series, filling common options from self.

    *name* is the short diagnostic name; the PNGs go into the '<name>_<plane>'
    subfolder with the '<job>_frame_<name>[...]' basename.
    """
    var_key = var_key or var_name
    scale, label, use_log, lo, hi = au.plot_params(var_key, self.units)
    norm = None
    if norm_from_params:
      norm = LogNorm(vmin=lo, vmax=hi) if use_log else Normalize(vmin=lo, vmax=hi)
    render_frames_all(self.output_dirs, self.figpath,
                      file_prefix=file_prefix, var_name=var_name,
                      cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, label=label,
                      out_prefix=self._frame_out(name), subdir=self._frame_dir(name),
                      xlim=self.xy_lim, ylim=self.xy_lim,
                      print_minmax=print_minmax,
                      show_trackers=self.show_trackers, show_refinement=True,
                      trackers=self.trackers,
                      show_horizon=self.show_horizon,
                      horizons=self.horizons if self.show_horizon else None,
                      scale_factor=scale,
                      skip_existing=self.skip_existing, use_abs=use_abs,
                      plane=self.plane,
                      coord_scale=self.coord_scale, coord_label=self.coord_label,
                      time_scale=self.time_scale, time_tex=self.time_tex,
                      time_axis=self.time_axis)

  # --- individual sections ------------------------------------------------

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
    self._render_field(f"{self.jobname}.mhd_w_bcc_{self.plane}", "dens",
                       "dens", cmap="inferno")

  def temperature(self):
    print("[temperature] Rendering temperature frames...")
    self._render_field(f"{self.jobname}.mhd_w_bcc_{self.plane}", "temperature",
                       "temperature", cmap="viridis", norm_from_params=False,
                       vmin=0.0, vmax=70.0)

  def s00(self):
    print("[s00] Rendering s_00 (Ye proxy) frames...")
    self._render_field(f"{self.jobname}.mhd_w_bcc_{self.plane}", "s_00",
                       "s00", cmap="plasma", norm_from_params=False,
                       vmin=0.0, vmax=0.5)

  def rad_E(self):
    print("[rad_E] Rendering radiation energy E:0 frames...")
    self._render_field(self._detect_rad_prefixes()['E'], "E:0", "rad_E0")

  def rad_N(self):
    print("[rad_N] Rendering radiation number density N:0 frames...")
    self._render_field(self._detect_rad_prefixes()['N'], "N:0", "rad_N0")

  def rad_N2(self):
    print("[rad_N2] Rendering radiation number density N:2 (heavy lepton) frames...")
    self._render_field(self._detect_rad_prefixes()['N'], "N:2", "rad_N2")

  def rad_abs(self):
    print("[rad_abs] Rendering absorption opacity abs_1:0 frames...")
    self._render_field(self._detect_rad_prefixes()['abs_1'], "abs_1:0", "abs1_0")

  def rad_abs0(self):
    print("[rad_abs0] Rendering absorption opacity abs_0:0 frames...")
    self._render_field(self._detect_rad_prefixes()['abs_0'], "abs_0:0", "abs0_0")

  def rad_eta0(self):
    print("[rad_eta0] Rendering emissivity eta_0:0 frames...")
    self._render_field(self._detect_rad_prefixes()['eta_0'], "eta_0:0", "eta0_0")

  def rad_eta1(self):
    print("[rad_eta1] Rendering emissivity eta_1:0 frames...")
    self._render_field(self._detect_rad_prefixes()['eta_1'], "eta_1:0", "eta1_0")

  def z4c(self):
    print("[z4c] Rendering con_H (Hamiltonian constraint) frames...")
    self._render_field(f"{self.jobname}.con_{self.plane}", "con_H",
                       "con_H", use_abs=True)

  def s00_avg(self):
    print("[s00_avg] Computing average Y_e (s_00) time series...")
    plot_s00_timeseries_all(self.output_dirs, self.figpath, self.jobname,
                            time_scale=self.time_scale, time_axis=self.time_axis)

  def avg_energy(self):
    print("[avg_energy] Rendering average neutrino energy frames...")
    render_avg_energy_all(self.output_dirs, self.figpath, print_minmax=True,
                          show_trackers=self.show_trackers, show_refinement=True,
                          trackers=self.trackers,
                          show_horizon=self.show_horizon,
                          horizons=self.horizons if self.show_horizon else None,
                          xlim=self.xy_lim, ylim=self.xy_lim,
                          units=self.units, jobname=self.jobname,
                          out_prefix=self._frame_out("avg_energy"),
                          subdir=self._frame_dir("avg_energy"),
                          plane=self.plane,
                          coord_scale=self.coord_scale, coord_label=self.coord_label,
                          time_scale=self.time_scale, time_tex=self.time_tex,
                          time_axis=self.time_axis)

  def _nu_energy(self, section, species, label):
    print(f"[{section}] Rendering neutrino energy E/{label} (species {species}) frames...")
    render_avg_energy_all(self.output_dirs, self.figpath, print_minmax=True,
                          show_trackers=self.show_trackers, show_refinement=True,
                          trackers=self.trackers,
                          show_horizon=self.show_horizon,
                          horizons=self.horizons if self.show_horizon else None,
                          xlim=self.xy_lim, ylim=self.xy_lim,
                          units=self.units, species=species, jobname=self.jobname,
                          out_prefix=self._frame_out(section),
                          subdir=self._frame_dir(section),
                          var_key='nu_energy',
                          plane=self.plane,
                          coord_scale=self.coord_scale, coord_label=self.coord_label,
                          time_scale=self.time_scale, time_tex=self.time_tex,
                          time_axis=self.time_axis)

  # --- dispatch -----------------------------------------------------------

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
      "z4c":         self.z4c,
      "s00_avg":     self.s00_avg,
      "avg_energy":  self.avg_energy,
    }
    for section, species, label in _NU_ENERGY_SECTIONS:
      table[section] = (lambda sec=section, sp=species, lb=label:
                        self._nu_energy(sec, sp, lb))
    return table

  def run(self, sections=("all",)):
    """Run the requested sections.

    Parameters:
    sections: iterable of section names, or containing 'all' to run everything.
    """
    if not self.output_dirs:
      print(f"ERROR: No output-XXXX directories found under {self.simpath}")
      raise SystemExit(1)

    configure_matplotlib()
    os.makedirs(self.figpath, exist_ok=True)

    sections = set(sections)
    run_all = "all" in sections

    print(f"SIMPATH  : {self.simpath}")
    print(f"FIGPATH  : {self.figpath}")
    print(f"JOBNAME  : {self.jobname}")
    print(f"Plane    : {self.plane}")
    print(f"Segments : {len(self.output_dirs)}")
    for d in self.output_dirs:
      print(f"  {d}")
    print(f"Sections : {', '.join(sorted(sections))}")
    print(f"Trackers : {'on' if self.show_trackers else 'off'}")
    print(f"Horizon  : {'on' if self.show_horizon else 'off'}")
    print(f"Domain   : {'full' if self.full_domain else '+/-60 Msun'}")
    print(f"Units    : {self.units}  (spatial axes in "
          f"{'M_sun' if self.coord_scale == 1.0 else 'km'})")
    print(f"Time     : {self.time_axis}\n")

    table = self._dispatch_table()
    for name, func in table.items():
      if run_all or name in sections:
        func()

    print("\nDone.")


# ---------------------------------------------------------------------------
# Command-line interface
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
  parser.add_argument("--plane", default="xy", choices=list(PLANES),
                      help="Slice plane to render: xy [default], xz, or yz")
  parser.add_argument("--time-units", dest="time_units", default="Msun",
                      choices=list(TIME_UNITS),
                      help="Time display units for titles/axes: Msun [default] or ms")
  parser.add_argument("--no-trackers", dest="no_trackers", action="store_true",
                      help="Disable tracker overlays on all 2-D slice plots")
  parser.add_argument("--show-horizon", dest="show_horizon", action="store_true",
                      help="Overlay black-hole apparent-horizon circles on 2-D slice plots")
  parser.add_argument("--full-domain", dest="full_domain", action="store_true",
                      help="Show full simulation domain instead of the default +/-60 Msun window")
  parser.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                      help="Skip frames whose output PNG already exists in figpath")
  parser.add_argument("--sections", nargs="+", default=["all"],
                      choices=SECTIONS + ["all"],
                      help="Which sections to run (default: all)")
  parser.add_argument("--units", default="cgs",
                      choices=["code", "cgs", "ngs"],
                      help="Unit system for physical quantities: "
                           "'code' = raw AthenaK GeometricKilometer units, "
                           "'cgs' = Gaussian CGS (g, cm, s, erg) [default], "
                           "'ngs' = bns_nurates units (nm, g, s, MeV)")
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  plotter = SystemPlotter(
    simpath=args.simpath,
    figpath=args.figpath,
    units=args.units,
    jobname=args.jobname,
    plane=args.plane,
    time_units=args.time_units,
    show_trackers=not args.no_trackers,
    show_horizon=args.show_horizon,
    full_domain=args.full_domain,
    skip_existing=args.skip_existing,
  )
  plotter.run(args.sections)


if __name__ == "__main__":
  main()
