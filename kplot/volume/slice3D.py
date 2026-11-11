"""
Functionality to slice 3D data along certain positions.

TODO: more general description, i.e. slice all three directions.
"""

# Built-in libraries.
import argparse
import os

# Third-party libraries.
import numpy as np

# KPlot utility.
from kplot.system.bin_convert import read_binary
from kplot.utils.bin3d import (find_snapshots, load_tracker, bin_time, tracker_center,
                               slice_cells)
from kplot.utils.memory import peak_rss_gb, reset_peak_rss
from kplot.utils.parallel import default_workers, pool_map


# ======================================================================
# MULTIPROCESS WORKER
# ======================================================================

# Per-worker state.
_W_TRACKER = None


def _init_worker(tracker: dict | None) -> None:
  """
  Pool initializer: store the tracker data.

  Parameters:
  tracker (dict): Tracker data from load_tracker, or None.
  
  Returns:
  None.
  """
  global _W_TRACKER
  _W_TRACKER = tracker


def out_file(outdir: str, tag: str, plane: str, snap: int, half_width: float) -> str:
  """
  Path of the .npz file of one slice.

  Parameters:
  outdir       (str): Output directory.
  tag          (str): Dump tag.
  plane        (str): 'xy' or 'xz'.
  snap         (int): Snapshot number.
  half_width (float): Half-width of the slice [code units].

  Returns:
  Path of the .npz file.
  """
  return os.path.join(outdir, f"{tag}_{plane}",
                      f"slice_{tag}_{snap:05d}_hw_{half_width:g}.npz")


def _slice_snapshot(task: tuple) -> None:
  """
  Per-snapshot worker: slice the dumps in all directions.

  Parameters:
  task (tuple): tag, file, snap, time, half_width, npts, center, outdir, print_memory

  Returns:
  None.

  Output: <outdir>/<tag>_<plane>/slice_<tag>_<snap>_hw_<half_width>.npz, with
    u      (npts,):        slice coordinates relative to the center [code units]
    time   ():             simulation time of the dump
    center (3,):           center (x, y, z) of the slice
    snap   ():             snapshot number
    <var>  (npts, npts):   float32 slice, NaN where not covered.
  Arrays are indexed [row, col] = [second axis, x], i.e. [y, x] for xy and
  [z, x] for xz.
  """
  # Unpack the tuple.
  tag, file, snap, time, half_width, npts, center, outdir, print_memory = task
  if print_memory: reset_peak_rss()
  c = tracker_center(_W_TRACKER, time, center)
  u = np.linspace(-half_width, half_width, npts)

  # Read the data.
  hw = np.abs(u).max()

  def keep(g: tuple) -> bool:
    """
    Whether a meshblock intersects the slice window.

    Parameters:
    g (tuple): Meshblock geometry (x1min, x1max, x2min, x2max, x3min, x3max).

    Returns:
    True if the block is read.
    """
    inside = (g[0] < c[0] + hw and g[1] > c[0] - hw and
              g[2] < c[1] + hw and g[3] > c[1] - hw and
              g[4] < c[2] + hw and g[5] > c[2] - hw)
    return inside and (g[2] <= c[1] < g[3] or g[4] <= c[2] < g[5])

  axes  = {"xy": (0, 1, 2), "xz": (0, 2, 1)}
  fd    = read_binary(file, keep)
  nx    = (fd["nx1_out_mb"], fd["nx2_out_mb"], fd["nx3_out_mb"])
  sel   = np.arange(fd["n_mbs"] * np.prod(nx))

  # Slice indices per plane.
  idx = {}
  for s, ax in axes.items():
    pos    = slice_cells(fd["mb_geometry"], nx, sel,
                         ax, c[ax[2]], c[ax[0]] + u, c[ax[1]] + u)
    idx[s] = (pos >= 0, np.maximum(pos, 0))

  out = {s: {} for s in axes}
  for var in fd["var_names"]:
    data = np.asarray(fd["mb_data"][var]).reshape(-1)
    for s, (ok, pos) in idx.items():
      out[s][var] = np.where(ok, data[pos], np.nan).astype(np.float32)
    del data

  # Save the file.
  for s in axes:
    path = out_file(outdir, tag, s, snap, half_width)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, u=u, time=time, center=c, snap=snap, **out[s])

  # Print memory diagnostics (optional).
  if print_memory:
    nbytes = sum(a.nbytes for v in fd["var_names"] for a in fd["mb_data"][v])
    print(f"$ [pid {os.getpid()}] {tag} {snap:05d}: {fd['n_mbs']} blocks, "
          f"read {nbytes / 1024**3:.2f} GB, peak RSS {peak_rss_gb():.2f} GB", flush=True)

  
def slice_all(simpath: str, batchtools: bool, suffix: str = "_3D", tags: list = None, 
          tracker: str = None, outdir: str = None, half_width: float = 100.0, 
          npts: int = 512, center: list = None, n_workers: int = None,
          overwrite: bool = False, print_memory: bool = False,
          snaps: list = None) -> None:
  """
  Slice 3D dumps along all three directions w.r.t. to a specified center. This is useful 
  if the remnant object is drifting, but also allows post-processing if no 2D slices were 
  specified in the output.

  Parameters:
  simpath       (str): Path to the 3D dumps.
  batchtools   (bool): Whether the dumps are in output-XXXX segments.
  suffix        (str): suffix of the 3D dump names.
  tags         (list): list of the tags to slice.
  tracker       (str): Path to the tracker file.
  outdir        (str): Path to the output directory of stored results.
  half_width  (float): Half-width of the slices around the center [code units].
  npts          (int): Number of points per slice axis.
  center       (list): Fixed center if no tracker file is given.
  n_workers     (int): Number of worker processes.
  overwrite    (bool): Overwrite already written slices.
  print_memory (bool): Print some memory diagnostics.
  snaps        (list): Snapshot numbers to slices (default: all).

  Returns:
  None.
  """
  # Find the files with the specified tags.
  files = {tag: find_snapshots(tag, simpath, batchtools, suffix) for tag in tags}

  # Restrict to selected snapshots.
  if snaps is not None:
    for tag in tags:
      missing = sorted(set(snaps) - files[tag].keys())
      if missing:
        print(f"$ Warning: {tag} has no snapshots {missing}!")
      files[tag] = {n: p for n, p in files[tag].items() if n in snaps}

  times = {t: {n: bin_time(p) for n, p in files[t].items()} for t in tags}

  # Set the tasks.
  tasks = []
  for tag in files.keys():
    for snap, file in files[tag].items():
      done = all(os.path.exists(out_file(outdir, tag, s, snap, half_width))
                 for s in ("xy", "xz"))
      if overwrite or not done:
        tasks.append((tag, file, snap, times[tag][snap], half_width, npts, center, outdir, 
                      print_memory))

  if not tasks:
    print("$ Nothing left to do.")
    return

  os.makedirs(outdir, exist_ok=True)

  # Initializer arguments and pool map.
  init_args = (load_tracker(tracker) if tracker is not None else None,)
  print(f"$ Slicing {len(tasks)} frames...")
  pool_map(_slice_snapshot, tasks, n_workers or default_workers(),
           initializer=_init_worker, initargs=init_args, progress=True)


def main(argv: list[str] | None = None) -> None:
  """
  CLI wrapper for the slicing routines.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  None.
  """
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--simpath", "--bindir", dest="simpath", required=True,
                  help="simulation directory holding the output-XXXX segments, "
                       "or (with --no-batchtools) the directory of the 3D bin files")
  ap.add_argument("--batchtools", action=argparse.BooleanOptionalAction, default=True,
                  help="search the 3D bin files in <simpath>/output-XXXX[/bin]; "
                       "--no-batchtools searches <simpath>[/bin] only")
  ap.add_argument("--suffix", default="_3D",
                  help="suffix of the 3D output ids, i.e. files <job>.<var><suffix>.NNNNN.bin "
                       "(use '' for unsuffixed ids)")
  ap.add_argument("--tags", required=True, nargs="+",
                  help="what tags to slice (convention: <job>.<tag><suffix>.NNNNN.bin)")
  ap.add_argument("--snaps", type=int, nargs="+", default=None,
                  help="snapshot numbers to slice (default: all)")
  ap.add_argument("--tracker", default=None,
                  help="path to the tracker file")
  ap.add_argument("--center", type=float, nargs=3, default=None,
                  help="fixed slice center if no tracker file is given")
  ap.add_argument("--outdir", required=True, help="where to write the .npz results")
  ap.add_argument("--slice-extent", type=float, default=100.0,
                  help="half-width of the slices around the center [code units]")
  ap.add_argument("--slice-npts", type=int, default=512,
                  help="number of points per axis of the slices")
  ap.add_argument("--n-workers", type=int, default=None,
                  help="number of worker processes")
  ap.add_argument("--overwrite", action="store_true", help="overwrite already written slices")
  ap.add_argument("--print-memory", action="store_true", help="print peak memory usage by workers")
  args = ap.parse_args(argv)

  slice_all(args.simpath, args.batchtools, args.suffix, args.tags,
        args.tracker, args.outdir, args.slice_extent, args.slice_npts,
        args.center, args.n_workers, args.overwrite, args.print_memory,
        args.snaps)


if __name__ == "__main__":
  main()