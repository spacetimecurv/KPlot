"""AthenaK 3D binary dumps: lookup, meshblock geometry and tracker/horizon files."""

# Built-in libraries.
import glob
import os
import re

# Third-party libraries.
import numpy as np

# KPlot utility.
from kplot.system.checkpoints import find_segments

# ======================================================================
# Helper functions for the 3D data.
# ======================================================================
def find_snapshots(tag: str, simpath: str, batchtools: bool = True, suffix: str = "_3D") -> dict:
  """
  Locates the 3D snapshots `<job>.<tag><suffix>.NNNNN.bin` as {snapshot number: path}.

  With batchtools, search every output-XXXX segment under simpath (in bin/
  if present). Take latest snapshot if multiple with same number are found.
  Without batchtools seach simpath directly.

  Parameters:
  tag         (str): variable tag, such as 'mhd_w_bcc'.
  simpath     (str): path to the data.
  batchtools (bool): flag whether output data is in output-XXXX segments.
  suffix      (str): suffix of the file name, i.e. <job>.<tag><suffix>.bin.

  Returns:
  dict with the snapshot number and its path.
  """
  pat = f"*.{tag}{suffix}.*.bin"
  name = re.compile(rf".+\.{re.escape(tag + suffix)}\.(\d+)\.bin")
  hits = {}
  for segment in find_segments(simpath, batchtools):
    for sub in ("bin", ""):
      files = glob.glob(os.path.join(segment, sub, pat))
      if files:
        break
    for f in sorted(files):
      m = name.fullmatch(os.path.basename(f))
      if m:
        hits[int(m.group(1))] = f
  if not hits:
    raise SystemExit(f"No file matching {pat} under {simpath}!")

  return hits


def snapshot_tag(path: str) -> float:
  """
  Snapshot number as it appears in the file name, e.g. '00010'.

  Parameters:
  path (str): path to the snapshot file.

  Returns:
  Snapshot number.
  """
  return path.split("/")[-1].split(".")[-2].strip()


def parse_par_dump(header: list) -> dict:
  """
  Turn the <block>/key = value par dump carried in the bin header into a dict.

  Parameters:
  header (list): list of the parameter header of a 3D dump from system.bin_convert

  Returns:
  dict holding the block, the keys in the block and their values.
  """
  pars, block = {}, None
  for line in header:
    line = line.strip()
    if line.startswith("<") and line.endswith(">"):
      block = line[1:-1]
      pars.setdefault(block, {})
    elif "=" in line and block is not None:
      key, val = line.split("=", 1)
      pars[block][key.strip()] = val.strip()
  return pars


def par(pars: dict, block: str, key: str, cast: type = float, default: type = None):
  """
  Try to case a value to a specific type from a dictionary.

  Parameters:
  pars    (dict): dict holding the parameter keys and values.
  block    (str): name of the block in the parfile.
  key      (str): key name in the block.
  case    (type): type of the case.
  default (type): default type if case fails.

  Returns:
  casted parameter key value.
  """
  try:
    return cast(pars[block][key])
  except (KeyError, ValueError):
    return default


def block_cell_geometry(fd: dict) -> tuple[float, float, float,
                                           float, float, float, float]:
  """
  Cell-centre coordinates and cell volume for every meshblock.

  Parameters:
  fd (dict): dict holding the cell geometry from the 3D dump.

  Returns:
  Cell-centered coordinates, the number of points, and the spacing of the MeshBlock.
  """
  geom = fd["mb_geometry"]
  nx1, nx2, nx3 = fd["nx1_out_mb"], fd["nx2_out_mb"], fd["nx3_out_mb"]
  dx1 = (geom[:, 1] - geom[:, 0]) / nx1
  dx2 = (geom[:, 3] - geom[:, 2]) / nx2
  dx3 = (geom[:, 5] - geom[:, 4]) / nx3
  xv = geom[:, 0, None] + dx1[:, None] * (np.arange(nx1) + 0.5)
  yv = geom[:, 2, None] + dx2[:, None] * (np.arange(nx2) + 0.5)
  zv = geom[:, 4, None] + dx3[:, None] * (np.arange(nx3) + 0.5)
  return xv, yv, zv, dx1 * dx2 * dx3, dx1, dx2, dx3


def slice_cells(geom: np.ndarray, nx: tuple[int, int, int], sel: np.ndarray,
                axes: tuple[int, int, int], fixed: float,
                ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
  """
  Index into `sel` of the cell holding each point (ca, cb) on the plane axes[2] = fixed.

  Parameters:
  geom (np.ndarray): MeshBlock bounds of shape (n_mbs, 6), (x1min, x1max, x2min, ...).
  nx        (tuple): number of cells per MeshBlock (nx1, nx2, nx3).
  sel  (np.ndarray): sorted flat cell indices of the selected cells.
  axes      (tuple): (a, b, c) axes, where a and b span the plane and c is its normal.
  fixed     (float): coordinate of the plane along axis c.
  ca   (np.ndarray): sorted sample coordinates along axis a.
  cb   (np.ndarray): sorted sample coordinates along axis b.

  Returns:
  np.ndarray of shape (cb.size, ca.size) with the index into `sel`, or -1 if the
  point is not covered by a selected cell.
  """
  a, b, c = axes
  lo, hi = geom[:, 0::2], geom[:, 1::2]
  d      = (hi - lo) / np.asarray(nx)
  ncell  = nx[0] * nx[1] * nx[2]
  stride = (1, nx[0], nx[0] * nx[1])
  out    = np.full((cb.size, ca.size), -1, dtype=np.int64)
  for m in np.flatnonzero((lo[:, c] <= fixed) & (fixed < hi[:, c])):
    ia = slice(*np.searchsorted(ca, (lo[m, a], hi[m, a])))
    ib = slice(*np.searchsorted(cb, (lo[m, b], hi[m, b])))
    if ia.start == ia.stop or ib.start == ib.stop:
      continue
    i_a = np.minimum(((ca[ia] - lo[m, a]) / d[m, a]).astype(np.int64), nx[a] - 1)
    i_b = np.minimum(((cb[ib] - lo[m, b]) / d[m, b]).astype(np.int64), nx[b] - 1)
    i_c = min(int((fixed - lo[m, c]) / d[m, c]), nx[c] - 1)
    out[ib, ia] = (m * ncell + i_c * stride[c] +
                   i_b[:, None] * stride[b] + i_a[None, :] * stride[a])
  pos = np.clip(np.searchsorted(sel, out), 0, sel.size - 1)
  return np.where((out >= 0) & (sel[pos] == out), pos, -1)


def bin_time(path: str) -> float:
  """
  Time of a 3D dump, read from the header.

  Parameters:
  path (str): path to the .bin file.

  Returns:
  Simulation time of the dump.
  """
  with open(path, "rb") as fp:
    fp.readline()
    n = int(fp.readline().split(b"=")[-1])
    for _ in range(n - 1):
      key, val = fp.readline().decode("utf-8").split("=")
      if key.strip() == "time":
        return float(val)
  raise ValueError(f"No time in the header of {path}!")


# ======================================================================
# Helper functions for tracker/horizon files.
# ======================================================================
def load_tracker(path: str) -> dict:
  """
  Loads the tracker data.

  Parameters:
  path (str): Path to the tracker file.

  Returns:
  dict with the time and position columns.
  """
  data = np.loadtxt(path, comments="#")
  return {"time": data[:, 1], "x": data[:, 2], "y": data[:, 3], "z": data[:, 4]}


def load_horizon(path: str) -> dict:
  """
  Load the horizon data.

  Parameters:
  path (str): Path to the horizon file.

  Returns:
  dict with the time, mass, and radius columns of the AH.
  """
  data = np.loadtxt(path, comments="#")
  return {"time": data[:, 1], "mass": data[:, 2], "radius": data[:, -2]}


def tracker_center(tracker: dict, time: float, center: list = None) -> np.array:
  """
  Tracker position closest to `time`, else the fixed `center`.

  Parameters:
  tracker (dict): dict holding the tracker time and position.
  time   (float): target time at which to get the tracker position.
  center  (list): list of the fixed center. Only if no tracker file.

  Returns:
  np.array with the tracker positions at 'time'.
  """
  if tracker is not None:
    i = np.argmin(np.abs(tracker["time"] - time))
    return np.array([tracker["x"][i], tracker["y"][i], tracker["z"][i]])
  if center:
    return np.array(center)
  raise SystemExit("Either tracker file or manual center has to be specified!")
