"""
Locating AthenaK spherical-surface (``file_type = sph``) VTK dumps.

AthenaK names a dump after the surfaces it holds:

    <job>.r=300.00.<var>.NNNNN.vtk          one surface at r = 300
    <job>.r=300.00-400.00.<var>.NNNNN.vtk   several surfaces spanning r = 300 .. 400
"""

# Built-in libraries.
import os
import re
from functools import lru_cache

# Third-party libaries.
import numpy as np

# Plot-tools utility.
from athplot.load_sph_vtk import SphericalData

# A stored surface counts as the requested radius when it lies within this distance
# [M_sun].
RADIUS_TOL = 0.05
CENTER_TOL = 0.5


# ======================================================================
# Helper functions for the file names and metadata.
# ======================================================================
@lru_cache(maxsize=None)
def _name_pattern(jobname: str, variable: str) -> re.Pattern:
  """
  Regex for `<jobname>.r=RMIN[-RMAX].<variable>.NNNNN.vtk`.

  Parameters:
  jobname  (str): job name the dumps are named after.
  variable (str): sph output variable, such as 'mhd_w_bcc'.

  Returns:
  compiled regex matching the dump names.
  """
  num = r'\d+(?:\.\d+)?'
  return re.compile(rf'^{re.escape(jobname)}\.r=({num})(?:-({num}))?'
                    rf'\.{re.escape(variable)}\.(\d+)\.vtk$')


def parse_name(fname: str, jobname: str, variable: str) -> tuple:
  """
  Parse the radius range and snapshot number from a dump name.

  Parameters:
  fname    (str): file name of the dump.
  jobname  (str): job name the dumps are named after.
  variable (str): sph output variable, such as 'mhd_w_bcc'.

  Returns:
  tuple (rmin, rmax, index) for a matching dump name, else None.
  Single-surface names give rmin == rmax.
  """
  match = _name_pattern(jobname, variable).match(fname)
  if match is None:
    return None
  rmin = float(match.group(1))
  rmax = float(match.group(2)) if match.group(2) is not None else rmin
  return rmin, rmax, int(match.group(3))


def spec_str(rmin: float, rmax: float) -> str:
  """
  The `r=...` part of a dump name for the given radius range.

  Parameters:
  rmin (float): smallest radius in the dump.
  rmax (float): largest radius in the dump.

  Returns:
  string 'r=RMIN' or 'r=RMIN-RMAX'.
  """
  return f'r={rmin:.2f}' if rmin == rmax else f'r={rmin:.2f}-{rmax:.2f}'


def read_header(path: str, encoding: str = 'latin_1') -> dict:
  """
  Parse the `key=value` metadata AthenaK writes on the second line.

  Parameters:
  path     (str): path to the dump.
  encoding (str): encoding of the file.

  Returns:
  dict with the metadata keys and their (string) values.
  """
  with open(path, 'r', encoding=encoding) as f:
    f.readline()
    meta = f.readline()
  header = {}
  for token in meta.split():
    key, sep, value = token.partition('=')
    if sep:
      header[key] = value
  return header


def read_time(path: str) -> float:
  """
  Simulation time of a dump, without parsing the field data.

  Parameters:
  path (str): path to the dump.

  Returns:
  simulation time [M_sun].
  """
  return float(read_header(path)['time'])


def dump_nradii(path: str, rmin: float, rmax: float) -> int:
  """
  How many surfaces a dump holds, from its metadata line.

  Parameters:
  path   (str): path to the dump.
  rmin (float): smallest radius in the dump.
  rmax (float): largest radius in the dump.

  Returns:
  number of surfaces in the dump.
  """
  if rmin == rmax:
    return 1
  nradii = read_header(path).get('nradii')
  return int(nradii) if nradii is not None else SphericalData(path).nradii


def _stored_radii(path: str, rmin: float, rmax: float) -> np.ndarray:
  """
  The radii a dump holds.

  Parameters:
  path   (str): path to the dump.
  rmin (float): smallest radius in the dump.
  rmax (float): largest radius in the dump.

  Returns:
  array of the stored radii.
  """
  if rmin == rmax:
    return np.array([rmin])
  return np.asarray(SphericalData(path).radii, dtype=float)


def _fmt_radii(radii: np.ndarray) -> str:
  """
  Format the radii for the error message, shortened if there are more than four.

  Parameters:
  radii (np.ndarray): array of the radii.

  Returns:
  comma-separated string of the radii.
  """
  if len(radii) <= 4:
    return ', '.join(f'{r:.2f}' for r in radii)
  return ', '.join(f'{r:.2f}' for r in radii[:3]) + f', ..., {radii[-1]:.2f}'


# ======================================================================
# Helper functions for the snapshot lookup.
# ======================================================================
class SnapshotSet:
  """
  The dumps of one variable that hold one requested extraction surface.

  Attributes:
  variable  (str): sph output variable, such as 'mhd_w_bcc' or 'rad_m1_F'.
  radius  (float): radius of the surface as stored in the files [M_sun].
  entries  (dict): {snapshot counter: (path, surface index)}.
  groups   (list): (name spec, nradii, surface index, count) per dump form, for the log.
  """

  def __init__(self, variable: str, radius: float, entries: dict, groups: list) -> None:
    self.variable = variable
    self.radius = radius
    self.entries = entries
    self.groups = groups

  @property
  def indices(self) -> list:
    """
    Snapshot counters, sorted.

    Returns:
    sorted list of the snapshot counters.
    """
    return sorted(self.entries)

  @property
  def paths(self) -> dict:
    """
    Snapshot paths, dropping the surface index.

    Returns:
    dict {snapshot counter: path}.
    """
    return {index: path for index, (path, _) in self.entries.items()}

  def shell(self, index: int) -> tuple:
    """
    Path and surface index of a snapshot, ready to hand to Shell.

    Parameters:
    index (int): snapshot counter.

    Returns:
    tuple (path, surface index).
    """
    return self.entries[index]

  def describe(self) -> str:
    """
    One-line summary for the analysis log.

    Returns:
    summary string of the dumps and surfaces found.
    """
    line = f'{len(self.entries):5d} {self.variable} dumps at r = {self.radius:.2f}'
    if len(self.groups) == 1 and self.groups[0][1] == 1:
      return line
    detail = ', '.join(
        f'{spec}: surface {iradius} of {nradii} ({count})' if nradii > 1
        else f'{spec} ({count})'
        for spec, nradii, iradius, count in self.groups)
    return f'{line}  [{detail}]'

  def __len__(self) -> int:
    return len(self.entries)


def locate(sph_dirs: list, jobname: str, variable: str, radius: float,
           tol: float = RADIUS_TOL, hint: str = '') -> SnapshotSet:
  """
  Find the dumps of `variable` that hold the extraction surface at `radius`.

  Parameters:
  sph_dirs (list): list of the AthenaK output-XXXX/sph directories.
  jobname   (str): job name the dumps are named after.
  variable  (str): sph output variable, such as 'mhd_w_bcc'.
  radius  (float): requested extraction radius [M_sun].
  tol     (float): how far a stored surface may sit from `radius` to still count [M_sun].
  hint      (str): appended to the error message when nothing matches, e.g. which
                   parfile output blocks the caller needs.

  Returns:
  SnapshotSet of the dumps holding the surface.

  Raises:
  RuntimeError if no dump of `variable` exists, or none holds a surface at `radius`.
  """
  # Dumps are grouped by the surfaces they hold, so that one surface index is
  # valid for every dump of a group.
  groups = {}   # (rmin, rmax, nradii) -> {snapshot counter: path}
  for d in sph_dirs:
    if not os.path.isdir(d):
      continue
    for fname in os.listdir(d):
      parsed = parse_name(fname, jobname, variable)
      if parsed is None:
        continue
      rmin, rmax, index = parsed
      path = os.path.join(d, fname)
      key = (rmin, rmax, dump_nradii(path, rmin, rmax))
      groups.setdefault(key, {})[index] = path

  if not groups:
    raise RuntimeError(
        f'No {jobname}.r=*.{variable}.*.vtk dumps found in {list(sph_dirs)}.\n'
        f'Check the sph directories, the job name and the extraction radius.{hint}')

  matches, stored = [], {}
  for key, paths in sorted(groups.items()):
    rmin, rmax, nradii = key
    if not (rmin - tol <= radius <= rmax + tol):
      continue
    radii = _stored_radii(paths[min(paths)], rmin, rmax)
    stored[key] = radii
    i = int(np.argmin(np.abs(radii - radius)))
    if abs(radii[i] - radius) <= tol:
      matches.append((abs(radii[i] - radius), nradii, key, i,
                      float(radii[i]), paths))

  if not matches:
    found = ', '.join(
        spec_str(rmin, rmax) + (f' (surfaces at {_fmt_radii(stored[key])})'
                                if len(stored.get(key, ())) > 1 else '')
        for key in sorted(groups) for rmin, rmax, _ in (key,))
    raise RuntimeError(
        f'No {variable} surface at r = {radius:g} (within {tol:g} M_sun) in '
        f'{list(sph_dirs)}.\nThe dumps found cover {found}.{hint}')

  matches.sort(key=lambda m: m[:2])
  entries, group_info = {}, []
  for _, nradii, (rmin, rmax, _), iradius, _, paths in matches:
    added = 0
    for index, path in paths.items():
      # A counter shared by two groups keeps the closer surface.
      if index not in entries:
        entries[index] = (path, iradius)
        added += 1
    group_info.append((spec_str(rmin, rmax), nradii, iradius, added))
  return SnapshotSet(variable, matches[0][4], entries, group_info)


# ======================================================================
# Helper functions for reading one surface.
# ======================================================================
class Shell:
  """
  One extraction surface of an sph dump, however many surfaces the file holds.

  Parameters:
  path               (str): path to the dump.
  iradius            (int): index of the surface in the dump.
  data (SphericalData): already loaded dump, read from `path` if None.
  """

  def __init__(self, path: str, iradius: int = 0, data: SphericalData = None) -> None:
    self._data = data if data is not None else SphericalData(path)
    if not 0 <= iradius < self._data.nradii:
      raise IndexError(f'{os.path.basename(path)} holds '
                       f'{self._data.nradii} surface(s), no index {iradius}.')
    self._iradius = iradius
    self.path = path
    self.iradius = iradius
    self.nradii = self._data.nradii
    self.radius = float(self._data.radii[iradius])
    self.center = np.array([getattr(self._data, k, 0.0)
                            for k in ('xc', 'yc', 'zc')])
    self.time = self._data.time
    self.cycle = self._data.cycle
    self.theta = self._data.theta
    self.phi = self._data.phi
    self.field_names = self._data.field_names

  def shell(self, var: str) -> np.ndarray:
    """
    Field `var` on the selected surface.

    Parameters:
    var (str): name of the field.

    Returns:
    array of the field with shape (Nphi, Ntheta).
    """
    return self._data.shell(var, self._iradius)
