"""
Generic per-snapshot functionality for sph surface analyses.

An analysis provides ``compute(surface, ctx) -> dict``; `run` locates the
dumps, builds one `Surface` per snapshot and radius in a worker pool and hands
back the results per radius, sorted by time:

  def compute(s, ctx):
    f = fluid(s.metric, s.alpha, s.beta, s.vel)
    return {'Mdot': -(sqrt_det(s.metric) * s.alpha * s['dens'] * s.radial(f['u_up']) * s.dA).sum()}

  res = run(sph_dirs, 'bns', [50, 100], compute)
"""

# Built-in libraries.
import os
from typing import Any, Callable

# Third-party libraries.
import numpy as np

# Plot-tools utility.
from athplot.load_sph_vtk import SphericalData

# KPlot utility.
from .parallel import pool_map
from .quadrature import calc_ur, calc_uphi, get_riemann_weights
from .sph import CENTER_TOL, Shell, locate, read_time

# The mhd variables needed for the most general computations.
MHD_VARIABLES = ('mhd_w_bcc', 'adm', 'z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')


# ======================================================================
# Helper functions for one extraction surface.
# ======================================================================
class Surface:
  """
  One extraction surface of one snapshot, gathered from the dumps of every variable
  needed for the underlying computation.

  s['dens'] returns a field corrected for the points AthenaK duplicates on the
  phi = 0 seam and the north pole; s.raw(name) returns it as stored.

  Parameters:
  shells (list): list of the Shell objects of one snapshot and radius.
  """

  def __init__(self, shells: list) -> None:
    first = shells[0]
    self.time, self.radius = first.time, first.radius
    self.theta, self.phi = first.theta, first.phi
    self.center = _check_centers(shells)
    self._shell = {}
    for sh in shells:
      for name in sh.field_names:
        self._shell.setdefault(name, sh)
    self._cache = {}
    self.dup = _dup_correction(self.raw('z4c_alpha'), self.theta, self.phi) \
      if 'z4c_alpha' in self._shell else 1.0


  def raw(self, name: str) -> np.ndarray:
    """
    Field as stored in the dump, without the duplicate correction.

    Parameters:
    name (str): name of the field.

    Returns:
    array of the field on the surface.
    """
    return self._shell[name].shell(name)


  def __getitem__(self, name: str) -> np.ndarray:
    """
    Field corrected for the duplicated points, cached after the first access.

    Parameters:
    name (str): name of the field.

    Returns:
    array of the corrected field on the surface.
    """
    if name not in self._cache:
      self._cache[name] = self.raw(name) * self.dup
    return self._cache[name]


  @property
  def metric(self) -> tuple:
    """
    Spatial metric of the surface.

    Returns:
    tuple of the six metric components (gxx, gxy, gxz, gyy, gyz, gzz).
    """
    return tuple(self[f'adm_g{k}'] for k in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'))


  @property
  def alpha(self) -> np.ndarray:
    """
    Lapse on the surface.

    Returns:
    array of the lapse.
    """
    return self['z4c_alpha']


  @property
  def beta(self) -> list:
    """
    Shift vector on the surface.

    Returns:
    list of the shift components (x, y, z).
    """
    return [self[f'z4c_beta{k}'] for k in 'xyz']


  @property
  def vel(self) -> list:
    """
    Fluid velocity on the surface.

    Returns:
    list of the velocity components (x, y, z).
    """
    return [self[f'vel{k}'] for k in 'xyz']


  @property
  def bcc(self) -> list:
    """
    Cell-centered magnetic field on the surface.

    Returns:
    list of the magnetic field components (1, 2, 3).
    """
    return [self[f'bcc{k}'] for k in '123']


  @property
  def pos(self) -> tuple:
    """
    Cartesian position relative to the sphere center.

    Returns:
    tuple of the coordinates (x, y, z).
    """
    st = np.sin(self.theta)
    return (self.radius * st * np.cos(self.phi), self.radius * st * np.sin(self.phi),
            self.radius * np.cos(self.theta))


  @property
  def theta_1d(self) -> np.ndarray:
    """
    Polar angles of the surface grid.

    Returns:
    1D array of theta.
    """
    return self.theta[0, :]


  @property
  def phi_1d(self) -> np.ndarray:
    """
    Azimuthal angles of the surface grid.

    Returns:
    1D array of phi.
    """
    return self.phi[:, 0]


  @property
  def theta_edges(self) -> np.ndarray:
    """
    Cell edges in theta, from 0 to pi.

    Returns:
    1D array of the theta edges.
    """
    t = np.sort(self.theta_1d)
    return np.concatenate([[0.0], 0.5 * (t[:-1] + t[1:]), [np.pi]])


  @property
  def dA(self) -> np.ndarray:
    """
    Coordinate area element r^2 sin(theta) dtheta dphi.

    Returns:
    array of the area element on the surface.
    """
    return (get_riemann_weights(self.theta, self.theta_1d, self.phi_1d)
            * np.sin(self.theta) * self.radius**2)


  def radial(self, v: list) -> np.ndarray:
    """
    Radial component of a Cartesian vector.

    Parameters:
    v (list): list of the vector components (x, y, z).

    Returns:
    array of the radial component.
    """
    return calc_ur(*v, self.theta, self.phi)


  def azimuthal(self, v: list) -> np.ndarray:
    """
    Azimuthal component of a Cartesian vector.

    Parameters:
    v (list): list of the vector components (x, y, z).

    Returns:
    array of the azimuthal component.
    """
    return calc_uphi(v[0], v[1], self.phi)


def _check_centers(shells: list) -> np.ndarray:
  """
  Check that all shells share the same sphere center.

  Parameters:
  shells (list): list of the Shell objects of one snapshot and radius.

  Returns:
  array of the common sphere center.

  Raises:
  RuntimeError if the centers differ by more than CENTER_TOL.
  """
  centers = np.array([sh.center for sh in shells])
  if not np.allclose(centers, centers[0], rtol=0.0, atol=CENTER_TOL):
    detail = '\n'.join(f'  {os.path.basename(sh.path)}: ({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})'
                       for sh, c in zip(shells, centers))
    raise RuntimeError(f'Sphere centers disagree at t = {shells[0].time:g} M_sun:\n{detail}\n'
                       'Every sph output block feeding one analysis must use the same '
                       'center_tracker.')
  return centers[0]


def _dup_correction(alpha: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
  """
  Correction factor for the points AthenaK duplicates on the phi = 0 seam and north pole.

  Parameters:
  alpha (np.ndarray): lapse on the surface.
  theta (np.ndarray): polar angle on the surface.
  phi   (np.ndarray): azimuthal angle on the surface.

  Returns:
  array of the correction factor (1, 0.5 on the seam, 0.25 on the pole).
  """
  # Duplicated points are stored summed; the lapse (< 1) shows where.
  phi_zero = (phi == phi.min())
  theta_north = (theta == theta.min())
  dup = np.ones(alpha.shape)
  if alpha[phi_zero & ~theta_north].max() > 1.0:
    dup[phi_zero] = 0.5
  if alpha[theta_north].max() > 1.0:
    dup[theta_north] = 0.25
  return dup


# ======================================================================
# Helper functions for the driver function.
# ======================================================================
_COMPUTE = None
_CTX     = None


def _init(compute: Callable, ctx: Any) -> None:
  """
  Worker initializer, storing the compute function and its context.

  Parameters:
  compute (Callable): analysis function compute(surface, ctx).
  ctx          (Any): context passed to compute.
  """
  global _COMPUTE, _CTX
  _COMPUTE, _CTX = compute, ctx


def _worker(task: list) -> dict:
  """
  Compute all radii of one snapshot; each dump is parsed once.

  Parameters:
  task (list): list of (radius, [(path, surface index), ...]) of one snapshot.

  Returns:
  dict {radius: result dict} with also the 'time' and 'center' of the surface.
  """
  parsed = {}

  def shell(entry: tuple) -> Shell:
    path, iradius = entry
    if path not in parsed:
      # Read the dump in large chunks first: the VTK reader's small reads are slow
      # on GPFS, but fast once the file sits in the page cache.
      with open(path, 'rb') as f:
        while f.read(64 << 20):
          pass
      parsed[path] = SphericalData(path)
    return Shell(path, iradius, data=parsed[path])

  out = {}
  for radius, entries in task:
    s = Surface([shell(e) for e in entries])
    out[radius] = {'time': s.time, 'center': s.center, **_COMPUTE(s, _CTX)}
  return out


def _nearest(times: np.ndarray, entries: list, t: float, tol: float = 4.0) -> tuple:
  """
  Entry closest in time to `t`.

  Parameters:
  times (np.ndarray): times of the entries [M_sun].
  entries     (list): list of the (path, surface index) entries.
  t          (float): target time [M_sun].
  tol        (float): maximum time difference [M_sun].

  Returns:
  tuple (path, surface index) of the closest entry, else None if further than tol.
  """
  i = int(np.argmin(np.abs(times - t)))
  return entries[i] if abs(times[i] - t) <= tol else None


def run(sph_dirs: list, jobname: str, radii: float | list, compute: Callable, ctx: Any = None,
        variables: tuple = MHD_VARIABLES, by_time: tuple = (), n_workers: int = None,
        hint: str = '') -> dict:
  """
  Run compute(surface, ctx) on every snapshot holding all `variables` at each radius.

  Parameters:
  sph_dirs        (list): list of the AthenaK output-XXXX/sph directories.
  jobname          (str): job name the dumps are named after.
  radii   (float | list): extraction radius or list of radii [M_sun].
  compute     (Callable): module-level analysis function compute(surface, ctx) -> dict.
  ctx              (Any): context passed to compute, such as the EOS.
  variables      (tuple): sph output variables needed on the same snapshot counter.
  by_time        (tuple): variables that may use another snapshot counter and are
                          matched to the nearest time instead.
  n_workers        (int): number of parallel workers.
  hint             (str): appended to the error message when nothing matches.

  Returns:
  dict {radius: [result dicts, sorted by time]}; each dict also holds 'time' and 'center'.
  """
  radii = sorted(set(float(r) for r in np.atleast_1d(radii)))
  variables, by_time = tuple(variables), tuple(by_time)
  tasks = {}
  for radius in radii:
    located = {v: locate(sph_dirs, jobname, v, radius, hint=hint) for v in variables + by_time}
    for s in located.values():
      print(f"  {s.describe()}")
    index_sets = [set(located[v].paths) for v in variables]
    complete = sorted(set.intersection(*index_sets))
    skipped = len(set.union(*index_sets)) - len(complete)
    timed = {}
    for v in by_time:
      entries = list(located[v].entries.values())
      timed[v] = (np.array([read_time(p) for p, _ in entries]), entries)
    found = 0
    for i in complete:
      entries = [located[v].shell(i) for v in variables]
      if by_time:
        t = read_time(entries[0][0])
        extra = [_nearest(*timed[v], t) for v in by_time]
        if any(e is None for e in extra):
          skipped += 1
          continue
        entries += extra
      tasks.setdefault(i, []).append((radius, entries))
      found += 1
    if skipped:
      print(f"  [skip] r = {radius:g}: {skipped} snapshots missing one of "
            f"{', '.join(variables + by_time)}")
    print(f"Found {found} snapshots at r = {radius:g}.")

  if not tasks:
    raise RuntimeError(
      f"No snapshot in {list(sph_dirs)} has all of {', '.join(variables + by_time)} "
      f"at any of r = {', '.join(f'{r:g}' for r in radii)}.\n"
      f"Check the sph directories, the job name and the extraction radii.{hint}")

  print(f"Processing {len(tasks)} snapshots...")
  results = pool_map(_worker, [tasks[i] for i in sorted(tasks)], n_workers,
                     initializer=_init, initargs=(compute, ctx))
  return {radius: sorted((r[radius] for r in results if radius in r), key=lambda d: d['time'])
          for radius in radii}


# ======================================================================
# Helper functions for the results.
# ======================================================================
def stack(results: list, key: str) -> np.ndarray:
  """
  Stack one key over a list of result dicts.

  Parameters:
  results (list): list of the result dicts.
  key      (str): key to stack.

  Returns:
  array of `key` over the results.
  """
  return np.array([r[key] for r in results])


def save_centers(path: str, results: list) -> None:
  """
  Write and report the sphere center over time.

  Parameters:
  path     (str): path to the output text file.
  results (list): list of the result dicts holding 'time' and 'center'.
  """
  t, c = stack(results, 'time'), stack(results, 'center')
  shift = np.linalg.norm(c - c[0], axis=1).max()
  print(f"Sphere center: ({c[0][0]:.6g}, {c[0][1]:.6g}, {c[0][2]:.6g}) -> "
        f"({c[-1][0]:.6g}, {c[-1][1]:.6g}, {c[-1][2]:.6g}) M_sun, max excursion {shift:.6g}")
  np.savetxt(path, np.column_stack((t, c)),
             header='time(Msun)    xc(Msun)    yc(Msun)    zc(Msun)', fmt='%.6e')
