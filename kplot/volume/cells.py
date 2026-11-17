"""
Cells of one snapshot for the volume analyses.

The cells come either from 3D bin dumps (AMR meshblocks) or from an sph volume
(concentric surfaces around a tracker). Analyses work on flat arrays of the kept cells;
the grid is only needed for gradients and plane samples, which `AmrGrid` and `SphGrid`
provide with the same interface.
"""

# Built-in libraries.
from dataclasses import dataclass

# Third-party libraries.
import numpy as np
from scipy.ndimage import uniform_filter

# KPlot utility.
from kplot.utils.bin3d import block_cell_geometry, parse_par_dump, par, read_binary, slice_cells
from kplot.utils.catalog import Snapshot
from kplot.utils.sph import CENTER_TOL, SphDump, dup_correction


@dataclass
class Cells:
  """
  Kept cells of one snapshot.

  Attributes:
  time  (float): simulation time [M_sun].
  tag     (str): snapshot tag for output file names.
  pars   (dict): parfile parameters (empty for sph dumps).
  origin (np.ndarray): center of the sph volume, None for bin dumps.
  x, y, z (np.ndarray): cell-center coordinates.
  dV (np.ndarray): coordinate cell volume.
  dz (np.ndarray): native grid spacing (NaN where unknown, i.e. on sph volumes).
  h  (np.ndarray): cell size used for grid-scale cuts.
  dz_min (float): finest native grid spacing on the mesh (NaN where unknown).
  data   (dict): {field: values on the kept cells}.
  grid (AmrGrid | SphGrid): grid of the snapshot.
  ncells_total (int): number of cells before the density cut.
  """
  time: float
  tag: str
  pars: dict
  origin: np.ndarray | None
  x: np.ndarray
  y: np.ndarray
  z: np.ndarray
  dV: np.ndarray
  dz: np.ndarray
  h: np.ndarray
  dz_min: float
  data: dict
  grid: object
  ncells_total: int


class AmrGrid:
  """
  Meshblock grid of a 3D bin dump, restricted to the kept cells `sel`.
  """

  def __init__(self, geom: np.ndarray, nx: tuple, sel: np.ndarray, dx: tuple) -> None:
    self.geom, self.nx, self.sel, self.dx = geom, nx, sel, dx
    self.ncells = len(geom) * nx[0] * nx[1] * nx[2]

  def gradient(self, q: np.ndarray, axes: str = 'xyz', smooth: int = 1) -> dict:
    """
    Cartesian derivatives of q per meshblock (one-sided at the block edges).

    Parameters:
    q    (np.ndarray): field on the kept cells.
    axes        (str): components to compute, e.g. 'z' or 'xyz'.
    smooth      (int): box-smoothing width in cells before differencing (1 = none).

    Returns:
    dict {axis: derivative on the kept cells}.
    """
    nx1, nx2, nx3 = self.nx
    full = np.full(self.ncells, np.nan, dtype=np.float32)
    full[self.sel] = q
    full = full.reshape(-1, nx3, nx2, nx1)
    if smooth > 1:
      full = _smooth(full, (1, smooth, smooth, smooth), 'nearest')
    out = {}
    for a in axes:
      i = 'xyz'.index(a)
      out[a] = (np.gradient(full, axis=3 - i) / self.dx[i][:, None, None, None]).reshape(-1)[self.sel]
    return out

  def plane(self, axes: tuple, fixed: float, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """
    Index into the kept cells of the cell holding each point (ca, cb) on axes[2] = fixed.

    Returns:
    array of shape (cb.size, ca.size), -1 where no kept cell covers the point.
    """
    return slice_cells(self.geom, self.nx, self.sel, axes, fixed, ca, cb)


class SphGrid:
  """
  Spherical grid (radius, phi, theta) of an sph volume, restricted to the kept cells `sel`.
  """

  def __init__(self, radii: np.ndarray, theta: np.ndarray, phi: np.ndarray,
               center: np.ndarray, sel: np.ndarray) -> None:
    self.r, self.theta, self.phi = radii, theta, phi
    self.center, self.sel = center, sel
    self.shape = (len(radii), len(phi), len(theta))
    self.r_edges = _radial_edges(radii)
    self.theta_edges = _theta_edges(theta)
    ir, ip, it = np.unravel_index(sel, self.shape)
    self._r, self._th, self._ph = radii[ir], theta[it], phi[ip]

  def gradient(self, q: np.ndarray, axes: str = 'xyz', smooth: int = 1) -> dict:
    """
    Cartesian derivatives of q from the spherical ones (periodic in phi).

    Parameters:
    q    (np.ndarray): field on the kept cells.
    axes        (str): components to compute, e.g. 'z' or 'xyz'.
    smooth      (int): box-smoothing width in grid points before differencing (1 = none).

    Returns:
    dict {axis: derivative on the kept cells}.
    """
    full = np.full(np.prod(self.shape), np.nan, dtype=np.float32)
    full[self.sel] = q
    full = full.reshape(self.shape)
    if smooth > 1:
      full = _smooth(full, smooth, ('nearest', 'wrap', 'nearest'))
    dphi = 2.0 * np.pi / self.shape[1]
    d_r = np.gradient(full, self.r, axis=0).reshape(-1)[self.sel]
    d_t = np.gradient(full, self.theta, axis=2).reshape(-1)[self.sel] / self._r
    with np.errstate(divide='ignore', invalid='ignore'):
      d_p = ((np.roll(full, -1, axis=1) - np.roll(full, 1, axis=1)) / (2.0 * dphi)
             ).reshape(-1)[self.sel] / (self._r * np.sin(self._th))
    st, ct, sp, cp = np.sin(self._th), np.cos(self._th), np.sin(self._ph), np.cos(self._ph)
    comp = {'x': lambda: st * cp * d_r + ct * cp * d_t - sp * d_p,
            'y': lambda: st * sp * d_r + ct * sp * d_t + cp * d_p,
            'z': lambda: ct * d_r - st * d_t}
    return {a: comp[a]() for a in axes}

  def plane(self, axes: tuple, fixed: float, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """
    Index into the kept cells of the grid point nearest to each point (ca, cb) on
    axes[2] = fixed.

    Returns:
    array of shape (cb.size, ca.size), -1 where no kept cell covers the point.
    """
    p = np.zeros((cb.size, ca.size, 3))
    p[..., axes[0]] = ca[None, :]
    p[..., axes[1]] = cb[:, None]
    p[..., axes[2]] = fixed
    p -= self.center
    r = np.linalg.norm(p, axis=-1)
    th = np.arccos(np.clip(p[..., 2] / np.maximum(r, 1e-30), -1.0, 1.0))
    ph = np.mod(np.arctan2(p[..., 1], p[..., 0]), 2.0 * np.pi)
    ir = np.digitize(r, self.r_edges) - 1
    it = np.clip(np.digitize(th, self.theta_edges) - 1, 0, self.shape[2] - 1)
    ip = np.rint(ph / (2.0 * np.pi / self.shape[1])).astype(np.int64) % self.shape[1]
    inside = (ir >= 0) & (ir < self.shape[0])
    flat = np.where(inside, (np.clip(ir, 0, None) * self.shape[1] + ip) * self.shape[2] + it, -1)
    pos = np.clip(np.searchsorted(self.sel, flat), 0, self.sel.size - 1)
    return np.where(inside & (self.sel[pos] == flat), pos, -1)


def read_cells(snap: Snapshot, rho_keep: float | None = None) -> Cells:
  """
  Read the cells of a snapshot with density above rho_keep.

  Parameters:
  snap      (Snapshot): snapshot from kplot.utils.catalog.find; its fields are read.
  rho_keep     (float): density cut in code units (default: 1.5 x the density floor).

  Returns:
  Cells.
  """
  first = snap.sources['dens'][0]
  reader = _read_amr if first.endswith('.bin') else _read_sph
  return reader(snap, rho_keep)


def read_blocks(snap: Snapshot, keep=None) -> dict:
  """
  Meshblock data of every field of a bin snapshot, merged into one read_binary dict.

  Parameters:
  snap (Snapshot): snapshot from kplot.utils.catalog.find.
  keep (callable): keep(mb_geometry) -> bool; meshblocks it rejects are skipped unread.

  Returns:
  dict as from read_binary, with the fields of all dumps in 'mb_data'.
  """
  fd = None
  for path, names in snap.by_path().items():
    f = read_binary(path, keep, variables=names)
    if fd is None:
      fd = f
      continue
    if not np.array_equal(f['mb_geometry'], fd['mb_geometry']):
      raise SystemExit(f'{path} has a different mesh than the other dumps at t = {snap.time:g}!')
    fd['mb_data'].update(f['mb_data'])
    fd['var_names'] += f['var_names']
  return fd


def _keep_threshold(pars: dict, rho_keep: float | None) -> float:
  dfloor = par(pars, 'mhd', 'dfloor', float, 1.0e-14)
  rho_keep = 1.5 * dfloor if rho_keep is None else rho_keep
  if rho_keep <= dfloor:
    raise SystemExit(f'rho_keep must exceed dfloor = {dfloor:.4g} (code units)')
  return rho_keep


def _read_amr(snap: Snapshot, rho_keep: float | None) -> Cells:
  groups = snap.by_path()
  first = snap.sources['dens'][0]
  fd = read_binary(first, variables=groups[first])
  pars = parse_par_dump(fd['header'])
  nx = (fd['nx1_out_mb'], fd['nx2_out_mb'], fd['nx3_out_mb'])
  geom = fd['mb_geometry']
  xv, yv, zv, dvol, dxv, dyv, dzv = block_cell_geometry(fd)

  rho_full = np.asarray(fd['mb_data']['dens']).reshape(-1)
  sel = np.flatnonzero(rho_full > _keep_threshold(pars, rho_keep))
  ncells_total = rho_full.size
  del rho_full

  def take(f: dict, names: list) -> dict:
    return {n: np.asarray(f['mb_data'][n]).reshape(-1)[sel] for n in names}

  data = take(fd, groups[first])
  time = fd['time']
  del fd
  for path, names in groups.items():
    if path == first:
      continue
    f = read_binary(path, variables=names)
    if not np.array_equal(f['mb_geometry'], geom):
      raise SystemExit(f'{path} has a different mesh than {first}!')
    data.update(take(f, names))
    del f

  i = sel % nx[0]
  j = (sel // nx[0]) % nx[1]
  k = (sel // (nx[0] * nx[1])) % nx[2]
  m = sel // (nx[0] * nx[1] * nx[2])
  dz = dzv[m].astype(np.float32)
  return Cells(time=time, tag=snap.tag, pars=pars, origin=None,
               x=xv[m, i], y=yv[m, j], z=zv[m, k],
               dV=dvol[m].astype(np.float32), dz=dz, h=dz, dz_min=float(np.min(dzv)),
               data={n: data[n] for n in snap.sources},
               grid=AmrGrid(geom, nx, sel, (dxv, dyv, dzv)), ncells_total=ncells_total)


def _read_sph(snap: Snapshot, rho_keep: float | None) -> Cells:
  dumps = {path: SphDump(path) for path in snap.by_path()}
  ref = dumps[snap.sources['dens'][0]]
  for d in dumps.values():
    if not np.allclose(d.center, ref.center, rtol=0.0, atol=CENTER_TOL):
      raise SystemExit(f'sph volumes at t = {ref.time:g} have different centers: '
                       f'{d.path} vs {ref.path}')
  radii = ref.radii.astype(float)
  theta = ref.theta[0].astype(float)
  phi = ref.phi[:, 0].astype(float)
  if len(radii) < 2:
    raise SystemExit(f'{ref.path} holds a single surface, not a volume.')

  fields = {n: dumps[p].field(n) for n, (p, _) in snap.sources.items()}
  if 'z4c_alpha' in fields:
    dup = np.stack([dup_correction(a, ref.theta, ref.phi) for a in fields['z4c_alpha']])
    if np.any(dup != 1.0):
      fields = {n: (v * dup).astype(np.float32) for n, v in fields.items()}

  rho_full = fields['dens'].reshape(-1)
  sel = np.flatnonzero(rho_full > _keep_threshold({}, rho_keep))
  ncells_total = rho_full.size

  r_e, t_e = _radial_edges(radii), _theta_edges(theta)
  dvol = np.broadcast_to(((r_e[1:]**3 - r_e[:-1]**3) / 3.0)[:, None, None]
                         * (2.0 * np.pi / len(phi))
                         * (np.cos(t_e[:-1]) - np.cos(t_e[1:]))[None, None, :],
                         fields['dens'].shape)
  ir, ip, it = np.unravel_index(sel, fields['dens'].shape)
  st = np.sin(theta[it])
  dV = dvol.reshape(-1)[sel].astype(np.float32)
  return Cells(time=ref.time, tag=snap.tag, pars={}, origin=ref.center,
               x=ref.xc + radii[ir] * st * np.cos(phi[ip]),
               y=ref.yc + radii[ir] * st * np.sin(phi[ip]),
               z=ref.zc + radii[ir] * np.cos(theta[it]),
               dV=dV, dz=np.full(sel.size, np.nan, dtype=np.float32), h=np.cbrt(dV),
               dz_min=np.nan, data={n: v.reshape(-1)[sel] for n, v in fields.items()},
               grid=SphGrid(radii, theta, phi, ref.center, sel), ncells_total=ncells_total)


def _radial_edges(r: np.ndarray) -> np.ndarray:
  log = np.all(r > 0) and np.allclose(np.diff(np.log(r)), np.log(r[1] / r[0]), rtol=1e-3)
  c = np.log(r) if log else r
  e = np.concatenate([[1.5 * c[0] - 0.5 * c[1]], 0.5 * (c[:-1] + c[1:]),
                      [1.5 * c[-1] - 0.5 * c[-2]]])
  return np.exp(e) if log else np.maximum(e, 0.0)


def _theta_edges(theta: np.ndarray) -> np.ndarray:
  if np.allclose(np.diff(theta), theta[1] - theta[0], rtol=1e-3):
    mid = 0.5 * (theta[:-1] + theta[1:])
    return np.clip(np.concatenate([[0.0], mid, [np.pi]]), 0.0, np.pi)
  mu = np.cos(theta)
  mid = 0.5 * (mu[:-1] + mu[1:])
  return np.arccos(np.clip(np.concatenate([[1.0], mid, [-1.0]]), -1.0, 1.0))


def _smooth(full: np.ndarray, size, mode) -> np.ndarray:
  good = np.isfinite(full).astype(np.float32)
  wsum = uniform_filter(good, size=size, mode=mode)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(good > 0, uniform_filter(np.nan_to_num(full), size=size, mode=mode) / wsum,
                    np.nan).astype(np.float32)
