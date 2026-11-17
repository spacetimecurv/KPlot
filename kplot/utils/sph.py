"""
Reading AthenaK spherical-surface (``file_type = sph``) VTK dumps.

A dump holds one or several concentric surfaces on a shared angular grid. Points are
stored with the radius varying fastest, then theta, then phi; every field is a
big-endian float32 block, so single fields are read without touching the rest.
"""

# Built-in libraries.
import os

# Third-party libaries.
import numpy as np

# A stored surface counts as the requested radius when it lies within this distance
# [M_sun].
RADIUS_TOL = 0.05
CENTER_TOL = 0.5


def read_header(path: str) -> dict:
  """
  Parse the `key=value` metadata AthenaK writes on the second line.

  Parameters:
  path (str): path to the dump.

  Returns:
  dict with the metadata keys and their (string) values.
  """
  with open(path, 'rb') as f:
    f.readline()
    meta = f.readline().decode('latin_1')
  return dict(tok.split('=', 1) for tok in meta.split() if '=' in tok)


def read_time(path: str) -> float:
  """
  Simulation time of a dump, without parsing the field data.

  Parameters:
  path (str): path to the dump.

  Returns:
  simulation time [M_sun].
  """
  return float(read_header(path)['time'])


def read_layout(path: str) -> dict:
  """
  Dimensions, metadata and byte offsets of a dump, read from its ASCII headers only.

  Parameters:
  path (str): path to the dump.

  Returns:
  dict with 'meta', 'dims' (nradii, ntheta, nphi), 'points' (offset), 'radii' (offset
  or None) and 'fields' ({name: offset}).
  """
  out = {'meta': read_header(path), 'radii': None, 'fields': {}}
  with open(path, 'rb') as f:
    f.readline()
    f.readline()
    while True:
      line = f.readline()
      if not line:
        break
      tok = line.split()
      if not tok:
        continue
      key = tok[0]
      if key == b'DIMENSIONS':
        out['dims'] = tuple(int(t) for t in tok[1:4])
      elif key == b'POINTS':
        out['points'] = f.tell()
        f.seek(12 * int(tok[1]), 1)
      elif key == b'FIELD':
        for _ in range(int(tok[2])):
          entry = f.readline().split()
          while not entry:
            entry = f.readline().split()
          name, ncomp, ntup = entry[:3]
          if name == b'RADII':
            out['radii'] = f.tell()
          f.seek(4 * int(ncomp) * int(ntup), 1)
      elif key == b'POINT_DATA':
        npoints = int(tok[1])
      elif key == b'SCALARS':
        f.readline()
        out['fields'][tok[1].decode()] = f.tell()
        f.seek(4 * npoints, 1)
  return out


class SphDump:
  """
  One sph dump. Fields are read on first access.

  Parameters:
  path (str): path to the dump.
  """

  def __init__(self, path: str) -> None:
    lay = read_layout(path)
    meta = lay['meta']
    self.path = path
    self.nradii, self.ntheta, self.nphi = lay['dims']
    self.npoints = self.nradii * self.ntheta * self.nphi
    self.time = float(meta['time'])
    self.cycle = int(meta.get('cycle', -1))
    self.xc, self.yc, self.zc = (float(meta.get(k, 0.0)) for k in ('xc', 'yc', 'zc'))
    self.theta_spacing = meta.get('theta_spacing')
    self.field_names = list(lay['fields'])
    self._offsets = lay['fields']
    self._cache = {}

    pts = np.memmap(path, dtype='>f4', mode='r', offset=lay['points'],
                    shape=(self.nphi, self.ntheta, self.nradii, 3))
    self.radii = np.asarray(pts[0, 0, :, 0], dtype=np.float32)
    self.theta = np.asarray(pts[:, :, 0, 1], dtype=np.float32)
    self.phi = np.asarray(pts[:, :, 0, 2], dtype=np.float32)
    del pts

  @property
  def center(self) -> np.ndarray:
    return np.array([self.xc, self.yc, self.zc])

  def field(self, name: str) -> np.ndarray:
    """
    Field on all surfaces.

    Parameters:
    name (str): name of the field.

    Returns:
    float32 array of shape (nradii, nphi, ntheta).
    """
    if name not in self._cache:
      raw = np.fromfile(self.path, dtype='>f4', count=self.npoints,
                        offset=self._offsets[name])
      self._cache[name] = np.moveaxis(
          raw.astype(np.float32).reshape(self.nphi, self.ntheta, self.nradii), -1, 0)
    return self._cache[name]

  def shell(self, name: str, iradius: int = 0) -> np.ndarray:
    """
    Field on one surface.

    Parameters:
    name    (str): name of the field.
    iradius (int): index of the surface.

    Returns:
    float32 array of shape (nphi, ntheta).
    """
    return self.field(name)[iradius]


class Shell:
  """
  One extraction surface of an sph dump, however many surfaces the file holds.

  Parameters:
  path          (str): path to the dump.
  iradius       (int): index of the surface in the dump.
  data    (SphDump): already loaded dump, read from `path` if None.
  """

  def __init__(self, path: str, iradius: int = 0, data: SphDump = None) -> None:
    self._data = data if data is not None else SphDump(path)
    if not 0 <= iradius < self._data.nradii:
      raise IndexError(f'{os.path.basename(path)} holds '
                       f'{self._data.nradii} surface(s), no index {iradius}.')
    self.path = path
    self.iradius = iradius
    self.nradii = self._data.nradii
    self.radius = float(self._data.radii[iradius])
    self.center = self._data.center
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
    return self._data.shell(var, self.iradius)


def dup_correction(alpha: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
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
