"""PyCompOSE HDF5 EOS tables."""

# Third-party libraries.
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Plot-tools utility.
from athplot.utils.units import conv_dens, cactus, cgs

# Neutron mass, needed table fields, and chunk size for efficient 3D
# processing.
MN_MEV     = 939.56535                 # neutron mass, must match the table's `mn` scalar
EOS_FIELDS = ("Q1", "Q2", "Q7", "cs2")
EOS_CHUNK  = 4_000_000                 # points per interpolation chunk


def load_eos(path: str) -> dict:
  """
  Load a PyCompOSE HDF5 table and build enthalpy interpolators.

  Parameters:
  path (str): path to the .h5 PyCompOSE EOS table.

  Returns:
  dict with (interp_e, interp_p_rho  - interpolators for Q7 and p/(nb*m_n)
             mn_mev, mn_gram         - neutron mass [MeV], [g]
             h_min                   - minimum specific enthalpy over the table
             nb_min, nb_max, ye_min,
             ye_max, t_min, t_max    - table bounds)
  """
  with h5py.File(path, 'r') as f:
    nb_axis = f['nb'][:]         # [fm^-3]
    yq_axis = f['yq'][:]         # [dimensionless]
    t_axis  = f['t'][:]          # [MeV]
    mn_mev  = float(f['mn'][()]) # neutron mass [MeV]
    Q1_grid = f['Q1'][:]         # p / nb  [MeV]
    Q7_grid = f['Q7'][:]         # e / (nb * m_n) - 1

  axes = (np.log(nb_axis), yq_axis, np.log(t_axis))
  return dict(
    interp_e=RegularGridInterpolator(axes, Q7_grid, method='linear',
                                     bounds_error=False, fill_value=None),
    interp_p_rho=RegularGridInterpolator(axes, Q1_grid / mn_mev, method='linear',
                                         bounds_error=False, fill_value=None),
    mn_mev=mn_mev,
    mn_gram=mn_mev * cgs.eV * 1e6 / cgs.light_speed**2,
    # Minimum of h = 1 + e/(nb*m_n) + p/(nb*m_n), for the Bernoulli criterion.
    h_min=np.min(1.0 + Q7_grid + Q1_grid / mn_mev),
    nb_min=nb_axis.min(), nb_max=nb_axis.max(),
    ye_min=yq_axis.min(), ye_max=yq_axis.max(),
    t_min=t_axis.min(), t_max=t_axis.max(),
  )


def enthalpy(eos: dict, dens: np.ndarray, ye: np.ndarray, temperature: np.ndarray) -> np.ndarray:
  """
  Specific enthalpy h = 1 + eps + p / rho of code-unit density `dens` from a load_eos table.

  Parameters:
  eos               (dict): loaded EOS table from load_eos().
  dens        (np.ndarray): density points of the sphere data.
  ye          (np.ndarray): ye points of the sphere data.
  temperature (np.ndarray): temperature points of the sphere data.

  Returns:
  np.ndarray of the computed enthalpy.
  """
  nb = conv_dens(cactus, cgs, dens) * 1e-39 / eos['mn_gram']
  pts = np.column_stack((
    np.log(np.clip(nb.ravel(), eos['nb_min'], eos['nb_max'])),
    np.clip(ye.ravel(), eos['ye_min'], eos['ye_max']),
    np.log(np.clip(temperature.ravel(), eos['t_min'], eos['t_max']))))
  h = 1.0 + eos['interp_e'](pts) + eos['interp_p_rho'](pts)
  return h.reshape(np.shape(dens))


class EOSTable:
  """
  Load a PyCompOSE HDF5 table and build interpolators for 3D dumps.

  Sets the following fields:
    interpolator - interpolator for the stacked Q1, Q2, Q7 and the sound
      speed squared cs2 (evaluate with interpolate())
    mn_mev - neutron mass [MeV]
    nb_min, nb_max, ye_min, ye_max, t_min, t_max - table bounds
  """
  def __init__(self, path):
    # Read the table.
    with h5py.File(path, 'r') as f:
      nb = f['nb'][:]         # [fm^-3]
      yq = f['yq'][:]         # [dim.less]
      t  = f['t'][:]          # [MeV]
      mn = float(f['mn'][()]) # neutron mass [MeV]
      Q1 = f['Q1'][:]         # p / nb [MeV]
      Q2 = f['Q2'][:]         # s [kB / baryon]
      Q7 = f['Q7'][:]         # e / (nb * m_n) - 1
      cs2 = f['cs2'][:]       # sound speed squared [c^2]

    # AthenaK convention.
    self.log_nb = np.log(nb)
    self.log_t  = np.log(t)
    self.ye     = yq

    self.nb_min, self.nb_max = self.log_nb.min(), self.log_nb.max()
    self.t_min, self.t_max   = self.log_t.min(), self.log_t.max()
    self.ye_min, self.ye_max = self.ye.min(), self.ye.max()

    if not np.isclose(mn, MN_MEV):
      raise SystemExit(f"Table's mn with {mn}MeV does not match {MN_MEV}MeV!")
    else:
      self.mb = mn

    self.hmin = np.min(1.0 + Q7 + Q1 / self.mb)

    # One interpolator over the stacked quantities, so the grid search and
    # corner weights are computed once per point instead of once per table.
    self.interpolator = RegularGridInterpolator(
      (self.log_nb, self.ye, self.log_t),
      np.stack((Q1, Q2, Q7, cs2), axis=-1),
      method='linear', bounds_error=False, fill_value=None,
    )

  def interpolate(self, log_nb, ye, log_t, chunk=EOS_CHUNK):
    """Interpolate Q1, Q2, Q7 and cs2 at the given points."""
    out = np.empty((log_nb.size, len(EOS_FIELDS)))
    for s in range(0, log_nb.size, chunk):
      e = s + chunk
      out[s:e] = self.interpolator(
        np.column_stack((log_nb[s:e], ye[s:e], log_t[s:e])))
    return {q: out[:, n] for n, q in enumerate(EOS_FIELDS)}
