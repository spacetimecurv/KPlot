"""Quadrature helpers for integrals over an AthenaK spherical extraction surface."""

# Third-party libaries.
import numpy as np


def calc_ur(ux: np.array, uy: np.array, uz: np.array,
            theta: np.array, phi: np.array) -> np.array:
  """
  Calculate the radial component of a vector.

  Parameters:
  ux, uy, uz (np.array): three components of the vector.
  theta      (np.array): theta surface array.
  phi        (np.array): phi surface array.

  Returns:
  array of the radial component of the vector.
  """
  return (ux * np.sin(theta) * np.cos(phi) +
          uy * np.sin(theta) * np.sin(phi) +
          uz * np.cos(theta))


def calc_uphi(ux: np.array, uy: np.array, phi: np.array) -> np.array:
  """
  Compute the azimuthal component of a vector.

  Parameters:
  ux, uy (np.array): planar vector components.
  phi    (np.array): phi surface array.

  Returns:
  array of the azimuthal component of the vector.
  """
  return -ux * np.sin(phi) + uy * np.cos(phi)


def sum_over_time(arr: np.ndarray, time: np.ndarray) -> np.ndarray:
  """
  Integrate array over time using a midpoint Riemann sum.

  Each snapshot contributes arr[i] * dt[i], where dt[i] is the midpoint
  width of the time interval around snapshot i.

  Parameters"
  arr  (np.ndarray): array to be integrated - shape (n_times, ...)
  time (np.ndarray): time array - shape (n_times,)

  Returns:
  the integrated np.ndarray.
  """
  dt = np.empty_like(time)
  dt[0]    = time[1] - time[0]
  dt[-1]   = time[-1] - time[-2]
  dt[1:-1] = 0.5 * (time[2:] - time[:-2])
  return np.sum(arr * dt[(slice(None),) + (None,) * (arr.ndim - 1)], axis=0)


def _get_widths(c: list) -> float:
  """
  Cell widths from center coordinates via midpoint rule.

  Parameters:
  c (list): list of centered coordinate points.

  Returns:
  cell width from the centered coordinates.
  """
  e = np.zeros(len(c) + 1)
  e[1:-1] = 0.5 * (c[:-1] + c[1:])
  e[0]  = c[0]  - (e[1]  - c[0])
  e[-1] = c[-1] + (c[-1] - e[-2])
  return np.abs(np.diff(e))


def _pole_weights(shape: tuple, theta_1d: np.array, nphi: int) -> np.ndarray:
  """
  Pole columns are shared by every phi, so they count 1/Nphi each,
  so compute the weights of each angle phi.

  Parameters:
  shape       (tuple): shape of the weight array.
  theta_1d (np.array): flattened theta array.
  nphi          (int): number of azimuthal points on the surface.

  Returns:
  weights for the pole columns.
  """
  # Pole columns are shared by every phi, so they count 1/Nphi each.
  weights = np.ones(shape)
  is_pole = np.isclose(theta_1d, 0, atol=1e-7) | np.isclose(theta_1d, np.pi, atol=1e-7)
  weights[:, is_pole] /= nphi
  return weights


def get_riemann_weights(integrand: np.ndarray, theta_1d: np.array,
                        phi_1d: np.array) -> np.ndarray:
  """
  Per-cell Riemann integration weights for a 2D spherical surface integral.

  Parameters:
  integrand (np.ndarray): integrand for the shape of the weights.
  theta_1d    (np.array): flattened theta array.
  phi_1d      (np.array): flattened phi array.

  Returns:
  an array of shape (Nphi, Ntheta) containing dtheta * dphi for each
  cell, with pole cells divided by Nphi to avoid double-counting the singular
  axis.
  """
  weights = _pole_weights(np.shape(integrand), theta_1d, len(phi_1d))
  return weights * _get_widths(theta_1d)[None, :] * _get_widths(phi_1d)[:, None]


def integrate_over_phi_riemann(integrand: np.ndarray, theta_1d: np.array,
                               phi_1d: np.array) -> np.ndarray:
  """
  Calculates d(dM/dtheta)/dt by integrating the 2D integrand over phi
  using a Riemann sum with polar weighting.

  Parameters:
  integrand (np.ndarray): integrand for the shape of the weights.
  theta_1d    (np.array): flattened theta array.
  phi_1d      (np.array): flattened phi array.

  Returns:
  the integrated array.
  """
  weights = _pole_weights(np.shape(integrand), theta_1d, len(phi_1d))
  return np.sum(integrand * weights * _get_widths(phi_1d)[:, None], axis=0)
