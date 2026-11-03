"""
3+1 GRMHD mathematical operations on arrays.

The spatial metric `g` is the tuple (gxx, gxy, gxz, gyy, gyz, gzz), vectors are
3-sequences of arrays.
"""

# Third-party libraries.
import numpy as np


def matrix(g: tuple) -> tuple:
  """
  Reorder the 3-metric components into a 3x3 matrix.

  Parameters:
  g (tuple): tuple of the six symmetric metric components.

  Returns:
  3x3 tuple of the full 3-metric.
  """
  gxx, gxy, gxz, gyy, gyz, gzz = g
  return ((gxx, gxy, gxz), (gxy, gyy, gyz), (gxz, gyz, gzz))


def det(g: tuple) -> float:
  """
  Compute the determinant of the 3-metric.

  Parameters:
  g (tuple): tuple of the six symmetric metric components.

  Returns:
  determinant of the 3-metric.
  """
  gxx, gxy, gxz, gyy, gyz, gzz = g
  return gxx * gyy * gzz + 2 * gxy * gxz * gyz - gxx * gyz**2 - gyy * gxz**2 - gzz * gxy**2


def sqrt_det(g: tuple) -> float:
  """
  Compute the squared-root of the determinant of the 3-metric.

  Parameters:
  g (tuple): tuple of the six symmetric metric components.

  Returns:
  square-root of the determinant of the 3-metric.
  """
  return np.sqrt(det(g))


def inverse(g: tuple) -> tuple:
  """
  Compute the inverse components of the 3-metric in the same
  order (xx, xy, xz, yy, yz, zz).

  Parameters:
  g (tuple): tuple of the six symmetric metric components.

  Returns:
  tuple of the inverse 3-metric components.
  """
  gxx, gxy, gxz, gyy, gyz, gzz = g
  d = det(g)
  return ((gyy * gzz - gyz**2) / d, (gxz * gyz - gxy * gzz) / d, (gxy * gyz - gxz * gyy) / d,
          (gxx * gzz - gxz**2) / d, (gxy * gxz - gxx * gyz) / d, (gxx * gyy - gxy**2) / d)


def lower(g: tuple, v: np.array) -> list:
  """
  Lower vector components with the 3-metric.

  Parameters:
  g    (tuple): tuple of the six symmetric metric components.
  v (np.array): array of the vector components.

  Returns:
  vector v with lower indices.
  """
  m = matrix(g)
  return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def dot(a: np.array, b: np.array) -> float:
  """
  Take the dot product between two vectors.

  Parameters:
  a (np.array): vector 1.
  b (np.array): vector 2.

  Returns:
  the dot product between both vectors a and b.
  """
  return sum(ai * bi for ai, bi in zip(a, b))


def fluid(g: tuple, alp: np.ndarray, bet: list, vel: list, bcc: list = None) -> dict:
  """
  Compute the fluid state on a surface, i.e. the fluid four-velocity from the
  Eulerian \tilde{u}^i = W v^i and, given bcc = \sqrt{\gamma} B^i, the
  magnetic four-vector b^\mu.

  Parameters:
  g        (tuple): tuple of the six symmetric metric components.
  alp (np.ndarray): array of the lapse.
  bet       (list): list of the shift components.
  vel       (list): list of the Eulerian velocity components.
  bcc       (list): list of the densitized Eulerian magnetic field components.

  Returns:
  dict holding the keys - W, u_d (u_i), u_t_up, u_up, u_t, g_tt and with
  bcc also B, b_t_up, b_up, b_d, b_t, b2.
  """
  # Lower the indices on the Eulerian 3-velocity, compute
  # the Lorentz factor, lower the indices on the shift,
  # compute g_tt (for the Penrose analysis), and compute
  # the 4-velocity components.
  u_d = lower(g, vel)
  W = np.sqrt(1.0 + dot(vel, u_d))
  u_t_up = W / alp
  bet_d = lower(g, bet)
  g_tt = -alp**2 + dot(bet, bet_d)
  f = {'W': W, 'u_d': u_d, 'u_t_up': u_t_up,
       'u_up': [vel[i] - bet[i] * u_t_up for i in range(3)],
       'u_t': -alp * W + dot(bet, u_d), 'g_tt': g_tt}
  if bcc is None:
    return f

  # Un-densitize the Eulerian magnetic field components,
  # and compute the fluid frame magnetic field components.
  B = [b / sqrt_det(g) for b in bcc]
  BWv = dot(B, u_d)
  b_t_up = BWv / alp
  b_up = [(B[i] + BWv * f['u_up'][i]) / W for i in range(3)]
  f.update(B=B, b_t_up=b_t_up, b_up=b_up,
           b_d=lower(g, [b_up[i] + bet[i] * b_t_up for i in range(3)]),
           b_t=g_tt * b_t_up + dot(bet_d, b_up),
           b2=(dot(B, lower(g, B)) + BWv**2) / W**2)
  return f
